# -*- coding: utf-8 -*-
"""
Prueba de punta a punta de la demo, sin UI.

Es la pregunta del stand: cuando alguien provoca un defecto, Eyve lo
encuentra, lo sigue mientras cruza el encuadre, y lo cuenta UNA vez?

Y la segunda pregunta, que es el argumento de la demo: cada motor hace lo
suyo.

    YOLO    nombra rayon y mancha, que son las que se cuentan
    Patron  encuentra los fallos de impresion, que NO son clases

Cada caso va con su caso negativo: tela limpia no cuenta ni marca nada.
"""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SCRATCH = Path(tempfile.gettempdir()) / "eyve_tests"
SCRATCH.mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np

from eyve.demo.textile import PRINT_FAULTS, TextilePattern, YOLO_CLASSES
from eyve.inference.tracker import InstanceTracker, RawDetection, TrackerConfig
from eyve.modules import CountingModule, PatternModule

WEIGHTS = (REPO / "projects" / "Demo_Textil" / "models" / "best.pt")
if not WEIGHTS.exists():
    WEIGHTS = (REPO / "projects" / "Demo_Textil" / "runs" / "train"
               / "weights" / "best.pt")

fails = []


def check_true(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


assert WEIGHTS.exists(), f"no hay pesos en {WEIGHTS}"
from ultralytics import YOLO
model = YOLO(str(WEIGHTS))
names = list(model.names.values())
print(f"modelo: {WEIGHTS.name}  clases={names}")
check_true("el modelo entrena solo las clases puntuales",
           sorted(names) == sorted(YOLO_CLASSES), str(names))


def nueva(motif="diamantes", weave="sarga"):
    p = TextilePattern(width=960, height=540, axis="x", motif=motif,
                       weave=weave, speed=130.0, tilt_deg=1.6, seed=7)
    p.offset = 300.0
    return p


def correr(pintar, pasos=26, dt=0.16, metodo="appear", guardar=None):
    """
    Simula `pasos` frames de la demo con los DOS motores.
    Devuelve (conteo_yolo, por_clase, max_en_pantalla, frames_con_anomalia).
    """
    p = nueva()
    # el modulo Patron se calibra con la tela limpia, antes de ensuciarla
    pat = PatternModule()
    pat.enabled = True
    pat.sensitivity = 70
    for _ in range(8):
        pat.calibrate(p.frame())
        p.advance(0.2)

    if pintar:
        pintar(p)

    tk = InstanceTracker(config=TrackerConfig(
        iou_match=0.20, max_lost_frames=18, min_confirm_frames=2,
        process_conf_min=0.30))
    cm = CountingModule()
    cm.enabled = True
    cm.set_method(metodo)

    max_pantalla = 0
    con_anomalia = 0
    tiras = []
    for i in range(pasos):
        frame = p.frame()
        res = model.predict(frame, conf=0.35, imgsz=512, verbose=False)
        dets = []
        for r in res:
            for b in r.boxes:
                ci = int(b.cls[0])
                xy = b.xyxy[0].tolist()
                dets.append(RawDetection(names[ci], float(b.conf[0]),
                                         int(xy[0]), int(xy[1]),
                                         int(xy[2]), int(xy[3])))
        tracks = tk.update(dets, frame)
        cm.update_tracks(tracks, expired=tk.last_expired, frame_wh=(960, 540))
        max_pantalla = max(max_pantalla, sum(1 for t in tracks if t.confirmed))
        regs = pat.analyze(frame)
        if regs:
            con_anomalia += 1
        if guardar and i % 6 == 0 and len(tiras) < 4:
            vis = frame.copy()
            for tr in tracks:
                if not tr.confirmed:
                    continue
                x1, y1, x2, y2 = tr.bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 215, 255), 2)
                cv2.putText(vis, f"{tr.label} #{tr.id}", (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
            pat.draw(vis)
            cv2.putText(vis, f"conteo: {cm.total}", (16, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 118), 2)
            tiras.append(cv2.resize(vis, (480, 270)))
        p.advance(dt)
    if guardar and tiras:
        while len(tiras) < 4:
            tiras.append(np.zeros((270, 480, 3), np.uint8))
        cv2.imwrite(str(SCRATCH / guardar),
                    np.vstack([np.hstack(tiras[:2]), np.hstack(tiras[2:4])]))
    return cm.total, dict(cm.counts), max_pantalla, con_anomalia


print("\n[1] CASO NEGATIVO: tela limpia, ningun motor marca nada")
total, por_cls, maxp, anom = correr(None, pasos=20)
check_true("YOLO no cuenta nada", total == 0, f"conteo={total} {por_cls}")
check_true("YOLO no ve nada en pantalla", maxp == 0, f"max={maxp}")
check_true("Patron tampoco marca", anom <= 1, f"{anom}/20 frames")

print("\n[2] YOLO: nombra y cuenta lo que le ensenaste")
for cls in YOLO_CLASSES:
    def pintar(p, _c=cls):
        fx, fy = p.screen_to_fabric(700, 270)
        if _c == "rayon":
            p.streak(fx, fy, length=150, thickness=6)
        else:
            p.blob(fx, fy, size=80)

    total, por_cls, maxp, _ = correr(pintar, guardar=f"e2e_{cls}.png")
    check_true(f"{cls}: lo encuentra", maxp >= 1, f"max en pantalla={maxp}")
    check_true(f"{cls}: lo cuenta UNA vez", total == 1, f"conteo={total} {por_cls}")
    check_true(f"{cls}: con la clase correcta", por_cls.get(cls, 0) == 1,
               str(por_cls))

print("\n[3] Patron: encuentra los fallos que NO son clases")
for fault, hacer in [
        ("fantasma",        lambda p, f: p.ghost(*f, size=170)),
        ("offset",          lambda p, f: p.misregister(*f, size=180)),
        ("falta_impresion", lambda p, f: p.ink_starved(*f, size=170,
                                                       severity=0.8))]:
    def pintar(p, _h=hacer):
        _h(p, p.screen_to_fabric(560, 270))

    total, por_cls, maxp, anom = correr(pintar, pasos=18,
                                        guardar=f"e2e_{fault}.png")
    check_true(f"{fault}: Patron lo marca", anom >= 3, f"{anom}/18 frames")
    # y YOLO NO lo confunde con una de sus clases: para eso se generaron
    # estos fallos sin etiqueta en el entrenamiento
    check_true(f"{fault}: YOLO no lo confunde con una clase", total == 0,
               f"conteo={total} {por_cls}")

print("\n[4] Los dos a la vez, que es el argumento de la demo")


def pintar_mixto(p):
    p.streak(*p.screen_to_fabric(720, 160), length=140, thickness=6)
    p.misregister(*p.screen_to_fabric(380, 360), size=170)


total, por_cls, maxp, anom = correr(pintar_mixto, pasos=22,
                                    guardar="e2e_mixto.png")
check_true("YOLO cuenta el rayon", total == 1, f"conteo={total} {por_cls}")
check_true("Patron marca el fallo de impresion", anom >= 4, f"{anom}/22")

print("\n[5] Metodo 'cruce de meta' con la linea en medio")


def correr_meta():
    p = nueva()
    pintar_mixto(p)
    tk = InstanceTracker(config=TrackerConfig(
        iou_match=0.20, max_lost_frames=18, min_confirm_frames=2))
    cm = CountingModule()
    cm.enabled = True
    cm.set_method("line")
    cm.line = (480, 0, 480, 540)
    for _ in range(34):
        frame = p.frame()
        res = model.predict(frame, conf=0.35, imgsz=512, verbose=False)
        dets = []
        for r in res:
            for b in r.boxes:
                ci = int(b.cls[0])
                xy = b.xyxy[0].tolist()
                dets.append(RawDetection(names[ci], float(b.conf[0]),
                                         int(xy[0]), int(xy[1]),
                                         int(xy[2]), int(xy[3])))
        tracks = tk.update(dets, frame)
        cm.update_tracks(tracks, expired=tk.last_expired, frame_wh=(960, 540))
        p.advance(0.16)
    return cm.total, dict(cm.counts), cm.dir_counts


total, por_cls, dirs = correr_meta()
check_true("el rayon cruza la meta y se cuenta una vez", total == 1,
           f"conteo={total} {por_cls} sentidos={dirs}")
check_true("todos en el mismo sentido (la tela va para un lado)",
           dirs["fwd"] == 0 or dirs["rev"] == 0, str(dirs))

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f_ in fails:
        print("  -", f_)
    sys.exit(1)
print("DEMO DE PUNTA A PUNTA: CADA MOTOR HACE LO SUYO")
