# -*- coding: utf-8 -*-
"""
Prueba de punta a punta de la demo, sin UI.

Es la pregunta que de verdad importa para el stand: cuando alguien dibuja
un defecto sobre la tela, ¿Eyve lo encuentra, lo sigue mientras cruza el
encuadre, y lo cuenta UNA vez?

Se simula exactamente eso: tela limpia -> dibujar -> dejar que la tela
viaje -> medir. Y como caso negativo, tela limpia sin dibujar nada, donde
no debe contar nada.
"""
import sys
import tempfile
from pathlib import Path

# El repo es el padre de tests/: nada de rutas absolutas, para que las
# pruebas corran en cualquier maquina y desde cualquier carpeta.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Salidas de las pruebas (hojas de contacto, proyectos temporales).
SCRATCH = Path(tempfile.gettempdir()) / "eyve_tests"
SCRATCH.mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np

from eyve.demo.textile import TextilePattern, DEFECT_CLASSES
from eyve.inference.tracker import InstanceTracker, RawDetection, TrackerConfig
from eyve.modules import CountingModule

SP = SCRATCH
ROOT = REPO
WEIGHTS = ROOT / "projects" / "Demo_Textil" / "runs" / "train" / "weights" / "best.pt"

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


def correr(pintar, pasos=26, dt=0.16, metodo="appear", guardar=None):
    """
    Simula `pasos` frames de la demo. `pintar(p)` dibuja en la tela antes
    de empezar el viaje. Devuelve (conteo, por_clase, max_en_pantalla).
    """
    p = TextilePattern(width=960, height=540, axis="x", motif="diamantes",
                       speed=130.0, tilt_deg=1.6, seed=7)
    p.offset = 300.0
    if pintar:
        pintar(p)

    tk = InstanceTracker(config=TrackerConfig(
        iou_match=0.20, max_lost_frames=18, min_confirm_frames=2,
        process_conf_min=0.30))
    cm = CountingModule()
    cm.enabled = True
    cm.set_method(metodo)

    max_pantalla = 0
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
        if guardar and i % 6 == 0 and len(tiras) < 4:
            vis = frame.copy()
            for tr in tracks:
                if not tr.confirmed:
                    continue
                x1, y1, x2, y2 = tr.bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 215, 255), 2)
                cv2.putText(vis, f"{tr.label} #{tr.id}", (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
            cv2.putText(vis, f"conteo: {cm.total}", (16, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 118), 2)
            tiras.append(cv2.resize(vis, (480, 270)))
        p.advance(dt)
    if guardar and tiras:
        while len(tiras) < 4:
            tiras.append(np.zeros((270, 480, 3), np.uint8))
        cv2.imwrite(str(SP / guardar),
                    np.vstack([np.hstack(tiras[:2]), np.hstack(tiras[2:4])]))
    return cm.total, dict(cm.counts), max_pantalla


print("\n[1] CASO NEGATIVO: tela limpia, no debe contar nada")
total, por_cls, maxp = correr(None, pasos=20)
check_true("tela limpia -> conteo 0", total == 0, f"conteo={total} {por_cls}")
check_true("tela limpia -> nada en pantalla", maxp == 0, f"max={maxp}")

print("\n[2] Un defecto de cada clase, dibujado y seguido mientras viaja")
for cls in DEFECT_CLASSES:
    def pintar(p, _c=cls):
        fx, fy = p.screen_to_fabric(700, 270)
        if _c == "rayon":
            p.streak(fx, fy, length=130, thickness=6)
        elif _c == "mancha":
            p.blob(fx, fy, size=70)
        else:
            p.missing_print(fx, fy, size=85)

    total, por_cls, maxp = correr(pintar, guardar=f"e2e_{cls}.png")
    check_true(f"{cls}: lo encuentra", maxp >= 1, f"max en pantalla={maxp}")
    check_true(f"{cls}: lo cuenta UNA vez", total == 1,
               f"conteo={total}  {por_cls}")
    acerto = por_cls.get(cls, 0) == 1
    check_true(f"{cls}: con la clase correcta", acerto, str(por_cls))

print("\n[3] Tres defectos a la vez: tres instancias, tres conteos")


def pintar3(p):
    p.streak(*p.screen_to_fabric(760, 140), length=120, thickness=6)
    p.blob(*p.screen_to_fabric(700, 300), size=72)
    p.missing_print(*p.screen_to_fabric(820, 430), size=85)


total, por_cls, maxp = correr(pintar3, pasos=30, guardar="e2e_tres.png")
check_true("ve los tres a la vez", maxp >= 3, f"max en pantalla={maxp}")
check_true("cuenta exactamente 3", total == 3, f"conteo={total}  {por_cls}")
check_true("una de cada clase", len(por_cls) == 3, str(por_cls))

print("\n[4] Metodo 'en pantalla' sigue la ocupacion, no acumula")
total, por_cls, maxp = correr(pintar3, pasos=30, metodo="screen")
check_true("en pantalla no acumula mas alla de lo visible",
           total <= 3, f"ultimo={total} max visto={maxp}")

print("\n[5] Metodo 'cruce de meta' con la linea en medio")


def correr_meta():
    p = TextilePattern(width=960, height=540, axis="x", motif="diamantes",
                       speed=130.0, tilt_deg=1.6, seed=7)
    p.offset = 300.0
    pintar3(p)
    tk = InstanceTracker(config=TrackerConfig(
        iou_match=0.20, max_lost_frames=18, min_confirm_frames=2))
    cm = CountingModule()
    cm.enabled = True
    cm.set_method("line")
    cm.line = (480, 0, 480, 540)      # meta vertical al centro
    for _ in range(34):
        frame = p.frame()
        res = model.predict(frame, conf=0.35, imgsz=512, verbose=False)
        dets = []
        for r in res:
            for b in r.boxes:
                ci = int(b.cls[0]); xy = b.xyxy[0].tolist()
                dets.append(RawDetection(names[ci], float(b.conf[0]),
                                         int(xy[0]), int(xy[1]),
                                         int(xy[2]), int(xy[3])))
        tracks = tk.update(dets, frame)
        cm.update_tracks(tracks, expired=tk.last_expired, frame_wh=(960, 540))
        p.advance(0.16)
    return cm.total, dict(cm.counts), cm.dir_counts


total, por_cls, dirs = correr_meta()
check_true("los 3 cruzan la meta y se cuentan una vez",
           total == 3, f"conteo={total} {por_cls} sentidos={dirs}")
check_true("todos en el mismo sentido (la tela va para un lado)",
           dirs["fwd"] == 0 or dirs["rev"] == 0, str(dirs))

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("DEMO DE PUNTA A PUNTA: EYVE ENCUENTRA, SIGUE Y CUENTA LO QUE SE DIBUJA")
