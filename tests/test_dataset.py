# -*- coding: utf-8 -*-
"""
Verifica el dataset generado LEYENDOLO DEL DISCO como lo leeria YOLO.

No vuelve a preguntarle al generador donde puso las cajas (eso solo
comprobaria que el generador es consistente consigo mismo). Carga el .jpg
y el .txt, convierte de YOLO normalizado a pixeles, y comprueba contra la
imagen que ahi hay un defecto: la region de la caja tiene que diferir de la
tela de alrededor. Si las etiquetas estuvieran corridas, esto falla.
"""
import sys
import random
import shutil
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
import yaml

from eyve.demo.dataset import build_dataset
from eyve.demo.textile import DEFECT_CLASSES, PRINT_FAULTS, YOLO_CLASSES

OUT = SCRATCH / "Demo_Verif"
fails = []


def check(name, got, want):
    ok = got == want
    print(f"  {'OK  ' if ok else 'FALLA'}  {name}: got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


def check_true(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


for stale in (OUT, SCRATCH / "Demo_Textil", SCRATCH / "Demo_Textil_test"):
    if stale.exists():
        shutil.rmtree(stale)

print("\n[1] Generar")
proj = build_dataset(OUT, frames=120, seed=99)
check("carpeta exacta (sin huerfanas)", proj.root.name, "Demo_Verif")
check_true("no se creo una carpeta extra",
           not (SCRATCH / "Demo_Textil").exists())
check("solo las clases entrenables", proj.class_names, list(YOLO_CLASSES))

print("\n[2] Estructura en disco")
p = proj.paths
imgs = sorted(p.raw_images.rglob("*.jpg"))
check("120 imagenes", len(imgs), 120)
lbls = sorted(p.tagged_labels.glob("*.txt"))
check("120 etiquetas", len(lbls), 120)
check_true("cada imagen tiene SU etiqueta canonica",
           all(p.label_file(i).exists() for i in imgs))

print("\n[3] data.yaml coherente con el proyecto")
dy = yaml.safe_load(p.dataset_yaml.read_text(encoding="utf-8"))
check("nc", dy["nc"], len(YOLO_CLASSES))
check("names en orden", [dy["names"][i] for i in range(len(YOLO_CLASSES))],
      list(YOLO_CLASSES))
tr = list((p.dataset / "images" / "train").glob("*.jpg"))
va = list((p.dataset / "images" / "val").glob("*.jpg"))
check_true("split train/val poblado", len(tr) > 0 and len(va) > 0,
           f"train={len(tr)} val={len(va)}")
check_true("cada imagen del split tiene su etiqueta",
           all((p.dataset / "labels" / s / (f.stem + ".txt")).exists()
               for s, fs in (("train", tr), ("val", va)) for f in fs))

print("\n[4] Formato YOLO valido")
n_box = 0
n_clean = 0
for lb in lbls:
    txt = lb.read_text(encoding="utf-8").strip()
    if not txt:
        n_clean += 1
        continue
    for line in txt.splitlines():
        parts = line.split()
        check_true_once = len(parts) == 5
        if not check_true_once:
            fails.append(f"linea mal formada en {lb.name}: {line}")
            continue
        ci = int(parts[0])
        vals = [float(v) for v in parts[1:]]
        if not (0 <= ci < len(YOLO_CLASSES)):
            fails.append(f"class_id fuera de rango en {lb.name}: {ci}")
        if not all(0.0 <= v <= 1.0 for v in vals):
            fails.append(f"coordenada fuera de [0,1] en {lb.name}: {vals}")
        if vals[2] <= 0 or vals[3] <= 0:
            fails.append(f"caja de area cero en {lb.name}")
        n_box += 1
print(f"  OK    {n_box} cajas, {n_clean} frames limpios, formato valido")
check_true("hay frames limpios (fondo negativo)", n_clean >= 5, f"n={n_clean}")
check_true("hay suficientes cajas", n_box > 100, f"n={n_box}")

print("\n[4b] Los fallos de impresion se generan y NO se etiquetan")
# Es el reparto entre los dos motores, llevado al dataset: YOLO aprende que
# un fallo de impresion es fondo, y quien lo encuentra es el modulo Patron.
# Se comprueba de dos formas, porque la ausencia de etiqueta por si sola no
# prueba que el fallo este ahi: (a) ninguna etiqueta los menciona, y (b) el
# modulo Patron, calibrado, los encuentra en varios frames del dataset.
ids_validos = set(range(len(YOLO_CLASSES)))
malas = []
for lb in lbls:
    for line in lb.read_text(encoding="utf-8").strip().splitlines():
        if line.strip() and int(line.split()[0]) not in ids_validos:
            malas.append(lb.name)
check("ninguna etiqueta fuera de las clases entrenables", malas, [])

from eyve.modules.pattern_module import PatternModule

_rng_pat = random.Random(7)
pm = PatternModule()
pm.enabled = True
pm.sensitivity = 70
limpias = [i for i in imgs if not proj.paths.label_file(i).read_text().strip()]
for i in limpias[:10]:
    pm.calibrate(cv2.imread(str(i)))
hallados = revisados = 0
for i in _rng_pat.sample(imgs, min(40, len(imgs))):
    revisados += 1
    if pm.analyze(cv2.imread(str(i))):
        hallados += 1
print(f"        el modulo Patron marca {hallados}/{revisados} frames del dataset")
check_true("el dataset SI trae fallos que YOLO no etiqueta",
           hallados >= 3, f"{hallados}/{revisados}")

print("\n[5] Las clases entrenables aparecen")
per_cls = {c: 0 for c in range(len(YOLO_CLASSES))}
for lb in lbls:
    for line in lb.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            per_cls[int(line.split()[0])] += 1
print("       ", {YOLO_CLASSES[k]: v for k, v in per_cls.items()})
check_true("ninguna clase vacia", all(v > 10 for v in per_cls.values()))

def firma(roi, cls):
    """Fraccion de pixeles de *roi* con la firma de color de *cls*."""
    b = roi[:, :, 0].astype(np.int16)
    g = roi[:, :, 1].astype(np.int16)
    r = roi[:, :, 2].astype(np.int16)
    if cls == "rayon":
        return float(((b + g + r) < 330).mean())
    if cls == "mancha":
        return float(((r > b + 25) & ((b + g + r) < 500)).mean())
    # solo quedan dos clases entrenables; cualquier otra es un error
    raise AssertionError("clase no entrenable en una etiqueta: " + cls)


def cajas_de(img_path, dx=0, dy=0, rel=0.0):
    """
    Cajas del .txt en pixeles, opcionalmente corridas.

    dx/dy  corrimiento fijo en pixeles
    rel    corrimiento PROPORCIONAL al tamano de cada caja. Hace falta
           porque los defectos van de 40 a 400 px: un corrimiento fijo de
           90 px saca de su sitio a uno chico pero deja al grande casi
           encima de si mismo, y el control negativo deja de controlar.
    """
    txt = p_.label_file(img_path).read_text(encoding="utf-8").strip()
    if not txt:
        return None, []
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    out = []
    for line in txt.splitlines():
        ci, cx, cy, bw, bh = line.split()
        cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
        ox = dx + int(rel * bw * W)
        oy = dy + int(rel * bh * H)
        x1 = int((cx - bw / 2) * W) + ox; x2 = int((cx + bw / 2) * W) + ox
        y1 = int((cy - bh / 2) * H) + oy; y2 = int((cy + bh / 2) * H) + oy
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 >= 10 and y2 - y1 >= 10:
            out.append((YOLO_CLASSES[int(ci)], x1, y1, x2, y2))
    return img, out


def tasa_acierto(dx=0, dy=0, rel=0.0, umbral=0.10):
    aciertos = total = 0
    detalle = []
    for ip in muestras:
        img, cajas = cajas_de(ip, dx, dy, rel)
        if img is None:
            continue
        for cls, x1, y1, x2, y2 in cajas:
            total += 1
            f = firma(img[y1:y2, x1:x2], cls)
            if f >= umbral:
                aciertos += 1
            else:
                detalle.append((ip.name, cls, round(f, 3)))
    return aciertos, total, detalle


p_ = p
rng = random.Random(0)
muestras = rng.sample(imgs, 40)

print("\n[6] PRUEBA DE FUEGO: la caja contiene la firma de SU clase")
ok_n, tot, peores = tasa_acierto()
print(f"       {ok_n}/{tot} cajas contienen la firma de su clase")
if peores[:5]:
    print("        peores:", peores[:5])
check_true("las etiquetas caen sobre el defecto correcto",
           tot > 0 and ok_n / tot > 0.95, f"{ok_n}/{tot}")

print("\n[7] El chequeo SI distingue: con etiquetas corridas debe desplomarse")
# Corrimiento proporcional: 1.3 veces el tamano de cada caja la saca por
# completo de su defecto, mida lo que mida.
for rel in (1.3, -1.3):
    bad_n, bad_tot, _ = tasa_acierto(rel=rel)
    tasa = bad_n / max(1, bad_tot)
    print(f"       corrimiento de {rel:+.1f} veces su tamano: "
          f"{bad_n}/{bad_tot} ({tasa:.0%})")
    check_true(f"rechaza corrimiento proporcional ({rel:+.1f}x)", tasa < 0.35,
               f"tasa={tasa:.0%}")

print("\n[8] Cada clase por separado")
por_cls = {}
for ip in muestras:
    img, cajas = cajas_de(ip)
    if img is None:
        continue
    for cls, x1, y1, x2, y2 in cajas:
        a, t = por_cls.get(cls, (0, 0))
        f = firma(img[y1:y2, x1:x2], cls)
        por_cls[cls] = (a + (1 if f >= 0.10 else 0), t + 1)
for cls in YOLO_CLASSES:
    a, t = por_cls.get(cls, (0, 0))
    if t:
        check_true(f"clase {cls}", a / t > 0.90, f"{a}/{t}")
    else:
        fails.append(f"clase {cls} sin muestras")

# hoja de contacto para revisar a ojo
hoja = []
for img_path in sorted(imgs)[:12]:
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    txt = p.label_file(img_path).read_text(encoding="utf-8").strip()
    for line in txt.splitlines():
        if not line.strip():
            continue
        ci, cx, cy, bw, bh = line.split()
        cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
        x1 = int((cx - bw/2) * W); y1 = int((cy - bh/2) * H)
        x2 = int((cx + bw/2) * W); y2 = int((cy + bh/2) * H)
        col = [(0,215,255),(255,190,0)][int(ci)]
        cv2.rectangle(img, (x1,y1), (x2,y2), col, 2)
        cv2.putText(img, YOLO_CLASSES[int(ci)], (x1, max(12,y1-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    hoja.append(cv2.resize(img, (320, 180)))
filas = [np.hstack(hoja[i:i+4]) for i in range(0, 12, 4)]
cv2.imwrite(str(SCRATCH / "dataset_hoja.png"), np.vstack(filas))
print("\n  hoja de contacto: dataset_hoja.png")

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails[:20]:
        print("  -", f)
    sys.exit(1)
print("DATASET: TODAS LAS PRUEBAS PASARON")
