# -*- coding: utf-8 -*-
"""
Las escenas del conteo cuentan lo que de verdad pasa en ellas.

No basta con que la demo se vea bien: si la escena muestra tres piezas
cruzando la meta y el contador dice cinco, la demo ensena algo falso — y es
peor que no tenerla, porque el visitante se lleva una idea equivocada.

Asi que la verdad se saca de la ESCENA, siguiendo los objetos Piece, y se
compara contra lo que cuenta el modulo. No se le pregunta al contador si
esta de acuerdo consigo mismo.
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

from eyve.demo.conveyor import PIECE_LABEL, SCENES, SCENE_INFO, CountingScene
from eyve.inference.tracker import InstanceTracker, TrackerConfig
from eyve.modules import CountingModule

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


DT = 1 / 30.0


# Las piezas se siguen por su `uid`, no por id() de Python: id() se
# reutiliza en cuanto un objeto se libera, y dos piezas distintas acababan
# compartiendolo. Con eso la "verdad" de la escena salia mal y la prueba
# acusaba al codigo de errores suyos.
def montar(metodo, seed=7):
    esc = CountingScene(method=metodo, seed=seed)
    tk = InstanceTracker(config=TrackerConfig(
        iou_match=0.25, max_lost_frames=10, min_confirm_frames=2))
    cm = CountingModule()
    cm.enabled = True
    cm.set_method(metodo)
    info = SCENE_INFO[metodo]
    if info.draws_line:
        cm.line = (esc.line_x, 0, esc.line_x, esc.height)
    if info.draws_zone:
        cm.zone = esc.zone
    return esc, tk, cm


print("\n[1] Las escenas producen imagen y detecciones")
for metodo in SCENES:
    esc, _, _ = montar(metodo)
    frame, dets = esc.step(DT)
    check(f"{metodo}: forma del frame", frame.shape, (540, 960, 3))
    check(f"{metodo}: dtype", frame.dtype, np.dtype("uint8"))
    check_true(f"{metodo}: las detecciones llevan la clase",
               all(d.label == PIECE_LABEL for d in dets))
    check_true(f"{metodo}: las cajas caen dentro del frame",
               all(0 <= d.x1 < d.x2 <= 960 and 0 <= d.y1 < d.y2 <= 540
                   for d in dets), str([d.bbox for d in dets][:2]))

print("\n[2] CRUCE DE META: cuenta las que de verdad cruzaron")
esc, tk, cm = montar("line")
cruzaron = set()
previas = {}
for _ in range(900):                       # 30 s
    frame, dets = esc.step(DT)
    for p in esc.pieces:
        ant = previas.get(p.uid)
        if ant is not None and ant >= esc.line_x > p.x:
            cruzaron.add(p.uid)
        previas[p.uid] = p.x
    tr = tk.update(dets)
    cm.update_tracks(tr, expired=tk.last_expired, frame_wh=(960, 540))
print(f"        cruzaron de verdad: {len(cruzaron)}   contadas: {cm.total}")
check("cuenta exactamente las que cruzaron", cm.total, len(cruzaron))
check_true("y cruzo mas de una, para que la prueba signifique algo",
           len(cruzaron) >= 5, str(len(cruzaron)))

print("\n[3] ZONA: cuenta las que de verdad entraron")
esc, tk, cm = montar("zone")
zx1, zy1, zx2, zy2 = esc.zone
entraron = set()
dentro_antes = {}
for _ in range(900):
    frame, dets = esc.step(DT)
    for p in esc.pieces:
        dentro = zx1 <= p.x <= zx2 and zy1 <= p.y <= zy2
        if dentro and dentro_antes.get(p.uid) is False:
            entraron.add(p.uid)
        dentro_antes[p.uid] = dentro
    tr = tk.update(dets)
    cm.update_tracks(tr, expired=tk.last_expired, frame_wh=(960, 540))
print(f"        entraron de verdad: {len(entraron)}   contadas: {cm.total}")
check("cuenta exactamente las que entraron", cm.total, len(entraron))
check_true("y entro mas de una", len(entraron) >= 5, str(len(entraron)))

print("\n[4] EN PANTALLA: sigue a las visibles, no acumula")
esc, tk, cm = montar("screen")
lecturas = []
reales = []
for i in range(900):
    frame, dets = esc.step(DT)
    tr = tk.update(dets)
    cm.update_tracks(tr, expired=tk.last_expired, frame_wh=(960, 540))
    if i > 60:                              # tras el arranque
        lecturas.append(cm.total)
        reales.append(len(dets))
check_true("nunca cuenta mas de las que hay",
           all(l <= max(reales) for l in lecturas),
           f"max leido={max(lecturas)} max real={max(reales)}")
check_true("el numero sube y baja (no acumula)",
           len(set(lecturas)) > 1 and max(lecturas) < 20,
           f"valores vistos: {sorted(set(lecturas))}")
# y el rango esperado veta cuando se sale
cm.expect_min, cm.expect_max = SCENE_INFO["screen"].expect
v = cm.update_tracks(tk.update(esc.step(DT)[1]), frame_wh=(960, 540))
check_true("con rango esperado el veredicto tiene sentido",
           isinstance(v.ok, bool))

print("\n[5] AL APARECER: una por pieza nueva, y solo una")
esc, tk, cm = montar("appear")
vistas = set()
for _ in range(900):
    frame, dets = esc.step(DT)
    for p in esc.pieces:
        if p.alpha >= 0.45:
            vistas.add(p.uid)
    tr = tk.update(dets)
    cm.update_tracks(tr, expired=tk.last_expired, frame_wh=(960, 540))
print(f"        aparecieron de verdad: {len(vistas)}   contadas: {cm.total}")
# El tracker pide ver una instancia dos frames seguidos antes de darla por
# real, asi que las que aparecen en los ultimos frames de la corrida aun no
# estan confirmadas cuando se mira el marcador. Es cola de la medicion, no
# un fallo del conteo: se admite esa diferencia y ninguna mas.
check_true("cuenta las que aparecieron (salvo las que aun no confirman)",
           len(vistas) - 2 <= cm.total <= len(vistas),
           f"{cm.total} contra {len(vistas)}")
check_true("y aparecieron varias", len(vistas) >= 8, str(len(vistas)))

print("\n[6] AL DESAPARECER: una por pieza que se fue")
esc, tk, cm = montar("disappear")
se_fueron = set()
estaban = set()
for _ in range(900):
    frame, dets = esc.step(DT)
    ahora = {p.uid for p in esc.pieces if p.alpha >= 0.45}
    se_fueron |= (estaban - ahora)
    estaban = ahora
    tr = tk.update(dets)
    cm.update_tracks(tr, expired=tk.last_expired, frame_wh=(960, 540))
print(f"        se fueron de verdad: {len(se_fueron)}   contadas: {cm.total}")
# el tracker necesita unos frames para dar por perdida una pieza, asi que
# las ultimas pueden quedar sin contar: se admite esa diferencia, no mas
check_true("cuenta las que se fueron (salvo las que aun no expiran)",
           len(se_fueron) - 2 <= cm.total <= len(se_fueron),
           f"{cm.total} contra {len(se_fueron)}")
check_true("y se fueron varias", len(se_fueron) >= 5, str(len(se_fueron)))

print("\n[7] Escena vacia no cuenta nada")
esc = CountingScene(method="appear", seed=1)
esc.pieces.clear()
tk = InstanceTracker(config=TrackerConfig(min_confirm_frames=2))
cm = CountingModule(); cm.enabled = True; cm.set_method("appear")
for _ in range(60):
    esc._next_spawn = 1e9           # que no genere ninguna
    frame, dets = esc.step(DT)
    tr = tk.update(dets)
    cm.update_tracks(tr, expired=tk.last_expired, frame_wh=(960, 540))
check("sin piezas, conteo cero", cm.total, 0)
check("y sin detecciones", len(dets), 0)

print("\n[8] Cambiar de metodo reinicia la escena")
esc = CountingScene(method="line", seed=3)
for _ in range(120):
    esc.step(DT)
n_antes = len(esc.pieces)
esc.set_method("appear")
check_true("la escena nueva arranca limpia", len(esc.pieces) == 0,
           f"antes {n_antes}, ahora {len(esc.pieces)}")
check("y con el metodo pedido", esc.method, "appear")

print("\n[9] No se dispara ni se queda sin piezas en una corrida larga")
for metodo in SCENES:
    esc, _, _ = montar(metodo)
    maximos = []
    for _ in range(1800):                  # 60 s
        _, dets = esc.step(DT)
        maximos.append(len(dets))
    check_true(f"{metodo}: la escena no se satura", max(maximos) <= 10,
               f"max {max(maximos)} piezas a la vez")
    check_true(f"{metodo}: y no se queda vacia", sum(maximos) > 0)

# hoja de contacto para revisar a ojo
tiras = []
for metodo in SCENES:
    esc, tk, cm = montar(metodo)
    for _ in range(150):
        frame, dets = esc.step(DT)
        tr = tk.update(dets)
        cm.update_tracks(tr, expired=tk.last_expired, frame_wh=(960, 540))
    vis = frame.copy()
    cm.draw(vis)
    for t in tr:
        if t.confirmed:
            x1, y1, x2, y2 = t.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 230, 118), 2)
    cv2.putText(vis, f"{metodo}: {cm.summary()}", (16, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    tiras.append(cv2.resize(vis, (480, 270)))
while len(tiras) % 2:
    tiras.append(np.zeros((270, 480, 3), np.uint8))
filas = [np.hstack(tiras[i:i + 2]) for i in range(0, len(tiras), 2)]
cv2.imwrite(str(SCRATCH / "conteo_escenas.png"), np.vstack(filas))
print(f"\n  hoja de contacto: {SCRATCH / 'conteo_escenas.png'}")

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f_ in fails:
        print("  -", f_)
    sys.exit(1)
print("ESCENAS DE CONTEO: CUENTAN LO QUE DE VERDAD PASA")
