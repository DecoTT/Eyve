# -*- coding: utf-8 -*-
"""
Las anomalias son defectos: se confirman como instancias y se cuentan.

Dos problemas vistos en la pantalla real:

  1. A veces marcaba por un instante una zona que estaba bien. El modulo
     analizaba cada frame por su cuenta y dibujaba el resultado crudo.
  2. Una anomalia no contaba, aunque es un defecto.

Los dos se arreglan igual: las regiones del Patron entran al MISMO tracker
con su propia clase, asi que heredan la confirmacion por frames (mata el
parpadeo) y el modulo de conteo las cuenta.

Cada caso va con su contrario: lo que aparece un solo frame NO debe
confirmarse, y lo que persiste SI.
"""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SCRATCH = Path(tempfile.gettempdir()) / "eyve_tests"
SCRATCH.mkdir(parents=True, exist_ok=True)

from eyve.inference.tracker import InstanceTracker, RawDetection, TrackerConfig
from eyve.modules import CountingModule, PatternModule

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


def modulo_con(regiones):
    """PatternModule con regiones ya puestas, sin analizar imagen."""
    m = PatternModule()
    m.enabled = True
    m._regions = list(regiones)
    return m


print("\n[1] as_detections convierte regiones en detecciones")
m = modulo_con([(100, 100, 200, 180), (400, 300, 500, 380)])
dets = m.as_detections()
check("dos regiones -> dos detecciones", len(dets), 2)
check("con la clase de anomalia", dets[0].label, PatternModule.ANOMALY_LABEL)
check("conserva la caja", dets[0].bbox, (100, 100, 200, 180))
check("sin regiones, sin detecciones", modulo_con([]).as_detections(), [])

print("\n[2] Lo que YOLO ya nombro, el Patron no lo repite")
# si no se filtrara, el mismo rayon se contaria dos veces: una por motor
yolo = [(105, 105, 195, 175)]
check("region que coincide con una caja de YOLO se descarta",
      len(m.as_detections(named=yolo)), 1)
check("la que queda es la otra",
      m.as_detections(named=yolo)[0].bbox, (400, 300, 500, 380))
# una caja lejana no suprime nada
check("una caja de YOLO lejana no suprime",
      len(m.as_detections(named=[(800, 50, 900, 120)])), 2)
# y un solape pequeno tampoco: son defectos distintos que se tocan
check("un solape pequeno no suprime",
      len(m.as_detections(named=[(190, 170, 280, 250)])), 2)

print("\n[3] PARPADEO: una region de un solo frame no se confirma")
tk = InstanceTracker(config=TrackerConfig(min_confirm_frames=2,
                                          max_lost_frames=4))
cm = CountingModule(); cm.enabled = True; cm.set_method("appear")
# frame 1: aparece algo
tr = tk.update(modulo_con([(100, 100, 200, 180)]).as_detections())
cm.update_tracks(tr, expired=tk.last_expired, frame_wh=(960, 540))
check_true("tras un frame no hay nada confirmado",
           not any(t.confirmed for t in tr))
check("y no se ha contado nada", cm.total, 0)
# frames 2..6: ya no esta
for _ in range(6):
    tr = tk.update([])
    cm.update_tracks(tr, expired=tk.last_expired, frame_wh=(960, 540))
check("el parpadeo nunca se conto", cm.total, 0)

print("\n[4] PERSISTENTE: una anomalia que se queda SI cuenta")
tk = InstanceTracker(config=TrackerConfig(min_confirm_frames=2,
                                          max_lost_frames=4))
cm = CountingModule(); cm.enabled = True; cm.set_method("appear")
region = [(100, 100, 200, 180)]
for i in range(5):
    tr = tk.update(modulo_con(region).as_detections())
    cm.update_tracks(tr, expired=tk.last_expired, frame_wh=(960, 540))
    if i == 0:
        check_true("al primer frame todavia no", cm.total == 0)
check("la anomalia persistente cuenta una vez", cm.total, 1)
check("y con su propia clase",
      list(cm.counts), [PatternModule.ANOMALY_LABEL])
check_true("tiene una instancia confirmada",
           any(t.confirmed for t in tr))

print("\n[5] No cuenta dos veces mientras sigue ahi")
for _ in range(8):
    tr = tk.update(modulo_con(region).as_detections())
    cm.update_tracks(tr, expired=tk.last_expired, frame_wh=(960, 540))
check("sigue siendo una", cm.total, 1)

print("\n[6] Un defecto nombrado y una anomalia se cuentan por separado")
tk = InstanceTracker(config=TrackerConfig(min_confirm_frames=2,
                                          max_lost_frames=4))
cm = CountingModule(); cm.enabled = True; cm.set_method("appear")
rayon = RawDetection("rayon", 0.9, 600, 200, 700, 300)
for _ in range(5):
    dets = [rayon] + modulo_con(region).as_detections(named=[rayon.bbox])
    tr = tk.update(dets)
    cm.update_tracks(tr, expired=tk.last_expired, frame_wh=(960, 540))
check("dos defectos distintos, dos cuentas", cm.total, 2)
check("uno de cada clase", sorted(cm.counts),
      sorted(["rayon", PatternModule.ANOMALY_LABEL]))

print("\n[7] El mismo defecto visto por los dos motores cuenta UNA vez")
tk = InstanceTracker(config=TrackerConfig(min_confirm_frames=2,
                                          max_lost_frames=4))
cm = CountingModule(); cm.enabled = True; cm.set_method("appear")
# el Patron ve lo mismo que YOLO: su region cae encima del rayon
misma = [(605, 205, 695, 295)]
for _ in range(5):
    dets = [rayon] + modulo_con(misma).as_detections(named=[rayon.bbox])
    tr = tk.update(dets)
    cm.update_tracks(tr, expired=tk.last_expired, frame_wh=(960, 540))
check("un solo defecto, una sola cuenta", cm.total, 1)
check("y lo nombra YOLO, no el Patron", list(cm.counts), ["rayon"])

print("\n[8] Y la prueba de que el filtro SIRVE: sin el, se contaria doble")
tk = InstanceTracker(config=TrackerConfig(min_confirm_frames=2,
                                          max_lost_frames=4))
cm = CountingModule(); cm.enabled = True; cm.set_method("appear")
for _ in range(5):
    # mismas regiones, pero SIN decirle que YOLO ya las nombro
    dets = [rayon] + modulo_con(misma).as_detections()
    tr = tk.update(dets)
    cm.update_tracks(tr, expired=tk.last_expired, frame_wh=(960, 540))
check("sin el filtro el mismo defecto cuenta dos veces", cm.total, 2)

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f_ in fails:
        print("  -", f_)
    sys.exit(1)
print("ANOMALIAS COMO INSTANCIAS: TODO PASO")
