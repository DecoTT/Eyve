"""
Prueba del modulo de conteo con tracks sinteticos.

Cada metodo se prueba con un caso que debe contar Y un caso que NO debe
contar, para que la prueba falle si la logica esta mal (una prueba que pasa
con cualquier implementacion no prueba nada).
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

from eyve.modules.counting_module import CountingModule
from eyve.inference.tracker import (InstanceTracker, RawDetection,
                                    TrackedInstance, TrackerConfig)

FW, FH = 640, 480
fails = []


def check(name, got, want):
    ok = got == want
    print(f"  {'OK  ' if ok else 'FALLA'}  {name}: got={got} want={want}")
    if not ok:
        fails.append(name)


def mk_track(tid, label, cx, cy, w=40, h=40, confirmed=True):
    """TrackedInstance falso sin pasar por el tracker."""
    det = RawDetection(label, 0.9, cx - w // 2, cy - h // 2,
                       cx + w // 2, cy + h // 2)
    tr = TrackedInstance(det)
    tr.id = tid
    tr.confirmed = confirmed
    return tr


def move(tr, cx, cy):
    w, h = tr.w, tr.h
    tr.bbox = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
    return tr


# ───────────────────────────────────────────────────────────────────────────
print("\n[1] SCREEN — cuenta lo visible, no acumula")
m = CountingModule()
m.enabled = True
m.set_method("screen")

a = mk_track(1, "pieza", 100, 100)
b = mk_track(2, "pieza", 200, 100)
c = mk_track(3, "pieza", 300, 100, confirmed=False)   # sin confirmar: no cuenta
m.update_tracks([a, b, c], frame_wh=(FW, FH))
check("3 tracks, 1 sin confirmar", m.total, 2)

m.update_tracks([a], frame_wh=(FW, FH))
check("baja a 1 (no acumula)", m.total, 1)

m.update_tracks([], frame_wh=(FW, FH))
check("encuadre vacio", m.total, 0)

# clase objetivo filtra
m.reset()
m.target_class = "pieza"
d = mk_track(4, "otra", 400, 100)
m.update_tracks([a, b, d], frame_wh=(FW, FH))
check("filtro por clase ignora 'otra'", m.total, 2)

# rango esperado veta el frame
m.target_class = None
m.expect_min, m.expect_max = 3, 5
v = m.update_tracks([a, b], frame_wh=(FW, FH))
check("2 visibles con esperado 3-5 -> veta", v.ok, False)
v = m.update_tracks([a, b, d], frame_wh=(FW, FH))
check("3 visibles con esperado 3-5 -> pasa", v.ok, True)

# ───────────────────────────────────────────────────────────────────────────
print("\n[2] LINE — cruce con sentido, una vez por instancia")
m = CountingModule()
m.enabled = True
m.set_method("line")
m.line = (320, 0, 320, 480)          # meta vertical en el centro

t1 = mk_track(1, "pieza", 280, 240)
m.update_tracks([t1], frame_wh=(FW, FH))
check("primer frame (sin prev) no cuenta", m.total, 0)

move(t1, 360, 240)                   # cruza izq -> der
m.update_tracks([t1], frame_wh=(FW, FH))
check("cruza la meta", m.total, 1)

move(t1, 440, 240)                   # sigue avanzando
m.update_tracks([t1], frame_wh=(FW, FH))
check("no recuenta al avanzar", m.total, 1)

move(t1, 280, 240)                   # regresa cruzando de vuelta
m.update_tracks([t1], frame_wh=(FW, FH))
check("misma instancia no cuenta dos veces", m.total, 1)

# una instancia que nunca cruza
m.reset()
t2 = mk_track(2, "pieza", 100, 240)
m.update_tracks([t2], frame_wh=(FW, FH))
move(t2, 200, 240)
m.update_tracks([t2], frame_wh=(FW, FH))
check("se mueve sin cruzar", m.total, 0)

# sentidos opuestos
m.reset()
m.direction = "both"
up = mk_track(10, "pieza", 280, 100)
dn = mk_track(11, "pieza", 360, 300)
m.update_tracks([up, dn], frame_wh=(FW, FH))
move(up, 360, 100)                   # izq -> der
move(dn, 280, 300)                   # der -> izq
m.update_tracks([up, dn], frame_wh=(FW, FH))
check("ambos sentidos: total", m.total, 2)
check("un cruce en cada sentido",
      sorted(m.dir_counts.values()), [1, 1])

# con un solo sentido, solo ese alimenta el total
m.reset()
m.direction = "fwd"
up = mk_track(20, "pieza", 280, 100)
dn = mk_track(21, "pieza", 360, 300)
m.update_tracks([up, dn], frame_wh=(FW, FH))
move(up, 360, 100)
move(dn, 280, 300)
m.update_tracks([up, dn], frame_wh=(FW, FH))
check("sentido unico: total cuenta 1 de 2", m.total, 1)
check("el otro sentido queda registrado",
      m.dir_counts["fwd"] + m.dir_counts["rev"], 2)

# ───────────────────────────────────────────────────────────────────────────
print("\n[3] ZONE — entrada / salida / ocupacion")
m = CountingModule()
m.enabled = True
m.set_method("zone")
m.zone = (200, 150, 440, 330)
m.zone_trigger = "enter"

z = mk_track(1, "pieza", 100, 240)          # fuera
m.update_tracks([z], frame_wh=(FW, FH))
check("primer frame fuera", m.total, 0)
check("ocupacion 0", m.occupancy, 0)

move(z, 320, 240)                            # entra
m.update_tracks([z], frame_wh=(FW, FH))
check("entra a la zona", m.total, 1)
check("ocupacion 1", m.occupancy, 1)

move(z, 400, 240)                            # se mueve dentro
m.update_tracks([z], frame_wh=(FW, FH))
check("moverse dentro no recuenta", m.total, 1)

move(z, 600, 240)                            # sale
m.update_tracks([z], frame_wh=(FW, FH))
check("salir no cuenta (trigger=enter)", m.total, 1)
check("ocupacion vuelve a 0", m.occupancy, 0)

# trigger = exit
m.reset()
m.zone_trigger = "exit"
z = mk_track(5, "pieza", 100, 240)
m.update_tracks([z], frame_wh=(FW, FH))
move(z, 320, 240)
m.update_tracks([z], frame_wh=(FW, FH))
check("trigger=exit: entrar no cuenta", m.total, 0)
move(z, 600, 240)
m.update_tracks([z], frame_wh=(FW, FH))
check("trigger=exit: salir cuenta", m.total, 1)

# trigger = both
m.reset()
m.zone_trigger = "both"
z = mk_track(7, "pieza", 100, 240)
m.update_tracks([z], frame_wh=(FW, FH))
move(z, 320, 240); m.update_tracks([z], frame_wh=(FW, FH))
move(z, 600, 240); m.update_tracks([z], frame_wh=(FW, FH))
check("trigger=both: entrada + salida", m.total, 2)

# ocupacion con rango esperado
m.reset()
m.expect_max = 1
o1 = mk_track(30, "pieza", 300, 240)
o2 = mk_track(31, "pieza", 350, 240)
m.update_tracks([o1, o2], frame_wh=(FW, FH))   # primer frame: sin estado previo
v = m.update_tracks([o1, o2], frame_wh=(FW, FH))
check("ocupacion 2 con maximo 1 -> veta", v.ok, False)

# ───────────────────────────────────────────────────────────────────────────
print("\n[4] APPEAR — cada instancia nueva una vez")
m = CountingModule()
m.enabled = True
m.set_method("appear")

p1 = mk_track(1, "pieza", 100, 100)
m.update_tracks([p1], frame_wh=(FW, FH))
check("primera instancia", m.total, 1)
m.update_tracks([p1], frame_wh=(FW, FH))
check("la misma no recuenta", m.total, 1)

p2 = mk_track(2, "pieza", 200, 100)
m.update_tracks([p1, p2], frame_wh=(FW, FH))
check("instancia nueva suma", m.total, 2)

m.update_tracks([], frame_wh=(FW, FH))
check("encuadre vacio conserva el acumulado", m.total, 2)

p3 = mk_track(3, "pieza", 300, 100, confirmed=False)
m.update_tracks([p3], frame_wh=(FW, FH))
check("sin confirmar no cuenta", m.total, 2)
p3.confirmed = True
m.update_tracks([p3], frame_wh=(FW, FH))
check("al confirmarse si cuenta", m.total, 3)

# ───────────────────────────────────────────────────────────────────────────
print("\n[5] DISAPPEAR — instancias purgadas, con filtro de borde")
m = CountingModule()
m.enabled = True
m.set_method("disappear")
m.edge = "any"

gone = mk_track(1, "pieza", 320, 240)
m.update_tracks([], expired=[gone], frame_wh=(FW, FH))
check("una expirada cuenta", m.total, 1)
m.update_tracks([], expired=[gone], frame_wh=(FW, FH))
check("no cuenta dos veces", m.total, 1)

flicker = mk_track(2, "pieza", 320, 240, confirmed=False)
m.update_tracks([], expired=[flicker], frame_wh=(FW, FH))
check("expirada sin confirmar = parpadeo, no cuenta", m.total, 1)

# filtro de borde
m.reset()
m.edge = "right"
centro = mk_track(10, "pieza", 320, 240)
m.update_tracks([], expired=[centro], frame_wh=(FW, FH))
check("borde=derecho: desaparecer en el centro no cuenta", m.total, 0)

derecha = mk_track(11, "pieza", FW - 10, 240)
m.update_tracks([], expired=[derecha], frame_wh=(FW, FH))
check("borde=derecho: salir por la derecha si cuenta", m.total, 1)

izquierda = mk_track(12, "pieza", 10, 240)
m.update_tracks([], expired=[izquierda], frame_wh=(FW, FH))
check("borde=derecho: salir por la izquierda no cuenta", m.total, 1)

m.reset()
m.edge = "left"
m.update_tracks([], expired=[mk_track(20, "pieza", 10, 240)], frame_wh=(FW, FH))
check("borde=izquierdo cuenta por la izquierda", m.total, 1)

# ───────────────────────────────────────────────────────────────────────────
print("\n[6] Cambio de metodo limpia el acumulado")
m = CountingModule()
m.enabled = True
m.set_method("appear")
m.update_tracks([mk_track(1, "pieza", 100, 100)], frame_wh=(FW, FH))
check("acumulado antes del cambio", m.total, 1)
m.set_method("line")
check("cambiar de metodo resetea", m.total, 0)

print("\n[7] Modulo apagado no cuenta nada")
m = CountingModule()
m.set_method("appear")
m.enabled = False
m.update_tracks([mk_track(1, "pieza", 100, 100)], frame_wh=(FW, FH))
check("enabled=False", m.total, 0)

# ───────────────────────────────────────────────────────────────────────────
print("\n[8] TrackerConfig configurable + last_expired")
cfg = TrackerConfig(iou_match=0.3, max_lost_frames=2, min_confirm_frames=1)
tk = InstanceTracker(config=cfg)
check("config aplicada", tk.config.max_lost_frames, 2)

d = RawDetection("pieza", 0.9, 100, 100, 140, 140)
tracks = tk.update([d])
check("confirma al primer frame (min_confirm=1)", tracks[0].confirmed, True)

for i in range(3):                   # 3 frames sin ver > max_lost_frames=2
    tk.update([])
check("purgada tras exceder la tolerancia", len(tk.get_all()), 0)
check("last_expired la entrega una vez", len(tk.last_expired), 1)

# clamp: la UI no puede romper el tracker
bad = TrackerConfig(iou_match=5.0, max_lost_frames=-3,
                    min_confirm_frames=0, process_conf_min=9.0).clamped()
check("clamp iou", bad.iou_match <= 0.95, True)
check("clamp max_lost", bad.max_lost_frames >= 0, True)
check("clamp min_confirm", bad.min_confirm_frames >= 1, True)
check("clamp conf", bad.process_conf_min <= 0.99, True)

# configure() en caliente
tk.configure(max_lost_frames=30)
check("configure en caliente", tk.config.max_lost_frames, 30)
tk.configure(max_lost_frames=None)   # None = no tocar
check("None no sobreescribe", tk.config.max_lost_frames, 30)

# tolerancia alta: la instancia sobrevive el parpadeo y conserva su ID
tk2 = InstanceTracker(config=TrackerConfig(max_lost_frames=10,
                                           min_confirm_frames=1))
d1 = RawDetection("pieza", 0.9, 100, 100, 140, 140)
first = tk2.update([d1])[0].id
for _ in range(5):
    tk2.update([])                   # parpadeo de 5 frames
again = tk2.update([d1])
check("conserva el ID tras 5 frames perdida",
      again[0].id if again else None, first)

# ───────────────────────────────────────────────────────────────────────────
print("\n[9] draw() no explota en ningun metodo")
import numpy as np
for meth in CountingModule.METHODS:
    mm = CountingModule()
    mm.enabled = True
    mm.set_method(meth)
    mm.line = (10, 10, 100, 100)
    mm.zone = (20, 20, 120, 120)
    for d_ in CountingModule.DIRECTIONS:
        mm.direction = d_
        for e_ in CountingModule.EDGES:
            mm.edge = e_
            canvas = np.zeros((FH, FW, 3), dtype=np.uint8)
            mm.draw(canvas)
    # sin geometria tampoco debe explotar
    mm.clear_geometry()
    mm.draw(np.zeros((FH, FW, 3), dtype=np.uint8))
print("  OK    draw() en los 5 metodos x 3 sentidos x 5 bordes")

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("TODAS LAS PRUEBAS PASARON")
