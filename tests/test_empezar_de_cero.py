# -*- coding: utf-8 -*-
"""
El boton de "empezar de cero" de la demo textil.

Limpiar es lo facil.  Lo que se comprueba aqui es que funciona DESDE
CUALQUIER ESTADO, y sobre todo el que deja basura si se hace mal: pulsarlo
a mitad de un trazo.  Si no se cierra el trazo y se olvida el ultimo
punto, el siguiente movimiento del raton pinta una raya desde donde estaba
la mano antes del reinicio.

Cada comprobacion lleva su condicion previa afirmada: si la tela no
estuviera sucia ANTES, "la tela queda limpia" no probaria nada.
"""
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("YOLO_OFFLINE", "1")

fails = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


# ── neutralizar camara, modelo, red y repintado ───────────────────────────
from eyve.inference import detector as _det
_det.VideoSource.start = lambda self: False
_det.VideoSource.stop = lambda self: None
_det.VideoSource.read = lambda self: None

from eyve.ui.screens import production_screen as _ps
_ps.ProductionScreen._try_autoload_model = lambda self: None
from eyve.ui.screens import demo_screen as _ds
_ds.DemoScreen._load_model = lambda self: None
_ds.DemoScreen._paint = lambda self, canvas, frame, attr: (960, 540, 960, 540, 0, 0)
from eyve.ui.screens import counting_demo_screen as _cds
_cds.CountingDemoScreen._pintar = lambda self, frame: None

from eyve.core import updater as _U
_U.check_async = lambda cb: None

from eyve.i18n import t
from eyve.modules import CountingModule
from eyve.ui.app import EyveApp


def settle(app, veces=3):
    for _ in range(veces):
        app.update()
        app.update_idletasks()


app = EyveApp()
app.geometry("1200x800+80+40")
app.navigate("nav_demo")
settle(app)
demo = app._screens["nav_demo"]
demo._running = False          # el bucle se mueve a mano en esta prueba
demo._cal_left = 0
# mapeo lienzo↔tela 1:1, para poder simular el raton
demo._view = (960, 540, 960, 540, 0, 0)


def una_vuelta_del_bucle():
    demo._running = True
    demo._loop()
    demo._running = False
    settle(app, 1)


def raton(x, y):
    return types.SimpleNamespace(x=x, y=y)


print("\n[1] El boton existe, es el mas grande y esta cableado")
check("existe el boton", hasattr(demo, "_restart_btn"))
alto_restart = demo._restart_btn.cget("height")
alto_otros = demo._pause_btn.cget("height")
check("es mas alto que los controles normales", alto_restart > alto_otros,
      f"{alto_restart} > {alto_otros}")
check("el texto es inconfundible", t("demo_restart") == t("demo_restart").upper()
      and len(t("demo_restart")) > 5, t("demo_restart"))

print("\n[2] Desde el peor estado: sucio, pausado y con el automatico apagado")
# dejar la demo hecha un desastre
demo._pattern.streak(100, 100, length=200, thickness=7)
demo._pattern.blob(300, 250, size=90)
demo._auto = False
demo._refresh_auto_label()
demo._source.set_paused(True)
demo._on_count_method(t("count_m_" + CountingModule.METHODS[4]))
demo._auto_placed = 3
# y con el Patron ya calibrado, para que el recalibrado se note
frame = demo._pattern.frame()
for _ in range(10):
    demo._patternmod.calibrate(frame)

# ── las condiciones previas, afirmadas: sin esto no se prueba nada ──────
check("PREVIO: la tela esta sucia", len(demo._pattern.defects) > 0,
      f"{len(demo._pattern.defects)} defectos")
check("PREVIO: el automatico esta apagado", not demo._auto)
check("PREVIO: la tela esta pausada", demo._source.paused)
check("PREVIO: el metodo de conteo no es el inicial",
      demo._counting.method != _ds._METODO_INICIAL, demo._counting.method)
check("PREVIO: el Patron esta calibrado", demo._patternmod.calibrated)

demo._restart_btn.invoke()          # el boton de verdad, no el metodo
settle(app)

check("la tela queda limpia", len(demo._pattern.defects) == 0,
      f"{len(demo._pattern.defects)} defectos")
check("el automatico queda encendido", demo._auto)
check("la tela vuelve a correr", not demo._source.paused)
check("el boton de pausa vuelve a decir Pausar",
      demo._pause_btn.cget("text") == t("demo_pause"),
      demo._pause_btn.cget("text"))
check("el metodo de conteo vuelve al inicial",
      demo._counting.method == _ds._METODO_INICIAL, demo._counting.method)
check("el contador queda a cero", demo._counting.total == 0,
      str(demo._counting.total))
check("el tracker queda vacio", len(demo._tracker.get_all()) == 0,
      str(len(demo._tracker.get_all())))
check("los defectos automaticos vuelven a contarse desde cero",
      demo._auto_placed == 0, str(demo._auto_placed))
check("el Patron queda SIN calibrar, esperando material bueno",
      not demo._patternmod.calibrated and demo._cal_left > 0,
      f"calibrado={demo._patternmod.calibrated} pendientes={demo._cal_left}")
check("avisa de lo que hizo", demo._hint.cget("text") == t("demo_restart_done"),
      demo._hint.cget("text"))
# Visto en la app: con 2 s, el primer defecto automatico cae antes de que
# quien atiende alcance a enseñar la tela limpia.
import time as _t
margen = demo._auto_next - _t.time()
check("deja unos segundos de tela limpia antes de que vuelva el automatico",
      margen >= 5.0, f"{margen:.1f} s")

print("\n[3] Y la calibracion LLEGA a completarse, no se queda a medias")
# Un recalibrado que nunca termina deja al modulo Patron mudo todo el dia,
# que es peor que no recalibrar.
# El tope no es decorativo: si el recalibrado pidiera un numero absurdo de
# frames, un bucle sin tope dejaria la PRUEBA colgada en vez de fallar, y
# una prueba colgada no informa de nada.
vueltas = 0
while demo._cal_left > 0 and vueltas < 40:
    una_vuelta_del_bucle()
    vueltas += 1
check("la calibracion se completa sola", demo._cal_left == 0,
      f"quedan {demo._cal_left} tras {vueltas} vueltas")
check("y el Patron vuelve a estar calibrado", demo._patternmod.calibrated)

print("\n[4] Pulsado MIENTRAS alguien dibuja, con la tela ya pausada")
# La tela se pausa ANTES a proposito.  Si se empieza a dibujar con la tela
# corriendo, `_pausa_previa` ya vale False y la comprobacion de "olvida
# que estaba pausada" pasaria sola, sin probar nada — que es justo lo que
# pasaba antes de escribirlo asi.
demo._toggle_pause()
check("PREVIO: la tela esta pausada antes de dibujar", demo._source.paused)
demo._on_press(raton(400, 300))
demo._on_drag(raton(430, 320))
check("PREVIO: esta dibujando", demo._drawing)
check("PREVIO: hay un punto anterior guardado", demo._last_pt is not None)
check("PREVIO: se recordo que la tela ya estaba pausada", demo._pausa_previa)
sucios_antes = len(demo._pattern.defects)
check("PREVIO: el trazo pinto algo", sucios_antes > 0, str(sucios_antes))

demo._restart_btn.invoke()
settle(app)
check("deja de estar dibujando", not demo._drawing)
check("olvida el punto anterior", demo._last_pt is None)
check("olvida que la tela estaba pausada", not demo._pausa_previa)
check("la tela vuelve a correr", not demo._source.paused)
check("la tela quedo limpia", len(demo._pattern.defects) == 0)

# ── lo que de verdad importa: mover el raton despues no pinta nada ──────
demo._on_drag(raton(800, 500))
demo._on_drag(raton(850, 520))
check("mover el raton despues NO pinta una raya",
      len(demo._pattern.defects) == 0,
      f"{len(demo._pattern.defects)} defectos aparecidos de la nada")
# y soltar el raton no vuelve a pausar la tela
demo._on_release(None)
check("soltar el raton no vuelve a pausar", not demo._source.paused)

print("\n[5] Pulsarlo dos veces seguidas no rompe nada")
demo._restart_btn.invoke()
demo._restart_btn.invoke()
settle(app)
check("sigue limpio y en automatico",
      len(demo._pattern.defects) == 0 and demo._auto
      and not demo._source.paused)
check("y sigue esperando calibracion", demo._cal_left > 0, str(demo._cal_left))

print("\n[6] Desde el estado recien abierto tampoco estorba")
vueltas = 0
while demo._cal_left > 0 and vueltas < 40:
    una_vuelta_del_bucle()
    vueltas += 1
check("PREVIO: calibrado y limpio",
      demo._patternmod.calibrated and not demo._pattern.defects)
demo._restart_btn.invoke()
settle(app)
check("vuelve a dejarlo listo", demo._auto and not demo._pattern.defects
      and demo._counting.total == 0)

# ── cierre ────────────────────────────────────────────────────────────────
try:
    app._teardown_all_screens()
    app.update()
    app.destroy()
    print("\n  OK    cierre limpio")
except Exception as e:
    fails.append(f"cierre: {e}")

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("EMPEZAR DE CERO: DEJA LA DEMO COMO RECIEN ABIERTA, DESDE CUALQUIER ESTADO")
