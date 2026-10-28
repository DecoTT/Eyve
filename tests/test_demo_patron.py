# -*- coding: utf-8 -*-
"""
Patron en la pantalla de demo: lo que se vio mal al grabar el video.

Grabando el tutorial del modulo Patron (16-oct-2026) salieron dos fallos
que el modulo, probado por su cuenta, no tenia:

  - la etiqueta de Patron se quedaba en la raya despues de limpiar la tela
    o de cambiar de material;
  - tras cambiar de estampado, Patron no marcaba casi nada.  No era el
    modulo: la calibracion se comia el ultimo frame de la tela ANTERIOR
    que aun tenia guardado la fuente, y con ese frame el umbral salia de
    2 a 9.  En puntos no se marcaba ni un offset.
"""
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import customtkinter as ctk
import numpy as np
from eyve.i18n import t, set_language
from eyve.ui import theme as T
T.apply("dark")
set_language("es")

from eyve.demo.source import SyntheticSource
from eyve.demo.textile import TextilePattern, YOLO_CLASSES

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


class FakeProject:
    classes = []
    ok_classes = []
    nok_classes = list(YOLO_CLASSES)
    active_model = None

    class paths:
        best_model = types.SimpleNamespace(exists=lambda: False)


class FakeApp(ctk.CTk):
    # Patron es Pro desde 2.1.6: estas pruebas son de lo que hace Patron,
    # asi que corren con licencia Pro.  Free esta en test_patron_pro.py.
    license = types.SimpleNamespace(permite_checks_avanzados=lambda: True)

    def get_project(self):
        return FakeProject()


app = FakeApp()
app.geometry("1600x950")

from eyve.ui.screens import demo_screen as DS
DS.DemoScreen._load_model = lambda self: None

scr = DS.DemoScreen(app, app)
scr.pack(fill="both", expand=True)
app.update()
# El hilo de la fuente NO se arranca: asi la tela no se mueve sola y cada
# frame es exactamente el que la prueba decide.  `start()` deja guardado un
# primer frame, y eso se imita a mano.
scr._source._frame = scr._source.pattern.frame()


def calibrar(frames=10):
    """Lo que hace el bucle de la pantalla mientras _cal_left > 0."""
    while scr._cal_left > 0:
        scr._patternmod.calibrate(scr._source.read())
        scr._cal_left -= 1
        scr._pattern.advance(0.07)
        scr._source._frame = scr._pattern.frame()


print("\n[1] La etiqueta de Patron se repinta despues de limpiar")
scr._start_calibration()
calibrar()
scr._inspect(scr._source.read())
antes = scr._pat_lbl.cget("text")
check("con la tela limpia dice que no hay anomalias", antes, "sin anomalías")
scr._on_clear_click()
check("limpiar la deja en la raya", scr._pat_lbl.cget("text"), "—")
scr._inspect(scr._source.read())
# El resumen es el MISMO que antes de limpiar, y por eso se quedaba pegada:
# reset_demo() pintaba la raya pero no olvidaba el ultimo resumen.
check("y el siguiente frame la vuelve a llenar",
      scr._pat_lbl.cget("text"), "sin anomalías")

print("\n[2] Cambiar de estampado calibra con la tela NUEVA")
# La fuente guarda un frame de diamantes, como en la pantalla real justo
# antes de elegir otro estampado.
check("arranca en diamantes", scr._pattern.motif, "diamantes")
scr._source._frame = scr._pattern.frame()
scr._on_motif(t("motif_puntos"))
check("estampado -> puntos", scr._pattern.motif, "puntos")
check_true("el primer frame que lee la calibracion ya es de puntos",
           np.array_equal(scr._source.read(), scr._pattern.frame()))
calibrar()
piso = scr._patternmod.baseline
check_true("el piso es el de los puntos, no el de los diamantes",
           piso is not None and piso < 1.5, f"piso={piso:.2f}")
# Y lo que se ve en pantalla: un offset en puntos se marca.
fx, fy = scr._pattern.screen_to_fabric(480, 270)
scr._pattern.misregister(fx, fy, size=170)
regs = scr._patternmod.analyze(scr._pattern.frame())
check_true("un offset en puntos se marca", len(regs) >= 1, str(regs))

print("\n[3] Empezar de cero calibra SIN los defectos de antes")
scr._source._frame = scr._pattern.frame()      # todavia con el offset
scr.empezar_de_cero()
check_true("el primer frame de la calibracion ya esta limpio",
           np.array_equal(scr._source.read(), scr._pattern.frame()))
calibrar()
check_true("el piso no se inflo con el defecto viejo",
           scr._patternmod.baseline < 1.5,
           f"piso={scr._patternmod.baseline:.2f}")

print("\n[4] El hilo de la fuente no pisa un refresh() con un frame viejo")
# El hilo renderiza FUERA del candado.  Si refresh() entra mientras tanto,
# el frame que el hilo termina despues es de la tela vieja y no debe
# guardarse encima del nuevo.


class TelaDePrueba:
    speed = 0.0

    def __init__(self):
        self.n = 0
        self.src = None

    def advance(self, dt):
        pass

    def frame(self):
        self.n += 1
        if self.n == 2:
            # el hilo esta a medio renderizar y entra un refresh()
            self.src.refresh()
            self.src._running = False      # una vuelta y basta
            return np.full((4, 4, 3), 1, np.uint8)       # viejo
        return np.full((4, 4, 3), 2 if self.n == 3 else 0, np.uint8)


tela = TelaDePrueba()
src = SyntheticSource(tela, fps=1000.0)
tela.src = src
src._running = True
src._loop()            # en este mismo hilo: determinista
check("se queda el frame de refresh(), no el viejo",
      int(src.read()[0, 0, 0]), 2)

scr.on_close()
app.destroy()

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f_ in fails:
        print("  -", f_)
    sys.exit(1)
print("PATRON EN LA DEMO: TODO PASO")
