# -*- coding: utf-8 -*-
"""
La pantalla de demo con la tela realista y la mascara aprendida.

Recorre lo que haria quien graba el video: prender la tela realista, aprender
la mascara con tela buena, calibrarla, revisarla en su ventana y aprobarla, y
ver que una doble impresion sale marcada como "mascara".  Tambien que cambiar
de tela tira la mascara (era de la tela anterior) y que rechazar la deja sin
mascara.
"""
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import customtkinter as ctk
from eyve.i18n import set_language
from eyve.ui import theme as T
T.apply("dark")
set_language("es")

from eyve.demo.tela_real import FALLAS_REALES, TelaRealista
from eyve.demo.textile import TextilePattern, YOLO_CLASSES

fails = []


def check(name, got, want):
    ok = got == want
    print(("  OK    " if ok else "  FALLA  ") + f"{name}: got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


def check_true(name, cond, extra=""):
    print(("  OK    " if cond else "  FALLA  ") + f"{name} {extra}")
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


def esperar(fase, tope=60.0):
    t0 = time.time()
    while scr._masc_fase != fase and time.time() - t0 < tope:
        scr._mask_poll()
        app.update()
        time.sleep(0.05)
    return scr._masc_fase


def alimentar(n):
    for _ in range(n):
        f = scr._pattern.frame()
        scr._pattern.advance(1 / 30.0)
        scr._mask_tick(f)


print("\n[1] Tela realista")
check("arranca con la demo de siempre", type(scr._pattern), TextilePattern)
scr._real_sw.select()
scr._on_realista()
app.update()
check("el interruptor cambia la tela", type(scr._pattern), TelaRealista)
check_true("se ven los botones de fallas de linea",
           scr._faults_real.winfo_manager() == "pack")
check_true("y se esconden los de siempre", scr._faults_basic.winfo_manager() == "")
scr._source.set_paused(True)
scr._place_fault("doble_impresion")
check("un boton de falla real pone esa falla",
      getattr(scr._pattern.defects[-1], "tipo", None), "doble_impresion")
scr.reset_demo()

print("\n[2] Aprender, calibrar, revisar y aprobar")
scr._auto = True
scr._mask_learn_click()
check("aprender calla el automatico", scr._auto, False)
check("fase aprender", scr._masc_fase, "aprender")
alimentar(DS._MASK_N * DS._MASK_CADA)
check("con los cuadros juntos pasa a calibrar", esperar("calibrar"), "calibrar")
alimentar(DS._CAL_N * DS._CAL_CADA)
check("y despues a revisar", esperar("revisar"), "revisar")
check_true("se abrio la ventana de revision", scr._mask_top is not None and
           scr._mask_top.winfo_exists())
check("aun sin aprobar no juzga", scr._mascara.lista, False)
scr._mask_decidir(True)
check("aprobada: activa", scr._masc_fase, "activa")
check_true("la mascara queda lista", scr._mascara.lista)
check("el automatico vuelve como estaba", scr._auto, True)

print("\n[3] Una doble impresion sale como 'mascara'")
scr._auto = False
p = scr._pattern
fx = int(p.offset + p.width + 300)
p.doble_impresion(fx, 270, largo=500, fuerza=0.6)
etiquetas = set()
for _ in range(220):
    p.advance(1 / 16.0)
    scr._inspect(p.frame())
    etiquetas |= {tr.label for tr in scr._tracker.get_all() if tr.confirmed}
    if DS._MASK_LABEL in etiquetas:
        break
check_true("la mascara marco la doble impresion", DS._MASK_LABEL in etiquetas,
           str(etiquetas))

print("\n[4] Cambiar de tela tira la mascara")
scr._rebuild_fabric("puntos", "sarga")
check("sin mascara", scr._mascara, None)
check("fase off", scr._masc_fase, "off")

print("\n[5] Rechazar deja sin mascara")
scr._mask_learn_click()
alimentar(DS._MASK_N * DS._MASK_CADA)
esperar("calibrar")
alimentar(DS._CAL_N * DS._CAL_CADA)
esperar("revisar")
scr._mask_decidir(False)
check("rechazada: off", scr._masc_fase, "off")
check("sin mascara", scr._mascara, None)

app.destroy()
if fails:
    print(f"\nFALLARON {len(fails)}: {fails}")
    sys.exit(1)
print("\ntodo bien")
