# -*- coding: utf-8 -*-
"""
Smoke test de la pantalla de demo.

Simula lo que hace un visitante del stand: elegir herramienta, arrastrar
sobre el lienzo, poner un fallo de impresion, cambiar el material, tomar el
control del modo automatico. Comprueba que cada gesto llega a la tela con
la CLASE correcta y en el LUGAR correcto, que es lo unico que no se puede
ver a ojo sin la expo enfrente.
"""
import sys
import types
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SCRATCH = Path(tempfile.gettempdir()) / "eyve_tests"
SCRATCH.mkdir(parents=True, exist_ok=True)

import customtkinter as ctk
import numpy as np
from eyve.i18n import t, set_language
from eyve.ui import theme as T
T.apply("dark")
set_language("es")

from eyve.demo.textile import MOTIFS, PRINT_FAULTS, WEAVES, YOLO_CLASSES
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


class FakeProject:
    classes = []
    ok_classes = []
    nok_classes = list(YOLO_CLASSES)
    active_model = None

    class paths:
        best_model = types.SimpleNamespace(exists=lambda: False)


class FakeApp(ctk.CTk):
    def get_project(self):
        return FakeProject()


app = FakeApp()
app.geometry("1600x950")

from eyve.ui.screens import demo_screen as DS
DS.DemoScreen._load_model = lambda self: None

scr = DS.DemoScreen(app, app)
scr.pack(fill="both", expand=True)
scr._source.start()        # on_show() no corre en la prueba
app.update()

print("\n[1] Arranca con los dos motores y el modo automatico")
check_true("hay patron de tela", scr._pattern is not None)
check_true("hay modulo de conteo", scr._counting.enabled)
check_true("hay modulo Patron", scr._patternmod.enabled)
check("herramienta inicial", scr._tool, "rayon")
check("el modo automatico arranca encendido", scr._auto, True)

print("\n[2] El lienzo mapea canvas -> frame")
scr._view = (960, 540, 960, 540, 0, 0)
check("centro", scr._canvas_to_frame(480, 270), (480, 270))
check("fuera del video -> None", scr._canvas_to_frame(-5, 270), None)
scr._view = (960, 540, 480, 270, 100, 50)
check("con letterbox", scr._canvas_to_frame(100 + 240, 50 + 135), (480, 270))
scr._view = (960, 540, 960, 540, 0, 0)

print("\n[3] Las dos herramientas pintan su clase")
scr._source.set_paused(True)
for cls in YOLO_CLASSES:
    scr.reset_demo()
    scr._set_tool(cls)
    check(f"herramienta activa {cls}", scr._tool, cls)
    scr._on_press(types.SimpleNamespace(x=200, y=200))
    for x in range(210, 320, 10):
        scr._on_drag(types.SimpleNamespace(x=x, y=200 + (x % 20)))
    scr._on_release(types.SimpleNamespace(x=320, y=200))
    check(f"un arrastre = un defecto ({cls})", len(scr._pattern.defects), 1)
    check(f"clase correcta ({cls})", scr._pattern.defects[0].cls, cls)

print("\n[4] Tocar el lienzo le quita el control al modo automatico")
scr._auto = True
scr._refresh_auto_label()
scr.reset_demo()
scr._on_press(types.SimpleNamespace(x=300, y=300))
scr._on_release(types.SimpleNamespace(x=300, y=300))
check("al dibujar se apaga el automatico", scr._auto, False)
check("y el aviso lo dice", scr._auto_lbl.cget("text"), t("demo_auto_off"))
scr._toggle_auto()
check("se puede volver a encender", scr._auto, True)
check("el aviso vuelve", scr._auto_lbl.cget("text"), t("demo_auto_on"))

print("\n[5] Los botones de fallo de impresion pintan sin ser clases")
for fault in PRINT_FAULTS:
    scr.reset_demo()
    scr._auto = True
    scr._place_fault(fault)
    check_true(f"{fault}: deja algo en la tela",
               len(scr._pattern.defects) >= 1, str(len(scr._pattern.defects)))
    check(f"{fault}: tambien quita el automatico", scr._auto, False)
    marcado = {d.cls for d in scr._pattern.defects}
    check_true(f"{fault}: no se etiqueta como clase de YOLO",
               not (marcado & set(YOLO_CLASSES)), str(marcado))

print("\n[6] El automatico es ESPACIADO, no una lluvia de defectos")
check_true("intervalo minimo de al menos 10 s", DS._AUTO_MIN >= 10.0,
           f"{DS._AUTO_MIN}s")
check_true("tope de defectos antes de limpiar", DS._AUTO_MAX_DEFECTS <= 6,
           str(DS._AUTO_MAX_DEFECTS))
scr.reset_demo()
scr._auto = True
scr._auto_next = 0.0
scr._auto_tick()
n1 = len(scr._pattern.defects)
check_true("coloca uno al vencer el tiempo", n1 >= 1, str(n1))
scr._auto_tick()      # sin esperar: no debe colocar otro
check("no coloca otro antes de tiempo", len(scr._pattern.defects), n1)
# al llegar al tope, limpia en vez de acumular
scr._auto_placed = DS._AUTO_MAX_DEFECTS
scr._auto_next = 0.0
scr._auto_tick()
check("al llegar al tope limpia la tela", len(scr._pattern.defects), 0)
check("y reinicia la cuenta", scr._auto_placed, 0)

print("\n[7] Los 5 metodos de conteo, con geometria automatica")
for m in CountingModule.METHODS:
    scr._on_count_method(t("count_m_" + m))
    check(f"metodo -> {m}", scr._counting.method, m)
    if m == "line":
        check_true("la meta se dibuja sola", scr._counting.line is not None)
    if m == "zone":
        check_true("la zona se dibuja sola", scr._counting.zone is not None)
    check_true(f"hay explicacion para {m}",
               bool(scr._count_help.cget("text")))

print("\n[8] Material: estampado y tejido, y recalibra al cambiar")
for w in WEAVES:
    scr._on_weave(t("weave_" + w))
    check(f"tejido -> {w}", scr._pattern.weave, w)
    check_true(f"el origen apunta a la tela nueva ({w})",
               scr._source.pattern is scr._pattern)
    check_true(f"recalibra al cambiar de tejido ({w})", scr._cal_left > 0,
               str(scr._cal_left))
for mo in MOTIFS:
    scr._on_motif(t("motif_" + mo))
    check(f"estampado -> {mo}", scr._pattern.motif, mo)

print("\n[9] Calibracion del modulo Patron al entrar")
scr._start_calibration(frames=3)
check("pide 3 frames", scr._cal_left, 3)
check_true("arranca sin calibrar", not scr._patternmod.calibrated)
for _ in range(3):
    f = scr._source.read()
    if f is not None:
        scr._patternmod.calibrate(f)
        scr._cal_left -= 1
check_true("queda calibrado", scr._patternmod.calibrated)

print("\n[10] El bucle corre sin modelo YOLO (solo con Patron)")
scr._worker = None
frame = scr._source.read()
check_true("hay frame", frame is not None)
out = scr._inspect(frame)
check_true("inspect devuelve imagen", out is not None and out.shape == (540, 960, 3))

print("\n[11] Las cadenas existen en los dos idiomas")
from eyve.i18n.es import STRINGS as ES
from eyve.i18n.en import STRINGS as EN
claves = (["demo_yolo_title", "demo_pattern_title", "demo_count_title",
           "demo_auto", "demo_auto_on", "demo_auto_off", "demo_material",
           "demo_calibrating", "demo_faults", "demo_nothing",
           "demo_count_reset", "demo_auto_stop"]
          + [f"demo_fault_{x}" for x in PRINT_FAULTS]
          + [f"motif_{m}" for m in MOTIFS]
          + [f"weave_{w}" for w in WEAVES]
          + [f"demo_tool_{c}" for c in YOLO_CLASSES])
faltan = [k for k in claves if k not in ES or k not in EN]
check("ninguna clave falta", faltan, [])

print("\n[12] Cierre limpio")
scr.on_hide()
check_true("el origen se detuvo", scr._source._thread is None)
scr.on_close()
app.destroy()
print("  OK    on_hide/on_close sin excepciones")

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f_ in fails:
        print("  -", f_)
    sys.exit(1)
print("PANTALLA DE DEMO: TODO PASO")
