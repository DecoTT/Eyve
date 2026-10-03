# -*- coding: utf-8 -*-
"""
Smoke test de la pantalla de demo.

Simula lo que hace un visitante del stand: elegir herramienta, arrastrar
sobre el lienzo, limpiar, pausar, cambiar de motivo. Comprueba que cada
gesto llega a la tela con la CLASE correcta y en el LUGAR correcto — que
es lo unico que no se puede ver a ojo sin la expo enfrente.
"""
import sys
import types
import tempfile
from pathlib import Path

# El repo es el padre de tests/: nada de rutas absolutas, para que las
# pruebas corran en cualquier maquina y desde cualquier carpeta.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Salidas de las pruebas (hojas de contacto, proyectos temporales).
SCRATCH = Path(tempfile.gettempdir()) / "eyve_tests"
SCRATCH.mkdir(parents=True, exist_ok=True)

import customtkinter as ctk
import numpy as np
from eyve.i18n import t, set_language
from eyve.ui import theme as T
T.apply("dark")
set_language("es")

from eyve.demo.textile import DEFECT_CLASSES

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
    nok_classes = list(DEFECT_CLASSES)
    active_model = None
    class paths:
        best_model = types.SimpleNamespace(exists=lambda: False)


class FakeApp(ctk.CTk):
    def get_project(self): return FakeProject()


app = FakeApp()
app.geometry("1400x900")

from eyve.ui.screens import demo_screen as DS
scr = DS.DemoScreen(app, app)
scr.pack(fill="both", expand=True)
app.update()

print("\n[1] Se construye y arranca la tela sintetica")
check_true("hay patron", scr._pattern is not None)
check_true("hay origen sintetico", scr._source is not None)
f = scr._source.pattern.frame()
check("frame del tamano pedido", f.shape, (540, 960, 3))
check("herramienta inicial", scr._tool, "rayon")

print("\n[2] El lienzo mapea canvas -> frame")
# forzar un mapeo conocido (en la expo lo pone _paint con el tamano real)
scr._view = (960, 540, 960, 540, 0, 0)
check("centro", scr._canvas_to_frame(480, 270), (480, 270))
check("esquina", scr._canvas_to_frame(0, 0), (0, 0))
check("fuera del video -> None", scr._canvas_to_frame(-5, 270), None)
check("fuera por abajo -> None", scr._canvas_to_frame(480, 999), None)
# con letterbox (video centrado y escalado a la mitad)
scr._view = (960, 540, 480, 270, 100, 50)
check("letterbox: centro", scr._canvas_to_frame(100 + 240, 50 + 135), (480, 270))

print("\n[3] Un arrastre pinta UN defecto de la clase elegida")
scr._view = (960, 540, 960, 540, 0, 0)
scr._source.set_paused(True)          # tela quieta, como al dibujar con calma
for cls in DEFECT_CLASSES:
    scr.reset_demo()
    scr._set_tool(cls)
    check(f"herramienta activa {cls}", scr._tool, cls)
    ev = types.SimpleNamespace(x=200, y=200)
    scr._on_press(ev)
    for x in range(210, 330, 10):
        scr._on_drag(types.SimpleNamespace(x=x, y=200 + (x % 20)))
    scr._on_release(ev)
    check(f"un arrastre = un defecto ({cls})", len(scr._pattern.defects), 1)
    check(f"clase correcta ({cls})", scr._pattern.defects[0].cls, cls)
    labs = scr._pattern.visible_labels()
    check(f"queda etiquetado ({cls})", len(labs), 1)
    check(f"la etiqueta dice {cls}", labs[0][0], cls)
    # y se ve en el frame
    limpio = DS.TextilePattern(width=960, height=540, axis="x",
                               motif=scr._pattern.motif)
    limpio.offset = scr._pattern.offset
    d = np.abs(scr._pattern.frame().astype(int) - limpio.frame().astype(int))
    check_true(f"la tinta se ve en el frame ({cls})",
               (d.sum(axis=2) > 40).sum() > 150,
               f"px={(d.sum(axis=2) > 40).sum()}")

print("\n[4] Dos arrastres = dos defectos (instancias distintas)")
scr.reset_demo()
scr._set_tool("mancha")
for cx in (200, 600):
    scr._on_press(types.SimpleNamespace(x=cx, y=300))
    scr._on_drag(types.SimpleNamespace(x=cx + 30, y=300))
    scr._on_release(types.SimpleNamespace(x=cx + 30, y=300))
check("dos defectos", len(scr._pattern.defects), 2)

print("\n[5] Arrastrar fuera del video no pinta ni revienta")
scr.reset_demo()
scr._on_press(types.SimpleNamespace(x=-50, y=-50))
check_true("press fuera no inicia trazo", scr._drawing is False)
scr._on_drag(types.SimpleNamespace(x=-40, y=-40))
scr._on_release(types.SimpleNamespace(x=-40, y=-40))
check("nada pintado", len(scr._pattern.defects), 0)
# press dentro, arrastre que se sale, release fuera
scr._on_press(types.SimpleNamespace(x=300, y=300))
scr._on_drag(types.SimpleNamespace(x=5000, y=5000))
scr._on_release(types.SimpleNamespace(x=5000, y=5000))
check("el trazo que se sale no revienta", len(scr._pattern.defects), 1)
check_true("el trazo quedo cerrado", scr._drawing is False)

print("\n[6] Limpiar deja todo en cero")
scr._set_tool("rayon")
scr._on_press(types.SimpleNamespace(x=300, y=300))
scr._on_drag(types.SimpleNamespace(x=400, y=320))
scr._on_release(types.SimpleNamespace(x=400, y=320))
scr._counting.counts["rayon"] = 5
scr._nok_total = 5
scr.reset_demo()
check("sin defectos", len(scr._pattern.defects), 0)
check("conteo en cero", scr._counting.total, 0)
check("contador en cero", scr._nok_total, 0)
check("sin tracks", len(scr._tracker.get_all()), 0)
check("etiqueta en cero", scr._count_lbl.cget("text"), t("demo_found", n=0))

print("\n[7] Pausa, velocidad y motivo")
scr._source.set_paused(False)
scr._toggle_pause()
check("pausado", scr._source.paused, True)
check("boton dice reanudar", scr._pause_btn.cget("text"), t("demo_resume"))
scr._toggle_pause()
check("reanudado", scr._source.paused, False)
check("boton dice pausar", scr._pause_btn.cget("text"), t("demo_pause"))

scr._on_speed(200.0)
check("velocidad al patron", scr._source.pattern.speed, 200.0)
scr._on_speed(0.0)
check("velocidad cero permitida", scr._source.pattern.speed, 0.0)

for motif in DS.MOTIFS:
    scr._on_motif(motif)
    check(f"motivo {motif}", scr._pattern.motif, motif)
    check_true(f"el origen apunta a la tela nueva ({motif})",
               scr._source.pattern is scr._pattern)
    check(f"cambiar motivo limpia ({motif})", len(scr._pattern.defects), 0)

print("\n[8] La tela VIAJA y el defecto viaja con ella")
scr._on_motif("diamantes")
scr._source.set_paused(True)
scr._set_tool("mancha")
scr._view = (960, 540, 960, 540, 0, 0)
scr._on_press(types.SimpleNamespace(x=700, y=270))
scr._on_drag(types.SimpleNamespace(x=720, y=280))
scr._on_release(types.SimpleNamespace(x=720, y=280))
xs = []
for _ in range(5):
    lab = scr._pattern.visible_labels()
    xs.append(lab[0][1] if lab else None)
    scr._pattern.advance(0.4)
vistos = [q for q in xs if q is not None]
check_true("el defecto se mueve con la tela", len(set(vistos)) > 1, f"x={vistos}")

print("\n[9] El bucle corre sin modelo cargado (no revienta)")
scr._worker = None
scr._running = True
scr._source.set_paused(False)
scr._source.start()
for _ in range(5):
    frame = scr._source.read()
    check_true("hay frame del origen", frame is not None)
    out = scr._inspect(frame) if frame is not None else None
    check_true("inspect devuelve imagen sin modelo",
               out is not None and out.shape == (540, 960, 3))
    break
app.update()

print("\n[10] Las cadenas de la demo existen en los dos idiomas")
from eyve.i18n.es import STRINGS as ES
from eyve.i18n.en import STRINGS as EN
claves = ["demo_title","demo_hint","demo_side_eyve","demo_side_you","demo_clear",
          "demo_pause","demo_resume","demo_speed","demo_found","demo_clean",
          "demo_defect","demo_loading","demo_no_model","demo_model_error",
          "nav_demo"] + [f"demo_tool_{c}" for c in DEFECT_CLASSES]
for k in claves:
    if k not in ES: fails.append(f"falta en es: {k}")
    if k not in EN: fails.append(f"falta en en: {k}")
print(f"  OK    {len(claves)} claves presentes en es y en")

print("\n[11] Cierre limpio")
scr.on_hide()
check_true("el origen se detuvo", scr._source._thread is None)
scr.on_close()
app.destroy()
print("  OK    on_hide/on_close sin excepciones")

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("PANTALLA DE DEMO: TODO PASO")
