# -*- coding: utf-8 -*-
"""
Smoke test de la UI del modulo de conteo.

Construye ProductionScreen con un app falso (sin camara, sin modelo) y
ejercita cada metodo, cada opcion y cada slider, comprobando que el estado
del modulo y del tracker CAMBIA como se espera. Si los handlers estuvieran
mal cableados, estos asserts fallan.
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

import customtkinter as ctk
from eyve.i18n import t, set_language
from eyve.ui import theme as T
T.apply("dark")
from eyve.modules.counting_module import CountingModule

fails = []


def check(name, got, want):
    ok = got == want
    print(f"  {'OK  ' if ok else 'FALLA'}  {name}: got={got!r} want={want!r}")
    if not ok:
        fails.append(name)


class FakeClass:
    def __init__(self, name):
        self.name = name
        self.color = "#00e676"


class FakeProject:
    classes = [FakeClass("rayon"), FakeClass("mancha")]
    ok_classes = ["rayon"]
    nok_classes = ["mancha"]
    active_model = None


class FakeLicense:
    nivel_label = "Pro"
    def permite_checks_avanzados(self): return True
    def es_pro(self): return True


class FakeApp(ctk.CTk):
    license = FakeLicense()
    def get_project(self): return FakeProject()


set_language("es")
app = FakeApp()
app.geometry("1280x880")

# evitar que arranque camara/modelo: parcheamos antes de construir
from eyve.ui.screens import production_screen as PS
PS.ProductionScreen._try_autoload_model = lambda self: None

scr = PS.ProductionScreen(app, app)
scr.pack(fill="both", expand=True)
app.update()

c = scr._counting

print("\n[1] Estado inicial")
check("metodo por omision", c.method, "line")
check("menu refleja el metodo", scr._count_method.get(), t("count_m_line"))
check("ayuda poblada", bool(scr._count_help.cget("text")), True)

print("\n[2] Cada metodo cambia el modulo y las filas visibles")
for key in CountingModule.METHODS:
    scr._on_count_method_change(t("count_m_" + key))
    app.update()
    check(f"metodo -> {key}", c.method, key)
    check(f"ayuda de {key}", scr._count_help.cget("text"), t("count_help_" + key))
    visible = {
        "dir":  scr._row_dir.winfo_ismapped(),
        "trig": scr._row_trig.winfo_ismapped(),
        "edge": scr._row_edge.winfo_ismapped(),
        "exp":  scr._row_exp.winfo_ismapped(),
        "draw": scr._row_draw.winfo_ismapped(),
    }
    want = {
        "screen":    {"dir":0,"trig":0,"edge":0,"exp":1,"draw":0},
        "line":      {"dir":1,"trig":0,"edge":0,"exp":0,"draw":1},
        "zone":      {"dir":0,"trig":1,"edge":0,"exp":1,"draw":1},
        "appear":    {"dir":0,"trig":0,"edge":0,"exp":0,"draw":0},
        "disappear": {"dir":0,"trig":0,"edge":1,"exp":0,"draw":0},
    }[key]
    check(f"filas visibles en {key}", visible, want)

print("\n[3] Etiqueta del boton de dibujo sigue a la geometria")
scr._on_count_method_change(t("count_m_line"))
check("linea -> 'dibujar meta'", scr._meta_btn.cget("text"), t("prod_draw_line"))
scr._on_count_method_change(t("count_m_zone"))
check("zona -> 'dibujar zona'", scr._meta_btn.cget("text"), t("prod_draw_zone"))
scr._arm_meta_draw()
check("armado pide rectangulo", scr._meta_kind, "rect")
scr._meta_drawing = False
scr._on_count_method_change(t("count_m_line"))
scr._arm_meta_draw()
check("armado pide linea", scr._meta_kind, "line")
scr._meta_drawing = False

print("\n[4] Opciones por metodo llegan al modulo")
for key in CountingModule.DIRECTIONS:
    scr._on_count_dir_change(t("count_d_" + key))
    check(f"sentido -> {key}", c.direction, key)
for key in CountingModule.ZONE_TRIGGERS:
    scr._on_count_trig_change(t("count_z_" + key))
    check(f"trigger -> {key}", c.zone_trigger, key)
for key in CountingModule.EDGES:
    scr._on_count_edge_change(t("count_e_" + key))
    check(f"borde -> {key}", c.edge, key)

print("\n[5] Rango esperado: vacio, valido y basura")
scr._exp_min.delete(0, "end"); scr._exp_min.insert(0, "12")
scr._exp_max.delete(0, "end"); scr._exp_max.insert(0, "18")
scr._on_count_expect_change()
check("min leido", c.expect_min, 12)
check("max leido", c.expect_max, 18)
scr._exp_min.delete(0, "end")
scr._on_count_expect_change()
check("vacio = sin limite", c.expect_min, None)
scr._exp_max.delete(0, "end"); scr._exp_max.insert(0, "abc")
scr._on_count_expect_change()
check("basura ignorada sin excepcion", c.expect_max, None)
scr._exp_max.delete(0, "end"); scr._exp_max.insert(0, "-5")
scr._on_count_expect_change()
check("negativo se clampea a 0 (= no debe haber nada)", c.expect_max, 0)

print("\n[6] Clase objetivo")
scr._on_count_class_change(t("prod_mod_all"))
check("(todas) = sin filtro", c.target_class, None)
scr._on_count_class_change("rayon")
check("clase concreta", c.target_class, "rayon")

print("\n[7] Sliders de persistencia llegan al tracker y a config")
from eyve.core import config as _cfg
scr._on_persist_change("track_lost", 25)
check("tolerancia al tracker", scr._tracker.config.max_lost_frames, 25)
check("tolerancia a config", _cfg.get("track_lost"), 25)
scr._on_persist_change("track_confirm", 5)
check("confirmar al tracker", scr._tracker.config.min_confirm_frames, 5)
scr._on_persist_change("track_iou", 40)
check("iou al tracker (0-1)", scr._tracker.config.iou_match, 0.40)
scr._on_persist_change("track_conf_min", 65)
check("conf minima al tracker", scr._tracker.config.process_conf_min, 0.65)
for k, (sl, vlbl, fmt) in scr._persist_sliders.items():
    if not vlbl.cget("text"):
        fails.append("etiqueta de slider vacia: " + k)

print("\n[8] Plegable de persistencia")
check("cerrado al inicio", scr._persist_box.winfo_ismapped(), 0)
scr._toggle_persist(); app.update()
check("abre", scr._persist_box.winfo_ismapped(), 1)
scr._toggle_persist(); app.update()
check("cierra", scr._persist_box.winfo_ismapped(), 0)

print("\n[9] Toggle y reset del modulo")
scr._count_var.set(True); scr._on_counting_toggle()
check("encendido", c.enabled, True)
c.counts["rayon"] = 7
scr._reset_counting()
check("reset limpia", c.total, 0)
scr._count_var.set(False); scr._on_counting_toggle()
check("apagado", c.enabled, False)

print("\n[10] Cambiar idioma y volver a ejercitar los menus")
set_language("en")
scr2 = PS.ProductionScreen(app, app)
app.update()
for key in CountingModule.METHODS:
    scr2._on_count_method_change(t("count_m_" + key))
    check(f"[en] metodo -> {key}", scr2._counting.method, key)
set_language("es")

app.destroy()

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("SMOKE TEST DE UI: TODO PASO")
