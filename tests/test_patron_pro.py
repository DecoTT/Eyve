# -*- coding: utf-8 -*-
"""
Patron es Pro desde 2.1.6, y la demo trae su modelo.

Lo que pidio el dueño: el modelo de la demo textil va en el instalador, y
el modulo Patron pasa a ser Pro "con ese demo model".  Quedo asi:

  - Produccion: la casilla de Patron se desactiva sin Pro, como Polaridad y
    Ensamble; calibrar no lo enciende; un job que lo trae encendido corre
    con marca de agua (eso ya lo hace Job.pro_modules con `pro = True`).
  - Demo textil: abre para todos y YOLO marca rayon y mancha; sin Pro
    Patron no corre, su recuadro lo dice y el modo automatico no pone
    fallos de impresion que nadie iba a marcar.
  - El modelo: si no hay proyecto, se busca el de desarrollo y luego el que
    build_release.ps1 mete en eyve/demo/modelo/.

Cada comprobacion con su contraria: lo que pasa con Pro y lo que pasa sin.
"""
import os
import shutil
import sys
import tempfile
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


from eyve.inference import detector as _det
_det.VideoSource.start = lambda self: False
_det.VideoSource.stop = lambda self: None
_det.VideoSource.read = lambda self: None
from eyve.ui.screens import production_screen as _ps
_ps.ProductionScreen._try_autoload_model = lambda self: None
_ps.ProductionScreen._auto_start_preview = lambda self: None
from eyve.ui.screens import demo_screen as DS
DS.DemoScreen._load_model = lambda self: None
from eyve.core import updater as _U
_U.check_async = lambda cb: None

from eyve.core import job as J
from eyve.core.project_manager import ClassDef, create_project
from eyve.demo.textile import PRINT_FAULTS, YOLO_CLASSES
from eyve.i18n import t
from eyve.modules import PatternModule
from eyve.ui.app import EyveApp

print("\n[1] El modulo es Pro, y un job lo cuenta como tal")
check("PatternModule.pro", PatternModule.pro is True)
pm = PatternModule()
pm.enabled = True
job = J.capture([pm], name="SKU-PAT")
check("job con Patron encendido: lo nombra como Pro",
      job.pro_modules([pm]) == ["Patrón"], str(job.pro_modules([pm])))
pm.enabled = False
job = J.capture([pm], name="SKU-PAT-OFF")
check("y apagado no", job.pro_modules([pm]) == [], str(job.pro_modules([pm])))

# ── la app real, con la licencia forzada ─────────────────────────────────
TMP = Path(tempfile.mkdtemp(prefix="eyve_patron_pro_"))
proj = create_project("PatronPro", TMP)
proj.classes = [ClassDef(name="pieza", kind="ok")]
proj.save()
app = EyveApp()
app.geometry("1280x900+40+40")


def settle(n=4):
    for _ in range(n):
        app.update()
        app.update_idletasks()


def con_licencia(pro: bool):
    app.license.permite_checks_avanzados = lambda: pro
    app.license.es_pro = lambda: pro


def produccion(pro: bool):
    con_licencia(pro)
    app.set_project(proj)            # tira la pantalla y la rehace
    app.navigate("nav_production")
    settle()
    return app._screens["nav_production"]


print("\n[2] Produccion CON Pro: Patron se enciende")
scr = produccion(True)
check("la casilla esta activa", scr._pat_chk.cget("state") == "normal",
      scr._pat_chk.cget("state"))
scr._pat_var.set(True)
scr._on_pattern_toggle()
check("al marcarla, el modulo se enciende", scr._pattern.enabled)
scr._pat_var.set(False)
scr._on_pattern_toggle()
scr._calibrate_pattern()
check("calibrar lo enciende", scr._pattern.enabled and scr._pat_cal_left > 0)

print("\n[3] Produccion SIN Pro: no se enciende a mano")
scr = produccion(False)
check("la casilla esta desactivada", scr._pat_chk.cget("state") == "disabled",
      scr._pat_chk.cget("state"))
check("y dice (Pro)", "(Pro)" in scr._pat_chk.cget("text"),
      scr._pat_chk.cget("text"))
scr._pat_var.set(True)                 # aunque llegue a marcarse
scr._on_pattern_toggle()
check("el modulo NO se enciende", not scr._pattern.enabled)
check("y la casilla vuelve a apagarse", not scr._pat_var.get())
scr._calibrate_pattern()
check("calibrar tampoco lo enciende",
      not scr._pattern.enabled and scr._pat_cal_left == 0)
check("y dice por que", scr._pat_status.cget("text") == t("lic_pro_required"),
      scr._pat_status.cget("text")[:40])
# un job que lo trae encendido si corre, con marca de agua
scr._pattern.enabled = True
scr._job = J.capture(scr._modulos(), name="SKU-CON-PATRON")
scr._avisos_del_job()
check("un job con Patron encendido deja marca de agua sin Pro",
      scr._wm_text == t("wm_pro"), repr(scr._wm_text))
scr._calibrate_pattern()
check("y ese job SI se puede calibrar", scr._pat_cal_left > 0)

# ── la demo ──────────────────────────────────────────────────────────────
print("\n[4] Demo SIN Pro: abre, YOLO trabaja, Patron no")
con_licencia(False)
app.navigate("nav_demo")
settle()
demo = app._screens["nav_demo"]
check("Patron apagado", not demo._patternmod.enabled)
check("el recuadro de Patron lo dice",
      demo._pat_lbl.cget("text") == t("demo_pattern_pro"),
      demo._pat_lbl.cget("text"))
demo._start_calibration()
check("no calibra nada", demo._cal_left == 0)
demo._inspect(demo._source.read())
check("tras inspeccionar sigue diciendolo",
      demo._pat_lbl.cget("text") == t("demo_pattern_pro"))
demo.reset_demo()
check("y tras limpiar la tela tambien",
      demo._pat_lbl.cget("text") == t("demo_pattern_pro"))
clases = set()
for _ in range(40):
    demo._pattern.clear_defects()
    demo._auto_placed = 0
    demo._auto_next = 0.0
    demo._auto_tick()
    clases |= {d.cls for d in demo._pattern.defects}
check("el automatico solo pone lo que YOLO marca",
      clases and clases <= set(YOLO_CLASSES), str(sorted(clases)))

print("\n[5] Demo CON Pro: Patron corre")
app._teardown_screen(app._screens.pop("nav_demo"))
con_licencia(True)
app.navigate("nav_demo")
settle()
demo = app._screens["nav_demo"]
check("Patron encendido", demo._patternmod.enabled)
check("el recuadro no habla de licencia",
      demo._pat_lbl.cget("text") != t("demo_pattern_pro"))
demo._start_calibration()
check("calibra", demo._cal_left > 0)
clases = set()
for _ in range(40):
    demo._pattern.clear_defects()
    demo._auto_placed = 0
    demo._auto_next = 0.0
    demo._auto_tick()
    clases |= {d.cls for d in demo._pattern.defects}
check("el automatico tambien pone fallos de impresion",
      bool(clases - set(YOLO_CLASSES)), str(sorted(clases)))

print("\n[6] El modelo de la demo, sin proyecto abierto")
demo._app = types.SimpleNamespace(get_project=lambda: None)
dev, emp = DS._MODELO_DEV, DS._MODELO_EMPAQUETADO
falso = TMP / "empaquetado" / "best.pt"
falso.parent.mkdir(parents=True)
falso.write_bytes(b"x")
try:
    DS._MODELO_DEV = TMP / "no_existe.pt"
    DS._MODELO_EMPAQUETADO = TMP / "tampoco.pt"
    check("sin ninguno: no hay modelo", demo._model_path() is None)
    DS._MODELO_EMPAQUETADO = falso
    check("con el empaquetado: lo usa", demo._model_path() == falso,
          str(demo._model_path()))
    DS._MODELO_DEV = falso.parent.parent / "dev.pt"
    DS._MODELO_DEV.write_bytes(b"x")
    check("si tambien esta el de desarrollo, manda ese",
          demo._model_path() == DS._MODELO_DEV)
finally:
    DS._MODELO_DEV, DS._MODELO_EMPAQUETADO = dev, emp
check("el empaquetado se busca DENTRO de eyve/ (el actualizador no toca "
      "projects/)", emp.relative_to(REPO).parts[:3] == ("eyve", "demo", "modelo"),
      str(emp.relative_to(REPO)))

for s in list(app._screens.values()):
    if hasattr(s, "on_close"):
        s.on_close()
app.destroy()
shutil.rmtree(TMP, ignore_errors=True)

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f_ in fails:
        print("  -", f_)
    sys.exit(1)
print("PATRON PRO: TODO PASO")
