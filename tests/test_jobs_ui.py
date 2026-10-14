# -*- coding: utf-8 -*-
"""
Los jobs vistos desde la pantalla de Producción.

`test_jobs.py` comprueba el núcleo —guardar, cargar, aplicar a los
módulos—. Lo que falta, y es lo que de verdad puede fallar sin ruido, es
que la PANTALLA se entere:

    `J.apply()` toca los MÓDULOS, no los widgets. Si al cargar un job los
    controles siguen enseñando los valores viejos, el operador lee una cosa
    y la inspección corre con otra. Eso es peor que no cargar nada, porque
    nadie lo nota hasta que salen piezas mal juzgadas.

Por eso aquí cada comprobación mira el WIDGET, no el módulo, y cada una
lleva delante una que exige que el widget estuviera en otro valor — si no,
"se refrescó" pasaría igual sin refrescar.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("YOLO_OFFLINE", "1")

fails = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


# ── neutralizar cámara, modelo y red ──────────────────────────────────────
from eyve.inference import detector as _det
_det.VideoSource.start = lambda self: False
_det.VideoSource.stop = lambda self: None
_det.VideoSource.read = lambda self: None

from eyve.ui.screens import production_screen as _ps
_ps.ProductionScreen._try_autoload_model = lambda self: None
_ps.ProductionScreen._auto_start_preview = lambda self: None
from eyve.ui.screens import demo_screen as _ds
_ds.DemoScreen._load_model = lambda self: None

from eyve.core import updater as _U
_U.check_async = lambda cb: None

from eyve.core import job as J
from eyve.core.project_manager import ClassDef, create_project
from eyve.i18n import t
from eyve.ui.app import EyveApp

TMP = Path(tempfile.mkdtemp(prefix="eyve_jobs_ui_"))
proj = create_project("JobsUI", TMP)
proj.classes = [ClassDef(name="IC", kind="nok"), ClassDef(name="pieza", kind="ok")]
proj.save()

app = EyveApp()
app.geometry("1280x900+40+40")
app.set_project(proj)
app.navigate("nav_production")
for _ in range(5):
    app.update()
    app.update_idletasks()
scr = app._screens["nav_production"]


def settle(n=3):
    for _ in range(n):
        app.update()
        app.update_idletasks()


print("\n[1] La barra de job existe y arranca sin job")
check("hay selector de job", hasattr(scr, "_job_menu"))
check("arranca en «sin job»", scr._job_menu.get() == t("job_none"),
      scr._job_menu.get())
check("y sin job cargado", scr._job is None)
check("la carpeta jobs/ está en el proyecto",
      proj.paths.jobs.name == "jobs", str(proj.paths.jobs))

# El caso que rompió test_ui_counting: un proyecto a medias —o un objeto que
# sólo se parece a uno— no puede tirar Producción abajo al construirse, sólo
# por no poder listar los jobs.
class ProyectoAMedias:
    pass


_real_get = app.get_project
app.get_project = lambda: ProyectoAMedias()
try:
    check("un proyecto sin paths no revienta el listado de jobs",
          scr._jobs_dir() is None)
    scr._refresh_job_list()
    check("y el selector sigue en pie", scr._job_menu.get() == t("job_none"))
except Exception as e:
    check("un proyecto sin paths no revienta el listado de jobs", False, str(e))
finally:
    app.get_project = _real_get

print("\n[2] Guardar un job con la configuración de ahora")
# dejar los módulos en algo reconocible
scr._counting.enabled = True
scr._counting.target_class = "pieza"
scr._counting.set_method("zone")
scr._counting.zone = (12, 34, 210, 180)
scr._counting.zone_trigger = "exit"
scr._counting.expect_min, scr._counting.expect_max = 3, 9
scr._pattern.enabled = True
scr._pattern.set_method("layout")
scr._pattern.sensitivity = 83.0
scr._tracker.configure(max_lost_frames=21)

# Sin modales: el nombre se escribe en la fila que abre "+"
scr._toggle_nuevo_job(True)
settle()
check("la fila del nombre aparece", scr._job_new_row.winfo_ismapped())
scr._job_new_entry.insert(0, "SKU-TEST-1")
scr._crear_job()
settle()
check("y se oculta al crear", not scr._job_new_row.winfo_ismapped())
check("el job queda cargado", scr._job is not None and scr._job.name == "SKU-TEST-1",
      scr._job.name if scr._job else "None")
check("y el YAML está en el proyecto",
      (proj.paths.jobs / "SKU-TEST-1.yaml").exists(),
      str(sorted(p.name for p in proj.paths.jobs.glob("*"))))
check("el selector lo muestra", scr._job_menu.get() == "SKU-TEST-1",
      scr._job_menu.get())
check("el proyecto recuerda el último job",
      proj.last_job == "SKU-TEST-1", proj.last_job)

print("\n[3] Con los controles en OTROS valores, cargar el job los devuelve")
# dejar los módulos y los widgets en valores distintos
scr._counting.set_method("screen")
scr._counting.zone = None
scr._counting.expect_min = scr._counting.expect_max = None
scr._pattern.set_method("periodo")
scr._pattern.sensitivity = 10.0
scr._pattern.enabled = False
scr._tracker.configure(max_lost_frames=5)
scr._count_method.set(t("count_m_screen"))
scr._pat_method.set(t("pat_m_periodo"))
scr._pat_sens.set(10)
scr._count_var.set(False)
scr._pat_var.set(False)
# y el deslizador de persistencia, que es el otro control que un job trae
_sl_lost = scr._persist_sliders["track_lost"][0]
_sl_lost.set(5)
scr._exp_min.delete(0, "end")
scr._exp_max.delete(0, "end")
settle()

# ── las guardas: si ya estuvieran bien, lo de abajo no probaría nada ─────
check("PREVIO: el widget del método de conteo NO es el guardado",
      scr._count_method.get() != t("count_m_zone"), scr._count_method.get())
check("PREVIO: el deslizador de sensibilidad NO es el guardado",
      abs(scr._pat_sens.get() - 83.0) > 1, str(scr._pat_sens.get()))
check("PREVIO: las casillas están apagadas",
      not scr._count_var.get() and not scr._pat_var.get())
check("PREVIO: el deslizador de persistencia NO es el guardado",
      abs(_sl_lost.get() - 21) > 1, str(_sl_lost.get()))

scr._job_menu.set("SKU-TEST-1")
scr._on_job_selected("SKU-TEST-1")
settle()

print("  — los módulos —")
check("conteo: método", scr._counting.method == "zone", scr._counting.method)
check("conteo: la zona vuelve", scr._counting.zone == (12, 34, 210, 180),
      str(scr._counting.zone))
check("conteo: rango esperado",
      (scr._counting.expect_min, scr._counting.expect_max) == (3, 9),
      f"{scr._counting.expect_min}-{scr._counting.expect_max}")
check("patrón: método y sensibilidad",
      scr._pattern.method == "layout" and scr._pattern.sensitivity == 83.0,
      f"{scr._pattern.method} {scr._pattern.sensitivity}")
check("tracker: persistencia", scr._tracker.config.max_lost_frames == 21,
      str(scr._tracker.config.max_lost_frames))

print("  — y los WIDGETS, que es lo que lee el operador —")
check("el menú de método de conteo lo muestra",
      scr._count_method.get() == t("count_m_zone"), scr._count_method.get())
check("el menú de disparo de zona lo muestra",
      scr._count_trig.get() == t("count_z_exit"), scr._count_trig.get())
check("el menú de método del patrón lo muestra",
      scr._pat_method.get() == t("pat_m_layout"), scr._pat_method.get())
check("el deslizador de sensibilidad lo muestra",
      abs(scr._pat_sens.get() - 83.0) < 1.5, str(scr._pat_sens.get()))
check("las casillas de los módulos quedan encendidas",
      scr._count_var.get() and scr._pat_var.get(),
      f"conteo={scr._count_var.get()} patrón={scr._pat_var.get()}")
check("el rango esperado aparece en las cajas",
      scr._exp_min.get() == "3" and scr._exp_max.get() == "9",
      f"{scr._exp_min.get()!r}-{scr._exp_max.get()!r}")
check("la clase objetivo del conteo",
      scr._count_class.get() == "pieza", scr._count_class.get())
check("el deslizador de persistencia lo muestra",
      abs(_sl_lost.get() - 21) < 1.5, str(_sl_lost.get()))

print("\n[4] La marca de agua: job Pro sin licencia Pro")
scr._polarity.enabled = True
scr._polarity.target_class = "IC"
scr._toggle_nuevo_job(True)
scr._job_new_entry.insert(0, "SKU-CON-PRO")
scr._crear_job()
settle()

scr._pro_ok = True          # licencia que SÍ cubre
scr._avisos_del_job()
check("PREVIO: con licencia Pro no hay marca", scr._wm_text == "",
      repr(scr._wm_text))
check("PREVIO: ni aviso de Pro", "Pro" not in scr._job_warn.cget("text")
      or t("job_pro_warn", mods="x")[:12] not in scr._job_warn.cget("text"))

scr._pro_ok = False         # licencia que NO lo cubre
scr._avisos_del_job()
check("sin licencia Pro aparece la marca de agua",
      scr._wm_text == t("wm_pro"), repr(scr._wm_text))
check("y el aviso nombra el módulo",
      "Polaridad" in scr._job_warn.cget("text"), scr._job_warn.cget("text")[:70])

print("\n[5] La marca de agua se dibuja de verdad sobre el frame")
import numpy as np
img = np.zeros((300, 500, 3), np.uint8)
antes = img.copy()
scr._draw_watermark(img)
check("el frame cambia", not np.array_equal(img, antes),
      f"{int(np.abs(img.astype(int) - antes.astype(int)).sum())} de diferencia")
check("y se dibuja abajo a la derecha",
      img[150:, 250:].any() and not img[:140, :240].any())
# ── el caso malo: sin marca, el frame no se toca ────────────────────────
scr._wm_text = ""
img2 = np.zeros((300, 500, 3), np.uint8)
scr._draw_watermark(img2)
check("sin marca, el frame queda intacto", not img2.any())

print("\n[6] Un job que pide un módulo Pro NO pierde sus parámetros")
# Guardarlos aunque la licencia no los permita: perderlos por abrir el job
# en una máquina sin Pro sería peor que no poder encenderlo.
jp = J.find(proj.paths.jobs, "SKU-CON-PRO")
check("el job guardó la polaridad encendida",
      (jp.modules.get("polarity") or {}).get("enabled") is True,
      str(jp.modules.get("polarity"))[:60])
check("con su clase objetivo",
      (jp.modules.get("polarity") or {}).get("target_class") == "IC")

print("\n[7] Al volver a Producción se reabre el último job")
scr._job = None
scr._job_menu.set(t("job_none"))
check("PREVIO: no hay job cargado", scr._job is None)
scr.on_show()
settle()
check("vuelve el último usado",
      scr._job is not None and scr._job.name == proj.last_job,
      f"{scr._job.name if scr._job else None} vs {proj.last_job}")

# Lo anterior usa el proyecto EN MEMORIA, así que pasaría igual aunque no
# se escribiera nada en disco. Lo que de verdad importa es cerrar Eyve y
# volver: se relee el project.yaml.
from eyve.core.project_manager import load_project
recargado = load_project(proj.root)
check("y sobrevive a releer el proyecto del disco",
      recargado.last_job == proj.last_job,
      f"{recargado.last_job!r} vs {proj.last_job!r}")

# ── cierre ────────────────────────────────────────────────────────────────
try:
    app._teardown_all_screens()
    app.update()
    app.destroy()
    print("\n  OK    cierre limpio")
except Exception as e:
    fails.append(f"cierre: {e}")
shutil.rmtree(TMP, ignore_errors=True)

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("JOBS EN PRODUCCIÓN: LA PANTALLA ENSEÑA LO QUE SE ESTÁ INSPECCIONANDO")
