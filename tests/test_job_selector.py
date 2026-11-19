# -*- coding: utf-8 -*-
"""
Selector de jobs por compañía → proyecto → SKU.

Con cientos de SKU un solo menú no se puede usar: el operador va
depurando. El caso que lo pidió (usuario, 18-nov-2026): una compañía con un
proyecto de 5 subensambles + 4 terminados y otro de 7 SKU, más los jobs de
siempre sin compañía.

Cada comprobación va con su contraparte que tendría que fallar si el filtro
no filtrara: contar «hay menos» no basta, se mira QUÉ queda.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("YOLO_OFFLINE", "1")

fails = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


from eyve.core import job as J

TMP = Path(tempfile.mkdtemp(prefix="eyve_job_selector_"))

# ── el caso del usuario ───────────────────────────────────────────────────
SUB = [f"SUB-{i}" for i in range(1, 6)]
TERM = [f"TERM-{i}" for i in range(1, 5)]
OTRO = [f"P2-{i}" for i in range(1, 8)]
VIEJOS = ["SKU-1234", "SKU-9"]


def jobs_del_caso():
    out = []
    for n in SUB + TERM:
        out.append(J.Job(name=n, company="Acme", product="Tablero"))
    for n in OTRO:
        out.append(J.Job(name=n, company="Acme", product="Arnés motor"))
    out.append(J.Job(name="B-1", company="Beta", product=""))
    for n in VIEJOS:
        out.append(J.Job(name=n))
    return out


print("\n[1] Agrupar y filtrar (núcleo)")
js = jobs_del_caso()
check("compañías en orden, «sin compañía» al final",
      J.companies(js) == ["Acme", "Beta", ""], J.companies(js))
check("proyectos de Acme, solo los suyos",
      J.products(js, "Acme") == ["Arnés motor", "Tablero"], J.products(js, "Acme"))
check("proyectos de Beta: solo «sin proyecto»", J.products(js, "Beta") == [""])
check("proyectos de todas", J.products(js) == ["Arnés motor", "Tablero", ""])

f = [j.name for j in J.filter_jobs(js, "Acme", "Tablero")]
check("Acme / Tablero = los 9 (5 sub + 4 terminados)", sorted(f) == sorted(SUB + TERM), f)
check("y ninguno del otro proyecto", not any(n in f for n in OTRO))
f = [j.name for j in J.filter_jobs(js, "Acme", "Tablero", "term")]
check("buscar «term» deja los 4 terminados", sorted(f) == TERM, f)
f = [j.name for j in J.filter_jobs(js, None, None, "acme motor")]
check("varias palabras exigen todas (compañía + proyecto)", sorted(f) == sorted(OTRO), f)
f = [j.name for j in J.filter_jobs(js, "", None)]
check("«sin compañía» = solo los viejos", sorted(f) == sorted(VIEJOS), f)
check("sin filtros = todos", len(J.filter_jobs(js)) == len(js))

print("\n[2] Guardar y leer los campos; los jobs viejos siguen abriendo")
d = TMP / "jobs"
for j in js:
    j.save(d)
leido = J.find(d, "TERM-2")
check("compañía y proyecto sobreviven al disco",
      (leido.company, leido.product) == ("Acme", "Tablero"),
      (leido.company, leido.product))
viejo = d / "VIEJO.yaml"
viejo.write_text(yaml.safe_dump({"name": "VIEJO", "modules": {}}), encoding="utf-8")
v = J.find(d, "VIEJO")
check("un YAML de antes, sin los campos, abre con «sin compañía»",
      v is not None and v.company == "" and v.product == "")

print("\n[3] capture() hereda el grupo del job base y respeta el explícito")
base = J.find(d, "SUB-1")
c = J.capture([], name="SUB-1", base=base)
check("hereda compañía/proyecto del base", (c.company, c.product) == ("Acme", "Tablero"))
c = J.capture([], name="X", base=base, company="Beta", product="")
check("el explícito manda, incluso vacío", (c.company, c.product) == ("Beta", ""))

print("\n[4] find() no devuelve el job de OTRO SKU con el mismo slug")
d2 = TMP / "slug"
J.Job(name="SKU 1", notes="soy el uno").save(d2)
J.Job(name="SKU/1", notes="soy el de la barra").save(d2)
check("PREVIO: comparten slug, el segundo quedó con sufijo",
      len(list(d2.glob("*.yaml"))) == 2 and J.slug("SKU 1") == J.slug("SKU/1"))
a = J.find(d2, "SKU 1")
b = J.find(d2, "SKU/1")
check("«SKU 1» es el suyo", a is not None and a.notes == "soy el uno")
check("«SKU/1» es el suyo, no el del slug", b is not None and b.notes == "soy el de la barra",
      b.notes if b else None)

print("\n[5] Listar sin cargar referencias; find() sí las trae")
r = J.Job(name="CON-REF")
r.reference_image = np.ones((4, 4), np.uint8)
r.save(d)
lista = {j.name: j for j in J.list_jobs(d, refs=False)}
check("listado ligero: sin imagen", lista["CON-REF"].reference_image is None)
check("PREVIO: el listado normal sí la trae",
      {j.name: j for j in J.list_jobs(d)}["CON-REF"].reference_image is not None)
check("find() la trae (es lo que se aplica)", J.find(d, "CON-REF").reference_image is not None)

# ── la pantalla ───────────────────────────────────────────────────────────
print("\n[6] La pantalla de Producción")
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

from eyve.core.project_manager import ClassDef, create_project
from eyve.i18n import t
from eyve.ui.app import EyveApp

proj = create_project("Selector", TMP)
proj.classes = [ClassDef(name="pieza", kind="ok")]
proj.save()
for j in jobs_del_caso():
    j.save(proj.paths.jobs)
N = len(jobs_del_caso())

app = EyveApp()
app.geometry("1280x900+40+40")
app.set_project(proj)
app.navigate("nav_production")


def settle(n=4):
    for _ in range(n):
        app.update()
        app.update_idletasks()


settle()
scr = app._screens["nav_production"]
scr._refresh_job_list()
settle()


def skus():
    return [v for v in scr._job_menu.cget("values") if v != t("job_none")]


check("arranca con todos", sorted(skus()) == sorted(j.name for j in jobs_del_caso()),
      len(skus()))
check("el contador lo dice", scr._job_count.cget("text") == t("job_count", k=N, n=N),
      scr._job_count.cget("text"))
vals = scr._job_company_menu.cget("values")
check("menú de compañía: todas + Acme + Beta + sin compañía",
      list(vals) == [t("job_all_f"), "Acme", "Beta", t("job_no_company")], vals)

scr._on_company_selected("Acme")
settle()
check("al elegir Acme quedan solo sus 16", sorted(skus()) == sorted(SUB + TERM + OTRO),
      len(skus()))
check("y el menú de proyecto trae solo los de Acme",
      list(scr._job_product_menu.cget("values")) == [t("job_all_m"), "Arnés motor", "Tablero"])

scr._on_product_selected("Tablero")
settle()
check("Acme / Tablero: 9", sorted(skus()) == sorted(SUB + TERM), len(skus()))
scr._job_search.insert(0, "term")
scr._refresh_job_list(releer=False)
settle()
check("y la búsqueda «term» deja 4", sorted(skus()) == TERM, skus())
check("contador 4 de N", scr._job_count.cget("text") == t("job_count", k=4, n=N))
scr._limpiar_busqueda()
settle()
check("Escape/limpiar regresa a los 9", len(skus()) == 9, len(skus()))

print("\n[7] Cargar un job y filtrar no lo descarga")
scr._job_menu.set("SUB-3")
scr._on_job_selected("SUB-3")
settle()
scr._on_company_selected(t("job_no_company"))
settle()
check("PREVIO: SUB-3 ya no está en la lista filtrada", "SUB-3" not in skus())
check("pero sigue cargado y a la vista",
      scr._job is not None and scr._job.name == "SUB-3" and scr._job_menu.get() == "SUB-3",
      scr._job_menu.get())

print("\n[8] Un job nuevo toma el grupo del filtro")
scr._on_company_selected("Acme")
scr._on_product_selected("Arnés motor")
scr._toggle_nuevo_job(True)
settle()
check("los campos vienen llenos con el filtro",
      scr._job_new_company.get() == "Acme" and scr._job_new_product.get() == "Arnés motor",
      (scr._job_new_company.get(), scr._job_new_product.get()))
scr._job_new_entry.insert(0, "P2-NUEVO")
scr._confirmar_job_row()
settle()
nuevo = J.find(proj.paths.jobs, "P2-NUEVO")
check("se guardó en Acme / Arnés motor",
      nuevo is not None and (nuevo.company, nuevo.product) == ("Acme", "Arnés motor"))
check("y aparece en la lista filtrada", "P2-NUEVO" in skus())

print("\n[9] Cambiar de grupo el job cargado toca solo el grupo")
antes = yaml.safe_load(nuevo.yaml_path(proj.paths.jobs).read_text(encoding="utf-8"))
# Un cambio en los controles SIN guardar no debe colarse al reagrupar.
scr._conf_slider.set(13)
scr._toggle_nuevo_job(True, regroup=True)
settle()
check("modo reagrupar: el nombre viene y no se edita",
      scr._job_new_entry.get() == "P2-NUEVO"
      and str(scr._job_new_entry.cget("state")) == "disabled")
scr._job_new_company.delete(0, "end")
scr._job_new_company.insert(0, "Beta")
scr._job_new_product.delete(0, "end")
scr._job_new_product.insert(0, "Arneses")
scr._confirmar_job_row()
settle()
despues = yaml.safe_load(nuevo.yaml_path(proj.paths.jobs).read_text(encoding="utf-8"))
check("en disco quedó en Beta / Arneses",
      (despues["company"], despues["product"]) == ("Beta", "Arneses"))
check("y lo demás del job, igual (la confianza sin guardar no se coló)",
      despues["source"] == antes["source"] and despues["modules"] == antes["modules"],
      despues["source"].get("conf"))
check("los filtros siguen al job", (scr._f_company, scr._f_product) == ("Beta", "Arneses"))
check("el siguiente «+» ya no está en modo reagrupar",
      (scr._toggle_nuevo_job(True), str(scr._job_new_entry.cget("state")))[1] == "normal")
scr._toggle_nuevo_job(False)

print("\n[10] Al volver, entra filtrado en el grupo del último job")
scr._job = None
scr._f_company = scr._f_product = None
scr._refresh_job_list()
proj.last_job = "TERM-1"
scr._cargar_ultimo_job()
settle()
check("filtros = Acme / Tablero", (scr._f_company, scr._f_product) == ("Acme", "Tablero"),
      (scr._f_company, scr._f_product))
check("y la lista muestra ese proyecto", sorted(skus()) == sorted(SUB + TERM))

app.destroy()
shutil.rmtree(TMP, ignore_errors=True)

print("\n" + "=" * 60)
print("SELECTOR DE JOBS: COMPAÑÍA → PROYECTO → SKU")
if fails:
    print(f"FALLARON {len(fails)}: {fails}")
    sys.exit(1)
print("todo bien")
