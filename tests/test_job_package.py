# -*- coding: utf-8 -*-
"""
Paquete de job (.eyvejob): exportar en una PC e importar en otra.

Lo que se comprueba es lo que haría fallar una semana de prueba en planta:
que llegue el MISMO modelo, las mismas clases y los mismos parámetros del
job (con su referencia), que la fuente no apunte a un video de la PC de
origen, que nunca se escriba encima de un proyecto existente y que un
paquete dañado o manipulado no deje nada a medias ni escriba fuera.
"""
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eyve.core import job as J  # noqa: E402
from eyve.core.job_package import PaqueteInvalido, exportar, importar  # noqa: E402
from eyve.core.project_manager import ClassDef, create_project  # noqa: E402
from eyve.modules.assembly_module import AssemblyModule  # noqa: E402

SCRATCH = Path(tempfile.mkdtemp(prefix="eyve_pkg_"))
fails = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


# ── PC de origen: proyecto entrenado con un job de Ensamble ─────────────────
orig = create_project("Tarjeta_A", SCRATCH / "pc1", target="antena y etiqueta")
orig.classes = [ClassDef("base", kind="ok"), ClassDef("antena", kind="ok"),
                ClassDef("etiqueta", kind="ok")]
orig.paths.best_model.parent.mkdir(parents=True, exist_ok=True)
orig.paths.best_model.write_bytes(np.random.default_rng(1).bytes(4096))
orig.active_model = str(orig.paths.best_model)
orig.save()
asm = AssemblyModule()
asm.enabled, asm.glove, asm.tolerance = True, "azul", 0.09
asm.reference = {"antena": [0.07, -0.5, 0.15, 0.09], "etiqueta": [0.53, 0.45, 0.81, 0.69]}
job = J.capture([asm], source={"camera_source": 0, "video_path": r"C:\videos\linea3.mp4",
                               "kind": "video", "conf": 0.45}, name="SKU 123/B")
job.reference_image = np.full((8, 8), 7, np.uint8)
job.save(orig.paths.jobs)

pkg = exportar(orig, "SKU 123/B", SCRATCH / "salida" / "tarjeta_a")
check("se escribe con extensión .eyvejob", pkg.suffix == ".eyvejob" and pkg.exists(), pkg.name)
with zipfile.ZipFile(pkg) as zf:
    nombres = zf.namelist()
check("no lleva imágenes ni datasets", all(n.split("/")[0] in ("manifest.json", "project.yaml",
      "classes.yaml", "models", "jobs") for n in nombres), str(nombres))

# ── PC de destino ────────────────────────────────────────────────────────────
destino = SCRATCH / "pc2"
p = importar(pkg, destino)
check("crea el proyecto en la carpeta de destino", p.root.parent == destino and p.root.exists(),
      str(p.root))
check("las mismas clases, en el mismo orden (es el índice del modelo)",
      [c.name for c in p.classes] == ["base", "antena", "etiqueta"], str(p.class_names))
check("el MISMO modelo, en su carpeta",
      Path(p.active_model) == p.paths.best_model
      and p.paths.best_model.read_bytes() == orig.paths.best_model.read_bytes())
j2 = J.find(p.paths.jobs, "SKU 123/B")
check("llega el job con su nombre real", j2 is not None)
m2 = AssemblyModule()
J.apply(j2, [m2])
check("con sus parámetros: referencia, tolerancia y guante",
      m2.reference == asm.reference and m2.tolerance == 0.09 and m2.glove == "azul",
      f"{m2.tolerance} {m2.glove}")
check("y la referencia del Patrón (.npy)",
      j2.reference_image is not None and int(j2.reference_image[0, 0]) == 7)
check("la fuente ya no apunta al video de la PC de origen",
      j2.source.get("video_path") == "" and "kind" not in j2.source, str(j2.source))
check("pero conserva el umbral", j2.source.get("conf") == 0.45, str(j2.source.get("conf")))
check("el job queda como el último, para abrir directo en Producción", p.last_job == "SKU 123/B")

p2 = importar(pkg, destino)
check("importar dos veces NO escribe encima: carpeta nueva",
      p2.root != p.root and p.paths.best_model.exists(), f"{p.root.name} / {p2.root.name}")

# ── paquetes malos ──────────────────────────────────────────────────────────
def copia_con(nombre, cambiar):
    out = SCRATCH / nombre
    with zipfile.ZipFile(pkg) as zi, zipfile.ZipFile(out, "w") as zo:
        for n in zi.namelist():
            data = cambiar(n, zi.read(n))
            if data is not None:
                zo.writestr(n, data)
        extra = cambiar("__extra__", None)
        if extra:
            zo.writestr(*extra)
    return out


def importar_falla(paquete):
    antes = sorted(x.name for x in destino.iterdir())
    try:
        importar(paquete, destino)
        return False, "no lanzó"
    except PaqueteInvalido as e:
        despues = sorted(x.name for x in destino.iterdir())
        return antes == despues, str(e)[:120]


danado = copia_con("danado.eyvejob", lambda n, d: (d[:-1] + b"X") if n == "models/best.pt" else
                   (None if n == "__extra__" else d))
ok, msg = importar_falla(danado)
check("modelo dañado → se rechaza y no queda carpeta a medias", ok, msg)

futuro = copia_con("futuro.eyvejob", lambda n, d: json.dumps(
    {**json.loads(d), "version_formato": 99}).encode() if n == "manifest.json" else
    (None if n == "__extra__" else d))
ok, msg = importar_falla(futuro)
check("formato de una versión más nueva → pide actualizar Eyve", ok and "actualiza" in msg, msg)

(SCRATCH / "no_es.eyvejob").write_bytes(b"hola")
ok, msg = importar_falla(SCRATCH / "no_es.eyvejob")
check("un archivo que no es paquete → se rechaza", ok, msg)

malo = copia_con("malo.eyvejob", lambda n, d: ("../../fuera.txt", b"x") if n == "__extra__" else d)
p3 = importar(malo, destino)
check("una entrada '../' en el ZIP se ignora: nada se escribe fuera",
      not (SCRATCH / "fuera.txt").exists() and not (destino / "fuera.txt").exists()
      and not any(p3.root.rglob("fuera.txt")))

shutil.rmtree(SCRATCH, ignore_errors=True)
print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("PAQUETE DE JOB: LO MISMO DE UNA PC A OTRA")
