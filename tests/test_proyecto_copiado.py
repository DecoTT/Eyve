# -*- coding: utf-8 -*-
"""
Un proyecto copiado a otra PC (otra carpeta) debe encontrar su modelo.

project.yaml guarda el modelo activo con ruta absoluta. Al copiar la carpeta
del proyecto a otra máquina esa ruta ya no existe y Producción decía "sin
modelo" aunque models/best.pt viajara con el proyecto.
"""
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eyve.core.project_manager import ClassDef, create_project, load_project  # noqa: E402

SCRATCH = Path(tempfile.mkdtemp(prefix="eyve_copia_"))
fails = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


orig = create_project("Original", SCRATCH / "pc1", target="prueba")
orig.classes = [ClassDef("base", kind="ok")]
orig.paths.create_all()
orig.paths.best_model.write_bytes(b"pesos")
orig.active_model = str(orig.paths.best_model)
orig.save()

check("PREVIO: en su carpeta, el modelo es el de su ruta",
      load_project(orig.root).active_model == str(orig.paths.best_model))

copia = SCRATCH / "pc2" / "Original"
shutil.copytree(orig.root, copia)
shutil.rmtree(SCRATCH / "pc1")          # la PC original ya no está
p = load_project(copia)
check("copiado a otra carpeta → usa su models/best.pt",
      p.active_model == str(copia / "models" / "best.pt") and Path(p.active_model).exists(),
      str(p.active_model))

# sin best.pt en la copia no se inventa nada: se queda la ruta guardada
sin = SCRATCH / "pc3" / "Original"
shutil.copytree(copia, sin)
(sin / "models" / "best.pt").unlink()
check("sin modelo en la copia → deja la ruta guardada", load_project(sin).active_model == str(orig.paths.best_model),
      str(load_project(sin).active_model))

shutil.rmtree(SCRATCH, ignore_errors=True)
print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("PROYECTO COPIADO: ENCUENTRA SU MODELO")
