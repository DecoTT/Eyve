# -*- coding: utf-8 -*-
"""
«Crear Paquete» tiene que servir en OTRA maquina.

El bug: data.yaml llevaba `path:` con la ruta absoluta de la PC de origen.
En otra PC, check_det_dataset de Ultralytics se iba a esa ruta y daba
FileNotFoundError.

Lo que hace que esta prueba distinga: la carpeta de origen SE BORRA antes de
validar. Si sigue en disco, el data.yaml viejo "funciona" en la misma PC
porque apunta a un origen que aun existe, y la prueba pasaria con el bug. El
control [2] mete la linea vieja a proposito y exige el FileNotFoundError,
para demostrar que el arnes si lo detecta.
"""
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SCRATCH = Path(tempfile.gettempdir()) / "eyve_tests"
SCRATCH.mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
import yaml

from eyve.core.project_manager import ClassDef, create_project, load_project
from eyve.training.dataset_builder import build_dataset
from eyve.training.training_packager import create_package

from ultralytics.data.utils import check_det_dataset

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


def proyecto(raiz: Path, n: int = 8):
    """Proyecto con *n* imagenes etiquetadas de verdad en disco."""
    (raiz / "projects").mkdir(parents=True)
    proj = create_project("Portable", raiz / "projects", target="prueba")
    proj.classes = [ClassDef(name="pieza", kind="nok")]
    proj.paths.create_all()
    proj.save()
    rng = np.random.default_rng(7)
    for i in range(n):
        img = proj.paths.raw_images / f"img_{i:03d}.jpg"
        img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img), rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        lbl = proj.paths.label_file(img)
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    return proj


base = SCRATCH / "paquete_portable"
if base.exists():
    shutil.rmtree(base)
origen = base / "pc_origen"
# Espacio y acento: una carpeta de usuario normal en la otra PC.
destino = base / "otra pc ñ"
destino.mkdir(parents=True)

proj = proyecto(origen)

# Un entrenamiento anterior deja sus .cache junto a labels/train y
# build_dataset no los borra: dentro van las rutas de esta PC.
build_dataset(proj)
cache = proj.paths.dataset / "labels" / "train.cache"
cache.write_text(str(proj.paths.dataset.resolve()), encoding="utf-8")

pkg = create_package(proj)
marca_origen = str(origen.resolve()).replace("\\", "/").lower()

print("\n[1] Nada en el paquete nombra la carpeta de origen")
with zipfile.ZipFile(pkg) as zf:
    nombres = zf.namelist()
    check_true("trae data.yaml", "dataset/data.yaml" in nombres)
    check_true("trae imagenes de train y val",
               any(n.startswith("dataset/images/train/") for n in nombres)
               and any(n.startswith("dataset/images/val/") for n in nombres))
    check("sin .cache de Ultralytics",
          [n for n in nombres if n.endswith(".cache")], [])
    for n in nombres:
        if n.endswith(".jpg"):
            continue
        txt = zf.read(n).decode("utf-8", "replace").replace("\\", "/")
        txt = txt.replace("\\\\", "/").lower()
        check_true(f"{n} no nombra el origen", marca_origen not in txt)
    meta = json.loads(zf.read("project_meta.json"))
    readme = zf.read("README.txt").decode("utf-8")
    check("metadatos del proyecto", meta["project"], "Portable")
    zf.extractall(destino)

datos = yaml.safe_load((destino / "dataset" / "data.yaml").read_text("utf-8"))
check("sin clave path en data.yaml", "path" in datos, False)
check("clases conservadas", datos["names"], {0: "pieza"})

# Sin esto la prueba no distingue nada: ver el docstring.
shutil.rmtree(origen)
check_true("el origen ya no existe", not origen.exists())

copia_yaml = destino / "dataset" / "data.yaml"
copia_ds = (destino / "dataset").resolve()

print("\n[2] Control: con la linea vieja, Ultralytics SI falla")
viejo = destino.parent / "control_linea_vieja"
shutil.copytree(destino / "dataset", viejo)
d = yaml.safe_load((viejo / "data.yaml").read_text("utf-8"))
d = {"path": str((origen / "x").as_posix()), **d}
(viejo / "data.yaml").write_text(yaml.safe_dump(d, sort_keys=False), "utf-8")
try:
    check_det_dataset(str(viejo / "data.yaml"))
    check_true("el control debio fallar", False)
except FileNotFoundError as e:
    check_true("FileNotFoundError con la ruta de origen", True, str(e)[:70])


def valida(nombre, dataset, cwd):
    antes = os.getcwd()
    os.chdir(cwd)
    try:
        info = check_det_dataset(dataset)
    except Exception as e:
        check_true(f"{nombre}: Ultralytics lo acepta", False,
                   f"{type(e).__name__}: {str(e)[:90]}")
        return
    finally:
        os.chdir(antes)
    check(f"{nombre}: train en la copia", Path(info["train"]),
          copia_ds / "images" / "train")
    check(f"{nombre}: val en la copia", Path(info["val"]),
          copia_ds / "images" / "val")
    check(f"{nombre}: clases", info["names"], {0: "pieza"})


print("\n[3] Ultralytics acepta la copia, sin el origen en disco")
valida("ruta absoluta", str(copia_yaml), base)
# Asi lo dice el README.txt del paquete.
valida("como el README", "dataset/data.yaml", destino)
valida("desde dataset/", "data.yaml", destino / "dataset")

print("\n[4] El README del paquete da el comando que se acaba de probar")
check_true("data=dataset/data.yaml", "data=dataset/data.yaml" in readme)

print("\n[5] Entrenar en Eyve no depende de la carpeta de trabajo")
# Sin `path:` en el yaml, la ruta del propio yaml es la que manda: si
# TrainManager pasara una relativa, un Ultralytics 8.3 la buscaria en su
# datasets_dir. Debe recibirla absoluta.
import eyve.training.train_manager as TM
import ultralytics

proj2 = proyecto(base / "pc_entrena", n=4)
visto = {}


class FakeYOLO:
    def __init__(self, path): pass
    def add_callback(self, *a, **kw): pass

    def train(self, **kw):
        visto.update(kw)
        raise RuntimeError("corte a proposito")


cwd_antes = os.getcwd()
os.chdir(base / "pc_entrena")
orig = ultralytics.YOLO
ultralytics.YOLO = FakeYOLO
try:
    rel = load_project(Path("projects/Portable"))
    check_true("proyecto abierto con ruta relativa",
               not Path(rel.root).is_absolute(), str(rel.root))
    TM.TrainManager(rel)._run(TM.TrainConfig(epochs=1))
finally:
    ultralytics.YOLO = orig
    os.chdir(cwd_antes)
check_true("data= es absoluto", Path(visto.get("data", "")).is_absolute(),
           str(visto.get("data")))
check("data= es el yaml del proyecto", Path(visto.get("data", "")),
      (base / "pc_entrena/projects/Portable").resolve()
      / "datasets" / "yolo_dataset" / "data.yaml")

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("PAQUETE PORTABLE: TODO PASO")
