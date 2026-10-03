# -*- coding: utf-8 -*-
"""
Un proyecto abierto con ruta RELATIVA debe entrenar en SU carpeta.

El bug: ultralytics resuelve un `project=` relativo contra su propio
directorio de runs, asi que un proyecto abierto como "projects/Demo" iba a
parar a "runs/detect/projects/Demo/runs/train". Despues TrainManager
buscaba los pesos en paths.runs/train —vacio— y la copia a models/best.pt
fallaba EN SILENCIO: se reportaba "listo" y el modelo del proyecto seguia
siendo el viejo.

No se entrena de verdad (tardaria una hora): se intercepta la llamada a
model.train() y se mira QUE RUTA recibe. Es lo unico que importa aqui.
"""
import sys
import os
import shutil
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SCRATCH = Path(tempfile.gettempdir()) / "eyve_tests"
SCRATCH.mkdir(parents=True, exist_ok=True)

from eyve.core.project_manager import ClassDef, create_project
from eyve.training.train_manager import TrainConfig, TrainManager

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


base = SCRATCH / "rutas"
if base.exists():
    shutil.rmtree(base)
(base / "projects").mkdir(parents=True)

proj = create_project("Demo", base / "projects", target="prueba de rutas")
proj.classes = [ClassDef(name="pieza", kind="nok")]
proj.paths.create_all()
proj.save()

print("\n[1] Con ruta ABSOLUTA, el project= que recibe ultralytics es absoluto")
cfg = TrainConfig(epochs=1)
mgr = TrainManager(proj)
visto = {}


def fake_train(self, **kw):
    visto.update(kw)
    raise RuntimeError("corte a proposito: solo queriamos ver project=")


# Interceptar: ni se carga YOLO ni se construye el dataset.
import eyve.training.train_manager as TM
TM.build_dataset = lambda project, since=None: project.paths.dataset_yaml


class FakeYOLO:
    def __init__(self, path): pass
    def add_callback(self, *a, **kw): pass
    def train(self, **kw):
        visto.update(kw)
        raise RuntimeError("corte a proposito")


import ultralytics
orig_yolo = ultralytics.YOLO
ultralytics.YOLO = FakeYOLO
try:
    mgr._run(cfg)
finally:
    ultralytics.YOLO = orig_yolo

got = Path(visto.get("project", ""))
check_true("project= es absoluto", got.is_absolute(), str(got))
check("apunta a runs/ del proyecto", got, proj.paths.runs.resolve())

print("\n[2] Con el proyecto abierto por ruta RELATIVA, sigue siendo absoluto")
# Este es el caso del bug: cwd distinto y ruta relativa.
cwd_antes = os.getcwd()
os.chdir(base)
try:
    from eyve.core.project_manager import load_project
    rel = load_project(Path("projects/Demo"))
    check_true("el proyecto quedo con ruta relativa (como en el bug)",
               not Path(rel.root).is_absolute(), str(rel.root))

    visto.clear()
    mgr2 = TrainManager(rel)
    ultralytics.YOLO = FakeYOLO
    try:
        mgr2._run(cfg)
    finally:
        ultralytics.YOLO = orig_yolo

    got2 = Path(visto.get("project", ""))
    check_true("project= sigue siendo absoluto", got2.is_absolute(), str(got2))
    check_true("no se anida bajo runs/detect",
               "runs" not in got2.parent.parts or "detect" not in got2.parts,
               str(got2))
    check("apunta a la carpeta real del proyecto",
          got2, (base / "projects" / "Demo" / "runs").resolve())
finally:
    os.chdir(cwd_antes)

print("\n[3] Si no aparecen los pesos, se dice — no se reporta 'listo'")
# El fallo silencioso: ultralytics "termina" pero no dejo weights/best.pt.


class YOLOSinPesos(FakeYOLO):
    def train(self, **kw):
        return object()          # termina sin dejar nada en disco


mgr3 = TrainManager(proj)
estados = []
mgr3.add_callback(lambda p: estados.append((p.status, p.message)))
ultralytics.YOLO = YOLOSinPesos
try:
    mgr3._run(cfg)
finally:
    ultralytics.YOLO = orig_yolo

final = estados[-1][0] if estados else "(sin eventos)"
check("reporta error, no 'done'", final, "error")
msg = estados[-1][1] if estados else ""
check_true("dice que faltan los pesos",
           "peso" in msg.lower() or "best.pt" in msg, msg[:80])
check_true("NO dejo un best.pt inventado en models/",
           not proj.paths.best_model.exists())

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("RUTAS DE ENTRENAMIENTO: TODO PASO")
