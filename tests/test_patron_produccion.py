# -*- coding: utf-8 -*-
"""
Patron en Produccion: sus anomalias pasan por el tracker, como en la demo.

El commit 6f36a9e hizo que las regiones de Patron entraran al MISMO tracker
que YOLO —confirmacion por frames, ID estable, conteo— pero solo cableo la
pantalla de la demo.  En Produccion el bloque de Patron decia que "analyze()
ya corrio arriba, antes de asociar", y no habia tal codigo: analyze() corria
DESPUES del tracker, dentro de process().  Resultado: el frame salia NO OK
por Patron, pero sin caja en pantalla y sin llegar al contador.

La prueba corre el bucle de Produccion de verdad (_loop), con una fuente y
un "detector" falsos que no nombran nada, sobre la tela de la demo.
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


from eyve.inference import detector as _det
_det.VideoSource.start = lambda self: False
_det.VideoSource.stop = lambda self: None
_det.VideoSource.read = lambda self: None
from eyve.ui.screens import production_screen as _ps
_ps.ProductionScreen._try_autoload_model = lambda self: None
_ps.ProductionScreen._auto_start_preview = lambda self: None
from eyve.core import updater as _U
_U.check_async = lambda cb: None

from eyve.core.project_manager import ClassDef, create_project
from eyve.demo.textile import TextilePattern
from eyve.inference.detector import InferenceResult
from eyve.modules import PatternModule
from eyve.production.ok_nok_logic import InspectionStatus
from eyve.ui.app import EyveApp

TMP = Path(tempfile.mkdtemp(prefix="eyve_patron_prod_"))
proj = create_project("PatronProd", TMP)
proj.classes = [ClassDef(name="rayon", kind="nok")]
proj.save()
app = EyveApp()
app.geometry("1280x900+40+40")
app.license.permite_checks_avanzados = lambda: True
app.license.es_pro = lambda: True
app.set_project(proj)
app.navigate("nav_production")
for _ in range(4):
    app.update()
scr = app._screens["nav_production"]


class Fuente:
    """La tela de la demo como camara: un frame por lectura."""
    def __init__(self):
        self.tela = TextilePattern(width=960, height=540, axis="x",
                                   motif="diamantes", weave="sarga",
                                   speed=110.0, seed=4)
        self.tela.offset = 300.0

    def read(self):
        self.tela.advance(1 / 16)
        return self.tela.frame()

    def stop(self):
        pass


class Detector:
    """Un YOLO que no nombra nada: lo que aparezca es cosa de Patron."""
    _running = True

    def push_frame(self, frame):
        pass

    def get_result(self):
        return InferenceResult(detections=[])

    def stop(self):
        pass


# cuantas veces se analiza cada frame
llamadas = {"n": 0}
_orig = PatternModule.analyze


def contar(self, frame):
    llamadas["n"] += 1
    return _orig(self, frame)


PatternModule.analyze = contar

fuente = Fuente()
scr._source = fuente
scr._worker = Detector()
scr._running = True
scr._paused = False
scr._counting.enabled = True
scr._counting.set_method("screen")


def vuelta():
    scr._loop()        # lo que reagenda con after() no corre: no hay update()


print("\n[1] Calibrar Patron desde la pantalla, con tela buena")
scr._calibrate_pattern()
for _ in range(16):
    vuelta()
check("queda calibrado", scr._pattern.calibrated and scr._pat_cal_left == 0)

print("\n[2] Tela buena: nada que reportar")
for _ in range(6):
    vuelta()
check("sin tracks de anomalia",
      not [t for t in scr._tracker.get_all()
           if t.label == PatternModule.ANOMALY_LABEL])

print("\n[3] Un offset cruzando: caja confirmada, contada, y NO OK")
t = fuente.tela
t.misregister(*t.screen_to_fabric(700, 270), size=170)
llamadas["n"] = 0
for _ in range(8):
    vuelta()
anom = [tr for tr in scr._tracker.get_all()
        if tr.label == PatternModule.ANOMALY_LABEL and tr.confirmed]
check("hay un track CONFIRMADO de la clase de Patron", len(anom) >= 1,
      str([(tr.id, tr.bbox) for tr in anom]))
check("el conteo lo ve", scr._counting.total >= 1, scr._counting.summary())
check("el frame sale NO OK", scr._last_result is not None
      and scr._last_result.status == InspectionStatus.NOT_OK,
      str(getattr(scr._last_result, "status", None)))
check("y lo dice Patron",
      "patr" in (scr._last_result.triggered_by or "").lower(),
      repr(scr._last_result.triggered_by))
check("se analiza UNA vez por frame, no dos", llamadas["n"] == 8,
      f"{llamadas['n']} analisis en 8 frames")

print("\n[4] Apagado, Patron no corre")
scr._pat_var.set(False)
scr._on_pattern_toggle()
llamadas["n"] = 0
for _ in range(3):
    vuelta()
check("sin analisis con el modulo apagado", llamadas["n"] == 0,
      str(llamadas["n"]))

scr._running = False
PatternModule.analyze = _orig
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
print("PATRON EN PRODUCCION: TODO PASO")
