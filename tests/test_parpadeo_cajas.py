# -*- coding: utf-8 -*-
"""
Las cajas de Producción no parpadean.

Lo arrastrábamos desde la 2.1: la instancia persistía (mismo #ID) pero su
caja se dibujaba, desaparecía y volvía. Eran dos cosas:

  1. En CPU el detector entrega un resultado cada varios ciclos de pantalla;
     en los ciclos sin resultado nuevo se mostraba el cuadro CRUDO.
  2. En los ciclos con resultado se dibujaba solo lo visto en ESE cuadro
     (update() devuelve frames_lost == 0): si el detector se saltaba la pieza
     una vez, la caja se borraba aunque la instancia siguiera viva.

Aquí un detector falso entrega resultado en 1 de cada 3 ciclos y se salta la
pieza en uno de ellos. Se mira cada cuadro que llega al canvas: la caja tiene
que estar en TODOS desde que la instancia se confirma, y tiene que irse cuando
la pieza se va de verdad (la persistencia no es eterna).
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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
from eyve.ui.screens import production_screen as _ps
_ps.ProductionScreen._try_autoload_model = lambda self: None
_ps.ProductionScreen._auto_start_preview = lambda self: None
from eyve.core import updater as _U
_U.check_async = lambda cb: None

from eyve.core.project_manager import ClassDef, create_project
from eyve.inference.detector import Detection, InferenceResult
from eyve.ui.app import EyveApp

TMP = Path(tempfile.mkdtemp(prefix="eyve_parpadeo_"))
proj = create_project("Parpadeo", TMP)
proj.classes = [ClassDef(name="pieza", kind="ok", color="#00ff00")]
proj.save()

app = EyveApp()
app.geometry("1280x900+40+40")
app.set_project(proj)
app.navigate("nav_production")
for _ in range(5):
    app.update()
scr = app._screens["nav_production"]

VERDE = np.array([0, 255, 0])            # BGR de "#00ff00"
CAJA = (100, 100, 220, 200)
FRAME = np.full((480, 640, 3), 90, np.uint8)


class Fuente:
    def read(self):
        return FRAME.copy()


class Detector:
    """Resultado solo en los ciclos de *agenda*; el resto, None (como en CPU)."""

    def __init__(self, agenda):
        self._running = True
        self.agenda = agenda
        self.tick = -1

    def push_frame(self, frame):
        self.tick += 1

    def get_result(self):
        if self.tick not in self.agenda:
            return None
        dets = ([Detection(0, "pieza", 0.9, *CAJA)] if self.agenda[self.tick] else [])
        return InferenceResult(detections=dets)


def hay_caja(img):
    # el borde izquierdo de la caja, a media altura (grosor 2)
    return bool(np.all(img[150, CAJA[0]] == VERDE))


def correr(agenda, ciclos, max_lost=2):
    vistos = []
    scr._update_canvas = lambda f: vistos.append(f.copy())
    scr._update_status = lambda *a, **k: None
    scr._session = None
    scr._source = Fuente()
    scr._worker = Detector(agenda)
    scr._running, scr._paused = True, False
    scr._last_result = None
    scr._tracker.reset()
    scr._tracker.configure(max_lost_frames=max_lost, min_confirm_frames=2)
    despues = scr.after
    scr.after = lambda *a, **k: None          # cada ciclo lo damos a mano
    try:
        for _ in range(ciclos):
            scr._loop()
    finally:
        scr.after = despues
        scr._running = False
    return vistos


print("\n[1] Pieza quieta; el detector entrega en 1 de cada 3 ciclos y se salta una vez")
# resultado en 0, 3, 6, 9, 12 (en el 6 NO ve la pieza); después la pieza se va
agenda = {0: True, 3: True, 6: False, 9: True, 12: True,
          15: False, 18: False, 21: False, 24: False}
v = correr(agenda, 27)
check("llegaron todos los cuadros al canvas", len(v) == 27, len(v))
confirmada = 3                 # 2 resultados con la pieza → confirmada en el ciclo 3
con_caja = [hay_caja(f) for f in v]
sin_resultado = [i for i in range(confirmada, 15) if i not in agenda]
check("PREVIO: hay ciclos sin resultado nuevo entre la confirmación y la salida",
      len(sin_resultado) >= 6, sin_resultado)
check("la caja está en los ciclos SIN resultado nuevo",
      all(con_caja[i] for i in sin_resultado),
      [i for i in sin_resultado if not con_caja[i]])
check("y en el ciclo en que el detector se saltó la pieza (6)", con_caja[6])
check("en ningún ciclo entre 3 y 14 desaparece",
      all(con_caja[confirmada:15]), [i for i in range(confirmada, 15) if not con_caja[i]])
check("antes de confirmarse no se dibuja (ciclos 0–2)", not any(con_caja[:confirmada]))
check("cuando la pieza se va de verdad, la caja se va (persistencia 2: desde el ciclo 21)",
      not any(con_caja[21:]), [i for i in range(21, 27) if con_caja[i]])

print("\n[3] Persistencia larga (30): la instancia sigue viva, pero sin fantasma en pantalla")
# la pieza se va en el 15; el job guarda la instancia 30 resultados (para la
# lógica), pero su caja se dibuja solo los 2 primeros resultados sin verla
v = correr(agenda, 27, max_lost=30)
con_caja = [hay_caja(f) for f in v]
check("PREVIO: la instancia sigue viva en el tracker al final",
      len(scr._tracker.get_all()) == 1, len(scr._tracker.get_all()))
check("con la pieza quieta la caja no parpadea (3 a 14)", all(con_caja[3:15]),
      [i for i in range(3, 15) if not con_caja[i]])
check("tras 2 resultados sin verla sigue (15 a 20): tapa saltos del detector", all(con_caja[15:21]),
      [i for i in range(15, 21) if not con_caja[i]])
check("al 3.er resultado sin verla la caja se va (21 en adelante): sin fantasma",
      not any(con_caja[21:]), [i for i in range(21, 27) if con_caja[i]])

print("\n[2] Sin ningún resultado todavía: cuadro crudo, sin inventar cajas")
v = correr({}, 5)
check("sin resultado no hay caja", not any(hay_caja(f) for f in v))

app.destroy()
shutil.rmtree(TMP, ignore_errors=True)

print("\n" + "=" * 60)
print("PRODUCCIÓN: LAS CAJAS NO PARPADEAN")
if fails:
    print(f"FALLARON {len(fails)}: {fails}")
    sys.exit(1)
print("todo bien")
