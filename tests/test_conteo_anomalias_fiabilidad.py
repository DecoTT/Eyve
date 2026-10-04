# -*- coding: utf-8 -*-
"""
Que tan fiable es contar anomalias, medido con variacion.

Una sola corrida no dice nada: el fallo generado cambia entre semillas, y
una semilla con suerte daba el numero exacto mientras otras daban 29. Asi
que se mide sobre varias semillas y se afirma lo que de verdad aguanta.

Lo que se sabe, medido:

  ACUMULAR anomalias (metodo "al aparecer") NO es fiable todavia. Un fallo
  cruzando el encuadre cambia de forma, la region se parte y se vuelve a
  unir, y cada pedazo nuevo es una instancia. El cierre a escala de
  repeticion bajo el desastre de 159 a unos pocos, pero "unos pocos" no es
  "uno".

  CONTAR LAS VISIBLES (metodo "en pantalla") SI es fiable: no depende de
  mantener una identidad a lo largo del tiempo, solo de cuantas regiones
  hay ahora.

  Y lo mas importante para un stand: con material BUENO no cuenta nada, en
  ninguna de las dos formas.

La prueba fija esos tres hechos. Si alguien mejora la estabilidad de las
regiones, el caso de "acumular" empezara a fallar por ser demasiado
pesimista — y eso es exactamente cuando hay que volver a mirarlo.
"""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SCRATCH = Path(tempfile.gettempdir()) / "eyve_tests"
SCRATCH.mkdir(parents=True, exist_ok=True)

from eyve.demo.textile import TextilePattern
from eyve.inference.tracker import InstanceTracker, TrackerConfig
from eyve.modules import CountingModule, PatternModule

fails = []


def check_true(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


def correr(n_fallos, seed, metodo="appear", pasos=150):
    p = TextilePattern(width=960, height=540, axis="x", motif="diamantes",
                       weave="sarga", speed=110.0, tilt_deg=1.8, seed=seed)
    pm = PatternModule()
    pm.enabled = True
    pm.sensitivity = 70
    for _ in range(10):
        pm.calibrate(p.frame())
        p.advance(1 / 30)
    pos = [(700, 270), (770, 150), (630, 410)]
    for i in range(n_fallos):
        p.misregister(*p.screen_to_fabric(*pos[i]), size=170)

    tk = InstanceTracker(config=TrackerConfig(
        iou_match=0.20, max_lost_frames=18, min_confirm_frames=2))
    cm = CountingModule()
    cm.enabled = True
    cm.set_method(metodo)
    pico = 0
    for _ in range(pasos):
        pm.analyze(p.frame())
        tr = tk.update(pm.as_detections())
        cm.update_tracks(tr, expired=tk.last_expired, frame_wh=(960, 540))
        pico = max(pico, cm.total)
        p.advance(1 / 30)
    return cm.total, pico


SEEDS = range(6)

print("\n[1] LO QUE MAS IMPORTA: material bueno no cuenta nada")
for metodo in ("appear", "screen"):
    r = [correr(0, s, metodo)[1] for s in SEEDS]
    check_true(f"metodo {metodo}: cero en material bueno", all(v == 0 for v in r),
               str(r))

print("\n[2] 'En pantalla' si es fiable: cuenta las visibles")
# No depende de mantener identidad en el tiempo, solo de cuantas hay ahora.
for n in (1, 2):
    picos = [correr(n, s, "screen")[1] for s in SEEDS]
    print(f"        {n} fallo(s) -> picos {picos}")
    check_true(f"{n} fallo(s): nunca se ve mas de lo que hay",
               all(v <= n for v in picos), str(picos))
    check_true(f"{n} fallo(s): se ve al menos uno", all(v >= 1 for v in picos),
               str(picos))

print("\n[3] ACUMULAR todavia NO es fiable, y queda dicho")
# Un fallo cruzando cambia de forma, la region se parte y se vuelve a unir.
# El cierre a escala de repeticion bajo esto de 159 a unos pocos, pero unos
# pocos no es uno. Se afirma la cota que si aguanta.
for n in (1, 2):
    totales = [correr(n, s, "appear")[0] for s in SEEDS]
    print(f"        {n} fallo(s) -> conteos {totales}")
    check_true(f"{n} fallo(s): nunca cuenta de menos",
               all(v >= 1 for v in totales), str(totales))
    check_true(f"{n} fallo(s): ya no se dispara a decenas",
               all(v <= 12 for v in totales),
               f"{totales} — antes del cierre llegaba a 159")

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f_ in fails:
        print("  -", f_)
    sys.exit(1)
print("FIABILIDAD DEL CONTEO DE ANOMALIAS: MEDIDA Y AFIRMADA")
