# -*- coding: utf-8 -*-
"""
Jobs: los parámetros de producción de un SKU, guardados y recuperados.

Lo que se comprueba no es "guarda y carga" —eso es fácil— sino las tres
formas que tiene esto de fallar sin hacer ruido:

  1. `PatternModule.set_method()` llama a `reset()`, que BORRA la
     calibración. Si al aplicar un job se pone el método DESPUÉS de
     restaurarla, el job se guarda bien y se carga vacío: el operador cree
     que tiene su calibración y el módulo está mirando con un piso de
     ruido inventado.

  2. Un SKU real trae puntos, espacios o barras, y el nombre de archivo
     tiene que colapsarlos. "SKU 1" y "SKU/1" caen en el mismo archivo: dos
     piezas distintas inspeccionadas con los parámetros de la otra.

  3. Un .npy de una referencia vieja que sobrevive a un job que ya no la
     usa: el módulo compararía contra material que no es el suyo.

Cada caso bueno va con el malo, y las comprobaciones de ida y vuelta llevan
delante una que exige que los dos estados sean DISTINTOS — si no, "se
restauró" pasaría igual sin restaurar nada.
"""
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("YOLO_OFFLINE", "1")

import numpy as np

from eyve.core import job as J
from eyve.inference.tracker import InstanceTracker, TrackerConfig
from eyve.modules import CountingModule, PatternModule, PolarityModule

fails = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


def modulos():
    return [PolarityModule(), CountingModule(), PatternModule()]


TMP = Path(tempfile.mkdtemp(prefix="eyve_jobs_"))


print("\n[1] Los tres módulos saben decir y recibir su configuración")
for m in modulos():
    c = m.config()
    check(f"{m.key}: config() es serializable",
          all(isinstance(k, str) for k in c)
          and "enabled" in c and "target_class" in c, str(sorted(c))[:60])

print("\n[2] Ida y vuelta completa, con los módulos bien revueltos")
pol, cnt, pat = modulos()
pol.enabled, pol.target_class = True, "IC"
pol.set_method("notch"); pol.set_reference("left"); pol.set_arc_thickness(0.33)
cnt.enabled, cnt.target_class = True, "pieza"
cnt.set_method("zone"); cnt.zone = (10, 20, 300, 400)
cnt.zone_trigger, cnt.edge, cnt.direction = "exit", "left", "rev"
cnt.expect_min, cnt.expect_max = 2, 7
pat.enabled = True
pat.set_method("layout"); pat.sensitivity = 81.0; pat.min_area_pct = 0.5
tracker = InstanceTracker(config=TrackerConfig(iou_match=0.44, max_lost_frames=19,
                                               min_confirm_frames=5,
                                               process_conf_min=0.66))
mods = [pol, cnt, pat]
job = J.capture(mods, tracker, source={"camera_source": 2, "conf": 0.71},
                name="IC-74HC595 rev.B", notes="linea 3")
ruta = job.save(TMP)
check("el YAML existe", ruta.exists(), ruta.name)

# ── módulos NUEVOS, en valores por defecto ───────────────────────────────
pol2, cnt2, pat2 = modulos()
mods2 = [pol2, cnt2, pat2]
tracker2 = InstanceTracker()
# la guarda: si ya fueran iguales, "se restauró" no probaría nada
check("PREVIO: los módulos nuevos NO son como los guardados",
      (pol2.config() != pol.config() and cnt2.config() != cnt.config()
       and pat2.config() != pat.config()))
check("PREVIO: el tracker nuevo tampoco",
      tracker2.config.iou_match != tracker.config.iou_match)

cargado = J.Job.load(ruta)
J.apply(cargado, mods2, tracker2)

check("polaridad vuelve igual", pol2.config() == pol.config(),
      f"{pol2.config()} vs {pol.config()}")
check("conteo vuelve igual", cnt2.config() == cnt.config())
check("la zona vuelve como TUPLA, no como lista de YAML",
      cnt2.zone == (10, 20, 300, 400) and isinstance(cnt2.zone, tuple),
      f"{cnt2.zone!r}")
check("patrón vuelve igual", pat2.config() == pat.config())
check("el tracker vuelve igual",
      (tracker2.config.iou_match, tracker2.config.max_lost_frames,
       tracker2.config.min_confirm_frames, tracker2.config.process_conf_min)
      == (0.44, 19, 5, 0.66),
      str(tracker2.config))
check("la fuente y el umbral vuelven",
      cargado.source["camera_source"] == 2 and cargado.source["conf"] == 0.71)
check("el nombre con puntos y espacios se conserva entero",
      cargado.name == "IC-74HC595 rev.B", cargado.name)

print("\n[3] La calibración del Patrón sobrevive — la trampa del orden")
# set_method() llama a reset() y borra la calibración. Si apply_config la
# restaurara ANTES de fijar el método, aquí llegaría sin calibrar.
pat3 = PatternModule()
pat3.enabled = True
pat3.set_method("referencia")
rng = np.random.default_rng(7)
frame = (rng.normal(128, 12, (540, 960, 3)).clip(0, 255)).astype(np.uint8)
# La referencia PRIMERO: con el método "referencia", calibrar sin ella no
# tiene contra qué medir y no calibra nada.
pat3.learn_reference(frame)
for _ in range(6):
    pat3.calibrate(frame)
check("PREVIO: el patrón está calibrado", pat3.calibrated)
check("PREVIO: y tiene referencia aprendida", pat3.reference_image is not None)
base_guardado = pat3._baseline

job3 = J.capture([pat3], name="con-calibracion")
r3 = job3.save(TMP)
check("la calibración va en el YAML",
      bool(job3.modules["pattern"].get("calib")),
      str(job3.modules["pattern"].get("calib"))[:60])
check("y la fecha de calibración queda apuntada", bool(job3.calibrated_at),
      job3.calibrated_at[:19])
check("la referencia va en un .npy aparte",
      job3.ref_path(TMP).exists(), job3.ref_path(TMP).name)

# un módulo en OTRO método y sin calibrar
pat4 = PatternModule()
pat4.set_method("periodo")
check("PREVIO: el módulo destino NO está calibrado", not pat4.calibrated)
check("PREVIO: y está en otro método", pat4.method != "referencia")

J.apply(J.Job.load(r3), [pat4])
check("queda en el método guardado", pat4.method == "referencia", pat4.method)
check("Y SIGUE CALIBRADO (si no, set_method borró la calibración)",
      pat4.calibrated, f"calibrado={pat4.calibrated}")
check("con el mismo piso de ruido", pat4._baseline == base_guardado,
      f"{pat4._baseline} vs {base_guardado}")
check("y con la referencia aprendida de vuelta",
      pat4.reference_image is not None
      and np.array_equal(pat4.reference_image, pat3.reference_image))

print("\n[4] Una calibración a medias no deja el módulo medio calibrado")
pat5 = PatternModule()
for mala in ({"baseline": 1.0}, {"norm": [1, 2]}, {"baseline": "x", "norm": [1, 2]},
             {"baseline": 1.0, "norm": "no-es-lista"}):
    pat5._baseline = None
    try:
        ok = pat5.load_calibration(mala)
    except Exception as e:
        # Lanzar tambien es no aceptarla, pero conviene decirlo distinto:
        # una excepcion aqui subiria hasta la pantalla de Produccion.
        ok, roto = True, f"lanzo {type(e).__name__}"
    else:
        roto = ""
    check(f"calib rota {str(mala)[:28]:<30} → no calibrado",
          not ok and not pat5.calibrated, roto)
check("y una calib vacía tampoco rompe", not pat5.load_calibration(None))

print("\n[5] Dos SKU que colapsan al mismo archivo NO se pisan")
a = J.Job(name="SKU 1")
b = J.Job(name="SKU/1")
check("PREVIO: los dos nombres dan el mismo slug",
      J.slug(a.name) == J.slug(b.name), J.slug(a.name))
pa = a.save(TMP)
pb = b.save(TMP)
check("se guardan en archivos DISTINTOS", pa != pb, f"{pa.name} / {pb.name}")
check("y cada uno conserva su nombre",
      J.Job.load(pa).name == "SKU 1" and J.Job.load(pb).name == "SKU/1",
      f"{J.Job.load(pa).name!r} / {J.Job.load(pb).name!r}")
# volver a guardar el primero no debe crear un tercero
pa2 = a.save(TMP)
check("regrabar el primero reusa su archivo", pa2 == pa, pa2.name)

print("\n[6] Un .npy viejo no sobrevive a un job que ya no lo usa")
jr = J.Job(name="con-ref")
jr.reference_image = np.zeros((4, 4), np.float32)
jr.save(TMP)
check("PREVIO: el .npy está", jr.ref_path(TMP).exists())
jr.reference_image = None
jr.save(TMP)
check("al guardar sin referencia, el .npy se borra",
      not jr.ref_path(TMP).exists(),
      "si quedara, el módulo compararía contra material que ya no es el suyo")

print("\n[7] El listado y la búsqueda")
nombres = [j.name for j in J.list_jobs(TMP)]
check("lista los jobs guardados",
      {"IC-74HC595 rev.B", "con-calibracion", "SKU 1", "SKU/1"} <= set(nombres),
      str(nombres))
check("encuentra por nombre",
      J.find(TMP, "con-calibracion") is not None)
check("y no inventa los que no hay", J.find(TMP, "no-existe") is None)

print("\n[8] Qué módulos Pro pide el job")
mods_pro = modulos()
jp = J.Job(name="pide-pro", modules={"polarity": {"enabled": True},
                                     "counting": {"enabled": True}})
check("detecta la polaridad encendida",
      jp.pro_modules(mods_pro) == ["Polaridad"], str(jp.pro_modules(mods_pro)))
jp2 = J.Job(name="sin-pro", modules={"polarity": {"enabled": False},
                                     "counting": {"enabled": True}})
check("y no la reporta si está apagada", jp2.pro_modules(mods_pro) == [],
      str(jp2.pro_modules(mods_pro)))

print("\n[9] Un job de una versión más nueva no tira Producción abajo")
# `period` como entero hace que tuple(5) lance TypeError: es lo que de
# verdad ejercita la red de apply(). Sin un valor asi, quitar el try/except
# no cambiaria nada y la comprobacion no probaria nada.
raro = J.Job(name="del-futuro",
             modules={"pattern": {"enabled": True, "metodo_nuevo": 7,
                                  "sensitivity": 44.0, "period": 5},
                      "modulo_que_no_existe": {"enabled": True}})
pat6 = PatternModule()
try:
    J.apply(raro, [pat6])
    check("aplica lo que entiende y se salta lo demás",
          pat6.enabled and pat6.sensitivity == 44.0,
          f"enabled={pat6.enabled} sens={pat6.sensitivity}")
except Exception as e:
    check("aplica lo que entiende y se salta lo demás", False, str(e))

# ── y la red de apply(), con un modulo que lanza de verdad ──────────────
# El modulo ya es defensivo campo a campo, asi que para probar la red hace
# falta algo que falle de todas formas: un modulo roto, o uno de un plugin
# futuro. Sin esto, quitar el try/except de apply() no cambiaria nada.
class ModuloRoto:
    key = "roto"
    name = "Roto"
    pro = False
    aplicado = False

    def apply_config(self, cfg):
        ModuloRoto.aplicado = True
        raise RuntimeError("este modulo no sabe leer su config")

roto = ModuloRoto()
pat7 = PatternModule()
jr2 = J.Job(name="con-modulo-roto",
            modules={"roto": {"enabled": True},
                     "pattern": {"enabled": True, "sensitivity": 55.0}})
try:
    J.apply(jr2, [roto, pat7])
    check("un módulo que lanza no tira el resto del job", True)
except Exception as e:
    check("un módulo que lanza no tira el resto del job", False, str(e))
check("PREVIO: el módulo roto llegó a ejecutarse", ModuloRoto.aplicado)
check("y los demás módulos sí quedaron aplicados",
      pat7.enabled and pat7.sensitivity == 55.0,
      f"enabled={pat7.enabled} sens={pat7.sensitivity}")

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("JOBS: GUARDAN, CARGAN Y NO SE PISAN ENTRE SKU")
