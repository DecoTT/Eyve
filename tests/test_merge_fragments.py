# -*- coding: utf-8 -*-
"""
La fusion de fragmentos debe unir pedazos de UNA pieza y NO unir piezas
vecinas.

El bug que la motivo: un rayon dibujado de un trazo salia con 2-3 cajas y
el conteo marcaba 3 en vez de 1. Lo peligroso de arreglarlo a lo bruto
(bajando el umbral de NMS de YOLO) es que las piezas PEGADAS —tortillas en
una charola, pines en un conector— se fusionarian tambien, y ahi el conteo
es el producto. Por eso cada caso que debe fusionar va con su caso que NO
debe.

Las cajas de fragmento de abajo son MEDIDAS, salidas del modelo de la demo
sobre un rayon largo de verdad. Las primeras que escribi me las invente y
resulto que no se parecian a las reales (su IoMin era 0.44 en vez de ~0.99),
asi que la prueba fallaba por la razon equivocada.

Limite conocido y a proposito: los dos EXTREMOS de un rayon muy largo no se
solapan (IoMin 0.00) y ninguna regla de solapamiento puede unirlos. Eso se
arregla en el dataset —generando rayones tan largos como los que dibuja la
gente— no aqui. La fusion es la red de seguridad, no la cura.
"""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SCRATCH = Path(tempfile.gettempdir()) / "eyve_tests"
SCRATCH.mkdir(parents=True, exist_ok=True)

from eyve.inference.tracker import (InstanceTracker, RawDetection,
                                    TrackerConfig, merge_fragments, _io_min)
from eyve.modules import CountingModule

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


def d(label, x1, y1, x2, y2, conf=0.9):
    return RawDetection(label, conf, x1, y1, x2, y2)


#: cajas MEDIDAS del modelo de la demo sobre un solo rayon de ~260 px
FRAGMENTOS_REALES = [d("rayon", 263, 191, 528, 355, 0.82),
                     d("rayon", 329, 267, 528, 351, 0.55),
                     d("rayon", 261, 193, 440, 296, 0.49),
                     d("rayon", 316, 229, 530, 351, 0.44)]

print("\n[1] _io_min distingue 'metida dentro' de 'al lado'")
check_true("contenida -> alto",
           _io_min((0, 0, 100, 100), (10, 10, 50, 50)) > 0.95,
           f"{_io_min((0,0,100,100),(10,10,50,50)):.2f}")
check_true("vecinas sin traslape -> 0",
           _io_min((0, 0, 50, 50), (50, 0, 100, 50)) == 0.0)
check_true("vecinas que se rozan -> bajo",
           _io_min((0, 0, 50, 50), (42, 0, 92, 50)) < 0.25,
           f"{_io_min((0,0,50,50),(42,0,92,50)):.2f}")
# los fragmentos reales, contra la caja grande
grande = FRAGMENTOS_REALES[0].bbox
for frag in FRAGMENTOS_REALES[1:]:
    v = _io_min(grande, frag.bbox)
    check_true(f"fragmento real {frag.bbox} -> alto", v > 0.9, f"{v:.2f}")

print("\n[2] SI fusiona: los fragmentos reales de un rayon")
out = merge_fragments(list(FRAGMENTOS_REALES))
check("4 fragmentos reales -> una instancia", len(out), 1)
check("la caja cubre todos",
      (out[0].x1, out[0].y1, out[0].x2, out[0].y2), (261, 191, 530, 355))
check("conserva la clase", out[0].label, "rayon")
check("se queda con la confianza mas alta", out[0].confidence, 0.82)

print("\n[3] NO fusiona: piezas pegadas de la misma clase")
fila = [d("tortilla", 0, 0, 100, 100),
        d("tortilla", 92, 0, 192, 100),
        d("tortilla", 184, 0, 284, 100),
        d("tortilla", 276, 0, 376, 100)]
check("4 tortillas pegadas siguen siendo 4", len(merge_fragments(fila)), 4)

fila2 = [d("pieza", 0, 0, 100, 100),
         d("pieza", 55, 0, 155, 100),
         d("pieza", 110, 0, 210, 100)]
check("piezas con 45% de traslape siguen separadas",
      len(merge_fragments(fila2)), 3)

# dos piezas una encima de otra en vertical (apiladas), que se rozan
apiladas = [d("caja", 0, 0, 100, 100), d("caja", 0, 88, 100, 188)]
check("piezas apiladas siguen separadas", len(merge_fragments(apiladas)), 2)

print("\n[4] NO fusiona clases distintas aunque se encimen")
mixto = [d("mancha", 100, 100, 200, 200), d("rayon", 110, 110, 150, 150)]
check("clases distintas no se tocan", len(merge_fragments(mixto)), 2)

print("\n[5] Casos de borde")
check("lista vacia", merge_fragments([]), [])
check("una sola deteccion", len(merge_fragments([d("a", 0, 0, 10, 10)])), 1)
check("iomin=0 desactiva la fusion",
      len(merge_fragments(list(FRAGMENTOS_REALES), iomin=0.0)), 4)
check("iomin=1 solo lo totalmente contenido",
      len(merge_fragments([d("a", 0, 0, 100, 100), d("a", 10, 10, 50, 50)],
                          iomin=1.0)), 1)

print("\n[6] Limite conocido: los extremos de un rayon muy largo no se tocan")
# medidas reales de un trazo de ~380 px: los dos extremos dan IoMin 0.00
extremos = [d("rayon", 294, 192, 468, 291), d("rayon", 501, 289, 686, 416)]
check_true("los extremos no se solapan",
           _io_min(extremos[0].bbox, extremos[1].bbox) == 0.0)
check("la fusion NO los une (y no debe inventarlo)",
      len(merge_fragments(extremos)), 2)
print("        -> esto se arregla generando rayones mas largos en el dataset,")
print("           no bajando el umbral: bajarlo fusionaria piezas vecinas.")

print("\n[7] Integrada en el tracker: un objeto fragmentado = un ID")
tk = InstanceTracker(config=TrackerConfig(min_confirm_frames=1,
                                          merge_iomin=0.65))
for _ in range(5):
    tracks = tk.update(list(FRAGMENTOS_REALES))
check("4 cajas del mismo objeto -> 1 track", len(tracks), 1)

print("\n[8] Y el caso que NO debe romper: 4 tortillas = 4 tracks")
tk2 = InstanceTracker(config=TrackerConfig(min_confirm_frames=1,
                                           merge_iomin=0.65))
for _ in range(5):
    tracks2 = tk2.update(list(fila))
check("4 tortillas pegadas -> 4 tracks", len(tracks2), 4)

print("\n[9] Se puede apagar desde la config")
tk3 = InstanceTracker(config=TrackerConfig(min_confirm_frames=1,
                                           merge_iomin=0.0))
for _ in range(5):
    tracks3 = tk3.update(list(FRAGMENTOS_REALES))
check("con merge_iomin=0 vuelven los 4 tracks", len(tracks3), 4)
tk3.configure(merge_iomin=0.65)
check("configure acepta merge_iomin", tk3.config.merge_iomin, 0.65)

print("\n[10] El conteo, que es lo que el usuario ve")
for iomin, esperado, etiqueta in [(0.0, 4, "sin fusion (el bug)"),
                                  (0.65, 1, "con fusion (arreglado)")]:
    tk4 = InstanceTracker(config=TrackerConfig(min_confirm_frames=2,
                                               merge_iomin=iomin))
    cm = CountingModule(); cm.enabled = True; cm.set_method("appear")
    for _ in range(6):
        tr = tk4.update(list(FRAGMENTOS_REALES))
        cm.update_tracks(tr, expired=tk4.last_expired, frame_wh=(960, 540))
    check(f"un rayon, {etiqueta}", cm.total, esperado)

# y contar tortillas sigue dando 4 con la fusion encendida
tk5 = InstanceTracker(config=TrackerConfig(min_confirm_frames=2,
                                           merge_iomin=0.65))
cm5 = CountingModule(); cm5.enabled = True; cm5.set_method("screen")
for _ in range(6):
    tr = tk5.update(list(fila))
    cm5.update_tracks(tr, frame_wh=(960, 540))
check("4 tortillas pegadas siguen contando 4", cm5.total, 4)

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("FUSION DE FRAGMENTOS: TODO PASO")
