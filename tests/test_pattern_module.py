# -*- coding: utf-8 -*-
"""
El modulo Patron: encontrar lo que nunca vio, sin marcar tela buena.

La promesa es "no necesita haberlo visto antes", asi que la prueba no
puede limitarse a los defectos que yo mismo generaria: incluye fallos que
no estan en ninguna lista de clases (fantasma, offset, falta de tinta) y
tambien formas arbitrarias.

Y lo que de verdad decide si sirve: la tela LIMPIA no debe marcar nada.
Un detector de anomalias que grita en cada frame se apaga el primer dia.
"""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SCRATCH = Path(tempfile.gettempdir()) / "eyve_tests"
SCRATCH.mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np

from eyve.demo.textile import TextilePattern, MOTIFS, WEAVES
from eyve.modules.pattern_module import PatternModule

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


def tela(motif="diamantes", weave="sarga", tilt=1.6, seed=5):
    p = TextilePattern(width=960, height=540, axis="x", motif=motif,
                       weave=weave, tilt_deg=tilt, speed=110.0, seed=seed)
    p.offset = 350.0
    return p


def modulo(sens=50.0):
    m = PatternModule()
    m.enabled = True
    m.sensitivity = sens
    return m


def cubre(regiones, caja, min_frac=0.25):
    """True si alguna region solapa la caja esperada."""
    x1, y1, x2, y2 = caja
    area = max(1, (x2 - x1) * (y2 - y1))
    for rx1, ry1, rx2, ry2 in regiones:
        ix = max(0, min(x2, rx2) - max(x1, rx1))
        iy = max(0, min(y2, ry2) - max(y1, ry1))
        if ix * iy / area >= min_frac:
            return True
    return False


print("\n[1] Estima el periodo del estampado")
for motif in MOTIFS:
    p = tela(motif=motif, tilt=0.0)
    m = modulo()
    g = m._prep(p.frame())
    per = m.estimate_period(g)
    check_true(f"periodo estimado ({motif})", per is not None, str(per))
    if per:
        check_true(f"periodo razonable ({motif})",
                   4 <= per[0] <= 120 and 4 <= per[1] <= 120, str(per))

print("\n[2] LO MAS IMPORTANTE: tela limpia no marca nada")
falsos = 0
total = 0
for motif in MOTIFS:
    for weave in ("sarga", "tafetan", "canasta", "ninguno"):
        p = tela(motif=motif, weave=weave)
        m = modulo()
        for _ in range(3):
            r = m.analyze(p.frame())
            total += 1
            if r:
                falsos += 1
            p.advance(0.2)
print(f"       {falsos}/{total} frames limpios con falsa alarma")
check_true("tela limpia practicamente sin falsas alarmas",
           falsos <= total * 0.08, f"{falsos}/{total}")

def calibrado(motif, weave="sarga", sens=70, frames=8, metodo="periodo"):
    """Modulo calibrado con material BUENO de ese mismo material."""
    m = PatternModule()
    m.enabled = True
    m.set_method(metodo)
    m.sensitivity = sens
    b = tela(motif=motif, weave=weave)
    for _ in range(frames):
        m.calibrate(b.frame())
        b.advance(0.2)
    return m


print("\n[3] Calibracion: aprende el ruido del material que tiene enfrente")
m = calibrado("diamantes")
check_true("queda calibrado", m.calibrated)
check("cuenta los frames de calibracion", m.calibration_count, 8)
check_true("el piso aprendido es positivo", (m.baseline or 0) > 0,
           f"{m.baseline:.2f}")
check_true("el umbral queda por encima del piso",
           m.threshold > (m.baseline or 0), f"umbral={m.threshold:.2f}")
# el ruido del material bueno NO es el mismo en todos los estampados: por
# eso un umbral fijo no puede servir y hay que calibrar
bases = {mo: (calibrado(mo).baseline or 0) for mo in MOTIFS}
print("        piso de ruido por estampado:",
      {k: round(v, 1) for k, v in bases.items()})
# El piso va de ~1.0 a ~1.6 segun el estampado: un 60% de diferencia. No
# parece mucho hasta que se traduce a umbral fijo — uno de 2.2 seria 2.2x
# el ruido de las rayas y solo 1.4x el de los diamantes, es decir dos
# sensibilidades distintas con el mismo numero. Por eso se calibra.
check_true("el piso depende del material",
           max(bases.values()) > 1.4 * min(bases.values()),
           f"{min(bases.values()):.1f} .. {max(bases.values()):.1f}")
m.clear_calibration()
check_true("se puede borrar la calibracion", not m.calibrated)

print("\n[4] CAPACIDAD MEDIDA, incluido lo que NO alcanza")
# Estos numeros salen de medir sobre los 4 estampados, no de desearlos.
# El modulo es complementario de YOLO, no un sustituto: rayon y mancha son
# clases entrenadas, y lo que aporta Patron es offset, falta de tinta y
# todo lo que nadie listo.
ESPERADO = [
    ("rayon",       lambda p, f: p.streak(*f, length=160, thickness=6),   4),
    ("offset",      lambda p, f: p.misregister(*f, size=160),             4),
    ("mancha",      lambda p, f: p.blob(*f, size=90),                     4),
    ("falta tinta", lambda p, f: p.ink_starved(*f, size=150, severity=0.6), 3),
    ("faltante severo", lambda p, f: p.ink_starved(*f, size=150, severity=0.95), 3),
]
for nombre, hacer, minimo in ESPERADO:
    n = 0
    for motif in MOTIFS:
        m = calibrado(motif)
        p = tela(motif=motif)
        hacer(p, p.screen_to_fabric(480, 270))
        if m.analyze(p.frame()):
            n += 1
    check_true(f"{nombre}: al menos {minimo} de 4 estampados", n >= minimo,
               f"{n}/4")

print("\n[4b] El fantasma, que era el punto ciego")
# El metodo compara el patron contra copias desplazadas de si mismo, y un
# fantasma ES una copia desplazada del patron, asi que parecia una ceguera
# estructural: se detectaba en 1 de 4 estampados.
#
# No era eso. Era que la reticula llega girada a la camara y un
# desplazamiento de un periodo sobre una reticula girada 1.6 grados deja
# 1.5 px de deriva — suficiente para enterrar un fallo tenue. Enderezando
# la reticula antes de analizar, el fantasma sube a 3 de 4.
n = 0
for motif in MOTIFS:
    m = calibrado(motif)
    p = tela(motif=motif)
    p.ghost(*p.screen_to_fabric(480, 270), size=160)
    if m.analyze(p.frame()):
        n += 1
print(f"        fantasma detectado en {n}/4 estampados")
check_true("el fantasma ya no es el punto ciego", n >= 3, f"{n}/4")

print("\n[4bb] Estima la inclinacion de la reticula")
for real in (0.0, 1.6, -4.0, 7.0):
    mm = PatternModule()
    mm.enabled = True
    pp = TextilePattern(width=960, height=540, axis="x", motif="diamantes",
                        weave="sarga", tilt_deg=real, speed=110.0, seed=5)
    pp.offset = 350.0
    mm._prep(pp.frame())
    check_true(f"inclinacion real {real:+.1f}",
               abs(abs(mm._angle) - abs(real)) < 1.0,
               f"estimado {mm._angle:+.2f}")

print("\n[4bbb] Metodo LAYOUT: dice de que lado esta el error")
# Mide menos que el metodo periodo (7 de 24 contra 21 de 24 en la misma
# tabla), pero aporta algo que periodo no puede: distinguir tinta de MAS
# de tinta que FALTA. Para el operador no es lo mismo — una es rodillo
# sucio y la otra es tinta que se acabo.
n = 0
tipos = set()
for motif in MOTIFS:
    m = calibrado(motif, metodo="layout")
    p = tela(motif=motif)
    p.streak(*p.screen_to_fabric(480, 270), length=160, thickness=6)
    if m.analyze(p.frame()):
        n += 1
        tipos.update(k for k in m.kinds if k)
check_true("layout encuentra tinta de mas", n >= 3, f"{n}/4")
check_true("y la etiqueta como tal",
           PatternModule.KIND_EXTRA in tipos, str(tipos))
check("sin anomalias no inventa tipos",
      calibrado("diamantes", metodo="layout").kinds, [])

print("\n[4c] Tras calibrar, el material bueno no da falsas alarmas")
fa = tot = 0
for motif in MOTIFS:
    for weave in ("sarga", "tafetan", "canasta", "ninguno"):
        m = calibrado(motif, weave)
        b = tela(motif=motif, weave=weave)
        for _ in range(4):
            if m.analyze(b.frame()):
                fa += 1
            tot += 1
            b.advance(0.2)
print(f"        {fa}/{tot} falsas alarmas tras calibrar")
check_true("cero o casi cero falsas alarmas", fa <= tot * 0.03, f"{fa}/{tot}")

print("\n[5] Y algo que nadie programo como defecto: un garabato cualquiera")
p = tela()
fx, fy = p.screen_to_fabric(480, 270)
# una espiral, que no es ninguna de las primitivas del generador
p.begin_stroke("rayon")
import math
prev = None
for i in range(90):
    a = i * 0.35
    r = 4 + i * 0.9
    q = p.screen_to_fabric(int(480 + math.cos(a) * r),
                           int(270 + math.sin(a) * r))
    if prev:
        p.paint_segment(prev, q, radius=4)
    prev = q
p.end_stroke()
m = calibrado("diamantes")
regs = m.analyze(p.frame())
check_true("encuentra una espiral que nunca vio", len(regs) >= 1, str(regs[:2]))

print("\n[6] La sensibilidad hace lo que dice")
p = tela()
p.misregister(*p.screen_to_fabric(480, 270), size=170)
frame = p.frame()
# El CONTEO de regiones no es monotono: al bajar el umbral las manchas se
# funden en menos piezas mas grandes. Lo que si debe serlo es el AREA
# marcada, que es lo que el operador percibe como "mas sensible".
def area_marcada(sens):
    mm = calibrado("diamantes", sens=sens)
    return sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in mm.analyze(frame))

a_baja, a_alta = area_marcada(5), area_marcada(95)
check_true("mas sensibilidad marca mas area", a_alta >= a_baja,
           f"baja={a_baja} alta={a_alta}")
check_true("en el minimo no marca nada en tela limpia",
           len(calibrado("diamantes", sens=5).analyze(tela().frame())) == 0)

print("\n[7] Metodo 'referencia': aprende de frames buenos")
p = tela()
m = PatternModule(); m.enabled = True; m.set_method("referencia")
check("sin referencia no inventa nada", len(m.analyze(p.frame())), 0)
check("lo dice en el resumen", m.summary(), "sin referencia")
for _ in range(6):
    m.learn_reference(p.frame())
check("cuenta los frames aprendidos", m.reference_count, 6)
check("tela buena contra su referencia: limpio", len(m.analyze(p.frame())), 0)
p.ink_starved(*p.screen_to_fabric(480, 270), size=160, severity=0.8)
regs = m.analyze(p.frame())
check_true("un fallo contra la referencia si se marca", len(regs) >= 1,
           str(regs[:2]))

print("\n[8] El veredicto veta el frame, como la polaridad")
p = tela()
m = calibrado("diamantes")
v = m.process(p.frame(), [], p.frame())
check("tela limpia pasa", v.ok, True)
p.misregister(*p.screen_to_fabric(480, 270), size=170)
v = m.process(p.frame(), [], p.frame())
check("con fallo, veta", v.ok, False)
check("dice quien lo veto", v.triggered_by, "patrón")

print("\n[9] No revienta con entradas raras")
m = modulo()
check("frame None", m.analyze(None), [])
check("frame vacio", m.analyze(np.zeros((0, 0, 3), np.uint8)), [])
check("frame de un color (sin periodo)",
      m.analyze(np.full((540, 960, 3), 200, np.uint8)), [])
m.analyze(np.zeros((24, 32, 3), np.uint8))
print("  OK    frame diminuto no revienta")

# muestra visual
# Dos defectos, que es el caso real: con cuatro a la vez lo anomalo pasa
# de un tercio del encuadre y la referencia robusta se degrada (ver el
# limite documentado en el modulo).
p = tela(motif="diamantes", weave="sarga")
p.misregister(*p.screen_to_fabric(300, 380), size=150)
p.streak(*p.screen_to_fabric(720, 300), length=170, thickness=6)
# calibrado, que es como se usa de verdad
m = calibrado("diamantes", weave="sarga", sens=70)
f = p.frame()
m.analyze(f)
vis = f.copy()
m.draw(vis)
cv2.imwrite(str(SCRATCH / "patron_hallazgos.png"), vis)
print(f"\n  muestra: {SCRATCH / 'patron_hallazgos.png'}")

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f_ in fails:
        print("  -", f_)
    sys.exit(1)
print("MODULO PATRON: TODO PASO")
