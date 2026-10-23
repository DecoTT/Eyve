# -*- coding: utf-8 -*-
"""
Ensamble: partes contra la base, un veredicto por pieza.

Sin modelo: las detecciones se arman a mano y los frames son planos, así que
lo que se prueba es la lógica del módulo, no el detector.

  1. Referencia relativa a la base: la misma pieza movida en el encuadre
     sigue siendo OK (si se midiera en píxeles absolutos, cualquier pieza
     colocada un poco más a la derecha saldría NOT OK).
  2. Falta una parte → NOT OK con "falta"; parte movida → "fuera de
     posición". Cada uno con su caso bueno delante, para que una prueba que
     siempre diga NOT OK no pase.
  3. Por pieza: con mano (guante azul) no hay veredicto; NOT OK + corrección
     + reinspección OK cierra como "corregida"; una pieza que se va con
     NOT OK cierra como "rechazo". Y una pieza se cuenta UNA vez.
  4. La referencia sobrevive al job (config → apply_config).
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("YOLO_OFFLINE", "1")

import numpy as np

from eyve.inference.detector import Detection
from eyve.modules import AssemblyModule

fails = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


def det(name, x1, y1, x2, y2, c=0.9):
    return Detection(class_id=0, class_name=name, confidence=c, x1=x1, y1=y1, x2=x2, y2=y2)


def pieza(dx=0, antena="ok", etiqueta=True):
    """Base 600×500 en (100+dx, 100); antena vertical a la izquierda."""
    d = [det("base", 100 + dx, 100, 700 + dx, 600)]
    if antena == "ok":
        d.append(det("antena", 130 + dx, 150, 170 + dx, 580))
    elif antena == "acostada":
        d.append(det("antena", 130 + dx, 540, 560 + dx, 580))
    if etiqueta:
        d.append(det("etiqueta", 400 + dx, 160, 560 + dx, 260))
    return d


GRIS = np.full((480, 640, 3), 160, np.uint8)
MANO = GRIS.copy()
MANO[200:400, 300:500] = (220, 110, 40)        # azul nitrilo (BGR)

# ── 1. referencia ────────────────────────────────────────────────────────
m = AssemblyModule()
m.enabled = True
check("PREVIO: sin referencia no está calibrado", not m.calibrated)
for _ in range(10):
    m.learn(pieza())
check("aprende la referencia", m.finish_learning() and m.calibrated)
check("la referencia trae las dos partes", set(m.reference) == {"antena", "etiqueta"},
      str(sorted(m.reference)))

# ── 2. defectos ──────────────────────────────────────────────────────────
base_d, parts = m._split(pieza(dx=0))
check("pieza OK en su lugar → sin defectos", m.check(base_d, parts) == frozenset())
base_d, parts = m._split(pieza(dx=180))
check("la misma pieza movida 180 px → sigue OK", m.check(base_d, parts) == frozenset())
base_d, parts = m._split(pieza(etiqueta=False))
check("sin etiqueta → falta:etiqueta", m.check(base_d, parts) == {"falta:etiqueta"},
      str(m.check(base_d, parts)))
base_d, parts = m._split(pieza(antena="acostada"))
check("antena acostada → pos:antena", m.check(base_d, parts) == {"pos:antena"},
      str(m.check(base_d, parts)))
fuera = pieza() + [det("etiqueta", 1100, 160, 1200, 260, c=0.99)]
base_d, parts = m._split(fuera)
check("una etiqueta lejos de la base no cuenta como la de la pieza",
      m.check(base_d, parts) == frozenset())

# antena que sobresale de la carcasa (tarjeta de minisplit): su centro queda
# arriba de la base, pero es de la pieza
alta = [det("base", 100, 300, 700, 600), det("antena", 130, 80, 170, 320)]
_, parts = m._split(alta)
check("una parte que sobresale de la base sí es de la pieza", "antena" in parts, str(parts))


# ── 3. por pieza ─────────────────────────────────────────────────────────
def correr(m, pasos, t0):
    """pasos: lista de (frame, detecciones, segundos). 10 frames por segundo."""
    t = t0
    for frame, dets, seg in pasos:
        for _ in range(int(seg * 10)):
            m.update(frame, dets, now=t)
            t += 0.1
    return t


m.reset()
t = correr(m, [(MANO, pieza(antena="acostada"), 1.0)], 0)
check("con la mano en zona no hay veredicto", m.verdict is None and m.state == "mano",
      m.state)
t = correr(m, [(GRIS, pieza(antena="acostada"), 1.5)], t)
check("se va la mano → NOT OK por la antena", m.verdict == "NOT_OK" and "pos:antena" in m.defects,
      f"{m.verdict} {set(m.defects)}")
check("pide corregir (no rechazo) si la parte está movida",
      "corregir" in m.action_text(), m.action_text())
t = correr(m, [(MANO, pieza(antena="acostada"), 0.8), (GRIS, pieza(), 1.5)], t)
check("reinspección tras corregir → OK", m.verdict == "OK", str(m.verdict))
check("todavía no cuenta: la pieza sigue en la estación", sum(m.counts.values()) == 0)
t = correr(m, [(GRIS, [], 1.0)], t)
check("al irse cierra como corregida", m.counts == {"ok": 0, "corregida": 1, "rechazo": 0},
      str(m.counts))

t = correr(m, [(MANO, pieza(etiqueta=False), 1.0), (GRIS, pieza(etiqueta=False), 1.5)], t)
check("sin etiqueta → a RECHAZO", "RECHAZO" in m.action_text(), m.action_text())
t = correr(m, [(MANO, pieza(etiqueta=False), 0.5), (GRIS, [], 1.0)], t)
check("se va con NOT OK → rechazo", m.counts["rechazo"] == 1, str(m.counts))

t = correr(m, [(MANO, pieza(), 1.0), (GRIS, pieza(), 3.0), (GRIS, [], 1.0)], t)
check("pieza OK larga cuenta UNA vez", m.counts == {"ok": 1, "corregida": 1, "rechazo": 1},
      str(m.counts))

# una parte delgada que se corrige despacio casi no cambia la imagen: no hay
# mano ni movimiento, pero lo que se ve ya no es el veredicto
m3 = AssemblyModule()
m3.apply_config(m.config())
m3.enabled = True
t = correr(m3, [(MANO, pieza(antena="acostada"), 1.0), (GRIS, pieza(antena="acostada"), 1.5)], 0)
check("PREVIO: NOT OK por la antena", m3.verdict == "NOT_OK", str(m3.verdict))
t = correr(m3, [(GRIS, pieza(), 1.5)], t)
check("se corrige sin movimiento visible → reinspecciona: corregida",
      m3.verdict == "OK" and "Corregida" in m3.action_text(), m3.action_text())
t = correr(m3, [(GRIS, pieza(etiqueta=False), 0.2), (GRIS, pieza(), 1.0)], t)
check("un parpadeo de 0.2 s no reinspecciona", len(m3._history) == 2, str(m3._history))

# la pieza trae una parte del color del guante (relevador azul): no es mano
RELE = GRIS.copy()
RELE[150:190, 180:220] = (220, 110, 40)        # ~0.5 % del cuadro, dentro de la base
m4 = AssemblyModule()
m4.apply_config(m.config())
m4.enabled, m4.glove = True, "azul"
base_px = [det("base", 0, 0, 400, 300)]
m4.update(RELE, base_px, now=0)
check("PREVIO: sin la base, el relevador azul parece mano", m4._signals(RELE)[0])
check("una parte azul DENTRO de la pieza no es mano", not m4._signals(RELE, (0, 0, 400, 300))[0])
check("una mano grande encima de la pieza sí es mano (cámara cerrada)",
      m4._signals(MANO, (0, 0, 640, 480))[0])
BRAZO = RELE.copy()
BRAZO[320:480, 420:560] = (220, 110, 40)       # el brazo, afuera de la pieza
check("el guante que entra desde afuera sí es mano", m4._signals(BRAZO, (0, 0, 400, 300))[0])

# ── 4. job ───────────────────────────────────────────────────────────────
m.tolerance = 0.11
m2 = AssemblyModule()
check("PREVIO: módulo nuevo sin referencia", not m2.calibrated)
m2.apply_config(m.config())
check("la referencia vuelve del job", m2.reference == m.reference)
check("y la tolerancia", m2.tolerance == 0.11, str(m2.tolerance))

# ── 5. Pro e instancias persistentes ─────────────────────────────────────
from types import SimpleNamespace  # noqa: E402

from eyve.modules.assembly_module import from_tracks  # noqa: E402

check("Ensamble es módulo Pro (sin licencia → marca de agua)", AssemblyModule.pro is True)


def track(label, lost, confirmed=True):
    return SimpleNamespace(label=label, confidence=0.9, bbox=(10, 10, 50, 50),
                           frames_lost=lost, confirmed=confirmed)


nombres = [d.class_name for d in from_tracks(
    [track("antena", 0), track("etiqueta", 2), track("base", 1, confirmed=False),
     track("etiqueta", 9)])]
check("una instancia perdida 2 frames sigue contando (no parpadea)", "etiqueta" in nombres,
      str(nombres))
check("PREVIO: una sin confirmar no cuenta", "base" not in nombres, str(nombres))
check("una perdida 9 frames ya no cuenta (la quitaron de verdad)",
      nombres.count("etiqueta") == 1, str(nombres))

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ENSAMBLE: PARTES CONTRA LA BASE, UN VEREDICTO POR PIEZA")
