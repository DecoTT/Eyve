# -*- coding: utf-8 -*-
"""
Las etiquetas sobre la imagen no salen con «??».

cv2.putText solo dibuja ASCII: en Produccion la etiqueta de Patron decia
«#3 anomal??a 100%», y pasaria igual con «rayón» o cualquier clase con
acento.  La polaridad pintaba «°», «→» y «⚠», y tambien salian como «?».

La prueba espia cv2.putText y mira el texto que de verdad le llega, y
compara pixeles: la etiqueta «anomalía» tiene que dibujarse igual que
«anomalia».
"""
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2
import numpy as np

fails = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


from eyve.core.texto_cv import ascii_cv

print("\n[1] ascii_cv")
for txt, esperado in (("anomalía", "anomalia"), ("rayón", "rayon"),
                      ("niño", "nino"), ("REF Q2 → OK", "REF Q2 -> OK"),
                      ("12.3°", "12.3 deg"), ("Q1 [80%] ⚠", "Q1 [80%] !"),
                      ("#3 pieza 97%", "#3 pieza 97%")):
    check(f"{txt!r} -> {esperado!r}", ascii_cv(txt) == esperado,
          repr(ascii_cv(txt)))

# ── espia: todo lo que llega a cv2.putText ────────────────────────────────
dibujado = []
_put = cv2.putText


def espia(img, text, *a, **k):
    dibujado.append(text)
    return _put(img, text, *a, **k)


cv2.putText = espia

from eyve.ui.screens.production_screen import ProductionScreen
from eyve.inference.polarity import PolarityAnalyzer


def todo_ascii(textos):
    return all(t.isascii() for t in textos)


def pista(label, id_=3):
    return types.SimpleNamespace(confirmed=True, label=label, id=id_,
                                 bbox=(60, 60, 260, 200), confidence=1.0)


falso = types.SimpleNamespace(_class_color=lambda l: (255, 120, 255))
frame = np.full((300, 400, 3), 90, np.uint8)

print("\n[2] Produccion: etiqueta de una clase con acento")
dibujado.clear()
con = ProductionScreen._annotate_tracks(falso, frame, [pista("anomalía")], None)
check("a cv2.putText solo le llega ASCII", todo_ascii(dibujado), repr(dibujado))
check("y dice anomalia, sin «?»",
      any("anomalia" in t for t in dibujado) and not any("?" in t for t in dibujado),
      repr(dibujado))
sin = ProductionScreen._annotate_tracks(falso, frame, [pista("anomalia")], None)
check("se dibuja igual que «anomalia», pixel a pixel", np.array_equal(con, sin))
dibujado.clear()
ProductionScreen._annotate_tracks(falso, frame, [pista("rayón", 7)], None)
check("«rayón» tambien", dibujado == ["#7 rayon 100%"], repr(dibujado))

print("\n[3] Produccion: la marca de agua")
dibujado.clear()
img = np.zeros((300, 500, 3), np.uint8)
ProductionScreen._draw_watermark(
    types.SimpleNamespace(_wm_text="Licencia de Eyve vencida - renuévala"), img)
check("solo ASCII", dibujado and todo_ascii(dibujado), repr(dibujado))
check("y se dibuja", img.any())

print("\n[4] Polaridad: «°», «→» y «⚠»")
dibujado.clear()
res = types.SimpleNamespace(quadrant=2, side="top", angle_deg=12.0,
                            confidence=0.8, is_correct=False)
PolarityAnalyzer.draw_on_frame(frame.copy(), (60, 60, 260, 200), res)
check("la etiqueta con ⚠ llega en ASCII", todo_ascii(dibujado), repr(dibujado))
check("sin «?»", not any("?" in t for t in dibujado), repr(dibujado))

cv2.putText = _put
print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f_ in fails:
        print("  -", f_)
    sys.exit(1)
print("TEXTO SOBRE LA IMAGEN: TODO PASO")
