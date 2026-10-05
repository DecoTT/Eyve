# -*- coding: utf-8 -*-
"""
Cuanto cuesta repintar la demo en ventana y a pantalla completa.

Salio solo: al probar el kiosco en un monitor de 3440x1440, la prueba se
quedaba girando porque repintar los dos lienzos tardaba mas que los 33 ms
del bucle.  El coste del repintado crece con el area del lienzo, y en
kiosco el lienzo es el triple de grande.

Se mide el `_paint` REAL de DemoScreen (el mismo cv2.resize, el mismo
cvtColor, el mismo PhotoImage y el mismo create_image), no una
reimplementacion.  Varias repeticiones y se reporta la distribucion: una
sola medida de tiempo en Windows no dice nada.

    .venv\\Scripts\\python.exe tests\\perf_kiosco.py
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import tkinter as tk
import customtkinter as ctk

from eyve.demo.textile import TextilePattern
from eyve.ui.screens.demo_screen import DemoScreen

#: lienzos a comparar, medidos en la app de verdad con una ventana de
#: 1200x800 y con el kiosco en un monitor de 3440x1440
LIENZOS = [
    ("ventana 1200x800", 486, 574),
    ("kiosco 3440x1440", 1696, 1392),
]
REPS = 40


class Falso:
    """`_paint` solo escribe el PhotoImage en un atributo del objeto."""


def medir(root, cw, ch, frame) -> list[float]:
    # La ventana TIENE que ser del tamano del lienzo. Con una ventana mas
    # chica el lienzo queda recortado, Tk no dibuja lo que no se ve y la
    # medida sale plana con cualquier tamano — ese fue el primer intento y
    # daba 1.5 ms para los dos casos.
    canvas = tk.Canvas(root, width=cw, height=ch, highlightthickness=0)
    canvas.pack()
    root.geometry(f"{cw}x{ch}+0+0")
    for _ in range(6):
        root.update()
        root.update_idletasks()
    assert canvas.winfo_width() >= cw - 4, (
        f"el lienzo salio de {canvas.winfo_width()}px y se pidio {cw}px: "
        f"la ventana lo esta recortando y la medida no valdria")
    falso = Falso()
    tiempos = []
    for i in range(REPS + 5):
        t0 = time.perf_counter()
        DemoScreen._paint(falso, canvas, frame, "_photo")
        root.update_idletasks()
        dt = (time.perf_counter() - t0) * 1000
        if i >= 5:          # las primeras son de calentamiento
            tiempos.append(dt)
    canvas.destroy()
    return tiempos


def main() -> int:
    pat = TextilePattern(width=960, height=540, axis="x", motif="diamantes",
                         weave="sarga", speed=110.0)
    frame = pat.frame()

    root = ctk.CTk()
    root.geometry("200x100+0+0")
    root.update()

    print(f"repintado de UN lienzo, {REPS} repeticiones\n")
    print(f"{'lienzo':<20} {'mediana':>9} {'p90':>8} {'min':>7} {'max':>7}")
    resumen = {}
    for nombre, cw, ch in LIENZOS:
        ts = medir(root, cw, ch, frame)
        ts_ord = sorted(ts)
        med = statistics.median(ts)
        p90 = ts_ord[int(0.9 * len(ts_ord)) - 1]
        resumen[nombre] = med
        print(f"{nombre:<20} {med:>8.1f}ms {p90:>7.1f}ms "
              f"{min(ts):>6.1f}ms {max(ts):>6.1f}ms")

    root.destroy()

    print("\nla demo textil repinta DOS lienzos por frame:\n")
    print(f"{'lienzo':<20} {'2 lienzos':>11} {'+ 42 ms de tela e inspeccion':>30}")
    for nombre, med in resumen.items():
        dos = med * 2
        total = dos + 42.0
        print(f"{nombre:<20} {dos:>10.1f}ms {total:>21.1f}ms "
              f"= {1000 / total:>4.1f} fps")

    a, b = (resumen[n] for n, _, _ in LIENZOS)
    print(f"\nel kiosco multiplica el repintado por {b / a:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
