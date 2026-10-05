# -*- coding: utf-8 -*-
"""
Presupuesto de tiempo por frame de la demo textil, en ventana y en kiosco.

Salio de una sorpresa: al probar el kiosco en un monitor de 3440x1440 la
demo se arrastraba, y el culpable no era la inspeccion sino generar la
tela y repintar los lienzos.  Esto mide las tres partes por separado para
que no haya que adivinar cual duele.

Se mide el codigo REAL (`TextilePattern.frame`, `PatternModule.analyze` y
`DemoScreen._paint`), no una reimplementacion, y con varias repeticiones:
una sola medida de tiempo en Windows no dice nada.

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
from eyve.modules import PatternModule
from eyve.ui.screens.demo_screen import DemoScreen

#: lienzos medidos en la app de verdad
LIENZOS = [
    ("ventana 1200x800", 486, 574),
    ("kiosco 3440x1440", 1696, 1392),
]
REPS = 30


class Falso:
    """`_paint` solo necesita poder guardarse atributos."""


def resumen(ts):
    ts = sorted(ts)
    return statistics.median(ts), ts[int(0.9 * len(ts)) - 1]


def medir(fn, reps=REPS, calentar=5):
    ts = []
    for i in range(reps + calentar):
        t = time.perf_counter()
        fn()
        dt = (time.perf_counter() - t) * 1000
        if i >= calentar:
            ts.append(dt)
    return resumen(ts)


def medir_repintado(root, cw, ch, frame):
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

    def una():
        DemoScreen._paint(falso, canvas, frame, "_photo")
        root.update_idletasks()

    r = medir(una)
    canvas.destroy()
    return r


def main() -> int:
    pat = TextilePattern(width=960, height=540, axis="x", motif="diamantes",
                         weave="sarga", speed=110.0, tilt_deg=1.8, seed=7)
    pat.streak(300, 200, length=180, thickness=6)
    pat.blob(600, 300, size=90)
    frame = pat.frame()

    print("── las dos partes que no dependen del tamano del lienzo ──\n")
    f_med, f_p90 = medir(lambda: pat.frame())
    print(f"{'generar la tela':<22} {f_med:>7.1f}ms  (p90 {f_p90:.1f})")

    pm = PatternModule()
    pm.enabled = True
    pm.sensitivity = 70.0
    for _ in range(10):
        pm.calibrate(pat.frame())
        pat.advance(1 / 30)
    p_med, p_p90 = medir(lambda: pm.analyze(frame), reps=15)
    print(f"{'inspeccion (Patron)':<22} {p_med:>7.1f}ms  (p90 {p_p90:.1f})")

    root = ctk.CTk()
    root.geometry("200x100+0+0")
    root.update()

    print("\n── repintar UN lienzo ──\n")
    print(f"{'lienzo':<20} {'mediana':>9} {'p90':>8}")
    repintado = {}
    for nombre, cw, ch in LIENZOS:
        med, p90 = medir_repintado(root, cw, ch, frame)
        repintado[nombre] = med
        print(f"{nombre:<20} {med:>8.1f}ms {p90:>7.1f}ms")
    root.destroy()

    print("\n── presupuesto por frame (la demo repinta DOS lienzos) ──\n")
    print(f"{'':<20} {'tela':>7} {'Patron':>8} {'2 lienzos':>10} "
          f"{'TOTAL':>8} {'fps':>6}")
    for nombre, _, _ in LIENZOS:
        dos = repintado[nombre] * 2
        total = f_med + p_med + dos
        print(f"{nombre:<20} {f_med:>6.1f}ms {p_med:>7.1f}ms {dos:>9.1f}ms "
              f"{total:>7.1f}ms {1000/total:>5.1f}")

    print("\nEl bucle pide 30 fps (after de 33 ms). Si el TOTAL pasa de 33 ms,")
    print("la demo va mas lenta y update() deja de drenar la cola de eventos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
