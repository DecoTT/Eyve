# -*- coding: utf-8 -*-
"""
El repintado que reutiliza el PhotoImage.

Dos riesgos, y los dos son silenciosos:

  1. Que `paste()` no actualice nada.  La app seguiria corriendo y la demo
     se veria CONGELADA, sin ningun error.  Por eso aqui no se comprueba
     que "se pinto algo": se leen los pixeles del PhotoImage de Tk y se
     exige que cambien cuando cambia el frame.

  2. Que se rompa el mapeo lienzo→tela, y entonces el visitante dibuje
     desplazado.  No vale una prueba de ida y vuelta: componer un mapeo
     con su inverso se cancela aunque los dos esten mal (paso en esta
     tanda con las etiquetas giradas el doble). Se comprueban puntos
     ABSOLUTOS calculados a mano, con el encuadre descentrado a proposito.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("YOLO_OFFLINE", "1")

import numpy as np
import tkinter as tk
import customtkinter as ctk

from eyve.ui.screens.demo_screen import DemoScreen

fails = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


class Falso:
    """`_paint` solo necesita poder guardarse atributos."""


def pixel(root, photo, x, y):
    """Lee un pixel del PhotoImage de Tk, por su nombre."""
    v = root.tk.call(str(photo), "get", x, y)
    # Tk devuelve "r g b" o ya una tupla, segun la version
    if isinstance(v, str):
        v = v.split()
    return tuple(int(c) for c in v)


root = ctk.CTk()
root.geometry("1000x1000+0+0")
canvas = tk.Canvas(root, width=1000, height=1000, highlightthickness=0)
canvas.pack()
for _ in range(6):
    root.update()
    root.update_idletasks()
check("el lienzo mide lo pedido", canvas.winfo_width() >= 996,
      str(canvas.winfo_width()))

obj = Falso()
rojo = np.zeros((540, 960, 3), np.uint8)
rojo[:, :] = (0, 0, 255)            # BGR: rojo
verde = np.zeros((540, 960, 3), np.uint8)
verde[:, :] = (0, 255, 0)

print("\n[1] Primera pasada: se crea el PhotoImage y el item")
vista = DemoScreen._paint(obj, canvas, rojo, "_ph")
root.update_idletasks()
photo1 = getattr(obj, "_ph", None)
check("devuelve el mapeo", vista is not None, str(vista))
check("hay PhotoImage", photo1 is not None)
check("hay un solo item en el lienzo", len(canvas.find_all()) == 1,
      str(len(canvas.find_all())))
check("el PhotoImage lleva la imagen en rojo",
      pixel(root, photo1, 10, 10) == (255, 0, 0),
      str(pixel(root, photo1, 10, 10)))

print("\n[2] Segunda pasada: se reutiliza, no se crea otro")
DemoScreen._paint(obj, canvas, verde, "_ph")
root.update_idletasks()
photo2 = getattr(obj, "_ph")
check("es EL MISMO PhotoImage", photo2 is photo1)
check("sigue habiendo un solo item", len(canvas.find_all()) == 1,
      str(len(canvas.find_all())))

# ── la comprobacion que importa: los pixeles cambiaron de verdad ────────
check("paste() SI actualizo los pixeles",
      pixel(root, photo2, 10, 10) == (0, 255, 0),
      f"{pixel(root, photo2, 10, 10)} (si saliera rojo, la demo se veria "
      f"congelada sin dar ningun error)")

print("\n[3] Al cambiar el tamano del lienzo se rehace")
canvas.configure(width=700, height=700)
root.geometry("700x700+0+0")
for _ in range(6):
    root.update()
    root.update_idletasks()
DemoScreen._paint(obj, canvas, rojo, "_ph")
root.update_idletasks()
photo3 = getattr(obj, "_ph")
check("es un PhotoImage nuevo", photo3 is not photo1)
check("del tamano nuevo", photo3.width() != photo1.width(),
      f"{photo1.width()} -> {photo3.width()}")
check("y sigue habiendo un solo item", len(canvas.find_all()) == 1,
      str(len(canvas.find_all())))

print("\n[4] El mapeo lienzo→tela, en puntos absolutos")
# Lienzo 1000x1000 y frame 960x540: la escala es min(1000/960, 1000/540)
# = 1.0417, asi que la imagen sale 1000x562 y queda centrada en vertical
# con una banda de (1000-562)//2 = 219 px arriba.
canvas.configure(width=1000, height=1000)
root.geometry("1000x1000+0+0")
for _ in range(6):
    root.update()
    root.update_idletasks()
obj2 = Falso()
vista = DemoScreen._paint(obj2, canvas, rojo, "_ph2")
fw, fh, nw, nh, ox, oy = vista
check("la imagen se escala a lo ancho", (nw, nh) == (1000, 562), f"{nw}x{nh}")
check("y se centra en vertical", (ox, oy) == (0, 219), f"({ox},{oy})")

esc = DemoScreen.__new__(DemoScreen)
esc._view = vista
check("la esquina de la imagen cae en la esquina de la tela",
      esc._canvas_to_frame(0, 219) == (0, 0),
      str(esc._canvas_to_frame(0, 219)))
check("el centro de la imagen cae en el centro de la tela",
      esc._canvas_to_frame(500, 219 + 281) == (480, 270),
      str(esc._canvas_to_frame(500, 219 + 281)))
# ── el caso malo: fuera de la imagen no debe devolver un punto ──────────
check("por encima de la imagen devuelve None",
      esc._canvas_to_frame(500, 10) is None,
      str(esc._canvas_to_frame(500, 10)))
check("por debajo de la imagen devuelve None",
      esc._canvas_to_frame(500, 990) is None,
      str(esc._canvas_to_frame(500, 990)))

try:
    root.destroy()
    print("\n  OK    cierre limpio")
except Exception as e:
    fails.append(f"cierre: {e}")

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("REPINTADO: REUTILIZA, ACTUALIZA DE VERDAD Y NO MUEVE EL MAPEO")
