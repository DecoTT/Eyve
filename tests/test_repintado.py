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

from eyve.core import config
from eyve.ui import theme as T
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


# La paleta del tema se aplica al crear EyveApp; aqui no hay app, asi
# que hay que aplicarla a mano o los colores quedan en cadena vacia y
# Tk se queja con 'unknown color name'.
T.apply("dark")

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

print("\n[4] El tope de ampliacion: no estirar lo que no tiene mas detalle")
# Lienzo 1000x1000 y frame 960x540. Sin tope la escala seria
# min(1000/960, 1000/540) = 1.0417 y la imagen saldria 1000x562; con el
# tope en 1.0 se queda en 960x540 y sobra margen por los cuatro lados.
canvas.configure(width=1000, height=1000)
root.geometry("1000x1000+0+0")
for _ in range(6):
    root.update()
    root.update_idletasks()
obj2 = Falso()
vista = DemoScreen._paint(obj2, canvas, rojo, "_ph2")
fw, fh, nw, nh, ox, oy = vista
check("el lienzo es MAS GRANDE que el frame (si no, no se prueba nada)",
      1000 > fw and 1000 > fh, f"lienzo 1000x1000 vs frame {fw}x{fh}")
check("no se amplia: la imagen se queda en su tamano",
      (nw, nh) == (fw, fh), f"{nw}x{nh} (frame {fw}x{fh})")
check("y queda centrada", (ox, oy) == (20, 230), f"({ox},{oy})")
check("por defecto el interruptor viene apagado",
      not config.get("demo_fill_screen", False))
check("y con el apagado el tope es 1.0", T.max_escala_demo() == 1.0,
      str(T.max_escala_demo()))

# ── el caso malo: encendido TIENE que ampliar ──────────────────────────
# Sin esto, "no se amplia" pasaria igual si el interruptor no estuviera
# conectado a nada: la prueba no distinguiria entre un tope que funciona
# y un tope clavado a 1.0.
# Se toca config.get y no config.set a proposito: set escribe en el
# config de verdad del usuario, y una prueba no debe dejar rastro.
_get = config.get
config.get = lambda k, d=None: (True if k == "demo_fill_screen"
                                else _get(k, d))
try:
    check("encendido, el tope desaparece", T.max_escala_demo() > 1.0,
          str(T.max_escala_demo()))
    obj_fill = Falso()
    v_fill = DemoScreen._paint(obj_fill, canvas, rojo, "_ph_fill")
    check("encendido, la imagen SI se amplia hasta llenar",
          v_fill[2] > v_fill[0], f"{v_fill[2]} > {v_fill[0]}")
    check("y sigue cabiendo en el lienzo",
          v_fill[2] <= 1000 and v_fill[3] <= 1000,
          f"{v_fill[2]}x{v_fill[3]}")
finally:
    config.get = _get
check("al apagarlo vuelve a no ampliar", T.max_escala_demo() == 1.0,
      str(T.max_escala_demo()))

print("\n[4c] El interruptor de Settings escribe la clave")
# Que la casilla exista no basta: tiene que escribir en la config, que
# es lo que lee el repintado.
guardado = {}
_set = config.set
config.set = lambda k, v: guardado.__setitem__(k, v)
try:
    from eyve.ui.screens.settings_popup import SettingsPopup
    pop = SettingsPopup(root, app=None)
    root.update()
    check("Settings tiene el interruptor", hasattr(pop, "_fill_var"))
    pop._fill_var.set(True)
    pop._toggle_fill()
    check("encenderlo guarda demo_fill_screen=True",
          guardado.get("demo_fill_screen") is True,
          str(guardado.get("demo_fill_screen")))
    pop._fill_var.set(False)
    pop._toggle_fill()
    check("apagarlo guarda demo_fill_screen=False",
          guardado.get("demo_fill_screen") is False,
          str(guardado.get("demo_fill_screen")))
    pop.destroy()
    root.update()
except Exception as e:
    import traceback
    traceback.print_exc()
    fails.append(f"Settings: {e}")
finally:
    config.set = _set

print("\n[4b] Reducir SI se permite: un lienzo pequeno encoge la imagen")
canvas.configure(width=480, height=480)
root.geometry("480x480+0+0")
for _ in range(6):
    root.update()
    root.update_idletasks()
obj3 = Falso()
v3 = DemoScreen._paint(obj3, canvas, rojo, "_ph3")
check("con lienzo chico la imagen se reduce", v3[2] < v3[0],
      f"{v3[2]} < {v3[0]}")

print("\n[5] El mapeo lienzo→tela, en puntos absolutos")
canvas.configure(width=1000, height=1000)
root.geometry("1000x1000+0+0")
for _ in range(6):
    root.update()
    root.update_idletasks()
esc = DemoScreen.__new__(DemoScreen)
esc._view = vista
check("la esquina de la imagen cae en la esquina de la tela",
      esc._canvas_to_frame(20, 230) == (0, 0),
      str(esc._canvas_to_frame(20, 230)))
check("el centro de la imagen cae en el centro de la tela",
      esc._canvas_to_frame(20 + 480, 230 + 270) == (480, 270),
      str(esc._canvas_to_frame(20 + 480, 230 + 270)))
# ── el caso malo: fuera de la imagen no debe devolver un punto ──────────
check("por encima de la imagen devuelve None",
      esc._canvas_to_frame(500, 10) is None,
      str(esc._canvas_to_frame(500, 10)))
check("por debajo de la imagen devuelve None",
      esc._canvas_to_frame(500, 990) is None,
      str(esc._canvas_to_frame(500, 990)))
check("a la izquierda de la imagen devuelve None",
      esc._canvas_to_frame(5, 500) is None,
      str(esc._canvas_to_frame(5, 500)))

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
