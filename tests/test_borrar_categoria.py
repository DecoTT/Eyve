# -*- coding: utf-8 -*-
"""
Borrar una categoria pide confirmacion.

Antes no preguntaba: un toque en la ✕ y la fila se iba. Y la pantalla
importaba los cuatro dialogos de dialogs.py sin llamar a ninguno, asi que
el import mentia sobre lo que hacia la pantalla.

Lo que, mal hecho, no se nota:

  * preguntar DESPUES de quitar la fila —la pregunta sale, el usuario dice
    que no, y la fila ya no esta—;
  * preguntar tambien cuando la categoria tiene imagenes etiquetadas, que
    es un caso en el que no se va a borrar de todas formas: el usuario
    confirma para nada;
  * cablear el boton a otro sitio: si la prueba llama a _delete_row en vez
    de pulsar la ✕, eso pasa desapercibido;
  * que falte la clave de traduccion: t() devuelve la clave tal cual, asi
    que el dialogo saldria con el texto "cls_del_confirm" y nadie se
    entera.

Cada caso bueno va con el malo: antes de exigir que la fila desaparezca se
exige que estuviera, y el "si" se mide contra el "no" con el mismo boton.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("YOLO_OFFLINE", "1")

fails = []


def check(name, cond, extra=""):
    print(f"  {'OK  ' if cond else 'FALLA'}  {name} {extra}")
    if not cond:
        fails.append(name)


import customtkinter as ctk
from eyve.i18n import t, set_language
from eyve.core.project_manager import ClassDef
from eyve.ui import theme as T
from eyve.ui.components import dialogs
from eyve.ui.screens import classes_screen as CS

T.apply("dark")

# ── el import, antes de tocar nada ────────────────────────────────────────
print("\n[1] El import dice la verdad")
# Que se importe ask_confirm no basta: tiene que ser EL de dialogs, no
# cualquier cosa con ese nombre. Y los que no se usan no deben estar.
check("ask_confirm viene de dialogs.py",
      CS.ask_confirm is dialogs.ask_confirm)
muertos = [n for n in ("show_error", "show_info", "ask_string")
           if hasattr(CS, n)]
check("no quedan dialogos importados sin usar", not muertos, str(muertos))
fuente = (REPO / "eyve/ui/screens/classes_screen.py").read_text(encoding="utf-8")
check("y ask_confirm se llama de verdad en el modulo",
      "ask_confirm(" in fuente.split("import", 1)[1])

print("\n[2] La traduccion existe en los dos idiomas")
# t() devuelve la clave cuando falta, sin avisar. Si no se mira esto, el
# dialogo puede salir diciendo "cls_del_confirm" y la prueba pasa igual.
for lang in ("es", "en"):
    set_language(lang)
    txt = t("cls_del_confirm", name="Rayon")
    check(f"[{lang}] no es la clave pelada", txt != "cls_del_confirm", repr(txt[:40]))
    check(f"[{lang}] mete el nombre dentro", "Rayon" in txt)
set_language("es")


# ── arnes: una pantalla de Categorias de verdad ──────────────────────────
class FakePaths:
    def __init__(self, root):
        self.raw_images = root

    def raw_class_dir(self, name):
        return self.raw_images / name


class FakeProject:
    name = "prueba"

    def __init__(self, root):
        self.paths = FakePaths(root)
        self.classes = []

    def save(self):
        pass


class FakeApp:
    def __init__(self, proj):
        self._p = proj

    def get_project(self):
        return self._p


import tempfile
tmp = Path(tempfile.mkdtemp(prefix="eyve_cat_"))
(tmp / "Rayon").mkdir()

root = ctk.CTk()
root.geometry("900x700+80+80")
proj = FakeProject(tmp)
pantalla = CS.ClassesScreen(root, app=FakeApp(proj))
pantalla.pack(fill="both", expand=True)
root.update()


def settle(n=3):
    for _ in range(n):
        root.update()
        root.update_idletasks()


def anadir(nombre):
    pantalla._new_name.delete(0, "end")
    pantalla._new_name.insert(0, nombre)
    pantalla._add_class()
    settle()


def boton_equis(fila):
    """La ✕ de verdad, no _delete_row. Si alguien la recablea, se nota."""
    for w in fila.winfo_children():
        try:
            if w.cget("text") == "✕":
                return w
        except Exception:
            pass
    return None


def nombres():
    return [r.name for r in pantalla._rows]


# ── lo que de verdad cambia ──────────────────────────────────────────────
preguntas = []
respuesta = {"v": False}
CS.ask_confirm = lambda master, msg: (preguntas.append(msg),
                                      respuesta["v"])[1]

print("\n[3] Diciendo que NO, la fila se queda")
anadir("Rayon")
anadir("Mancha")
check("PREVIO: estan las dos", nombres() == ["Rayon", "Mancha"], str(nombres()))
fila = pantalla._rows[1]
equis = boton_equis(fila)
check("PREVIO: la fila tiene su ✕", equis is not None)
respuesta["v"] = False
preguntas.clear()
equis.invoke()
settle()
check("pregunto", len(preguntas) == 1, str(len(preguntas)))
check("y el texto lleva el nombre",
      preguntas and "Mancha" in preguntas[0], repr(preguntas[0][:50]) if preguntas else "")
check("la fila SIGUE ahi", nombres() == ["Rayon", "Mancha"], str(nombres()))
check("y el contador no se movio",
      "2" in pantalla._count_lbl.cget("text")
      if hasattr(pantalla, "_count_lbl") else True)

print("\n[4] Diciendo que SI, se va — mismo boton")
respuesta["v"] = True
preguntas.clear()
equis.invoke()
settle()
check("pregunto otra vez", len(preguntas) == 1, str(len(preguntas)))
check("ahora si se fue", nombres() == ["Rayon"], str(nombres()))

print("\n[5] Con imagenes etiquetadas no pregunta siquiera")
# 'Rayon' tiene una imagen: el aviso de "en uso" corta antes. Preguntar
# aqui seria pedir confirmacion de algo que no se va a borrar igualmente.
(tmp / "Rayon" / "una.jpg").write_bytes(b"x")
fila_r = pantalla._rows[0]
preguntas.clear()
respuesta["v"] = True          # diria que si, para que solo el corte lo salve
boton_equis(fila_r).invoke()
settle()
check("no pregunto", len(preguntas) == 0, str(len(preguntas)))
check("y la fila sigue", nombres() == ["Rayon"], str(nombres()))
check("con el aviso de 'en uso' puesto",
      "Rayon" in pantalla._msg.cget("text"), repr(pantalla._msg.cget("text")[:50]))

print("\n[6] El dialogo de verdad devuelve lo que se pulsa")
# Lo de arriba usa un doble. Esto mira el ask_confirm autentico: que el
# boton de "no" devuelva False y el de "si" True. Sin esto, el doble
# podria estar mintiendo sobre la forma de la respuesta.
CS.ask_confirm = dialogs.ask_confirm
for etiqueta, quiero in ((t("no"), False), (t("yes"), True)):
    salida = {}

    def pulsar():
        for w in root.winfo_children():
            if w.winfo_class() == "Toplevel":
                for sub in w.winfo_children():
                    for b in sub.winfo_children():
                        try:
                            if b.cget("text") == etiqueta:
                                b.invoke()
                                return
                        except Exception:
                            pass

    root.after(400, pulsar)
    salida["r"] = dialogs.ask_confirm(root, "¿seguro?")
    settle()
    check(f"pulsar '{etiqueta}' devuelve {quiero}",
          salida["r"] is quiero, str(salida["r"]))

# ── cierre ────────────────────────────────────────────────────────────────
try:
    root.update()
    root.destroy()
    print("\n  OK    cierre limpio")
except Exception as e:
    fails.append(f"cierre: {e}")

import shutil
shutil.rmtree(tmp, ignore_errors=True)

print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("BORRAR CATEGORIA: PREGUNTA ANTES, Y NO PREGUNTA DE BALDE")
