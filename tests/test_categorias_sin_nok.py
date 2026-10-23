# -*- coding: utf-8 -*-
"""
Categorias todas OK: se guardan, con aviso.

Antes Guardar exigia al menos una NO OK. Eso choca con el esquema por
partes del modulo Ensamble (base, antena, etiqueta): el modelo solo dice
que hay y donde, y el veredicto lo da el modulo. Ahora se guarda y se
avisa en amarillo.

Lo que, mal hecho, no se nota:

  * mostrar el aviso pero no guardar —el mensaje sale y proj.classes
    sigue vacio—;
  * guardar sin avisar: la prueba de "se guardo" pasa igual;
  * avisar tambien cuando SI hay una NO OK;
  * aflojar de paso la otra regla: sin ninguna OK tiene que seguir
    bloqueado;
  * que falte la clave: t() devuelve "cls_saved_no_nok" pelado.

Se pulsa el boton Guardar de verdad, no _save.
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
from eyve.ui import theme as T
from eyve.ui.screens import classes_screen as CS

T.apply("dark")

print("\n[1] La traduccion existe en los dos idiomas")
for lang in ("es", "en"):
    set_language(lang)
    txt = t("cls_saved_no_nok")
    check(f"[{lang}] no es la clave pelada", txt != "cls_saved_no_nok", repr(txt[:40]))
    check(f"[{lang}] nombra el modulo", "Ensamble" in txt)
set_language("es")


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
        self.guardados = 0

    def save(self):
        self.guardados += 1


class FakeApp:
    def __init__(self, proj):
        self._p = proj

    def get_project(self):
        return self._p


import tempfile
tmp = Path(tempfile.mkdtemp(prefix="eyve_sinnok_"))

root = ctk.CTk()
root.geometry("900x700+80+80")


def settle(n=3):
    for _ in range(n):
        root.update()
        root.update_idletasks()


def pantalla_con(filas):
    """Pantalla nueva con las filas (nombre, tipo) agregadas por la UI."""
    proj = FakeProject(tmp)
    p = CS.ClassesScreen(root, app=FakeApp(proj))
    p.pack(fill="both", expand=True)
    settle()
    etiquetas = {"ok": t("cls_type_ok"), "nok": t("cls_type_nok")}
    for nombre, tipo in filas:
        p._new_kind.set(etiquetas[tipo])
        p._new_name.delete(0, "end")
        p._new_name.insert(0, nombre)
        p._add_class()
    settle()
    return p, proj


def boton_guardar(p):
    pila = [p]
    while pila:
        w = pila.pop()
        try:
            if w.cget("text") == t("cls_save"):
                return w
        except Exception:
            pass
        pila.extend(w.winfo_children())
    return None


def guardar(p):
    b = boton_guardar(p)
    check("PREVIO: se encontro el boton Guardar", b is not None)
    b.invoke()
    settle()
    return p._msg.cget("text"), p._msg.cget("text_color")


print("\n[2] Tres partes, todas OK: se guarda y avisa")
p, proj = pantalla_con([("base", "ok"), ("antena", "ok"), ("etiqueta", "ok")])
check("PREVIO: las tres filas son OK",
      [r.kind for r in p._rows] == ["ok", "ok", "ok"], str([r.kind for r in p._rows]))
check("PREVIO: nada guardado aun", proj.guardados == 0 and proj.classes == [])
txt, color = guardar(p)
check("se guardo el proyecto", proj.guardados == 1, str(proj.guardados))
check("con las tres categorias",
      [c.name for c in proj.classes] == ["base", "antena", "etiqueta"],
      str([c.name for c in proj.classes]))
check("el mensaje es el aviso", txt == t("cls_saved_no_nok"), repr(txt[:50]))
check("en color de aviso, no de exito", color == T.WARN, str(color))
p.destroy()

print("\n[3] Con una NO OK: mensaje normal, sin aviso")
p, proj = pantalla_con([("pieza", "ok"), ("rayon", "nok")])
txt, color = guardar(p)
check("se guardo", proj.guardados == 1, str(proj.guardados))
check("mensaje de guardado normal", txt == t("cls_saved_ok"), repr(txt[:50]))
check("en color de exito", color == T.ACCENT, str(color))
p.destroy()

print("\n[4] Sin ninguna OK sigue bloqueado")
p, proj = pantalla_con([("rayon", "nok"), ("mancha", "nok")])
txt, color = guardar(p)
check("NO se guardo", proj.guardados == 0 and proj.classes == [],
      f"{proj.guardados} {proj.classes}")
check("con el error de 'falta OK'", txt == t("cls_need_ok"), repr(txt[:50]))
p.destroy()

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
print("CATEGORIAS TODAS OK: SE GUARDAN, CON AVISO")
