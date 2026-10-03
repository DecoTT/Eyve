# -*- coding: utf-8 -*-
"""
Smoke test del dialogo de actualizacion y su cableado.

Simula los tres estados que puede ver un cliente —al dia, hay version
nueva, sin red— y comprueba que el dialogo dice lo correcto y habilita o
no el boton. Tambien comprueba el aviso clicable de la barra de estado.
"""
import sys
import types
import tempfile
from pathlib import Path

# El repo es el padre de tests/: nada de rutas absolutas, para que las
# pruebas corran en cualquier maquina y desde cualquier carpeta.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Salidas de las pruebas (hojas de contacto, proyectos temporales).
SCRATCH = Path(tempfile.gettempdir()) / "eyve_tests"
SCRATCH.mkdir(parents=True, exist_ok=True)

import customtkinter as ctk
from eyve.i18n import t, set_language
from eyve.ui import theme as T
T.apply("dark")
set_language("es")

from eyve.core import updater as U
from eyve.ui.screens.update_dialog import UpdateDialog, offer_update
from eyve.ui.components.status_bar import StatusBar

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


root = ctk.CTk()
root.geometry("900x600")

print("\n[1] Hay version nueva")
info = U.UpdateInfo(available=True, version="2.2.0", current="2.1.0",
                    url="http://x/a.zip", sums_url="http://x/s.txt",
                    size=200000, notes="- conteo con 5 metodos\n- demo de expo")
d = UpdateDialog(root, info)
root.update()
check("dice la version nueva", d._state_lbl.cget("text"),
      t("upd_available", v="2.2.0"))
check("dice de donde a donde", d._ver_lbl.cget("text"),
      t("upd_from_to", a="2.1.0", b="2.2.0"))
check("boton habilitado", d._go_btn.cget("state"), "normal")
check("boton dice instalar", d._go_btn.cget("text"), t("upd_install"))
notas = d._notes.get("1.0", "end").strip()
check_true("muestra las notas del release", "5 metodos" in notas, notas[:50])
d._on_close()
root.update()

print("\n[2] Ya esta al dia")
info = U.UpdateInfo(available=False, version="2.1.0", current="2.1.0")
d = UpdateDialog(root, info)
root.update()
check("dice que esta al dia", d._state_lbl.cget("text"), t("upd_uptodate"))
check("boton deshabilitado", d._go_btn.cget("state"), "disabled")
d._on_close()
root.update()

print("\n[3] Sin red")
info = U.UpdateInfo(available=False, version="", current="2.1.0",
                    error="sin red")
d = UpdateDialog(root, info)
root.update()
check("dice que no pudo consultar", d._state_lbl.cget("text"), t("upd_offline"))
check("boton deshabilitado", d._go_btn.cget("state"), "disabled")
d._on_close()
root.update()

print("\n[4] No se puede cerrar a media instalacion")
info = U.UpdateInfo(available=True, version="2.2.0", current="2.1.0",
                    url="http://x/a.zip", sums_url="http://x/s.txt")
d = UpdateDialog(root, info)
root.update()
d._busy = True
d._on_close()
root.update()
check_true("el dialogo sigue abierto mientras instala", d.winfo_exists())
d._busy = False
d._on_close()
root.update()

print("\n[5] Resultado fallido se muestra y deja reintentar")
d = UpdateDialog(root, info)
root.update()
d._info = info
d._finish(U.ApplyResult(ok=False, error="el archivo descargado no coincide"))
root.update()
check_true("muestra el error", "no coincide" in d._state_lbl.cget("text"),
           d._state_lbl.cget("text")[:60])
check("ofrece reintentar", d._go_btn.cget("text"), t("upd_retry"))
check("boton habilitado para reintentar", d._go_btn.cget("state"), "normal")
d._on_close(); root.update()

print("\n[6] Resultado bueno pide reiniciar, y avisa si cambiaron librerias")
d = UpdateDialog(root, info)
root.update()
d._info = info
d._finish(U.ApplyResult(ok=True, replaced=["eyve"], requirements_changed=False))
root.update()
check("pide reiniciar", d._go_btn.cget("text"), t("upd_restart"))
check_true("dice que quedo listo", "2.2.0" in d._state_lbl.cget("text"),
           d._state_lbl.cget("text")[:60])
check_true("sin aviso de librerias",
           t("upd_deps_changed") not in d._state_lbl.cget("text"))
d._on_close(); root.update()

d = UpdateDialog(root, info)
root.update()
d._info = info
d._finish(U.ApplyResult(ok=True, replaced=["eyve"], requirements_changed=True))
root.update()
check_true("avisa que hay que correr setup.bat",
           t("upd_deps_changed") in d._state_lbl.cget("text"))
d._on_close(); root.update()

print("\n[7] Aviso clicable en la barra de estado")
bar = StatusBar(root)
bar.pack(fill="x")
root.update()
check("empieza vacio", bar._update_lbl.cget("text"), "")

pulsado = {"n": 0}
bar.set_update("Version 2.2.0 disponible", lambda: pulsado.__setitem__("n", 1))
root.update()
check("muestra el aviso", bar._update_lbl.cget("text"), "Version 2.2.0 disponible")
check("se vuelve clicable", str(bar._update_lbl.cget("cursor")), "hand2")
# CTkLabel.bind enlaza al label INTERNO, no al frame externo: un clic
# real cae en el interno, asi que ahi hay que generar el evento.
bar._update_lbl._label.event_generate("<Button-1>")
root.update()
check("el clic llama al callback", pulsado["n"], 1)

bar.set_update("")
root.update()
check("se puede borrar", bar._update_lbl.cget("text"), "")
check("deja de ser clicable", str(bar._update_lbl.cget("cursor")), "arrow")

print("\n[8] offer_update solo avisa cuando hay algo que ofrecer")


class FakeApp:
    def __init__(self, bar): self.status_bar = bar


bar.set_update("")
offer_update(FakeApp(bar), U.UpdateInfo(available=False, version="2.1.0"))
root.update()
check("sin version nueva no avisa", bar._update_lbl.cget("text"), "")
offer_update(FakeApp(bar), U.UpdateInfo(available=True, version="2.3.0"))
root.update()
check("con version nueva avisa", bar._update_lbl.cget("text"),
      t("upd_banner", v="2.3.0"))

# app sin barra de estado: no debe reventar
offer_update(types.SimpleNamespace(), U.UpdateInfo(available=True, version="2.3.0"))
print("  OK    offer_update sin barra de estado no revienta")

print("\n[9] should_check respeta la casilla y el intervalo")
from eyve.core import config as _cfg
prev_dis = _cfg.get("update_check_disabled", False)
prev_last = _cfg.get("update_last_check", 0)
try:
    _cfg.set("update_check_disabled", True)
    check("desactivado -> no consulta", U.should_check(), False)
    _cfg.set("update_check_disabled", False)
    _cfg.set("update_last_check", 0)
    check("nunca consultado -> si consulta", U.should_check(), True)
    import time as _t
    _cfg.set("update_last_check", _t.time())
    check("recien consultado -> no consulta", U.should_check(), False)
    _cfg.set("update_last_check", _t.time() - 25 * 3600)
    check("hace 25 h -> si consulta", U.should_check(), True)
finally:
    _cfg.set("update_check_disabled", prev_dis)
    _cfg.set("update_last_check", prev_last)

print("\n[10] Las cadenas existen en los dos idiomas")
from eyve.i18n.es import STRINGS as ES
from eyve.i18n.en import STRINGS as EN
claves = [k for k in ES if k.startswith("upd_") or k.startswith("settings_update")
          or k in ("settings_updates", "settings_version")]
for k in claves:
    if k not in EN:
        fails.append("falta en en: " + k)
check_true(f"{len(claves)} claves en ambos", len(claves) >= 18, str(len(claves)))

root.destroy()
print("\n" + "=" * 60)
if fails:
    print(f"FALLARON {len(fails)}:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("UI DEL UPDATER: TODO PASO")
