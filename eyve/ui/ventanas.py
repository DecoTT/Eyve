"""
Ventanas secundarias (CTkToplevel) que SE VEN aunque Produccion este corriendo.

El fallo
────────
Con la pantalla de Produccion a `prod_fps_cap: 0` (bucle a `after(1)`),
cualquier CTkToplevel —los cuatro dialogos de components/dialogs.py,
Ajustes, el gestor de modelos— se creaba, agarraba el foco y se quedaba
INVISIBLE, con el titulo todavia en "CTkToplevel".  `ask_string` y
`ask_confirm` esperaban para siempre a una ventana que nadie podia cerrar.

La causa esta en CustomTkinter 6.0, no en Eyve.  En Windows, el constructor
de CTkToplevel pinta la barra de titulo oscura asi:

    withdraw()      # esconder
    update()        # procesar TODO lo pendiente
    DwmSetWindowAttribute(...)
    after(5, deiconify)

`update()` no vuelve mientras haya eventos listos, y con un bucle a 1 ms
siempre los hay: cuando Tk termina de repintar lo que dejo una vuelta de
Produccion, la siguiente ya esta vencida.  El constructor no terminaba
nunca, ni el `title()` ni el `deiconify()` llegaban a correr.  Medido en
la app: con `after(33)` o `after(100)` el dialogo aparece; con `after(1)`,
nunca.  En un arnes sin repintado no se reproducia, porque sin nada que
procesar entre vuelta y vuelta `update()` si encuentra la cola vacia.

El arreglo
──────────
Se reemplaza ese metodo por uno que hace lo mismo SIN esconder la ventana
ni llamar a `update()`: pone el atributo de modo oscuro cuando la ventana
ya existe (al mapearse) y fuerza que Windows repinte el marco.  Se instala
una vez, desde `theme.apply()`.

La ventana principal (CTk) no se toca: cuando ya existe solo se esconde y
se restaura con un `after`, sin `update()`, y no se cuelga.
"""
from __future__ import annotations

import ctypes
import sys

import customtkinter as ctk

_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19
# SetWindowPos: solo recalcular y repintar el marco
_SWP = 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020   # NOSIZE NOMOVE NOZORDER NOACTIVATE FRAMECHANGED

_instalado = False


def _poner_barra(ventana, valor: int) -> bool:
    """Barra de titulo oscura (1) o clara (0). False si aun no hay ventana."""
    try:
        hwnd = ctypes.windll.user32.GetParent(ventana.winfo_id())
    except Exception:
        return True                   # ventana destruida: nada que hacer
    if not hwnd:
        return False
    v = ctypes.c_int(valor)
    for attr in (_DWMWA_USE_IMMERSIVE_DARK_MODE,
                 _DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1):
        if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(v), ctypes.sizeof(v)) == 0:
            break
    ctypes.windll.user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, _SWP)
    return True


def _barra_sin_update(self, color_mode: str) -> None:
    """Reemplazo de CTkToplevel._windows_set_titlebar_color, sin update()."""
    if (not sys.platform.startswith("win")
            or self._deactivate_windows_window_header_manipulation):
        return
    modo = color_mode.lower()
    if modo not in ("dark", "light"):
        return
    valor = 1 if modo == "dark" else 0
    if self.winfo_ismapped() and _poner_barra(self, valor):
        self._eyve_barra = None
        return
    self._eyve_barra = valor

    # Todavia no se ve: se pone al mapearse.  El <Map> de los hijos tambien
    # llega aqui (comparten bindtag con la ventana), por eso el valor
    # pendiente se borra en cuanto se aplica.
    if not getattr(self, "_eyve_barra_bind", False):
        self._eyve_barra_bind = True

        def al_mapear(_e=None):
            pendiente = getattr(self, "_eyve_barra", None)
            if pendiente is not None and _poner_barra(self, pendiente):
                self._eyve_barra = None

        self.bind("<Map>", al_mapear, add="+")


def instalar() -> None:
    """Una vez por proceso. Sin efecto fuera de Windows."""
    global _instalado
    if _instalado or not sys.platform.startswith("win"):
        return
    if not hasattr(ctk.CTkToplevel, "_windows_set_titlebar_color"):
        return                        # otra version de CustomTkinter
    ctk.CTkToplevel._windows_set_titlebar_color = _barra_sin_update
    _instalado = True
