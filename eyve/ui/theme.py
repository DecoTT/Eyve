"""Eyve 2.1 visual theme — dark and light palettes."""
from __future__ import annotations
import customtkinter as ctk

from eyve.core import config

# ── palettes ──────────────────────────────────────────────────────────────────
_DARK = {
    "BG_DARK":  "#0f1117",
    "BG_CARD":  "#1a1d27",
    "BG_PANEL": "#13151f",
    "BG_INPUT": "#222535",
    "TEXT_PRI": "#f0f2f5",
    "TEXT_SEC": "#8b92a8",
    "TEXT_DIM": "#4a5068",
    "BORDER":   "#2a2e42",
}

_LIGHT = {
    "BG_DARK":  "#e8ecf5",   # main bg — deeper blue-gray so cards/buttons pop
    "BG_CARD":  "#f5f7ff",   # cards/panels — off-white, distinct from bg
    "BG_PANEL": "#d5daf0",   # sidebar (clearly darker for separation)
    "BG_INPUT": "#c4ccdf",   # inputs & secondary buttons
    "TEXT_PRI": "#0d0f1a",   # primary text
    "TEXT_SEC": "#3a4265",   # secondary text
    "TEXT_DIM": "#58637f",   # dim text — passes AA contrast
    "BORDER":   "#5c6fa8",   # borders/outlines — dark enough to see on light bg
}

# ── current mode ──────────────────────────────────────────────────────────────
_MODE = "dark"

# ── colour globals (updated by set_mode / apply) ──────────────────────────────
BG_DARK = BG_CARD = BG_PANEL = BG_INPUT = ""
TEXT_PRI = TEXT_SEC = TEXT_DIM = BORDER = ""

# ── accent / status (invariant across modes) ──────────────────────────────────
ACCENT   = "#00e676"
ACCENT2  = "#00bcd4"
WARN     = "#ffb300"
DANGER   = "#ff4444"

COLOR_OK     = "#00e676"
COLOR_NOK    = "#ff4444"
COLOR_REVIEW = "#ffb300"
COLOR_NONE   = "#4a5068"
COLOR_ERROR  = "#ff6e6e"

# ── font sizes ────────────────────────────────────────────────────────────────
FONT_XS  = 10
FONT_SM  = 12
FONT_MD  = 14
FONT_LG  = 17
FONT_XL  = 22
FONT_XXL = 32

SIDEBAR_W = 180

#: Tope de ampliacion del frame de la demo cuando NO se llena la
#: pantalla, que es lo normal.  Lo elige max_escala_demo() segun la
#: config; aqui solo esta el valor.
#:
#: Cuanto se permite AMPLIAR el frame de la demo al pintarlo.
#:
#: La tela se genera a 960x540.  En un monitor grande el lienzo pide
#: 1696 px de ancho, y estirar hasta ahi triplica los pixeles que hay
#: que volcar a Tk —de 2.8 a 18.9 ms por lienzo, y la demo pinta dos
#: por frame— sin anadir ni un detalle, porque no hay mas detalle que
#: anadir.  Con el tope en 1.0 la imagen se queda en su tamano y sobra
#: margen alrededor.
#:
#: En pantallas donde el panel mide 960 px o menos esto no cambia nada:
#: ahi ya se reducia, y reducir si se permite.  Subirlo da una demo mas
#: grande y mas lenta; es una decision de como se ve, no un ajuste
#: tecnico.
MAX_ESCALA_DEMO = 1.0

#: Con el interruptor encendido no hay tope de verdad: se llena el
#: panel.  Un numero grande es mas simple que un None que haya que
#: filtrar en el min() de cada repintado.
_SIN_TOPE = 1e6


def max_escala_demo() -> float:
    """
    Cuanto puede ampliarse el frame de la demo, segun la config.

    Se consulta en CADA repintado —es una busqueda en un dict, no
    cuesta nada— para que cambiarlo en Settings se vea al momento y no
    haya que reiniciar en mitad de una expo.
    """
    if config.get("demo_fill_screen", False):
        return _SIN_TOPE
    return MAX_ESCALA_DEMO


def _apply_palette(mode: str) -> None:
    global BG_DARK, BG_CARD, BG_PANEL, BG_INPUT, TEXT_PRI, TEXT_SEC, TEXT_DIM, BORDER
    p = _LIGHT if mode == "light" else _DARK
    BG_DARK  = p["BG_DARK"]
    BG_CARD  = p["BG_CARD"]
    BG_PANEL = p["BG_PANEL"]
    BG_INPUT = p["BG_INPUT"]
    TEXT_PRI = p["TEXT_PRI"]
    TEXT_SEC = p["TEXT_SEC"]
    TEXT_DIM = p["TEXT_DIM"]
    BORDER   = p["BORDER"]


def set_mode(mode: str) -> None:
    """Switch between 'dark' and 'light'. Updates globals and ctk."""
    global _MODE
    _MODE = mode
    _apply_palette(mode)
    ctk.set_appearance_mode(mode)


def get_mode() -> str:
    return _MODE


def apply(mode: str | None = None) -> None:
    """Configure CustomTkinter and palette. Call once at startup."""
    global _MODE
    if mode is not None:
        _MODE = mode
    _apply_palette(_MODE)
    ctk.set_appearance_mode(_MODE)
    ctk.set_default_color_theme("dark-blue")


def font(size: int = FONT_MD, weight: str = "normal") -> tuple:
    return ("Segoe UI", size, weight)


def bold(size: int = FONT_MD) -> tuple:
    return font(size, "bold")
