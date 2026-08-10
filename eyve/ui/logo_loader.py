"""
Logo loader — loads pre-made transparent-background PNGs and returns a
CTkImage that CustomTkinter swaps automatically on theme switch.

Expected files in eyve/assets/:
    logo_black.png  — black logo, transparent bg  → displayed in LIGHT mode
    logo_white.png  — white logo, transparent bg  → displayed in DARK  mode

If only one file is present the same image is used for both modes.
If neither file is present every function returns None (app shows text fallback).
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Optional

import customtkinter as ctk

_ASSETS      = Path(__file__).parent.parent / "assets"
_LOGO_BLACK  = _ASSETS / "logo_black.png"   # light mode variant
_LOGO_WHITE  = _ASSETS / "logo_white.png"   # dark  mode variant


@lru_cache(maxsize=16)
def get_logo(size: tuple[int, int] = (120, 120)) -> Optional[ctk.CTkImage]:
    """
    Return a CTkImage (light=black logo, dark=white logo).
    Size is (width, height) in logical pixels.
    Returns None if no logo file is found.
    """
    black_exists = _LOGO_BLACK.exists()
    white_exists = _LOGO_WHITE.exists()

    if not black_exists and not white_exists:
        return None

    try:
        from PIL import Image

        light_img = Image.open(_LOGO_BLACK).convert("RGBA") if black_exists else None
        dark_img  = Image.open(_LOGO_WHITE).convert("RGBA") if white_exists else None

        # fall back: use whichever is available for both modes
        if light_img is None:
            light_img = dark_img
        if dark_img is None:
            dark_img = light_img

        return ctk.CTkImage(
            light_image=light_img,
            dark_image=dark_img,
            size=size,
        )
    except Exception:
        return None


# ── convenience sizes ─────────────────────────────────────────────────────────

def sidebar_logo() -> Optional[ctk.CTkImage]:
    """110 × 110 px logo for the sidebar header."""
    return get_logo((110, 110))


def about_logo() -> Optional[ctk.CTkImage]:
    """64 × 64 px logo for the Settings / About block."""
    return get_logo((64, 64))


def logo_available() -> bool:
    return _LOGO_BLACK.exists() or _LOGO_WHITE.exists()
