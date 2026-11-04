# -*- coding: utf-8 -*-
"""
Rollos distintos de la misma tela, para no medir sobre lo aprendido.

TextilePattern siembra su ruido de hilo con semillas fijas (7 y 13): dos telas
con el mismo estampado y tejido son la MISMA tela, pixel a pixel, sin importar
la `seed`, que solo mueve los defectos.  Aprender y probar sobre eso seria
medir sobre lo aprendido aunque las posiciones cambiaran.

Aqui cada rollo cambia esas dos semillas: mismo estampado y mismo tejido, otro
hilo.  El rollo 0 es la tela tal cual, la de la medicion del video.
"""
from __future__ import annotations

import numpy as np

from eyve.demo.textile import TextilePattern

W, H = 960, 540


def rollo(motif, weave, numero=0, seed=None, **kw) -> TextilePattern:
    if numero == 0:
        return TextilePattern(width=W, height=H, axis="x", motif=motif,
                              weave=weave, speed=110.0, seed=seed, **kw)
    real = np.random.default_rng

    def otra(s=None):
        return real(None if s is None else int(s) + 1000 * int(numero))

    np.random.default_rng = otra
    try:
        return TextilePattern(width=W, height=H, axis="x", motif=motif,
                              weave=weave, speed=110.0, seed=seed, **kw)
    finally:
        np.random.default_rng = real


def cuadros_buenos(p: TextilePattern, n: int = 24, paso: float = 0.33) -> list:
    """n cuadros buenos separados `paso` s: unos 8 s de linea."""
    out = []
    for _ in range(n):
        out.append(p.frame())
        p.advance(paso)
    return out
