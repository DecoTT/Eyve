"""
Texto sobre la imagen con cv2.putText, sin «??».

Las fuentes Hershey de OpenCV solo traen ASCII: cualquier otro caracter
sale como «?».  En pantalla, la etiqueta de Patron decia «anomal??a», y
pasaria igual con «rayón» o con cualquier clase con acento que ponga el
usuario.  Aqui se quitan los acentos (NFKD) y se cambian los pocos simbolos
que se usan por su equivalente ASCII.

Solo cambia lo que se DIBUJA: los nombres de clase en los datos, en el
proyecto y en los jobs se quedan como estan.
"""
from __future__ import annotations

import unicodedata

import cv2

#: simbolos sin descomposicion ASCII, cambiados por algo legible
_SUSTITUTOS = {
    "→": "->", "←": "<-", "°": " deg", "×": "x", "⚠": "!",
    "«": '"', "»": '"', "“": '"', "”": '"', "‘": "'", "’": "'",
    "–": "-", "—": "-", "…": "...", "¿": "", "¡": "",
}


def ascii_cv(texto) -> str:
    """El texto como lo puede dibujar cv2.putText: solo ASCII, nunca «?»."""
    s = str(texto)
    for k, v in _SUSTITUTOS.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s)
    return s.encode("ascii", "ignore").decode("ascii")


def put_text(img, texto, org, font, scale, color, thickness=1,
             line_type=cv2.LINE_8) -> None:
    """cv2.putText con el texto pasado por ascii_cv()."""
    cv2.putText(img, ascii_cv(texto), org, font, scale, color, thickness,
                line_type)


def text_size(texto, font, scale, thickness):
    """cv2.getTextSize del texto TAL COMO se dibuja."""
    return cv2.getTextSize(ascii_cv(texto), font, scale, thickness)
