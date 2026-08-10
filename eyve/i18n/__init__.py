"""
i18n — simple bilingual string manager.

Usage:
    from eyve.i18n import t, set_language
    set_language("es")
    print(t("app_title"))
    print(t("cls_count", n=3))
"""
from __future__ import annotations
from typing import Any

_lang: str = "en"
_strings: dict[str, str] = {}


def set_language(lang: str) -> None:
    global _lang, _strings
    _lang = lang
    if lang == "es":
        from eyve.i18n.es import STRINGS
    else:
        from eyve.i18n.en import STRINGS
    _strings = STRINGS


def get_language() -> str:
    return _lang


def t(key: str, **kwargs: Any) -> str:
    """Return translated string, interpolating any kwargs."""
    if not _strings:
        set_language(_lang)
    text = _strings.get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return text


def available_languages() -> list[tuple[str, str]]:
    return [("en", "English"), ("es", "Español")]


# Load default language at import time
set_language(_lang)
