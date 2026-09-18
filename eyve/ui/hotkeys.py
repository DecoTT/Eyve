"""
Toplevel hotkey guard shared by screens that bind keys on the root window.

Screens register hotkeys with ``toplevel.bind(seq, fn, add=True)`` because
CustomTkinter blocks ``bind_all``.  That means the binding outlives the
screen unless explicitly unbound, and it fires no matter which screen is
visible.  ``guard_hotkey`` wraps a handler so it only runs when:

  * the screen widget still exists      (not a stale, destroyed instance),
  * the screen is currently mapped      (not hidden behind another screen —
                                         'S' on Home must not save labels),
  * the key wasn't typed into an Entry  (typing "s" in a name field must
                                         not trigger Save).

app._teardown_screen() unbinds properly; this guard is defense in depth so
a missed unbind degrades to a silent no-op instead of a stale handler that
runs real side effects with obsolete state and then raises TclError.
"""
from __future__ import annotations
import tkinter as tk
from typing import Callable


def guard_hotkey(screen: tk.Misc, fn: Callable) -> Callable:
    def _guarded(event=None, _fn=fn):
        try:
            if not screen.winfo_exists() or not screen.winfo_ismapped():
                return
        except Exception:
            return   # widget half-destroyed — treat as gone
        w = getattr(event, "widget", None)
        if isinstance(w, (tk.Entry, tk.Text)):
            return
        return _fn(event)
    return _guarded
