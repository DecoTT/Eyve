"""Reusable modal dialogs."""
from __future__ import annotations
from typing import Optional
import customtkinter as ctk
from eyve.ui import theme as T
from eyve.i18n import t


def show_error(master, message: str) -> None:
    dlg = ctk.CTkToplevel(master)
    dlg.title(t("error"))
    dlg.resizable(False, False)
    dlg.grab_set()
    dlg.configure(fg_color=T.BG_CARD)
    ctk.CTkLabel(dlg, text=message, font=T.font(T.FONT_MD),
                 text_color=T.TEXT_PRI, wraplength=340).pack(padx=24, pady=20)
    ctk.CTkButton(dlg, text=t("ok"), fg_color=T.DANGER,
                  command=dlg.destroy).pack(pady=(0, 16))


def show_info(master, message: str, title: str | None = None) -> None:
    dlg = ctk.CTkToplevel(master)
    dlg.title(title or t("info"))
    dlg.resizable(False, False)
    dlg.grab_set()
    dlg.configure(fg_color=T.BG_CARD)
    ctk.CTkLabel(dlg, text=message, font=T.font(T.FONT_MD),
                 text_color=T.TEXT_PRI, wraplength=340).pack(padx=24, pady=20)
    ctk.CTkButton(dlg, text=t("ok"), fg_color=T.ACCENT,
                  text_color="#000", command=dlg.destroy).pack(pady=(0, 16))


def ask_confirm(master, message: str) -> bool:
    result = [False]
    dlg = ctk.CTkToplevel(master)
    dlg.title(t("warning"))
    dlg.resizable(False, False)
    dlg.grab_set()
    dlg.configure(fg_color=T.BG_CARD)
    ctk.CTkLabel(dlg, text=message, font=T.font(T.FONT_MD),
                 text_color=T.TEXT_PRI, wraplength=340).pack(padx=24, pady=20)
    row = ctk.CTkFrame(dlg, fg_color="transparent")
    row.pack(pady=(0, 16))

    def _yes():
        result[0] = True
        dlg.destroy()

    ctk.CTkButton(row, text=t("yes"), fg_color=T.DANGER,
                  width=90, command=_yes).pack(side="left", padx=6)
    ctk.CTkButton(row, text=t("no"), fg_color=T.BG_INPUT,
                  width=90, command=dlg.destroy).pack(side="left", padx=6)
    dlg.wait_window()
    return result[0]


def ask_string(master, prompt: str, title: str = "", initial: str = "") -> Optional[str]:
    result = [None]
    dlg = ctk.CTkToplevel(master)
    dlg.title(title or prompt)
    dlg.resizable(False, False)
    dlg.grab_set()
    dlg.configure(fg_color=T.BG_CARD)
    ctk.CTkLabel(dlg, text=prompt, font=T.font(T.FONT_MD),
                 text_color=T.TEXT_PRI).pack(padx=24, pady=(20, 8))
    entry = ctk.CTkEntry(dlg, width=260, fg_color=T.BG_INPUT,
                          border_color=T.BORDER)
    entry.insert(0, initial)
    entry.pack(padx=24, pady=(0, 12))
    row = ctk.CTkFrame(dlg, fg_color="transparent")
    row.pack(pady=(0, 16))

    def _ok():
        result[0] = entry.get().strip()
        dlg.destroy()

    ctk.CTkButton(row, text=t("ok"), fg_color=T.ACCENT, text_color="#000",
                  width=90, command=_ok).pack(side="left", padx=6)
    ctk.CTkButton(row, text=t("cancel"), fg_color=T.BG_INPUT,
                  width=90, command=dlg.destroy).pack(side="left", padx=6)
    dlg.wait_window()
    return result[0]
