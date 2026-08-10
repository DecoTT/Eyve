"""WinRAR-style license reminder dialog."""
from __future__ import annotations
import webbrowser
import customtkinter as ctk
from eyve.ui import theme as T
from eyve.i18n import t
from eyve.license.license_manager import LicenseManager


class LicenseDialog(ctk.CTkToplevel):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.title(t("lic_reminder_title"))
        self.geometry("440x360")
        self.resizable(False, False)
        self.grab_set()
        self.configure(fg_color=T.BG_CARD)
        self._lm = LicenseManager()
        self._build()

    def _build(self) -> None:
        ctk.CTkLabel(self, text=t("lic_reminder_title"),
                     font=T.bold(T.FONT_XL), text_color=T.WARN).pack(pady=(20, 8))

        ctk.CTkLabel(self, text=t("lic_reminder_body"),
                     font=T.font(T.FONT_SM), text_color=T.TEXT_PRI,
                     wraplength=380, justify="center").pack(padx=24, pady=8)

        sep = ctk.CTkFrame(self, height=1, fg_color=T.BORDER)
        sep.pack(fill="x", padx=24, pady=12)

        # key entry
        key_row = ctk.CTkFrame(self, fg_color="transparent")
        key_row.pack(padx=24, fill="x")
        self._key_entry = ctk.CTkEntry(key_row, placeholder_text="EYVE-XXXX-XXXX",
                                        fg_color=T.BG_INPUT, border_color=T.BORDER,
                                        width=220)
        self._key_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(key_row, text=t("lic_enter_key"), width=120,
                      fg_color=T.ACCENT, text_color="#000",
                      command=self._activate).pack(side="left")

        self._msg = ctk.CTkLabel(self, text="",
                                  font=T.font(T.FONT_SM), text_color=T.TEXT_DIM)
        self._msg.pack(pady=4)

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(pady=(8, 20))

        ctk.CTkButton(btn_row, text=t("lic_remind_later"),
                      fg_color=T.BG_INPUT, width=160,
                      command=self.destroy).pack(side="left", padx=6)

        ctk.CTkButton(btn_row, text="eyve.app",
                      fg_color=T.BG_INPUT, width=100,
                      command=lambda: webbrowser.open("https://eyve.app/license")).pack(
            side="left", padx=6)

    def _activate(self) -> None:
        key = self._key_entry.get().strip()
        if self._lm.activate(key):
            self._msg.configure(text=t("lic_key_valid"), text_color=T.ACCENT)
            self.after(1500, self.destroy)
        else:
            self._msg.configure(text=t("lic_key_invalid"), text_color=T.DANGER)
