"""Left navigation sidebar."""
from __future__ import annotations
from typing import Callable, TYPE_CHECKING
import customtkinter as ctk
from eyve.ui import theme as T
from eyve.ui.logo_loader import sidebar_logo, logo_available
from eyve.i18n import t, get_language

if TYPE_CHECKING:
    from eyve.ui.app import EyveApp


NAV_ITEMS = [
    ("nav_home",       "⌂"),
    ("nav_classes",    "◈"),
    ("nav_capture",    "⊙"),
    ("nav_tagging",    "⊞"),
    ("nav_training",   "▶"),
    ("nav_production", "◉"),
    ("nav_demo",       "◱"),
]


class Sidebar(ctk.CTkFrame):
    def __init__(self, master, on_navigate: Callable[[str], None], **kwargs):
        super().__init__(
            master,
            width=T.SIDEBAR_W,
            corner_radius=0,
            fg_color=T.BG_PANEL,
            **kwargs,
        )
        self._on_navigate = on_navigate
        self._active: str = "nav_home"
        self._buttons: dict[str, ctk.CTkButton] = {}
        self._app = None  # set after construction via set_app()
        self.grid_propagate(False)
        self._build()

    def set_app(self, app: "EyveApp") -> None:
        self._app = app

    def _build(self) -> None:
        # row weight: push gear to bottom
        self.grid_rowconfigure(len(NAV_ITEMS) + 3, weight=1)

        # ── logo / brand header ───────────────────────────────────────────────
        img = sidebar_logo()
        if img is not None:
            # image logo — auto-swaps black ↔ white with theme
            logo_lbl = ctk.CTkLabel(self, image=img, text="")
            logo_lbl.grid(row=0, column=0, padx=0, pady=(16, 0))
            ver = ctk.CTkLabel(
                self,
                text="2.1",
                font=T.font(T.FONT_XS),
                text_color=T.TEXT_DIM,
            )
            ver.grid(row=1, column=0, padx=0, pady=(0, 12))
        else:
            # text fallback when logo file is not present
            ctk.CTkLabel(
                self,
                text="EYVE",
                font=T.bold(T.FONT_XL),
                text_color=T.ACCENT,
            ).grid(row=0, column=0, padx=20, pady=(20, 4), sticky="w")
            ctk.CTkLabel(
                self,
                text="2.1",
                font=T.font(T.FONT_SM),
                text_color=T.TEXT_DIM,
            ).grid(row=1, column=0, padx=20, pady=(0, 16), sticky="w")

        for i, (key, icon) in enumerate(NAV_ITEMS):
            btn = ctk.CTkButton(
                self,
                text=f"  {icon}  {t(key)}",
                anchor="w",
                font=T.font(T.FONT_SM),
                corner_radius=8,
                height=38,
                fg_color="transparent",
                hover_color=T.BG_CARD,
                text_color=T.TEXT_SEC,
                command=lambda k=key: self._click(k),
            )
            btn.grid(row=i + 2, column=0, padx=8, pady=2, sticky="ew")
            self._buttons[key] = btn

        self.columnconfigure(0, weight=1)
        self._set_active("nav_home")

        # ── bottom: language pill + settings gear ────────────────────────────
        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.grid(row=len(NAV_ITEMS) + 3, column=0, sticky="ew", padx=8, pady=(0, 12))
        bottom.columnconfigure(0, weight=1)

        # language quick toggle (EN | ES)
        lang_row = ctk.CTkFrame(bottom, fg_color=T.BG_INPUT, corner_radius=8)
        lang_row.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self._lang_btns: dict[str, ctk.CTkButton] = {}
        for code, label in [("en", "EN"), ("es", "ES")]:
            b = ctk.CTkButton(
                lang_row,
                text=label,
                font=T.font(T.FONT_XS),
                width=40, height=26,
                fg_color="transparent",
                hover_color=T.BG_CARD,
                text_color=T.TEXT_DIM,
                corner_radius=6,
                command=lambda c=code: self._switch_lang(c),
            )
            b.pack(side="left", padx=2, pady=2)
            self._lang_btns[code] = b
        self._refresh_lang_btns()

        # settings gear
        self._gear_btn = ctk.CTkButton(
            bottom,
            text="⚙  Settings",
            anchor="w",
            font=T.font(T.FONT_XS),
            height=30,
            fg_color="transparent",
            hover_color=T.BG_CARD,
            text_color=T.TEXT_DIM,
            corner_radius=8,
            command=self._open_settings,
        )
        self._gear_btn.grid(row=1, column=0, sticky="ew")

    # ── actions ───────────────────────────────────────────────────────────────
    def _click(self, key: str) -> None:
        self._set_active(key)
        self._on_navigate(key)

    def _set_active(self, key: str) -> None:
        if self._active in self._buttons:
            self._buttons[self._active].configure(
                fg_color="transparent", text_color=T.TEXT_SEC
            )
        self._active = key
        if key in self._buttons:
            self._buttons[key].configure(
                fg_color=T.BG_CARD, text_color=T.ACCENT
            )

    def _switch_lang(self, code: str) -> None:
        if self._app:
            self._app.switch_language(code)
        self._refresh_lang_btns()

    def _refresh_lang_btns(self) -> None:
        current = get_language()
        for code, btn in self._lang_btns.items():
            if code == current:
                btn.configure(fg_color=T.ACCENT, text_color="#000")
            else:
                btn.configure(fg_color="transparent", text_color=T.TEXT_DIM)

    def _open_settings(self) -> None:
        if self._app:
            from eyve.ui.screens.settings_popup import SettingsPopup
            SettingsPopup(self._app, app=self._app)

    # ── public ────────────────────────────────────────────────────────────────
    def set_active(self, key: str) -> None:
        self._set_active(key)

    def refresh_labels(self) -> None:
        for key, btn in self._buttons.items():
            icon = next(ic for k, ic in NAV_ITEMS if k == key)
            btn.configure(text=f"  {icon}  {t(key)}")
        self._refresh_lang_btns()
