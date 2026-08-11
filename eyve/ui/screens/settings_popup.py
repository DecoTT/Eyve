"""
Settings popup — language, appearance (dark/light), license status + activation.
Opens from the gear icon in the sidebar.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import threading
import webbrowser

import customtkinter as ctk

from eyve.ui import theme as T
from eyve.ui.logo_loader import about_logo
from eyve.i18n import t, get_language, available_languages
from eyve.core import config
from eyve.core import model_manager as mm
from eyve.ui.screens.production_screen import FPS_OPTIONS, FPS_LABELS
from eyve.license.license_manager import LicenseManager, device_fingerprint

if TYPE_CHECKING:
    from eyve.ui.app import EyveApp

_COMPANY      = "Servicios Bonacril SA de CV"
_COMPANY_WEB  = "sbcgroup.com.mx"
_COMPANY_URL  = "https://sbcgroup.com.mx"


class SettingsPopup(ctk.CTkToplevel):
    def __init__(self, master, app: "EyveApp", **kwargs):
        super().__init__(master, **kwargs)
        self._app = app
        self._lm = LicenseManager()
        self.title("⚙  Settings")
        self.resizable(False, True)
        self.grab_set()
        self.configure(fg_color=T.BG_CARD)
        self._build()
        # Size to 90% of screen height, capped at 1100px
        self.update_idletasks()
        screen_h = self.winfo_screenheight()
        win_h    = min(1100, int(screen_h * 0.90))
        self.geometry(f"420x{win_h}")
        self.lift()

    def _build(self) -> None:
        PX = 24  # horizontal padding constant

        # ── header ────────────────────────────────────────────────────────────
        ctk.CTkLabel(self, text=t("set_title"),
                     font=T.bold(T.FONT_LG), text_color=T.ACCENT).pack(pady=(18, 4))
        self._sep()

        # ── language ─────────────────────────────────────────────────────────
        # Always bilingual — intentionally NOT translated
        ctk.CTkLabel(self, text="Language / Lenguaje",
                     font=T.bold(T.FONT_SM), text_color=T.TEXT_SEC).pack(
            anchor="w", padx=PX, pady=(10, 4))

        lang_row = ctk.CTkFrame(self, fg_color="transparent")
        lang_row.pack(fill="x", padx=PX, pady=(0, 8))
        current_lang = get_language()
        for code, label in available_languages():
            active = code == current_lang
            ctk.CTkButton(
                lang_row, text=label, width=120, height=34,
                font=T.bold(T.FONT_SM) if active else T.font(T.FONT_SM),
                fg_color=T.ACCENT if active else T.BG_INPUT,
                text_color="#000" if active else T.TEXT_PRI,
                border_width=2 if active else 0, border_color=T.ACCENT,
                command=lambda c=code: self._set_lang(c),
            ).pack(side="left", padx=(0, 8))

        self._sep()

        # ── appearance ────────────────────────────────────────────────────────
        ctk.CTkLabel(self, text=t("settings_appearance"),
                     font=T.bold(T.FONT_SM), text_color=T.TEXT_SEC).pack(
            anchor="w", padx=PX, pady=(10, 4))

        mode_row = ctk.CTkFrame(self, fg_color="transparent")
        mode_row.pack(fill="x", padx=PX, pady=(0, 8))
        current_mode = T.get_mode()
        for m, key in [("dark", "settings_dark"), ("light", "settings_light")]:
            active = m == current_mode
            ctk.CTkButton(
                mode_row, text=t(key), width=120, height=34,
                font=T.bold(T.FONT_SM) if active else T.font(T.FONT_SM),
                fg_color=T.ACCENT if active else T.BG_INPUT,
                text_color="#000" if active else T.TEXT_PRI,
                border_width=2 if active else 0, border_color=T.ACCENT,
                command=lambda mv=m: self._set_theme(mv),
            ).pack(side="left", padx=(0, 8))

        self._sep()

        # ── performance ───────────────────────────────────────────────────────
        self._build_performance_section(PX)
        self._sep()

        # ── license ───────────────────────────────────────────────────────────
        ctk.CTkLabel(self, text=t("settings_license"),
                     font=T.bold(T.FONT_SM), text_color=T.TEXT_SEC).pack(
            anchor="w", padx=PX, pady=(10, 2))

        status = self._lm.status()
        if status == "activated":
            exp = self._lm.expiry_date()
            badge_text  = f"{t('settings_status_active')}  ·  expires {exp}" if exp else t("settings_status_active")
            badge_color = T.COLOR_OK
        elif status == "license_expired":
            badge_text  = f"License expired ({self._lm.expiry_date()}) — please renew"
            badge_color = T.DANGER
        elif status == "trial":
            badge_text  = t("settings_status_trial", days=self._lm.days_remaining())
            badge_color = T.WARN
        else:
            badge_text  = t("settings_status_expired")
            badge_color = T.DANGER

        ctk.CTkLabel(self, text=badge_text,
                     font=T.bold(T.FONT_SM), text_color=badge_color).pack(
            anchor="w", padx=PX, pady=(0, 6))

        # device ID
        fp = device_fingerprint()
        ctk.CTkLabel(self, text=t("settings_device_id"),
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(
            anchor="w", padx=PX)

        fp_row = ctk.CTkFrame(self, fg_color="transparent")
        fp_row.pack(fill="x", padx=PX, pady=(2, 2))
        self._fp_entry = ctk.CTkEntry(
            fp_row, fg_color=T.BG_INPUT, border_color=T.BORDER,
            font=T.font(T.FONT_SM))
        self._fp_entry.insert(0, fp)
        self._fp_entry.configure(state="readonly")
        self._fp_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(
            fp_row, text="⎘", width=30, height=28,
            fg_color=T.BG_INPUT, border_width=1, border_color=T.BORDER,
            font=T.font(T.FONT_SM), text_color=T.TEXT_SEC,
            command=lambda: self._copy(fp),
        ).pack(side="left")

        ctk.CTkLabel(self, text=t("settings_device_hint"),
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM,
                     wraplength=340).pack(anchor="w", padx=PX, pady=(0, 6))

        # activate section — hidden once activated
        if status != "activated":
            ctk.CTkLabel(self, text=t("settings_activate"),
                         font=T.bold(T.FONT_XS), text_color=T.TEXT_SEC).pack(
                anchor="w", padx=PX, pady=(4, 2))

            act_row = ctk.CTkFrame(self, fg_color="transparent")
            act_row.pack(fill="x", padx=PX, pady=(0, 2))
            self._key_entry = ctk.CTkEntry(
                act_row, placeholder_text=t("settings_key_hint"),
                fg_color=T.BG_INPUT, border_color=T.BORDER,
                font=T.font(T.FONT_SM))
            self._key_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
            ctk.CTkButton(
                act_row, text=t("settings_activate_btn"),
                width=80, height=28,
                fg_color=T.ACCENT, text_color="#000",
                font=T.bold(T.FONT_XS),
                command=self._activate,
            ).pack(side="left")

            self._act_lbl = ctk.CTkLabel(self, text="",
                                          font=T.font(T.FONT_XS), text_color=T.DANGER)
            self._act_lbl.pack(anchor="w", padx=PX)

        self._sep()

        # ── models / offline mode ─────────────────────────────────────────────
        self._build_models_section(PX)
        self._sep()

        # ── about ─────────────────────────────────────────────────────────────
        ctk.CTkLabel(self, text=t("settings_about"),
                     font=T.bold(T.FONT_SM), text_color=T.TEXT_SEC).pack(
            anchor="w", padx=PX, pady=(8, 6))

        # logo image (if available)
        img = about_logo()
        if img is not None:
            ctk.CTkLabel(self, image=img, text="").pack(pady=(0, 6))

        # version + tagline
        from eyve import __version__
        ctk.CTkLabel(self, text=f"Eyve  v{__version__}",
                     font=T.bold(T.FONT_SM), text_color=T.TEXT_PRI).pack(
            anchor="center", pady=(0, 2))
        ctk.CTkLabel(self, text=t("set_tagline"),
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(
            anchor="center", pady=(0, 6))

        # company info separator
        ctk.CTkFrame(self, height=1, fg_color=T.BG_INPUT).pack(
            fill="x", padx=40, pady=(4, 8))

        # company name
        ctk.CTkLabel(self, text=_COMPANY,
                     font=T.bold(T.FONT_XS), text_color=T.TEXT_SEC).pack(
            anchor="center", pady=(0, 2))

        # website button — opens browser
        ctk.CTkButton(
            self, text=f"🌐  {_COMPANY_WEB}",
            font=T.font(T.FONT_XS),
            height=26, width=180,
            fg_color=T.BG_INPUT,
            border_width=1, border_color=T.BORDER,
            text_color=T.ACCENT2,
            hover_color=T.BG_DARK,
            command=lambda: webbrowser.open(_COMPANY_URL),
        ).pack(anchor="center", pady=(0, 6))

        # license line
        ctk.CTkLabel(self, text="Licensed under AGPL-3.0  ·  eyve.app",
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(
            anchor="center", pady=(0, 2))

        # ── close ─────────────────────────────────────────────────────────────
        ctk.CTkButton(self, text=t("close"),
                      fg_color=T.BG_INPUT, width=100,
                      command=self.destroy).pack(pady=(12, 16))

    # ── performance section ───────────────────────────────────────────────────
    def _build_performance_section(self, PX: int) -> None:
        # Always bilingual for clarity
        ctk.CTkLabel(self, text=t("set_perf"),
                     font=T.bold(T.FONT_SM), text_color=T.TEXT_SEC).pack(
            anchor="w", padx=PX, pady=(10, 2))

        ctk.CTkLabel(
            self,
            text=(
                "Production loop FPS cap  —  límite de cuadros por segundo.\n"
                "'Sin límite' deja que la cámara y la inferencia marquen el ritmo."
            ),
            font=T.font(T.FONT_XS), text_color=T.TEXT_DIM, wraplength=360, anchor="w",
        ).pack(anchor="w", padx=PX, pady=(0, 6))

        current_cap = int(config.get("prod_fps_cap", 30))

        fps_row = ctk.CTkFrame(self, fg_color="transparent")
        fps_row.pack(fill="x", padx=PX, pady=(0, 8))

        self._fps_btns: dict[int, ctk.CTkButton] = {}

        def _set_cap(cap: int) -> None:
            config.set("prod_fps_cap", cap)
            for v, btn in self._fps_btns.items():
                active = (v == cap)
                btn.configure(
                    fg_color=T.ACCENT if active else T.BG_INPUT,
                    text_color="#000" if active else T.TEXT_PRI,
                    font=T.bold(T.FONT_XS) if active else T.font(T.FONT_XS),
                    border_width=2 if active else 0,
                )

        for cap in FPS_OPTIONS:
            active = (cap == current_cap)
            btn = ctk.CTkButton(
                fps_row,
                text=FPS_LABELS[cap],
                width=82, height=30,
                font=T.bold(T.FONT_XS) if active else T.font(T.FONT_XS),
                fg_color=T.ACCENT if active else T.BG_INPUT,
                text_color="#000" if active else T.TEXT_PRI,
                border_width=2 if active else 0,
                border_color=T.ACCENT,
                command=lambda c=cap: _set_cap(c),
            )
            btn.pack(side="left", padx=(0, 6))
            self._fps_btns[cap] = btn

    # ── models section (compact) ──────────────────────────────────────────────
    def _build_models_section(self, PX: int) -> None:
        ctk.CTkLabel(self, text=t("set_models_title"),
                     font=T.bold(T.FONT_SM), text_color=T.TEXT_SEC).pack(
            anchor="w", padx=PX, pady=(10, 4))

        # offline toggle row
        off_row = ctk.CTkFrame(self, fg_color="transparent")
        off_row.pack(fill="x", padx=PX, pady=(0, 2))
        ctk.CTkLabel(off_row, text=t("set_offline"),
                     font=T.font(T.FONT_SM), text_color=T.TEXT_PRI).pack(side="left")
        self._offline_var = ctk.BooleanVar(value=bool(config.get("offline_mode", False)))
        ctk.CTkSwitch(
            off_row, text="", variable=self._offline_var,
            width=46, height=24,
            fg_color=T.BG_INPUT, progress_color=T.ACCENT,
            command=self._toggle_offline,
        ).pack(side="right")

        self._offline_hint = ctk.CTkLabel(
            self, text=self._offline_hint_text(),
            font=T.font(T.FONT_XS), text_color=T.TEXT_DIM, wraplength=360, anchor="w")
        self._offline_hint.pack(fill="x", padx=PX, pady=(0, 2))

        # warning label — shown when offline but nothing downloaded
        self._offline_warn = ctk.CTkLabel(
            self, text="", font=T.bold(T.FONT_XS),
            text_color=T.WARN, wraplength=360, anchor="w")
        self._offline_warn.pack(fill="x", padx=PX, pady=(0, 4))
        self._refresh_offline_warn()

        # "Manage Models" button opens full dialog
        ctk.CTkButton(
            self, text=t("set_manage_models"),
            font=T.font(T.FONT_XS), height=28,
            fg_color=T.BG_INPUT, border_width=1, border_color=T.BORDER,
            text_color=T.TEXT_SEC,
            command=self._open_model_manager,
        ).pack(fill="x", padx=PX, pady=(0, 8))

    def _offline_hint_text(self) -> str:
        if config.get("offline_mode", False):
            return "🔒 Offline — ultralytics will not access the internet."
        return "🌐 Online — models auto-downloaded by ultralytics on first use."

    def _refresh_offline_warn(self) -> None:
        if config.get("offline_mode", False) and not any(mm.is_downloaded(n) for n in mm.MODELS):
            self._offline_warn.configure(
                text=t("set_no_models"))
        else:
            self._offline_warn.configure(text="")

    def _toggle_offline(self) -> None:
        val = self._offline_var.get()
        config.set("offline_mode", val)
        mm.apply_offline_env()
        self._offline_hint.configure(text=self._offline_hint_text())
        self._refresh_offline_warn()

    def _open_model_manager(self) -> None:
        ModelManagerDialog(self)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _sep(self) -> None:
        ctk.CTkFrame(self, height=1, fg_color=T.BORDER).pack(fill="x", padx=20, pady=4)

    def _copy(self, text: str) -> None:
        self.clipboard_clear()
        self.clipboard_append(text)

    def _set_lang(self, code: str) -> None:
        self.destroy()
        self._app.switch_language(code)

    def _set_theme(self, mode: str) -> None:
        self.destroy()
        self._app.switch_theme(mode)

    def _activate(self) -> None:
        key = self._key_entry.get().strip()
        ok, msg = self._lm.activate(key)
        if ok:
            exp = f"{msg[:4]}-{msg[4:6]}-{msg[6:8]}"
            self._act_lbl.configure(
                text=f"{t('settings_activated_ok')}  (expires {exp})",
                text_color=T.COLOR_OK)
            self._key_entry.configure(state="disabled")
        else:
            self._act_lbl.configure(text=f"✗  {msg}", text_color=T.DANGER)


# ── Model Manager Dialog ───────────────────────────────────────────────────────

class ModelManagerDialog(ctk.CTkToplevel):
    """
    Standalone dialog for downloading / managing local YOLO base weights.
    Opened from Settings → "Manage downloaded models".
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.title("⬇  Model Manager")
        self.geometry("480x340")
        self.resizable(False, False)
        self.grab_set()
        self.configure(fg_color=T.BG_CARD)
        self._build()
        self.lift()

    def _build(self) -> None:
        PX = 20

        ctk.CTkLabel(self, text=t("mm_title"),
                     font=T.bold(T.FONT_MD), text_color=T.ACCENT).pack(pady=(16, 2))
        ctk.CTkLabel(
            self,
            text=t("mm_stored_in", path=str(mm.models_dir())),
            font=T.font(T.FONT_XS), text_color=T.TEXT_DIM,
        ).pack(pady=(0, 10))

        # ── model selector (dropdown) ─────────────────────────────────────────
        sel_row = ctk.CTkFrame(self, fg_color="transparent")
        sel_row.pack(fill="x", padx=PX, pady=(0, 6))

        ctk.CTkLabel(sel_row, text=t("mm_model"),
                     font=T.font(T.FONT_SM), text_color=T.TEXT_SEC).pack(side="left", padx=(0, 8))

        # Build dropdown labels: "YOLOv8 Nano (★)  —  6 MB  ✓" etc.
        self._model_keys = list(mm.MODELS.keys())
        dropdown_labels = [self._make_label(n) for n in self._model_keys]

        self._selected = ctk.StringVar(value=dropdown_labels[0])
        self._dropdown = ctk.CTkOptionMenu(
            sel_row,
            values=dropdown_labels,
            variable=self._selected,
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD,
            font=T.font(T.FONT_SM),
            command=self._on_select,
            width=300,
        )
        self._dropdown.pack(side="left", fill="x", expand=True)

        # ── info card ─────────────────────────────────────────────────────────
        self._card = ctk.CTkFrame(self, fg_color=T.BG_DARK, corner_radius=8)
        self._card.pack(fill="x", padx=PX, pady=(4, 8))

        self._info_lbl  = ctk.CTkLabel(self._card, text="",
                                        font=T.font(T.FONT_XS), text_color=T.TEXT_DIM,
                                        wraplength=420, anchor="w")
        self._info_lbl.pack(anchor="w", padx=12, pady=(8, 2))

        self._status_lbl = ctk.CTkLabel(self._card, text="",
                                         font=T.bold(T.FONT_XS), text_color=T.TEXT_PRI)
        self._status_lbl.pack(anchor="w", padx=12, pady=(0, 4))

        self._prog = ctk.CTkProgressBar(self._card, height=8,
                                         fg_color=T.BG_INPUT, progress_color=T.ACCENT)
        self._prog.set(0)

        # ── action buttons ────────────────────────────────────────────────────
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=PX, pady=(0, 16))

        self._dl_btn = ctk.CTkButton(
            btn_row, text=t("mm_download"),
            fg_color=T.ACCENT, text_color="#000",
            font=T.bold(T.FONT_SM), height=36, width=160,
            command=self._download,
        )
        self._dl_btn.pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_row, text=t("mm_close"),
            fg_color=T.BG_INPUT, height=36, width=80,
            font=T.font(T.FONT_SM),
            command=self.destroy,
        ).pack(side="left")

        # initialise card for first model
        self._refresh_card()

    # ── helpers ────────────────────────────────────────────────────────────────
    def _make_label(self, name: str) -> str:
        info = mm.MODELS[name]
        star = " ★" if info.get("recommended") else ""
        tick = "  ✓" if mm.is_downloaded(name) else ""
        return f"{info['label']}{star}  —  {info['size']}{tick}"

    def _selected_name(self) -> str:
        idx = [self._make_label(n) for n in self._model_keys].index(self._selected.get())
        return self._model_keys[idx]

    def _on_select(self, _: str) -> None:
        self._refresh_card()

    def _refresh_card(self) -> None:
        name = self._selected_name()
        info = mm.MODELS[name]
        downloaded = mm.is_downloaded(name)

        self._info_lbl.configure(
            text=f"{info['size']}  ·  {info['desc']}")
        if downloaded:
            self._status_lbl.configure(text=t("mm_ready_offline"),
                                        text_color=T.COLOR_OK)
            self._dl_btn.configure(text=t("mm_downloaded"), state="disabled",
                                    fg_color=T.BG_INPUT, text_color=T.TEXT_DIM)
        else:
            self._status_lbl.configure(text=t("mm_not_downloaded"), text_color=T.TEXT_DIM)
            self._dl_btn.configure(text=t("mm_download"), state="normal",
                                    fg_color=T.ACCENT, text_color="#000")
        self._prog.pack_forget()

    def _refresh_dropdown(self) -> None:
        """Rebuild dropdown labels after a download completes."""
        new_labels = [self._make_label(n) for n in self._model_keys]
        cur_name   = self._selected_name()
        new_val    = self._make_label(cur_name)
        self._dropdown.configure(values=new_labels)
        self._selected.set(new_val)

    def _download(self) -> None:
        name = self._selected_name()

        self._dl_btn.configure(state="disabled", text=t("mm_downloading"))
        self._status_lbl.configure(text=t("mm_downloading"), text_color=T.TEXT_SEC)
        self._prog.set(0)
        self._prog.pack(fill="x", padx=12, pady=(0, 8))

        def _progress(frac: float) -> None:
            self.after(0, lambda f=frac: (
                self._prog.set(f),
                self._status_lbl.configure(text=f"Downloading…  {f*100:.0f}%"),
            ))

        def _done(ok: bool, msg: str) -> None:
            def _ui():
                self._prog.pack_forget()
                if ok:
                    self._refresh_dropdown()
                    self._refresh_card()
                    # Notify parent settings popup to refresh its warning label
                    try:
                        self.master._refresh_offline_warn()
                    except Exception:
                        pass
                else:
                    self._status_lbl.configure(
                        text=f"✗  Failed: {msg}", text_color=T.DANGER)
                    self._dl_btn.configure(state="normal", text=t("mm_retry"),
                                            fg_color=T.DANGER, text_color="#fff")
            self.after(0, _ui)

        mm.download_model(name, on_progress=_progress, on_done=_done)
