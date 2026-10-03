"""
Diálogo de actualización.

Qué ve el usuario: qué versión tiene, cuál hay, qué trae, y un botón.  Lo
que NO ve es el detalle del mecanismo (ZIP, checksum, respaldo), porque no
le sirve para decidir — pero sí ve, antes de apretar, que sus proyectos no
se tocan.  Esa es la pregunta que de verdad frena a la gente a actualizar.

Al terminar, Eyve se tiene que reiniciar: el paquete `eyve` ya está
cargado en memoria y recargarlo a mano es una fuente de errores raros.  Se
cierra y se vuelve a abrir, que además es lo que el usuario espera.
"""
from __future__ import annotations

import subprocess
import threading
from typing import TYPE_CHECKING, Optional

import customtkinter as ctk

from eyve.core import config as _cfg
from eyve.core import updater as U
from eyve.core.logger import log
from eyve.i18n import t
from eyve.ui import theme as T

if TYPE_CHECKING:
    from eyve.ui.app import EyveApp


class UpdateDialog(ctk.CTkToplevel):
    def __init__(self, master, info: Optional[U.UpdateInfo] = None, **kwargs):
        super().__init__(master, **kwargs)
        self.title(t("upd_title"))
        self.resizable(False, False)
        self.configure(fg_color=T.BG_CARD)
        self.geometry("460x400")
        self._info = info
        self._busy = False
        self._result: Optional[U.ApplyResult] = None
        self._build()
        self.after(200, self.grab_set)
        if info is None:
            self.after(120, self._check)
        else:
            self._show_info(info)

    # ── layout ────────────────────────────────────────────────────────────
    def _build(self) -> None:
        ctk.CTkLabel(self, text=t("upd_title"), font=T.bold(T.FONT_LG),
                     text_color=T.ACCENT).pack(pady=(18, 2))

        self._ver_lbl = ctk.CTkLabel(
            self, text=t("upd_current", v=U.__version__),
            font=T.font(T.FONT_SM), text_color=T.TEXT_SEC)
        self._ver_lbl.pack(pady=(0, 10))

        self._state_lbl = ctk.CTkLabel(
            self, text=t("upd_checking"), font=T.bold(T.FONT_MD),
            text_color=T.TEXT_PRI, wraplength=400, justify="left")
        self._state_lbl.pack(padx=24, pady=(0, 6))

        self._notes = ctk.CTkTextbox(self, height=130, fg_color=T.BG_INPUT,
                                     font=T.font(T.FONT_XS),
                                     text_color=T.TEXT_SEC, wrap="word")
        self._notes.pack(fill="x", padx=24, pady=(0, 8))
        self._notes.configure(state="disabled")

        # Lo que de verdad pregunta la gente antes de actualizar.
        ctk.CTkLabel(self, text=t("upd_safe"), font=T.font(T.FONT_XS),
                     text_color=T.TEXT_DIM, wraplength=400,
                     justify="left").pack(padx=24, anchor="w")

        self._bar = ctk.CTkProgressBar(self, height=8)
        self._bar.set(0)

        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.pack(fill="x", padx=24, pady=(12, 16))
        self._go_btn = ctk.CTkButton(btns, text=t("upd_install"), height=36,
                                     fg_color=T.ACCENT, text_color="#000",
                                     font=T.bold(T.FONT_SM),
                                     command=self._start)
        self._go_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._go_btn.configure(state="disabled")
        self._close_btn = ctk.CTkButton(btns, text=t("upd_later"), height=36,
                                        fg_color=T.BG_INPUT,
                                        text_color=T.TEXT_PRI,
                                        font=T.font(T.FONT_SM),
                                        command=self._on_close)
        self._close_btn.pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── consulta ──────────────────────────────────────────────────────────
    def _check(self) -> None:
        self._state_lbl.configure(text=t("upd_checking"), text_color=T.TEXT_PRI)

        def done(info: U.UpdateInfo):
            self.after(0, lambda: self._show_info(info))

        U.check_async(done)

    def _show_info(self, info: U.UpdateInfo) -> None:
        if not self.winfo_exists():
            return
        self._info = info
        if info.error and not info.version:
            self._state_lbl.configure(text=t("upd_offline"), text_color=T.WARN)
            return
        if not info.available:
            self._state_lbl.configure(text=t("upd_uptodate"),
                                      text_color=T.COLOR_OK)
            return
        self._state_lbl.configure(text=t("upd_available", v=info.version),
                                  text_color=T.ACCENT)
        self._ver_lbl.configure(
            text=t("upd_from_to", a=info.current, b=info.version))
        if info.notes:
            self._notes.configure(state="normal")
            self._notes.delete("1.0", "end")
            self._notes.insert("1.0", info.notes[:4000])
            self._notes.configure(state="disabled")
        self._go_btn.configure(state="normal")

    # ── instalación ───────────────────────────────────────────────────────
    def _start(self) -> None:
        if self._busy or not self._info or not self._info.available:
            return
        self._busy = True
        self._go_btn.configure(state="disabled", text=t("upd_working"))
        self._close_btn.configure(state="disabled")
        self._bar.pack(fill="x", padx=24, pady=(0, 6))
        self._bar.set(0)
        self._state_lbl.configure(text=t("upd_downloading"), text_color=T.TEXT_PRI)

        def on_progress(done: int, total: int) -> None:
            frac = (done / total) if total else 0.0
            self.after(0, lambda: self._bar.winfo_exists() and self._bar.set(frac))

        def work():
            res = U.update_now(self._info, on_progress)
            self.after(0, lambda: self._finish(res))

        threading.Thread(target=work, daemon=True).start()

    def _finish(self, res: U.ApplyResult) -> None:
        if not self.winfo_exists():
            return
        self._busy = False
        self._close_btn.configure(state="normal")
        if not res.ok:
            self._state_lbl.configure(text=t("upd_failed", err=res.error[:200]),
                                      text_color=T.COLOR_NOK)
            self._go_btn.configure(state="normal", text=t("upd_retry"))
            self._bar.pack_forget()
            return

        self._bar.set(1.0)
        msg = t("upd_done", v=self._info.version if self._info else "")
        if res.requirements_changed:
            # No se corre pip solo: puede tardar minutos y bajar gigas, y si
            # el usuario esta en una planta con red medida esa decision es
            # suya. setup.bat lo hace y es idempotente.
            msg += "\n\n" + t("upd_deps_changed")
        self._state_lbl.configure(text=msg, text_color=T.COLOR_OK)
        self._go_btn.configure(state="normal", text=t("upd_restart"),
                               command=self._restart)
        _cfg.set("update_last_installed", self._info.version if self._info else "")

    def _restart(self) -> None:
        try:
            subprocess.Popen(U.restart_command(), close_fds=True)
        except Exception as e:
            log.error(f"Restart failed: {e}")
        self.master.quit()
        self.master.destroy()

    def _on_close(self) -> None:
        if self._busy:
            return          # no cerrar a media instalacion
        self.grab_release()
        self.destroy()


def offer_update(app: "EyveApp", info: U.UpdateInfo) -> None:
    """
    Aviso discreto al arrancar cuando hay versión nueva.

    No se abre el diálogo de golpe: el usuario abrió Eyve para trabajar, no
    para actualizar.  Se avisa en la barra de estado y se actualiza cuando
    él quiera.
    """
    if not info.available:
        return
    try:
        bar = getattr(app, "status_bar", None)
        if bar is None or not hasattr(bar, "set_update"):
            return
        bar.set_update(t("upd_banner", v=info.version),
                       lambda: UpdateDialog(app, info))
    except Exception as e:
        log.debug(f"offer_update: {e}")
