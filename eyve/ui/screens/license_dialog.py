"""
Diálogo "Licencia" — pegar llave, ver nivel / titular / vence / equipos,
Comprobar ahora, liberar equipos (abre la cuenta en sbcsuite).

Nada aquí bloquea: se abre desde Settings, desde la barra de estado, o solo
una vez por arranque cuando la llave está vencida (aviso, no cierre).
"""
from __future__ import annotations
import webbrowser
from typing import Optional

import customtkinter as ctk

from eyve.ui import theme as T
from eyve.i18n import t
from eyve.license.license_manager import (
    LicenseManager, LlaveInvalida, EstadoServidor, hash_equipo,
    URL_CUENTA_LICENCIAS, URL_COMPRAR,
)


class LicenseDialog(ctk.CTkToplevel):
    def __init__(self, master, lm: Optional[LicenseManager] = None,
                 on_change=None, **kwargs):
        super().__init__(master, **kwargs)
        self._lm = lm or LicenseManager()
        self._on_change = on_change          # app refresca barra de estado
        self._busy = False
        self.title(t("lic_title"))
        self.geometry("520x620")
        self.resizable(False, False)
        self.grab_set()
        self.configure(fg_color=T.BG_CARD)
        self._build()
        self._refresh()
        self.lift()

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build(self) -> None:
        PX = 24
        ctk.CTkLabel(self, text=t("lic_title"),
                     font=T.bold(T.FONT_XL), text_color=T.ACCENT).pack(pady=(18, 2))
        ctk.CTkLabel(self, text=t("lic_honor"),
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM,
                     wraplength=460, justify="center").pack(padx=PX, pady=(0, 10))

        # estado actual
        card = ctk.CTkFrame(self, fg_color=T.BG_PANEL, corner_radius=8)
        card.pack(fill="x", padx=PX)
        self._lbl_nivel = ctk.CTkLabel(card, text="", font=T.bold(T.FONT_LG),
                                       text_color=T.TEXT_PRI, anchor="w")
        self._lbl_nivel.pack(fill="x", padx=14, pady=(10, 0))
        self._rows: dict[str, ctk.CTkLabel] = {}
        for key in ("lic_titular", "lic_vence", "lic_equipos", "lic_ultimo_checkin"):
            row = ctk.CTkFrame(card, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=1)
            ctk.CTkLabel(row, text=t(key), width=120, anchor="w",
                         font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(side="left")
            lbl = ctk.CTkLabel(row, text="—", anchor="w",
                               font=T.font(T.FONT_SM), text_color=T.TEXT_PRI)
            lbl.pack(side="left", fill="x", expand=True)
            self._rows[key] = lbl
        self._lbl_aviso = ctk.CTkLabel(card, text="", font=T.font(T.FONT_XS),
                                       text_color=T.WARN, wraplength=440, justify="left",
                                       anchor="w")
        self._lbl_aviso.pack(fill="x", padx=14, pady=(4, 10))

        # pegar llave
        ctk.CTkLabel(self, text=t("lic_paste_label"),
                     font=T.bold(T.FONT_XS), text_color=T.TEXT_SEC).pack(
            anchor="w", padx=PX, pady=(12, 2))
        self._key_box = ctk.CTkTextbox(self, height=74, fg_color=T.BG_INPUT,
                                       border_color=T.BORDER, border_width=1,
                                       font=("Consolas", 10), wrap="char")
        self._key_box.pack(fill="x", padx=PX)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=PX, pady=(6, 0))
        self._btn_save = ctk.CTkButton(row, text=t("lic_btn_save"), width=140,
                                       fg_color=T.ACCENT, text_color="#000",
                                       font=T.bold(T.FONT_XS), command=self._guardar)
        self._btn_save.pack(side="left")
        self._btn_check = ctk.CTkButton(row, text=t("lic_btn_check"), width=140,
                                        fg_color=T.BG_INPUT, border_width=1,
                                        border_color=T.BORDER, text_color=T.TEXT_PRI,
                                        font=T.font(T.FONT_XS), command=self._comprobar)
        self._btn_check.pack(side="left", padx=6)
        self._btn_remove = ctk.CTkButton(row, text=t("lic_btn_remove"), width=90,
                                         fg_color=T.BG_INPUT, border_width=1,
                                         border_color=T.BORDER, text_color=T.TEXT_SEC,
                                         font=T.font(T.FONT_XS), command=self._quitar)
        self._btn_remove.pack(side="right")

        self._msg = ctk.CTkLabel(self, text="", font=T.font(T.FONT_XS),
                                 text_color=T.TEXT_DIM, wraplength=460, justify="left",
                                 anchor="w")
        self._msg.pack(fill="x", padx=PX, pady=(6, 0))

        # enlaces
        links = ctk.CTkFrame(self, fg_color="transparent")
        links.pack(fill="x", padx=PX, pady=(10, 0))
        ctk.CTkButton(links, text=t("lic_btn_account"), fg_color="transparent",
                      hover=False, text_color=T.ACCENT, anchor="w",
                      font=T.font(T.FONT_XS),
                      command=lambda: webbrowser.open(URL_CUENTA_LICENCIAS)).pack(side="left")
        ctk.CTkButton(links, text=t("lic_btn_buy"), fg_color="transparent",
                      hover=False, text_color=T.ACCENT, anchor="w",
                      font=T.font(T.FONT_XS),
                      command=lambda: webbrowser.open(URL_COMPRAR)).pack(side="left", padx=8)

        # id del equipo (para soporte / liberar)
        idrow = ctk.CTkFrame(self, fg_color="transparent")
        idrow.pack(fill="x", padx=PX, pady=(8, 0))
        ctk.CTkLabel(idrow, text=t("lic_device"), font=T.font(T.FONT_XS),
                     text_color=T.TEXT_DIM).pack(side="left")
        h = hash_equipo()
        ctk.CTkLabel(idrow, text=h[:16] + "…", font=("Consolas", 10),
                     text_color=T.TEXT_SEC).pack(side="left", padx=6)
        ctk.CTkButton(idrow, text="⎘", width=28, height=24, fg_color=T.BG_INPUT,
                      border_width=1, border_color=T.BORDER, text_color=T.TEXT_SEC,
                      command=lambda: self._copy(h)).pack(side="left")

        ctk.CTkButton(self, text=t("ok"), width=120, fg_color=T.BG_INPUT,
                      border_width=1, border_color=T.BORDER, text_color=T.TEXT_PRI,
                      command=self.destroy).pack(pady=(14, 16))

    # ── estado → widgets ─────────────────────────────────────────────────────
    def _refresh(self) -> None:
        lm = self._lm
        lic = lm.licencia
        srv = lm.servidor
        st = lm.status()
        if lic is None:
            self._lbl_nivel.configure(text="Eyve Free", text_color=T.TEXT_PRI)
            for l in self._rows.values():
                l.configure(text="—")
            self._lbl_aviso.configure(
                text=t("lic_invalid_stored") if st == "invalida" else t("lic_no_key"),
                text_color=T.WARN if st == "invalida" else T.TEXT_DIM)
            self._btn_check.configure(state="disabled")
            self._btn_remove.configure(state="normal" if st == "invalida" else "disabled")
            return

        color = T.WARN if lic.vencida else T.COLOR_OK
        self._lbl_nivel.configure(text=f"Eyve {lm.nivel_label}", text_color=color)
        self._rows["lic_titular"].configure(text=lic.titular + (f"  ({lic.sub})" if lic.nombre else ""))
        self._rows["lic_vence"].configure(
            text=lic.vence_str + (f"   ·   {t('lic_expired_tag')}" if lic.vencida else ""),
            text_color=T.WARN if lic.vencida else T.TEXT_PRI)
        if srv.consultado_en:
            self._rows["lic_equipos"].configure(text=f"{srv.equipos_activos} / {srv.max_equipos}")
        else:
            self._rows["lic_equipos"].configure(text=f"— / {lic.maxeq}")
        last = lm.ultimo_checkin
        self._rows["lic_ultimo_checkin"].configure(
            text=last.astimezone().strftime("%Y-%m-%d %H:%M") if last else t("lic_never"))

        aviso, col = "", T.TEXT_DIM
        if lic.vencida:
            aviso, col = t("lic_expired_notice", date=lic.vence_str), T.WARN
        elif lm.activacion_pendiente:
            aviso, col = t("lic_pending"), T.TEXT_DIM
        elif srv.motivo == "sin_cupo":
            eq = ", ".join(e.get("nombre", "?") for e in srv.equipos) or "—"
            aviso, col = f"{srv.mensaje or t('lic_no_slots')}\n{t('lic_devices_list')}: {eq}", T.WARN
        elif srv.estado == "revocada":
            aviso, col = t("lic_revoked"), T.DANGER
        elif srv.estado == "pendiente":
            aviso, col = t("lic_student_pending"), T.TEXT_DIM
        elif srv.motivo == "no_registrada":
            aviso, col = t("lic_not_registered"), T.WARN
        self._lbl_aviso.configure(text=aviso, text_color=col)
        self._btn_check.configure(state="normal")
        self._btn_remove.configure(state="normal")
        if self._on_change:
            self._on_change()

    # ── acciones ─────────────────────────────────────────────────────────────
    def _guardar(self) -> None:
        raw = self._key_box.get("1.0", "end").strip()
        if not raw:
            self._msg.configure(text=t("lic_paste_first"), text_color=T.WARN)
            return
        try:
            lic = self._lm.guardar_llave(raw)
        except LlaveInvalida:
            self._msg.configure(text=t("lic_key_invalid"), text_color=T.DANGER)
            return
        self._key_box.delete("1.0", "end")
        self._msg.configure(text=t("lic_key_saved", level=lic.nivel.capitalize()),
                            text_color=T.COLOR_OK)
        self._refresh()
        self._comprobar()

    def _comprobar(self) -> None:
        if self._busy or not self._lm.licencia:
            return
        self._busy = True
        self._btn_check.configure(state="disabled", text=t("lic_checking"))

        def done(srv: Optional[EstadoServidor], err: Optional[str]) -> None:
            self.after(0, lambda: self._done(srv, err))

        self._lm.comprobar_en_hilo(done)

    def _done(self, srv: Optional[EstadoServidor], err: Optional[str]) -> None:
        self._busy = False
        if not self.winfo_exists():
            return
        self._btn_check.configure(state="normal", text=t("lic_btn_check"))
        if err:
            self._msg.configure(text=t("lic_offline"), text_color=T.TEXT_DIM)
        elif srv and not srv.motivo:
            self._msg.configure(text=t("lic_server_ok", n=srv.equipos_activos, max=srv.max_equipos),
                                text_color=T.COLOR_OK)
        elif srv and srv.motivo == "sin_cupo":
            self._msg.configure(text=t("lic_no_slots"), text_color=T.WARN)
        elif srv and srv.motivo == "llave_invalida":
            self._msg.configure(text=t("lic_key_invalid"), text_color=T.DANGER)
        else:
            self._msg.configure(text=f"{t('lic_server_err')}: {srv.motivo if srv else ''}",
                                text_color=T.WARN)
        self._refresh()

    def _quitar(self) -> None:
        self._lm.quitar_llave()
        self._msg.configure(text=t("lic_removed"), text_color=T.TEXT_DIM)
        self._refresh()

    def _copy(self, text: str) -> None:
        self.clipboard_clear()
        self.clipboard_append(text)
