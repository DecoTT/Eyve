"""Bottom status bar."""
from __future__ import annotations
import customtkinter as ctk
from eyve.ui import theme as T
from eyve.i18n import t


class StatusBar(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, height=28, corner_radius=0,
                         fg_color=T.BG_PANEL, **kwargs)
        self.grid_propagate(False)

        self._project_lbl = ctk.CTkLabel(
            self, text=t("status_no_project"),
            font=T.font(T.FONT_XS), text_color=T.TEXT_SEC, anchor="w"
        )
        self._project_lbl.pack(side="left", padx=12)

        self._right_lbl = ctk.CTkLabel(
            self, text="",
            font=T.font(T.FONT_XS), text_color=T.TEXT_DIM, anchor="e"
        )
        self._right_lbl.pack(side="right", padx=12)

        # Aviso de version nueva. Va en la barra y no en un dialogo al
        # arrancar: el usuario abrio Eyve para trabajar, no para actualizar.
        self._update_lbl = ctk.CTkLabel(
            self, text="",
            font=T.font(T.FONT_XS), text_color=T.ACCENT, anchor="e"
        )
        self._update_lbl.pack(side="right", padx=4)
        self._update_cb = None
        self._update_lbl.bind(
            "<Button-1>", lambda e: (self._update_cb or (lambda: None))())

        self._license_lbl = ctk.CTkLabel(
            self, text="",
            font=T.font(T.FONT_XS), text_color=T.WARN, anchor="e"
        )
        self._license_lbl.pack(side="right", padx=4)
        # click → diálogo de licencia (la app lo expone como _show_license_dialog)
        self._license_lbl.bind("<Button-1>", lambda e: getattr(master, "_show_license_dialog", lambda: None)())
        self._license_lbl.configure(cursor="hand2")

    def set_project(self, name: str | None) -> None:
        if name:
            self._project_lbl.configure(text=t("status_project", name=name))
        else:
            self._project_lbl.configure(text=t("status_no_project"))

    def set_right(self, text: str) -> None:
        self._right_lbl.configure(text=text)

    def set_update(self, text: str, on_click=None) -> None:
        """Muestra (o borra, con texto vacio) el aviso de version nueva."""
        self._update_cb = on_click
        self._update_lbl.configure(
            text=text, cursor="hand2" if text else "arrow")

    def set_license(self, text: str, warn: bool = False) -> None:
        color = T.WARN if warn else T.TEXT_DIM
        self._license_lbl.configure(text=text, text_color=color)
