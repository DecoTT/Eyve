"""Classes screen — define inspection categories and mark OK / NOT OK / Ignore."""
from __future__ import annotations
from typing import TYPE_CHECKING
import customtkinter as ctk

from eyve.ui import theme as T
from eyve.i18n import t
from eyve.core.project_manager import ClassDef
from eyve.ui.components.dialogs import show_error, show_info, ask_confirm, ask_string

if TYPE_CHECKING:
    from eyve.ui.app import EyveApp

_KIND_COLORS = {"ok": T.COLOR_OK, "nok": T.COLOR_NOK, "ignore": T.TEXT_DIM}
_KIND_LABELS = {"ok": "cls_type_ok", "nok": "cls_type_nok", "ignore": "cls_type_ignore"}

# cycling order when user clicks the type badge
_KIND_CYCLE = ["ok", "nok", "ignore"]


class ClassesScreen(ctk.CTkFrame):
    def __init__(self, master, app: "EyveApp", **kwargs):
        super().__init__(master, fg_color=T.BG_DARK, **kwargs)
        self._app = app
        self._rows: list[_ClassRow] = []
        self._build()
        self._load_existing()

    # ── layout ────────────────────────────────────────────────────────────────
    def _build(self) -> None:
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # header
        header = ctk.CTkFrame(self, fg_color=T.BG_CARD, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 1))
        ctk.CTkLabel(header, text=t("cls_title"),
                     font=T.bold(T.FONT_XL), text_color=T.TEXT_PRI).pack(
            side="left", padx=24, pady=14)

        self._count_lbl = ctk.CTkLabel(header, text="",
                                        font=T.font(T.FONT_SM), text_color=T.TEXT_DIM)
        self._count_lbl.pack(side="left", padx=8)

        # warning banner
        warn = ctk.CTkFrame(self, fg_color="#2a1a00", corner_radius=0)
        warn.grid(row=1, column=0, sticky="ew")
        ctk.CTkLabel(warn, text=t("cls_warning_permanent"),
                     font=T.font(T.FONT_SM), text_color=T.WARN,
                     wraplength=800, justify="left").pack(padx=20, pady=8, anchor="w")

        # body — split into left (class list) + right (add / legend)
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=2, column=0, sticky="nsew", padx=20, pady=16)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0)
        body.grid_rowconfigure(0, weight=1)

        # scrollable class list
        self._scroll = ctk.CTkScrollableFrame(body, fg_color=T.BG_CARD,
                                               corner_radius=10, label_text="")
        self._scroll.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self._scroll.grid_columnconfigure(0, weight=1)

        # right panel
        right = ctk.CTkFrame(body, fg_color=T.BG_CARD, corner_radius=10, width=260)
        right.grid(row=0, column=1, sticky="ns")
        right.grid_propagate(False)
        self._build_right_panel(right)

        # bottom bar
        bot = ctk.CTkFrame(self, fg_color="transparent")
        bot.grid(row=3, column=0, sticky="ew", pady=(0, 16), padx=20)
        ctk.CTkButton(bot, text=t("cls_save"),
                      fg_color=T.ACCENT, text_color="#000", height=40, width=180,
                      font=T.bold(T.FONT_MD),
                      command=self._save).pack(side="left")
        self._msg = ctk.CTkLabel(bot, text="",
                                  font=T.font(T.FONT_SM), text_color=T.TEXT_DIM)
        self._msg.pack(side="left", padx=16)

    def _build_right_panel(self, parent) -> None:
        ctk.CTkLabel(parent, text=t("cls_add"),
                     font=T.bold(T.FONT_MD), text_color=T.TEXT_PRI).pack(padx=16, pady=(16, 8))

        self._new_name = ctk.CTkEntry(parent, placeholder_text=t("cls_name_hint"),
                                       fg_color=T.BG_INPUT, border_color=T.BORDER)
        self._new_name.pack(padx=16, fill="x")
        self._new_name.bind("<Return>", lambda e: self._add_class())

        self._new_kind = ctk.CTkOptionMenu(
            parent,
            values=[t("cls_type_ok"), t("cls_type_nok"), t("cls_type_ignore")],
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD,
        )
        self._new_kind.set(t("cls_type_nok"))
        self._new_kind.pack(padx=16, pady=6, fill="x")

        ctk.CTkButton(parent, text=t("cls_add"),
                      fg_color=T.ACCENT, text_color="#000",
                      command=self._add_class).pack(padx=16, pady=(0, 16), fill="x")

        sep = ctk.CTkFrame(parent, height=1, fg_color=T.BORDER)
        sep.pack(fill="x", padx=16, pady=8)

        # legend
        for kind, color in _KIND_COLORS.items():
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=2)
            ctk.CTkFrame(row, width=12, height=12, fg_color=color,
                         corner_radius=2).pack(side="left", padx=(0, 8))
            ctk.CTkLabel(row, text=t(_KIND_LABELS[kind]),
                         font=T.font(T.FONT_SM), text_color=T.TEXT_SEC).pack(side="left")

    # ── load / populate ───────────────────────────────────────────────────────
    def _load_existing(self) -> None:
        proj = self._app.get_project()
        if not proj:
            return
        for cls in proj.classes:
            self._add_row(cls)
        self._update_count()

    def _add_row(self, cls: ClassDef) -> None:
        row = _ClassRow(self._scroll, cls_def=cls, on_delete=self._delete_row)
        row.grid(row=len(self._rows), column=0, sticky="ew", padx=4, pady=2)
        self._rows.append(row)

    def _update_count(self) -> None:
        n = len(self._rows)
        self._count_lbl.configure(text=t("cls_count", n=n))

    # ── actions ───────────────────────────────────────────────────────────────
    def _add_class(self) -> None:
        name = self._new_name.get().strip()
        if not name:
            self._msg.configure(text=t("cls_name_empty"), text_color=T.DANGER)
            return
        existing = [r.name for r in self._rows]
        if name in existing:
            self._msg.configure(text=t("cls_name_dup"), text_color=T.DANGER)
            return

        kind_label = self._new_kind.get()
        kind_map = {t("cls_type_ok"): "ok", t("cls_type_nok"): "nok", t("cls_type_ignore"): "ignore"}
        kind = kind_map.get(kind_label, "nok")
        color = _KIND_COLORS[kind]
        cls = ClassDef(name=name, kind=kind, color=color)
        self._add_row(cls)
        self._new_name.delete(0, "end")
        self._update_count()
        self._msg.configure(text="")

    def _delete_row(self, row: "_ClassRow") -> None:
        proj = self._app.get_project()
        if proj:
            count = sum(1 for _ in proj.paths.raw_images.rglob(f"{row.name}/**/*.jpg"))
            if count > 0:
                self._msg.configure(
                    text=t("cls_in_use", name=row.name, n=count),
                    text_color=T.WARN,
                )
                return
        row.grid_forget()
        row.destroy()
        self._rows.remove(row)
        # re-grid remaining rows
        for i, r in enumerate(self._rows):
            r.grid(row=i, column=0, sticky="ew", padx=4, pady=2)
        self._update_count()

    def _save(self) -> None:
        proj = self._app.get_project()
        if not proj:
            return
        if len(self._rows) < 2:
            self._msg.configure(text=t("cls_empty"), text_color=T.DANGER)
            return
        kinds = [r.kind for r in self._rows]
        if "ok" not in kinds:
            self._msg.configure(text=t("cls_need_ok"), text_color=T.DANGER)
            return
        if "nok" not in kinds:
            self._msg.configure(text=t("cls_need_nok"), text_color=T.DANGER)
            return

        proj.classes = [ClassDef(name=r.name, kind=r.kind, color=_KIND_COLORS[r.kind])
                        for r in self._rows]
        # ensure raw capture dirs exist
        for c in proj.classes:
            proj.paths.raw_class_dir(c.name).mkdir(parents=True, exist_ok=True)
        proj.save()
        self._msg.configure(text=t("cls_saved_ok"), text_color=T.ACCENT)


# ── Single class row widget ───────────────────────────────────────────────────

class _ClassRow(ctk.CTkFrame):
    def __init__(self, master, cls_def: ClassDef, on_delete, **kwargs):
        super().__init__(master, fg_color=T.BG_INPUT, corner_radius=6, height=40, **kwargs)
        self._cls = cls_def
        self._on_delete = on_delete
        self.grid_propagate(False)
        self.grid_columnconfigure(1, weight=1)
        self._build()

    def _build(self) -> None:
        # color dot
        self._dot = ctk.CTkFrame(self, width=10, height=10,
                                   fg_color=_KIND_COLORS.get(self._cls.kind, T.TEXT_DIM),
                                   corner_radius=5)
        self._dot.grid(row=0, column=0, padx=(10, 6), pady=12)

        # name label
        self._name_lbl = ctk.CTkLabel(self, text=self._cls.name,
                                       font=T.bold(T.FONT_SM), text_color=T.TEXT_PRI, anchor="w")
        self._name_lbl.grid(row=0, column=1, sticky="w")

        # kind badge — click to cycle
        self._kind_btn = ctk.CTkButton(
            self,
            text=t(_KIND_LABELS[self._cls.kind]),
            font=T.font(T.FONT_XS), width=70, height=24,
            fg_color=_KIND_COLORS.get(self._cls.kind, T.TEXT_DIM),
            text_color="#000" if self._cls.kind != "ignore" else T.TEXT_PRI,
            corner_radius=4,
            command=self._cycle_kind,
        )
        self._kind_btn.grid(row=0, column=2, padx=6)

        # delete button
        ctk.CTkButton(self, text="✕", width=28, height=28,
                      fg_color="transparent", hover_color=T.DANGER,
                      text_color=T.TEXT_DIM, font=T.bold(T.FONT_SM),
                      command=lambda: self._on_delete(self)).grid(row=0, column=3, padx=(0, 6))

    def _cycle_kind(self) -> None:
        idx = (_KIND_CYCLE.index(self._cls.kind) + 1) % len(_KIND_CYCLE)
        self._cls.kind = _KIND_CYCLE[idx]
        color = _KIND_COLORS[self._cls.kind]
        self._dot.configure(fg_color=color)
        self._kind_btn.configure(
            text=t(_KIND_LABELS[self._cls.kind]),
            fg_color=color,
            text_color="#000" if self._cls.kind != "ignore" else T.TEXT_PRI,
        )

    @property
    def name(self) -> str:
        return self._cls.name

    @property
    def kind(self) -> str:
        return self._cls.kind
