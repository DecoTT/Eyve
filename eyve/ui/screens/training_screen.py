"""Training screen — local training + dataset package export."""
from __future__ import annotations
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import customtkinter as ctk

from eyve.ui import theme as T
from eyve.i18n import t
from tkinter import filedialog

from eyve.training.dataset_builder import validate_dataset
from eyve.training.train_manager import (TrainConfig, TrainManager, TrainProgress,
                                         last_training_date)
from eyve.training.training_packager import create_package, detect_size_label
from eyve.ui.components.dialogs import show_error, show_info

if TYPE_CHECKING:
    from eyve.ui.app import EyveApp


def _fmt_time(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m}m {sec:02d}s"
    h, mn = divmod(m, 60)
    return f"{h}h {mn:02d}m"


class TrainingScreen(ctk.CTkFrame):
    def __init__(self, master, app: "EyveApp", **kwargs):
        super().__init__(master, fg_color=T.BG_DARK, **kwargs)
        self._app = app
        self._manager: TrainManager | None = None
        self._build()
        self._check_dataset()

    # ── lifecycle (navigation) ────────────────────────────────────────────────
    def on_show(self) -> None:
        """
        Re-validate the dataset every time the user navigates here.

        Screens are cached by app._navigate(); without this, the label count
        shown was frozen at whatever it was when the screen was FIRST built —
        e.g. 'no labeled images' from before the user tagged anything.
        """
        if not (self._manager and self._manager.is_running()):
            self._check_dataset()

    # ── layout ────────────────────────────────────────────────────────────────
    def _build(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(self, fg_color=T.BG_CARD, corner_radius=0)
        hdr.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(hdr, text=t("train_title"),
                     font=T.bold(T.FONT_XL), text_color=T.TEXT_PRI).pack(
            side="left", padx=24, pady=14)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=20, pady=16)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self._build_local_panel(body)
        self._build_package_panel(body)

    # ── local training panel ──────────────────────────────────────────────────
    def _build_local_panel(self, parent) -> None:
        panel = ctk.CTkFrame(parent, fg_color=T.BG_CARD, corner_radius=10)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(panel, text=t("train_local"),
                     font=T.bold(T.FONT_LG), text_color=T.TEXT_PRI).pack(padx=20, pady=(16, 4))

        sep = ctk.CTkFrame(panel, height=1, fg_color=T.BORDER)
        sep.pack(fill="x", padx=16, pady=8)

        # dataset info
        self._dataset_lbl = ctk.CTkLabel(panel, text="",
                                          font=T.font(T.FONT_SM), text_color=T.TEXT_SEC,
                                          wraplength=320, justify="left")
        self._dataset_lbl.pack(padx=20, anchor="w")

        # warnings
        self._warn_frame = ctk.CTkFrame(panel, fg_color="transparent")
        self._warn_frame.pack(fill="x", padx=16, pady=(4, 8))

        sep2 = ctk.CTkFrame(panel, height=1, fg_color=T.BORDER)
        sep2.pack(fill="x", padx=16, pady=4)

        # settings grid
        settings = ctk.CTkFrame(panel, fg_color="transparent")
        settings.pack(fill="x", padx=16, pady=8)
        settings.grid_columnconfigure(1, weight=1)

        def row(label_key, widget_fn, r):
            ctk.CTkLabel(settings, text=t(label_key),
                         font=T.font(T.FONT_SM), text_color=T.TEXT_SEC).grid(
                row=r, column=0, sticky="w", pady=4, padx=(0, 12))
            w = widget_fn(settings)
            w.grid(row=r, column=1, sticky="ew", pady=4)

        # -- base model: project best.pt (keep improving) / stock / any .pt --
        # label -> what TrainConfig.base_model receives (stock name or abs path)
        self._model_choices: dict[str, str] = {}
        self._model_var = ctk.StringVar()
        self._model_menu = ctk.CTkOptionMenu(
            settings, variable=self._model_var, values=["yolov8n.pt"],
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD, command=self._on_model_choice)
        ctk.CTkLabel(settings, text=t("train_model_base"),
                     font=T.font(T.FONT_SM), text_color=T.TEXT_SEC).grid(
            row=0, column=0, sticky="w", pady=4, padx=(0, 12))
        self._model_menu.grid(row=0, column=1, sticky="ew", pady=4)
        self._refresh_model_choices()

        # -- only new labels since last training (fine-tuning) --------------
        self._only_new_var = ctk.BooleanVar(value=False)
        self._only_new_chk = ctk.CTkCheckBox(
            settings, text=t("train_only_new_short"), variable=self._only_new_var,
            font=T.font(T.FONT_XS), text_color=T.TEXT_SEC,
            command=self._check_dataset)
        self._only_new_chk.grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self._only_new_note = ctk.CTkLabel(
            settings, text="", font=T.font(T.FONT_XS), text_color=T.TEXT_DIM,
            wraplength=320, justify="left")
        self._only_new_note.grid(row=6, column=0, columnspan=2, sticky="w")

        self._epochs = ctk.CTkEntry(settings, fg_color=T.BG_INPUT, border_color=T.BORDER)
        self._epochs.insert(0, "50")
        ctk.CTkLabel(settings, text=t("train_epochs"),
                     font=T.font(T.FONT_SM), text_color=T.TEXT_SEC).grid(
            row=1, column=0, sticky="w", pady=4, padx=(0, 12))
        self._epochs.grid(row=1, column=1, sticky="ew", pady=4)

        self._imgsz_var = ctk.StringVar(value="640")
        ctk.CTkLabel(settings, text=t("train_imgsz"),
                     font=T.font(T.FONT_SM), text_color=T.TEXT_SEC).grid(
            row=2, column=0, sticky="w", pady=4, padx=(0, 12))
        ctk.CTkOptionMenu(settings, variable=self._imgsz_var,
                           values=["416", "480", "512", "640", "768"],
                           fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
                           dropdown_fg_color=T.BG_CARD).grid(row=2, column=1, sticky="ew", pady=4)

        self._batch_var = ctk.StringVar(value="8")
        ctk.CTkLabel(settings, text=t("train_batch"),
                     font=T.font(T.FONT_SM), text_color=T.TEXT_SEC).grid(
            row=3, column=0, sticky="w", pady=4, padx=(0, 12))
        ctk.CTkOptionMenu(settings, variable=self._batch_var,
                           values=["4", "8", "16", "32"],
                           fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
                           dropdown_fg_color=T.BG_CARD).grid(row=3, column=1, sticky="ew", pady=4)

        self._device_var = ctk.StringVar(value="auto")
        ctk.CTkLabel(settings, text=t("train_device"),
                     font=T.font(T.FONT_SM), text_color=T.TEXT_SEC).grid(
            row=4, column=0, sticky="w", pady=4, padx=(0, 12))
        ctk.CTkOptionMenu(settings, variable=self._device_var,
                           values=["auto", "cpu", "0"],
                           fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
                           dropdown_fg_color=T.BG_CARD).grid(row=4, column=1, sticky="ew", pady=4)

        # hardware note
        ctk.CTkLabel(panel, text=t("train_hardware_note"),
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM,
                     wraplength=320, justify="left").pack(padx=20, pady=(4, 8), anchor="w")

        sep3 = ctk.CTkFrame(panel, height=1, fg_color=T.BORDER)
        sep3.pack(fill="x", padx=16, pady=4)

        # progress
        self._prog_bar = ctk.CTkProgressBar(panel, mode="determinate",
                                             progress_color=T.ACCENT,
                                             fg_color=T.BG_INPUT)
        self._prog_bar.set(0)
        self._prog_bar.pack(fill="x", padx=16, pady=(8, 4))

        self._status_lbl = ctk.CTkLabel(panel, text=t("train_status_idle"),
                                         font=T.font(T.FONT_SM), text_color=T.TEXT_SEC)
        self._status_lbl.pack(padx=20, anchor="w")

        self._eta_lbl = ctk.CTkLabel(panel, text="",
                                      font=T.font(T.FONT_SM), text_color=T.TEXT_DIM)
        self._eta_lbl.pack(padx=20, pady=(0, 8), anchor="w")

        # log output
        self._log_box = ctk.CTkTextbox(panel, height=120, fg_color=T.BG_INPUT,
                                        text_color=T.TEXT_SEC, font=("Courier New", 10),
                                        state="disabled")
        self._log_box.pack(fill="x", padx=16, pady=(0, 8))

        # buttons
        btn_row = ctk.CTkFrame(panel, fg_color="transparent")
        btn_row.pack(padx=16, pady=(0, 16), anchor="w")
        self._start_btn = ctk.CTkButton(btn_row, text=t("train_start"),
                                         fg_color=T.ACCENT, text_color="#000",
                                         height=40, width=160, font=T.bold(T.FONT_MD),
                                         command=self._start_training)
        self._start_btn.pack(side="left", padx=(0, 8))
        self._stop_btn = ctk.CTkButton(btn_row, text=t("train_stop"),
                                        fg_color=T.BG_INPUT, height=40, width=120,
                                        state="disabled", command=self._stop_training)
        self._stop_btn.pack(side="left")

    # ── package panel ─────────────────────────────────────────────────────────
    def _build_package_panel(self, parent) -> None:
        panel = ctk.CTkFrame(parent, fg_color=T.BG_CARD, corner_radius=10)
        panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(panel, text=t("train_package"),
                     font=T.bold(T.FONT_LG), text_color=T.TEXT_PRI).pack(padx=20, pady=(16, 4))

        sep = ctk.CTkFrame(panel, height=1, fg_color=T.BORDER)
        sep.pack(fill="x", padx=16, pady=8)

        ctk.CTkLabel(panel, text=t("pkg_description"),
                     font=T.font(T.FONT_SM), text_color=T.TEXT_SEC,
                     wraplength=300, justify="left").pack(padx=20, anchor="w")

        sep2 = ctk.CTkFrame(panel, height=1, fg_color=T.BORDER)
        sep2.pack(fill="x", padx=16, pady=12)

        # size tiers
        ctk.CTkLabel(panel, text=t("pkg_detected_size", label="—"),
                     font=T.bold(T.FONT_SM), text_color=T.TEXT_PRI).pack(padx=20, anchor="w")
        self._pkg_size_lbl = ctk.CTkLabel(panel, text="",
                                           font=T.font(T.FONT_SM), text_color=T.TEXT_DIM)
        self._pkg_size_lbl.pack(padx=20, pady=(2, 12), anchor="w")

        for key, lbl_key in [
            ("small",   "pkg_size_small"),
            ("medium",  "pkg_size_medium"),
            ("large",   "pkg_size_large"),
            ("complex", "pkg_size_complex"),
        ]:
            ctk.CTkLabel(panel, text=t(lbl_key),
                         font=T.font(T.FONT_SM), text_color=T.TEXT_SEC).pack(
                padx=28, anchor="w", pady=1)

        sep3 = ctk.CTkFrame(panel, height=1, fg_color=T.BORDER)
        sep3.pack(fill="x", padx=16, pady=12)

        ctk.CTkLabel(panel, text=t("pkg_note"),
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM,
                     wraplength=300, justify="left").pack(padx=20, anchor="w")

        btn_row = ctk.CTkFrame(panel, fg_color="transparent")
        btn_row.pack(padx=16, pady=16, anchor="w")
        self._pkg_btn = ctk.CTkButton(btn_row, text=t("pkg_create"),
                                       fg_color=T.ACCENT2, text_color="#000",
                                       height=40, width=180, font=T.bold(T.FONT_MD),
                                       command=self._create_package)
        self._pkg_btn.pack(side="left", padx=(0, 8))
        self._pkg_open_btn = ctk.CTkButton(btn_row, text=t("pkg_open_folder"),
                                            fg_color=T.BG_INPUT, height=40, width=140,
                                            state="disabled",
                                            command=self._open_pkg_folder)
        self._pkg_open_btn.pack(side="left")

        self._pkg_status = ctk.CTkLabel(panel, text="",
                                         font=T.font(T.FONT_SM), text_color=T.TEXT_SEC,
                                         wraplength=300)
        self._pkg_status.pack(padx=20, pady=(0, 8), anchor="w")

        self._last_pkg: Path | None = None

    # ── base model choices ────────────────────────────────────────────────────
    _STOCK = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]

    def _refresh_model_choices(self, keep: str | None = None) -> None:
        """
        Rebuild the dropdown.  The project's own best.pt goes FIRST and is
        the default when it exists — "seguir mejorando el mismo modelo" is
        the normal second round, not starting from a stock YOLO again.
        """
        proj = self._app.get_project()
        self._model_choices = {}
        labels: list[str] = []
        best = proj.paths.best_model if proj else None
        if best and best.exists():
            lbl = t("train_model_project")
            self._model_choices[lbl] = str(best)
            labels.append(lbl)
        for name in self._STOCK:
            self._model_choices[name] = name
            labels.append(name)
        browse = t("train_model_browse")
        self._model_choices[browse] = ""       # sentinel -> file dialog
        labels.append(browse)
        self._model_menu.configure(values=labels)
        self._model_var.set(keep if keep in self._model_choices else labels[0])

    def _on_model_choice(self, label: str) -> None:
        if self._model_choices.get(label, None) != "":
            return                              # a real choice
        # "Elegir archivo .pt..."
        path = filedialog.askopenfilename(
            title=t("train_model_browse"),
            filetypes=[("YOLO model", "*.pt"), ("All", "*.*")])
        if not path:
            self._refresh_model_choices()       # back to default
            return
        lbl = f"{Path(path).name}  ({Path(path).parent.name})"
        self._refresh_model_choices()
        self._model_choices[lbl] = path
        vals = list(self._model_menu.cget("values"))
        vals.insert(-1, lbl)                    # before "Elegir..."
        self._model_menu.configure(values=vals)
        self._model_var.set(lbl)

    def _selected_base_model(self) -> str:
        return self._model_choices.get(self._model_var.get(), "yolov8n.pt") or "yolov8n.pt"

    def _since(self):
        """Cutoff datetime when 'only new labels' is on, else None."""
        proj = self._app.get_project()
        if not (proj and self._only_new_var.get()):
            return None
        return last_training_date(proj)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _check_dataset(self) -> None:
        proj = self._app.get_project()
        if not proj:
            return
        # the checkbox only makes sense after a first training
        last = last_training_date(proj)
        if last is None:
            self._only_new_var.set(False)
            self._only_new_chk.configure(state="disabled",
                                         text=t("train_only_new_short"))
            self._only_new_note.configure(text="")
        else:
            self._only_new_chk.configure(
                state="normal",
                text=t("train_only_new", date=last.strftime("%Y-%m-%d %H:%M")))
            self._only_new_note.configure(
                text=t("train_only_new_warn") if self._only_new_var.get() else "")

        v = validate_dataset(proj, since=self._since())
        if v.errors:
            self._dataset_lbl.configure(text=t(v.errors[0]), text_color=T.DANGER)
        elif self._only_new_var.get():
            self._dataset_lbl.configure(
                text=t("train_dataset_ok_new", n=v.labeled_images,
                       total=v.total_labeled, c=len(proj.classes)),
                text_color=T.ACCENT)
        else:
            self._dataset_lbl.configure(
                text=t("train_dataset_ok", n=v.labeled_images, c=len(proj.classes)),
                text_color=T.ACCENT
            )
        for w in self._warn_frame.winfo_children():
            w.destroy()
        for warn in v.warnings[:5]:
            ctk.CTkLabel(self._warn_frame, text=f"⚠ {warn}",
                         font=T.font(T.FONT_XS), text_color=T.WARN).pack(anchor="w")

        # package size
        n_imgs = sum(1 for _ in proj.paths.raw_images.rglob("*.jpg"))
        _, lbl_key = detect_size_label(n_imgs)
        self._pkg_size_lbl.configure(text=t(lbl_key))

    def _log(self, text: str) -> None:
        self._log_box.configure(state="normal")
        self._log_box.insert("end", text + "\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    # ── training ──────────────────────────────────────────────────────────────
    def _start_training(self) -> None:
        proj = self._app.get_project()
        if not proj:
            return
        v = validate_dataset(proj, since=self._since())
        if v.errors:
            show_error(self._app, t(v.errors[0]))
            return

        cfg = TrainConfig(
            base_model=self._selected_base_model(),
            only_new_since=self._since(),
            epochs=int(self._epochs.get() or 50),
            imgsz=int(self._imgsz_var.get()),
            batch=int(self._batch_var.get()),
            device=self._device_var.get(),
        )
        self._manager = TrainManager(proj)
        self._manager.add_callback(self._on_progress)
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._prog_bar.set(0)
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")
        self._manager.start(cfg)

    def _stop_training(self) -> None:
        if self._manager:
            self._manager.stop()

    def _on_progress(self, p: TrainProgress) -> None:
        def _update():
            if p.status == "training" and p.total_epochs > 0:
                self._prog_bar.set(p.epoch / p.total_epochs)
                self._status_lbl.configure(
                    text=t("train_status_training", epoch=p.epoch, total=p.total_epochs),
                    text_color=T.TEXT_SEC
                )
                eta = t("train_eta", time=_fmt_time(p.eta_s)) if p.eta_s > 0 else ""
                elapsed = t("train_elapsed", time=_fmt_time(p.elapsed_s))
                self._eta_lbl.configure(text=f"{elapsed}  {eta}")
                self._log(f"Epoch {p.epoch}/{p.total_epochs}  mAP50={p.map50:.3f}")
            elif p.status == "preparing":
                self._status_lbl.configure(
                    text=t("train_status_preparing"), text_color=T.TEXT_SEC)
                self._prog_bar.configure(mode="indeterminate")
                self._prog_bar.start()
            elif p.status == "done":
                self._prog_bar.stop()
                self._prog_bar.configure(mode="determinate")
                self._prog_bar.set(1.0)
                self._status_lbl.configure(
                    text=t("train_status_done"), text_color=T.ACCENT)
                if p.best_model:
                    self._eta_lbl.configure(
                        text=t("train_best_model", path=p.best_model))
                self._log(f"Done. Best model: {p.best_model}")
                self._start_btn.configure(state="normal")
                self._stop_btn.configure(state="disabled")
                # best.pt now exists / is newer: offer it as the base for the
                # next round and re-read the dataset status (checkbox state)
                self._refresh_model_choices(keep=t("train_model_project"))
                self._check_dataset()
            elif p.status == "error":
                self._prog_bar.stop()
                self._prog_bar.configure(mode="determinate")
                self._prog_bar.set(0)
                self._status_lbl.configure(
                    text=t("train_status_error", error=p.message), text_color=T.DANGER)
                self._start_btn.configure(state="normal")
                self._stop_btn.configure(state="disabled")
            elif p.status == "idle":
                self._start_btn.configure(state="normal")
                self._stop_btn.configure(state="disabled")

        self.after(0, _update)

    # ── package ───────────────────────────────────────────────────────────────
    def _create_package(self) -> None:
        proj = self._app.get_project()
        if not proj:
            return
        try:
            self._pkg_btn.configure(state="disabled", text=t("loading"))
            self._pkg_status.configure(text=t("loading"), text_color=T.TEXT_DIM)
            self.update()
            pkg = create_package(proj)
            self._last_pkg = pkg
            self._pkg_status.configure(text=t("pkg_done", path=pkg.name), text_color=T.ACCENT)
            self._pkg_open_btn.configure(state="normal")
        except Exception as e:
            self._pkg_status.configure(text=str(e), text_color=T.DANGER)
        finally:
            self._pkg_btn.configure(state="normal", text=t("pkg_create"))

    def _open_pkg_folder(self) -> None:
        if self._last_pkg and self._last_pkg.parent.exists():
            os.startfile(str(self._last_pkg.parent))
