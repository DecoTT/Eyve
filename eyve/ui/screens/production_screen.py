"""
Production / Live Feed screen.

Loads a model, starts the camera, shows live detections,
and displays OK / NOT OK / NO DETECTION / REVIEW status.

Camera starts automatically on screen entry.
If no project model is found, yolov8n.pt is downloaded in the
background from the ultralytics hub (Nano — fast enough for most hardware).
"""
from __future__ import annotations
import threading
import time
from pathlib import Path
from tkinter import filedialog
from typing import TYPE_CHECKING, Optional

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

from eyve.core import config as _cfg
from eyve.ui import theme as T
from eyve.i18n import t
from eyve.inference.detector import YOLOWorker, VideoSource
from eyve.inference.polarity import PolarityAnalyzer
from eyve.inference.tracker import InstanceTracker, RawDetection, TrackerConfig
from eyve.modules import PolarityModule, CountingModule, PatternModule
from eyve.production.ok_nok_logic import decide, InspectionStatus, InspectionResult
from eyve.production.production_session import ProductionSession
from eyve.ui.components.dialogs import show_error
from eyve.camera.camera_enum import get_camera_labels, label_to_index
from eyve.ui.hotkeys import guard_hotkey

if TYPE_CHECKING:
    from eyve.ui.app import EyveApp

_STATUS_COLORS = {
    InspectionStatus.OK:        T.COLOR_OK,
    InspectionStatus.NOT_OK:    T.COLOR_NOK,
    InspectionStatus.NO_DETECT: T.COLOR_NONE,
    InspectionStatus.REVIEW:    T.COLOR_REVIEW,
    InspectionStatus.ERROR:     T.COLOR_ERROR,
}
_STATUS_KEYS = {
    InspectionStatus.OK:        "prod_status_ok",
    InspectionStatus.NOT_OK:    "prod_status_nok",
    InspectionStatus.NO_DETECT: "prod_status_none",
    InspectionStatus.REVIEW:    "prod_status_review",
    InspectionStatus.ERROR:     "prod_status_error",
}

_SPINNER = ["◐", "◓", "◑", "◒"]

# FPS-cap values offered in Settings, in display order
FPS_OPTIONS: list[int] = [0, 15, 30, 60, 120]   # 0 = uncapped
FPS_LABELS:  dict[int, str] = {
    0:   "Sin límite",
    15:  "15 FPS",
    30:  "30 FPS  ★",
    60:  "60 FPS",
    120: "120 FPS",
}


def _loop_delay() -> int:
    """
    Return the after() delay (ms) for the production loop based on the
    'prod_fps_cap' config key.

    cap = 0  → 1 ms  (uncapped — camera & inference are the real bottleneck)
    cap = n  → 1000 // n  ms  (e.g. 30 → 33 ms, 60 → 16 ms)

    Reading config fresh every tick means the user can change the setting in
    Settings and the new value takes effect immediately without restart.
    """
    cap = int(_cfg.get("prod_fps_cap", 30))
    if cap <= 0:
        return 1
    return max(1, 1000 // cap)


def _color_tint(hex_color: str, alpha: float = 0.15) -> str:
    """
    Blend *hex_color* (e.g. '#00e676') over a near-black background at
    *alpha* opacity and return a valid 6-char Tkinter hex color.
    Tkinter does NOT support 8-character #RRGGBBAA colors.
    """
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        bg = 18   # ≈ #121212 dark background
        r2 = int(r * alpha + bg * (1 - alpha))
        g2 = int(g * alpha + bg * (1 - alpha))
        b2 = int(b * alpha + bg * (1 - alpha))
        return f"#{r2:02x}{g2:02x}{b2:02x}"
    except Exception:
        return T.BG_INPUT


class ProductionScreen(ctk.CTkFrame):
    def __init__(self, master, app: "EyveApp", **kwargs):
        super().__init__(master, fg_color=T.BG_DARK, **kwargs)
        self._app = app
        self._worker: Optional[YOLOWorker] = None
        self._source: Optional[VideoSource] = None
        self._session: Optional[ProductionSession] = None
        self._running = False
        self._paused = False
        self._photo = None
        self._last_result = None
        self._fps_times: list[float] = []
        self._frame_count = 0
        self._start_time = 0.0
        self._worker_loading = False   # True while loading model in background
        self._spin_step = 0            # spinner animation step
        self._warming = False          # True while camera is opening
        self._cam_starting = False     # True while VideoSource.start() is in-flight
        # Video-file source (remote validation without a camera)
        self._prod_video_path: Optional[Path] = None
        # Persistent instance tracking (ported from 2.0 Persistent_Instance):
        # stable IDs across frames kill detection flicker; modules process
        # each instance ONCE instead of every frame.
        self._tracker = InstanceTracker(
            modules=["polarity"],
            config=TrackerConfig(
                iou_match          = float(_cfg.get("track_iou", 0.25)),
                max_lost_frames    = int(_cfg.get("track_lost", 12)),
                min_confirm_frames = int(_cfg.get("track_confirm", 2)),
                process_conf_min   = float(_cfg.get("track_conf_min", 0.40)),
            ))
        # Inspection modules (Eyve Pro)
        self._polarity = PolarityModule()
        self._counting = CountingModule()
        self._pattern = PatternModule()
        self._pat_last = ""
        self._pat_cal_left = 0
        self._mod_last_status = ""     # cache to avoid configure() every frame
        self._count_last = ""
        # canvas→frame mapping for drawing the counting line (set per frame)
        self._view: Optional[tuple] = None   # (fw, fh, nw, nh, ox, oy)
        self._meta_drawing = False           # True while user drags the geometry
        self._meta_start: Optional[tuple[int, int]] = None
        self._meta_kind = "line"             # "line" | "rect" — set when armed
        self._build()

        # Space-bar hot-key (bound before auto-start so it's always available)
        top = self.winfo_toplevel()
        self._hotkey_ids = [top.bind(
            "<space>", guard_hotkey(self, lambda e: self._space_action()), add=True)]

        # Model load happens once at creation time (background thread).
        # Camera auto-start is handled by on_show() so it fires exactly once
        # per navigation, with no duplicate scheduling from __init__.
        self.after(150, self._try_autoload_model)

    # ── layout ────────────────────────────────────────────────────────────────
    def _build(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # header
        hdr = ctk.CTkFrame(self, fg_color=T.BG_CARD, corner_radius=0)
        hdr.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(hdr, text=t("prod_title"),
                     font=T.bold(T.FONT_XL), text_color=T.TEXT_PRI).pack(
            side="left", padx=24, pady=12)

        # body
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0)
        body.grid_rowconfigure(0, weight=1)

        # canvas
        canvas_frame = ctk.CTkFrame(body, fg_color="#000", corner_radius=8)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.grid_propagate(False)
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(canvas_frame, bg="#0a0a0a", highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        # finish-line drawing for the Counting module
        self._canvas.bind("<ButtonPress-1>",   self._meta_press)
        self._canvas.bind("<B1-Motion>",       self._meta_drag)
        self._canvas.bind("<ButtonRelease-1>", self._meta_release)

        # right panel.  Scrollable: con los 5 metodos de conteo y el panel de
        # persistencia, el contenido pasa de la altura de la ventana en 1080p.
        right = ctk.CTkScrollableFrame(body, fg_color=T.BG_CARD, corner_radius=10,
                                       width=250, scrollbar_button_color=T.BG_INPUT,
                                       scrollbar_button_hover_color=T.BORDER)
        right.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        self._build_right(right)

    def _build_right(self, parent) -> None:
        # ── big status display ────────────────────────────────────────────────
        self._status_frame = ctk.CTkFrame(parent, fg_color=T.BG_INPUT,
                                           corner_radius=10, height=120)
        self._status_frame.pack(fill="x", padx=12, pady=(16, 8))
        self._status_frame.pack_propagate(False)

        self._status_lbl = ctk.CTkLabel(
            self._status_frame, text="—",
            font=T.bold(T.FONT_XXL), text_color=T.TEXT_DIM,
        )
        self._status_lbl.place(relx=0.5, rely=0.5, anchor="center")

        # ── counters row ──────────────────────────────────────────────────────
        count_row = ctk.CTkFrame(parent, fg_color="transparent")
        count_row.pack(fill="x", padx=12, pady=(4, 8))
        self._ok_lbl = ctk.CTkLabel(count_row, text=t("prod_count_ok", n=0),
                                     font=T.bold(T.FONT_SM), text_color=T.COLOR_OK)
        self._ok_lbl.pack(side="left", expand=True)
        self._nok_lbl = ctk.CTkLabel(count_row, text=t("prod_count_nok", n=0),
                                      font=T.bold(T.FONT_SM), text_color=T.COLOR_NOK)
        self._nok_lbl.pack(side="left", expand=True)
        ctk.CTkButton(count_row, text="↺", width=28, height=22,
                      font=T.font(T.FONT_XS), fg_color="transparent",
                      text_color=T.TEXT_DIM, hover_color=T.BG_INPUT,
                      command=self._reset_counts).pack(side="right")

        # ── model section ─────────────────────────────────────────────────────
        ctk.CTkLabel(parent, text=t("prod_load_model"),
                     font=T.bold(T.FONT_SM), text_color=T.TEXT_SEC).pack(
            padx=12, anchor="w")

        self._model_lbl = ctk.CTkLabel(parent, text=t("prod_no_model"),
                                        font=T.font(T.FONT_XS), text_color=T.TEXT_DIM,
                                        wraplength=200, anchor="w", justify="left")
        self._model_lbl.pack(padx=12, pady=(2, 4), anchor="w")

        ctk.CTkButton(parent, text=t("prod_load_model"),
                      fg_color=T.BG_INPUT, border_width=1, border_color=T.BORDER,
                      height=30, command=self._load_model).pack(fill="x", padx=12, pady=(0, 8))

        sep = ctk.CTkFrame(parent, height=1, fg_color=T.BORDER)
        sep.pack(fill="x", padx=12, pady=4)

        # ── source: camera or video file ──────────────────────────────────────
        src_row = ctk.CTkFrame(parent, fg_color="transparent")
        src_row.pack(fill="x", padx=12, pady=(4, 0))
        self._prod_src_var = ctk.StringVar(value="camera")
        ctk.CTkRadioButton(src_row, text=t("cap_source_camera"),
                           variable=self._prod_src_var, value="camera",
                           font=T.font(T.FONT_XS),
                           command=self._on_prod_source_change).pack(side="left")
        ctk.CTkRadioButton(src_row, text=t("cap_source_video"),
                           variable=self._prod_src_var, value="video",
                           font=T.font(T.FONT_XS),
                           command=self._on_prod_source_change).pack(side="left", padx=10)

        # ── camera / conf ─────────────────────────────────────────────────────
        cam_row = ctk.CTkFrame(parent, fg_color="transparent")
        cam_row.pack(fill="x", padx=12, pady=4)
        ctk.CTkLabel(cam_row, text=t("cap_camera_id"),
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(anchor="w")
        self._cam_idx = ctk.CTkOptionMenu(
            cam_row, values=[t("cam_loading")],
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD,
            font=T.font(T.FONT_XS),
            state="disabled",
        )
        self._cam_idx.pack(fill="x")
        self._cam_row = cam_row
        threading.Thread(target=self._load_cameras_bg, daemon=True).start()

        # video-file picker (hidden while source == camera)
        self._vid_row = ctk.CTkFrame(parent, fg_color="transparent")
        self._prod_vid_btn = ctk.CTkButton(
            self._vid_row, text=t("prod_select_video"),
            fg_color=T.BG_INPUT, height=28, font=T.font(T.FONT_XS),
            command=self._browse_prod_video,
        )
        self._prod_vid_btn.pack(fill="x")
        self._prod_vid_lbl = ctk.CTkLabel(
            self._vid_row, text="", font=T.font(T.FONT_XS),
            text_color=T.TEXT_DIM, wraplength=200, anchor="w",
        )
        self._prod_vid_lbl.pack(anchor="w")

        conf_row = ctk.CTkFrame(parent, fg_color="transparent")
        self._conf_row = conf_row   # anchor for packing the video picker above
        conf_row.pack(fill="x", padx=12, pady=4)
        ctk.CTkLabel(conf_row, text="Conf %",
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(side="left")
        self._conf_slider = ctk.CTkSlider(conf_row, from_=10, to=95,
                                           width=120, command=self._on_conf_change)
        self._conf_slider.set(50)
        self._conf_slider.pack(side="right")
        self._conf_val = ctk.CTkLabel(conf_row, text="50%",
                                       font=T.font(T.FONT_XS), text_color=T.TEXT_SEC)
        self._conf_val.pack(side="right", padx=4)

        sep2 = ctk.CTkFrame(parent, height=1, fg_color=T.BORDER)
        sep2.pack(fill="x", padx=12, pady=4)

        # ── inspection modules (Eyve Pro glimpse) ─────────────────────────────
        ctk.CTkLabel(parent, text=t("prod_modules"),
                     font=T.bold(T.FONT_SM), text_color=T.TEXT_SEC).pack(
            padx=12, anchor="w")

        # Puerta por nivel (licencia por honor: es lo único que cambia entre
        # niveles). Los módulos de check de SBC son Pro; conteo y log son de todos.
        lm = getattr(self._app, "license", None)
        self._pro_ok = lm.permite_checks_avanzados() if lm else False
        if lm and not lm.es_pro():
            ctk.CTkLabel(parent, text=t("lic_prod_notice", level=lm.nivel_label),
                         font=T.font(T.FONT_XS), text_color=T.WARN,
                         wraplength=220, justify="left").pack(padx=12, anchor="w", pady=(0, 4))

        self._pol_var = ctk.BooleanVar(value=False)
        self._pol_chk = ctk.CTkCheckBox(parent, text=self._polarity.name,
                        variable=self._pol_var,
                        font=T.font(T.FONT_XS), text_color=T.TEXT_PRI,
                        command=self._on_polarity_toggle)
        self._pol_chk.pack(padx=12, anchor="w", pady=(2, 0))
        if not self._pro_ok:
            self._pol_chk.configure(state="disabled", text=f"{self._polarity.name}  (Pro)")
            ctk.CTkLabel(parent, text=t("lic_pro_required"),
                         font=T.font(T.FONT_XS), text_color=T.TEXT_DIM,
                         wraplength=220, justify="left").pack(padx=12, anchor="w", pady=(0, 2))

        pol_cfg = ctk.CTkFrame(parent, fg_color="transparent")
        pol_cfg.pack(fill="x", padx=12, pady=(2, 2))

        proj = self._app.get_project()
        cls_names = [c.name for c in proj.classes] if proj and proj.classes else ["—"]
        row_c = ctk.CTkFrame(pol_cfg, fg_color="transparent"); row_c.pack(fill="x", pady=1)
        ctk.CTkLabel(row_c, text=t("prod_mod_class"), width=64, anchor="w",
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(side="left")
        self._pol_class = ctk.CTkOptionMenu(
            row_c, values=cls_names, width=120, height=24,
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD, font=T.font(T.FONT_XS),
            command=lambda v: setattr(self._polarity, "target_class", v))
        self._pol_class.pack(side="right")

        row_m = ctk.CTkFrame(pol_cfg, fg_color="transparent"); row_m.pack(fill="x", pady=1)
        ctk.CTkLabel(row_m, text=t("prod_mod_method"), width=64, anchor="w",
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(side="left")
        self._pol_method = ctk.CTkOptionMenu(
            row_m, values=list(PolarityModule.METHODS), width=120, height=24,
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD, font=T.font(T.FONT_XS),
            command=self._on_pol_method_change)
        self._pol_method.pack(side="right")

        row_r = ctk.CTkFrame(pol_cfg, fg_color="transparent"); row_r.pack(fill="x", pady=1)
        ctk.CTkLabel(row_r, text=t("prod_mod_ref"), width=64, anchor="w",
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(side="left")
        self._pol_ref = ctk.CTkOptionMenu(
            row_r, values=list(PolarityModule.REFERENCES), width=120, height=24,
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD, font=T.font(T.FONT_XS),
            command=self._on_pol_ref_change)
        self._pol_ref.pack(side="right")

        # arc thickness — on-the-fly tuning for the stripe method
        row_a = ctk.CTkFrame(pol_cfg, fg_color="transparent"); row_a.pack(fill="x", pady=1)
        ctk.CTkLabel(row_a, text=t("prod_arc"), width=64, anchor="w",
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(side="left")
        self._pol_arc_val = ctk.CTkLabel(row_a, text="15%", width=34,
                                          font=T.font(T.FONT_XS), text_color=T.TEXT_SEC)
        self._pol_arc_val.pack(side="right")
        self._pol_arc = ctk.CTkSlider(row_a, from_=8, to=35, width=86,
                                       command=self._on_pol_arc_change)
        self._pol_arc.set(15)
        self._pol_arc.pack(side="right", padx=4)

        self._pol_status = ctk.CTkLabel(parent, text="",
                                         font=T.font(T.FONT_XS), text_color=T.TEXT_DIM,
                                         wraplength=200, anchor="w")
        self._pol_status.pack(padx=12, anchor="w")

        # ── counting module ───────────────────────────────────────────────────
        self._count_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(parent, text=self._counting.name,
                        variable=self._count_var,
                        font=T.font(T.FONT_XS), text_color=T.TEXT_PRI,
                        command=self._on_counting_toggle).pack(
            padx=12, anchor="w", pady=(4, 0))

        cnt_cfg = ctk.CTkFrame(parent, fg_color="transparent")
        cnt_cfg.pack(fill="x", padx=12, pady=(2, 0))
        row_cc = ctk.CTkFrame(cnt_cfg, fg_color="transparent"); row_cc.pack(fill="x", pady=1)
        ctk.CTkLabel(row_cc, text=t("prod_mod_class"), width=64, anchor="w",
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(side="left")
        self._count_class = ctk.CTkOptionMenu(
            row_cc, values=[t("prod_mod_all")] + (cls_names if cls_names != ["—"] else []),
            width=120, height=24,
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD, font=T.font(T.FONT_XS),
            command=self._on_count_class_change)
        self._count_class.pack(side="right")

        # metodo de conteo — cambia que evento se cuenta y que controles salen
        row_cm = ctk.CTkFrame(cnt_cfg, fg_color="transparent"); row_cm.pack(fill="x", pady=1)
        ctk.CTkLabel(row_cm, text=t("prod_mod_method"), width=64, anchor="w",
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(side="left")
        self._count_method = ctk.CTkOptionMenu(
            row_cm, values=[t("count_m_" + k) for k in CountingModule.METHODS],
            width=120, height=24,
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD, font=T.font(T.FONT_XS),
            command=self._on_count_method_change)
        self._count_method.set(t("count_m_" + self._counting.method))
        self._count_method.pack(side="right")

        # explicacion de una linea del metodo activo — nadie adivina que
        # significa "aparicion" sin leerla
        self._count_help = ctk.CTkLabel(
            cnt_cfg, text="", font=T.font(T.FONT_XS), text_color=T.TEXT_DIM,
            wraplength=210, justify="left", anchor="w")
        self._count_help.pack(fill="x", pady=(2, 2))

        # ── filas condicionales (una por metodo) ──────────────────────────────
        # Se crean todas y se muestran con pack/pack_forget segun el metodo:
        # reconstruir widgets al cambiar de metodo deja huerfanos en CTk.
        self._row_dir = ctk.CTkFrame(cnt_cfg, fg_color="transparent")
        ctk.CTkLabel(self._row_dir, text=t("count_sense"), width=64, anchor="w",
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(side="left")
        self._count_dir = ctk.CTkOptionMenu(
            self._row_dir,
            values=[t("count_d_" + k) for k in CountingModule.DIRECTIONS],
            width=120, height=24,
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD, font=T.font(T.FONT_XS),
            command=self._on_count_dir_change)
        self._count_dir.set(t("count_d_" + self._counting.direction))
        self._count_dir.pack(side="right")

        self._row_trig = ctk.CTkFrame(cnt_cfg, fg_color="transparent")
        ctk.CTkLabel(self._row_trig, text=t("count_counts"), width=64, anchor="w",
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(side="left")
        self._count_trig = ctk.CTkOptionMenu(
            self._row_trig,
            values=[t("count_z_" + k) for k in CountingModule.ZONE_TRIGGERS],
            width=120, height=24,
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD, font=T.font(T.FONT_XS),
            command=self._on_count_trig_change)
        self._count_trig.set(t("count_z_" + self._counting.zone_trigger))
        self._count_trig.pack(side="right")

        self._row_edge = ctk.CTkFrame(cnt_cfg, fg_color="transparent")
        ctk.CTkLabel(self._row_edge, text=t("count_exit_by"), width=64, anchor="w",
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(side="left")
        self._count_edge = ctk.CTkOptionMenu(
            self._row_edge,
            values=[t("count_e_" + k) for k in CountingModule.EDGES],
            width=120, height=24,
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD, font=T.font(T.FONT_XS),
            command=self._on_count_edge_change)
        self._count_edge.set(t("count_e_" + self._counting.edge))
        self._count_edge.pack(side="right")

        # rango esperado — es lo que convierte el conteo en criterio OK/NOK
        self._row_exp = ctk.CTkFrame(cnt_cfg, fg_color="transparent")
        ctk.CTkLabel(self._row_exp, text=t("count_expected"), width=64, anchor="w",
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(side="left")
        self._exp_max = ctk.CTkEntry(self._row_exp, width=42, height=24,
                                     font=T.font(T.FONT_XS), fg_color=T.BG_INPUT,
                                     border_width=1, border_color=T.BORDER,
                                     justify="center", placeholder_text="max")
        self._exp_max.pack(side="right")
        ctk.CTkLabel(self._row_exp, text="–", font=T.font(T.FONT_XS),
                     text_color=T.TEXT_DIM).pack(side="right", padx=3)
        self._exp_min = ctk.CTkEntry(self._row_exp, width=42, height=24,
                                     font=T.font(T.FONT_XS), fg_color=T.BG_INPUT,
                                     border_width=1, border_color=T.BORDER,
                                     justify="center", placeholder_text="min")
        self._exp_min.pack(side="right")
        for _e in (self._exp_min, self._exp_max):
            _e.bind("<KeyRelease>", self._on_count_expect_change)
            _e.bind("<FocusOut>",   self._on_count_expect_change)

        # ── dibujar geometria ─────────────────────────────────────────────────
        self._row_draw = ctk.CTkFrame(cnt_cfg, fg_color="transparent")
        self._meta_btn = ctk.CTkButton(
            self._row_draw, text=t("prod_draw_line"), height=26,
            fg_color=T.BG_INPUT, border_width=1, border_color=T.BORDER,
            font=T.font(T.FONT_XS), command=self._arm_meta_draw)
        self._meta_btn.pack(side="left", fill="x", expand=True)

        # ── lectura + reinicio ────────────────────────────────────────────────
        row_cr = ctk.CTkFrame(cnt_cfg, fg_color="transparent"); row_cr.pack(fill="x", pady=2)
        self._count_lbl = ctk.CTkLabel(row_cr, text="",
                                        font=T.bold(T.FONT_SM), text_color=T.ACCENT2,
                                        wraplength=170, anchor="w", justify="left")
        self._count_lbl.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(row_cr, text="↺", width=28, height=26,
                      font=T.font(T.FONT_XS), fg_color="transparent",
                      text_color=T.TEXT_DIM, hover_color=T.BG_INPUT,
                      command=self._reset_counting).pack(side="right")

        # ── persistencia de instancia (plegable) ──────────────────────────────
        # "Que tanto seguimos reconociendo la misma instancia": hasta 2.1.0
        # estos cuatro valores solo se podian cambiar editando tracker.py.
        self._persist_open = False
        self._persist_btn = ctk.CTkButton(
            cnt_cfg, text="▸  " + t("count_persistence"), height=24, anchor="w",
            fg_color="transparent", hover_color=T.BG_INPUT,
            text_color=T.TEXT_DIM, font=T.font(T.FONT_XS),
            command=self._toggle_persist)
        self._persist_btn.pack(fill="x", pady=(4, 0))

        self._persist_box = ctk.CTkFrame(cnt_cfg, fg_color="transparent")
        _tc = self._tracker.config
        self._persist_sliders = {}
        for _key, _label, _lo, _hi, _val, _fmt in [
            ("track_lost",     t("count_tolerance"), 0,  40,
             _tc.max_lost_frames,         lambda v: "%d f" % int(v)),
            ("track_confirm",  t("count_confirm"),   1,  10,
             _tc.min_confirm_frames,      lambda v: "%d f" % int(v)),
            ("track_iou",      t("count_iou"),       5,  80,
             _tc.iou_match * 100,         lambda v: "%d%%" % int(v)),
            ("track_conf_min", t("count_conf_min"),  10, 95,
             _tc.process_conf_min * 100,  lambda v: "%d%%" % int(v)),
        ]:
            _r = ctk.CTkFrame(self._persist_box, fg_color="transparent")
            _r.pack(fill="x", pady=1)
            ctk.CTkLabel(_r, text=_label, width=72, anchor="w",
                         font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(side="left")
            _vlbl = ctk.CTkLabel(_r, text=_fmt(_val), width=32,
                                 font=T.font(T.FONT_XS), text_color=T.TEXT_SEC)
            _vlbl.pack(side="right")
            _sl = ctk.CTkSlider(_r, from_=_lo, to=_hi, width=84,
                                command=lambda v, k=_key: self._on_persist_change(k, v))
            _sl.set(_val)
            _sl.pack(side="right", padx=4)
            self._persist_sliders[_key] = (_sl, _vlbl, _fmt)

        ctk.CTkLabel(self._persist_box, text=t("count_persistence_hint"),
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM,
                     wraplength=210, justify="left", anchor="w").pack(
            fill="x", pady=(2, 4))

        self._refresh_count_rows()

        sep2a = ctk.CTkFrame(parent, height=1, fg_color=T.BORDER)
        sep2a.pack(fill="x", padx=12, pady=4)

        # ── modulo Patron: inspeccion sin clases ──────────────────────────────
        self._pat_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(parent, text=t("pat_title"), variable=self._pat_var,
                        font=T.font(T.FONT_XS), text_color=T.TEXT_PRI,
                        command=self._on_pattern_toggle).pack(
            padx=12, anchor="w", pady=(2, 0))

        pat_cfg = ctk.CTkFrame(parent, fg_color="transparent")
        pat_cfg.pack(fill="x", padx=12, pady=(2, 0))

        row_pm = ctk.CTkFrame(pat_cfg, fg_color="transparent")
        row_pm.pack(fill="x", pady=1)
        ctk.CTkLabel(row_pm, text=t("pat_method"), width=64, anchor="w",
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(side="left")
        self._pat_method = ctk.CTkOptionMenu(
            row_pm, values=[t("pat_m_" + k) for k in PatternModule.METHODS],
            width=120, height=24, fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD, font=T.font(T.FONT_XS),
            command=self._on_pat_method)
        self._pat_method.set(t("pat_m_" + self._pattern.method))
        self._pat_method.pack(side="right")

        self._pat_help = ctk.CTkLabel(
            pat_cfg, text=t("pat_help_" + self._pattern.method),
            font=T.font(T.FONT_XS), text_color=T.TEXT_DIM,
            wraplength=210, justify="left", anchor="w")
        self._pat_help.pack(fill="x", pady=(2, 2))

        row_ps = ctk.CTkFrame(pat_cfg, fg_color="transparent")
        row_ps.pack(fill="x", pady=1)
        ctk.CTkLabel(row_ps, text=t("pat_sens"), width=72, anchor="w",
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(side="left")
        self._pat_sens_val = ctk.CTkLabel(row_ps, text="70", width=28,
                                           font=T.font(T.FONT_XS),
                                           text_color=T.TEXT_SEC)
        self._pat_sens_val.pack(side="right")
        self._pat_sens = ctk.CTkSlider(row_ps, from_=0, to=100, width=86,
                                        command=self._on_pat_sens)
        self._pat_sens.set(70)
        self._pat_sens.pack(side="right", padx=4)
        self._pattern.sensitivity = 70.0

        self._pat_cal_btn = ctk.CTkButton(
            pat_cfg, text=t("pat_calibrate"), height=26,
            fg_color=T.BG_INPUT, border_width=1, border_color=T.BORDER,
            font=T.font(T.FONT_XS), command=self._calibrate_pattern)
        self._pat_cal_btn.pack(fill="x", pady=2)

        self._pat_status = ctk.CTkLabel(
            pat_cfg, text=t("pat_uncalibrated"), font=T.font(T.FONT_XS),
            text_color=T.WARN, wraplength=210, justify="left", anchor="w")
        self._pat_status.pack(fill="x")

        sep2b = ctk.CTkFrame(parent, height=1, fg_color=T.BORDER)
        sep2b.pack(fill="x", padx=12, pady=4)

        # ── options ───────────────────────────────────────────────────────────
        self._save_nok_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(parent, text=t("prod_save_nok"),
                        variable=self._save_nok_var,
                        font=T.font(T.FONT_XS), text_color=T.TEXT_SEC).pack(
            padx=12, anchor="w", pady=2)

        self._save_log_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(parent, text=t("prod_save_logs"),
                        variable=self._save_log_var,
                        font=T.font(T.FONT_XS), text_color=T.TEXT_SEC).pack(
            padx=12, anchor="w", pady=2)

        sep3 = ctk.CTkFrame(parent, height=1, fg_color=T.BORDER)
        sep3.pack(fill="x", padx=12, pady=8)

        # ── start / stop / pause ─────────────────────────────────────────────
        self._toggle_btn = ctk.CTkButton(
            parent, text=t("prod_start"),
            fg_color=T.ACCENT, text_color="#000",
            height=44, font=T.bold(T.FONT_MD),
            command=self._toggle,
        )
        self._toggle_btn.pack(fill="x", padx=12, pady=(4, 2))

        self._pause_btn = ctk.CTkButton(
            parent, text=t("prod_pause"),
            fg_color=T.BG_INPUT, height=32,
            font=T.font(T.FONT_SM), state="disabled",
            command=self._toggle_pause,
        )
        self._pause_btn.pack(fill="x", padx=12, pady=(0, 4))

        sep4 = ctk.CTkFrame(parent, height=1, fg_color=T.BORDER)
        sep4.pack(fill="x", padx=12, pady=8)

        # ── stats ─────────────────────────────────────────────────────────────
        self._fps_lbl = ctk.CTkLabel(parent, text=t("prod_fps", fps=0),
                                      font=T.font(T.FONT_SM), text_color=T.TEXT_DIM)
        self._fps_lbl.pack(padx=12, anchor="w", pady=1)

        self._session_lbl = ctk.CTkLabel(parent, text="",
                                          font=T.font(T.FONT_XS), text_color=T.TEXT_DIM,
                                          wraplength=200, anchor="w")
        self._session_lbl.pack(padx=12, anchor="w", pady=1)

        self._elapsed_lbl = ctk.CTkLabel(parent, text="",
                                          font=T.font(T.FONT_XS), text_color=T.TEXT_DIM)
        self._elapsed_lbl.pack(padx=12, anchor="w", pady=1)

        self._det_lbl = ctk.CTkLabel(parent, text="",
                                      font=T.font(T.FONT_XS), text_color=T.TEXT_DIM,
                                      wraplength=200, anchor="w")
        self._det_lbl.pack(padx=12, anchor="w", pady=(1, 12))

    # ── camera enumeration (async) ────────────────────────────────────────────
    def _load_cameras_bg(self) -> None:
        labels = get_camera_labels()
        self.after(0, lambda: self._on_cameras_loaded(labels))

    def _on_cameras_loaded(self, labels: list[str]) -> None:
        if not self.winfo_exists():
            return
        if not labels:
            labels = [t("cam_none")]
        self._cam_idx.configure(values=labels, state="normal")
        self._cam_idx.set(labels[0])

    # ── source selector (camera / video file) ────────────────────────────────
    def _on_prod_source_change(self) -> None:
        if self._running or self._cam_starting:
            self._stop()
        if self._prod_src_var.get() == "video":
            self._cam_row.pack_forget()
            self._vid_row.pack(fill="x", padx=12, pady=4, before=self._conf_row)
        else:
            self._vid_row.pack_forget()
            self._cam_row.pack(fill="x", padx=12, pady=4, before=self._conf_row)
            self.after(100, self._auto_start_preview)

    def _browse_prod_video(self) -> None:
        path = filedialog.askopenfilename(
            title=t("prod_select_video"),
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")]
        )
        if not path:
            return
        self._prod_video_path = Path(path)
        self._prod_vid_lbl.configure(text=path, text_color=T.TEXT_DIM)
        # switch playback to the selected file immediately
        if self._running or self._cam_starting:
            self._stop()
        self._start()

    # ── polarity module UI ───────────────────────────────────────────────────
    def _on_polarity_toggle(self) -> None:
        if self._pol_var.get() and not self._pro_ok:
            self._pol_var.set(False)
            return
        if self._pol_var.get():
            cls = self._pol_class.get()
            if not cls or cls == "—":
                self._pol_var.set(False)
                self._pol_status.configure(text=t("prod_mod_need_class"),
                                            text_color=T.WARN)
                return
            self._polarity.target_class = cls
            self._polarity.set_method(self._pol_method.get())
            self._polarity.set_reference(self._pol_ref.get())
            self._polarity.enabled = True
            self._pol_status.configure(text=self._polarity.learning_status,
                                        text_color=T.TEXT_DIM)
        else:
            self._polarity.enabled = False
            self._mod_last_status = ""
            self._pol_status.configure(text="")

    def _rearm_polarity(self) -> None:
        """Re-analyze live tracks after an on-the-fly parameter change."""
        for tr in self._tracker.get_all():
            tr._proc_state["polarity"] = "pending"

    def _on_pol_method_change(self, method: str) -> None:
        self._polarity.set_method(method)
        self._rearm_polarity()

    def _on_pol_ref_change(self, ref: str) -> None:
        self._polarity.set_reference(ref)
        self._rearm_polarity()

    def _on_pol_arc_change(self, val) -> None:
        v = float(val) / 100.0
        self._polarity.set_arc_thickness(v)
        self._pol_arc_val.configure(text=f"{int(float(val))}%")
        self._rearm_polarity()

    # ── counting module UI ───────────────────────────────────────────────────
    def _on_counting_toggle(self) -> None:
        self._counting.enabled = self._count_var.get()
        if not self._counting.enabled:
            self._count_lbl.configure(text="")
        else:
            self._refresh_count_rows()

    def _on_count_class_change(self, val: str) -> None:
        self._counting.target_class = None if val == t("prod_mod_all") else val
        self._counting.reset()
        self._count_last = ""

    def _on_count_method_change(self, label: str) -> None:
        """Translated label back to the internal method key."""
        key = next((k for k in CountingModule.METHODS
                    if t("count_m_" + k) == label), None)
        if key is None:
            return
        self._counting.set_method(key)     # resetea el acumulado a proposito
        self._count_last = ""
        self._refresh_count_rows()

    def _on_count_dir_change(self, label: str) -> None:
        key = next((k for k in CountingModule.DIRECTIONS
                    if t("count_d_" + k) == label), None)
        if key:
            self._counting.direction = key
            self._count_last = ""

    def _on_count_trig_change(self, label: str) -> None:
        key = next((k for k in CountingModule.ZONE_TRIGGERS
                    if t("count_z_" + k) == label), None)
        if key:
            self._counting.zone_trigger = key
            self._counting.reset()
            self._count_last = ""

    def _on_count_edge_change(self, label: str) -> None:
        key = next((k for k in CountingModule.EDGES
                    if t("count_e_" + k) == label), None)
        if key:
            self._counting.edge = key
            self._count_last = ""

    def _on_count_expect_change(self, _event=None) -> None:
        """
        Empty field = no expectation on that end (the module only vetoes
        when at least one bound is set).  Garbage text is ignored, not an
        error dialog: the user is mid-typing on every KeyRelease.
        """
        def parse(entry):
            txt = entry.get().strip()
            if not txt:
                return None
            try:
                return max(0, int(txt))
            except ValueError:
                return None
        self._counting.expect_min = parse(self._exp_min)
        self._counting.expect_max = parse(self._exp_max)
        self._count_last = ""

    def _toggle_persist(self) -> None:
        self._persist_open = not self._persist_open
        if self._persist_open:
            self._persist_box.pack(fill="x", pady=(2, 0))
            self._persist_btn.configure(text="\u25be  " + t("count_persistence"))
        else:
            self._persist_box.pack_forget()
            self._persist_btn.configure(text="\u25b8  " + t("count_persistence"))

    #: config key -> TrackerConfig field
    _PERSIST_FIELDS = {
        "track_lost":     "max_lost_frames",
        "track_confirm":  "min_confirm_frames",
        "track_iou":      "iou_match",
        "track_conf_min": "process_conf_min",
    }

    def _on_persist_change(self, key: str, value: float) -> None:
        """
        Tracker persistence sliders.  Applied live to the running tracker and
        persisted in app config, so the next session starts with the line
        already tuned instead of back at the defaults.
        """
        _sl, vlbl, fmt = self._persist_sliders[key]
        vlbl.configure(text=fmt(value))
        if key in ("track_iou", "track_conf_min"):
            stored = round(value / 100.0, 2)
        else:
            stored = int(value)
        _cfg.set(key, stored)
        self._tracker.configure(**{self._PERSIST_FIELDS[key]: stored})

    def _refresh_count_rows(self) -> None:
        """
        Show only the controls the active method actually uses, and update the
        one-line explanation + draw-button label.
        """
        m = self._counting.method
        for row in (self._row_dir, self._row_trig, self._row_edge,
                    self._row_exp, self._row_draw):
            row.pack_forget()
        if m == "line":
            self._row_dir.pack(fill="x", pady=1)
            self._row_draw.pack(fill="x", pady=2)
        elif m == "zone":
            self._row_trig.pack(fill="x", pady=1)
            self._row_exp.pack(fill="x", pady=1)
            self._row_draw.pack(fill="x", pady=2)
        elif m == "disappear":
            self._row_edge.pack(fill="x", pady=1)
        elif m == "screen":
            self._row_exp.pack(fill="x", pady=1)

        self._count_help.configure(text=t("count_help_" + m))

        kind = self._counting.geometry_kind
        if kind == "rect":
            self._meta_btn.configure(text=t("prod_draw_zone"))
        else:
            self._meta_btn.configure(text=t("prod_draw_line"))

        if self._counting.enabled and self._counting.needs_geometry:
            hint = t("prod_zone_hint") if kind == "rect" else t("prod_line_hint")
            self._count_lbl.configure(text=hint, text_color=T.TEXT_DIM)
        elif self._counting.enabled:
            self._count_lbl.configure(text=self._counting.summary(),
                                      text_color=T.ACCENT2)

    # ── modulo Patron ────────────────────────────────────────────────────────
    def _on_pattern_toggle(self) -> None:
        self._pattern.enabled = self._pat_var.get()
        if not self._pattern.enabled:
            self._pat_status.configure(text="", text_color=T.TEXT_DIM)
        else:
            self._refresh_pat_status()

    def _on_pat_method(self, label: str) -> None:
        key = next((k for k in PatternModule.METHODS
                    if t("pat_m_" + k) == label), None)
        if key is None:
            return
        self._pattern.set_method(key)
        self._pat_help.configure(text=t("pat_help_" + key))
        self._refresh_pat_status()

    def _on_pat_sens(self, value: float) -> None:
        self._pattern.sensitivity = float(value)
        self._pat_sens_val.configure(text=str(int(value)))

    def _calibrate_pattern(self) -> None:
        """
        Aprende el ruido del material BUENO que esta pasando ahora.

        Se toman varios frames seguidos y no uno: un frame trae su propio
        ruido, y ese ruido se volveria el criterio de lo correcto.
        """
        self._pattern.clear_calibration()
        self._pat_cal_left = 15
        self._pat_var.set(True)
        self._pattern.enabled = True
        self._refresh_pat_status()

    def _refresh_pat_status(self) -> None:
        if self._pat_cal_left > 0:
            self._pat_status.configure(
                text=t("pat_calibrating", n=self._pat_cal_left),
                text_color=T.ACCENT2)
        elif self._pattern.calibrated:
            self._pat_status.configure(
                text=t("pat_calibrated", n=self._pattern.calibration_count),
                text_color=T.TEXT_DIM)
        else:
            self._pat_status.configure(text=t("pat_uncalibrated"),
                                        text_color=T.WARN)

    def _reset_counting(self) -> None:
        self._counting.reset()
        self._count_last = ""
        self._count_lbl.configure(text="0", text_color=T.ACCENT2)

    # ── geometry drawing on the canvas (line or zone rect) ───────────────────
    def _arm_meta_draw(self) -> None:
        self._meta_kind = self._counting.geometry_kind or "line"
        self._meta_drawing = True
        self._meta_btn.configure(fg_color=T.ACCENT2, text_color="#000")
        hint = t("prod_zone_hint") if self._meta_kind == "rect" else t("prod_line_hint")
        self._count_lbl.configure(text=hint, text_color=T.ACCENT2)

    def _meta_press(self, event) -> None:
        if self._meta_drawing:
            self._meta_start = (event.x, event.y)

    def _meta_drag(self, event) -> None:
        if not (self._meta_drawing and self._meta_start):
            return
        self._canvas.delete("meta_tmp")
        if self._meta_kind == "rect":
            self._canvas.create_rectangle(*self._meta_start, event.x, event.y,
                                          outline="#00bffe", width=2,
                                          tags="meta_tmp")
        else:
            self._canvas.create_line(*self._meta_start, event.x, event.y,
                                     fill="#ffd700", width=2, tags="meta_tmp")

    def _meta_release(self, event) -> None:
        if not (self._meta_drawing and self._meta_start):
            return
        self._canvas.delete("meta_tmp")
        start, self._meta_start = self._meta_start, None
        self._meta_drawing = False
        self._meta_btn.configure(fg_color=T.BG_INPUT, text_color=T.TEXT_PRI)
        if self._view is None:
            return
        fw, fh, nw, nh, ox, oy = self._view

        def to_frame(cx, cy):
            fx = (cx - ox) * fw / max(nw, 1)
            fy = (cy - oy) * fh / max(nh, 1)
            return (int(max(0, min(fw - 1, fx))), int(max(0, min(fh - 1, fy))))

        p1 = to_frame(*start)
        p2 = to_frame(event.x, event.y)
        if self._meta_kind == "rect":
            # un rectangulo de 3 px no es una zona: fue un clic suelto
            if abs(p1[0] - p2[0]) < 12 or abs(p1[1] - p2[1]) < 12:
                self._refresh_count_rows()
                return
            self._counting.zone = (min(p1[0], p2[0]), min(p1[1], p2[1]),
                                   max(p1[0], p2[0]), max(p1[1], p2[1]))
        else:
            if abs(p1[0] - p2[0]) < 5 and abs(p1[1] - p2[1]) < 5:
                self._refresh_count_rows()
                return   # accidental click, not a line
            self._counting.line = (*p1, *p2)
        self._counting.reset()
        self._count_var.set(True)
        self._counting.enabled = True
        self._count_last = ""
        self._count_lbl.configure(text="0", text_color=T.ACCENT2)

    # ── model loading ─────────────────────────────────────────────────────────
    def _try_autoload_model(self) -> None:
        """Load project model if available, otherwise download yolov8n in bg."""
        proj = self._app.get_project()
        if proj and proj.active_model:
            model_path = Path(proj.active_model)
            if model_path.exists():
                self._start_model_load(str(model_path))
                return
        # Fallback: auto-download YOLO nano (general-purpose preview)
        self._model_lbl.configure(
            text=t("prod_downloading_generic"), text_color=T.WARN)
        self._start_model_load("yolov8n.pt")

    def _start_model_load(self, model_path: str) -> None:
        """Kick off model load in a background thread."""
        if self._worker_loading:
            return
        self._worker_loading = True
        threading.Thread(
            target=self._load_model_bg, args=(model_path,), daemon=True
        ).start()

    def _load_model_bg(self, model_path: str) -> None:
        """Background thread: load (and download if needed) the model."""
        try:
            conf = self._conf_slider.get() / 100
            worker = YOLOWorker(model_path, conf=conf)
            worker.load()           # may trigger ultralytics hub download
            self.after(0, lambda: self._on_model_ready(worker, model_path))
        except Exception as e:
            from eyve.core.logger import log
            log.error(f"Model load failed: {e}")
            self.after(0, lambda: self._on_model_error(str(e)))

    def _on_model_ready(self, worker: YOLOWorker, model_path: str) -> None:
        """Main-thread callback: model is loaded and ready."""
        if not self.winfo_exists():
            return
        self._worker_loading = False
        self._worker = worker
        name = Path(model_path).name
        # A bare "yolov8n.pt" means the COCO fallback, not a project model —
        # it must be clearly labeled as generic (PRD §11.1: the generic stays,
        # what's forbidden is using it silently).
        proj = self._app.get_project()
        is_generic = (name == "yolov8n.pt"
                      and not (proj and proj.active_model
                               and Path(proj.active_model).exists()))
        if is_generic:
            self._model_lbl.configure(
                text=t("prod_generic_model"), text_color=T.WARN)
        else:
            self._model_lbl.configure(
                text=t("prod_model_loaded", name=name), text_color=T.ACCENT)
        # Warn if class mismatch with project (skip for the generic model —
        # COCO never matches user classes; the generic label already says it)
        if proj and not is_generic:
            model_cls = set(worker.class_names)
            proj_cls  = set(proj.class_names)
            if model_cls != proj_cls:
                self._model_lbl.configure(
                    text=t("prod_class_mismatch",
                           model=", ".join(sorted(model_cls)),
                           project=", ".join(sorted(proj_cls))),
                    text_color=T.WARN,
                )
        # If camera is already running in preview mode, start inference worker
        if self._running and not self._paused:
            worker.start()
            # Create session now that we have a model
            if self._session is None:
                self._create_session()

    def _on_model_error(self, err: str) -> None:
        if not self.winfo_exists():
            return
        self._worker_loading = False
        self._model_lbl.configure(
            text=t("prod_model_error", err=err[:80]), text_color=T.DANGER)

    def _load_model(self) -> None:
        """Manual model load via file dialog."""
        path = filedialog.askopenfilename(
            title=t("prod_load_model"),
            filetypes=[("YOLO model", "*.pt"), ("All", "*.*")]
        )
        if not path:
            return
        self._model_lbl.configure(text=t("prod_loading_model"), text_color=T.TEXT_DIM)
        self._start_model_load(path)

    # ── session helpers ───────────────────────────────────────────────────────
    def _create_session(self) -> None:
        if self._session is not None:
            return
        proj = self._app.get_project()
        self._session = ProductionSession(
            sessions_dir=(proj.paths.sessions if proj else Path("sessions")),
            screenshots_dir=(proj.paths.screenshots if proj else Path("screenshots")),
            model_path=self._worker._model_path if self._worker else "preview",
            save_nok=self._save_nok_var.get(),
            save_log=self._save_log_var.get(),
        )
        self._session_lbl.configure(text=t("prod_session", id=self._session.id))

    # ── auto-start camera preview ─────────────────────────────────────────────
    def _auto_start_preview(self) -> None:
        """
        Start the camera feed automatically when the screen opens.
        No model required — shows raw frames until inference model is ready.
        """
        if self._running or self._cam_starting:
            return   # already live or open in-flight
        if self._prod_src_var.get() == "video":
            # video mode: only auto-start when a file is already selected
            if self._prod_video_path and self._prod_video_path.exists():
                self._start()
            return
        # Wait until cameras are enumerated ("⟳ …" = still loading)
        cam_val = self._cam_idx.get()
        if cam_val.startswith("⟳"):
            self.after(300, self._auto_start_preview)
            return
        if cam_val == t("cam_none"):
            return
        self._start()

    # ── start / stop ──────────────────────────────────────────────────────────
    def _toggle(self) -> None:
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        if self._running or self._cam_starting:
            return   # already live or open in-flight — ignore duplicate call

        # Resolve source: camera index or video file path (remote validation)
        if self._prod_src_var.get() == "video":
            if not (self._prod_video_path and self._prod_video_path.exists()):
                self._prod_vid_lbl.configure(text=t("prod_video_missing"),
                                              text_color=T.WARN)
                return
            source: int | str = str(self._prod_video_path)
        else:
            source = label_to_index(self._cam_idx.get())

        self._cam_starting = True
        # Show opening animation on canvas
        self._warming = True
        self._spin_step = 0
        self._animate_warmup()

        def _open_bg():
            src = VideoSource(source)
            ok = src.start()
            self.after(0, lambda: self._on_source_ready(src, ok))

        threading.Thread(target=_open_bg, daemon=True).start()

    def _on_source_ready(self, src: VideoSource, ok: bool) -> None:
        """Main-thread: camera opened (or failed)."""
        # The screen may have been destroyed while the open thread ran
        # (navigation / set_project) — touching widgets then raises TclError.
        if not self.winfo_exists():
            src.stop()
            return
        self._warming = False
        self._cam_starting = False
        if not ok:
            # Don't show a modal error for auto-start failures — the camera
            # may simply need a moment (DSHOW async release).  Update the
            # button state so the user can retry manually, and schedule one
            # silent retry after 1.5 s.
            self._toggle_btn.configure(
                text=t("prod_start"), fg_color=T.ACCENT, text_color="#000")
            self._pause_btn.configure(state="disabled")
            self._model_lbl.configure(
                text=t("prod_cam_retry"),
                text_color=T.WARN)
            self.after(1500, self._auto_start_preview)
            return
        self._source = src
        self._running = True
        self._paused = False
        self._start_time = time.time()
        self._fps_times.clear()
        self._toggle_btn.configure(text=t("prod_stop"), fg_color=T.DANGER, text_color=T.TEXT_PRI)
        self._pause_btn.configure(state="normal", text=t("prod_pause"))
        # Start YOLO worker only if model is already loaded
        if self._worker and not self._worker_loading:
            self._worker.start()
            self._create_session()
        self._loop()

    def _stop(self) -> None:
        self._running = False
        self._paused = False
        self._warming = False
        self._cam_starting = False
        self._polarity.reset()
        self._mod_last_status = ""
        self._tracker.reset()
        self._counting.reset()
        self._pat_last = ""
        self._count_last = ""
        if self._worker:
            self._worker.stop()
        if self._source:
            self._source.stop()
            self._source = None
        if self._session:
            self._session.close()
            self._session = None
        self._toggle_btn.configure(text=t("prod_start"), fg_color=T.ACCENT, text_color="#000")
        self._pause_btn.configure(state="disabled", text=t("prod_pause"))
        self._status_lbl.configure(text="—", text_color=T.TEXT_DIM)
        self._status_frame.configure(fg_color=T.BG_INPUT)

    # ── canvas warmup animation ───────────────────────────────────────────────
    def _animate_warmup(self) -> None:
        if not getattr(self, "_warming", False) or self._running:
            return
        cw = self._canvas.winfo_width()  or 640
        ch = self._canvas.winfo_height() or 480
        self._canvas.delete("all")
        self._canvas.create_rectangle(0, 0, cw, ch, fill="#0a0a0a", outline="")
        spin = _SPINNER[self._spin_step % len(_SPINNER)]
        self._canvas.create_text(cw // 2, ch // 2 - 24, text=spin,
                                  fill=T.ACCENT, font=("Segoe UI", 36), anchor="center")
        self._canvas.create_text(cw // 2, ch // 2 + 20,
                                  text=t("prod_connecting_cam"), fill=T.TEXT_SEC,
                                  font=("Segoe UI", 13), anchor="center")
        self._spin_step += 1
        self.after(200, self._animate_warmup)

    # ── main loop ─────────────────────────────────────────────────────────────
    def _loop(self) -> None:
        if not self._running:
            return

        if self._paused:
            # draw pause overlay on the last shown frame
            self._canvas.create_rectangle(
                0, 0, self._canvas.winfo_width(), self._canvas.winfo_height(),
                fill="#000000", stipple="gray50", tags="pause_overlay"
            )
            self._canvas.create_text(
                self._canvas.winfo_width() // 2, self._canvas.winfo_height() // 2,
                text=t("prod_paused"), fill=T.WARN,
                font=("Segoe UI", 20, "bold"), tags="pause_overlay"
            )
            self.after(100, self._loop)
            return

        self._canvas.delete("pause_overlay")
        frame = self._source.read() if self._source else None

        if frame is not None:
            if self._worker and self._worker._running:
                # Inference mode: push frame, draw annotated result
                self._worker.push_frame(frame)
                result = self._worker.get_result()
                if result is not None:
                    proj = self._app.get_project()
                    ok_cls  = proj.ok_classes  if proj else []
                    nok_cls = proj.nok_classes if proj else []
                    conf_thresh = self._conf_slider.get() / 100
                    inspection = decide(result.detections, ok_cls, nok_cls, conf_thresh)

                    # ── persistent tracking (kills detection flicker) ─────
                    raw = [RawDetection(d.class_name, d.confidence,
                                        int(d.x1), int(d.y1),
                                        int(d.x2), int(d.y2))
                           for d in result.detections]
                    tracks = self._tracker.update(raw, frame)
                    annotated = self._annotate_tracks(frame, tracks, inspection)

                    # ── polarity: analyze each instance ONCE, verdict is ──
                    #    sticky for the life of the track (no flicker)
                    pol_wrong = 0
                    if self._polarity.enabled:
                        for tr in tracks:
                            if (tr.label == self._polarity.target_class
                                    and tr.needs_processing("polarity")):
                                res = self._polarity.analyze_roi(tr.get_roi(frame))
                                if res is not None:
                                    tr.set_result("polarity", res)
                                else:
                                    tr.skip_module("polarity")
                        last_lbl = ""
                        for tr in tracks:
                            res = tr.get_result("polarity")
                            if res is None:
                                continue
                            PolarityAnalyzer.draw_on_frame(annotated, tr.bbox, res)
                            if res.quadrant is not None:
                                last_lbl = (f"#{tr.id} {res.side} "
                                            f"{res.confidence:.0%}")
                            if res.is_correct is False:
                                pol_wrong += 1
                        if pol_wrong and inspection.status != InspectionStatus.NOT_OK:
                            inspection = InspectionResult(
                                status=InspectionStatus.NOT_OK,
                                triggered_by=f"polaridad ×{pol_wrong}",
                                confidence=1.0,
                                detections=result.detections,
                            )
                        st = f"{self._polarity.learning_status}   {last_lbl}".strip()
                        if st != self._mod_last_status:
                            self._mod_last_status = st
                            self._pol_status.configure(
                                text=st,
                                text_color=(T.COLOR_NOK if pol_wrong
                                            else T.TEXT_DIM))

                    # ── Patron: lo que el detector no sabe nombrar ───────
                    if self._pattern.enabled:
                        if self._pat_cal_left > 0:
                            self._pattern.calibrate(frame)
                            self._pat_cal_left -= 1
                            self._refresh_pat_status()
                        else:
                            # analyze() ya corrio arriba, antes de asociar, y
                            # sus regiones entraron al tracker: aqui solo se
                            # recoge el veredicto. Las cajas las dibuja
                            # _annotate_tracks como cualquier otra instancia.
                            pv = self._pattern.process(frame, result.detections,
                                                       annotated)
                            if not pv.ok and inspection.status != InspectionStatus.NOT_OK:
                                inspection = InspectionResult(
                                    status=InspectionStatus.NOT_OK,
                                    triggered_by=pv.triggered_by or "patron",
                                    confidence=1.0,
                                    detections=result.detections,
                                )
                            ps = self._pattern.summary()
                            if ps != self._pat_last:
                                self._pat_last = ps
                                self._pat_status.configure(
                                    text=ps,
                                    text_color=(T.COLOR_NOK if not pv.ok
                                                else T.TEXT_DIM))

                    # ── counting (5 metodos sobre el mismo tracker) ───────
                    if self._counting.enabled:
                        fh_, fw_ = frame.shape[:2]
                        cv_ = self._counting.update_tracks(
                            tracks,
                            expired=self._tracker.last_expired,
                            frame_wh=(fw_, fh_))
                        self._counting.draw(annotated)
                        # Un conteo fuera del rango esperado es un defecto:
                        # veta el frame igual que la polaridad.
                        if not cv_.ok and inspection.status != InspectionStatus.NOT_OK:
                            inspection = InspectionResult(
                                status=InspectionStatus.NOT_OK,
                                triggered_by=cv_.triggered_by or "conteo",
                                confidence=1.0,
                                detections=result.detections,
                            )
                        cs = self._counting.summary()
                        if cs != self._count_last:
                            self._count_last = cs
                            self._count_lbl.configure(
                                text=cs,
                                text_color=(T.COLOR_NOK if not cv_.ok
                                            else T.ACCENT2))

                    # ── polarity preview (picture-in-picture, top-right) ──
                    if self._polarity.enabled and self._polarity.last_debug is not None:
                        dbg = self._polarity.last_debug
                        dh, dw = dbg.shape[:2]
                        sc = 200.0 / max(dw, dh)
                        small = cv2.resize(dbg, (max(1, int(dw * sc)),
                                                 max(1, int(dh * sc))))
                        sh, sw = small.shape[:2]
                        fh2, fw2 = annotated.shape[:2]
                        if sh + 16 < fh2 and sw + 16 < fw2:
                            annotated[8:8 + sh, fw2 - sw - 8:fw2 - 8] = small

                    self._last_result = inspection
                    if self._session:
                        self._session.record(
                            inspection,
                            frame if inspection.status == InspectionStatus.NOT_OK else None,
                        )
                        self._update_counts()
                    self._update_canvas(annotated)
                    self._update_status(inspection)
                else:
                    # Waiting for first inference result — show raw frame
                    self._update_canvas(frame)
            else:
                # Preview mode — no model yet, show raw feed
                self._update_canvas(frame)
            self._update_fps()

        self.after(_loop_delay(), self._loop)

    # ── drawing ───────────────────────────────────────────────────────────────
    def _class_color(self, class_name: str) -> tuple:
        proj = self._app.get_project()
        if proj:
            cls_def = next((c for c in proj.classes if c.name == class_name), None)
            if cls_def:
                h = cls_def.color
                return (int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16))
        return (0, 230, 118)

    def _annotate_tracks(self, frame, tracks, inspection) -> np.ndarray:
        """
        Draw CONFIRMED tracked instances: smoothed bbox + persistent #ID.
        Tracks replace raw detections here — the smoothing (alpha 0.65) and
        the 2-frame confirmation are what kill the box/count flicker.
        """
        out = frame.copy()
        for tr in tracks:
            if not tr.confirmed:
                continue
            color = self._class_color(tr.label)
            x1, y1, x2, y2 = tr.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"#{tr.id} {tr.label} {tr.confidence:.0%}"
            cv2.rectangle(out, (x1, y1 - 18), (x1 + len(label) * 8, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return out

    def _update_canvas(self, frame) -> None:
        try:
            cw = self._canvas.winfo_width() or 640
            ch = self._canvas.winfo_height() or 480
            # Resize in OpenCV (C++ SIMD) before PIL to avoid the full-res
            # array going through Python — critical for 1080p+ camera feeds.
            fh, fw = frame.shape[:2]
            scale = min(cw / fw, ch / fh)
            if scale < 0.99:
                nw = max(1, int(fw * scale))
                nh = max(1, int(fh * scale))
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            else:
                nw, nh = fw, fh
            # canvas↔frame mapping used by the finish-line drawing
            self._view = (fw, fh, nw, nh, (cw - nw) // 2, (ch - nh) // 2)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._photo = photo
            self._canvas.create_image(cw // 2, ch // 2, image=photo, anchor="center")
        except Exception:
            pass

    def _update_status(self, inspection) -> None:
        status = inspection.status
        color = _STATUS_COLORS.get(status, T.TEXT_DIM)
        label = t(_STATUS_KEYS.get(status, "prod_status_error"))
        self._status_lbl.configure(text=label, text_color=color)
        self._status_frame.configure(fg_color=_color_tint(color))
        if inspection.triggered_by:
            self._det_lbl.configure(
                text=f"{inspection.triggered_by}  {inspection.confidence:.0%}",
                text_color=color
            )
        else:
            self._det_lbl.configure(text="")

    def _update_fps(self) -> None:
        now = time.time()
        self._fps_times.append(now)
        cutoff = now - 2.0
        self._fps_times = [t_ for t_ in self._fps_times if t_ >= cutoff]
        fps = len(self._fps_times) / 2.0 if len(self._fps_times) > 1 else 0
        self._fps_lbl.configure(text=t("prod_fps", fps=f"{fps:.1f}"))
        elapsed = int(now - self._start_time)
        m, s = divmod(elapsed, 60)
        self._elapsed_lbl.configure(
            text=t("prod_elapsed", t=f"{m:02d}:{s:02d}"))

    def _space_action(self) -> None:
        """SPACE key: pause/resume when running, start when stopped."""
        if self._running:
            self._toggle_pause()
        else:
            self._start()

    def _toggle_pause(self) -> None:
        if not self._running:
            return
        self._paused = not self._paused
        if self._paused:
            self._pause_btn.configure(text=t("prod_resume"), fg_color=T.WARN)
        else:
            self._canvas.delete("pause_overlay")
            self._pause_btn.configure(text=t("prod_pause"), fg_color=T.BG_INPUT)

    def _update_counts(self) -> None:
        if self._session:
            s = self._session.stats
            self._ok_lbl.configure(text=t("prod_count_ok", n=s["ok"]))
            self._nok_lbl.configure(text=t("prod_count_nok", n=s["nok"]))

    def _reset_counts(self) -> None:
        if self._session:
            self._session._count_ok = 0
            self._session._count_nok = 0
            self._session._count_review = 0
        self._ok_lbl.configure(text=t("prod_count_ok", n=0))
        self._nok_lbl.configure(text=t("prod_count_nok", n=0))

    def _on_conf_change(self, val) -> None:
        v = int(float(val))
        self._conf_val.configure(text=f"{v}%")
        if self._worker:
            self._worker.set_conf(v / 100)

    # ── lifecycle (navigation) ────────────────────────────────────────────────
    def on_hide(self) -> None:
        """Stop camera and inference when the user navigates away."""
        self._stop()

    def on_show(self) -> None:
        """Auto-start camera preview when the user navigates back."""
        self.after(100, self._auto_start_preview)

    def on_close(self) -> None:
        top = self.winfo_toplevel()
        for bid in getattr(self, "_hotkey_ids", []):
            try:
                top.unbind("<space>", bid)
            except Exception:
                pass
        self._hotkey_ids = []
        self._stop()
