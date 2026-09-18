"""
Tagging screen — manual bounding box annotation.

Sources
-------
Images  (default): browse through captured images in raw/images/**
Camera  (live):    capture frames from camera, freeze and tag each one
Video   (file):    play a video file, capture frames, freeze and tag

Keyboard shortcuts:
  1-9      select class
  N        next image
  P        previous image
  S        save current labels
  Delete   remove last / selected box
  Escape   cancel current draw / freeze camera
  G        grab frame from camera/video
"""
from __future__ import annotations
import cv2
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from tkinter import filedialog
from typing import TYPE_CHECKING, Optional

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk

from eyve.ui import theme as T
from eyve.i18n import t
from eyve.ui.components.dialogs import show_error
from eyve.camera.camera_enum import get_camera_labels, label_to_index
from eyve.camera.release import release_async
from eyve.ui.hotkeys import guard_hotkey

if TYPE_CHECKING:
    from eyve.ui.app import EyveApp


def _open_camera(idx: int) -> cv2.VideoCapture:
    """Open camera with DSHOW (fast on Windows) → MSMF → ANY fallback."""
    import platform
    import time as _t
    backends = (
        [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        if platform.system() == "Windows" else [cv2.CAP_ANY]
    )
    for api in backends:
        # DSHOW releases asynchronously: retry before falling back to MSMF
        # (which takes 5+ s and renegotiates on every cap.set call).
        max_tries = 5 if api == cv2.CAP_DSHOW else 1
        cap = None
        for attempt in range(max_tries):
            cap = cv2.VideoCapture(idx, api)
            if cap.isOpened():
                break
            cap.release()
            cap = None
            if attempt < max_tries - 1:
                _t.sleep(0.35)
        if cap and cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # MJPEG once at open so USB 2.0 can carry 1080p at 30 fps;
            # then request HD — camera silently caps at its physical max.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            return cap
        if cap:
            cap.release()
    return cv2.VideoCapture(idx)


class _BBox:
    """One annotation bounding box (pixel coords on display canvas)."""
    def __init__(self, x1: int, y1: int, x2: int, y2: int,
                 class_idx: int, class_name: str, color: str):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.class_idx = class_idx
        self.class_name = class_name
        self.color = color
        self.canvas_id: Optional[int] = None
        self.label_id: Optional[int] = None


class TaggingScreen(ctk.CTkFrame):
    def __init__(self, master, app: "EyveApp", **kwargs):
        super().__init__(master, fg_color=T.BG_DARK, **kwargs)
        self._app = app

        # ── image list + annotation state ──────────────────────────────────
        self._images: list[Path] = []
        self._idx: int = 0
        self._boxes: list[_BBox] = []
        self._draw_start: Optional[tuple[int, int]] = None
        self._active_rect: Optional[int] = None
        self._selected_class: int = 0
        self._display_size: tuple[int, int] = (640, 480)
        self._orig_size: tuple[int, int] = (640, 480)
        self._photo = None
        self._offset: tuple[int, int] = (0, 0)

        # ── live camera / video state ──────────────────────────────────────
        self._cap: Optional[cv2.VideoCapture] = None
        self._live_running = False
        self._live_frame: Optional[object] = None   # latest numpy frame
        self._frame_lock = threading.Lock()
        self._live_thread: Optional[threading.Thread] = None
        self._frozen_frame: Optional[object] = None  # frame awaiting annotation
        self._video_path: Optional[Path] = None
        self._warming = False
        self._anim_step = 0
        # ── video playback control (pause / seek) ──────────────────────────
        self._video_paused = False
        self._seek_target: Optional[int] = None   # frame index requested by slider
        self._video_pos = 0                       # current frame index (grab thread writes)
        self._video_total = 0
        self._video_fps = 30.0
        self._frame_dirty = False                 # paused: repaint only after a seek

        self._build()
        self._load_images()

    # ── layout ────────────────────────────────────────────────────────────────
    def _build(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # header
        hdr = ctk.CTkFrame(self, fg_color=T.BG_CARD, corner_radius=0)
        hdr.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(hdr, text=t("tag_title"),
                     font=T.bold(T.FONT_XL), text_color=T.TEXT_PRI).pack(
            side="left", padx=24, pady=12)
        self._progress_lbl = ctk.CTkLabel(hdr, text="",
                                           font=T.font(T.FONT_SM), text_color=T.TEXT_DIM)
        self._progress_lbl.pack(side="left", padx=8)
        hint = ctk.CTkLabel(hdr, text=t("tag_shortcuts"),
                            font=T.font(T.FONT_XS), text_color=T.TEXT_DIM)
        hint.pack(side="right", padx=16)

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
        canvas_frame.grid_rowconfigure(1, weight=0)
        canvas_frame.grid_columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(canvas_frame, bg="#0a0a0a", cursor="crosshair",
                                  highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nsew")

        # ── video playback bar (VLC-style, under the player) ───────────────
        # Full-width: [⏸/▶]  [──────slider──────]  [00:05 / 01:23]
        # Shown only while a video source is live; pause freezes the frame
        # for tagging, the slider seeks anywhere in the file.
        self._playbar = ctk.CTkFrame(canvas_frame, fg_color="#141414",
                                      corner_radius=0, height=44)
        self._vid_pause_btn = ctk.CTkButton(
            self._playbar, text="⏸", width=44, height=30,
            fg_color=T.BG_INPUT, font=T.bold(T.FONT_MD),
            command=self._toggle_video_pause,
        )
        self._vid_pause_btn.pack(side="left", padx=(10, 8), pady=7)
        self._vid_time_lbl = ctk.CTkLabel(
            self._playbar, text="00:00 / 00:00",
            font=T.font(T.FONT_SM), text_color=T.TEXT_SEC, width=110,
        )
        self._vid_time_lbl.pack(side="right", padx=(8, 12))
        self._vid_slider = ctk.CTkSlider(
            self._playbar, from_=0, to=1,
            command=self._on_video_seek,
        )
        self._vid_slider.set(0)
        self._vid_slider.pack(side="left", fill="x", expand=True, pady=7)
        self._playbar.grid(row=1, column=0, sticky="ew")
        self._playbar.grid_remove()   # hidden until a video starts
        self._canvas.bind("<ButtonPress-1>",   self._on_press)
        self._canvas.bind("<B1-Motion>",        self._on_drag)
        self._canvas.bind("<ButtonRelease-1>",  self._on_release)
        self._canvas.bind("<Configure>",        self._on_resize)

        # right panel
        right = ctk.CTkFrame(body, fg_color=T.BG_CARD, corner_radius=10, width=230)
        right.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        right.grid_propagate(False)
        self._build_right(right)

        # bottom nav bar
        nav = ctk.CTkFrame(self, fg_color=T.BG_CARD, corner_radius=0)
        nav.grid(row=2, column=0, sticky="ew")
        self._build_nav(nav)

        # keyboard bindings
        top = self.winfo_toplevel()
        self._hotkey_ids: list[tuple[str, str]] = []

        def _hk(seq: str, fn) -> None:
            bid = top.bind(seq, guard_hotkey(self, fn), add=True)
            self._hotkey_ids.append((seq, bid))

        _hk("<n>",      lambda e: self._next())
        _hk("<N>",      lambda e: self._next())
        _hk("<p>",      lambda e: self._prev())
        _hk("<P>",      lambda e: self._prev())
        _hk("<s>",      lambda e: self._save())
        _hk("<S>",      lambda e: self._save())
        _hk("<Delete>", lambda e: self._delete_last())
        _hk("<Escape>", lambda e: self._escape_action())
        _hk("<g>",      lambda e: self._grab_frame())
        _hk("<G>",      lambda e: self._grab_frame())
        for i in range(1, 10):
            _hk(str(i), lambda e, idx=i-1: self._select_class(idx))

    def _build_right(self, parent) -> None:
        # ── source selector ────────────────────────────────────────────────
        src_lbl = ctk.CTkLabel(parent, text=t("tag_source"),
                                font=T.bold(T.FONT_SM), text_color=T.TEXT_PRI)
        src_lbl.pack(pady=(14, 4), padx=12)

        self._src_var = ctk.StringVar(value="images")

        row1 = ctk.CTkFrame(parent, fg_color="transparent")
        row1.pack(fill="x", padx=12, pady=2)
        ctk.CTkRadioButton(row1, text=t("tag_source_images"),
                           variable=self._src_var, value="images",
                           font=T.font(T.FONT_XS),
                           command=self._on_source_change).pack(side="left")
        ctk.CTkRadioButton(row1, text=t("cap_source_camera"),
                           variable=self._src_var, value="camera",
                           font=T.font(T.FONT_XS),
                           command=self._on_source_change).pack(side="left", padx=8)

        row2 = ctk.CTkFrame(parent, fg_color="transparent")
        row2.pack(fill="x", padx=12, pady=(0, 6))
        ctk.CTkRadioButton(row2, text=t("cap_source_video"),
                           variable=self._src_var, value="video",
                           font=T.font(T.FONT_XS),
                           command=self._on_source_change).pack(side="left")

        ctk.CTkFrame(parent, height=1, fg_color=T.BORDER).pack(fill="x", padx=12, pady=4)

        # ── camera controls (shown when source = camera or video) ──────────
        self._cam_panel = ctk.CTkFrame(parent, fg_color="transparent")
        self._cam_panel.pack(fill="x", padx=12, pady=(0, 4))

        ctk.CTkLabel(self._cam_panel, text=t("cap_camera_id"),
                     font=T.font(T.FONT_XS), text_color=T.TEXT_SEC).pack(anchor="w")
        self._cam_idx = ctk.CTkOptionMenu(
            self._cam_panel, values=[t("cam_loading")],
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD,
            font=T.font(T.FONT_XS), state="disabled",
        )
        self._cam_idx.pack(fill="x", pady=(0, 4))
        threading.Thread(target=self._load_cameras_bg, daemon=True).start()

        self._video_btn = ctk.CTkButton(
            self._cam_panel, text=t("prod_select_video"),
            fg_color=T.BG_INPUT, height=28, font=T.font(T.FONT_XS),
            command=self._browse_video,
        )
        self._video_path_lbl = ctk.CTkLabel(
            self._cam_panel, text="", font=T.font(T.FONT_XS),
            text_color=T.TEXT_DIM, wraplength=160, anchor="w",
        )

        # (video playback controls live in self._playbar under the canvas —
        #  built in _build(); nothing to add here)

        self._cam_btns = ctk.CTkFrame(self._cam_panel, fg_color="transparent")
        cam_btns = self._cam_btns
        cam_btns.pack(fill="x", pady=2)
        self._cam_start_btn = ctk.CTkButton(
            cam_btns, text=t("tag_start"),
            fg_color=T.ACCENT, text_color="#000", height=30,
            font=T.bold(T.FONT_XS), command=self._start_live,
        )
        self._cam_start_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._cam_stop_btn = ctk.CTkButton(
            cam_btns, text=t("tag_stop"),
            fg_color=T.BG_INPUT, height=30,
            font=T.font(T.FONT_XS), state="disabled",
            command=self._stop_live,
        )
        self._cam_stop_btn.pack(side="left", fill="x", expand=True)

        self._grab_btn = ctk.CTkButton(
            self._cam_panel,
            text=t("tag_grab_btn"),
            fg_color=T.ACCENT2, text_color="#000",
            height=36, font=T.bold(T.FONT_SM),
            state="disabled", command=self._grab_frame,
        )
        self._grab_btn.pack(fill="x", pady=(4, 0))

        self._cam_status = ctk.CTkLabel(
            self._cam_panel, text="", font=T.font(T.FONT_XS),
            text_color=T.TEXT_DIM, wraplength=180,
        )
        self._cam_status.pack(pady=2)

        self._cam_panel.pack_forget()   # hidden by default (source = images)

        # Anchor separator: _on_source_change re-packs _cam_panel BEFORE this
        # widget so the panel returns to its original slot.  A plain pack()
        # appends it at the BOTTOM of the right panel where the playback
        # controls get clipped off-screen.
        self._sep_after_cam = ctk.CTkFrame(parent, height=1, fg_color=T.BORDER)
        self._sep_after_cam.pack(fill="x", padx=12, pady=4)

        # ── class selector ─────────────────────────────────────────────────
        ctk.CTkLabel(parent, text=t("tag_class"),
                     font=T.bold(T.FONT_SM), text_color=T.TEXT_PRI).pack(
            pady=(8, 4), padx=12)

        self._class_frame = ctk.CTkScrollableFrame(parent, fg_color="transparent",
                                                    label_text="")
        self._class_frame.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        self._msg = ctk.CTkLabel(parent, text="",
                                  font=T.font(T.FONT_SM), text_color=T.TEXT_DIM,
                                  wraplength=180)
        self._msg.pack(padx=12, pady=2)

        ctk.CTkLabel(parent, text=t("tag_class_stats"),
                     font=T.font(T.FONT_XS), text_color=T.TEXT_DIM).pack(
            padx=12, anchor="w")
        self._stats_lbl = ctk.CTkLabel(parent, text="",
                                        font=T.font(T.FONT_XS), text_color=T.TEXT_DIM,
                                        anchor="w", justify="left")
        self._stats_lbl.pack(padx=12, pady=(2, 12), anchor="w")

        self._class_btns: list[ctk.CTkButton] = []
        self._refresh_class_buttons()

    def _build_nav(self, parent) -> None:
        ctk.CTkButton(parent, text=t("tag_prev"),
                      fg_color=T.BG_INPUT, width=110, height=36,
                      command=self._prev).pack(side="left", padx=8, pady=6)

        self._img_lbl = ctk.CTkLabel(parent, text="",
                                      font=T.font(T.FONT_SM), text_color=T.TEXT_SEC)
        self._img_lbl.pack(side="left", padx=12)

        ctk.CTkButton(parent, text=t("tag_next"),
                      fg_color=T.BG_INPUT, width=110, height=36,
                      command=self._next).pack(side="left", padx=4, pady=6)

        ctk.CTkButton(parent, text=t("tag_save"),
                      fg_color=T.ACCENT, text_color="#000", width=110, height=36,
                      font=T.bold(T.FONT_SM),
                      command=self._save).pack(side="left", padx=16, pady=6)

        ctk.CTkButton(parent, text=t("tag_delete_box"),
                      fg_color=T.BG_INPUT, width=130, height=36,
                      command=self._delete_last).pack(side="left", padx=4, pady=6)

        ctk.CTkButton(parent, text=t("tag_clear"),
                      fg_color=T.BG_INPUT, width=120, height=36,
                      command=self._clear_boxes).pack(side="left", padx=4, pady=6)

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

    # ── source selector ───────────────────────────────────────────────────────
    def _on_source_change(self) -> None:
        src = self._src_var.get()
        if src == "images":
            self._stop_live()
            self._cam_panel.pack_forget()
            self._frozen_frame = None
            self._load_images()
        elif src == "camera":
            self._cam_panel.pack(fill="x", padx=12, pady=(0, 4),
                                 before=self._sep_after_cam)
            self._video_btn.pack_forget()
            self._video_path_lbl.pack_forget()
            self._playbar.grid_remove()
            # Don't auto-start; user clicks ▶ Iniciar
        else:  # video
            self._cam_panel.pack(fill="x", padx=12, pady=(0, 4),
                                 before=self._sep_after_cam)
            # Video widgets go ABOVE the Iniciar/Parar row (packing them
            # after the status label pushed them below the window edge)
            self._video_btn.pack(fill="x", pady=(0, 2), before=self._cam_btns)
            self._video_path_lbl.pack(anchor="w", before=self._cam_btns)

    def _browse_video(self) -> None:
        p = filedialog.askopenfilename(
            title="Seleccionar Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")]
        )
        if not p:
            return
        self._video_path = Path(p)
        self._video_path_lbl.configure(text=p)
        # Selecting a new file switches playback to it immediately — before,
        # a video already playing kept running and the new selection appeared
        # to be ignored (the user had to know to press Parar + Iniciar).
        if self._src_var.get() == "video":
            if self._live_running:
                self._stop_live()
            self._start_live()

    # ── live camera / video ───────────────────────────────────────────────────
    def _start_live(self) -> None:
        src = self._src_var.get()
        if src == "camera":
            label = self._cam_idx.get()
            # latent bug fixed: the old guard checked for "Loading" while the
            # placeholder said "Cargando" — it never matched.
            if label.startswith("⟳") or label == t("cam_none"):
                return
            idx = label_to_index(label)
            if idx < 0:
                return
            self._cam_status.configure(text="⟳ " + t("cap_opening_cam"), text_color=T.TEXT_SEC)
            self._warming = True
            self._anim_step = 0
            self._animate_warmup()
            def _bg():
                cap = _open_camera(idx)
                self.after(0, lambda: self._on_cap_ready(cap))
            threading.Thread(target=_bg, daemon=True).start()
        else:  # video
            if not self._video_path or not self._video_path.exists():
                self._cam_status.configure(text=t("tag_select_video_first"), text_color=T.WARN)
                return
            cap = cv2.VideoCapture(str(self._video_path))
            if not cap.isOpened():
                self._cam_status.configure(text=t("tag_video_open_error"), text_color=T.DANGER)
                return
            self._on_cap_ready(cap)

    def _on_cap_ready(self, cap: cv2.VideoCapture) -> None:
        self._warming = False
        if not cap.isOpened():
            self._cam_status.configure(text=t("cap_no_camera"), text_color=T.DANGER)
            return
        self._cap = cap
        self._live_running = True
        self._frozen_frame = None
        if self._src_var.get() == "video":
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._video_total = max(total, 1)
            self._video_fps = fps if fps > 0 else 30.0
            self._video_paused = False
            self._seek_target = None
            self._video_pos = 0
            self._vid_slider.configure(to=max(self._video_total - 1, 1))
            self._vid_slider.set(0)
            self._vid_pause_btn.configure(text="⏸")
            self._playbar.grid()   # show VLC-style bar under the player
        self._cam_start_btn.configure(state="disabled")
        self._cam_stop_btn.configure(state="normal")
        self._grab_btn.configure(state="normal")
        self._cam_status.configure(text=t("tag_live"), text_color=T.ACCENT)
        self._live_thread = threading.Thread(target=self._live_grab_loop, daemon=True)
        self._live_thread.start()
        self._live_display_loop()

    def _stop_live(self) -> None:
        self._live_running = False
        self._warming = False
        self._video_paused = False
        self._seek_target = None
        self._playbar.grid_remove()
        # The grab thread owns the capture and releases it on exit.  Never
        # release here on the Tk thread: DSHOW's CoUninitialize() would kill
        # the COM apartment behind every file dialog (eyve/camera/release.py).
        cap, self._cap = self._cap, None
        if cap is not None and not (self._live_thread and self._live_thread.is_alive()):
            release_async(cap)
        self._frozen_frame = None
        self._cam_start_btn.configure(state="normal")
        self._cam_stop_btn.configure(state="disabled")
        self._grab_btn.configure(state="disabled")
        self._cam_status.configure(text="", text_color=T.TEXT_DIM)
        # Show static image if one is loaded
        if self._images:
            self._show_image(self._idx)

    def _live_grab_loop(self) -> None:
        """
        Background: continuously read frames.

        Video extras (this thread is the ONLY one touching self._cap):
          - seek: consumes self._seek_target set by the slider and reads one
            frame there, even while paused (scrub preview);
          - pause: stops reading, keeping the last frame on screen so the
            user can draw boxes on it.
        """
        is_video = self._src_var.get() == "video"
        # This thread OWNS the capture: local ref, released on exit (COM-safe).
        cap = self._cap
        if cap is None:
            return
        try:
            while self._live_running and cap.isOpened():
                if is_video:
                    seek = self._seek_target
                    if seek is not None:
                        self._seek_target = None
                        cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
                        ret, frame = cap.read()
                        if ret:
                            with self._frame_lock:
                                self._live_frame = frame
                            self._video_pos = seek
                            self._frame_dirty = True   # repaint even while paused
                        continue
                    if self._video_paused:
                        time.sleep(0.05)
                        continue
                ret, frame = cap.read()
                if not ret:
                    if is_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                with self._frame_lock:
                    self._live_frame = frame
                if is_video:
                    self._video_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    time.sleep(1 / 30)
        finally:
            try:
                cap.release()   # on the grab thread, never on Tk's
            except Exception:
                pass

    def _live_display_loop(self) -> None:
        """Main thread: paint latest live frame (unless frozen for annotation)."""
        if not self._live_running:
            return
        if self._frozen_frame is None:
            with self._frame_lock:
                frame = self._live_frame
            # While paused we must NOT repaint every tick — _paint_frame does
            # delete("all") and would erase the boxes being drawn.  We repaint
            # only when the grab thread flags a fresh frame (seek preview).
            if frame is not None and (not self._video_paused or self._frame_dirty):
                self._frame_dirty = False
                self._paint_frame(frame)
        # keep slider/time in sync while playing (diff-guard avoids fighting
        # the user's drag — we only push when clearly out of sync)
        if self._src_var.get() == "video" and self._video_total > 1:
            pos = self._video_pos
            if self._seek_target is None and abs(self._vid_slider.get() - pos) > 3:
                self._vid_slider.set(pos)
            cur_s = pos / self._video_fps
            tot_s = self._video_total / self._video_fps
            self._vid_time_lbl.configure(
                text=f"{int(cur_s//60):02d}:{int(cur_s%60):02d} / "
                     f"{int(tot_s//60):02d}:{int(tot_s%60):02d}")
        self.after(16, self._live_display_loop)

    def _paint_frame(self, frame) -> None:
        """Draw a numpy BGR frame on the canvas (no boxes)."""
        try:
            cw = self._canvas.winfo_width()  or 640
            ch = self._canvas.winfo_height() or 480
            fh, fw = frame.shape[:2]
            scale = min(cw / fw, ch / fh)
            if scale < 0.99:
                nw = max(1, int(fw * scale))
                nh = max(1, int(fh * scale))
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._photo = photo
            # Record geometry so boxes drawn over a paused/live frame convert
            # correctly to YOLO coords in _save_labels (same fields that
            # _show_image sets for static images).
            dh, dw = frame.shape[:2]
            self._orig_size = (fw, fh)
            self._display_size = (dw, dh)
            self._offset = ((cw - dw) // 2, (ch - dh) // 2)
            self._canvas.delete("all")
            self._canvas.create_image(cw // 2, ch // 2, image=photo, anchor="center")
        except Exception:
            pass

    # ── canvas warmup animation ───────────────────────────────────────────────
    _SPINNER = ["◐", "◓", "◑", "◒"]

    def _animate_warmup(self) -> None:
        if not self._warming or self._live_running:
            return
        cw = self._canvas.winfo_width()  or 640
        ch = self._canvas.winfo_height() or 480
        self._canvas.delete("all")
        self._canvas.create_rectangle(0, 0, cw, ch, fill="#0a0a0a", outline="")
        spin = self._SPINNER[self._anim_step % len(self._SPINNER)]
        self._canvas.create_text(cw // 2, ch // 2 - 24, text=spin,
                                  fill=T.ACCENT, font=("Segoe UI", 36), anchor="center")
        self._canvas.create_text(cw // 2, ch // 2 + 20,
                                  text=t("cap_opening_cam"), fill=T.TEXT_SEC,
                                  font=("Segoe UI", 13), anchor="center")
        self._anim_step += 1
        self.after(200, self._animate_warmup)

    # ── video playback control ────────────────────────────────────────────────
    def _toggle_video_pause(self) -> None:
        """⏸/▶: pausing freezes the current frame so it can be annotated."""
        if not self._live_running or self._src_var.get() != "video":
            return
        self._video_paused = not self._video_paused
        self._vid_pause_btn.configure(text="▶" if self._video_paused else "⏸")
        if self._video_paused:
            self._cam_status.configure(text=t("tag_vid_paused"), text_color=T.WARN)
        else:
            self._clear_boxes()
            self._cam_status.configure(text=t("tag_live"), text_color=T.ACCENT)

    def _on_video_seek(self, value) -> None:
        """Slider: request a seek (grab thread owns the VideoCapture)."""
        if not self._live_running or self._src_var.get() != "video":
            return
        # New frame → previous annotations no longer apply
        if self._boxes:
            self._clear_boxes()
        self._cancel_draw()
        self._seek_target = int(float(value))

    # ── grab frame ────────────────────────────────────────────────────────────
    def _grab_frame(self) -> None:
        """
        Freeze the current live frame for annotation.  Hotkey: G

        NOTHING is written to disk here — the jpg + labels are saved together
        when the user presses Guardar.  Grabbing repeatedly just refreshes
        the freeze to the latest frame, so skimming through a video with G
        never leaves orphan untagged images in the project (the bug this
        replaces: every grab wrote a jpg immediately, tagged or not).
        Escape discards the freeze.
        """
        if not self._live_running:
            return
        with self._frame_lock:
            frame = self._live_frame
        if frame is None:
            return
        self._frozen_frame = frame.copy()
        self._clear_boxes()
        self._paint_frame(self._frozen_frame)   # also records geometry for save
        self._cam_status.configure(text=t("tag_frozen_hint"), text_color=T.ACCENT)

    def _escape_action(self) -> None:
        if self._frozen_frame is not None:
            # Un-freeze: go back to live feed
            self._frozen_frame = None
            self._cam_status.configure(text=t("tag_live"), text_color=T.ACCENT)
        else:
            self._cancel_draw()

    # ── load images ───────────────────────────────────────────────────────────
    def _load_images(self) -> None:
        proj = self._app.get_project()
        if not proj:
            return
        raw = proj.paths.raw_images
        try:
            self._images = sorted(raw.rglob("*.jpg")) if raw.exists() else []
        except Exception:
            self._images = []
        self._update_progress()
        if self._images:
            self._show_image(0)
        else:
            cw = self._canvas.winfo_width()  or 640
            ch = self._canvas.winfo_height() or 480
            self._canvas.delete("all")
            self._canvas.create_text(
                cw // 2, ch // 2,
                text=t("tag_no_images"),
                fill=T.TEXT_DIM, font=("Segoe UI", 14),
            )

    def _refresh_class_buttons(self) -> None:
        for b in self._class_btns:
            b.destroy()
        self._class_btns.clear()

        proj = self._app.get_project()
        classes = proj.classes if proj else []
        for i, cls in enumerate(classes):
            color = (T.COLOR_OK  if cls.kind == "ok"  else
                     T.COLOR_NOK if cls.kind == "nok" else T.TEXT_DIM)
            btn = ctk.CTkButton(
                self._class_frame,
                text=f"  {i+1}. {cls.name}",
                anchor="w",
                fg_color=(color if i == self._selected_class else T.BG_INPUT),
                text_color=("#000" if (i == self._selected_class and cls.kind != "ignore")
                            else T.TEXT_PRI),
                font=T.font(T.FONT_SM),
                height=32,
                command=lambda idx=i: self._select_class(idx),
            )
            btn.pack(fill="x", padx=4, pady=2)
            self._class_btns.append(btn)

    def _select_class(self, idx: int) -> None:
        proj = self._app.get_project()
        if not proj or idx >= len(proj.classes):
            return
        self._selected_class = idx
        self._refresh_class_buttons()

    # ── image display ─────────────────────────────────────────────────────────
    def _show_image(self, idx: int) -> None:
        if not self._images:
            return
        self._idx = max(0, min(idx, len(self._images) - 1))
        self._boxes.clear()
        self._canvas.delete("all")
        path = self._images[self._idx]
        try:
            img = Image.open(path)
        except Exception:
            return
        self._orig_size = img.size
        cw = self._canvas.winfo_width()  or 640
        ch = self._canvas.winfo_height() or 480
        img.thumbnail((cw, ch))
        self._display_size = img.size
        photo = ImageTk.PhotoImage(img)
        self._photo = photo
        ox = (cw - img.width)  // 2
        oy = (ch - img.height) // 2
        self._offset = (ox, oy)
        self._canvas.create_image(ox, oy, image=photo, anchor="nw")
        self._load_labels()
        self._update_nav()

    def _on_resize(self, event) -> None:
        if self._live_running and self._frozen_frame is None:
            return   # live mode — display loop handles redraws
        if self._images:
            self._show_image(self._idx)

    # ── YOLO label I/O ────────────────────────────────────────────────────────
    def _label_path(self) -> Optional[Path]:
        proj = self._app.get_project()
        if not proj or not self._images:
            return None
        img_path = self._images[self._idx]
        # Canonical stem — single source of truth shared with dataset_builder
        # (two conventions here is what silently broke training: BUG-01)
        unique_stem = proj.paths.label_stem(img_path)

        label_dir = proj.paths.tagged_labels
        label_dir.mkdir(parents=True, exist_ok=True)
        # Also ensure tagged/images has the image (copy with unique name if needed)
        dest_name = unique_stem + img_path.suffix
        tagged_img = proj.paths.tagged_images / dest_name
        tagged_img.parent.mkdir(parents=True, exist_ok=True)
        if not tagged_img.exists():
            try:
                shutil.copy2(str(img_path), str(tagged_img))
            except Exception:
                pass
        return proj.paths.label_file(img_path)

    def _load_labels(self) -> None:
        proj = self._app.get_project()
        lp = self._label_path()
        if not lp or not lp.exists():
            return
        classes = proj.classes if proj else []
        with open(lp) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                ci, xc, yc, bw, bh = (int(parts[0]),
                                       float(parts[1]), float(parts[2]),
                                       float(parts[3]), float(parts[4]))
                if ci >= len(classes):
                    continue
                ox, oy = self._offset
                dw, dh = self._display_size
                x1 = int((xc - bw / 2) * dw) + ox
                y1 = int((yc - bh / 2) * dh) + oy
                x2 = int((xc + bw / 2) * dw) + ox
                y2 = int((yc + bh / 2) * dh) + oy
                cls = classes[ci]
                color = (T.COLOR_OK  if cls.kind == "ok"  else
                         T.COLOR_NOK if cls.kind == "nok" else T.TEXT_DIM)
                box = _BBox(x1, y1, x2, y2, ci, cls.name, color)
                self._draw_box(box)
                self._boxes.append(box)

    def _save_labels(self) -> None:
        lp = self._label_path()
        if not lp:
            return
        dw, dh = self._display_size
        lines = []
        for box in self._boxes:
            ox, oy = self._offset
            xc = ((box.x1 + box.x2) / 2 - ox) / dw
            yc = ((box.y1 + box.y2) / 2 - oy) / dh
            bw = (box.x2 - box.x1) / dw
            bh = (box.y2 - box.y1) / dh
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            bw = max(0.01, min(1.0, bw))
            bh = max(0.01, min(1.0, bh))
            lines.append(f"{box.class_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        with open(lp, "w") as f:
            f.write("\n".join(lines))

    # ── mouse draw ────────────────────────────────────────────────────────────
    def _on_press(self, event) -> None:
        # Drawing is allowed on: static images, a frozen (grabbed) frame,
        # or a PAUSED video frame.  Only a moving live feed rejects clicks.
        if (self._live_running and self._frozen_frame is None
                and not self._video_paused):
            return
        self._draw_start = (event.x, event.y)

    def _on_drag(self, event) -> None:
        if self._draw_start is None:
            return
        if self._active_rect:
            self._canvas.delete(self._active_rect)
        x1, y1 = self._draw_start
        proj = self._app.get_project()
        if not proj or not proj.classes:
            return
        cls = proj.classes[min(self._selected_class, len(proj.classes) - 1)]
        color = (T.COLOR_OK  if cls.kind == "ok"  else
                 T.COLOR_NOK if cls.kind == "nok" else T.TEXT_DIM)
        self._active_rect = self._canvas.create_rectangle(
            x1, y1, event.x, event.y,
            outline=color, width=2, dash=(4, 2)
        )

    def _on_release(self, event) -> None:
        if self._draw_start is None:
            return
        x1, y1 = self._draw_start
        x2, y2 = event.x, event.y
        self._draw_start = None
        if self._active_rect:
            self._canvas.delete(self._active_rect)
            self._active_rect = None
        if abs(x2 - x1) < 8 or abs(y2 - y1) < 8:
            return  # too small
        proj = self._app.get_project()
        if not proj or not proj.classes:
            return
        idx = min(self._selected_class, len(proj.classes) - 1)
        cls = proj.classes[idx]
        color = (T.COLOR_OK  if cls.kind == "ok"  else
                 T.COLOR_NOK if cls.kind == "nok" else T.TEXT_DIM)
        box = _BBox(x1, y1, x2, y2, idx, cls.name, color)
        self._draw_box(box)
        self._boxes.append(box)

    def _draw_box(self, box: _BBox) -> None:
        box.canvas_id = self._canvas.create_rectangle(
            box.x1, box.y1, box.x2, box.y2,
            outline=box.color, width=2
        )
        box.label_id = self._canvas.create_text(
            box.x1 + 4, box.y1 + 4,
            text=box.class_name, anchor="nw",
            fill=box.color, font=("Segoe UI", 9, "bold")
        )

    def _cancel_draw(self) -> None:
        self._draw_start = None
        if self._active_rect:
            self._canvas.delete(self._active_rect)
            self._active_rect = None

    # ── actions ───────────────────────────────────────────────────────────────
    def _write_grab(self, frame) -> bool:
        """
        Write *frame* to the project as a new image and point self._idx at it,
        WITHOUT clearing the boxes the user already drew.

        This is the ONLY place live/video frames touch the disk — always
        together with their labels (from _save), never as orphans.
        Returns False (with a user-facing message) when saving isn't possible.
        """
        proj = self._app.get_project()
        if not proj:
            self._msg.configure(text=t("tag_open_project_first"), text_color=T.WARN)
            return False
        classes = proj.classes
        if not classes:
            self._msg.configure(text=t("tag_define_classes_first"), text_color=T.WARN)
            return False
        if frame is None:
            self._msg.configure(text=t("tag_no_frame"), text_color=T.WARN)
            return False

        cls = classes[min(self._selected_class, len(classes) - 1)]
        dest_dir = proj.paths.raw_class_dir(cls.name)
        dest_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_path = dest_dir / f"tag_{ts}.jpg"
        cv2.imwrite(str(img_path), frame)

        self._images.append(img_path)
        self._idx = len(self._images) - 1
        # NOTE: no _show_image() here — it would clear the drawn boxes.
        # _paint_frame already recorded display geometry for _save_labels.
        return True

    def _save(self) -> None:
        live = self._src_var.get() in ("camera", "video")
        if live:
            # Frame + labels are written together, exactly once, on Guardar.
            if not self._boxes:
                self._msg.configure(text=t("tag_draw_box_first"), text_color=T.WARN)
                return
            if self._frozen_frame is not None:
                frame = self._frozen_frame
            else:
                with self._frame_lock:
                    frame = self._live_frame
            if not self._write_grab(frame):
                return
        elif not self._images:
            # Static-image mode with nothing loaded: nothing to save onto.
            self._msg.configure(text=t("tag_no_frame"), text_color=T.WARN)
            return

        self._save_labels()
        self._msg.configure(text=t("tag_saved"), text_color=T.ACCENT)
        self._update_progress()
        # After saving in live mode: un-freeze and clean up for the next frame
        if live:
            self._frozen_frame = None
            self._clear_boxes()
            if self._live_running and not self._video_paused:
                self._cam_status.configure(text=t("tag_live"), text_color=T.ACCENT)

    def _delete_last(self) -> None:
        if not self._boxes:
            return
        box = self._boxes.pop()
        if box.canvas_id:
            self._canvas.delete(box.canvas_id)
        if box.label_id:
            self._canvas.delete(box.label_id)

    def _clear_boxes(self) -> None:
        for box in self._boxes:
            if box.canvas_id:
                self._canvas.delete(box.canvas_id)
            if box.label_id:
                self._canvas.delete(box.label_id)
        self._boxes.clear()

    def _prev(self) -> None:
        if self._live_running:
            return
        self._save_labels()
        self._show_image(self._idx - 1)

    def _next(self) -> None:
        if self._live_running:
            return
        self._save_labels()
        self._show_image(self._idx + 1)

    def _update_nav(self) -> None:
        total = len(self._images)
        self._img_lbl.configure(text=t("tag_image", current=self._idx + 1, total=total))

    def _update_progress(self) -> None:
        proj = self._app.get_project()
        if not proj:
            return
        total = len(self._images)
        labeled = 0
        for img in self._images:
            lf = proj.paths.label_file(img)
            # An EMPTY .txt means "seen, no boxes drawn" — navigating with
            # N/P auto-saves those.  Counting them as labeled is what made
            # the header say 71/71 while the dataset only had 40 real pairs.
            try:
                if lf.exists() and lf.stat().st_size > 0:
                    labeled += 1
            except OSError:
                pass
        self._progress_lbl.configure(
            text=t("tag_progress", labeled=labeled, total=total))
        if proj.classes:
            stats = proj.images_per_class()
            lines = [f"  {k}: {v}" for k, v in stats.items()]
            self._stats_lbl.configure(text="\n".join(lines))

    # ── lifecycle (navigation) ────────────────────────────────────────────────
    def on_hide(self) -> None:
        """Stop live camera when the user navigates away to save CPU."""
        self._stop_live()

    def on_show(self) -> None:
        """Reload image list when the user navigates back (picks up new captures)."""
        if self._src_var.get() == "images":
            self._load_images()

    def on_close(self) -> None:
        """Unbind hotkeys and stop camera so they don't leak to other screens."""
        top = self.winfo_toplevel()
        for seq, bid in getattr(self, "_hotkey_ids", []):
            try:
                top.unbind(seq, bid)
            except Exception:
                pass
        self._hotkey_ids = []
        self._stop_live()
