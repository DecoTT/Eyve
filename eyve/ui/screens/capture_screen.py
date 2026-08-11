"""
Capture screen — live camera preview or video file.
Frames are captured and saved into the project's raw/images/<class>/ folder.

Architecture
------------
_grab_loop  (background thread)
    cap.read() blocks until the camera delivers a frame — no sleep needed.
    Writes latest frame to _latest_frame/_last_frame under _frame_lock.
    When recording: pushes frame.copy() to _rec_queue (never blocks camera).

_display_loop  (main thread, 30 fps poll)
    Reads _latest_frame, resizes with cv2 (SIMD), converts to PhotoImage.
    Reuses a single canvas image item via itemconfig() — never accumulates
    canvas objects (that was the cause of progressive slowdown in Eyve 2.0).

_record_writer_loop  (dedicated writer thread)
    Drains _rec_queue and calls VideoWriter.write() independently of the
    camera loop.  Slow encoding never throttles the preview or capture FPS.
"""
from __future__ import annotations
import cv2
import queue
import threading
import time
from collections import deque
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
from eyve.camera.camera_enum import get_camera_labels, label_to_index, label_to_device

if TYPE_CHECKING:
    from eyve.ui.app import EyveApp


def _open_camera(idx: int) -> cv2.VideoCapture:
    """
    Open a camera as fast as possible on Windows.

    Backend priority (Windows):
      1. CAP_DSHOW  — DirectShow, fastest open (~200 ms), no pipeline renegotiation.
                      Used by Eyve 2.0 (main_prod.py, train_cameravideo.py) — proven fast.
      2. CAP_MSMF   — Modern Windows Media Foundation, may take 5-6 s just to open,
                      and any cap.set(WIDTH/HEIGHT/FPS) triggers a full pipeline
                      renegotiation adding another 15-20 s.  Used as fallback only.
      3. CAP_ANY    — Last resort.

    Rules:
      - Never call cap.set(WIDTH/HEIGHT/FPS) during startup.
      - Resize frames in software for UI/inference if needed.
      - Show a loading animation in the canvas while this runs.
    """
    import platform as _platform
    import time as _t
    from eyve.core.logger import log as _log

    t0 = _t.monotonic()

    on_windows = _platform.system() == "Windows"
    backends = (
        [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        if on_windows else
        [cv2.CAP_ANY]
    )

    cap = None
    for api in backends:
        name = {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF",
                cv2.CAP_ANY: "ANY"}.get(api, str(api))
        # DSHOW releases asynchronously: the filter graph keeps tearing down
        # for 200-500 ms after cap.release().  Retry before falling back to
        # MSMF (which takes 5+ s to open and renegotiates on every cap.set).
        max_tries = 5 if api == cv2.CAP_DSHOW else 1
        c = None
        for attempt in range(max_tries):
            c = cv2.VideoCapture(idx, api)
            if c.isOpened():
                break
            c.release()
            c = None
            if attempt < max_tries - 1:
                _t.sleep(0.35)
        elapsed = (_t.monotonic() - t0) * 1000
        _log.debug(f"  VideoCapture({idx}, {name})  opened={bool(c and c.isOpened())}  {elapsed:.0f} ms")
        if c and c.isOpened():
            cap = c
            break
        if c:
            c.release()

    if cap is None:
        cap = cv2.VideoCapture(idx)   # absolute last resort

    if cap.isOpened():
        # BUFFERSIZE=1 keeps latency low (drop stale frames).
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Request MJPEG once, here, as part of the initial open sequence.
        # USB cameras compress frames on-chip with MJPEG; without it the
        # driver sends uncompressed YUV which saturates USB 2.0 bandwidth
        # (~5 fps at 1080p).  Setting FOURCC here bundles the renegotiation
        # with the camera open rather than doing it separately in
        # _apply_resolution — that was the double-renegotiation bug that
        # caused 5 fps.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        # Default to 720p ("Auto HD").  720p MJPEG over USB 2.0 is fast and
        # responsive — same baseline as Eyve 2.0 train_cameravideo.py.
        # The user can select 1920×1080 explicitly in the dropdown if they
        # need full resolution (costs more USB bandwidth and CPU to display).
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

        _log.debug(f"  _open_camera({idx}) ready  "
                   f"{(_t.monotonic()-t0)*1000:.0f} ms total  "
                   f"({cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
                   f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
                   f"@ {cap.get(cv2.CAP_PROP_FPS):.0f} fps)")
    return cap


class CaptureScreen(ctk.CTkFrame):
    def __init__(self, master, app: "EyveApp", **kwargs):
        super().__init__(master, fg_color=T.BG_DARK, **kwargs)
        self._app = app
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._photo = None              # prevent GC of current PhotoImage
        self._latest_frame = None       # producer writes, consumer reads
        self._last_frame = None         # captured on demand
        self._frame_lock = threading.Lock()
        self._captured_count = 0
        self._video_path: Optional[Path] = None
        # Pre-warm: open camera in background as soon as it's selected so
        # clicking Start is nearly instant.
        self._prewarm_cap: Optional[cv2.VideoCapture] = None
        self._prewarm_idx: int = -1      # -1 = none; >=0 = warming OR warm
        self._prewarm_pending: bool = False  # True = Start clicked, waiting for warm
        # Loading animation while camera is opening
        self._warming: bool = False
        self._anim_step: int = 0
        # Recording state
        self._recording: bool = False
        self._writer: Optional[cv2.VideoWriter] = None
        self._rec_lock = threading.Lock()
        self._rec_file: Optional[Path] = None
        self._rec_frames: int = 0
        self._last_rec_dir: Optional[Path] = None   # folder of last saved recording
        # Dedicated writer thread + bounded queue (decouples encoding from camera)
        self._rec_queue: queue.Queue = queue.Queue(maxsize=90)
        self._rec_writer_thread: Optional[threading.Thread] = None
        # Canvas image reuse — one item, updated via itemconfig() every frame
        self._canvas_img_id: Optional[int] = None
        # Dual FPS counters: camera (grab loop) + render (display loop)
        self._cam_fps_times:  deque = deque(maxlen=60)   # camera delivery rate
        self._fps_times:      deque = deque(maxlen=60)   # UI render rate
        self._last_fps_ui_update: float = 0.0            # rate-limit label updates
        self._build()

    # ── layout ────────────────────────────────────────────────────────────────
    def _build(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(self, fg_color=T.BG_CARD, corner_radius=0)
        hdr.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(hdr, text=t("cap_title"),
                     font=T.bold(T.FONT_XL), text_color=T.TEXT_PRI).pack(
            side="left", padx=24, pady=14)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=16, pady=12)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0)
        body.grid_rowconfigure(0, weight=1)

        self._canvas = tk.Canvas(body, bg="#000000", highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nsew")

        right = ctk.CTkFrame(body, fg_color=T.BG_CARD, corner_radius=10, width=240)
        right.grid(row=0, column=1, sticky="ns", padx=(12, 0))
        right.grid_propagate(False)
        self._build_controls(right)

    def _build_controls(self, parent) -> None:
        pad = {"padx": 16, "pady": 6}

        ctk.CTkLabel(parent, text=t("cap_title"),
                     font=T.bold(T.FONT_MD), text_color=T.TEXT_PRI).pack(pady=(16, 4))
        ctk.CTkFrame(parent, height=1, fg_color=T.BORDER).pack(fill="x", padx=12, pady=4)

        # source radio
        self._source_var = ctk.StringVar(value="camera")
        ctk.CTkRadioButton(parent, text=t("cap_source_camera"),
                           variable=self._source_var, value="camera",
                           font=T.font(T.FONT_SM),
                           command=self._on_source_change).pack(anchor="w", **pad)
        ctk.CTkRadioButton(parent, text=t("cap_source_video"),
                           variable=self._source_var, value="video",
                           font=T.font(T.FONT_SM),
                           command=self._on_source_change).pack(anchor="w", padx=16, pady=(0, 4))

        self._vid_row = ctk.CTkFrame(parent, fg_color="transparent")
        self._vid_row.pack(fill="x", padx=12, pady=(0, 6))
        self._vid_path_lbl = ctk.CTkLabel(self._vid_row, text="…",
                                           font=T.font(T.FONT_XS), text_color=T.TEXT_DIM,
                                           wraplength=160, anchor="w")
        self._vid_path_lbl.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(self._vid_row, text="…", width=28,
                      fg_color=T.BG_INPUT, command=self._browse_video).pack(side="right")
        self._vid_row.pack_forget()

        # camera selector — populated async so the UI doesn't freeze
        self._cam_row = ctk.CTkFrame(parent, fg_color="transparent")
        self._cam_row.pack(fill="x", padx=12, pady=(0, 6))
        ctk.CTkLabel(self._cam_row, text=t("cap_camera_id"),
                     font=T.font(T.FONT_XS), text_color=T.TEXT_SEC).pack(anchor="w")
        self._cam_idx = ctk.CTkOptionMenu(
            self._cam_row, values=[t("cam_loading")],
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD,
            font=T.font(T.FONT_XS),
            state="disabled",
            command=self._on_cam_selection_change,
        )
        self._cam_idx.pack(fill="x")
        ctk.CTkButton(
            self._cam_row, text=t("cap_refresh_cams"),
            font=T.font(T.FONT_XS), height=24, fg_color="transparent",
            text_color=T.TEXT_DIM, hover_color=T.BG_INPUT,
            command=self._refresh_cameras,
        ).pack(anchor="e")
        # kick off background enumeration
        threading.Thread(target=self._load_cameras_bg, daemon=True).start()

        # resolution selector
        res_row = ctk.CTkFrame(parent, fg_color="transparent")
        res_row.pack(fill="x", padx=12, pady=(0, 6))
        ctk.CTkLabel(res_row, text=t("cap_resolution"),
                     font=T.font(T.FONT_XS), text_color=T.TEXT_SEC).pack(anchor="w")
        _RES_OPTS = ["Auto HD", "1920×1080", "1280×720", "640×480"]
        self._res_menu = ctk.CTkOptionMenu(
            res_row, values=_RES_OPTS,
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD,
            font=T.font(T.FONT_XS),
        )
        self._res_menu.set("Auto HD")
        self._res_menu.pack(fill="x")

        ctk.CTkFrame(parent, height=1, fg_color=T.BORDER).pack(fill="x", padx=12, pady=6)

        # start / stop
        self._start_btn = ctk.CTkButton(parent, text=t("cap_start"),
                                         fg_color=T.ACCENT, text_color="#000",
                                         command=self._start)
        self._start_btn.pack(fill="x", padx=12, pady=4)

        self._stop_btn = ctk.CTkButton(parent, text=t("cap_stop"),
                                        fg_color=T.BG_INPUT,
                                        state="disabled", command=self._stop)
        self._stop_btn.pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkFrame(parent, height=1, fg_color=T.BORDER).pack(fill="x", padx=12, pady=4)

        # class selector
        ctk.CTkLabel(parent, text=t("cap_session_class"),
                     font=T.font(T.FONT_SM), text_color=T.TEXT_SEC).pack(padx=12, anchor="w")
        self._class_menu = ctk.CTkOptionMenu(parent, values=[t("cap_no_class")],
                                              fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
                                              dropdown_fg_color=T.BG_CARD,
                                              font=T.font(T.FONT_SM))
        self._class_menu.pack(fill="x", padx=12, pady=(4, 8))
        self._refresh_classes()

        # capture button
        self._cap_btn = ctk.CTkButton(parent, text=t("cap_capture"),
                                       fg_color=T.ACCENT2, text_color="#000",
                                       height=44, font=T.bold(T.FONT_MD),
                                       state="disabled", command=self._capture)
        self._cap_btn.pack(fill="x", padx=12, pady=4)

        # record button
        self._rec_btn = ctk.CTkButton(parent, text=t("cap_record"),
                                       fg_color=T.DANGER, text_color="#fff",
                                       height=44, font=T.bold(T.FONT_MD),
                                       state="disabled", command=self._toggle_record)
        self._rec_btn.pack(fill="x", padx=12, pady=(0, 4))

        self._rec_lbl = ctk.CTkLabel(parent, text="",
                                      font=T.font(T.FONT_XS), text_color=T.DANGER)
        self._rec_lbl.pack(pady=(0, 2))

        # Shown only after a recording is saved
        self._open_folder_btn = ctk.CTkButton(
            parent, text=t("cap_open_folder"),
            font=T.font(T.FONT_XS), height=22,
            fg_color="transparent", text_color=T.ACCENT2,
            hover_color=T.BG_INPUT,
            command=self._open_rec_folder,
        )
        # hidden until first recording completes
        self._open_folder_btn.pack(fill="x", padx=12, pady=(0, 4))
        self._open_folder_btn.pack_forget()

        ctk.CTkFrame(parent, height=1, fg_color=T.BORDER).pack(fill="x", padx=12, pady=4)

        # counter + status
        self._count_lbl = ctk.CTkLabel(parent, text=t("cap_captured", n=0),
                                        font=T.bold(T.FONT_MD), text_color=T.TEXT_PRI)
        self._count_lbl.pack(pady=(6, 2))

        self._status = ctk.CTkLabel(parent, text="",
                                     font=T.font(T.FONT_SM), text_color=T.TEXT_DIM,
                                     wraplength=200)
        self._status.pack(padx=12, pady=4)

        # FPS display
        self._fps_lbl = ctk.CTkLabel(parent, text="",
                                      font=T.font(T.FONT_XS), text_color=T.TEXT_DIM)
        self._fps_lbl.pack(pady=(0, 8))

        # bind_all is blocked by CustomTkinter — bind to the real Tk toplevel instead
        top = self.winfo_toplevel()
        self._hotkey_ids = [
            top.bind("<c>", lambda e: self._capture(), add=True),
            top.bind("<C>", lambda e: self._capture(), add=True),
        ]

    def _load_cameras_bg(self) -> None:
        """Background thread: enumerate cameras without blocking the UI."""
        labels = get_camera_labels()
        self.after(0, lambda: self._on_cameras_loaded(labels))

    def _on_cameras_loaded(self, labels: list[str]) -> None:
        """Main thread: update dropdown once enumeration completes."""
        if not self.winfo_exists():
            return
        if not labels:
            labels = [t("cam_none")]
        self._cam_idx.configure(values=labels, state="normal")
        self._cam_idx.set(labels[0])
        # Kick off pre-warm for the first camera immediately
        if labels and labels[0] != t("cam_none"):
            self._start_prewarm(label_to_index(labels[0]))

    def _on_cam_selection_change(self, label: str) -> None:
        """Called when the user picks a different camera from the dropdown."""
        if self._source_var.get() == "camera" and not self._running:
            self._start_prewarm(label_to_index(label))

    def _start_prewarm(self, idx: int) -> None:
        """
        Open camera idx in a background thread so clicking Start is instant.

        Guard: if _prewarm_idx == idx the camera is EITHER already warming OR
        already warm — either way there is nothing to do, so we return early.
        This prevents launching two concurrent VideoCapture() calls for the
        same index (which was the previous double-open bug).
        """
        if self._prewarm_idx == idx:
            return  # already warming or warmed — do not start a second thread
        # Cancel / release a prewarm for a different camera
        self._prewarm_pending = False
        self._warming = False
        if self._prewarm_cap is not None:
            self._prewarm_cap.release()
            self._prewarm_cap = None
        self._prewarm_idx = idx

        # Start canvas animation immediately
        if not self._running:
            self._warming = True
            self._anim_step = 0
            self._animate_warmup()

        def _bg() -> None:
            cap = _open_camera(idx)
            self.after(0, lambda: self._on_prewarm_ready(idx, cap))

        threading.Thread(target=_bg, daemon=True).start()

    def _on_prewarm_ready(self, idx: int, cap: cv2.VideoCapture) -> None:
        """Main-thread callback: prewarm thread finished."""
        from eyve.core.logger import log
        self._warming = False   # stop spinner regardless of outcome
        if self._prewarm_idx != idx:
            cap.release()   # user switched camera while we were warming
            return
        if self._prewarm_pending:
            # Start was clicked while the camera was still warming — use it now
            self._prewarm_pending = False
            self._prewarm_idx = -1
            log.debug(f"  Camera {idx} warm — starting (was pending)")
            self._on_camera_ready(cap)
        elif not self._running:
            self._prewarm_idx = -1
            self._prewarm_cap = None
            log.debug(f"  Camera {idx} pre-warmed ✓ — auto-starting preview")
            self._on_camera_ready(cap)   # auto-start live preview, no click needed
        else:
            cap.release()   # camera already active (user was very fast)

    # ── canvas loading animation ──────────────────────────────────────────────
    _SPINNER = ["◐", "◓", "◑", "◒"]

    @staticmethod
    def _canvas_size(canvas: tk.Canvas) -> tuple[int, int]:
        """Return canvas dimensions, falling back to defaults if not yet realized."""
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        return (cw if cw > 8 else 640), (ch if ch > 8 else 480)

    def _animate_warmup(self) -> None:
        """Draws a spinner on the canvas every 200 ms while warming up."""
        if not self._warming or self._running:
            return
        cw, ch = self._canvas_size(self._canvas)
        self._canvas.delete("all")
        self._canvas.create_rectangle(0, 0, cw, ch, fill="#0a0a0a", outline="")
        spin = self._SPINNER[self._anim_step % len(self._SPINNER)]
        self._canvas.create_text(
            cw // 2, ch // 2 - 24,
            text=spin, fill=T.ACCENT, font=("Segoe UI", 36), anchor="center",
        )
        self._canvas.create_text(
            cw // 2, ch // 2 + 20,
            text=t("cap_opening_cam"), fill=T.TEXT_SEC,
            font=("Segoe UI", 13), anchor="center",
        )
        self._anim_step += 1
        self.after(200, self._animate_warmup)

    def _refresh_cameras(self) -> None:
        self._cam_idx.configure(values=[t("cam_refreshing")], state="disabled")
        threading.Thread(target=self._load_cameras_bg, daemon=True).start()

    def _refresh_classes(self) -> None:
        proj = self._app.get_project()
        names = [c.name for c in proj.classes] if proj else []
        if names:
            self._class_menu.configure(values=names)
            self._class_menu.set(names[0])
        else:
            self._class_menu.configure(values=[t("cap_no_class")])
            self._class_menu.set(t("cap_no_class"))

    # ── source switch ─────────────────────────────────────────────────────────
    def _on_source_change(self) -> None:
        if self._source_var.get() == "video":
            self._cam_row.pack_forget()
            self._vid_row.pack(fill="x", padx=12, pady=(0, 6))
        else:
            self._vid_row.pack_forget()
            self._cam_row.pack(fill="x", padx=12, pady=(0, 6))

    def _browse_video(self) -> None:
        p = filedialog.askopenfilename(
            title=t("cap_select_video"),
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All", "*.*")]
        )
        if p:
            self._video_path = Path(p)
            self._vid_path_lbl.configure(text=p)

    # ── camera control ────────────────────────────────────────────────────────
    def _start(self) -> None:
        self._stop()

        if self._source_var.get() == "camera":
            idx = label_to_index(self._cam_idx.get())
            is_camera = True
        else:
            if not self._video_path:
                self._status.configure(text=t("cap_select_video"), text_color=T.WARN)
                return
            idx = str(self._video_path)
            is_camera = False

        if is_camera:
            if self._prewarm_cap is not None and self._prewarm_idx == idx:
                # Camera already open — instant start
                cap = self._prewarm_cap
                self._prewarm_cap = None
                self._prewarm_idx = -1
                self._on_camera_ready(cap)
                return
            if self._prewarm_idx == idx:
                # Prewarm in progress — hook into it instead of opening a 2nd time
                self._start_btn.configure(state="disabled")
                self._status.configure(text="⏳ " + t("cap_opening_cam"), text_color=T.TEXT_SEC)
                self._prewarm_pending = True
                return

        # No prewarm available — open in background
        self._start_btn.configure(state="disabled")
        self._status.configure(text="⏳ " + t("cap_opening_cam"), text_color=T.TEXT_SEC)

        def _open_bg() -> None:
            cap = _open_camera(idx) if is_camera else cv2.VideoCapture(idx)
            self.after(0, lambda: self._on_camera_ready(cap))

        threading.Thread(target=_open_bg, daemon=True).start()

    # ── resolution map ────────────────────────────────────────────────────────
    _RES_MAP = {"1920×1080": (1920, 1080), "1280×720": (1280, 720), "640×480": (640, 480)}

    def _apply_resolution(self, cap: cv2.VideoCapture) -> None:
        """
        Apply the selected resolution dropdown value to an already-open camera.

        FOURCC is NOT set here — it was already set to MJPEG inside
        _open_camera() as part of the initial open sequence.  Setting it
        again would trigger a second DSHOW renegotiation (the 5-fps bug).

        "Nativa" means: keep the 1080p (or camera-max) that _open_camera()
        already negotiated — nothing to do.
        Any other option changes only the frame dimensions.
        """
        from eyve.core.logger import log
        res = getattr(self, "_res_menu", None)
        if res is None:
            return
        sel = res.get()

        if sel not in self._RES_MAP:
            # "Auto HD" — _open_camera already set 1920×1080 (or camera max)
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            log.debug(f"  Resolution=Auto HD  actual={w}x{h} @ {fps:.0f} fps")
            return

        # Specific resolution — only change dimensions; FOURCC already MJPG.
        w, h = self._RES_MAP[sel]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        log.debug(
            f"  Resolution requested={w}x{h}  actual={actual_w}x{actual_h}"
            f"  @ {actual_fps:.0f} fps"
        )

    def _on_camera_ready(self, cap: cv2.VideoCapture) -> None:
        """Called on the main thread once the camera has been opened in background."""
        if not cap.isOpened():
            cap.release()
            self._status.configure(text=t("cap_no_camera"), text_color=T.DANGER)
            self._start_btn.configure(state="normal")
            return

        self._apply_resolution(cap)
        self._cap = cap
        self._running = True
        self._fps_times.clear()
        self._cam_fps_times.clear()
        self._canvas_img_id = None   # force create_image on first frame
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._cap_btn.configure(state="normal")
        self._rec_btn.configure(state="normal")
        self._status.configure(text=t("status_camera_ok"), text_color=T.ACCENT)

        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        self._display_loop()   # start main-thread render poller

    def _stop(self) -> None:
        self._running = False
        self._prewarm_pending = False
        self._stop_recording()          # finalize any active recording
        if self._cap:
            self._cap.release()
            self._cap = None
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._cap_btn.configure(state="disabled")
        self._rec_btn.configure(state="disabled")
        self._fps_lbl.configure(text="")
        self._canvas_img_id = None   # next start creates a fresh canvas item
        # NOTE: do NOT re-prewarm here — that would auto-restart the preview
        # right after the user explicitly clicked Stop.  When Start is clicked
        # again, _start() will open the camera fresh (still fast via DSHOW).

    # ── grab loop (background thread) ─────────────────────────────────────────
    def _grab_loop(self) -> None:
        """
        Runs in a daemon thread.
        cap.read() blocks until the camera delivers a frame — no sleep needed.
        For video files we add a 33 ms sleep to avoid spinning at 100% CPU.

        Recording: frames are pushed to _rec_queue; the dedicated
        _record_writer_loop thread drains it and calls VideoWriter.write().
        This decouples slow encoding from the camera read loop so that a slow
        disk or codec never throttles the preview or capture FPS.
        """
        is_video = self._source_var.get() == "video"
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                if is_video:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            with self._frame_lock:
                self._latest_frame = frame
                self._last_frame   = frame   # for still capture
            # Camera FPS counter (rolling window)
            self._cam_fps_times.append(time.monotonic())
            # Push to recording queue — never block the camera loop
            if self._recording:
                try:
                    self._rec_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass   # encoder is behind — drop this frame silently
            if is_video:
                time.sleep(1 / 30)

    # ── recording writer loop (dedicated thread) ──────────────────────────────
    def _record_writer_loop(self) -> None:
        """
        Runs in its own daemon thread while recording is active.
        Drains _rec_queue and calls VideoWriter.write() — completely isolated
        from the camera grab loop so a slow codec/disk never stalls the preview.
        Exits when it receives a None sentinel from _stop_recording().
        """
        while True:
            try:
                item = self._rec_queue.get(timeout=0.2)
            except queue.Empty:
                if not self._recording:
                    break   # recording stopped and queue is drained
                continue
            if item is None:
                break       # explicit sentinel — stop now
            with self._rec_lock:
                writer = self._writer
            if writer is not None:
                writer.write(item)
                with self._rec_lock:
                    self._rec_frames += 1

    # ── display loop (main thread) ────────────────────────────────────────────
    def _display_loop(self) -> None:
        """
        Polls the latest frame every ~33 ms (≈30 fps) on the main thread.
        Targeting 30 fps matches camera delivery and halves Tk render work
        compared to the old 60 fps poll.

        Canvas strategy: one persistent image item created on the first frame,
        then updated via itemconfig(image=…) + coords().  This avoids the
        canvas-object accumulation bug where thousands of stacked create_image()
        calls caused progressive slowdown over time.

        Resize pipeline:
          1. cv2.resize  — C++ SIMD, 1-3 ms for 720p/1080p → display size
          2. cv2.cvtColor — on the already-small frame
          3. Image.fromarray → ImageTk.PhotoImage — negligible at display size
        """
        if not self._running:
            return

        frame = None
        with self._frame_lock:
            if self._latest_frame is not None:
                frame = self._latest_frame
                self._latest_frame = None

        if frame is not None:
            try:
                cw = self._canvas.winfo_width()  or 640
                ch = self._canvas.winfo_height() or 480

                # ── 1. resize with OpenCV ─────────────────────────────────────
                fh, fw = frame.shape[:2]
                scale = min(cw / fw, ch / fh)
                if scale < 0.99:
                    nw = max(1, int(fw * scale))
                    nh = max(1, int(fh * scale))
                    frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

                # ── 2. convert + PhotoImage ───────────────────────────────────
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                self._photo = photo   # keep GC reference

                # ── 3. reuse canvas item ──────────────────────────────────────
                cx, cy = cw // 2, ch // 2
                if self._canvas_img_id is None:
                    self._canvas.delete("all")
                    self._canvas_img_id = self._canvas.create_image(
                        cx, cy, image=photo, anchor="center"
                    )
                else:
                    self._canvas.itemconfig(self._canvas_img_id, image=photo)
                    self._canvas.coords(self._canvas_img_id, cx, cy)

                # ── 4. FPS counters (rate-limited label update) ───────────────
                now = time.monotonic()
                self._fps_times.append(now)
                if now - self._last_fps_ui_update >= 0.5:
                    self._last_fps_ui_update = now
                    # Render FPS
                    r_fps = 0.0
                    if len(self._fps_times) >= 2:
                        w = self._fps_times[-1] - self._fps_times[0]
                        if w > 0:
                            r_fps = (len(self._fps_times) - 1) / w
                    # Camera FPS
                    c_fps = 0.0
                    ct = self._cam_fps_times
                    if len(ct) >= 2:
                        w = ct[-1] - ct[0]
                        if w > 0:
                            c_fps = (len(ct) - 1) / w
                    self._fps_lbl.configure(
                        text=f"Cam {c_fps:.0f} fps  ·  UI {r_fps:.0f} fps"
                    )
            except Exception:
                pass

        self.after(33, self._display_loop)   # ~30 fps poll

    # ── capture ───────────────────────────────────────────────────────────────
    def _capture(self) -> None:
        if not self._running:
            return
        with self._frame_lock:
            frame = self._last_frame
        if frame is None:
            return
        proj = self._app.get_project()

        # Resolve save directory: class subfolder if one is selected, else
        # the project root raw/images/ folder.  No project → temp folder.
        if proj:
            class_name = self._class_menu.get()
            if class_name and class_name != t("cap_no_class"):
                dest_dir = proj.paths.raw_class_dir(class_name)
            else:
                dest_dir = proj.paths.raw_images
        else:
            from pathlib import Path
            dest_dir = Path.home() / "eyve_capture"

        dest_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = dest_dir / f"img_{ts}.jpg"
        cv2.imwrite(str(filename), frame)
        self._captured_count += 1
        self._count_lbl.configure(text=t("cap_captured", n=self._captured_count))
        self._status.configure(text=t("cap_saved"), text_color=T.ACCENT)

    # ── recording ─────────────────────────────────────────────────────────────
    def _toggle_record(self) -> None:
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        if not self._running:
            return

        # Get frame dimensions from the last frame
        with self._frame_lock:
            frame = self._last_frame
        if frame is None:
            self._status.configure(text=t("cap_no_frame_yet"), text_color=T.WARN)
            return
        h, w = frame.shape[:2]

        proj = self._app.get_project()

        # Resolve save directory: class subfolder if selected, else root raw/images/.
        if proj:
            class_name = self._class_menu.get()
            if class_name and class_name != t("cap_no_class"):
                dest_dir = proj.paths.raw_class_dir(class_name)
            else:
                dest_dir = proj.paths.raw_images
        else:
            from pathlib import Path
            dest_dir = Path.home() / "eyve_capture"

        dest_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_path = dest_dir / f"vid_{ts}.mp4"

        fps = self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 30.0
        if fps <= 0:
            fps = 30.0
        writer = cv2.VideoWriter(
            str(rec_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        if not writer.isOpened():
            self._status.configure(text=t("cap_rec_file_error"), text_color=T.DANGER)
            return

        # Flush any leftover frames from a previous recording
        while not self._rec_queue.empty():
            try:
                self._rec_queue.get_nowait()
            except queue.Empty:
                break

        with self._rec_lock:
            self._writer = writer
            self._rec_frames = 0
            self._recording = True
            self._rec_file = rec_path
            self._rec_start = time.monotonic()

        # Dedicated writer thread — encoding never blocks the grab loop
        self._rec_writer_thread = threading.Thread(
            target=self._record_writer_loop, daemon=True
        )
        self._rec_writer_thread.start()

        self._rec_btn.configure(text=t("cap_stop_save"), fg_color="#8B0000")
        self._rec_lbl.configure(text="● REC  0s")
        self._status.configure(text=t("cap_recording"), text_color=T.DANGER)
        self._update_rec_timer()

    def _stop_recording(self) -> None:
        if not self._recording:
            return

        with self._rec_lock:
            self._recording = False
            writer = self._writer
            self._writer = None
            saved = self._rec_file

        # Send sentinel so the writer thread drains the queue and exits
        try:
            self._rec_queue.put_nowait(None)
        except queue.Full:
            # Queue full — clear it first, then send sentinel
            while not self._rec_queue.empty():
                try:
                    self._rec_queue.get_nowait()
                except queue.Empty:
                    break
            try:
                self._rec_queue.put_nowait(None)
            except queue.Full:
                pass

        # Wait for writer thread to finish flushing (max 5 s)
        if self._rec_writer_thread and self._rec_writer_thread.is_alive():
            self._rec_writer_thread.join(timeout=5.0)
        self._rec_writer_thread = None

        # Read frame count after join (thread has finished incrementing)
        with self._rec_lock:
            frames = self._rec_frames

        if writer:
            writer.release()

        self._rec_btn.configure(text=t("cap_record"), fg_color=T.DANGER)
        self._rec_lbl.configure(text="")
        if saved and frames > 0:
            self._last_rec_dir = saved.parent
            self._status.configure(
                text=f"✓ {saved.name}\n{saved.parent}",
                text_color=T.ACCENT,
            )
            self._open_folder_btn.pack(fill="x", padx=12, pady=(0, 4))

    def _update_rec_timer(self) -> None:
        """Tick the REC timer label every second while recording."""
        if not self._recording:
            return
        elapsed = int(time.monotonic() - self._rec_start)
        m, s = divmod(elapsed, 60)
        self._rec_lbl.configure(text=f"● REC  {m:02d}:{s:02d}")
        self.after(1000, self._update_rec_timer)

    def _open_rec_folder(self) -> None:
        """Open the folder containing the last saved recording in the OS file manager."""
        folder = self._last_rec_dir
        if not folder or not folder.exists():
            return
        import platform, subprocess
        sys = platform.system()
        try:
            if sys == "Windows":
                import os
                os.startfile(str(folder))
            elif sys == "Darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception:
            pass

    # ── lifecycle (navigation) ────────────────────────────────────────────────
    def on_hide(self) -> None:
        """
        Called by app._navigate() when the user leaves this screen.

        Stops the camera, display loop and prewarm so we don't burn CPU
        while another screen is visible (same approach as ProductionScreen).
        """
        self._stop()
        # Release any camera that finished warming while we weren't watching.
        if self._prewarm_cap is not None:
            self._prewarm_cap.release()
            self._prewarm_cap = None
        self._prewarm_idx = -1
        self._warming = False

    def on_show(self) -> None:
        """
        Called by app._navigate() when the user arrives at this screen.

        Kicks off a background prewarm so the preview auto-starts without
        a manual click — same as the initial load path.
        """
        if self._source_var.get() != "camera":
            return
        label = self._cam_idx.get()
        # "⟳ …" = still enumerating; t("cam_none") = nothing to open
        if label and not label.startswith("⟳") and label != t("cam_none"):
            self._start_prewarm(label_to_index(label))

    def on_close(self) -> None:
        # Remove hotkey bindings so they don't fire on other screens
        top = self.winfo_toplevel()
        for bid in getattr(self, "_hotkey_ids", []):
            try:
                top.unbind("<c>", bid)
                top.unbind("<C>", bid)
            except Exception:
                pass
        self._hotkey_ids = []
        self._stop()
        # Release pre-warmed cap if the screen is torn down before Start is clicked
        if self._prewarm_cap is not None:
            self._prewarm_cap.release()
            self._prewarm_cap = None
