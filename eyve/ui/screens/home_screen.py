"""Home screen — create new project or open existing one."""
from __future__ import annotations
import re
import threading
import time
from pathlib import Path
from tkinter import filedialog
from typing import TYPE_CHECKING

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import cv2

from eyve.ui import theme as T
from eyve.i18n import t
from eyve.core import config
from eyve.core.project_manager import (
    create_project, load_project, validate_name,
)
from eyve.ui.components.dialogs import show_error, show_info
from eyve.camera.release import release_async
from eyve.camera.camera_enum import get_camera_labels, label_to_index
from eyve.ui.screens.capture_screen import _open_camera

if TYPE_CHECKING:
    from eyve.ui.app import EyveApp


class HomeScreen(ctk.CTkFrame):
    def __init__(self, master, app: "EyveApp", **kwargs):
        super().__init__(master, fg_color=T.BG_DARK, **kwargs)
        self._app = app
        self._build()

    def _build(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.grid(row=0, column=0)

        # ── header ────────────────────────────────────────────────────────────
        ctk.CTkLabel(
            outer, text=t("home_welcome"),
            font=T.bold(T.FONT_XXL), text_color=T.ACCENT,
        ).pack(pady=(0, 6))
        ctk.CTkLabel(
            outer, text=t("home_tagline"),
            font=T.font(T.FONT_LG), text_color=T.TEXT_SEC,
        ).pack(pady=(0, 36))

        # ── main action buttons ───────────────────────────────────────────────
        row = ctk.CTkFrame(outer, fg_color="transparent")
        row.pack()
        ctk.CTkButton(
            row, text=t("home_new_project"),
            font=T.bold(T.FONT_MD), height=48, width=200,
            fg_color=T.ACCENT, text_color="#000",
            command=self._new_project,
        ).pack(side="left", padx=8)
        ctk.CTkButton(
            row, text=t("home_open_project"),
            font=T.bold(T.FONT_MD), height=48, width=200,
            fg_color=T.BG_CARD, border_width=1, border_color=T.BORDER,
            command=self._open_project,
        ).pack(side="left", padx=8)

        # ── secondary buttons ─────────────────────────────────────────────────
        row2 = ctk.CTkFrame(outer, fg_color="transparent")
        row2.pack(pady=(12, 0))
        ctk.CTkButton(
            row2, text=t("home_test_camera"),
            font=T.font(T.FONT_SM), height=34, width=140,
            fg_color=T.BG_CARD, border_width=1, border_color=T.BORDER,
            command=self._test_camera,
        ).pack(side="left", padx=6)
        ctk.CTkButton(
            row2, text=t("home_help"),
            font=T.font(T.FONT_SM), height=34, width=140,
            fg_color=T.BG_CARD, border_width=1, border_color=T.BORDER,
            command=self._open_help,
        ).pack(side="left", padx=6)

        # ── language toggle ───────────────────────────────────────────────────
        lang_row = ctk.CTkFrame(outer, fg_color="transparent")
        lang_row.pack(pady=(8, 0))
        ctk.CTkLabel(lang_row, text="🌐", font=T.font(T.FONT_SM),
                     text_color=T.TEXT_DIM).pack(side="left", padx=(0, 4))
        for code, label in [("en", "EN"), ("es", "ES")]:
            ctk.CTkButton(
                lang_row, text=label,
                font=T.font(T.FONT_XS), width=36, height=24,
                fg_color=T.BG_INPUT if code != t("nav_home")[:2] else T.ACCENT,
                command=lambda c=code: self._app.switch_language(c),
            ).pack(side="left", padx=2)

        # ── recent projects ───────────────────────────────────────────────────
        sep = ctk.CTkFrame(outer, height=1, fg_color=T.BORDER)
        sep.pack(fill="x", pady=28)

        ctk.CTkLabel(
            outer, text=t("home_recent"),
            font=T.bold(T.FONT_MD), text_color=T.TEXT_SEC,
        ).pack(anchor="w")

        self._recent_frame = ctk.CTkFrame(outer, fg_color="transparent")
        self._recent_frame.pack(fill="x", pady=(8, 0))
        self._populate_recent()

    def _populate_recent(self) -> None:
        for w in self._recent_frame.winfo_children():
            w.destroy()

        recent = config.get_recent_projects()
        if not recent:
            ctk.CTkLabel(
                self._recent_frame, text=t("home_no_recent"),
                font=T.font(T.FONT_SM), text_color=T.TEXT_DIM,
            ).pack(anchor="w")
            return

        for path_str in recent[:6]:
            p = Path(path_str)
            row = ctk.CTkFrame(self._recent_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkButton(
                row,
                text=f"  {p.name}  —  {str(p.parent)}",
                font=T.font(T.FONT_SM),
                anchor="w",
                fg_color=T.BG_CARD,
                hover_color=T.BG_INPUT,
                height=32,
                command=lambda ps=path_str: self._load_recent(ps),
            ).pack(side="left", fill="x", expand=True)

    # ── actions ───────────────────────────────────────────────────────────────
    def _new_project(self) -> None:
        dialog = _NewProjectDialog(self._app, app=self._app)
        self.wait_window(dialog)

    def _open_project(self) -> None:
        default_dir = Path.home() / "Documents" / "Eyve Projects"
        folder = filedialog.askdirectory(
            title=t("home_open_project"),
            initialdir=str(default_dir) if default_dir.exists() else str(Path.home()))
        if not folder:
            return
        self._load_path(Path(folder))

    def _load_recent(self, path_str: str) -> None:
        self._load_path(Path(path_str))

    def _load_path(self, path: Path) -> None:
        try:
            proj = load_project(path)
            config.add_recent_project(str(path))
            self._app.set_project(proj)
            self._app.navigate("nav_classes")
        except Exception as e:
            show_error(self._app, str(e))

    def _test_camera(self) -> None:
        _CameraTestDialog(self._app)

    def _open_help(self) -> None:
        import webbrowser
        webbrowser.open("https://eyve.app/docs")


# ── New project dialog ────────────────────────────────────────────────────────

class _NewProjectDialog(ctk.CTkToplevel):
    def __init__(self, master, app: "EyveApp", **kwargs):
        super().__init__(master, **kwargs)
        self._app = app
        self.title(t("proj_create_title"))
        self.resizable(False, False)
        self.grab_set()
        self.configure(fg_color=T.BG_CARD)
        self.geometry("480x520")
        self._build()

    def _build(self) -> None:
        pad = {"padx": 28, "pady": 4}

        ctk.CTkLabel(self, text=t("proj_create_title"),
                     font=T.bold(T.FONT_LG), text_color=T.ACCENT).pack(pady=(20, 12))

        # project name
        ctk.CTkLabel(self, text=t("proj_name"),
                     font=T.font(T.FONT_SM), text_color=T.TEXT_SEC, anchor="w").pack(fill="x", **pad)
        self._name = ctk.CTkEntry(self, placeholder_text=t("proj_name_hint"),
                                   fg_color=T.BG_INPUT, border_color=T.BORDER)
        self._name.pack(fill="x", **pad)

        # target
        ctk.CTkLabel(self, text=t("proj_target"),
                     font=T.font(T.FONT_SM), text_color=T.TEXT_SEC, anchor="w").pack(fill="x", **pad)
        self._target = ctk.CTkEntry(self, placeholder_text=t("proj_target_hint"),
                                     fg_color=T.BG_INPUT, border_color=T.BORDER)
        self._target.pack(fill="x", **pad)

        # camera source — async populated
        ctk.CTkLabel(self, text=t("proj_camera"),
                     font=T.font(T.FONT_SM), text_color=T.TEXT_SEC, anchor="w").pack(fill="x", **pad)
        self._camera = ctk.CTkOptionMenu(
            self, values=["⟳  Loading cameras…"],
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD,
            font=("Segoe UI", 11),
            state="disabled",
        )
        self._camera.pack(fill="x", **pad)
        threading.Thread(target=self._load_cameras_bg, daemon=True).start()

        # folder
        ctk.CTkLabel(self, text=t("proj_folder"),
                     font=T.font(T.FONT_SM), text_color=T.TEXT_SEC, anchor="w").pack(fill="x", **pad)
        folder_row = ctk.CTkFrame(self, fg_color="transparent")
        folder_row.pack(fill="x", **pad)
        self._folder = ctk.CTkEntry(folder_row, fg_color=T.BG_INPUT, border_color=T.BORDER)
        self._folder.pack(side="left", fill="x", expand=True, padx=(0, 6))
        # default to Documents/Eyve Projects
        default = Path.home() / "Documents" / "Eyve Projects"
        self._folder.insert(0, str(default))
        ctk.CTkButton(folder_row, text=t("proj_folder_browse"), width=80,
                      fg_color=T.BG_INPUT, border_width=1, border_color=T.BORDER,
                      command=self._browse).pack(side="left")

        # error label
        self._err = ctk.CTkLabel(self, text="", font=T.font(T.FONT_SM), text_color=T.DANGER)
        self._err.pack(pady=(4, 0))

        # buttons
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(pady=(8, 20))
        ctk.CTkButton(btn_row, text=t("proj_create_btn"),
                      fg_color=T.ACCENT, text_color="#000", width=160,
                      command=self._create).pack(side="left", padx=6)
        ctk.CTkButton(btn_row, text=t("proj_cancel"),
                      fg_color=T.BG_INPUT, width=100,
                      command=self.destroy).pack(side="left", padx=6)

    def _load_cameras_bg(self) -> None:
        labels = get_camera_labels()
        self.after(0, lambda: (
            self._camera.configure(
                values=labels if labels else ["No cameras found"],
                state="normal" if labels else "disabled"),
            self._camera.set((labels if labels else ["No cameras found"])[0])
        ))

    def _browse(self) -> None:
        current = Path(self._folder.get().strip() or Path.home())
        folder = filedialog.askdirectory(
            initialdir=str(current) if current.exists() else str(Path.home()))
        if folder:
            self._folder.delete(0, "end")
            self._folder.insert(0, folder)

    def _create(self) -> None:
        name = self._name.get().strip()
        folder = self._folder.get().strip()
        target = self._target.get().strip()

        err = validate_name(name)
        if err:
            self._err.configure(text=t(err))
            return
        if not folder:
            self._err.configure(text=t("proj_folder_required"))
            return

        try:
            camera = label_to_index(self._camera.get())
            proj = create_project(name, Path(folder), target=target, camera_source=camera)
            config.add_recent_project(str(proj.root))
            self._app.set_project(proj)
            self.destroy()
            self._app.navigate("nav_classes")
        except FileExistsError:
            self._err.configure(text=t("proj_exists"))
        except Exception as e:
            self._err.configure(text=str(e))


# ── Camera test dialog ────────────────────────────────────────────────────────

class _CameraTestDialog(ctk.CTkToplevel):
    """Quick camera test — same producer/consumer architecture as CaptureScreen."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.title(t("home_test_camera"))
        self.geometry("640x540")
        self.grab_set()
        self.configure(fg_color=T.BG_CARD)

        self._cap = None
        self._running = False
        self._thread = None
        self._photo = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._prewarm_cap = None
        self._prewarm_idx = -1
        self._prewarm_pending = False

        ctk.CTkLabel(self, text=t("cap_camera_id"),
                     font=T.font(T.FONT_SM), text_color=T.TEXT_SEC).pack(pady=(12, 4))

        cam_row = ctk.CTkFrame(self, fg_color="transparent")
        cam_row.pack(fill="x", padx=20)
        self._cam_idx = ctk.CTkOptionMenu(
            cam_row, values=["⟳  Loading cameras…"],
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD,
            font=("Segoe UI", 11),
            state="disabled",
        )
        self._cam_idx.pack(side="left", padx=6, fill="x", expand=True)
        ctk.CTkButton(cam_row, text="⟳", width=32, height=28,
                      fg_color=T.BG_INPUT,
                      command=self._refresh_cam_test).pack(side="left")
        threading.Thread(target=self._load_cams_bg, daemon=True).start()
        ctk.CTkButton(cam_row, text=t("cap_start"),
                      fg_color=T.ACCENT, text_color="#000",
                      command=self._start).pack(side="left", padx=6)
        ctk.CTkButton(cam_row, text=t("cap_stop"),
                      fg_color=T.BG_INPUT, command=self._stop).pack(side="left", padx=6)

        self._canvas = tk.Canvas(self, bg="#000", width=580, height=380)
        self._canvas.pack(pady=10)

        self._status = ctk.CTkLabel(self, text="", font=T.font(T.FONT_SM),
                                     text_color=T.TEXT_SEC)
        self._status.pack()
        self._fps_lbl = ctk.CTkLabel(self, text="", font=T.font(T.FONT_XS),
                                      text_color=T.TEXT_DIM)
        self._fps_lbl.pack()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_cams_bg(self) -> None:
        labels = get_camera_labels()
        self.after(0, lambda: self._on_cams_loaded(labels))

    def _on_cams_loaded(self, labels: list) -> None:
        labels = labels if labels else ["No cameras found"]
        self._cam_idx.configure(
            values=labels,
            state="normal" if labels[0] != "No cameras found" else "disabled")
        self._cam_idx.set(labels[0])
        # Pre-warm the first camera so Start is instant
        if labels[0] != "No cameras found":
            self._start_prewarm(label_to_index(labels[0]))

    def _start_prewarm(self, idx: int) -> None:
        if self._prewarm_idx == idx:
            return  # already warming or warmed — don't open a second time
        self._prewarm_pending = False
        if self._prewarm_cap is not None:
            release_async(self._prewarm_cap)
            self._prewarm_cap = None
        self._prewarm_idx = idx

        def _bg() -> None:
            cap = _open_camera(idx)
            self.after(0, lambda: self._on_prewarm_ready(idx, cap))

        threading.Thread(target=_bg, daemon=True).start()

    def _on_prewarm_ready(self, idx: int, cap: cv2.VideoCapture) -> None:
        if self._prewarm_idx != idx:
            release_async(cap)
            return
        if self._prewarm_pending:
            self._prewarm_pending = False
            self._prewarm_idx = -1
            self._on_camera_ready(cap)
        elif not self._running:
            self._prewarm_cap = cap
        else:
            release_async(cap)

    def _refresh_cam_test(self) -> None:
        self._cam_idx.configure(values=["⟳  Refreshing…"], state="disabled")
        threading.Thread(target=self._load_cams_bg, daemon=True).start()

    def _start(self) -> None:
        self._stop()
        idx = label_to_index(self._cam_idx.get())

        if self._prewarm_cap is not None and self._prewarm_idx == idx:
            # Camera already open — instant
            cap = self._prewarm_cap
            self._prewarm_cap = None
            self._prewarm_idx = -1
            self._on_camera_ready(cap)
            return
        if self._prewarm_idx == idx:
            # Still warming — hook into it
            self._status.configure(text="⏳ " + t("cap_opening_cam"), text_color=T.TEXT_SEC)
            self._prewarm_pending = True
            return

        self._status.configure(text="⏳ " + t("cap_opening_cam"), text_color=T.TEXT_SEC)

        def _open_bg() -> None:
            cap = _open_camera(idx)
            self.after(0, lambda: self._on_camera_ready(cap))

        threading.Thread(target=_open_bg, daemon=True).start()

    def _on_camera_ready(self, cap: cv2.VideoCapture) -> None:
        if not cap.isOpened():
            release_async(cap)
            self._status.configure(text=t("cap_no_camera"), text_color=T.DANGER)
            return
        self._cap = cap
        self._running = True
        self._fps_counter = 0
        self._fps_ts = time.monotonic()
        self._status.configure(text=t("status_camera_ok"), text_color=T.ACCENT)
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        self._display_loop()

    def _grab_loop(self) -> None:
        # This thread OWNS the capture: released here, never on Tk's thread
        # (DSHOW CoUninitialize on main breaks file dialogs — camera/release.py)
        cap = self._cap
        if cap is None:
            return
        try:
            while self._running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                with self._frame_lock:
                    self._latest_frame = frame
        finally:
            try:
                cap.release()
            except Exception:
                pass

    def _display_loop(self) -> None:
        if not self._running:
            return
        frame = None
        with self._frame_lock:
            if self._latest_frame is not None:
                frame = self._latest_frame
                self._latest_frame = None
        if frame is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                img.thumbnail((580, 380), Image.NEAREST)
                photo = ImageTk.PhotoImage(img)
                self._photo = photo
                self._canvas.create_image(290, 190, image=photo, anchor="center")
                self._fps_counter += 1
                now = time.monotonic()
                if now - self._fps_ts >= 1.0:
                    fps = self._fps_counter / (now - self._fps_ts)
                    self._fps_lbl.configure(text=f"FPS: {fps:.0f}")
                    self._fps_counter = 0
                    self._fps_ts = now
            except Exception:
                pass
        self.after(16, self._display_loop)

    def _stop(self) -> None:
        self._running = False
        cap, self._cap = self._cap, None
        if cap is not None and not (self._thread and self._thread.is_alive()):
            release_async(cap)
        self._fps_lbl.configure(text="")

    def _on_close(self) -> None:
        self._stop()
        if self._prewarm_cap is not None:
            release_async(self._prewarm_cap)
            self._prewarm_cap = None
        self.destroy()
