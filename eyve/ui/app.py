"""
Main application window.

Hosts the sidebar + a content area where screens are swapped.
Screens are created lazily on first navigation.
"""
from __future__ import annotations
from typing import Optional
import customtkinter as ctk

from eyve.ui import theme as T
from eyve.ui.components.sidebar import Sidebar
from eyve.ui.components.status_bar import StatusBar
from eyve.i18n import t, set_language, get_language
from eyve.core import config
from eyve.core.project_manager import Project
from eyve.core.logger import log


class EyveApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        # apply language + theme from saved config
        lang = config.get("language", "en")
        set_language(lang)

        mode = config.get("theme", "dark")
        T.apply(mode)

        # apply offline mode env var (sets/clears YOLO_OFFLINE)
        from eyve.core.model_manager import apply_offline_env
        apply_offline_env()
        self.configure(fg_color=T.BG_DARK)
        # "[dev]" marks the development copy — the frozen 2.1 backup shows
        # the plain title, so it's obvious at a glance which build is running
        self.title(t("app_title") + "  [dev]")

        w = config.get("window_width", 1280)
        h = config.get("window_height", 880)   # +10 % vs 800
        self.geometry(f"{w}x{h}")
        self.minsize(960, 640)

        self._project: Optional[Project] = None
        self._screens: dict[str, ctk.CTkFrame] = {}

        self._build_layout()
        self._navigate("nav_home")
        self._check_license()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── layout ───────────────────────────────────────────────────────────────
    def _build_layout(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        self.sidebar = Sidebar(self, on_navigate=self._navigate)
        self.sidebar.set_app(self)
        self.sidebar.grid(row=0, column=0, sticky="nsw")

        self.content = ctk.CTkFrame(self, fg_color=T.BG_DARK, corner_radius=0)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self.status_bar = StatusBar(self)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

    # ── navigation ───────────────────────────────────────────────────────────
    def _navigate(self, key: str) -> None:
        # Pause every currently-visible screen before hiding it.
        # grid_remove() only hides the widget — without on_hide() all camera
        # grab loops, display pollers and production inference loops keep
        # firing in the background, which is the cause of high idle CPU.
        for screen in self._screens.values():
            if screen.winfo_ismapped():
                if hasattr(screen, "on_hide"):
                    screen.on_hide()
                screen.grid_remove()

        # Lazy-create on first visit
        if key not in self._screens:
            self._screens[key] = self._make_screen(key)

        screen = self._screens[key]
        screen.grid(row=0, column=0, sticky="nsew")
        # Resume the screen (restart camera / loops as needed)
        if hasattr(screen, "on_show"):
            screen.on_show()
        self.sidebar.set_active(key)
        log.debug(f"Navigate → {key}")

    def _make_screen(self, key: str) -> ctk.CTkFrame:
        from eyve.ui.screens.home_screen import HomeScreen
        from eyve.ui.screens.classes_screen import ClassesScreen
        from eyve.ui.screens.capture_screen import CaptureScreen
        from eyve.ui.screens.tagging_screen import TaggingScreen
        from eyve.ui.screens.training_screen import TrainingScreen
        from eyve.ui.screens.production_screen import ProductionScreen

        map_ = {
            "nav_home":       HomeScreen,
            "nav_classes":    ClassesScreen,
            "nav_capture":    CaptureScreen,
            "nav_tagging":    TaggingScreen,
            "nav_training":   TrainingScreen,
            "nav_production": ProductionScreen,
        }
        cls = map_.get(key)
        if cls is None:
            raise ValueError(f"Unknown screen key: {key}")
        return cls(self.content, app=self)

    # ── project helpers ──────────────────────────────────────────────────────
    def set_project(self, project: Project) -> None:
        self._project = project
        self.status_bar.set_project(project.name)
        # invalidate cached screens that depend on project state
        for key in ("nav_classes", "nav_capture", "nav_tagging",
                    "nav_training", "nav_production"):
            if key in self._screens:
                self._screens[key].destroy()
                del self._screens[key]
        log.info(f"App: project set to {project.name}")

    def get_project(self) -> Optional[Project]:
        return self._project

    def navigate(self, key: str) -> None:
        self._navigate(key)

    # ── language ─────────────────────────────────────────────────────────────
    def switch_language(self, lang: str) -> None:
        set_language(lang)
        config.set("language", lang)
        self.sidebar.refresh_labels()
        for s in self._screens.values():
            s.destroy()
        self._screens.clear()
        self._navigate("nav_home")

    # ── theme ─────────────────────────────────────────────────────────────────
    def switch_theme(self, mode: str) -> None:
        T.set_mode(mode)
        config.set("theme", mode)
        # update persistent containers
        self.configure(fg_color=T.BG_DARK)
        self.content.configure(fg_color=T.BG_DARK)
        # rebuild sidebar and status bar with new palette
        self.sidebar.destroy()
        self.status_bar.destroy()
        self.sidebar = Sidebar(self, on_navigate=self._navigate)
        self.sidebar.set_app(self)
        self.sidebar.grid(row=0, column=0, sticky="nsw")
        self.status_bar = StatusBar(self)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        if self._project:
            self.status_bar.set_project(self._project.name)
        self._check_license()
        # rebuild screens
        for s in self._screens.values():
            s.destroy()
        self._screens.clear()
        self._navigate("nav_home")

    # ── license ──────────────────────────────────────────────────────────────
    def _check_license(self) -> None:
        from eyve.license.license_manager import LicenseManager
        lm = LicenseManager()
        status = lm.status()
        if status == "trial":
            days = lm.days_remaining()
            self.status_bar.set_license(t("lic_trial_active", days=days))
        elif status == "expired":
            self.status_bar.set_license(t("lic_watermark"), warn=True)
            self.after(1500, self._show_license_reminder)
        # "activated" → no badge

    def _show_license_reminder(self) -> None:
        from eyve.ui.screens.license_dialog import LicenseDialog
        LicenseDialog(self)

    # ── close ────────────────────────────────────────────────────────────────
    def _on_close(self) -> None:
        config.set("window_width", self.winfo_width())
        config.set("window_height", self.winfo_height())
        # stop any running camera/workers
        for screen in self._screens.values():
            if hasattr(screen, "on_close"):
                screen.on_close()
        self.destroy()
