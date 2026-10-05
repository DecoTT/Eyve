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
        self.title(t("app_title"))

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

        # Version nueva: se consulta una vez al dia, en segundo plano. El
        # arranque no espera a la red — una planta con el proxy caido no
        # puede quedarse mirando un splash.
        self.after(2500, self.check_for_updates)

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

    # ── actualizaciones ──────────────────────────────────────────────────────
    def check_for_updates(self, force: bool = False) -> None:
        """
        Consulta si hay version nueva y lo avisa en la barra de estado.
        force=True ignora el intervalo (lo usa el boton de Settings).
        """
        from eyve.core import updater as U
        if not force and not U.should_check():
            return

        def done(info) -> None:
            try:
                self.after(0, lambda: self._on_update_info(info))
            except Exception:
                pass

        U.check_async(done)

    def _on_update_info(self, info) -> None:
        from eyve.ui.screens.update_dialog import offer_update
        offer_update(self, info)

    def open_update_dialog(self) -> None:
        from eyve.ui.screens.update_dialog import UpdateDialog
        UpdateDialog(self)

    def _make_screen(self, key: str) -> ctk.CTkFrame:
        from eyve.ui.screens.home_screen import HomeScreen
        from eyve.ui.screens.classes_screen import ClassesScreen
        from eyve.ui.screens.capture_screen import CaptureScreen
        from eyve.ui.screens.tagging_screen import TaggingScreen
        from eyve.ui.screens.training_screen import TrainingScreen
        from eyve.ui.screens.production_screen import ProductionScreen
        from eyve.ui.screens.demo_screen import DemoScreen
        from eyve.ui.screens.counting_demo_screen import CountingDemoScreen

        map_ = {
            "nav_home":       HomeScreen,
            "nav_classes":    ClassesScreen,
            "nav_capture":    CaptureScreen,
            "nav_tagging":    TaggingScreen,
            "nav_training":   TrainingScreen,
            "nav_production": ProductionScreen,
            "nav_demo":       DemoScreen,
            "nav_cdemo":      CountingDemoScreen,
        }
        cls = map_.get(key)
        if cls is None:
            raise ValueError(f"Unknown screen key: {key}")
        return cls(self.content, app=self)

    # ── screen teardown ──────────────────────────────────────────────────────
    def _teardown_screen(self, screen) -> None:
        """
        Destroy a cached screen PROPERLY.

        A bare destroy() leaves the screen's toplevel hotkey bindings alive
        (they were registered with bind(add=True) on the root) pointing at
        dead widgets.  Seen in the field after re-opening a project: every
        keypress ran the STALE instance's handler too.  That is worse than
        the TclError noise it produces — the stale handler executes real
        logic with obsolete state (e.g. _save_labels() against its old
        _idx/_boxes, overwriting someone else's label file) before it dies
        touching a destroyed widget.
        on_close() unbinds hotkeys and releases cameras/threads first.
        """
        try:
            if hasattr(screen, "on_hide") and screen.winfo_ismapped():
                screen.on_hide()
        except Exception:
            log.debug("on_hide failed during teardown", exc_info=True)
        try:
            if hasattr(screen, "on_close"):
                screen.on_close()
        except Exception:
            log.debug("on_close failed during teardown", exc_info=True)
        screen.destroy()

    def _teardown_all_screens(self) -> None:
        for s in list(self._screens.values()):
            self._teardown_screen(s)
        self._screens.clear()

    # ── project helpers ──────────────────────────────────────────────────────
    def set_project(self, project: Project) -> None:
        self._project = project
        self.status_bar.set_project(project.name)
        # invalidate cached screens that depend on project state
        for key in ("nav_classes", "nav_capture", "nav_tagging",
                    "nav_training", "nav_production"):
            if key in self._screens:
                self._teardown_screen(self._screens[key])
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
        self._teardown_all_screens()
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
        self._teardown_all_screens()
        self._navigate("nav_home")

    # ── license ──────────────────────────────────────────────────────────────
    # Por honor: nada de esto bloquea. Se lee la llave sin red, se pinta el
    # nivel en la barra y, si toca (activación pendiente o +7 días), se hace
    # el check-in en un hilo sin esperar la respuesta.
    def _check_license(self) -> None:
        from eyve.license.license_manager import LicenseManager
        self.license = LicenseManager()
        self._paint_license()
        st = self.license.status()
        if st in ("vencida", "invalida") and not getattr(self, "_lic_warned", False):
            self._lic_warned = True          # una vez por arranque
            self.after(1500, self._show_license_dialog)
        if not getattr(self, "_lic_checked", False):
            self._lic_checked = True
            self.license.comprobar_en_hilo(
                lambda srv, err: self.after(0, self._paint_license), solo_si_toca=True)

    def _paint_license(self) -> None:
        lm = self.license
        text, warn = lm.resumen(), False
        if lm.status() == "invalida":
            text, warn = t("lic_invalid_stored"), True
        elif lm.status() == "vencida" or lm.servidor.motivo == "sin_cupo":
            warn = True
        self.status_bar.set_license(text, warn=warn)

    def _show_license_dialog(self) -> None:
        from eyve.ui.screens.license_dialog import LicenseDialog
        LicenseDialog(self, lm=self.license, on_change=self._paint_license)

    # ── close ────────────────────────────────────────────────────────────────
    def _on_close(self) -> None:
        config.set("window_width", self.winfo_width())
        config.set("window_height", self.winfo_height())
        # stop any running camera/workers
        for screen in self._screens.values():
            if hasattr(screen, "on_close"):
                screen.on_close()
        self.destroy()
