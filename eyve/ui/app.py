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

#: Pantallas que pueden ir a pantalla completa.  Solo las dos de demo: en
#: las demas, esconder la navegacion deja al usuario encerrado sin forma
#: obvia de volver, que es exactamente el fallo que el kiosco evita en el
#: stand.
KIOSK_SCREENS = ("nav_demo", "nav_cdemo")


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
        #: pantalla visible ahora; el kiosco necesita saberlo para no
        #: dejarse activar donde no toca
        self._current: Optional[str] = None

        # ── kiosco ───────────────────────────────────────────────────────
        self._kiosk = False
        #: tamano Y posicion de antes de entrar, para devolver la ventana
        #: donde estaba al salir
        self._kiosk_geom: Optional[str] = None
        self._kiosk_zoomed = False
        self._kiosk_toast: Optional[ctk.CTkFrame] = None
        self._kiosk_toast_job: Optional[str] = None

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

        # Kiosco: F11 entra y sale, Esc solo sale.  Con add=True a
        # proposito: las pantallas tambien enlazan en la raiz (ver
        # eyve/ui/hotkeys.py) y un bind sin add les borraria el suyo —
        # Etiquetado perderia su Escape.
        self.bind("<F11>", self._on_f11, add=True)
        self.bind("<Escape>", self._on_escape, add=True)

        # Version nueva: se consulta una vez al dia, en segundo plano. El
        # arranque no espera a la red — una planta con el proxy caido no
        # puede quedarse mirando un splash.
        self.after(2500, self.check_for_updates)

    # ── navigation ───────────────────────────────────────────────────────────
    def _navigate(self, key: str, force: bool = False) -> None:
        # En kiosco no se sale de la demo con el teclado.  Es el fallo mas
        # probable del stand: un visitante acaba en "Etiquetado" y no sabe
        # volver.  force=True es para quien apaga el kiosco a proposito
        # (cambio de idioma o de tema), que reconstruye todo.
        if self._kiosk and not force and key != self._current:
            log.debug(f"Kiosco activo: navegacion a {key} ignorada")
            return

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
        self._current = key
        log.debug(f"Navigate → {key}")

    # ── kiosco ───────────────────────────────────────────────────────────────
    # Pantalla completa sin barra lateral ni barra de estado, para el stand.
    # No es solo cosmetico: con la navegacion a la vista, un visitante se
    # mete en "Etiquetado", no sabe volver y la demo se queda muerta hasta
    # que alguien del stand se da cuenta.

    @property
    def kiosk(self) -> bool:
        return self._kiosk

    def kiosk_available(self) -> bool:
        """Si la pantalla visible admite pantalla completa."""
        return self._current in KIOSK_SCREENS

    def _on_f11(self, _event=None) -> None:
        self.toggle_kiosk()

    def _on_escape(self, _event=None) -> None:
        # Esc NO entra, solo sale.  Etiquetado usa Esc para lo suyo y su
        # handler esta guardado por pantalla visible (guard_hotkey), asi
        # que basta con no hacer nada cuando el kiosco esta apagado.
        if self._kiosk:
            self.exit_kiosk()

    def toggle_kiosk(self) -> None:
        if self._kiosk:
            self.exit_kiosk()
        else:
            self.enter_kiosk()

    def enter_kiosk(self) -> None:
        if self._kiosk or not self.kiosk_available():
            return
        # Tamano Y posicion: al salir la ventana vuelve donde estaba, no
        # solo del tamano que tenia.
        #
        # Se guarda con geometry() y no con winfo_*: el geometry() de
        # CustomTkinter divide por el escalado de DPI al leer y multiplica
        # al escribir, asi que leer y escribir por ahi es exacto.  Mezclar
        # winfo_* (pixeles de verdad) con el setter escalado devolveria la
        # ventana con el tamano multiplicado por la escala del monitor.
        self._kiosk_zoomed = self.state() == "zoomed"
        self._kiosk_geom = self.geometry()
        # grid_remove (no grid_forget) conserva la configuracion de fila y
        # columna, asi que al volver se restaura en el mismo sitio.  La
        # columna de la barra lateral tiene weight=0 y sin minsize, de modo
        # que al quedarse vacia se encoge a cero y no deja franja.
        self.sidebar.grid_remove()
        self.status_bar.grid_remove()
        self.attributes("-fullscreen", True)
        self._kiosk = True
        self._show_kiosk_toast()
        self._notify_kiosk()
        log.info("Kiosco: pantalla completa")

    def exit_kiosk(self) -> None:
        if not self._kiosk:
            return
        self._kiosk = False
        self._hide_kiosk_toast()
        self.attributes("-fullscreen", False)
        self.sidebar.grid(row=0, column=0, sticky="nsw")
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        if self._kiosk_zoomed:
            self.state("zoomed")
        elif self._kiosk_geom:
            self.geometry(self._kiosk_geom)
        self._notify_kiosk()
        log.info("Kiosco: ventana normal")

    def _notify_kiosk(self) -> None:
        """
        Avisar a la pantalla visible de que el kiosco cambio, para que
        ajuste su pista.  El aviso grande se desvanece a los 6 s y quien
        llega al stand mas tarde no vio nada: la pista de la esquina
        tiene que decir en todo momento como se sale.
        """
        screen = self._screens.get(self._current or "")
        fn = getattr(screen, "on_kiosk", None)
        if callable(fn):
            try:
                fn(self._kiosk)
            except Exception:
                log.debug("on_kiosk fallo", exc_info=True)

    def _show_kiosk_toast(self) -> None:
        """
        Pista de como salir, encima de la demo y unos segundos.

        Se muestra CADA vez que se entra y no solo la primera: en un stand
        quien atiende cambia a lo largo del dia y no es necesariamente
        quien puso la pantalla completa.  Va con place() y no con grid()
        para no tocar el layout que el kiosco acaba de dejar limpio.
        """
        self._hide_kiosk_toast()
        box = ctk.CTkFrame(self, fg_color=T.BG_CARD, corner_radius=10)
        ctk.CTkLabel(box, text=t("kiosk_exit_hint"), font=T.bold(T.FONT_SM),
                     text_color=T.TEXT_PRI).pack(padx=18, pady=10)
        box.place(relx=0.5, y=16, anchor="n")
        self._kiosk_toast = box
        self._kiosk_toast_job = self.after(6000, self._hide_kiosk_toast)

    def _hide_kiosk_toast(self) -> None:
        if self._kiosk_toast_job is not None:
            try:
                self.after_cancel(self._kiosk_toast_job)
            except Exception:
                pass
            self._kiosk_toast_job = None
        if self._kiosk_toast is not None:
            try:
                self._kiosk_toast.destroy()
            except Exception:
                pass
            self._kiosk_toast = None

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
        # Fuera del kiosco primero: esto tira todas las pantallas y vuelve
        # a Inicio, que en pantalla completa y sin navegacion seria una
        # ventana sin salida.
        self.exit_kiosk()
        set_language(lang)
        config.set("language", lang)
        self.sidebar.refresh_labels()
        self._teardown_all_screens()
        self._navigate("nav_home", force=True)

    # ── theme ─────────────────────────────────────────────────────────────────
    def switch_theme(self, mode: str) -> None:
        # Antes de nada: aqui se destruyen y se recrean la barra lateral y
        # la de estado, y el kiosco las tiene fuera del grid.
        self.exit_kiosk()
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
        self._navigate("nav_home", force=True)

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
        # Si se cierra estando en kiosco, winfo_width() es el monitor
        # entero: guardarlo dejaria la ventana del proximo arranque del
        # tamano de la pantalla.  Se guarda lo de antes de entrar.
        if self._kiosk and self._kiosk_geom:
            tam = self._kiosk_geom.split("+")[0].split("x")
            config.set("window_width", int(tam[0]))
            config.set("window_height", int(tam[1]))
        else:
            config.set("window_width", self.winfo_width())
            config.set("window_height", self.winfo_height())
        # stop any running camera/workers
        for screen in self._screens.values():
            if hasattr(screen, "on_close"):
                screen.on_close()
        self.destroy()
