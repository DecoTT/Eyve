"""
Pantalla de demo para stand de expo.

La pantalla se parte en dos:

    IZQUIERDA — Eyve corriendo de verdad.  El mismo YOLOWorker, el mismo
                InstanceTracker, el mismo CountingModule y la misma lógica
                OK/NOT OK que la pantalla de Producción.  Lo único distinto
                es de dónde salen los frames.

    DERECHA   — el lienzo del visitante: la misma tela, sin anotaciones,
                donde dibuja defectos con el dedo o el mouse.  Lo que pinta
                entra en la tela, viaja con ella hacia el encuadre de Eyve,
                y Eyve lo encuentra.

Por qué no se reusa ProductionScreen entera: en un stand la gente mira de
lejos y toca la pantalla.  Lo que se necesita es estado grande, contador
grande y tres botones gordos — no el panel de 40 controles de producción.
El motor sí se reusa; la cáscara no.

"Resetear": cuando la tela vuelve a estar limpia (el visitante borra, o el
defecto sale del encuadre), Eyve vuelve a OK solo.  El botón Limpiar deja
la tela como nueva y pone los contadores en cero.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import customtkinter as ctk
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk

from eyve.core import config as _cfg
from eyve.core.logger import log
from eyve.demo.source import SyntheticSource
from eyve.demo.textile import DEFECT_CLASSES, MOTIFS, TextilePattern
from eyve.i18n import t
from eyve.inference.detector import YOLOWorker
from eyve.inference.tracker import InstanceTracker, RawDetection, TrackerConfig
from eyve.modules import CountingModule
from eyve.production.ok_nok_logic import InspectionStatus, decide
from eyve.ui import theme as T

if TYPE_CHECKING:
    from eyve.ui.app import EyveApp

#: herramientas del lienzo → clase de defecto, color del botón
_TOOLS = [
    ("rayon",           "#2b2a2e"),
    ("mancha",          "#5c4a96"),
    ("falta_impresion", "#c8cdd4"),
]

_STATUS_COLORS = {
    InspectionStatus.OK:        T.COLOR_OK,
    InspectionStatus.NOT_OK:    T.COLOR_NOK,
    InspectionStatus.NO_DETECT: T.COLOR_OK,     # tela limpia = bien, en la demo
    InspectionStatus.REVIEW:    T.COLOR_REVIEW,
    InspectionStatus.ERROR:     T.COLOR_ERROR,
}

#: tamaño de la tela que ve "la cámara"
_W, _H = 960, 540


class DemoScreen(ctk.CTkFrame):
    def __init__(self, master, app: "EyveApp", **kwargs):
        super().__init__(master, fg_color=T.BG_DARK, **kwargs)
        self._app = app
        self._running = False
        self._worker: Optional[YOLOWorker] = None
        self._worker_loading = False
        self._photo_eyve = None
        self._photo_canvas = None

        self._pattern = TextilePattern(width=_W, height=_H, axis="x",
                                       motif="diamantes", speed=110.0)
        self._source = SyntheticSource(self._pattern, fps=30.0)

        self._tracker = InstanceTracker(config=TrackerConfig(
            iou_match=0.20, max_lost_frames=18, min_confirm_frames=2,
            process_conf_min=0.30))
        self._counting = CountingModule()
        self._counting.enabled = True
        self._counting.set_method("appear")

        self._tool = "rayon"
        self._drawing = False
        self._last_pt: Optional[tuple[int, int]] = None
        self._view: Optional[tuple] = None      # mapeo lienzo↔frame
        self._nok_total = 0
        self._last_status: Optional[InspectionStatus] = None
        self._per_class: dict[str, int] = {}

        self._build()

    # ── layout ────────────────────────────────────────────────────────────
    def _build(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(self, fg_color=T.BG_CARD, corner_radius=0)
        hdr.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(hdr, text=t("demo_title"), font=T.bold(T.FONT_XL),
                     text_color=T.TEXT_PRI).pack(side="left", padx=24, pady=12)
        self._hint = ctk.CTkLabel(hdr, text=t("demo_hint"),
                                  font=T.font(T.FONT_SM), text_color=T.TEXT_SEC)
        self._hint.pack(side="left", padx=8)
        self._fps_lbl = ctk.CTkLabel(hdr, text="", font=T.font(T.FONT_XS),
                                     text_color=T.TEXT_DIM)
        self._fps_lbl.pack(side="right", padx=24)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1, uniform="half")
        body.grid_columnconfigure(1, weight=1, uniform="half")

        # ── izquierda: Eyve ───────────────────────────────────────────────
        left = ctk.CTkFrame(body, fg_color=T.BG_CARD, corner_radius=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        lh = ctk.CTkFrame(left, fg_color="transparent")
        lh.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        ctk.CTkLabel(lh, text=t("demo_side_eyve"), font=T.bold(T.FONT_MD),
                     text_color=T.ACCENT).pack(side="left")
        self._status_lbl = ctk.CTkLabel(lh, text="—", font=T.bold(T.FONT_XXL),
                                        text_color=T.TEXT_DIM)
        self._status_lbl.pack(side="right")

        self._eyve_canvas = tk.Canvas(left, bg="#0a0a0a", highlightthickness=0)
        self._eyve_canvas.grid(row=1, column=0, sticky="nsew", padx=12)

        lf = ctk.CTkFrame(left, fg_color="transparent")
        lf.grid(row=2, column=0, sticky="ew", padx=12, pady=(6, 12))
        self._count_lbl = ctk.CTkLabel(lf, text=t("demo_found", n=0),
                                       font=T.bold(T.FONT_LG),
                                       text_color=T.COLOR_NOK)
        self._count_lbl.pack(side="left")
        self._detail_lbl = ctk.CTkLabel(lf, text="", font=T.font(T.FONT_SM),
                                        text_color=T.TEXT_SEC)
        self._detail_lbl.pack(side="left", padx=12)

        # ── derecha: el lienzo del visitante ──────────────────────────────
        right = ctk.CTkFrame(body, fg_color=T.BG_CARD, corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        rh = ctk.CTkFrame(right, fg_color="transparent")
        rh.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        ctk.CTkLabel(rh, text=t("demo_side_you"), font=T.bold(T.FONT_MD),
                     text_color=T.ACCENT2).pack(side="left")

        self._draw_canvas = tk.Canvas(right, bg="#0a0a0a", highlightthickness=0,
                                      cursor="pencil")
        self._draw_canvas.grid(row=1, column=0, sticky="nsew", padx=12)
        self._draw_canvas.bind("<ButtonPress-1>",   self._on_press)
        self._draw_canvas.bind("<B1-Motion>",       self._on_drag)
        self._draw_canvas.bind("<ButtonRelease-1>", self._on_release)

        tools = ctk.CTkFrame(right, fg_color="transparent")
        tools.grid(row=2, column=0, sticky="ew", padx=12, pady=(6, 12))
        self._tool_btns: dict[str, ctk.CTkButton] = {}
        for cls, color in _TOOLS:
            b = ctk.CTkButton(tools, text=t("demo_tool_" + cls), height=44,
                              font=T.bold(T.FONT_SM), corner_radius=10,
                              fg_color=color,
                              text_color="#fff" if cls != "falta_impresion" else "#111",
                              command=lambda c=cls: self._set_tool(c))
            b.pack(side="left", expand=True, fill="x", padx=3)
            self._tool_btns[cls] = b

        ctrl = ctk.CTkFrame(right, fg_color="transparent")
        ctrl.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 12))
        ctk.CTkButton(ctrl, text=t("demo_clear"), height=38, corner_radius=10,
                      font=T.bold(T.FONT_SM), fg_color=T.BG_INPUT,
                      text_color=T.TEXT_PRI, command=self.reset_demo).pack(
            side="left", expand=True, fill="x", padx=3)
        self._pause_btn = ctk.CTkButton(
            ctrl, text=t("demo_pause"), height=38, corner_radius=10,
            font=T.bold(T.FONT_SM), fg_color=T.BG_INPUT, text_color=T.TEXT_PRI,
            command=self._toggle_pause)
        self._pause_btn.pack(side="left", expand=True, fill="x", padx=3)

        spd = ctk.CTkFrame(right, fg_color="transparent")
        spd.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 10))
        ctk.CTkLabel(spd, text=t("demo_speed"), font=T.font(T.FONT_XS),
                     text_color=T.TEXT_DIM).pack(side="left")
        self._speed = ctk.CTkSlider(spd, from_=0, to=260,
                                    command=self._on_speed)
        self._speed.set(110)
        self._speed.pack(side="left", fill="x", expand=True, padx=8)
        self._motif = ctk.CTkOptionMenu(
            spd, values=list(MOTIFS), width=110, height=26,
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD, font=T.font(T.FONT_XS),
            command=self._on_motif)
        self._motif.set(self._pattern.motif)
        self._motif.pack(side="right")

        self._set_tool("rayon")

    # ── ciclo de vida de la pantalla ──────────────────────────────────────
    def on_show(self) -> None:
        self._source.start()
        if self._worker is None and not self._worker_loading:
            self._load_model()
        if not self._running:
            self._running = True
            self.after(30, self._loop)

    def on_hide(self) -> None:
        self._running = False
        self._source.stop()

    def on_close(self) -> None:
        self.on_hide()
        if self._worker:
            self._worker.stop()
            self._worker = None

    # ── modelo ────────────────────────────────────────────────────────────
    def _model_path(self) -> Optional[Path]:
        """
        Modelo de la demo: el del proyecto abierto si es el proyecto demo,
        y si no, el best.pt del proyecto Demo_Textil junto a la instalación.
        """
        proj = self._app.get_project()
        if proj and proj.active_model and Path(proj.active_model).exists():
            return Path(proj.active_model)
        if proj and proj.paths.best_model.exists():
            return proj.paths.best_model
        here = Path(__file__).resolve().parents[3]
        cand = here / "projects" / "Demo_Textil" / "models" / "best.pt"
        return cand if cand.exists() else None

    def _load_model(self) -> None:
        path = self._model_path()
        if path is None:
            self._hint.configure(text=t("demo_no_model"), text_color=T.WARN)
            return
        self._worker_loading = True
        self._hint.configure(text=t("demo_loading"), text_color=T.TEXT_SEC)

        def work():
            try:
                w = YOLOWorker(str(path), conf=0.35, imgsz=512)
                w.start()
                self._worker = w
                self.after(0, lambda: self._hint.configure(
                    text=t("demo_hint"), text_color=T.TEXT_SEC))
            except Exception as e:
                log.error(f"Demo model load failed: {e}")
                self.after(0, lambda: self._hint.configure(
                    text=t("demo_model_error", err=str(e)[:60]),
                    text_color=T.COLOR_NOK))
            finally:
                self._worker_loading = False

        threading.Thread(target=work, daemon=True).start()

    # ── herramientas del lienzo ───────────────────────────────────────────
    def _set_tool(self, cls: str) -> None:
        self._tool = cls
        for c, btn in self._tool_btns.items():
            btn.configure(border_width=3 if c == cls else 0,
                          border_color=T.ACCENT)

    def _canvas_to_frame(self, cx: int, cy: int) -> Optional[tuple[int, int]]:
        if self._view is None:
            return None
        fw, fh, nw, nh, ox, oy = self._view
        fx = (cx - ox) * fw / max(nw, 1)
        fy = (cy - oy) * fh / max(nh, 1)
        if not (0 <= fx < fw and 0 <= fy < fh):
            return None
        return int(fx), int(fy)

    def _radius(self) -> int:
        return {"rayon": 5, "mancha": 16, "falta_impresion": 20}[self._tool]

    def _on_press(self, event) -> None:
        pt = self._canvas_to_frame(event.x, event.y)
        if pt is None:
            return
        self._drawing = True
        self._pattern.begin_stroke(self._tool)
        fab = self._pattern.screen_to_fabric(*pt)
        self._pattern.paint(*fab, self._radius(), self._tool)
        self._last_pt = fab

    def _on_drag(self, event) -> None:
        if not self._drawing:
            return
        pt = self._canvas_to_frame(event.x, event.y)
        if pt is None:
            return
        fab = self._pattern.screen_to_fabric(*pt)
        if self._last_pt is not None:
            # Si la tela se movió entre dos eventos del mouse, unir los dos
            # puntos EN LA TELA dejaría una raya larguísima cruzando el
            # encuadre.  Con el viaje en marcha se pinta punto a punto.
            dx = abs(fab[0] - self._last_pt[0])
            dy = abs(fab[1] - self._last_pt[1])
            if dx < 120 and dy < 120:
                self._pattern.paint_segment(self._last_pt, fab,
                                            self._radius(), self._tool)
            else:
                self._pattern.paint(*fab, self._radius(), self._tool)
        self._last_pt = fab

    def _on_release(self, _event) -> None:
        if self._drawing:
            self._pattern.end_stroke()
        self._drawing = False
        self._last_pt = None

    # ── controles ─────────────────────────────────────────────────────────
    def reset_demo(self) -> None:
        """Tela como nueva: sin defectos, sin tracks, contadores en cero."""
        self._pattern.clear_defects()
        self._tracker.reset()
        self._counting.reset()
        self._nok_total = 0
        self._per_class = {}
        self._count_lbl.configure(text=t("demo_found", n=0))
        self._detail_lbl.configure(text="")

    def _toggle_pause(self) -> None:
        self._source.set_paused(not self._source.paused)
        self._pause_btn.configure(
            text=t("demo_resume") if self._source.paused else t("demo_pause"))

    def _on_speed(self, value: float) -> None:
        self._source.set_speed(float(value))

    def _on_motif(self, motif: str) -> None:
        """Cambiar de motivo re-teje la tela: los defectos no sobreviven."""
        self._pattern = TextilePattern(width=_W, height=_H, axis="x",
                                       motif=motif,
                                       speed=float(self._speed.get()))
        self._source.pattern = self._pattern
        self.reset_demo()

    # ── bucle ─────────────────────────────────────────────────────────────
    def _loop(self) -> None:
        if not self._running:
            return
        frame = self._source.read()
        if frame is not None:
            # lienzo del visitante: la tela limpia, sin anotaciones
            self._view = self._paint(self._draw_canvas, frame, "_photo_canvas")
            # lado de Eyve: detección + seguimiento + conteo
            annotated = self._inspect(frame)
            self._paint(self._eyve_canvas, annotated, "_photo_eyve")
        self.after(33, self._loop)

    def _inspect(self, frame: np.ndarray) -> np.ndarray:
        if self._worker is None:
            return frame
        self._worker.push_frame(frame)
        result = self._worker.get_result()
        if result is None:
            return frame

        proj = self._app.get_project()
        nok = proj.nok_classes if proj else list(DEFECT_CLASSES)
        if not nok:
            nok = list(DEFECT_CLASSES)
        inspection = decide(result.detections, [], nok, 0.35)

        raw = [RawDetection(d.class_name, d.confidence,
                            int(d.x1), int(d.y1), int(d.x2), int(d.y2))
               for d in result.detections]
        tracks = self._tracker.update(raw, frame)

        annotated = frame.copy()
        for tr in tracks:
            if not tr.confirmed:
                continue
            x1, y1, x2, y2 = tr.bbox
            col = {"rayon": (0, 215, 255), "mancha": (255, 190, 0),
                   "falta_impresion": (120, 255, 120)}.get(tr.label,
                                                           (0, 230, 118))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
            cv2.putText(annotated, f"{tr.label} #{tr.id}",
                        (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, col, 2, cv2.LINE_AA)

        fh, fw = frame.shape[:2]
        self._counting.update_tracks(tracks, expired=self._tracker.last_expired,
                                     frame_wh=(fw, fh))
        self._counting.draw(annotated)

        total = self._counting.total
        if total != self._nok_total:
            self._nok_total = total
            self._count_lbl.configure(text=t("demo_found", n=total))
            detail = "  ".join(f"{k}: {v}"
                               for k, v in sorted(self._counting.counts.items()))
            self._detail_lbl.configure(text=detail)

        status = inspection.status
        if status != self._last_status:
            self._last_status = status
            if status == InspectionStatus.NOT_OK:
                txt, col = t("demo_defect"), T.COLOR_NOK
            else:
                txt, col = t("demo_clean"), T.COLOR_OK
            self._status_lbl.configure(text=txt, text_color=col)
        return annotated

    def _paint(self, canvas: tk.Canvas, frame: np.ndarray, attr: str):
        """Dibuja *frame* ajustado al canvas; devuelve el mapeo canvas↔frame."""
        try:
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw < 10 or ch < 10:
                return None
            canvas.delete("all")
            fh, fw = frame.shape[:2]
            scale = min(cw / fw, ch / fh)
            nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
            small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            setattr(self, attr, photo)       # evita que el GC se lo lleve
            canvas.create_image(cw // 2, ch // 2, image=photo, anchor="center")
            return (fw, fh, nw, nh, (cw - nw) // 2, (ch - nh) // 2)
        except Exception:
            return None
