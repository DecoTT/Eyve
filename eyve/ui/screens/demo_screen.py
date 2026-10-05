"""
Pantalla de demo para stand de expo.

La pantalla se parte en dos:

    IZQUIERDA — Eyve corriendo de verdad.  El mismo YOLOWorker, el mismo
                InstanceTracker, el mismo CountingModule, el mismo
                PatternModule y la misma lógica OK/NOT OK que la pantalla
                de Producción.  Lo único distinto es de dónde salen los
                frames.

    DERECHA   — el lienzo del visitante: la misma tela, sin anotaciones,
                donde dibuja defectos con el dedo o el mouse.

Lo que la demo tiene que dejar claro, y por eso corren los dos módulos a
la vez:

    YOLO    nombra lo que le enseñaste        "rayón", "mancha"
    Patrón  descubre lo que nunca vio         "aquí algo no cuadra"
    Conteo  convierte instancias en un número que sirve para producir

Por qué no se reusa ProductionScreen entera: en un stand la gente mira de
lejos y toca la pantalla.  Lo que se necesita es estado grande, contador
grande y botones gordos — no el panel de 40 controles de producción.  El
motor sí se reusa; la cáscara no.

Modo automático: con el stand vacío, Eyve tiene que estar haciendo algo o
nadie se acerca.  Cada cierto rato aparece un defecto solo.  Se apaga en
cuanto alguien toca el lienzo: a partir de ahí manda la persona.
"""
from __future__ import annotations

import random
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import customtkinter as ctk
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk

from eyve.core.logger import log
from eyve.demo.source import SyntheticSource
from eyve.demo.textile import (MOTIFS, PRINT_FAULTS, TextilePattern, WEAVES,
                               YOLO_CLASSES)
from eyve.i18n import t
from eyve.inference.detector import YOLOWorker
from eyve.inference.tracker import InstanceTracker, RawDetection, TrackerConfig
from eyve.modules import CountingModule, PatternModule
from eyve.production.ok_nok_logic import InspectionStatus, decide
from eyve.ui import theme as T

if TYPE_CHECKING:
    from eyve.ui.app import EyveApp

#: herramientas del lienzo → (clase, color del botón)
_TOOLS = [("rayon", "#2b2a2e"), ("mancha", "#5c4a96")]

#: cada cuánto aparece un defecto solo, en segundos.  Generoso a propósito:
#: uno cada 5 s llenaría la tela en medio minuto y el equipo estaría
#: inferiendo sobre un encuadre saturado todo el día.
_AUTO_MIN, _AUTO_MAX = 11.0, 18.0
#: defectos automáticos antes de limpiar la tela
_AUTO_MAX_DEFECTS = 4

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
        self._rng = random.Random()

        self._pattern = TextilePattern(width=_W, height=_H, axis="x",
                                       motif="diamantes", weave="sarga",
                                       speed=110.0)
        self._source = SyntheticSource(self._pattern, fps=30.0)

        self._tracker = InstanceTracker(config=TrackerConfig(
            iou_match=0.20, max_lost_frames=18, min_confirm_frames=2,
            process_conf_min=0.30))
        self._counting = CountingModule()
        self._counting.enabled = True
        # "En pantalla" y no "al aparecer" a proposito. Medido: acumular
        # anomalias no es exacto —un defecto que cruza cambia de forma, su
        # region se parte y cada pedazo es una instancia— mientras que
        # contar las VISIBLES AHORA no depende de mantener una identidad en
        # el tiempo y nunca ve de mas. En un stand ademas se explica mejor
        # en voz alta: "cuantos defectos hay ahora mismo en la tela".
        self._counting.set_method("screen")
        self._patternmod = PatternModule()
        self._patternmod.enabled = True
        self._patternmod.sensitivity = 70.0

        self._tool = "rayon"
        self._drawing = False
        self._last_pt: Optional[tuple[int, int]] = None
        #: si la tela ya estaba pausada antes de empezar a dibujar, para no
        #: reanudarla al soltar y pisarle la decision al visitante
        self._pausa_previa = False
        self._view: Optional[tuple] = None
        self._count_total = -1
        self._last_status: Optional[InspectionStatus] = None
        self._pat_last = ""
        #: ultimas detecciones de YOLO, para que el Patron no vuelva a
        #: levantar la mano por algo que ya tiene nombre
        self._ultimas_dets: list = []
        # modo automático
        self._auto = True
        self._auto_next = time.time() + 4.0
        self._auto_placed = 0
        # calibración del módulo Patrón
        self._cal_left = 0

        self._build()

    # ── layout ────────────────────────────────────────────────────────────
    def _build(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(self, fg_color=T.BG_CARD, corner_radius=0)
        hdr.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(hdr, text=t("demo_title"), font=T.bold(T.FONT_XL),
                     text_color=T.TEXT_PRI).pack(side="left", padx=20, pady=10)
        self._hint = ctk.CTkLabel(hdr, text=t("demo_hint"),
                                  font=T.font(T.FONT_SM), text_color=T.TEXT_SEC)
        self._hint.pack(side="left", padx=8)
        self._auto_lbl = ctk.CTkLabel(hdr, text="", font=T.bold(T.FONT_SM),
                                      text_color=T.ACCENT2)
        self._auto_lbl.pack(side="right", padx=20)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=6)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1, uniform="half")
        body.grid_columnconfigure(1, weight=1, uniform="half")

        self._build_left(body)
        self._build_right(body)
        self._refresh_auto_label()
        self._refresh_count_label()

    def _build_left(self, body) -> None:
        left = ctk.CTkFrame(body, fg_color=T.BG_CARD, corner_radius=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
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

        # ── lo que encontró cada motor, lado a lado ──────────────────────
        found = ctk.CTkFrame(left, fg_color="transparent")
        found.grid(row=2, column=0, sticky="ew", padx=12, pady=(8, 4))
        found.grid_columnconfigure(0, weight=1, uniform="f")
        found.grid_columnconfigure(1, weight=1, uniform="f")

        yolo_box = ctk.CTkFrame(found, fg_color=T.BG_INPUT, corner_radius=8)
        yolo_box.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkLabel(yolo_box, text=t("demo_yolo_title"),
                     font=T.bold(T.FONT_XS), text_color=T.ACCENT).pack(
            anchor="w", padx=10, pady=(6, 0))
        ctk.CTkLabel(yolo_box, text=t("demo_yolo_sub"), font=T.font(T.FONT_XS),
                     text_color=T.TEXT_DIM, wraplength=260, justify="left",
                     anchor="w").pack(anchor="w", padx=10)
        self._yolo_lbl = ctk.CTkLabel(yolo_box, text="—", font=T.bold(T.FONT_LG),
                                      text_color=T.COLOR_NOK)
        self._yolo_lbl.pack(anchor="w", padx=10, pady=(0, 8))

        pat_box = ctk.CTkFrame(found, fg_color=T.BG_INPUT, corner_radius=8)
        pat_box.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ctk.CTkLabel(pat_box, text=t("demo_pattern_title"),
                     font=T.bold(T.FONT_XS), text_color="#ff78ff").pack(
            anchor="w", padx=10, pady=(6, 0))
        ctk.CTkLabel(pat_box, text=t("demo_pattern_sub"), font=T.font(T.FONT_XS),
                     text_color=T.TEXT_DIM, wraplength=260, justify="left",
                     anchor="w").pack(anchor="w", padx=10)
        self._pat_lbl = ctk.CTkLabel(pat_box, text="—", font=T.bold(T.FONT_LG),
                                     text_color="#ff78ff")
        self._pat_lbl.pack(anchor="w", padx=10, pady=(0, 8))

        # ── conteo: el módulo que convierte instancias en producción ─────
        cnt = ctk.CTkFrame(left, fg_color=T.BG_INPUT, corner_radius=8)
        cnt.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 12))
        row1 = ctk.CTkFrame(cnt, fg_color="transparent")
        row1.pack(fill="x", padx=10, pady=(8, 2))
        ctk.CTkLabel(row1, text=t("demo_count_title"), font=T.bold(T.FONT_XS),
                     text_color=T.ACCENT2).pack(side="left")
        self._count_method = ctk.CTkOptionMenu(
            row1, values=[t("count_m_" + k) for k in CountingModule.METHODS],
            width=130, height=24, fg_color=T.BG_CARD, button_color=T.BG_CARD,
            dropdown_fg_color=T.BG_CARD, font=T.font(T.FONT_XS),
            command=self._on_count_method)
        self._count_method.set(t("count_m_" + self._counting.method))
        self._count_method.pack(side="right")

        self._count_help = ctk.CTkLabel(cnt, text="", font=T.font(T.FONT_XS),
                                        text_color=T.TEXT_DIM, wraplength=430,
                                        justify="left", anchor="w")
        self._count_help.pack(fill="x", padx=10)

        row2 = ctk.CTkFrame(cnt, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=(2, 8))
        self._count_lbl = ctk.CTkLabel(row2, text="", font=T.bold(T.FONT_XL),
                                       text_color=T.ACCENT2)
        self._count_lbl.pack(side="left")
        ctk.CTkButton(row2, text=t("demo_count_reset"), width=80, height=26,
                      font=T.font(T.FONT_XS), fg_color=T.BG_CARD,
                      text_color=T.TEXT_SEC,
                      command=self._reset_counting).pack(side="right")

    def _build_right(self, body) -> None:
        right = ctk.CTkFrame(body, fg_color=T.BG_CARD, corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        rh = ctk.CTkFrame(right, fg_color="transparent")
        rh.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        ctk.CTkLabel(rh, text=t("demo_side_you"), font=T.bold(T.FONT_MD),
                     text_color=T.ACCENT2).pack(side="left")
        self._cal_lbl = ctk.CTkLabel(rh, text="", font=T.font(T.FONT_XS),
                                     text_color=T.WARN)
        self._cal_lbl.pack(side="right")

        self._draw_canvas = tk.Canvas(right, bg="#0a0a0a", highlightthickness=0,
                                      cursor="pencil")
        self._draw_canvas.grid(row=1, column=0, sticky="nsew", padx=12)
        self._draw_canvas.bind("<ButtonPress-1>",   self._on_press)
        self._draw_canvas.bind("<B1-Motion>",       self._on_drag)
        self._draw_canvas.bind("<ButtonRelease-1>", self._on_release)

        tools = ctk.CTkFrame(right, fg_color="transparent")
        tools.grid(row=2, column=0, sticky="ew", padx=12, pady=(8, 2))
        self._tool_btns: dict[str, ctk.CTkButton] = {}
        for cls, color in _TOOLS:
            b = ctk.CTkButton(tools, text=t("demo_tool_" + cls), height=44,
                              font=T.bold(T.FONT_SM), corner_radius=10,
                              fg_color=color, text_color="#fff",
                              command=lambda c=cls: self._set_tool(c))
            b.pack(side="left", expand=True, fill="x", padx=3)
            self._tool_btns[cls] = b

        faults = ctk.CTkFrame(right, fg_color="transparent")
        faults.grid(row=3, column=0, sticky="ew", padx=12, pady=(4, 2))
        ctk.CTkLabel(faults, text=t("demo_faults"), font=T.font(T.FONT_XS),
                     text_color=T.TEXT_DIM).pack(anchor="w")
        frow = ctk.CTkFrame(faults, fg_color="transparent")
        frow.pack(fill="x", pady=(2, 0))
        for fault in PRINT_FAULTS:
            ctk.CTkButton(frow, text=t("demo_fault_" + fault), height=34,
                          font=T.font(T.FONT_XS), corner_radius=8,
                          fg_color=T.BG_INPUT, text_color=T.TEXT_PRI,
                          border_width=1, border_color="#ff78ff",
                          command=lambda f=fault: self._place_fault(f)).pack(
                side="left", expand=True, fill="x", padx=2)

        ctrl = ctk.CTkFrame(right, fg_color="transparent")
        ctrl.grid(row=4, column=0, sticky="ew", padx=12, pady=(6, 2))
        ctk.CTkButton(ctrl, text=t("demo_clear"), height=36, corner_radius=10,
                      font=T.bold(T.FONT_SM), fg_color=T.BG_INPUT,
                      text_color=T.TEXT_PRI, command=self.reset_demo).pack(
            side="left", expand=True, fill="x", padx=3)
        self._pause_btn = ctk.CTkButton(
            ctrl, text=t("demo_pause"), height=36, corner_radius=10,
            font=T.bold(T.FONT_SM), fg_color=T.BG_INPUT, text_color=T.TEXT_PRI,
            command=self._toggle_pause)
        self._pause_btn.pack(side="left", expand=True, fill="x", padx=3)
        self._auto_btn = ctk.CTkButton(
            ctrl, text=t("demo_auto"), height=36, corner_radius=10,
            font=T.bold(T.FONT_SM), fg_color=T.ACCENT2, text_color="#000",
            command=self._toggle_auto)
        self._auto_btn.pack(side="left", expand=True, fill="x", padx=3)

        # ── material: estampado y tejido ─────────────────────────────────
        mat = ctk.CTkFrame(right, fg_color="transparent")
        mat.grid(row=5, column=0, sticky="ew", padx=12, pady=(2, 10))
        ctk.CTkLabel(mat, text=t("demo_material"), font=T.font(T.FONT_XS),
                     text_color=T.TEXT_DIM).pack(side="left")
        self._weave = ctk.CTkOptionMenu(
            mat, values=[t("weave_" + w) for w in WEAVES], width=110, height=26,
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD, font=T.font(T.FONT_XS),
            command=self._on_weave)
        self._weave.set(t("weave_" + self._pattern.weave))
        self._weave.pack(side="right", padx=(4, 0))
        self._motif = ctk.CTkOptionMenu(
            mat, values=[t("motif_" + m) for m in MOTIFS], width=110, height=26,
            fg_color=T.BG_INPUT, button_color=T.BG_INPUT,
            dropdown_fg_color=T.BG_CARD, font=T.font(T.FONT_XS),
            command=self._on_motif)
        self._motif.set(t("motif_" + self._pattern.motif))
        self._motif.pack(side="right", padx=(4, 0))
        self._speed = ctk.CTkSlider(mat, from_=0, to=260, width=120,
                                    command=self._on_speed)
        self._speed.set(110)
        self._speed.pack(side="right", padx=8)

        self._set_tool("rayon")

    # ── ciclo de vida ─────────────────────────────────────────────────────
    def on_show(self) -> None:
        self._source.start()
        if self._worker is None and not self._worker_loading:
            self._load_model()
        self._start_calibration()
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

    # ── calibración del módulo Patrón ─────────────────────────────────────
    def _start_calibration(self, frames: int = 10) -> None:
        """
        El módulo Patrón necesita ver material BUENO unos segundos antes de
        poder decir qué es anómalo.  Se hace al entrar a la pantalla, con la
        tela limpia, y se rehace cuando cambia el material.
        """
        self._patternmod.clear_calibration()
        self._cal_left = frames

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
        return {"rayon": 5, "mancha": 16}.get(self._tool, 8)

    def _on_press(self, event) -> None:
        pt = self._canvas_to_frame(event.x, event.y)
        if pt is None:
            return
        self._hand_over()
        # La tela se detiene mientras se dibuja. Sin esto, entre dos eventos
        # del mouse la tela avanza y el trazo sale partido en dos pedazos
        # separados — y entonces un solo gesto del visitante se cuenta como
        # dos defectos, que es justo lo que no debe pasar en el stand.
        self._pausa_previa = self._source.paused
        self._source.set_paused(True)
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
            # puntos EN LA TELA dejaría una raya cruzando todo el encuadre.
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
            if not self._pausa_previa:
                self._source.set_paused(False)
        self._drawing = False
        self._last_pt = None

    def _place_fault(self, fault: str) -> None:
        """
        Pone un fallo de impresión.  Estos NO son clases entrenadas: los
        encuentra el módulo Patrón, y ese es justo el punto que la demo
        tiene que dejar claro.
        """
        self._hand_over()
        sx = self._rng.randint(150, _W - 150)
        sy = self._rng.randint(110, _H - 110)
        fx, fy = self._pattern.screen_to_fabric(sx, sy)
        self._paint_fault(fault, fx, fy)

    def _paint_fault(self, fault: str, fx: int, fy: int) -> None:
        if fault == "fantasma":
            self._pattern.ghost(fx, fy, size=self._rng.randint(130, 200))
        elif fault == "offset":
            self._pattern.misregister(fx, fy, size=self._rng.randint(140, 220))
        else:
            self._pattern.ink_starved(fx, fy, size=self._rng.randint(120, 200),
                                      severity=self._rng.uniform(0.5, 1.0))

    # ── modo automático ───────────────────────────────────────────────────
    def _hand_over(self) -> None:
        """Alguien tocó: a partir de aquí manda la persona, no el automático."""
        if self._auto:
            self._auto = False
            self._refresh_auto_label()

    def _toggle_auto(self) -> None:
        self._auto = not self._auto
        if self._auto:
            self._auto_next = time.time() + 2.0
        self._refresh_auto_label()

    def _refresh_auto_label(self) -> None:
        if self._auto:
            self._auto_lbl.configure(text=t("demo_auto_on"), text_color=T.ACCENT2)
            self._auto_btn.configure(fg_color=T.ACCENT2, text_color="#000",
                                     text=t("demo_auto_stop"))
        else:
            self._auto_lbl.configure(text=t("demo_auto_off"), text_color=T.TEXT_DIM)
            self._auto_btn.configure(fg_color=T.BG_INPUT, text_color=T.TEXT_PRI,
                                     text=t("demo_auto"))

    def _auto_tick(self) -> None:
        if not self._auto or time.time() < self._auto_next:
            return
        self._auto_next = time.time() + self._rng.uniform(_AUTO_MIN, _AUTO_MAX)
        if self._auto_placed >= _AUTO_MAX_DEFECTS:
            # Tela nueva en vez de acumular: con el encuadre saturado el
            # módulo Patrón pierde su referencia de material sano, y además
            # ver la tela volver a LIMPIO es parte de lo que se demuestra.
            self._pattern.clear_defects()
            self._auto_placed = 0
            return
        self._auto_placed += 1
        sx = self._rng.randint(150, _W - 150)
        sy = self._rng.randint(110, _H - 110)
        fx, fy = self._pattern.screen_to_fabric(sx, sy)
        # mezcla a propósito: unos los nombra YOLO, otros solo los ve Patrón
        if self._rng.random() < 0.5:
            cls = self._rng.choice(YOLO_CLASSES)
            if cls == "rayon":
                self._pattern.streak(fx, fy,
                                     length=self._rng.randint(90, 280),
                                     thickness=self._rng.randint(4, 9))
            else:
                self._pattern.blob(fx, fy, size=self._rng.randint(50, 130))
        else:
            self._paint_fault(self._rng.choice(PRINT_FAULTS), fx, fy)

    # ── controles ─────────────────────────────────────────────────────────
    def reset_demo(self) -> None:
        self._pattern.clear_defects()
        self._tracker.reset()
        self._counting.reset()
        self._ultimas_dets = []
        self._count_total = -1
        self._auto_placed = 0
        self._refresh_count_label()
        self._yolo_lbl.configure(text="—")
        self._pat_lbl.configure(text="—")

    def _reset_counting(self) -> None:
        self._counting.reset()
        self._count_total = -1
        self._refresh_count_label()

    def _toggle_pause(self) -> None:
        self._source.set_paused(not self._source.paused)
        self._pause_btn.configure(
            text=t("demo_resume") if self._source.paused else t("demo_pause"))

    def _on_speed(self, value: float) -> None:
        self._source.set_speed(float(value))

    def _on_count_method(self, label: str) -> None:
        key = next((k for k in CountingModule.METHODS
                    if t("count_m_" + k) == label), None)
        if key is None:
            return
        self._counting.set_method(key)
        # geometría automática: en un stand nadie va a dibujar la meta
        if key == "line":
            self._counting.line = (_W // 2, 0, _W // 2, _H)
        elif key == "zone":
            self._counting.zone = (_W // 4, _H // 5, 3 * _W // 4, 4 * _H // 5)
        self._count_total = -1
        self._refresh_count_label()

    def _refresh_count_label(self) -> None:
        self._count_help.configure(text=t("count_help_" + self._counting.method))
        self._count_lbl.configure(text=self._counting.summary() or "0")

    def _rebuild_fabric(self, motif: str, weave: str) -> None:
        self._pattern = TextilePattern(width=_W, height=_H, axis="x",
                                       motif=motif, weave=weave,
                                       speed=float(self._speed.get()))
        self._source.pattern = self._pattern
        self.reset_demo()
        # material nuevo, calibración nueva: el ruido de una tela de rayas
        # no se parece al de una de diamantes
        self._start_calibration()

    def _on_motif(self, label: str) -> None:
        key = next((m for m in MOTIFS if t("motif_" + m) == label), None)
        if key:
            self._rebuild_fabric(key, self._pattern.weave)
            self._motif.set(t("motif_" + key))

    def _on_weave(self, label: str) -> None:
        key = next((w for w in WEAVES if t("weave_" + w) == label), None)
        if key:
            self._rebuild_fabric(self._pattern.motif, key)
            self._weave.set(t("weave_" + key))

    # ── bucle ─────────────────────────────────────────────────────────────
    def _loop(self) -> None:
        if not self._running:
            return
        frame = self._source.read()
        if frame is not None:
            if self._cal_left > 0:
                self._patternmod.calibrate(frame)
                self._cal_left -= 1
                self._cal_lbl.configure(
                    text=t("demo_calibrating") if self._cal_left else "")
            else:
                self._auto_tick()
            self._view = self._paint(self._draw_canvas, frame, "_photo_canvas")
            annotated = self._inspect(frame)
            self._paint(self._eyve_canvas, annotated, "_photo_eyve")
        self.after(33, self._loop)

    def _inspect(self, frame: np.ndarray) -> np.ndarray:
        annotated = frame.copy()
        fh, fw = frame.shape[:2]

        # ── YOLO: nombra lo que le ensenaron ─────────────────────────────
        raw: list[RawDetection] = []
        nombres: list[str] = []
        if self._worker is not None:
            self._worker.push_frame(frame)
            result = self._worker.get_result()
            if result is not None:
                self._ultimas_dets = [
                    RawDetection(d.class_name, d.confidence,
                                 int(d.x1), int(d.y1), int(d.x2), int(d.y2))
                    for d in result.detections if d.confidence >= 0.35]
                nombres = sorted({d.label for d in self._ultimas_dets})
                self._yolo_lbl.configure(
                    text=", ".join(nombres) if nombres else t("demo_nothing"))
        raw = list(self._ultimas_dets)

        # ── Patron: corre siempre, no necesita modelo ────────────────────
        if self._patternmod.enabled and self._patternmod.calibrated:
            try:
                self._patternmod.analyze(frame)
                # Las anomalias entran al MISMO tracker, con su propia
                # clase. Asi heredan la confirmacion por frames —que es lo
                # que mata el parpadeo de una region que aparece un instante—
                # y el modulo de conteo las cuenta como lo que son: defectos.
                # Las que YOLO ya nombro se descartan para no contar dos
                # veces el mismo defecto.
                raw += self._patternmod.as_detections(
                    named=[d.bbox for d in self._ultimas_dets])
                resumen = self._patternmod.summary()
                if resumen != self._pat_last:
                    self._pat_last = resumen
                    self._pat_lbl.configure(text=resumen or "—")
            except Exception as e:
                log.debug(f"demo pattern: {e}")

        # ── un solo tracker para los dos motores ─────────────────────────
        tracks = self._tracker.update(raw, frame)
        hay_defecto = False
        for tr in tracks:
            if not tr.confirmed:
                continue
            hay_defecto = True
            x1, y1, x2, y2 = tr.bbox
            if tr.label == PatternModule.ANOMALY_LABEL:
                col, etq = (255, 120, 255), f"? #{tr.id}"
            else:
                col = {"rayon": (0, 215, 255),
                       "mancha": (255, 190, 0)}.get(tr.label, (0, 230, 118))
                etq = f"{tr.label} #{tr.id}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
            cv2.putText(annotated, etq, (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

        self._counting.update_tracks(
            tracks, expired=self._tracker.last_expired, frame_wh=(fw, fh))
        self._counting.draw(annotated)

        total = self._counting.total
        if total != self._count_total:
            self._count_total = total
            self._refresh_count_label()

        estado = (InspectionStatus.NOT_OK if hay_defecto
                  else InspectionStatus.OK)
        if estado != self._last_status:
            self._last_status = estado
            if estado == InspectionStatus.NOT_OK:
                self._status_lbl.configure(text=t("demo_defect"),
                                           text_color=T.COLOR_NOK)
            else:
                self._status_lbl.configure(text=t("demo_clean"),
                                           text_color=T.COLOR_OK)
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
