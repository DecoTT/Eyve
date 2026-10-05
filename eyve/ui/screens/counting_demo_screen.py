"""
Demo del módulo de conteo: los cinco métodos, en bucle.

Por qué existe esta pantalla y no un párrafo en un folleto: "contar" suena
a una sola cosa hasta que alguien pregunta *qué* cuenta.  Son cinco
preguntas distintas, y la diferencia entre ellas se entiende en tres
segundos de verlas y en tres minutos de explicarlas.

Cada método se muestra con la escena que lo hace obvio —una banda, un área,
piezas que aparecen, piezas que alguien retira— con su número en grande y
una línea que dice para qué sirve en una planta de verdad.  El bucle avanza
solo cada pocos segundos; en un stand nadie va a estar dándole a un botón.

Qué es real y qué no, dicho sin letra chica: las piezas las genera un
simulador, así que las detecciones son exactas.  El seguimiento y el conteo
SÍ son los de producción —el mismo `InstanceTracker` y el mismo
`CountingModule`—, así que la lógica que el visitante ve contar es la que
corre en una línea.  Meter errores de detección en medio sólo enturbiaría
lo que esta pantalla explica.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import customtkinter as ctk
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk

from eyve.demo.conveyor import SCENE_INFO, SCENES, CountingScene
from eyve.i18n import t
from eyve.inference.tracker import InstanceTracker, TrackerConfig
from eyve.modules import CountingModule
from eyve.ui import theme as T

if TYPE_CHECKING:
    from eyve.ui.app import EyveApp

#: cuánto dura cada método antes de pasar al siguiente
_SEGUNDOS_POR_METODO = 14.0

_W, _H = 960, 540


class CountingDemoScreen(ctk.CTkFrame):
    def __init__(self, master, app: "EyveApp", **kwargs):
        super().__init__(master, fg_color=T.BG_DARK, **kwargs)
        self._app = app
        self._running = False
        self._photo = None
        self._idx = 0
        self._auto = True
        self._t_metodo = 0.0
        self._ultimo = time.perf_counter()
        self._resumen_last = ""

        self._scene = CountingScene(method=SCENES[0], width=_W, height=_H)
        self._tracker = InstanceTracker(config=TrackerConfig(
            iou_match=0.25, max_lost_frames=10, min_confirm_frames=2))
        self._counting = CountingModule()
        self._counting.enabled = True

        self._build()
        self._aplicar_metodo(SCENES[0])

    # ── layout ────────────────────────────────────────────────────────────
    def _build(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(self, fg_color=T.BG_CARD, corner_radius=0)
        hdr.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(hdr, text=t("cdemo_title"), font=T.bold(T.FONT_XL),
                     text_color=T.TEXT_PRI).pack(side="left", padx=20, pady=10)
        ctk.CTkLabel(hdr, text=t("cdemo_sub"), font=T.font(T.FONT_SM),
                     text_color=T.TEXT_SEC).pack(side="left", padx=6)
        # TEXT_SEC y no TEXT_DIM: a un metro de la pantalla el gris
        # oscuro sobre fondo oscuro no se lee.
        self._kiosk_lbl = ctk.CTkLabel(hdr, text=t("kiosk_enter_hint"),
                                       font=T.font(T.FONT_XS),
                                       text_color=T.TEXT_SEC)
        self._kiosk_lbl.pack(side="right", padx=16)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)

        # ── la escena ────────────────────────────────────────────────────
        izq = ctk.CTkFrame(body, fg_color=T.BG_CARD, corner_radius=10)
        izq.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        izq.grid_rowconfigure(0, weight=1)
        izq.grid_columnconfigure(0, weight=1)
        self._canvas = tk.Canvas(izq, bg="#0a0a0a", highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        # ── el panel que explica ─────────────────────────────────────────
        der = ctk.CTkFrame(body, fg_color=T.BG_CARD, corner_radius=10)
        der.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        der.grid_columnconfigure(0, weight=1)

        self._paso_lbl = ctk.CTkLabel(der, text="", font=T.font(T.FONT_XS),
                                      text_color=T.TEXT_DIM, anchor="w")
        self._paso_lbl.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 0))

        self._metodo_lbl = ctk.CTkLabel(der, text="", font=T.bold(T.FONT_XXL),
                                        text_color=T.ACCENT2, anchor="w",
                                        wraplength=420, justify="left")
        self._metodo_lbl.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 2))

        self._pregunta_lbl = ctk.CTkLabel(der, text="", font=T.bold(T.FONT_MD),
                                          text_color=T.TEXT_PRI, anchor="w",
                                          wraplength=420, justify="left")
        self._pregunta_lbl.grid(row=2, column=0, sticky="ew", padx=20, pady=(4, 10))

        self._como_lbl = ctk.CTkLabel(der, text="", font=T.font(T.FONT_SM),
                                      text_color=T.TEXT_SEC, anchor="w",
                                      wraplength=420, justify="left")
        self._como_lbl.grid(row=3, column=0, sticky="ew", padx=20)

        sep = ctk.CTkFrame(der, height=1, fg_color=T.BORDER)
        sep.grid(row=4, column=0, sticky="ew", padx=20, pady=14)

        ctk.CTkLabel(der, text=t("cdemo_count"), font=T.font(T.FONT_XS),
                     text_color=T.TEXT_DIM, anchor="w").grid(
            row=5, column=0, sticky="ew", padx=20)
        self._num_lbl = ctk.CTkLabel(der, text="0", anchor="w",
                                     font=ctk.CTkFont(size=64, weight="bold"),
                                     text_color=T.ACCENT)
        self._num_lbl.grid(row=6, column=0, sticky="ew", padx=20)
        self._detalle_lbl = ctk.CTkLabel(der, text="", font=T.font(T.FONT_SM),
                                         text_color=T.TEXT_SEC, anchor="w",
                                         wraplength=420, justify="left")
        self._detalle_lbl.grid(row=7, column=0, sticky="ew", padx=20, pady=(0, 10))

        sep2 = ctk.CTkFrame(der, height=1, fg_color=T.BORDER)
        sep2.grid(row=8, column=0, sticky="ew", padx=20, pady=10)

        ctk.CTkLabel(der, text=t("cdemo_donde"), font=T.font(T.FONT_XS),
                     text_color=T.TEXT_DIM, anchor="w").grid(
            row=9, column=0, sticky="ew", padx=20)
        self._uso_lbl = ctk.CTkLabel(der, text="", font=T.font(T.FONT_SM),
                                     text_color=T.TEXT_PRI, anchor="w",
                                     wraplength=420, justify="left")
        self._uso_lbl.grid(row=10, column=0, sticky="ew", padx=20, pady=(2, 12))

        der.grid_rowconfigure(11, weight=1)

        # ── controles: por si alguien quiere detenerse en uno ────────────
        ctrl = ctk.CTkFrame(der, fg_color="transparent")
        ctrl.grid(row=12, column=0, sticky="ew", padx=16, pady=(0, 16))
        ctk.CTkButton(ctrl, text="‹", width=48, height=40,
                      font=T.bold(T.FONT_LG), fg_color=T.BG_INPUT,
                      text_color=T.TEXT_PRI,
                      command=lambda: self._saltar(-1)).pack(side="left", padx=3)
        self._auto_btn = ctk.CTkButton(
            ctrl, text=t("cdemo_auto_on"), height=40, corner_radius=10,
            font=T.bold(T.FONT_SM), fg_color=T.ACCENT2, text_color="#000",
            command=self._toggle_auto)
        self._auto_btn.pack(side="left", expand=True, fill="x", padx=3)
        ctk.CTkButton(ctrl, text="›", width=48, height=40,
                      font=T.bold(T.FONT_LG), fg_color=T.BG_INPUT,
                      text_color=T.TEXT_PRI,
                      command=lambda: self._saltar(1)).pack(side="left", padx=3)

        self._barra = ctk.CTkProgressBar(der, height=4,
                                         progress_color=T.ACCENT2)
        self._barra.set(0)
        self._barra.grid(row=13, column=0, sticky="ew", padx=20, pady=(0, 16))

    def on_kiosk(self, activo: bool) -> None:
        """La pista dice como SALIR cuando ya se esta dentro."""
        self._kiosk_lbl.configure(
            text=t("kiosk_exit_hint") if activo else t("kiosk_enter_hint"))

    # ── ciclo de vida ─────────────────────────────────────────────────────
    def on_show(self) -> None:
        self._ultimo = time.perf_counter()
        if not self._running:
            self._running = True
            self.after(30, self._loop)

    def on_hide(self) -> None:
        self._running = False

    def on_close(self) -> None:
        self.on_hide()

    # ── método activo ─────────────────────────────────────────────────────
    def _aplicar_metodo(self, metodo: str) -> None:
        info = SCENE_INFO[metodo]
        self._scene.set_method(metodo)
        self._tracker.reset()
        self._counting.set_method(metodo)
        self._counting.reset()
        self._counting.line = None
        self._counting.zone = None
        self._counting.expect_min, self._counting.expect_max = info.expect
        if info.draws_line:
            self._counting.line = (self._scene.line_x, 0,
                                   self._scene.line_x, self._scene.height)
        if info.draws_zone:
            self._counting.zone = self._scene.zone
        self._t_metodo = 0.0
        self._resumen_last = ""

        self._paso_lbl.configure(
            text=t("cdemo_paso", n=self._idx + 1, total=len(SCENES)))
        self._metodo_lbl.configure(text=t("count_m_" + metodo))
        self._pregunta_lbl.configure(text=t("cdemo_q_" + metodo))
        # texto propio y no el de Produccion: alli termina en "dibuja la
        # linea sobre el video", y aqui la geometria se pone sola
        self._como_lbl.configure(text=t("cdemo_como_" + metodo))
        self._uso_lbl.configure(text=t("cdemo_uso_" + metodo))
        self._num_lbl.configure(text="0", text_color=T.ACCENT)
        self._detalle_lbl.configure(text="")

    def _saltar(self, paso: int) -> None:
        self._idx = (self._idx + paso) % len(SCENES)
        self._aplicar_metodo(SCENES[self._idx])

    def _toggle_auto(self) -> None:
        self._auto = not self._auto
        self._auto_btn.configure(
            text=t("cdemo_auto_on") if self._auto else t("cdemo_auto_off"),
            fg_color=T.ACCENT2 if self._auto else T.BG_INPUT,
            text_color="#000" if self._auto else T.TEXT_PRI)

    # ── bucle ─────────────────────────────────────────────────────────────
    def _loop(self) -> None:
        if not self._running:
            return
        ahora = time.perf_counter()
        dt = min(ahora - self._ultimo, 0.2)
        self._ultimo = ahora

        frame, dets = self._scene.step(dt)
        tracks = self._tracker.update(dets, frame)
        verdict = self._counting.update_tracks(
            tracks, expired=self._tracker.last_expired,
            frame_wh=(frame.shape[1], frame.shape[0]))

        anotado = frame.copy()
        for tr in tracks:
            if not tr.confirmed:
                continue
            x1, y1, x2, y2 = tr.bbox
            cv2.rectangle(anotado, (x1, y1), (x2, y2), (0, 230, 118), 2)
            cv2.putText(anotado, f"#{tr.id}", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 118), 1,
                        cv2.LINE_AA)
        self._counting.draw(anotado)
        self._pintar(anotado)

        total = self._counting.total
        color = T.COLOR_NOK if not verdict.ok else T.ACCENT
        self._num_lbl.configure(text=str(total), text_color=color)
        resumen = self._counting.summary()
        if resumen != self._resumen_last:
            self._resumen_last = resumen
            # el numero grande ya dice el total; aqui va el detalle que
            # cada metodo aporta (sentidos, ocupacion, rango esperado)
            extra = ""
            if self._counting.method == "line":
                d = self._counting.dir_counts
                extra = t("cdemo_dir", a=d["fwd"], b=d["rev"])
            elif self._counting.method == "zone":
                extra = t("cdemo_zona", n=self._counting.occupancy)
            elif self._counting.method == "screen":
                lo, hi = self._counting.expect_min, self._counting.expect_max
                if lo is not None and hi is not None:
                    extra = t("cdemo_esperado", lo=lo, hi=hi)
            self._detalle_lbl.configure(text=extra)

        self._t_metodo += dt
        self._barra.set(min(1.0, self._t_metodo / _SEGUNDOS_POR_METODO))
        if self._auto and self._t_metodo >= _SEGUNDOS_POR_METODO:
            self._saltar(1)

        self.after(33, self._loop)

    def _pintar(self, frame: np.ndarray) -> None:
        try:
            cw = self._canvas.winfo_width()
            ch = self._canvas.winfo_height()
            if cw < 10 or ch < 10:
                return
            self._canvas.delete("all")
            fh, fw = frame.shape[:2]
            esc = min(cw / fw, ch / fh)
            nw, nh = max(1, int(fw * esc)), max(1, int(fh * esc))
            chico = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(chico, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._photo = photo        # evita que el GC se lo lleve
            self._canvas.create_image(cw // 2, ch // 2, image=photo,
                                      anchor="center")
        except Exception:
            pass
