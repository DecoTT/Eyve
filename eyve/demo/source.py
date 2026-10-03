"""
Origen de video sintético — compatible con eyve.inference.detector.VideoSource.

Expone la misma interfaz (start / stop / read) que la cámara y el archivo de
video, así que el bucle de producción no sabe ni le importa que no haya
cámara: la demo de expo corre sin webcam, sin OBS y sin cámara virtual, con
latencia cero y a FPS estable.

El hilo propio existe por una razón: `read()` lo llama el hilo de Tk en cada
tick, y renderizar la tela ahí haría que el avance de la tela dependiera del
jitter del bucle de UI. Con hilo propio la tela viaja a velocidad real en
px/segundo, y `read()` solo devuelve el último frame listo.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from eyve.demo.textile import TextilePattern


class SyntheticSource:
    """Tela sintética como origen de video."""

    def __init__(self, pattern: Optional[TextilePattern] = None,
                 fps: float = 30.0) -> None:
        self.pattern = pattern or TextilePattern()
        self.fps = max(1.0, float(fps))
        self._frame: Optional[np.ndarray] = None
        self._running = False
        self._paused = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # ── interfaz de VideoSource ───────────────────────────────────────────
    def start(self) -> bool:
        if self._running:
            return True
        self._running = True
        # un primer frame listo antes de volver, para que la pantalla no
        # tenga que esperar un periodo completo para pintar algo
        with self._lock:
            self._frame = self.pattern.frame()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        t = self._thread
        if t is not None and t.is_alive() and threading.current_thread() is not t:
            t.join(timeout=2.0)
        self._thread = None

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    # ── control de la demo ────────────────────────────────────────────────
    @property
    def paused(self) -> bool:
        return self._paused

    def set_paused(self, value: bool) -> None:
        """
        Pausa el VIAJE de la tela, no el render: el visitante necesita que
        la tela se quede quieta para dibujar un defecto con calma, y el
        frame debe seguir actualizándose para que vea su trazo aparecer.
        """
        self._paused = bool(value)

    def set_speed(self, px_per_s: float) -> None:
        self.pattern.speed = max(0.0, float(px_per_s))

    # ── hilo ──────────────────────────────────────────────────────────────
    def _loop(self) -> None:
        period = 1.0 / self.fps
        last = time.perf_counter()
        while self._running:
            now = time.perf_counter()
            dt = now - last
            last = now
            # dt acotado: tras un congelamiento de la UI, un dt enorme
            # haría saltar la tela media vuelta de golpe
            if not self._paused:
                self.pattern.advance(min(dt, 0.2))
            frame = self.pattern.frame()
            with self._lock:
                self._frame = frame
            slack = period - (time.perf_counter() - now)
            if slack > 0:
                time.sleep(slack)
