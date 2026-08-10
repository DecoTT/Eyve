"""
YOLO inference worker — runs detection in a background thread.
Ported and generalized from Eyve 2.0 v7/eyve_workers.py.
"""
from __future__ import annotations
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from eyve.core.logger import log


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class InferenceResult:
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    frame_idx: int = 0


class YOLOWorker:
    """
    Async YOLO detection worker.

    Push frames with push_frame(), pull results with get_result().
    """

    def __init__(self, model_path: str, conf: float = 0.50, imgsz: int = 640):
        self._model_path = model_path
        self._conf = conf
        self._imgsz = imgsz
        self._model = None
        self._class_names: list[str] = []

        self._in_q: queue.Queue = queue.Queue(maxsize=2)
        self._out_q: queue.Queue = queue.Queue(maxsize=8)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_idx = 0

    def load(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(self._model_path)
        self._class_names = list(self._model.names.values())
        log.info(f"Model loaded: {self._model_path}  classes={self._class_names}")

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    def start(self) -> None:
        if not self._model:
            self.load()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def push_frame(self, frame: np.ndarray) -> None:
        try:
            self._in_q.put_nowait((self._frame_idx, frame))
            self._frame_idx += 1
        except queue.Full:
            pass  # drop frame if worker can't keep up

    def get_result(self) -> Optional[InferenceResult]:
        try:
            return self._out_q.get_nowait()
        except queue.Empty:
            return None

    def set_conf(self, conf: float) -> None:
        self._conf = conf

    def _loop(self) -> None:
        while self._running:
            try:
                frame_idx, frame = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            detections = self._infer(frame)
            ms = (time.perf_counter() - t0) * 1000
            result = InferenceResult(detections=detections, inference_ms=ms, frame_idx=frame_idx)
            try:
                self._out_q.put_nowait(result)
            except queue.Full:
                try:
                    self._out_q.get_nowait()
                    self._out_q.put_nowait(result)
                except queue.Empty:
                    pass

    def _infer(self, frame: np.ndarray) -> list[Detection]:
        if self._model is None:
            return []
        try:
            results = self._model.predict(
                frame, conf=self._conf, imgsz=self._imgsz, verbose=False
            )
            detections = []
            for r in results:
                for box in r.boxes:
                    ci = int(box.cls[0])
                    name = self._class_names[ci] if ci < len(self._class_names) else str(ci)
                    xyxy = box.xyxy[0].tolist()
                    detections.append(Detection(
                        class_id=ci,
                        class_name=name,
                        confidence=float(box.conf[0]),
                        x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3],
                    ))
            return detections
        except Exception as e:
            log.error(f"Inference error: {e}")
            return []


class VideoSource:
    """Camera or video file frame grabber (background thread)."""

    def __init__(self, source: int | str):
        self._source = source
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._running = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        is_cam = isinstance(self._source, int)
        backend = cv2.CAP_DSHOW if is_cam else 0

        # DSHOW releases the camera asynchronously: cap.release() returns
        # immediately but the DirectShow filter graph keeps tearing down for
        # 200-500 ms.  If another screen just released this device (on_hide →
        # on_show), DSHOW reports "can't capture by index".
        # Retry a few times before giving up.
        max_tries = 5 if is_cam else 1
        self._cap = None
        for attempt in range(max_tries):
            cap = cv2.VideoCapture(self._source, backend)
            if cap.isOpened():
                self._cap = cap
                break
            cap.release()
            if attempt < max_tries - 1:
                time.sleep(0.35)

        if not (self._cap and self._cap.isOpened()):
            return False

        if is_cam:
            # MJPEG once at open so USB 2.0 can carry 1080p at 30 fps,
            # then request HD.  YOLO resizes internally to imgsz (640).
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _loop(self) -> None:
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                if hasattr(self._source, '__len__') or isinstance(self._source, str):
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.01)
