"""Production session logger — CSV + optional NOK screenshots."""
from __future__ import annotations
import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from eyve.production.ok_nok_logic import InspectionResult, InspectionStatus
from eyve.core.logger import log


class ProductionSession:
    def __init__(
        self,
        sessions_dir: Path,
        screenshots_dir: Path,
        model_path: str,
        save_nok: bool = True,
        save_log: bool = True,
    ):
        self.id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
        self._save_nok = save_nok
        self._save_log = save_log
        self._model_path = model_path
        self._started = datetime.now()

        self._session_dir = sessions_dir / f"session_{self.id}"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._nok_dir = self._session_dir / "nok_screenshots"
        if save_nok:
            self._nok_dir.mkdir(exist_ok=True)

        self._csv_path = self._session_dir / "detections.csv"
        self._csv_file = None
        self._csv_writer = None
        if save_log:
            self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "timestamp", "status", "class", "confidence",
                "x1", "y1", "x2", "y2", "image_path", "model_path"
            ])

        self._count_ok = 0
        self._count_nok = 0
        self._count_review = 0
        log.info(f"Production session started: {self.id}")

    def record(self, result: InspectionResult, frame: Optional[np.ndarray] = None) -> None:
        ts = datetime.now().isoformat()
        status = result.status.value
        cls_name = result.triggered_by or ""
        conf = result.confidence

        if result.status == InspectionStatus.OK:
            self._count_ok += 1
        elif result.status == InspectionStatus.NOT_OK:
            self._count_nok += 1
        else:
            self._count_review += 1

        img_path = ""
        if result.status == InspectionStatus.NOT_OK and self._save_nok and frame is not None:
            fname = f"nok_{datetime.now().strftime('%H%M%S_%f')}.jpg"
            img_path = str(self._nok_dir / fname)
            cv2.imwrite(img_path, frame)

        if self._csv_writer:
            det = result.detections[0] if result.detections else None
            x1 = det.x1 if det else ""
            y1 = det.y1 if det else ""
            x2 = det.x2 if det else ""
            y2 = det.y2 if det else ""
            self._csv_writer.writerow([
                ts, status, cls_name, f"{conf:.3f}",
                x1, y1, x2, y2, img_path, self._model_path
            ])

    def close(self) -> None:
        if self._csv_file:
            self._csv_file.close()
        elapsed = (datetime.now() - self._started).total_seconds()
        summary = (
            f"Session: {self.id}\n"
            f"Duration: {int(elapsed)}s\n"
            f"OK: {self._count_ok}  NOK: {self._count_nok}  Review: {self._count_review}\n"
            f"Model: {self._model_path}\n"
        )
        (self._session_dir / "summary.txt").write_text(summary, encoding="utf-8")
        log.info(f"Session closed: {self.id}  OK={self._count_ok} NOK={self._count_nok}")

    @property
    def stats(self) -> dict:
        return {
            "id": self.id,
            "ok": self._count_ok,
            "nok": self._count_nok,
            "review": self._count_review,
        }
