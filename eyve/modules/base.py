"""
Inspection module interface — the extension point beyond plain detection.

An InspectionModule runs INSIDE the production loop, after YOLO detection,
and operates on the detections it declares interest in (via `target_class`).
It can veto the frame verdict (e.g. polarity wrong → NOT_OK) and draw its
own overlay on the annotated frame.

This is the reference contract for future Eyve Pro modules:
counting, measurement, edge/area checks, assembly verification — they all
follow the same shape:

    frame + detections in  →  per-detection analysis  →  verdict + overlay

Design rules for implementors:
  - process() runs on every frame at production FPS: keep per-ROI work
    in the low milliseconds (cv2 primitives, no model loads here).
  - Never raise from process(); return a failed ModuleVerdict instead.
  - Draw only on the frame passed in (it is already a copy).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ModuleVerdict:
    """Outcome of one module pass over one frame."""
    ok: bool = True                 # False → module vetoes the frame to NOT_OK
    label: str = ""                 # short text for the side panel ("Q1 top 87%")
    triggered_by: str = ""          # what caused a veto ("polaridad IC")
    analyzed: int = 0               # detections this module actually analyzed
    details: dict = field(default_factory=dict)


class InspectionModule(ABC):
    """Base class for production-time inspection modules."""

    #: human-readable module name (shown in UI)
    name: str = "module"

    def __init__(self) -> None:
        self.enabled: bool = False
        self.target_class: Optional[str] = None   # class name this module inspects

    @abstractmethod
    def process(self, frame: np.ndarray, detections: list,
                annotated: np.ndarray) -> ModuleVerdict:
        """
        Analyze *frame* (clean BGR) for every detection whose class_name
        matches self.target_class, drawing overlays on *annotated*.

        detections: list of eyve.inference.detector.Detection
        Returns a ModuleVerdict; ok=False vetoes the frame to NOT_OK.
        """

    def reset(self) -> None:
        """Clear any learned state (called when production stops)."""
