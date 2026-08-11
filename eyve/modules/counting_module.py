"""
Counting module — counts tracked instances that cross a user-drawn line.

Second Eyve Pro module.  Works ON TOP of InstanceTracker: an instance is
counted exactly once, when its center crosses the finish line ("meta")
after being a confirmed track.  This is what makes the count immune to
detection flicker — a part that blinks out for a few frames keeps its ID
(MAX_LOST_FRAMES) and can't be counted twice.

The line is defined in FRAME coordinates (the production screen converts
the user's canvas drag).  Crossing = the segment between the track's
previous and current center intersects the line segment.
"""
from __future__ import annotations
from typing import Optional, Tuple

import cv2
import numpy as np

from eyve.modules.base import InspectionModule, ModuleVerdict

_LINE_COLOR = (0, 215, 255)      # amarillo (BGR)
_LINE_THICK = 2


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """True if segment p1-p2 intersects segment p3-p4 (2D)."""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    return False


class CountingModule(InspectionModule):
    name = "Conteo"

    def __init__(self) -> None:
        super().__init__()
        #: finish line in frame coords: (x1, y1, x2, y2) — None until drawn
        self.line: Optional[Tuple[int, int, int, int]] = None
        self.counts: dict[str, int] = {}
        self._counted_ids: set[int] = set()
        self._prev_centers: dict[int, Tuple[int, int]] = {}

    def reset(self) -> None:
        self.counts.clear()
        self._counted_ids.clear()
        self._prev_centers.clear()

    def clear_line(self) -> None:
        self.line = None
        self.reset()

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def summary(self) -> str:
        if not self.counts:
            return "0"
        return "  ".join(f"{k}: {v}" for k, v in sorted(self.counts.items()))

    # ── per-frame update (tracker-based, not detection-based) ─────────────
    def update_tracks(self, tracks: list) -> int:
        """
        Feed this frame's active TrackedInstances.  Counts every CONFIRMED
        track whose center path crossed the line since last frame.
        target_class None/"(todas)" counts every class.
        Returns how many new crossings happened this frame.
        """
        if not self.enabled or self.line is None:
            return 0
        lx1, ly1, lx2, ly2 = self.line
        new = 0
        for tr in tracks:
            if self.target_class and tr.label != self.target_class:
                continue
            cur = (tr.cx, tr.cy)
            prev = self._prev_centers.get(tr.id)
            self._prev_centers[tr.id] = cur
            if prev is None or not tr.confirmed or tr.id in self._counted_ids:
                continue
            if _segments_intersect(prev, cur, (lx1, ly1), (lx2, ly2)):
                self._counted_ids.add(tr.id)
                self.counts[tr.label] = self.counts.get(tr.label, 0) + 1
                new += 1
        # forget centers of purged tracks so the dict doesn't grow forever
        alive = {tr.id for tr in tracks}
        for tid in list(self._prev_centers):
            if tid not in alive:
                self._prev_centers.pop(tid, None)
                # keep _counted_ids: IDs are never reused within a session
        return new

    def draw(self, annotated: np.ndarray) -> None:
        """Draw the finish line + running total on the annotated frame."""
        if self.line is None:
            return
        x1, y1, x2, y2 = self.line
        cv2.line(annotated, (x1, y1), (x2, y2), _LINE_COLOR, _LINE_THICK)
        cv2.circle(annotated, (x1, y1), 4, _LINE_COLOR, -1)
        cv2.circle(annotated, (x2, y2), 4, _LINE_COLOR, -1)
        cv2.putText(annotated, f"{self.total}",
                    (min(x1, x2), min(y1, y2) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _LINE_COLOR, 2, cv2.LINE_AA)

    # InspectionModule contract — counting never vetoes the verdict
    def process(self, frame, detections, annotated) -> ModuleVerdict:
        return ModuleVerdict(ok=True, label=self.summary())
