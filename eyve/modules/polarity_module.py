"""
Polarity inspection module — first Eyve Pro module, ported from 2.0.

Wraps eyve.inference.polarity.PolarityAnalyzer (the proven 2.0 engine:
stripe/half/vector/dot/notch with weighted voting and auto-learned
reference) behind the InspectionModule contract so the production loop
can run it per-detection without knowing anything about polarity.

Typical use: capacitores SMD en cinta — detect class "IC", the stripe
method finds the cathode band, the learned reference says which side is
correct, and any flipped part vetoes the frame to NOT_OK.
"""
from __future__ import annotations
from typing import Optional

import numpy as np

from eyve.inference.polarity import PolarityAnalyzer, PolarityResult
from eyve.modules.base import InspectionModule, ModuleVerdict


class PolarityModule(InspectionModule):
    name = "Polaridad"

    #: analyzer methods surfaced in the UI (subset of PolarityAnalyzer.METHODS)
    METHODS = ("stripe", "auto", "dot", "notch")
    #: reference options: auto-learn from first confident samples, or fixed side
    REFERENCES = ("auto", "top", "bottom", "left", "right")
    _SIDE_TO_QUAD = {"top": 1, "right": 2, "bottom": 3, "left": 4}

    def __init__(self) -> None:
        super().__init__()
        self._analyzer = PolarityAnalyzer(method="stripe")
        self._reference = "auto"
        self._last: Optional[PolarityResult] = None
        #: debug image of the most recent analysis (picture-in-picture preview)
        self.last_debug = None

    # ── configuration (called from UI) ────────────────────────────────────
    def set_method(self, method: str) -> None:
        self._analyzer.set_method(method)

    def set_reference(self, ref: str) -> None:
        """'auto' re-enters learning mode; a side name fixes the reference."""
        self._reference = ref
        if ref == "auto":
            self._analyzer.reset_reference()
        elif ref in self._SIDE_TO_QUAD:
            self._analyzer.set_reference(self._SIDE_TO_QUAD[ref])

    def set_arc_thickness(self, value: float) -> None:
        """Stripe arc thickness (fraction of cap radius sampled at the rim)."""
        self._analyzer.set_arc_thickness(value)

    @property
    def arc_thickness(self) -> float:
        return self._analyzer.arc_thickness

    def reset(self) -> None:
        if self._reference == "auto":
            self._analyzer.reset_reference()
        self._last = None
        self.last_debug = None

    # ── per-track analysis (used with InstanceTracker) ────────────────────
    def analyze_roi(self, roi) -> Optional[PolarityResult]:
        """
        Analyze one ROI directly (per-track path: the production loop calls
        this ONCE per tracked instance instead of every frame).
        Stores the debug image for the picture-in-picture preview.
        """
        if roi is None:
            return None
        try:
            result = self._analyzer.analyze(roi)
        except Exception:
            return None
        self._last = result
        if result.debug_img is not None:
            self.last_debug = result.debug_img
        return result

    @property
    def learning_status(self) -> str:
        """Short status for the UI: learning progress or locked reference."""
        a = self._analyzer
        if a._reference is not None:
            side = {1: "top", 2: "right", 3: "bottom", 4: "left"}.get(a._reference, "?")
            return f"ref: {side}"
        return f"aprendiendo {len(a._ref_samples)}/6"

    # ── production-loop hook ──────────────────────────────────────────────
    def process(self, frame: np.ndarray, detections: list,
                annotated: np.ndarray) -> ModuleVerdict:
        if not self.enabled or not self.target_class:
            return ModuleVerdict()

        fh, fw = frame.shape[:2]
        analyzed = 0
        wrong = 0
        label = ""

        for det in detections:
            if det.class_name != self.target_class:
                continue
            x1 = max(0, int(det.x1)); y1 = max(0, int(det.y1))
            x2 = min(fw, int(det.x2)); y2 = min(fh, int(det.y2))
            if x2 - x1 < 12 or y2 - y1 < 12:
                continue   # ROI too small to analyze
            try:
                result = self._analyzer.analyze(frame[y1:y2, x1:x2])
            except Exception:
                continue   # never let a module kill the production loop
            analyzed += 1
            self._last = result
            if result.quadrant is not None:
                label = f"{result.side} Q{result.quadrant} {result.confidence:.0%}"
                if result.is_correct is False:
                    wrong += 1
            PolarityAnalyzer.draw_on_frame(annotated, (x1, y1, x2, y2), result)

        if wrong:
            return ModuleVerdict(
                ok=False,
                label=f"⚠ polaridad invertida ×{wrong}",
                triggered_by=f"polaridad {self.target_class}",
                analyzed=analyzed,
            )
        return ModuleVerdict(ok=True, label=label, analyzed=analyzed)
