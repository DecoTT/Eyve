"""
OK / NOT OK decision logic.

Reads the project's ok_classes / nok_classes and applies them
to a list of detections to produce a final inspection status.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from eyve.inference.detector import Detection


class InspectionStatus(str, Enum):
    OK         = "OK"
    NOT_OK     = "NOT_OK"
    NO_DETECT  = "NO_DETECTION"
    REVIEW     = "REVIEW"
    ERROR      = "ERROR"


@dataclass
class InspectionResult:
    status: InspectionStatus
    triggered_by: Optional[str] = None   # class name that triggered NOK
    confidence: float = 0.0
    detections: list[Detection] = None

    def __post_init__(self):
        if self.detections is None:
            self.detections = []


def decide(
    detections: list[Detection],
    ok_classes: list[str],
    nok_classes: list[str],
    min_confidence: float = 0.50,
    no_detection_behavior: str = "no_detect",   # "no_detect" | "review"
) -> InspectionResult:
    """Apply OK/NOK rules and return an InspectionResult."""

    # filter by confidence
    filtered = [d for d in detections if d.confidence >= min_confidence]

    if not filtered:
        status = (InspectionStatus.REVIEW
                  if no_detection_behavior == "review"
                  else InspectionStatus.NO_DETECT)
        return InspectionResult(status=status, detections=detections)

    # check for any NOK detection first (worst-case)
    nok_hits = [d for d in filtered if d.class_name in nok_classes]
    if nok_hits:
        worst = max(nok_hits, key=lambda d: d.confidence)
        return InspectionResult(
            status=InspectionStatus.NOT_OK,
            triggered_by=worst.class_name,
            confidence=worst.confidence,
            detections=detections,
        )

    # any OK detection
    ok_hits = [d for d in filtered if d.class_name in ok_classes]
    if ok_hits:
        best = max(ok_hits, key=lambda d: d.confidence)
        return InspectionResult(
            status=InspectionStatus.OK,
            triggered_by=best.class_name,
            confidence=best.confidence,
            detections=detections,
        )

    # detections exist but none are OK or NOK (all ignored)
    return InspectionResult(status=InspectionStatus.REVIEW, detections=detections)
