"""
Mock vision detector for testing and local development.

Returns deterministic detection results with common sensitive object types.
"""

from __future__ import annotations

import logging

from picture.domain.enums import FindingType, VisionObjectType
from picture.domain.models import BBox, PictureFinding, RegionMask
from picture.providers.base import VisionDetector

logger = logging.getLogger(__name__)


class MockVisionDetector(VisionDetector):
    """
    Mock vision detector that returns pre-defined object detections.

    Simulates typical detection results including face, QR code, and signature.
    """

    def __init__(self, return_detections: bool = True) -> None:
        self._return_detections = return_detections

    @property
    def name(self) -> str:
        return "MockVision"

    def detect(self, image_path: str) -> list[PictureFinding]:
        """Return mock vision detection results."""
        logger.info("[MockVision] Detecting objects in: %s", image_path)

        if not self._return_detections:
            return []

        return [
            PictureFinding(
                finding_type=FindingType.VISION_OBJECT,
                category=VisionObjectType.FACE.value,
                label="Face detected",
                score=0.92,
                region=RegionMask(
                    bbox=BBox(x=200, y=80, w=120, h=150),
                    confidence=0.92,
                ),
                reason_code="VISION_FACE",
                provider=self.name,
            ),
            PictureFinding(
                finding_type=FindingType.VISION_OBJECT,
                category=VisionObjectType.QR_CODE.value,
                label="QR code detected",
                score=0.88,
                region=RegionMask(
                    bbox=BBox(x=500, y=400, w=100, h=100),
                    confidence=0.88,
                ),
                reason_code="VISION_QR_CODE",
                provider=self.name,
            ),
            PictureFinding(
                finding_type=FindingType.VISION_OBJECT,
                category=VisionObjectType.SIGNATURE.value,
                label="Signature detected",
                score=0.75,
                region=RegionMask(
                    bbox=BBox(x=300, y=500, w=200, h=60),
                    confidence=0.75,
                ),
                reason_code="VISION_SIGNATURE",
                provider=self.name,
            ),
        ]
