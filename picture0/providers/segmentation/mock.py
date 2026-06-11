"""
Mock segmentation provider for testing.

Passes through bounding boxes with slight polygon refinement simulation.
"""

from __future__ import annotations

import logging

from picture.domain.models import Polygon, RegionMask
from picture.providers.base import SegmentationProvider

logger = logging.getLogger(__name__)


class MockSegmentationProvider(SegmentationProvider):
    """
    Mock segmentation provider that converts bounding boxes to simple
    rectangular polygons (simulating segmentation refinement).
    """

    @property
    def name(self) -> str:
        return "MockSegmentation"

    def refine(self, image_path: str, regions: list[RegionMask]) -> list[RegionMask]:
        """Refine regions by adding polygon representations from bbox."""
        logger.info("[MockSegmentation] Refining %d regions", len(regions))

        refined: list[RegionMask] = []
        for region in regions:
            bbox = region.bbox
            # Create a polygon from the bounding box (simulating SAM output)
            polygon = Polygon(points=[
                (bbox.x, bbox.y),
                (bbox.x + bbox.w, bbox.y),
                (bbox.x + bbox.w, bbox.y + bbox.h),
                (bbox.x, bbox.y + bbox.h),
            ])
            refined.append(RegionMask(
                bbox=bbox,
                polygon=polygon,
                confidence=region.confidence,
            ))

        return refined
