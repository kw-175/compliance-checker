"""
SAM 2 segmentation provider skeleton.

Requires: segment-anything-2 (or sam2), torch
"""

from __future__ import annotations

import logging
from typing import Any

from picture.domain.models import RegionMask
from picture.providers.base import SegmentationProvider

logger = logging.getLogger(__name__)


class SAM2SegmentationProvider(SegmentationProvider):
    """SAM 2 based segmentation refinement provider."""

    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-large",
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._kwargs = kwargs
        self._predictor: Any = None

    def _get_predictor(self) -> Any:
        """Lazy initialization of SAM 2."""
        if self._predictor is None:
            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import-untyped]
                self._predictor = SAM2ImagePredictor.from_pretrained(self._model_id)
            except ImportError:
                from picture.domain.exceptions import ProviderNotAvailableError
                raise ProviderNotAvailableError("SAM 2 (segment-anything-2)")
        return self._predictor

    @property
    def name(self) -> str:
        return "SAM2"

    def refine(self, image_path: str, regions: list[RegionMask]) -> list[RegionMask]:
        """Refine regions using SAM 2 prompted segmentation."""
        predictor = self._get_predictor()

        try:
            from PIL import Image
            import numpy as np
            image = np.array(Image.open(image_path).convert("RGB"))
        except ImportError:
            from picture.domain.exceptions import ProviderNotAvailableError
            raise ProviderNotAvailableError("Pillow (PIL)")

        predictor.set_image(image)

        refined: list[RegionMask] = []
        for region in regions:
            bbox = region.bbox
            input_box = np.array([bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h])

            masks, scores, _ = predictor.predict(
                box=input_box[None, :],
                multimask_output=False,
            )

            # Use highest-scoring mask
            best_mask = masks[0]
            best_score = float(scores[0])

            # Convert mask to polygon (simplified)
            from picture.domain.models import Polygon
            ys, xs = np.where(best_mask)
            if len(xs) > 0:
                polygon = Polygon(points=[
                    (float(xs.min()), float(ys.min())),
                    (float(xs.max()), float(ys.min())),
                    (float(xs.max()), float(ys.max())),
                    (float(xs.min()), float(ys.max())),
                ])
            else:
                polygon = None

            refined.append(RegionMask(
                bbox=bbox,
                polygon=polygon,
                confidence=best_score,
            ))

        return refined
