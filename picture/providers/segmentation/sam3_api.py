from __future__ import annotations

from typing import Any

import httpx

from picture.domain.models import BBox, Polygon, RegionMask
from picture.providers.base import SegmentationProvider


class SAM3APISegmentationProvider(SegmentationProvider):
    """SAM3 bbox refinement backed by the SAM3 FastAPI service."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8218",
        timeout_seconds: float = 180.0,
        confidence_threshold: float = 0.35,
        **kwargs: Any,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._confidence_threshold = confidence_threshold

    @property
    def name(self) -> str:
        return "SAM3API"

    def warmup(self) -> dict[str, Any]:
        response = httpx.post(f"{self._base_url}/warmup", timeout=self._timeout_seconds)
        response.raise_for_status()
        return response.json()

    def refine(self, image_path: str, regions: list[RegionMask]) -> list[RegionMask]:
        if not regions:
            return []
        payload = {
            "image_path": image_path,
            "threshold": self._confidence_threshold,
            "regions": [
                {
                    "bbox": region.bbox.model_dump(),
                    "polygon": region.polygon.model_dump(mode="json")["points"] if region.polygon is not None else None,
                    "confidence": region.confidence,
                    "region_kind": "ocr_text",
                    "text_prompt": "printed text characters",
                    "refine_mode": "text_region",
                }
                for region in regions
            ],
        }
        response = httpx.post(f"{self._base_url}/v1/sam3/refine", json=payload, timeout=self._timeout_seconds)
        response.raise_for_status()
        refined: list[RegionMask] = []
        for original, item in zip(regions, response.json().get("regions", [])):
            bbox_data = item.get("bbox") or {}
            polygon = _polygon_from_api(item.get("polygon"))
            refined.append(
                original.model_copy(
                    update={
                        "bbox": BBox(
                            x=float(bbox_data.get("x", original.bbox.x)),
                            y=float(bbox_data.get("y", original.bbox.y)),
                            w=float(bbox_data.get("w", original.bbox.w)),
                            h=float(bbox_data.get("h", original.bbox.h)),
                        ),
                        "polygon": polygon if polygon is not None else original.polygon,
                        "mask_path": str(item.get("mask_path") or original.mask_path or "") or None,
                        "confidence": max(original.confidence, float(item.get("confidence") or 0.0)),
                    }
                )
            )
        if len(refined) < len(regions):
            refined.extend(regions[len(refined):])
        return refined


def _polygon_from_api(value: Any) -> Polygon | None:
    if not isinstance(value, list):
        return None
    points: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            points.append((float(item[0]), float(item[1])))
        except (TypeError, ValueError):
            continue
    return Polygon(points=points) if points else None
