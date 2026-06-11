from __future__ import annotations

from typing import Any

import httpx

from picture.domain.enums import FindingType
from picture.domain.models import BBox, PictureFinding, Polygon, RegionMask
from picture.providers.base import VisionDetector
from picture.providers.vision.sam3 import DEFAULT_PROMPTS, _dedupe_findings


class SAM3APIVisionDetector(VisionDetector):
    """SAM3 sensitive object detector backed by the SAM3 FastAPI service."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8218",
        confidence_threshold: float = 0.35,
        timeout_seconds: float = 180.0,
        prompts: dict[str, str | list[str] | tuple[str, ...]] | None = None,
        **kwargs: Any,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._confidence_threshold = confidence_threshold
        self._timeout_seconds = timeout_seconds
        self._prompts = _normalize_prompt_map(prompts or DEFAULT_PROMPTS)

    @property
    def name(self) -> str:
        return "SAM3API"

    def warmup(self) -> dict[str, Any]:
        response = httpx.post(f"{self._base_url}/warmup", timeout=self._timeout_seconds)
        response.raise_for_status()
        return response.json()

    def detect(
        self,
        image_path: str,
        target_types: list[str] | set[str] | tuple[str, ...] | None = None,
    ) -> list[PictureFinding]:
        return self.detect_with_prompts(image_path, target_types=target_types)

    def detect_with_prompts(
        self,
        image_path: str,
        target_types: list[str] | set[str] | tuple[str, ...] | None = None,
        extra_prompts: dict[str, list[str] | tuple[str, ...]] | None = None,
        confidence_thresholds: dict[str, float] | None = None,
    ) -> list[PictureFinding]:
        prompts = self._select_prompts(target_types)
        if extra_prompts:
            prompts = _merge_prompt_map(prompts, extra_prompts)
        confidence_thresholds = confidence_thresholds or {}
        prompt_payload = [
            {
                "category": category,
                "text": prompt,
                "threshold": float(confidence_thresholds.get(category, self._confidence_threshold)),
            }
            for category, values in prompts.items()
            for prompt in values
        ]
        if not prompt_payload:
            return []
        response = httpx.post(
            f"{self._base_url}/v1/sam3/detect",
            json={"image_path": image_path, "prompts": prompt_payload, "return_masks": True, "return_polygons": True},
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        findings: list[PictureFinding] = []
        for item in response.json().get("detections", []):
            bbox_data = item.get("bbox") or {}
            bbox = BBox(
                x=float(bbox_data.get("x", 0.0)),
                y=float(bbox_data.get("y", 0.0)),
                w=float(bbox_data.get("w", 0.0)),
                h=float(bbox_data.get("h", 0.0)),
            )
            score = float(item.get("score") or 0.0)
            category = str(item.get("category") or "object")
            polygon = _polygon_from_api(item.get("polygon"))
            findings.append(
                PictureFinding(
                    finding_type=FindingType.VISION_OBJECT,
                    category=category,
                    label=f"SAM3 detected {category}",
                    score=score,
                    region=RegionMask(
                        bbox=bbox,
                        polygon=polygon,
                        mask_path=str(item.get("mask_path") or "") or None,
                        confidence=score,
                    ),
                    reason_code=f"VISION_{category.upper()}",
                    provider=self.name,
                    threshold_used=float(item.get("threshold") or self._confidence_threshold),
                    explanation="SAM3 API text-prompt sensitive object detection for education privacy governance.",
                    metadata=_detection_metadata(item, self._base_url),
                )
            )
        return _dedupe_findings(findings)

    def detect_exact_prompts(
        self,
        image_path: str,
        prompts: list[dict[str, Any]],
    ) -> list[PictureFinding]:
        payload = [
            {
                "category": str(item.get("category") or "object"),
                "text": str(item.get("text") or item.get("prompt") or "").strip(),
                "threshold": float(item.get("threshold", self._confidence_threshold)),
            }
            for item in prompts
            if str(item.get("text") or item.get("prompt") or "").strip()
        ]
        if not payload:
            return []
        response = httpx.post(
            f"{self._base_url}/v1/sam3/detect",
            json={"image_path": image_path, "prompts": payload, "return_masks": True, "return_polygons": True},
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        return self._findings_from_detections(response.json().get("detections", []))

    def detect_with_points(
        self,
        image_path: str,
        point_prompts: list[dict[str, Any]],
    ) -> list[PictureFinding]:
        payload = []
        for item in point_prompts:
            point = item.get("point") or item.get("center_point")
            if not isinstance(point, list) or len(point) != 2:
                continue
            payload.append(
                {
                    "category": str(item.get("category") or "object"),
                    "text": str(item.get("text") or item.get("prompt") or "visual").strip() or "visual",
                    "point": {"x": float(point[0]), "y": float(point[1]), "label": bool(item.get("label", True))},
                    "threshold": float(item.get("threshold", self._confidence_threshold)),
                    "box_size_ratio": float(item.get("box_size_ratio", 0.06)),
                }
            )
        if not payload:
            return []
        response = httpx.post(
            f"{self._base_url}/v1/sam3/point-detect",
            json={"image_path": image_path, "point_prompts": payload, "return_masks": True, "return_polygons": True},
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        return self._findings_from_detections(response.json().get("detections", []))

    def refine_regions(self, image_path: str, regions: list[RegionMask]) -> list[RegionMask]:
        if not regions:
            return []
        payload = {
            "image_path": image_path,
            "threshold": self._confidence_threshold,
            "regions": [
                {
                    "bbox": region.bbox.model_dump(),
                    "confidence": region.confidence,
                }
                for region in regions
            ],
        }
        response = httpx.post(f"{self._base_url}/v1/sam3/refine", json=payload, timeout=self._timeout_seconds)
        response.raise_for_status()
        refined: list[RegionMask] = []
        for original, item in zip(regions, response.json().get("regions", [])):
            bbox_data = item.get("bbox") or {}
            refined.append(
                original.model_copy(
                    update={
                        "bbox": BBox(
                            x=float(bbox_data.get("x", original.bbox.x)),
                            y=float(bbox_data.get("y", original.bbox.y)),
                            w=float(bbox_data.get("w", original.bbox.w)),
                            h=float(bbox_data.get("h", original.bbox.h)),
                        ),
                        "polygon": _polygon_from_api(item.get("polygon")) or original.polygon,
                        "mask_path": str(item.get("mask_path") or original.mask_path or "") or None,
                        "confidence": max(original.confidence, float(item.get("confidence") or 0.0)),
                    }
                )
            )
        if len(refined) < len(regions):
            refined.extend(regions[len(refined):])
        return refined

    def _findings_from_detections(self, detections: list[dict[str, Any]]) -> list[PictureFinding]:
        findings: list[PictureFinding] = []
        for item in detections:
            bbox_data = item.get("bbox") or {}
            bbox = BBox(
                x=float(bbox_data.get("x", 0.0)),
                y=float(bbox_data.get("y", 0.0)),
                w=float(bbox_data.get("w", 0.0)),
                h=float(bbox_data.get("h", 0.0)),
            )
            score = float(item.get("score") or 0.0)
            category = str(item.get("category") or "object")
            polygon = _polygon_from_api(item.get("polygon"))
            findings.append(
                PictureFinding(
                    finding_type=FindingType.VISION_OBJECT,
                    category=category,
                    label=f"SAM3 detected {category}",
                    score=score,
                    region=RegionMask(
                        bbox=bbox,
                        polygon=polygon,
                        mask_path=str(item.get("mask_path") or "") or None,
                        confidence=score,
                    ),
                    reason_code=f"VISION_{category.upper()}",
                    provider=self.name,
                    threshold_used=float(item.get("threshold") or self._confidence_threshold),
                    explanation="SAM3 API prompt-guided visual localization.",
                    metadata=_detection_metadata(item, self._base_url),
                )
            )
        return _dedupe_findings(findings)

    def _select_prompts(
        self,
        target_types: list[str] | set[str] | tuple[str, ...] | None,
    ) -> dict[str, tuple[str, ...]]:
        if not target_types:
            return self._prompts
        selected = {str(item).strip().lower().replace(".", "_").replace("-", "_") for item in target_types if str(item).strip()}
        return {
            category: values
            for category, values in self._prompts.items()
            if category.lower().replace(".", "_").replace("-", "_") in selected
        }


def _normalize_prompt_map(
    prompts: dict[str, str | list[str] | tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    normalized: dict[str, tuple[str, ...]] = {}
    for category, value in prompts.items():
        if isinstance(value, str):
            items = (value,)
        else:
            items = tuple(str(item) for item in value)
        normalized[category] = tuple(item.strip() for item in items if item.strip())
    return normalized


def _merge_prompt_map(
    base: dict[str, tuple[str, ...]],
    extra: dict[str, list[str] | tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    merged = dict(base)
    for category, values in extra.items():
        normalized_category = str(category).strip().lower().replace(".", "_").replace("-", "_")
        existing = list(merged.get(normalized_category, ()))
        for value in values:
            prompt = str(value).strip()
            if prompt and prompt not in existing:
                existing.append(prompt)
        if existing:
            merged[normalized_category] = tuple(existing)
    return merged


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


def _detection_metadata(item: dict[str, Any], base_url: str) -> dict[str, Any]:
    metadata = {
        "prompt": item.get("prompt", ""),
        "prompt_type": item.get("prompt_type"),
        "point": item.get("point"),
        "point_prompts": item.get("point_prompts"),
        "point_anchor_bbox": item.get("point_anchor_bbox"),
        "point_anchor_bboxes": item.get("point_anchor_bboxes"),
        "sam3_api_url": base_url,
        "mask_path": item.get("mask_path"),
        "polygons": item.get("polygons"),
        "mask_area": item.get("mask_area"),
        "mask_area_ratio": item.get("mask_area_ratio"),
        "mask_bbox_fill_ratio": item.get("mask_bbox_fill_ratio"),
    }
    return {key: value for key, value in metadata.items() if value not in (None, "", [])}
