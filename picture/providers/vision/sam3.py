from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from picture.domain.enums import FindingType, VisionObjectType
from picture.domain.exceptions import ProviderNotAvailableError
from picture.domain.models import BBox, PictureFinding, RegionMask
from picture.providers.base import VisionDetector
from picture.providers.sam3_runtime import get_sam3_runtime

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = {
    VisionObjectType.FACE.value: (
        "face",
        "human face",
        "person face",
        "student face",
        "visible face",
        "portrait face",
        "ID photo face",
        "child face",
    ),
    VisionObjectType.ID_CARD.value: ("identity card", "student ID card", "exam ticket"),
    VisionObjectType.BADGE.value: ("badge", "name tag", "school badge"),
    VisionObjectType.SIGNATURE.value: ("signature", "handwritten signature"),
    VisionObjectType.STAMP.value: ("official stamp", "school seal", "red stamp"),
    VisionObjectType.QR_CODE.value: ("QR code",),
    VisionObjectType.BARCODE.value: ("barcode",),
    VisionObjectType.LICENSE_PLATE.value: ("license plate",),
    "avatar": ("profile avatar", "user portrait"),
    "account_region": ("username", "account ID", "profile header"),
    "school_class_identifier": ("school name", "classroom sign", "school logo"),
}


class SAM3SensitiveObjectDetector(VisionDetector):
    """
    Prompt-based sensitive object detector for the documented education pipeline.

    This class owns visual identity regions: faces, ID photos, badges, signatures,
    stamps, QR codes, barcodes and similar objects. It expects an installed SAM3
    runtime that can perform text-prompt image prediction from local weights.
    """

    def __init__(
        self,
        model_dir: str,
        confidence_threshold: float = 0.35,
        device: str = "auto",
        prompts: dict[str, str | list[str] | tuple[str, ...]] | None = None,
        **kwargs: Any,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._confidence_threshold = confidence_threshold
        self._device = device
        self._prompts = _normalize_prompt_map(prompts or DEFAULT_PROMPTS)
        self._kwargs = kwargs
        self._predictor: Any = None

    @property
    def name(self) -> str:
        return "SAM3SensitiveObjectDetector"

    def _get_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        self._predictor = get_sam3_runtime(self._model_dir, self._device)
        return self._predictor

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
        predictor = self._get_predictor()
        try:
            from PIL import Image
        except ImportError as exc:
            raise ProviderNotAvailableError("Pillow for SAM3") from exc

        image = Image.open(image_path).convert("RGB")

        prompts = self._select_prompts(target_types)
        if extra_prompts:
            prompts = _merge_prompt_map(prompts, extra_prompts)
        confidence_thresholds = confidence_thresholds or {}
        findings: list[PictureFinding] = []
        for category, category_prompts in prompts.items():
            category_findings: list[PictureFinding] = []
            category_threshold = float(confidence_thresholds.get(category, self._confidence_threshold))
            for prompt in category_prompts:
                if not self._prompt_fits_model(predictor, prompt):
                    logger.warning("Skip SAM3 prompt beyond text length limit: category=%s prompt=%r", category, prompt)
                    continue
                boxes, scores = self._predict_prompt(predictor, image, prompt)
                for box, score in zip(boxes, scores):
                    score_value = float(score)
                    if score_value < category_threshold:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box]
                    category_findings.append(
                        PictureFinding(
                            finding_type=FindingType.VISION_OBJECT,
                            category=category,
                            label=f"SAM3 detected {category}",
                            score=score_value,
                            region=RegionMask(
                                bbox=BBox(x=x1, y=y1, w=max(0.0, x2 - x1), h=max(0.0, y2 - y1)),
                                confidence=score_value,
                            ),
                            reason_code=f"VISION_{category.upper()}",
                            provider=self.name,
                            threshold_used=category_threshold,
                            explanation="SAM3 text-prompt sensitive object detection for education privacy governance.",
                            metadata={"prompt": prompt},
                        )
                    )
            findings.extend(_dedupe_findings(category_findings))
        return findings

    def _select_prompts(
        self,
        target_types: list[str] | set[str] | tuple[str, ...] | None,
    ) -> dict[str, tuple[str, ...]]:
        if not target_types:
            return self._prompts
        selected = {str(item).strip().lower() for item in target_types if str(item).strip()}
        if not selected:
            return self._prompts
        normalized = {item.replace(".", "_").replace("-", "_") for item in selected}
        return {
            category: prompt
            for category, prompt in self._prompts.items()
            if category.lower().replace(".", "_").replace("-", "_") in normalized
        }

    @staticmethod
    def _predict_prompt(predictor: dict[str, Any], image: Any, prompt: str) -> tuple[list[Any], list[float]]:
        processor = predictor["processor"]
        model = predictor["model"]
        device = predictor["device"]
        torch = predictor["torch"]

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        if hasattr(boxes, "detach"):
            boxes = boxes.detach().cpu().tolist()
        if hasattr(scores, "detach"):
            scores = scores.detach().cpu().tolist()
        return list(boxes), [float(item) for item in scores]

    @staticmethod
    def _prompt_fits_model(predictor: dict[str, Any], prompt: str) -> bool:
        tokenizer = getattr(predictor["processor"], "tokenizer", None)
        if tokenizer is None:
            return True
        model_max_length = getattr(tokenizer, "model_max_length", None)
        if not isinstance(model_max_length, int) or model_max_length <= 0 or model_max_length > 10_000:
            return True
        tokenized = tokenizer(prompt, return_tensors="pt", truncation=False)
        input_ids = tokenized.get("input_ids")
        if input_ids is None:
            return True
        return int(input_ids.shape[-1]) <= model_max_length


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
    for category, prompts in extra.items():
        normalized_category = str(category).strip().lower().replace(".", "_").replace("-", "_")
        if not normalized_category:
            continue
        existing = list(merged.get(normalized_category, ()))
        for prompt in prompts:
            text = str(prompt).strip()
            if text and text not in existing:
                existing.append(text)
        if existing:
            merged[normalized_category] = tuple(existing)
    return merged


def _dedupe_findings(findings: list[PictureFinding], iou_threshold: float = 0.82) -> list[PictureFinding]:
    kept: list[PictureFinding] = []
    for finding in sorted(findings, key=lambda item: item.score, reverse=True):
        if finding.region is None:
            kept.append(finding)
            continue
        duplicate_index = _duplicate_index(kept, finding, iou_threshold)
        if duplicate_index is not None:
            existing = kept[duplicate_index]
            if _prefer_candidate(finding, existing):
                kept[duplicate_index] = finding
            continue
        kept.append(finding)
    return kept


def _duplicate_index(
    kept: list[PictureFinding],
    candidate: PictureFinding,
    iou_threshold: float,
) -> int | None:
    if candidate.region is None:
        return None
    for index, existing in enumerate(kept):
        if existing.region is None:
            continue
        if candidate.category != existing.category or candidate.finding_type != existing.finding_type:
            continue
        candidate_box = candidate.region.bbox
        existing_box = existing.region.bbox
        if _bbox_iou(candidate_box, existing_box) >= iou_threshold:
            return index
        if _bbox_containment(candidate_box, existing_box) >= 0.85 and _bbox_area_ratio(candidate_box, existing_box) >= 2.0:
            return index
    return None


def _prefer_candidate(candidate: PictureFinding, existing: PictureFinding) -> bool:
    if candidate.region is None or existing.region is None:
        return candidate.score > existing.score
    if candidate.category == VisionObjectType.FACE.value and _bbox_containment(candidate.region.bbox, existing.region.bbox) >= 0.85:
        candidate_area = _bbox_area(candidate.region.bbox)
        existing_area = _bbox_area(existing.region.bbox)
        if candidate_area < existing_area:
            return candidate.score >= existing.score * 0.70
        if existing_area < candidate_area:
            return False
    return candidate.score > existing.score


def _bbox_iou(left: BBox, right: BBox) -> float:
    left_x2 = left.x + left.w
    left_y2 = left.y + left.h
    right_x2 = right.x + right.w
    right_y2 = right.y + right.h
    inter_w = max(0.0, min(left_x2, right_x2) - max(left.x, right.x))
    inter_h = max(0.0, min(left_y2, right_y2) - max(left.y, right.y))
    intersection = inter_w * inter_h
    union = left.w * left.h + right.w * right.h - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _bbox_containment(left: BBox, right: BBox) -> float:
    intersection = _bbox_intersection_area(left, right)
    smaller = min(_bbox_area(left), _bbox_area(right))
    if smaller <= 0:
        return 0.0
    return intersection / smaller


def _bbox_area_ratio(left: BBox, right: BBox) -> float:
    left_area = _bbox_area(left)
    right_area = _bbox_area(right)
    smaller = min(left_area, right_area)
    larger = max(left_area, right_area)
    if smaller <= 0:
        return 0.0
    return larger / smaller


def _bbox_intersection_area(left: BBox, right: BBox) -> float:
    left_x2 = left.x + left.w
    left_y2 = left.y + left.h
    right_x2 = right.x + right.w
    right_y2 = right.y + right.h
    inter_w = max(0.0, min(left_x2, right_x2) - max(left.x, right.x))
    inter_h = max(0.0, min(left_y2, right_y2) - max(left.y, right.y))
    return inter_w * inter_h


def _bbox_area(box: BBox) -> float:
    return max(0.0, box.w) * max(0.0, box.h)
