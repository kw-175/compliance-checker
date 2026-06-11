"""Temporal aggregation for video risk annotations."""

from __future__ import annotations

from copy import deepcopy

from video.domain.models import VideoRiskAnnotation


def aggregate_risks(
    risks: list[VideoRiskAnnotation],
    gap_tolerance_ms: int = 1000,
    iou_threshold: float = 0.4,
) -> list[VideoRiskAnnotation]:
    """Merge adjacent frame-level risks into timeline-level annotations."""
    aggregated: list[VideoRiskAnnotation] = []
    counters: dict[str, int] = {}
    for risk in sorted(risks, key=lambda item: (item.span.start_ms, item.category, item.source_modality)):
        merged = False
        for existing in reversed(aggregated):
            if not _can_merge(existing, risk, gap_tolerance_ms, iou_threshold):
                continue
            existing.span.end_ms = max(existing.span.end_ms, risk.span.end_ms)
            existing.confidence = max(existing.confidence, risk.confidence)
            existing.severity = _max_severity(existing.severity, risk.severity)
            existing.frame_ids = _dedupe(existing.frame_ids + risk.frame_ids)
            existing.regions.extend(deepcopy(risk.regions))
            existing.evidence_refs = _dedupe(existing.evidence_refs + risk.evidence_refs)
            existing.reason_codes = _dedupe(existing.reason_codes + risk.reason_codes)
            existing.recommended_actions = _dedupe(existing.recommended_actions + risk.recommended_actions)
            existing.metadata["merged_risk_count"] = int(existing.metadata.get("merged_risk_count") or 1) + 1
            if _is_object_instance(existing) or _is_object_instance(risk):
                existing.metadata["video_role"] = "object_instance"
                existing.metadata["instance_key"] = existing.metadata.get("instance_key") or _object_identity_key(existing)
            merged = True
            break
        if merged:
            continue
        item = risk.model_copy(deep=True)
        counters[item.category] = counters.get(item.category, 0) + 1
        item.track_id = item.track_id or f"track_{_safe_category(item.category)}_{counters[item.category]:04d}"
        aggregated.append(item)
    return aggregated


def risk_summary(risks: list[VideoRiskAnnotation]) -> dict[str, object]:
    by_category: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for risk in risks:
        by_category[risk.category] = by_category.get(risk.category, 0) + 1
        by_severity[risk.severity] = by_severity.get(risk.severity, 0) + 1
    return {
        "risk_count": len(risks),
        "by_category": by_category,
        "by_severity": by_severity,
    }


def _can_merge(left: VideoRiskAnnotation, right: VideoRiskAnnotation, gap_tolerance_ms: int, iou_threshold: float) -> bool:
    if left.source_modality != right.source_modality:
        return False
    if left.category != right.category:
        return False
    if (left.text_span or "") != (right.text_span or ""):
        return False
    if right.span.start_ms - left.span.end_ms > gap_tolerance_ms:
        return False
    if _is_object_instance(left) != _is_object_instance(right):
        return False
    if left.source_modality in {"visual", "picture", "safety", "visual_safety", "video_clip"}:
        return _visual_regions_compatible(left, right, iou_threshold)
    return _regions_compatible(left, right, iou_threshold)


def _visual_regions_compatible(left: VideoRiskAnnotation, right: VideoRiskAnnotation, iou_threshold: float) -> bool:
    left_box = _last_bbox(left)
    right_box = _first_bbox(right)
    if left_box is not None and right_box is not None:
        if _compute_iou(left_box, right_box) >= iou_threshold:
            return True
        return _object_instances_compatible(left, right, left_box, right_box)
    if _is_event_level_content_risk(left) and _is_event_level_content_risk(right):
        return True
    if left_box is None and right_box is None:
        left_frames = {str(item) for item in left.frame_ids if item}
        right_frames = {str(item) for item in right.frame_ids if item}
        return bool(left_frames and right_frames and left_frames.intersection(right_frames))
    return False


def _regions_compatible(left: VideoRiskAnnotation, right: VideoRiskAnnotation, iou_threshold: float) -> bool:
    left_box = _last_bbox(left)
    right_box = _first_bbox(right)
    if left_box is None or right_box is None:
        return True
    return _compute_iou(left_box, right_box) >= iou_threshold


def _first_bbox(risk: VideoRiskAnnotation) -> dict[str, float] | None:
    if not risk.regions:
        return None
    box = risk.regions[0].get("bbox")
    return box if isinstance(box, dict) else None


def _last_bbox(risk: VideoRiskAnnotation) -> dict[str, float] | None:
    if not risk.regions:
        return None
    box = risk.regions[-1].get("bbox")
    return box if isinstance(box, dict) else None


def _compute_iou(left: dict[str, float], right: dict[str, float]) -> float:
    lx, ly, lw, lh = float(left.get("x", 0)), float(left.get("y", 0)), float(left.get("w", 0)), float(left.get("h", 0))
    rx, ry, rw, rh = float(right.get("x", 0)), float(right.get("y", 0)), float(right.get("w", 0)), float(right.get("h", 0))
    x1 = max(lx, rx)
    y1 = max(ly, ry)
    x2 = min(lx + lw, rx + rw)
    y2 = min(ly + lh, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union else 0.0


def _is_event_level_content_risk(risk: VideoRiskAnnotation) -> bool:
    if _is_object_instance(risk):
        return False
    if not risk.category.startswith("content."):
        return False
    if any(isinstance(region, dict) and isinstance(region.get("bbox"), dict) for region in risk.regions):
        return False
    return risk.source_modality in {"safety", "visual_safety", "video_clip"}


def _is_object_instance(risk: VideoRiskAnnotation) -> bool:
    role = str(risk.metadata.get("video_role") or "").strip().lower()
    if role == "object_instance":
        return True
    if not any(isinstance(region, dict) and isinstance(region.get("bbox"), dict) for region in risk.regions):
        return False
    return risk.category.startswith("privacy.") or risk.source_modality in {"visual", "ocr_text", "video_object"}


def _object_instances_compatible(
    left: VideoRiskAnnotation,
    right: VideoRiskAnnotation,
    left_box: dict[str, float],
    right_box: dict[str, float],
) -> bool:
    if not (_is_object_instance(left) and _is_object_instance(right)):
        return False
    if _is_strict_iou_only_instance(left) or _is_strict_iou_only_instance(right):
        return False
    if _object_identity_key(left) != _object_identity_key(right):
        return False
    if _area_similarity(left_box, right_box) < 0.20:
        return False
    return _center_distance(left_box, right_box) <= max(120.0, _bbox_diagonal(left_box) * 3.0)


def _is_strict_iou_only_instance(risk: VideoRiskAnnotation) -> bool:
    if risk.category in {"privacy.face"}:
        return True
    if risk.source_modality == "ocr_text":
        return True
    return False


def _object_identity_key(risk: VideoRiskAnnotation) -> str:
    metadata = risk.metadata if isinstance(risk.metadata, dict) else {}
    label = str(
        metadata.get("instance_label_en")
        or metadata.get("entity_label_en")
        or metadata.get("instance_label_zh")
        or metadata.get("entity_label_zh")
        or metadata.get("label")
        or risk.target_type
        or risk.category
    ).strip().lower()
    return "|".join([risk.category, risk.source_operator_id, label])


def _center_distance(left: dict[str, float], right: dict[str, float]) -> float:
    lx = float(left.get("x", 0)) + float(left.get("w", 0)) / 2.0
    ly = float(left.get("y", 0)) + float(left.get("h", 0)) / 2.0
    rx = float(right.get("x", 0)) + float(right.get("w", 0)) / 2.0
    ry = float(right.get("y", 0)) + float(right.get("h", 0)) / 2.0
    return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5


def _bbox_diagonal(bbox: dict[str, float]) -> float:
    return (float(bbox.get("w", 0)) ** 2 + float(bbox.get("h", 0)) ** 2) ** 0.5


def _area_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    left_area = max(1.0, float(left.get("w", 0)) * float(left.get("h", 0)))
    right_area = max(1.0, float(right.get("w", 0)) * float(right.get("h", 0)))
    return min(left_area, right_area) / max(left_area, right_area)


def _max_severity(left: str, right: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    return left if order.get(left, 0) >= order.get(right, 0) else right


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _safe_category(category: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in category).strip("_") or "risk"
