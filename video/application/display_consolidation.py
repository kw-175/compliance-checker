"""Build platform-facing video risks from model-centered events and tracks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from video.domain.models import VideoRiskAnnotation


PHYSICAL_CONFLICT_CATEGORY = "content.graphic_violence"
PHYSICAL_CONFLICT_LABEL_ZH = "暴力斗殴"


def consolidate_display_risks(risks: list[VideoRiskAnnotation]) -> list[VideoRiskAnnotation]:
    """Return user-facing risks without mutating or hiding raw model records."""
    clip_events = [_as_display_event(risk, risks) for risk in risks if _is_clip_event(risk)]
    if clip_events:
        display = clip_events + [_as_display_copy(risk, "primary_privacy") for risk in risks if _is_identifiable_face(risk)]
        return _dedupe_display(display)

    display: list[VideoRiskAnnotation] = []
    for risk in risks:
        if _is_invalid_localization(risk):
            continue
        if _is_identifiable_face(risk):
            display.append(_as_display_copy(risk, "primary_privacy"))
            continue
        if _is_frame_event(risk):
            display.append(_as_display_copy(risk, "primary_event"))
            continue
        if _is_concrete_object(risk):
            display.append(_as_display_copy(risk, "primary_object"))
    return _dedupe_display(display)


def _as_display_event(event: VideoRiskAnnotation, risks: list[VideoRiskAnnotation]) -> VideoRiskAnnotation:
    item = event.model_copy(deep=True)
    item.metadata = dict(item.metadata or {})
    event_type_zh = _event_label_zh(item)
    item.metadata.update({
        "display_role": "primary_event",
        "display_label_zh": event_type_zh,
        "event_type_zh": event_type_zh,
        "event_source": "qwen_video_event",
    })
    supporting = [risk for risk in risks if risk.risk_id != event.risk_id and _supports_event(risk, event)]
    if supporting:
        item.metadata["supporting_risk_ids"] = [risk.risk_id for risk in supporting]
        item.metadata["supporting_risk_count"] = len(supporting)
        item.regions = _dedupe_regions([*item.regions, *[region for risk in supporting for region in _valid_regions(risk)]])
        item.frame_ids = _dedupe([*item.frame_ids, *[frame_id for risk in supporting for frame_id in risk.frame_ids]])
        item.evidence_refs = _dedupe([*item.evidence_refs, *[ref for risk in supporting for ref in risk.evidence_refs]])
        item.evidence_points = _dedupe_dicts([*item.evidence_points, *[point for risk in supporting for point in risk.evidence_points]])
        _attach_supporting_tracking(item, supporting)
        if item.regions:
            item.spatial_precision = "bbox"
            item.localization_status = "event_with_tracked_regions" if _has_sam3_region(item.regions) else "event_with_keyframe_regions"
            item.target_type = "conflict_region" if _is_physical_conflict_event(item) else item.target_type
            item.metadata.setdefault("trackable_target_type", item.target_type)
    return item


def _as_display_copy(risk: VideoRiskAnnotation, role: str) -> VideoRiskAnnotation:
    item = risk.model_copy(deep=True)
    item.metadata = dict(item.metadata or {})
    item.metadata["display_role"] = role
    if role == "primary_privacy":
        item.metadata.setdefault("display_label_zh", "可识别人脸")
    elif role == "primary_object":
        item.metadata.setdefault("display_label_zh", _object_label_zh(item))
    elif role == "primary_event":
        item.metadata.setdefault("display_label_zh", _event_label_zh(item))
    return item


def _is_clip_event(risk: VideoRiskAnnotation) -> bool:
    return risk.source_modality == "video_clip" and str(risk.metadata.get("video_role") or "event") == "event"


def _is_frame_event(risk: VideoRiskAnnotation) -> bool:
    if risk.source_modality not in {"safety", "visual_safety"}:
        return False
    if not risk.category.startswith("content."):
        return False
    return not any(isinstance(region, dict) and isinstance(region.get("bbox"), dict) for region in risk.regions)


def _supports_event(risk: VideoRiskAnnotation, event: VideoRiskAnnotation) -> bool:
    if not _overlaps(risk, event):
        return False
    if _is_invalid_localization(risk):
        return False
    if risk.metadata.get("parent_risk_id") == event.risk_id:
        return True
    if _is_physical_conflict_event(event) and _is_conflict_spatial_evidence(risk):
        return True
    if risk.source_modality == "video_object" and risk.category == event.category:
        return True
    if risk.source_modality in {"safety", "visual_safety"} and risk.category == event.category:
        return True
    return False


def _is_physical_conflict_event(risk: VideoRiskAnnotation) -> bool:
    return risk.category in {PHYSICAL_CONFLICT_CATEGORY, "content.violence"} or any(
        token in _metadata_text(risk)
        for token in ("physical_conflict", "肢体冲突", "斗殴", "打架", "推搡", "拉扯")
    )


def _is_conflict_spatial_evidence(risk: VideoRiskAnnotation) -> bool:
    if risk.category not in {PHYSICAL_CONFLICT_CATEGORY, "content.violence"}:
        return False
    if not _valid_regions(risk):
        return False
    text = _metadata_text(risk)
    return any(token in text for token in ("conflict_region", "fighting group", "fight", "打斗", "斗殴", "肢体冲突", "推搡", "拉扯"))


def _is_identifiable_face(risk: VideoRiskAnnotation) -> bool:
    if _norm(risk.category) != "privacy.face" and _norm(risk.target_type) != "face":
        return False
    if _is_invalid_localization(risk):
        return False
    metadata = risk.metadata if isinstance(risk.metadata, dict) else {}
    if str(metadata.get("face_filter_decision") or "").strip().lower() == "drop":
        return False
    if not any(isinstance(region, dict) and isinstance(region.get("bbox"), dict) for region in risk.regions):
        return False
    identifiability = _safe_float(metadata.get("identifiability_score"), 0.0)
    if identifiability < 0.70:
        return False
    return bool(metadata.get("is_identifiable_face", False)) or str(metadata.get("face_filter_decision") or "").strip().lower() == "keep"


def _is_concrete_object(risk: VideoRiskAnnotation) -> bool:
    if risk.source_modality not in {"visual", "ocr_text", "video_object"}:
        return False
    if risk.category == "visual.dangerous":
        return bool(_valid_regions(risk)) and not _is_abstract_dangerous(risk)
    return bool(_valid_regions(risk))


def _is_invalid_localization(risk: VideoRiskAnnotation) -> bool:
    region_items = [region for region in risk.regions if isinstance(region, dict)]
    valid_region_count = len(_valid_regions(risk)) if region_items else 0
    if region_items and valid_region_count == 0:
        return True
    metadata = risk.metadata if isinstance(risk.metadata, dict) else {}
    if _metadata_marks_invalid(metadata) and valid_region_count == 0:
        return True
    if risk.category == "visual.dangerous" and _is_abstract_dangerous(risk):
        return True
    if risk.category == "privacy.face" and not any(isinstance(region, dict) and isinstance(region.get("bbox"), dict) for region in risk.regions):
        return True
    return False


def _is_abstract_dangerous(risk: VideoRiskAnnotation) -> bool:
    text = _metadata_text(risk)
    return any(token in text for token in ("肢体冲突", "斗殴", "打架", "暴力行为", "多人冲突", "physical_conflict", "fight"))


def _event_label_zh(risk: VideoRiskAnnotation) -> str:
    text = _metadata_text(risk)
    if risk.category == PHYSICAL_CONFLICT_CATEGORY or any(token in text for token in ("physical_conflict", "肢体冲突", "斗殴", "打架")):
        return PHYSICAL_CONFLICT_LABEL_ZH
    labels = {
        "content.sexual": "色情裸露",
        "content.hate": "仇恨极端内容",
        "content.self_harm": "自伤风险",
    }
    return labels.get(risk.category, "视频风险事件")


def _object_label_zh(risk: VideoRiskAnnotation) -> str:
    metadata = risk.metadata if isinstance(risk.metadata, dict) else {}
    for key in ("instance_label_zh", "object_name_zh", "label"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return risk.target_type or risk.category


def _valid_regions(risk: VideoRiskAnnotation) -> list[dict[str, Any]]:
    return [deepcopy(region) for region in risk.regions if isinstance(region, dict) and isinstance(region.get("bbox"), dict) and not _region_is_invalid(region)]


def _region_is_invalid(region: dict[str, Any]) -> bool:
    return _metadata_marks_invalid(region)


def _metadata_marks_invalid(metadata: dict[str, Any]) -> bool:
    source = str(metadata.get("source") or "").strip().lower()
    status = str(metadata.get("localization_status") or "").strip().lower()
    if source in {"qwen_rough_bbox_fallback", "rough_bbox_fallback"}:
        return True
    if "coarse_localization" in status or "sam3_rejections" in status:
        return True
    quality = metadata.get("mask_quality_score")
    if quality is not None and _safe_float(quality, 1.0) < 0.35:
        return True
    return False


def _has_sam3_region(regions: list[dict[str, Any]]) -> bool:
    return any(str(region.get("source") or "") == "sam3_video_tracker" for region in regions)


def _attach_supporting_tracking(item: VideoRiskAnnotation, supporting: list[VideoRiskAnnotation]) -> None:
    bbox_series: list[dict[str, Any]] = []
    redaction_series: list[dict[str, Any]] = []
    mask_keyframes: list[dict[str, Any]] = []
    quality_flags: list[str] = []
    backends: list[str] = []
    for risk in supporting:
        tracking = risk.metadata.get("tracking") if isinstance(risk.metadata.get("tracking"), dict) else {}
        bbox_series.extend(_series_from_tracking(tracking, "bbox_series"))
        redaction_series.extend(_series_from_tracking(tracking, "redaction_series"))
        mask_keyframes.extend(_series_from_tracking(tracking, "mask_keyframes"))
        quality_flags.extend(str(flag) for flag in tracking.get("quality_flags", []) if flag)
        backend = str(risk.metadata.get("tracking_backend") or tracking.get("tracking_backend") or "")
        if backend:
            backends.append(backend)
    region_series = [
        deepcopy(region)
        for risk in supporting
        for region in _valid_regions(risk)
        if isinstance(region.get("bbox"), dict)
    ]
    bbox_series = _dedupe_regions([*bbox_series, *region_series])
    redaction_series = _dedupe_regions([*redaction_series, *region_series])
    if not bbox_series and not redaction_series:
        return
    tracking_backend = "sam3_video_tracker" if any(backend == "sam3_video_tracker" for backend in backends) else (backends[0] if backends else "qwen_keyframe_regions")
    existing = item.metadata.get("tracking") if isinstance(item.metadata.get("tracking"), dict) else {}
    existing.update({
        "method": "event_spatial_evidence_fusion",
        "tracking_backend": tracking_backend,
        "bbox_series": _sort_series(bbox_series),
        "redaction_series": _sort_series(redaction_series),
        "mask_keyframes": _sort_series(mask_keyframes),
        "quality_flags": _dedupe([*quality_flags, *existing.get("quality_flags", [])]) if isinstance(existing.get("quality_flags"), list) else _dedupe(quality_flags),
        "has_spatial_track": True,
        "redaction_scope": "sam3_video_track" if tracking_backend == "sam3_video_tracker" else "sampled_frame_track",
        "redaction_ready": False,
    })
    item.metadata["tracking"] = existing
    item.metadata["tracking_backend"] = tracking_backend


def _series_from_tracking(tracking: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = tracking.get(key)
    return [deepcopy(item) for item in value if isinstance(item, dict) and isinstance(item.get("bbox"), dict)] if isinstance(value, list) else []


def _sort_series(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(series, key=lambda item: (_safe_float(item.get("pts_ms"), 0.0), str(item.get("frame_id") or "")))


def _overlaps(left: VideoRiskAnnotation, right: VideoRiskAnnotation, tolerance_ms: int = 250) -> bool:
    left_span = left.display_span or left.span
    right_span = right.display_span or right.span
    return left_span.start_ms <= right_span.end_ms + tolerance_ms and right_span.start_ms <= left_span.end_ms + tolerance_ms


def _dedupe_display(risks: list[VideoRiskAnnotation]) -> list[VideoRiskAnnotation]:
    seen: set[str] = set()
    result: list[VideoRiskAnnotation] = []
    for risk in sorted(risks, key=lambda item: ((item.display_span or item.span).start_ms, _display_rank(item), item.risk_id)):
        key = risk.risk_id
        if key in seen:
            continue
        seen.add(key)
        result.append(risk)
    return result


def _display_rank(risk: VideoRiskAnnotation) -> int:
    role = str(risk.metadata.get("display_role") or "")
    return {"primary_event": 0, "primary_privacy": 1, "primary_object": 2}.get(role, 3)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _dedupe_dicts(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for value in values:
        key = repr(sorted(value.items()))
        if key in seen:
            continue
        seen.add(key)
        result.append(deepcopy(value))
    return result


def _dedupe_regions(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    result: list[dict[str, Any]] = []
    for region in values:
        bbox = region.get("bbox") if isinstance(region.get("bbox"), dict) else {}
        key = (
            str(region.get("frame_id") or ""),
            (
                round(_safe_float(bbox.get("x"), 0.0) * 1000),
                round(_safe_float(bbox.get("y"), 0.0) * 1000),
                round(_safe_float(bbox.get("w"), 0.0) * 1000),
                round(_safe_float(bbox.get("h"), 0.0) * 1000),
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(deepcopy(region))
    return result


def _metadata_text(risk: VideoRiskAnnotation) -> str:
    metadata = risk.metadata if isinstance(risk.metadata, dict) else {}
    values: list[str] = [risk.category, risk.target_type, *risk.reason_codes, str(risk.text_span or "")]
    raw_event = metadata.get("raw_event") if isinstance(metadata.get("raw_event"), dict) else {}
    values.extend(str(raw_event.get(key) or "") for key in ("category", "label", "evidence"))
    for key in (
        "instance_label_zh",
        "instance_label_en",
        "risk_subtype_zh",
        "risk_subtype",
        "object_name_zh",
        "label",
        "explanation",
        "redaction_target",
    ):
        value = metadata.get(key)
        if value is not None:
            values.append(str(value))
    return " ".join(values).lower()


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
