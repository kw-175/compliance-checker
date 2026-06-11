"""Risk-track enrichment for video compliance results."""

from __future__ import annotations

from typing import Any

from video.domain.models import FrameReference, TimeSpan, VideoRiskAnnotation


def enrich_risk_tracks(
    risks: list[VideoRiskAnnotation],
    frames: list[FrameReference],
    gap_tolerance_ms: int = 1000,
    iou_threshold: float = 0.35,
) -> None:
    """Attach bbox series and interpolation metadata to risk annotations."""
    frame_by_id = {frame.frame_id: frame for frame in frames}
    for risk in risks:
        bbox_series = _bbox_series(risk, frame_by_id)
        interpolated = _interpolate_bbox_series(bbox_series, frames, risk.span.start_ms, risk.span.end_ms, gap_tolerance_ms)
        display_span = _display_span(risk, bbox_series, frame_by_id)
        temporal_precision = _temporal_precision(risk, bbox_series, frame_by_id)
        spatial_precision = _spatial_precision(risk, bbox_series)
        evidence_points = _evidence_points(risk, bbox_series, frame_by_id)
        redaction_series = _redaction_series(bbox_series, interpolated)
        mask_keyframes = _mask_keyframes(risk, frame_by_id)
        tracking_backend = str(risk.metadata.get("tracking_backend") or "sam3_image_iou_interpolation")
        redaction_scope = _redaction_scope(risk, bbox_series, interpolated, mask_keyframes, tracking_backend)
        quality_flags = _track_quality_flags(risk, bbox_series, interpolated, redaction_scope, tracking_backend)
        if _has_blocking_quality_flags(quality_flags):
            redaction_scope = "manual_review"
        redaction_ready = (
            redaction_scope in {"sampled_frame", "sampled_frame_track", "interpolated_track", "sam3_video_track", "mask_keyframes", "audio_segment"}
            and not _has_blocking_quality_flags(quality_flags)
        )
        risk.display_span = display_span
        risk.temporal_precision = temporal_precision
        risk.spatial_precision = spatial_precision
        risk.localization_status = _localization_status(temporal_precision, spatial_precision)
        risk.evidence_points = evidence_points
        risk.metadata["tracking"] = {
            "track_id": risk.track_id or "",
            "method": "category_iou_temporal_interpolation",
            "iou_threshold": iou_threshold,
            "gap_tolerance_ms": gap_tolerance_ms,
            "bbox_series": bbox_series,
            "interpolated_bbox_series": interpolated,
            "display_span": display_span.model_dump(mode="json"),
            "temporal_precision": temporal_precision,
            "spatial_precision": spatial_precision,
            "localization_status": risk.localization_status,
            "evidence_points": evidence_points,
            "redaction_ready": redaction_ready,
            "redaction_scope": redaction_scope,
            "redaction_series": redaction_series,
            "mask_keyframes": mask_keyframes,
            "quality_flags": quality_flags,
            "tracking_backend": tracking_backend,
            "detection_count": len(bbox_series),
            "interpolated_count": len(interpolated),
            "has_spatial_track": bool(bbox_series),
            "requires_full_frame_review": spatial_precision == "full_frame",
        }


def build_risk_tracks(risks: list[VideoRiskAnnotation], frames: list[FrameReference]) -> list[dict[str, Any]]:
    """Return platform-friendly track records with temporal and spatial series."""
    frame_by_id = {frame.frame_id: frame for frame in frames}
    tracks: list[dict[str, Any]] = []
    for index, risk in enumerate(risks):
        if not risk.track_id:
            risk.track_id = f"track_{index + 1:04d}"
        tracking = risk.metadata.get("tracking") if isinstance(risk.metadata.get("tracking"), dict) else {}
        bbox_series = tracking.get("bbox_series") if isinstance(tracking.get("bbox_series"), list) else _bbox_series(risk, frame_by_id)
        interpolated = tracking.get("interpolated_bbox_series") if isinstance(tracking.get("interpolated_bbox_series"), list) else []
        mask_keyframes = tracking.get("mask_keyframes") if isinstance(tracking.get("mask_keyframes"), list) else _mask_keyframes(risk, frame_by_id)
        redaction_series = tracking.get("redaction_series") if isinstance(tracking.get("redaction_series"), list) else _redaction_series(bbox_series, interpolated)
        tracks.append({
            "track_id": risk.track_id,
            "risk_id": risk.risk_id,
            "video_role": risk.metadata.get("video_role", "event" if risk.category.startswith("content.") and not bbox_series else "object_instance"),
            "parent_risk_id": risk.metadata.get("parent_risk_id", ""),
            "instance_label_zh": risk.metadata.get("instance_label_zh", ""),
            "instance_label_en": risk.metadata.get("instance_label_en", ""),
            "category": risk.category,
            "source_modality": risk.source_modality,
            "operator_id": risk.operator_id,
            "source_operator_id": risk.source_operator_id,
            "target_type": risk.target_type,
            "severity": risk.severity,
            "confidence": risk.confidence,
            "span": risk.span.model_dump(mode="json"),
            "display_span": (risk.display_span or risk.span).model_dump(mode="json"),
            "temporal_precision": risk.temporal_precision or tracking.get("temporal_precision", ""),
            "spatial_precision": risk.spatial_precision or tracking.get("spatial_precision", ""),
            "localization_status": risk.localization_status or tracking.get("localization_status", ""),
            "evidence_points": risk.evidence_points or tracking.get("evidence_points", []),
            "frame_ids": risk.frame_ids,
            "representative_frame_id": risk.metadata.get("representative_frame_id", ""),
            "representative_frame_uri": risk.metadata.get("representative_frame_uri", ""),
            "region_count": len(risk.regions),
            "bbox_series": bbox_series,
            "interpolated_bbox_series": interpolated,
            "mask_keyframes": mask_keyframes,
            "redaction_ready": bool(tracking.get("redaction_ready", False)) if isinstance(tracking, dict) else False,
            "redaction_scope": tracking.get("redaction_scope", "") if isinstance(tracking, dict) else "",
            "redaction_series": redaction_series,
            "quality_flags": tracking.get("quality_flags", []) if isinstance(tracking.get("quality_flags"), list) else [],
            "tracking_backend": tracking.get("tracking_backend", "sam3_image_iou_interpolation") if isinstance(tracking, dict) else "sam3_image_iou_interpolation",
            "audio_text": (risk.audio_segment or {}).get("text", "") if risk.audio_segment else "",
            "text_span": risk.text_span or "",
            "tracking_method": tracking.get("method", "none") if isinstance(tracking, dict) else "none",
            "requires_full_frame_review": bool(tracking.get("requires_full_frame_review", False)) if isinstance(tracking, dict) else False,
        })
    return tracks


def _display_span(risk: VideoRiskAnnotation, bbox_series: list[dict[str, Any]], frame_by_id: dict[str, FrameReference]) -> TimeSpan:
    if bbox_series:
        points = [(int(point.get("pts_ms", 0) or 0), _duration_for_point(point, frame_by_id)) for point in bbox_series]
        start = min(point_time for point_time, _ in points)
        end = max(point_time + max(1, duration) for point_time, duration in points)
        return TimeSpan(start_ms=max(0, int(start)), end_ms=max(int(start) + 1, int(end)))
    frame_points = _frame_points(risk, frame_by_id)
    if frame_points:
        start = min(point_time for point_time, _ in frame_points)
        end = max(point_time + max(1, duration) for point_time, duration in frame_points)
        return TimeSpan(start_ms=max(0, int(start)), end_ms=max(int(start) + 1, int(end)))
    return TimeSpan(start_ms=risk.span.start_ms, end_ms=max(risk.span.start_ms + 1, risk.span.end_ms))


def _temporal_precision(risk: VideoRiskAnnotation, bbox_series: list[dict[str, Any]], frame_by_id: dict[str, FrameReference]) -> str:
    if risk.source_modality == "audio":
        return "precise"
    if bbox_series:
        return "precise" if len(bbox_series) > 1 else "frame"
    if _frame_points(risk, frame_by_id):
        return "frame"
    if risk.source_modality == "video_clip":
        return "window"
    return "global"


def _spatial_precision(risk: VideoRiskAnnotation, bbox_series: list[dict[str, Any]]) -> str:
    if risk.source_modality == "audio":
        return "none"
    if bbox_series:
        return "bbox"
    if any(isinstance(region, dict) and region.get("mask_path") for region in risk.regions):
        return "mask"
    return "full_frame"


def _localization_status(temporal_precision: str, spatial_precision: str) -> str:
    if temporal_precision == "precise" and spatial_precision in {"bbox", "mask"}:
        return "precise_spatial_temporal"
    if temporal_precision == "frame":
        return "frame_review"
    if temporal_precision == "window":
        return "window_review"
    if spatial_precision == "full_frame":
        return "full_frame_review"
    return "localized"


def _evidence_points(risk: VideoRiskAnnotation, bbox_series: list[dict[str, Any]], frame_by_id: dict[str, FrameReference]) -> list[dict[str, Any]]:
    if bbox_series:
        return bbox_series
    points: list[dict[str, Any]] = []
    for frame_id in risk.frame_ids:
        frame = frame_by_id.get(str(frame_id))
        if frame:
            points.append({"frame_id": frame.frame_id, "pts_ms": frame.pts_ms, "source": "frame"})
    if not points and risk.metadata.get("representative_frame_id"):
        points.append({
            "frame_id": str(risk.metadata.get("representative_frame_id") or ""),
            "pts_ms": int(risk.metadata.get("representative_frame_pts_ms") or risk.span.start_ms),
            "source": "representative_frame",
        })
    points.sort(key=lambda item: (int(item.get("pts_ms", 0)), str(item.get("frame_id", ""))))
    return points


def _redaction_series(bbox_series: list[dict[str, Any]], interpolated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    series = []
    for point in bbox_series + interpolated:
        if not isinstance(point, dict) or not isinstance(point.get("bbox"), dict):
            continue
        series.append({
            "frame_id": str(point.get("frame_id") or ""),
            "pts_ms": int(point.get("pts_ms", 0) or 0),
            "bbox": point.get("bbox"),
            "confidence": float(point.get("confidence", 0.0) or 0.0),
            "source": str(point.get("source") or "detected"),
        })
    series.sort(key=lambda item: (int(item.get("pts_ms", 0)), str(item.get("frame_id", "")), str(item.get("source", ""))))
    return series


def _redaction_scope(
    risk: VideoRiskAnnotation,
    bbox_series: list[dict[str, Any]],
    interpolated: list[dict[str, Any]],
    mask_keyframes: list[dict[str, Any]],
    tracking_backend: str = "sam3_image_iou_interpolation",
) -> str:
    if risk.source_modality == "audio" and risk.audio_segment:
        return "audio_segment"
    if tracking_backend == "sam3_video_tracker" and len(bbox_series) >= 2:
        return "sam3_video_track"
    if mask_keyframes:
        return "mask_keyframes"
    if len(bbox_series) >= 2:
        return "interpolated_track" if interpolated else "sampled_frame_track"
    if len(bbox_series) == 1:
        return "sampled_frame"
    return "manual_review"


def _track_quality_flags(
    risk: VideoRiskAnnotation,
    bbox_series: list[dict[str, Any]],
    interpolated: list[dict[str, Any]],
    redaction_scope: str,
    tracking_backend: str,
) -> list[str]:
    flags: list[str] = []
    if redaction_scope == "manual_review":
        flags.append("missing_spatial_localization")
    if redaction_scope == "sampled_frame":
        flags.append("single_frame_localization")
    if redaction_scope == "sampled_frame_track" and not interpolated:
        flags.append("sampled_frames_only")
    if bbox_series:
        low_confidence = [point for point in bbox_series if float(point.get("confidence", risk.confidence) or 0.0) < 0.45]
        if low_confidence:
            flags.append("low_confidence_spatial_points")
    if risk.source_modality == "video_clip" and not bbox_series:
        flags.append("semantic_window_without_region")
    flags.extend(_coarse_localization_flags(risk, bbox_series))
    if tracking_backend == "sam3_video_tracker":
        flags.extend(_sam3_tracker_flags(risk, bbox_series))
    return _dedupe_strings(flags)


def _coarse_localization_flags(risk: VideoRiskAnnotation, bbox_series: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    metadata = risk.metadata if isinstance(risk.metadata, dict) else {}
    if _metadata_marks_coarse_localization(metadata):
        flags.append("low_quality_coarse_localization")
    evidence_regions = metadata.get("evidence_regions")
    if isinstance(evidence_regions, list):
        for region in evidence_regions:
            if isinstance(region, dict) and _metadata_marks_coarse_localization(region):
                flags.append("low_quality_coarse_localization")
                break
    for point in bbox_series:
        if _metadata_marks_coarse_localization(point):
            flags.append("low_quality_coarse_localization")
            break
    return flags


def _sam3_tracker_flags(risk: VideoRiskAnnotation, bbox_series: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    tracker_meta = risk.metadata.get("sam3_video_tracking") if isinstance(risk.metadata.get("sam3_video_tracking"), dict) else {}
    meta_flags = tracker_meta.get("quality_flags") if isinstance(tracker_meta.get("quality_flags"), list) else []
    flags.extend(str(flag) for flag in meta_flags if flag)
    if tracker_meta.get("skipped"):
        flags.append("sam3_track_skipped")
    frame_counts: dict[str, int] = {}
    for point in bbox_series:
        frame_id = str(point.get("frame_id") or "")
        if not frame_id:
            continue
        frame_counts[frame_id] = frame_counts.get(frame_id, 0) + 1
    if any(count > 1 for count in frame_counts.values()):
        flags.append("duplicate_sam3_points_same_frame")
    if _has_unstable_bbox_transition(bbox_series):
        flags.append("unstable_sam3_track")
    return flags


def _has_unstable_bbox_transition(bbox_series: list[dict[str, Any]]) -> bool:
    if len(bbox_series) < 2:
        return False
    for left, right in zip(bbox_series, bbox_series[1:]):
        left_box = left.get("bbox") if isinstance(left.get("bbox"), dict) else None
        right_box = right.get("bbox") if isinstance(right.get("bbox"), dict) else None
        if left_box is None or right_box is None:
            continue
        left_norm = _normalize_bbox(left_box)
        right_norm = _normalize_bbox(right_box)
        if _bbox_iou(left_norm, right_norm) >= 0.02:
            continue
        if _area_similarity(left_norm, right_norm) < 0.18:
            return True
        if _center_distance(left_norm, right_norm) > max(80.0, _bbox_diagonal(left_norm) * 4.0):
            return True
    return False


def _metadata_marks_coarse_localization(metadata: dict[str, Any]) -> bool:
    source = str(metadata.get("source") or "").strip().lower()
    status = str(metadata.get("localization_status") or "").strip().lower()
    if source in {"qwen_rough_bbox_fallback", "rough_bbox_fallback"}:
        return True
    if "coarse_localization" in status or "sam3_rejections" in status:
        return True
    quality = metadata.get("mask_quality_score")
    if quality is not None:
        try:
            if float(quality) < 0.35:
                return True
        except (TypeError, ValueError):
            return False
    return False


def _has_blocking_quality_flags(flags: list[str]) -> bool:
    blocking = {
        "ambiguous_sam3_points_same_frame",
        "duplicate_sam3_points_same_frame",
        "low_quality_coarse_localization",
        "sam3_track_skipped",
        "unstable_sam3_track",
    }
    return any(flag in blocking for flag in flags)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _bbox_iou(left: dict[str, float], right: dict[str, float]) -> float:
    lx, ly, lw, lh = left["x"], left["y"], left["w"], left["h"]
    rx, ry, rw, rh = right["x"], right["y"], right["w"], right["h"]
    x1 = max(lx, rx)
    y1 = max(ly, ry)
    x2 = min(lx + lw, rx + rw)
    y2 = min(ly + lh, ry + rh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance(left: dict[str, float], right: dict[str, float]) -> float:
    lx = left["x"] + left["w"] / 2.0
    ly = left["y"] + left["h"] / 2.0
    rx = right["x"] + right["w"] / 2.0
    ry = right["y"] + right["h"] / 2.0
    return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5


def _bbox_diagonal(bbox: dict[str, float]) -> float:
    return (bbox["w"] ** 2 + bbox["h"] ** 2) ** 0.5


def _area_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    left_area = max(1.0, left["w"] * left["h"])
    right_area = max(1.0, right["w"] * right["h"])
    return min(left_area, right_area) / max(left_area, right_area)


def _frame_points(risk: VideoRiskAnnotation, frame_by_id: dict[str, FrameReference]) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for frame_id in risk.frame_ids:
        frame = frame_by_id.get(str(frame_id))
        if frame:
            points.append((frame.pts_ms, _frame_duration(frame)))
    if not points and risk.metadata.get("representative_frame_id"):
        points.append((int(risk.metadata.get("representative_frame_pts_ms") or risk.span.start_ms), max(1, risk.span.end_ms - risk.span.start_ms)))
    return points


def _frame_duration(frame: FrameReference) -> int:
    value = frame.metadata.get("duration_ms", 0)
    try:
        return max(1, int(value or 0))
    except (TypeError, ValueError):
        return 1


def _duration_for_point(point: dict[str, Any], frame_by_id: dict[str, FrameReference]) -> int:
    frame = frame_by_id.get(str(point.get("frame_id") or ""))
    return _frame_duration(frame) if frame else 1


def _bbox_series(risk: VideoRiskAnnotation, frame_by_id: dict[str, FrameReference]) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for index, region in enumerate(risk.regions):
        if not isinstance(region, dict):
            continue
        bbox = region.get("bbox")
        if not isinstance(bbox, dict):
            continue
        frame_id = str(region.get("frame_id") or "")
        frame = frame_by_id.get(frame_id)
        pts_ms = frame.pts_ms if frame else _fallback_pts(risk, index)
        series.append({
            "frame_id": frame_id,
            "pts_ms": pts_ms,
            "bbox": _normalize_bbox(bbox),
            "confidence": float(region.get("confidence", risk.confidence) or 0.0),
            "source": str(region.get("source") or "detected"),
        })
    series.sort(key=lambda item: (int(item.get("pts_ms", 0)), str(item.get("frame_id", ""))))
    return series


def _interpolate_bbox_series(
    bbox_series: list[dict[str, Any]],
    frames: list[FrameReference],
    start_ms: int,
    end_ms: int,
    gap_tolerance_ms: int,
) -> list[dict[str, Any]]:
    if len(bbox_series) < 2:
        return []
    existing_frame_ids = {str(item.get("frame_id") or "") for item in bbox_series}
    interpolated: list[dict[str, Any]] = []
    for left, right in zip(bbox_series, bbox_series[1:]):
        left_time = int(left.get("pts_ms", 0) or 0)
        right_time = int(right.get("pts_ms", 0) or 0)
        if right_time <= left_time or right_time - left_time > max(1, gap_tolerance_ms * 4):
            continue
        left_box = left.get("bbox") if isinstance(left.get("bbox"), dict) else {}
        right_box = right.get("bbox") if isinstance(right.get("bbox"), dict) else {}
        for frame in frames:
            if frame.frame_id in existing_frame_ids:
                continue
            if frame.pts_ms <= left_time or frame.pts_ms >= right_time:
                continue
            if frame.pts_ms < start_ms or frame.pts_ms > end_ms:
                continue
            ratio = (frame.pts_ms - left_time) / max(1, right_time - left_time)
            interpolated.append({
                "frame_id": frame.frame_id,
                "pts_ms": frame.pts_ms,
                "bbox": {
                    "x": _lerp(left_box.get("x", 0), right_box.get("x", 0), ratio),
                    "y": _lerp(left_box.get("y", 0), right_box.get("y", 0), ratio),
                    "w": _lerp(left_box.get("w", 0), right_box.get("w", 0), ratio),
                    "h": _lerp(left_box.get("h", 0), right_box.get("h", 0), ratio),
                },
                "confidence": min(float(left.get("confidence", 0) or 0), float(right.get("confidence", 0) or 0)),
                "source": "interpolated",
            })
    interpolated.sort(key=lambda item: (int(item.get("pts_ms", 0)), str(item.get("frame_id", ""))))
    return interpolated


def _mask_keyframes(risk: VideoRiskAnnotation, frame_by_id: dict[str, FrameReference]) -> list[dict[str, Any]]:
    keyframes: list[dict[str, Any]] = []
    for region in risk.regions:
        if not isinstance(region, dict) or not region.get("mask_path"):
            continue
        frame_id = str(region.get("frame_id") or "")
        frame = frame_by_id.get(frame_id)
        keyframes.append({
            "frame_id": frame_id,
            "pts_ms": frame.pts_ms if frame else None,
            "mask_path": region.get("mask_path"),
            "polygon": region.get("polygon"),
        })
    return keyframes


def _fallback_pts(risk: VideoRiskAnnotation, index: int) -> int:
    if not risk.regions:
        return risk.span.start_ms
    span = max(1, risk.span.end_ms - risk.span.start_ms)
    return risk.span.start_ms + round(span * index / max(1, len(risk.regions) - 1))


def _normalize_bbox(bbox: dict[str, Any]) -> dict[str, float]:
    return {
        "x": float(bbox.get("x", 0) or 0),
        "y": float(bbox.get("y", 0) or 0),
        "w": float(bbox.get("w", 0) or 0),
        "h": float(bbox.get("h", 0) or 0),
    }


def _lerp(left: Any, right: Any, ratio: float) -> float:
    return float(left or 0) + (float(right or 0) - float(left or 0)) * ratio
