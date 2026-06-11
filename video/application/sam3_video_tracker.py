"""Optional SAM3 video tracker integration for visual risk tracks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from video.domain.models import FrameReference, VideoRiskAnnotation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sam3VideoTrackerConfig:
    """Connection settings for a SAM3 video tracking service."""

    base_url: str = "http://127.0.0.1:8218"
    endpoint: str = "/v1/sam3/video-track"
    timeout_seconds: int = 300
    fail_on_error: bool = False
    return_masks: bool = True


def enrich_with_sam3_video_tracking(
    risks: list[VideoRiskAnnotation],
    frames: list[FrameReference],
    config: Sam3VideoTrackerConfig,
) -> dict[str, Any]:
    """Use SAM3 video tracking to expand seeded visual risks across frames.

    This adapter intentionally treats SAM3 video tracking as an optional
    external capability. If the endpoint is unavailable, the caller can keep
    the existing sampled-frame/Iou interpolation path.
    """
    candidates = _tracking_candidates(risks)
    report: dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "backend": "sam3_video_tracker",
        "endpoint": _join_url(config.base_url, config.endpoint),
        "candidate_count": len(candidates),
        "updated_risk_count": 0,
        "point_count": 0,
    }
    if not frames:
        report["reason"] = "no_frames"
        return report
    if not candidates:
        report["reason"] = "no_seeded_visual_risks"
        return report

    payload = _build_payload(frames, candidates, config)
    try:
        import httpx

        response = httpx.post(report["endpoint"], json=payload, timeout=config.timeout_seconds)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        report.update({"reason": "tracker_request_failed", "error": str(exc)})
        _mark_tracker_fallback(candidates, report)
        if config.fail_on_error:
            raise
        logger.warning("SAM3 video tracker unavailable, falling back to sampled-frame tracking: %s", exc)
        return report

    updated_count, point_count = _apply_tracker_response(candidates, frames, data)
    report.update({
        "applied": updated_count > 0,
        "updated_risk_count": updated_count,
        "point_count": point_count,
        "response_track_count": len(_response_tracks(data)),
    })
    if updated_count == 0:
        report["reason"] = "empty_tracker_result"
        _mark_tracker_fallback(candidates, report)
    return report


def _tracking_candidates(risks: list[VideoRiskAnnotation]) -> list[VideoRiskAnnotation]:
    candidates: list[VideoRiskAnnotation] = []
    for risk in risks:
        if risk.source_modality == "audio":
            continue
        if risk.excluded_by_operator_selection or not risk.eligible_for_redaction:
            continue
        if not _is_trackable_instance_seed(risk):
            continue
        if _has_unreliable_seed_localization(risk):
            risk.metadata["sam3_video_tracking"] = {
                "applied": False,
                "skipped": True,
                "reason": "unreliable_seed_localization",
            }
            continue
        if not any(isinstance(region, dict) and isinstance(region.get("bbox"), dict) for region in risk.regions):
            continue
        candidates.append(risk)
    return candidates


def _build_payload(
    frames: list[FrameReference],
    risks: list[VideoRiskAnnotation],
    config: Sam3VideoTrackerConfig,
) -> dict[str, Any]:
    return {
        "frames": [
            {
                "frame_id": frame.frame_id,
                "frame_index": frame.frame_index,
                "pts_ms": frame.pts_ms,
                "duration_ms": _safe_int(frame.metadata.get("duration_ms"), 0),
                "image_path": frame.image_uri,
                "image_uri": frame.image_uri,
            }
            for frame in frames
        ],
        "tracks": [_seed_track_payload(risk) for risk in risks],
        "return_masks": config.return_masks,
        "return_polygons": True,
        "coordinate_format": "xywh",
    }


def _seed_track_payload(risk: VideoRiskAnnotation) -> dict[str, Any]:
    track_target = _track_target_type(risk)
    return {
        "risk_id": risk.risk_id,
        "track_id": risk.track_id or risk.risk_id,
        "category": track_target or risk.category,
        "semantic_category": risk.category,
        "operator_id": risk.operator_id,
        "source_operator_id": risk.source_operator_id,
        "target_type": track_target or risk.target_type,
        "confidence": risk.confidence,
        "seed_regions": _seed_regions(risk),
        "prompt": {
            "category": track_target or risk.category,
            "semantic_category": risk.category,
            "target_type": track_target or risk.target_type,
            "text": _tracking_prompt_text(risk, track_target),
        },
    }


def _seed_regions(risk: VideoRiskAnnotation) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    for region in risk.regions:
        if not isinstance(region, dict) or not isinstance(region.get("bbox"), dict):
            continue
        seeds.append({
            "frame_id": str(region.get("frame_id") or ""),
            "bbox": _normalize_bbox(region.get("bbox")),
            "confidence": _safe_float(region.get("confidence"), risk.confidence),
            "mask_path": str(region.get("mask_path") or ""),
            "polygon": region.get("polygon") if isinstance(region.get("polygon"), list) else [],
            "source": str(region.get("source") or ""),
            "localization_status": str(region.get("localization_status") or ""),
            "instance_label_zh": str(region.get("instance_label_zh") or risk.metadata.get("instance_label_zh") or ""),
            "instance_label_en": str(region.get("instance_label_en") or risk.metadata.get("instance_label_en") or ""),
        })
    return seeds


def _apply_tracker_response(
    risks: list[VideoRiskAnnotation],
    frames: list[FrameReference],
    data: dict[str, Any],
) -> tuple[int, int]:
    risk_by_id = {risk.risk_id: risk for risk in risks}
    risk_by_track = {str(risk.track_id or risk.risk_id): risk for risk in risks}
    frame_by_id = {frame.frame_id: frame for frame in frames}
    updated_risks: set[str] = set()
    point_count = 0

    for track in _response_tracks(data):
        if not isinstance(track, dict):
            continue
        risk = risk_by_id.get(str(track.get("risk_id") or "")) or risk_by_track.get(str(track.get("track_id") or ""))
        if risk is None:
            continue
        added = _apply_track_points(risk, frame_by_id, _track_points(track))
        if added:
            updated_risks.add(risk.risk_id)
            point_count += added
            tracker_meta = risk.metadata.get("sam3_video_tracking") if isinstance(risk.metadata.get("sam3_video_tracking"), dict) else {}
            tracker_meta.update({
                "applied": True,
                "track_id": str(track.get("track_id") or risk.track_id or risk.risk_id),
                "point_count": added,
            })
            risk.metadata["sam3_video_tracking"] = tracker_meta
            risk.metadata["tracking_backend"] = "sam3_video_tracker"
    return len(updated_risks), point_count


def _apply_track_points(
    risk: VideoRiskAnnotation,
    frame_by_id: dict[str, FrameReference],
    points: list[dict[str, Any]],
) -> int:
    seen = {
        (str(region.get("frame_id") or ""), _bbox_key(region.get("bbox")))
        for region in risk.regions
        if isinstance(region, dict) and isinstance(region.get("bbox"), dict)
    }
    selected_points, quality_flags = _select_single_object_points(risk, frame_by_id, points)
    added = 0
    for point in selected_points:
        frame_id = str(point.get("frame_id") or "")
        normalized_bbox = _normalize_bbox(point.get("bbox") if isinstance(point.get("bbox"), dict) else {})
        dedupe_key = (frame_id, _bbox_key(normalized_bbox))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        frame = frame_by_id[frame_id]
        risk.regions.append({
            "frame_id": frame_id,
            "pts_ms": frame.pts_ms,
            "bbox": normalized_bbox,
            "confidence": _safe_float(point.get("confidence"), risk.confidence),
            "mask_path": str(point.get("mask_path") or ""),
            "polygon": point.get("polygon") if isinstance(point.get("polygon"), list) else [],
            "source": "sam3_video_tracker",
        })
        if frame_id not in risk.frame_ids:
            risk.frame_ids.append(frame_id)
        added += 1
    tracker_meta = risk.metadata.get("sam3_video_tracking") if isinstance(risk.metadata.get("sam3_video_tracking"), dict) else {}
    tracker_meta["candidate_point_count"] = len(points)
    tracker_meta["selected_point_count"] = len(selected_points)
    if quality_flags:
        existing_flags = tracker_meta.get("quality_flags") if isinstance(tracker_meta.get("quality_flags"), list) else []
        tracker_meta["quality_flags"] = _dedupe_strings([*existing_flags, *quality_flags])
    risk.metadata["sam3_video_tracking"] = tracker_meta
    if added:
        risk.frame_ids.sort(key=lambda frame_id: frame_by_id.get(str(frame_id)).pts_ms if frame_by_id.get(str(frame_id)) else 0)
    return added


def _select_single_object_points(
    risk: VideoRiskAnnotation,
    frame_by_id: dict[str, FrameReference],
    points: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep one coherent SAM3 video point per frame for a single seeded risk."""
    quality_flags: list[str] = []
    seed_box = _seed_reference_box(risk)
    normalized: list[dict[str, Any]] = []
    for point in points:
        bbox = point.get("bbox") if isinstance(point.get("bbox"), dict) else None
        if bbox is None:
            continue
        frame_id = str(point.get("frame_id") or "")
        if not frame_id and point.get("pts_ms") is not None:
            frame_id = _nearest_frame_id(frame_by_id, _safe_int(point.get("pts_ms"), 0))
        if not frame_id or frame_id not in frame_by_id:
            continue
        frame = frame_by_id[frame_id]
        item = dict(point)
        item["frame_id"] = frame_id
        item["pts_ms"] = frame.pts_ms
        item["bbox"] = _normalize_bbox(bbox)
        normalized.append(item)
    if not normalized:
        return [], quality_flags

    groups: dict[str, list[dict[str, Any]]] = {}
    for point in normalized:
        groups.setdefault(str(point.get("frame_id") or ""), []).append(point)
    if any(len(group) > 1 for group in groups.values()):
        quality_flags.append("ambiguous_sam3_points_same_frame")

    selected: list[dict[str, Any]] = []
    previous_box = seed_box
    for frame_id in sorted(groups, key=lambda key: frame_by_id[key].pts_ms):
        candidates = groups[frame_id]
        chosen = _best_candidate_for_reference(candidates, previous_box)
        if chosen is None:
            quality_flags.append("unstable_sam3_track")
            continue
        if previous_box is not None and not _bbox_transition_plausible(previous_box, chosen["bbox"]):
            quality_flags.append("unstable_sam3_track")
            continue
        selected.append(chosen)
        previous_box = chosen["bbox"]
    return selected, _dedupe_strings(quality_flags)


def _best_candidate_for_reference(candidates: list[dict[str, Any]], reference_box: dict[str, float] | None) -> dict[str, Any] | None:
    if not candidates:
        return None
    if reference_box is None:
        return max(candidates, key=lambda item: _safe_float(item.get("confidence"), 0.0))
    best = max(candidates, key=lambda item: _candidate_score(reference_box, item))
    if _candidate_score(reference_box, best) < 0.08:
        return None
    return best


def _candidate_score(reference_box: dict[str, float], point: dict[str, Any]) -> float:
    bbox = point.get("bbox") if isinstance(point.get("bbox"), dict) else {}
    iou = _bbox_iou(reference_box, bbox)
    center_score = max(0.0, 1.0 - _center_distance(reference_box, bbox) / max(1.0, _bbox_diagonal(reference_box) * 4.0))
    area_score = _area_similarity(reference_box, bbox)
    confidence = _safe_float(point.get("confidence"), 0.0)
    return iou * 0.45 + center_score * 0.3 + area_score * 0.15 + confidence * 0.1


def _bbox_transition_plausible(left: dict[str, float], right: dict[str, float]) -> bool:
    if _bbox_iou(left, right) >= 0.02:
        return True
    if _area_similarity(left, right) < 0.18:
        return False
    return _center_distance(left, right) <= max(80.0, _bbox_diagonal(left) * 4.0)


def _seed_reference_box(risk: VideoRiskAnnotation) -> dict[str, float] | None:
    best_region = None
    best_confidence = -1.0
    for region in risk.regions:
        if isinstance(region, dict) and isinstance(region.get("bbox"), dict):
            confidence = _safe_float(region.get("confidence"), risk.confidence)
            if confidence > best_confidence:
                best_region = region
                best_confidence = confidence
    return _normalize_bbox(best_region.get("bbox")) if isinstance(best_region, dict) else None


def _is_trackable_instance_seed(risk: VideoRiskAnnotation) -> bool:
    if not any(isinstance(region, dict) and isinstance(region.get("bbox"), dict) for region in risk.regions):
        return False
    role = str(risk.metadata.get("video_role") or "").strip().lower()
    if risk.category.startswith("content.") and not _track_target_type(risk):
        return False
    if role == "object_instance":
        return True
    if role in {"event", "localized_candidate", "unlocalized_instance"}:
        return False
    if risk.category.startswith("privacy."):
        return True
    if risk.source_modality in {"visual", "ocr_text", "video_object"}:
        return bool(_track_target_type(risk)) or not risk.category.startswith("content.")
    return False


def _track_target_type(risk: VideoRiskAnnotation) -> str:
    metadata = risk.metadata if isinstance(risk.metadata, dict) else {}
    values = [
        risk.target_type,
        metadata.get("trackable_target_type"),
        metadata.get("instance_label_en"),
        metadata.get("instance_label_zh"),
        metadata.get("entity_label_en"),
        metadata.get("entity_label_zh"),
        metadata.get("label"),
    ]
    text = " ".join(str(value or "") for value in values).strip().lower()
    if any(token in text for token in ("conflict_region", "fighting group", "fight", "physical_conflict", "斗殴区域", "打斗人群", "肢体冲突")):
        return "conflict_region"
    if any(token in text for token in ("weapon", "gun", "knife", "pistol", "rifle", "枪", "刀", "武器")):
        return "weapon"
    if any(token in text for token in ("blood", "wound", "injury", "血迹", "伤口", "受伤")):
        return "blood_or_wound"
    if any(token in text for token in ("exposed_body", "breast", "genital", "nude", "裸露", "胸部", "性器官")):
        return "exposed_body"
    if any(token in text for token in ("hate_symbol", "nazi", "flag", "gesture", "仇恨符号", "旗帜", "手势")):
        return "hate_symbol"
    return ""


def _tracking_prompt_text(risk: VideoRiskAnnotation, track_target: str) -> str:
    label_zh = str(risk.metadata.get("instance_label_zh") or "").strip()
    if track_target == "conflict_region":
        return label_zh or "暴力斗殴区域"
    if track_target:
        return label_zh or track_target
    return risk.text_span or risk.target_type or risk.category


def _has_unreliable_seed_localization(risk: VideoRiskAnnotation) -> bool:
    metadata = risk.metadata if isinstance(risk.metadata, dict) else {}
    if _metadata_marks_coarse_localization(metadata):
        return True
    for region in risk.regions:
        if isinstance(region, dict) and _metadata_marks_coarse_localization(region):
            return True
    evidence_regions = metadata.get("evidence_regions")
    if isinstance(evidence_regions, list):
        for region in evidence_regions:
            if isinstance(region, dict) and _metadata_marks_coarse_localization(region):
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
    if quality is not None and _safe_float(quality, 1.0) < 0.35:
        return True
    return False


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


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _response_tracks(data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(data.get("tracks"), list):
        return data["tracks"]
    result = data.get("result") if isinstance(data.get("result"), dict) else {}
    if isinstance(result.get("tracks"), list):
        return result["tracks"]
    if isinstance(data.get("data"), dict) and isinstance(data["data"].get("tracks"), list):
        return data["data"]["tracks"]
    return []


def _track_points(track: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("points", "regions", "detections", "frames"):
        value = track.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _mark_tracker_fallback(risks: list[VideoRiskAnnotation], report: dict[str, Any]) -> None:
    for risk in risks:
        risk.metadata["sam3_video_tracking"] = {
            "applied": False,
            "fallback": "sampled_frame_tracking",
            "reason": report.get("reason", ""),
            "endpoint": report.get("endpoint", ""),
        }


def _join_url(base_url: str, endpoint: str) -> str:
    return base_url.rstrip("/") + "/" + endpoint.lstrip("/")


def _nearest_frame_id(frame_by_id: dict[str, FrameReference], pts_ms: int) -> str:
    if not frame_by_id:
        return ""
    frame = min(frame_by_id.values(), key=lambda item: abs(item.pts_ms - pts_ms))
    return frame.frame_id


def _normalize_bbox(bbox: dict[str, Any] | None) -> dict[str, float]:
    bbox = bbox or {}
    return {
        "x": _safe_float(bbox.get("x"), 0.0),
        "y": _safe_float(bbox.get("y"), 0.0),
        "w": _safe_float(bbox.get("w"), 0.0),
        "h": _safe_float(bbox.get("h"), 0.0),
    }


def _bbox_key(bbox: Any) -> tuple[int, int, int, int]:
    if not isinstance(bbox, dict):
        return (0, 0, 0, 0)
    normalized = _normalize_bbox(bbox)
    return (
        round(normalized["x"] * 1000),
        round(normalized["y"] * 1000),
        round(normalized["w"] * 1000),
        round(normalized["h"] * 1000),
    )


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
