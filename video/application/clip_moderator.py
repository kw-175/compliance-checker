"""Short-window video moderation through the shared Qwen video service."""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageChops

from video.application.scene_sampling import select_evenly_spaced_frames
from video.domain.models import FrameReference, TimeSpan, VideoRiskAnnotation
from video.domain.taxonomy import recommended_actions_for_category, severity_for_category

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClipModerationConfig:
    base_url: str = "http://127.0.0.1:8200"
    endpoint: str = "/video/action-recognition"
    timeout_seconds: int = 120
    max_frames: int = 8
    confidence_threshold: float = 0.55
    fail_on_error: bool = False
    labels: list[str] = field(default_factory=lambda: [
        "physical_conflict",
        "graphic_violence",
        "weapon_threat",
        "sexual_exposure",
        "hate_symbol",
        "self_harm",
        "normal_education",
    ])
    label_definitions: dict[str, str] = field(default_factory=lambda: {
        "physical_conflict": "多人肢体冲突、打架、斗殴、追打、明显攻击行为。",
        "graphic_violence": "血腥、严重暴力伤害或极端暴力画面。",
        "weapon_threat": "刀具、棍棒、枪械等武器威胁或攻击行为。",
        "sexual_exposure": "色情暴露、明显性行为或低俗裸露。",
        "hate_symbol": "仇恨、极端、纳粹、恐怖主义等符号、旗帜、文字或手势。",
        "self_harm": "自伤、自杀、危险自残行为。",
        "normal_education": "正常教学、讨论、走动、课堂互动或无明显安全风险。",
    })


def moderate_clip_windows(
    frames: list[FrameReference],
    clip_windows: list[dict[str, Any]],
    config: ClipModerationConfig,
    asset_id: str = "",
    operator_selection: Any | None = None,
) -> tuple[list[VideoRiskAnnotation], list[dict[str, Any]]]:
    """Run temporal moderation over bounded windows and return risk annotations."""
    if not frames or not clip_windows:
        return [], []
    frame_by_id = {frame.frame_id: frame for frame in frames}
    risks: list[VideoRiskAnnotation] = []
    audits: list[dict[str, Any]] = []
    for window in clip_windows:
        window_frames = [frame_by_id[item] for item in window.get("frame_ids", []) if item in frame_by_id]
        selected = select_evenly_spaced_frames(window_frames, max(1, config.max_frames))
        if not selected:
            continue
        payload = _build_payload(selected, frames, window, config)
        try:
            response = _post_json(config, payload)
        except Exception as exc:
            logger.warning("Clip moderation failed for %s: %s", window.get("window_id"), exc)
            audits.append({
                "window_id": window.get("window_id", ""),
                "status": "failed",
                "error": str(exc),
            })
            if config.fail_on_error:
                raise
            continue
        events = response.get("events") if isinstance(response.get("events"), list) else []
        localization_audits: list[dict[str, Any]] = []
        for event_index, event in enumerate(events):
            if isinstance(event, dict):
                localization_audit = _ensure_event_instances(event, event_index, selected, frames, window, config)
                if localization_audit:
                    localization_audits.append(localization_audit)
        audits.append({
            "window_id": window.get("window_id", ""),
            "status": "completed",
            "request_frame_count": len(selected),
            "event_count": len(events),
            "raw_response": response,
            "localization_attempts": localization_audits,
        })
        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            risk = _risk_from_event(event, event_index, window, selected, asset_id, config)
            if risk is None:
                continue
            if _risk_allowed(risk, operator_selection):
                risks.append(risk)
                risks.extend(
                    instance
                    for instance in _instance_risks_from_event(event, event_index, window, selected, asset_id, config, risk)
                    if _risk_allowed(instance, operator_selection)
                )
    return risks, audits


def _build_payload(
    selected: list[FrameReference],
    all_frames: list[FrameReference],
    window: dict[str, Any],
    config: ClipModerationConfig,
) -> dict[str, Any]:
    duration_seconds = max(0.001, _video_end_ms(all_frames) / 1000.0)
    window_start = int(window.get("start_ms", selected[0].pts_ms) or 0) / 1000.0
    window_end = int(window.get("end_ms", selected[-1].pts_ms) or 0) / 1000.0
    fps = len(selected) / max(0.001, window_end - window_start)
    return {
        "frames": [
            {
                "frame_index": frame.frame_index,
                "timestamp": frame.pts_ms / 1000.0,
                "image_base64": _image_to_base64(frame.image_uri),
            }
            for frame in selected
        ],
        "fps": fps,
        "duration": duration_seconds,
        "labels": config.labels,
        "label_definitions": config.label_definitions,
        "window_start": window_start,
        "window_end": window_end,
        "task": "video_safety_detection_with_temporal_localization",
        "coordinate_format": "xywh_pixels",
        "localization_required": True,
        "tracking_required": True,
        "output_language": "zh-CN",
        "output_schema": _qwen_localization_schema(),
        "localization_targets": _qwen_localization_targets(),
        "instructions": [
            "必须同时输出违规事件的 start_time/end_time，以及与该事件对应的可见违规要素实例。",
            "instances 中必须尽量给出 keyframes；每个 keyframe 包含 frame_index 或 timestamp，以及 bbox=[x,y,w,h]，坐标基于当前输入帧像素。",
            "不要把 violence、sexual、hate 等抽象词当成可追踪对象；应框出冲突交互区域、武器、血迹/伤口、裸露敏感区域、仇恨符号/文字/手势等具体可见要素。",
            "physical_conflict 只需要输出一个 conflict_region：框住发生推搡、攻击、倒地或身体接触的核心区域，不要拆成多个人脸或多个人员标签。",
            "如果只能判断事件但无法可靠框选要素，仍输出事件，但 instances 为空并设置 review_required=true。",
        ],
    }


def _build_event_localization_payload(
    selected: list[FrameReference],
    all_frames: list[FrameReference],
    window: dict[str, Any],
    event: dict[str, Any],
    config: ClipModerationConfig,
) -> dict[str, Any]:
    payload = _build_payload(selected, all_frames, window, config)
    payload.update({
        "task": "video_safety_event_region_localization",
        "labels": ["physical_conflict", "graphic_violence"],
        "event_to_localize": {
            "category": event.get("category"),
            "start_time": event.get("start_time"),
            "end_time": event.get("end_time"),
            "evidence": event.get("evidence"),
        },
        "output_schema": {
            "instances": [
                {
                    "category": "conflict_region",
                    "entity_label_zh": "暴力斗殴区域",
                    "entity_label_en": "conflict_region",
                    "confidence": 0.0,
                    "keyframes": [
                        {
                            "frame_index": 0,
                            "timestamp": 0.0,
                            "bbox": [0, 0, 0, 0],
                            "confidence": 0.0,
                        }
                    ],
                }
            ]
        },
        "instructions": [
            "只做结构化定位，不重新判断是否违规。",
            "根据 event_to_localize 的中文证据和关键帧，输出一个 conflict_region。",
            "conflict_region 必须框住发生推搡、攻击、倒地压制或身体接触的核心区域。",
            "不要输出 violence、graphic_violence、physical_conflict 作为可追踪对象；不要输出人脸或单个普通人员。",
            "如果无法可靠框选，instances 返回空数组。",
        ],
    })
    return payload


def _qwen_localization_schema() -> dict[str, Any]:
    return {
        "events": [
            {
                "category": "physical_conflict|graphic_violence|weapon_threat|sexual_exposure|hate_symbol|self_harm|normal_education",
                "confidence": 0.0,
                "start_time": 0.0,
                "end_time": 0.0,
                "evidence": "中文证据说明",
                "review_required": False,
                "instances": [
                    {
                        "instance_id": "可选稳定实例ID",
                        "category": "conflict_region|weapon|blood_or_wound|exposed_body|hate_symbol|harmful_object|text_region|other_visible_risk",
                        "entity_label_zh": "具体可见要素中文名",
                        "entity_label_en": "specific visible noun, e.g. conflict_region, pistol, knife, exposed_body, nazi_symbol",
                        "confidence": 0.0,
                        "start_time": 0.0,
                        "end_time": 0.0,
                        "keyframes": [
                            {
                                "frame_index": 0,
                                "timestamp": 0.0,
                                "bbox": [0, 0, 0, 0],
                                "confidence": 0.0,
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _qwen_localization_targets() -> dict[str, list[str]]:
    return {
        "physical_conflict": ["打斗/推搡交互区域", "攻击动作接触区域", "可见武器", "倒地/受伤者"],
        "graphic_violence": ["血迹", "伤口", "受伤身体区域", "可见武器", "暴力接触区域"],
        "weapon_threat": ["枪械", "刀具", "棍棒", "弓弩", "其他具体危险器具"],
        "sexual_exposure": ["裸露敏感区域", "性器官/胸部/臀部等敏感身体区域", "性行为相关可见区域"],
        "hate_symbol": ["仇恨符号", "极端组织标志", "仇恨文字", "仇恨手势", "相关旗帜"],
        "self_harm": ["自伤工具", "伤口/血迹", "危险接触区域", "自伤行为主体"],
    }


def _post_json(config: ClipModerationConfig, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = config.base_url.rstrip("/") + "/" + config.endpoint.lstrip("/")
    request = Request(url, data=body, method="POST")
    request.add_header("Accept", "application/json")
    request.add_header("Content-Type", "application/json; charset=utf-8")
    try:
        with urlopen(request, timeout=max(1, config.timeout_seconds)) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen video API HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Qwen video API request failed: {exc.reason}") from exc
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError(f"Qwen video API returned unexpected response: {data!r}")
    return data


def _ensure_event_instances(
    event: dict[str, Any],
    event_index: int,
    selected: list[FrameReference],
    all_frames: list[FrameReference],
    window: dict[str, Any],
    config: ClipModerationConfig,
) -> dict[str, Any] | None:
    if not _needs_secondary_localization(event):
        return None
    payload = _build_event_localization_payload(selected, all_frames, window, event, config)
    audit: dict[str, Any] = {
        "event_index": event_index,
        "category": str(event.get("category") or ""),
        "status": "requested",
    }
    try:
        response = _post_json(config, payload)
    except Exception as exc:
        event["localization_attempted"] = True
        event["localization_status"] = "secondary_localization_failed"
        event["review_required"] = True
        audit.update({"status": "failed", "error": str(exc)})
        if config.fail_on_error:
            raise
        return audit
    instances = _localized_instances_from_response(response)
    if instances:
        event["instances"] = instances
        event["localization_attempted"] = True
        event["localization_status"] = "secondary_localization_succeeded"
        event["review_required"] = False
        audit.update({"status": "completed", "instance_count": len(instances), "raw_response": response})
    else:
        fallback_instances = _fallback_event_instances_from_motion(event, selected, window)
        event["localization_attempted"] = True
        event["review_required"] = True
        if fallback_instances:
            event["instances"] = fallback_instances
            event["localization_status"] = "motion_saliency_seeded_after_secondary_empty"
            audit.update({
                "status": "fallback_motion_seeded",
                "instance_count": len(fallback_instances),
                "raw_response": response,
                "fallback": {
                    "method": "frame_motion_saliency",
                    "reason": "secondary_localization_empty",
                },
            })
        else:
            event["localization_status"] = "secondary_localization_empty"
            audit.update({"status": "empty", "instance_count": 0, "raw_response": response})
    return audit


def _needs_secondary_localization(event: dict[str, Any]) -> bool:
    category = str(event.get("category") or event.get("label") or "").strip().lower()
    if category not in {"physical_conflict", "graphic_violence"}:
        return False
    return not _event_has_structured_instances(event)


def _event_has_structured_instances(event: dict[str, Any]) -> bool:
    if event.get("bbox") is not None or event.get("rough_bbox") is not None or isinstance(event.get("keyframes"), list):
        return True
    for key in ("instances", "objects", "localized_objects", "localized_violations", "evidence_regions", "regions", "tracks", "object_tracks"):
        value = event.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict) and (item.get("bbox") is not None or item.get("rough_bbox") is not None or isinstance(item.get("keyframes"), list)):
                return True
    return False


def _localized_instances_from_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for key in ("instances", "objects", "localized_objects", "localized_violations", "evidence_regions", "regions", "tracks", "object_tracks"):
        value = response.get(key)
        if isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict))
    events = response.get("events")
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            for key in ("instances", "objects", "localized_objects", "localized_violations", "evidence_regions", "regions", "tracks", "object_tracks"):
                value = event.get(key)
                if isinstance(value, list):
                    candidates.extend(item for item in value if isinstance(item, dict))
            if event.get("bbox") is not None or event.get("rough_bbox") is not None or isinstance(event.get("keyframes"), list):
                candidates.append(event)
    result: list[dict[str, Any]] = []
    for item in candidates:
        if not _event_has_structured_instances(item):
            continue
        record = dict(item)
        role = _instance_role_text(record)
        if "conflict_region" not in role and "斗殴区域" not in role and "冲突" not in role and "打斗" not in role:
            continue
        record.setdefault("category", "conflict_region")
        record.setdefault("entity_label_zh", "暴力斗殴区域")
        record.setdefault("entity_label_en", "conflict_region")
        record.setdefault("source", "qwen_video_secondary_localization")
        record.setdefault("localization_status", "localized_by_qwen_video_secondary_localization")
        result.append(record)
    return _normalize_event_instances({"category": "physical_conflict"}, result)


def _fallback_event_instances_from_motion(
    event: dict[str, Any],
    selected_frames: list[FrameReference],
    window: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create a concrete seed for dynamic conflict events when Qwen returns text only."""
    category = str(event.get("category") or event.get("label") or "").strip().lower()
    if category not in {"physical_conflict", "graphic_violence"}:
        return []
    if _confidence(event) < 0.70:
        return []
    event_frames = _frames_for_event(event, window, selected_frames)
    keyframes = _motion_saliency_keyframes(event_frames)
    if not keyframes:
        return []
    return [{
        "instance_id": "conflict_region_motion_seed_1",
        "category": "conflict_region",
        "entity_label_zh": "暴力斗殴区域",
        "entity_label_en": "conflict_region",
        "confidence": min(0.82, max(0.56, _confidence(event) * 0.76)),
        "start_time": event.get("start_time"),
        "end_time": event.get("end_time"),
        "keyframes": keyframes,
        "source": "frame_motion_saliency_seed",
        "localization_status": "localized_by_frame_motion_saliency_seed",
        "review_required": True,
        "evidence": str(event.get("evidence") or ""),
    }]


def _frames_for_event(
    event: dict[str, Any],
    window: dict[str, Any],
    selected_frames: list[FrameReference],
) -> list[FrameReference]:
    start_ms, end_ms = _event_span(event, window)
    frames = [
        frame for frame in selected_frames
        if frame.pts_ms <= end_ms and _frame_end_ms(frame) >= start_ms
    ]
    return frames or selected_frames


def _motion_saliency_keyframes(frames: list[FrameReference]) -> list[dict[str, Any]]:
    if len(frames) < 2:
        return []
    keyframes: list[dict[str, Any]] = []
    previous = _load_gray_frame(frames[0])
    for left, right in zip(frames, frames[1:]):
        current = _load_gray_frame(right)
        if previous is None or current is None:
            previous = current
            continue
        bbox = _motion_bbox(previous, current)
        previous = current
        if bbox is None:
            continue
        keyframes.append({
            "frame_id": right.frame_id,
            "frame_index": right.frame_index,
            "timestamp": right.pts_ms / 1000.0,
            "bbox": bbox,
            "confidence": 0.62,
            "source": "frame_motion_saliency_seed",
            "localization_status": "localized_by_frame_motion_saliency_seed",
        })
    return _dedupe_motion_keyframes(keyframes)


def _load_gray_frame(frame: FrameReference) -> Image.Image | None:
    try:
        with Image.open(Path(frame.image_uri)) as image:
            return image.convert("L")
    except Exception:
        return None


def _motion_bbox(left: Image.Image, right: Image.Image) -> list[float] | None:
    if left.size != right.size:
        left = left.resize(right.size)
    diff = ImageChops.difference(left, right)
    width, height = right.size
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    histogram = diff.histogram()
    for percentile in (0.88, 0.94, 0.975, 0.985, 0.992):
        threshold = max(8, _histogram_percentile(histogram, percentile))
        mask = diff.point(lambda value, t=threshold: 255 if value >= t else 0)
        bbox = mask.getbbox()
        if bbox is None:
            continue
        box_w = bbox[2] - bbox[0]
        box_h = bbox[3] - bbox[1]
        area_ratio = (box_w * box_h) / max(1, width * height)
        if area_ratio >= 0.006 and area_ratio < 0.92:
            candidates.append((area_ratio, bbox))
    if not candidates:
        return None
    compact = [item for item in candidates if item[0] <= 0.72]
    if compact:
        _, bbox = min(compact, key=lambda item: abs(item[0] - 0.32))
    else:
        _, bbox = min(candidates, key=lambda item: item[0])
    raw_area_ratio = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / max(1, width * height)
    expand_ratio = 0.18 if raw_area_ratio < 0.45 else 0.04
    x1, y1, x2, y2 = _expand_bbox_xyxy(bbox, width, height, expand_ratio)
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None
    area_ratio = (box_w * box_h) / max(1, width * height)
    if area_ratio < 0.006 or area_ratio > 0.88:
        return None
    return [float(x1), float(y1), float(box_w), float(box_h)]


def _histogram_percentile(histogram: list[int], percentile: float) -> int:
    total = sum(histogram)
    if total <= 0:
        return 255
    target = total * max(0.0, min(1.0, percentile))
    cumulative = 0
    for level, count in enumerate(histogram):
        cumulative += count
        if cumulative >= target:
            return level
    return 255


def _expand_bbox_xyxy(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    pad_x = int(round((x2 - x1) * ratio))
    pad_y = int(round((y2 - y1) * ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
    )


def _dedupe_motion_keyframes(keyframes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in keyframes:
        frame_id = str(item.get("frame_id") or "")
        if not frame_id or frame_id in seen:
            continue
        seen.add(frame_id)
        result.append(item)
    return result


def _risk_from_event(
    event: dict[str, Any],
    event_index: int,
    window: dict[str, Any],
    selected_frames: list[FrameReference],
    asset_id: str,
    config: ClipModerationConfig,
) -> VideoRiskAnnotation | None:
    category, source_operator_id = _category_for_event(event)
    if not category:
        return None
    confidence = _confidence(event)
    if confidence < config.confidence_threshold and not bool(event.get("review_required", False)):
        return None
    start_ms, end_ms = _event_span(event, window)
    event_frames = [
        frame.frame_id for frame in selected_frames
        if frame.pts_ms <= end_ms and _frame_end_ms(frame) >= start_ms
    ] or [frame.frame_id for frame in selected_frames]
    severity = severity_for_category(category, confidence, event)
    return VideoRiskAnnotation(
        asset_id=asset_id,
        source_modality="video_clip",
        category=category,
        operator_id=_video_operator_id(source_operator_id),
        source_operator_id=source_operator_id,
        target_type=category,
        severity=severity,
        confidence=confidence,
        span=TimeSpan(start_ms=start_ms, end_ms=end_ms),
        frame_ids=event_frames,
        text_span=str(event.get("evidence") or ""),
        evidence_refs=[f"clip_moderation:{window.get('window_id', '')}:{event_index}"],
        provider="qwen_video_action_recognition",
        provider_version="Qwen3.5-9B@8200",
        reason_codes=[f"CLIP_{source_operator_id}" if source_operator_id else "CLIP_VIDEO_RISK"],
        recommended_actions=recommended_actions_for_category(category),
        metadata={
            "video_role": "event",
            "window_id": window.get("window_id", ""),
            "scene_id": window.get("scene_id", ""),
            "raw_event": event,
            "evidence": str(event.get("evidence") or ""),
            "review_required": bool(event.get("review_required", False)),
        },
    )


def _instance_risks_from_event(
    event: dict[str, Any],
    event_index: int,
    window: dict[str, Any],
    selected_frames: list[FrameReference],
    asset_id: str,
    config: ClipModerationConfig,
    parent_risk: VideoRiskAnnotation,
) -> list[VideoRiskAnnotation]:
    """Parse optional Qwen video object/localization output into trackable instances."""
    records = _event_instance_records(event)
    if not records:
        return []
    risks: list[VideoRiskAnnotation] = []
    for instance_index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        regions = _regions_from_instance_record(record, selected_frames)
        if not regions:
            continue
        confidence = _confidence(record) or parent_risk.confidence
        if confidence < config.confidence_threshold and not bool(record.get("review_required", False)):
            continue
        category, source_operator_id = _category_for_instance(record, parent_risk)
        start_ms, end_ms = _instance_span(record, parent_risk.span.start_ms, parent_risk.span.end_ms)
        label_zh = str(record.get("entity_label_zh") or record.get("label_zh") or record.get("label") or record.get("object_name_zh") or "").strip()
        label_en = str(record.get("entity_label_en") or record.get("label_en") or record.get("object") or record.get("category") or "").strip()
        risks.append(
            VideoRiskAnnotation(
                asset_id=asset_id,
                source_modality="video_object",
                category=category,
                operator_id=_video_operator_id(source_operator_id),
                source_operator_id=source_operator_id,
                target_type=label_en or category,
                severity=severity_for_category(category, confidence, record),
                confidence=confidence,
                span=TimeSpan(start_ms=start_ms, end_ms=end_ms),
                frame_ids=_dedupe([str(region.get("frame_id") or "") for region in regions]),
                regions=regions,
                text_span=str(record.get("evidence") or record.get("description") or parent_risk.text_span or ""),
                evidence_refs=[f"clip_moderation:{window.get('window_id', '')}:{event_index}:instance:{instance_index}"],
                provider="qwen_video_action_recognition",
                provider_version="Qwen3.5-9B@8200",
                reason_codes=[f"CLIP_OBJECT_{source_operator_id}" if source_operator_id else "CLIP_VIDEO_OBJECT"],
                recommended_actions=recommended_actions_for_category(category),
                metadata={
                    "video_role": "object_instance",
                    "parent_risk_id": parent_risk.risk_id,
                    "window_id": window.get("window_id", ""),
                    "scene_id": window.get("scene_id", ""),
                    "raw_instance": record,
                    "instance_label_zh": label_zh,
                    "instance_label_en": label_en,
                    "localization_status": str(record.get("localization_status") or "localized_by_qwen_video_keyframes"),
                    "review_required": bool(record.get("review_required", False)),
                    "tracking_seed_source": str(record.get("source") or "qwen_video_keyframes"),
                },
            )
        )
    return risks


def _event_instance_records(event: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if event.get("bbox") is not None or event.get("rough_bbox") is not None or isinstance(event.get("keyframes"), list):
        records.append(event)
    for key in ("instances", "objects", "localized_objects", "localized_violations", "evidence_regions", "regions", "tracks", "object_tracks"):
        value = event.get(key)
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
    return _normalize_event_instances(event, records)


def _normalize_event_instances(event: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    category_text = str(event.get("category") or event.get("label") or "").strip().lower()
    if category_text not in {"physical_conflict", "graphic_violence"}:
        return records
    conflict_records: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for record in records:
        role = _instance_role_text(record)
        if any(token in role for token in ("weapon", "gun", "knife", "pistol", "rifle", "枪", "刀", "武器")):
            kept.append(record)
        elif any(token in role for token in ("blood", "wound", "injured", "血迹", "伤口", "受伤")):
            kept.append(record)
        elif any(token in role for token in ("conflict", "participant", "person", "fight", "body", "contact", "冲突", "打斗", "人员", "身体", "接触", "推搡")):
            conflict_records.append(record)
        else:
            kept.append(record)
    conflict_region = _merge_conflict_records(event, conflict_records)
    if conflict_region:
        return [conflict_region, *kept]
    return kept


def _merge_conflict_records(event: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any] | None:
    keyframes: list[dict[str, Any]] = []
    direct_boxes: list[dict[str, Any]] = []
    source = ""
    localization_status = ""
    review_required = False
    for record in records:
        if not source and record.get("source"):
            source = str(record.get("source") or "")
        if not localization_status and record.get("localization_status"):
            localization_status = str(record.get("localization_status") or "")
        review_required = review_required or bool(record.get("review_required", False))
        if isinstance(record.get("keyframes"), list):
            keyframes.extend(item for item in record["keyframes"] if isinstance(item, dict))
        if record.get("bbox") is not None or record.get("rough_bbox") is not None:
            direct_boxes.append(record)
    merged_keyframes = _merge_keyframe_boxes(keyframes)
    if not merged_keyframes and direct_boxes:
        merged_keyframes = _merge_keyframe_boxes(direct_boxes)
    if not merged_keyframes:
        return None
    return {
        "instance_id": str(event.get("instance_id") or "conflict_region_1"),
        "category": "conflict_region",
        "entity_label_zh": "暴力斗殴区域",
        "entity_label_en": "conflict_region",
        "confidence": _confidence(event),
        "start_time": event.get("start_time"),
        "end_time": event.get("end_time"),
        "keyframes": merged_keyframes,
        "source": source or "qwen_video_conflict_region",
        "localization_status": localization_status or "localized_by_qwen_video_conflict_region",
        "review_required": review_required,
    }


def _merge_keyframe_boxes(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        bbox = _bbox_value(item.get("bbox") or item.get("rough_bbox"))
        if bbox is None:
            continue
        key = str(item.get("frame_index") if item.get("frame_index") is not None else item.get("timestamp", item.get("time", item.get("pts", ""))))
        grouped.setdefault(key, []).append({**item, "bbox": bbox})
    merged: list[dict[str, Any]] = []
    for _, group in grouped.items():
        first = group[0]
        xs = [item["bbox"][0] for item in group]
        ys = [item["bbox"][1] for item in group]
        x2s = [item["bbox"][0] + item["bbox"][2] for item in group]
        y2s = [item["bbox"][1] + item["bbox"][3] for item in group]
        item = {
            "bbox": [min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys)],
            "confidence": max(_confidence(item) for item in group),
        }
        for key in ("frame_id", "frame_index", "timestamp", "time", "pts"):
            if first.get(key) is not None:
                item[key] = first.get(key)
        merged.append(item)
    return merged


def _instance_role_text(record: dict[str, Any]) -> str:
    return " ".join(
        str(record.get(key) or "")
        for key in ("category", "entity_label_zh", "entity_label_en", "label", "label_zh", "label_en", "object", "object_name_zh")
    ).lower()


def _regions_from_instance_record(record: dict[str, Any], selected_frames: list[FrameReference]) -> list[dict[str, Any]]:
    frame_by_id = {frame.frame_id: frame for frame in selected_frames}
    regions: list[dict[str, Any]] = []
    keyframes = record.get("keyframes")
    if isinstance(keyframes, list):
        for item in keyframes:
            if isinstance(item, dict):
                region = _region_from_qwen_bbox(item, selected_frames, frame_by_id, record)
                if region:
                    regions.append(region)
    region = _region_from_qwen_bbox(record, selected_frames, frame_by_id, record)
    if region:
        regions.append(region)
    regions.sort(key=lambda item: (int(item.get("pts_ms", 0) or 0), str(item.get("frame_id", ""))))
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    for region in regions:
        bbox = region.get("bbox") if isinstance(region.get("bbox"), dict) else {}
        key = (
            str(region.get("frame_id") or ""),
            (
                round(float(bbox.get("x", 0)) * 1000),
                round(float(bbox.get("y", 0)) * 1000),
                round(float(bbox.get("w", 0)) * 1000),
                round(float(bbox.get("h", 0)) * 1000),
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(region)
    return deduped


def _region_from_qwen_bbox(
    item: dict[str, Any],
    selected_frames: list[FrameReference],
    frame_by_id: dict[str, FrameReference],
    parent: dict[str, Any],
) -> dict[str, Any] | None:
    bbox = _bbox_value(item.get("bbox") or item.get("rough_bbox"))
    if bbox is None:
        return None
    frame = _frame_for_instance_item(item, selected_frames, frame_by_id)
    if frame is None:
        return None
    label_zh = str(parent.get("entity_label_zh") or parent.get("label_zh") or parent.get("label") or parent.get("object_name_zh") or "")
    label_en = str(parent.get("entity_label_en") or parent.get("label_en") or parent.get("object") or parent.get("category") or "")
    return {
        "frame_id": frame.frame_id,
        "pts_ms": frame.pts_ms,
        "bbox": {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]},
        "confidence": _confidence(item) or _confidence(parent) or 0.0,
        "source": str(item.get("source") or parent.get("source") or "qwen_video_keyframe_bbox"),
        "localization_status": str(item.get("localization_status") or parent.get("localization_status") or "localized_by_qwen_video_keyframes"),
        "mask_quality_score": item.get("mask_quality_score", parent.get("mask_quality_score")),
        "instance_label_zh": label_zh,
        "instance_label_en": label_en,
        "violation_id": str(parent.get("violation_id") or parent.get("instance_id") or ""),
    }


def _frame_for_instance_item(
    item: dict[str, Any],
    selected_frames: list[FrameReference],
    frame_by_id: dict[str, FrameReference],
) -> FrameReference | None:
    frame_id = str(item.get("frame_id") or "")
    if frame_id and frame_id in frame_by_id:
        return frame_by_id[frame_id]
    if item.get("frame_index") is not None:
        try:
            frame_index = int(item.get("frame_index"))
            for frame in selected_frames:
                if frame.frame_index == frame_index:
                    return frame
        except (TypeError, ValueError):
            pass
    timestamp = item.get("timestamp", item.get("time", item.get("pts")))
    if timestamp is not None:
        try:
            pts_ms = int(round(float(timestamp) * 1000))
            return min(selected_frames, key=lambda frame: abs(frame.pts_ms - pts_ms))
        except (TypeError, ValueError):
            pass
    return selected_frames[0] if selected_frames else None


def _bbox_value(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        try:
            x = float(value.get("x", 0))
            y = float(value.get("y", 0))
            w = float(value.get("w", 0))
            h = float(value.get("h", 0))
        except (TypeError, ValueError):
            return None
    elif isinstance(value, list) and len(value) == 4:
        try:
            x, y, w, h = [float(item) for item in value]
        except (TypeError, ValueError):
            return None
    else:
        return None
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


def _category_for_instance(record: dict[str, Any], parent_risk: VideoRiskAnnotation) -> tuple[str, str]:
    raw = " ".join(str(record.get(key) or "") for key in ("category", "label", "entity_label_en", "entity_label_zh", "object")).lower()
    if any(token in raw for token in ("face", "人脸")):
        return "privacy.face", "VPI_001"
    if any(token in raw for token in ("text", "ocr", "文字", "文本")):
        return "privacy.screen_sensitive", "PII_009"
    if any(token in raw for token in ("weapon", "gun", "knife", "pistol", "rifle", "枪", "刀", "武器")):
        return "content.graphic_violence", "CSA_003"
    if any(token in raw for token in ("exposed", "nude", "sexual", "breast", "genital", "裸露", "色情", "胸部", "性器官")):
        return "content.sexual", "CSA_002"
    if any(token in raw for token in ("hate", "nazi", "symbol", "flag", "gesture", "仇恨", "极端", "符号", "旗帜", "手势")):
        return "content.hate", "CSA_004"
    if any(token in raw for token in ("blood", "wound", "injured", "participant", "person", "fight", "conflict_region", "血迹", "伤口", "受伤", "打斗", "参与者", "人员", "斗殴区域")):
        return parent_risk.category, parent_risk.source_operator_id
    return parent_risk.category, parent_risk.source_operator_id


def _instance_span(record: dict[str, Any], default_start_ms: int, default_end_ms: int) -> tuple[int, int]:
    span = record.get("time_span") or record.get("timestamp_range")
    if isinstance(span, list) and len(span) >= 2:
        start_ms = int(round(_float_value(span[0], default_start_ms / 1000.0) * 1000))
        end_ms = int(round(_float_value(span[1], default_end_ms / 1000.0) * 1000))
    else:
        start_ms = int(round(_float_value(record.get("start_time"), default_start_ms / 1000.0) * 1000))
        end_ms = int(round(_float_value(record.get("end_time"), default_end_ms / 1000.0) * 1000))
    if end_ms <= start_ms:
        return default_start_ms, default_end_ms
    return max(default_start_ms, start_ms), min(default_end_ms, end_ms)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _category_for_event(event: dict[str, Any]) -> tuple[str, str]:
    text = " ".join(
        str(event.get(key) or "")
        for key in ("category", "label", "primary_activity", "evidence")
    ).lower()
    if any(token in text for token in ("normal", "education", "teacherlecture", "studentactivity", "groupdiscussion", "transition", "正常", "教学", "课堂")):
        return "", ""
    if any(token in text for token in ("graphic_violence", "weapon", "fight", "assault", "physical_conflict", "violence", "斗殴", "打架", "暴力", "肢体冲突", "攻击", "武器")):
        return "content.graphic_violence", "CSA_003"
    if any(token in text for token in ("sexual", "porn", "nudity", "exposure", "裸露", "色情", "低俗")):
        return "content.sexual", "CSA_002"
    if any(token in text for token in ("hate", "nazi", "extremist", "terror", "symbol", "仇恨", "极端", "纳粹", "恐怖", "符号")):
        return "content.hate", "CSA_004"
    if any(token in text for token in ("self_harm", "suicide", "自伤", "自杀")):
        return "content.self_harm", "CSA_006"
    return "", ""


def _event_span(event: dict[str, Any], window: dict[str, Any]) -> tuple[int, int]:
    window_start = int(window.get("start_ms", 0) or 0)
    window_end = int(window.get("end_ms", window_start + 1) or window_start + 1)
    span = event.get("time_span") or event.get("timestamp_range")
    if isinstance(span, list) and len(span) >= 2:
        start_value = _float_value(span[0], window_start / 1000.0)
        end_value = _float_value(span[1], window_end / 1000.0)
    else:
        start_value = _float_value(event.get("start_time"), window_start / 1000.0)
        end_value = _float_value(event.get("end_time"), window_end / 1000.0)
    start_ms = int(round(start_value * 1000))
    end_ms = int(round(end_value * 1000))
    if end_ms <= start_ms:
        start_ms, end_ms = window_start, window_end
    start_ms = max(window_start, min(start_ms, window_end))
    end_ms = max(start_ms + 1, min(end_ms, window_end))
    return start_ms, end_ms


def _confidence(event: dict[str, Any]) -> float:
    value = event.get("confidence")
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return 0.7 if bool(event.get("review_required", False)) else 0.0


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _video_operator_id(source_operator_id: str) -> str:
    return f"VVIS_{source_operator_id}" if source_operator_id else ""


def _risk_allowed(risk: VideoRiskAnnotation, operator_selection: Any | None) -> bool:
    if operator_selection is None or not hasattr(operator_selection, "risk_allowed"):
        return True
    return bool(operator_selection.risk_allowed(
        operator_id=risk.operator_id,
        source_operator_id=risk.source_operator_id,
        target_type=risk.target_type,
        category=risk.category,
    ))


def _image_to_base64(image_uri: str) -> str:
    with Image.open(Path(image_uri)) as image:
        image = image.convert("RGB")
        image.thumbnail((768, 768))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=82)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _video_end_ms(frames: list[FrameReference]) -> int:
    if not frames:
        return 1
    return max(_frame_end_ms(frame) for frame in frames)


def _frame_end_ms(frame: FrameReference) -> int:
    return frame.pts_ms + max(1, int(frame.metadata.get("duration_ms", 0) or 0))
