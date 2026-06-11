"""Quality and review artifacts for video compliance jobs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from picture.domain.models import PictureJob
from video.application.services import SequenceBundle
from video.domain.models import VideoActionPlan, VideoGovernancePolicyResult, VideoRiskAnnotation


def build_review_queue(
    risks: list[VideoRiskAnnotation],
    policy: VideoGovernancePolicyResult,
    action_plan: VideoActionPlan,
) -> list[dict[str, Any]]:
    """Build a compact human-review queue from risk and policy outputs."""
    queue: list[dict[str, Any]] = []
    if policy.requires_review:
        queue.append({
            "review_id": "policy_review",
            "priority": "high",
            "review_type": "policy_decision",
            "reason": "策略要求人工复核",
            "reason_codes": policy.reason_codes,
            "span": {"start_ms": 0, "end_ms": 0},
        })
    for risk in risks:
        tracking = risk.metadata.get("tracking") if isinstance(risk.metadata.get("tracking"), dict) else {}
        priority = _priority(risk, tracking)
        if priority == "none":
            continue
        queue.append({
            "review_id": f"review_{risk.risk_id}",
            "priority": priority,
            "review_type": "risk_annotation",
            "risk_id": risk.risk_id,
            "track_id": risk.track_id,
            "category": risk.category,
            "operator_id": risk.operator_id,
            "source_modality": risk.source_modality,
            "severity": risk.severity,
            "confidence": risk.confidence,
            "span": (risk.display_span or risk.span).model_dump(mode="json"),
            "representative_frame_uri": risk.metadata.get("representative_frame_uri", ""),
            "reason_codes": risk.reason_codes,
            "review_reasons": _review_reasons(risk, tracking),
            "redaction_scope": tracking.get("redaction_scope", ""),
            "redaction_ready": bool(tracking.get("redaction_ready", False)),
            "quality_flags": tracking.get("quality_flags", []) if isinstance(tracking.get("quality_flags"), list) else [],
        })
    for operation in action_plan.operations:
        if not operation.regions and operation.modality not in {"audio"}:
            queue.append({
                "review_id": f"operation_region_{operation.operation_id}",
                "priority": "medium",
                "review_type": "redaction_region_missing",
                "risk_id": operation.risk_id,
                "track_id": operation.track_id,
                "operation": operation.operation,
                "span": {"start_ms": operation.start_ms, "end_ms": operation.end_ms},
                "review_reasons": ["脱敏操作缺少空间区域，需要人工确认或补框"],
            })
    return queue


def build_quality_report(
    sequence: SequenceBundle,
    frame_jobs: list[PictureJob],
    risks: list[VideoRiskAnnotation],
    scene_windows: list[dict[str, Any]],
    clip_windows: list[dict[str, Any]],
    clip_audits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize runtime coverage, cache behavior, and residual risk."""
    cached_frames = [frame for frame in sequence.frames if frame.metadata.get("cached_from_frame_id")]
    by_category = Counter(risk.category for risk in risks)
    by_modality = Counter(risk.source_modality for risk in risks)
    failed_clip_windows = [item for item in clip_audits if item.get("status") == "failed"]
    frame_findings = sum(len(job.findings) for job in frame_jobs)
    analyzed_frame_ids = {
        str(job.source.uri)
        for job in frame_jobs
        if job.source and job.source.uri
    }
    return {
        "source_kind": sequence.source_kind,
        "duration_ms": sequence.total_duration_ms,
        "input_frame_count": sequence.total_input_frames,
        "sampled_frame_count": len(sequence.frames),
        "picture_result_count": len(frame_jobs),
        "frame_cache": {
            "cached_frame_count": len(cached_frames),
            "cache_ratio": round(len(cached_frames) / max(1, len(sequence.frames)), 4),
            "cache_reasons": dict(Counter(str(frame.metadata.get("cache_reason") or "none") for frame in cached_frames)),
        },
        "scene_count": len(scene_windows),
        "clip_window_count": len(clip_windows),
        "clip_moderation": {
            "completed_windows": sum(1 for item in clip_audits if item.get("status") == "completed"),
            "failed_windows": len(failed_clip_windows),
            "failed_window_ids": [str(item.get("window_id") or "") for item in failed_clip_windows],
        },
        "risk_count": len(risks),
        "risk_by_category": dict(by_category),
        "risk_by_modality": dict(by_modality),
        "frame_finding_count": frame_findings,
        "spatial_track_count": sum(1 for risk in risks if risk.metadata.get("tracking", {}).get("has_spatial_track")),
        "needs_full_frame_review_count": sum(1 for risk in risks if risk.metadata.get("tracking", {}).get("requires_full_frame_review")),
        "redaction_ready_count": sum(1 for risk in risks if risk.metadata.get("tracking", {}).get("redaction_ready")),
        "manual_redaction_review_count": sum(1 for risk in risks if _manual_redaction_review_needed(risk)),
        "redaction_scope_count": dict(Counter(str(risk.metadata.get("tracking", {}).get("redaction_scope") or "unknown") for risk in risks)),
        "analyzed_frame_uri_count": len(analyzed_frame_ids),
        "quality_flags": _quality_flags(sequence, risks, failed_clip_windows),
    }


def _priority(risk: VideoRiskAnnotation, tracking: dict[str, Any]) -> str:
    if risk.severity in {"critical", "high"}:
        return "high"
    if risk.confidence < 0.65:
        return "medium"
    if tracking.get("requires_full_frame_review"):
        return "medium"
    if risk.source_modality == "video_clip":
        return "medium"
    return "none"


def _review_reasons(risk: VideoRiskAnnotation, tracking: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if risk.severity in {"critical", "high"}:
        reasons.append("高风险或严重风险")
    if risk.confidence < 0.65:
        reasons.append("模型置信度较低")
    if tracking.get("requires_full_frame_review"):
        reasons.append("缺少空间定位，需要人工确认")
    if risk.category.startswith("privacy.") and not tracking.get("redaction_ready", False) and risk.source_modality != "audio":
        reasons.append("隐私风险缺少可自动脱敏的空间轨道，需要人工框选或复核")
    for flag in tracking.get("quality_flags", []) if isinstance(tracking.get("quality_flags"), list) else []:
        reasons.append(_quality_flag_zh(str(flag)))
    if risk.source_modality == "video_clip":
        reasons.append("短片段行为审核命中，需要人工复核动作语义")
    return reasons or ["需要抽样复核"]


def _quality_flags(sequence: SequenceBundle, risks: list[VideoRiskAnnotation], failed_clip_windows: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    if sequence.total_duration_ms and len(sequence.frames) < max(2, sequence.total_duration_ms // 5000):
        flags.append("采样帧较少，短暂风险漏检概率较高")
    if failed_clip_windows:
        flags.append("部分短片段多模态审核失败")
    if any(risk.source_modality == "video_clip" and not risk.regions for risk in risks):
        flags.append("存在片段级风险但缺少空间框，需要结合关键帧复核")
    if any(_manual_redaction_review_needed(risk) for risk in risks):
        flags.append("存在隐私风险缺少可自动脱敏区域，需要人工框选或复核")
    if not risks:
        flags.append("未检出风险，建议对高风险数据集抽样复核")
    return flags


def _manual_redaction_review_needed(risk: VideoRiskAnnotation) -> bool:
    if not risk.category.startswith("privacy.") or risk.source_modality == "audio":
        return False
    tracking = risk.metadata.get("tracking") if isinstance(risk.metadata.get("tracking"), dict) else {}
    return not bool(tracking.get("redaction_ready", False))


def _quality_flag_zh(flag: str) -> str:
    return {
        "missing_spatial_localization": "没有获得空间定位结果",
        "single_frame_localization": "仅有单帧定位，连续脱敏需复核",
        "sampled_frames_only": "仅覆盖采样帧，未形成连续视频轨道",
        "low_confidence_spatial_points": "部分空间定位置信度较低",
        "semantic_window_without_region": "只有片段语义命中，缺少具体区域",
        "ambiguous_sam3_points_same_frame": "同一帧存在多个疑似追踪目标，需要人工确认",
        "duplicate_sam3_points_same_frame": "同一帧存在重复追踪框，需要人工确认",
        "low_quality_coarse_localization": "定位来源为低质量粗框，不能直接自动脱敏",
        "sam3_track_skipped": "SAM3 时序追踪已跳过该低可信定位",
        "unstable_sam3_track": "时序轨迹跳变较大，需要人工确认",
    }.get(flag, flag)
