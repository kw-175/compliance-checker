"""Core data models for the video compliance blueprint."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from picture.domain.models import PictureFinding, PictureModerationResult
from video.domain.enums import VideoDecisionType, VideoJobStatus, VideoRouteType


def _utcnow() -> datetime:
    # 统一使用 UTC 时间，便于跨时区审计。
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    # 生成可读性较好的短 ID。
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class TimeSpan(BaseModel):
    """Time range in milliseconds."""

    # 起止时间均使用毫秒单位。
    start_ms: int = 0
    end_ms: int = 0


class FrameReference(BaseModel):
    """Reference to a sampled or decoded frame."""

    # frame_id 用于在各清单和审计报告间关联同一帧。
    frame_id: str = Field(default_factory=lambda: _new_id("frame"))
    frame_index: int
    pts_ms: int
    image_uri: str = ""
    route: VideoRouteType | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoSegment(BaseModel):
    """A contiguous video segment used for routing and tracking."""

    # segment 代表同一路由或同类语义的连续时间片段。
    segment_id: str = Field(default_factory=lambda: _new_id("segment"))
    span: TimeSpan
    route: VideoRouteType = VideoRouteType.MIXED
    frame_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoAsset(BaseModel):
    """Materialized assets produced during video processing."""

    # 统一收敛所有产物 URI，便于 API 一次性返回。
    asset_id: str = Field(default_factory=lambda: _new_id("asset"))
    original_uri: str = ""
    normalized_uri: str | None = None
    audio_uri: str | None = None
    compliant_video_uri: str | None = None
    report_uri: str | None = None
    frame_manifest_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoFinding(BaseModel):
    """Compliance finding anchored to video time and optional frame evidence."""

    # finding 既可来自 picture，也可来自 safety 聚合结果。
    finding_id: str = Field(default_factory=lambda: _new_id("vf"))
    span: TimeSpan
    frame_id: str | None = None
    track_id: str | None = None
    source_modality: str = "picture"
    picture_finding: PictureFinding | None = None
    moderation: PictureModerationResult | None = None
    reason_code: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    # ── 新增：provider 追溯与可解释性 ──
    provider_version: str = ""
    confidence: float = 0.0
    explanation: str = ""


class VideoPolicyResult(BaseModel):
    """Final policy outcome after frame, track, and audio aggregation."""

    # decision 是视频任务最终可交付判定。
    decision: VideoDecisionType = VideoDecisionType.PASS_RAW
    reason_codes: list[str] = Field(default_factory=list)
    profile: str = "default_cn_enterprise"
    evaluated_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # ── 新增：可信等级与评分分解 ──
    trust_level: str = "full"
    score_breakdown: dict[str, Any] | None = None
    degrade_summary: str = ""


class VideoReport(BaseModel):
    """Audit report for a video compliance job."""

    # report 面向审计归档，包含发现、原因码与产物索引。
    job_id: str
    route: VideoRouteType = VideoRouteType.MIXED
    decision: VideoDecisionType = VideoDecisionType.PASS_RAW
    findings: list[VideoFinding] = Field(default_factory=list)
    provider_info: dict[str, str] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    # ── 新增：降级事件、可信等级与复核摘要 ──
    degrade_events: list[dict[str, Any]] = Field(default_factory=list)
    trust_level: str = "full"
    review_summary: str = ""


class VideoJob(BaseModel):
    """Top-level entity for a video compliance task."""

    # 顶层任务对象：串起输入、执行状态、产物与错误信息。
    job_id: str = Field(default_factory=lambda: _new_id("job"))
    tenant_id: str = ""
    source_uri: str = ""
    mime_type: str = "video/mp4"
    profile: str = "default_cn_enterprise"
    status: VideoJobStatus = VideoJobStatus.CREATED
    route: VideoRouteType | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    asset: VideoAsset | None = None
    frames: list[FrameReference] = Field(default_factory=list)
    segments: list[VideoSegment] = Field(default_factory=list)
    findings: list[VideoFinding] = Field(default_factory=list)
    policy_result: VideoPolicyResult | None = None

    # created/updated/completed 用于外部轮询与 SLA 统计。
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None

    error: str | None = None
    error_detail: str | None = None

    step_latencies: dict[str, float] = Field(default_factory=dict)
    provider_versions: dict[str, str] = Field(default_factory=dict)
    # ── 新增：降级事件、可信等级与双轨交付物 ──
    degrade_events: list[dict[str, Any]] = Field(default_factory=list)
    trust_level: str = "full"
    annotation_package_uri: str | None = None
    audit_package_uri: str | None = None
