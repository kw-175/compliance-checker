"""Core data models for the video compliance blueprint."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from picture.domain.models import PictureFinding, PictureModerationResult
from video.domain.enums import VideoDecisionType, VideoGovernanceDecision, VideoJobStatus, VideoRouteType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class TimeSpan(BaseModel):
    """Time range in milliseconds."""

    start_ms: int = 0
    end_ms: int = 0


class FrameReference(BaseModel):
    """Reference to a sampled or decoded frame."""

    frame_id: str = Field(default_factory=lambda: _new_id("frame"))
    frame_index: int
    pts_ms: int
    image_uri: str = ""
    route: VideoRouteType | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoSegment(BaseModel):
    """A contiguous video segment used for routing and tracking."""

    segment_id: str = Field(default_factory=lambda: _new_id("segment"))
    span: TimeSpan
    route: VideoRouteType = VideoRouteType.MIXED
    frame_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoAsset(BaseModel):
    """Materialized assets produced during video processing."""

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

    finding_id: str = Field(default_factory=lambda: _new_id("vf"))
    span: TimeSpan
    frame_id: str | None = None
    track_id: str | None = None
    source_modality: str = "picture"
    picture_finding: PictureFinding | None = None
    moderation: PictureModerationResult | None = None
    reason_code: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoPolicyResult(BaseModel):
    """Final policy outcome after frame, track, and audio aggregation."""

    decision: VideoDecisionType = VideoDecisionType.PASS_RAW
    reason_codes: list[str] = Field(default_factory=list)
    profile: str = "default_cn_enterprise"
    evaluated_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskContext(BaseModel):
    """Downstream use context that controls governance and redaction choices."""

    stage: str = "annotation"
    task_type: str = "generic_video"
    release_scope: str = "internal_controlled"
    needs_face: bool = False
    needs_audio: bool = False
    needs_screen_text: bool = False
    needs_identity_features: bool = False
    allow_restricted_use: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ComplianceOperatorSelection(BaseModel):
    """Selected video compliance labels/operators for the current training task."""

    visual_sensitive_object_operator_ids: list[str] = Field(default_factory=list)
    visual_sensitive_object_types: list[str] = Field(default_factory=list)
    visual_safety_operator_ids: list[str] = Field(default_factory=list)
    visual_safety_target_labels: list[str] = Field(default_factory=list)
    privacy_operator_ids: list[str] = Field(default_factory=list)
    privacy_target_types: list[str] = Field(default_factory=list)
    content_safety_operator_ids: list[str] = Field(default_factory=list)
    content_safety_target_labels: list[str] = Field(default_factory=list)
    audio_operator_ids: list[str] = Field(default_factory=list)
    audio_privacy_operator_ids: list[str] = Field(default_factory=list)
    audio_content_safety_operator_ids: list[str] = Field(default_factory=list)
    disabled_operator_ids: list[str] = Field(default_factory=list)
    disabled_target_types: list[str] = Field(default_factory=list)
    preserved_training_targets: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreservedTrainingTarget(BaseModel):
    """A target intentionally excluded from compliance risk detection."""

    target_id: str = Field(default_factory=lambda: _new_id("preserve"))
    target_type: str = ""
    source_operator_id: str = ""
    video_operator_id: str = ""
    source_modality: str = ""
    reason: str = "excluded_by_operator_selection"
    preserved_for_task: str = ""
    span: TimeSpan | None = None
    frame_ids: list[str] = Field(default_factory=list)
    regions: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoRiskAnnotation(BaseModel):
    """A non-destructive compliance risk annotation anchored on the video timeline."""

    risk_id: str = Field(default_factory=lambda: _new_id("risk"))
    asset_id: str = ""
    source_modality: str = "visual"
    category: str = ""
    operator_id: str = ""
    source_operator_id: str = ""
    target_type: str = ""
    severity: str = "low"
    confidence: float = 0.0
    span: TimeSpan
    display_span: TimeSpan | None = None
    temporal_precision: str = ""
    spatial_precision: str = ""
    localization_status: str = ""
    evidence_points: list[dict[str, Any]] = Field(default_factory=list)
    frame_ids: list[str] = Field(default_factory=list)
    track_id: str | None = None
    regions: list[dict[str, Any]] = Field(default_factory=list)
    text_span: str | None = None
    audio_segment: dict[str, Any] | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    provider: str = ""
    provider_version: str = ""
    reason_codes: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    eligible_for_redaction: bool = True
    excluded_by_operator_selection: bool = False
    task_impact: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoGovernancePolicyResult(BaseModel):
    """Use-aware governance decision built from risks and task context."""

    decision: VideoGovernanceDecision = VideoGovernanceDecision.ALLOW
    requires_review: bool = False
    requires_transformation: bool = False
    requires_restricted_dataset: bool = False
    allow_original_for_annotation: bool = True
    reason_codes: list[str] = Field(default_factory=list)
    matched_rules: list[str] = Field(default_factory=list)
    profile: str = "default_cn_enterprise"
    evaluated_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoRedactionOperation(BaseModel):
    """A planned redaction operation. It is not executed by detection."""

    operation_id: str = Field(default_factory=lambda: _new_id("op"))
    risk_id: str = ""
    modality: str = "visual"
    operation: str = "blur"
    start_ms: int = 0
    end_ms: int = 0
    track_id: str | None = None
    regions: list[dict[str, Any]] = Field(default_factory=list)
    task_impact: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoActionPlan(BaseModel):
    """Governance action plan produced after policy evaluation."""

    action_plan_id: str = Field(default_factory=lambda: _new_id("action"))
    default_action: str = "allow_with_risk_labels"
    preserve_original: bool = True
    render_redacted_asset: bool = True
    original_access_level: str = "internal_controlled"
    operations: list[VideoRedactionOperation] = Field(default_factory=list)
    derived_outputs: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoReport(BaseModel):
    """Audit report for a video compliance job."""

    job_id: str
    route: VideoRouteType = VideoRouteType.MIXED
    decision: VideoDecisionType = VideoDecisionType.PASS_RAW
    findings: list[VideoFinding] = Field(default_factory=list)
    provider_info: dict[str, str] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    risk_summary: dict[str, Any] = Field(default_factory=dict)
    display_risks: list[VideoRiskAnnotation] = Field(default_factory=list)
    governance_decision: dict[str, Any] = Field(default_factory=dict)
    action_plan: dict[str, Any] = Field(default_factory=dict)
    preserved_training_targets: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class VideoJob(BaseModel):
    """Top-level entity for a video compliance task."""

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
    risk_annotations: list[VideoRiskAnnotation] = Field(default_factory=list)
    display_risks: list[VideoRiskAnnotation] = Field(default_factory=list)
    risk_tracks: list[dict[str, Any]] = Field(default_factory=list)
    preserved_training_targets: list[PreservedTrainingTarget] = Field(default_factory=list)
    policy_result: VideoPolicyResult | None = None
    governance_result: VideoGovernancePolicyResult | None = None
    action_plan: VideoActionPlan | None = None
    task_context: TaskContext = Field(default_factory=TaskContext)
    operator_selection: ComplianceOperatorSelection = Field(default_factory=ComplianceOperatorSelection)

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None

    error: str | None = None
    error_detail: str | None = None

    step_latencies: dict[str, float] = Field(default_factory=dict)
    provider_versions: dict[str, str] = Field(default_factory=dict)
