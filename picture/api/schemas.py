"""
API request/response schemas for the picture compliance engine.
"""
# 中文说明：这里定义 API 层使用的 Pydantic 模型，用于隔离 HTTP 协议与领域对象。

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class SourceInput(BaseModel):
    """Source specification in a job creation request."""

    # 中文说明：type 表示输入的载体形式，而不是图片内容类别。
    # 当前主要使用 file，其他类型为后续扩展保留。
    type: str = "file"  # file | url | base64
    uri: str
    mime_type: str = "image/png"


class JobOptions(BaseModel):
    """Processing options for a job."""

    contract_version: str = "compliance-job.v1"
    platform_task_id: str = ""
    idempotency_key: str = ""
    operator_id: str = ""
    operator_catalog_version: str = "image-compliance-operators.v1"
    # 中文说明：route_hint 用于提示系统优先走哪条处理链。
    route_hint: str = "auto"
    # 中文说明：下面这些字段是请求级细粒度开关，用于覆盖全局默认行为。
    ocr_engine: str = "auto"
    detector_mode: str = "auto"
    enable_open_vocab_detection: bool = False
    redaction_mode_text: str = "black_box"
    redaction_mode_face: str = "gaussian_blur"
    redaction_mode_qr: str = "black_box"
    redaction_mode_signature: str = "solid_fill"
    redaction_mode_default: str = "black_box"
    max_redaction_area_ratio: Optional[float] = None
    drop_on_explicit_content: bool = True
    enable_total_compliance: bool = True
    enable_text_privacy_detection: Optional[bool] = None
    enable_text_content_detection: Optional[bool] = None
    enable_visual_safety_detection: Optional[bool] = None
    enable_visual_sensitive_object_detection: Optional[bool] = None
    disable_ocr: bool = False
    disable_visual_safety: bool = False
    disable_visual_sensitive_objects: bool = False
    ordinary_dataset_enabled: bool = True
    restricted_dataset_enabled: bool = False
    restricted_use_case: str = ""
    authorized_sensitive_use: bool = False
    privacy_operator_ids: list[str] = Field(default_factory=list)
    privacy_target_types: list[str] = Field(default_factory=list)
    content_safety_operator_ids: list[str] = Field(default_factory=list)
    content_safety_target_labels: list[str] = Field(default_factory=list)
    visual_safety_operator_ids: list[str] = Field(default_factory=list)
    visual_safety_target_labels: list[str] = Field(default_factory=list)
    visual_sensitive_object_operator_ids: list[str] = Field(default_factory=list)
    visual_sensitive_object_types: list[str] = Field(default_factory=list)
    picture_mode: str = "comprehensive"


class CreateJobRequest(BaseModel):
    """Request body for POST /v1/picture/jobs."""

    # 中文说明：tenant_id 让系统具备多租户归属能力，方便审计和隔离。
    tenant_id: str = "default"
    source: SourceInput
    # 中文说明：mode 当前主要是语义占位，未来可以扩展为只检测、只脱敏等模式。
    mode: str = "compliance_only"
    profile: str = "default_cn_enterprise"
    options: JobOptions = Field(default_factory=JobOptions)


class ManualBBox(BaseModel):
    """Manual review bbox in original image pixel coordinates."""

    x: float
    y: float
    w: float
    h: float


class ManualPolygon(BaseModel):
    """Manual review polygon in original image pixel coordinates."""

    points: list[tuple[float, float]] = Field(default_factory=list)


class ManualRegion(BaseModel):
    """Manual review region."""

    bbox: ManualBBox
    polygon: Optional[list[tuple[float, float]] | ManualPolygon] = None
    mask_path: Optional[str] = None
    confidence: float = 1.0


class ManualFindingInput(BaseModel):
    """Finding supplied by a human reviewer for re-redaction."""

    finding_id: Optional[str] = None
    finding_type: str = "vision_object"
    category: str = "manual_region"
    label: str = ""
    score: float = 1.0
    reason_code: str = "MANUAL_REVIEW"
    provider: str = "human_review"
    text_span: Optional[str] = None
    region: ManualRegion
    redaction_mode: str = "black_box"
    explanation: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ManualRedactionRequest(BaseModel):
    """Manual review payload used to regenerate image redaction artifacts."""

    findings: list[ManualFindingInput] = Field(default_factory=list)
    conclusion: Optional[str] = None
    review_note: str = ""
    reviewed_by: str = ""


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class CreateJobResponse(BaseModel):
    """Response for POST /v1/picture/jobs."""

    job_id: str
    status: str
    contract_version: str = "compliance-job.v1"
    platform_task_id: str = ""
    idempotency_key: str = ""
    modality: str = "image"
    status_label: str = ""
    effective_request: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class JobStatusResponse(BaseModel):
    """Response for GET /v1/picture/jobs/{job_id}."""

    job_id: str
    status: str
    contract_version: str = "compliance-job.v1"
    platform_task_id: str = ""
    idempotency_key: str = ""
    modality: str = "image"
    status_label: str = ""
    stage: str = ""
    progress: int = 0
    route: Optional[str] = None
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    error_info: Optional[dict[str, Any]] = None
    effective_request: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    current_step: Optional[str] = None
    current_provider: Optional[str] = None
    current_step_started_at: Optional[str] = None


class ArtifactURIs(BaseModel):
    """URIs for generated artifacts."""

    # 中文说明：这里聚合了任务最常用的产物地址，方便调用方直接拿去展示或下载。
    original_uri: Optional[str] = None
    compliant_uri: Optional[str] = None
    overlay_uri: Optional[str] = None
    report_uri: Optional[str] = None


class JobResultStats(BaseModel):
    """Timing and statistics for a completed job."""

    total_findings: int = 0
    total_redactions: int = 0
    # 中文说明：latency_ms 以步骤维度记录耗时，而不是只有一个粗粒度总时长。
    latency_ms: dict[str, float] = Field(default_factory=dict)
    # 中文说明：provider_versions 用于记录这次任务实际调用了哪些 provider。
    provider_versions: dict[str, str] = Field(default_factory=dict)


class JobResultResponse(BaseModel):
    """Response for GET /v1/picture/jobs/{job_id}/result."""

    job_id: str
    contract_version: str = "compliance-result.v1"
    platform_task_id: str = ""
    idempotency_key: str = ""
    modality: str = "image"
    decision: str
    dataset_action: str = ""
    review_required: bool = False
    requires_restricted_dataset: bool = False
    annotation_guidance_zh: str = ""
    reason_codes: list[str] = Field(default_factory=list)
    artifacts: ArtifactURIs = Field(default_factory=ArtifactURIs)
    stats: JobResultStats = Field(default_factory=JobResultStats)
    effective_request: dict[str, Any] = Field(default_factory=dict)
    result_consistency: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error_info: Optional[dict[str, Any]] = None


class FindingResponse(BaseModel):
    """A single finding in the findings list."""

    finding_id: str
    finding_type: str
    category: str
    label: str
    score: float
    reason_code: str
    provider: str
    text_span: Optional[str] = None
    localization_status: Optional[str] = None
    boundary_status: Optional[str] = None
    review_required: Optional[bool] = None
    mask_quality_score: Optional[float] = None
    # 中文说明：region 是 API 层输出的通用字典结构，而不是领域层的嵌套对象。
    region: Optional[dict[str, Any]] = None


class FindingsListResponse(BaseModel):
    """Response for GET /v1/picture/jobs/{job_id}/findings."""

    job_id: str
    total: int
    findings: list[FindingResponse] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Unified error response structure."""

    error: str
    code: str
    detail: Optional[str] = None


class ManualRedactionResponse(BaseModel):
    """Response after applying manual region review and re-redaction."""

    job_id: str
    artifacts: ArtifactURIs = Field(default_factory=ArtifactURIs)
    redaction_operations: list[dict[str, Any]] = Field(default_factory=list)
    manual_review: dict[str, Any] = Field(default_factory=dict)
    policy_snapshot: dict[str, Any] = Field(default_factory=dict)
