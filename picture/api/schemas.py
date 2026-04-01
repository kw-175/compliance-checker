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

    # 中文说明：route_hint 用于提示系统优先走哪条处理链。
    route_hint: str = "auto"
    # 中文说明：下面这些字段是请求级细粒度开关，用于覆盖全局默认行为。
    ocr_engine: str = "auto"
    detector_mode: str = "auto"
    enable_open_vocab_detection: bool = False
    redaction_mode_text: str = "black_box"
    redaction_mode_face: str = "gaussian_blur"
    drop_on_explicit_content: bool = True


class CreateJobRequest(BaseModel):
    """Request body for POST /v1/picture/jobs."""

    # 中文说明：tenant_id 让系统具备多租户归属能力，方便审计和隔离。
    tenant_id: str = "default"
    source: SourceInput
    # 中文说明：mode 当前主要是语义占位，未来可以扩展为只检测、只脱敏等模式。
    mode: str = "compliance_only"
    profile: str = "default_cn_enterprise"
    options: JobOptions = Field(default_factory=JobOptions)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class CreateJobResponse(BaseModel):
    """Response for POST /v1/picture/jobs."""

    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    """Response for GET /v1/picture/jobs/{job_id}."""

    job_id: str
    status: str
    route: Optional[str] = None
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


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
    decision: str
    reason_codes: list[str] = Field(default_factory=list)
    artifacts: ArtifactURIs = Field(default_factory=ArtifactURIs)
    stats: JobResultStats = Field(default_factory=JobResultStats)


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
