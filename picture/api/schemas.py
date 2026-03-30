"""
API request/response schemas for the picture compliance engine.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from picture.domain.enums import DecisionType, FindingType, JobStatus, RouteType


# ─────────────────────────────────────────────────────────────────────
# Request schemas
# ─────────────────────────────────────────────────────────────────────

class SourceInput(BaseModel):
    """Source specification in a job creation request."""
    type: str = "file"  # file | url | base64
    uri: str
    mime_type: str = "image/png"


class JobOptions(BaseModel):
    """Processing options for a job."""
    route_hint: str = "auto"
    ocr_engine: str = "auto"
    detector_mode: str = "auto"
    enable_open_vocab_detection: bool = False
    redaction_mode_text: str = "black_box"
    redaction_mode_face: str = "gaussian_blur"
    drop_on_explicit_content: bool = True


class CreateJobRequest(BaseModel):
    """Request body for POST /v1/picture/jobs."""
    tenant_id: str = "default"
    source: SourceInput
    mode: str = "compliance_only"
    profile: str = "default_cn_enterprise"
    options: JobOptions = Field(default_factory=JobOptions)


# ─────────────────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────────────────

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
    original_uri: Optional[str] = None
    compliant_uri: Optional[str] = None
    overlay_uri: Optional[str] = None
    report_uri: Optional[str] = None


class JobResultStats(BaseModel):
    """Timing and statistics for a completed job."""
    total_findings: int = 0
    total_redactions: int = 0
    latency_ms: dict[str, float] = Field(default_factory=dict)
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
