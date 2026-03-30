"""
Core domain models for the picture compliance engine.

All models use Pydantic v2 for serialization, validation, and OpenAPI schema generation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from picture.domain.enums import (
    DecisionType,
    FindingType,
    JobStatus,
    RedactionMode,
    RouteType,
    SafetyCategory,
)


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _new_id() -> str:
    """Generate a short unique ID."""
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

class BBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates (x, y, w, h)."""
    x: float
    y: float
    w: float
    h: float


class Polygon(BaseModel):
    """Arbitrary polygon as list of (x, y) points."""
    points: list[tuple[float, float]] = Field(default_factory=list)


class RegionMask(BaseModel):
    """
    Region on an image, using bbox, optional polygon refinement,
    and optional binary mask path.
    """
    bbox: BBox
    polygon: Optional[Polygon] = None
    mask_path: Optional[str] = None
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Source & asset
# ---------------------------------------------------------------------------

class SourceSpec(BaseModel):
    """Describes the source of an image to process."""
    type: str = "file"  # "file" | "url" | "base64"
    uri: str
    mime_type: str = "image/png"


class PictureAsset(BaseModel):
    """Represents a single image asset within a job."""
    asset_id: str = Field(default_factory=_new_id)
    original_uri: str = ""
    preprocessed_uri: Optional[str] = None
    width: int = 0
    height: int = 0
    mime_type: str = ""
    page_index: Optional[int] = None  # for multi-page PDF
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# OCR results
# ---------------------------------------------------------------------------

class OCRTextBlock(BaseModel):
    """A single text block from OCR."""
    text: str
    bbox: BBox
    confidence: float = 0.0
    language: str = ""


class LayoutRegion(BaseModel):
    """A layout region (paragraph, table, figure, etc.)."""
    region_type: str = "text"  # text | table | figure | header | footer
    bbox: BBox
    text_blocks: list[OCRTextBlock] = Field(default_factory=list)


class OCRLayoutResult(BaseModel):
    """Result from OCR + layout analysis."""
    full_text: str = ""
    text_blocks: list[OCRTextBlock] = Field(default_factory=list)
    layout_regions: list[LayoutRegion] = Field(default_factory=list)
    engine_name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

class PictureFinding(BaseModel):
    """A single compliance finding on an image."""
    finding_id: str = Field(default_factory=_new_id)
    finding_type: FindingType
    category: str = ""          # e.g. "person_name", "face", "explicit"
    label: str = ""             # human-readable label
    score: float = 0.0
    region: Optional[RegionMask] = None
    text_span: Optional[str] = None  # original text for PII
    reason_code: str = ""       # e.g. "PII_PHONE", "SAFETY_EXPLICIT"
    provider: str = ""          # which provider produced this
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Moderation / safety
# ---------------------------------------------------------------------------

class PictureModerationResult(BaseModel):
    """Safety moderation result for an image."""
    is_safe: bool = True
    categories: list[SafetyCategory] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    provider: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Redaction operation record
# ---------------------------------------------------------------------------

class RedactionOperation(BaseModel):
    """Records a single redaction action performed on an image."""
    finding_id: str = ""
    region: RegionMask
    mode: RedactionMode = RedactionMode.BLACK_BOX
    applied: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Policy result
# ---------------------------------------------------------------------------

class PicturePolicyResult(BaseModel):
    """Final policy evaluation result."""
    decision: DecisionType = DecisionType.PASS_RAW
    reason_codes: list[str] = Field(default_factory=list)
    profile: str = "default_cn_enterprise"
    evaluated_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class PictureReport(BaseModel):
    """Full audit report for a picture compliance job."""
    job_id: str
    route: RouteType = RouteType.MIXED
    decision: DecisionType = DecisionType.PASS_RAW
    findings: list[PictureFinding] = Field(default_factory=list)
    moderation: Optional[PictureModerationResult] = None
    redaction_operations: list[RedactionOperation] = Field(default_factory=list)
    provider_info: dict[str, str] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    timestamps: dict[str, str] = Field(default_factory=dict)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

class PictureJob(BaseModel):
    """Top-level job entity for picture compliance processing."""
    job_id: str = Field(default_factory=lambda: f"job_{uuid.uuid4().hex[:12]}")
    tenant_id: str = ""
    status: JobStatus = JobStatus.CREATED
    source: SourceSpec = Field(default_factory=lambda: SourceSpec(uri=""))
    route: Optional[RouteType] = None
    profile: str = "default_cn_enterprise"
    options: dict[str, Any] = Field(default_factory=dict)

    # Intermediate results
    asset: Optional[PictureAsset] = None
    ocr_result: Optional[OCRLayoutResult] = None
    moderation_result: Optional[PictureModerationResult] = None
    findings: list[PictureFinding] = Field(default_factory=list)
    redaction_operations: list[RedactionOperation] = Field(default_factory=list)
    policy_result: Optional[PicturePolicyResult] = None

    # Output URIs
    compliant_image_uri: Optional[str] = None
    overlay_image_uri: Optional[str] = None
    report_uri: Optional[str] = None

    # Timing
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = None

    # Error
    error: Optional[str] = None
    error_detail: Optional[str] = None

    # Audit
    step_latencies: dict[str, float] = Field(default_factory=dict)
    provider_versions: dict[str, str] = Field(default_factory=dict)
