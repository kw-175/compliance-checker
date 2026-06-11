"""
Core domain models for the picture compliance engine.

All models use Pydantic v2 for serialization, validation, and OpenAPI schema generation.
"""
# 中文说明：该文件定义 picture 模块的核心数据结构。
# 从输入源、OCR 结果、finding、策略结果到最终任务对象，都会在这里统一建模。
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
    # 中文说明：统一使用 UTC 时间，避免不同服务器时区导致审计时间混乱。
    return datetime.now(timezone.utc)


def _new_id() -> str:
    """Generate a short unique ID."""
    # 中文说明：这里生成 16 位短 ID，兼顾唯一性和可读性。
    return uuid.uuid4().hex[:16]


class BBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates (x, y, w, h)."""

    x: float
    y: float
    w: float
    h: float


class Polygon(BaseModel):
    """Arbitrary polygon as list of (x, y) points."""

    # 中文说明：当 bbox 不够精确时，可以用 polygon 表示更贴合目标边缘的区域。
    points: list[tuple[float, float]] = Field(default_factory=list)


class RegionMask(BaseModel):
    """
    Region on an image, using bbox, optional polygon refinement,
    and optional binary mask path.
    """

    # 中文说明：bbox 是最基本的区域表达。
    bbox: BBox

    # 中文说明：polygon 用于表达更细粒度边界，常由分割模型提供。
    polygon: Optional[Polygon] = None

    # 中文说明：mask_path 可指向二值 mask 文件，适合更复杂的图像区域表达。
    mask_path: Optional[str] = None

    confidence: float = 0.0


class SourceSpec(BaseModel):
    """Describes the source of an image to process."""

    # 中文说明：type 表示输入的来源形式，目前常见是 file / url / base64。
    type: str = "file"

    # 中文说明：uri 保存具体资源定位信息。
    uri: str

    # 中文说明：mime_type 用于在任务开始前做基础输入校验。
    mime_type: str = "image/png"


class PictureAsset(BaseModel):
    """Represents a single image asset within a job."""

    asset_id: str = Field(default_factory=_new_id)
    original_uri: str = ""
    preprocessed_uri: Optional[str] = None
    width: int = 0
    height: int = 0
    mime_type: str = ""

    # 中文说明：page_index 用于多页 PDF 等场景，标记当前资产对应的页码。
    page_index: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OCRTextBlock(BaseModel):
    """A single text block from OCR."""

    text: str
    bbox: BBox
    polygon: Optional[Polygon] = None
    confidence: float = 0.0
    language: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class LayoutRegion(BaseModel):
    """A layout region (paragraph, table, figure, etc.)."""

    # 中文说明：region_type 用来区分段落、表格、图片、页眉页脚等版面区域。
    region_type: str = "text"
    bbox: BBox
    text_blocks: list[OCRTextBlock] = Field(default_factory=list)


class OCRLayoutResult(BaseModel):
    """Result from OCR + layout analysis."""

    # 中文说明：full_text 是把所有 OCR 文本拼成的完整文本，便于整体送入 PII 检测器。
    full_text: str = ""
    text_blocks: list[OCRTextBlock] = Field(default_factory=list)
    layout_regions: list[LayoutRegion] = Field(default_factory=list)
    engine_name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class PictureFinding(BaseModel):
    """A single compliance finding on an image."""

    finding_id: str = Field(default_factory=_new_id)

    # 中文说明：finding_type 区分该 finding 来自文本 PII、视觉检测还是其他能力。
    finding_type: FindingType

    # 中文说明：category 是更细粒度的命中类别，例如 phone_number、face、qr_code。
    category: str = ""
    label: str = ""
    score: float = 0.0

    # 中文说明：region 是图像坐标信息，脱敏时主要依赖这里。
    region: Optional[RegionMask] = None

    # 中文说明：text_span 记录原始文本片段，通常只在文本 PII 场景下有值。
    text_span: Optional[str] = None

    # 中文说明：reason_code 用于策略与报告中的稳定代码表达。
    reason_code: str = ""

    # 中文说明：provider 记录该 finding 来自哪个具体能力实现。
    provider: str = ""
    # ── 新增：provider 追溯与可解释性 ──
    provider_version: str = ""
    threshold_used: float = 0.0
    explanation: str = ""              # 人类可读的判定原因
    metadata: dict[str, Any] = Field(default_factory=dict)


class PictureModerationResult(BaseModel):
    """Safety moderation result for an image."""

    is_safe: bool = True

    # 中文说明：categories 记录命中的审核类别枚举。
    categories: list[SafetyCategory] = Field(default_factory=list)

    # 中文说明：scores 按类别保存打分，便于策略层做阈值比较。
    scores: dict[str, float] = Field(default_factory=dict)

    # 中文说明：reason_codes 是更适合审计与对外输出的规则代码。
    reason_codes: list[str] = Field(default_factory=list)
    provider: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RedactionOperation(BaseModel):
    """Records a single redaction action performed on an image."""

    # 中文说明：finding_id 建立“发现 -> 脱敏动作”的可追溯关系。
    finding_id: str = ""
    region: RegionMask
    mode: RedactionMode = RedactionMode.BLACK_BOX

    # 中文说明：applied 可用于标记规划过但最终未实际执行的操作。
    applied: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class PicturePolicyResult(BaseModel):
    """Final policy evaluation result."""

    decision: DecisionType = DecisionType.PASS_RAW
    dataset_action: str = "deliver_raw"
    compliance_decision: str = "pass_raw"
    review_required: bool = False
    requires_restricted_dataset: bool = False
    authorization_required: bool = False
    access_control_required: bool = False
    audit_required: bool = False
    restricted_reason: str = ""
    redaction_strategy: str = "minimal_identity_redaction"
    education_value_preserved: bool = True
    annotation_guidance_zh: str = ""
    preserved_learning_content: list[str] = Field(default_factory=list)
    redacted_identity_content: list[str] = Field(default_factory=list)
    executed_steps: list[str] = Field(default_factory=list)
    skipped_steps: list[dict[str, Any]] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    profile: str = "default_cn_enterprise"
    evaluated_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    # ── 新增：降级事件、可信等级与复核摘要 ──
    degrade_events: list[dict[str, Any]] = Field(default_factory=list)
    trust_level: str = "full"
    score_breakdown: Optional[dict[str, Any]] = None
    review_summary: str = ""
    precheck: dict[str, Any] = Field(default_factory=dict)
    step_audits: list[dict[str, Any]] = Field(default_factory=list)
    policy_snapshot: dict[str, Any] = Field(default_factory=dict)


class PictureJob(BaseModel):
    """Top-level job entity for picture compliance processing."""

    # 中文说明：job_id 使用带前缀的短 ID，日志和接口里更容易辨认。
    job_id: str = Field(default_factory=lambda: f"job_{uuid.uuid4().hex[:12]}")
    tenant_id: str = ""
    status: JobStatus = JobStatus.CREATED
    source: SourceSpec = Field(default_factory=lambda: SourceSpec(uri=""))
    route: Optional[RouteType] = None
    profile: str = "default_cn_enterprise"
    options: dict[str, Any] = Field(default_factory=dict)
    contract_version: str = "compliance-job.v1"
    platform_task_id: str = ""
    idempotency_key: str = ""
    effective_request: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error_info: Optional[dict[str, Any]] = None

    # 中文说明：以下字段是在各处理阶段逐步填充的中间结果。
    asset: Optional[PictureAsset] = None
    ocr_result: Optional[OCRLayoutResult] = None
    moderation_result: Optional[PictureModerationResult] = None
    findings: list[PictureFinding] = Field(default_factory=list)
    text_content_findings: list[PictureFinding] = Field(default_factory=list)
    redaction_operations: list[RedactionOperation] = Field(default_factory=list)
    policy_result: Optional[PicturePolicyResult] = None
    precheck: dict[str, Any] = Field(default_factory=dict)
    step_audits: list[dict[str, Any]] = Field(default_factory=list)

    # 中文说明：以下字段指向最终可交付产物及审计产物的位置。
    compliant_image_uri: Optional[str] = None
    overlay_image_uri: Optional[str] = None
    report_uri: Optional[str] = None

    # 中文说明：时间字段分别反映任务创建、最近更新、最终完成的时刻。
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = None

    # 中文说明：失败时记录可直接展示的错误摘要和更细的错误类型。
    error: Optional[str] = None
    error_detail: Optional[str] = None

    # 中文说明：step_latencies 用于性能诊断，provider_versions 用于审计 provider 版本来源。
    step_latencies: dict[str, float] = Field(default_factory=dict)
    provider_versions: dict[str, str] = Field(default_factory=dict)

    # ── 新增：降级事件、可信等级与双轨交付物 ──
    degrade_events: list[dict[str, Any]] = Field(default_factory=list)
    trust_level: str = "full"
    annotation_package_uri: Optional[str] = None
    audit_package_uri: Optional[str] = None
