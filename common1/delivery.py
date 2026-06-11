# ──────────────────────────────────────────────────────────────
# 统一交付物模型
# ──────────────────────────────────────────────────────────────
#
# 定义双轨交付物：AnnotationPackage（标注样本包）和
# AuditPackage（审计证据包），以及统一的 ReleasePackage。
#
# 核心理念：
#   - 标注样本包保留原始内容可用性，不做破坏性替换
#   - 审计证据包包含完整的决策链路和可追溯信息
#   - 两者通过 ReleasePackage 统一交付给下游系统
# ──────────────────────────────────────────────────────────────

"""跨模态统一交付物模型：AnnotationPackage / AuditPackage / ReleasePackage。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from common.enums import Modality, TrustLevel, UnifiedDecision
from common.evidence import DegradeEvent, EvidenceUnit
from common.policy import PolicyResult, ScoreBreakdown


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── 风险片段 ─────────────────────────────────────────────


class RiskSegment(BaseModel):
    """
    结构化风险片段。

    描述内容中被检测到风险的片段，保留原始内容（不做替换），
    同时附带标注建议和替换建议。

    与现有 redacted_text 中直接替换占位符的区别：
    - 原始内容被保留，标注系统可以决定是否展示
    - 附带 replacement_suggestion 而非直接执行替换
    - 支持争议标记（is_disputed）
    """
    segment_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    evidence_id: str = ""             # 关联的 EvidenceUnit
    original_content: str = ""        # 原始内容片段
    replacement_suggestion: str = ""  # 建议的替换内容（如 <PHONE>）
    risk_level: str = ""              # high / medium / low
    category: str = ""                # pii / safety / license 等
    sub_category: str = ""            # person_name / phone_number 等
    is_disputed: bool = False         # 是否为争议片段
    dispute_reason: str = ""          # 争议原因
    annotation_hint: str = ""         # 给标注员的提示信息


# ── 标注样本包 ───────────────────────────────────────────


class AnnotationPackage(BaseModel):
    """
    标注样本包。

    面向下游标注系统的直接可消费产物。关键设计原则：
    1. 保留原始内容的可用性（不做破坏性替换）
    2. 以结构化风险片段形式标注风险区域
    3. 附带标注建议和争议说明

    与当前系统输出 redacted_text 的区别：
    - clean_content_uri 指向原始内容（非脱敏版本）
    - risk_segments 以非侵入式方式标记风险
    - 标注系统可自行决定展示策略
    """
    package_id: str = Field(default_factory=lambda: f"ann_{uuid.uuid4().hex[:12]}")
    modality: Modality
    pipeline_run_id: str = ""

    # 内容引用
    clean_content_uri: str = ""       # 可标注正文/音频/图像/视频的 URI
    content_format: str = ""          # text/plain, audio/wav, image/png, video/mp4

    # 风险标注
    risk_segments: list[RiskSegment] = Field(default_factory=list)
    dispute_segments: list[RiskSegment] = Field(default_factory=list)

    # 标注建议
    annotation_hints: list[str] = Field(default_factory=list)
    review_priority: str = "normal"   # critical / high / normal / low

    # 决策上下文（让标注系统了解为什么需要标注）
    decision: UnifiedDecision = UnifiedDecision.REVIEW
    trust_level: TrustLevel = TrustLevel.FULL

    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── 审计证据包 ───────────────────────────────────────────


class AuditPackage(BaseModel):
    """
    审计证据包。

    面向审计系统和复核人员的完整可验证产物。包含：
    - 所有证据单元（精确定位 + provider 信息）
    - 降级事件链（每次 fallback/mock/failure 的记录）
    - 完整评分分解（每个维度的分数和规则命中）
    - provider 版本清单
    - 处理时间线

    与当前系统仅输出 decision + reasons 的区别：
    - 可回答"为什么判成 review / reject"
    - 可回答"具体命中了哪里"
    - 可回答"用了哪个模型、哪个 provider、哪个阈值"
    - 可回答"为什么发生降级"
    """
    package_id: str = Field(default_factory=lambda: f"aud_{uuid.uuid4().hex[:12]}")
    modality: Modality
    pipeline_run_id: str = ""

    # 证据链
    evidence_units: list[EvidenceUnit] = Field(default_factory=list)
    degrade_events: list[DegradeEvent] = Field(default_factory=list)

    # 评分与决策
    score_breakdown: Optional[ScoreBreakdown] = None
    policy_result: Optional[PolicyResult] = None

    # Provider 清单
    provider_manifest: dict[str, str] = Field(default_factory=dict)

    # 处理时间线
    processing_timeline: dict[str, float] = Field(default_factory=dict)  # step → ms

    # 复核摘要
    review_summary: str = ""          # 人类可读的复核摘要
    review_suggestions: list[str] = Field(default_factory=list)

    # 可信性
    trust_level: TrustLevel = TrustLevel.FULL
    trust_explanation: str = ""

    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── 统一发布包 ───────────────────────────────────────────


class ReleasePackage(BaseModel):
    """
    统一发布包。

    包含标注样本包和审计证据包，作为流水线的一等交付物。

    下游系统可分别消费：
    - 标注系统 → annotation_package
    - 审计系统 → audit_package
    - 风控系统 → 两者均需
    """
    release_id: str = Field(default_factory=lambda: f"rel_{uuid.uuid4().hex[:12]}")
    modality: Modality
    pipeline_run_id: str = ""

    # 双轨交付物
    annotation_package: Optional[AnnotationPackage] = None
    audit_package: Optional[AuditPackage] = None

    # 顶层决策摘要
    decision: UnifiedDecision = UnifiedDecision.REVIEW
    trust_level: TrustLevel = TrustLevel.FULL

    # URI 索引
    annotation_package_uri: str = ""
    audit_package_uri: str = ""

    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
