"""
Pydantic schemas for the audio compliance pipeline.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    # 统一使用 UTC 时间戳，避免跨时区比较带来的歧义。
    return datetime.now(timezone.utc)


# 输入源类型枚举：用于前期分类与后续步骤分流。
class SourceType(str, Enum):
    AUDIO = "audio"
    ARCHIVE = "archive"
    REPO = "repo"
    MIXED = "mixed"


class Decision(str, Enum):
    ALLOW = "allow"
    REVIEW = "review"
    QUARANTINE = "quarantine"
    REJECT = "reject"


class SafetyLevel(str, Enum):
    SAFE = "safe"
    CONTROVERSIAL = "controversial"
    UNSAFE = "unsafe"


class RenderStrategy(str, Enum):
    SILENCE = "silence"
    BEEP = "beep"
    COPY = "copy"


class DeliveryStatus(str, Enum):
    DELIVER = "deliver"
    HOLD = "hold"
    BLOCK = "block"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# 源文件登记信息（入口步骤输出）。
class SourceRecord(BaseModel):
    source_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    path: str
    size_bytes: int = 0
    sha256: str = ""
    mime_type: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# 源分类结果，补充类型与元数据。
class SourceProfile(BaseModel):
    source_id: str
    path: str
    source_type: SourceType
    mime_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# 密钥/凭据类命中记录。
class SecretHit(BaseModel):
    source_id: str
    detector_type: str = ""
    decoder_type: str = ""
    raw_value: str = ""
    redacted: str = ""
    file_path: str = ""
    line_number: int = 0
    verified: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


# 单条许可证匹配片段。
class LicenseMatch(BaseModel):
    license_expression: str = ""
    spdx_id: str = ""
    score: float = 0.0
    matched_text: str = ""
    start_line: int = 0
    end_line: int = 0


# 合规扫描命中，可能来自同一源中的多个文件。
class ComplianceHit(BaseModel):
    source_id: str
    file_path: str = ""
    licenses: list[LicenseMatch] = Field(default_factory=list)
    copyrights: list[str] = Field(default_factory=list)
    scan_errors: list[str] = Field(default_factory=list)


# 归一化音频信息（后续 ASR/脱敏基准输入）。
class NormalizedAudioRecord(BaseModel):
    source_id: str
    original_path: str
    normalized_path: str
    sample_rate: int = 0
    channels: int = 0
    codec: str = ""
    duration_seconds: float = 0.0
    engine_name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ASR 时间片段。
class ASRSegment(BaseModel):
    segment_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_id: str
    start_time: float
    end_time: float
    text: str
    confidence: float = 0.0
    engine_name: str = ""
    language: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# 说话人分离片段。
class SpeakerSegment(BaseModel):
    speaker_segment_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_id: str
    speaker_id: str
    start_time: float
    end_time: float
    confidence: float = 0.0
    engine_name: str = ""


# 统一转写单元（融合 ASR 与说话人信息）。
class TranscriptUnit(BaseModel):
    unit_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_id: str
    start_time: float
    end_time: float
    speaker_id: str = "speaker_0"
    text: str = ""
    confidence: float = 0.0
    engine_name: str = ""
    language: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# 去重后的转写单元，记录重复关系。
class DedupTranscriptUnit(BaseModel):
    unit_id: str
    source_id: str
    start_time: float
    end_time: float
    speaker_id: str
    text: str
    confidence: float = 0.0
    engine_name: str = ""
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None


# 去重映射记录（当前 unit 指向原始 unit）。
class DedupMapEntry(BaseModel):
    unit_id: str
    duplicate_of: str
    jaccard_similarity: float = 0.0


# 关键词命中证据。
class KeywordHit(BaseModel):
    unit_id: str
    keyword: str
    start_pos: int = 0
    end_pos: int = 0
    context: str = ""


# 正则命中证据。
class RegexHit(BaseModel):
    unit_id: str
    pattern_name: str
    pattern: str = ""
    matched_text: str = ""
    start_pos: int = 0
    end_pos: int = 0
    context: str = ""


# 单条 PII 实体识别结果。
class PIIEntity(BaseModel):
    entity_type: str
    start: int
    end: int
    score: float = 0.0
    original_text: str = ""


# 需要在音频层进行替换/静音的时间区间。
class RedactionSpan(BaseModel):
    source_id: str
    unit_id: str
    start_time: float
    end_time: float
    entity_type: str
    original_text: str
    replacement: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# 文本隐私检测与脱敏结果。
class PrivacyResult(BaseModel):
    unit_id: str
    source_id: str
    original_text: str = ""
    redacted_text: str = ""
    pii_entities: list[PIIEntity] = Field(default_factory=list)
    pii_count: int = 0
    # ── 新增：保留原文标记与 provider 追溯 ──
    original_text_preserved: bool = True
    provider_name: str = ""
    provider_version: str = ""
    is_degraded: bool = False


# 文本安全审核结果。
class SafetyResult(BaseModel):
    unit_id: str
    source_id: str
    safety_level: SafetyLevel = SafetyLevel.SAFE
    harm_categories: list[str] = Field(default_factory=list)
    raw_output: str = ""
    score: float = 1.0
    # ── 新增：可解释性与 provider 追溯 ──
    explanation: str = ""
    provider_name: str = ""
    model_version: str = ""
    threshold_used: float = 0.0
    is_degraded: bool = False


# 面向策略引擎的单转写单元证据聚合结构。
# Hard-case adjudication output for uncertain audio units.
class AudioHardCaseJudgement(BaseModel):
    content_status: str = "clear"
    privacy_status: str = "clear"
    confidence: float = 0.0
    rationale: str = ""
    recommended_decision: Decision = Decision.REVIEW
    requires_manual_review: bool = True
    final_reasons: list[str] = Field(default_factory=list)


class AudioHardCaseResult(BaseModel):
    run_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    unit_id: str
    source_id: str
    trigger_sources: list[str] = Field(default_factory=list)
    trigger_reasons: list[str] = Field(default_factory=list)
    model_name: str = ""
    provider_name: str = ""
    prompt_version: str = ""
    adjudicated: bool = False
    is_degraded: bool = False
    uncertainty: float = 1.0
    judgement: AudioHardCaseJudgement = Field(default_factory=AudioHardCaseJudgement)
    raw_response: str = ""
    notes: list[str] = Field(default_factory=list)


# Aggregated evidence for policy decision.
class TranscriptEvidence(BaseModel):
    unit_id: str
    source_id: str
    text: str = ""
    speaker_id: str = "speaker_0"
    is_duplicate: bool = False
    secret_hits: list[SecretHit] = Field(default_factory=list)
    compliance_hits: list[ComplianceHit] = Field(default_factory=list)
    keyword_hits: list[KeywordHit] = Field(default_factory=list)
    regex_hits: list[RegexHit] = Field(default_factory=list)
    privacy: Optional[PrivacyResult] = None
    safety: Optional[SafetyResult] = None
    hard_case: Optional[AudioHardCaseResult] = None
    # ── 新增：降级事件与可信等级 ──
    degrade_events: list[dict[str, Any]] = Field(default_factory=list)
    trust_level: str = "full"


# 全量证据包：策略决策输入主对象。
class EvidenceBundle(BaseModel):
    pipeline_run_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    transcript_units: list[TranscriptEvidence] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    # ── 新增：全局降级与可信等级 ──
    degrade_events: list[dict[str, Any]] = Field(default_factory=list)
    trust_level: str = "full"


# 单转写单元的最终策略决策。
class UnitDecision(BaseModel):
    unit_id: str
    decision: Decision = Decision.REVIEW
    reasons: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)


# 全局策略决策输出。
class PolicyDecision(BaseModel):
    pipeline_run_id: str
    overall_decision: Decision = Decision.REVIEW
    unit_decisions: list[UnitDecision] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=_utcnow)
    # ── 新增：可信等级与降级摘要 ──
    trust_level: str = "full"
    degrade_summary: str = ""
    profile_name: str = "default"


class AudioAnnotationRecord(BaseModel):
    run_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    package_record_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    unit_id: str
    source_id: str
    original_text: str = ""
    redacted_view: str = ""
    delivery_status: DeliveryStatus = DeliveryStatus.HOLD
    decision: Decision = Decision.REVIEW
    review_priority: str = "normal"
    start_time: float = 0.0
    end_time: float = 0.0
    speaker_id: str = "speaker_0"
    redaction_spans: list[RedactionSpan] = Field(default_factory=list)
    annotation_hints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudioAuditRecord(BaseModel):
    run_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    audit_record_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    unit_id: str
    source_id: str
    transcript: TranscriptUnit
    privacy_result: Optional[PrivacyResult] = None
    safety_result: Optional[SafetyResult] = None
    hard_case_result: Optional[AudioHardCaseResult] = None
    redaction_spans: list[RedactionSpan] = Field(default_factory=list)
    evidence: Optional[TranscriptEvidence] = None
    decision: Optional[UnitDecision] = None
    provider_manifest: dict[str, str] = Field(default_factory=dict)
    trust_level: str = "full"
    audit_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudioRunSummaryRecord(BaseModel):
    run_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    summary_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    processed_units: int = 0
    processed_sources: int = 0
    overall_decision: Decision = Decision.ALLOW
    counts_by_decision: dict[str, int] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    review_suggestions: list[str] = Field(default_factory=list)
    explanation_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# 脱敏音频产物记录。
class RedactedAudioRecord(BaseModel):
    source_id: str
    original_audio_path: str
    redacted_audio_path: str
    render_strategy: RenderStrategy = RenderStrategy.COPY
    span_count: int = 0
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReleasePackage(BaseModel):
    pipeline_run_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    original_audio: list[NormalizedAudioRecord] = Field(default_factory=list)
    transcript_summary: dict[str, Any] = Field(default_factory=dict)
    decision: Optional[PolicyDecision] = None
    evidence_summary: dict[str, Any] = Field(default_factory=dict)
    redacted_audio: list[RedactedAudioRecord] = Field(default_factory=list)
    audit_metadata: dict[str, Any] = Field(default_factory=dict)
    # ── 新增：双轨交付物 URI ──
    annotation_package_uri: str = ""
    audit_package_uri: str = ""
    trust_level: str = "full"


class CheckRequest(BaseModel):
    contract_version: str = "compliance-job.v1"
    platform_task_id: str = ""
    idempotency_key: str = ""
    modality: str = "audio"
    operator_id: str = ""
    operator_catalog_version: str = "audio-compliance-operators.v1"
    input_paths: list[str]
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class CheckTaskInfo(BaseModel):
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    contract_version: str = "compliance-job.v1"
    platform_task_id: str = ""
    idempotency_key: str = ""
    modality: str = "audio"
    stage: str = ""
    progress: int = 0
    status_label: str = ""
    effective_request: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error_info: Optional[dict[str, Any]] = None
