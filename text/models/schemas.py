from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from common.enums import Modality, TrustLevel, UnifiedDecision


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionStatus(str, Enum):
    CLEAR = "clear"
    FLAGGED = "flagged"
    HARD_CASE = "hard_case"


class DispositionLevel(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"
    P5 = "P5"


class DeliveryStatus(str, Enum):
    DELIVER = "deliver"
    HOLD = "hold"
    BLOCK = "block"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ArtifactRecord(BaseModel):
    run_id: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class PackageAsset(BaseModel):
    asset_type: str
    uri: str
    role: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestUnit(ArtifactRecord):
    package_id: str
    doc_id: str
    source_path: str
    text: str
    text_hash: str
    language: str = ""
    source_type: str = ""
    task_id: str = ""
    tenant_id: str = ""
    profile_id: str = ""
    file_hash: str = ""
    package_kind: str = ""
    parser_name: str = ""
    candidate_profiles: list[str] = Field(default_factory=list)
    raw_text_refs: list[PackageAsset] = Field(default_factory=list)
    cleaned_data_refs: list[PackageAsset] = Field(default_factory=list)
    metadata_refs: list[PackageAsset] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    extensions: dict[str, Any] = Field(default_factory=dict)


class TextSpan(BaseModel):
    start: int
    end: int
    text: str = ""
    context_before: str = ""
    context_after: str = ""


class DetectionFinding(BaseModel):
    finding_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    doc_id: str
    finding_type: str
    risk_type: str
    policy_tag: str
    severity: Severity
    confidence: float
    explanation: str
    source_tool: str
    remediation_suggestion: str = ""
    redaction_suggestion: str = ""
    needs_adjudication: bool = False
    hard_case_reason: str = ""
    span: TextSpan | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class ContentSafetyResult(ArtifactRecord):
    doc_id: str
    text_hash: str
    status: DetectionStatus = DetectionStatus.CLEAR
    risk_score: float = 0.0
    summary: str = ""
    findings: list[DetectionFinding] = Field(default_factory=list)
    needs_adjudication: bool = False
    hard_case_reasons: list[str] = Field(default_factory=list)
    provider_name: str = "rule_safety_detector"
    provider_version: str = "builtin-2026.04"
    is_degraded: bool = False


class PrivacyDetectionResult(ArtifactRecord):
    doc_id: str
    text_hash: str
    pii_count: int = 0
    risk_score: float = 0.0
    summary: str = ""
    findings: list[DetectionFinding] = Field(default_factory=list)
    needs_adjudication: bool = False
    hard_case_reasons: list[str] = Field(default_factory=list)
    provider_name: str = "rule_pii_detector"
    provider_version: str = "builtin-2026.04"
    is_degraded: bool = False


class HardCaseJudgement(BaseModel):
    content_status: str = "clear"
    privacy_status: str = "clear"
    confidence: float = 0.0
    rationale: str = ""
    recommended_disposition: DispositionLevel = DispositionLevel.P3
    requires_manual_review: bool = True
    final_findings: list[DetectionFinding] = Field(default_factory=list)


class HardCaseAdjudicationResult(ArtifactRecord):
    doc_id: str
    trigger_sources: list[str] = Field(default_factory=list)
    model_name: str
    provider_name: str
    prompt_version: str
    adjudicated: bool = False
    is_degraded: bool = False
    uncertainty: float = 1.0
    judgement: HardCaseJudgement = Field(default_factory=HardCaseJudgement)
    raw_response: str = ""
    notes: list[str] = Field(default_factory=list)


class EvidenceEvent(ArtifactRecord):
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    doc_id: str
    category: str
    risk_type: str
    policy_tag: str
    severity: Severity
    confidence_summary: float
    source_tools: list[str] = Field(default_factory=list)
    finding_refs: list[str] = Field(default_factory=list)
    disputed: bool = False
    hard_case_applied: bool = False
    remediation_suggestion: str = ""
    explanation: str = ""
    primary_span: TextSpan | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RedactionTarget(BaseModel):
    finding_id: str
    event_id: str
    start: int
    end: int
    original_text: str = ""
    replacement: str = ""
    pii_type: str = ""


class PolicyDecisionRecord(ArtifactRecord):
    decision_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    doc_id: str
    disposition_level: DispositionLevel
    unified_decision: UnifiedDecision
    risk_score: float = 0.0
    required_actions: list[str] = Field(default_factory=list)
    redaction_targets: list[RedactionTarget] = Field(default_factory=list)
    redaction_method: str = ""
    blocked_reason: str = ""
    review_priority: str = "normal"
    reason_codes: list[str] = Field(default_factory=list)
    evidence_event_ids: list[str] = Field(default_factory=list)
    summary: str = ""
    policy_version: str = ""
    trust_level: TrustLevel = TrustLevel.FULL


class AnnotationPackageRecord(ArtifactRecord):
    package_record_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    doc_id: str
    original_text: str
    redacted_view: str
    delivery_status: DeliveryStatus
    disposition_level: DispositionLevel
    unified_decision: UnifiedDecision
    review_priority: str = "normal"
    span_annotations: list[RedactionTarget] = Field(default_factory=list)
    evidence_event_ids: list[str] = Field(default_factory=list)
    annotation_hints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditPackageRecord(ArtifactRecord):
    audit_record_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    doc_id: str
    ingest_unit: IngestUnit
    safety_result: ContentSafetyResult | None = None
    privacy_result: PrivacyDetectionResult | None = None
    hard_case_result: HardCaseAdjudicationResult | None = None
    evidence_events: list[EvidenceEvent] = Field(default_factory=list)
    decision: PolicyDecisionRecord
    provider_manifest: dict[str, str] = Field(default_factory=dict)
    trust_level: TrustLevel = TrustLevel.FULL
    degrade_reasons: list[str] = Field(default_factory=list)
    audit_summary: str = ""


class RunSummaryRecord(ArtifactRecord):
    summary_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    modality: Modality = Modality.TEXT
    processed_documents: int = 0
    overall_disposition: DispositionLevel = DispositionLevel.P0
    unified_decision: UnifiedDecision = UnifiedDecision.ALLOW
    trust_level: TrustLevel = TrustLevel.FULL
    counts_by_disposition: dict[str, int] = Field(default_factory=dict)
    counts_by_decision: dict[str, int] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    review_suggestions: list[str] = Field(default_factory=list)
    explanation_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckRequest(BaseModel):
    package_paths: list[str] = Field(default_factory=list)
    input_paths: list[str] = Field(default_factory=list)
    config_overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_paths(self) -> "CheckRequest":
        if not self.package_paths and self.input_paths:
            self.package_paths = list(self.input_paths)
        if not self.package_paths:
            raise ValueError("package_paths is required")
        return self


class CheckTaskInfo(BaseModel):
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
