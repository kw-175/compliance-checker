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


class DocumentContextRecord(ArtifactRecord):
    doc_id: str
    text_hash: str
    topic: str = ""
    document_type: str = "unknown"
    scene_type: str = "unknown"
    subject_type: str = "unknown"
    source_type: str = ""
    usage_target: str = "training_dataset"
    contains_education_context: bool = False
    contains_minor_context: bool = False
    confidence: float = 0.0
    summary: str = ""
    explanation: str = ""
    provider_name: str = "heuristic_document_context"
    provider_version: str = "builtin-2026.05"
    is_degraded: bool = False
    attributes: dict[str, Any] = Field(default_factory=dict)


class ContentCandidateWindowRecord(ArtifactRecord):
    doc_id: str
    window_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    start: int = 0
    end: int = 0
    text: str = ""
    candidate_labels: list[str] = Field(default_factory=list)
    candidate_score: float = 0.0
    recall_sources: list[str] = Field(default_factory=list)
    rule_hits: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContentLocalizedFragmentRecord(ArtifactRecord):
    doc_id: str
    fragment_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    window_id: str = ""
    risk_type: str = ""
    policy_tag: str = ""
    severity: Severity = Severity.MEDIUM
    confidence: float = 0.0
    explanation: str = ""
    span: TextSpan | None = None
    source_tool: str = ""
    is_degraded: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class PrivacyFragmentAdjudicationRecord(ArtifactRecord):
    doc_id: str
    finding_id: str
    risk_type: str = ""
    fragment_truth: str = "uncertain"
    governance_action: str = "manual_review"
    can_keep: bool = False
    requires_manual_review: bool = True
    training_impact: str = ""
    annotation_impact: str = ""
    explanation: str = ""
    confidence: float = 0.0
    provider_name: str = "privacy_fragment_adjudicator"
    provider_version: str = "builtin-2026.05"
    is_degraded: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContentFragmentAdjudicationRecord(ArtifactRecord):
    doc_id: str
    finding_id: str
    risk_type: str = ""
    semantic_role: str = "uncertain"
    operationality: str = "medium"
    audience_risk: str = "normal"
    protective_context: bool = False
    recommended_action: str = "manual_review"
    training_eligibility: str = "restricted"
    allow_downstream_annotation: bool = False
    requires_manual_review: bool = True
    explanation: str = ""
    confidence: float = 0.0
    provider_name: str = "content_fragment_adjudicator"
    provider_version: str = "builtin-2026.05"
    is_degraded: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class PrivacyDocumentAssessmentRecord(ArtifactRecord):
    doc_id: str
    text_hash: str
    overall_risk_level: str = "low"
    combination_risk: bool = False
    training_suitability: str = "restricted"
    annotation_suitability: str = "restricted"
    recommended_action: str = "manual_review"
    requires_manual_review: bool = True
    explanation: str = ""
    confidence: float = 0.0
    provider_name: str = "privacy_document_assessor"
    provider_version: str = "builtin-2026.05"
    is_degraded: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContentDocumentAssessmentRecord(ArtifactRecord):
    doc_id: str
    text_hash: str
    overall_stance: str = "uncertain"
    operational_risk: str = "medium"
    training_suitability: str = "restricted"
    annotation_suitability: str = "restricted"
    recommended_action: str = "manual_review"
    requires_manual_review: bool = True
    explanation: str = ""
    confidence: float = 0.0
    provider_name: str = "content_document_assessor"
    provider_version: str = "builtin-2026.05"
    is_degraded: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


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


class RedactionConflict(BaseModel):
    conflict_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    conflict_type: str = "overlap"
    start: int
    end: int
    text: str = ""
    selected_finding_id: str = ""
    selected_risk_type: str = ""
    suppressed_finding_ids: list[str] = Field(default_factory=list)
    suppressed_risk_types: list[str] = Field(default_factory=list)
    resolution_source: str = "deterministic_priority"
    rationale: str = ""


class SpanConflictResolutionResult(ArtifactRecord):
    doc_id: str
    text_hash: str
    input_finding_count: int = 0
    selected_span_count: int = 0
    suppressed_finding_count: int = 0
    redaction_targets: list[RedactionTarget] = Field(default_factory=list)
    conflicts: list[RedactionConflict] = Field(default_factory=list)
    needs_model_resolution: bool = False
    is_degraded: bool = False
    provider_name: str = "deterministic_span_conflict_resolver"
    provider_version: str = "builtin-2026.04"
    summary: str = ""


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
    explanation: str = ""
    policy_version: str = ""
    trust_level: TrustLevel = TrustLevel.FULL
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    document_context: DocumentContextRecord | None = None
    redaction_plan: SpanConflictResolutionResult | None = None
    hard_case_result: HardCaseAdjudicationResult | None = None
    content_candidate_windows: list[ContentCandidateWindowRecord] = Field(default_factory=list)
    content_localized_fragments: list[ContentLocalizedFragmentRecord] = Field(default_factory=list)
    privacy_fragment_adjudications: list[PrivacyFragmentAdjudicationRecord] = Field(default_factory=list)
    content_fragment_adjudications: list[ContentFragmentAdjudicationRecord] = Field(default_factory=list)
    privacy_document_assessment: PrivacyDocumentAssessmentRecord | None = None
    content_document_assessment: ContentDocumentAssessmentRecord | None = None
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


class ContentSafetyUpstreamInfo(BaseModel):
    data_stage: str = "cleaned"
    cleaning_version: str = ""
    cleaning_batch_id: str = ""
    dataset_id: str = ""
    sample_id: str = ""


class ContentSafetySceneMetadata(BaseModel):
    scene: str = "other"
    source_type: str = "other"
    audience: str = "unknown"
    visibility: str = "internal"
    speaker_role: str = "unknown"
    channel: str = "api"
    language: str = "unknown"
    extras: dict[str, Any] = Field(default_factory=dict)


class ContentSafetyTrainingContext(BaseModel):
    downstream_use: str = "other"
    training_split: str = ""
    target_model_stage: str = "unknown"
    curriculum_domain: str = "other"
    extras: dict[str, Any] = Field(default_factory=dict)


class ContentSafetyCheckRecord(BaseModel):
    doc_id: str = ""
    text: str
    upstream: ContentSafetyUpstreamInfo = Field(default_factory=ContentSafetyUpstreamInfo)
    metadata: ContentSafetySceneMetadata = Field(default_factory=ContentSafetySceneMetadata)
    training_context: ContentSafetyTrainingContext = Field(default_factory=ContentSafetyTrainingContext)


class ContentSafetyPolicyHit(BaseModel):
    policy_id: str
    hit: bool = True
    confidence: float = 0.0
    reason: str = ""
    evidence: list[str] = Field(default_factory=list)


class ContentSafetyEvidenceItem(BaseModel):
    label: str
    risk_type: str
    policy_tag: str
    severity: str
    confidence: float = 0.0
    text: str = ""
    start: int | None = None
    end: int | None = None
    explanation: str = ""
    source: str = ""


class ContentSafetyDecisionRecord(BaseModel):
    doc_id: str
    text_hash: str
    labels: list[str] = Field(default_factory=list)
    policy_hits: list[ContentSafetyPolicyHit] = Field(default_factory=list)
    context_type: str = ""
    risk_level: str = "C0"
    decision: str = "P0"
    training_eligibility: str = "T0"
    dataset_route: str = "general_training"
    allow_downstream_annotation: bool = True
    needs_manual_review: bool = False
    confidence: float = 0.0
    summary: str = ""
    evidence: list[ContentSafetyEvidenceItem] = Field(default_factory=list)
    explanation: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContentSafetyBatchCheckRequest(BaseModel):
    records: list[ContentSafetyCheckRecord]
    target_labels: list[str] = Field(default_factory=list)
    target_policies: list[str] = Field(default_factory=list)
    custom_policy: str = ""
    custom_policy_config: dict[str, Any] = Field(default_factory=dict)
    mode: str = "sync"


class ContentSafetyBatchCheckResponse(BaseModel):
    request_id: str
    overall_decision: str = "P0"
    overall_risk_level: str = "C0"
    overall_training_eligibility: str = "T0"
    overall_dataset_route: str = "general_training"
    review_suggestions: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    results: list[ContentSafetyDecisionRecord] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)


class CheckRequest(BaseModel):
    contract_version: str = "compliance-job.v1"
    platform_task_id: str = ""
    idempotency_key: str = ""
    modality: str = "text"
    operator_id: str = ""
    operator_catalog_version: str = "text-compliance-operators.v1"
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
    contract_version: str = "compliance-job.v1"
    platform_task_id: str = ""
    idempotency_key: str = ""
    modality: str = "text"
    stage: str = ""
    progress: int = 0
    status_label: str = ""
    effective_request: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error_info: dict[str, Any] | None = None
