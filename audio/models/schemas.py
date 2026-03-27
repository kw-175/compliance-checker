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
    return datetime.now(timezone.utc)


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


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SourceRecord(BaseModel):
    source_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    path: str
    size_bytes: int = 0
    sha256: str = ""
    mime_type: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class SourceProfile(BaseModel):
    source_id: str
    path: str
    source_type: SourceType
    mime_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


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


class LicenseMatch(BaseModel):
    license_expression: str = ""
    spdx_id: str = ""
    score: float = 0.0
    matched_text: str = ""
    start_line: int = 0
    end_line: int = 0


class ComplianceHit(BaseModel):
    source_id: str
    file_path: str = ""
    licenses: list[LicenseMatch] = Field(default_factory=list)
    copyrights: list[str] = Field(default_factory=list)
    scan_errors: list[str] = Field(default_factory=list)


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


class SpeakerSegment(BaseModel):
    speaker_segment_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_id: str
    speaker_id: str
    start_time: float
    end_time: float
    confidence: float = 0.0
    engine_name: str = ""


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


class DedupMapEntry(BaseModel):
    unit_id: str
    duplicate_of: str
    jaccard_similarity: float = 0.0


class KeywordHit(BaseModel):
    unit_id: str
    keyword: str
    start_pos: int = 0
    end_pos: int = 0
    context: str = ""


class RegexHit(BaseModel):
    unit_id: str
    pattern_name: str
    pattern: str = ""
    matched_text: str = ""
    start_pos: int = 0
    end_pos: int = 0
    context: str = ""


class PIIEntity(BaseModel):
    entity_type: str
    start: int
    end: int
    score: float = 0.0
    original_text: str = ""


class RedactionSpan(BaseModel):
    source_id: str
    unit_id: str
    start_time: float
    end_time: float
    entity_type: str
    original_text: str
    replacement: str


class PrivacyResult(BaseModel):
    unit_id: str
    source_id: str
    original_text: str = ""
    redacted_text: str = ""
    pii_entities: list[PIIEntity] = Field(default_factory=list)
    pii_count: int = 0


class SafetyResult(BaseModel):
    unit_id: str
    source_id: str
    safety_level: SafetyLevel = SafetyLevel.SAFE
    harm_categories: list[str] = Field(default_factory=list)
    raw_output: str = ""
    score: float = 1.0


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


class EvidenceBundle(BaseModel):
    pipeline_run_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    transcript_units: list[TranscriptEvidence] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class UnitDecision(BaseModel):
    unit_id: str
    decision: Decision = Decision.REVIEW
    reasons: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)


class PolicyDecision(BaseModel):
    pipeline_run_id: str
    overall_decision: Decision = Decision.REVIEW
    unit_decisions: list[UnitDecision] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=_utcnow)


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


class CheckRequest(BaseModel):
    input_paths: list[str]
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class CheckTaskInfo(BaseModel):
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = None
    result: Optional[PolicyDecision] = None
    error: Optional[str] = None
