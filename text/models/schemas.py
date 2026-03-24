"""
Pydantic data models for every JSONL / JSON artefact in the pipeline.

Each model corresponds to one line in the respective .jsonl output file,
or one top-level object in a .json output file.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ────────────────────────────────────────────────────────────
# Enums
# ────────────────────────────────────────────────────────────
class SourceType(str, Enum):
    CODE = "code"
    REPO = "repo"
    PACKAGE = "package"
    BINARY = "binary"
    WEB_TEXT = "web_text"
    PDF_TEXT = "pdf_text"
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


# ────────────────────────────────────────────────────────────
# Step A  –  source_registry.jsonl
# ────────────────────────────────────────────────────────────
class SourceRecord(BaseModel):
    """One row in source_registry.jsonl."""
    source_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    path: str
    size_bytes: int = 0
    sha256: str = ""
    mime_type: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


# ────────────────────────────────────────────────────────────
# Step B1  –  source_profile.jsonl
# ────────────────────────────────────────────────────────────
class SourceProfile(BaseModel):
    """Enriched source record with classification."""
    source_id: str
    path: str
    source_type: SourceType
    mime_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ────────────────────────────────────────────────────────────
# Step B2a  –  raw_secret_hits.jsonl
# ────────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────────
# Step B2b  –  source_compliance.jsonl
# ────────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────────
# Step C  –  cleaned_documents.jsonl
# ────────────────────────────────────────────────────────────
class CleanedDocument(BaseModel):
    doc_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_id: str
    text: str
    char_count: int = 0
    language: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ────────────────────────────────────────────────────────────
# Step D  –  deduped_documents.jsonl  +  dedup_map.jsonl
# ────────────────────────────────────────────────────────────
class DedupDocument(BaseModel):
    doc_id: str
    source_id: str
    text: str
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    minhash_signature: Optional[list[int]] = None


class DedupMapEntry(BaseModel):
    doc_id: str
    duplicate_of: str
    jaccard_similarity: float = 0.0


# ────────────────────────────────────────────────────────────
# Step E1a  –  keyword_hits.jsonl
# ────────────────────────────────────────────────────────────
class KeywordHit(BaseModel):
    doc_id: str
    keyword: str
    start_pos: int = 0
    end_pos: int = 0
    context: str = ""


# ────────────────────────────────────────────────────────────
# Step E1b  –  regex_hits.jsonl
# ────────────────────────────────────────────────────────────
class RegexHit(BaseModel):
    doc_id: str
    pattern_name: str
    pattern: str = ""
    matched_text: str = ""
    start_pos: int = 0
    end_pos: int = 0
    context: str = ""


# ────────────────────────────────────────────────────────────
# Step F  –  privacy_checked.jsonl
# ────────────────────────────────────────────────────────────
class PIIEntity(BaseModel):
    entity_type: str
    start: int
    end: int
    score: float = 0.0
    original_text: str = ""


class PrivacyResult(BaseModel):
    doc_id: str
    original_text: str = ""
    redacted_text: str = ""
    pii_entities: list[PIIEntity] = Field(default_factory=list)
    pii_count: int = 0


# ────────────────────────────────────────────────────────────
# Step G  –  safety_checked.jsonl
# ────────────────────────────────────────────────────────────
class SafetyResult(BaseModel):
    doc_id: str
    safety_level: SafetyLevel = SafetyLevel.SAFE
    harm_categories: list[str] = Field(default_factory=list)
    raw_output: str = ""
    score: float = 1.0


# ────────────────────────────────────────────────────────────
# Step H  –  evidence_bundle.json
# ────────────────────────────────────────────────────────────
class DocumentEvidence(BaseModel):
    doc_id: str
    source_id: str
    secret_hits: list[SecretHit] = Field(default_factory=list)
    compliance_hits: list[ComplianceHit] = Field(default_factory=list)
    is_duplicate: bool = False
    keyword_hits: list[KeywordHit] = Field(default_factory=list)
    regex_hits: list[RegexHit] = Field(default_factory=list)
    privacy: Optional[PrivacyResult] = None
    safety: Optional[SafetyResult] = None


class EvidenceBundle(BaseModel):
    pipeline_run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    documents: list[DocumentEvidence] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


# ────────────────────────────────────────────────────────────
# Step I  –  decision.json
# ────────────────────────────────────────────────────────────
class DocumentDecision(BaseModel):
    doc_id: str
    decision: Decision = Decision.REVIEW
    reasons: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)


class PolicyDecision(BaseModel):
    pipeline_run_id: str
    overall_decision: Decision = Decision.REVIEW
    document_decisions: list[DocumentDecision] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


# ────────────────────────────────────────────────────────────
# Pipeline task models (for the FastAPI service)
# ────────────────────────────────────────────────────────────
class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CheckRequest(BaseModel):
    """Body for POST /api/v1/check."""
    input_paths: list[str] = Field(
        ..., description="File paths, directory paths, or URLs to check"
    )
    config_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional overrides for pipeline settings",
    )


class CheckTaskInfo(BaseModel):
    """Returned by the service to track a running check."""
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    result: Optional[PolicyDecision] = None
    error: Optional[str] = None
