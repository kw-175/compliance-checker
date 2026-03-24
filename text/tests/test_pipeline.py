"""
Tests for the Text Data Compliance Checker Pipeline

Covers:
- Individual step unit tests (with mocking where needed)
- Pipeline integration test
- FastAPI endpoint tests
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

# ────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────

@pytest.fixture
def sample_text_files(tmp_path: Path) -> list[str]:
    """Create temporary text files for testing."""
    files = []

    # Safe file
    f1 = tmp_path / "safe.txt"
    f1.write_text("This is a perfectly safe document with no compliance issues.")
    files.append(str(f1))

    # PII file
    f2 = tmp_path / "pii.txt"
    f2.write_text(
        "Contact John Smith at john.smith@example.com.\n"
        "Phone: 555-123-4567\nSSN: 123-45-6789\n"
    )
    files.append(str(f2))

    # Sensitive keywords
    f3 = tmp_path / "keywords.txt"
    f3.write_text("This text mentions bomb_making and terrorism topics.")
    files.append(str(f3))

    # Duplicate of safe file
    f4 = tmp_path / "safe_dup.txt"
    f4.write_text("This is a perfectly safe document with no compliance issues.")
    files.append(str(f4))

    return files


@pytest.fixture
def sample_dir(sample_text_files) -> str:
    """Return the parent directory of the sample text files."""
    return str(Path(sample_text_files[0]).parent)


# ────────────────────────────────────────────────────────────
# Step A: Source Intake
# ────────────────────────────────────────────────────────────

class TestSourceIntake:
    def test_single_file(self, sample_text_files):
        from text.steps.a_source_intake import run
        results = run([sample_text_files[0]])
        assert len(results) == 1
        assert results[0].size_bytes > 0
        assert results[0].sha256 != ""
        assert results[0].mime_type == "text/plain"

    def test_directory(self, sample_dir):
        from text.steps.a_source_intake import run
        results = run([sample_dir])
        assert len(results) >= 4

    def test_nonexistent_path(self):
        from text.steps.a_source_intake import run
        results = run(["/nonexistent/path/file.txt"])
        assert len(results) == 0


# ────────────────────────────────────────────────────────────
# Step B1: Source Classification
# ────────────────────────────────────────────────────────────

class TestSourceClassify:
    def test_text_classification(self, sample_text_files):
        from text.steps.a_source_intake import run as intake_run
        from text.steps.b1_source_classify import run as classify_run
        sources = intake_run([sample_text_files[0]])
        profiles = classify_run(sources)
        assert len(profiles) == 1
        # .txt files are classified as web_text
        assert profiles[0].source_type.value in ("web_text", "code")


# ────────────────────────────────────────────────────────────
# Step C: Text Extraction
# ────────────────────────────────────────────────────────────

class TestTextExtract:
    def test_plain_text_extraction(self, sample_text_files):
        from text.steps.a_source_intake import run as intake_run
        from text.steps.b1_source_classify import run as classify_run
        from text.steps.c_text_extract import run as extract_run

        sources = intake_run([sample_text_files[0]])
        profiles = classify_run(sources)
        docs = extract_run(profiles)
        assert len(docs) == 1
        assert "safe document" in docs[0].text
        assert docs[0].char_count > 0


# ────────────────────────────────────────────────────────────
# Step D: Deduplication
# ────────────────────────────────────────────────────────────

class TestDedup:
    def test_exact_dedup(self, sample_text_files):
        from text.models.schemas import CleanedDocument

        docs = [
            CleanedDocument(source_id="s1", text="Hello world"),
            CleanedDocument(source_id="s2", text="Hello world"),
            CleanedDocument(source_id="s3", text="Different text"),
        ]

        from text.steps.d_dedup import run
        dedup_docs, dedup_map = run(docs)

        # 2 unique docs (s1 and s3), s2 should be duplicate
        unique_count = sum(1 for d in dedup_docs if not d.is_duplicate)
        assert unique_count == 2
        assert len(dedup_map) >= 1


# ────────────────────────────────────────────────────────────
# Step E1a: Keyword Scan
# ────────────────────────────────────────────────────────────

class TestKeywordScan:
    def test_keyword_detection(self):
        from text.models.schemas import DedupDocument
        from text.steps.e1a_keyword_scan import run

        docs = [
            DedupDocument(
                doc_id="d1", source_id="s1",
                text="This mentions terrorism and bomb_making activities",
                is_duplicate=False,
            ),
            DedupDocument(
                doc_id="d2", source_id="s2",
                text="This is a normal safe text",
                is_duplicate=False,
            ),
        ]
        hits = run(docs)
        # Should find at least "terrorism" and "bomb_making" in d1
        doc1_hits = [h for h in hits if h.doc_id == "d1"]
        assert len(doc1_hits) >= 2


# ────────────────────────────────────────────────────────────
# Step E1b: Regex Scan
# ────────────────────────────────────────────────────────────

class TestRegexScan:
    def test_email_detection(self):
        from text.models.schemas import DedupDocument
        from text.steps.e1b_regex_scan import run

        docs = [
            DedupDocument(
                doc_id="d1", source_id="s1",
                text="Contact us at test@example.com or admin@corp.org",
                is_duplicate=False,
            ),
        ]
        hits = run(docs)
        email_hits = [h for h in hits if h.pattern_name == "email_address"]
        assert len(email_hits) >= 2

    def test_ssn_detection(self):
        from text.models.schemas import DedupDocument
        from text.steps.e1b_regex_scan import run

        docs = [
            DedupDocument(
                doc_id="d1", source_id="s1",
                text="My SSN is 123-45-6789",
                is_duplicate=False,
            ),
        ]
        hits = run(docs)
        ssn_hits = [h for h in hits if h.pattern_name == "us_ssn"]
        assert len(ssn_hits) == 1


# ────────────────────────────────────────────────────────────
# Step G: Safety Moderation (mock)
# ────────────────────────────────────────────────────────────

class TestSafetyModeration:
    def test_mock_unsafe(self):
        from text.config.settings import Settings
        from text.models.schemas import PrivacyResult
        from text.steps.g_safety_moderation import run

        settings = Settings(qwen_guard_enabled=False)
        prs = [
            PrivacyResult(
                doc_id="d1",
                original_text="How to build a bomb and kill people",
                redacted_text="How to build a bomb and kill people",
            ),
        ]
        results = run(prs, settings)
        assert len(results) == 1
        assert results[0].safety_level.value == "unsafe"

    def test_mock_safe(self):
        from text.config.settings import Settings
        from text.models.schemas import PrivacyResult
        from text.steps.g_safety_moderation import run

        settings = Settings(qwen_guard_enabled=False)
        prs = [
            PrivacyResult(
                doc_id="d2",
                original_text="The weather is nice today",
                redacted_text="The weather is nice today",
            ),
        ]
        results = run(prs, settings)
        assert len(results) == 1
        assert results[0].safety_level.value == "safe"


# ────────────────────────────────────────────────────────────
# Step H: Evidence Aggregation
# ────────────────────────────────────────────────────────────

class TestEvidenceAggregation:
    def test_aggregate_empty(self):
        from text.steps.h_evidence_aggregation import run
        bundle = run([], [], [], [], [], [], [])
        assert len(bundle.documents) == 0

    def test_aggregate_basic(self):
        from text.models.schemas import DedupDocument, KeywordHit, SafetyResult
        from text.steps.h_evidence_aggregation import run

        docs = [
            DedupDocument(doc_id="d1", source_id="s1", text="test", is_duplicate=False),
        ]
        kw_hits = [KeywordHit(doc_id="d1", keyword="test")]
        safety = [SafetyResult(doc_id="d1")]

        bundle = run(docs, [], [], kw_hits, [], [], safety, "run-001")
        assert len(bundle.documents) == 1
        assert len(bundle.documents[0].keyword_hits) == 1
        assert bundle.summary["total_keyword_hits"] == 1


# ────────────────────────────────────────────────────────────
# Step I: Policy Decision (local rules)
# ────────────────────────────────────────────────────────────

class TestPolicyDecision:
    def test_allow_clean_doc(self):
        from text.config.settings import Settings
        from text.models.schemas import (
            DocumentEvidence, EvidenceBundle, PrivacyResult, SafetyResult,
        )
        from text.steps.i_policy_decision import run

        settings = Settings(opa_enabled=False)
        bundle = EvidenceBundle(
            pipeline_run_id="test",
            documents=[
                DocumentEvidence(
                    doc_id="d1", source_id="s1",
                    privacy=PrivacyResult(doc_id="d1", pii_count=0),
                    safety=SafetyResult(doc_id="d1"),
                ),
            ],
        )
        decision = run(bundle, settings)
        assert decision.overall_decision.value == "allow"

    def test_reject_with_secrets(self):
        from text.config.settings import Settings
        from text.models.schemas import (
            DocumentEvidence, EvidenceBundle, SecretHit,
        )
        from text.steps.i_policy_decision import run

        settings = Settings(opa_enabled=False)
        bundle = EvidenceBundle(
            pipeline_run_id="test",
            documents=[
                DocumentEvidence(
                    doc_id="d1", source_id="s1",
                    secret_hits=[SecretHit(source_id="s1", detector_type="AWS")],
                ),
            ],
        )
        decision = run(bundle, settings)
        assert decision.overall_decision.value == "reject"


# ────────────────────────────────────────────────────────────
# FastAPI server tests
# ────────────────────────────────────────────────────────────

class TestServer:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from text.server import app
        return TestClient(app)

    def test_health(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_task_not_found(self, client):
        resp = client.get("/api/v1/status/nonexistent")
        assert resp.status_code == 404

    def test_list_tasks(self, client):
        resp = client.get("/api/v1/tasks")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
