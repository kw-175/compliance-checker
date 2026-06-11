"""
Tests for the picture compliance orchestrator.

Covers all three processing chains (document, natural, mixed)
using mock providers to ensure CI compatibility.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from picture.application.orchestrator import PictureComplianceOrchestrator
from picture.application.use_cases import create_orchestrator
from picture.domain.enums import DecisionType, JobStatus, RouteType
from picture.domain.models import PictureJob, SourceSpec
from picture.domain.policy import ConfigurablePolicyEngine
from picture.infra.config import PictureSettings
from picture.infra.repository import InMemoryJobRepository
from picture.infra.storage import LocalFileStorageBackend
from picture.providers.ocr.mock import MockOCRLayoutProvider
from picture.providers.pii.mock import MockPIIDetector
from picture.providers.preprocess import DefaultPreprocessor
from picture.providers.redaction.opencv_redactor import OpenCVRedactor
from picture.providers.router import HeuristicRouter
from picture.providers.safety.mock import MockSafetyModerator
from picture.providers.segmentation.mock import MockSegmentationProvider
from picture.providers.vision.mock import MockVisionDetector

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def settings(tmp_path: Path) -> PictureSettings:
    """Create test settings with temp directory."""
    return PictureSettings(
        work_dir=tmp_path / "work",
        storage_base_path=tmp_path / "storage",
        policy_config_dir=Path(__file__).resolve().parent.parent / "configs",
    )


@pytest.fixture
def repo() -> InMemoryJobRepository:
    return InMemoryJobRepository()


@pytest.fixture
def orchestrator(settings: PictureSettings, repo: InMemoryJobRepository) -> PictureComplianceOrchestrator:
    """Create orchestrator with all mock providers."""
    return PictureComplianceOrchestrator(
        router=HeuristicRouter(),
        preprocessor=DefaultPreprocessor(),
        ocr_provider=MockOCRLayoutProvider(return_pii=True),
        pii_detector=MockPIIDetector(),
        safety_moderator=MockSafetyModerator(),
        vision_detector=MockVisionDetector(),
        segmentation_provider=MockSegmentationProvider(),
        redactor=OpenCVRedactor(),
        policy_engine=ConfigurablePolicyEngine(settings.policy_config_dir),
        storage=LocalFileStorageBackend(settings.storage_base_path),
        repository=repo,
        settings=settings,
    )


def _make_job(image_path: str, route_hint: str = "auto") -> PictureJob:
    """Helper to create a test job."""
    return PictureJob(
        tenant_id="test-tenant",
        source=SourceSpec(uri=image_path, mime_type="image/png"),
        profile="default_cn_enterprise",
        options={"route_hint": route_hint},
    )


class TestDocumentChain:
    """Tests for the document image processing chain."""

    def test_document_chain_produces_redacted(self, orchestrator: PictureComplianceOrchestrator):
        """Document chain with PII should produce pass_redacted."""
        image_path = str(FIXTURES_DIR / "sample_document.png")
        job = _make_job(image_path, route_hint="document")
        result = orchestrator.execute(job)

        assert result.status in (JobStatus.DONE, JobStatus.DROPPED)
        assert result.policy_result is not None
        assert result.policy_result.decision == DecisionType.PASS_REDACTED
        assert len(result.findings) > 0
        assert result.report_uri is not None

    def test_document_chain_records_latencies(self, orchestrator: PictureComplianceOrchestrator):
        """Document chain should record step latencies."""
        image_path = str(FIXTURES_DIR / "sample_document.png")
        job = _make_job(image_path, route_hint="document")
        result = orchestrator.execute(job)

        assert "total" in result.step_latencies
        assert "preprocess" in result.step_latencies
        assert "ocr_layout" in result.step_latencies


class TestNaturalChain:
    """Tests for the natural image processing chain."""

    def test_natural_safe_image_redacted(self, orchestrator: PictureComplianceOrchestrator):
        """Natural safe image with vision findings → pass_redacted."""
        image_path = str(FIXTURES_DIR / "sample_natural.png")
        job = _make_job(image_path, route_hint="natural")
        result = orchestrator.execute(job)

        assert result.status == JobStatus.DONE
        assert result.policy_result is not None
        # MockVisionDetector returns face/qr/signature → should be redacted
        assert result.policy_result.decision == DecisionType.PASS_REDACTED

    def test_natural_explicit_image_dropped(self, orchestrator: PictureComplianceOrchestrator):
        """Natural explicit image should be dropped."""
        image_path = str(FIXTURES_DIR / "sample_unsafe_explicit.png")
        job = _make_job(image_path, route_hint="natural")
        result = orchestrator.execute(job)

        assert result.status == JobStatus.DROPPED
        assert result.policy_result is not None
        assert result.policy_result.decision == DecisionType.DROP


class TestMixedChain:
    """Tests for the mixed screenshot processing chain."""

    def test_mixed_chain_runs_parallel(self, orchestrator: PictureComplianceOrchestrator):
        """Mixed chain should run OCR and safety in parallel."""
        image_path = str(FIXTURES_DIR / "sample_mixed.png")
        job = _make_job(image_path, route_hint="mixed")
        result = orchestrator.execute(job)

        assert result.status in (JobStatus.DONE, JobStatus.DROPPED)
        assert result.route == RouteType.MIXED
        assert result.ocr_result is not None
        assert result.moderation_result is not None
        assert "phase1_parallel" in result.step_latencies

    def test_mixed_chain_merges_findings(self, orchestrator: PictureComplianceOrchestrator):
        """Mixed chain should merge PII and vision findings."""
        image_path = str(FIXTURES_DIR / "sample_mixed.png")
        job = _make_job(image_path, route_hint="mixed")
        result = orchestrator.execute(job)

        # Should have both PII and vision findings
        pii_findings = [f for f in result.findings if f.finding_type.value == "text_pii"]
        vision_findings = [f for f in result.findings if f.finding_type.value == "vision_object"]
        assert len(pii_findings) > 0
        assert len(vision_findings) > 0


class TestAutoRouting:
    """Tests for automatic routing."""

    def test_auto_route_document(self, orchestrator: PictureComplianceOrchestrator):
        """Document-like image should be auto-routed to document chain."""
        image_path = str(FIXTURES_DIR / "sample_document.png")
        job = _make_job(image_path, route_hint="auto")
        result = orchestrator.execute(job)

        assert result.route is not None
        # The exact route depends on heuristics, but it should complete
        assert result.status in (JobStatus.DONE, JobStatus.DROPPED)

    def test_auto_route_natural(self, orchestrator: PictureComplianceOrchestrator):
        """Natural image should be auto-routed."""
        image_path = str(FIXTURES_DIR / "sample_natural.png")
        job = _make_job(image_path, route_hint="auto")
        result = orchestrator.execute(job)

        assert result.route is not None
        assert result.status in (JobStatus.DONE, JobStatus.DROPPED)


class TestOrchestrationWithFactory:
    """Tests using the factory function."""

    def test_create_orchestrator_and_run(self, settings: PictureSettings):
        """Test creating orchestrator via factory and running a job."""
        orchestrator = create_orchestrator(settings)
        image_path = str(FIXTURES_DIR / "sample_document.png")
        job = _make_job(image_path, route_hint="document")
        result = orchestrator.execute(job)

        assert result.status in (JobStatus.DONE, JobStatus.DROPPED)
        assert result.policy_result is not None


class TestErrorHandling:
    """Tests for error handling in the orchestrator."""

    def test_nonexistent_file(self, orchestrator: PictureComplianceOrchestrator):
        """Job with nonexistent file should fail gracefully."""
        job = _make_job("/nonexistent/image.png", route_hint="document")
        result = orchestrator.execute(job)

        assert result.status == JobStatus.FAILED
        assert result.error is not None

    def test_unsupported_mime_type(self, orchestrator: PictureComplianceOrchestrator):
        """Job with unsupported MIME type should fail."""
        job = PictureJob(
            tenant_id="test",
            source=SourceSpec(uri=str(FIXTURES_DIR / "sample_document.png"), mime_type="video/mp4"),
            profile="default_cn_enterprise",
        )
        result = orchestrator.execute(job)

        assert result.status == JobStatus.FAILED
        assert "unsupported" in (result.error or "").lower()

    def test_nonexistent_profile(self, settings: PictureSettings, repo: InMemoryJobRepository):
        """Job with nonexistent profile should fail."""
        orch = PictureComplianceOrchestrator(
            router=HeuristicRouter(),
            preprocessor=DefaultPreprocessor(),
            ocr_provider=MockOCRLayoutProvider(return_pii=True),
            pii_detector=MockPIIDetector(),
            safety_moderator=MockSafetyModerator(),
            vision_detector=MockVisionDetector(),
            segmentation_provider=MockSegmentationProvider(),
            redactor=OpenCVRedactor(),
            policy_engine=ConfigurablePolicyEngine(settings.policy_config_dir),
            storage=LocalFileStorageBackend(settings.storage_base_path),
            repository=repo,
            settings=settings,
        )

        image_path = str(FIXTURES_DIR / "sample_document.png")
        job = PictureJob(
            tenant_id="test",
            source=SourceSpec(uri=image_path, mime_type="image/png"),
            profile="nonexistent_profile",
            options={"route_hint": "document"},
        )
        result = orch.execute(job)

        assert result.status == JobStatus.FAILED
        assert "profile" in (result.error or "").lower()
