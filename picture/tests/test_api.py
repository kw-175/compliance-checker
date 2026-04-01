"""
Tests for the FastAPI endpoints.

Covers:
- Health check
- Job creation
- Status query
- Result retrieval
- Findings retrieval
- Error cases (not found, unsupported type)
"""
# 中文说明：这组测试验证 picture API 对外行为是否稳定，
# 包括路由路径、返回码、基本响应结构，以及仓储中已有任务时的读取行为。
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from picture.api.app import app
from picture.application.use_cases import _get_default_repo
from picture.domain.enums import DecisionType, FindingType, JobStatus
from picture.domain.models import (
    PictureFinding,
    PictureJob,
    PicturePolicyResult,
    SourceSpec,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_returns_ok(self, client: TestClient):
        """Health endpoint should return 200 with status healthy."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "picture-compliance-checker"


class TestJobCreation:
    """Test job creation endpoint."""

    def test_create_job(self, client: TestClient):
        """Should create a job and return job_id."""
        image_path = str(FIXTURES_DIR / "sample_document.png")
        response = client.post(
            "/v1/picture/jobs",
            json={
                "tenant_id": "test-tenant",
                "source": {
                    "type": "file",
                    "uri": image_path,
                    "mime_type": "image/png",
                },
                "profile": "default_cn_enterprise",
                "options": {
                    "route_hint": "document",
                },
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "CREATED"

    def test_create_job_minimal(self, client: TestClient):
        """Should create a job with minimal request."""
        # 中文说明：这里验证最小请求体是否能被 schema 与默认值正确补齐。
        response = client.post(
            "/v1/picture/jobs",
            json={
                "source": {
                    "uri": str(FIXTURES_DIR / "sample_document.png"),
                },
            },
        )
        assert response.status_code == 201


class TestJobStatus:
    """Test job status endpoint."""

    def test_get_job_status(self, client: TestClient):
        """Should return job status."""
        image_path = str(FIXTURES_DIR / "sample_document.png")
        create_resp = client.post(
            "/v1/picture/jobs",
            json={
                "source": {"uri": image_path, "mime_type": "image/png"},
                "options": {"route_hint": "document"},
            },
        )
        job_id = create_resp.json()["job_id"]

        response = client.get(f"/v1/picture/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id

    def test_get_nonexistent_job(self, client: TestClient):
        """Should return 404 for nonexistent job."""
        response = client.get("/v1/picture/jobs/nonexistent_id")
        assert response.status_code == 404


class TestJobResult:
    """Test job result endpoint."""

    def test_get_result_of_completed_job(self, client: TestClient):
        """Should return results for a completed job."""
        repo = _get_default_repo()

        # 中文说明：这里直接往仓储里塞一个已完成任务，
        # 只测试结果接口本身，不依赖真实处理流程。
        job = PictureJob(
            tenant_id="test",
            source=SourceSpec(uri="test.png", mime_type="image/png"),
            status=JobStatus.DONE,
            policy_result=PicturePolicyResult(
                decision=DecisionType.PASS_REDACTED,
                reason_codes=["PII_PHONE"],
            ),
            findings=[
                PictureFinding(
                    finding_type=FindingType.TEXT_PII,
                    category="phone_number",
                    label="Phone",
                    score=0.95,
                    reason_code="PII_PHONE",
                    provider="MockPII",
                ),
            ],
        )
        repo.save_job(job)

        response = client.get(f"/v1/picture/jobs/{job.job_id}/result")
        assert response.status_code == 200
        data = response.json()
        assert data["decision"] == "pass_redacted"
        assert "PII_PHONE" in data["reason_codes"]

    def test_get_result_not_found(self, client: TestClient):
        """Should return 404 for nonexistent job."""
        response = client.get("/v1/picture/jobs/nonexistent/result")
        assert response.status_code == 404


class TestJobFindings:
    """Test findings endpoint."""

    def test_get_findings(self, client: TestClient):
        """Should return findings for a job."""
        repo = _get_default_repo()

        job = PictureJob(
            tenant_id="test",
            source=SourceSpec(uri="test.png", mime_type="image/png"),
            status=JobStatus.DONE,
            findings=[
                PictureFinding(
                    finding_type=FindingType.TEXT_PII,
                    category="email",
                    label="Email",
                    score=0.92,
                    text_span="test@example.com",
                    reason_code="PII_EMAIL",
                    provider="MockPII",
                ),
                PictureFinding(
                    finding_type=FindingType.VISION_OBJECT,
                    category="face",
                    label="Face",
                    score=0.88,
                    reason_code="VISION_FACE",
                    provider="MockVision",
                ),
            ],
        )
        repo.save_job(job)

        response = client.get(f"/v1/picture/jobs/{job.job_id}/findings")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["findings"]) == 2

    def test_get_findings_not_found(self, client: TestClient):
        """Should return 404 for nonexistent job."""
        response = client.get("/v1/picture/jobs/nonexistent/findings")
        assert response.status_code == 404


class TestJobRerun:
    """Test rerun endpoint."""

    def test_rerun_creates_new_job(self, client: TestClient):
        """Rerun should create a new job with same parameters."""
        repo = _get_default_repo()

        job = PictureJob(
            tenant_id="test",
            source=SourceSpec(
                uri=str(FIXTURES_DIR / "sample_document.png"),
                mime_type="image/png",
            ),
            status=JobStatus.DONE,
        )
        repo.save_job(job)

        response = client.post(f"/v1/picture/jobs/{job.job_id}/rerun")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] != job.job_id
        assert data["status"] == "CREATED"
