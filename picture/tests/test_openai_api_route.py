"""
Tests for the GPT-5.2 external API picture route.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from picture.application.use_cases import create_orchestrator
from picture.domain.enums import DecisionType, JobStatus
from picture.domain.models import PictureJob, SourceSpec
from picture.infra.config import PictureSettings
from picture.providers.openai_shared import OpenAIPictureAnalysis, OpenAIPictureAnalyzer

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def openai_settings(tmp_path: Path) -> PictureSettings:
    return PictureSettings(
        work_dir=tmp_path / "work",
        storage_base_path=tmp_path / "storage",
        policy_config_dir=Path(__file__).resolve().parent.parent / "configs",
        ocr_provider="openai_gpt52",
        pii_provider="openai_gpt52",
        safety_provider="openai_gpt52",
        vision_provider="openai_gpt52",
        segmentation_provider="mock",
        openai_api_key="test-key",
    )


def _analysis_for(path: str) -> OpenAIPictureAnalysis:
    filename = Path(path).name.lower()
    if "unsafe" in filename or "explicit" in filename:
        return OpenAIPictureAnalysis(
            route_suggestion="natural",
            image_summary="Explicit unsafe image.",
            safety={
                "is_safe": False,
                "categories": [
                    {
                        "name": "explicit",
                        "score": 0.98,
                        "reason": "Explicit nudity is visible.",
                        "reason_code": "SAFETY_EXPLICIT",
                    }
                ],
            },
            text_blocks=[],
            pii_findings=[],
            visual_findings=[
                {
                    "category": "face",
                    "label": "Face",
                    "bbox_norm": {"x1": 220, "y1": 120, "x2": 420, "y2": 430},
                    "score": 0.86,
                    "reason": "Face is visible.",
                    "reason_code": "VISION_FACE",
                }
            ],
            recommended_decision="drop",
            decision_reason_codes=["SAFETY_EXPLICIT"],
            notes=["Unsafe content should be dropped."],
            raw_response={"mock": True},
        )

    return OpenAIPictureAnalysis(
        route_suggestion="document",
        image_summary="Document image with visible phone and QR code.",
        safety={"is_safe": True, "categories": []},
        text_blocks=[
            {
                "id": "tb-1",
                "text": "姓名：张三",
                "language": "zh",
                "bbox_norm": {"x1": 80, "y1": 100, "x2": 360, "y2": 170},
                "confidence": 0.95,
            },
            {
                "id": "tb-2",
                "text": "手机号：13800138000",
                "language": "zh",
                "bbox_norm": {"x1": 80, "y1": 180, "x2": 520, "y2": 260},
                "confidence": 0.96,
            },
        ],
        pii_findings=[
            {
                "category": "person_name",
                "label": "Person name",
                "text_span": "张三",
                "block_ids": ["tb-1"],
                "bbox_norm": {"x1": 180, "y1": 105, "x2": 300, "y2": 170},
                "score": 0.89,
                "reason": "Visible personal name on the document.",
                "reason_code": "PII_NAME",
            },
            {
                "category": "phone_number",
                "label": "Phone number",
                "text_span": "13800138000",
                "block_ids": ["tb-2"],
                "bbox_norm": {"x1": 210, "y1": 185, "x2": 500, "y2": 255},
                "score": 0.97,
                "reason": "Visible phone number on the document.",
                "reason_code": "PII_PHONE",
            },
        ],
        visual_findings=[
            {
                "category": "qr_code",
                "label": "QR code",
                "bbox_norm": {"x1": 650, "y1": 540, "x2": 860, "y2": 780},
                "score": 0.84,
                "reason": "QR code visible in the lower-right region.",
                "reason_code": "VISION_QR_CODE",
            }
        ],
        recommended_decision="pass_redacted",
        decision_reason_codes=["PII_PHONE", "VISION_QR_CODE"],
        notes=["Local policy should still decide the final output."],
        raw_response={"mock": True},
    )


def test_openai_document_route_passes_redacted(
    monkeypatch: pytest.MonkeyPatch,
    openai_settings: PictureSettings,
):
    monkeypatch.setattr(
        OpenAIPictureAnalyzer,
        "_request_and_parse",
        lambda self, image_path: _analysis_for(image_path),
    )
    orchestrator = create_orchestrator(openai_settings)
    job = PictureJob(
        tenant_id="test-tenant",
        source=SourceSpec(
            uri=str(FIXTURES_DIR / "sample_document.png"),
            mime_type="image/png",
        ),
        profile="default_cn_enterprise",
        options={"route_hint": "document"},
    )

    result = orchestrator.execute(job)

    assert result.status == JobStatus.DONE
    assert result.policy_result is not None
    assert result.policy_result.decision == DecisionType.PASS_REDACTED
    assert any(f.category == "phone_number" for f in result.findings)
    assert any(f.category == "qr_code" for f in result.findings)
    assert result.compliant_image_uri is not None
    assert result.report_uri is not None


def test_openai_natural_route_can_drop_image(
    monkeypatch: pytest.MonkeyPatch,
    openai_settings: PictureSettings,
):
    monkeypatch.setattr(
        OpenAIPictureAnalyzer,
        "_request_and_parse",
        lambda self, image_path: _analysis_for(image_path),
    )
    orchestrator = create_orchestrator(openai_settings)
    job = PictureJob(
        tenant_id="test-tenant",
        source=SourceSpec(
            uri=str(FIXTURES_DIR / "sample_unsafe_explicit.png"),
            mime_type="image/png",
        ),
        profile="default_cn_enterprise",
        options={"route_hint": "natural"},
    )

    result = orchestrator.execute(job)

    assert result.status == JobStatus.DROPPED
    assert result.policy_result is not None
    assert result.policy_result.decision == DecisionType.DROP
    assert result.moderation_result is not None
    assert "SAFETY_EXPLICIT" in result.moderation_result.reason_codes
