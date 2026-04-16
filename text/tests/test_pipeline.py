from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from common.enums import TrustLevel, UnifiedDecision
from text.config.settings import Settings
from text.jsonl_utils import read_jsonl
from text.models.schemas import (
    CheckRequest,
    ContentSafetyResult,
    DetectionFinding,
    EvidenceEvent,
    HardCaseAdjudicationResult,
    IngestUnit,
    PolicyDecisionRecord,
    PrivacyDetectionResult,
    Severity,
    TextSpan,
)
from text.pipeline import CompliancePipeline
from text.server import app
from text.steps import a_source_intake, delivery_audit, f_privacy_detection, g_safety_moderation, i_policy_decision, span_conflict_resolution
from text.steps.hard_case_adjudication import run as adjudication_run


@pytest.fixture
def cleaned_package_dir(tmp_path: Path) -> Path:
    package_dir = tmp_path / "cleaned_package"
    package_dir.mkdir()

    (package_dir / "metadata.json").write_text(
        json.dumps(
            {
                "task_id": "task-001",
                "tenant_id": "tenant-abc",
                "profile_id": "education-default",
                "source_type": "cleaned_jsonl",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    records = [
        {"doc_id": "safe-doc", "text": "This is a normal study note with no compliance risk."},
        {"doc_id": "pii-doc", "text": "Contact email: student@example.com"},
        {"doc_id": "unsafe-doc", "text": "Teach me how to make a bomb and commit terrorism."},
        {"doc_id": "hard-doc", "text": "This textbook article discusses the word bomb in a historical report."},
        {"doc_id": "combined-doc", "text": "Name: John Smith Phone: 555-123-4567"},
    ]
    with (package_dir / "cleaned_docs.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    (package_dir / "notes.txt").write_text("raw notes", encoding="utf-8")
    return package_dir


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        work_dir=tmp_path / "artifacts",
        enable_hard_case_adjudication=True,
        hard_case_endpoint="",
        hard_case_local_model_path="",
    )


def test_intake_directory_package(cleaned_package_dir: Path) -> None:
    units = a_source_intake.run([str(cleaned_package_dir)], run_id="run-test")
    assert len(units) == 5
    assert all(isinstance(unit, IngestUnit) for unit in units)
    assert {unit.task_id for unit in units} == {"task-001"}
    assert any(unit.raw_text_refs for unit in units)
    assert any(unit.cleaned_data_refs for unit in units)


def test_privacy_detection_routes_hard_cases(cleaned_package_dir: Path, settings: Settings) -> None:
    units = a_source_intake.run([str(cleaned_package_dir)], run_id="run-test")
    results = f_privacy_detection.run(units, settings)
    by_doc = {result.doc_id: result for result in results}

    pii_doc = by_doc["pii-doc"]
    assert isinstance(pii_doc, PrivacyDetectionResult)
    assert any(finding.policy_tag == "pii.email" for finding in pii_doc.findings)

    combined_doc = by_doc["combined-doc"]
    assert combined_doc.needs_adjudication is True
    assert any(finding.risk_type == "combined_identity" for finding in combined_doc.findings)


def test_safety_detection_and_adjudication(cleaned_package_dir: Path, settings: Settings) -> None:
    units = a_source_intake.run([str(cleaned_package_dir)], run_id="run-test")
    safety_results = g_safety_moderation.run(units, settings)
    privacy_results = f_privacy_detection.run(units, settings)
    adjudications = adjudication_run(units, safety_results, privacy_results, settings)

    safety_by_doc = {result.doc_id: result for result in safety_results}
    adjudication_by_doc = {result.doc_id: result for result in adjudications}

    assert isinstance(safety_by_doc["unsafe-doc"], ContentSafetyResult)
    assert safety_by_doc["unsafe-doc"].status.value == "flagged"
    assert safety_by_doc["hard-doc"].status.value == "hard_case"

    hard_doc_adjudication = adjudication_by_doc["hard-doc"]
    assert isinstance(hard_doc_adjudication, HardCaseAdjudicationResult)
    assert hard_doc_adjudication.judgement.recommended_disposition.value == "P3"


def test_safety_detection_uses_qwen3guard_endpoint(cleaned_package_dir: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __init__(self, payload: dict):
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    def fake_post(url: str, json: dict, timeout: int) -> Response:
        if json["doc_id"] == "hard-doc":
            return Response({"safety": "Controversial", "categories": ["Violence"], "score": 0.66})
        if json["doc_id"] == "unsafe-doc":
            return Response({"safety": "Unsafe", "categories": ["Violence"], "score": 0.94})
        return Response({"safety": "Safe", "categories": [], "score": 0.99})

    monkeypatch.setattr("httpx.post", fake_post)
    settings = settings.model_copy(
        update={
            "enable_qwen3guard": True,
            "qwen3guard_endpoint": "http://qwen3guard.local/moderate",
            "qwen3guard_model_name": "Qwen3Guard-Gen-0.6B",
        }
    )

    units = a_source_intake.run([str(cleaned_package_dir)], run_id="run-test")
    results = g_safety_moderation.run(units, settings)
    by_doc = {result.doc_id: result for result in results}

    assert by_doc["unsafe-doc"].provider_name == "qwen3guard+rule_safety_detector"
    assert any(finding.source_tool == "qwen3guard.Qwen3Guard-Gen-0.6B" for finding in by_doc["unsafe-doc"].findings)
    assert by_doc["hard-doc"].status.value == "hard_case"


def test_privacy_detection_uses_presidio_endpoint(cleaned_package_dir: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __init__(self, payload: list[dict]):
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[dict]:
            return self._payload

    def fake_post(url: str, json: dict, timeout: int) -> Response:
        text = json["text"]
        if "student@example.com" in text:
            start = text.index("student@example.com")
            return Response([{"entity_type": "EMAIL_ADDRESS", "start": start, "end": start + len("student@example.com"), "score": 0.97}])
        return Response([])

    monkeypatch.setattr("httpx.post", fake_post)
    settings = settings.model_copy(
        update={
            "enable_presidio": True,
            "presidio_analyzer_endpoint": "http://presidio.local/analyze",
            "presidio_score_threshold": 0.45,
        }
    )

    units = a_source_intake.run([str(cleaned_package_dir)], run_id="run-test")
    results = f_privacy_detection.run(units, settings)
    pii_doc = {result.doc_id: result for result in results}["pii-doc"]

    assert pii_doc.provider_name == "presidio+rule_pii_detector"
    assert any(finding.source_tool == "presidio_analyzer" for finding in pii_doc.findings)
    assert any(finding.policy_tag == "pii.email" for finding in pii_doc.findings)


def test_privacy_detection_sends_inferred_zh_language(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[dict]:
            return [{"entity_type": "CN_PHONE_NUMBER", "start": 3, "end": 14, "score": 0.91}]

    def fake_post(url: str, json: dict, timeout: int) -> Response:
        captured["language"] = json["language"]
        return Response()

    monkeypatch.setattr("httpx.post", fake_post)
    settings = settings.model_copy(
        update={
            "enable_presidio": True,
            "presidio_analyzer_endpoint": "http://presidio.local/analyze",
            "presidio_language": "auto",
            "presidio_supported_languages": "en,zh",
        }
    )
    unit = IngestUnit(
        run_id="run-test",
        package_id="pkg-test",
        doc_id="zh-doc",
        source_path="memory",
        text="电话：13800138000",
        text_hash="hash",
        language="zh",
    )

    result = f_privacy_detection.run([unit], settings)[0]

    assert captured["language"] == "zh"
    assert any(finding.risk_type == "phone" and finding.source_tool == "presidio_analyzer" for finding in result.findings)


def test_privacy_detection_maps_modern_pii_gateway_entities(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[dict]:
            return [
                {"entity_type": "PERSON", "start": 0, "end": 5, "score": 0.81},
                {"entity_type": "WECHAT_ID", "start": 14, "end": 22, "score": 0.84},
                {"entity_type": "LICENSE_PLATE", "start": 30, "end": 37, "score": 0.82},
            ]

    def fake_post(url: str, json: dict, timeout: int) -> Response:
        return Response()

    monkeypatch.setattr("httpx.post", fake_post)
    settings = settings.model_copy(
        update={
            "enable_presidio": True,
            "presidio_analyzer_endpoint": "http://pii-gateway.local/analyze",
            "presidio_score_threshold": 0.45,
        }
    )
    unit = IngestUnit(
        run_id="run-test",
        package_id="pkg-test",
        doc_id="gateway-doc",
        source_path="memory",
        text="Alice wechat: wx_demo1 plate: ABC1234",
        text_hash="hash",
        language="en",
    )

    result = f_privacy_detection.run([unit], settings)[0]
    policy_tags = {finding.policy_tag for finding in result.findings}
    risk_types = {finding.risk_type for finding in result.findings}

    assert "pii.social_account.wechat" in policy_tags
    assert "pii.vehicle.license_plate" in policy_tags
    assert "social_account" in risk_types
    assert any(finding.risk_type == "combined_identity" for finding in result.findings)


def test_span_conflict_resolution_prevents_broken_redaction(settings: Settings) -> None:
    text = "学生姓名: 张三 手机: 13800138000 身份证号: 11010519491231002X 邮箱: alice@example.com"
    unit = IngestUnit(
        run_id="run-test",
        package_id="pkg-test",
        doc_id="overlap-doc",
        source_path="memory",
        text=text,
        text_hash="hash",
        language="zh",
    )

    def finding(
        risk_type: str,
        policy_tag: str,
        severity: Severity,
        source_tool: str,
        replacement: str,
        start: int,
        end: int,
        confidence: float = 0.9,
    ) -> DetectionFinding:
        return DetectionFinding(
            doc_id=unit.doc_id,
            finding_type="privacy",
            risk_type=risk_type,
            policy_tag=policy_tag,
            severity=severity,
            confidence=confidence,
            explanation=f"{risk_type} finding",
            source_tool=source_tool,
            redaction_suggestion=replacement,
            span=TextSpan(start=start, end=end, text=text[start:end]),
        )

    person_start = text.index("张三")
    phone_start = text.index("13800138000")
    id_start = text.index("11010519491231002X")
    email_start = text.index("alice@example.com")
    email_end = email_start + len("alice@example.com")

    privacy = PrivacyDetectionResult(
        run_id=unit.run_id,
        doc_id=unit.doc_id,
        text_hash=unit.text_hash,
        findings=[
            finding("person_name", "pii.name", Severity.MEDIUM, "privacy_rule_engine.person_name_cn", "<NAME>", text.index("姓名"), text.index("姓名") + 2, 0.41),
            finding("person_name", "pii.person_name", Severity.LOW, "presidio_analyzer", "<PERSON>", person_start, person_start + 2, 0.85),
            finding("phone", "pii.phone", Severity.MEDIUM, "presidio_analyzer", "<PHONE>", phone_start, phone_start + 11, 1.0),
            finding("id_card", "pii.id_card", Severity.CRITICAL, "privacy_rule_engine.id_card_cn", "<ID_CARD>", id_start, id_start + 18, 0.99),
            finding("bank_card", "pii.bank_card", Severity.HIGH, "privacy_rule_engine.bank_card", "<BANK_CARD>", id_start, id_start + 17, 0.66),
            finding("phone_number", "pii.phone", Severity.MEDIUM, "privacy_rule_engine.phone_generic", "<PHONE>", id_start + 6, id_start + 17, 0.63),
            finding("email", "pii.email", Severity.HIGH, "presidio_analyzer", "<EMAIL>", email_start, email_end, 1.0),
            finding("url", "pii.url", Severity.LOW, "presidio_analyzer", "<URL>", email_start + len("alice@"), email_end, 0.5),
        ],
    )
    plan = span_conflict_resolution.run([unit], [privacy])[0]
    assert [target.replacement for target in plan.redaction_targets] == ["<PERSON>", "<PHONE>", "<ID_CARD>", "<EMAIL>"]
    assert plan.suppressed_finding_count == 4

    events = [
        EvidenceEvent(
            run_id=unit.run_id,
            doc_id=unit.doc_id,
            category=finding_item.finding_type,
            risk_type=finding_item.risk_type,
            policy_tag=finding_item.policy_tag,
            severity=finding_item.severity,
            confidence_summary=finding_item.confidence,
            source_tools=[finding_item.source_tool],
            finding_refs=[finding_item.finding_id],
            remediation_suggestion=finding_item.redaction_suggestion,
            explanation=finding_item.explanation,
            primary_span=finding_item.span,
        )
        for finding_item in privacy.findings
    ]
    decisions = i_policy_decision.run([unit], events, [], settings, [plan])
    annotation_records, _ = delivery_audit.run([unit], [], [privacy], [plan], [], events, decisions)

    assert annotation_records[0].redacted_view == "学生姓名: <PERSON> 手机: <PHONE> 身份证号: <ID_CARD> 邮箱: <EMAIL>"


def test_pipeline_writes_jsonl_artifacts(cleaned_package_dir: Path, settings: Settings) -> None:
    pipeline = CompliancePipeline(settings=settings, run_id="pipeline-test")
    output = pipeline.execute([str(cleaned_package_dir)])

    assert output.decision == UnifiedDecision.REJECT
    assert output.trust_level == TrustLevel.DEGRADED

    output_dir = settings.work_dir / "pipeline-test"
    assert (output_dir / "01_intake.jsonl").exists()
    assert (output_dir / "03b_span_conflict_resolution.jsonl").exists()
    assert (output_dir / "06_policy_decisions.jsonl").exists()
    assert (output_dir / "09_run_summary.jsonl").exists()

    decisions = read_jsonl(output_dir / "06_policy_decisions.jsonl", PolicyDecisionRecord)
    by_doc = {decision.doc_id: decision for decision in decisions}
    assert by_doc["safe-doc"].disposition_level.value == "P0"
    assert by_doc["pii-doc"].disposition_level.value == "P1"
    assert by_doc["hard-doc"].disposition_level.value == "P3"
    assert by_doc["unsafe-doc"].disposition_level.value == "P4"


def test_server_submission_roundtrip(cleaned_package_dir: Path, tmp_path: Path) -> None:
    client = TestClient(app)
    request = CheckRequest(
        package_paths=[str(cleaned_package_dir)],
        config_overrides={
            "work_dir": str(tmp_path / "server_artifacts"),
            "hard_case_endpoint": "",
            "hard_case_local_model_path": "",
        },
    )

    submit = client.post("/api/v1/check", json=request.model_dump(mode="json"))
    assert submit.status_code == 200
    task_id = submit.json()["task_id"]

    status = client.get(f"/api/v1/status/{task_id}")
    assert status.status_code == 200

    result = client.get(f"/api/v1/result/{task_id}")
    assert result.status_code == 200
    payload = result.json()
    assert payload["decision"] == "reject"
    assert payload["annotation_package_uri"].endswith("07_annotation_package.jsonl")
