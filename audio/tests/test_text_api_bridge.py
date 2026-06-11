from __future__ import annotations

import json
from pathlib import Path

from audio.config.settings import Settings
from audio.models.schemas import ASRSegment, TranscriptUnit
from audio.text_api_bridge import AudioTextApiBridgeExecutor


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_text_api_bridge_maps_findings_back_to_audio_timeline(tmp_path: Path):
    settings = Settings(work_dir=tmp_path)
    output_dir = tmp_path / "run-001"
    executor = AudioTextApiBridgeExecutor(settings=settings, run_id="run-001", output_dir=output_dir)

    transcript_units = [
        TranscriptUnit(
            unit_id="unit-1",
            source_id="source-1",
            start_time=0.0,
            end_time=2.0,
            speaker_id="speaker_a",
            text="张三的手机号是13800138000",
            confidence=0.95,
            engine_name="faster-whisper",
            language="zh",
        ),
        TranscriptUnit(
            unit_id="unit-2",
            source_id="source-1",
            start_time=2.0,
            end_time=4.0,
            speaker_id="speaker_a",
            text="请不要传播未核实的校园谣言",
            confidence=0.94,
            engine_name="faster-whisper",
            language="zh",
        ),
    ]
    source_docs = executor._build_source_documents(
        transcript_units,
        [type("Source", (), {"source_id": "source-1", "path": str(tmp_path / "sample.wav")})()],
    )

    privacy_path = tmp_path / "03_privacy_detection.jsonl"
    redaction_path = tmp_path / "03b_span_conflict_resolution.jsonl"
    safety_path = tmp_path / "02_content_safety.jsonl"

    _write_jsonl(
        privacy_path,
        [
            {
                "doc_id": "source-1",
                "findings": [
                    {
                        "finding_id": "f-1",
                        "risk_type": "phone_number",
                        "policy_tag": "pii.phone_number",
                        "severity": "high",
                        "confidence": 0.98,
                        "explanation": "phone detected",
                        "remediation_suggestion": "redact",
                        "span": {"start": 7, "end": 18, "text": "13800138000"},
                    }
                ],
            }
        ],
    )
    _write_jsonl(
        redaction_path,
        [
            {
                "doc_id": "source-1",
                "redaction_targets": [
                    {
                        "finding_id": "f-1",
                        "start": 7,
                        "end": 18,
                        "original_text": "13800138000",
                        "replacement": "<PHONE>",
                    }
                ],
                "conflicts": [],
            }
        ],
    )
    _write_jsonl(
        safety_path,
        [
            {
                "doc_id": "source-1",
                "status": "hard_case",
                "findings": [
                    {
                        "finding_id": "s-1",
                        "risk_type": "general_content_safety",
                        "policy_tag": "content.misinformation",
                        "severity": "medium",
                        "confidence": 0.7,
                        "explanation": "rumor detected",
                        "remediation_suggestion": "manual_review",
                        "span": {"start": 26, "end": 35, "text": "未核实的校园谣言"},
                    }
                ],
            }
        ],
    )

    text_api_result = {
        "task_id": "text-task-1",
        "metadata": {
            "artifact_paths": {
                "privacy": str(privacy_path),
                "redaction_plan": str(redaction_path),
                "content_safety": str(safety_path),
            }
        },
        "review_suggestions": ["source-1: review"],
    }

    report = executor._build_audio_report(
        operator_id="CMP_008",
        dataset_name="audio-demo",
        source_documents=source_docs,
        text_api_result=text_api_result,
        local_artifacts=executor._local_artifact_paths(),
    )

    assert report["modality"] == "audio"
    assert report["operator_id"] == "CMP_008"
    assert report["total_documents"] == 1
    assert report["total_findings"] == 2
    assert report["transcript_views"][0]["original_text"] == source_docs[0]["text"]
    assert report["transcript_views"][0]["segments"][0]["start_time"] == 0.0
    assert report["transcript_views"][0]["highlights"][0]["time_label"] != "-"
    assert report["redaction_views"][0]["redacted_text"].startswith("张三的手机号是<PHONE>")
    assert report["findings"][0]["time_label"] != "-"
    assert report["findings"][0]["source_id"] == "source-1"

    redaction_records = [
        {
            "doc_id": "source-1",
            "redaction_targets": [
                {
                    "finding_id": "f-1",
                    "start": 7,
                    "end": 18,
                    "original_text": "13800138000",
                    "replacement": "<PHONE>",
                    "pii_type": "phone_number",
                }
            ],
        }
    ]
    spans = executor._build_audio_redaction_spans(source_docs, redaction_records)
    assert len(spans) == 1
    assert spans[0].source_id == "source-1"
    assert spans[0].entity_type == "phone_number"
    assert spans[0].start_time <= 2.0
    assert spans[0].end_time >= spans[0].start_time

    governance = {
        "privacy": [
            {
                "doc_id": "source-1",
                "findings": [
                    {
                        "finding_id": "f-1",
                        "risk_type": "phone_number",
                        "policy_tag": "pii.phone_number",
                        "severity": "high",
                        "confidence": 0.98,
                        "explanation": "phone detected",
                        "span": {"start": 7, "end": 18, "text": "13800138000"},
                    }
                ],
            }
        ],
        "content_safety": [],
        "redaction_plan": redaction_records,
        "privacy_fragment_adjudications": [
            {
                "doc_id": "source-1",
                "finding_id": "f-1",
                "governance_action": "redact",
                "training_impact": "mask before training",
                "annotation_impact": "redacted annotation view",
                "explanation": "该片段包含电话号码，应脱敏后使用。",
            }
        ],
        "content_fragment_adjudications": [],
        "privacy_document_assessments": [
            {
                "doc_id": "source-1",
                "recommended_action": "redact",
                "training_suitability": "restricted",
                "annotation_suitability": "restricted",
                "explanation": "文档包含局部隐私信息。",
            }
        ],
        "content_document_assessments": [],
        "policy": [
            {
                "doc_id": "source-1",
                "disposition_level": "P1",
                "unified_decision": "allow",
                "required_actions": ["apply_redaction", "release"],
                "redaction_targets": redaction_records[0]["redaction_targets"],
                "explanation": "Apply redaction before release.",
                "trust_level": "full",
            }
        ],
        "evidence": [
            {
                "event_id": "ev-1",
                "doc_id": "source-1",
                "category": "privacy",
                "risk_type": "phone_number",
                "policy_tag": "pii.phone_number",
                "severity": "high",
                "confidence_summary": 0.98,
                "finding_refs": ["f-1"],
                "explanation": "phone detected",
                "primary_span": {"start": 7, "end": 18, "text": "13800138000"},
            }
        ],
        "summary": [],
        "annotation": [],
        "audit": [],
    }
    risk_records = executor._build_audio_text_risk_records(source_docs, governance)
    assert len(risk_records) == 1
    assert risk_records[0]["risk_source"] == "transcript_text"
    assert risk_records[0]["recommended_content_action"] == "transcript_redaction"
    assert risk_records[0]["audio_span"]["mapping_status"] == "mapped"
    assert risk_records[0]["fragment_adjudication"]["governance_action"] == "redact"

    audio_policy = executor._build_audio_policy_decisions(source_docs, governance, risk_records)
    assert audio_policy[0]["workflow_action"] == "normal_flow"
    assert audio_policy[0]["training_eligibility"] == "allow"


def test_text_api_bridge_marks_long_single_segment_as_degraded(tmp_path: Path):
    settings = Settings(work_dir=tmp_path)
    executor = AudioTextApiBridgeExecutor(settings=settings, run_id="run-quality", output_dir=tmp_path / "run-quality")
    transcript_units = [
        TranscriptUnit(
            unit_id="unit-1",
            source_id="source-1",
            start_time=0.0,
            end_time=90.0,
            text="这是一段很长的单片段转写。",
            confidence=0.58,
            engine_name="qwen3-asr-vllm",
            language="zh",
        )
    ]

    source_docs = executor._build_source_documents(
        transcript_units,
        [type("Source", (), {"source_id": "source-1", "path": str(tmp_path / "sample.wav")})()],
    )
    quality = source_docs[0]["asr_quality"]

    assert quality["is_degraded"] is True
    assert quality["requires_review"] is True
    assert {item["code"] for item in quality["warnings"]} >= {"asr_single_long_segment", "asr_low_confidence"}


def test_text_api_bridge_uses_qwen_asr_settings(monkeypatch, tmp_path: Path):
    from audio.steps import c1_asr_transcribe

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF")
    settings = Settings(
        work_dir=tmp_path,
        qwen_asr_enabled=True,
        faster_whisper_enabled=False,
        pyannote_enabled=False,
        text_api_base_url="http://text.local",
    )
    executor = AudioTextApiBridgeExecutor(settings=settings, run_id="run-qwen", output_dir=tmp_path / "run-qwen")
    captured = {}

    def fake_asr(records, received_settings):
        captured["qwen_asr_enabled"] = received_settings.qwen_asr_enabled
        captured["faster_whisper_enabled"] = received_settings.faster_whisper_enabled
        return [
            ASRSegment(
                source_id=records[0].source_id,
                start_time=0.0,
                end_time=1.0,
                text="张三的手机号是13800138000",
                confidence=0.9,
                engine_name="qwen3-asr",
                language="zh",
            )
        ]

    monkeypatch.setattr(c1_asr_transcribe, "run", fake_asr)
    monkeypatch.setattr(
        executor,
        "_run_text_api",
        lambda *args, **kwargs: {"task_id": "text-task", "metadata": {"artifact_paths": {}}},
    )

    report = executor.execute([str(audio_path)], operator_id="CMP_008", dataset_name="audio-demo")

    assert captured == {"qwen_asr_enabled": True, "faster_whisper_enabled": False}
    assert report["execution_engine"] == "qwen3_asr_plus_text_api"
    assert report["raw_artifacts"]["transcript_sources"][0]["segments"][0]["engine_name"] == "qwen3-asr"
    assert (executor.output_dir / "07c_audio_text_alignment_index.jsonl").exists()
    assert (executor.output_dir / "24_audio_text_risk_records.jsonl").exists()
    assert (executor.output_dir / "29_audio_run_summary.json").exists()


def test_text_api_bridge_marks_whole_audio_asr_mapping_as_coarse(tmp_path: Path):
    settings = Settings(work_dir=tmp_path, audio_redaction_enabled=True)
    executor = AudioTextApiBridgeExecutor(settings=settings, run_id="run-coarse", output_dir=tmp_path / "run-coarse")
    paths = executor._local_artifact_paths()

    transcript_units = [
        TranscriptUnit(
            unit_id="unit-1",
            source_id="source-1",
            start_time=0.0,
            end_time=10.0,
            text="张三的手机号是13800138000",
            confidence=0.9,
            engine_name="qwen3-asr-vllm",
            language="zh",
            metadata={"timestamp_granularity": "whole_audio"},
        )
    ]
    source_docs = executor._build_source_documents(
        transcript_units,
        [type("Source", (), {"source_id": "source-1", "path": str(tmp_path / "sample.wav")})()],
    )
    redaction_records = [
        {
            "doc_id": "source-1",
            "redaction_targets": [
                {
                    "finding_id": "f-1",
                    "start": 7,
                    "end": 18,
                    "original_text": "13800138000",
                    "replacement": "<PHONE>",
                    "pii_type": "phone_number",
                }
            ],
        }
    ]
    governance = {
        "privacy": [],
        "content_safety": [],
        "redaction_plan": redaction_records,
        "privacy_fragment_adjudications": [],
        "content_fragment_adjudications": [],
        "privacy_document_assessments": [],
        "content_document_assessments": [],
        "policy": [
            {
                "doc_id": "source-1",
                "redaction_targets": redaction_records[0]["redaction_targets"],
            }
        ],
        "evidence": [
            {
                "event_id": "ev-1",
                "doc_id": "source-1",
                "category": "privacy",
                "finding_refs": ["f-1"],
                "primary_span": {"start": 7, "end": 18, "text": "13800138000"},
            }
        ],
    }

    risk_records = executor._build_audio_text_risk_records(source_docs, governance)
    assert risk_records[0]["audio_span"]["mapping_status"] == "mapped"
    assert risk_records[0]["audio_span"]["mapping_precision"] == "coarse"
    assert risk_records[0]["audio_span"]["timestamp_granularity"] == "whole_audio"

    spans = executor._build_audio_redaction_spans(source_docs, redaction_records)
    assert len(spans) == 1
    assert spans[0].metadata["mapping_precision"] == "coarse"
    assert executor._render_redacted_audio([], spans, paths) == []
    assert paths["redacted_audio"].with_suffix(".skipped.json").exists()
