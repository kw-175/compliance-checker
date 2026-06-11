from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio.config.settings import Settings
from audio.models.schemas import (
    ASRSegment,
    AudioHardCaseJudgement,
    AudioHardCaseResult,
    Decision,
    DedupTranscriptUnit,
    EvidenceBundle,
    NormalizedAudioRecord,
    PIIEntity,
    PrivacyResult,
    SafetyLevel,
    SafetyResult,
    SourceType,
    SpeakerSegment,
    TranscriptEvidence,
    TranscriptUnit,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_AUDIO = FIXTURES_DIR / "sample_audio.wav"


@pytest.fixture
def sample_audio_path() -> str:
    return str(SAMPLE_AUDIO.resolve())


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@pytest.fixture
def core_cleaned_audio_package(tmp_path: Path) -> Path:
    package_dir = tmp_path / "core_audio_package"
    audio_dir = package_dir / "normalized_audio"
    audio_dir.mkdir(parents=True)
    target_audio = audio_dir / "aud_001.wav"
    shutil.copy2(SAMPLE_AUDIO, target_audio)

    (package_dir / "metadata.json").write_text(
        json.dumps(
            {
                "package_id": "pkg-core-audio",
                "package_contract_version": "audio-clean-package-v1",
                "package_level": "core",
                "task_id": "task-audio-001",
                "tenant_id": "tenant-demo",
                "profile_id": "education-default",
                "modality": "audio",
                "source_type": "cleaned_audio_package",
                "provided_files": ["audio_manifest.jsonl"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        package_dir / "audio_manifest.jsonl",
        [
            {
                "audio_id": "aud_001",
                "source_id": "raw_src_001",
                "clean_audio_path": "normalized_audio/aud_001.wav",
                "original_ref": "raw/audio_001.mp3",
                "duration_seconds": 2.0,
                "sample_rate": 16000,
                "channels": 1,
                "codec": "pcm_s16le",
                "quality_status": "unknown",
            }
        ],
    )
    return package_dir


@pytest.fixture
def extended_cleaned_audio_package(core_cleaned_audio_package: Path) -> Path:
    package_dir = core_cleaned_audio_package
    metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["package_level"] = "extended"
    metadata["provided_files"] = [
        "audio_manifest.jsonl",
        "quality_report.jsonl",
        "segments_manifest.jsonl",
        "transcript_segments.jsonl",
        "speaker_segments.jsonl",
        "lineage.jsonl",
    ]
    (package_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    _write_jsonl(package_dir / "quality_report.jsonl", [{"audio_id": "aud_001", "quality_status": "pass", "speech_ratio": 0.9}])
    _write_jsonl(package_dir / "segments_manifest.jsonl", [{"audio_id": "aud_001", "segment_id": "seg_001", "start_time": 0.0, "end_time": 2.0}])
    _write_jsonl(
        package_dir / "transcript_segments.jsonl",
        [
            {
                "audio_id": "aud_001",
                "segment_id": "seg_001",
                "start_time": 0.0,
                "end_time": 2.0,
                "text": "Send wire transfer to john@example.com immediately.",
                "confidence": 0.96,
                "engine_name": "cleaner-asr",
                "language": "en",
            }
        ],
    )
    _write_jsonl(
        package_dir / "speaker_segments.jsonl",
        [
            {
                "audio_id": "aud_001",
                "speaker_id": "speaker_teacher",
                "start_time": 0.0,
                "end_time": 2.0,
                "confidence": 0.91,
                "engine_name": "cleaner-diarization",
            }
        ],
    )
    _write_jsonl(
        package_dir / "lineage.jsonl",
        [
            {
                "audio_id": "aud_001",
                "source_id": "raw_src_001",
                "input_ref": "raw/audio_001.mp3",
                "output_ref": "normalized_audio/aud_001.wav",
                "operations": [{"name": "resample", "sample_rate": 16000}],
            }
        ],
    )
    return package_dir


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        work_dir=tmp_path / "work",
        qwen_asr_enabled=False,
        faster_whisper_enabled=False,
        pyannote_enabled=False,
        opa_enabled=False,
        qwen_guard_enabled=False,
        pii_enable_presidio=False,
        pii_enable_gliner=False,
        hard_case_local_model_path="",
        hard_case_endpoint="",
    )


def test_source_intake(sample_audio_path: str) -> None:
    from audio.steps.a_source_intake import run

    records = run([sample_audio_path])
    assert len(records) == 1
    assert records[0].size_bytes > 0


def test_cleaned_audio_package_intake(core_cleaned_audio_package: Path) -> None:
    from audio.steps.a_source_intake import run

    records = run([str(core_cleaned_audio_package)])
    assert len(records) == 1
    assert records[0].source_id == "aud_001"
    assert records[0].metadata["cleaned_audio_package"] is True
    assert records[0].metadata["package_level"] == "core"
    assert records[0].metadata["original_ref"] == "raw/audio_001.mp3"


def test_cleaned_audio_package_normalize_reuses_audio(core_cleaned_audio_package: Path, settings: Settings, tmp_path: Path) -> None:
    from audio.steps import a_source_intake, b1_source_classify, c0_audio_normalize

    sources = a_source_intake.run([str(core_cleaned_audio_package)])
    profiles = b1_source_classify.run(sources)
    records = c0_audio_normalize.run(profiles, settings, tmp_path)
    assert len(records) == 1
    assert records[0].engine_name == "cleaned_package"
    assert records[0].source_id == "aud_001"
    assert records[0].sample_rate == 16000
    assert Path(records[0].normalized_path).name == "aud_001.wav"


def test_extended_audio_package_sidecars(extended_cleaned_audio_package: Path, settings: Settings, tmp_path: Path) -> None:
    from audio.steps import a_source_intake, b1_source_classify, c0_audio_normalize, c1_asr_transcribe, c1b_diarization, c2_transcript_build

    sources = a_source_intake.run([str(extended_cleaned_audio_package)])
    profiles = b1_source_classify.run(sources)
    normalized = c0_audio_normalize.run(profiles, settings, tmp_path)
    asr_segments = c1_asr_transcribe.run(normalized, settings)
    speaker_segments = c1b_diarization.run(normalized, settings)
    transcript_units = c2_transcript_build.run(asr_segments, speaker_segments)

    assert asr_segments[0].engine_name == "cleaner-asr"
    assert speaker_segments[0].speaker_id == "speaker_teacher"
    assert transcript_units[0].speaker_id == "speaker_teacher"
    assert "john@example.com" in transcript_units[0].text


def test_audio_normalize(sample_audio_path: str, settings: Settings, tmp_path: Path) -> None:
    from audio.models.schemas import SourceProfile
    from audio.steps.c0_audio_normalize import run

    profiles = [SourceProfile(source_id="src-001", path=sample_audio_path, source_type=SourceType.AUDIO, mime_type="audio/wav")]
    records = run(profiles, settings, tmp_path)
    assert len(records) == 1
    assert Path(records[0].normalized_path).exists()


def test_asr_fallback(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    from audio.steps import c1_asr_transcribe

    record = NormalizedAudioRecord(source_id="src-001", original_path="orig.wav", normalized_path="norm.wav", duration_seconds=2.0)

    def fail_qwen(*args, **kwargs):
        raise RuntimeError("qwen unavailable")

    def fake_whisper(*args, **kwargs):
        return [ASRSegment(source_id="src-001", start_time=0.0, end_time=1.0, text="fallback transcript", confidence=0.8, engine_name="faster-whisper")]

    monkeypatch.setattr(c1_asr_transcribe, "_run_qwen_asr", fail_qwen)
    monkeypatch.setattr(c1_asr_transcribe, "_run_faster_whisper", fake_whisper)

    result = c1_asr_transcribe.run([record], settings.model_copy(update={"qwen_asr_enabled": True, "faster_whisper_enabled": True}))
    assert len(result) == 1
    assert result[0].engine_name == "faster-whisper"


def test_asr_required_rejects_placeholder_transcript(settings: Settings) -> None:
    from audio.steps import c1_asr_transcribe

    record = NormalizedAudioRecord(source_id="src-001", original_path="orig.wav", normalized_path="norm.wav", duration_seconds=2.0)

    with pytest.raises(RuntimeError, match="ASR transcript unavailable"):
        c1_asr_transcribe.run(
            [record],
            settings.model_copy(
                update={
                    "qwen_asr_enabled": False,
                    "faster_whisper_enabled": False,
                    "asr_required": True,
                    "asr_unavailable_fallback_enabled": False,
                }
            ),
        )


def test_qwen_asr_pipeline_is_cached(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    from audio.adapters import qwen_asr_adapter
    from audio.steps import c1_asr_transcribe

    class FakeASRPipeline:
        def __call__(self, audio_path: str) -> dict:
            return {"text": f"transcript for {Path(audio_path).stem}", "chunks": []}

    build_count = 0

    def fake_build(*args, **kwargs):
        nonlocal build_count
        build_count += 1
        return FakeASRPipeline()

    qwen_asr_adapter.reset_cache()
    monkeypatch.setattr(qwen_asr_adapter, "build_pipeline", fake_build)

    records = [
        NormalizedAudioRecord(source_id="src-001", original_path="orig-a.wav", normalized_path="norm-a.wav", duration_seconds=1.0),
        NormalizedAudioRecord(source_id="src-002", original_path="orig-b.wav", normalized_path="norm-b.wav", duration_seconds=1.0),
    ]
    result = c1_asr_transcribe.run(
        records,
        settings.model_copy(
            update={
                "qwen_asr_enabled": True,
                "faster_whisper_enabled": False,
                "qwen_asr_device": "cpu",
            }
        ),
    )

    qwen_asr_adapter.reset_cache()
    assert build_count == 1
    assert [segment.source_id for segment in result] == ["src-001", "src-002"]
    assert all(segment.engine_name == "qwen3-asr" for segment in result)


def test_qwen_asr_uses_endpoint_without_local_model(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    from audio.adapters import qwen_asr_adapter
    from audio.steps import c1_asr_transcribe

    def fake_endpoint(record: NormalizedAudioRecord, settings: Settings) -> list[ASRSegment]:
        assert settings.qwen_asr_endpoint == "http://asr.local/transcribe"
        return [
            ASRSegment(
                source_id=record.source_id,
                start_time=0.0,
                end_time=1.0,
                text="endpoint transcript",
                confidence=0.9,
                engine_name="qwen3-asr",
            )
        ]

    def fail_local(*args, **kwargs):
        raise AssertionError("local Qwen ASR model should not load when endpoint returns segments")

    monkeypatch.setattr(qwen_asr_adapter, "transcribe_endpoint", fake_endpoint)
    monkeypatch.setattr(qwen_asr_adapter, "transcribe_local", fail_local)
    record = NormalizedAudioRecord(source_id="src-001", original_path="orig.wav", normalized_path="norm.wav", duration_seconds=1.0)
    result = c1_asr_transcribe.run(
        [record],
        settings.model_copy(update={"qwen_asr_enabled": True, "qwen_asr_endpoint": "http://asr.local/transcribe"}),
    )

    assert len(result) == 1
    assert result[0].text == "endpoint transcript"


def test_safety_moderation_uses_qwen_guard_adapter(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    from audio.adapters import qwen_guard_adapter
    from audio.steps.g_safety_moderation import run

    monkeypatch.setattr(qwen_guard_adapter, "load_model", lambda settings: (object(), object(), "cpu"))

    def fake_moderate(text: str, settings: Settings) -> qwen_guard_adapter.QwenGuardResult:
        assert text == "redacted safety text"
        return qwen_guard_adapter.QwenGuardResult(
            level=SafetyLevel.UNSAFE,
            categories=["violent_content"],
            raw_output="Safety: Unsafe; Categories: violent_content",
            model_version="fake-qwen-guard",
        )

    monkeypatch.setattr(qwen_guard_adapter, "moderate", fake_moderate)
    results = run(
        [
            PrivacyResult(
                unit_id="u1",
                source_id="src-001",
                original_text="raw safety text",
                redacted_text="redacted safety text",
            )
        ],
        settings.model_copy(update={"qwen_guard_enabled": True}),
    )

    assert results[0].provider_name == "qwen_guard"
    assert results[0].model_version == "fake-qwen-guard"
    assert results[0].safety_level == SafetyLevel.UNSAFE
    assert results[0].harm_categories == ["violent_content"]


def test_safety_moderation_uses_guard_endpoint_without_local_load(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    from audio.adapters import qwen_guard_adapter
    from audio.steps.g_safety_moderation import run

    def fail_load_model(settings: Settings):
        raise AssertionError("local Qwen Guard model should not load when endpoint is configured")

    def fake_moderate(text: str, settings: Settings) -> qwen_guard_adapter.QwenGuardResult:
        assert settings.qwen_guard_endpoint == "http://guard.local/moderate"
        return qwen_guard_adapter.QwenGuardResult(
            level=SafetyLevel.CONTROVERSIAL,
            categories=["politically_sensitive"],
            raw_output="Safety: Controversial; Categories: politically_sensitive",
            model_version="guard-endpoint",
        )

    monkeypatch.setattr(qwen_guard_adapter, "load_model", fail_load_model)
    monkeypatch.setattr(qwen_guard_adapter, "moderate", fake_moderate)
    results = run(
        [PrivacyResult(unit_id="u1", source_id="src-001", redacted_text="borderline text")],
        settings.model_copy(update={"qwen_guard_enabled": True, "qwen_guard_endpoint": "http://guard.local/moderate"}),
    )

    assert results[0].provider_name == "qwen_guard_endpoint"
    assert results[0].model_version == "guard-endpoint"
    assert results[0].safety_level == SafetyLevel.CONTROVERSIAL


def test_diarization_fallback(settings: Settings) -> None:
    from audio.steps.c1b_diarization import run

    record = NormalizedAudioRecord(source_id="src-001", original_path="orig.wav", normalized_path="norm.wav", duration_seconds=3.2)
    result = run([record], settings)
    assert len(result) == 1
    assert result[0].speaker_id == "speaker_0"


def test_transcript_build() -> None:
    from audio.steps.c2_transcript_build import run

    asr_segments = [ASRSegment(source_id="src-001", start_time=0.0, end_time=1.0, text="hello world", confidence=0.9, engine_name="test")]
    speaker_segments = [SpeakerSegment(source_id="src-001", speaker_id="speaker_a", start_time=0.0, end_time=2.0, engine_name="test")]
    units = run(asr_segments, speaker_segments)
    assert len(units) == 1
    assert units[0].speaker_id == "speaker_a"


def test_legacy_keyword_and_regex_scan(settings: Settings) -> None:
    from audio.legacy_steps.e1a_keyword_scan import run as keyword_run
    from audio.legacy_steps.e1b_regex_scan import run as regex_run

    units = [DedupTranscriptUnit(unit_id="u1", source_id="src-001", start_time=0.0, end_time=1.0, speaker_id="speaker_0", text="Send wire transfer to john@example.com immediately.")]
    keyword_hits = keyword_run(units, settings)
    regex_hits = regex_run(units, settings)
    assert any(hit.keyword == "wire transfer" for hit in keyword_hits)
    assert any(hit.pattern_name == "email_address" for hit in regex_hits)


def test_audio_pii_local_engine_regex_detects_bilingual(settings: Settings) -> None:
    from audio.steps.f_privacy_detection import run

    units = [
        TranscriptUnit(
            unit_id="u1",
            source_id="src-001",
            start_time=0.0,
            end_time=2.0,
            speaker_id="speaker_0",
            text="Contact john@example.com, phone 13800138000, student id STU20240001.",
            language="zh",
        )
    ]

    results, spans = run(units, settings)
    entity_types = {entity.entity_type for entity in results[0].pii_entities}
    assert {"EMAIL_ADDRESS", "CN_PHONE_NUMBER", "STUDENT_ID"}.issubset(entity_types)
    assert "<EMAIL>" in results[0].redacted_text
    assert "<PHONE>" in results[0].redacted_text
    assert len(spans) == results[0].pii_count


def test_audio_privacy_pre_detection_pruning_keeps_only_selected_entities(settings: Settings) -> None:
    from audio.steps.f_privacy_detection import run

    units = [
        TranscriptUnit(
            unit_id="u1",
            source_id="src-001",
            start_time=0.0,
            end_time=2.0,
            speaker_id="speaker_0",
            text="Contact john@example.com, phone 13800138000, student id STU20240001.",
            language="zh",
        )
    ]

    results, spans = run(units, settings, target_entity_types=["EMAIL_ADDRESS"])

    assert [entity.entity_type for entity in results[0].pii_entities] == ["EMAIL_ADDRESS"]
    assert results[0].redacted_text == "Contact <EMAIL>, phone 13800138000, student id STU20240001."
    assert len(spans) == 1
    assert spans[0].entity_type == "EMAIL_ADDRESS"


def test_audio_content_safety_pre_detection_pruning_limits_mock_categories(settings: Settings) -> None:
    from audio.steps.g_safety_moderation import run

    privacy_results = [
        PrivacyResult(
            unit_id="u1",
            source_id="src-001",
            original_text="political protest with bomb threat",
            redacted_text="political protest with bomb threat",
        )
    ]

    political = run(privacy_results, settings.model_copy(update={"qwen_guard_enabled": False}), target_labels=["content.political"])
    violence = run(privacy_results, settings.model_copy(update={"qwen_guard_enabled": False}), target_labels=["content.violent"])

    assert political[0].harm_categories == ["politically_sensitive"]
    assert violence[0].harm_categories == ["violent_content"]


def test_audio_content_safety_mock_recalls_chinese_self_harm(settings: Settings) -> None:
    from audio.steps.g_safety_moderation import run

    privacy_results = [
        PrivacyResult(
            unit_id="u1",
            source_id="src-001",
            original_text="学生哭着说自己不想活了，还想伤害自己。",
            redacted_text="学生哭着说自己不想活了，还想伤害自己。",
        )
    ]

    results = run(
        privacy_results,
        settings.model_copy(update={"qwen_guard_enabled": False}),
        target_labels=["content.self_harm"],
    )

    assert results[0].safety_level == SafetyLevel.UNSAFE
    assert results[0].harm_categories == ["suicide_self_harm"]


def test_privacy_detection_uses_pii_endpoint(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    from audio.adapters import pii_service_adapter
    from audio.steps.f_privacy_detection import run
    from audio.steps.pii_local_engine import LocalPIIDetection

    def fake_detect(text: str, settings: Settings, language: str = "") -> LocalPIIDetection:
        assert settings.pii_endpoint == "http://pii.local/analyze"
        return LocalPIIDetection(
            entities=[
                PIIEntity(
                    entity_type="EMAIL_ADDRESS",
                    start=text.index("john@example.com"),
                    end=text.index("john@example.com") + len("john@example.com"),
                    score=0.95,
                    original_text="john@example.com",
                )
            ],
            provider_name="pii_endpoint",
            provider_version=settings.pii_endpoint,
            is_degraded=False,
        )

    monkeypatch.setattr(pii_service_adapter, "detect", fake_detect)
    units = [
        TranscriptUnit(
            unit_id="u1",
            source_id="src-001",
            start_time=0.0,
            end_time=1.0,
            text="Contact john@example.com.",
        )
    ]
    results, spans = run(units, settings.model_copy(update={"pii_endpoint": "http://pii.local/analyze"}))

    assert results[0].provider_name == "pii_endpoint"
    assert results[0].provider_version == "http://pii.local/analyze"
    assert results[0].redacted_text == "Contact <EMAIL>."
    assert len(spans) == 1


def test_hard_case_adjudication_heuristic_for_uncertain_safety(settings: Settings) -> None:
    from audio.steps.hard_case_adjudication import run

    units = [
        TranscriptUnit(
            unit_id="u1",
            source_id="src-001",
            start_time=0.0,
            end_time=1.0,
            speaker_id="speaker_0",
            text="This is a borderline safety statement.",
            confidence=0.55,
            engine_name="fallback",
        )
    ]
    safety_results = [
        SafetyResult(
            unit_id="u1",
            source_id="src-001",
            safety_level=SafetyLevel.CONTROVERSIAL,
            score=0.5,
            raw_output="borderline",
        )
    ]

    results = run(units, [], safety_results, settings, run_id="run-001")
    assert len(results) == 1
    assert results[0].provider_name == "heuristic_fallback"
    assert results[0].is_degraded is True
    assert "content_safety" in results[0].trigger_sources
    assert results[0].judgement.recommended_decision == Decision.REVIEW


def test_hard_case_adjudication_uses_qwen_adapter(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    from audio.adapters import qwen_hard_case_adapter
    from audio.steps.hard_case_adjudication import run

    def fake_adjudicate(prompt: str, settings: Settings) -> qwen_hard_case_adapter.HardCaseAdapterResult:
        assert "audio_transcript_unit" in prompt
        return qwen_hard_case_adapter.HardCaseAdapterResult(
            judgement=AudioHardCaseJudgement(
                content_status="borderline",
                privacy_status="clear",
                confidence=0.82,
                rationale="Resolved by adapter.",
                recommended_decision=Decision.REVIEW,
                requires_manual_review=True,
                final_reasons=["adapter_review"],
            ),
            provider_name="qwen_local",
            raw_response='{"recommended_decision":"review"}',
        )

    monkeypatch.setattr(qwen_hard_case_adapter, "adjudicate", fake_adjudicate)
    units = [
        TranscriptUnit(
            unit_id="u1",
            source_id="src-001",
            start_time=0.0,
            end_time=1.0,
            speaker_id="speaker_0",
            text="This is a borderline safety statement.",
            confidence=0.8,
        )
    ]
    safety_results = [
        SafetyResult(
            unit_id="u1",
            source_id="src-001",
            safety_level=SafetyLevel.CONTROVERSIAL,
            score=0.5,
            raw_output="borderline",
        )
    ]

    results = run(units, [], safety_results, settings, run_id="run-001")
    assert len(results) == 1
    assert results[0].provider_name == "qwen_local"
    assert results[0].is_degraded is False
    assert results[0].judgement.final_reasons == ["adapter_review"]


def test_evidence_aggregation() -> None:
    from audio.steps.h_evidence_aggregation import run

    units = [TranscriptUnit(unit_id="u1", source_id="src-001", start_time=0.0, end_time=1.0, speaker_id="speaker_0", text="text")]
    bundle = run(
        units,
        [PrivacyResult(unit_id="u1", source_id="src-001", pii_count=1)],
        [SafetyResult(unit_id="u1", source_id="src-001", safety_level=SafetyLevel.UNSAFE)],
        "run-001",
    )
    assert bundle.summary["total_units"] == 1
    assert bundle.summary["total_pii_entities"] == 1
    assert bundle.summary["unsafe_units"] == 1


def test_policy_uses_hard_case_recommendation(settings: Settings) -> None:
    from audio.steps.i_policy_decision import run

    hard_case = AudioHardCaseResult(
        run_id="run-001",
        unit_id="u1",
        source_id="src-001",
        trigger_sources=["privacy"],
        trigger_reasons=["privacy_score_band_uncertain"],
        provider_name="qwen_local",
        adjudicated=True,
        uncertainty=0.2,
        judgement=AudioHardCaseJudgement(
            confidence=0.8,
            recommended_decision=Decision.QUARANTINE,
            requires_manual_review=True,
            rationale="Dense uncertain PII.",
        ),
    )
    bundle = EvidenceBundle(
        pipeline_run_id="run-001",
        transcript_units=[
            TranscriptEvidence(
                unit_id="u1",
                source_id="src-001",
                text="borderline pii",
                hard_case=hard_case,
            )
        ],
        summary={},
    )

    decision = run(bundle, settings)
    assert decision.overall_decision == Decision.QUARANTINE
    assert any("hard-case adjudication" in reason for reason in decision.unit_decisions[0].reasons)


def test_local_rule_decision(settings: Settings) -> None:
    from audio.steps.i_policy_decision import run

    bundle = EvidenceBundle(
        pipeline_run_id="run-001",
        transcript_units=[
            TranscriptEvidence(
                unit_id="u1",
                source_id="src-001",
                text="dangerous",
                safety=SafetyResult(unit_id="u1", source_id="src-001", safety_level=SafetyLevel.UNSAFE),
            )
        ],
        summary={},
    )
    decision = run(bundle, settings)
    assert decision.overall_decision.value == "reject"


def test_policy_opa_config_uses_local_privacy_safety_rules(settings: Settings) -> None:
    from audio.steps import i_policy_decision

    bundle = EvidenceBundle(
        pipeline_run_id="run-opa",
        transcript_units=[
            TranscriptEvidence(
                unit_id="u1",
                source_id="src-001",
                text="safe text",
            )
        ],
        summary={},
    )
    decision = i_policy_decision.run(bundle, settings.model_copy(update={"opa_enabled": True}))
    assert decision.overall_decision.value == "allow"


def test_legacy_audio_redaction_copy_strategy(settings: Settings, tmp_path: Path) -> None:
    from audio.legacy_steps.k_audio_redaction import run
    from audio.models.schemas import RedactionSpan, RenderStrategy

    record = NormalizedAudioRecord(
        source_id="src-001",
        original_path=str(SAMPLE_AUDIO),
        normalized_path=str(SAMPLE_AUDIO),
        duration_seconds=2.0,
    )
    spans = [
        RedactionSpan(
            source_id="src-001",
            unit_id="u1",
            start_time=0.1,
            end_time=0.5,
            entity_type="EMAIL_ADDRESS",
            original_text="john@example.com",
            replacement="<EMAIL>",
        )
    ]

    outputs = run([record], spans, settings.model_copy(update={"redaction_strategy": "copy"}), tmp_path)
    assert len(outputs) == 1
    assert outputs[0].render_strategy == RenderStrategy.COPY
    assert Path(outputs[0].redacted_audio_path).exists()


def test_pipeline_integration(sample_audio_path: str, settings: Settings) -> None:
    from audio.pipeline import AudioCompliancePipeline

    pipeline = AudioCompliancePipeline(settings=settings)
    decision = pipeline.execute([sample_audio_path])
    assert decision.pipeline_run_id == pipeline.run_id
    assert (pipeline.output_dir / "12_annotation_package.jsonl").exists()
    assert (pipeline.output_dir / "13_audit_package.jsonl").exists()
    assert (pipeline.output_dir / "14_run_summary.jsonl").exists()
    assert (pipeline.output_dir / "09b_hard_case_adjudication.jsonl").exists()
    assert (pipeline.output_dir / "compliance_output.json").exists()


def test_pipeline_integration_cleaned_package(extended_cleaned_audio_package: Path, settings: Settings) -> None:
    from audio.pipeline import AudioCompliancePipeline

    pipeline = AudioCompliancePipeline(settings=settings)
    decision = pipeline.execute([str(extended_cleaned_audio_package)])
    asr_rows = [
        json.loads(line)
        for line in (pipeline.output_dir / "04_asr_segments.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert decision.pipeline_run_id == pipeline.run_id
    assert asr_rows[0]["engine_name"] == "cleaner-asr"
    assert asr_rows[0]["source_id"] == "aud_001"
    assert (pipeline.output_dir / "12_annotation_package.jsonl").exists()
    assert (pipeline.output_dir / "13_audit_package.jsonl").exists()


def test_health_endpoint() -> None:
    from audio.server import app

    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
