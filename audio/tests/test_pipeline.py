from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio.config.settings import Settings
from audio.models.schemas import (
    ASRSegment,
    ComplianceHit,
    DedupTranscriptUnit,
    EvidenceBundle,
    KeywordHit,
    NormalizedAudioRecord,
    PrivacyResult,
    RegexHit,
    SafetyLevel,
    SafetyResult,
    SecretHit,
    SourceType,
    SpeakerSegment,
    TranscriptEvidence,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_AUDIO = FIXTURES_DIR / "sample_audio.wav"


@pytest.fixture
def sample_audio_path() -> str:
    return str(SAMPLE_AUDIO.resolve())


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        work_dir=tmp_path / "work",
        qwen_asr_enabled=False,
        faster_whisper_enabled=False,
        pyannote_enabled=False,
        opa_enabled=False,
        qwen_guard_enabled=False,
    )


def test_source_intake(sample_audio_path: str):
    from audio.steps.a_source_intake import run

    records = run([sample_audio_path])
    assert len(records) == 1
    assert records[0].size_bytes > 0


def test_audio_normalize(sample_audio_path: str, settings: Settings, tmp_path: Path):
    from audio.models.schemas import SourceProfile
    from audio.steps.c0_audio_normalize import run

    profiles = [SourceProfile(source_id="src-001", path=sample_audio_path, source_type=SourceType.AUDIO, mime_type="audio/wav")]
    records = run(profiles, settings, tmp_path)
    assert len(records) == 1
    assert Path(records[0].normalized_path).exists()


def test_asr_fallback(monkeypatch, settings: Settings):
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


def test_diarization_fallback(settings: Settings):
    from audio.steps.c1b_diarization import run

    record = NormalizedAudioRecord(source_id="src-001", original_path="orig.wav", normalized_path="norm.wav", duration_seconds=3.2)
    result = run([record], settings)
    assert len(result) == 1
    assert result[0].speaker_id == "speaker_0"


def test_transcript_build():
    from audio.steps.c2_transcript_build import run

    asr_segments = [ASRSegment(source_id="src-001", start_time=0.0, end_time=1.0, text="hello world", confidence=0.9, engine_name="test")]
    speaker_segments = [SpeakerSegment(source_id="src-001", speaker_id="speaker_a", start_time=0.0, end_time=2.0, engine_name="test")]
    units = run(asr_segments, speaker_segments)
    assert len(units) == 1
    assert units[0].speaker_id == "speaker_a"


def test_keyword_and_regex_scan(settings: Settings):
    from audio.steps.e1a_keyword_scan import run as keyword_run
    from audio.steps.e1b_regex_scan import run as regex_run

    units = [DedupTranscriptUnit(unit_id="u1", source_id="src-001", start_time=0.0, end_time=1.0, speaker_id="speaker_0", text="Send wire transfer to john@example.com immediately.")]
    keyword_hits = keyword_run(units, settings)
    regex_hits = regex_run(units, settings)
    assert any(hit.keyword == "wire transfer" for hit in keyword_hits)
    assert any(hit.pattern_name == "email_address" for hit in regex_hits)


def test_evidence_aggregation():
    from audio.steps.h_evidence_aggregation import run

    units = [DedupTranscriptUnit(unit_id="u1", source_id="src-001", start_time=0.0, end_time=1.0, speaker_id="speaker_0", text="text")]
    bundle = run(
        units,
        [SecretHit(source_id="src-001", detector_type="test")],
        [ComplianceHit(source_id="src-001")],
        [KeywordHit(unit_id="u1", keyword="secret")],
        [RegexHit(unit_id="u1", pattern_name="email_address")],
        [PrivacyResult(unit_id="u1", source_id="src-001", pii_count=1)],
        [SafetyResult(unit_id="u1", source_id="src-001", safety_level=SafetyLevel.UNSAFE)],
        "run-001",
    )
    assert bundle.summary["total_units"] == 1
    assert bundle.summary["total_secret_hits"] == 1


def test_local_rule_decision(settings: Settings):
    from audio.steps.i_policy_decision import run

    bundle = EvidenceBundle(pipeline_run_id="run-001", transcript_units=[TranscriptEvidence(unit_id="u1", source_id="src-001", text="dangerous", secret_hits=[SecretHit(source_id="src-001", detector_type="aws")])], summary={})
    decision = run(bundle, settings)
    assert decision.overall_decision.value == "reject"


def test_policy_opa_fallback(settings: Settings, monkeypatch):
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

    def fake_query(*args, **kwargs):
        return None

    monkeypatch.setattr(i_policy_decision, "_query_opa", fake_query)
    decision = i_policy_decision.run(bundle, settings.model_copy(update={"opa_enabled": True}))
    assert decision.overall_decision.value in {"allow", "review", "quarantine", "reject"}


def test_audio_redaction_copy_strategy(settings: Settings, tmp_path: Path):
    from audio.models.schemas import RenderStrategy
    from audio.steps.k_audio_redaction import run

    record = NormalizedAudioRecord(
        source_id="src-001",
        original_path=str(SAMPLE_AUDIO),
        normalized_path=str(SAMPLE_AUDIO),
        duration_seconds=2.0,
    )
    spans = [
        {
            "source_id": "src-001",
            "unit_id": "u1",
            "start_time": 0.1,
            "end_time": 0.5,
            "entity_type": "EMAIL_ADDRESS",
            "original_text": "john@example.com",
            "replacement": "<EMAIL>",
        }
    ]
    from audio.models.schemas import RedactionSpan

    outputs = run(
        [record],
        [RedactionSpan(**item) for item in spans],
        settings.model_copy(update={"redaction_strategy": "copy"}),
        tmp_path,
    )
    assert len(outputs) == 1
    assert outputs[0].render_strategy == RenderStrategy.COPY
    assert Path(outputs[0].redacted_audio_path).exists()


def test_pipeline_integration(sample_audio_path: str, settings: Settings):
    from audio.pipeline import AudioCompliancePipeline

    pipeline = AudioCompliancePipeline(settings=settings)
    decision = pipeline.execute([sample_audio_path])
    assert decision.pipeline_run_id == pipeline.run_id
    assert (pipeline.output_dir / "release_package.json").exists()


def test_health_endpoint():
    from audio.server import app

    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
