from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from audio.config.settings import Settings
from audio.models.schemas import (
    ASRSegment,
    DedupTranscriptUnit,
    EvidenceBundle,
    NormalizedAudioRecord,
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
