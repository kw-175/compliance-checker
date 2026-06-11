from __future__ import annotations

import json
from pathlib import Path

from audio.config.settings import Settings
from audio.models.schemas import NormalizedAudioRecord
from audio.steps import c1_asr_transcribe


def test_asr_transcribe_loads_whisper_json_sidecar(tmp_path: Path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF")

    sidecar_path = tmp_path / "sample.transcript.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "language": "zh",
                "segments": [
                    {"segment_id": "seg-1", "start": 0.0, "end": 1.2, "text": "第一句", "confidence": 0.92},
                    {"segment_id": "seg-2", "start": 1.2, "end": 2.4, "text": "第二句", "confidence": 0.87},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    record = NormalizedAudioRecord(
        source_id="audio-001",
        original_path=str(audio_path),
        normalized_path=str(audio_path),
        duration_seconds=2.4,
        metadata={},
    )

    settings = Settings()
    segments = c1_asr_transcribe.run([record], settings)

    assert [segment.segment_id for segment in segments] == ["seg-1", "seg-2"]
    assert [segment.text for segment in segments] == ["第一句", "第二句"]
    assert segments[0].engine_name == "whisper_json"
    assert segments[0].language == "zh"
