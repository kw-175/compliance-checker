"""
Step C1: ASR transcription.
"""

from __future__ import annotations

import logging
from pathlib import Path

from audio.config.settings import Settings
from audio.models.schemas import ASRSegment, NormalizedAudioRecord
from audio.steps import load_jsonl

logger = logging.getLogger(__name__)


def _load_sidecar_segments(record: NormalizedAudioRecord) -> list[ASRSegment]:
    # 优先读取旁路转写文件，便于离线测试与无模型环境回放。
    candidates = [
        Path(record.original_path).with_suffix(".transcript.jsonl"),
        Path(record.original_path).with_name(f"{Path(record.original_path).stem}_transcript.jsonl"),
        Path(record.original_path).with_name("sample_transcript.jsonl"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        rows = load_jsonl(candidate)
        segments: list[ASRSegment] = []
        for row in rows:
            payload = {
                "source_id": record.source_id,
                "start_time": float(row.get("start_time", 0.0)),
                "end_time": float(row.get("end_time", 0.0)),
                "text": str(row.get("text", "")),
                "confidence": float(row.get("confidence", 0.0)),
                "engine_name": str(row.get("engine_name", "sidecar")),
                "language": str(row.get("language", "")),
                "metadata": {"sidecar_path": str(candidate)},
            }
            if row.get("segment_id"):
                payload["segment_id"] = str(row.get("segment_id"))
            segments.append(ASRSegment(**payload))
        if segments:
            # 只要某个候选文件有有效片段即立即返回。
            return segments
    return []


def _run_qwen_asr(record: NormalizedAudioRecord, settings: Settings) -> list[ASRSegment]:
    # 主 ASR 路径：基于 transformers + Qwen ASR 模型推理。
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError as exc:
        raise RuntimeError("transformers unavailable for Qwen ASR") from exc

    model = AutoModelForSpeechSeq2Seq.from_pretrained(settings.qwen_asr_model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(settings.qwen_asr_model, trust_remote_code=True)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True,
    )
    output = asr(record.normalized_path)
    chunks = output.get("chunks") or []
    if not chunks and output.get("text"):
        # 若模型未返回分段，退化为单段覆盖整段音频。
        chunks = [{"timestamp": (0.0, record.duration_seconds), "text": output.get("text", "")}]
    return [
        ASRSegment(
            source_id=record.source_id,
            start_time=float(chunk.get("timestamp", (0.0, 0.0))[0] or 0.0),
            end_time=float(chunk.get("timestamp", (0.0, 0.0))[1] or record.duration_seconds),
            text=str(chunk.get("text", "")).strip(),
            confidence=0.9,
            engine_name="qwen3-asr",
            language=str(output.get("language", "")),
        )
        for chunk in chunks
        if str(chunk.get("text", "")).strip()
    ]


def _run_faster_whisper(record: NormalizedAudioRecord, settings: Settings) -> list[ASRSegment]:
    # 次级 ASR 路径：Qwen 不可用时回退到 faster-whisper。
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper unavailable") from exc

    model = WhisperModel(settings.faster_whisper_model, device="cpu", compute_type="int8")
    segments, info = model.transcribe(record.normalized_path, vad_filter=True)
    return [
        ASRSegment(
            source_id=record.source_id,
            start_time=float(segment.start),
            end_time=float(segment.end),
            text=str(segment.text).strip(),
            confidence=float(getattr(segment, "avg_logprob", 0.0) or 0.0),
            engine_name="faster-whisper",
            language=str(getattr(info, "language", "")),
        )
        for segment in segments
        if str(segment.text).strip()
    ]


def _fallback_segments(record: NormalizedAudioRecord) -> list[ASRSegment]:
    # 最终兜底：旁路转写 > 占位文本，保证下游结构完整。
    sidecar = _load_sidecar_segments(record)
    if sidecar:
        return sidecar
    return [
        ASRSegment(
            source_id=record.source_id,
            start_time=0.0,
            end_time=record.duration_seconds,
            text="Audio transcript unavailable",
            confidence=0.0,
            engine_name="fallback",
        )
    ]


def run(records: list[NormalizedAudioRecord], settings: Settings) -> list[ASRSegment]:
    # 每条音频按“Qwen -> Whisper -> Fallback”顺序尝试。
    all_segments: list[ASRSegment] = []
    for record in records:
        segments: list[ASRSegment] = []
        if settings.qwen_asr_enabled:
            try:
                segments = _run_qwen_asr(record, settings)
            except Exception as exc:
                # 单引擎失败只记录告警，不中断当前音频处理。
                logger.warning("Qwen ASR failed for %s: %s", record.source_id, exc)
        if not segments and settings.faster_whisper_enabled:
            try:
                segments = _run_faster_whisper(record, settings)
            except Exception as exc:
                logger.warning("faster-whisper failed for %s: %s", record.source_id, exc)
        if not segments:
            segments = _fallback_segments(record)
        all_segments.extend(segments)
    return all_segments
