from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

from audio.config.settings import Settings
from audio.models.schemas import NormalizedAudioRecord, RedactionSpan, TranscriptUnit

logger = logging.getLogger(__name__)

_OPERATOR_PIPELINE_PROFILE = {
    "CMP_001": "privacy_only",
    "CMP_002": "safety_only",
    "CMP_008": "full",
}
_OPERATOR_NAMES = {
    "CMP_001": "Sensitive information detection",
    "CMP_002": "Content safety detection",
    "CMP_008": "Full compliance detection",
}
_SUPPLEMENTAL_FINDING_TYPES = {"api_privacy_invalid_span", "combined_identity"}
_SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
_DISPOSITION_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5}
_DECISION_RANK = {"allow": 0, "quarantine": 1, "review": 2, "reject": 3}
_RISK_TYPE_LABELS_ZH = {
    "person_name": "姓名",
    "phone": "手机号",
    "phone_number": "手机号",
    "email": "邮箱",
    "id_card": "身份证件",
    "address": "地址",
    "student_id": "学号",
    "parent_contact": "监护人联系方式",
    "political_sensitive": "政治与公共安全敏感",
    "politically_sensitive": "政治与公共安全敏感",
    "pornographic_content": "色情低俗内容",
    "violence": "暴力与危险行为",
    "hate_speech": "仇恨歧视",
    "harassment": "辱骂骚扰与霸凌",
    "self_harm": "自伤自杀风险",
    "illegal_instruction": "违法危险教程",
    "minor_harmful": "未成年人有害内容",
    "misleading": "误导欺诈",
    "values_violation": "教育价值观风险",
    "jailbreak_attempt": "提示注入与越狱",
    "general_content_safety": "内容安全风险",
}
_POLICY_TAG_LABELS_ZH = {
    "pii.person_name": "姓名",
    "pii.name": "姓名",
    "pii.phone": "手机号",
    "pii.phone_number": "手机号",
    "pii.email": "邮箱",
    "pii.id_card": "身份证件",
    "pii.address": "地址",
    "pii.student_id": "学号",
    "pii.parent_contact": "监护人联系方式",
    "content.political": "政治与公共安全敏感",
    "content.pornographic": "色情低俗内容",
    "content.pornographic.explicit": "色情低俗内容",
    "content.violent": "暴力与危险行为",
    "content.violent.encouragement": "暴力与危险行为",
    "content.violent.graphic_description": "暴力与危险行为",
    "content.hate": "仇恨歧视",
    "content.harassment": "辱骂骚扰与霸凌",
    "content.self_harm": "自伤自杀风险",
    "content.illegal_instruction": "违法危险教程",
    "content.minor_harmful": "未成年人有害内容",
    "content.misleading": "误导欺诈",
    "content.values_violation": "教育价值观风险",
    "content.jailbreak": "提示注入与越狱",
}
_RISK_LEVEL_LABELS_ZH = {
    "none": "无风险",
    "low": "低风险",
    "medium": "中风险",
    "high": "高风险",
    "critical": "极高风险",
}
_SPOKEN_DIGIT_MAP = {
    "零": "0",
    "〇": "0",
    "○": "0",
    "洞": "0",
    "O": "0",
    "o": "0",
    "幺": "1",
    "一": "1",
    "二": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
}
_SPOKEN_DIGIT_CHARS = "零〇○洞Oo幺一二两三四五六七八九0-9"
_SPOKEN_NUMBER_CHARS = _SPOKEN_DIGIT_CHARS + r"\s\-—_杠"
_ASR_LOW_CONFIDENCE_THRESHOLD = 0.65
_ASR_LONG_SINGLE_SEGMENT_SECONDS = 45.0
_ASR_MIN_TEXT_CHARS_PER_MINUTE = 12.0


def _write_jsonl(records: list[Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            if hasattr(record, "model_dump_json"):
                handle.write(record.model_dump_json() + "\n")
            else:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(record: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        if hasattr(record, "model_dump_json"):
            handle.write(record.model_dump_json(indent=2))
        else:
            json.dump(record, handle, indent=2, ensure_ascii=False)


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL row in %s", path)
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Skipping malformed JSON artifact: %s", path)
        return {}
    return payload if isinstance(payload, dict) else {}


def _artifact_paths(metadata: dict[str, Any]) -> dict[str, Path]:
    results: dict[str, Path] = {}
    raw_paths = metadata.get("artifact_paths")
    if not isinstance(raw_paths, dict):
        return results
    for key, value in raw_paths.items():
        if value:
            results[str(key)] = Path(str(value))
    return results


def _text_api_headers(settings: Settings) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if settings.text_api_key:
        headers["Authorization"] = f"Bearer {settings.text_api_key}"
        headers["X-API-Key"] = settings.text_api_key
    return headers


def _normalize_route(value: str) -> str:
    route = str(value or "").strip().lower()
    if route in {"api", "bridge", "text_api_bridge", "external_api", "compat"}:
        return "api"
    return "local"


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    total_ms = max(int(round(float(value) * 1000.0)), 0)
    minutes, remainder = divmod(total_ms, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def _format_time_label(start_time: float | None, end_time: float | None) -> str:
    if start_time is None or end_time is None:
        return "-"
    return f"{_format_seconds(start_time)} - {_format_seconds(end_time)}"


def _risk_label_zh(policy_tag: str, risk_type: str) -> str:
    return (
        _POLICY_TAG_LABELS_ZH.get(str(policy_tag or ""))
        or _RISK_TYPE_LABELS_ZH.get(str(risk_type or ""))
        or "合规风险"
    )


def _risk_level_label_zh(risk_level: str) -> str:
    return _RISK_LEVEL_LABELS_ZH.get(str(risk_level or "").lower(), "风险")


def _apply_redactions(text: str, targets: list[dict[str, Any]]) -> str:
    redacted = text
    ordered = sorted(
        [
            target for target in targets
            if isinstance(target, dict) and isinstance(target.get("start"), int) and isinstance(target.get("end"), int)
        ],
        key=lambda item: item["start"],
        reverse=True,
    )
    for target in ordered:
        start = target["start"]
        end = target["end"]
        replacement = str(target.get("replacement") or "<REDACTED>")
        if start < 0 or end <= start or end > len(redacted):
            continue
        redacted = redacted[:start] + replacement + redacted[end:]
    return redacted


def _replacement_for(finding: dict[str, Any], redaction_by_finding: dict[str, dict[str, Any]]) -> str:
    finding_id = str(finding.get("finding_id") or "")
    target = redaction_by_finding.get(finding_id)
    if target and target.get("replacement"):
        return str(target.get("replacement"))
    return str(finding.get("redaction_suggestion") or "")


def _severity_rank(value: str) -> int:
    return _SEVERITY_RANK.get(str(value or "").strip().lower(), 0)


def _risk_level(findings: list[dict[str, Any]]) -> str:
    highest = max((_severity_rank(item.get("risk_level")) for item in findings), default=0)
    if highest <= 0:
        return "none"
    if highest == 1:
        return "low"
    if highest == 2:
        return "medium"
    if highest == 3:
        return "high"
    return "critical"


def _conclusion(risk_level: str, total_findings: int) -> str:
    if total_findings <= 0:
        return "passed"
    if risk_level in {"high", "critical"}:
        return "failed"
    return "review"


class AudioTextApiBridgeExecutor:
    def __init__(self, settings: Settings, run_id: str, output_dir: Path):
        self.settings = settings
        self.run_id = run_id
        self.output_dir = output_dir

    def execute(
        self,
        input_paths: list[str],
        *,
        operator_id: str,
        dataset_name: str,
        config_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from audio.steps import (
            a_source_intake,
            b1_source_classify,
            c0_audio_normalize,
            c1_asr_transcribe,
            c1b_diarization,
            c1c_alignment,
            c2_transcript_build,
        )

        operator_id = str(operator_id or "").strip().upper()
        if operator_id not in _OPERATOR_PIPELINE_PROFILE:
            raise ValueError(f"Unsupported audio/text bridge operator: {operator_id}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = self._local_artifact_paths()

        sources = a_source_intake.run(input_paths)
        _write_jsonl(sources, paths["intake"])
        if not sources:
            report = self._empty_report(operator_id, dataset_name, "No audio sources were discovered in the supplied paths.")
            _write_json(report, paths["audio_report"])
            return report

        profiles = b1_source_classify.run(sources)
        _write_jsonl(profiles, paths["source_profile"])

        normalized = c0_audio_normalize.run(profiles, self.settings, self.output_dir)
        _write_jsonl(normalized, paths["normalized_audio"])
        if not normalized:
            report = self._empty_report(operator_id, dataset_name, "No usable audio records were discovered after intake.")
            _write_json(report, paths["audio_report"])
            return report

        asr_segments = c1_asr_transcribe.run(normalized, self.settings)
        _write_jsonl(asr_segments, paths["asr"])

        try:
            speaker_segments = c1b_diarization.run(normalized, self.settings)
        except Exception as exc:
            logger.warning("Audio bridge diarization failed, fallback to single speaker segments: %s", exc)
            speaker_segments = []
        _write_jsonl(speaker_segments, paths["speaker"])

        aligned_segments = c1c_alignment.run(asr_segments)
        _write_jsonl(aligned_segments, paths["aligned"])

        transcript_units = c2_transcript_build.run(aligned_segments, speaker_segments)
        _write_jsonl(transcript_units, paths["transcript"])
        if not transcript_units:
            report = self._empty_report(operator_id, dataset_name, "No transcript units were available for compliance detection.")
            _write_json(report, paths["audio_report"])
            return report

        source_documents = self._build_source_documents(transcript_units, sources)
        if not source_documents:
            report = self._empty_report(operator_id, dataset_name, "Audio transcript was empty after ASR transcription.")
            _write_json(report, paths["audio_report"])
            return report

        asr_payload = {
            "run_id": self.run_id,
            "operator_id": operator_id,
            "provider": "qwen3_asr_preferred",
            "sources": [
                {
                    "source_id": item["source_id"],
                    "source_path": item["source_path"],
                    "doc_text": item["text"],
                    "segments": item["segments"],
                }
                for item in source_documents
            ],
        }
        _write_json(asr_payload, paths["asr_json"])
        alignment_index = self._build_alignment_index(source_documents)
        _write_jsonl(alignment_index, paths["alignment_index"])

        text_input_rows = [
            {
                "doc_id": item["source_id"],
                "text": item["text"],
                "metadata": {
                    "source_path": item["source_path"],
                    "unit_count": len(item["segments"]),
                    "source_id": item["source_id"],
                    "asr_quality": item.get("asr_quality") or {},
                },
            }
            for item in source_documents
        ]
        _write_jsonl(text_input_rows, paths["text_api_input"])
        _write_json(source_documents, paths["text_api_source_map"])

        text_api_result = self._run_text_api(paths["text_api_input"], operator_id, dataset_name, config_overrides or {})
        _write_json(text_api_result, paths["text_api_result"])

        text_artifact_paths = _artifact_paths(text_api_result.get("metadata") or {})
        text_governance = self._load_text_governance_artifacts(text_artifact_paths)
        redaction_records = text_governance["redaction_plan"]
        audio_text_risk_records = self._build_audio_text_risk_records(source_documents, text_governance)
        _write_jsonl(audio_text_risk_records, paths["audio_text_risk_records"])
        audio_document_assessments = self._build_audio_document_assessments(source_documents, text_governance, audio_text_risk_records)
        _write_jsonl(audio_document_assessments, paths["audio_document_assessments"])
        audio_policy_decisions = self._build_audio_policy_decisions(source_documents, text_governance, audio_text_risk_records)
        _write_jsonl(audio_policy_decisions, paths["audio_policy_decisions"])
        audio_annotation_records = self._build_audio_annotation_records(source_documents, text_governance, audio_text_risk_records)
        _write_jsonl(audio_annotation_records, paths["audio_annotation"])
        audio_audit_records = self._build_audio_audit_records(source_documents, text_governance, audio_text_risk_records)
        _write_jsonl(audio_audit_records, paths["audio_audit"])
        audio_summary = self._build_audio_summary(
            source_documents=source_documents,
            text_api_result=text_api_result,
            text_governance=text_governance,
            risk_records=audio_text_risk_records,
            artifact_paths=paths,
        )
        _write_json(audio_summary, paths["audio_summary"])
        audio_redaction_spans = self._build_audio_redaction_spans(source_documents, redaction_records)
        _write_jsonl(audio_redaction_spans, paths["audio_redaction_spans"])

        redacted_audio_records = self._render_redacted_audio(normalized, audio_redaction_spans, paths)
        _write_jsonl(redacted_audio_records, paths["redacted_audio"])

        report = self._build_audio_report(
            operator_id=operator_id,
            dataset_name=dataset_name,
            source_documents=source_documents,
            text_api_result=text_api_result,
            local_artifacts=paths,
            config_overrides=config_overrides or {},
        )
        _write_json(report, paths["audio_report"])
        return report

    def _local_artifact_paths(self) -> dict[str, Path]:
        return {
            "intake": self.output_dir / "01_source_registry.jsonl",
            "source_profile": self.output_dir / "02_source_profile.jsonl",
            "normalized_audio": self.output_dir / "03_normalized_audio_manifest.jsonl",
            "asr": self.output_dir / "04_asr_segments.jsonl",
            "speaker": self.output_dir / "05_speaker_segments.jsonl",
            "aligned": self.output_dir / "06_aligned_segments.jsonl",
            "transcript": self.output_dir / "07_transcript_units.jsonl",
            "asr_json": self.output_dir / "07b_asr_transcript.json",
            "alignment_index": self.output_dir / "07c_audio_text_alignment_index.jsonl",
            "text_api_input": self.output_dir / "20_text_api_input.jsonl",
            "text_api_source_map": self.output_dir / "20b_text_api_source_map.json",
            "text_api_result": self.output_dir / "23_text_api_result.json",
            "audio_text_risk_records": self.output_dir / "24_audio_text_risk_records.jsonl",
            "audio_document_assessments": self.output_dir / "25_audio_document_assessments.jsonl",
            "audio_policy_decisions": self.output_dir / "26_audio_policy_decisions.jsonl",
            "audio_annotation": self.output_dir / "27_audio_annotation_package.jsonl",
            "audio_audit": self.output_dir / "28_audio_audit_package.jsonl",
            "audio_summary": self.output_dir / "29_audio_run_summary.json",
            "speech_privacy_fallback": self.output_dir / "29b_speech_privacy_fallback_findings.jsonl",
            "audio_redaction_spans": self.output_dir / "30_audio_redaction_spans.jsonl",
            "redacted_audio": self.output_dir / "31_redacted_audio_manifest.jsonl",
            "audio_report": self.output_dir / "32_audio_compliance_report.json",
        }

    def _run_text_api(
        self,
        text_input_path: Path,
        operator_id: str,
        dataset_name: str,
        config_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        submit_url = self.settings.text_api_base_url.rstrip("/") + self.settings.text_api_submit_path
        status_url = self.settings.text_api_base_url.rstrip("/") + self.settings.text_api_status_path
        result_url = self.settings.text_api_base_url.rstrip("/") + self.settings.text_api_result_path

        remote_overrides: dict[str, Any] = {}
        nested_remote = config_overrides.get("remoteConfig") or config_overrides.get("remote_config")
        if isinstance(nested_remote, dict):
            remote_overrides.update(nested_remote)
        for key in (
            "pipeline_profile",
            "operator_id",
            "dataset_name",
            "tenant_id",
            "profile_id",
            "api_compliance_base_url",
            "api_compliance_model",
            "local_compliance_base_url",
            "local_compliance_model",
            "content_safety_operator_ids",
            "content_safety_target_labels",
            "content_safety_custom_policy",
            "content_safety_custom_policy_config",
            "content_safety_metadata",
            "content_safety_training_context",
            "privacy_operator_ids",
            "privacy_target_types",
            "privacy_custom_policy",
            "privacy_custom_policy_config",
            "privacy_metadata",
            "privacy_training_context",
        ):
            if key in config_overrides and config_overrides[key] not in (None, ""):
                remote_overrides[key] = config_overrides[key]
        remote_overrides["operator_id"] = operator_id
        remote_overrides["dataset_name"] = dataset_name
        remote_overrides.setdefault("pipeline_profile", _OPERATOR_PIPELINE_PROFILE.get(operator_id, "full"))

        payload = {
            "package_paths": [str(text_input_path.resolve())],
            "config_overrides": remote_overrides,
        }

        headers = _text_api_headers(self.settings)
        submit_timeout = httpx.Timeout(self.settings.text_api_submit_timeout_seconds)
        poll_timeout = httpx.Timeout(self.settings.text_api_poll_timeout_seconds)

        with httpx.Client(headers=headers, timeout=submit_timeout) as client:
            response = client.post(submit_url, json=payload)
            response.raise_for_status()
            submit_body = response.json()
        remote_task_id = str(submit_body.get("task_id") or "").strip()
        if not remote_task_id:
            raise RuntimeError("text API bridge did not return task_id")

        deadline = time.monotonic() + max(float(self.settings.text_api_task_timeout_seconds or 0), 1.0)
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"text API bridge task {remote_task_id} did not complete within "
                    f"{self.settings.text_api_task_timeout_seconds}s"
                )
            with httpx.Client(headers=headers, timeout=poll_timeout) as client:
                response = client.get(f"{status_url}/{remote_task_id}")
                response.raise_for_status()
                status_body = response.json()

            remote_status = str(status_body.get("status") or "").strip().lower()
            if remote_status == "completed":
                with httpx.Client(headers=headers, timeout=poll_timeout) as client:
                    response = client.get(f"{result_url}/{remote_task_id}")
                    response.raise_for_status()
                    result_body = response.json()
                result_body["remote_task_id"] = remote_task_id
                return result_body
            if remote_status == "failed":
                raise RuntimeError(str(status_body.get("error") or "text API bridge task failed"))
            time.sleep(max(self.settings.text_api_poll_interval_millis, 500) / 1000.0)

    def _build_alignment_index(self, source_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for source_doc in source_documents:
            source_id = str(source_doc.get("source_id") or "")
            source_path = str(source_doc.get("source_path") or "")
            for segment in source_doc.get("segments") or []:
                rows.append({
                    "source_id": source_id,
                    "doc_id": source_id,
                    "source_path": source_path,
                    "unit_id": str(segment.get("unit_id") or ""),
                    "speaker_id": str(segment.get("speaker_id") or "speaker_0"),
                    "start_time": float(segment.get("start_time") or 0.0),
                    "end_time": float(segment.get("end_time") or 0.0),
                    "text_start": int(segment.get("text_start") or 0),
                    "text_end": int(segment.get("text_end") or 0),
                    "text": str(segment.get("text") or ""),
                    "confidence": float(segment.get("confidence") or 0.0),
                    "engine_name": str(segment.get("engine_name") or ""),
                    "language": str(segment.get("language") or ""),
                })
        return rows

    def _load_text_governance_artifacts(self, text_artifact_paths: dict[str, Path]) -> dict[str, list[dict[str, Any]]]:
        names = [
            "intake",
            "document_context",
            "content_safety",
            "content_candidate_windows",
            "content_fragment_localization",
            "content_fragment_adjudications",
            "content_document_assessments",
            "privacy",
            "redaction_plan",
            "privacy_fragment_adjudications",
            "privacy_document_assessments",
            "hard_case",
            "evidence",
            "policy",
            "annotation",
            "audit",
            "summary",
            "downstream_annotation_requests",
            "downstream_annotation_map",
            "downstream_annotation_manifest",
        ]
        return {name: _read_jsonl(text_artifact_paths.get(name)) for name in names}

    def _finding_maps(self, governance: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
        finding_by_id: dict[str, dict[str, Any]] = {}
        for record in governance.get("privacy", []):
            doc_id = str(record.get("doc_id") or "")
            for finding in record.get("findings") or []:
                if not isinstance(finding, dict):
                    continue
                finding_id = str(finding.get("finding_id") or "")
                if not finding_id:
                    continue
                finding_by_id[finding_id] = {
                    **finding,
                    "doc_id": doc_id,
                    "chain": "privacy",
                }
        for record in governance.get("content_safety", []):
            doc_id = str(record.get("doc_id") or "")
            for finding in record.get("findings") or []:
                if not isinstance(finding, dict):
                    continue
                finding_id = str(finding.get("finding_id") or "")
                if not finding_id:
                    continue
                finding_by_id[finding_id] = {
                    **finding,
                    "doc_id": doc_id,
                    "chain": "content_safety",
                }
        return finding_by_id

    def _fragment_maps(self, governance: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        privacy_by_finding = {
            str(item.get("finding_id") or ""): item
            for item in governance.get("privacy_fragment_adjudications", [])
            if str(item.get("finding_id") or "")
        }
        content_by_finding = {
            str(item.get("finding_id") or ""): item
            for item in governance.get("content_fragment_adjudications", [])
            if str(item.get("finding_id") or "")
        }
        return privacy_by_finding, content_by_finding

    def _doc_record_map(self, records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            str(record.get("doc_id") or ""): record
            for record in records
            if str(record.get("doc_id") or "")
        }

    def _redaction_target_map(self, governance: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for record in governance.get("redaction_plan", []):
            for target in record.get("redaction_targets") or []:
                finding_id = str(target.get("finding_id") or "")
                if finding_id:
                    results[finding_id] = target
        return results

    def _content_action_for(
        self,
        *,
        chain: str,
        finding_id: str,
        decision: dict[str, Any],
        redaction_targets_by_finding: dict[str, dict[str, Any]],
    ) -> str:
        disposition = str(decision.get("disposition_level") or "")
        if chain == "privacy" and finding_id in redaction_targets_by_finding:
            return "transcript_redaction"
        if disposition in {"P4", "P5"}:
            return "whole_audio_quarantine"
        if disposition == "P3":
            return "machine_suggested_manual_review"
        return "keep"

    def _workflow_action_for(self, decision: dict[str, Any]) -> str:
        disposition = str(decision.get("disposition_level") or "")
        if disposition in {"P4", "P5"}:
            return "exclude_training"
        if disposition == "P3":
            return "manual_review"
        if disposition == "P2":
            return "controlled_flow"
        return "normal_flow"

    def _build_audio_text_risk_records(
        self,
        source_documents: list[dict[str, Any]],
        governance: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        source_map = {str(item.get("source_id") or ""): item for item in source_documents}
        finding_by_id = self._finding_maps(governance)
        privacy_fragment_by_finding, content_fragment_by_finding = self._fragment_maps(governance)
        privacy_doc_by_doc = self._doc_record_map(governance.get("privacy_document_assessments", []))
        content_doc_by_doc = self._doc_record_map(governance.get("content_document_assessments", []))
        policy_by_doc = self._doc_record_map(governance.get("policy", []))
        redaction_targets_by_finding = self._redaction_target_map(governance)

        records: list[dict[str, Any]] = []
        seen_event_ids: set[str] = set()
        for event in governance.get("evidence", []):
            category = str(event.get("category") or "")
            if category not in {"privacy", "content_safety"}:
                continue
            span = event.get("primary_span") or {}
            if not isinstance(span, dict):
                span = {}
            start = span.get("start")
            end = span.get("end")
            if not isinstance(start, int) or not isinstance(end, int):
                continue

            doc_id = str(event.get("doc_id") or "")
            source_doc = source_map.get(doc_id)
            overlap = self._locate_span(source_doc, start, end)
            finding_id = str((event.get("finding_refs") or [""])[0] or "")
            finding = finding_by_id.get(finding_id, {})
            chain = "privacy" if category == "privacy" else "content_safety"
            fragment_adjudication = (
                privacy_fragment_by_finding.get(finding_id, {})
                if chain == "privacy"
                else content_fragment_by_finding.get(finding_id, {})
            )
            document_assessment = privacy_doc_by_doc.get(doc_id, {}) if chain == "privacy" else content_doc_by_doc.get(doc_id, {})
            decision = policy_by_doc.get(doc_id, {})
            event_id = str(event.get("event_id") or "")
            if event_id:
                seen_event_ids.add(event_id)
            finding_span = finding.get("span") if isinstance(finding.get("span"), dict) else {}
            records.append({
                "risk_record_id": event_id or finding_id,
                "run_id": self.run_id,
                "source_id": doc_id,
                "doc_id": doc_id,
                "risk_source": "transcript_text",
                "chain": chain,
                "finding_id": finding_id,
                "event_id": event_id,
                "risk_type": str(event.get("risk_type") or finding.get("risk_type") or ""),
                "policy_tag": str(event.get("policy_tag") or finding.get("policy_tag") or ""),
                "severity": str(event.get("severity") or finding.get("severity") or ""),
                "confidence": float(event.get("confidence_summary") or finding.get("confidence") or 0.0),
                "text_span": {
                    "start": start,
                    "end": end,
                    "text": str(span.get("text") or finding_span.get("text") or ""),
                    "context_before": str(span.get("context_before") or ""),
                    "context_after": str(span.get("context_after") or ""),
                },
                "audio_span": {
                    "start_time": overlap.get("start_time"),
                    "end_time": overlap.get("end_time"),
                    "time_label": _format_time_label(overlap.get("start_time"), overlap.get("end_time")),
                    "speaker_id": overlap.get("speaker_id", ""),
                    "unit_ids": overlap.get("unit_ids", []),
                    "mapping_status": "mapped" if overlap.get("start_time") is not None else "unmapped",
                    "mapping_precision": overlap.get("mapping_precision", "unmapped"),
                    "timestamp_granularity": overlap.get("timestamp_granularity", ""),
                    "mapping_note": overlap.get("mapping_note", ""),
                },
                "fragment_adjudication": fragment_adjudication,
                "document_assessment": document_assessment,
                "policy_decision": decision,
                "recommended_content_action": self._content_action_for(
                    chain=chain,
                    finding_id=finding_id,
                    decision=decision,
                    redaction_targets_by_finding=redaction_targets_by_finding,
                ),
                "recommended_workflow_action": self._workflow_action_for(decision),
                "training_impact": str(fragment_adjudication.get("training_impact") or document_assessment.get("training_suitability") or ""),
                "annotation_impact": str(fragment_adjudication.get("annotation_impact") or document_assessment.get("annotation_suitability") or ""),
                "explanation": str(
                    fragment_adjudication.get("explanation")
                    or event.get("explanation")
                    or finding.get("explanation")
                    or decision.get("explanation")
                    or ""
                ),
                "raw_event": event,
            })

        for decision in governance.get("policy", []):
            doc_id = str(decision.get("doc_id") or "")
            for target in decision.get("redaction_targets") or []:
                finding_id = str(target.get("finding_id") or "")
                if not finding_id:
                    continue
                if any(record.get("finding_id") == finding_id for record in records):
                    continue
                source_doc = source_map.get(doc_id)
                start = target.get("start")
                end = target.get("end")
                if not isinstance(start, int) or not isinstance(end, int):
                    continue
                overlap = self._locate_span(source_doc, start, end)
                records.append({
                    "risk_record_id": finding_id,
                    "run_id": self.run_id,
                    "source_id": doc_id,
                    "doc_id": doc_id,
                    "risk_source": "transcript_text",
                    "chain": "privacy",
                    "finding_id": finding_id,
                    "event_id": str(target.get("event_id") or ""),
                    "risk_type": str(target.get("pii_type") or "privacy"),
                    "policy_tag": "privacy.redaction_target",
                    "severity": "medium",
                    "confidence": 0.0,
                    "text_span": {"start": start, "end": end, "text": str(target.get("original_text") or "")},
                    "audio_span": {
                        "start_time": overlap.get("start_time"),
                        "end_time": overlap.get("end_time"),
                        "time_label": _format_time_label(overlap.get("start_time"), overlap.get("end_time")),
                        "speaker_id": overlap.get("speaker_id", ""),
                        "unit_ids": overlap.get("unit_ids", []),
                        "mapping_status": "mapped" if overlap.get("start_time") is not None else "unmapped",
                        "mapping_precision": overlap.get("mapping_precision", "unmapped"),
                        "timestamp_granularity": overlap.get("timestamp_granularity", ""),
                        "mapping_note": overlap.get("mapping_note", ""),
                    },
                    "fragment_adjudication": privacy_fragment_by_finding.get(finding_id, {}),
                    "document_assessment": privacy_doc_by_doc.get(doc_id, {}),
                    "policy_decision": decision,
                    "recommended_content_action": "transcript_redaction",
                    "recommended_workflow_action": self._workflow_action_for(decision),
                    "training_impact": str(privacy_doc_by_doc.get(doc_id, {}).get("training_suitability") or ""),
                    "annotation_impact": str(privacy_doc_by_doc.get(doc_id, {}).get("annotation_suitability") or ""),
                    "explanation": str(decision.get("explanation") or decision.get("summary") or ""),
                    "raw_event": {},
                })
        return records

    def _build_audio_document_assessments(
        self,
        source_documents: list[dict[str, Any]],
        governance: dict[str, list[dict[str, Any]]],
        risk_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        privacy_doc_by_doc = self._doc_record_map(governance.get("privacy_document_assessments", []))
        content_doc_by_doc = self._doc_record_map(governance.get("content_document_assessments", []))
        policy_by_doc = self._doc_record_map(governance.get("policy", []))
        records: list[dict[str, Any]] = []
        for source_doc in source_documents:
            doc_id = str(source_doc.get("source_id") or "")
            privacy_assessment = privacy_doc_by_doc.get(doc_id, {})
            content_assessment = content_doc_by_doc.get(doc_id, {})
            decision = policy_by_doc.get(doc_id, {})
            doc_risks = [item for item in risk_records if item.get("doc_id") == doc_id]
            mapped = sum(1 for item in doc_risks if item.get("audio_span", {}).get("mapping_status") == "mapped")
            records.append({
                "run_id": self.run_id,
                "source_id": doc_id,
                "doc_id": doc_id,
                "risk_source": "transcript_text",
                "overall_disposition": str(decision.get("disposition_level") or "P0"),
                "unified_decision": str(decision.get("unified_decision") or "allow"),
                "risk_score": float(decision.get("risk_score") or 0.0),
                "training_eligibility": self._training_eligibility(decision),
                "annotation_eligibility": self._annotation_eligibility(decision),
                "requires_manual_review": self._workflow_action_for(decision) == "manual_review",
                "risk_record_count": len(doc_risks),
                "mapped_risk_record_count": mapped,
                "unmapped_risk_record_count": len(doc_risks) - mapped,
                "privacy_document_assessment": privacy_assessment,
                "content_document_assessment": content_assessment,
                "policy_decision": decision,
                "explanation": str(decision.get("explanation") or decision.get("summary") or ""),
            })
        return records

    def _build_audio_policy_decisions(
        self,
        source_documents: list[dict[str, Any]],
        governance: dict[str, list[dict[str, Any]]],
        risk_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        policy_by_doc = self._doc_record_map(governance.get("policy", []))
        records: list[dict[str, Any]] = []
        for source_doc in source_documents:
            doc_id = str(source_doc.get("source_id") or "")
            decision = policy_by_doc.get(doc_id, {})
            records.append({
                "run_id": self.run_id,
                "source_id": doc_id,
                "doc_id": doc_id,
                "modality": "audio",
                "decision_source": "text_policy_decision_enriched_with_audio_timeline",
                "disposition_level": str(decision.get("disposition_level") or "P0"),
                "unified_decision": str(decision.get("unified_decision") or "allow"),
                "required_actions": list(decision.get("required_actions") or []),
                "content_actions": sorted({str(item.get("recommended_content_action") or "") for item in risk_records if item.get("doc_id") == doc_id and item.get("recommended_content_action")}),
                "workflow_action": self._workflow_action_for(decision),
                "training_eligibility": self._training_eligibility(decision),
                "annotation_eligibility": self._annotation_eligibility(decision),
                "trust_level": str(decision.get("trust_level") or "full"),
                "review_priority": str(decision.get("review_priority") or "low"),
                "reason_codes": list(decision.get("reason_codes") or []),
                "explanation": str(decision.get("explanation") or decision.get("summary") or ""),
                "text_policy_decision": decision,
            })
        return records

    def _build_audio_annotation_records(
        self,
        source_documents: list[dict[str, Any]],
        governance: dict[str, list[dict[str, Any]]],
        risk_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        annotation_by_doc = self._doc_record_map(governance.get("annotation", []))
        return [
            {
                "run_id": self.run_id,
                "source_id": str(source_doc.get("source_id") or ""),
                "doc_id": str(source_doc.get("source_id") or ""),
                "source_path": str(source_doc.get("source_path") or ""),
                "segments": source_doc.get("segments") or [],
                "risk_records": [item for item in risk_records if item.get("doc_id") == str(source_doc.get("source_id") or "")],
                "text_annotation_record": annotation_by_doc.get(str(source_doc.get("source_id") or ""), {}),
            }
            for source_doc in source_documents
        ]

    def _build_audio_audit_records(
        self,
        source_documents: list[dict[str, Any]],
        governance: dict[str, list[dict[str, Any]]],
        risk_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        audit_by_doc = self._doc_record_map(governance.get("audit", []))
        return [
            {
                "run_id": self.run_id,
                "source_id": str(source_doc.get("source_id") or ""),
                "doc_id": str(source_doc.get("source_id") or ""),
                "source_path": str(source_doc.get("source_path") or ""),
                "risk_record_count": sum(1 for item in risk_records if item.get("doc_id") == str(source_doc.get("source_id") or "")),
                "alignment_segments": source_doc.get("segments") or [],
                "text_audit_record": audit_by_doc.get(str(source_doc.get("source_id") or ""), {}),
            }
            for source_doc in source_documents
        ]

    def _training_eligibility(self, decision: dict[str, Any]) -> str:
        disposition = str(decision.get("disposition_level") or "")
        if disposition in {"P4", "P5"}:
            return "exclude"
        if disposition in {"P2", "P3"}:
            return "restricted"
        return "allow"

    def _annotation_eligibility(self, decision: dict[str, Any]) -> str:
        disposition = str(decision.get("disposition_level") or "")
        if disposition in {"P4", "P5"}:
            return "block"
        if disposition in {"P2", "P3"}:
            return "review"
        return "allow"

    def _risk_level_from_governance(
        self,
        disposition: str,
        decision: str,
        fallback_findings: list[dict[str, Any]],
    ) -> str:
        disposition = str(disposition or "")
        decision = str(decision or "").lower()
        if disposition in {"P4", "P5"} or decision == "reject":
            return "critical"
        if disposition == "P3" or decision == "review":
            return "high"
        if disposition == "P2" or decision == "quarantine":
            return "medium"
        if disposition == "P1":
            return "low"
        return _risk_level(fallback_findings)

    def _conclusion_from_governance(
        self,
        disposition: str,
        decision: str,
        risk_level: str,
        total_findings: int,
    ) -> str:
        disposition = str(disposition or "")
        decision = str(decision or "").lower()
        if disposition in {"P4", "P5"} or decision == "reject":
            return "failed"
        if disposition in {"P2", "P3"} or decision in {"review", "quarantine"}:
            return "review"
        return _conclusion(risk_level, total_findings)

    def _overall_text_decision(self, text_api_result: dict[str, Any], governance: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        summary = governance.get("summary", [])
        if summary:
            record = summary[0]
            return {
                "overall_disposition": str(record.get("overall_disposition") or ""),
                "unified_decision": str(record.get("unified_decision") or ""),
                "trust_level": str(record.get("trust_level") or ""),
                "explanation_summary": str(record.get("explanation_summary") or ""),
                "counts_by_disposition": record.get("counts_by_disposition") or {},
                "counts_by_decision": record.get("counts_by_decision") or {},
            }
        legacy = text_api_result.get("legacy_decision") or {}
        return {
            "overall_disposition": str(legacy.get("overall_disposition") or ""),
            "unified_decision": str(text_api_result.get("decision") or legacy.get("overall_decision") or ""),
            "trust_level": str(text_api_result.get("trust_level") or ""),
            "explanation_summary": str(text_api_result.get("explanation_summary") or ""),
            "counts_by_disposition": legacy.get("counts_by_disposition") or {},
            "counts_by_decision": legacy.get("counts_by_decision") or {},
        }

    def _build_audio_summary(
        self,
        *,
        source_documents: list[dict[str, Any]],
        text_api_result: dict[str, Any],
        text_governance: dict[str, list[dict[str, Any]]],
        risk_records: list[dict[str, Any]],
        artifact_paths: dict[str, Path],
    ) -> dict[str, Any]:
        text_decision = self._overall_text_decision(text_api_result, text_governance)
        mapped = sum(1 for item in risk_records if item.get("audio_span", {}).get("mapping_status") == "mapped")
        unmapped = len(risk_records) - mapped
        policy_decisions = text_governance.get("policy", [])
        worst_disposition = max(
            (str(item.get("disposition_level") or "P0") for item in policy_decisions),
            key=lambda value: _DISPOSITION_RANK.get(value, 0),
            default=str(text_decision.get("overall_disposition") or "P0"),
        )
        worst_decision = max(
            (str(item.get("unified_decision") or "allow") for item in policy_decisions),
            key=lambda value: _DECISION_RANK.get(value, 0),
            default=str(text_decision.get("unified_decision") or "allow"),
        )
        explanation = text_decision.get("explanation_summary") or (
            f"Processed {len(source_documents)} audio source(s) through Qwen3-ASR and text compliance. "
            f"{len(risk_records)} transcript text risk record(s) were found."
        )
        return {
            "run_id": self.run_id,
            "modality": "audio",
            "decision_source": "text_governance_enriched_with_audio_timeline",
            "processed_sources": len(source_documents),
            "processed_transcript_units": sum(len(item.get("segments") or []) for item in source_documents),
            "overall_disposition": text_decision.get("overall_disposition") or worst_disposition,
            "unified_decision": text_decision.get("unified_decision") or worst_decision,
            "trust_level": text_decision.get("trust_level") or "full",
            "training_eligibility": "exclude" if worst_disposition in {"P4", "P5"} else "restricted" if worst_disposition in {"P2", "P3"} else "allow",
            "annotation_eligibility": "block" if worst_disposition in {"P4", "P5"} else "review" if worst_disposition in {"P2", "P3"} else "allow",
            "risk_record_count": len(risk_records),
            "mapped_risk_record_count": mapped,
            "unmapped_risk_record_count": unmapped,
            "counts_by_disposition": text_decision.get("counts_by_disposition") or {},
            "counts_by_decision": text_decision.get("counts_by_decision") or {},
            "explanation_summary": explanation,
            "artifact_paths": {name: str(path) for name, path in artifact_paths.items()},
        }

    def _build_audio_redaction_spans(
        self,
        source_documents: list[dict[str, Any]],
        redaction_records: list[dict[str, Any]],
    ) -> list[RedactionSpan]:
        source_map = {str(item.get("source_id") or ""): item for item in source_documents}
        spans: list[RedactionSpan] = []
        padding_seconds = max(float(getattr(self.settings, "audio_redaction_padding_ms", 0) or 0), 0.0) / 1000.0

        for record in redaction_records:
            source_id = str(record.get("doc_id") or "")
            source_doc = source_map.get(source_id)
            if source_doc is None:
                continue
            duration = self._source_duration(source_doc)
            for target in record.get("redaction_targets") or []:
                if not isinstance(target, dict):
                    continue
                start = target.get("start")
                end = target.get("end")
                if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                    continue
                overlap = self._locate_span(source_doc, start, end)
                start_time = overlap.get("start_time")
                end_time = overlap.get("end_time")
                if start_time is None or end_time is None:
                    continue
                padded_start = max(float(start_time) - padding_seconds, 0.0)
                padded_end = min(float(end_time) + padding_seconds, duration) if duration > 0 else float(end_time) + padding_seconds
                if padded_end <= padded_start:
                    continue
                unit_ids = [str(item) for item in overlap.get("unit_ids") or [] if str(item)]
                spans.append(
                    RedactionSpan(
                        source_id=source_id,
                        unit_id=unit_ids[0] if unit_ids else str(target.get("finding_id") or ""),
                        start_time=padded_start,
                        end_time=padded_end,
                        entity_type=str(target.get("pii_type") or target.get("risk_type") or "pii"),
                        original_text=str(target.get("original_text") or ""),
                        replacement=str(target.get("replacement") or "<REDACTED>"),
                        metadata={
                            "finding_id": str(target.get("finding_id") or ""),
                            "mapping_precision": overlap.get("mapping_precision", "unmapped"),
                            "timestamp_granularity": overlap.get("timestamp_granularity", ""),
                            "mapping_note": overlap.get("mapping_note", ""),
                        },
                    )
                )

        return self._merge_audio_redaction_spans(spans)

    def _merge_audio_redaction_spans(self, spans: list[RedactionSpan]) -> list[RedactionSpan]:
        merged: list[RedactionSpan] = []
        for span in sorted(spans, key=lambda item: (item.source_id, item.start_time, item.end_time)):
            if not merged or merged[-1].source_id != span.source_id or span.start_time > merged[-1].end_time:
                merged.append(span)
                continue
            previous = merged[-1]
            previous.end_time = max(previous.end_time, span.end_time)
            previous.entity_type = ",".join(sorted({*previous.entity_type.split(","), span.entity_type}))
            if span.original_text and span.original_text not in previous.original_text:
                previous.original_text = " ".join(part for part in (previous.original_text, span.original_text) if part)
            previous_precision = str((previous.metadata or {}).get("mapping_precision") or "")
            span_precision = str((span.metadata or {}).get("mapping_precision") or "")
            if "coarse" in {previous_precision, span_precision}:
                previous.metadata["mapping_precision"] = "coarse"
            previous_granularity = str((previous.metadata or {}).get("timestamp_granularity") or "")
            span_granularity = str((span.metadata or {}).get("timestamp_granularity") or "")
            granularities = {item for item in (previous_granularity, span_granularity) if item}
            if granularities:
                previous.metadata["timestamp_granularity"] = ",".join(sorted(granularities))
        return merged

    def _source_duration(self, source_doc: dict[str, Any]) -> float:
        return max(
            (float(segment.get("end_time") or 0.0) for segment in source_doc.get("segments") or []),
            default=0.0,
        )

    def _render_redacted_audio(
        self,
        normalized: list[NormalizedAudioRecord],
        redaction_spans: list[RedactionSpan],
        paths: dict[str, Path],
    ) -> list[Any]:
        if not redaction_spans or not self.settings.audio_redaction_enabled:
            return []
        renderable_spans = [
            span for span in redaction_spans
            if str((span.metadata or {}).get("mapping_precision") or "").lower() != "coarse"
        ]
        skipped_spans = [span for span in redaction_spans if span not in renderable_spans]
        if skipped_spans:
            skipped_manifest = paths["redacted_audio"].with_suffix(".skipped.json")
            _write_json(
                {
                    "reason": "coarse_asr_timestamps",
                    "message": "Audio redaction rendering was skipped for spans mapped from whole-audio ASR timestamps.",
                    "skipped_span_count": len(skipped_spans),
                    "renderable_span_count": len(renderable_spans),
                    "skipped_spans": [span.model_dump(mode="json") for span in skipped_spans],
                },
                skipped_manifest,
            )
        if not renderable_spans:
            return []
        try:
            from audio.legacy_steps.k_audio_redaction import run as render_audio_redaction

            return render_audio_redaction(normalized, renderable_spans, self.settings, self.output_dir)
        except Exception as exc:
            logger.warning("Audio redaction rendering failed: %s", exc)
            failure_manifest = paths["redacted_audio"].with_suffix(".failure.json")
            _write_json({"error": str(exc), "span_count": len(redaction_spans)}, failure_manifest)
            return []

    def _build_source_documents(
        self,
        transcript_units: list[TranscriptUnit],
        sources: list[Any],
    ) -> list[dict[str, Any]]:
        source_paths: dict[str, str] = {}
        for source in sources:
            source_paths[getattr(source, "source_id", "")] = getattr(source, "path", "")

        units_by_source: dict[str, list[TranscriptUnit]] = defaultdict(list)
        for unit in transcript_units:
            units_by_source[unit.source_id].append(unit)

        documents: list[dict[str, Any]] = []
        for source_id, source_units in sorted(units_by_source.items()):
            sorted_units = sorted(source_units, key=lambda item: (item.start_time, item.end_time, item.unit_id))
            parts: list[str] = []
            segments: list[dict[str, Any]] = []
            cursor = 0
            for index, unit in enumerate(sorted_units):
                text = unit.text or ""
                start_offset = cursor
                end_offset = start_offset + len(text)
                segments.append({
                    "unit_id": unit.unit_id,
                    "speaker_id": unit.speaker_id,
                    "start_time": unit.start_time,
                    "end_time": unit.end_time,
                    "confidence": unit.confidence,
                    "engine_name": unit.engine_name,
                    "language": unit.language,
                    "metadata": unit.metadata,
                    "text_start": start_offset,
                    "text_end": end_offset,
                    "text": text,
                })
                parts.append(text)
                cursor = end_offset
                if index < len(sorted_units) - 1:
                    parts.append("\n")
                    cursor += 1
            documents.append({
                "source_id": source_id,
                "source_path": source_paths.get(source_id, ""),
                "text": "".join(parts),
                "original_text": "".join(parts),
                "segments": segments,
                "asr_quality": self._source_asr_quality(segments, "".join(parts)),
            })
        return documents

    def _source_asr_quality(self, segments: list[dict[str, Any]], text: str) -> dict[str, Any]:
        duration = max((float(segment.get("end_time") or 0.0) for segment in segments), default=0.0)
        confidences = [
            float(segment.get("confidence") or 0.0)
            for segment in segments
            if segment.get("confidence") not in (None, "")
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        granularities = {
            str(((segment.get("metadata") or {}) if isinstance(segment.get("metadata"), dict) else {}).get("timestamp_granularity") or "")
            for segment in segments
        }
        granularities.discard("")
        warnings: list[dict[str, Any]] = []
        text_len = len(str(text or "").strip())
        minutes = max(duration / 60.0, 1.0 / 60.0)

        if not segments or text_len == 0:
            warnings.append({
                "code": "asr_empty_transcript",
                "level": "critical",
                "message": "ASR 未生成可用于合规检测的转写文本。",
            })
        if len(segments) == 1 and duration >= _ASR_LONG_SINGLE_SEGMENT_SECONDS:
            warnings.append({
                "code": "asr_single_long_segment",
                "level": "medium",
                "message": "ASR 仅返回一个长片段，风险片段的时间定位只能按文本比例估算。",
            })
        if confidences and avg_confidence < _ASR_LOW_CONFIDENCE_THRESHOLD:
            warnings.append({
                "code": "asr_low_confidence",
                "level": "high",
                "message": "ASR 平均置信度偏低，建议重新转写或人工复核。",
            })
        if duration >= 30.0 and text_len / minutes < _ASR_MIN_TEXT_CHARS_PER_MINUTE:
            warnings.append({
                "code": "asr_sparse_transcript",
                "level": "medium",
                "message": "音频时长与转写文本长度不匹配，可能存在漏转写。",
            })
        if granularities & {"whole_audio", "whole_audio_reference"}:
            warnings.append({
                "code": "asr_whole_audio_timestamp",
                "level": "medium",
                "message": "ASR 时间戳为整段级，风险时间点需要按比例估算。",
            })

        return {
            "segment_count": len(segments),
            "duration_seconds": duration,
            "text_length": text_len,
            "avg_confidence": round(avg_confidence, 4),
            "timestamp_granularity": ",".join(sorted(granularities)) if granularities else "segment",
            "warning_count": len(warnings),
            "warnings": warnings,
            "requires_review": any(item["level"] in {"high", "critical"} for item in warnings),
            "is_degraded": bool(warnings),
        }

    def _speech_privacy_fallback_findings(
        self,
        *,
        source_documents: list[dict[str, Any]],
        existing_findings: list[dict[str, Any]],
        config_overrides: dict[str, Any],
    ) -> list[dict[str, Any]]:
        enabled_types = self._enabled_privacy_types(config_overrides)
        results: list[dict[str, Any]] = []

        def enabled(risk_type: str) -> bool:
            if not enabled_types:
                return True
            aliases = {
                "phone_number": {"phone", "phone_number", "parent_contact"},
                "bank_card": {"bank_card", "bank_account", "payment_account"},
                "password": {"password", "secret", "api_key", "token"},
            }.get(risk_type, {risk_type})
            return bool(enabled_types & aliases)

        number = rf"[{_SPOKEN_NUMBER_CHARS}]"
        detectors = [
            ("id_card", "pii.id_card", "high", rf"(?:身份证号|身份证)[^。；\n]{{0,18}}?(?P<value>{number}{{15,40}})", 15),
            ("phone_number", "pii.phone_number", "high", rf"(?:备用手机号|手机号|联系电话|联系方式|家长电话|电话)[^。；\n]{{0,12}}?(?P<value>{number}{{11,24}})", 11),
            ("bank_card", "pii.bank_card", "high", rf"(?:银行卡号|银行卡|银行账号|卡号)[^。；\n]{{0,12}}?(?P<value>{number}{{12,32}})", 12),
            ("student_id", "pii.student_id", "medium", rf"(?:学号)[^。；\n]{{0,12}}?(?P<value>[A-Za-z]?{number}{{5,30}})", 5),
        ]

        all_existing = [*existing_findings]
        for source_doc in source_documents:
            text = str(source_doc.get("text") or source_doc.get("original_text") or "")
            source_id = str(source_doc.get("source_id") or source_doc.get("doc_id") or "")
            if not text or not source_id:
                continue

            for risk_type, policy_tag, severity, pattern, min_digits in detectors:
                if not enabled(risk_type):
                    continue
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    value = match.group("value").strip()
                    digits = self._spoken_digits(value)
                    if len(digits) < min_digits:
                        continue
                    item = self._speech_privacy_item(
                        source_doc=source_doc,
                        source_id=source_id,
                        risk_type=risk_type,
                        policy_tag=policy_tag,
                        severity=severity,
                        confidence=0.91 if severity == "high" else 0.82,
                        start=match.start("value"),
                        end=match.end("value"),
                        text=value,
                        normalized_value=digits,
                    )
                    if self._is_new_audio_finding(item, all_existing):
                        results.append(item)
                        all_existing.append(item)

            if enabled("address"):
                for match in re.finditer(r"(?:家庭地址|住址|地址)(?:还是|是|为|：|:)?(?P<value>[^，。；\n]{6,60})", text):
                    value = match.group("value").strip()
                    if not any(token in value for token in ("省", "市", "区", "县", "路", "街", "号", "栋", "室")):
                        continue
                    item = self._speech_privacy_item(
                        source_doc=source_doc,
                        source_id=source_id,
                        risk_type="address",
                        policy_tag="pii.address",
                        severity="high",
                        confidence=0.88,
                        start=match.start("value"),
                        end=match.end("value"),
                        text=value,
                    )
                    if self._is_new_audio_finding(item, all_existing):
                        results.append(item)
                        all_existing.append(item)

            if enabled("email"):
                email_pattern = r"(?P<value>(?:[A-Za-z]\s*){2,}(?:点|\.)?(?:[A-Za-z]\s*){0,20}\s*(?:at|@)\s*(?:[A-Za-z]\s*){2,}(?:点|\.)(?:[A-Za-z]\s*){2,})"
                for match in re.finditer(email_pattern, text, flags=re.IGNORECASE):
                    value = match.group("value").strip()
                    item = self._speech_privacy_item(
                        source_doc=source_doc,
                        source_id=source_id,
                        risk_type="email",
                        policy_tag="pii.email",
                        severity="medium",
                        confidence=0.84,
                        start=match.start("value"),
                        end=match.end("value"),
                        text=value,
                    )
                    if self._is_new_audio_finding(item, all_existing):
                        results.append(item)
                        all_existing.append(item)

            if enabled("password"):
                for match in re.finditer(r"(?:临时系统口令|系统口令|口令|密码|密钥|token|api[_ -]?key)[是为：:\\s]*(?P<value>[^，。；\n]{4,36})", text, flags=re.IGNORECASE):
                    value = match.group("value").strip()
                    if not re.search(r"[A-Za-z0-9零〇○洞幺一二两三四五六七八九#井号]", value):
                        continue
                    item = self._speech_privacy_item(
                        source_doc=source_doc,
                        source_id=source_id,
                        risk_type="password",
                        policy_tag="pii.password",
                        severity="high",
                        confidence=0.9,
                        start=match.start("value"),
                        end=match.end("value"),
                        text=value,
                    )
                    if self._is_new_audio_finding(item, all_existing):
                        results.append(item)
                        all_existing.append(item)

        return results

    def _enabled_privacy_types(self, config_overrides: dict[str, Any]) -> set[str]:
        privacy = config_overrides.get("privacyCompliance") or config_overrides.get("privacy_compliance") or {}
        if not isinstance(privacy, dict):
            return set()
        raw = privacy.get("target_types") or privacy.get("targetTypes") or []
        return {str(item).strip() for item in raw if str(item).strip()} if isinstance(raw, list) else set()

    def _spoken_digits(self, value: str) -> str:
        digits: list[str] = []
        for char in str(value or ""):
            if char.isdigit():
                digits.append(char)
            elif char in _SPOKEN_DIGIT_MAP:
                digits.append(_SPOKEN_DIGIT_MAP[char])
        return "".join(digits)

    def _speech_privacy_item(
        self,
        *,
        source_doc: dict[str, Any],
        source_id: str,
        risk_type: str,
        policy_tag: str,
        severity: str,
        confidence: float,
        start: int,
        end: int,
        text: str,
        normalized_value: str = "",
    ) -> dict[str, Any]:
        overlap = self._locate_span(source_doc, start, end)
        seed = f"{source_id}:{risk_type}:{start}:{end}:{text}"
        finding_id = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
        label = _risk_label_zh(policy_tag, risk_type)
        return {
            "finding_id": finding_id,
            "doc_id": source_id,
            "source_id": source_id,
            "source_path": source_doc.get("source_path", ""),
            "type": risk_type,
            "risk_type": risk_type,
            "policy_tag": policy_tag,
            "risk_level": severity,
            "severity": severity,
            "confidence": confidence,
            "text": text,
            "start": start,
            "end": end,
            "start_time": overlap.get("start_time"),
            "end_time": overlap.get("end_time"),
            "time_label": _format_time_label(overlap.get("start_time"), overlap.get("end_time")),
            "speaker_id": overlap.get("speaker_id", ""),
            "unit_ids": overlap.get("unit_ids", []),
            "replacement": f"<{risk_type.upper()}>",
            "suggestion": "manual_review",
            "source_tool": "audio_speech_privacy_fallback",
            "explanation": f"语音转写中检测到疑似{label}，该类信息常以中文读数或口述形式出现，需要纳入音频合规证据。",
            "context_before": str(source_doc.get("text", ""))[max(0, start - 20):start],
            "context_after": str(source_doc.get("text", ""))[end:end + 20],
            "speech_normalized_value": normalized_value,
        }

    def _is_new_audio_finding(self, item: dict[str, Any], existing_findings: list[dict[str, Any]]) -> bool:
        start = item.get("start")
        end = item.get("end")
        risk_type = str(item.get("risk_type") or "")
        if not isinstance(start, int) or not isinstance(end, int):
            return True
        for existing in existing_findings:
            if str(existing.get("doc_id") or existing.get("source_id") or "") != str(item.get("doc_id") or ""):
                continue
            ex_start = existing.get("start")
            ex_end = existing.get("end")
            if not isinstance(ex_start, int) or not isinstance(ex_end, int):
                continue
            overlaps = start < ex_end and ex_start < end
            same_type = risk_type == str(existing.get("risk_type") or "")
            if overlaps and same_type:
                return False
        return True

    def _build_asr_quality_summary(self, source_documents: list[dict[str, Any]]) -> dict[str, Any]:
        sources: list[dict[str, Any]] = []
        warning_count = 0
        requires_review = False
        for source_doc in source_documents:
            quality = dict(source_doc.get("asr_quality") or {})
            warnings = [item for item in quality.get("warnings") or [] if isinstance(item, dict)]
            warning_count += len(warnings)
            requires_review = requires_review or bool(quality.get("requires_review"))
            sources.append({
                "source_id": str(source_doc.get("source_id") or ""),
                "source_path": str(source_doc.get("source_path") or ""),
                **quality,
                "warnings": warnings,
            })
        return {
            "source_count": len(source_documents),
            "warning_count": warning_count,
            "requires_review": requires_review,
            "is_degraded": warning_count > 0,
            "sources": sources,
        }

    def _build_audio_report(
        self,
        *,
        operator_id: str,
        dataset_name: str,
        source_documents: list[dict[str, Any]],
        text_api_result: dict[str, Any],
        local_artifacts: dict[str, Path],
        config_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        text_artifact_paths = _artifact_paths(text_api_result.get("metadata") or {})
        intake_records = _read_jsonl(text_artifact_paths.get("intake"))
        privacy_records = _read_jsonl(text_artifact_paths.get("privacy"))
        safety_records = _read_jsonl(text_artifact_paths.get("content_safety"))
        redaction_records = _read_jsonl(text_artifact_paths.get("redaction_plan"))
        annotation_records = _read_jsonl(text_artifact_paths.get("annotation"))
        policy_records = _read_jsonl(text_artifact_paths.get("policy"))
        summary_records = _read_jsonl(text_artifact_paths.get("summary"))
        audio_text_risk_records = _read_jsonl(local_artifacts.get("audio_text_risk_records"))
        audio_document_assessments = _read_jsonl(local_artifacts.get("audio_document_assessments"))
        audio_policy_decisions = _read_jsonl(local_artifacts.get("audio_policy_decisions"))
        audio_annotation_records = _read_jsonl(local_artifacts.get("audio_annotation"))
        audio_audit_records = _read_jsonl(local_artifacts.get("audio_audit"))
        audio_summary = _read_json(local_artifacts.get("audio_summary"))
        audio_redaction_spans = _read_jsonl(local_artifacts.get("audio_redaction_spans"))
        redacted_audio_records = _read_jsonl(local_artifacts.get("redacted_audio"))

        source_map = {item["source_id"]: item for item in source_documents}
        redaction_by_finding = self._redaction_targets_by_finding(redaction_records)
        asr_quality = self._build_asr_quality_summary(source_documents)

        privacy_view = self._build_privacy_view(source_map, privacy_records, redaction_records, redaction_by_finding)
        content_view = self._build_content_view(source_map, safety_records)
        speech_privacy_fallback = self._speech_privacy_fallback_findings(
            source_documents=source_documents,
            existing_findings=[*privacy_view["findings"], *privacy_view["supplemental_findings"]],
            config_overrides=config_overrides or {},
        )
        if speech_privacy_fallback:
            privacy_view["findings"].extend(speech_privacy_fallback)
            privacy_view["summary"]["privacy_hits"] = privacy_view["summary"].get("privacy_hits", 0) + len(speech_privacy_fallback)
            for item in speech_privacy_fallback:
                risk_type = str(item.get("risk_type") or "")
                if risk_type:
                    privacy_view["summary"][risk_type] = privacy_view["summary"].get(risk_type, 0) + 1
            _write_jsonl(speech_privacy_fallback, local_artifacts["speech_privacy_fallback"])

        if operator_id == "CMP_001":
            findings = privacy_view["findings"]
            supplemental_findings = privacy_view["supplemental_findings"]
            summary = privacy_view["summary"]
            redaction_views = privacy_view["redaction_views"]
            raw_artifacts = {
                "transcript_sources": source_documents,
                "asr_quality": asr_quality,
                "privacy_detection": privacy_records,
                "redaction_plan": redaction_records,
                "audio_redaction_spans": audio_redaction_spans,
                "redacted_audio": redacted_audio_records,
                "audio_text_risk_records": audio_text_risk_records,
                "speech_privacy_fallback": speech_privacy_fallback,
                "audio_policy_decisions": audio_policy_decisions,
                "annotation_package": annotation_records,
                "text_api_result": text_api_result,
            }
        elif operator_id == "CMP_002":
            findings = content_view["findings"]
            supplemental_findings = content_view["supplemental_findings"]
            summary = content_view["summary"]
            redaction_views = []
            raw_artifacts = {
                "transcript_sources": source_documents,
                "asr_quality": asr_quality,
                "content_safety": safety_records,
                "audio_text_risk_records": audio_text_risk_records,
                "audio_policy_decisions": audio_policy_decisions,
                "text_api_result": text_api_result,
            }
        else:
            findings = [*privacy_view["findings"], *content_view["findings"]]
            supplemental_findings = [*privacy_view["supplemental_findings"], *content_view["supplemental_findings"]]
            summary = {
                "processed_sources": len(source_documents),
                "processed_units": sum(len(item["segments"]) for item in source_documents),
                "privacy_hits": len(privacy_view["findings"]) + len(privacy_view["supplemental_findings"]),
                "safety_hits": len(content_view["findings"]) + len(content_view["supplemental_findings"]),
                "redacted_sources": len(privacy_view["redaction_views"]),
            }
            redaction_views = privacy_view["redaction_views"]
            raw_artifacts = {
                "transcript_sources": source_documents,
                "asr_quality": asr_quality,
                "privacy_detection": privacy_records,
                "redaction_plan": redaction_records,
                "audio_redaction_spans": audio_redaction_spans,
                "redacted_audio": redacted_audio_records,
                "audio_text_risk_records": audio_text_risk_records,
                "speech_privacy_fallback": speech_privacy_fallback,
                "audio_policy_decisions": audio_policy_decisions,
                "annotation_package": annotation_records,
                "content_safety": safety_records,
                "text_api_result": text_api_result,
            }

        all_findings = [*findings, *supplemental_findings]
        transcript_views = self._build_transcript_views(source_documents, all_findings)
        text_summary = summary_records[0] if summary_records else {}
        final_decision = str(audio_summary.get("unified_decision") or text_api_result.get("decision") or text_summary.get("unified_decision") or "")
        final_disposition = str(audio_summary.get("overall_disposition") or text_summary.get("overall_disposition") or "")
        normalized_decision = final_decision or "allow"
        normalized_disposition = final_disposition or "P0"
        trust_level = str(audio_summary.get("trust_level") or text_api_result.get("trust_level") or text_summary.get("trust_level") or "full")
        if asr_quality.get("is_degraded") and trust_level == "full":
            trust_level = "degraded"
        risk_level = self._risk_level_from_governance(normalized_disposition, normalized_decision, all_findings)
        conclusion = self._conclusion_from_governance(normalized_disposition, normalized_decision, risk_level, len(all_findings))
        if asr_quality.get("requires_review") and conclusion == "passed":
            conclusion = "review"
            risk_level = "low" if risk_level == "none" else risk_level
        review_suggestions = list(text_api_result.get("review_suggestions") or [])
        review_suggestions.extend(
            f"{item.get('source_id')}: {warning.get('message')}"
            for item in asr_quality.get("sources", [])
            for warning in item.get("warnings", [])
            if warning.get("level") in {"high", "critical"}
        )
        report = {
            "job_id": self.run_id,
            "remote_task_id": str(text_api_result.get("remote_task_id") or text_api_result.get("task_id") or ""),
            "operator_id": operator_id,
            "operator_name": _OPERATOR_NAMES.get(operator_id, operator_id),
            "dataset_name": dataset_name,
            "modality": "audio",
            "execution_route": "api_audio_bridge",
            "execution_engine": "qwen3_asr_plus_text_api",
            "source_service": "audio.server",
            "transcript_view_contract": {
                "display_mode": "single_full_text",
                "text_field": "transcript_views[].text",
                "highlight_field": "transcript_views[].highlights",
                "audio_link_fields": ["start_time", "end_time", "time_label"],
                "interaction_targets": ["asr_text_highlight", "evidence_card", "audio_player"],
            },
            "conclusion": conclusion,
            "risk_level": risk_level,
            "is_compliant": (
                normalized_decision == "allow"
                and normalized_disposition in {"P0", "P1"}
                and not asr_quality.get("requires_review")
            ),
            "decision": normalized_decision,
            "overall_disposition": normalized_disposition,
            "trust_level": trust_level,
            "training_eligibility": audio_summary.get("training_eligibility") or self._training_eligibility(policy_records[0] if policy_records else {}),
            "annotation_eligibility": audio_summary.get("annotation_eligibility") or self._annotation_eligibility(policy_records[0] if policy_records else {}),
            "total_documents": len(source_documents) or len(intake_records),
            "total_findings": len(all_findings),
            "total_risk_records": len(audio_text_risk_records),
            "visible_findings_count": len(findings),
            "supplemental_findings_count": len(supplemental_findings),
            "summary": {**summary, "audio_governance": audio_summary, "asr_quality": asr_quality},
            "findings": findings,
            "supplemental_findings": supplemental_findings,
            "risk_records": audio_text_risk_records,
            "audio_document_assessments": audio_document_assessments,
            "audio_policy_decisions": audio_policy_decisions,
            "audio_annotation_package": audio_annotation_records,
            "audio_audit_package": audio_audit_records,
            "transcript_views": transcript_views,
            "redaction_views": redaction_views,
            "audio_redaction_spans": audio_redaction_spans,
            "redacted_audio": redacted_audio_records,
            "review_suggestions": review_suggestions,
            "artifact_paths": {
                "audio": {name: str(path) for name, path in local_artifacts.items()},
                "text_api": {name: str(path) for name, path in text_artifact_paths.items()},
            },
            "raw_artifacts": raw_artifacts,
        }
        return report

    def _build_privacy_view(
        self,
        source_map: dict[str, dict[str, Any]],
        privacy_records: list[dict[str, Any]],
        redaction_records: list[dict[str, Any]],
        redaction_by_finding: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        summary: dict[str, int] = {
            "processed_sources": len(source_map),
            "processed_units": sum(len(item["segments"]) for item in source_map.values()),
            "privacy_hits": 0,
            "redacted_sources": 0,
        }
        findings: list[dict[str, Any]] = []
        supplemental_findings: list[dict[str, Any]] = []

        for record in privacy_records:
            source_id = str(record.get("doc_id") or "")
            source_doc = source_map.get(source_id)
            for finding in record.get("findings") or []:
                finding_id = str(finding.get("finding_id") or "")
                if redaction_by_finding and isinstance(finding.get("span"), dict) and finding_id not in redaction_by_finding:
                    continue
                item = self._map_finding(
                    finding=finding,
                    source_doc=source_doc,
                    source_id=source_id,
                    replacement=_replacement_for(finding, redaction_by_finding),
                )
                risk_type = str(item.get("risk_type") or "")
                if risk_type:
                    summary[risk_type] = summary.get(risk_type, 0) + 1
                summary["privacy_hits"] += 1
                if self._is_supplemental(item):
                    supplemental_findings.append(item)
                else:
                    findings.append(item)

        redaction_views = self._build_redaction_views(source_map, redaction_records)
        summary["redacted_sources"] = len(redaction_views)
        return {
            "summary": summary,
            "findings": findings,
            "supplemental_findings": supplemental_findings,
            "redaction_views": redaction_views,
        }

    def _build_content_view(
        self,
        source_map: dict[str, dict[str, Any]],
        safety_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        summary: dict[str, int] = {
            "processed_sources": len(source_map),
            "processed_units": sum(len(item["segments"]) for item in source_map.values()),
            "safe": 0,
            "controversial": 0,
            "unsafe": 0,
            "safety_hits": 0,
        }
        findings: list[dict[str, Any]] = []
        supplemental_findings: list[dict[str, Any]] = []

        for record in safety_records:
            source_id = str(record.get("doc_id") or "")
            source_doc = source_map.get(source_id)
            status = str(record.get("status") or "clear")
            if status == "hard_case":
                summary["controversial"] += 1
            elif status == "flagged":
                summary["unsafe"] += 1
            else:
                summary["safe"] += 1

            for finding in record.get("findings") or []:
                item = self._map_finding(
                    finding=finding,
                    source_doc=source_doc,
                    source_id=source_id,
                    replacement="",
                )
                item["category"] = status
                risk_type = str(item.get("risk_type") or "")
                if risk_type:
                    summary[risk_type] = summary.get(risk_type, 0) + 1
                summary["safety_hits"] += 1
                if self._is_supplemental(item):
                    supplemental_findings.append(item)
                else:
                    findings.append(item)

        return {
            "summary": summary,
            "findings": findings,
            "supplemental_findings": supplemental_findings,
        }

    def _map_finding(
        self,
        *,
        finding: dict[str, Any],
        source_doc: dict[str, Any] | None,
        source_id: str,
        replacement: str,
    ) -> dict[str, Any]:
        span = finding.get("span") or {}
        start = span.get("start")
        end = span.get("end")
        overlap = self._locate_span(source_doc, start, end)
        item = {
            "finding_id": str(finding.get("finding_id") or ""),
            "doc_id": source_id,
            "source_id": source_id,
            "source_path": "" if source_doc is None else source_doc.get("source_path", ""),
            "type": str(finding.get("risk_type") or ""),
            "risk_type": str(finding.get("risk_type") or ""),
            "policy_tag": str(finding.get("policy_tag") or ""),
            "risk_level": str(finding.get("severity") or "medium"),
            "confidence": float(finding.get("confidence") or 0.0),
            "text": str(span.get("text") or ""),
            "start": start if isinstance(start, int) else None,
            "end": end if isinstance(end, int) else None,
            "start_time": overlap.get("start_time"),
            "end_time": overlap.get("end_time"),
            "time_label": _format_time_label(overlap.get("start_time"), overlap.get("end_time")),
            "speaker_id": overlap.get("speaker_id", ""),
            "unit_ids": overlap.get("unit_ids", []),
            "replacement": replacement,
            "suggestion": str(finding.get("remediation_suggestion") or ""),
            "source_tool": str(finding.get("source_tool") or ""),
            "explanation": str(finding.get("explanation") or ""),
            "context_before": str(span.get("context_before") or ""),
            "context_after": str(span.get("context_after") or ""),
        }
        return item

    def _locate_span(
        self,
        source_doc: dict[str, Any] | None,
        start: Any,
        end: Any,
    ) -> dict[str, Any]:
        if source_doc is None or not isinstance(start, int) or not isinstance(end, int):
            return {
                "start_time": None,
                "end_time": None,
                "speaker_id": "",
                "unit_ids": [],
                "mapping_precision": "unmapped",
                "timestamp_granularity": "",
                "mapping_note": "source document or text span was unavailable",
            }

        overlaps = [
            segment for segment in source_doc.get("segments", [])
            if int(segment.get("text_end", 0)) > start and int(segment.get("text_start", 0)) < end
        ]
        if not overlaps:
            return {
                "start_time": None,
                "end_time": None,
                "speaker_id": "",
                "unit_ids": [],
                "mapping_precision": "unmapped",
                "timestamp_granularity": "",
                "mapping_note": "text span did not overlap any transcript unit",
            }

        first = overlaps[0]
        last = overlaps[-1]
        start_time = self._offset_to_time(first, start)
        end_time = self._offset_to_time(last, end)
        speaker_ids = {str(segment.get("speaker_id") or "") for segment in overlaps if str(segment.get("speaker_id") or "")}
        granularities = {
            str(((segment.get("metadata") or {}) if isinstance(segment.get("metadata"), dict) else {}).get("timestamp_granularity") or "")
            for segment in overlaps
        }
        granularities.discard("")
        timestamp_granularity = ",".join(sorted(granularities)) if granularities else "segment"
        coarse = bool(granularities & {"whole_audio", "whole_audio_reference"})
        return {
            "start_time": start_time,
            "end_time": end_time,
            "speaker_id": next(iter(speaker_ids)) if len(speaker_ids) == 1 else "multiple",
            "unit_ids": [str(segment.get("unit_id") or "") for segment in overlaps if str(segment.get("unit_id") or "")],
            "mapping_precision": "coarse" if coarse else "segment",
            "timestamp_granularity": timestamp_granularity,
            "mapping_note": (
                "ASR returned only whole-audio timestamps; character spans were linearly projected onto audio duration."
                if coarse
                else "Mapped through transcript unit timestamps."
            ),
        }

    def _offset_to_time(self, segment: dict[str, Any], doc_offset: int) -> float:
        text = str(segment.get("text") or "")
        start_offset = int(segment.get("text_start") or 0)
        end_offset = int(segment.get("text_end") or start_offset)
        duration = max(float(segment.get("end_time") or 0.0) - float(segment.get("start_time") or 0.0), 0.0)
        if duration <= 0.0 or end_offset <= start_offset or not text:
            return float(segment.get("start_time") or 0.0)
        local_offset = min(max(doc_offset - start_offset, 0), len(text))
        return float(segment.get("start_time") or 0.0) + duration * (local_offset / max(len(text), 1))

    def _redaction_targets_by_finding(self, redaction_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for record in redaction_records:
            for target in record.get("redaction_targets") or []:
                finding_id = str(target.get("finding_id") or "")
                if finding_id:
                    results[finding_id] = target
        return results

    def _build_redaction_views(
        self,
        source_map: dict[str, dict[str, Any]],
        redaction_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        plans_by_doc = {
            str(record.get("doc_id") or ""): record
            for record in redaction_records
            if str(record.get("doc_id") or "")
        }
        views: list[dict[str, Any]] = []
        for source_id, source_doc in source_map.items():
            plan = plans_by_doc.get(source_id) or {}
            targets = [target for target in plan.get("redaction_targets") or [] if isinstance(target, dict)]
            if not targets:
                continue
            enriched_targets: list[dict[str, Any]] = []
            for target in targets:
                overlap = self._locate_span(source_doc, target.get("start"), target.get("end"))
                enriched_targets.append({
                    **target,
                    "source_id": source_id,
                    "source_path": source_doc.get("source_path", ""),
                    "start_time": overlap.get("start_time"),
                    "end_time": overlap.get("end_time"),
                    "time_label": _format_time_label(overlap.get("start_time"), overlap.get("end_time")),
                    "speaker_id": overlap.get("speaker_id", ""),
                    "unit_ids": overlap.get("unit_ids", []),
                    "mapping_precision": overlap.get("mapping_precision", "unmapped"),
                    "timestamp_granularity": overlap.get("timestamp_granularity", ""),
                    "mapping_note": overlap.get("mapping_note", ""),
                })
            views.append({
                "doc_id": source_id,
                "source_path": source_doc.get("source_path", ""),
                "original_text": source_doc.get("text", ""),
                "redacted_text": _apply_redactions(source_doc.get("text", ""), targets),
                "redaction_targets": enriched_targets,
                "conflicts": plan.get("conflicts") or [],
            })
        return views

    def _is_supplemental(self, finding: dict[str, Any]) -> bool:
        risk_type = str(finding.get("risk_type") or "")
        if risk_type in _SUPPLEMENTAL_FINDING_TYPES:
            return True
        has_range = isinstance(finding.get("start"), int) and isinstance(finding.get("end"), int)
        return not has_range and not str(finding.get("text") or "").strip()

    def _empty_report(self, operator_id: str, dataset_name: str, explanation: str) -> dict[str, Any]:
        return {
            "job_id": self.run_id,
            "remote_task_id": "",
            "operator_id": operator_id,
            "operator_name": _OPERATOR_NAMES.get(operator_id, operator_id),
            "dataset_name": dataset_name,
            "modality": "audio",
            "execution_route": "api_audio_bridge",
            "execution_engine": "qwen3_asr_plus_text_api",
            "source_service": "audio.server",
            "transcript_view_contract": {
                "display_mode": "single_full_text",
                "text_field": "transcript_views[].text",
                "highlight_field": "transcript_views[].highlights",
                "audio_link_fields": ["start_time", "end_time", "time_label"],
                "interaction_targets": ["asr_text_highlight", "evidence_card", "audio_player"],
            },
            "conclusion": "passed",
            "risk_level": "none",
            "is_compliant": True,
            "decision": "allow",
            "overall_disposition": "P0",
            "trust_level": "full",
            "training_eligibility": "allow",
            "annotation_eligibility": "allow",
            "total_documents": 0,
            "total_findings": 0,
            "total_risk_records": 0,
            "visible_findings_count": 0,
            "supplemental_findings_count": 0,
            "summary": {
                "processed_sources": 0,
                "processed_units": 0,
            },
            "findings": [],
            "supplemental_findings": [],
            "risk_records": [],
            "audio_document_assessments": [],
            "audio_policy_decisions": [],
            "audio_annotation_package": [],
            "audio_audit_package": [],
            "redaction_views": [],
            "audio_redaction_spans": [],
            "redacted_audio": [],
            "transcript_views": [],
            "review_suggestions": [],
            "artifact_paths": {
                "audio": {name: str(path) for name, path in self._local_artifact_paths().items()},
                "text_api": {},
            },
            "raw_artifacts": {},
            "explanation": explanation,
        }

    def _build_transcript_views(
        self,
        source_documents: list[dict[str, Any]],
        findings: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        findings_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for finding in findings:
            doc_id = str(finding.get("doc_id") or finding.get("source_id") or "")
            if doc_id:
                findings_by_doc[doc_id].append(finding)

        views: list[dict[str, Any]] = []
        for source_doc in source_documents:
            source_id = str(source_doc.get("source_id") or source_doc.get("doc_id") or "")
            original_text = str(source_doc.get("text") or source_doc.get("original_text") or "")
            highlights: list[dict[str, Any]] = []
            for finding in findings_by_doc.get(source_id, []):
                start = finding.get("start")
                end = finding.get("end")
                if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                    continue
                if start < 0 or end > len(original_text):
                    continue
                risk_type = str(finding.get("risk_type") or finding.get("type") or "")
                policy_tag = str(finding.get("policy_tag") or "")
                risk_level = str(finding.get("risk_level") or "")
                type_label = _risk_label_zh(policy_tag, risk_type)
                time_label = finding.get("time_label") or _format_time_label(
                    finding.get("start_time"),
                    finding.get("end_time"),
                )
                highlights.append({
                    "finding_id": finding.get("finding_id") or "",
                    "start": start,
                    "end": end,
                    "text": original_text[start:end],
                    "risk_type": risk_type,
                    "risk_type_label": type_label,
                    "display_type_label": type_label,
                    "policy_tag": policy_tag,
                    "policy_tag_label": _risk_label_zh(policy_tag, policy_tag),
                    "risk_level": risk_level,
                    "risk_level_label": _risk_level_label_zh(risk_level),
                    "display_risk_level": _risk_level_label_zh(risk_level),
                    "confidence": finding.get("confidence"),
                    "start_time": finding.get("start_time"),
                    "end_time": finding.get("end_time"),
                    "time_label": time_label,
                    "speaker_id": finding.get("speaker_id") or "",
                    "unit_ids": finding.get("unit_ids") or [],
                    "explanation": finding.get("explanation") or finding.get("suggestion") or "",
                    "display_reason": f"该音频转写片段命中“{type_label}”风险，对应音频时间为{time_label}。",
                })

            views.append({
                "doc_id": source_id,
                "source_id": source_id,
                "source_path": source_doc.get("source_path", ""),
                "original_text": original_text,
                "text": original_text,
                "segments": source_doc.get("segments") or [],
                "display_mode": "single_full_text",
                "highlight_linkage": "text_span_to_audio_time",
                "highlights": sorted(highlights, key=lambda item: (int(item.get("start") or 0), int(item.get("end") or 0))),
            })
        return views
