"""
Pipeline orchestrator for audio privacy and content-safety checking.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from audio.config.settings import Settings, get_settings
from audio.models.schemas import AudioRunSummaryRecord, Decision
from common.contracts import ComplianceOutput
from common.enums import Modality, TrustLevel, UnifiedDecision

logger = logging.getLogger(__name__)

DECISION_PRIORITY = {
    Decision.ALLOW: 0,
    Decision.REVIEW: 1,
    Decision.QUARANTINE: 2,
    Decision.REJECT: 3,
}
DECISION_TO_UNIFIED = {
    Decision.ALLOW: UnifiedDecision.ALLOW,
    Decision.REVIEW: UnifiedDecision.REVIEW,
    Decision.QUARANTINE: UnifiedDecision.QUARANTINE,
    Decision.REJECT: UnifiedDecision.REJECT,
}


def _write_jsonl(records: list, output_path: Path) -> None:
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


def _write_single_jsonl(record: Any, output_path: Path) -> None:
    _write_jsonl([record], output_path)


class AudioCompliancePipeline:
    def __init__(self, settings: Settings | None = None, run_id: str | None = None):
        self.settings = settings or get_settings()
        self.run_id = run_id or uuid.uuid4().hex
        self.output_dir = self.settings.work_dir / self.run_id

    def execute(self, input_paths: list[str]) -> ComplianceOutput:
        from audio.steps import (
            a_source_intake,
            b1_source_classify,
            c0_audio_normalize,
            c1_asr_transcribe,
            c1b_diarization,
            c1c_alignment,
            c2_transcript_build,
            f_privacy_detection,
            g_safety_moderation,
            hard_case_adjudication,
            h_evidence_aggregation,
            i_policy_decision,
        )
        from audio.steps.delivery_audit import run as delivery_audit_run

        logger.info("Audio privacy/safety pipeline run %s started", self.run_id[:8])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        artifact_paths = {
            "intake": self.output_dir / "01_source_registry.jsonl",
            "source_profile": self.output_dir / "02_source_profile.jsonl",
            "normalized_audio": self.output_dir / "03_normalized_audio_manifest.jsonl",
            "asr": self.output_dir / "04_asr_segments.jsonl",
            "speaker": self.output_dir / "05_speaker_segments.jsonl",
            "aligned": self.output_dir / "06_aligned_segments.jsonl",
            "transcript": self.output_dir / "07_transcript_units.jsonl",
            "privacy": self.output_dir / "08_privacy_detection.jsonl",
            "redaction_spans": self.output_dir / "08b_redaction_spans.jsonl",
            "content_safety": self.output_dir / "09_content_safety.jsonl",
            "hard_case": self.output_dir / "09b_hard_case_adjudication.jsonl",
            "evidence": self.output_dir / "10_evidence_bundle.json",
            "policy": self.output_dir / "11_policy_decision.json",
            "annotation": self.output_dir / "12_annotation_package.jsonl",
            "audit": self.output_dir / "13_audit_package.jsonl",
            "summary": self.output_dir / "14_run_summary.jsonl",
            "compliance_output": self.output_dir / "compliance_output.json",
        }

        sources = a_source_intake.run(input_paths)
        _write_jsonl(sources, artifact_paths["intake"])
        if not sources:
            return self._empty_output(artifact_paths, "No audio sources were discovered in the supplied paths.")

        profiles = b1_source_classify.run(sources)
        _write_jsonl(profiles, artifact_paths["source_profile"])

        normalized = c0_audio_normalize.run(profiles, self.settings, self.output_dir)
        _write_jsonl(normalized, artifact_paths["normalized_audio"])
        if not normalized:
            return self._empty_output(artifact_paths, "No usable audio records were discovered after intake.")

        asr_segments = c1_asr_transcribe.run(normalized, self.settings)
        _write_jsonl(asr_segments, artifact_paths["asr"])

        speaker_segments = c1b_diarization.run(normalized, self.settings)
        _write_jsonl(speaker_segments, artifact_paths["speaker"])

        aligned_segments = c1c_alignment.run(asr_segments)
        _write_jsonl(aligned_segments, artifact_paths["aligned"])

        transcript_units = c2_transcript_build.run(aligned_segments, speaker_segments)
        _write_jsonl(transcript_units, artifact_paths["transcript"])
        if not transcript_units:
            return self._empty_output(artifact_paths, "No transcript units were available for compliance detection.")

        privacy_results, redaction_spans = f_privacy_detection.run(transcript_units, self.settings)
        _write_jsonl(privacy_results, artifact_paths["privacy"])
        _write_jsonl(redaction_spans, artifact_paths["redaction_spans"])

        safety_results = g_safety_moderation.run(privacy_results, self.settings)
        _write_jsonl(safety_results, artifact_paths["content_safety"])

        hard_case_results = hard_case_adjudication.run(
            transcript_units,
            privacy_results,
            safety_results,
            self.settings,
            run_id=self.run_id,
        )
        _write_jsonl(hard_case_results, artifact_paths["hard_case"])

        extension_events = self._run_detection_extensions(
            transcript_units=transcript_units,
            privacy_results=privacy_results,
            safety_results=safety_results,
            hard_case_results=hard_case_results,
        )
        evidence_bundle = h_evidence_aggregation.run(
            transcript_units,
            privacy_results,
            safety_results,
            self.run_id,
            hard_case_results=hard_case_results,
            extension_events=extension_events,
        )
        _write_json(evidence_bundle, artifact_paths["evidence"])

        policy_decision = i_policy_decision.run(evidence_bundle, self.settings)
        _write_json(policy_decision, artifact_paths["policy"])

        annotation_records, audit_records = delivery_audit_run(
            transcript_units,
            privacy_results,
            safety_results,
            redaction_spans,
            evidence_bundle,
            policy_decision,
            hard_case_results=hard_case_results,
        )
        _write_jsonl(annotation_records, artifact_paths["annotation"])
        _write_jsonl(audit_records, artifact_paths["audit"])

        counts_by_decision: dict[str, int] = {}
        for unit_decision in policy_decision.unit_decisions:
            counts_by_decision[unit_decision.decision.value] = counts_by_decision.get(unit_decision.decision.value, 0) + 1

        review_suggestions = [
            f"{item.unit_id}: {item.decision.value} / {', '.join(item.reasons)}"
            for item in policy_decision.unit_decisions
            if item.decision != Decision.ALLOW
        ]
        explanation_summary = (
            f"Processed {len(transcript_units)} audio transcript units from "
            f"{len({unit.source_id for unit in transcript_units})} source(s). "
            f"Decision distribution: {counts_by_decision}."
        )
        summary_record = AudioRunSummaryRecord(
            run_id=self.run_id,
            processed_units=len(transcript_units),
            processed_sources=len({unit.source_id for unit in transcript_units}),
            overall_decision=policy_decision.overall_decision,
            counts_by_decision=counts_by_decision,
            artifact_paths={name: str(path) for name, path in artifact_paths.items()},
            review_suggestions=review_suggestions[:20],
            explanation_summary=explanation_summary,
            metadata={
                "active_detectors": ["privacy", "content_safety", "hard_case_adjudication"],
                "hard_case_units": len(hard_case_results),
                "reserved_extension_events": len(extension_events),
            },
        )
        _write_single_jsonl(summary_record, artifact_paths["summary"])

        output = self._build_compliance_output(
            summary_record=summary_record,
            policy_decision=policy_decision,
            artifact_paths=artifact_paths,
        )
        _write_json(output, artifact_paths["compliance_output"])
        logger.info("Audio privacy/safety pipeline run %s completed", self.run_id[:8])
        return output

    def _run_detection_extensions(self, **_: Any) -> list[dict[str, Any]]:
        return []

    def _empty_output(self, artifact_paths: dict[str, Path], explanation: str) -> ComplianceOutput:
        summary = AudioRunSummaryRecord(
            run_id=self.run_id,
            processed_units=0,
            processed_sources=0,
            overall_decision=Decision.ALLOW,
            counts_by_decision={Decision.ALLOW.value: 0},
            artifact_paths={name: str(path) for name, path in artifact_paths.items()},
            explanation_summary=explanation,
            metadata={"active_detectors": ["privacy", "content_safety", "hard_case_adjudication"]},
        )
        _write_single_jsonl(summary, artifact_paths["summary"])
        output = ComplianceOutput(
            pipeline_run_id=self.run_id,
            modality=Modality.AUDIO,
            decision=UnifiedDecision.ALLOW,
            trust_level=TrustLevel.FULL,
            annotation_package_uri=str(artifact_paths["annotation"]),
            audit_package_uri=str(artifact_paths["audit"]),
            explanation_summary=explanation,
            legacy_decision={
                "overall_decision": Decision.ALLOW.value,
                "unit_decisions": [],
            },
            metadata={"artifact_paths": summary.artifact_paths},
        )
        _write_json(output, artifact_paths["compliance_output"])
        return output

    def _build_compliance_output(
        self,
        *,
        summary_record: AudioRunSummaryRecord,
        policy_decision,
        artifact_paths: dict[str, Path],
    ) -> ComplianceOutput:
        unified_decision = DECISION_TO_UNIFIED.get(policy_decision.overall_decision, UnifiedDecision.REVIEW)
        trust_level = TrustLevel.DEGRADED if policy_decision.trust_level != "full" else TrustLevel.FULL
        return ComplianceOutput(
            pipeline_run_id=self.run_id,
            modality=Modality.AUDIO,
            decision=unified_decision,
            trust_level=trust_level,
            annotation_package_uri=str(artifact_paths["annotation"]),
            audit_package_uri=str(artifact_paths["audit"]),
            degrade_summary=policy_decision.degrade_summary,
            review_suggestions=summary_record.review_suggestions,
            explanation_summary=summary_record.explanation_summary,
            legacy_decision=policy_decision.model_dump(mode="json"),
            metadata={
                "artifact_paths": summary_record.artifact_paths,
                "active_detectors": ["privacy", "content_safety", "hard_case_adjudication"],
                "extension_interface": "AudioCompliancePipeline._run_detection_extensions",
            },
        )
