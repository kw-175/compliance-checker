"""
Pipeline orchestrator for audio compliance checking.
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from audio.config.settings import Settings, get_settings
from audio.models.schemas import EvidenceBundle, PolicyDecision, ReleasePackage

logger = logging.getLogger(__name__)


def _write_jsonl(records: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def _write_json(record: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        if hasattr(record, "model_dump_json"):
            handle.write(record.model_dump_json(indent=2))
        else:
            import json
            json.dump(record, handle, indent=2, ensure_ascii=False)


class AudioCompliancePipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.run_id = uuid.uuid4().hex
        self.output_dir = self.settings.work_dir / self.run_id
        self._tracker = None

    @property
    def tracker(self):
        if self._tracker is None:
            from audio.steps.j_lineage_audit import LineageTracker
            self._tracker = LineageTracker(self.settings)
        return self._tracker

    def _run_step(self, step_name: str, func, *args, output_file: str | None = None, **kwargs):
        run_id = self.tracker.start_step(step_name, outputs=[{"name": output_file}] if output_file else None)
        try:
            result = func(*args, **kwargs)
            self.tracker.complete_step(step_name, run_id, outputs=[{"name": output_file}] if output_file else None)
            return result
        except Exception as exc:
            self.tracker.fail_step(step_name, run_id, str(exc))
            raise

    def execute(self, input_paths: list[str]) -> PolicyDecision:
        from audio.steps import (
            a_source_intake,
            b1_source_classify,
            b2a_trufflehog_scan,
            b2b_scancode_scan,
            c0_audio_normalize,
            c1_asr_transcribe,
            c1b_diarization,
            c1c_alignment,
            c2_transcript_build,
            d_dedup,
            e1a_keyword_scan,
            e1b_regex_scan,
            f_privacy_detection,
            g_safety_moderation,
            h_evidence_aggregation,
            i_policy_decision,
            k_audio_redaction,
            l_release_package,
        )

        logger.info("Audio pipeline run %s started", self.run_id[:8])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sources = self._run_step("step_a_source_intake", a_source_intake.run, input_paths, output_file="source_registry.jsonl")
        _write_jsonl(sources, self.output_dir / "source_registry.jsonl")
        if not sources:
            return PolicyDecision(pipeline_run_id=self.run_id)

        profiles = self._run_step("step_b1_source_classify", b1_source_classify.run, sources, output_file="source_profile.jsonl")
        _write_jsonl(profiles, self.output_dir / "source_profile.jsonl")

        secret_hits = []
        compliance_hits = []
        with ThreadPoolExecutor(max_workers=max(1, min(2, int(self.settings.max_workers or 2)))) as executor:
            secret_future = executor.submit(self._run_step, "step_b2a_trufflehog", b2a_trufflehog_scan.run, sources, self.settings, output_file="raw_secret_hits.jsonl")
            compliance_future = executor.submit(self._run_step, "step_b2b_scancode", b2b_scancode_scan.run, profiles, self.settings, output_file="source_compliance.jsonl")
            try:
                secret_hits = secret_future.result()
            except Exception as exc:
                logger.warning("TruffleHog step degraded: %s", exc)
            try:
                compliance_hits = compliance_future.result()
            except Exception as exc:
                logger.warning("ScanCode step degraded: %s", exc)
        _write_jsonl(secret_hits, self.output_dir / "raw_secret_hits.jsonl")
        _write_jsonl(compliance_hits, self.output_dir / "source_compliance.jsonl")

        normalized = self._run_step("step_c0_audio_normalize", c0_audio_normalize.run, profiles, self.settings, self.output_dir, output_file="normalized_audio_manifest.jsonl")
        _write_jsonl(normalized, self.output_dir / "normalized_audio_manifest.jsonl")
        if not normalized:
            return PolicyDecision(pipeline_run_id=self.run_id)

        asr_segments = self._run_step("step_c1_asr_transcribe", c1_asr_transcribe.run, normalized, self.settings, output_file="asr_segments.jsonl")
        _write_jsonl(asr_segments, self.output_dir / "asr_segments.jsonl")

        speaker_segments = self._run_step("step_c1b_diarization", c1b_diarization.run, normalized, self.settings, output_file="speaker_segments.jsonl")
        _write_jsonl(speaker_segments, self.output_dir / "speaker_segments.jsonl")

        aligned_segments = self._run_step("step_c1c_alignment", c1c_alignment.run, asr_segments, output_file="aligned_segments.jsonl")
        _write_jsonl(aligned_segments, self.output_dir / "aligned_segments.jsonl")

        transcript_units = self._run_step("step_c2_transcript_build", c2_transcript_build.run, aligned_segments, speaker_segments, output_file="transcript_units.jsonl")
        _write_jsonl(transcript_units, self.output_dir / "transcript_units.jsonl")

        deduped_units, dedup_map = self._run_step("step_d_dedup", d_dedup.run, transcript_units, self.settings, output_file="deduped_transcript_units.jsonl")
        _write_jsonl(deduped_units, self.output_dir / "deduped_transcript_units.jsonl")
        _write_jsonl(dedup_map, self.output_dir / "dedup_map.jsonl")

        keyword_hits = []
        regex_hits = []
        with ThreadPoolExecutor(max_workers=max(1, min(2, int(self.settings.max_workers or 2)))) as executor:
            keyword_future = executor.submit(self._run_step, "step_e1a_keyword_scan", e1a_keyword_scan.run, deduped_units, self.settings, output_file="keyword_hits.jsonl")
            regex_future = executor.submit(self._run_step, "step_e1b_regex_scan", e1b_regex_scan.run, deduped_units, self.settings, output_file="regex_hits.jsonl")
            try:
                keyword_hits = keyword_future.result()
            except Exception as exc:
                logger.warning("Keyword scan degraded: %s", exc)
            try:
                regex_hits = regex_future.result()
            except Exception as exc:
                logger.warning("Regex scan degraded: %s", exc)
        _write_jsonl(keyword_hits, self.output_dir / "keyword_hits.jsonl")
        _write_jsonl(regex_hits, self.output_dir / "regex_hits.jsonl")

        privacy_results, redaction_spans = self._run_step("step_f_privacy_detection", f_privacy_detection.run, deduped_units, self.settings, output_file="privacy_checked.jsonl")
        _write_jsonl(privacy_results, self.output_dir / "privacy_checked.jsonl")
        _write_jsonl(redaction_spans, self.output_dir / "redaction_spans.jsonl")

        safety_results = self._run_step("step_g_safety_moderation", g_safety_moderation.run, privacy_results, self.settings, output_file="safety_checked.jsonl")
        _write_jsonl(safety_results, self.output_dir / "safety_checked.jsonl")

        evidence_bundle: EvidenceBundle = self._run_step(
            "step_h_evidence_aggregation",
            h_evidence_aggregation.run,
            deduped_units,
            secret_hits,
            compliance_hits,
            keyword_hits,
            regex_hits,
            privacy_results,
            safety_results,
            self.run_id,
            output_file="evidence_bundle.json",
        )
        _write_json(evidence_bundle, self.output_dir / "evidence_bundle.json")

        decision = self._run_step("step_i_policy_decision", i_policy_decision.run, evidence_bundle, self.settings, output_file="decision.json")
        _write_json(decision, self.output_dir / "decision.json")

        redacted_audio = self._run_step("step_k_audio_redaction", k_audio_redaction.run, normalized, redaction_spans, self.settings, self.output_dir, output_file="redacted_audio_manifest.jsonl")
        _write_jsonl(redacted_audio, self.output_dir / "redacted_audio_manifest.jsonl")

        release_package: ReleasePackage = self._run_step(
            "step_l_release_package",
            l_release_package.run,
            self.run_id,
            normalized,
            evidence_bundle,
            decision,
            redacted_audio,
            {"output_dir": str(self.output_dir.resolve())},
            output_file="release_package.json",
        )
        _write_json(release_package, self.output_dir / "release_package.json")

        logger.info("Audio pipeline run %s completed with decision=%s", self.run_id[:8], decision.overall_decision.value)
        return decision

