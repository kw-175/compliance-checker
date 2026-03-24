"""
Pipeline Orchestrator

Wires steps A → J into a single ``CompliancePipeline`` that supports:
  - Sequential and parallel execution (B2a/b parallel, E1a/b parallel)
  - Per-step JSONL output persistence
  - OpenLineage lineage tracking for every step
"""

from __future__ import annotations

import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from text.config.settings import Settings, get_settings
from text.models.schemas import (
    CleanedDocument,
    ComplianceHit,
    DedupDocument,
    DedupMapEntry,
    EvidenceBundle,
    KeywordHit,
    PolicyDecision,
    PrivacyResult,
    RegexHit,
    SafetyResult,
    SecretHit,
    SourceProfile,
    SourceRecord,
)

logger = logging.getLogger(__name__)


def _write_jsonl(records: list, output_path: Path) -> None:
    """Write a list of Pydantic models to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
    logger.debug("Wrote %d records to %s", len(records), output_path)


def _write_json(obj: Any, output_path: Path) -> None:
    """Write a single Pydantic model to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(obj.model_dump_json(indent=2))
    logger.debug("Wrote JSON to %s", output_path)


class CompliancePipeline:
    """
    Orchestrates the full A→J compliance checking workflow.

    Parameters
    ----------
    settings : Settings, optional
        Pipeline configuration.  Defaults to env-loaded settings.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.run_id = uuid.uuid4().hex
        self.output_dir = self.settings.work_dir / self.run_id

        # Lazy-import lineage tracker
        self._tracker = None

    @property
    def tracker(self):
        if self._tracker is None:
            from text.steps.j_lineage_audit import LineageTracker
            self._tracker = LineageTracker(self.settings)
        return self._tracker

    # ── Step runners with lineage integration ───────────────

    def _run_step(self, step_name: str, func, *args, output_file: str | None = None, **kwargs):
        """Generic step runner with lineage tracking."""
        run_id = self.tracker.start_step(
            step_name,
            outputs=[{"name": output_file}] if output_file else None,
        )
        try:
            result = func(*args, **kwargs)
            self.tracker.complete_step(
                step_name, run_id,
                outputs=[{"name": output_file}] if output_file else None,
            )
            return result
        except Exception as e:
            self.tracker.fail_step(step_name, run_id, str(e))
            raise

    def execute(self, input_paths: list[str]) -> PolicyDecision:
        """
        Run the full compliance pipeline.

        Parameters
        ----------
        input_paths : list[str]
            Files / directories / URLs to check.

        Returns
        -------
        PolicyDecision
        """
        logger.info(
            "═══ Pipeline run %s started ═══ (%d input paths)",
            self.run_id[:8], len(input_paths),
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Step A: Source Intake ─────────────────────────
        from text.steps import a_source_intake
        sources: list[SourceRecord] = self._run_step(
            "step_a_source_intake",
            a_source_intake.run,
            input_paths,
            output_file="source_registry.jsonl",
        )
        _write_jsonl(sources, self.output_dir / "source_registry.jsonl")

        if not sources:
            logger.warning("No sources found – aborting pipeline")
            return PolicyDecision(pipeline_run_id=self.run_id)

        # ── Step B1: Source Classification ────────────────
        from text.steps import b1_source_classify
        profiles: list[SourceProfile] = self._run_step(
            "step_b1_source_classify",
            b1_source_classify.run,
            sources,
            output_file="source_profile.jsonl",
        )
        _write_jsonl(profiles, self.output_dir / "source_profile.jsonl")

        # ── Step B2: Raw Object Scans (parallel) ─────────
        from text.steps import b2a_trufflehog_scan, b2b_scancode_scan

        secret_hits: list[SecretHit] = []
        compliance_hits: list[ComplianceHit] = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_secrets = executor.submit(
                self._run_step,
                "step_b2a_trufflehog",
                b2a_trufflehog_scan.run,
                sources, self.settings,
                output_file="raw_secret_hits.jsonl",
            )
            future_compliance = executor.submit(
                self._run_step,
                "step_b2b_scancode",
                b2b_scancode_scan.run,
                profiles, self.settings,
                output_file="source_compliance.jsonl",
            )

            try:
                secret_hits = future_secrets.result()
            except Exception as e:
                logger.error("TruffleHog scan failed: %s", e)

            try:
                compliance_hits = future_compliance.result()
            except Exception as e:
                logger.error("ScanCode scan failed: %s", e)

        _write_jsonl(secret_hits, self.output_dir / "raw_secret_hits.jsonl")
        _write_jsonl(compliance_hits, self.output_dir / "source_compliance.jsonl")

        # ── Step C: Text Extract & Preprocess ────────────
        from text.steps import c_text_extract
        cleaned_docs: list[CleanedDocument] = self._run_step(
            "step_c_text_extract",
            c_text_extract.run,
            profiles, self.settings,
            output_file="cleaned_documents.jsonl",
        )
        _write_jsonl(cleaned_docs, self.output_dir / "cleaned_documents.jsonl")

        if not cleaned_docs:
            logger.warning("No text extracted – aborting pipeline")
            return PolicyDecision(pipeline_run_id=self.run_id)

        # ── Step D: Early Dedup ──────────────────────────
        from text.steps import d_dedup
        dedup_docs, dedup_map = self._run_step(
            "step_d_dedup",
            d_dedup.run,
            cleaned_docs, self.settings,
            output_file="deduped_documents.jsonl",
        )
        _write_jsonl(dedup_docs, self.output_dir / "deduped_documents.jsonl")
        _write_jsonl(dedup_map, self.output_dir / "dedup_map.jsonl")

        # Non-duplicate documents for subsequent steps
        active_docs = [d for d in dedup_docs if not d.is_duplicate]
        logger.info("Active (non-duplicate) documents: %d", len(active_docs))

        # ── Step E1: Deterministic Text Scans (parallel) ─
        from text.steps import e1a_keyword_scan, e1b_regex_scan

        keyword_hits: list[KeywordHit] = []
        regex_hits: list[RegexHit] = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_kw = executor.submit(
                self._run_step,
                "step_e1a_keyword_scan",
                e1a_keyword_scan.run,
                dedup_docs, self.settings,
                output_file="keyword_hits.jsonl",
            )
            future_rx = executor.submit(
                self._run_step,
                "step_e1b_regex_scan",
                e1b_regex_scan.run,
                dedup_docs, self.settings,
                output_file="regex_hits.jsonl",
            )

            try:
                keyword_hits = future_kw.result()
            except Exception as e:
                logger.error("Keyword scan failed: %s", e)

            try:
                regex_hits = future_rx.result()
            except Exception as e:
                logger.error("Regex scan failed: %s", e)

        _write_jsonl(keyword_hits, self.output_dir / "keyword_hits.jsonl")
        _write_jsonl(regex_hits, self.output_dir / "regex_hits.jsonl")

        # ── Step F: Privacy Detection & Redaction ────────
        from text.steps import f_privacy_detection
        privacy_results: list[PrivacyResult] = self._run_step(
            "step_f_privacy",
            f_privacy_detection.run,
            dedup_docs, self.settings,
            output_file="privacy_checked.jsonl",
        )
        _write_jsonl(privacy_results, self.output_dir / "privacy_checked.jsonl")

        # ── Step G: Semantic Safety Moderation ────────────
        from text.steps import g_safety_moderation
        safety_results: list[SafetyResult] = self._run_step(
            "step_g_safety",
            g_safety_moderation.run,
            privacy_results, self.settings,
            output_file="safety_checked.jsonl",
        )
        _write_jsonl(safety_results, self.output_dir / "safety_checked.jsonl")

        # ── Step H: Evidence Aggregation ─────────────────
        from text.steps import h_evidence_aggregation
        evidence_bundle: EvidenceBundle = self._run_step(
            "step_h_evidence_aggregation",
            h_evidence_aggregation.run,
            dedup_docs,
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

        # ── Step I: Policy Decision ──────────────────────
        from text.steps import i_policy_decision
        decision: PolicyDecision = self._run_step(
            "step_i_policy_decision",
            i_policy_decision.run,
            evidence_bundle, self.settings,
            output_file="decision.json",
        )
        _write_json(decision, self.output_dir / "decision.json")

        logger.info(
            "═══ Pipeline run %s completed ═══  Overall decision: %s",
            self.run_id[:8], decision.overall_decision.value,
        )
        return decision
