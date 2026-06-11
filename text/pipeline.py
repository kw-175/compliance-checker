from __future__ import annotations

import logging
import uuid
from pathlib import Path

from common.contracts import ComplianceOutput
from common.enums import Modality, TrustLevel, UnifiedDecision
from text.config.settings import Settings, get_settings
from text.jsonl_utils import write_jsonl, write_single_jsonl
from text.models.schemas import (
    DispositionLevel,
    RunSummaryRecord,
)
from text.steps import (
    a_source_intake,
    downstream_annotation_export,
    f_privacy_detection,
    g_safety_moderation,
    h_evidence_aggregation,
    i_policy_decision,
    span_conflict_resolution,
)
from text.steps.delivery_audit import run as delivery_audit_run
from text.steps.hard_case_adjudication import run as hard_case_adjudication_run

logger = logging.getLogger(__name__)

DISPOSITION_PRIORITY = {
    DispositionLevel.P0: 0,
    DispositionLevel.P1: 1,
    DispositionLevel.P2: 2,
    DispositionLevel.P3: 3,
    DispositionLevel.P4: 4,
    DispositionLevel.P5: 5,
}


class CompliancePipeline:
    def __init__(self, settings: Settings | None = None, run_id: str | None = None):
        self.settings = settings or get_settings()
        self.run_id = run_id or uuid.uuid4().hex
        self.output_dir = self.settings.work_dir / self.run_id

    def execute(self, package_paths: list[str]) -> ComplianceOutput:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        artifact_paths = {
            "intake": self.output_dir / "01_intake.jsonl",
            "content_safety": self.output_dir / "02_content_safety.jsonl",
            "privacy": self.output_dir / "03_privacy_detection.jsonl",
            "redaction_plan": self.output_dir / "03b_span_conflict_resolution.jsonl",
            "hard_case": self.output_dir / "04_hard_case_adjudication.jsonl",
            "evidence": self.output_dir / "05_evidence_events.jsonl",
            "policy": self.output_dir / "06_policy_decisions.jsonl",
            "annotation": self.output_dir / "07_annotation_package.jsonl",
            "audit": self.output_dir / "08_audit_package.jsonl",
            "summary": self.output_dir / "09_run_summary.jsonl",
            "downstream_annotation_requests": self.output_dir / "10_downstream_annotation_requests.jsonl",
            "downstream_annotation_map": self.output_dir / "11_downstream_annotation_text_id_map.jsonl",
            "downstream_annotation_manifest": self.output_dir / "12_downstream_annotation_manifest.jsonl",
        }
        downstream_export_dir = self.output_dir / "10_annotation_exports"

        ingest_units = a_source_intake.run(package_paths, run_id=self.run_id)
        write_jsonl(ingest_units, artifact_paths["intake"])
        if not ingest_units:
            return self._empty_output(artifact_paths)

        safety_results = g_safety_moderation.run(ingest_units, self.settings)
        write_jsonl(safety_results, artifact_paths["content_safety"])

        privacy_results = f_privacy_detection.run(ingest_units, self.settings)
        write_jsonl(privacy_results, artifact_paths["privacy"])

        redaction_plans = span_conflict_resolution.run(ingest_units, privacy_results)
        write_jsonl(redaction_plans, artifact_paths["redaction_plan"])

        adjudications = hard_case_adjudication_run(ingest_units, safety_results, privacy_results, self.settings)
        write_jsonl(adjudications, artifact_paths["hard_case"])

        evidence_events = h_evidence_aggregation.run(ingest_units, safety_results, privacy_results, adjudications)
        write_jsonl(evidence_events, artifact_paths["evidence"])

        decisions = i_policy_decision.run(ingest_units, evidence_events, adjudications, self.settings, redaction_plans)
        write_jsonl(decisions, artifact_paths["policy"])

        annotation_records, audit_records = delivery_audit_run(
            ingest_units,
            safety_results,
            privacy_results,
            redaction_plans,
            adjudications,
            evidence_events,
            decisions,
        )
        write_jsonl(annotation_records, artifact_paths["annotation"])
        write_jsonl(audit_records, artifact_paths["audit"])

        downstream_exports = downstream_annotation_export.run(
            annotation_records,
            ingest_units,
            audit_records,
            downstream_export_dir,
            self.settings,
        )
        write_jsonl(downstream_exports.request_records, artifact_paths["downstream_annotation_requests"])
        write_jsonl(downstream_exports.mapping_records, artifact_paths["downstream_annotation_map"])
        write_jsonl(downstream_exports.manifest_records, artifact_paths["downstream_annotation_manifest"])

        overall_disposition = max(
            (decision.disposition_level for decision in decisions),
            key=lambda item: DISPOSITION_PRIORITY[item],
            default=DispositionLevel.P0,
        )
        overall_decision = max(
            (decision.unified_decision for decision in decisions),
            key=lambda item: list(UnifiedDecision).index(item),
            default=UnifiedDecision.ALLOW,
        )
        trust_level = TrustLevel.DEGRADED if any(decision.trust_level != TrustLevel.FULL for decision in decisions) else TrustLevel.FULL

        counts_by_disposition: dict[str, int] = {}
        counts_by_decision: dict[str, int] = {}
        for decision in decisions:
            counts_by_disposition[decision.disposition_level.value] = counts_by_disposition.get(decision.disposition_level.value, 0) + 1
            counts_by_decision[decision.unified_decision.value] = counts_by_decision.get(decision.unified_decision.value, 0) + 1

        review_suggestions = [
            f"{decision.doc_id}: {decision.disposition_level.value} / {decision.summary}"
            for decision in decisions
            if decision.disposition_level in {DispositionLevel.P2, DispositionLevel.P3, DispositionLevel.P4, DispositionLevel.P5}
        ]
        explanation_summary = (
            f"Processed {len(ingest_units)} cleaned documents. "
            f"Disposition distribution: {counts_by_disposition}."
        )

        summary_record = RunSummaryRecord(
            run_id=self.run_id,
            processed_documents=len(ingest_units),
            overall_disposition=overall_disposition,
            unified_decision=overall_decision,
            trust_level=trust_level,
            counts_by_disposition=counts_by_disposition,
            counts_by_decision=counts_by_decision,
            artifact_paths={
                **{name: str(path) for name, path in artifact_paths.items()},
                "downstream_annotation_export_dir": str(downstream_export_dir),
            },
            review_suggestions=review_suggestions[:20],
            explanation_summary=explanation_summary,
        )
        write_single_jsonl(summary_record, artifact_paths["summary"])

        return ComplianceOutput(
            pipeline_run_id=self.run_id,
            modality=Modality.TEXT,
            decision=overall_decision,
            trust_level=trust_level,
            annotation_package_uri=str(artifact_paths["annotation"]),
            audit_package_uri=str(artifact_paths["audit"]),
            degrade_summary="" if trust_level == TrustLevel.FULL else "Hard-case adjudication used a degraded fallback provider.",
            review_suggestions=summary_record.review_suggestions,
            explanation_summary=explanation_summary,
            legacy_decision={
                "overall_disposition": overall_disposition.value,
                "overall_decision": overall_decision.value,
                "counts_by_disposition": counts_by_disposition,
                "counts_by_decision": counts_by_decision,
                "documents": [decision.model_dump(mode="json") for decision in decisions],
            },
            metadata={"artifact_paths": summary_record.artifact_paths},
        )

    def _empty_output(self, artifact_paths: dict[str, Path]) -> ComplianceOutput:
        summary = RunSummaryRecord(
            run_id=self.run_id,
            processed_documents=0,
            overall_disposition=DispositionLevel.P0,
            unified_decision=UnifiedDecision.ALLOW,
            trust_level=TrustLevel.FULL,
            artifact_paths={
                **{name: str(path) for name, path in artifact_paths.items()},
                "downstream_annotation_export_dir": str(self.output_dir / "10_annotation_exports"),
            },
            explanation_summary="No cleaned documents were discovered in the supplied package paths.",
        )
        write_single_jsonl(summary, artifact_paths["summary"])
        return ComplianceOutput(
            pipeline_run_id=self.run_id,
            modality=Modality.TEXT,
            decision=UnifiedDecision.ALLOW,
            trust_level=TrustLevel.FULL,
            annotation_package_uri=str(artifact_paths["annotation"]),
            audit_package_uri=str(artifact_paths["audit"]),
            explanation_summary=summary.explanation_summary,
            legacy_decision={"overall_disposition": "P0", "overall_decision": "allow", "documents": []},
            metadata={"artifact_paths": summary.artifact_paths},
        )
