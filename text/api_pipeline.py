from __future__ import annotations

import logging
import uuid
from enum import Enum
from pathlib import Path

from common.contracts import ComplianceOutput
from common.enums import Modality, TrustLevel, UnifiedDecision
from text.api_clients import resolve_provider_config
from text.api_steps import api_hard_case_adjudication, api_privacy_detection, api_safety_moderation
from text.config.settings import Settings, get_settings
from text.engines.privacy_decision_engine import aggregate_privacy_document, decide_privacy_finding
from text.engines.privacy_policy_engine import load_privacy_policies, match_privacy_policy_hits
from text.engines.privacy_rule_engine import load_privacy_entity_catalog, privacy_rule_hit
from text.jsonl_utils import write_jsonl, write_single_jsonl
from text.models.schemas import DispositionLevel, RunSummaryRecord
from text.steps import (
    a_source_intake,
    b_document_context,
    b_content_candidate_windows,
    b_content_fragment_localization,
    c_content_fragment_adjudication,
    c_privacy_fragment_adjudication,
    content_safety_review,
    d_content_document_assessment,
    d_privacy_document_assessment,
    downstream_annotation_export,
    h_evidence_aggregation,
    i_policy_decision,
    privacy_review,
    span_conflict_resolution,
)
from text.steps.delivery_audit import run as delivery_audit_run

logger = logging.getLogger(__name__)

DISPOSITION_PRIORITY = {
    DispositionLevel.P0: 0,
    DispositionLevel.P1: 1,
    DispositionLevel.P2: 2,
    DispositionLevel.P3: 3,
    DispositionLevel.P4: 4,
    DispositionLevel.P5: 5,
}


class PipelineProfile(str, Enum):
    FULL = "full"
    PRIVACY_ONLY = "privacy_only"
    SAFETY_ONLY = "safety_only"


class APICompliancePipeline:
    """Provider-neutral text compliance pipeline with local-model-first support."""

    def __init__(self, settings: Settings | None = None, run_id: str | None = None):
        self.settings = settings or get_settings()
        self.run_id = run_id or uuid.uuid4().hex
        self.output_dir = self.settings.work_dir / self.run_id

    def execute(self, package_paths: list[str], profile: str | PipelineProfile = PipelineProfile.FULL) -> ComplianceOutput:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifact_paths = self._artifact_paths()
        downstream_export_dir = self.output_dir / "10_annotation_exports"
        active_profile = self._normalize_profile(profile)
        provider_meta = self._provider_metadata()

        ingest_units = a_source_intake.run(package_paths, run_id=self.run_id)
        write_jsonl(ingest_units, artifact_paths["intake"])
        document_views = self._document_views(ingest_units)
        write_jsonl(document_views, artifact_paths["document_views"])
        if not ingest_units:
            return self._empty_output(artifact_paths, active_profile, provider_meta)

        document_contexts = self._build_document_contexts(ingest_units, provider_meta)
        write_jsonl(document_contexts, artifact_paths["document_context"])

        if active_profile == PipelineProfile.PRIVACY_ONLY:
            privacy_results = api_privacy_detection.run(ingest_units, self.settings, document_contexts=document_contexts)
            write_jsonl(privacy_results, artifact_paths["privacy"])
            privacy_fragment_adjudications = c_privacy_fragment_adjudication.run(
                ingest_units,
                privacy_results,
                document_contexts,
                self.settings,
            )
            write_jsonl(privacy_fragment_adjudications, artifact_paths["privacy_fragment_adjudications"])
            privacy_document_assessments = d_privacy_document_assessment.run(
                ingest_units,
                privacy_results,
                privacy_fragment_adjudications,
                document_contexts,
                self.settings,
            )
            write_jsonl(privacy_document_assessments, artifact_paths["privacy_document_assessments"])
            redaction_plans = span_conflict_resolution.run(ingest_units, privacy_results)
            write_jsonl(redaction_plans, artifact_paths["redaction_plan"])
            self._write_privacy_governance_artifacts(
                ingest_units,
                privacy_results,
                redaction_plans,
                artifact_paths,
                document_contexts=document_contexts,
                privacy_fragment_adjudications=privacy_fragment_adjudications,
                privacy_document_assessments=privacy_document_assessments,
            )
            adjudications = []
            write_jsonl(adjudications, artifact_paths["hard_case"])
            return self._finalize_governance_output(
                profile=active_profile,
                ingest_units=ingest_units,
                artifact_paths=artifact_paths,
                downstream_export_dir=downstream_export_dir,
                provider_meta=provider_meta,
                document_views=document_views,
                safety_results=[],
                privacy_results=privacy_results,
                redaction_plans=redaction_plans,
                adjudications=adjudications,
                document_contexts=document_contexts,
                content_candidate_windows=[],
                content_localized_fragments=[],
                privacy_fragment_adjudications=privacy_fragment_adjudications,
                content_fragment_adjudications=[],
                privacy_document_assessments=privacy_document_assessments,
                content_document_assessments=[],
            )

        if active_profile == PipelineProfile.SAFETY_ONLY:
            content_candidate_windows = []
            localized_fragments = []
            if provider_meta["mode"] == "local_model":
                content_candidate_windows = b_content_candidate_windows.run(ingest_units, document_contexts, self.settings)
                write_jsonl(content_candidate_windows, artifact_paths["content_candidate_windows"])
                localized_fragments, safety_results = b_content_fragment_localization.run(
                    ingest_units,
                    content_candidate_windows,
                    document_contexts,
                    self.settings,
                )
                write_jsonl(localized_fragments, artifact_paths["content_fragment_localization"])
            else:
                safety_results = api_safety_moderation.run(ingest_units, self.settings, document_contexts=document_contexts)
            write_jsonl(safety_results, artifact_paths["content_safety"])
            content_fragment_adjudications = c_content_fragment_adjudication.run(
                ingest_units,
                safety_results,
                document_contexts,
                self.settings,
            )
            write_jsonl(content_fragment_adjudications, artifact_paths["content_fragment_adjudications"])
            content_document_assessments = d_content_document_assessment.run(
                ingest_units,
                safety_results,
                content_fragment_adjudications,
                document_contexts,
                self.settings,
            )
            write_jsonl(content_document_assessments, artifact_paths["content_document_assessments"])
            self._write_content_safety_governance_artifacts(
                safety_results,
                artifact_paths,
                document_contexts=document_contexts,
                content_candidate_windows=content_candidate_windows,
                content_localized_fragments=localized_fragments,
                content_fragment_adjudications=content_fragment_adjudications,
                content_document_assessments=content_document_assessments,
            )
            redaction_plans = []
            adjudications = []
            write_jsonl(adjudications, artifact_paths["redaction_plan"])
            write_jsonl(adjudications, artifact_paths["hard_case"])
            return self._finalize_governance_output(
                profile=active_profile,
                ingest_units=ingest_units,
                artifact_paths=artifact_paths,
                downstream_export_dir=downstream_export_dir,
                provider_meta=provider_meta,
                document_views=document_views,
                safety_results=safety_results,
                privacy_results=[],
                redaction_plans=redaction_plans,
                adjudications=adjudications,
                document_contexts=document_contexts,
                content_candidate_windows=content_candidate_windows,
                content_localized_fragments=localized_fragments,
                privacy_fragment_adjudications=[],
                content_fragment_adjudications=content_fragment_adjudications,
                privacy_document_assessments=[],
                content_document_assessments=content_document_assessments,
            )

        content_candidate_windows = []
        localized_fragments = []
        if provider_meta["mode"] == "local_model":
            content_candidate_windows = b_content_candidate_windows.run(ingest_units, document_contexts, self.settings)
            write_jsonl(content_candidate_windows, artifact_paths["content_candidate_windows"])
            localized_fragments, safety_results = b_content_fragment_localization.run(
                ingest_units,
                content_candidate_windows,
                document_contexts,
                self.settings,
            )
            write_jsonl(localized_fragments, artifact_paths["content_fragment_localization"])
        else:
            safety_results = api_safety_moderation.run(ingest_units, self.settings, document_contexts=document_contexts)
        write_jsonl(safety_results, artifact_paths["content_safety"])
        content_fragment_adjudications = c_content_fragment_adjudication.run(
            ingest_units,
            safety_results,
            document_contexts,
            self.settings,
        )
        write_jsonl(content_fragment_adjudications, artifact_paths["content_fragment_adjudications"])
        content_document_assessments = d_content_document_assessment.run(
            ingest_units,
            safety_results,
            content_fragment_adjudications,
            document_contexts,
            self.settings,
        )
        write_jsonl(content_document_assessments, artifact_paths["content_document_assessments"])
        self._write_content_safety_governance_artifacts(
            safety_results,
            artifact_paths,
            document_contexts=document_contexts,
            content_candidate_windows=content_candidate_windows,
            content_localized_fragments=localized_fragments,
            content_fragment_adjudications=content_fragment_adjudications,
            content_document_assessments=content_document_assessments,
        )

        privacy_results = api_privacy_detection.run(ingest_units, self.settings, document_contexts=document_contexts)
        write_jsonl(privacy_results, artifact_paths["privacy"])
        privacy_fragment_adjudications = c_privacy_fragment_adjudication.run(
            ingest_units,
            privacy_results,
            document_contexts,
            self.settings,
        )
        write_jsonl(privacy_fragment_adjudications, artifact_paths["privacy_fragment_adjudications"])
        privacy_document_assessments = d_privacy_document_assessment.run(
            ingest_units,
            privacy_results,
            privacy_fragment_adjudications,
            document_contexts,
            self.settings,
        )
        write_jsonl(privacy_document_assessments, artifact_paths["privacy_document_assessments"])

        redaction_plans = span_conflict_resolution.run(ingest_units, privacy_results)
        write_jsonl(redaction_plans, artifact_paths["redaction_plan"])
        self._write_privacy_governance_artifacts(
            ingest_units,
            privacy_results,
            redaction_plans,
            artifact_paths,
            document_contexts=document_contexts,
            privacy_fragment_adjudications=privacy_fragment_adjudications,
            privacy_document_assessments=privacy_document_assessments,
        )

        adjudications = api_hard_case_adjudication.run(
            ingest_units,
            safety_results,
            privacy_results,
            self.settings,
            document_contexts=document_contexts,
            content_document_assessments=content_document_assessments,
            privacy_document_assessments=privacy_document_assessments,
        )
        write_jsonl(adjudications, artifact_paths["hard_case"])

        return self._finalize_governance_output(
            profile=active_profile,
            ingest_units=ingest_units,
            artifact_paths=artifact_paths,
            downstream_export_dir=downstream_export_dir,
            provider_meta=provider_meta,
            document_views=document_views,
            safety_results=safety_results,
            privacy_results=privacy_results,
            redaction_plans=redaction_plans,
            adjudications=adjudications,
            document_contexts=document_contexts,
            content_candidate_windows=content_candidate_windows,
            content_localized_fragments=localized_fragments,
            privacy_fragment_adjudications=privacy_fragment_adjudications,
            content_fragment_adjudications=content_fragment_adjudications,
            privacy_document_assessments=privacy_document_assessments,
            content_document_assessments=content_document_assessments,
        )

    def _finalize_governance_output(
        self,
        *,
        profile: PipelineProfile,
        ingest_units: list,
        artifact_paths: dict[str, Path],
        downstream_export_dir: Path,
        provider_meta: dict[str, str],
        document_views: list[dict[str, Any]],
        safety_results: list,
        privacy_results: list,
        redaction_plans: list,
        adjudications: list,
        document_contexts: list | None = None,
        content_candidate_windows: list | None = None,
        content_localized_fragments: list | None = None,
        privacy_fragment_adjudications: list | None = None,
        content_fragment_adjudications: list | None = None,
        privacy_document_assessments: list | None = None,
        content_document_assessments: list | None = None,
    ) -> ComplianceOutput:
        """Route every profile through the same final governance output.

        Single-chain profiles pass empty inputs for the inactive chain. This keeps
        final policy, annotation, audit, and summary semantics aligned with the
        full profile without fabricating inactive-chain findings.
        """
        evidence_events = h_evidence_aggregation.run(
            ingest_units,
            safety_results,
            privacy_results,
            adjudications,
            document_contexts=document_contexts,
            privacy_fragment_adjudications=privacy_fragment_adjudications,
            content_fragment_adjudications=content_fragment_adjudications,
            privacy_document_assessments=privacy_document_assessments,
            content_document_assessments=content_document_assessments,
        )
        write_jsonl(evidence_events, artifact_paths["evidence"])

        decisions = i_policy_decision.run(
            ingest_units,
            evidence_events,
            adjudications,
            self.settings,
            redaction_plans,
            document_contexts=document_contexts,
            privacy_fragment_adjudications=privacy_fragment_adjudications,
            content_fragment_adjudications=content_fragment_adjudications,
            content_candidate_windows=content_candidate_windows,
            content_localized_fragments=content_localized_fragments,
            privacy_document_assessments=privacy_document_assessments,
            content_document_assessments=content_document_assessments,
        )
        write_jsonl(decisions, artifact_paths["policy"])

        annotation_records, audit_records = delivery_audit_run(
            ingest_units,
            safety_results,
            privacy_results,
            redaction_plans,
            adjudications,
            evidence_events,
            decisions,
            document_contexts=document_contexts,
            content_candidate_windows=content_candidate_windows,
            content_localized_fragments=content_localized_fragments,
            privacy_fragment_adjudications=privacy_fragment_adjudications,
            content_fragment_adjudications=content_fragment_adjudications,
            privacy_document_assessments=privacy_document_assessments,
            content_document_assessments=content_document_assessments,
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

        return self._output_from_decisions(
            profile=profile,
            decisions=decisions,
            artifact_paths=artifact_paths,
            downstream_export_dir=downstream_export_dir,
            processed_documents=len(ingest_units),
            provider_meta=provider_meta,
            document_views=document_views,
        )

    def _artifact_paths(self) -> dict[str, Path]:
        return {
            "intake": self.output_dir / "01_intake.jsonl",
            "document_views": self.output_dir / "01c_document_views.jsonl",
            "document_context": self.output_dir / "01b_document_context.jsonl",
            "content_safety": self.output_dir / "02_content_safety.jsonl",
            "content_candidate_windows": self.output_dir / "02a_content_candidate_windows.jsonl",
            "content_fragment_localization": self.output_dir / "02aa_content_fragment_localization.jsonl",
            "content_safety_decisions": self.output_dir / "02b_content_safety_decisions.jsonl",
            "content_safety_audit": self.output_dir / "02c_content_safety_audit.jsonl",
            "content_safety_review_tasks": self.output_dir / "02d_content_safety_review_tasks.jsonl",
            "content_safety_review_results": self.output_dir / "02e_content_safety_review_results.jsonl",
            "content_safety_final_decisions": self.output_dir / "02f_content_safety_final_decisions.jsonl",
            "content_fragment_adjudications": self.output_dir / "02g_content_fragment_adjudications.jsonl",
            "content_document_assessments": self.output_dir / "02h_content_document_assessments.jsonl",
            "privacy": self.output_dir / "03_privacy_detection.jsonl",
            "redaction_plan": self.output_dir / "03b_span_conflict_resolution.jsonl",
            "privacy_decisions": self.output_dir / "03c_privacy_policy_decisions.jsonl",
            "privacy_audit": self.output_dir / "03d_privacy_audit.jsonl",
            "privacy_review_tasks": self.output_dir / "03e_privacy_review_tasks.jsonl",
            "privacy_fragment_adjudications": self.output_dir / "03f_privacy_fragment_adjudications.jsonl",
            "privacy_document_assessments": self.output_dir / "03g_privacy_document_assessments.jsonl",
            "privacy_review_results": self.output_dir / "03h_privacy_review_results.jsonl",
            "privacy_final_decisions": self.output_dir / "03i_privacy_final_decisions.jsonl",
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

    def _artifact_path_metadata(self, artifact_paths: dict[str, Path], downstream_export_dir: Path) -> dict[str, str]:
        return {
            **{name: str(path) for name, path in artifact_paths.items()},
            "downstream_annotation_export_dir": str(downstream_export_dir),
        }

    def _document_views(self, ingest_units: list[Any]) -> list[dict[str, Any]]:
        views: list[dict[str, Any]] = []
        for index, unit in enumerate(ingest_units, start=1):
            text = str(getattr(unit, "text", "") or "")
            if not text:
                continue
            views.append(
                {
                    "doc_id": str(getattr(unit, "doc_id", "") or f"document-{index}"),
                    "source_path": str(getattr(unit, "source_path", "") or ""),
                    "text": text,
                    "original_text": text,
                }
            )
        return views

    def _write_content_safety_governance_artifacts(
        self,
        safety_results: list,
        artifact_paths: dict[str, Path],
        document_contexts: list | None = None,
        content_candidate_windows: list | None = None,
        content_localized_fragments: list | None = None,
        content_fragment_adjudications: list | None = None,
        content_document_assessments: list | None = None,
    ) -> None:
        decision_records, audit_records = self._content_safety_governance_records(
            safety_results,
            document_contexts=document_contexts,
            content_candidate_windows=content_candidate_windows,
            content_localized_fragments=content_localized_fragments,
            content_fragment_adjudications=content_fragment_adjudications,
            content_document_assessments=content_document_assessments,
        )
        document_context_by_doc = {
            getattr(item, "doc_id", ""): item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            for item in document_contexts or []
        }
        fragment_adjudication_by_finding = {
            getattr(item, "finding_id", ""): item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            for item in content_fragment_adjudications or []
            if getattr(item, "finding_id", "")
        }
        document_assessment_by_doc = {
            getattr(item, "doc_id", ""): item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            for item in content_document_assessments or []
        }
        candidate_windows_by_doc = {
            doc_id: [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
                for item in content_candidate_windows or []
                if getattr(item, "doc_id", "") == doc_id
            ]
            for doc_id in {getattr(item, "doc_id", "") for item in content_candidate_windows or []}
        }
        localized_fragment_by_finding: dict[str, dict] = {}
        for result in safety_results:
            for finding in list(getattr(result, "findings", []) or []):
                finding_id = getattr(finding, "finding_id", "")
                localized = dict(getattr(finding, "attributes", {}) or {}).get("localized_fragment", {})
                if finding_id and isinstance(localized, dict):
                    localized_fragment_by_finding[finding_id] = localized
        review_tasks = content_safety_review.build_review_tasks(
            safety_results,
            document_context_by_doc=document_context_by_doc,
            candidate_windows_by_doc=candidate_windows_by_doc,
            localized_fragment_by_finding=localized_fragment_by_finding,
            fragment_adjudication_by_finding=fragment_adjudication_by_finding,
            document_assessment_by_doc=document_assessment_by_doc,
        )
        final_decisions = content_safety_review.build_final_decisions(decision_records, review_tasks)
        write_jsonl(decision_records, artifact_paths["content_safety_decisions"])
        write_jsonl(audit_records, artifact_paths["content_safety_audit"])
        write_jsonl(review_tasks, artifact_paths["content_safety_review_tasks"])
        if not artifact_paths["content_safety_review_results"].exists():
            write_jsonl([], artifact_paths["content_safety_review_results"])
        write_jsonl(final_decisions, artifact_paths["content_safety_final_decisions"])

    def _content_safety_governance_records(
        self,
        safety_results: list,
        document_contexts: list | None = None,
        content_candidate_windows: list | None = None,
        content_localized_fragments: list | None = None,
        content_fragment_adjudications: list | None = None,
        content_document_assessments: list | None = None,
    ) -> tuple[list[dict], list[dict]]:
        document_context_by_doc = {
            getattr(item, "doc_id", ""): item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            for item in document_contexts or []
        }
        fragment_adjudication_by_finding = {
            getattr(item, "finding_id", ""): item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            for item in content_fragment_adjudications or []
            if getattr(item, "finding_id", "")
        }
        document_assessment_by_doc = {
            getattr(item, "doc_id", ""): item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            for item in content_document_assessments or []
        }
        candidate_windows_by_doc: dict[str, list[dict]] = {}
        for item in content_candidate_windows or []:
            candidate_windows_by_doc.setdefault(getattr(item, "doc_id", ""), []).append(
                item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            )
        localized_fragments_by_doc: dict[str, list[dict]] = {}
        localized_fragments_by_window: dict[str, list[dict]] = {}
        for item in content_localized_fragments or []:
            dumped = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            localized_fragments_by_doc.setdefault(getattr(item, "doc_id", ""), []).append(dumped)
            localized_fragments_by_window.setdefault(str(getattr(item, "window_id", "")), []).append(dumped)
        decision_records: list[dict] = []
        audit_records: list[dict] = []
        for result in safety_results:
            doc_id = getattr(result, "doc_id", "")
            document_context = document_context_by_doc.get(doc_id, {})
            document_assessment = document_assessment_by_doc.get(doc_id, {})
            candidate_windows = candidate_windows_by_doc.get(doc_id, [])
            localized_fragments = localized_fragments_by_doc.get(doc_id, [])
            findings = list(getattr(result, "findings", []) or [])
            labels: dict[str, None] = {}
            policy_hits: list[dict] = []
            evidence: list[dict] = []
            decision_paths: list[dict] = []
            risk_level = "C0"
            decision = "P0"
            training = "T0"
            dataset_route = "general_training"
            allow_annotation = True
            needs_manual_review = bool(getattr(result, "needs_adjudication", False))
            confidence = 0.0

            for finding in findings:
                attrs = dict(getattr(finding, "attributes", {}) or {}).get("content_safety", {}) or {}
                policy_tag = getattr(finding, "policy_tag", "")
                matched_label = str(attrs.get("matched_label") or policy_tag)
                labels[matched_label] = None
                labels[policy_tag] = None
                risk_level = self._max_code(risk_level, str(attrs.get("risk_level_code") or "C0"), self._risk_rank())
                decision = self._max_code(decision, str(attrs.get("action") or "P0"), self._decision_rank())
                training = self._max_code(training, str(attrs.get("training_eligibility") or "T0"), self._training_rank())
                if attrs.get("dataset_route"):
                    dataset_route = str(attrs.get("dataset_route"))
                allow_annotation = allow_annotation and bool(attrs.get("allow_downstream_annotation", True))
                needs_manual_review = needs_manual_review or bool(attrs.get("requires_manual_review", False))
                confidence = max(confidence, float(getattr(finding, "confidence", 0.0) or 0.0))
                for hit in attrs.get("policy_hits", []) or []:
                    if isinstance(hit, dict):
                        policy_hits.append(hit)
                for step in attrs.get("decision_path", []) or []:
                    if isinstance(step, dict):
                        decision_paths.append({"finding_id": getattr(finding, "finding_id", ""), **step})

                span = getattr(finding, "span", None)
                localized_fragment = dict(getattr(finding, "attributes", {}) or {}).get("localized_fragment", {}) or {}
                window_id = str(localized_fragment.get("window_id") or attrs.get("candidate_window_id") or "")
                candidate_window = next(
                    (item for item in candidate_windows if str(item.get("window_id") or "") == window_id),
                    {},
                )
                evidence_item = {
                    "finding_id": getattr(finding, "finding_id", ""),
                    "label": matched_label,
                    "risk_type": getattr(finding, "risk_type", ""),
                    "policy_tag": policy_tag,
                    "severity": getattr(getattr(finding, "severity", ""), "value", str(getattr(finding, "severity", ""))),
                    "confidence": getattr(finding, "confidence", 0.0),
                    "text": getattr(span, "text", "") if span else "",
                    "start": getattr(span, "start", None) if span else None,
                    "end": getattr(span, "end", None) if span else None,
                    "explanation": getattr(finding, "explanation", ""),
                    "source": getattr(finding, "source_tool", ""),
                    "candidate_window_id": window_id,
                    "localized_fragment_id": localized_fragment.get("fragment_id", ""),
                }
                evidence.append(evidence_item)
                audit_records.append(
                    {
                        "run_id": getattr(result, "run_id", self.run_id),
                        "doc_id": getattr(result, "doc_id", ""),
                        "text_hash": getattr(result, "text_hash", ""),
                        "finding_id": getattr(finding, "finding_id", ""),
                        "risk_type": getattr(finding, "risk_type", ""),
                        "policy_tag": policy_tag,
                        "matched_label": matched_label,
                        "severity": evidence_item["severity"],
                        "confidence": evidence_item["confidence"],
                        "span": {
                            "start": evidence_item["start"],
                            "end": evidence_item["end"],
                            "text": evidence_item["text"],
                        },
                        "rule_hits": attrs.get("rule_hits", []),
                        "policy_hits": attrs.get("policy_hits", []),
                        "decision_path": attrs.get("decision_path", []),
                        "risk_level_code": attrs.get("risk_level_code", ""),
                        "action": attrs.get("action", ""),
                        "training_eligibility": attrs.get("training_eligibility", ""),
                        "dataset_route": attrs.get("dataset_route", ""),
                        "allow_downstream_annotation": attrs.get("allow_downstream_annotation", True),
                        "requires_manual_review": attrs.get("requires_manual_review", False),
                        "api_context_type": attrs.get("api_context_type", ""),
                        "api_context_rationale": attrs.get("api_context_rationale", ""),
                        "semantic_adjudication": attrs.get("semantic_adjudication", {}),
                        "candidate_window": candidate_window,
                        "localized_fragment": localized_fragment,
                        "semantic_decision": attrs.get("semantic_decision", ""),
                        "semantic_context_type": attrs.get("semantic_context_type", ""),
                        "semantic_reasoning_summary": attrs.get("semantic_reasoning_summary", ""),
                        "document_context": document_context,
                        "fragment_adjudication": fragment_adjudication_by_finding.get(getattr(finding, "finding_id", ""), {}),
                        "document_assessment": document_assessment,
                        "api_payload": dict(getattr(finding, "attributes", {}) or {}).get("api_payload", {}),
                        "decision_engine_version": attrs.get("decision_engine_version", ""),
                        "versions": self._content_safety_versions(attrs),
                    }
                )

            decision_records.append(
                {
                    "run_id": getattr(result, "run_id", self.run_id),
                    "doc_id": getattr(result, "doc_id", ""),
                    "text_hash": getattr(result, "text_hash", ""),
                    "status": getattr(getattr(result, "status", ""), "value", str(getattr(result, "status", ""))),
                    "labels": [label for label in labels if label],
                    "risk_level": risk_level,
                    "decision": decision,
                    "training_eligibility": training,
                    "dataset_route": dataset_route,
                    "allow_downstream_annotation": allow_annotation,
                    "needs_manual_review": needs_manual_review,
                    "confidence": round(confidence, 4),
                    "risk_score": getattr(result, "risk_score", 0.0),
                    "summary": str(document_assessment.get("explanation") or getattr(result, "summary", "")),
                    "policy_hits": self._dedupe_policy_hits(policy_hits),
                    "evidence": evidence,
                    "explanation": {
                        "document_context": document_context,
                        "document_assessment": document_assessment,
                        "candidate_windows": candidate_windows,
                        "localized_fragments": localized_fragments,
                        "decision_path": decision_paths,
                        "hard_case_reasons": getattr(result, "hard_case_reasons", []),
                        "fragment_adjudications": [
                            fragment_adjudication_by_finding.get(getattr(finding, "finding_id", ""), {})
                            for finding in findings
                            if fragment_adjudication_by_finding.get(getattr(finding, "finding_id", ""), {})
                        ],
                        "semantic_adjudications": [
                            {
                                "finding_id": getattr(finding, "finding_id", ""),
                                **(dict(getattr(finding, "attributes", {}) or {}).get("content_safety", {}) or {}).get(
                                    "semantic_adjudication",
                                    {},
                                ),
                            }
                            for finding in findings
                            if (
                                dict(getattr(finding, "attributes", {}) or {}).get("content_safety", {}) or {}
                            ).get("semantic_adjudication")
                        ],
                    },
                    "metadata": {
                        "provider_name": getattr(result, "provider_name", ""),
                        "provider_version": getattr(result, "provider_version", ""),
                        "is_degraded": getattr(result, "is_degraded", False),
                        "document_context": document_context,
                        "document_assessment": document_assessment,
                        "content_candidate_window_count": len(candidate_windows),
                        "content_localized_fragment_count": len(localized_fragments),
                        "candidate_windows": candidate_windows,
                        "localized_fragments": localized_fragments,
                        "versions": self._content_safety_versions({}),
                        "review_task_count": sum(
                            1
                            for finding in findings
                            if (
                                dict(getattr(finding, "attributes", {}) or {}).get("content_safety", {}) or {}
                            ).get("requires_manual_review")
                            or getattr(finding, "needs_adjudication", False)
                        ),
                    },
                }
            )
        return decision_records, audit_records

    def _write_privacy_governance_artifacts(
        self,
        ingest_units: list,
        privacy_results: list,
        redaction_plans: list,
        artifact_paths: dict[str, Path],
        document_contexts: list | None = None,
        privacy_fragment_adjudications: list | None = None,
        privacy_document_assessments: list | None = None,
    ) -> None:
        decision_records, audit_records, review_tasks = self._privacy_governance_records(
            ingest_units,
            privacy_results,
            redaction_plans,
            document_contexts=document_contexts,
            privacy_fragment_adjudications=privacy_fragment_adjudications,
            privacy_document_assessments=privacy_document_assessments,
        )
        write_jsonl(decision_records, artifact_paths["privacy_decisions"])
        write_jsonl(audit_records, artifact_paths["privacy_audit"])
        write_jsonl(review_tasks, artifact_paths["privacy_review_tasks"])
        if not artifact_paths["privacy_review_results"].exists():
            write_jsonl([], artifact_paths["privacy_review_results"])
        final_decisions = privacy_review.build_final_decisions(decision_records, review_tasks)
        write_jsonl(final_decisions, artifact_paths["privacy_final_decisions"])

    def _privacy_governance_records(
        self,
        ingest_units: list,
        privacy_results: list,
        redaction_plans: list,
        document_contexts: list | None = None,
        privacy_fragment_adjudications: list | None = None,
        privacy_document_assessments: list | None = None,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        catalog = load_privacy_entity_catalog(self.settings.privacy_entity_catalog_path)
        policies = load_privacy_policies(self.settings.privacy_policies_path)
        context = dict(self.settings.privacy_metadata or {})
        training_context = dict(self.settings.privacy_training_context or {})
        privacy_by_doc = {getattr(result, "doc_id", ""): result for result in privacy_results}
        redaction_by_finding = self._redaction_by_finding(redaction_plans)
        document_context_by_doc = {
            getattr(item, "doc_id", ""): item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            for item in document_contexts or []
        }
        fragment_adjudication_by_finding = {
            getattr(item, "finding_id", ""): item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            for item in privacy_fragment_adjudications or []
            if getattr(item, "finding_id", "")
        }
        document_assessment_by_doc = {
            getattr(item, "doc_id", ""): item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            for item in privacy_document_assessments or []
        }
        decision_records: list[dict] = []
        audit_records: list[dict] = []
        review_tasks: list[dict] = []

        for unit in ingest_units:
            result = privacy_by_doc.get(getattr(unit, "doc_id", ""))
            findings = list(getattr(result, "findings", []) or []) if result else []
            if redaction_by_finding:
                findings = [
                    finding for finding in findings
                    if getattr(finding, "span", None) is None
                    or getattr(finding, "finding_id", "") in redaction_by_finding
                ]
            finding_decisions: list[dict] = []
            evidence: list[dict] = []
            for finding in findings:
                rule_hit = privacy_rule_hit(finding, catalog, context)
                policy_hits = match_privacy_policy_hits(
                    str(rule_hit.get("entity_type") or finding.risk_type),
                    policies,
                    context,
                    training_context,
                )
                governance = decide_privacy_finding(
                    finding=finding,
                    rule_hit=rule_hit,
                    policy_hits=policy_hits,
                    context=context,
                    training_context=training_context,
                    custom_policy=str(self.settings.privacy_custom_policy or ""),
                    custom_policy_config=self.settings.privacy_custom_policy_config,
                )
                finding_decisions.append(governance)
                span = getattr(finding, "span", None)
                redaction = redaction_by_finding.get(getattr(finding, "finding_id", ""))
                evidence_item = {
                    "finding_id": getattr(finding, "finding_id", ""),
                    "doc_id": getattr(finding, "doc_id", ""),
                    "entity_type": rule_hit.get("entity_type") or getattr(finding, "risk_type", ""),
                    "operator_id": rule_hit.get("operator_id", ""),
                    "operator_name_zh": rule_hit.get("operator_name_zh", ""),
                    "risk_type": getattr(finding, "risk_type", ""),
                    "policy_tag": getattr(finding, "policy_tag", ""),
                    "sensitivity_level": governance.get("sensitivity_level", ""),
                    "training_admissibility": governance.get("training_admissibility", ""),
                    "annotation_admissibility": governance.get("annotation_admissibility", ""),
                    "privacy_action": governance.get("action", ""),
                    "dataset_route": governance.get("dataset_route", ""),
                    "allow_downstream_annotation": governance.get("allow_downstream_annotation", True),
                    "requires_manual_review": governance.get("requires_manual_review", False),
                    "redaction_required": governance.get("redaction_required", False),
                    "replacement": (redaction or {}).get("replacement", getattr(finding, "redaction_suggestion", "")),
                    "severity": getattr(getattr(finding, "severity", ""), "value", str(getattr(finding, "severity", ""))),
                    "confidence": getattr(finding, "confidence", 0.0),
                    "text": getattr(span, "text", "") if span else "",
                    "start": getattr(span, "start", None) if span else None,
                    "end": getattr(span, "end", None) if span else None,
                    "explanation": getattr(finding, "explanation", ""),
                    "reason_zh": governance.get("reason_zh", ""),
                    "source": getattr(finding, "source_tool", ""),
                    "rule_hits": governance.get("rule_hits", []),
                    "policy_hits": governance.get("policy_hits", []),
                    "decision_path": governance.get("decision_path", []),
                    "decision_engine_version": governance.get("decision_engine_version", ""),
                }
                evidence.append(evidence_item)
                audit_records.append(
                    {
                        "run_id": getattr(result, "run_id", self.run_id) if result else self.run_id,
                        "doc_id": getattr(finding, "doc_id", ""),
                        "text_hash": getattr(result, "text_hash", getattr(unit, "text_hash", "")) if result else getattr(unit, "text_hash", ""),
                        **evidence_item,
                        "api_payload": dict(getattr(finding, "attributes", {}) or {}).get("api_payload", {}),
                        "versions": self._privacy_versions(),
                    }
                )
                if governance.get("requires_manual_review") or governance.get("training_admissibility") in {"T2", "T3"}:
                    review_tasks.append(
                        {
                            "review_task_id": f"{self.run_id}-{getattr(finding, 'finding_id', '')}",
                            "run_id": self.run_id,
                            "doc_id": getattr(finding, "doc_id", ""),
                            "finding_id": getattr(finding, "finding_id", ""),
                            "entity_type": rule_hit.get("entity_type") or getattr(finding, "risk_type", ""),
                            "operator_name_zh": rule_hit.get("operator_name_zh", ""),
                            "sensitivity_level": governance.get("sensitivity_level", ""),
                            "training_admissibility": governance.get("training_admissibility", ""),
                            "privacy_action": governance.get("action", ""),
                            "text": getattr(span, "text", "") if span else "",
                            "reason_zh": governance.get("reason_zh", ""),
                            "document_context": document_context_by_doc.get(getattr(finding, "doc_id", ""), {}),
                            "fragment_adjudication": fragment_adjudication_by_finding.get(getattr(finding, "finding_id", ""), {}),
                            "document_assessment": document_assessment_by_doc.get(getattr(finding, "doc_id", ""), {}),
                            "status": "pending",
                        }
                    )

            aggregate = aggregate_privacy_document(getattr(unit, "doc_id", ""), finding_decisions)
            decision_records.append(
                {
                    "run_id": self.run_id,
                    "doc_id": getattr(unit, "doc_id", ""),
                    "text_hash": getattr(unit, "text_hash", ""),
                    **aggregate,
                    "evidence": evidence,
                    "policy_hits": self._dedupe_policy_hits(
                        [
                            hit
                            for item in evidence
                            for hit in item.get("policy_hits", [])
                            if isinstance(hit, dict)
                        ]
                    ),
                    "summary_zh": str(
                        document_assessment_by_doc.get(getattr(unit, "doc_id", ""), {}).get("explanation")
                        or aggregate.get("summary_zh", "")
                    ),
                    "custom_policy": self.settings.privacy_custom_policy,
                    "custom_policy_config": self.settings.privacy_custom_policy_config,
                    "document_context": document_context_by_doc.get(getattr(unit, "doc_id", ""), {}),
                    "document_assessment": document_assessment_by_doc.get(getattr(unit, "doc_id", ""), {}),
                    "fragment_adjudications": [
                        fragment_adjudication_by_finding.get(getattr(finding, "finding_id", ""), {})
                        for finding in findings
                        if fragment_adjudication_by_finding.get(getattr(finding, "finding_id", ""), {})
                    ],
                    "metadata": {
                        "provider_name": getattr(result, "provider_name", "") if result else "",
                        "provider_version": getattr(result, "provider_version", "") if result else "",
                        "is_degraded": getattr(result, "is_degraded", False) if result else False,
                        "document_context": document_context_by_doc.get(getattr(unit, "doc_id", ""), {}),
                        "document_assessment": document_assessment_by_doc.get(getattr(unit, "doc_id", ""), {}),
                        "versions": self._privacy_versions(),
                    },
                }
            )
        return decision_records, audit_records, review_tasks

    def _redaction_by_finding(self, redaction_plans: list) -> dict[str, dict]:
        mapping: dict[str, dict] = {}
        for plan in redaction_plans:
            for target in getattr(plan, "redaction_targets", []) or []:
                finding_id = getattr(target, "finding_id", "")
                if finding_id:
                    mapping[finding_id] = target.model_dump(mode="json") if hasattr(target, "model_dump") else dict(target)
        return mapping

    def _privacy_versions(self) -> dict[str, str]:
        provider_meta = self._provider_metadata()
        return {
            "provider_mode": provider_meta["mode"],
            "provider_model": provider_meta["model"],
            "privacy_prompt": self.settings.api_privacy_detection_prompt_path.name,
            "entity_catalog": self.settings.privacy_entity_catalog_path.name,
            "policy_bundle": self.settings.privacy_policies_path.name,
            "decision_engine": "privacy-decision-v1",
        }

    def _content_safety_versions(self, attrs: dict) -> dict[str, str]:
        provider_meta = self._provider_metadata()
        return {
            "provider_mode": provider_meta["mode"],
            "provider_model": provider_meta["model"],
            "base_prompt": self.settings.api_content_safety_prompt_path.name,
            "semantic_prompt": self.settings.api_content_safety_semantic_prompt_path.name,
            "label_catalog": self.settings.content_safety_labels_path.name,
            "rule_bundle": self.settings.content_rules_path.name,
            "policy_bundle": self.settings.content_safety_policies_path.name,
            "policy_version": str(attrs.get("policy_version") or self.settings.policy_version),
            "decision_engine": str(attrs.get("decision_engine_version") or "content-decision-v2"),
        }

    def _dedupe_policy_hits(self, policy_hits: list[dict]) -> list[dict]:
        seen: set[str] = set()
        deduped: list[dict] = []
        for hit in policy_hits:
            key = str(hit.get("policy_id", "")) + "|" + str(hit.get("reason", ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
        return deduped

    def _max_code(self, left: str, right: str, rank: dict[str, int]) -> str:
        return right if rank.get(right, -1) > rank.get(left, -1) else left

    def _risk_rank(self) -> dict[str, int]:
        return {"C0": 0, "C1": 1, "C2": 2, "C3": 3}

    def _decision_rank(self) -> dict[str, int]:
        return {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5}

    def _training_rank(self) -> dict[str, int]:
        return {"T0": 0, "T1": 1, "T2": 2, "T3": 3}

    def _output_from_decisions(
        self,
        *,
        profile: PipelineProfile,
        decisions: list,
        artifact_paths: dict[str, Path],
        downstream_export_dir: Path,
        processed_documents: int,
        provider_meta: dict[str, str],
        document_views: list[dict[str, Any]],
    ) -> ComplianceOutput:
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
            f"{decision.doc_id}: {decision.explanation or decision.summary}"
            for decision in decisions
            if decision.disposition_level in {DispositionLevel.P2, DispositionLevel.P3, DispositionLevel.P4, DispositionLevel.P5}
        ]
        blocked_count = counts_by_disposition.get("P4", 0) + counts_by_disposition.get("P5", 0)
        review_count = counts_by_disposition.get("P3", 0)
        restricted_count = counts_by_disposition.get("P1", 0) + counts_by_disposition.get("P2", 0)
        explanation_summary = (
            f"Processed {processed_documents} cleaned documents with {provider_meta['label']}. "
            f"{blocked_count} documents were blocked, {review_count} require manual review, "
            f"and {restricted_count} require restricted handling or redaction before normal use."
        )
        artifact_metadata = self._artifact_path_metadata(artifact_paths, downstream_export_dir)
        summary_record = RunSummaryRecord(
            run_id=self.run_id,
            processed_documents=processed_documents,
            overall_disposition=overall_disposition,
            unified_decision=overall_decision,
            trust_level=trust_level,
            counts_by_disposition=counts_by_disposition,
            counts_by_decision=counts_by_decision,
            artifact_paths=artifact_metadata,
            review_suggestions=review_suggestions[:20],
            explanation_summary=explanation_summary,
            metadata={
                "execution_mode": provider_meta["mode"],
                "pipeline_profile": profile.value,
                "provider_base_url": provider_meta["base_url"],
                "provider_model": provider_meta["model"],
                "api_base_url": self.settings.api_compliance_base_url,
                "api_model": self.settings.api_compliance_model,
                "local_base_url": self.settings.local_compliance_base_url,
                "local_model": self.settings.local_compliance_model,
            },
        )
        write_single_jsonl(summary_record, artifact_paths["summary"])

        return ComplianceOutput(
            pipeline_run_id=self.run_id,
            modality=Modality.TEXT,
            decision=overall_decision,
            trust_level=trust_level,
            annotation_package_uri=str(artifact_paths["annotation"]),
            audit_package_uri=str(artifact_paths["audit"]),
            degrade_summary="" if trust_level == TrustLevel.FULL else "API compliance operators used degraded fallback handling.",
            review_suggestions=summary_record.review_suggestions,
            explanation_summary=explanation_summary,
            legacy_decision={
                "overall_disposition": overall_disposition.value,
                "overall_decision": overall_decision.value,
                "counts_by_disposition": counts_by_disposition,
                "counts_by_decision": counts_by_decision,
                "documents": [decision.model_dump(mode="json") for decision in decisions],
            },
            metadata={
                "artifact_paths": artifact_metadata,
                "execution_mode": provider_meta["mode"],
                "pipeline_profile": profile.value,
                "document_views": document_views,
            },
        )

    def _output_from_partial_findings(
        self,
        *,
        profile: PipelineProfile,
        ingest_units: list,
        artifact_paths: dict[str, Path],
        downstream_export_dir: Path,
        provider_meta: dict[str, str],
        safety_results: list,
        privacy_results: list,
    ) -> ComplianceOutput:
        decision = UnifiedDecision.ALLOW
        review_suggestions: list[str] = []

        if safety_results:
            for result in safety_results:
                if result.status.value == "flagged":
                    decision = UnifiedDecision.REJECT
                elif result.status.value == "hard_case" and decision != UnifiedDecision.REJECT:
                    decision = UnifiedDecision.REVIEW
                if result.status.value in {"flagged", "hard_case"}:
                    review_suggestions.append(f"{result.doc_id}: {result.summary}")

        if privacy_results:
            privacy_hits = [result for result in privacy_results if result.findings]
            if privacy_hits and decision == UnifiedDecision.ALLOW:
                decision = UnifiedDecision.QUARANTINE
            for result in privacy_hits:
                review_suggestions.append(f"{result.doc_id}: {result.summary}")

        disposition = self._disposition_for_partial_decision(decision)
        counts_by_disposition = {disposition.value: len(ingest_units)} if ingest_units else {}
        counts_by_decision = {decision.value: len(ingest_units)} if ingest_units else {}
        explanation_summary = (
            f"Processed {len(ingest_units)} cleaned documents with {provider_meta['label']} "
            f"for the {profile.value} workflow."
        )
        artifact_metadata = self._artifact_path_metadata(artifact_paths, downstream_export_dir)
        summary_record = RunSummaryRecord(
            run_id=self.run_id,
            processed_documents=len(ingest_units),
            overall_disposition=disposition,
            unified_decision=decision,
            trust_level=TrustLevel.FULL,
            counts_by_disposition=counts_by_disposition,
            counts_by_decision=counts_by_decision,
            artifact_paths=artifact_metadata,
            review_suggestions=review_suggestions[:20],
            explanation_summary=explanation_summary,
            metadata={
                "execution_mode": provider_meta["mode"],
                "pipeline_profile": profile.value,
                "provider_base_url": provider_meta["base_url"],
                "provider_model": provider_meta["model"],
                "api_base_url": self.settings.api_compliance_base_url,
                "api_model": self.settings.api_compliance_model,
                "local_base_url": self.settings.local_compliance_base_url,
                "local_model": self.settings.local_compliance_model,
            },
        )
        write_single_jsonl(summary_record, artifact_paths["summary"])

        return ComplianceOutput(
            pipeline_run_id=self.run_id,
            modality=Modality.TEXT,
            decision=decision,
            trust_level=TrustLevel.FULL,
            annotation_package_uri=str(artifact_paths["annotation"]),
            audit_package_uri=str(artifact_paths["audit"]),
            review_suggestions=summary_record.review_suggestions,
            explanation_summary=explanation_summary,
            legacy_decision={
                "overall_disposition": disposition.value,
                "overall_decision": decision.value,
                "counts_by_disposition": counts_by_disposition,
                "counts_by_decision": counts_by_decision,
                "documents": [],
            },
            metadata={
                "artifact_paths": artifact_metadata,
                "execution_mode": provider_meta["mode"],
                "pipeline_profile": profile.value,
                "document_views": [],
            },
        )

    def _empty_output(self, artifact_paths: dict[str, Path], profile: PipelineProfile, provider_meta: dict[str, str]) -> ComplianceOutput:
        downstream_export_dir = self.output_dir / "10_annotation_exports"
        artifact_metadata = self._artifact_path_metadata(artifact_paths, downstream_export_dir)
        summary = RunSummaryRecord(
            run_id=self.run_id,
            processed_documents=0,
            overall_disposition=DispositionLevel.P0,
            unified_decision=UnifiedDecision.ALLOW,
            trust_level=TrustLevel.FULL,
            artifact_paths=artifact_metadata,
            explanation_summary="No cleaned documents were discovered in the supplied package paths.",
            metadata={
                "execution_mode": provider_meta["mode"],
                "pipeline_profile": profile.value,
                "provider_base_url": provider_meta["base_url"],
                "provider_model": provider_meta["model"],
                "api_base_url": self.settings.api_compliance_base_url,
                "api_model": self.settings.api_compliance_model,
                "local_base_url": self.settings.local_compliance_base_url,
                "local_model": self.settings.local_compliance_model,
            },
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
            metadata={
                "artifact_paths": artifact_metadata,
                "execution_mode": provider_meta["mode"],
                "pipeline_profile": profile.value,
            },
        )

    def _provider_metadata(self) -> dict[str, str]:
        try:
            provider = resolve_provider_config(self.settings)
            label = "local compliance models" if provider.mode == "local_model" else "API compliance operators"
            return {
                "mode": provider.mode,
                "base_url": provider.base_url,
                "model": provider.model,
                "label": label,
            }
        except Exception:
            return {
                "mode": "heuristic_only",
                "base_url": "",
                "model": "",
                "label": "heuristic compliance operators",
            }

    def _build_document_contexts(self, ingest_units: list, provider_meta: dict[str, str]) -> list:
        if provider_meta["mode"] != "local_model":
            return []
        return b_document_context.run(ingest_units, self.settings)

    def _normalize_profile(self, profile: str | PipelineProfile) -> PipelineProfile:
        if isinstance(profile, PipelineProfile):
            return profile
        try:
            return PipelineProfile(str(profile).strip().lower())
        except ValueError:
            logger.warning("Unknown API pipeline profile '%s', fallback to full", profile)
            return PipelineProfile.FULL

    def _disposition_for_partial_decision(self, decision: UnifiedDecision) -> DispositionLevel:
        if decision == UnifiedDecision.REJECT:
            return DispositionLevel.P4
        if decision == UnifiedDecision.REVIEW:
            return DispositionLevel.P3
        if decision == UnifiedDecision.QUARANTINE:
            return DispositionLevel.P1
        return DispositionLevel.P0
