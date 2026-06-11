from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.enums import UnifiedDecision
from text.config.settings import Settings, get_settings
from text.models.schemas import (
    AnnotationPackageRecord,
    AuditPackageRecord,
    DeliveryStatus,
    DispositionLevel,
    IngestUnit,
    RedactionTarget,
)

logger = logging.getLogger(__name__)

TEXT_TOOL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "tool_name": "text_label_agent",
        "endpoint": "/api/v1/text/label-agent/jobs",
        "filename": "text_label_agent_request.json",
        "extra": {"labels": None},
    },
    {
        "tool_name": "text_ner_prelabel_agent",
        "endpoint": "/api/v1/text/ner-prelabel-agent/jobs",
        "filename": "text_ner_prelabel_agent_request.json",
        "extra": {"labels": None},
    },
    {
        "tool_name": "text_re_agent",
        "endpoint": "/api/v1/text/re-agent/jobs",
        "filename": "text_re_agent_request.json",
        "extra": {"entity_types": None, "relation_types": None},
    },
    {
        "tool_name": "text_sequence_label_agent",
        "endpoint": "/api/v1/text/sequence-label-agent/jobs",
        "filename": "text_sequence_label_agent_request.json",
        "extra": {"ner_labels": None, "pos_labels": None},
    },
    {
        "tool_name": "text_intent_slot_agent",
        "endpoint": "/api/v1/text/intent-slot-agent/jobs",
        "filename": "text_intent_slot_agent_request.json",
        "extra": {"intents": None, "slot_labels": None},
    },
    {
        "tool_name": "text_sentiment_emotion_agent",
        "endpoint": "/api/v1/text/sentiment-emotion-agent/jobs",
        "filename": "text_sentiment_emotion_agent_request.json",
        "extra": {
            "sentiment_labels": ["positive", "negative", "neutral"],
            "emotion_labels": ["喜悦", "愤怒", "悲伤", "恐惧", "厌恶", "惊讶", "期待"],
        },
    },
    {
        "tool_name": "text_summary_keyword_agent",
        "endpoint": "/api/v1/text/summary-keyword-agent/jobs",
        "filename": "text_summary_keyword_agent_request.json",
        "extra": {"keywords": None},
    },
)

DOCUMENT_TOOL_SPEC = {
    "tool_name": "document_annotation_agent",
    "endpoint": "/api/v1/document/annotation-agent/jobs",
    "filename": "document_annotation_agent_request.json",
    "extra": {"labels": None},
}

DOCUMENT_FILE_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".doc",
    ".docx",
    ".html",
    ".htm",
    ".md",
    ".rtf",
}


@dataclass(frozen=True)
class DownstreamAnnotationExport:
    request_records: list[dict[str, Any]]
    mapping_records: list[dict[str, Any]]
    manifest_records: list[dict[str, Any]]


def _full_endpoint(settings: Settings, endpoint: str) -> str:
    base_url = settings.downstream_annotation_base_url.strip().rstrip("/")
    if not base_url:
        return endpoint
    return f"{base_url}{endpoint}"


def _dataset_name(run_id: str, settings: Settings) -> str:
    configured = settings.downstream_annotation_dataset_name.strip()
    if configured:
        return configured
    return f"text-compliance-{run_id or 'run'}"


def _should_export(record: AnnotationPackageRecord, settings: Settings) -> tuple[bool, str]:
    if record.delivery_status == DeliveryStatus.BLOCK:
        return False, "blocked_by_compliance_policy"
    if record.disposition_level in {DispositionLevel.P4, DispositionLevel.P5}:
        return False, "high_risk_content_not_exported"
    if record.delivery_status == DeliveryStatus.HOLD and not settings.downstream_annotation_include_hold:
        return False, "hold_records_disabled"
    return True, ""


def _select_annotation_text(record: AnnotationPackageRecord, settings: Settings) -> tuple[str, str]:
    mode = settings.downstream_annotation_text_mode.strip().lower()
    if mode not in {"graded", "original", "redacted"}:
        mode = "graded"

    if mode == "redacted":
        return record.redacted_view, "redacted_view"

    if mode == "original":
        if (
            record.delivery_status == DeliveryStatus.HOLD
            and not settings.downstream_annotation_trusted_original_for_hold
        ):
            return record.redacted_view, "redacted_view"
        return record.original_text, "original_text"

    if record.disposition_level in {DispositionLevel.P0, DispositionLevel.P1}:
        return record.original_text, "original_text"
    if settings.downstream_annotation_trusted_original_for_hold:
        return record.original_text, "original_text"
    return record.redacted_view, "redacted_view"


def _audit_by_doc(audit_records: list[AuditPackageRecord] | None) -> dict[str, AuditPackageRecord]:
    return {record.doc_id: record for record in audit_records or []}


def _pii_types(record: AnnotationPackageRecord, audit: AuditPackageRecord | None) -> list[str]:
    values = {target.pii_type for target in record.span_annotations if target.pii_type}
    if audit and audit.privacy_result:
        values.update(finding.risk_type for finding in audit.privacy_result.findings if finding.risk_type)
    return sorted(values)


def _content_risk_tags(audit: AuditPackageRecord | None) -> list[str]:
    if not audit or not audit.safety_result:
        return []
    return sorted({finding.policy_tag for finding in audit.safety_result.findings if finding.policy_tag})


def _content_risk_types(audit: AuditPackageRecord | None) -> list[str]:
    if not audit or not audit.safety_result:
        return []
    return sorted({finding.risk_type for finding in audit.safety_result.findings if finding.risk_type})


def _span_payloads(spans: list[RedactionTarget]) -> list[dict[str, Any]]:
    return [
        {
            "finding_id": span.finding_id,
            "event_id": span.event_id,
            "start": span.start,
            "end": span.end,
            "pii_type": span.pii_type,
            "replacement": span.replacement,
        }
        for span in spans
    ]


def _text_item(
    record: AnnotationPackageRecord,
    audit: AuditPackageRecord | None,
    settings: Settings,
) -> dict[str, Any]:
    text, text_source = _select_annotation_text(record, settings)
    pii_types = _pii_types(record, audit)
    content_tags = _content_risk_tags(audit)
    content_types = _content_risk_types(audit)
    contains_pii = bool(pii_types)
    contains_content_safety_risk = bool(content_tags)

    return {
        "source_doc_id": record.doc_id,
        "annotation_text": text,
        "annotation_text_source": text_source,
        "offset_base": text_source,
        "delivery_status": record.delivery_status.value,
        "disposition_level": record.disposition_level.value,
        "unified_decision": record.unified_decision.value
        if isinstance(record.unified_decision, UnifiedDecision)
        else str(record.unified_decision),
        "review_required": record.delivery_status == DeliveryStatus.HOLD,
        "review_priority": record.review_priority,
        "contains_pii": contains_pii,
        "contains_content_safety_risk": contains_content_safety_risk,
        "pii_types": pii_types,
        "content_risk_types": content_types,
        "content_risk_tags": content_tags,
        "pii_spans": _span_payloads(record.span_annotations),
        "post_annotation_compliance_required": (
            contains_pii or contains_content_safety_risk or record.delivery_status != DeliveryStatus.DELIVER
        ),
        "metadata": dict(record.metadata),
    }


def _request_record(
    *,
    run_id: str,
    tool_name: str,
    endpoint: str,
    request_path: Path | None,
    request_body: dict[str, Any] | None,
    item_count: int,
    status: str,
    skipped_reason: str = "",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "tool_name": tool_name,
        "endpoint": endpoint,
        "request_path": str(request_path) if request_path else "",
        "request_body": request_body,
        "item_count": item_count,
        "status": status,
        "skipped_reason": skipped_reason,
    }


def _write_request_json(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _candidate_path_values(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _document_file_paths(unit: IngestUnit | None) -> list[str]:
    if unit is None:
        return []

    candidates: list[str] = []
    for source in (unit.metadata, unit.extensions):
        for key in (
            "file_path",
            "file_paths",
            "document_path",
            "document_paths",
            "original_file_path",
            "original_file_paths",
            "raw_file_path",
            "raw_file_paths",
            "source_file_path",
            "source_file_paths",
        ):
            candidates.extend(_candidate_path_values(source.get(key)))

    candidates.extend(asset.uri for asset in unit.raw_text_refs)
    eligible: list[str] = []
    for raw_path in candidates:
        suffix = Path(raw_path).suffix.lower()
        if suffix in DOCUMENT_FILE_SUFFIXES:
            eligible.append(raw_path)
    return sorted(dict.fromkeys(eligible))


def _build_text_requests(
    *,
    run_id: str,
    dataset_name: str,
    text_items: list[dict[str, Any]],
    export_dir: Path,
    settings: Settings,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    request_records: list[dict[str, Any]] = []
    mapping_records: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    texts = [item["annotation_text"] for item in text_items]

    for spec in TEXT_TOOL_SPECS:
        tool_name = str(spec["tool_name"])
        endpoint = _full_endpoint(settings, str(spec["endpoint"]))
        request_path = export_dir / str(spec["filename"])
        if not text_items:
            request_records.append(
                _request_record(
                    run_id=run_id,
                    tool_name=tool_name,
                    endpoint=endpoint,
                    request_path=None,
                    request_body=None,
                    item_count=0,
                    status="skipped",
                    skipped_reason="no_eligible_text_records",
                )
            )
            manifest_records.append(
                {
                    "run_id": run_id,
                    "tool_name": tool_name,
                    "status": "skipped",
                    "item_count": 0,
                    "request_path": "",
                    "skipped_reason": "no_eligible_text_records",
                }
            )
            continue

        body = {"dataset_name": dataset_name, "texts": texts}
        body.update(dict(spec.get("extra", {})))
        _write_request_json(request_path, body)

        request_records.append(
            _request_record(
                run_id=run_id,
                tool_name=tool_name,
                endpoint=endpoint,
                request_path=request_path,
                request_body=body,
                item_count=len(text_items),
                status="ready",
            )
        )
        manifest_records.append(
            {
                "run_id": run_id,
                "tool_name": tool_name,
                "status": "ready",
                "item_count": len(text_items),
                "request_path": str(request_path),
                "endpoint": endpoint,
            }
        )

        for index, item in enumerate(text_items):
            mapping_records.append(
                {
                    "run_id": run_id,
                    "tool_name": tool_name,
                    "downstream_text_id": f"text_{index}",
                    "source_doc_id": item["source_doc_id"],
                    "annotation_text_source": item["annotation_text_source"],
                    "offset_base": item["offset_base"],
                    "delivery_status": item["delivery_status"],
                    "disposition_level": item["disposition_level"],
                    "unified_decision": item["unified_decision"],
                    "review_required": item["review_required"],
                    "contains_pii": item["contains_pii"],
                    "contains_content_safety_risk": item["contains_content_safety_risk"],
                    "pii_types": item["pii_types"],
                    "content_risk_types": item["content_risk_types"],
                    "content_risk_tags": item["content_risk_tags"],
                    "pii_spans": item["pii_spans"],
                    "post_annotation_compliance_required": item["post_annotation_compliance_required"],
                    "metadata": item["metadata"],
                }
            )

    return request_records, mapping_records, manifest_records


def _build_document_request(
    *,
    run_id: str,
    dataset_name: str,
    records: list[AnnotationPackageRecord],
    units_by_doc: dict[str, IngestUnit],
    export_dir: Path,
    settings: Settings,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    spec = DOCUMENT_TOOL_SPEC
    tool_name = str(spec["tool_name"])
    endpoint = _full_endpoint(settings, str(spec["endpoint"]))
    file_entries: list[tuple[str, AnnotationPackageRecord]] = []
    mapping_records: list[dict[str, Any]] = []

    for record in records:
        unit = units_by_doc.get(record.doc_id)
        for file_path in _document_file_paths(unit):
            if file_path not in {entry[0] for entry in file_entries}:
                file_entries.append((file_path, record))

    file_paths = [file_path for file_path, _ in file_entries]
    for index, (file_path, record) in enumerate(file_entries):
        mapping_records.append(
            {
                "run_id": run_id,
                "tool_name": tool_name,
                "downstream_file_id": f"file_{index}",
                "source_doc_id": record.doc_id,
                "source_file_path": file_path,
                "delivery_status": record.delivery_status.value,
                "disposition_level": record.disposition_level.value,
                "unified_decision": record.unified_decision.value
                if isinstance(record.unified_decision, UnifiedDecision)
                else str(record.unified_decision),
                "review_required": record.delivery_status == DeliveryStatus.HOLD,
                "contains_pii": bool(record.span_annotations),
                "pii_spans": _span_payloads(record.span_annotations),
                "post_annotation_compliance_required": bool(record.span_annotations)
                or record.delivery_status != DeliveryStatus.DELIVER,
                "metadata": dict(record.metadata),
            }
        )
    if not file_paths:
        request_record = _request_record(
            run_id=run_id,
            tool_name=tool_name,
            endpoint=endpoint,
            request_path=None,
            request_body=None,
            item_count=0,
            status="skipped",
            skipped_reason="no_eligible_document_file_paths",
        )
        manifest_record = {
            "run_id": run_id,
            "tool_name": tool_name,
            "status": "skipped",
            "item_count": 0,
            "request_path": "",
            "skipped_reason": "no_eligible_document_file_paths",
        }
        return request_record, [], manifest_record

    body = {"dataset_name": dataset_name, "file_paths": file_paths}
    body.update(dict(spec["extra"]))
    request_path = export_dir / str(spec["filename"])
    _write_request_json(request_path, body)
    request_record = _request_record(
        run_id=run_id,
        tool_name=tool_name,
        endpoint=endpoint,
        request_path=request_path,
        request_body=body,
        item_count=len(file_paths),
        status="ready",
    )
    manifest_record = {
        "run_id": run_id,
        "tool_name": tool_name,
        "status": "ready",
        "item_count": len(file_paths),
        "request_path": str(request_path),
        "endpoint": endpoint,
    }
    return request_record, mapping_records, manifest_record


def run(
    annotation_records: list[AnnotationPackageRecord],
    ingest_units: list[IngestUnit],
    audit_records: list[AuditPackageRecord] | None = None,
    export_dir: Path | None = None,
    settings: Settings | None = None,
) -> DownstreamAnnotationExport:
    settings = settings or get_settings()
    run_id = annotation_records[0].run_id if annotation_records else ""
    dataset_name = _dataset_name(run_id, settings)
    export_dir = export_dir or settings.work_dir / run_id / "10_annotation_exports"
    audit_lookup = _audit_by_doc(audit_records)
    units_by_doc = {unit.doc_id: unit for unit in ingest_units}

    eligible_records: list[AnnotationPackageRecord] = []
    skipped_records: list[dict[str, Any]] = []
    text_items: list[dict[str, Any]] = []

    for record in annotation_records:
        should_export, reason = _should_export(record, settings)
        if not should_export:
            skipped_records.append(
                {
                    "run_id": record.run_id,
                    "doc_id": record.doc_id,
                    "delivery_status": record.delivery_status.value,
                    "disposition_level": record.disposition_level.value,
                    "skipped_reason": reason,
                }
            )
            continue
        eligible_records.append(record)
        text_items.append(_text_item(record, audit_lookup.get(record.doc_id), settings))

    request_records, mapping_records, manifest_records = _build_text_requests(
        run_id=run_id,
        dataset_name=dataset_name,
        text_items=text_items,
        export_dir=export_dir,
        settings=settings,
    )
    document_request, document_mappings, document_manifest = _build_document_request(
        run_id=run_id,
        dataset_name=dataset_name,
        records=eligible_records,
        units_by_doc=units_by_doc,
        export_dir=export_dir,
        settings=settings,
    )
    request_records.append(document_request)
    mapping_records.extend(document_mappings)
    manifest_records.append(document_manifest)

    manifest_records.append(
        {
            "run_id": run_id,
            "tool_name": "downstream_annotation_export",
            "status": "completed",
            "eligible_record_count": len(eligible_records),
            "skipped_record_count": len(skipped_records),
            "skipped_records": skipped_records,
            "text_mode": settings.downstream_annotation_text_mode,
            "include_hold": settings.downstream_annotation_include_hold,
            "trusted_original_for_hold": settings.downstream_annotation_trusted_original_for_hold,
            "export_dir": str(export_dir),
        }
    )

    logger.info(
        "Downstream annotation export completed: %d request records, %d mapping records",
        len(request_records),
        len(mapping_records),
    )
    return DownstreamAnnotationExport(
        request_records=request_records,
        mapping_records=mapping_records,
        manifest_records=manifest_records,
    )
