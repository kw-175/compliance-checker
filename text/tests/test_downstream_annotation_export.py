from __future__ import annotations

from pathlib import Path

from common.enums import UnifiedDecision
from text.config.settings import Settings
from text.models.schemas import (
    AnnotationPackageRecord,
    DeliveryStatus,
    DispositionLevel,
    IngestUnit,
    RedactionTarget,
)
from text.steps import downstream_annotation_export


def test_downstream_annotation_export_builds_tool_requests_and_mapping(tmp_path: Path) -> None:
    settings = Settings(work_dir=tmp_path, downstream_annotation_base_url="http://127.0.0.1:8100")
    records = [
        AnnotationPackageRecord(
            run_id="run-export",
            doc_id="safe-doc",
            original_text="safe text",
            redacted_view="safe text",
            delivery_status=DeliveryStatus.DELIVER,
            disposition_level=DispositionLevel.P0,
            unified_decision=UnifiedDecision.ALLOW,
        ),
        AnnotationPackageRecord(
            run_id="run-export",
            doc_id="hold-doc",
            original_text="Name: Alice Phone: 13800138000",
            redacted_view="Name: <PERSON> Phone: <PHONE>",
            delivery_status=DeliveryStatus.HOLD,
            disposition_level=DispositionLevel.P3,
            unified_decision=UnifiedDecision.REVIEW,
            span_annotations=[
                RedactionTarget(
                    finding_id="finding-1",
                    event_id="event-1",
                    start=6,
                    end=11,
                    original_text="Alice",
                    replacement="<PERSON>",
                    pii_type="person_name",
                )
            ],
        ),
        AnnotationPackageRecord(
            run_id="run-export",
            doc_id="blocked-doc",
            original_text="unsafe text",
            redacted_view="unsafe text",
            delivery_status=DeliveryStatus.BLOCK,
            disposition_level=DispositionLevel.P4,
            unified_decision=UnifiedDecision.REJECT,
        ),
    ]
    units = [
        IngestUnit(
            run_id="run-export",
            package_id="pkg",
            doc_id=record.doc_id,
            source_path="memory",
            text=record.original_text,
            text_hash=record.doc_id,
        )
        for record in records
    ]

    export = downstream_annotation_export.run(records, units, [], tmp_path / "exports", settings)
    requests_by_tool = {record["tool_name"]: record for record in export.request_records}
    label_request = requests_by_tool["text_label_agent"]

    assert label_request["status"] == "ready"
    assert label_request["endpoint"] == "http://127.0.0.1:8100/api/v1/text/label-agent/jobs"
    assert label_request["request_body"]["texts"] == ["safe text", "Name: <PERSON> Phone: <PHONE>"]
    assert (tmp_path / "exports" / "text_label_agent_request.json").exists()

    label_mappings = [item for item in export.mapping_records if item["tool_name"] == "text_label_agent"]
    assert [item["source_doc_id"] for item in label_mappings] == ["safe-doc", "hold-doc"]
    assert label_mappings[0]["annotation_text_source"] == "original_text"
    assert label_mappings[1]["annotation_text_source"] == "redacted_view"
    assert label_mappings[1]["post_annotation_compliance_required"] is True

    summary = [item for item in export.manifest_records if item["tool_name"] == "downstream_annotation_export"][0]
    assert summary["skipped_records"][0]["doc_id"] == "blocked-doc"
