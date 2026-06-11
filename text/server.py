from __future__ import annotations

import logging
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from text.config.settings import get_settings
from text.jsonl_utils import read_jsonl, write_jsonl
from text.models.schemas import CheckRequest, CheckTaskInfo, TaskStatus
from text.pipeline import CompliancePipeline
from text.steps import a_source_intake, f_privacy_detection, g_safety_moderation, span_conflict_resolution

logger = logging.getLogger(__name__)

_tasks: dict[str, dict[str, Any]] = {}

PLATFORM_OPERATOR_NAMES = {
    "CMP_001": "Sensitive information detection",
    "CMP_002": "Content safety detection",
}

PLATFORM_ARTIFACTS = {
    "intake": "01_intake.jsonl",
    "document_views": "01c_document_views.jsonl",
    "content_safety": "02_content_safety.jsonl",
    "privacy_detection": "03_privacy_detection.jsonl",
    "redaction_plan": "03b_span_conflict_resolution.jsonl",
    "annotation_package": "07_annotation_package.jsonl",
    "report": "platform_report.json",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    yield


app = FastAPI(
    title="Text Cleaned-Package Compliance Checker",
    description="Accepts cleaned data packages, runs JSONL-native compliance detection, and returns annotation/audit package URIs.",
    version="0.2.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _status_value(status: TaskStatus | str) -> str:
    return status.value if isinstance(status, TaskStatus) else str(status)


def _safe_filename(filename: str | None) -> str:
    name = Path(filename or "input.txt").name
    return name or "input.txt"


def _parse_config(config: str | None) -> dict[str, Any]:
    if not config or not config.strip():
        return {}
    try:
        payload = json.loads(config)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid config JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="config must be a JSON object")
    return payload


def _settings_with_overrides(config_overrides: dict[str, Any]):
    settings = get_settings()
    if not config_overrides:
        return settings
    valid_overrides = {key: value for key, value in config_overrides.items() if hasattr(settings, key)}
    for key, value in list(valid_overrides.items()):
        if key.endswith("_path") or key in {"work_dir", "upload_dir"}:
            valid_overrides[key] = Path(value)
    return settings.model_copy(update=valid_overrides)


async def _require_platform_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    expected = get_settings().platform_api_key
    if not expected:
        return
    bearer = ""
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()
    if expected not in {bearer, x_api_key or ""}:
        raise HTTPException(status_code=401, detail="Invalid platform API key")


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {name: output_dir / filename for name, filename in PLATFORM_ARTIFACTS.items()}


def _read_artifact(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _apply_redactions(text: str, targets: list[dict[str, Any]]) -> str:
    redacted = text
    for target in sorted(targets, key=lambda item: int(item.get("start", 0)), reverse=True):
        start = int(target.get("start", 0))
        end = int(target.get("end", 0))
        replacement = str(target.get("replacement") or "<REDACTED>")
        if 0 <= start < end <= len(redacted):
            redacted = redacted[:start] + replacement + redacted[end:]
    return redacted


def _severity_rank(value: str) -> int:
    return {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(value, 0)


def _risk_from_findings(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return "none"
    highest = max((_severity_rank(str(item.get("risk_level") or item.get("severity") or "")) for item in findings), default=0)
    return {1: "low", 2: "medium", 3: "high", 4: "critical"}.get(highest, "medium")


def _decision_from_risk(risk_level: str, total_findings: int) -> str:
    if total_findings <= 0:
        return "passed"
    if risk_level in {"critical", "high"}:
        return "failed"
    return "review"


def _finding_span(finding: dict[str, Any]) -> dict[str, Any]:
    span = finding.get("span") if isinstance(finding.get("span"), dict) else {}
    return {
        "start": span.get("start"),
        "end": span.get("end"),
        "text": span.get("text", ""),
        "context_before": span.get("context_before", ""),
        "context_after": span.get("context_after", ""),
    }


def _build_redaction_views(ingest_units: list[Any], redaction_plans: list[Any]) -> list[dict[str, Any]]:
    plans_by_doc = {item.doc_id: item for item in redaction_plans}
    views: list[dict[str, Any]] = []
    for unit in ingest_units:
        plan = plans_by_doc.get(unit.doc_id)
        targets = [target.model_dump(mode="json") for target in (plan.redaction_targets if plan else [])]
        views.append(
            {
                "doc_id": unit.doc_id,
                "source_path": unit.source_path,
                "original_text": unit.text,
                "redacted_text": _apply_redactions(unit.text, targets),
                "redaction_targets": targets,
                "conflicts": [conflict.model_dump(mode="json") for conflict in (plan.conflicts if plan else [])],
            }
        )
    return views


def _document_views(ingest_units: list[Any]) -> list[dict[str, Any]]:
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


def _build_privacy_report(
    *,
    job_id: str,
    operator_id: str,
    dataset_name: str,
    ingest_units: list[Any],
    privacy_results: list[Any],
    redaction_plans: list[Any],
    redaction_views: list[dict[str, Any]],
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    redaction_by_finding: dict[str, dict[str, Any]] = {}
    for plan in redaction_plans:
        for target in plan.redaction_targets:
            redaction_by_finding[target.finding_id] = target.model_dump(mode="json")

    summary: dict[str, int] = {}
    findings: list[dict[str, Any]] = []
    for result in privacy_results:
        for finding in result.findings:
            span = finding.span.model_dump(mode="json") if finding.span else {}
            target = redaction_by_finding.get(finding.finding_id, {})
            summary[finding.risk_type] = summary.get(finding.risk_type, 0) + 1
            findings.append(
                {
                    "finding_id": finding.finding_id,
                    "doc_id": finding.doc_id,
                    "type": finding.risk_type,
                    "policy_tag": finding.policy_tag,
                    "risk_level": finding.severity.value,
                    "confidence": finding.confidence,
                    "text": span.get("text", ""),
                    "start": span.get("start"),
                    "end": span.get("end"),
                    "context_before": span.get("context_before", ""),
                    "context_after": span.get("context_after", ""),
                    "replacement": target.get("replacement") or finding.redaction_suggestion,
                    "suggestion": finding.remediation_suggestion,
                    "source_tool": finding.source_tool,
                    "explanation": finding.explanation,
                }
            )

    risk_level = _risk_from_findings(findings)
    total_findings = len(findings)
    return {
        "job_id": job_id,
        "operator_id": operator_id,
        "operator_name": PLATFORM_OPERATOR_NAMES[operator_id],
        "dataset_name": dataset_name,
        "conclusion": _decision_from_risk(risk_level, total_findings),
        "risk_level": risk_level,
        "is_compliant": total_findings == 0,
        "total_documents": len(ingest_units),
        "total_findings": total_findings,
        "summary": summary,
        "findings": findings,
        "document_views": _document_views(ingest_units),
        "redaction_views": redaction_views,
        "review_suggestions": [
            "Sensitive information was found. Review redaction views before publishing."
        ] if total_findings else [],
        "artifact_paths": {name: str(path) for name, path in artifacts.items()},
        "raw_artifacts": {
            "privacy_detection": _read_artifact(artifacts["privacy_detection"]),
            "redaction_plan": _read_artifact(artifacts["redaction_plan"]),
            "annotation_package": _read_artifact(artifacts["annotation_package"]),
        },
    }


def _build_content_report(
    *,
    job_id: str,
    operator_id: str,
    dataset_name: str,
    ingest_units: list[Any],
    safety_results: list[Any],
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    status_summary = {"safe": 0, "controversial": 0, "unsafe": 0}
    risk_summary: dict[str, int] = {}
    findings: list[dict[str, Any]] = []

    for result in safety_results:
        if result.status.value == "clear":
            status_summary["safe"] += 1
        elif result.status.value == "hard_case":
            status_summary["controversial"] += 1
        else:
            status_summary["unsafe"] += 1

        for finding in result.findings:
            span = finding.span.model_dump(mode="json") if finding.span else {}
            risk_summary[finding.risk_type] = risk_summary.get(finding.risk_type, 0) + 1
            findings.append(
                {
                    "finding_id": finding.finding_id,
                    "doc_id": finding.doc_id,
                    "category": result.status.value,
                    "risk_type": finding.risk_type,
                    "policy_tag": finding.policy_tag,
                    "risk_level": finding.severity.value,
                    "confidence": finding.confidence,
                    "text": span.get("text", ""),
                    "start": span.get("start"),
                    "end": span.get("end"),
                    "context_before": span.get("context_before", ""),
                    "context_after": span.get("context_after", ""),
                    "suggestion": finding.remediation_suggestion,
                    "source_tool": finding.source_tool,
                    "explanation": finding.explanation,
                }
            )

    risk_level = _risk_from_findings(findings)
    total_findings = len(findings)
    return {
        "job_id": job_id,
        "operator_id": operator_id,
        "operator_name": PLATFORM_OPERATOR_NAMES[operator_id],
        "dataset_name": dataset_name,
        "conclusion": _decision_from_risk(risk_level, total_findings),
        "risk_level": risk_level,
        "is_compliant": total_findings == 0,
        "total_documents": len(ingest_units),
        "total_findings": total_findings,
        "summary": {**status_summary, **risk_summary},
        "findings": findings,
        "document_views": _document_views(ingest_units),
        "review_suggestions": [
            "Content safety risks were found. Send flagged samples to manual review."
        ] if total_findings else [],
        "artifact_paths": {name: str(path) for name, path in artifacts.items()},
        "raw_artifacts": {
            "content_safety": _read_artifact(artifacts["content_safety"]),
        },
    }


def _write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_pipeline(task_id: str, package_paths: list[str], config_overrides: dict[str, Any]) -> None:
    task = _tasks[task_id]
    task["status"] = TaskStatus.RUNNING
    try:
        settings = get_settings()
        if config_overrides:
            valid_overrides = {key: value for key, value in config_overrides.items() if hasattr(settings, key)}
            for key, value in list(valid_overrides.items()):
                if key.endswith("_path") or key == "work_dir":
                    valid_overrides[key] = Path(value)
            settings = settings.model_copy(update=valid_overrides)

        pipeline = CompliancePipeline(settings=settings, run_id=task_id)
        compliance_output = pipeline.execute(package_paths)

        task["status"] = TaskStatus.COMPLETED
        task["completed_at"] = _utcnow()
        task["result"] = compliance_output.legacy_decision
        task["compliance_output"] = compliance_output
    except Exception as exc:
        logger.exception("Task %s failed", task_id)
        task["status"] = TaskStatus.FAILED
        task["completed_at"] = _utcnow()
        task["error"] = str(exc)


def _run_platform_compliance_job(
    job_id: str,
    operator_id: str,
    dataset_name: str,
    input_path: str,
    config_overrides: dict[str, Any],
) -> None:
    task = _tasks[job_id]
    task["status"] = TaskStatus.RUNNING
    task["progress"] = 10
    task["current_step"] = "intake"
    try:
        settings = _settings_with_overrides(config_overrides)
        output_dir = settings.work_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = _artifact_paths(output_dir)

        ingest_units = a_source_intake.run([input_path], run_id=job_id)
        write_jsonl(ingest_units, artifacts["intake"])
        write_jsonl(_document_views(ingest_units), artifacts["document_views"])
        if not ingest_units:
            report = {
                "job_id": job_id,
                "operator_id": operator_id,
                "operator_name": PLATFORM_OPERATOR_NAMES[operator_id],
                "dataset_name": dataset_name,
                "conclusion": "passed",
                "risk_level": "none",
                "is_compliant": True,
                "total_documents": 0,
                "total_findings": 0,
                "summary": {},
                "findings": [],
                "redaction_views": [],
                "review_suggestions": ["No readable text documents were found in the uploaded file."],
                "artifact_paths": {name: str(path) for name, path in artifacts.items()},
                "raw_artifacts": {},
            }
        elif operator_id == "CMP_001":
            task["current_step"] = "privacy_detection"
            task["progress"] = 35
            privacy_results = f_privacy_detection.run(ingest_units, settings)
            write_jsonl(privacy_results, artifacts["privacy_detection"])

            task["current_step"] = "span_conflict_resolution"
            task["progress"] = 65
            redaction_plans = span_conflict_resolution.run(ingest_units, privacy_results)
            write_jsonl(redaction_plans, artifacts["redaction_plan"])

            redaction_views = _build_redaction_views(ingest_units, redaction_plans)
            write_jsonl(redaction_views, artifacts["annotation_package"])
            report = _build_privacy_report(
                job_id=job_id,
                operator_id=operator_id,
                dataset_name=dataset_name,
                ingest_units=ingest_units,
                privacy_results=privacy_results,
                redaction_plans=redaction_plans,
                redaction_views=redaction_views,
                artifacts=artifacts,
            )
        elif operator_id == "CMP_002":
            task["current_step"] = "content_safety"
            task["progress"] = 55
            safety_results = g_safety_moderation.run(ingest_units, settings)
            write_jsonl(safety_results, artifacts["content_safety"])
            report = _build_content_report(
                job_id=job_id,
                operator_id=operator_id,
                dataset_name=dataset_name,
                ingest_units=ingest_units,
                safety_results=safety_results,
                artifacts=artifacts,
            )
        else:
            raise ValueError(f"Unsupported platform operator: {operator_id}")

        _write_report(report, artifacts["report"])
        task["status"] = TaskStatus.COMPLETED
        task["progress"] = 100
        task["current_step"] = "completed"
        task["completed_at"] = _utcnow()
        task["result"] = report
        task["artifact_paths"] = {name: str(path) for name, path in artifacts.items()}
    except Exception as exc:
        logger.exception("Platform compliance job %s failed", job_id)
        task["status"] = TaskStatus.FAILED
        task["progress"] = max(1, int(task.get("progress") or 1))
        task["current_step"] = "failed"
        task["completed_at"] = _utcnow()
        task["error"] = str(exc)


@app.get("/api/v1/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "text-cleaned-package-checker",
        "service_mode": "legacy",
        "preferred_service": "text.api_server",
        "preferred_port": get_settings().api_server_port,
        "active_tasks": sum(1 for task in _tasks.values() if task["status"] == TaskStatus.RUNNING),
    }


@app.post("/api/v1/text/compliance-agent/jobs")
async def submit_platform_compliance_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    operator_id: str = Form(...),
    dataset_name: str = Form(default=""),
    config: str = Form(default="{}"),
    _: None = Depends(_require_platform_key),
) -> dict[str, Any]:
    operator_id = operator_id.strip().upper()
    if operator_id not in PLATFORM_OPERATOR_NAMES:
        raise HTTPException(status_code=400, detail=f"Unsupported operator_id: {operator_id}")

    config_overrides = _parse_config(config)
    settings = _settings_with_overrides(config_overrides)
    job_id = uuid.uuid4().hex
    upload_dir = settings.upload_dir / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / _safe_filename(file.filename)
    with upload_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    await file.close()

    resolved_dataset_name = dataset_name.strip() or upload_path.name
    _tasks[job_id] = {
        "task_id": job_id,
        "job_id": job_id,
        "operator_id": operator_id,
        "dataset_name": resolved_dataset_name,
        "status": TaskStatus.PENDING,
        "progress": 5,
        "current_step": "queued",
        "created_at": _utcnow(),
        "completed_at": None,
        "result": None,
        "error": None,
        "artifact_paths": {},
        "input_path": str(upload_path),
    }
    background_tasks.add_task(
        _run_platform_compliance_job,
        job_id,
        operator_id,
        resolved_dataset_name,
        str(upload_path),
        config_overrides,
    )
    return {
        "code": 200,
        "data": {
            "job_id": job_id,
            "operator_id": operator_id,
            "status": "pending",
            "progress": 5,
            "current_step": "queued",
        },
    }


@app.get("/api/v1/text/compliance-agent/jobs/{job_id}")
async def get_platform_compliance_job(
    job_id: str,
    _: None = Depends(_require_platform_key),
) -> dict[str, Any]:
    task = _tasks.get(job_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    status = _status_value(task.get("status", "pending"))
    data = {
        "job_id": job_id,
        "operator_id": task.get("operator_id", ""),
        "dataset_name": task.get("dataset_name", ""),
        "status": status,
        "progress": int(task.get("progress") or 0),
        "current_step": task.get("current_step", ""),
        "completed_tasks": 1 if status == "completed" else 0,
        "total_tasks": 1,
        "result": task.get("result"),
        "error": task.get("error"),
        "artifact_paths": task.get("artifact_paths", {}),
        "created_at": task["created_at"].isoformat() if task.get("created_at") else None,
        "completed_at": task["completed_at"].isoformat() if task.get("completed_at") else None,
    }
    return {"code": 200, "data": data}


@app.get("/api/v1/text/compliance-agent/jobs/{job_id}/artifacts/{artifact_name}")
async def get_platform_compliance_artifact(
    job_id: str,
    artifact_name: str,
    _: None = Depends(_require_platform_key),
) -> dict[str, Any]:
    task = _tasks.get(job_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    artifact_paths = task.get("artifact_paths") or {}
    raw_path = artifact_paths.get(artifact_name)
    if not raw_path:
        raise HTTPException(status_code=404, detail=f"Artifact {artifact_name} not found")
    path = Path(raw_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact file {artifact_name} not found")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = _read_artifact(path)
    return {"code": 200, "data": {"job_id": job_id, "artifact_name": artifact_name, "records": payload}}


@app.post("/api/v1/check", response_model=CheckTaskInfo)
async def submit_check(request: CheckRequest, background_tasks: BackgroundTasks) -> CheckTaskInfo:
    task_id = uuid.uuid4().hex
    _tasks[task_id] = {
        "task_id": task_id,
        "status": TaskStatus.PENDING,
        "created_at": _utcnow(),
        "completed_at": None,
        "result": None,
        "error": None,
        "compliance_output": None,
    }
    background_tasks.add_task(_run_pipeline, task_id, request.package_paths, request.config_overrides)
    return CheckTaskInfo(task_id=task_id, status=TaskStatus.PENDING, created_at=_tasks[task_id]["created_at"])


@app.get("/api/v1/status/{task_id}", response_model=CheckTaskInfo)
async def get_task_status(task_id: str) -> CheckTaskInfo:
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return CheckTaskInfo(
        task_id=task["task_id"],
        status=task["status"],
        created_at=task["created_at"],
        completed_at=task["completed_at"],
        result=task["result"],
        error=task["error"],
    )


@app.get("/api/v1/result/{task_id}")
async def get_task_result(task_id: str) -> dict[str, Any]:
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    if task["status"] in {TaskStatus.PENDING, TaskStatus.RUNNING}:
        raise HTTPException(status_code=202, detail=f"Task is {task['status'].value}")
    if task["status"] == TaskStatus.FAILED:
        raise HTTPException(status_code=500, detail=task["error"] or "Task failed")

    compliance_output = task["compliance_output"]
    return {
        "task_id": task_id,
        "status": task["status"].value,
        "created_at": task["created_at"].isoformat(),
        "completed_at": task["completed_at"].isoformat() if task["completed_at"] else None,
        "decision": compliance_output.decision.value,
        "trust_level": compliance_output.trust_level.value,
        "annotation_package_uri": compliance_output.annotation_package_uri,
        "audit_package_uri": compliance_output.audit_package_uri,
        "review_suggestions": compliance_output.review_suggestions,
        "explanation_summary": compliance_output.explanation_summary,
        "legacy_decision": compliance_output.legacy_decision,
        "metadata": {
            **dict(compliance_output.metadata or {}),
            "service_mode": "legacy",
            "preferred_service": "text.api_server",
            "preferred_port": get_settings().api_server_port,
        },
    }


@app.get("/api/v1/tasks")
async def list_tasks(limit: int = 50) -> list[dict[str, Any]]:
    ordered = sorted(_tasks.values(), key=lambda task: task["created_at"], reverse=True)[:limit]
    return [
        {
            "task_id": task["task_id"],
            "status": task["status"].value,
            "created_at": task["created_at"].isoformat(),
            "completed_at": task["completed_at"].isoformat() if task["completed_at"] else None,
        }
        for task in ordered
    ]


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("text.server:app", host=settings.server_host, port=settings.server_port, reload=True)
