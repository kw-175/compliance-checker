from __future__ import annotations

import logging
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from text.api_clients import resolve_provider_config
from text.api_content_safety_service import check_content_safety
from text.api_pipeline import APICompliancePipeline
from text.config.settings import get_settings
from text.jsonl_utils import read_jsonl, write_jsonl
from text.models.schemas import (
    CheckRequest,
    CheckTaskInfo,
    ContentSafetyBatchCheckRequest,
    ContentSafetyBatchCheckResponse,
    TaskStatus,
)
from text.steps.content_safety_review import build_final_decisions, merge_review_results
from text.steps.privacy_review import build_final_decisions as build_privacy_final_decisions, merge_review_results as merge_privacy_review_results

logger = logging.getLogger(__name__)

_tasks: dict[str, dict[str, Any]] = {}
_OPERATOR_PIPELINE_PROFILE = {
    "CMP_001": "privacy_only",
    "CMP_002": "safety_only",
    "CMP_008": "full",
}
_CONTRACT_VERSION = "compliance-job.v1"
_RESULT_CONTRACT_VERSION = "compliance-result.v1"
_OPERATOR_CATALOG_VERSION = "text-compliance-operators.v1"


def _status_label(status: TaskStatus | str) -> str:
    value = status.value if isinstance(status, TaskStatus) else str(status)
    return {
        TaskStatus.PENDING.value: "等待执行",
        TaskStatus.RUNNING.value: "检测中",
        TaskStatus.COMPLETED.value: "检测完成",
        TaskStatus.FAILED.value: "检测失败",
    }.get(value, "处理中")


def _effective_request(modality: str, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    operator_id = str(payload.get("operator_id") or "").strip()
    requested = [operator_id] if operator_id else []
    return {
        "contract_version": str(payload.get("contract_version") or _CONTRACT_VERSION),
        "platform_task_id": str(payload.get("platform_task_id") or ""),
        "idempotency_key": str(payload.get("idempotency_key") or ""),
        "remote_task_id": task_id,
        "modality": modality,
        "operator_id": operator_id,
        "operator_catalog_version": str(payload.get("operator_catalog_version") or _OPERATOR_CATALOG_VERSION),
        "requested_operator_ids": requested,
        "effective_operator_ids": requested,
        "profile": str(payload.get("profile") or payload.get("profile_id") or "default_cn_enterprise"),
    }


def _error_info(code: str, stage: str, message: str, retryable: bool = False) -> dict[str, Any]:
    return {
        "code": code,
        "stage": stage,
        "message": message,
        "retryable": retryable,
    }


def _task_info(task_id: str) -> CheckTaskInfo:
    task = _tasks[task_id]
    status = task["status"]
    effective = dict(task.get("effective_request") or {})
    return CheckTaskInfo(
        task_id=task["task_id"],
        status=status,
        created_at=task["created_at"],
        completed_at=task["completed_at"],
        result=task["result"],
        error=task["error"],
        contract_version=effective.get("contract_version", _CONTRACT_VERSION),
        platform_task_id=effective.get("platform_task_id", ""),
        idempotency_key=effective.get("idempotency_key", ""),
        modality="text",
        stage=task.get("stage", ""),
        progress=task.get("progress", 0),
        status_label=_status_label(status),
        effective_request=effective,
        warnings=list(task.get("warnings") or []),
        error_info=task.get("error_info"),
    )


def _existing_task_id_for(payload: dict[str, Any]) -> str:
    idempotency_key = str(payload.get("idempotency_key") or "").strip()
    platform_task_id = str(payload.get("platform_task_id") or "").strip()
    if not idempotency_key and not platform_task_id:
        return ""
    for task_id, task in _tasks.items():
        effective = dict(task.get("effective_request") or {})
        if idempotency_key and effective.get("idempotency_key") == idempotency_key:
            return task_id
        if platform_task_id and effective.get("platform_task_id") == platform_task_id:
            return task_id
    return ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    yield


app = FastAPI(
    title="Text API Compliance Checker",
    description="Runs the JSONL-native text compliance workflow through local-model-first compliance operators.",
    version="0.1.0",
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


def _settings_with_overrides(config_overrides: dict[str, Any]):
    settings = get_settings()
    if not config_overrides:
        return settings
    valid_overrides = {key: value for key, value in config_overrides.items() if hasattr(settings, key)}
    for key, value in list(valid_overrides.items()):
        if key.endswith("_path") or key in {"work_dir", "upload_dir"}:
            valid_overrides[key] = Path(value)
    return settings.model_copy(update=valid_overrides)


def _resolve_pipeline_profile(config_overrides: dict[str, Any]) -> str:
    explicit_profile = str(config_overrides.get("pipeline_profile") or "").strip().lower()
    if explicit_profile:
        return explicit_profile
    operator_id = str(config_overrides.get("operator_id") or "").strip().upper()
    return _OPERATOR_PIPELINE_PROFILE.get(operator_id, "full")


def _run_api_pipeline(task_id: str, package_paths: list[str], config_overrides: dict[str, Any]) -> None:
    task = _tasks[task_id]
    task["status"] = TaskStatus.RUNNING
    task["stage"] = "pipeline"
    task["progress"] = 35
    try:
        settings = _settings_with_overrides(config_overrides)
        pipeline_profile = _resolve_pipeline_profile(config_overrides)
        pipeline = APICompliancePipeline(settings=settings, run_id=task_id)
        compliance_output = pipeline.execute(package_paths, profile=pipeline_profile)

        task["status"] = TaskStatus.COMPLETED
        task["completed_at"] = _utcnow()
        task["result"] = compliance_output.legacy_decision
        task["compliance_output"] = compliance_output
        task["artifact_paths"] = compliance_output.metadata.get("artifact_paths", {})
        task["stage"] = "completed"
        task["progress"] = 100
    except Exception as exc:
        logger.exception("API task %s failed", task_id)
        task["status"] = TaskStatus.FAILED
        task["completed_at"] = _utcnow()
        task["error"] = str(exc)
        task["error_info"] = _error_info("TEXT_PIPELINE_FAILED", "text_pipeline", str(exc), retryable=False)


@app.get("/api/v1/health")
async def health_check() -> dict[str, Any]:
    settings = get_settings()
    try:
        provider = resolve_provider_config(settings)
        provider_mode = provider.mode
        provider_model = provider.model
    except Exception:
        provider_mode = "unconfigured"
        provider_model = ""
    return {
        "status": "healthy",
        "service": "text-api-compliance-checker",
        "service_mode": "preferred",
        "api_base_url_configured": bool(settings.api_compliance_base_url),
        "api_model_configured": bool(settings.api_compliance_model),
        "local_base_url_configured": bool(settings.local_compliance_base_url),
        "local_model_configured": bool(settings.local_compliance_model),
        "provider_mode": provider_mode,
        "provider_model": provider_model,
        "active_tasks": sum(1 for task in _tasks.values() if task["status"] == TaskStatus.RUNNING),
    }


@app.post("/api/v1/check", response_model=CheckTaskInfo)
async def submit_check(request: CheckRequest, background_tasks: BackgroundTasks) -> CheckTaskInfo:
    contract_payload = {
        **request.config_overrides,
        "contract_version": request.contract_version,
        "platform_task_id": request.platform_task_id,
        "idempotency_key": request.idempotency_key,
        "operator_id": request.operator_id,
        "operator_catalog_version": request.operator_catalog_version,
    }
    existing_task_id = _existing_task_id_for(contract_payload)
    if existing_task_id:
        return _task_info(existing_task_id)
    task_id = uuid.uuid4().hex
    _tasks[task_id] = {
        "task_id": task_id,
        "status": TaskStatus.PENDING,
        "created_at": _utcnow(),
        "completed_at": None,
        "result": None,
        "error": None,
        "compliance_output": None,
        "stage": "accepted",
        "progress": 0,
        "effective_request": _effective_request("text", task_id, contract_payload),
        "warnings": [],
        "error_info": None,
    }
    background_tasks.add_task(_run_api_pipeline, task_id, request.package_paths, request.config_overrides)
    return _task_info(task_id)


@app.post("/api/v1/check-file", response_model=CheckTaskInfo)
async def submit_check_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    config_overrides: str = Form("{}"),
) -> CheckTaskInfo:
    try:
        overrides = json.loads(config_overrides or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"config_overrides must be JSON: {exc}")
    if not isinstance(overrides, dict):
        raise HTTPException(status_code=400, detail="config_overrides must be a JSON object")
    existing_task_id = _existing_task_id_for(overrides)
    if existing_task_id:
        return _task_info(existing_task_id)

    task_id = uuid.uuid4().hex
    settings = get_settings()
    filename = Path(file.filename or "input.dat").name
    upload_dir = Path(settings.upload_dir) / task_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / filename
    with input_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    _tasks[task_id] = {
        "task_id": task_id,
        "status": TaskStatus.PENDING,
        "created_at": _utcnow(),
        "completed_at": None,
        "result": None,
        "error": None,
        "compliance_output": None,
        "stage": "accepted",
        "progress": 0,
        "effective_request": _effective_request("text", task_id, overrides),
        "warnings": [],
        "error_info": None,
    }
    background_tasks.add_task(_run_api_pipeline, task_id, [str(input_path)], overrides)
    return _task_info(task_id)


@app.post("/api/v2/text/content-safety/check", response_model=ContentSafetyBatchCheckResponse)
async def content_safety_check(request: ContentSafetyBatchCheckRequest) -> ContentSafetyBatchCheckResponse:
    settings = get_settings()
    return check_content_safety(request, settings=settings)


@app.get("/api/v1/status/{task_id}", response_model=CheckTaskInfo)
async def get_task_status(task_id: str) -> CheckTaskInfo:
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return _task_info(task_id)


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
    final_decision_path = _artifact_path(task_id, "content_safety_final_decisions")
    final_decisions = read_jsonl(final_decision_path) if final_decision_path.exists() else []
    privacy_final_decision_path = _artifact_path(task_id, "privacy_final_decisions")
    privacy_final_decisions = read_jsonl(privacy_final_decision_path) if privacy_final_decision_path.exists() else []
    user_document_explanations = [
        {
            "doc_id": item.get("doc_id", ""),
            "explanation": item.get("explanation", "") or item.get("summary", "") or "",
        }
        for item in compliance_output.legacy_decision.get("documents", [])
        if isinstance(item, dict)
    ]
    result_payload = {
        "task_id": task_id,
        "contract_version": _RESULT_CONTRACT_VERSION,
        "platform_task_id": task.get("effective_request", {}).get("platform_task_id", ""),
        "idempotency_key": task.get("effective_request", {}).get("idempotency_key", ""),
        "modality": "text",
        "status": task["status"].value,
        "status_label": _status_label(task["status"]),
        "created_at": task["created_at"].isoformat(),
        "completed_at": task["completed_at"].isoformat() if task["completed_at"] else None,
        "decision": compliance_output.decision.value,
        "trust_level": compliance_output.trust_level.value,
        "annotation_package_uri": compliance_output.annotation_package_uri,
        "audit_package_uri": compliance_output.audit_package_uri,
        "review_suggestions": compliance_output.review_suggestions,
        "explanation_summary": compliance_output.explanation_summary,
        "user_summary": compliance_output.explanation_summary,
        "document_explanations": user_document_explanations,
        "content_safety_final_decisions_uri": str(final_decision_path),
        "content_safety_final_decisions": final_decisions,
        "content_safety_review_task_count": sum(int(item.get("review_task_count", 0) or 0) for item in final_decisions),
        "privacy_final_decisions_uri": str(privacy_final_decision_path),
        "privacy_final_decisions": privacy_final_decisions,
        "privacy_review_task_count": sum(int(item.get("review_task_count", 0) or 0) for item in privacy_final_decisions),
        "legacy_decision": compliance_output.legacy_decision,
        "artifact_records": _inline_artifact_records(task_id),
        "effective_request": task.get("effective_request") or {},
        "warnings": task.get("warnings") or [],
        "error_info": task.get("error_info"),
        "metadata": compliance_output.metadata,
    }
    result_payload["result_consistency"] = _result_consistency(task_id, result_payload)
    return result_payload


def _artifact_path(task_id: str, name: str) -> Path:
    task = _tasks.get(task_id)
    artifact_paths = dict((task or {}).get("artifact_paths") or {})
    if artifact_paths.get(name):
        return Path(str(artifact_paths[name]))
    settings = get_settings()
    default_names = {
        "content_safety_decisions": "02b_content_safety_decisions.jsonl",
        "content_safety_review_tasks": "02d_content_safety_review_tasks.jsonl",
        "content_safety_review_results": "02e_content_safety_review_results.jsonl",
        "content_safety_final_decisions": "02f_content_safety_final_decisions.jsonl",
        "privacy_review_tasks": "03e_privacy_review_tasks.jsonl",
        "privacy_review_results": "03h_privacy_review_results.jsonl",
        "privacy_final_decisions": "03i_privacy_final_decisions.jsonl",
    }
    return settings.work_dir / task_id / default_names[name]


def _inline_artifact_records(task_id: str) -> dict[str, Any]:
    task = _tasks.get(task_id) or {}
    artifact_paths = dict(task.get("artifact_paths") or {})
    records: dict[str, Any] = {}
    for name, raw_path in artifact_paths.items():
        path = Path(str(raw_path))
        if not path.exists() or not path.is_file():
            continue
        if path.suffix.lower() == ".jsonl":
            records[name] = read_jsonl(path)
        elif path.suffix.lower() == ".json":
            try:
                records[name] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Unable to inline JSON artifact: %s", path)
    return records


def _result_consistency(task_id: str, result_payload: dict[str, Any]) -> dict[str, Any]:
    artifact_records = _inline_artifact_records(task_id)
    evidence_count = 0
    for key in ("privacy", "content_safety", "annotation"):
        value = artifact_records.get(key)
        if isinstance(value, list):
            evidence_count += len(value)
    review_task_count = 0
    for key in ("privacy_review_tasks", "content_safety_review_tasks"):
        value = artifact_records.get(key)
        if isinstance(value, list):
            review_task_count += len(value)
    return {
        "modality": "text",
        "evidence_artifact_keys": sorted(artifact_records.keys()),
        "evidence_record_count": evidence_count,
        "review_required": review_task_count > 0,
        "review_task_count": review_task_count,
        "decision": result_payload.get("decision", ""),
        "trust_level": result_payload.get("trust_level", ""),
    }


@app.get("/api/v1/review/tasks/{task_id}")
async def get_content_safety_review_tasks(task_id: str) -> dict[str, Any]:
    task = _tasks.get(task_id)
    if task is not None and task["status"] in {TaskStatus.PENDING, TaskStatus.RUNNING}:
        raise HTTPException(status_code=202, detail=f"Task is {task['status'].value}")
    review_path = _artifact_path(task_id, "content_safety_review_tasks")
    if not review_path.exists():
        raise HTTPException(status_code=404, detail=f"Review tasks for {task_id} not found")
    tasks = read_jsonl(review_path)
    return {
        "task_id": task_id,
        "review_task_count": len(tasks),
        "tasks": tasks,
        "artifact_path": str(review_path),
    }


@app.post("/api/v1/review/tasks/{task_id}/decisions")
async def submit_content_safety_review_decisions(task_id: str, submissions: list[dict[str, Any]]) -> dict[str, Any]:
    review_path = _artifact_path(task_id, "content_safety_review_tasks")
    result_path = _artifact_path(task_id, "content_safety_review_results")
    initial_decision_path = _artifact_path(task_id, "content_safety_decisions")
    final_decision_path = _artifact_path(task_id, "content_safety_final_decisions")
    if not review_path.exists():
        raise HTTPException(status_code=404, detail=f"Review tasks for {task_id} not found")
    tasks = read_jsonl(review_path)
    previous_results = read_jsonl(result_path) if result_path.exists() else []
    results = list(previous_results) + merge_review_results(tasks, submissions)
    initial_decisions = read_jsonl(initial_decision_path) if initial_decision_path.exists() else []
    final_decisions = build_final_decisions(initial_decisions, tasks, results)
    write_jsonl(results, result_path)
    write_jsonl(final_decisions, final_decision_path)
    return {
        "task_id": task_id,
        "review_result_count": len(results),
        "artifact_path": str(result_path),
        "final_decision_artifact_path": str(final_decision_path),
        "results": results,
        "final_decisions": final_decisions,
    }


@app.get("/api/v1/privacy/review/tasks/{task_id}")
async def get_privacy_review_tasks(task_id: str) -> dict[str, Any]:
    task = _tasks.get(task_id)
    if task is not None and task["status"] in {TaskStatus.PENDING, TaskStatus.RUNNING}:
        raise HTTPException(status_code=202, detail=f"Task is {task['status'].value}")
    review_path = _artifact_path(task_id, "privacy_review_tasks")
    if not review_path.exists():
        raise HTTPException(status_code=404, detail=f"Privacy review tasks for {task_id} not found")
    tasks = read_jsonl(review_path)
    return {
        "task_id": task_id,
        "review_task_count": len(tasks),
        "tasks": tasks,
        "artifact_path": str(review_path),
    }


@app.post("/api/v1/privacy/review/tasks/{task_id}/decisions")
async def submit_privacy_review_decisions(task_id: str, submissions: list[dict[str, Any]]) -> dict[str, Any]:
    review_path = _artifact_path(task_id, "privacy_review_tasks")
    result_path = _artifact_path(task_id, "privacy_review_results")
    initial_decision_path = _artifact_path(task_id, "privacy_decisions")
    final_decision_path = _artifact_path(task_id, "privacy_final_decisions")
    if not review_path.exists():
        raise HTTPException(status_code=404, detail=f"Privacy review tasks for {task_id} not found")
    tasks = read_jsonl(review_path)
    previous_results = read_jsonl(result_path) if result_path.exists() else []
    results = list(previous_results) + merge_privacy_review_results(tasks, submissions)
    initial_decisions = read_jsonl(initial_decision_path) if initial_decision_path.exists() else []
    final_decisions = build_privacy_final_decisions(initial_decisions, tasks, results)
    write_jsonl(results, result_path)
    write_jsonl(final_decisions, final_decision_path)
    return {
        "task_id": task_id,
        "review_result_count": len(results),
        "artifact_path": str(result_path),
        "final_decision_artifact_path": str(final_decision_path),
        "results": results,
        "final_decisions": final_decisions,
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
    uvicorn.run("text.api_server:app", host=settings.server_host, port=settings.api_server_port, reload=True)
