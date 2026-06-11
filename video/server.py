"""FastAPI server for video compliance checking."""

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

from video.config.settings import get_settings
from video.models.schemas import CheckRequest, CheckTaskInfo, TaskStatus
from video.pipeline import VideoCompliancePipeline

logger = logging.getLogger(__name__)
_tasks: dict[str, CheckTaskInfo] = {}
_CONTRACT_VERSION = "compliance-job.v1"
_RESULT_CONTRACT_VERSION = "compliance-result.v1"
_OPERATOR_CATALOG_VERSION = "video-compliance-operators.v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Video compliance service starting")
    yield
    logger.info("Video compliance service stopping")


app = FastAPI(
    title="Video Data Compliance Checker",
    description="Video compliance service built on top of picture and audio sub-engines.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def _status_label(status: TaskStatus | str) -> str:
    value = status.value if isinstance(status, TaskStatus) else str(status)
    return {
        TaskStatus.PENDING.value: "等待执行",
        TaskStatus.RUNNING.value: "检测中",
        TaskStatus.COMPLETED.value: "检测完成",
        TaskStatus.FAILED.value: "检测失败",
    }.get(value, "处理中")


def _operator_ids_from_selection(operator_id: str, selection: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    if operator_id:
        ids.append(operator_id)
    for value in selection.values():
        if isinstance(value, list):
            ids.extend(str(item) for item in value if str(item).strip())
    return list(dict.fromkeys(ids))


def _effective_request(task_id: str, request: CheckRequest) -> dict[str, Any]:
    operator_id = str(request.operator_id or request.options.get("operator_id") or "").strip()
    requested = _operator_ids_from_selection(operator_id, request.operator_selection)
    return {
        "contract_version": request.contract_version or _CONTRACT_VERSION,
        "platform_task_id": request.platform_task_id,
        "idempotency_key": request.idempotency_key,
        "remote_task_id": task_id,
        "modality": "video",
        "operator_id": operator_id,
        "operator_catalog_version": request.operator_catalog_version or _OPERATOR_CATALOG_VERSION,
        "requested_operator_ids": requested,
        "effective_operator_ids": requested,
        "profile": request.profile,
        "task_context": request.task_context,
        "operator_selection": request.operator_selection,
        "options": request.options,
    }


def _error_info(code: str, stage: str, message: str, retryable: bool = False) -> dict[str, Any]:
    return {
        "code": code,
        "stage": stage,
        "message": message,
        "retryable": retryable,
    }


def _apply_task_contract(task: CheckTaskInfo, effective_request: dict[str, Any]) -> None:
    task.contract_version = effective_request.get("contract_version", _CONTRACT_VERSION)
    task.platform_task_id = effective_request.get("platform_task_id", "")
    task.idempotency_key = effective_request.get("idempotency_key", "")
    task.modality = "video"
    task.stage = "accepted"
    task.progress = 0
    task.status_label = _status_label(task.status)
    task.effective_request = effective_request


def _result_consistency(result: dict[str, Any] | None) -> dict[str, Any]:
    payload = result or {}
    risks = payload.get("risk_annotations")
    review_queue = payload.get("review_queue")
    artifacts = payload.get("asset") if isinstance(payload.get("asset"), dict) else {}
    governance = payload.get("governance_result") if isinstance(payload.get("governance_result"), dict) else {}
    policy = payload.get("policy_result") if isinstance(payload.get("policy_result"), dict) else {}
    evidence_count = len(risks) if isinstance(risks, list) else 0
    review_count = len(review_queue) if isinstance(review_queue, list) else 0
    return {
        "modality": "video",
        "evidence_record_count": evidence_count,
        "review_required": bool(governance.get("requires_review")) or review_count > 0,
        "review_task_count": review_count,
        "artifact_keys": sorted(artifacts.keys()) if isinstance(artifacts, dict) else [],
        "decision": str(governance.get("decision") or policy.get("decision") or ""),
    }


def _existing_task_for(effective_request: dict[str, Any]) -> CheckTaskInfo | None:
    idempotency_key = str(effective_request.get("idempotency_key") or "").strip()
    platform_task_id = str(effective_request.get("platform_task_id") or "").strip()
    if not idempotency_key and not platform_task_id:
        return None
    for task in _tasks.values():
        effective = dict(task.effective_request or {})
        if idempotency_key and effective.get("idempotency_key") == idempotency_key:
            return task
        if platform_task_id and effective.get("platform_task_id") == platform_task_id:
            return task
    return None


def _existing_task_for_payload(payload: dict[str, Any]) -> CheckTaskInfo | None:
    return _existing_task_for({
        "idempotency_key": payload.get("idempotency_key", ""),
        "platform_task_id": payload.get("platform_task_id", ""),
    })


def _run_pipeline(task_id: str, request: CheckRequest) -> None:
    task = _tasks[task_id]
    task.status = TaskStatus.RUNNING
    task.stage = "pipeline"
    task.progress = 35
    task.status_label = _status_label(task.status)
    try:
        settings = get_settings()
        if request.config_overrides:
            settings = settings.model_copy(
                update={key: value for key, value in request.config_overrides.items() if hasattr(settings, key)}
            )
        pipeline = VideoCompliancePipeline(settings=settings)
        pipeline.run_id = task_id
        pipeline.output_dir = settings.work_dir / task_id
        result = pipeline.execute(
            input_path=request.input_path,
            tenant_id=request.tenant_id,
            profile=request.profile,
            dataset_id=request.dataset_id,
            asset_id=request.asset_id,
            cleaning_run_id=request.cleaning_run_id,
            task_context=request.task_context,
            operator_selection=request.operator_selection,
            options=request.options,
        )
        task.status = TaskStatus.COMPLETED
        task.result = result.model_dump(mode="json")
        task.completed_at = datetime.now(timezone.utc)
        task.stage = "completed"
        task.progress = 100
        task.status_label = _status_label(task.status)
    except Exception as exc:
        logger.exception("Task %s failed", task_id)
        task.status = TaskStatus.FAILED
        task.error = str(exc)
        task.error_info = _error_info("VIDEO_PIPELINE_FAILED", "video_pipeline", str(exc), retryable=False)
        task.completed_at = datetime.now(timezone.utc)
        task.status_label = _status_label(task.status)


@app.get("/api/v1/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "video-compliance-checker",
        "version": "0.1.0",
        "active_tasks": sum(1 for task in _tasks.values() if task.status == TaskStatus.RUNNING),
    }


@app.post("/api/v1/check", response_model=CheckTaskInfo)
async def submit_check(request: CheckRequest, background_tasks: BackgroundTasks) -> CheckTaskInfo:
    effective_request = _effective_request(uuid.uuid4().hex, request)
    existing_task = _existing_task_for(effective_request)
    if existing_task is not None:
        return existing_task
    task_id = uuid.uuid4().hex
    effective_request["remote_task_id"] = task_id
    task = CheckTaskInfo(task_id=task_id, status=TaskStatus.PENDING)
    _apply_task_contract(task, effective_request)
    _tasks[task_id] = task
    background_tasks.add_task(_run_pipeline, task_id, request)
    return task


@app.post("/api/v1/check-file", response_model=CheckTaskInfo)
async def submit_check_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: str = Form("{}"),
) -> CheckTaskInfo:
    try:
        payload = json.loads(request or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"request must be JSON: {exc}")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request must be a JSON object")
    existing_task = _existing_task_for_payload(payload)
    if existing_task is not None:
        return existing_task

    task_id = uuid.uuid4().hex
    settings = get_settings()
    filename = Path(file.filename or "input.dat").name
    upload_dir = settings.work_dir / "uploads" / task_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / filename
    with input_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    payload["input_path"] = str(input_path)

    check_request = CheckRequest.model_validate(payload)
    effective_request = _effective_request(task_id, check_request)
    task = CheckTaskInfo(task_id=task_id, status=TaskStatus.PENDING)
    effective_request["remote_task_id"] = task_id
    _apply_task_contract(task, effective_request)
    _tasks[task_id] = task
    background_tasks.add_task(_run_pipeline, task_id, check_request)
    return task


@app.get("/api/v1/status/{task_id}", response_model=CheckTaskInfo)
async def get_status(task_id: str) -> CheckTaskInfo:
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task


@app.get("/api/v1/result/{task_id}")
async def get_result(task_id: str) -> dict[str, Any]:
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    if task.status in {TaskStatus.PENDING, TaskStatus.RUNNING}:
        raise HTTPException(status_code=202, detail="Task is not finished yet")
    if task.status == TaskStatus.FAILED:
        raise HTTPException(status_code=500, detail=task.error or "Task failed")
    return {
        "task_id": task.task_id,
        "contract_version": _RESULT_CONTRACT_VERSION,
        "platform_task_id": task.platform_task_id,
        "idempotency_key": task.idempotency_key,
        "modality": "video",
        "status": task.status.value,
        "status_label": _status_label(task.status),
        "created_at": task.created_at.isoformat(),
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "result": task.result,
        "result_consistency": _result_consistency(task.result),
        "effective_request": task.effective_request,
        "warnings": task.warnings,
        "error_info": task.error_info,
    }


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("video.server:app", host=settings.server_host, port=settings.server_port, reload=True)
