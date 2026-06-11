"""
FastAPI server for audio compliance checking.
"""

from __future__ import annotations

import logging
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from audio.config.settings import Settings, get_settings
from audio.models.schemas import CheckRequest, CheckTaskInfo, TaskStatus
from audio.pipeline import AudioCompliancePipeline
from audio.text_api_bridge import AudioTextApiBridgeExecutor

logger = logging.getLogger(__name__)
# 进程内任务表：用于追踪异步检查任务状态与结果。
_tasks: dict[str, CheckTaskInfo] = {}
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_CONTRACT_VERSION = "compliance-job.v1"
_RESULT_CONTRACT_VERSION = "compliance-result.v1"
_OPERATOR_CATALOG_VERSION = "audio-compliance-operators.v1"


app = FastAPI(
    title="Audio Data Compliance Checker",
    description="Audio compliance checking microservice with normalization, Qwen3-ASR, text compliance bridge, and timeline reporting.",
    version="0.1.0",
)
# 开放 CORS，便于前端或其他服务直接调用本地 API。
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def _status_label(status: TaskStatus | str) -> str:
    value = status.value if isinstance(status, TaskStatus) else str(status)
    return {
        TaskStatus.PENDING.value: "等待执行",
        TaskStatus.RUNNING.value: "检测中",
        TaskStatus.COMPLETED.value: "检测完成",
        TaskStatus.FAILED.value: "检测失败",
    }.get(value, "处理中")


def _effective_request(task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    operator_id = str(payload.get("operator_id") or "").strip()
    requested = [operator_id] if operator_id else []
    return {
        "contract_version": str(payload.get("contract_version") or _CONTRACT_VERSION),
        "platform_task_id": str(payload.get("platform_task_id") or ""),
        "idempotency_key": str(payload.get("idempotency_key") or ""),
        "remote_task_id": task_id,
        "modality": "audio",
        "operator_id": operator_id,
        "operator_catalog_version": str(payload.get("operator_catalog_version") or _OPERATOR_CATALOG_VERSION),
        "requested_operator_ids": requested,
        "effective_operator_ids": requested,
        "profile": str(payload.get("profile") or "default_cn_enterprise"),
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
    task.modality = "audio"
    task.stage = "accepted"
    task.progress = 0
    task.status_label = _status_label(task.status)
    task.effective_request = effective_request


def _existing_task_for(payload: dict[str, Any]) -> CheckTaskInfo | None:
    idempotency_key = str(payload.get("idempotency_key") or "").strip()
    platform_task_id = str(payload.get("platform_task_id") or "").strip()
    if not idempotency_key and not platform_task_id:
        return None
    for task in _tasks.values():
        effective = dict(task.effective_request or {})
        if idempotency_key and effective.get("idempotency_key") == idempotency_key:
            return task
        if platform_task_id and effective.get("platform_task_id") == platform_task_id:
            return task
    return None


def _apply_config_overrides(settings: Settings, config_overrides: dict[str, Any]) -> Settings:
    valid_overrides = {
        key: value
        for key, value in config_overrides.items()
        if hasattr(settings, key)
    }
    if not valid_overrides:
        return settings

    payload = settings.model_dump()
    payload.update(valid_overrides)
    return Settings.model_validate(payload)


def _resolve_execution_route(settings: Settings, config_overrides: dict[str, Any]) -> str:
    for key in ("compliance_route", "execution_route", "audio_execution_route", "route"):
        value = str(config_overrides.get(key) or "").strip().lower()
        if value:
            if value in {"api", "bridge", "text_api_bridge", "external_api", "compat"}:
                return "api"
            return "local"
    default_route = str(settings.audio_execution_route or "").strip().lower()
    if default_route in {"api", "bridge", "text_api_bridge", "external_api", "compat"}:
        return "api"
    return "local"


def _serialize_result(result: Any) -> Any:
    if result is None:
        return None
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    if isinstance(result, dict):
        return result
    return result


def _result_consistency(result: Any) -> dict[str, Any]:
    payload = _serialize_result(result)
    if not isinstance(payload, dict):
        return {"modality": "audio", "evidence_record_count": 0, "review_required": False}
    findings = payload.get("findings")
    supplemental = payload.get("supplemental_findings")
    review_tasks = payload.get("review_tasks")
    evidence_count = (len(findings) if isinstance(findings, list) else 0) + (
        len(supplemental) if isinstance(supplemental, list) else 0
    )
    review_count = len(review_tasks) if isinstance(review_tasks, list) else 0
    return {
        "modality": "audio",
        "evidence_record_count": evidence_count,
        "review_required": bool(payload.get("review_required")) or review_count > 0,
        "review_task_count": review_count,
        "artifact_keys": sorted((payload.get("artifacts") or {}).keys()) if isinstance(payload.get("artifacts"), dict) else [],
        "decision": str(payload.get("decision") or payload.get("conclusion") or ""),
    }



def _run_pipeline(task_id: str, input_paths: list[str], config_overrides: dict[str, Any]) -> None:
    # 后台任务入口：更新状态并执行完整管线。
    task = _tasks[task_id]
    task.status = TaskStatus.RUNNING
    task.stage = "pipeline"
    task.progress = 35
    task.status_label = _status_label(task.status)
    try:
        settings = get_settings()
        if config_overrides:
            # 仅允许覆盖 Settings 中存在的字段，避免注入无效配置。
            settings = _apply_config_overrides(settings, config_overrides)
        execution_route = _resolve_execution_route(settings, config_overrides)
        output_dir = settings.work_dir / task_id

        if execution_route == "api":
            operator_id = str(config_overrides.get("operator_id") or "").strip().upper()
            dataset_name = str(config_overrides.get("dataset_name") or "").strip() or f"{operator_id}-{task_id}"
            executor = AudioTextApiBridgeExecutor(settings=settings, run_id=task_id, output_dir=output_dir)
            decision = executor.execute(
                input_paths,
                operator_id=operator_id,
                dataset_name=dataset_name,
                config_overrides=config_overrides,
            )
        else:
            pipeline = AudioCompliancePipeline(settings=settings)
            # 让 pipeline run_id 与 task_id 对齐，便于 API 查询与目录索引一致。
            pipeline.run_id = task_id
            pipeline.output_dir = output_dir
            decision = pipeline.execute(input_paths)

        task.status = TaskStatus.COMPLETED
        task.result = decision
        task.completed_at = datetime.now(timezone.utc)
        task.stage = "completed"
        task.progress = 100
        task.status_label = _status_label(task.status)
    except Exception as exc:
        # 失败时记录异常堆栈并同步更新任务状态。
        logger.exception("Task %s failed", task_id)
        task.status = TaskStatus.FAILED
        task.error = str(exc)
        task.error_info = _error_info("AUDIO_PIPELINE_FAILED", "audio_pipeline", str(exc), retryable=False)
        task.completed_at = datetime.now(timezone.utc)
        task.status_label = _status_label(task.status)


@app.get("/api/v1/health")
async def health() -> dict[str, Any]:
    # 健康检查接口：返回服务版本与当前运行中任务计数。
    return {
        "status": "healthy",
        "service": "audio-compliance-checker",
        "version": "0.1.0",
        "active_tasks": sum(1 for task in _tasks.values() if task.status == TaskStatus.RUNNING),
    }


@app.post("/api/v1/check", response_model=CheckTaskInfo)
async def submit_check(request: CheckRequest, background_tasks: BackgroundTasks) -> CheckTaskInfo:
    # 提交任务仅入队并立即返回，不阻塞 HTTP 请求线程。
    contract_payload = {
        **request.config_overrides,
        "contract_version": request.contract_version,
        "platform_task_id": request.platform_task_id,
        "idempotency_key": request.idempotency_key,
        "operator_id": request.operator_id,
        "operator_catalog_version": request.operator_catalog_version,
    }
    existing_task = _existing_task_for(contract_payload)
    if existing_task is not None:
        return existing_task
    task_id = uuid.uuid4().hex
    task = CheckTaskInfo(task_id=task_id, status=TaskStatus.PENDING)
    _apply_task_contract(task, _effective_request(task_id, contract_payload))
    _tasks[task_id] = task
    background_tasks.add_task(_run_pipeline, task_id, request.input_paths, request.config_overrides)
    return task


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
    existing_task = _existing_task_for(overrides)
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

    task = CheckTaskInfo(task_id=task_id, status=TaskStatus.PENDING)
    _apply_task_contract(task, _effective_request(task_id, overrides))
    _tasks[task_id] = task
    background_tasks.add_task(_run_pipeline, task_id, [str(input_path)], overrides)
    return task


@app.get("/api/v1/status/{task_id}", response_model=CheckTaskInfo)
async def get_status(task_id: str) -> CheckTaskInfo:
    # 状态查询仅返回任务元信息，不携带最终结果体。
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return CheckTaskInfo(
        task_id=task.task_id,
        status=task.status,
        created_at=task.created_at,
        completed_at=task.completed_at,
        error=task.error,
        contract_version=task.contract_version,
        platform_task_id=task.platform_task_id,
        idempotency_key=task.idempotency_key,
        modality=task.modality,
        stage=task.stage,
        progress=task.progress,
        status_label=_status_label(task.status),
        effective_request=task.effective_request,
        warnings=task.warnings,
        error_info=task.error_info,
    )


@app.get("/api/v1/result/{task_id}")
async def get_result(task_id: str) -> dict[str, Any]:
    # 结果查询：按任务状态返回 202/500/200 语义化响应。
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
        "modality": "audio",
        "status": task.status.value,
        "status_label": _status_label(task.status),
        "created_at": task.created_at.isoformat(),
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "result": _serialize_result(task.result),
        "result_consistency": _result_consistency(task.result),
        "effective_request": task.effective_request,
        "warnings": task.warnings,
        "error_info": task.error_info,
    }


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    # 本地开发入口，生产场景通常由外部进程管理器拉起。
    uvicorn.run("audio.server:app", host=settings.server_host, port=settings.server_port, reload=True)
