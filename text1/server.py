from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from text.config.settings import get_settings
from text.models.schemas import CheckRequest, CheckTaskInfo, TaskStatus
from text.pipeline import CompliancePipeline

logger = logging.getLogger(__name__)

_tasks: dict[str, dict[str, Any]] = {}


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


@app.get("/api/v1/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "text-cleaned-package-checker",
        "active_tasks": sum(1 for task in _tasks.values() if task["status"] == TaskStatus.RUNNING),
    }


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
        "metadata": compliance_output.metadata,
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
