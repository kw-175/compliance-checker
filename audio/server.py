"""
FastAPI server for audio compliance checking.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from audio.config.settings import get_settings
from audio.models.schemas import CheckRequest, CheckTaskInfo, TaskStatus
from audio.pipeline import AudioCompliancePipeline

logger = logging.getLogger(__name__)
_tasks: dict[str, CheckTaskInfo] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Audio compliance service starting")
    yield
    logger.info("Audio compliance service stopping")


app = FastAPI(
    title="Audio Data Compliance Checker",
    description="Audio compliance checking microservice with normalization, ASR, privacy, safety, and policy stages.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])



def _run_pipeline(task_id: str, input_paths: list[str], config_overrides: dict[str, Any]) -> None:
    task = _tasks[task_id]
    task.status = TaskStatus.RUNNING
    try:
        settings = get_settings()
        if config_overrides:
            settings = settings.model_copy(update={key: value for key, value in config_overrides.items() if hasattr(settings, key)})
        pipeline = AudioCompliancePipeline(settings=settings)
        pipeline.run_id = task_id
        pipeline.output_dir = settings.work_dir / task_id
        decision = pipeline.execute(input_paths)
        task.status = TaskStatus.COMPLETED
        task.result = decision
        task.completed_at = datetime.now(timezone.utc)
    except Exception as exc:
        logger.exception("Task %s failed", task_id)
        task.status = TaskStatus.FAILED
        task.error = str(exc)
        task.completed_at = datetime.now(timezone.utc)


@app.get("/api/v1/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "audio-compliance-checker",
        "version": "0.1.0",
        "active_tasks": sum(1 for task in _tasks.values() if task.status == TaskStatus.RUNNING),
    }


@app.post("/api/v1/check", response_model=CheckTaskInfo)
async def submit_check(request: CheckRequest, background_tasks: BackgroundTasks) -> CheckTaskInfo:
    task_id = uuid.uuid4().hex
    task = CheckTaskInfo(task_id=task_id, status=TaskStatus.PENDING)
    _tasks[task_id] = task
    background_tasks.add_task(_run_pipeline, task_id, request.input_paths, request.config_overrides)
    return task


@app.get("/api/v1/status/{task_id}", response_model=CheckTaskInfo)
async def get_status(task_id: str) -> CheckTaskInfo:
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return CheckTaskInfo(
        task_id=task.task_id,
        status=task.status,
        created_at=task.created_at,
        completed_at=task.completed_at,
        error=task.error,
    )


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
        "status": task.status.value,
        "created_at": task.created_at.isoformat(),
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "result": task.result.model_dump() if task.result else None,
    }


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("audio.server:app", host=settings.server_host, port=settings.server_port, reload=True)
