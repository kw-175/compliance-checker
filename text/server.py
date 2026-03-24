"""
FastAPI Microservice for Text Data Compliance Checking

Endpoints
---------
POST /api/v1/check         Submit a compliance check task
GET  /api/v1/status/{id}   Query task status
GET  /api/v1/result/{id}   Retrieve check result
GET  /api/v1/health        Health check
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from text.config.settings import Settings, get_settings
from text.models.schemas import (
    CheckRequest,
    CheckTaskInfo,
    PolicyDecision,
    TaskStatus,
)
from text.pipeline import CompliancePipeline

logger = logging.getLogger(__name__)

# ── In-memory task store ───────────────────────────────────
_tasks: dict[str, CheckTaskInfo] = {}


# ── Lifespan ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown hooks."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Text Compliance Checker service starting up")
    yield
    logger.info("Text Compliance Checker service shutting down")


# ── App ────────────────────────────────────────────────────
app = FastAPI(
    title="Text Data Compliance Checker",
    description=(
        "A microservice that orchestrates compliance checks on text data, "
        "including secret scanning, license compliance, PII detection, "
        "safety moderation, and policy decision."
    ),
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


# ── Background task runner ─────────────────────────────────
def _run_pipeline(task_id: str, input_paths: list[str], config_overrides: dict[str, Any]):
    """Execute the pipeline in a background thread."""
    task = _tasks[task_id]
    task.status = TaskStatus.RUNNING

    try:
        settings = get_settings()
        # Apply any runtime overrides
        for key, value in config_overrides.items():
            if hasattr(settings, key):
                setattr(settings, key, value)

        pipeline = CompliancePipeline(settings=settings)
        pipeline.run_id = task_id
        decision = pipeline.execute(input_paths)

        task.result = decision
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.utcnow()
        logger.info("Task %s completed: %s", task_id[:8], decision.overall_decision.value)

    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error = str(e)
        task.completed_at = datetime.utcnow()
        logger.exception("Task %s failed: %s", task_id[:8], e)


# ── Endpoints ──────────────────────────────────────────────

@app.get("/api/v1/health")
async def health_check():
    """Service health check."""
    return {
        "status": "healthy",
        "service": "text-compliance-checker",
        "version": "0.1.0",
        "active_tasks": sum(1 for t in _tasks.values() if t.status == TaskStatus.RUNNING),
    }


@app.post("/api/v1/check", response_model=CheckTaskInfo)
async def submit_check(request: CheckRequest, background_tasks: BackgroundTasks):
    """Submit a new compliance check task."""
    task_id = uuid.uuid4().hex
    task = CheckTaskInfo(task_id=task_id, status=TaskStatus.PENDING)
    _tasks[task_id] = task

    background_tasks.add_task(
        _run_pipeline,
        task_id,
        request.input_paths,
        request.config_overrides,
    )

    logger.info(
        "Accepted task %s with %d input paths",
        task_id[:8], len(request.input_paths),
    )
    return task


@app.get("/api/v1/status/{task_id}", response_model=CheckTaskInfo)
async def get_task_status(task_id: str):
    """Query the status of a running / completed task."""
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    # Return without full result to keep response small
    return CheckTaskInfo(
        task_id=task.task_id,
        status=task.status,
        created_at=task.created_at,
        completed_at=task.completed_at,
        error=task.error,
    )


@app.get("/api/v1/result/{task_id}")
async def get_task_result(task_id: str):
    """Retrieve the result of a completed task."""
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    if task.status == TaskStatus.PENDING:
        raise HTTPException(status_code=202, detail="Task is still pending")
    if task.status == TaskStatus.RUNNING:
        raise HTTPException(status_code=202, detail="Task is still running")
    if task.status == TaskStatus.FAILED:
        raise HTTPException(status_code=500, detail=f"Task failed: {task.error}")

    return {
        "task_id": task.task_id,
        "status": task.status.value,
        "created_at": task.created_at.isoformat(),
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "result": task.result.model_dump() if task.result else None,
    }


@app.get("/api/v1/tasks")
async def list_tasks(limit: int = 50):
    """List recent tasks."""
    tasks = sorted(_tasks.values(), key=lambda t: t.created_at, reverse=True)[:limit]
    return [
        {
            "task_id": t.task_id,
            "status": t.status.value,
            "created_at": t.created_at.isoformat(),
            "completed_at": t.completed_at.isoformat() if t.completed_at else None,
        }
        for t in tasks
    ]


# ── CLI entry point ───────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "text.server:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
    )
