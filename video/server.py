"""FastAPI server for video compliance checking."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from video.config.settings import get_settings
from video.models.schemas import CheckRequest, CheckTaskInfo, TaskStatus
from video.pipeline import VideoCompliancePipeline

logger = logging.getLogger(__name__)
# 进程内任务表：维护异步提交任务的状态与结果。
_tasks: dict[str, CheckTaskInfo] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 统一初始化日志格式，方便串联排查请求与后台任务。
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
# 为本地调试和跨域调用放开 CORS。
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def _run_pipeline(task_id: str, request: CheckRequest) -> None:
    # 后台任务入口：执行完整视频流水线并回写结果。
    task = _tasks[task_id]
    task.status = TaskStatus.RUNNING
    try:
        settings = get_settings()
        if request.config_overrides:
            # 仅允许覆盖存在于 Settings 的字段，避免脏配置注入。
            settings = settings.model_copy(
                update={key: value for key, value in request.config_overrides.items() if hasattr(settings, key)}
            )
        pipeline = VideoCompliancePipeline(settings=settings)
        # 让任务 ID 与 pipeline 运行目录保持一致，便于定位产物。
        pipeline.run_id = task_id
        pipeline.output_dir = settings.work_dir / task_id
        result = pipeline.execute(
            input_path=request.input_path,
            tenant_id=request.tenant_id,
            profile=request.profile,
            options=request.options,
        )
        task.status = TaskStatus.COMPLETED
        task.result = result.model_dump(mode="json")
        task.completed_at = datetime.now(timezone.utc)
    except Exception as exc:
        # 捕获异常并把任务标记为失败，避免后台线程静默崩溃。
        logger.exception("Task %s failed", task_id)
        task.status = TaskStatus.FAILED
        task.error = str(exc)
        task.completed_at = datetime.now(timezone.utc)


@app.get("/api/v1/health")
async def health() -> dict[str, Any]:
    # 健康检查：返回服务状态和当前运行中任务数。
    return {
        "status": "healthy",
        "service": "video-compliance-checker",
        "version": "0.1.0",
        "active_tasks": sum(1 for task in _tasks.values() if task.status == TaskStatus.RUNNING),
    }


@app.post("/api/v1/check", response_model=CheckTaskInfo)
async def submit_check(request: CheckRequest, background_tasks: BackgroundTasks) -> CheckTaskInfo:
    # 创建待执行任务并交给后台线程池处理。
    task_id = uuid.uuid4().hex
    task = CheckTaskInfo(task_id=task_id, status=TaskStatus.PENDING)
    _tasks[task_id] = task
    background_tasks.add_task(_run_pipeline, task_id, request)
    return task


@app.get("/api/v1/status/{task_id}", response_model=CheckTaskInfo)
async def get_status(task_id: str) -> CheckTaskInfo:
    # 状态查询直接返回任务对象快照。
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task


@app.get("/api/v1/result/{task_id}")
async def get_result(task_id: str) -> dict[str, Any]:
    # 结果查询按任务状态返回 202/500/200 的标准语义。
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
        "result": task.result,
    }


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    # 本地开发启动方式；生产环境建议使用进程管理器。
    uvicorn.run("video.server:app", host=settings.server_host, port=settings.server_port, reload=True)
