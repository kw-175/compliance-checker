# ──────────────────────────────────────────────────────────────
# FastAPI 微服务入口 (API Server)
# ──────────────────────────────────────────────────────────────
#
# 功能：
#   提供 RESTful API 接口，用于提交合规检查任务、查询状态、
#   获取结果。使用 FastAPI 异步框架 + BackgroundTasks 后台执行。
#
# API 端点：
#   POST /api/v1/check         提交新的合规检查任务
#   GET  /api/v1/status/{id}   查询任务状态
#   GET  /api/v1/result/{id}   获取检查结果
#   GET  /api/v1/tasks         列出近期任务
#   GET  /api/v1/health        服务健康检查
#
# 任务生命周期：
#   1. 客户端 POST /api/v1/check → 创建任务（PENDING）
#   2. 后台线程执行流水线（RUNNING）
#   3. 执行完成（COMPLETED）或失败（FAILED）
#   4. 客户端 GET 查询状态和结果
#
# 注意事项：
#   - 任务状态存储在内存字典中，重启后丢失
#   - 生产环境应替换为 Redis / 数据库存储
# ──────────────────────────────────────────────────────────────

"""
FastAPI 微服务入口。

提供 REST API 接口用于提交合规检查任务、查询状态和获取结果。
使用 BackgroundTasks 在后台执行流水线。
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
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

# ── 内存任务存储 ─────────────────────────────────────────
# 使用字典存储所有任务的状态和结果
# 注意：仅适用于单实例部署，生产环境应替换为持久化存储
_tasks: dict[str, CheckTaskInfo] = {}


# ── 应用生命周期管理 ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用启动/关闭钩子。

    在启动时配置日志系统，关闭时执行清理操作。
    使用 FastAPI 新版的 lifespan 上下文管理器替代
    已弃用的 on_startup/on_shutdown。
    """
    # 启动时配置全局日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("文本合规检测服务启动中...")
    yield  # 应用运行期间
    logger.info("文本合规检测服务关闭中...")


# ── FastAPI 应用实例 ─────────────────────────────────────
app = FastAPI(
    title="Text Data Compliance Checker",
    description=(
        "文本数据合规检测微服务，集成密钥扫描、许可证合规、"
        "PII 检测、安全审核和策略决策等功能。"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 中间件：允许跨域请求（开发环境配置，生产环境应限制来源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 允许所有来源（生产应限制）
    allow_credentials=True,
    allow_methods=["*"],       # 允许所有 HTTP 方法
    allow_headers=["*"],       # 允许所有请求头
)


# ── 后台任务执行器 ───────────────────────────────────────
def _run_pipeline(task_id: str, input_paths: list[str], config_overrides: dict[str, Any]):
    """
    在后台线程中执行合规检测流水线。

    修正 Bug 7：使用 model_copy 创建 Settings 副本，
    避免 setattr 修改共享实例导致的并发竞态条件。

    Args:
        task_id: 任务 ID
        input_paths: 输入路径列表
        config_overrides: 运行时配置覆盖项
    """
    task = _tasks[task_id]
    task.status = TaskStatus.RUNNING

    try:
        # 修正 Bug 7：创建 Settings 副本而非修改共享实例
        # 多个并发任务使用独立的配置实例，避免互相干扰
        settings = get_settings()
        if config_overrides:
            # 使用 Pydantic model_copy(update=...) 安全创建副本
            valid_overrides = {
                k: v for k, v in config_overrides.items()
                if hasattr(settings, k)
            }
            settings = settings.model_copy(update=valid_overrides)

        # 创建并执行流水线
        pipeline = CompliancePipeline(settings=settings)
        pipeline.run_id = task_id  # 使用任务 ID 作为运行 ID
        compliance_output = pipeline.execute(input_paths)

        # 更新任务状态为已完成
        # 存储完整的 ComplianceOutput 对象
        task.result = compliance_output.legacy_decision  # 保留旧接口兼容性
        task._compliance_output = compliance_output       # 新增：完整输出
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now(timezone.utc)
        logger.info(
            "任务 %s 完成: decision=%s, trust=%s",
            task_id[:8],
            compliance_output.decision.value,
            compliance_output.trust_level.value,
        )

    except Exception as e:
        # 更新任务状态为失败
        task.status = TaskStatus.FAILED
        task.error = str(e)
        task.completed_at = datetime.now(timezone.utc)
        logger.exception("任务 %s 失败: %s", task_id[:8], e)


# ── API 端点 ─────────────────────────────────────────────

@app.get("/api/v1/health")
async def health_check():
    """
    服务健康检查端点。

    返回服务运行状态、版本和当前活跃任务数。
    用于负载均衡和容器编排的健康探测。
    """
    return {
        "status": "healthy",
        "service": "text-compliance-checker",
        "version": "0.1.0",
        "active_tasks": sum(1 for t in _tasks.values() if t.status == TaskStatus.RUNNING),
    }


@app.post("/api/v1/check", response_model=CheckTaskInfo)
async def submit_check(request: CheckRequest, background_tasks: BackgroundTasks):
    """
    提交新的合规检查任务。

    创建一个新任务并加入后台执行队列。立即返回任务 ID，
    客户端可通过 /status 和 /result 端点查询进度和结果。

    Args:
        request: 包含输入路径和配置覆盖的请求体
        background_tasks: FastAPI 后台任务管理器

    Returns:
        CheckTaskInfo 包含任务 ID 和初始状态
    """
    task_id = uuid.uuid4().hex
    task = CheckTaskInfo(task_id=task_id, status=TaskStatus.PENDING)
    _tasks[task_id] = task

    # 将流水线执行加入后台任务队列
    background_tasks.add_task(
        _run_pipeline,
        task_id,
        request.input_paths,
        request.config_overrides,
    )

    logger.info(
        "已接受任务 %s: %d 个输入路径",
        task_id[:8], len(request.input_paths),
    )
    return task


@app.get("/api/v1/status/{task_id}", response_model=CheckTaskInfo)
async def get_task_status(task_id: str):
    """
    查询任务状态。

    返回任务的当前状态、创建时间和完成时间，
    不包含完整结果数据（避免响应过大）。
    """
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 未找到")
    # 返回精简信息（不含完整 result 字段）
    return CheckTaskInfo(
        task_id=task.task_id,
        status=task.status,
        created_at=task.created_at,
        completed_at=task.completed_at,
        error=task.error,
    )


@app.get("/api/v1/result/{task_id}")
async def get_task_result(task_id: str):
    """
    获取已完成任务的检查结果。

    仅在任务状态为 COMPLETED 时返回结果。
    PENDING/RUNNING 返回 202，FAILED 返回 500。
    """
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 未找到")
    if task.status == TaskStatus.PENDING:
        raise HTTPException(status_code=202, detail="任务仍在等待中")
    if task.status == TaskStatus.RUNNING:
        raise HTTPException(status_code=202, detail="任务正在运行中")
    if task.status == TaskStatus.FAILED:
        raise HTTPException(status_code=500, detail=f"任务失败: {task.error}")

    # 构建增强结果（包含统一契约字段 + 向后兼容字段）
    response = {
        "task_id": task.task_id,
        "status": task.status.value,
        "created_at": task.created_at.isoformat(),
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }

    # 统一输出（新接口）
    co = getattr(task, "_compliance_output", None)
    if co:
        response.update({
            "decision": co.decision.value,
            "trust_level": co.trust_level.value,
            "degrade_summary": co.degrade_summary,
            "review_suggestions": co.review_suggestions,
            "explanation_summary": co.explanation_summary,
            "annotation_package_uri": co.annotation_package_uri,
            "audit_package_uri": co.audit_package_uri,
        })

    # 向后兼容（旧接口）
    response["legacy_decision"] = task.result.model_dump() if task.result else co.legacy_decision if co else None

    return response


@app.get("/api/v1/tasks")
async def list_tasks(limit: int = 50):
    """
    列出近期任务。

    按创建时间倒序排列，默认返回最近 50 条。
    仅返回摘要信息，不含完整结果。
    """
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


# ── CLI 入口点 ───────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "text.server:app",          # 模块路径
        host=settings.server_host,   # 监听地址
        port=settings.server_port,   # 监听端口
        reload=True,                 # 开发模式自动重载
    )
