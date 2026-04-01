"""Pydantic schemas for the video compliance service."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    # 统一使用 UTC 时间戳。
    return datetime.now(timezone.utc)


class TaskStatus(str, Enum):
    """Task lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CheckRequest(BaseModel):
    """Request body for video checks."""

    # input_path 支持视频文件、动画图或帧目录。
    input_path: str
    tenant_id: str = "default"
    profile: str = "default_cn_enterprise"
    options: dict[str, Any] = Field(default_factory=dict)
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class CheckTaskInfo(BaseModel):
    """Task status model."""

    # result 在任务完成后回填完整 job 输出。
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
