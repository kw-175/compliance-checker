"""Pydantic schemas for the video compliance service."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TaskStatus(str, Enum):
    """Task lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CheckRequest(BaseModel):
    """Request body for video checks."""

    contract_version: str = "compliance-job.v1"
    platform_task_id: str = ""
    idempotency_key: str = ""
    modality: str = "video"
    operator_id: str = ""
    operator_catalog_version: str = "video-compliance-operators.v1"
    input_path: str
    dataset_id: str = ""
    asset_id: str = ""
    cleaning_run_id: str = ""
    tenant_id: str = "default"
    profile: str = "default_cn_enterprise"
    task_context: dict[str, Any] = Field(default_factory=dict)
    operator_selection: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class CheckTaskInfo(BaseModel):
    """Task status model."""

    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    contract_version: str = "compliance-job.v1"
    platform_task_id: str = ""
    idempotency_key: str = ""
    modality: str = "video"
    stage: str = ""
    progress: int = 0
    status_label: str = ""
    effective_request: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error_info: dict[str, Any] | None = None
