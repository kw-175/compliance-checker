"""
In-memory job repository implementation.

Provides thread-safe in-memory storage for PictureJob entities.
Can be replaced with a database-backed implementation.
"""
# 中文说明：该文件实现了最简单的任务仓储，用于开发、测试和单机运行场景。
# 如果后续接数据库，只需要保持接口不变即可替换实现。
from __future__ import annotations

import logging
import threading
from typing import Any

from picture.domain.exceptions import JobNotFoundError
from picture.domain.models import PictureJob
from picture.providers.base import JobRepository

logger = logging.getLogger(__name__)


class InMemoryJobRepository(JobRepository):
    """Thread-safe in-memory job repository."""

    def __init__(self) -> None:
        # 中文说明：_jobs 用 job_id 做主键，_lock 用于保证多线程访问安全。
        self._jobs: dict[str, PictureJob] = {}
        self._lock = threading.Lock()

    def save_job(self, job: PictureJob) -> None:  # type: ignore[override]
        """Persist or update a job."""
        with self._lock:
            self._jobs[job.job_id] = job
            logger.debug("Job %s saved (status=%s)", job.job_id, job.status.value)

    def get_job(self, job_id: str) -> PictureJob:  # type: ignore[override]
        """Retrieve a job by ID."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job

    def list_jobs(  # type: ignore[override]
        self,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[PictureJob]:
        """List jobs, optionally filtered by tenant."""
        with self._lock:
            jobs = list(self._jobs.values())

        # 中文说明：如果指定 tenant_id，则只返回该租户的任务。
        if tenant_id:
            jobs = [j for j in jobs if j.tenant_id == tenant_id]

        # 中文说明：默认按创建时间倒序返回最近任务，并限制数量防止列表过大。
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)[:limit]
