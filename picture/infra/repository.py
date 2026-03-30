"""
In-memory job repository implementation.

Provides thread-safe in-memory storage for PictureJob entities.
Can be replaced with a database-backed implementation.
"""

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

    def list_jobs(self, tenant_id: str | None = None, limit: int = 100) -> list[PictureJob]:  # type: ignore[override]
        """List jobs, optionally filtered by tenant."""
        with self._lock:
            jobs = list(self._jobs.values())
        if tenant_id:
            jobs = [j for j in jobs if j.tenant_id == tenant_id]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)[:limit]
