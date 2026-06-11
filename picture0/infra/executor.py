"""
Task executor for synchronous and async job execution.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Simple executor that supports sync and async (background) execution."""

    def __init__(self, max_workers: int = 4) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Submit a task for background execution. Returns a Future."""
        return self._pool.submit(fn, *args, **kwargs)

    def run_sync(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a function synchronously."""
        return fn(*args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the executor."""
        self._pool.shutdown(wait=wait)
