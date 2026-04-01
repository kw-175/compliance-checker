"""
Task executor for synchronous and async job execution.
"""
# 中文说明：该执行器把“同步执行”和“后台执行”两种模式统一包装起来，
# 方便 API 层按需要选择立即执行或异步提交。
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Simple executor that supports sync and async (background) execution."""

    def __init__(self, max_workers: int = 4) -> None:
        # 中文说明：底层直接使用线程池，适合当前以 I/O 和外部 provider 调用为主的场景。
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Submit a task for background execution. Returns a Future."""
        return self._pool.submit(fn, *args, **kwargs)

    def run_sync(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute a function synchronously."""
        # 中文说明：同步模式用于单测、脚本调试或希望立即拿到结果的场景。
        return fn(*args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the executor."""
        self._pool.shutdown(wait=wait)
