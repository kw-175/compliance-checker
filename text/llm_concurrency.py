from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

from text.config.settings import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def map_concurrently(
    fn: Callable[[T], list[R]],
    items: list[T],
    *,
    step_name: str = "llm_step",
    max_workers: int | None = None,
) -> list[R]:
    """对彼此独立的单元并发执行 fn，并按输入顺序拼接各单元返回的记录。

    单元内部各自捕获异常并返回降级记录，语义与原串行 for 循环一致；
    max_workers <= 1 时完全退化为串行。并发度默认取 settings.llm_max_concurrency。
    """
    if not items:
        return []
    if max_workers is None:
        max_workers = get_settings().llm_max_concurrency
    if max_workers <= 1:
        merged: list[R] = []
        for item in items:
            merged.extend(fn(item))
        return merged

    merged: list[R] = []
    workers = min(max_workers, len(items))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=step_name) as executor:
        for records in executor.map(fn, items):
            merged.extend(records)
    return merged
