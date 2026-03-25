# ──────────────────────────────────────────────────────────────
# 步骤 J – 血缘与审计 (Lineage & Audit)
# ──────────────────────────────────────────────────────────────
#
# 功能：
#   使用 OpenLineage Python 客户端为流水线的每个步骤记录
#   数据血缘事件（RunEvent），追踪数据在各步骤间的流转。
#
# 事件类型：
#   - START：步骤开始执行
#   - COMPLETE：步骤执行成功
#   - FAIL：步骤执行失败
#
# 传输方式：
#   - ConsoleTransport（默认/开发）：事件输出到标准输出
#   - HttpTransport（生产）：事件推送到 Marquez 后端
#
# Fallback 策略：
#   - openlineage-python 未安装 → 仅通过日志记录事件
#
# 在流水线中的位置：
#   贯穿整个流水线，为 B2/C/D/E1/F/G/H/I 每个步骤提供血缘追踪。
#   在 pipeline.py 的 _run_step 中统一调用。
# ──────────────────────────────────────────────────────────────

"""
步骤 J – 血缘与审计 (OpenLineage)。

为每个流水线步骤记录 START/COMPLETE/FAIL 事件。
支持 ConsoleTransport（开发）和 HttpTransport（生产 → Marquez）。
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from text.config.settings import Settings

logger = logging.getLogger(__name__)

# 模块级单例（OpenLineage 客户端缓存）
_client = None
_namespace = "compliance-checker"


def _get_client(settings: Settings):
    """
    延迟初始化 OpenLineage 客户端。

    根据配置选择传输方式：
    - openlineage_url 非空 → HttpTransport（推送到 Marquez）
    - openlineage_url 为空 → ConsoleTransport（输出到控制台）

    若 openlineage-python 未安装，返回 None（降级为纯日志记录）。

    Args:
        settings: 配置对象

    Returns:
        OpenLineageClient 实例或 None
    """
    global _client, _namespace
    if _client is not None:
        return _client

    try:
        from openlineage.client import OpenLineageClient
        from openlineage.client.transport.console import ConsoleTransport
        from openlineage.client.transport.http import HttpConfig, HttpTransport

        _namespace = settings.openlineage_namespace

        if settings.openlineage_url:
            # 生产模式：推送到 Marquez 等 OpenLineage 后端
            transport = HttpTransport(
                HttpConfig.from_dict({"url": settings.openlineage_url})
            )
            logger.info("OpenLineage: 使用 HttpTransport → %s", settings.openlineage_url)
        else:
            # 开发模式：输出到控制台
            transport = ConsoleTransport()
            logger.info("OpenLineage: 使用 ConsoleTransport（stdout）")

        _client = OpenLineageClient(transport=transport)
        return _client

    except ImportError:
        logger.warning(
            "openlineage-python 未安装；血缘事件将仅通过日志记录。"
            "安装: pip install openlineage-python"
        )
        return None


class LineageTracker:
    """
    血缘追踪器——用于在流水线执行过程中记录 OpenLineage 事件。

    封装了 OpenLineage RunEvent 的创建和发送逻辑，
    为每个步骤提供 start/complete/fail 三种事件方法。

    使用示例::

        tracker = LineageTracker(settings)
        run_id = tracker.start_step("step_c_text_extract")
        # ... 执行步骤 ...
        tracker.complete_step("step_c_text_extract", run_id)

    Attributes:
        settings: 配置对象
        client: OpenLineage 客户端（可能为 None）
        _runs: 步骤名 → run_id 的映射，用于自动关联事件
    """

    def __init__(self, settings: Settings):
        """
        初始化血缘追踪器。

        Args:
            settings: 配置对象
        """
        self.settings = settings
        self.client = _get_client(settings)
        self._runs: dict[str, str] = {}  # 步骤名 → run_id

    def start_step(
        self,
        step_name: str,
        inputs: list[dict[str, str]] | None = None,
        outputs: list[dict[str, str]] | None = None,
    ) -> str:
        """
        发送 START 事件并返回 run_id。

        记录一个步骤的开始执行，包括其输入和预期输出数据集。

        Args:
            step_name: 步骤名称（如 "step_c_text_extract"）
            inputs: 输入数据集列表（可选）
            outputs: 输出数据集列表（可选）

        Returns:
            生成的 run_id（用于后续的 complete/fail 调用）
        """
        run_id = uuid.uuid4().hex
        self._runs[step_name] = run_id

        # OpenLineage 客户端不可用时，仅记录日志
        if self.client is None:
            logger.info("[血缘] START  %s (run_id=%s)", step_name, run_id[:8])
            return run_id

        try:
            from openlineage.client.run import (
                InputDataset,
                Job,
                OutputDataset,
                Run,
                RunEvent,
                RunState,
            )

            # 构造 RunEvent
            event = RunEvent(
                eventType=RunState.START,
                eventTime=datetime.now(timezone.utc).isoformat(),
                run=Run(runId=run_id),
                job=Job(namespace=_namespace, name=step_name),
                inputs=[
                    InputDataset(namespace=_namespace, name=i.get("name", ""))
                    for i in (inputs or [])
                ],
                outputs=[
                    OutputDataset(namespace=_namespace, name=o.get("name", ""))
                    for o in (outputs or [])
                ],
                producer=f"compliance-checker/{step_name}",
            )
            self.client.emit(event)
            logger.debug("[血缘] 已发送 START 事件: %s", step_name)
        except Exception as e:
            logger.warning("[血缘] START 事件发送失败 %s: %s", step_name, e)

        return run_id

    def complete_step(
        self,
        step_name: str,
        run_id: str | None = None,
        outputs: list[dict[str, str]] | None = None,
    ) -> None:
        """
        发送 COMPLETE 事件。

        记录一个步骤的成功完成。

        Args:
            step_name: 步骤名称
            run_id: 运行 ID（可选，默认使用 start_step 返回的 ID）
            outputs: 输出数据集列表（可选）
        """
        run_id = run_id or self._runs.get(step_name, uuid.uuid4().hex)

        if self.client is None:
            logger.info("[血缘] COMPLETE %s (run_id=%s)", step_name, run_id[:8])
            return

        try:
            from openlineage.client.run import (
                Job,
                OutputDataset,
                Run,
                RunEvent,
                RunState,
            )

            event = RunEvent(
                eventType=RunState.COMPLETE,
                eventTime=datetime.now(timezone.utc).isoformat(),
                run=Run(runId=run_id),
                job=Job(namespace=_namespace, name=step_name),
                outputs=[
                    OutputDataset(namespace=_namespace, name=o.get("name", ""))
                    for o in (outputs or [])
                ],
                producer=f"compliance-checker/{step_name}",
            )
            self.client.emit(event)
            logger.debug("[血缘] 已发送 COMPLETE 事件: %s", step_name)
        except Exception as e:
            logger.warning("[血缘] COMPLETE 事件发送失败 %s: %s", step_name, e)

    def fail_step(
        self,
        step_name: str,
        run_id: str | None = None,
        error: str = "",
    ) -> None:
        """
        发送 FAIL 事件。

        记录一个步骤的执行失败。

        Args:
            step_name: 步骤名称
            run_id: 运行 ID（可选）
            error: 错误信息
        """
        run_id = run_id or self._runs.get(step_name, uuid.uuid4().hex)

        if self.client is None:
            logger.info("[血缘] FAIL %s (run_id=%s): %s", step_name, run_id[:8], error)
            return

        try:
            from openlineage.client.run import (
                Job,
                Run,
                RunEvent,
                RunState,
            )

            event = RunEvent(
                eventType=RunState.FAIL,
                eventTime=datetime.now(timezone.utc).isoformat(),
                run=Run(runId=run_id),
                job=Job(namespace=_namespace, name=step_name),
                producer=f"compliance-checker/{step_name}",
            )
            self.client.emit(event)
            logger.debug("[血缘] 已发送 FAIL 事件 %s: %s", step_name, error)
        except Exception as e:
            logger.warning("[血缘] FAIL 事件发送失败 %s: %s", step_name, e)
