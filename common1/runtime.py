# ──────────────────────────────────────────────────────────────
# 运行语义模型
# ──────────────────────────────────────────────────────────────
#
# 解决当前系统的"静默伪安全"问题：
# - provider 失效时默默 fallback → 显式记录 DegradeEvent
# - mock 在生产语义中存在 → 强制标记 UNTRUSTED
# - 降级事件未被写入证据链 → StepExecutionRecord 全程记录
# - 失败没有影响最终可信等级 → TrustEvaluator 计算综合可信度
# ──────────────────────────────────────────────────────────────

"""运行语义：StepExecutionRecord / PipelineExecutionContext / TrustEvaluator。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from common.enums import FailurePolicy, TrustLevel
from common.evidence import DegradeEvent

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StepExecutionRecord(BaseModel):
    """
    单步执行记录。

    追踪流水线中每个步骤的执行状态、provider 信息和耗时。
    相比仅用 logger.info 的现有方式，StepExecutionRecord
    是结构化数据，写入审计包后可供事后复盘。
    """
    step_name: str
    status: str = "pending"          # pending / running / completed / failed / degraded
    provider: str = ""               # 实际使用的 provider
    provider_version: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0
    input_count: int = 0             # 输入样本数
    output_count: int = 0            # 输出样本数
    error: Optional[str] = None
    degraded: bool = False           # 是否发生了降级
    degrade_event: Optional[DegradeEvent] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineExecutionContext(BaseModel):
    """
    流水线执行上下文。

    在整个流水线执行期间收集所有步骤的执行记录和降级事件。
    流水线结束后，由 TrustEvaluator 评估整体可信度。

    使用方式：
        ctx = PipelineExecutionContext(pipeline_run_id="abc")
        ctx.record_step_start("step_a")
        try:
            result = step_a.run(...)
            ctx.record_step_complete("step_a", provider="presidio")
        except Exception as e:
            ctx.record_step_failure("step_a", error=str(e), ...)
    """
    pipeline_run_id: str = ""
    step_records: list[StepExecutionRecord] = Field(default_factory=list)
    degrade_events: list[DegradeEvent] = Field(default_factory=list)
    failure_policy: FailurePolicy = FailurePolicy.FAIL_CLOSED

    def record_step_start(self, step_name: str, **kwargs) -> StepExecutionRecord:
        """记录步骤开始执行。"""
        record = StepExecutionRecord(
            step_name=step_name,
            status="running",
            started_at=_utcnow(),
            **kwargs,
        )
        self.step_records.append(record)
        return record

    def record_step_complete(
        self,
        step_name: str,
        provider: str = "",
        provider_version: str = "",
        output_count: int = 0,
        **kwargs,
    ) -> None:
        """记录步骤成功完成。"""
        record = self._find_record(step_name)
        if record:
            record.status = "completed"
            record.provider = provider
            record.provider_version = provider_version
            record.output_count = output_count
            record.completed_at = _utcnow()
            if record.started_at:
                record.duration_ms = (
                    record.completed_at - record.started_at
                ).total_seconds() * 1000

    def record_step_failure(
        self,
        step_name: str,
        error: str = "",
        fallback_provider: str = "",
        is_mock: bool = False,
    ) -> DegradeEvent:
        """
        记录步骤失败并生成降级事件。

        在 fail-closed 模式下，失败会上抬风险等级。
        """
        record = self._find_record(step_name)
        trust_impact = TrustLevel.UNTRUSTED if is_mock else TrustLevel.DEGRADED

        degrade = DegradeEvent(
            step_name=step_name,
            provider=record.provider if record else "",
            fallback_provider=fallback_provider,
            error_type=type(error).__name__ if not isinstance(error, str) else "",
            error_message=str(error),
            is_mock=is_mock,
            trust_impact=trust_impact,
        )
        self.degrade_events.append(degrade)

        if record:
            record.status = "degraded" if fallback_provider else "failed"
            record.degraded = True
            record.error = str(error)
            record.degrade_event = degrade
            record.completed_at = _utcnow()
            if record.started_at:
                record.duration_ms = (
                    record.completed_at - record.started_at
                ).total_seconds() * 1000

        logger.warning(
            "步骤 %s 降级: error=%s, fallback=%s, is_mock=%s",
            step_name, error, fallback_provider, is_mock,
        )
        return degrade

    def _find_record(self, step_name: str) -> Optional[StepExecutionRecord]:
        """查找最近的指定步骤记录。"""
        for rec in reversed(self.step_records):
            if rec.step_name == step_name:
                return rec
        return None

    def to_processing_timeline(self) -> dict[str, float]:
        """导出步骤 → 耗时(ms) 映射，用于审计包。"""
        return {
            rec.step_name: rec.duration_ms
            for rec in self.step_records
            if rec.duration_ms > 0
        }

    def get_provider_manifest(self) -> dict[str, str]:
        """导出步骤 → provider 版本映射，用于审计包。"""
        manifest: dict[str, str] = {}
        for rec in self.step_records:
            if rec.provider:
                key = rec.step_name
                value = f"{rec.provider}@{rec.provider_version}" if rec.provider_version else rec.provider
                manifest[key] = value
        return manifest


class TrustEvaluator:
    """
    可信等级评估器。

    根据执行上下文中的降级事件和步骤失败情况，
    综合评估流水线结果的可信程度。

    评估逻辑：
    - 无任何降级 → FULL
    - 存在 mock provider → UNTRUSTED
    - 关键步骤失败（如 safety/privacy）→ PARTIAL
    - 非关键步骤降级 → DEGRADED
    """

    # 关键步骤列表：这些步骤的降级会导致更严重的信任降级
    CRITICAL_STEPS = {
        "step_f_privacy", "step_f_privacy_detection",
        "step_g_safety", "step_g_safety_moderation",
        "step_i_policy_decision",
    }

    @classmethod
    def evaluate(cls, ctx: PipelineExecutionContext) -> TrustLevel:
        """评估整体可信等级。"""
        if not ctx.degrade_events:
            return TrustLevel.FULL

        has_mock = any(e.is_mock for e in ctx.degrade_events)
        if has_mock:
            return TrustLevel.UNTRUSTED

        has_critical_failure = any(
            e.step_name in cls.CRITICAL_STEPS
            for e in ctx.degrade_events
        )
        if has_critical_failure:
            return TrustLevel.PARTIAL

        return TrustLevel.DEGRADED

    @classmethod
    def build_explanation(cls, ctx: PipelineExecutionContext) -> str:
        """生成可信等级的人类可读解释。"""
        if not ctx.degrade_events:
            return "所有步骤均由生产级 provider 正常完成，结果完全可信。"

        parts = []
        for event in ctx.degrade_events:
            if event.is_mock:
                parts.append(
                    f"步骤 {event.step_name} 使用了 mock provider，"
                    f"结果不应用于生产决策。"
                )
            elif event.fallback_provider:
                parts.append(
                    f"步骤 {event.step_name} 的 provider 失败（{event.error_message}），"
                    f"已降级到 {event.fallback_provider}。"
                )
            else:
                parts.append(
                    f"步骤 {event.step_name} 执行失败（{event.error_message}），"
                    f"该步骤的检测结果缺失。"
                )
        return " ".join(parts)
