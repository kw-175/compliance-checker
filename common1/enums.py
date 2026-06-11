# ──────────────────────────────────────────────────────────────
# 统一枚举定义
# ──────────────────────────────────────────────────────────────
#
# 将四个模态中反复出现且语义一致的枚举统一到此处，
# 各模态自身的枚举继续保留，但跨模态接口优先使用此处定义。
# ──────────────────────────────────────────────────────────────

"""跨模态统一枚举定义。"""

from __future__ import annotations

from enum import Enum


class Modality(str, Enum):
    """数据模态枚举。"""
    TEXT = "text"
    AUDIO = "audio"
    PICTURE = "picture"
    VIDEO = "video"


class UnifiedDecision(str, Enum):
    """
    统一决策枚举。

    统一了 text/audio 的 Decision（allow/review/quarantine/reject）
    与 picture/video 的 DecisionType（pass_raw/pass_redacted/drop）。

    映射关系：
      allow      ↔ pass_raw
      review     ↔ （需人工复核，暂不输出）
      quarantine ↔ pass_redacted（脱敏后输出）
      reject     ↔ drop
    """
    ALLOW = "allow"
    REVIEW = "review"
    QUARANTINE = "quarantine"
    REJECT = "reject"


class TrustLevel(str, Enum):
    """
    可信等级枚举。

    描述当前处理结果的整体可信程度：
    - FULL: 所有步骤均由生产级 provider 正常完成
    - DEGRADED: 存在降级（fallback provider 或部分步骤失败）
    - PARTIAL: 多个步骤降级或关键步骤失败
    - UNKNOWN: 无法确定可信等级（如 mock 混入生产）
    - UNTRUSTED: 使用了 mock provider 或关键 provider 不可用
    """
    FULL = "full"
    DEGRADED = "degraded"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    UNTRUSTED = "untrusted"


class FailurePolicy(str, Enum):
    """
    失败处理策略枚举。

    - FAIL_CLOSED: 步骤失败时趋向保守（上抬风险、阻止放行）
    - FAIL_OPEN: 步骤失败时趋向放行（继续流程、不上抬风险）
    """
    FAIL_CLOSED = "fail_closed"
    FAIL_OPEN = "fail_open"


class ProcessingMode(str, Enum):
    """
    处理模式枚举。

    - CLEAN_SAMPLE: 承接上游清洗后的数据，跳过原始采集与清洗步骤
    - RAW_INPUT: 从原始输入开始，执行完整流水线
    """
    CLEAN_SAMPLE = "clean_sample"
    RAW_INPUT = "raw_input"
