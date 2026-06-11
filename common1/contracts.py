# ──────────────────────────────────────────────────────────────
# 统一输入/输出契约接口
# ──────────────────────────────────────────────────────────────
#
# 定义 clean_sample 和 raw_input 两种入口的统一接口，
# 以及流水线统一输出契约（包含 ReleasePackage）。
# ──────────────────────────────────────────────────────────────

"""跨模态统一输入/输出契约接口。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from common.delivery import ReleasePackage
from common.enums import Modality, ProcessingMode, TrustLevel, UnifiedDecision


class CleanSampleInput(BaseModel):
    """
    清洗后样本的统一输入契约。

    当上游系统已完成数据清洗（去噪、格式化、基础过滤等），
    交付给本系统的样本应遵循此契约。

    与直接传入原始文件路径的 RAW_INPUT 模式的区别：
    - 不需要重做上游已完成的清洗或扫描
    - 跳过 source_intake / source_classify 步骤
    - 减少误伤率
    - 明确区分"生产入口"和"治理入口"
    """
    modality: Modality
    sample_id: str = ""              # 上游系统分配的样本 ID
    content_uri: str = ""            # 清洗后内容的 URI
    content_format: str = ""         # MIME type
    upstream_metadata: dict[str, Any] = Field(default_factory=dict)
    upstream_clean_hash: str = ""    # 上游清洗后的内容 hash
    processing_mode: ProcessingMode = ProcessingMode.CLEAN_SAMPLE

    # 上游已完成的检查结果（可选，避免重复检测）
    upstream_license_clear: bool = False    # 上游已确认无许可证问题
    upstream_secret_clear: bool = False     # 上游已确认无密钥泄露
    upstream_provenance: dict[str, Any] = Field(default_factory=dict)


class RawInput(BaseModel):
    """
    原始输入契约。

    从原始文件路径/URL 开始的完整流水线输入。
    """
    modality: Modality
    input_paths: list[str] = Field(default_factory=list)
    processing_mode: ProcessingMode = ProcessingMode.RAW_INPUT
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class ComplianceOutput(BaseModel):
    """
    统一输出契约。

    流水线的统一输出结构，包含：
    - 决策结果
    - 双轨交付物（ReleasePackage）
    - 可信等级
    - 降级摘要

    与现有系统仅返回 PolicyDecision 的区别：
    - 包含标注样本包和审计证据包的 URI
    - 包含可信等级
    - 包含降级摘要和复核建议
    """
    pipeline_run_id: str = ""
    modality: Modality = Modality.TEXT
    decision: UnifiedDecision = UnifiedDecision.REVIEW
    trust_level: TrustLevel = TrustLevel.FULL

    # 双轨交付物
    release_package: Optional[ReleasePackage] = None

    # URI 快捷索引（避免下游深度解引用）
    annotation_package_uri: str = ""
    audit_package_uri: str = ""

    # 摘要信息
    degrade_summary: str = ""
    review_suggestions: list[str] = Field(default_factory=list)
    explanation_summary: str = ""

    # 向后兼容：保留原始的 decision 结构供现有下游使用
    legacy_decision: Optional[dict[str, Any]] = None

    metadata: dict[str, Any] = Field(default_factory=dict)
