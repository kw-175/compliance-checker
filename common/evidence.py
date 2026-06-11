# ──────────────────────────────────────────────────────────────
# 统一证据模型
# ──────────────────────────────────────────────────────────────
#
# 定义跨模态通用的证据基本单元、span 定位类型和降级事件。
# 各模态的 finding/hit 最终应能映射为 EvidenceUnit，
# 从而让下游系统以统一结构消费所有模态的检测发现。
# ──────────────────────────────────────────────────────────────

"""跨模态统一证据模型：EvidenceUnit / DegradeEvent / Span 类型。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from common.enums import Modality, TrustLevel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_evidence_id() -> str:
    return f"ev_{uuid.uuid4().hex[:12]}"


# ── Span 定位类型 ────────────────────────────────────────


class TextSpan(BaseModel):
    """文本片段定位：字符级起止位置。"""
    start: int
    end: int
    text_snippet: str = ""           # 命中的原始文本片段
    context_before: str = ""         # 命中前 N 字符的上下文
    context_after: str = ""          # 命中后 N 字符的上下文


class AudioTimeSpan(BaseModel):
    """音频片段定位：毫秒级起止时间。"""
    start_ms: int
    end_ms: int
    transcript_snippet: str = ""     # 对应转写文本


class ImageRegion(BaseModel):
    """图像区域定位：bbox + 可选 polygon。"""
    x: float
    y: float
    w: float
    h: float
    polygon_points: Optional[list[tuple[float, float]]] = None
    mask_uri: Optional[str] = None


class VideoTimeRegion(BaseModel):
    """视频时间+空间定位：时间跨度 + 帧内区域。"""
    start_ms: int
    end_ms: int
    frame_id: Optional[str] = None
    region: Optional[ImageRegion] = None


# ── 统一证据单元 ─────────────────────────────────────────


class EvidenceUnit(BaseModel):
    """
    统一证据单元。

    每个 EvidenceUnit 代表一条带有精确定位、可追溯、可解释的检测发现。
    无论来自文本 PII、音频转写、图像 OCR 还是视频帧级检测，
    最终都应能表达为 EvidenceUnit。

    与现有系统中 KeywordHit / RegexHit / PIIEntity / PictureFinding 的区别：
    - 统一了定位结构（span 字段根据模态使用对应类型）
    - 强制携带 provider/版本/阈值/上下文
    - 支持可解释性（explanation 字段）
    """
    evidence_id: str = Field(default_factory=_new_evidence_id)
    modality: Modality

    # 检测类别与细分
    category: str = ""               # 如 pii/safety/license/secret/keyword/regex
    sub_category: str = ""           # 如 person_name/phone_number/gpl/face/explicit

    # 精确定位（根据模态选择对应 span 类型）
    text_span: Optional[TextSpan] = None
    audio_span: Optional[AudioTimeSpan] = None
    image_region: Optional[ImageRegion] = None
    video_region: Optional[VideoTimeRegion] = None

    # 检测元数据
    confidence: float = 0.0
    provider: str = ""               # provider 名称，如 "presidio" / "qwen3guard"
    provider_version: str = ""       # provider 版本号
    model_version: str = ""          # 使用的具体模型版本
    threshold_used: float = 0.0      # 做出判定时使用的阈值
    processing_time_ms: float = 0.0  # 该证据产出的处理耗时

    # 可解释性
    explanation: str = ""            # 人类可读的判定原因
    reason_code: str = ""            # 机器可读的原因代码

    # 关联信息
    source_id: str = ""              # 关联的来源 ID
    doc_id: str = ""                 # 关联的文档/单元 ID
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── 降级事件 ─────────────────────────────────────────────


class DegradeEvent(BaseModel):
    """
    降级事件记录。

    当步骤中的 provider 失败、fallback 启用、mock 被使用等情况发生时，
    必须生成一条 DegradeEvent 写入证据链。

    与当前系统仅 logger.warning 的区别：
    - DegradeEvent 是结构化数据，会写入最终审计包
    - 影响 TrustLevel 计算
    - 在 fail-closed 模式下会上抬风险等级
    """
    event_id: str = Field(default_factory=lambda: f"deg_{uuid.uuid4().hex[:10]}")
    step_name: str                   # 发生降级的步骤名称
    provider: str = ""               # 原始 provider
    fallback_provider: str = ""      # 实际使用的 fallback provider
    error_type: str = ""             # 错误类型
    error_message: str = ""          # 错误详情
    is_mock: bool = False            # 是否使用了 mock provider
    trust_impact: TrustLevel = TrustLevel.DEGRADED  # 对可信等级的影响
    timestamp: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
