"""Enum definitions for the video compliance blueprint."""

from __future__ import annotations

from enum import Enum


class VideoRouteType(str, Enum):
    """High-level route for video content."""

    # 以文档/屏幕内容为主的路线。
    SCREENCAST = "screencast"
    # 自然场景路线。
    NATURAL = "natural"
    # 混合场景路线。
    MIXED = "mixed"


class VideoJobStatus(str, Enum):
    """Lifecycle states for a video compliance job."""

    # 任务创建完成，尚未处理。
    CREATED = "created"
    PREPROCESSING = "preprocessing"
    SAMPLING = "sampling"
    DETECTING = "detecting"
    TRACKING = "tracking"
    AUDIO_PROCESSING = "audio_processing"
    POLICY_EVALUATING = "policy_evaluating"
    RENDERING = "rendering"
    DONE = "done"
    DROPPED = "dropped"
    FAILED = "failed"


class VideoDecisionType(str, Enum):
    """Final compliance decision."""

    # 可直接放行原视频。
    PASS_RAW = "pass_raw"
    # 需脱敏后放行。
    PASS_REDACTED = "pass_redacted"
    # 不可放行，直接拦截。
    DROP = "drop"
