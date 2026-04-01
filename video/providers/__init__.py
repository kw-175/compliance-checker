"""Provider interfaces for the video compliance blueprint."""

# 统一导出 provider 抽象接口，供实现层按需落地。
from video.providers.base import (
    AudioExtractor,
    FrameSampler,
    PictureFrameAnalyzer,
    SceneSegmenter,
    TemporalTracker,
    VideoRenderer,
    VideoSourceLoader,
)

__all__ = [
    "AudioExtractor",
    "FrameSampler",
    "PictureFrameAnalyzer",
    "SceneSegmenter",
    "TemporalTracker",
    "VideoRenderer",
    "VideoSourceLoader",
]
