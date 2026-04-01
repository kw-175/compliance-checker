"""Abstract interfaces for the video compliance blueprint."""

from __future__ import annotations

import abc
from typing import Any

from picture.domain.models import PictureJob
from video.domain.models import FrameReference, TimeSpan, VideoFinding, VideoSegment


class VideoSourceLoader(abc.ABC):
    """Normalize and materialize a video source for downstream processing."""

    @abc.abstractmethod
    def prepare(self, source_uri: str, output_dir: str) -> str:
        # 将输入源标准化为本地可处理路径。
        """Return a local, normalized video path."""
        ...


class FrameSampler(abc.ABC):
    """Extract representative frames from a video."""

    @abc.abstractmethod
    def sample(self, video_path: str, output_dir: str) -> list[FrameReference]:
        # 负责抽帧并返回帧引用清单。
        """Return sampled frames saved under output_dir."""
        ...


class SceneSegmenter(abc.ABC):
    """Split a video into routeable temporal segments."""

    @abc.abstractmethod
    def segment(self, video_path: str, frames: list[FrameReference]) -> list[VideoSegment]:
        # 按时间轴把帧聚合为可路由片段。
        """Return temporal segments for routing and aggregation."""
        ...


class PictureFrameAnalyzer(abc.ABC):
    """Delegate a single frame to the existing picture compliance engine."""

    @abc.abstractmethod
    def analyze(
        self,
        frame: FrameReference,
        profile: str,
        options: dict[str, Any] | None = None,
    ) -> PictureJob:
        # 对单帧执行 picture 合规分析并返回 job 结果。
        """Return a picture job result for the provided frame."""
        ...


class TemporalTracker(abc.ABC):
    """Turn per-frame findings into track-aware video findings."""

    @abc.abstractmethod
    def track(
        self,
        frames: list[FrameReference],
        frame_jobs: list[PictureJob],
    ) -> list[VideoFinding]:
        # 将离散帧发现合并为时序连续发现。
        """Aggregate frame-level detections into time-aware findings."""
        ...


class AudioExtractor(abc.ABC):
    """Extract an audio track for reuse by the audio compliance engine."""

    @abc.abstractmethod
    def extract(self, video_path: str, output_dir: str) -> str | None:
        # 从视频中抽取可供 audio 流水线处理的音轨。
        """Return the extracted audio path, or None if no audio track exists."""
        ...


class VideoRenderer(abc.ABC):
    """Apply temporal redaction results back onto the video stream."""

    @abc.abstractmethod
    def render(
        self,
        video_path: str,
        findings: list[VideoFinding],
        output_path: str,
    ) -> str:
        # 根据时序发现回写合规视频。
        """Render a compliant video and return the output path."""
        ...

    def render_preview(
        self,
        video_path: str,
        spans: list[TimeSpan],
        output_path: str,
    ) -> str:
        # 预览默认可选，子类可覆盖为叠框/高亮效果。
        """Optional preview artifact for audit and QA."""
        return output_path
