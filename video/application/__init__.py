"""Application layer for the video compliance blueprint."""

# 统一从 application 层导出蓝图与阶段模型。
from video.application.orchestrator import PipelineStage, VideoComplianceBlueprint

__all__ = ["PipelineStage", "VideoComplianceBlueprint"]
