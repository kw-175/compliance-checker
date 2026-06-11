"""Optional video redaction execution.

Detection and policy do not call this module unless the action plan explicitly
requests a rendered derivative asset.
"""

from __future__ import annotations

from pathlib import Path

from picture.domain.models import PictureJob
from video.application.services import SequenceBundle, render_sequence_outputs
from video.domain.enums import VideoDecisionType
from video.domain.models import VideoActionPlan


def render_redacted_derivative(
    sequence: SequenceBundle,
    frame_jobs: list[PictureJob],
    action_plan: VideoActionPlan,
    output_dir: Path,
    render_preview: bool = True,
    ffmpeg_bin: str = "ffmpeg",
    audio_path: str | None = None,
) -> tuple[str | None, str | None]:
    """Render a derivative only when the action plan allows it."""
    if not action_plan.render_redacted_asset:
        return None, None
    return render_sequence_outputs(
        sequence=sequence,
        frame_jobs=frame_jobs,
        output_dir=output_dir,
        decision=VideoDecisionType.PASS_REDACTED,
        render_preview=render_preview,
        ffmpeg_bin=ffmpeg_bin,
        audio_path=audio_path,
        action_plan=action_plan,
    )
