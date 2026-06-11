"""Scene windows and clip windows for video compliance analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from video.domain.models import FrameReference


def detect_scene_windows(
    frames: list[FrameReference],
    change_threshold: float = 0.18,
    min_scene_duration_ms: int = 1000,
) -> list[dict[str, Any]]:
    """Build lightweight scene windows from sampled frame visual differences."""
    if not frames:
        return []
    if len(frames) == 1:
        return [_scene_record(0, frames, 0, 0.0, "single_frame")]

    hashes = [_thumbnail_vector(Path(frame.image_uri)) for frame in frames]
    boundaries = [0]
    last_boundary_ms = frames[0].pts_ms
    scores: list[float] = [0.0]
    for index in range(1, len(frames)):
        score = _mean_abs_diff(hashes[index - 1], hashes[index])
        scores.append(score)
        elapsed = frames[index].pts_ms - last_boundary_ms
        if score >= change_threshold and elapsed >= min_scene_duration_ms:
            boundaries.append(index)
            last_boundary_ms = frames[index].pts_ms
    if boundaries[-1] != len(frames):
        boundaries.append(len(frames))

    scenes: list[dict[str, Any]] = []
    for scene_index, start_index in enumerate(boundaries[:-1]):
        end_exclusive = boundaries[scene_index + 1]
        end_index = max(start_index, end_exclusive - 1)
        reason = "scene_change" if scene_index > 0 else "start"
        scenes.append(_scene_record(scene_index, frames, start_index, scores[start_index], reason, end_index=end_index))
    return scenes


def build_clip_windows(
    frames: list[FrameReference],
    scene_windows: list[dict[str, Any]] | None = None,
    max_window_ms: int = 5000,
    overlap_ms: int = 1000,
) -> list[dict[str, Any]]:
    """Split scenes into bounded clip windows for short temporal moderation."""
    if not frames:
        return []
    frame_by_id = {frame.frame_id: frame for frame in frames}
    sources = scene_windows or [_scene_record(0, frames, 0, 0.0, "all_frames", end_index=len(frames) - 1)]
    windows: list[dict[str, Any]] = []
    for scene in sources:
        scene_frame_ids = [item for item in scene.get("frame_ids", []) if item in frame_by_id]
        if not scene_frame_ids:
            continue
        scene_frames = [frame_by_id[frame_id] for frame_id in scene_frame_ids]
        scene_start = int(scene.get("start_ms", scene_frames[0].pts_ms) or 0)
        scene_end = int(scene.get("end_ms", _frame_end_ms(scene_frames[-1])) or 0)
        step = max(1, max_window_ms - max(0, overlap_ms))
        cursor = scene_start
        while cursor < scene_end:
            window_end = min(scene_end, cursor + max_window_ms)
            window_frames = [
                frame for frame in scene_frames
                if frame.pts_ms < window_end and _frame_end_ms(frame) > cursor
            ]
            if window_frames:
                windows.append({
                    "window_id": f"clip_{len(windows) + 1:04d}",
                    "scene_id": scene.get("scene_id", ""),
                    "start_ms": cursor,
                    "end_ms": window_end,
                    "frame_ids": [frame.frame_id for frame in window_frames],
                    "frame_count": len(window_frames),
                    "source": "scene_window",
                })
            if window_end >= scene_end:
                break
            cursor += step
    return windows


def select_evenly_spaced_frames(frames: list[FrameReference], max_frames: int) -> list[FrameReference]:
    """Select a bounded set of representative frames without changing order."""
    if max_frames <= 0 or len(frames) <= max_frames:
        return list(frames)
    if max_frames == 1:
        return [frames[len(frames) // 2]]
    indexes = [
        round(index * (len(frames) - 1) / (max_frames - 1))
        for index in range(max_frames)
    ]
    selected: list[FrameReference] = []
    seen: set[int] = set()
    for index in indexes:
        if index in seen:
            continue
        seen.add(index)
        selected.append(frames[index])
    return selected


def _scene_record(
    scene_index: int,
    frames: list[FrameReference],
    start_index: int,
    change_score: float,
    reason: str,
    end_index: int | None = None,
) -> dict[str, Any]:
    end_index = len(frames) - 1 if end_index is None else end_index
    start_frame = frames[start_index]
    end_frame = frames[end_index]
    return {
        "scene_id": f"scene_{scene_index + 1:04d}",
        "start_ms": start_frame.pts_ms,
        "end_ms": _frame_end_ms(end_frame),
        "start_frame_id": start_frame.frame_id,
        "end_frame_id": end_frame.frame_id,
        "frame_ids": [frame.frame_id for frame in frames[start_index:end_index + 1]],
        "frame_count": max(0, end_index - start_index + 1),
        "change_score": round(float(change_score), 6),
        "boundary_reason": reason,
    }


def _frame_end_ms(frame: FrameReference) -> int:
    return frame.pts_ms + max(1, int(frame.metadata.get("duration_ms", 0) or 0))


def _thumbnail_vector(path: Path, size: int = 32) -> list[int]:
    try:
        with Image.open(path) as image:
            return list(image.convert("L").resize((size, size)).getdata())
    except Exception:
        return []


def _mean_abs_diff(left: list[int], right: list[int]) -> float:
    if not left or not right:
        return 0.0
    total = min(len(left), len(right))
    return sum(abs(left[index] - right[index]) for index in range(total)) / (255.0 * total)
