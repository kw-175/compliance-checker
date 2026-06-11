"""Reusable service helpers for the video compliance pipeline."""

from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from audio.config.settings import get_settings as get_audio_settings
from audio.models.schemas import Decision as AudioDecision
from audio.pipeline import AudioCompliancePipeline
from audio.text_api_bridge import AudioTextApiBridgeExecutor
from audio.steps import load_jsonl, run_command
from picture.domain.enums import DecisionType as PictureDecisionType
from picture.domain.enums import RedactionMode
from picture.domain.enums import RouteType
from picture.domain.models import BBox, PictureFinding, PictureJob, RedactionOperation, RegionMask
from picture.providers.redaction.opencv_redactor import OpenCVRedactor
from video.application.picture_api_client import PictureApiConfig, PictureComplianceApiClient
from video.domain.enums import VideoDecisionType, VideoRouteType
from video.domain.models import FrameReference, TimeSpan, VideoActionPlan, VideoFinding, VideoPolicyResult, VideoSegment

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_ANIMATED_SUFFIXES = {".gif", ".webp", ".png"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


@dataclass
class SequenceBundle:
    """Materialized frame sequence used by the video pipeline."""

    source_kind: str
    frames: list[FrameReference]
    frame_durations_ms: list[int]
    total_input_frames: int
    total_duration_ms: int
    source_path: str = ""
    fps: float = 0.0
    has_native_audio: bool = False


def write_json(record: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        if hasattr(record, "model_dump_json"):
            handle.write(record.model_dump_json(indent=2))
        else:
            json.dump(record, handle, indent=2, ensure_ascii=False)


def write_jsonl(records: list[object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            if hasattr(record, "model_dump_json"):
                handle.write(record.model_dump_json() + "\n")
            else:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_input_source(input_path: str, output_dir: Path) -> str:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input path not found: {source}")
    if source.is_dir():
        return str(source.resolve())

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / source.name
    shutil.copy2(source, destination)
    return str(destination.resolve())


def load_sequence(
    input_path: str,
    output_dir: Path,
    frame_stride: int = 1,
    sample_fps: float = 0.0,
    max_frames: int = 0,
    default_frame_duration_ms: int = 250,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> SequenceBundle:
    source = Path(input_path)
    if source.is_dir():
        return _load_from_directory(source, output_dir, max(1, frame_stride), max(0, max_frames), max(1, default_frame_duration_ms))
    if source.suffix.lower() in _VIDEO_SUFFIXES:
        return _load_from_video_container(
            source,
            output_dir,
            max(1, frame_stride),
            max(0.0, sample_fps),
            max(0, max_frames),
            max(1, default_frame_duration_ms),
            ffmpeg_bin,
            ffprobe_bin,
        )
    return _load_from_animated_image(source, output_dir, max(1, frame_stride), max(0, max_frames), max(1, default_frame_duration_ms))


def _load_from_directory(source: Path, output_dir: Path, frame_stride: int, max_frames: int, default_frame_duration_ms: int) -> SequenceBundle:
    candidates = [path for path in sorted(source.iterdir()) if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES]
    if not candidates:
        raise ValueError(f"No image frames found in directory: {source}")

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    selected: list[Path] = []
    for index, path in enumerate(candidates):
        if index % frame_stride != 0:
            continue
        selected.append(path)
        if max_frames and len(selected) >= max_frames:
            break

    frame_refs: list[FrameReference] = []
    durations: list[int] = []
    pts_ms = 0
    for sampled_index, path in enumerate(selected):
        destination = frames_dir / f"{source.stem}_frame_{sampled_index:05d}.png"
        with Image.open(path) as image:
            image.convert("RGB").save(destination)
        duration_ms = default_frame_duration_ms
        frame_refs.append(FrameReference(frame_index=sampled_index, pts_ms=pts_ms, image_uri=str(destination.resolve()), metadata={"source_index": sampled_index, "duration_ms": duration_ms, "source_path": str(path.resolve())}))
        durations.append(duration_ms)
        pts_ms += duration_ms

    return SequenceBundle("frame_directory", frame_refs, durations, len(candidates), sum(durations), str(source.resolve()))


def _load_from_animated_image(source: Path, output_dir: Path, frame_stride: int, max_frames: int, default_frame_duration_ms: int) -> SequenceBundle:
    if source.suffix.lower() not in _ANIMATED_SUFFIXES:
        raise ValueError(
            f"Unsupported video source '{source.suffix}'. Current runtime supports animated GIF/WebP/APNG, frame directories, or FFmpeg-backed video containers."
        )

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_refs: list[FrameReference] = []
    frame_durations: list[int] = []
    pts_ms = 0
    total_frames = 0
    pending_duration = 0

    with Image.open(source) as image:
        n_frames = getattr(image, "n_frames", 1)
        for source_index in range(n_frames):
            image.seek(source_index)
            frame = image.convert("RGB")
            duration_ms = max(1, int(image.info.get("duration") or default_frame_duration_ms))
            pending_duration += duration_ms
            total_frames += 1
            if source_index % frame_stride != 0:
                continue
            if max_frames and len(frame_refs) >= max_frames:
                continue
            output_path = frames_dir / f"{source.stem}_frame_{len(frame_refs):05d}.png"
            frame.save(output_path)
            frame_refs.append(FrameReference(frame_index=len(frame_refs), pts_ms=pts_ms, image_uri=str(output_path.resolve()), metadata={"source_index": source_index, "duration_ms": pending_duration}))
            frame_durations.append(pending_duration)
            pts_ms += pending_duration
            pending_duration = 0

    if pending_duration and frame_durations:
        frame_durations[-1] += pending_duration
        frame_refs[-1].metadata["duration_ms"] = frame_durations[-1]
    if not frame_refs:
        raise ValueError(f"No frames were sampled from source: {source}")

    return SequenceBundle("animated_image", frame_refs, frame_durations, total_frames, sum(frame_durations), str(source.resolve()))


def _load_from_video_container(
    source: Path,
    output_dir: Path,
    frame_stride: int,
    sample_fps: float,
    max_frames: int,
    default_frame_duration_ms: int,
    ffmpeg_bin: str,
    ffprobe_bin: str,
) -> SequenceBundle:
    metadata = probe_media(source, ffprobe_bin=ffprobe_bin)
    if not metadata:
        raise RuntimeError("FFmpeg/ffprobe metadata probing failed. Install FFmpeg and ensure ffprobe is available in PATH.")
    fps = float(metadata.get("fps", 0.0) or 0.0)
    if sample_fps > 0 and fps > 0:
        frame_stride = max(frame_stride, int(round(fps / sample_fps)))

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = frames_dir / f"{source.stem}_frame_%05d.png"
    result = run_command([
        ffmpeg_bin,
        "-y",
        "-i",
        str(source),
        "-vf",
        f"select=not(mod(n\\,{frame_stride}))",
        "-vsync",
        "vfr",
        *( ["-frames:v", str(max_frames)] if max_frames else [] ),
        str(pattern),
    ], timeout=600)
    if result is None:
        raise RuntimeError("FFmpeg frame extraction failed. Install FFmpeg or verify the input container is readable.")

    extracted = sorted(frames_dir.glob(f"{source.stem}_frame_*.png"))
    if not extracted:
        raise RuntimeError(f"No frames were extracted from video source: {source}")

    total_duration_ms = int(metadata.get("duration_ms", 0) or 0)
    sampled_frame_duration_ms = max(1, int(round((1000.0 / fps) * frame_stride))) if fps > 0 else default_frame_duration_ms

    frame_refs: list[FrameReference] = []
    durations: list[int] = []
    pts_ms = 0
    for index, path in enumerate(extracted):
        duration_ms = sampled_frame_duration_ms
        frame_refs.append(FrameReference(frame_index=index, pts_ms=pts_ms, image_uri=str(path.resolve()), metadata={"source_index": index * frame_stride, "duration_ms": duration_ms, "source_path": str(source.resolve()), "sample_frame_stride": frame_stride, "sample_fps": sample_fps}))
        durations.append(duration_ms)
        pts_ms += duration_ms

    if total_duration_ms > 0 and durations:
        assigned = sum(durations[:-1])
        durations[-1] = max(1, total_duration_ms - assigned) if len(durations) > 1 else max(1, total_duration_ms)
        frame_refs[-1].metadata["duration_ms"] = durations[-1]

    return SequenceBundle(
        "video_container",
        frame_refs,
        durations,
        int(metadata.get("frame_count", len(extracted)) or len(extracted)),
        total_duration_ms or sum(durations),
        str(source.resolve()),
        fps,
        bool(metadata.get("has_audio", False)),
    )


def _parse_fraction(raw: str | None) -> float:
    if not raw:
        return 0.0
    text = str(raw)
    if "/" in text:
        left, right = text.split("/", 1)
        try:
            numerator = float(left)
            denominator = float(right)
            return numerator / denominator if denominator else 0.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def probe_media(path: Path, ffprobe_bin: str = "ffprobe") -> dict[str, object]:
    result = run_command([ffprobe_bin, "-v", "error", "-show_streams", "-show_format", "-print_format", "json", str(path)], timeout=120)
    if result is None:
        return {}
    try:
        payload = json.loads(result.stdout)
    except Exception:
        logger.warning("ffprobe output parse failed for %s", path)
        return {}

    streams = payload.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
    fmt = payload.get("format", {})
    duration_raw = fmt.get("duration") or video_stream.get("duration") or 0.0
    try:
        duration_ms = int(round(float(duration_raw) * 1000))
    except Exception:
        duration_ms = 0
    try:
        frame_count = int(video_stream.get("nb_frames") or 0)
    except Exception:
        frame_count = 0
    return {
        "fps": _parse_fraction(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")),
        "duration_ms": duration_ms,
        "frame_count": frame_count,
        "width": int(video_stream.get("width", 0) or 0),
        "height": int(video_stream.get("height", 0) or 0),
        "has_audio": bool(audio_stream),
        "video_codec": str(video_stream.get("codec_name", "")),
        "audio_codec": str(audio_stream.get("codec_name", "")),
    }


def extract_audio_track(video_path: str, output_dir: Path, ffmpeg_bin: str = "ffmpeg") -> str | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / (Path(video_path).stem + "_audio.wav")
    result = run_command([
        ffmpeg_bin,
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(target_path),
    ], timeout=600)
    if result is None or not target_path.exists():
        logger.warning("Native audio extraction failed for %s", video_path)
        return None
    return str(target_path.resolve())


def analyze_frames(
    frames: list[FrameReference],
    tenant_id: str,
    profile: str,
    options: dict[str, object] | None = None,
    max_workers: int = 1,
    picture_api_config: PictureApiConfig | None = None,
) -> list[PictureJob]:
    options = dict(options or {})
    frame_options = {
        "route_hint": str(options.get("route_hint", "auto")),
        "redaction_mode_text": str(options.get("redaction_mode_text", "black_box")),
        "redaction_mode_face": str(options.get("redaction_mode_face", "gaussian_blur")),
    }
    for key in (
        "enable_total_compliance",
        "enable_text_privacy_detection",
        "enable_text_content_detection",
        "enable_visual_safety_detection",
        "enable_visual_sensitive_object_detection",
        "disable_ocr",
        "disable_visual_safety",
        "disable_visual_sensitive_objects",
        "ordinary_dataset_enabled",
        "restricted_dataset_enabled",
        "privacy_operator_ids",
        "privacy_target_types",
        "content_safety_operator_ids",
        "content_safety_target_labels",
        "visual_safety_operator_ids",
        "visual_safety_target_labels",
        "visual_sensitive_object_operator_ids",
        "visual_sensitive_object_types",
        "picture_mode",
    ):
        if key in options:
            frame_options[key] = options[key]
    workers = max(1, min(max_workers, len(frames)))
    client = PictureComplianceApiClient(picture_api_config)
    client.check_health()
    skip_duplicates = bool(options.get("skip_duplicate_frames", True))
    diff_threshold = float(options.get("frame_difference_threshold", 0.02) or 0.02)
    submission_frames = _select_submission_frames(frames, diff_threshold) if skip_duplicates else frames
    if len(submission_frames) < len(frames):
        logger.info("Video frame duplicate cache reduced picture API calls from %s to %s", len(frames), len(submission_frames))

    def _run(frame: FrameReference) -> PictureJob:
        return client.run_frame(
            image_uri=frame.image_uri,
            tenant_id=tenant_id,
            profile=profile,
            options=frame_options,
        )

    if workers == 1:
        unique_results = [_run(frame) for frame in submission_frames]
        return _expand_cached_frame_results(frames, submission_frames, unique_results)

    workers = max(1, min(workers, len(submission_frames)))
    results: list[PictureJob | None] = [None] * len(submission_frames)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_run, frame): index for index, frame in enumerate(submission_frames)}
        for future in as_completed(future_map):
            results[future_map[future]] = future.result()
    unique_results = [result for result in results if result is not None]
    return _expand_cached_frame_results(frames, submission_frames, unique_results)


def _select_submission_frames(frames: list[FrameReference], difference_threshold: float) -> list[FrameReference]:
    selected: list[FrameReference] = []
    first_frame_by_hash: dict[tuple[int, ...], str] = {}
    last_hash: tuple[int, ...] | None = None
    for frame in frames:
        current_hash = _average_hash(Path(frame.image_uri))
        if current_hash is not None:
            frame.metadata["analysis_hash"] = _hash_to_hex(current_hash)
            cached_from = first_frame_by_hash.get(current_hash)
            if cached_from:
                frame.metadata["cached_from_frame_id"] = cached_from
                frame.metadata["cache_reason"] = "exact_frame_hash"
                continue
        if last_hash is not None and current_hash is not None:
            distance = _hash_distance(last_hash, current_hash)
            if distance <= difference_threshold:
                frame.metadata["cached_from_frame_id"] = selected[-1].frame_id if selected else ""
                frame.metadata["cache_reason"] = "similar_previous_frame"
                frame.metadata["cache_distance"] = distance
                continue
        selected.append(frame)
        if current_hash is not None:
            first_frame_by_hash.setdefault(current_hash, frame.frame_id)
            last_hash = current_hash
    return selected


def _average_hash(path: Path, size: int = 8) -> tuple[int, ...] | None:
    try:
        with Image.open(path) as image:
            pixels = list(image.convert("L").resize((size, size)).getdata())
    except Exception:
        return None
    avg = sum(pixels) / max(len(pixels), 1)
    return tuple(1 if pixel >= avg else 0 for pixel in pixels)


def _hash_distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    total = min(len(left), len(right))
    if total <= 0:
        return 1.0
    return sum(1 for a, b in zip(left, right) if a != b) / total


def _hash_to_hex(value: tuple[int, ...]) -> str:
    bits = "".join("1" if item else "0" for item in value)
    if not bits:
        return ""
    return f"{int(bits, 2):0{max(1, len(bits) // 4)}x}"


def _expand_cached_frame_results(
    all_frames: list[FrameReference],
    submitted_frames: list[FrameReference],
    submitted_results: list[PictureJob],
) -> list[PictureJob]:
    result_by_frame_id = {
        frame.frame_id: result
        for frame, result in zip(submitted_frames, submitted_results)
    }
    expanded: list[PictureJob] = []
    last_result: PictureJob | None = None
    for frame in all_frames:
        result = result_by_frame_id.get(frame.frame_id)
        if result is None:
            cached_from = str(frame.metadata.get("cached_from_frame_id") or "")
            result = result_by_frame_id.get(cached_from) or last_result
        if result is None:
            continue
        expanded.append(result)
        last_result = result
    return expanded


def derive_segments(frames: list[FrameReference], frame_jobs: list[PictureJob]) -> list[VideoSegment]:
    if not frames or not frame_jobs:
        return []
    route_map = {RouteType.DOCUMENT: VideoRouteType.SCREENCAST, RouteType.NATURAL: VideoRouteType.NATURAL, RouteType.MIXED: VideoRouteType.MIXED}
    segments: list[VideoSegment] = []
    current_route = route_map.get(frame_jobs[0].route or RouteType.MIXED, VideoRouteType.MIXED)
    current_ids = [frames[0].frame_id]
    start_ms = frames[0].pts_ms
    end_ms = frames[0].pts_ms + int(frames[0].metadata.get("duration_ms", 0))
    for frame, frame_job in zip(frames[1:], frame_jobs[1:]):
        route = route_map.get(frame_job.route or RouteType.MIXED, VideoRouteType.MIXED)
        frame_end_ms = frame.pts_ms + int(frame.metadata.get("duration_ms", 0))
        if route == current_route:
            current_ids.append(frame.frame_id)
            end_ms = frame_end_ms
            continue
        segments.append(VideoSegment(span=TimeSpan(start_ms=start_ms, end_ms=end_ms), route=current_route, frame_ids=list(current_ids)))
        current_route = route
        current_ids = [frame.frame_id]
        start_ms = frame.pts_ms
        end_ms = frame_end_ms
    segments.append(VideoSegment(span=TimeSpan(start_ms=start_ms, end_ms=end_ms), route=current_route, frame_ids=list(current_ids)))
    return segments


def build_video_findings(frames: list[FrameReference], frame_jobs: list[PictureJob], iou_threshold: float = 0.4, gap_tolerance_ms: int = 1000) -> list[VideoFinding]:
    aggregated: list[VideoFinding] = []
    for frame, frame_job in zip(frames, frame_jobs):
        frame_start = frame.pts_ms
        frame_end = frame.pts_ms + int(frame.metadata.get("duration_ms", 0))
        if frame_job.moderation_result and not frame_job.moderation_result.is_safe:
            for reason_code in frame_job.moderation_result.reason_codes:
                _merge_video_finding(aggregated, VideoFinding(span=TimeSpan(start_ms=frame_start, end_ms=frame_end), frame_id=frame.frame_id, source_modality="safety", moderation=frame_job.moderation_result, reason_code=reason_code), iou_threshold, gap_tolerance_ms)
        for picture_finding in frame_job.findings:
            _merge_video_finding(aggregated, VideoFinding(span=TimeSpan(start_ms=frame_start, end_ms=frame_end), frame_id=frame.frame_id, source_modality="picture", picture_finding=picture_finding, reason_code=picture_finding.reason_code), iou_threshold, gap_tolerance_ms)
    return aggregated


def _merge_video_finding(aggregated: list[VideoFinding], candidate: VideoFinding, iou_threshold: float, gap_tolerance_ms: int) -> None:
    for existing in reversed(aggregated):
        if not _same_finding_signature(existing, candidate):
            continue
        if candidate.span.start_ms - existing.span.end_ms > gap_tolerance_ms:
            continue
        if not _regions_compatible(existing.picture_finding, candidate.picture_finding, iou_threshold):
            continue
        existing.span.end_ms = max(existing.span.end_ms, candidate.span.end_ms)
        return
    aggregated.append(candidate)


def _same_finding_signature(left: VideoFinding, right: VideoFinding) -> bool:
    if left.source_modality != right.source_modality:
        return False
    if left.source_modality == "safety":
        return left.reason_code == right.reason_code
    if left.picture_finding is None or right.picture_finding is None:
        return False
    if left.picture_finding.finding_type != right.picture_finding.finding_type:
        return False
    if left.picture_finding.category != right.picture_finding.category:
        return False
    return (left.picture_finding.text_span or "") == (right.picture_finding.text_span or "")


def _regions_compatible(left: PictureFinding | None, right: PictureFinding | None, iou_threshold: float) -> bool:
    if left is None or right is None or left.region is None or right.region is None:
        return True
    return _compute_iou(left.region.bbox, right.region.bbox) >= iou_threshold


def _compute_iou(left_bbox, right_bbox) -> float:  # type: ignore[no-untyped-def]
    x1 = max(left_bbox.x, right_bbox.x)
    y1 = max(left_bbox.y, right_bbox.y)
    x2 = min(left_bbox.x + left_bbox.w, right_bbox.x + right_bbox.w)
    y2 = min(left_bbox.y + left_bbox.h, right_bbox.y + right_bbox.h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    union = left_bbox.w * left_bbox.h + right_bbox.w * right_bbox.h - intersection
    return intersection / union if union else 0.0


def resolve_video_route(frame_jobs: list[PictureJob]) -> VideoRouteType:
    if not frame_jobs:
        return VideoRouteType.MIXED
    mapping = {RouteType.DOCUMENT: VideoRouteType.SCREENCAST, RouteType.NATURAL: VideoRouteType.NATURAL, RouteType.MIXED: VideoRouteType.MIXED}
    return Counter(mapping.get(job.route or RouteType.MIXED, VideoRouteType.MIXED) for job in frame_jobs).most_common(1)[0][0]


def aggregate_policy(frame_jobs: list[PictureJob], profile: str, audio_decision: AudioDecision | None = None) -> VideoPolicyResult:
    decision = VideoDecisionType.PASS_RAW
    reason_codes: list[str] = []
    for frame_job in frame_jobs:
        if frame_job.policy_result is None:
            continue
        reason_codes.extend(frame_job.policy_result.reason_codes)
        if frame_job.policy_result.decision == PictureDecisionType.DROP:
            decision = VideoDecisionType.DROP
        elif frame_job.policy_result.decision == PictureDecisionType.PASS_REDACTED and decision != VideoDecisionType.DROP:
            decision = VideoDecisionType.PASS_REDACTED
    if audio_decision in {AudioDecision.REJECT, AudioDecision.QUARANTINE}:
        decision = VideoDecisionType.DROP
        reason_codes.append(f"AUDIO_{audio_decision.value.upper()}")
    elif audio_decision == AudioDecision.REVIEW and decision != VideoDecisionType.DROP:
        decision = VideoDecisionType.PASS_REDACTED
        reason_codes.append("AUDIO_REVIEW")
    deduped: list[str] = []
    seen: set[str] = set()
    for code in reason_codes:
        if code not in seen:
            seen.add(code)
            deduped.append(code)
    return VideoPolicyResult(decision=decision, reason_codes=deduped, profile=profile)


def render_sequence_outputs(
    sequence: SequenceBundle,
    frame_jobs: list[PictureJob],
    output_dir: Path,
    decision: VideoDecisionType,
    render_preview: bool = True,
    ffmpeg_bin: str = "ffmpeg",
    audio_path: str | None = None,
    action_plan: VideoActionPlan | None = None,
) -> tuple[str | None, str | None]:
    if decision == VideoDecisionType.DROP:
        return None, None
    compliant_dir = output_dir / "rendered" / "compliant_frames"
    preview_dir = output_dir / "rendered" / "preview_frames"
    compliant_dir.mkdir(parents=True, exist_ok=True)
    if render_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)
    compliant_paths: list[str] = []
    preview_paths: list[str] = []
    for frame, frame_job in zip(sequence.frames, frame_jobs):
        frame_path = Path(frame.image_uri)
        compliant_target = compliant_dir / frame_path.name
        preview_target = preview_dir / frame_path.name
        compliant_source = _resolve_artifact_path(frame_job.compliant_image_uri) if frame_job.compliant_image_uri else frame.image_uri
        shutil.copy2(compliant_source, compliant_target)
        _apply_track_redactions(compliant_target, frame, action_plan)
        compliant_paths.append(str(compliant_target))
        if render_preview:
            preview_source = _resolve_artifact_path(frame_job.overlay_image_uri) if frame_job.overlay_image_uri else frame.image_uri
            shutil.copy2(preview_source, preview_target)
            _apply_track_redactions(preview_target, frame, action_plan)
            preview_paths.append(str(preview_target))
    if sequence.source_kind == "video_container":
        compliant_video = output_dir / "compliant_video.mp4"
        if compose_video_container(compliant_paths, sequence.frame_durations_ms, compliant_video, ffmpeg_bin, audio_path):
            preview_video: str | None = None
            if render_preview:
                preview_path = output_dir / "preview.mp4"
                if compose_video_container(preview_paths, sequence.frame_durations_ms, preview_path, ffmpeg_bin, None):
                    preview_video = str(preview_path.resolve())
            return str(compliant_video.resolve()), preview_video
        logger.warning("Falling back to GIF rendering because MP4 composition failed")
    compliant_animation = output_dir / "compliant_video.gif"
    compose_gif(compliant_paths, sequence.frame_durations_ms, compliant_animation)
    preview_animation: str | None = None
    if render_preview:
        preview_animation_path = output_dir / "preview.gif"
        compose_gif(preview_paths, sequence.frame_durations_ms, preview_animation_path)
        preview_animation = str(preview_animation_path.resolve())
    return str(compliant_animation.resolve()), preview_animation


def compose_video_container(frame_paths: list[str], durations_ms: list[int], output_path: Path, ffmpeg_bin: str = "ffmpeg", audio_path: str | None = None) -> bool:
    if not frame_paths:
        return False
    concat_path = output_path.parent / (output_path.stem + "_concat.txt")
    concat_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for frame_path, duration_ms in zip(frame_paths, durations_ms):
        escaped = str(Path(frame_path).resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
        lines.append(f"duration {max(1, duration_ms) / 1000.0:.6f}")
    last_path = str(Path(frame_paths[-1]).resolve()).replace("'", "'\\''")
    lines.append(f"file '{last_path}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    command = [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_path)]
    if audio_path and Path(audio_path).exists():
        command.extend(["-i", audio_path])
    command.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    if audio_path and Path(audio_path).exists():
        command.extend(["-c:a", "aac", "-shortest"])
    else:
        command.append("-an")
    command.append(str(output_path))
    result = run_command(command, timeout=600)
    return result is not None and output_path.exists()


def _resolve_artifact_path(uri: str) -> str:
    return uri.replace("local://", "") if uri.startswith("local://") else uri


def _apply_track_redactions(image_path: Path, frame: FrameReference, action_plan: VideoActionPlan | None) -> None:
    if action_plan is None or not action_plan.operations:
        return
    operations = _frame_redaction_operations(image_path, frame, action_plan)
    if not operations:
        return
    OpenCVRedactor().redact(str(image_path), operations, str(image_path))


def _frame_redaction_operations(image_path: Path, frame: FrameReference, action_plan: VideoActionPlan) -> list[RedactionOperation]:
    width, height = _image_size(image_path)
    if width <= 0 or height <= 0:
        return []
    operations: list[RedactionOperation] = []
    for plan_op in action_plan.operations:
        if plan_op.modality == "audio" or plan_op.operation == "mute":
            continue
        mode = _redaction_mode(plan_op.operation)
        metadata = plan_op.metadata if isinstance(plan_op.metadata, dict) else {}
        for point in _points_for_frame(frame, metadata):
            bbox = _bbox_from_point(point, width, height)
            if bbox is None:
                continue
            operations.append(
                RedactionOperation(
                    finding_id=plan_op.risk_id,
                    region=RegionMask(
                        bbox=bbox,
                        confidence=float(point.get("confidence", 1.0) or 1.0),
                        mask_path=str(point.get("mask_path") or ""),
                    ),
                    mode=mode,
                    metadata={
                        "video_operation_id": plan_op.operation_id,
                        "track_id": plan_op.track_id or "",
                        "pts_ms": int(point.get("pts_ms", frame.pts_ms) or frame.pts_ms),
                        "source": str(point.get("source") or "track_redaction"),
                    },
                )
            )
        if not metadata.get("redaction_series"):
            for region in plan_op.regions:
                if str(region.get("frame_id") or "") != frame.frame_id:
                    continue
                bbox = _bbox_from_point(region, width, height)
                if bbox is None:
                    continue
                operations.append(
                    RedactionOperation(
                        finding_id=plan_op.risk_id,
                        region=RegionMask(
                            bbox=bbox,
                            confidence=float(region.get("confidence", 1.0) or 1.0),
                            mask_path=str(region.get("mask_path") or ""),
                        ),
                        mode=mode,
                        metadata={"video_operation_id": plan_op.operation_id, "track_id": plan_op.track_id or "", "source": "operation_region"},
                    )
                )
    return operations


def _points_for_frame(frame: FrameReference, metadata: dict[str, object]) -> list[dict[str, object]]:
    points = metadata.get("redaction_series")
    if not isinstance(points, list):
        return []
    frame_start = frame.pts_ms
    frame_end = frame.pts_ms + _frame_duration_ms(frame)
    matched: list[dict[str, object]] = []
    for item in points:
        if not isinstance(item, dict):
            continue
        item_frame_id = str(item.get("frame_id") or "")
        if item_frame_id and item_frame_id == frame.frame_id:
            matched.append(item)
            continue
        pts = _int_value(item.get("pts_ms"), -1)
        if not item_frame_id and frame_start <= pts < frame_end:
            matched.append(item)
    return matched


def _bbox_from_point(point: dict[str, object], width: int, height: int) -> BBox | None:
    raw = point.get("bbox") if isinstance(point.get("bbox"), dict) else point
    if not isinstance(raw, dict):
        return None
    x = _float_value(raw.get("x"), _float_value(raw.get("left"), 0.0))
    y = _float_value(raw.get("y"), _float_value(raw.get("top"), 0.0))
    w = _float_value(raw.get("w"), 0.0)
    h = _float_value(raw.get("h"), 0.0)
    if w <= 0 and "x2" in raw:
        w = _float_value(raw.get("x2"), 0.0) - x
    if h <= 0 and "y2" in raw:
        h = _float_value(raw.get("y2"), 0.0) - y
    normalized = max(abs(x), abs(y), abs(w), abs(h)) <= 1.5
    if normalized:
        x, y, w, h = x * width, y * height, w * width, h * height
    x, y, w, h = _expand_bbox(x, y, w, h, width, height)
    if w <= 1 or h <= 1:
        return None
    return BBox(x=x, y=y, w=w, h=h)


def _expand_bbox(x: float, y: float, w: float, h: float, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    pad_x = max(2.0, w * 0.08)
    pad_y = max(2.0, h * 0.08)
    x1 = max(0.0, x - pad_x)
    y1 = max(0.0, y - pad_y)
    x2 = min(float(image_width), x + w + pad_x)
    y2 = min(float(image_height), y + h + pad_y)
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def _redaction_mode(operation: str) -> RedactionMode:
    value = str(operation or "").strip().lower()
    if value == "gaussian_blur":
        return RedactionMode.GAUSSIAN_BLUR
    if value == "pixelate":
        return RedactionMode.PIXELATE
    if value == "solid_fill":
        return RedactionMode.SOLID_FILL
    return RedactionMode.BLACK_BOX


def _image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        logger.warning("Cannot read frame size for redaction: %s", path)
        return 0, 0


def _frame_duration_ms(frame: FrameReference) -> int:
    return max(1, _int_value(frame.metadata.get("duration_ms"), 1))


def _int_value(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _float_value(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def compose_gif(frame_paths: list[str], durations_ms: list[int], output_path: Path) -> None:
    if not frame_paths:
        raise ValueError("No frames available for GIF rendering")
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    first, rest = images[0], images[1:]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first.save(output_path, save_all=True, append_images=rest, duration=durations_ms, loop=0, optimize=False)
    for image in images:
        image.close()


def resolve_sidecar_audio(input_path: str, options: dict[str, object] | None, enabled: bool = True, sidecar_extensions: list[str] | None = None) -> str | None:
    options = options or {}
    explicit = options.get("sidecar_audio_path")
    if explicit:
        explicit_path = Path(str(explicit))
        if explicit_path.exists():
            return str(explicit_path.resolve())
    if not enabled:
        return None
    source = Path(input_path)
    if source.is_dir():
        for extension in sidecar_extensions or []:
            candidate = source / f"audio{extension}"
            if candidate.exists():
                return str(candidate.resolve())
        return None
    for extension in sidecar_extensions or []:
        candidate = source.with_suffix(extension)
        if candidate.exists():
            return str(candidate.resolve())
    return None


def run_audio_sidecar(audio_path: str, work_dir: Path, config_overrides: dict[str, object] | None = None) -> tuple[AudioDecision | None, Path, str | None]:
    base_settings = get_audio_settings()
    overrides = {
        key: value
        for key, value in dict(config_overrides or {}).items()
        if hasattr(base_settings, key)
    }
    overrides["work_dir"] = work_dir
    settings = base_settings.model_copy(update=overrides)
    route = str(getattr(settings, "audio_execution_route", "") or "").strip().lower()
    if route == "api":
        run_id = Path(audio_path).stem + "_audio_bridge"
        output_dir = work_dir / run_id
        executor = AudioTextApiBridgeExecutor(settings=settings, run_id=run_id, output_dir=output_dir)
        report = executor.execute(
            [audio_path],
            operator_id=str(dict(config_overrides or {}).get("operator_id") or "CMP_008"),
            dataset_name=str(dict(config_overrides or {}).get("dataset_name") or Path(audio_path).stem),
            config_overrides=dict(config_overrides or {}),
        )
        decision_value = str(report.get("decision") or report.get("conclusion") or "").lower()
        if decision_value in {"reject", "failed", "non_compliant"}:
            decision = AudioDecision.REJECT
        elif decision_value in {"review", "quarantine"}:
            decision = AudioDecision.REVIEW
        else:
            decision = AudioDecision.ALLOW
        pipeline_output_dir = output_dir
    else:
        pipeline = AudioCompliancePipeline(settings=settings)
        decision_model = pipeline.execute([audio_path])
        decision = decision_model.overall_decision
        pipeline_output_dir = pipeline.output_dir
    redacted_audio_path = None
    manifest_path = pipeline_output_dir / "redacted_audio_manifest.jsonl"
    if not manifest_path.exists():
        manifest_path = pipeline_output_dir / "31_redacted_audio_manifest.jsonl"
    if manifest_path.exists():
        records = load_jsonl(manifest_path)
        if records:
            redacted_audio_path = records[0].get("redacted_audio_path")
    return decision, pipeline_output_dir, redacted_audio_path
