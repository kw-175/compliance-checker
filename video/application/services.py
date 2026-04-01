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
from audio.steps import load_jsonl, run_command
from picture.application.use_cases import create_orchestrator
from picture.domain.enums import DecisionType as PictureDecisionType
from picture.domain.enums import RouteType
from picture.domain.models import PictureFinding, PictureJob, SourceSpec
from picture.infra.config import PictureSettings, get_fresh_settings
from video.domain.enums import VideoDecisionType, VideoRouteType
from video.domain.models import FrameReference, TimeSpan, VideoFinding, VideoPolicyResult, VideoSegment

logger = logging.getLogger(__name__)

# 支持按文件后缀快速分流输入类型。
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_ANIMATED_SUFFIXES = {".gif", ".webp", ".png"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


@dataclass
class SequenceBundle:
    """Materialized frame sequence used by the video pipeline."""

    # source_kind 标识帧来源：目录、动画图或视频容器。
    source_kind: str
    frames: list[FrameReference]
    frame_durations_ms: list[int]
    total_input_frames: int
    total_duration_ms: int
    source_path: str = ""
    fps: float = 0.0
    has_native_audio: bool = False


def write_json(record: object, output_path: Path) -> None:
    # 通用 JSON 落盘：支持 pydantic 模型与普通对象。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        if hasattr(record, "model_dump_json"):
            handle.write(record.model_dump_json(indent=2))
        else:
            json.dump(record, handle, indent=2, ensure_ascii=False)


def write_jsonl(records: list[object], output_path: Path) -> None:
    # 通用 JSONL 落盘：逐行写入，便于流式读取。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            if hasattr(record, "model_dump_json"):
                handle.write(record.model_dump_json() + "\n")
            else:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_input_source(input_path: str, output_dir: Path) -> str:
    # 目录输入保持原引用；文件输入复制到工作目录隔离处理。
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
    max_frames: int = 0,
    default_frame_duration_ms: int = 250,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> SequenceBundle:
    # 根据输入形态自动分派到对应加载器。
    source = Path(input_path)
    if source.is_dir():
        return _load_from_directory(source, output_dir, max(1, frame_stride), max(0, max_frames), max(1, default_frame_duration_ms))
    if source.suffix.lower() in _VIDEO_SUFFIXES:
        return _load_from_video_container(
            source,
            output_dir,
            max(1, frame_stride),
            max(0, max_frames),
            max(1, default_frame_duration_ms),
            ffmpeg_bin,
            ffprobe_bin,
        )
    return _load_from_animated_image(source, output_dir, max(1, frame_stride), max(0, max_frames), max(1, default_frame_duration_ms))


def _load_from_directory(source: Path, output_dir: Path, frame_stride: int, max_frames: int, default_frame_duration_ms: int) -> SequenceBundle:
    # 目录模式：把静态图片序列标准化为统一帧清单。
    candidates = [path for path in sorted(source.iterdir()) if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES]
    if not candidates:
        raise ValueError(f"No image frames found in directory: {source}")

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    selected: list[Path] = []
    for index, path in enumerate(candidates):
        # 按 stride 采样，必要时受 max_frames 限制。
        if index % frame_stride != 0:
            continue
        selected.append(path)
        if max_frames and len(selected) >= max_frames:
            break

    frame_refs: list[FrameReference] = []
    durations: list[int] = []
    pts_ms = 0
    for sampled_index, path in enumerate(selected):
        # 统一转为 RGB PNG，避免后续处理受格式差异影响。
        destination = frames_dir / f"{source.stem}_frame_{sampled_index:05d}.png"
        with Image.open(path) as image:
            image.convert("RGB").save(destination)
        duration_ms = default_frame_duration_ms
        frame_refs.append(FrameReference(frame_index=sampled_index, pts_ms=pts_ms, image_uri=str(destination.resolve()), metadata={"source_index": sampled_index, "duration_ms": duration_ms, "source_path": str(path.resolve())}))
        durations.append(duration_ms)
        pts_ms += duration_ms

    return SequenceBundle("frame_directory", frame_refs, durations, len(candidates), sum(durations), str(source.resolve()))


def _load_from_animated_image(source: Path, output_dir: Path, frame_stride: int, max_frames: int, default_frame_duration_ms: int) -> SequenceBundle:
    # 动画图模式：逐帧读取并保留原始帧时长信息。
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
            # 对未采样帧累积时长，保证采样帧时间轴连续。
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
        # 尾部残余时长并入最后一个采样帧。
        frame_durations[-1] += pending_duration
        frame_refs[-1].metadata["duration_ms"] = frame_durations[-1]
    if not frame_refs:
        raise ValueError(f"No frames were sampled from source: {source}")

    return SequenceBundle("animated_image", frame_refs, frame_durations, total_frames, sum(frame_durations), str(source.resolve()))


def _load_from_video_container(
    source: Path,
    output_dir: Path,
    frame_stride: int,
    max_frames: int,
    default_frame_duration_ms: int,
    ffmpeg_bin: str,
    ffprobe_bin: str,
) -> SequenceBundle:
    # 视频容器模式：先探测元数据，再用 ffmpeg 按 stride 抽帧。
    metadata = probe_media(source, ffprobe_bin=ffprobe_bin)
    if not metadata:
        raise RuntimeError("FFmpeg/ffprobe metadata probing failed. Install FFmpeg and ensure ffprobe is available in PATH.")

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

    fps = float(metadata.get("fps", 0.0) or 0.0)
    total_duration_ms = int(metadata.get("duration_ms", 0) or 0)
    # 采样帧默认时长由 fps 与 stride 推导，缺失时回退配置值。
    sampled_frame_duration_ms = max(1, int(round((1000.0 / fps) * frame_stride))) if fps > 0 else default_frame_duration_ms

    frame_refs: list[FrameReference] = []
    durations: list[int] = []
    pts_ms = 0
    for index, path in enumerate(extracted):
        duration_ms = sampled_frame_duration_ms
        frame_refs.append(FrameReference(frame_index=index, pts_ms=pts_ms, image_uri=str(path.resolve()), metadata={"source_index": index * frame_stride, "duration_ms": duration_ms, "source_path": str(source.resolve())}))
        durations.append(duration_ms)
        pts_ms += duration_ms

    if total_duration_ms > 0 and durations:
        # 使用探测到的总时长校正最后一帧时长，减少累计误差。
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
    # 兼容 ffprobe 返回的分数字符串（如 30000/1001）。
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
    # 读取视频/音频流关键信息，为抽帧与音轨策略提供依据。
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
    # 从视频容器抽取单声道 16k WAV，便于复用 audio 流水线。
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


def build_picture_settings(base_dir: Path) -> PictureSettings:
    # 每次运行创建独立 picture 工作目录与存储目录。
    return get_fresh_settings(work_dir=base_dir / "picture_work", storage_base_path=base_dir / "picture_storage")


def analyze_frames(
    frames: list[FrameReference],
    tenant_id: str,
    profile: str,
    picture_settings: PictureSettings,
    options: dict[str, object] | None = None,
    max_workers: int = 1,
) -> list[PictureJob]:
    # 将视频级 options 映射到 picture 单帧分析参数。
    options = dict(options or {})
    frame_options = {
        "route_hint": str(options.get("route_hint", "auto")),
        "redaction_mode_text": str(options.get("redaction_mode_text", picture_settings.redaction_mode_text)),
        "redaction_mode_face": str(options.get("redaction_mode_face", picture_settings.redaction_mode_face)),
    }
    workers = max(1, min(max_workers, len(frames)))

    def _run(frame: FrameReference) -> PictureJob:
        # 每个并发任务独立创建 orchestrator，避免共享状态干扰。
        orchestrator = create_orchestrator(settings=picture_settings)
        job = PictureJob(tenant_id=tenant_id, source=SourceSpec(uri=frame.image_uri, mime_type="image/png"), profile=profile, options=frame_options)
        return orchestrator.execute(job)

    if workers == 1:
        # 单线程路径便于调试并保持结果顺序。
        return [_run(frame) for frame in frames]

    results: list[PictureJob | None] = [None] * len(frames)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # 并发执行后按原始帧序恢复结果顺序。
        future_map = {executor.submit(_run, frame): index for index, frame in enumerate(frames)}
        for future in as_completed(future_map):
            results[future_map[future]] = future.result()
    return [result for result in results if result is not None]


def derive_segments(frames: list[FrameReference], frame_jobs: list[PictureJob]) -> list[VideoSegment]:
    # 把连续同 route 帧压缩为片段，减少后续聚合复杂度。
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
        # route 切换时收束上一段并开启新段。
        segments.append(VideoSegment(span=TimeSpan(start_ms=start_ms, end_ms=end_ms), route=current_route, frame_ids=list(current_ids)))
        current_route = route
        current_ids = [frame.frame_id]
        start_ms = frame.pts_ms
        end_ms = frame_end_ms
    segments.append(VideoSegment(span=TimeSpan(start_ms=start_ms, end_ms=end_ms), route=current_route, frame_ids=list(current_ids)))
    return segments


def build_video_findings(frames: list[FrameReference], frame_jobs: list[PictureJob], iou_threshold: float = 0.4, gap_tolerance_ms: int = 1000) -> list[VideoFinding]:
    # 将逐帧 picture/safety 结果提升为带时间跨度的视频发现。
    aggregated: list[VideoFinding] = []
    for frame, frame_job in zip(frames, frame_jobs):
        frame_start = frame.pts_ms
        frame_end = frame.pts_ms + int(frame.metadata.get("duration_ms", 0))
        if frame_job.moderation_result and not frame_job.moderation_result.is_safe:
            for reason_code in frame_job.moderation_result.reason_codes:
                # 不安全审核结果也转为时序 finding，便于统一策略评估。
                _merge_video_finding(aggregated, VideoFinding(span=TimeSpan(start_ms=frame_start, end_ms=frame_end), frame_id=frame.frame_id, source_modality="safety", moderation=frame_job.moderation_result, reason_code=reason_code), iou_threshold, gap_tolerance_ms)
        for picture_finding in frame_job.findings:
            _merge_video_finding(aggregated, VideoFinding(span=TimeSpan(start_ms=frame_start, end_ms=frame_end), frame_id=frame.frame_id, source_modality="picture", picture_finding=picture_finding, reason_code=picture_finding.reason_code), iou_threshold, gap_tolerance_ms)
    return aggregated


def _merge_video_finding(aggregated: list[VideoFinding], candidate: VideoFinding, iou_threshold: float, gap_tolerance_ms: int) -> None:
    # 逆序扫描最近 finding，命中同签名且时空相邻则并段。
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
    # 签名判定：模态 + 类型/类别 + 文本片段（如有）。
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
    # 无区域信息时视为可合并；有区域时使用 IoU 过滤。
    if left is None or right is None or left.region is None or right.region is None:
        return True
    return _compute_iou(left.region.bbox, right.region.bbox) >= iou_threshold


def _compute_iou(left_bbox, right_bbox) -> float:  # type: ignore[no-untyped-def]
    # 计算矩形框 IoU，用于跨帧同目标聚合。
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
    # 采用多数投票得到视频级 route。
    if not frame_jobs:
        return VideoRouteType.MIXED
    mapping = {RouteType.DOCUMENT: VideoRouteType.SCREENCAST, RouteType.NATURAL: VideoRouteType.NATURAL, RouteType.MIXED: VideoRouteType.MIXED}
    return Counter(mapping.get(job.route or RouteType.MIXED, VideoRouteType.MIXED) for job in frame_jobs).most_common(1)[0][0]


def aggregate_policy(frame_jobs: list[PictureJob], profile: str, audio_decision: AudioDecision | None = None) -> VideoPolicyResult:
    # 视频决策按风险优先级折叠：DROP > PASS_REDACTED > PASS_RAW。
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
        # 音频高风险直接提升为 DROP。
        decision = VideoDecisionType.DROP
        reason_codes.append(f"AUDIO_{audio_decision.value.upper()}")
    elif audio_decision == AudioDecision.REVIEW and decision != VideoDecisionType.DROP:
        # 音频需复核时至少要求视频脱敏后放行。
        decision = VideoDecisionType.PASS_REDACTED
        reason_codes.append("AUDIO_REVIEW")
    deduped: list[str] = []
    seen: set[str] = set()
    # 保持原因码顺序去重，方便前端展示。
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
) -> tuple[str | None, str | None]:
    # 先拷贝逐帧合规图，再按来源类型回写为 MP4 或 GIF。
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
        compliant_paths.append(str(compliant_target))
        if render_preview:
            # 预览优先使用 overlay，缺失时回退原帧。
            preview_source = _resolve_artifact_path(frame_job.overlay_image_uri) if frame_job.overlay_image_uri else frame.image_uri
            shutil.copy2(preview_source, preview_target)
            preview_paths.append(str(preview_target))
    if sequence.source_kind == "video_container":
        compliant_video = output_dir / "compliant_video.mp4"
        # 容器输入优先回写 MP4，并可携带音轨。
        if compose_video_container(compliant_paths, sequence.frame_durations_ms, compliant_video, ffmpeg_bin, audio_path):
            preview_video: str | None = None
            if render_preview:
                preview_path = output_dir / "preview.mp4"
                if compose_video_container(preview_paths, sequence.frame_durations_ms, preview_path, ffmpeg_bin, None):
                    preview_video = str(preview_path.resolve())
            return str(compliant_video.resolve()), preview_video
        logger.warning("Falling back to GIF rendering because MP4 composition failed")
    # 非容器或 MP4 回写失败时回退 GIF，保证有可用产物。
    compliant_animation = output_dir / "compliant_video.gif"
    compose_gif(compliant_paths, sequence.frame_durations_ms, compliant_animation)
    preview_animation: str | None = None
    if render_preview:
        preview_animation_path = output_dir / "preview.gif"
        compose_gif(preview_paths, sequence.frame_durations_ms, preview_animation_path)
        preview_animation = str(preview_animation_path.resolve())
    return str(compliant_animation.resolve()), preview_animation


def compose_video_container(frame_paths: list[str], durations_ms: list[int], output_path: Path, ffmpeg_bin: str = "ffmpeg", audio_path: str | None = None) -> bool:
    # 通过 concat 清单按帧时长编码视频，可选复用音轨。
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
        # 音轨存在时追加第二输入并启用 AAC 编码。
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
    # 将 local:// URI 还原为本地路径。
    return uri.replace("local://", "") if uri.startswith("local://") else uri


def compose_gif(frame_paths: list[str], durations_ms: list[int], output_path: Path) -> None:
    # 将帧序列按给定时长拼接为 GIF。
    if not frame_paths:
        raise ValueError("No frames available for GIF rendering")
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    first, rest = images[0], images[1:]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first.save(output_path, save_all=True, append_images=rest, duration=durations_ms, loop=0, optimize=False)
    for image in images:
        # 显式关闭句柄，避免文件锁影响后续流程。
        image.close()


def resolve_sidecar_audio(input_path: str, options: dict[str, object] | None, enabled: bool = True, sidecar_extensions: list[str] | None = None) -> str | None:
    # sidecar 解析优先级：显式指定 > 目录 audio.* > 同名音轨。
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


def run_audio_sidecar(audio_path: str, work_dir: Path) -> tuple[AudioDecision | None, Path, str | None]:
    # 调用 audio 子流水线并解析脱敏音频清单。
    settings = get_audio_settings().model_copy(update={"work_dir": work_dir})
    pipeline = AudioCompliancePipeline(settings=settings)
    decision = pipeline.execute([audio_path])
    redacted_audio_path = None
    manifest_path = pipeline.output_dir / "redacted_audio_manifest.jsonl"
    if manifest_path.exists():
        records = load_jsonl(manifest_path)
        if records:
            redacted_audio_path = records[0].get("redacted_audio_path")
    return decision.overall_decision, pipeline.output_dir, redacted_audio_path
