from __future__ import annotations

import json
import shutil
import subprocess
import uuid
import wave
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from audio.models.schemas import Decision as AudioDecision
from picture.domain.models import PictureJob, SourceSpec
from video.application import services
from video.config.settings import Settings
from video.domain.enums import VideoDecisionType, VideoJobStatus
from video.pipeline import VideoCompliancePipeline


def _make_frame(color: tuple[int, int, int], offset: int) -> Image.Image:
    # 生成带几何块的测试帧，便于模拟视觉检测场景。
    image = Image.new("RGB", (800, 600), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle([100 + offset, 60, 350 + offset, 240], fill=(20, 20, 20))
    draw.rectangle([420, 360, 600, 520], fill=(240, 240, 240))
    return image


def _write_gif(path: Path) -> None:
    # 生成短动画 GIF 作为视频输入样本。
    frames = [_make_frame((240, 240, 240), 0), _make_frame((230, 230, 230), 20), _make_frame((220, 220, 220), 40)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[150, 150, 150], loop=0)


def _write_frame_directory(path: Path) -> None:
    # 生成帧目录输入样本。
    path.mkdir(parents=True, exist_ok=True)
    for index, color in enumerate([(240, 240, 240), (230, 230, 230), (220, 220, 220)]):
        _make_frame(color, index * 15).save(path / f"frame_{index:03d}.png")


def _write_sidecar_wav(path: Path) -> None:
    # 生成静音 wav，验证 sidecar 音频链路。
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)


def _make_case_dir(name: str) -> Path:
    # 为每个测试创建独立临时目录，避免产物互相污染。
    base = Path("video_test_runs") / f"{name}_{uuid.uuid4().hex[:8]}"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _make_settings(base: Path) -> Settings:
    # 统一测试配置，限制并发提高稳定性。
    return Settings(work_dir=base / "video_work", storage_base_path=base / "video_storage", max_workers=1)


def test_gif_pipeline_produces_redacted_output():
    # 验证 GIF 输入能走通完整流水线并产出脱敏放行结果。
    base = _make_case_dir("gif")
    input_path = base / "sample_video.gif"
    _write_gif(input_path)
    pipeline = VideoCompliancePipeline(settings=_make_settings(base))
    result = pipeline.execute(str(input_path), tenant_id="test-tenant", options={"route_hint": "mixed"})
    assert result.status == VideoJobStatus.DONE
    assert result.policy_result is not None
    assert result.policy_result.decision == VideoDecisionType.PASS_REDACTED
    assert result.asset is not None
    assert result.asset.compliant_video_uri is not None
    assert result.asset.report_uri is not None
    assert len(result.findings) > 0


def test_gif_pipeline_drops_explicit_content():
    # 验证显式不安全场景会触发 DROP。
    base = _make_case_dir("unsafe")
    input_path = base / "sample_unsafe_explicit.gif"
    _write_gif(input_path)
    pipeline = VideoCompliancePipeline(settings=_make_settings(base))
    result = pipeline.execute(str(input_path), tenant_id="test-tenant", options={"route_hint": "natural"})
    assert result.status == VideoJobStatus.DROPPED
    assert result.policy_result is not None
    assert result.policy_result.decision == VideoDecisionType.DROP


def test_directory_input_and_audio_sidecar():
    # 验证帧目录输入 + sidecar 音轨可正常接入 audio 流水线。
    base = _make_case_dir("dir_audio")
    frame_dir = base / "sequence"
    _write_frame_directory(frame_dir)
    _write_sidecar_wav(frame_dir / "audio.wav")
    pipeline = VideoCompliancePipeline(settings=_make_settings(base))
    result = pipeline.execute(str(frame_dir), tenant_id="test-tenant", options={"route_hint": "document"})
    assert result.status == VideoJobStatus.DONE
    assert result.asset is not None
    assert result.asset.audio_uri is not None
    assert "audio_pipeline" in result.step_latencies


def test_mp4_sequence_support_with_mocked_ffmpeg(monkeypatch: pytest.MonkeyPatch):
    # 通过 mock ffprobe/ffmpeg 验证容器抽帧逻辑，无需真实 FFmpeg 依赖。
    base = _make_case_dir("mp4_sequence")
    input_path = base / "sample_video.mp4"
    input_path.write_bytes(b"fake-mp4")

    def fake_run_command(command, timeout=300, ok_returncodes=(0,)):
        # 分别模拟 ffprobe 元数据输出和 ffmpeg 抽帧输出。
        if "ffprobe" in command[0]:
            payload = {
                "streams": [
                    {"codec_type": "video", "avg_frame_rate": "12/1", "width": 800, "height": 600, "nb_frames": "24"},
                    {"codec_type": "audio", "codec_name": "aac"},
                ],
                "format": {"duration": "2.0"},
            }
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
        if "ffmpeg" in command[0]:
            pattern = Path(command[-1])
            pattern.parent.mkdir(parents=True, exist_ok=True)
            for index in range(3):
                _make_frame((240 - index * 10, 240, 240), index * 10).save(pattern.parent / f"sample_video_frame_{index:05d}.png")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return None

    monkeypatch.setattr(services, "run_command", fake_run_command)
    sequence = services.load_sequence(str(input_path), base / "work", frame_stride=4, ffmpeg_bin="ffmpeg", ffprobe_bin="ffprobe")
    assert sequence.source_kind == "video_container"
    assert sequence.has_native_audio is True
    assert len(sequence.frames) == 3
    assert sequence.total_duration_ms == 2000


def test_mp4_render_support_with_mocked_ffmpeg(monkeypatch: pytest.MonkeyPatch):
    # 验证 MP4 回写渲染路径可生成 compliant/preview 视频。
    base = _make_case_dir("mp4_render")
    frames_dir = base / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    frame_refs = []
    for index in range(2):
        frame_path = frames_dir / f"frame_{index:03d}.png"
        _make_frame((240, 240 - index * 10, 240), index * 20).save(frame_path)
        frame_paths.append(str(frame_path.resolve()))
        frame_refs.append(services.FrameReference(frame_index=index, pts_ms=index * 500, image_uri=str(frame_path.resolve()), metadata={"duration_ms": 500}))
    sequence = services.SequenceBundle("video_container", frame_refs, [500, 500], 2, 1000, str(base / "sample.mp4"), 2.0, True)
    frame_jobs = [PictureJob(source=SourceSpec(uri=frame_path, mime_type="image/png")) for frame_path in frame_paths]

    def fake_run_command(command, timeout=300, ok_returncodes=(0,)):
        # 模拟 ffmpeg 成功写出目标视频文件。
        if "ffmpeg" in command[0]:
            output = Path(command[-1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"mp4")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return None

    monkeypatch.setattr(services, "run_command", fake_run_command)
    compliant_path, preview_path = services.render_sequence_outputs(sequence, frame_jobs, base / "rendered_output", decision=VideoDecisionType.PASS_REDACTED, ffmpeg_bin="ffmpeg")
    assert compliant_path is not None and compliant_path.endswith(".mp4")
    assert Path(compliant_path).exists()
    assert preview_path is not None and Path(preview_path).exists()


def test_mp4_pipeline_support_with_mocked_ffmpeg(monkeypatch: pytest.MonkeyPatch):
    # 端到端验证 MP4 输入在 mock 媒体工具下可完整运行。
    base = _make_case_dir("mp4_pipeline")
    input_path = base / "sample_video.mp4"
    input_path.write_bytes(b"fake-mp4")

    def fake_run_command(command, timeout=300, ok_returncodes=(0,)):
        # 按命令尾部输出类型模拟抽帧/抽音轨/封装视频三类行为。
        if "ffprobe" in command[0]:
            payload = {
                "streams": [
                    {"codec_type": "video", "avg_frame_rate": "10/1", "width": 800, "height": 600, "nb_frames": "20"},
                    {"codec_type": "audio", "codec_name": "aac"},
                ],
                "format": {"duration": "2.0"},
            }
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
        if "ffmpeg" in command[0]:
            output = Path(command[-1])
            if "%05d" in str(output):
                output.parent.mkdir(parents=True, exist_ok=True)
                for index in range(3):
                    _make_frame((240, 240 - index * 10, 240), index * 20).save(output.parent / f"sample_video_frame_{index:05d}.png")
            elif output.suffix == ".wav":
                output.parent.mkdir(parents=True, exist_ok=True)
                _write_sidecar_wav(output)
            elif output.suffix == ".mp4":
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"mp4")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return None

    monkeypatch.setattr(services, "run_command", fake_run_command)
    # 避免测试依赖真实 audio 模块重计算，直接固定音频决策。
    monkeypatch.setattr("video.pipeline.run_audio_sidecar", lambda audio_path, work_dir: (AudioDecision.ALLOW, work_dir, None))
    pipeline = VideoCompliancePipeline(settings=_make_settings(base).model_copy(update={"ffmpeg_bin": "ffmpeg", "ffprobe_bin": "ffprobe"}))
    result = pipeline.execute(str(input_path), tenant_id="test-tenant", options={"route_hint": "mixed"})
    assert result.status == VideoJobStatus.DONE
    assert result.asset is not None
    assert result.asset.compliant_video_uri is not None
    assert result.asset.compliant_video_uri.endswith(".mp4")
