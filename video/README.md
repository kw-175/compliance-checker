# Video Compliance Engine

`video/` 现在已经是一套可运行的视频数据合规处理引擎，支持两类执行模式：

- 纯 Pillow 路径：动画 GIF / WebP / APNG、帧目录输入
- FFmpeg 正式视频容器路径：`mp4 / mov / mkv / avi / m4v / webm`

它不是复制 `picture/` 的代码，而是把仓库里已经成熟的能力重新编排成一条视频时序流水线：

- 单帧视觉合规复用 `picture`
- sidecar 或 native 音轨合规复用 `audio`
- `video` 自己负责抽帧、时序聚合、视频级策略汇总、视频/动画回写渲染

## 1. 设计目标

这个模块解决的问题不是“识别一张图是否合规”，而是“给定一段视频型输入，如何在时间轴上持续检测、持续脱敏、持续输出可审计结果”。

实现时我坚持了几个原则：

1. 不重复实现 `picture` 已经做好的 OCR、PII、视觉检测、分割、脱敏能力。
2. 把 `video` 定位成时序编排层，而不是第四套独立模型堆栈。
3. 保持和 `audio/text/picture` 一致的工程风格：`pipeline.py`、`server.py`、Pydantic 模型、JSONL/JSON 落盘、pytest 测试。
4. 在没有 `ffmpeg` 时继续支持 GIF/帧目录；一旦环境有 `ffmpeg/ffprobe`，自动扩展到正式视频容器。

## 2. 当前已支持的输入类型

### 2.1 无 FFmpeg 也可运行

- 动画 GIF
- 动画 WebP / APNG（取决于 Pillow 能否正确打开）
- 帧目录输入（一个目录内包含连续 PNG/JPG 帧）

### 2.2 安装 FFmpeg 后可运行

- MP4
- MOV
- MKV
- AVI
- M4V
- WEBM

### 2.3 音轨来源

模块支持两种音轨接入方式：

- sidecar 音轨：
  - 目录输入自动寻找 `audio.wav` / `audio.mp3`
  - 文件输入自动寻找同名 `.wav` / `.mp3`
  - 也可以通过 `options.sidecar_audio_path` 显式指定
- native 音轨：
  - 当输入是 `mp4/mov/mkv/...`，且系统存在 `ffmpeg` 时，会自动抽取原生音轨

## 3. 为什么这样复用 `picture`

`picture/` 已经是一个完整的单帧合规引擎，具备：

- route 机制：`document / natural / mixed`
- OCR 与 layout
- 文本 PII 检测
- 视觉检测
- 安全审核
- 分割精修
- 脱敏渲染
- 策略评估

所以在 `video` 里，正确的做法不是重写这些步骤，而是：

1. 把视频拆成帧
2. 对每一帧直接调用 `picture` 的编排器
3. 把结果重新映射回时间轴

这就是本模块的核心复用思想。

## 4. 为什么还要复用 `audio`

视频是天然多模态对象，仅做画面检测是不够的。一个录屏视频可能画面完全合规，但音轨中出现手机号、身份证号、辱骂、涉政或暴恐表达。

因此当前实现支持两类音频流程：

1. 发现 sidecar 音轨并调用 `audio.pipeline.AudioCompliancePipeline`
2. 对正式视频容器用 `ffmpeg` 抽取 native 音轨，再调用同一套 `audio` 流水线

视频级策略会把音轨结果折叠到最终决策中：

- `ALLOW` 不升级风险
- `REVIEW` 会把视频整体提升到 `pass_redacted`
- `REJECT / QUARANTINE` 会把视频整体提升到 `drop`

## 5. 模块结构

```text
video/
├── __init__.py
├── README.md
├── pipeline.py
├── server.py
├── requirements.txt
├── application/
│   ├── __init__.py
│   ├── orchestrator.py
│   └── services.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── domain/
│   ├── __init__.py
│   ├── enums.py
│   └── models.py
├── models/
│   ├── __init__.py
│   └── schemas.py
├── providers/
│   ├── __init__.py
│   └── base.py
└── tests/
    ├── __init__.py
    └── test_pipeline.py
```

各层职责如下：

- `config/`: 视频模块配置，包括 `ffmpeg/ffprobe` 路径
- `domain/`: 视频任务、片段、时间跨度、视频发现等核心模型
- `models/`: FastAPI 请求/响应 schema
- `application/services.py`: 抽帧、逐帧分析、时序聚合、渲染、音轨接入等可复用服务函数
- `pipeline.py`: 主流水线编排器
- `server.py`: FastAPI 服务入口
- `tests/`: 真实可运行回归测试

## 6. 核心执行链路

### 6.1 输入标准化

`prepare_input_source()` 负责把输入复制到工作目录（目录输入则直接引用目录）。这样后续产物和原始输入有明确边界。

### 6.2 抽帧

`load_sequence()` 现在支持三种来源：

- 动画图像：使用 Pillow 逐帧读取，并保留每帧时长
- 帧目录：按文件序列加载并附加默认帧时长
- 正式视频容器：
  - 用 `ffprobe` 获取 fps、duration、codec、是否有音轨
  - 用 `ffmpeg` 结合 `select=not(mod(n,stride))` 做抽帧
  - 为采样帧构造时间戳和持续时长

这意味着 `video` 现在已经具备真正的 `mp4 + ffmpeg` 容器处理能力，而不是只处理 GIF。

### 6.3 单帧视觉合规

`analyze_frames()` 会对每个采样帧调用图片合规 API，而不是在视频进程内直接加载 `picture` pipeline。视频服务要求图片合规 API 可访问，默认地址为 `http://127.0.0.1:19012`。

每个采样帧会提交到：

- `POST /v1/picture/jobs`
- `GET /v1/picture/jobs/{job_id}`
- `GET /v1/picture/jobs/{job_id}/report`

视频侧会把图片 API 的完整 report 还原为 `PictureJob` 等价结构，再继续执行视频时间轴聚合、治理策略和可选渲染。

视频侧的算子裁剪会随每个帧任务传入图片 API 的 `options`，包括：

- `visual_sensitive_object_operator_ids`
- `visual_sensitive_object_types`
- `visual_safety_operator_ids`
- `visual_safety_target_labels`
- `privacy_operator_ids`
- `privacy_target_types`
- `content_safety_operator_ids`
- `content_safety_target_labels`

复用的是整条图像合规链，所以每帧天然支持：

- `route_hint`
- OCR
- PII
- 安全审核
- 视觉检测
- segmentation
- redaction
- policy decision

### 6.4 时序聚合

`build_video_findings()` 把逐帧结果提升为视频时间轴上的发现：

- 同类别、相邻时间段、区域 IoU 足够高的 finding 会被合并
- 安全审核结果也会被拉平为时间跨度 finding
- 最终得到 `VideoFinding(span, frame_id, picture_finding/moderation)`

这里的实现仍然是轻量级 track 聚合，但已经能把离散帧结果提升成视频级连续区段。

### 6.5 音轨处理

音轨流程分为两步：

1. `resolve_sidecar_audio()` 尝试找到 sidecar 音轨
2. 如果没有 sidecar 且输入是视频容器，则 `extract_audio_track()` 使用 `ffmpeg` 抽取 native 音轨

之后统一调用 `run_audio_sidecar()`，内部直接复用 `audio.pipeline.AudioCompliancePipeline`。

如果音频流水线产出了 `redacted_audio_manifest.jsonl`，视频渲染优先使用音频脱敏后的产物，而不是原音轨。

### 6.6 视频级策略

`aggregate_policy()` 采用保守策略：

- 任何帧被 `picture` 判成 `drop`，视频整体 `drop`
- 任意帧需要脱敏，则视频整体 `pass_redacted`
- sidecar/native 音轨若为 `reject/quarantine`，视频整体 `drop`
- sidecar/native 音轨若为 `review`，视频整体 `pass_redacted`

### 6.7 回写渲染

`render_sequence_outputs()` 会根据输入来源自动选择输出方式：

- 对 GIF / 帧目录：输出 `compliant_video.gif` 与 `preview.gif`
- 对 MP4 / MOV / MKV 等容器：
  - 用 concat 清单拼接脱敏帧
  - 用 `ffmpeg` 渲染 `compliant_video.mp4`
  - 如果有可用音轨，自动 mux 到输出视频中

如果 MP4 合成失败，会自动回退到 GIF 预览路径，保证至少还能产出视觉审计结果。

## 7. 关键数据模型

在 `video/domain/models.py` 里，最重要的模型有：

- `TimeSpan`: 视频时间范围
- `FrameReference`: 单帧引用
- `VideoSegment`: 连续片段
- `VideoFinding`: 带时间维度的视频发现
- `VideoPolicyResult`: 视频级最终策略结果
- `VideoJob`: 顶层任务实体
- `VideoReport`: 可审计报告模型

这里的设计取舍很明确：

- 空间信息继续复用 `picture` 的 `PictureFinding` / `BBox` / `RegionMask`
- 视频层只补时间轴，而不复制空间结构

这样后期维护成本最低。

## 8. 配置项

核心配置定义在 `video/config/settings.py`，包括：

- `VIDEO_WORK_DIR`
- `VIDEO_STORAGE_BASE_PATH`
- `VIDEO_FRAME_STRIDE`
- `VIDEO_MAX_FRAMES`
- `VIDEO_DEFAULT_FRAME_DURATION_MS`
- `VIDEO_MAX_WORKERS`
- `VIDEO_SCENE_DETECTION_ENABLED`
- `VIDEO_SCENE_CHANGE_THRESHOLD`
- `VIDEO_CLIP_WINDOW_MS`
- `VIDEO_CLIP_WINDOW_OVERLAP_MS`
- `VIDEO_CLIP_MODERATION_ENABLED`
- `VIDEO_CLIP_MODERATION_BASE_URL`
- `VIDEO_CLIP_MODERATION_ENDPOINT`
- `VIDEO_CLIP_MODERATION_MAX_FRAMES`
- `VIDEO_CLIP_MODERATION_CONFIDENCE_THRESHOLD`
- `VIDEO_TRACK_GAP_TOLERANCE_MS`
- `VIDEO_TRACK_IOU_THRESHOLD`
- `VIDEO_ENABLE_AUDIO_SIDECAR`
- `VIDEO_EXTRACT_NATIVE_AUDIO`
- `VIDEO_FAIL_ON_AUDIO_ERROR`
- `VIDEO_FFMPEG_BIN`
- `VIDEO_FFPROBE_BIN`
- `VIDEO_SERVER_HOST`
- `VIDEO_SERVER_PORT`

其中最关键的运行参数是：

- `frame_stride`: 每隔多少帧取一帧
- `max_frames`: 最多分析多少帧
- `scene_change_threshold`: 基于采样帧差异进行轻量镜头切分的阈值
- `clip_window_ms`: 短片段多模态审核的时间窗长度
- `clip_moderation_enabled`: 是否调用 8200 的 `/video/action-recognition` 做片段级行为风险审核
- `track_gap_tolerance_ms / track_iou_threshold`: 风险轨合并和框序列插值的时间/空间阈值
- `render_preview`: 是否生成叠框预览
- `enable_audio_sidecar`: 是否尝试处理 sidecar 音轨
- `extract_native_audio`: 是否对视频容器抽取内嵌音轨
- `ffmpeg_bin / ffprobe_bin`: FFmpeg 工具路径

## 9. 运行方式

### 9.1 直接用 Python

```python
from video.pipeline import VideoCompliancePipeline

pipeline = VideoCompliancePipeline()
job = pipeline.execute(
    input_path="./sample_video.mp4",
    tenant_id="demo",
    options={"route_hint": "mixed"},
)

print(job.status.value)
print(job.policy_result.decision.value)
print(job.asset.compliant_video_uri)
```

### 9.2 启动 HTTP 服务

```bash
uvicorn video.server:app --host 0.0.0.0 --port 19003
```

提交任务示例：

```bash
curl -X POST http://localhost:19003/api/v1/check \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "./sample_video.mp4",
    "tenant_id": "demo",
    "profile": "default_cn_enterprise",
    "options": {
      "route_hint": "mixed"
    }
  }'
```

## 10. 产物说明

单次运行的主目录为：

```text
{VIDEO_WORK_DIR}/{run_id}/
```

其中主要产物有：

- `input/`: 工作副本
- `frames/`: 采样帧
- `frame_manifest.jsonl`: 帧级结果摘要
- `scene_manifest.jsonl`: 轻量镜头切分结果
- `clip_windows.jsonl`: 短片段审核窗口
- `clip_moderation_audit.jsonl`: 8200 片段级行为审核请求结果
- `segment_manifest.jsonl`: 视频片段划分
- `video_findings.jsonl`: 时序聚合后的发现
- `risk_tracks.json`: 带 bbox 序列、插值框、mask 关键帧的视频风险轨
- `review_queue.jsonl`: 面向人工复核的高风险、低置信、缺空间定位任务
- `quality_report.json`: 采样覆盖、缓存命中、片段审核和风险分布质量报告
- `rendered/compliant_frames/`: 合规帧
- `rendered/preview_frames/`: 叠框预览帧
- `compliant_video.gif` 或 `compliant_video.mp4`
- `preview.gif` 或 `preview.mp4`
- `report.json`: 视频级审计报告
- `audio_work/`: 音频合规产物
- `native_audio/`: 从正式视频容器抽取出来的音轨（如启用）

## 11. 测试与验证

当前模块带了 6 条真实回归测试：

```bash
python -m pytest video/tests/test_pipeline.py -q -p no:cacheprovider
```

覆盖场景：

1. 普通 GIF 输入，结果应为 `pass_redacted`
2. 文件名带 `explicit` 的 GIF，结果应为 `drop`
3. 帧目录 + sidecar 音轨，验证音轨接入路径可运行
4. mock `ffprobe/ffmpeg` 的 MP4 抽帧路径
5. mock `ffmpeg` 的 MP4 回写渲染路径
6. mock `ffmpeg` 的 MP4 端到端流水线路径

当前环境里我已经跑通：

- `python -m compileall video`
- `python -m pytest video/tests/test_pipeline.py -q -p no:cacheprovider`

测试结果：`6 passed`

## 12. 当前边界与后续扩展

这版引擎已经支持正式视频容器，但仍有清晰边界。

### 当前已完成

- GIF / WebP / APNG / 帧目录输入处理
- MP4 / MOV / MKV / AVI / M4V / WEBM 的 FFmpeg 抽帧支持
- 逐帧调用 `picture`
- 视频级时间聚合
- GIF 与 MP4 两条回写渲染路径
- sidecar 音轨检测与 native 音轨抽取
- 音频流水线结果并入视频决策
- FastAPI 服务
- pytest 回归

### 当前未完成

- 更复杂的目标跟踪器（当前是类别 + IoU + 时间间隔的轻量聚合）
- 复杂镜头切分与自适应采样
- 原始帧级高帧率无损回写
- GPU 优化和大批量调度

### 后续最自然的演进方向

1. 接入真正的 scene detection，按镜头动态采样
2. 为 `VideoFinding` 引入真实 track id 和目标跟踪器
3. 对 redacted audio 与视频进行更精细的 mux 策略控制
4. 支持批量任务、异步队列和 webhook
5. 对接更强的视觉/多模态模型作为 `picture` 的上层增强

## 13. 总结

这个 `video` 模块的核心不是“又做了一套检测器”，而是把现有仓库里的能力组织成一条可执行的视频合规流水线。

一句话概括就是：

- `picture` 负责看懂每一帧
- `audio` 负责看懂每一段声音
- `video` 负责把它们组织到同一条时间轴上，并输出最终可交付的合规结果

现在这套实现已经不再局限于 GIF，而是可以在安装 FFmpeg 的环境里正式处理 `mp4 + ffmpeg` 视频容器路径。
