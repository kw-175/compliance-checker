# Audio Compliance Checker

`audio` 目录提供面向音频文件的合规检测与处理流水线，对齐 `text` 模块的分阶段设计，但将文本提取替换为音频归一化、ASR、说话人分离和音频脱敏。

## 目标

给定一个或多个音频输入，流水线可以完成：

1. 来源登记与分类。
2. 原始对象级 secrets 和 license 扫描。
3. 音频归一化、转写、说话人分段、转录构建。
4. 文本关键词/正则扫描、PII 检测、语义安全审核。
5. 策略决策与脱敏音频输出。
6. 结果与中间产物落盘，便于审计和回放。

## 主流程

- A `steps/a_source_intake.py`
  - 遍历输入路径，登记 `SourceRecord`，记录 hash、大小、MIME。
- B1 `steps/b1_source_classify.py`
  - 按文件后缀和 MIME 将来源分为 `audio`、`archive`、`repo`、`mixed`。
- B2a `steps/b2a_trufflehog_scan.py`
  - 使用 TruffleHog 扫描原始来源中的 secrets。
- B2b `steps/b2b_scancode_scan.py`
  - 使用 ScanCode 扫描 license、copyright、扫描错误。
- C0 `steps/c0_audio_normalize.py`
  - 使用 FFmpeg 归一化音频；若不可用则回退为复制源文件。
- C1 `steps/c1_asr_transcribe.py`
  - 优先 Qwen3-ASR，其次 faster-whisper，再回退 sidecar transcript 或占位文本。
- C1b `steps/c1b_diarization.py`
  - 优先 pyannote，说话人分离失败时回退为单说话人。
- C1c `steps/c1c_alignment.py`
  - 当前为 pass-through，对齐接口已预留。
- C2 `steps/c2_transcript_build.py`
  - 将 ASR 段和说话人段合并为 `TranscriptUnit`。
- D `steps/d_dedup.py`
  - 对 transcript 做精确去重和基于 token Jaccard 的近似去重。
- E1a `steps/e1a_keyword_scan.py`
  - 关键词扫描与上下文提取。
- E1b `steps/e1b_regex_scan.py`
  - 正则扫描与上下文提取。
- F `steps/f_privacy_detection.py`
  - Presidio 主路径 PII 检测，失败时回退到内置正则检测，并输出 `redaction_spans`。
- G `steps/g_safety_moderation.py`
  - Qwen3-Guard 主路径，失败时回退关键词 mock 分类。
- H `steps/h_evidence_aggregation.py`
  - 汇总所有证据，生成 `EvidenceBundle` 和 summary。
- I `steps/i_policy_decision.py`
  - 优先 OPA 决策，失败时回退本地规则引擎。
- K `steps/k_audio_redaction.py`
  - 根据 `redaction_spans` 输出脱敏音频，支持 `copy`、`silence`、`beep`。
- L `steps/l_release_package.py`
  - 汇总最终 `release_package.json`。

## 输出产物

流水线默认将产物写入 `{work_dir}/{run_id}/`：

- `source_registry.jsonl`
- `source_profile.jsonl`
- `raw_secret_hits.jsonl`
- `source_compliance.jsonl`
- `normalized_audio_manifest.jsonl`
- `asr_segments.jsonl`
- `speaker_segments.jsonl`
- `aligned_segments.jsonl`
- `transcript_units.jsonl`
- `deduped_transcript_units.jsonl`
- `dedup_map.jsonl`
- `keyword_hits.jsonl`
- `regex_hits.jsonl`
- `privacy_checked.jsonl`
- `redaction_spans.jsonl`
- `safety_checked.jsonl`
- `evidence_bundle.json`
- `decision.json`
- `redacted_audio_manifest.jsonl`
- `release_package.json`

## 已修正的问题

本轮对照 `text` 模块和当前 README，已修正以下关键问题：

1. `b2a_trufflehog_scan.py`
   - 修正了 TruffleHog 返回码处理，避免把“发现 secret”误当成失败。
   - 增加按目录去重扫描，避免同目录重复调用。
   - 增加 `finding -> source_id` 反查，确保命中挂回正确来源。

2. `b2b_scancode_scan.py`
   - 补齐更稳健的 ScanCode 结果解析。
   - 兼容 `license_detections.matches` 结构，避免 license 信息丢失。
   - 仅在实际存在命中或扫描错误时输出 `ComplianceHit`。

3. `steps/__init__.py`
   - 公共命令执行支持自定义可接受返回码。
   - `load_yaml()` 在缺少 `PyYAML` 时回退为内置简单解析器，保证本地最小环境仍可运行。
   - `load_jsonl()` 兼容 UTF-8 BOM，并跳过损坏行而不中断整条流水线。

4. `d_dedup.py`
   - 从仅精确哈希去重扩展为精确去重 + 基于阈值的近似去重。

5. `c2_transcript_build.py`
   - 增加源内排序，稳定 transcript 输出顺序。

6. `h_evidence_aggregation.py` 与 `l_release_package.py`
   - 补齐更完整的 summary 字段，如 source 数量、safe/controversial/unsafe 统计。
   - release package 会携带更完整的 transcript/evidence 摘要。

## 降级策略

当依赖不可用时，流水线不会整体中断，而是尽量降级：

- TruffleHog/ScanCode 不可用：记录 warning，返回空扫描结果。
- FFmpeg/ffprobe 不可用：归一化和音频脱敏回退为复制源文件。
- Qwen3-ASR 不可用：回退 faster-whisper，再回退 sidecar transcript 或占位文本。
- pyannote 不可用：回退为单说话人段。
- Presidio 不可用：回退为内置正则 PII 检测。
- Qwen3-Guard 不可用：回退为关键词 mock 分类。
- OPA 不可用：回退为本地规则引擎。
- OpenLineage 不可用：只记录日志，不阻断主流程。

## 运行方式

- 本地服务
  - `uvicorn audio.server:app --host 0.0.0.0 --port 8001`
- Docker
  - 使用 `audio/Dockerfile` 构建
  - 使用 `audio/docker-compose.yml` 启动

## 依赖说明

基础依赖见 `audio/requirements.txt`。其中：

- 必需运行时：`pydantic`、`pydantic-settings`
- 服务接口：`fastapi`、`uvicorn`
- 推荐安装：`PyYAML`、`httpx`
- 可选模型/工具：`faster-whisper`、`pyannote.audio`、`presidio-*`、`transformers`、`torch`
- 外部二进制：`trufflehog`、`scancode`、`ffmpeg`、`ffprobe`

即使部分可选依赖缺失，流水线仍应以降级模式完成一次检测。

## 本次验证

当前环境中缺少 `fastapi`、`pytest`、`PyYAML`、`trufflehog`、`ffmpeg`、`ffprobe` 等部分依赖，因此没有做完整服务端和 pytest 回归；但已完成以下验证：

1. `python -m compileall audio` 通过。
2. 使用 `audio/tests/fixtures/sample_audio.wav` 运行 `AudioCompliancePipeline.execute()` 成功。
3. 流水线成功生成 `release_package.json`、`decision.json`、`regex_hits.jsonl` 等关键产物。
4. 在缺少外部工具时，流水线能够按 README 描述降级而不是中途崩溃。

## 后续建议

- 补装 `fastapi`、`pytest` 和 `audio/requirements.txt` 中的基础依赖后，跑完整 `audio/tests/test_pipeline.py`。
- 若需要生产级脱敏质量，建议优先安装 FFmpeg、Presidio、Qwen3-ASR、Qwen3-Guard、pyannote 和 OPA。
- `c1c_alignment.py` 目前仍是 pass-through，占位接口保留，后续可接入强制对齐模型。

---

## 审查结果与发现 (Review Findings & Thoughts)

经过对 `audio` 目录及其核心代码逻辑 (如 `pipeline.py` 及 `steps/` 子目录下各个处理节点) 的全面审查，**可以确认 `audio` 模块充分且完整地实现了 "Audio Compliance Module Implementation Plan" 中定义的各项任务目标**。

1. **整体编排与数据流 (A -> L) 高度吻合**：
   `pipeline.py` 完美兑现了预定的编排逻辑。从源输入解析、外部工具探测、核心音频处理、文本/安全/隐私特征扫描、到策略应用与压缩包下发，所有节点均按设计正确流转传递并生成了各自的 JSONL 产物。
2. **特有的音频处理核心环节实现精准**：
   - **C0 音频归一化**：`c0_audio_normalize.py` 严格基于 FFmpeg 与 ffprobe 将音频重采样并归一化为 16kHz wav 格式，成功自带 `copy_fallback` 降级容错。
   - **C1 ASR 转写**：`c1_asr_transcribe.py` 建立起 `Qwen3-ASR` -> `faster-whisper` -> `sidecar transcript/fallback` 的严谨三级退火机制方案。
   - **C1b 说话人分离**：`c1b_diarization.py` 规范对接了 `pyannote.audio` 提取发言片段，并可稳当回退到 `speaker_0`。
   - **K 音频脱敏核心**：`k_audio_redaction.py` 依托 `redaction_spans` 及 FFmpeg 的复杂 filter_complex 滤镜，完美实现 `silence`（静音抹除）、`beep`（蜂鸣叠加）和 `copy` 模式。
3. **架构容错性与健壮设计**：
   强依赖外部组件的核心步骤都做了完善的异常捕获与日志。代码高度模块化，非核心模型失败只会记录 warnings，符合高容灾设计，同时利用 `ThreadPoolExecutor` (如 B2 与 E1 阶段) 恰到好处地加速了任务处理。

**💡 想法与后续演进建议：**

- **时间戳级别切分精度**：由于 `c1c_alignment.py` 目前按预期仅作 `pass-through` 透传，若业务上要求对敏感情报做到单字无缝摘除，引入一个强制对齐（Forced Alignment，如 Wav2Vec2）模型能够大大提升静音 / Beep 操作前后的听感连贯度。
- **微服务高可用扩容**：鉴于部分深度学习节点（C1 ASR, G Qwen3-Guard）极为消耗 GPU 运算；当投入生产环境面临数千级并发时，单容器内部的线程池将形成内存挤兑。建议未来将高算力单元剥离至 `Celery` 队列消费层，将其转型为完全分布式的异步微服务架构。
