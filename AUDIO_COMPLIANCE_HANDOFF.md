# 音频合规检测后端修正交接

本文档总结本轮对音频合规检测系统的设计修正，供后续修改平台前后端时作为上下文。

## 目标定位

当前音频合规检测只覆盖“可转写为文本的语音内容”，暂不做非语音音频内容检测。

核心链路：

```text
音频输入
  -> FFmpeg 标准化
  -> Qwen3-ASR 转写
  -> 转写文本送文本合规检测 API
  -> 文本风险结果回映射到音频时间轴
  -> 生成音频合规报告 / 风险记录 / 脱敏计划
```

## 核心架构

- ASR 模型：`Qwen3-ASR-0.6B`
- 默认 ASR 后端：vLLM
- ASR vLLM 适配器：`ops/qwen3asr_vllm_adapter.py`
- 一键启动脚本：`scripts/start_audio_local_stack.sh`
- 真实音频测试脚本：`scripts/test_audio_real_wav_stack.sh`

## 启动设计

`scripts/start_audio_local_stack.sh` 当前默认配置：

- `ASR_BACKEND=vllm`
- ASR 服务端口：`19011`
- Audio API 端口：`19001`
- Text API 端口：`19002`
- Qwen3-ASR 虚拟环境：`/data/kw/compliance-checker/qwen-serving/asr-vllm/.venv`
- Qwen3-ASR 模型路径：`/data/kw/compliance-checker/models/Qwen/Qwen3-ASR-0.6B`
- FFmpeg/FFprobe 路径：`/data/kw/compliance-checker/models/ffmpeg-7.0.2-amd64-static/`

默认单卡参数：

| 模型 | GPU | gpu_memory_utilization |
| --- | --- | --- |
| Qwen3.5 | 0 | `0.68` |
| Qwen3Guard | 0 | `0.06` |
| Qwen3-ASR vLLM | 0 | `0.12` |

Qwen3-ASR 额外默认参数：

- `QWEN3ASR_MAX_INFERENCE_BATCH_SIZE=1`
- `QWEN3ASR_MAX_NEW_TOKENS=2048`
- `QWEN3ASR_MAX_MODEL_LEN=4096`
- `QWEN3ASR_TENSOR_PARALLEL_SIZE=1`

注意：单卡显存 OOM 风险仍然存在，暂时没有彻底解决。

## 关键可靠性修正

### 1. ASR 失败不再伪装成合规文本

此前 Qwen ASR 失败后可能生成占位 transcript：

```text
Audio transcript unavailable
```

然后继续送入文本合规检测，可能误判为合规。

现在默认禁止这种行为：

- `COMPLIANCE_ASR_REQUIRED=true`
- `COMPLIANCE_ASR_UNAVAILABLE_FALLBACK_ENABLED=false`

如果 Qwen ASR 和其他可用 ASR 都失败，音频任务会失败，不再继续文本合规检测。

对应代码：

- `audio/config/settings.py`
- `audio/steps/c1_asr_transcribe.py`
- `scripts/start_audio_local_stack.sh`

### 2. Qwen3-ASR vLLM 增加真实 ready 检查

ASR 适配器现在区分：

- `GET /health`：进程健康，不代表模型已加载。
- `GET /ready`：实际加载 Qwen3-ASR vLLM 模型，加载失败返回 `503`。

启动脚本行为：

- `ASR_BACKEND=vllm` 时等待 `/ready`
- `ASR_BACKEND=transformers` 时等待 `/health`

对应代码：

- `ops/qwen3asr_vllm_adapter.py`
- `scripts/start_audio_local_stack.sh`

### 3. vLLM 路线拒绝 CPU 配置

`Qwen3ASRModel.LLM(...)` 内部使用 vLLM，当前按 CUDA 路线运行。

如果：

```text
ASR_BACKEND=vllm
QWEN_ASR_DEVICE=cpu
```

系统会提前失败，并提示 CPU ASR 应改用 transformers 路线。

### 4. 文本 API 轮询增加总超时

新增配置：

```text
COMPLIANCE_TEXT_API_TASK_TIMEOUT_SECONDS=1800
```

避免文本合规服务卡住后，音频任务无限保持 `running`。

对应代码：

- `audio/config/settings.py`
- `audio/text_api_bridge.py`
- `scripts/start_audio_local_stack.sh`

## 时间轴与脱敏设计

当前本地没有 Qwen3-ASR forced aligner 模型，因此默认 ASR 结果是“整段音频一个 transcript segment”。

影响：

- 文本风险可以回映射到音频时间轴。
- 但时间戳是按字符 offset 线性投影到整段音频时长的粗粒度估计。
- 这种时间戳可用于风险定位提示，但不适合直接精确消音。

因此新增标记：

```json
{
  "timestamp_granularity": "whole_audio",
  "mapping_precision": "coarse",
  "mapping_note": "ASR returned only whole-audio timestamps; character spans were linearly projected onto audio duration."
}
```

如果将来接入 forced aligner 或 ASR 原生分段时间戳，可变成：

- `timestamp_granularity=forced_alignment`
- `timestamp_granularity=segment`
- `mapping_precision=segment`

## 音频脱敏策略

默认只生成脱敏计划：

- `30_audio_redaction_spans.jsonl`

默认不直接渲染修改后的音频。

如果开启：

```text
COMPLIANCE_AUDIO_REDACTION_ENABLED=true
```

但 redaction span 来自：

```text
mapping_precision=coarse
```

系统会跳过实际消音渲染，并写出 `.skipped.json`，避免用不精确时间戳误消音。

对应代码：

- `audio/models/schemas.py`
- `audio/text_api_bridge.py`

## 平台前后端需要关注的接口语义

Audio API：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/v1/health` | 音频服务健康检查 |
| `POST` | `/api/v1/check` | 提交音频检测任务 |
| `GET` | `/api/v1/status/{task_id}` | 查询任务状态 |
| `GET` | `/api/v1/result/{task_id}` | 获取任务结果 |

音频任务提交示例：

```json
{
  "input_paths": ["/path/to/audio.wav"],
  "config_overrides": {
    "operator_id": "CMP_001",
    "dataset_name": "privacy-real-audio",
    "pipeline_profile": "privacy_only",
    "audio_execution_route": "api",
    "qwen_asr_enabled": true,
    "faster_whisper_enabled": false,
    "pyannote_enabled": false
  }
}
```

`operator_id` 映射：

| operator_id | 含义 |
| --- | --- |
| `CMP_001` | 隐私检测 |
| `CMP_002` | 内容安全检测 |
| `CMP_008` | 完整检测 |

## 前端展示建议

前端应明确展示：

- 音频检测基于“语音转写文本”。
- ASR 状态：成功 / 失败 / 未转写。
- 如果任务失败原因为 ASR unavailable，不要展示为合规通过。
- 风险记录中的文本风险类型。
- 风险对应的 transcript span。
- 风险对应的音频时间段。
- `mapping_precision`。

当：

```text
mapping_precision=coarse
```

UI 应提示：

```text
音频时间位置为粗略估计，不适合直接精确消音。
```

如果存在 `.skipped.json`，UI 应提示：

```text
由于 ASR 时间戳不精确，系统未自动生成消音音频。
```

## 主要产物

音频运行目录中的关键文件：

| 文件 | 说明 |
| --- | --- |
| `04_asr_segments.jsonl` | ASR 转写片段 |
| `07_transcript_units.jsonl` | 统一 transcript units |
| `07c_audio_text_alignment_index.jsonl` | 文本到音频时间轴索引 |
| `20_text_api_input.jsonl` | 送入文本合规 API 的文本 |
| `23_text_api_result.json` | 文本 API 返回结果 |
| `24_audio_text_risk_records.jsonl` | 音频侧主风险记录 |
| `29_audio_run_summary.json` | 音频检测摘要 |
| `30_audio_redaction_spans.jsonl` | 音频脱敏时间段计划 |
| `31_redacted_audio_manifest.jsonl` | 实际渲染出的脱敏音频 manifest |
| `32_audio_compliance_report.json` | 最终音频合规报告 |

## 已验证内容

未做真实 GPU 运行测试。

已完成非 GPU 验证：

```bash
bash -n scripts/start_audio_local_stack.sh scripts/test_audio_real_wav_stack.sh
python -m py_compile audio/config/settings.py audio/steps/c1_asr_transcribe.py audio/models/schemas.py audio/text_api_bridge.py ops/qwen3asr_vllm_adapter.py
timeout 60s python -m pytest audio/tests/test_text_api_bridge.py audio/tests/test_pipeline.py::test_asr_fallback audio/tests/test_pipeline.py::test_asr_required_rejects_placeholder_transcript audio/tests/test_pipeline.py::test_qwen_asr_uses_endpoint_without_local_model -q
```

定向测试结果：

```text
6 passed
```

## 后续平台改造重点

1. 平台后端接入 Audio API 的异步任务模型。
2. 前端将音频检测结果展示成“转写文本合规 + 音频时间轴映射”。
3. 明确区分检测失败、ASR 失败、检测通过、检测有风险。
4. 对 `mapping_precision=coarse` 做特殊提示。
5. 不要把 `Audio transcript unavailable` 或 ASR 失败结果当成合规通过。
6. 音频消音/脱敏功能先作为“计划展示”，不要默认承诺可精确生成。
