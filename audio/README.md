# Audio Compliance Checker

`audio` 目录现在作为文本合规系统的音频适配层。当前默认主链路很简单：

```text
audio input -> ffmpeg normalize -> Qwen3-ASR transcript -> text.api_server compliance -> audio timeline report
```

最终隐私合规、内容安全、片段裁决、文档裁决和 policy decision 都由 `text` 模块完成。音频侧只负责把音频稳定转写成文本、把文本合规结果映射回音频时间轴，并在需要时生成可选的音频脱敏产物。

## 本地资源

默认使用仓库内模型和工具：

- Qwen3-ASR: `/data/kw/compliance-checker/models/Qwen/Qwen3-ASR-0.6B`
- FFmpeg: `/data/kw/compliance-checker/models/ffmpeg-7.0.2-amd64-static/ffmpeg`
- FFprobe: `/data/kw/compliance-checker/models/ffmpeg-7.0.2-amd64-static/ffprobe`
- 可选说话人分离模型: `/data/kw/compliance-checker/models/pyannote/speaker-diarization-3.1`

## 默认服务链路

音频服务：

```bash
uvicorn audio.server:app --host 0.0.0.0 --port 19001
```

文本合规服务需要先启动，默认地址：

```text
http://127.0.0.1:19002
```

文本本地栈启动方式见仓库根目录的 `TEXT_COMPLIANCE_HANDOFF.md`，通常是：

```bash
bash scripts/start_text_local_stack.sh
```

提交音频检测：

```json
{
  "input_paths": ["/path/to/audio.wav"],
  "config_overrides": {
    "operator_id": "CMP_008"
  }
}
```

operator 映射与文本系统一致：

- `CMP_001` -> `privacy_only`
- `CMP_002` -> `safety_only`
- `CMP_008` -> `full`

也可以直接覆盖：

```json
{
  "config_overrides": {
    "pipeline_profile": "privacy_only"
  }
}
```

## 核心产物

音频侧默认输出到 `{work_dir}/{task_id}/`：

- `01_source_registry.jsonl`: 输入源登记。
- `02_source_profile.jsonl`: 音频源分类。
- `03_normalized_audio_manifest.jsonl`: ffmpeg 归一化后的音频清单。
- `04_asr_segments.jsonl`: Qwen3-ASR 转写片段。
- `05_speaker_segments.jsonl`: 说话人片段，默认单说话人 fallback。
- `06_aligned_segments.jsonl`: ASR 对齐片段。
- `07_transcript_units.jsonl`: 统一转写单元。
- `07b_asr_transcript.json`: 按音频源聚合的转写文档和 segment map。
- `07c_audio_text_alignment_index.jsonl`: 音频时间轴与文本字符 offset 的双向索引。
- `20_text_api_input.jsonl`: 提交给文本合规服务的 JSONL。
- `20b_text_api_source_map.json`: 文本 doc 与音频 segment 的映射。
- `23_text_api_result.json`: 文本合规 API 返回。
- `24_audio_text_risk_records.jsonl`: 文本合规风险、Qwen3.5 裁决和音频时间段合并后的主风险视图。
- `25_audio_document_assessments.jsonl`: 音频级文档判断包装，结论来自文本最终治理结果。
- `26_audio_policy_decisions.jsonl`: 音频级 policy decision 包装，保留训练/标注流转建议。
- `27_audio_annotation_package.jsonl`: 面向音频标注/播放器的风险片段包。
- `28_audio_audit_package.jsonl`: 音频审计包，包含文本审计记录与时间轴证据。
- `29_audio_run_summary.json`: 音频 run 级汇总，最终结论跟随文本 policy/summary。
- `30_audio_redaction_spans.jsonl`: 文本 redaction targets 映射出的音频时间区间。
- `31_redacted_audio_manifest.jsonl`: 可选脱敏音频清单。
- `32_audio_compliance_report.json`: 面向调用方的音频合规报告。

文本侧的 `02/03/05/06/07/08/09/10/11/12` 产物仍由 `text.api_server` 生成，并会通过报告里的 `artifact_paths.text_api` 暴露。

## 音频脱敏策略

默认只生成 `30_audio_redaction_spans.jsonl`，不直接改音频：

```text
audio_redaction_enabled = false
```

如果开启：

```json
{
  "config_overrides": {
    "audio_redaction_enabled": true,
    "redaction_strategy": "silence"
  }
}
```

音频侧会根据文本系统 `03b_span_conflict_resolution.jsonl` 中的隐私 redaction targets，把字符 span 映射回 ASR 时间轴，前后加 `audio_redaction_padding_ms`，再调用 FFmpeg 生成静音或 beep 版本。

建议口径：

- 隐私 PII：允许自动静音或 beep。
- 内容安全高风险：默认 `hold/reject`，不建议用局部消音伪装成合规音频。
- ASR 时间戳缺失：只输出 redaction plan，不渲染音频。

## 兼容旧链路

`audio.pipeline.AudioCompliancePipeline` 和部分 `audio.legacy_steps` 仍保留，便于回归和兼容旧测试。但生产默认入口已经是：

```text
Qwen3-ASR + text API bridge
```

如需强制旧链路，可在请求中覆盖：

```json
{
  "config_overrides": {
    "audio_execution_route": "local"
  }
}
```

## 验证

推荐的轻量验证：

```bash
python -m py_compile audio/config/settings.py audio/text_api_bridge.py audio/server.py
pytest -q audio/tests/test_text_api_bridge.py audio/tests/test_asr_sidecar_json.py
```
