# Text API Compliance Pipeline

> Note:
> This document records the API-compatible operator layer and its history.
> For the preferred local-model-first startup path, use:
> [`text/LOCAL_MODEL_PIPELINE.md`](./LOCAL_MODEL_PIPELINE.md)

## 目的

本模块新增一条旁路式全量 API 文本合规检测链路，用于本地模型服务器不稳定时进行平台联调和算子验证。

原本地模型链路 `text/pipeline.py` 不变；新增链路 `text/api_pipeline.py` 只替换三类算子：

1. 内容安全检测
2. PII / 个人信息检测
3. 困难情形重审

后续的 span 冲突处理、证据聚合、策略决策、交付审计、下游文本标注导出全部复用原有代码。

## 新增文件

```text
text/api_clients.py
text/api_pipeline.py
text/api_server.py
text/api_steps/__init__.py
text/api_steps/api_safety_moderation.py
text/api_steps/api_privacy_detection.py
text/api_steps/api_hard_case_adjudication.py
text/prompts/content_safety_api_v1.md
text/prompts/privacy_detection_api_v1.md
text/prompts/hard_case_adjudication_api_v1.md
text/prompt_loader.py
text/tests/test_api_pipeline.py
```

## API 配置填写位置

所有 API 都使用同一个 OpenAI-compatible `/v1/chat/completions` 接口。

推荐在项目根目录 `.env` 中填写：

```bash
COMPLIANCE_API_COMPLIANCE_BASE_URL=http://your-api-host:port/v1
COMPLIANCE_API_COMPLIANCE_API_KEY=your-api-key
COMPLIANCE_API_COMPLIANCE_MODEL=your-model-name
```

三个 API 算子的 prompt 已经文件化，默认路径为：

```text
text/prompts/content_safety_api_v1.md
text/prompts/privacy_detection_api_v1.md
text/prompts/hard_case_adjudication_api_v1.md
```

如需替换 prompt，可以在 `.env` 中覆盖：

```bash
COMPLIANCE_API_CONTENT_SAFETY_PROMPT_PATH=/path/to/content_safety_prompt.md
COMPLIANCE_API_PRIVACY_DETECTION_PROMPT_PATH=/path/to/privacy_prompt.md
COMPLIANCE_API_HARD_CASE_PROMPT_PATH=/path/to/hard_case_prompt.md
```

也可以在请求 `config_overrides` 中临时传入：

```json
{
  "package_paths": ["/data/kw/input/cleaned_package"],
  "config_overrides": {
    "work_dir": "/data/kw/compliance-checker/compliance_output/text_api",
    "api_compliance_base_url": "http://your-api-host:port/v1",
    "api_compliance_api_key": "your-api-key",
    "api_compliance_model": "your-model-name",
    "api_content_safety_prompt_path": "/path/to/content_safety_prompt.md",
    "api_privacy_detection_prompt_path": "/path/to/privacy_prompt.md",
    "api_hard_case_prompt_path": "/path/to/hard_case_prompt.md"
  }
}
```

如果 `api_compliance_base_url` 填写为 `http://host:port/v1`，代码会自动请求：

```text
http://host:port/v1/chat/completions
```

## 运行 API 专用服务

API 备用链路独立于原本地模型服务，建议使用单独端口，例如 19002：

Windows 一键启动脚本：

```powershell
.\scripts\start_text_api.ps1
```

等价手动启动命令：

```bash
python -m uvicorn text.api_server:app --host 0.0.0.0 --port 19002
```

健康检查：

```bash
curl http://127.0.0.1:19002/api/v1/health
```

## 一键测试

先保持 `scripts/start_text_api.ps1` 启动窗口运行，然后另开一个 PowerShell 窗口执行：

```powershell
.\scripts\test_text_api.ps1
```

测试脚本会自动创建清洗包：

```text
temp/text_api_smoke_pkg
```

并把输出写入：

```text
temp/text_api_smoke_output/<task_id>
```

如需更严格地要求 API 必须识别出至少一个内容安全风险和一个 PII 风险：

```powershell
.\scripts\test_text_api.ps1 -StrictFindings
```

提交任务：

```bash
curl -X POST http://127.0.0.1:19002/api/v1/check \
  -H "Content-Type: application/json" \
  -d '{
    "package_paths": ["/data/kw/input/cleaned_package"],
    "config_overrides": {
      "work_dir": "/data/kw/compliance-checker/compliance_output/text_api"
    }
  }'
```

查询结果：

```bash
curl http://127.0.0.1:19002/api/v1/status/<task_id>
curl http://127.0.0.1:19002/api/v1/result/<task_id>
```

## 输出结果

API pipeline 仍然生成与原本地模型 pipeline 一致的 JSONL 产物：

```text
01_intake.jsonl
02_content_safety.jsonl
03_privacy_detection.jsonl
03b_span_conflict_resolution.jsonl
04_hard_case_adjudication.jsonl
05_evidence_events.jsonl
06_policy_decisions.jsonl
07_annotation_package.jsonl
08_audit_package.jsonl
09_run_summary.jsonl
10_downstream_annotation_requests.jsonl
11_downstream_annotation_text_id_map.jsonl
12_downstream_annotation_manifest.jsonl
```

因此前端、平台和下游标注工具不需要关心底层是本地模型链路还是 API 链路。

## API 返回格式要求

三个 API 算子都通过同一个 `/v1/chat/completions` 调用。代码会在 prompt 中要求模型返回 JSON。

内容安全检测需要返回：

```json
{
  "status": "clear|flagged|hard_case",
  "risk_score": 0.0,
  "summary": "short summary",
  "needs_adjudication": false,
  "hard_case_reasons": [],
  "findings": []
}
```

PII 检测需要返回：

```json
{
  "pii_count": 0,
  "risk_score": 0.0,
  "summary": "short summary",
  "needs_adjudication": false,
  "hard_case_reasons": [],
  "findings": [
    {
      "risk_type": "phone_number",
      "policy_tag": "pii.phone",
      "severity": "high",
      "confidence": 0.95,
      "explanation": "Detected phone number.",
      "redaction_suggestion": "<PHONE>",
      "span": {"start": 0, "end": 11, "text": "13800138000"}
    }
  ]
}
```

困难情形重审需要返回：

```json
{
  "content_status": "clear|unsafe|borderline",
  "privacy_status": "clear|contains_pii|borderline",
  "confidence": 0.86,
  "rationale": "short explanation",
  "recommended_disposition": "P0|P1|P2|P3|P4|P5",
  "requires_manual_review": true,
  "final_findings": []
}
```

## 与原本地模型链路的关系

```text
text/pipeline.py
  正式本地模型链路，继续使用 Qwen3Guard、Presidio、Qwen hard-case。

text/api_pipeline.py
  备用 API 链路，使用同一个 OpenAI-compatible API 完成内容安全、PII 和困难重审。
```

二者输出保持一致，但入口服务建议分开：

```text
9000: 原本地模型 text.server
19002: 新增 API 备用 text.api_server
```
