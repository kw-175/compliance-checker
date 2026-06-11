# 文本合规检测当前实现交接文档

生成时间：2026-05-08

本文用于在新的上下文窗口中无缝衔接当前已经完成的文本合规检测工作。当前文本合规检测已经基本达到两个最终形态设计文档的核心目标：

- `文本内容安全最终形态设计.md`
- `文本隐私合规检测最终形态设计.md`

## 1. 当前总体结论

当前文本合规检测已经实现本地模型闭环运行，不再依赖外部 API。

核心能力：

- 隐私合规链：PII 召回 -> Qwen3.5 片段级上下文裁决 -> Qwen3.5 文档级综合判断 -> policy decision -> audit / annotation / downstream 输出。
- 内容安全链：文档上下文 -> Qwen3Guard + 规则召回候选窗口 -> Qwen3.5 精确风险片段定位 -> Qwen3.5 片段级上下文裁决 -> Qwen3.5 文档级综合判断 -> policy decision -> audit / annotation / downstream 输出。
- `full`、`privacy_only`、`safety_only` 三种 profile 现在都走统一最终治理出口，都会生成 `05/06/07/08/09/10/11/12` 产物。
- `full` 模式中，内容安全链和隐私合规链已经做了链路隔离：隐私文档不会被内容安全链当成内容风险处理，内容文档中的疑似 PII 会被隐私链上下文复核，而不是直接污染最终内容安全决策。

## 2. 本地服务与端口

一键启动脚本：

```bash
bash scripts/start_text_local_stack.sh
```

当前服务拓扑：

```text
PII Gateway -> Qwen3Guard vLLM -> Qwen3Guard Adapter -> Qwen3.5 vLLM -> text.api_server
```

端口：

- `5002`: PII Gateway，接口 `POST /analyze`
- `8212`: Qwen3Guard vLLM，OpenAI-compatible vLLM backend
- `8215`: Qwen3Guard Adapter，接口 `POST /moderate`
- `8301`: Qwen3.5 vLLM，OpenAI-compatible `/v1/chat/completions`
- `19002`: `text.api_server`，接口 `/api/v1/check`

## 3. GPU 与虚拟环境

当前启动脚本默认使用 0 号 GPU：

```bash
QWEN3GUARD_GPU="${QWEN3GUARD_GPU:-0}"
QWEN35_GPU="${QWEN35_GPU:-0}"
```

显存参数：

```bash
QWEN3GUARD_GPU_MEMORY_UTILIZATION="${QWEN3GUARD_GPU_MEMORY_UTILIZATION:-0.08}"
QWEN35_GPU_MEMORY_UTILIZATION="${QWEN35_GPU_MEMORY_UTILIZATION:-0.78}"
QWEN3GUARD_MAX_MODEL_LEN="${QWEN3GUARD_MAX_MODEL_LEN:-1536}"
QWEN35_MAX_MODEL_LEN="${QWEN35_MAX_MODEL_LEN:-6144}"
```

虚拟环境：

- Qwen3Guard vLLM 使用：`qwen-serving/asr-vllm/.venv`
- Qwen3.5 vLLM 使用：`qwen-serving/text-vllm/.venv`
- Qwen3Guard Adapter 使用：`.venv`
- text API 使用：`.venv`
- PII Gateway 使用：`.venvs/compliance-pii`

启动脚本会在启动前检查：

- venv 激活文件是否存在
- `vllm` 是否存在
- `from transformers import GenerationConfig` 是否成功
- 模型目录是否存在

临时切换 GPU 的方式：

```bash
QWEN35_GPU=1 QWEN3GUARD_GPU=1 bash scripts/start_text_local_stack.sh
```

## 4. 关键脚本

### 4.1 一键启动

文件：

```text
scripts/start_text_local_stack.sh
```

功能：

- 创建 tmux session：默认 `text-local`
- 依次启动 PII、Qwen3.5 vLLM、Qwen3Guard vLLM、Qwen3Guard Adapter、text API
- 启动顺序是串行等待，避免两个 Qwen 模型同时抢显存
- 通过 endpoint readiness 检查后再启动下一项

常用命令：

```bash
bash scripts/start_text_local_stack.sh start
bash scripts/start_text_local_stack.sh restart
bash scripts/start_text_local_stack.sh stop
bash scripts/start_text_local_stack.sh status
bash scripts/start_text_local_stack.sh attach
```

### 4.2 常规冒烟测试

文件：

```text
scripts/test_text_local_stack.sh
```

用途：

- 使用内置合成样例测试 `privacy_only`、`safety_only`、`full`
- 检查服务探活、核心产物、片段裁决、文档裁决和 policy decision

### 4.3 真实 JSON 测试

文件：

```text
scripts/test_text_real_json_stack.sh
```

默认测试数据：

```text
test_data/content_safety_11_targets_single_text.json
test_data/privacy_compliance_11_targets_single_text.json
```

运行：

```bash
bash scripts/test_text_real_json_stack.sh
```

该脚本会分别跑：

- `safety_only`: 只提交内容安全测试 JSON
- `privacy_only`: 只提交隐私合规测试 JSON
- `full`: 两个 JSON 一起提交

它会强制检查：

- 内容安全链 `02a/02aa/02/02g/02h`
- 隐私链 `03/03f/03g/03i`
- 单链和 full 都必须生成 `05/06/07/08/09`
- `09_run_summary.jsonl` 必须和 `06_policy_decisions.jsonl` 对齐
- `privacy_only` 不得消费内容安全 localized fragments
- `safety_only` 不得消费隐私 fragment adjudications
- `full` 模式下隐私文档不得进入内容安全候选窗口、定位片段和内容片段裁决

可调阈值：

```bash
MIN_CONTENT_FRAGMENTS=8
MIN_CONTENT_TARGET_COVERAGE=7
MIN_PRIVACY_FINDINGS=8
MIN_PRIVACY_TARGET_COVERAGE=8
```

## 5. 文本 API 输入方式

提交到：

```text
POST http://127.0.0.1:19002/api/v1/check
```

请求结构：

```json
{
  "package_paths": ["/path/to/package_or_json"],
  "config_overrides": {
    "work_dir": "/data/kw/compliance-checker/temp/output",
    "pipeline_profile": "full"
  }
}
```

`pipeline_profile` 支持：

- `full`
- `privacy_only`
- `safety_only`

也可以通过 `operator_id` 间接映射：

- `CMP_001` -> `privacy_only`
- `CMP_002` -> `safety_only`
- `CMP_008` -> `full`

## 6. 文本合规处理主流程

主入口：

```text
text/api_pipeline.py
```

核心 profile：

```python
PipelineProfile.FULL
PipelineProfile.PRIVACY_ONLY
PipelineProfile.SAFETY_ONLY
```

### 6.1 公共前置

所有 profile 都会执行：

```text
01_intake.jsonl
01b_document_context.jsonl
```

对应代码：

- `text/steps/a_source_intake.py`
- `text/steps/b_document_context.py`

`01b_document_context.jsonl` 由 Qwen3.5 生成文档上下文，用于后续隐私链和内容链判断。

### 6.2 隐私合规链

适用于 `privacy_only` 和 `full`。

流程：

```text
03_privacy_detection.jsonl
03f_privacy_fragment_adjudications.jsonl
03g_privacy_document_assessments.jsonl
03b_span_conflict_resolution.jsonl
03c_privacy_policy_decisions.jsonl
03d_privacy_audit.jsonl
03e_privacy_review_tasks.jsonl
03h_privacy_review_results.jsonl
03i_privacy_final_decisions.jsonl
```

关键代码：

- `text/api_steps/api_privacy_detection.py`
- `text/steps/c_privacy_fragment_adjudication.py`
- `text/steps/d_privacy_document_assessment.py`
- `text/steps/span_conflict_resolution.py`

关键逻辑：

- PII Gateway 做初始 PII 召回。
- 每一个真实召回的隐私 finding 都必须经过 Qwen3.5 上下文片段裁决。
- Qwen3.5 为每个隐私片段输出自然语言解释、`governance_action`、`training_impact`、`annotation_impact`。
- 文档级隐私判断由 Qwen3.5 综合片段裁决、文档上下文和组合识别风险输出。
- 无隐私证据时使用 `privacy_document_scope_guard`，`can_raise_disposition=false`，避免内容安全风险污染隐私链。

### 6.3 内容安全链

适用于 `safety_only` 和 `full`。

流程：

```text
02a_content_candidate_windows.jsonl
02aa_content_fragment_localization.jsonl
02_content_safety.jsonl
02g_content_fragment_adjudications.jsonl
02h_content_document_assessments.jsonl
02b_content_safety_decisions.jsonl
02c_content_safety_audit.jsonl
02d_content_safety_review_tasks.jsonl
02e_content_safety_review_results.jsonl
02f_content_safety_final_decisions.jsonl
```

关键代码：

- `text/steps/b_content_candidate_windows.py`
- `text/steps/b_content_fragment_localization.py`
- `text/steps/c_content_fragment_adjudication.py`
- `text/steps/d_content_document_assessment.py`

关键逻辑：

- Qwen3Guard 只承担粗召回角色，不承担最终合规判断。
- 内容安全候选窗口由 Qwen3Guard + 规则召回。
- Qwen3.5 对候选窗口进行精确 span 定位，生成 localized fragments。
- 每个 localized fragment 进入 Qwen3.5 片段级上下文裁决。
- Qwen3.5 输出自然语言解释、`semantic_role`、`operationality`、`protective_context`、`recommended_action`。
- Qwen3.5 再做文档级整体立场判断，输出 `overall_stance`、`operational_risk`、`training_suitability`、`recommended_action`。
- 无内容安全证据时使用 `content_document_scope_guard`，`can_raise_disposition=false`，避免隐私风险污染内容安全链。

## 7. 统一最终治理出口

这是本轮最后修正的重点。

以前问题：

- `privacy_only` 和 `safety_only` 只走旧的 `_output_from_partial_findings()`。
- 单链 `09_run_summary.jsonl` 只看召回结果，没有消费 Qwen3.5 片段裁决和文档级裁决。
- 结果表现为：`privacy_only` 被低估成 `P1/quarantine`，`safety_only` 被低估成 `P3/review`。

现在修正：

- `full`、`privacy_only`、`safety_only` 都调用：

```text
APICompliancePipeline._finalize_governance_output()
```

统一生成：

```text
05_evidence_events.jsonl
06_policy_decisions.jsonl
07_annotation_package.jsonl
08_audit_package.jsonl
09_run_summary.jsonl
10_downstream_annotation_requests.jsonl
11_downstream_annotation_text_id_map.jsonl
12_downstream_annotation_manifest.jsonl
```

关键代码：

- `text/steps/h_evidence_aggregation.py`
- `text/steps/i_policy_decision.py`
- `text/steps/delivery_audit.py`
- `text/steps/downstream_annotation_export.py`

最终 policy decision 会消费：

- 文档上下文
- 隐私片段裁决
- 隐私文档级判断
- 内容候选窗口
- 内容定位片段
- 内容片段裁决
- 内容文档级判断
- redaction plan

单链模式只传入当前链的产物，另一条链传空列表，避免伪造 inactive-chain 结果。

## 8. Qwen3Guard 在当前系统中的角色

Qwen3Guard 不是最终裁决模型。

当前角色：

- 内容安全候选窗口粗召回。
- 提供安全/不安全标签和粗类别提示。
- 帮助减少 Qwen3.5 的全文扫描成本。

最终合规与否取决于：

- Qwen3.5 对片段的上下文裁决。
- Qwen3.5 对文档整体立场/风险的判断。
- `i_policy_decision.py` 的统一治理决策。

## 9. 当前关键产物语义

内容安全：

- `02a_content_candidate_windows.jsonl`: 候选窗口，粗召回结果。
- `02aa_content_fragment_localization.jsonl`: Qwen3.5 精确定位的内容风险片段。
- `02_content_safety.jsonl`: 内容安全 finding，引用 localized fragment。
- `02g_content_fragment_adjudications.jsonl`: 每个内容风险片段的 Qwen3.5 上下文裁决。
- `02h_content_document_assessments.jsonl`: 内容安全文档级判断。
- `02b/02c/02d/02e/02f`: 内容安全治理、审计、复核任务和最终链路视图。

隐私合规：

- `03_privacy_detection.jsonl`: PII 召回。
- `03f_privacy_fragment_adjudications.jsonl`: 每个隐私 finding 的 Qwen3.5 上下文裁决。
- `03g_privacy_document_assessments.jsonl`: 隐私文档级综合判断。
- `03b_span_conflict_resolution.jsonl`: 脱敏 span 冲突消解和 redaction plan。
- `03i_privacy_final_decisions.jsonl`: 隐私最终治理视图。

统一出口：

- `05_evidence_events.jsonl`: 统一证据事件。
- `06_policy_decisions.jsonl`: 最终 policy decision。
- `07_annotation_package.jsonl`: 给标注/交付侧的包。
- `08_audit_package.jsonl`: 审计主视图，包含 content candidate windows、localized fragments、privacy adjudications 等。
- `09_run_summary.jsonl`: run 级汇总，必须与 `06_policy_decisions.jsonl` 对齐。

## 10. 第二轮真实 JSON 测试结果

最新测试目录：

```text
temp/text_real_json_test_output/20260508_153247
```

测试文件：

```text
test_data/content_safety_11_targets_single_text.json
test_data/privacy_compliance_11_targets_single_text.json
```

结果：

```text
safety_only:
  content_candidate_windows = 3
  content_localized_fragments = 10
  content_findings = 10
  content_fragment_adjudications = 10
  content expected targets = 11/11 covered
  content_document_assessment = exclude_from_training
  policy = P4 / reject / full
  summary = P4 / reject / full

privacy_only:
  privacy_findings = 31
  privacy_fragment_adjudications = 31
  privacy_redaction_targets = 20
  privacy expected targets = 11/11 covered
  privacy_document_assessment = critical / exclude_from_training
  policy = P3 / review / full
  summary = P3 / review / full

full:
  content_candidate_windows = 3
  content_localized_fragments = 11
  content_fragment_adjudications = 11
  privacy_findings = 31
  privacy_fragment_adjudications = 31
  content policy = P4 / reject / full
  privacy policy = P3 / review / full
  summary = P4 / reject / full
```

链路隔离结果：

```text
privacy_doc_content_candidate_windows = 0
privacy_doc_content_localized_fragments = 0
privacy_doc_content_fragment_adjudications = 0
privacy_doc_content_can_raise_disposition = false
privacy_policy_content_localized_fragment_count = 0
content_doc_privacy_finding_count = 5
```

解释：

- 隐私文档没有进入内容安全风险片段链。
- 内容文档中被隐私链召回的 5 个疑似 PII，被 Qwen3.5 判断为合成内容/上下文样例或误报，没有污染内容安全最终决策。
- `trust_level` 全部为 `full`。
- hard-case 本轮没有触发，`04_hard_case_adjudication.jsonl` 为空，这是预期结果。

## 11. 已经解决过的重要问题

### 11.1 vLLM / transformers 环境问题

曾出现：

```text
ImportError: cannot import name 'GenerationConfig' from 'transformers'
ImportError: libcudart.so.13
libcusparse.so.12 undefined symbol __nvJitLinkGetErrorLogSize_12_9
```

处理结论：

- Qwen3Guard 最终使用 `qwen-serving/asr-vllm/.venv`。
- Qwen3.5 最终使用 `qwen-serving/text-vllm/.venv`。
- 启动脚本增加了 `GenerationConfig` 和 `vllm` 可用性检查。

### 11.2 GPU 显存冲突

曾使用 1 号 GPU，后来因 1 号 GPU 被占用切到 0 号 GPU。

当前默认：

```text
Qwen3Guard GPU = 0
Qwen3.5 GPU = 0
```

### 11.3 Qwen3Guard Adapter 空 Bearer 问题

曾出现：

```text
httpx.LocalProtocolError: Illegal header value b'Bearer '
```

已处理：adapter 不再发送非法空 Authorization header。

### 11.4 冒烟测试 JSON 构造问题

曾出现 shell 嵌套 JSON/换行导致：

```text
SyntaxError: unterminated string literal
JSON decode error: Invalid control character
```

已处理：测试脚本改用 Python 生成 JSON payload。

### 11.5 Qwen3.5 上下文长度问题

曾出现：

```text
maximum context length is 4096 tokens
requested 4096 output tokens
```

已处理：

- 降低 prompt 输入体积。
- 增加 `COMPLIANCE_LOCAL_COMPLIANCE_MAX_CHARS=4800`
- 设置 `COMPLIANCE_LOCAL_COMPLIANCE_MAX_TOKENS=1024`
- Qwen3.5 max model len 提升到 `6144`

### 11.6 内容安全链与隐私链污染

曾出现内容安全文档级判断处理 PII 风险的问题。

已处理：

- 内容安全文档判断 scope constraint：只判断 content safety。
- 隐私文档判断 scope constraint：只判断 privacy/re-identification。
- 无对应链证据时使用 scope guard，`can_raise_disposition=false`。

### 11.7 单链 summary 低估风险

曾出现：

```text
privacy_only summary = P1 / quarantine
safety_only summary = P3 / review
```

但 Qwen3.5 文档级判断已经更高风险。

已处理：

- 单链不再使用旧 `_output_from_partial_findings()`。
- `privacy_only`、`safety_only` 和 `full` 都走 `_finalize_governance_output()`。
- 当前结果：

```text
privacy_only = P3 / review
safety_only = P4 / reject
```

## 12. 当前仍需注意的点

1. `text/LOCAL_MODEL_PIPELINE.md` 中曾写过 GPU `1`，如果还未同步文档，需要后续改为 GPU `0` 或说明可通过环境变量覆盖。
2. Qwen3.5 片段定位存在自然波动，例如同一内容安全 JSON 在 `safety_only` 中定位 10 个片段，在 `full` 中定位 11 个片段，但目标覆盖和最终决策一致，当前可接受。
3. Qwen3Guard 召回不是最终裁决，后续如需提升内容安全召回质量，可以替换或补充更专注内容安全 span recall 的模型，但当前系统已通过 Qwen3.5 定位和裁决兜底。
4. `privacy_only` 当前对 `exclude_from_training` 映射为 `P3/review`，不是 `P4/reject`。这是现有 policy decision 的治理口径：隐私高风险通常进入人工复核/隔离路径，内容高风险阻断为 `P4`。如果产品上要求隐私 critical 直接 block，需要修改 `text/steps/i_policy_decision.py` 的映射策略。
5. 单链 profile 中 inactive chain 不生成对应 `02` 或 `03` 原始检测产物，但统一出口 `05/06/07/08/09/10/11/12` 一定生成。

## 13. 如果在新上下文中继续文本问题

优先阅读这些文件：

```text
scripts/start_text_local_stack.sh
scripts/test_text_real_json_stack.sh
text/api_pipeline.py
text/steps/b_content_candidate_windows.py
text/steps/b_content_fragment_localization.py
text/steps/c_content_fragment_adjudication.py
text/steps/d_content_document_assessment.py
text/steps/c_privacy_fragment_adjudication.py
text/steps/d_privacy_document_assessment.py
text/steps/h_evidence_aggregation.py
text/steps/i_policy_decision.py
text/steps/delivery_audit.py
```

快速验证命令：

```bash
python -m py_compile text/api_pipeline.py
bash -n scripts/start_text_local_stack.sh
bash -n scripts/test_text_real_json_stack.sh
pytest -q text/tests/test_api_pipeline.py
```

服务启动后做真实 JSON 验证：

```bash
bash scripts/start_text_local_stack.sh restart
bash scripts/test_text_real_json_stack.sh
```

