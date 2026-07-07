# 多模态数据合规检测引擎

本仓库是一个面向文本、音频、图片、视频的多模态合规检测工程。当前代码按模态拆分为独立服务，同时通过 `common/` 中的统一合同、证据、策略和交付结构沉淀审计产物。

仓库只保存代码、配置、启动脚本、轻量测试数据和模型目录占位文件。虚拟环境、模型权重、运行输出、缓存和大体积媒体文件不应提交到 Git。

## 目录结构

```text
.
├── common/                 # 多模态共享合同、枚举、证据、策略和运行时上下文
├── text/                   # 文本合规流水线、文本 API 服务和文本规则/提示词
├── audio/                  # 音频合规流水线、音频 API 服务、ASR/文本桥接逻辑
├── picture/                # 图片合规编排、领域模型、API、视觉/ OCR /脱敏提供方
├── video/                  # 视频合规编排、帧/片段/音频侧路处理和视频 API 服务
├── ops/                    # 本地模型服务适配器，例如 Qwen、PaddleOCR-VL、SAM3、PII 网关
├── scripts/                # 本地服务栈启动脚本
├── requirements/           # 按运行环境拆分的依赖清单
├── models/                 # 模型目录占位；只跟踪 .gitkeep，不跟踪真实权重
├── test_data/              # 轻量测试数据
├── pytest.ini              # 默认跳过 integration 和 slow 测试
└── README.md
```

## 统一合同

`common/contracts.py` 定义了跨模态统一输出 `ComplianceOutput`，核心字段包括：

- `pipeline_run_id`：本次流水线运行标识。
- `modality`：`text`、`audio`、`picture` 或 `video`。
- `decision`：统一决策，取值来自 `allow`、`review`、`quarantine`、`reject`。
- `trust_level`：运行可信度，支持 `full`、`degraded`、`partial`、`unknown`、`untrusted`。
- `release_package`、`annotation_package_uri`、`audit_package_uri`：交付、标注和审计产物位置。
- `degrade_summary`、`review_suggestions`、`explanation_summary`：降级、复核和解释信息。

`common/enums.py`、`common/evidence.py`、`common/policy.py`、`common/delivery.py` 和 `common/runtime.py` 提供了统一枚举、证据事件、策略评估、交付包与运行上下文支持。

## 文本合规

文本模块的当前主入口是 `text/api_pipeline.py` 中的 `APICompliancePipeline`，服务入口是 `text/api_server.py`。`text/pipeline.py` 中仍保留经典本地流水线，主要用于兼容较早的规则式处理链。

### 输入

文本 intake 支持目录、JSONL、JSON、TXT、TEXT、Markdown 文件。结构化记录中会优先读取这些文本字段：

```text
cleaned_text, text, content, body, document_text, normalized_text, payload_text
```

样本 ID 会优先读取：

```text
doc_id, id, record_id, sample_id, uid
```

目录输入会尝试识别 `metadata*`、`manifest*`、cleaned JSON/JSONL 以及原始 TXT/Markdown 文件。

### 处理画像

API 流水线支持三类处理画像：

- `full`：隐私识别、内容安全、硬案例裁决、策略汇总和交付产物。
- `privacy_only`：只执行隐私/PII 相关检测和后续策略。
- `safety_only`：只执行内容安全相关检测和后续策略。

### 主要产物

文本 API 流水线会在工作目录中写出 JSONL 审计链，常见产物包括：

```text
01_intake.jsonl
01b_document_context.jsonl
01c_document_views.jsonl
02_content_safety.jsonl
02b_content_safety_decisions.jsonl
02d_content_safety_review_tasks.jsonl
02f_content_safety_final_decisions.jsonl
03_privacy_detection.jsonl
03c_privacy_policy_decisions.jsonl
03e_privacy_review_tasks.jsonl
03i_privacy_final_decisions.jsonl
04_hard_case_adjudication.jsonl
05_evidence_events.jsonl
06_policy_decisions.jsonl
07_annotation_package.jsonl
08_audit_package.jsonl
09_run_summary.jsonl
10_annotation_exports/
```

### API

默认端口来自 `text/config/settings.py`，当前为 `19002`。

```bash
python -m uvicorn text.api_server:app --host 0.0.0.0 --port 19002
```

主要接口：

```text
GET  /api/v1/health
POST /api/v1/check
POST /api/v1/check-file
POST /api/v2/text/content-safety/check
GET  /api/v1/status/{task_id}
GET  /api/v1/result/{task_id}
GET  /api/v1/review/tasks/{task_id}
POST /api/v1/review/tasks/{task_id}/decisions
GET  /api/v1/privacy/review/tasks/{task_id}
POST /api/v1/privacy/review/tasks/{task_id}/decisions
GET  /api/v1/tasks
```

`text/server.py` 还提供兼容平台侧 compliance-agent 的接口，包括 `/api/v1/text/compliance-agent/jobs`。

## 音频合规

音频模块入口是 `audio/server.py` 和 `audio/pipeline.py`。当前配置中 `audio_execution_route` 默认是 `api`，即音频服务先做标准化、ASR 和转录单元构建，再通过 `audio/text_api_bridge.py` 调用文本 API 复用文本合规能力。

当 `audio_execution_route=local` 时，会使用 `AudioCompliancePipeline` 执行本地完整流水线。

### 操作类型映射

音频桥接层支持以下操作标识：

```text
CMP_001 -> 敏感信息检测 -> text profile privacy_only
CMP_002 -> 内容安全检测 -> text profile safety_only
CMP_008 -> 完整合规检测 -> text profile full
```

### 本地流水线

本地音频流水线包括：

```text
源登记 -> 音频画像 -> 格式标准化 -> ASR -> 说话人分离
-> 时间对齐 -> 转录单元 -> 隐私检测/脱敏 span
-> 内容安全 -> 硬案例裁决 -> 证据包 -> 策略决策
-> 标注包 -> 审计包 -> 运行摘要 -> compliance_output.json
```

常见产物：

```text
01_source_registry.jsonl
02_source_profile.jsonl
03_normalized_audio_manifest.jsonl
04_asr_segments.jsonl
05_speaker_segments.jsonl
06_aligned_segments.jsonl
07_transcript_units.jsonl
08_privacy_detection.jsonl
08b_redaction_spans.jsonl
09_content_safety.jsonl
09b_hard_case_adjudication.jsonl
10_evidence_bundle.json
11_policy_decision.json
12_annotation_package.jsonl
13_audit_package.jsonl
14_run_summary.jsonl
compliance_output.json
```

### API

默认端口是 `19001`。

```bash
python -m uvicorn audio.server:app --host 0.0.0.0 --port 19001
```

主要接口：

```text
GET  /api/v1/health
POST /api/v1/check
POST /api/v1/check-file
GET  /api/v1/status/{task_id}
GET  /api/v1/result/{task_id}
```

默认 API 路由依赖文本服务 `http://127.0.0.1:19002`。音频/视频处理还依赖可用的 `ffmpeg` 和 `ffprobe`。

## 图片合规

图片模块入口是 `picture/api/app.py`，核心编排在 `picture/application/orchestrator.py`，依赖注入工厂在 `picture/application/use_cases.py`。

图片处理链采用统一路由，结合 OCR、文本合规、视觉安全、敏感目标检测、分割细化和 OpenCV 脱敏输出。默认配置位于 `picture/infra/config.py`。

### 默认提供方

当前默认配置倾向于本地服务组合：

```text
OCR                 -> paddleocr_vl_api, http://127.0.0.1:8217
PII/text compliance -> text_compliance / text_api, http://127.0.0.1:19002
safety              -> qwen_sam3_safety_fusion
vision              -> qwen_sam3_api_fusion
segmentation        -> sam3_api, http://127.0.0.1:8218
```

代码也保留了 mock、OpenAI、Presidio、ShieldGemma2、YOLO、GroundingDINO、SAM2/SAM3 等 provider 适配路径，具体由环境变量和配置选择。

### 处理结果

图片任务会生成：

- 原图、合规图、可选 overlay 图。
- `PictureFinding` 风险发现和区域定位。
- `PicturePolicyResult` 策略结果。
- 结构化报告、标注包和审计包 URI。
- provider 版本、降级事件和 step audit。

### API

默认端口是 `19012`。

```bash
python -m uvicorn picture.api.app:app --host 0.0.0.0 --port 19012
```

主要接口：

```text
GET  /api/v1/health
GET  /api/v1/readiness
GET  /v1/picture/admin/runtime
POST /v1/picture/admin/warmup
POST /v1/picture/jobs
POST /v1/picture/jobs/file
GET  /v1/picture/jobs/{job_id}
GET  /v1/picture/jobs/{job_id}/result
GET  /v1/picture/jobs/{job_id}/artifact/{kind}
GET  /v1/picture/jobs/{job_id}/findings
GET  /v1/picture/jobs/{job_id}/report
POST /v1/picture/jobs/{job_id}/rerun
POST /v1/picture/jobs/{job_id}/manual-redaction
```

`artifact/{kind}` 支持 `original`、`compliant`、`overlay` 和 `report`。

## 视频合规

视频模块入口是 `video/server.py` 和 `video/pipeline.py`。当前视频栈通过抽帧调用图片 API 进行帧级合规分析，并可选抽取原生音轨调用音频侧路。代码中没有把图片能力作为默认内联 fallback，因此视频服务运行时应先启动图片服务。

### 输入

`video/application/services.py` 支持：

- 帧目录。
- 动态图文件，例如 GIF、WebP、带多帧的 PNG/APNG。
- 视频容器：MP4、MOV、MKV、AVI、M4V、WebM。

### 处理链

视频流水线包括：

```text
输入准备 -> 帧/片段加载 -> 场景窗口 -> clip 窗口
-> 图片 API 帧分析 -> 可选音频抽取和音频合规
-> 风险聚合 -> 可选 SAM3 视频跟踪
-> 治理策略 -> 动作计划 -> 复核队列
-> 质量报告 -> 可选脱敏衍生视频 -> 审计/标注/最终输出
```

常见产物包括帧清单、场景清单、clip 审计、视频发现、风险标注、风险轨迹、复核队列、质量报告、策略决策、动作计划、annotation overlay、audit package、`compliance_output` 和 report。

### API

默认端口是 `19003`。

```bash
python -m uvicorn video.server:app --host 0.0.0.0 --port 19003
```

主要接口：

```text
GET  /api/v1/health
POST /api/v1/check
POST /api/v1/check-file
GET  /api/v1/status/{task_id}
GET  /api/v1/result/{task_id}
```

视频默认依赖：

```text
Picture API -> http://127.0.0.1:19012
Audio API   -> http://127.0.0.1:19001
SAM3 API    -> http://127.0.0.1:8218
ffmpeg/ffprobe
```

## 启动脚本

`scripts/` 中提供了面向本地模型服务的组合启动脚本。脚本主要通过 tmux 管理多个进程，默认路径和端口按当前仓库布局配置。

```bash
# 文本本地栈：PII 网关、Qwen3Guard、硬案例模型和文本 API
bash scripts/start_text_local_stack.sh start

# 音频栈：可选启动文本栈、ASR 适配器和音频 API
bash scripts/start_audio_local_stack.sh start

# 图片栈：图片 API；可选启动文本栈
bash scripts/start_picture_local_stack.sh start

# 视频完整栈：文本、PaddleOCR-VL、SAM3、音频、图片和视频服务
bash scripts/start_video_compliance_stack.sh start

# 单独启动 PaddleOCR-VL serving
bash scripts/start_paddleocr_vl_serving.sh start

# 单独启动 SAM3 API
bash scripts/start_sam3_api.sh start
```

部分脚本和配置默认使用 `/data/kw/compliance-checker` 下的模型目录。如果仓库移动到其他路径，需要同步调整环境变量或脚本中的默认路径。

## 模型目录和 Git 策略

`models/` 目录用于保留项目结构，但实际模型权重不提交。当前 `.gitignore` 规则会忽略 `models/**` 下的真实内容，只允许目录和 `.gitkeep` 占位文件进入 Git。

常见模型/工具目录包括：

```text
models/Qwen/Qwen3-ASR-0.6B/
models/Qwen/Qwen3Guard-Gen-0.6B/
models/Qwen/Qwen3.5-9B/
models/paddleocr_vl/PaddleOCR-VL-1.5/
models/paddleocr_vl/PP-DocLayoutV2/
models/facebook/sam3/
models/facebook/sam2-hiera-large/
models/compliance-pii/
models/Meddies/meddies-pii/
models/ffmpeg-7.0.2-amd64-static/
models/pyannote/
models/yolo/
models/visual_privacy/
```

不要提交以下内容：

```text
.venv/
.venvs/
venv/
env/
__pycache__/
.pytest_cache/
compliance_output*/
outputs/
video_test_runs/
*.pt
*.pth
*.safetensors
*.onnx
*.gguf
*.bin
*.mp4
*.wav
*.jpg
*.png
```

如果新增模型目录，只提交目录下的 `.gitkeep` 或其他轻量说明文件，不提交模型权重。

## 依赖安装

仓库按模态和运行环境拆分依赖。建议按需要安装，不要在一个环境中盲目合并全部重型依赖。

轻量模块依赖可按当前要运行的模态选择安装：

```bash
python -m venv .venv
source .venv/bin/activate

# 按需选择，不要求全部装进同一个环境
pip install -r text/requirements.txt
pip install -r audio/requirements.txt
pip install -r picture/requirements.txt
pip install -r picture/requirements-local.txt
pip install -r video/requirements.txt
```

本地模型服务和专项环境依赖位于 `requirements/`，例如：

```text
requirements/root-venv.txt
requirements/compliance-pii.txt
requirements/paddleocr-vl.txt
requirements/paddleocr-vl-vllm.txt
requirements/qwen-serving-audio.txt
requirements/qwen-serving-guard.txt
requirements/qwen-serving-hardcase.txt
requirements/sam3.txt
```

PaddleOCR-VL、SAM3、Qwen、ASR、视频渲染等能力通常需要独立 Python 环境、GPU/CUDA 支持和对应模型权重。

## 配置

文本和音频配置使用 `COMPLIANCE_` 环境变量前缀：

```text
COMPLIANCE_WORK_DIR
COMPLIANCE_POLICY_VERSION
COMPLIANCE_SERVER_HOST
COMPLIANCE_API_SERVER_PORT
COMPLIANCE_SERVER_PORT
COMPLIANCE_AUDIO_EXECUTION_ROUTE
COMPLIANCE_TEXT_API_BASE_URL
COMPLIANCE_QWEN_ASR_MODEL_PATH
COMPLIANCE_FFMPEG_BIN
COMPLIANCE_FFPROBE_BIN
```

图片配置使用 `PICTURE_` 前缀：

```text
PICTURE_WORK_DIR
PICTURE_SERVER_PORT
PICTURE_OCR_PROVIDER
PICTURE_PADDLEOCR_VL_API_URL
PICTURE_TEXT_API_BASE_URL
PICTURE_SAM3_API_URL
PICTURE_STORAGE_BASE_PATH
```

视频配置使用 `VIDEO_` 前缀：

```text
VIDEO_WORK_DIR
VIDEO_SERVER_PORT
VIDEO_PICTURE_API_BASE_URL
VIDEO_AUDIO_API_BASE_URL
VIDEO_SAM3_API_BASE_URL
VIDEO_ENABLE_AUDIO_SIDECAR
VIDEO_FFMPEG_BIN
VIDEO_FFPROBE_BIN
```

## 测试

默认测试配置会跳过 `integration` 和 `slow` 标记：

```bash
python -m pytest
```

运行指定模态测试：

```bash
python -m pytest text
python -m pytest audio
python -m pytest picture
python -m pytest video
```

需要外部服务或大模型的测试应显式使用对应 marker 或单独运行。Windows 环境可参考 `scripts/run_pytest.ps1`。

## 当前运行注意事项

- 文本 API 是其他模态复用文本合规能力的基础服务，建议优先启动。
- 音频默认 API 路由依赖文本 API；切换成本地完整流水线需要配置 `COMPLIANCE_AUDIO_EXECUTION_ROUTE=local`。
- 图片默认依赖 PaddleOCR-VL API、文本 API 和 SAM3 API；未启动外部 provider 时会发生降级或失败，具体取决于配置。
- 视频当前依赖图片 API 进行帧级分析，并可选依赖音频 API 处理原生音轨。
- 图片和视频服务中的任务状态主要保存在进程内存或本地存储中，服务重启后不要假定仍保留完整内存态。
- 运行输出目录和缓存目录已经被 `.gitignore` 排除，排查问题时应直接查看本地工作目录中的 JSONL、报告和日志。
