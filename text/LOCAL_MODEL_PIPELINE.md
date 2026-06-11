# Text Local-Model Pipeline

## Purpose

This document describes the preferred local-model-first startup path for text compliance on this repository.

The runtime topology is:

`PII gateway -> Qwen3Guard vLLM -> Qwen3Guard adapter -> Qwen3.5 vLLM -> text.api_server`

The service entrypoint is [`text.api_server`](./api_server.py), which runs [`text/api_pipeline.py`](./api_pipeline.py) and prefers the local provider mode introduced for:

- document context building
- privacy detection with contextual governance
- content-safety recall plus semantic adjudication
- hard-case adjudication

## Ports

Recommended local ports:

- `5002`: PII gateway
- `8212`: Qwen3Guard vLLM
- `8215`: Qwen3Guard adapter
- `8301`: Qwen3.5-9B vLLM
- `19002`: text.api_server

Recommended GPU split:

- `Qwen3Guard`: GPU `1`
- `Qwen3.5-9B`: GPU `1`

Recommended safe default startup parameters for `Qwen3.5-9B`:

- `QWEN35_MAX_MODEL_LEN=6144`
- `QWEN35_MAX_NUM_SEQS=1`
- `QWEN35_GPU_MEMORY_UTILIZATION=0.78`

Recommended low-footprint startup parameters for `Qwen3Guard` when it still runs on vLLM:

- `QWEN3GUARD_MAX_MODEL_LEN=1536`
- `QWEN3GUARD_MAX_NUM_SEQS=1`
- `QWEN3GUARD_GPU_MEMORY_UTILIZATION=0.08`

## Required models

Expected local model directories:

```text
models/Qwen/Qwen3Guard-Gen-0.6B
models/Qwen/Qwen3.5-9B
models/compliance-pii/gliner-pii-large-v1.0
models/compliance-pii/stanza_resources
```

## Required environments

Default environment layout used by the startup script:

```text
.venv
.venvs/compliance-pii
qwen-serving/text-vllm/.venv
qwen-serving/asr-vllm/.venv
```

Environment roles:

- `.venvs/compliance-pii`: PII gateway
- `qwen-serving/asr-vllm/.venv`: Qwen3Guard vLLM and Qwen3 ASR runtime
- `qwen-serving/text-vllm/.venv`: Qwen3.5-9B vLLM
- `.venv`: Qwen3Guard adapter and text.api_server

The local text stack uses `asr-vllm` for `Qwen3Guard` and `text-vllm` for `Qwen3.5-9B`. The startup script validates that `vllm` is present and that `from transformers import GenerationConfig` succeeds before launching either text model service.

## Start

Preferred startup command:

```bash
bash scripts/start_text_local_stack.sh start
```

The script opens a tmux session with five windows:

1. `pii-gateway`
2. `qwen35-vllm`
3. `qwen3guard-vllm`
4. `qwen3guard-adapter`
5. `text-api-server`

Startup order is serialized to reduce GPU contention on the single A100:

1. `pii-gateway`
2. `qwen35-vllm`
3. wait for `http://127.0.0.1:8301/v1/models`
4. `qwen3guard-vllm`
5. wait for `http://127.0.0.1:8212/v1/models`
6. `qwen3guard-adapter`
7. wait for `http://127.0.0.1:8215/health`
8. `text-api-server`

Useful commands:

```bash
bash scripts/start_text_local_stack.sh status
bash scripts/start_text_local_stack.sh attach
bash scripts/start_text_local_stack.sh stop
```

## Health checks

Important endpoints:

```text
http://127.0.0.1:5002/analyze
http://127.0.0.1:8215/health
http://127.0.0.1:8215/moderate
http://127.0.0.1:8301/v1/models
http://127.0.0.1:19002/api/v1/health
```

Expected `text.api_server` health signal:

- `provider_mode = local_model`

## Smoke test

Run:

```bash
bash scripts/test_text_local_stack.sh
```

The smoke test:

- probes the PII gateway
- probes the Qwen3Guard adapter
- probes the Qwen3.5 vLLM service
- probes `text.api_server`
- submits `privacy_only`
- submits `safety_only`
- submits `full`

Expected output artifacts include:

```text
01_intake.jsonl
01b_document_context.jsonl
02_content_safety.jsonl
03_privacy_detection.jsonl
04_hard_case_adjudication.jsonl
09_run_summary.jsonl
```

## Key environment variables

The startup script sets these for `text.api_server`:

```bash
COMPLIANCE_COMPLIANCE_PROVIDER_MODE=local
COMPLIANCE_LOCAL_COMPLIANCE_BASE_URL=http://127.0.0.1:8301/v1
COMPLIANCE_LOCAL_COMPLIANCE_MODEL=Qwen3.5-9B
COMPLIANCE_LOCAL_COMPLIANCE_MAX_CHARS=4800
COMPLIANCE_LOCAL_COMPLIANCE_MAX_TOKENS=1024
COMPLIANCE_ENABLE_PRESIDIO=true
COMPLIANCE_PRESIDIO_ANALYZER_ENDPOINT=http://127.0.0.1:5002/analyze
COMPLIANCE_ENABLE_QWEN3GUARD=true
COMPLIANCE_QWEN3GUARD_ENDPOINT=http://127.0.0.1:8215/moderate
COMPLIANCE_API_SERVER_PORT=19002
```

## Relationship to legacy text server

Legacy stack:

- [`scripts/start_text_4window.sh`](../scripts/start_text_4window.sh)
- [`text.server`](./server.py)

Preferred local-model-first stack:

- [`scripts/start_text_local_stack.sh`](../scripts/start_text_local_stack.sh)
- [`text.api_server`](./api_server.py)

The legacy stack remains available, but it is not the preferred path for the final local-model compliance design.
