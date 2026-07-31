from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException


app = FastAPI(title="Qwen3.5 Local Video Action Adapter")

VLLM_URL = os.getenv(
    "QWEN35_VIDEO_VLLM_URL",
    "http://127.0.0.1:8301/v1/chat/completions",
)
API_KEY = os.getenv("QWEN35_VIDEO_API_KEY", "")
MODEL = os.getenv("QWEN35_VIDEO_MODEL", "Qwen3.5-9B")
TIMEOUT_SECONDS = int(os.getenv("QWEN35_VIDEO_TIMEOUT_SECONDS", "180"))
MAX_TOKENS = int(os.getenv("QWEN35_VIDEO_MAX_TOKENS", "1536"))
MAX_FRAMES = int(os.getenv("QWEN35_VIDEO_MAX_FRAMES", "4"))

SYSTEM_PROMPT = """你是严格的视频内容安全审核器。
你会收到按时间顺序排列的短视频关键帧和结构化任务说明。
必须综合连续帧判断动作语义，不得仅根据单帧外观猜测行为。
只返回一个符合请求 output_schema 的 JSON 对象，不要返回 Markdown 或额外解释。
时间字段使用秒，bbox 使用输入帧像素坐标 [x,y,w,h]。
证据不足时不要编造风险；可设置 review_required=true。
"""


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_json(text: str) -> dict[str, Any]:
    stripped = _strip_code_fence(text)
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        try:
            payload, _ = decoder.raw_decode(stripped[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("No JSON object found in model output.")


def build_chat_payload(request_payload: dict[str, Any]) -> dict[str, Any]:
    frames = [item for item in request_payload.get("frames", []) if isinstance(item, dict)]
    frames = frames[: max(1, MAX_FRAMES)]
    task_payload = {key: value for key, value in request_payload.items() if key != "frames"}
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "任务参数：\n" + json.dumps(task_payload, ensure_ascii=False),
        }
    ]
    for frame in frames:
        image_base64 = str(frame.get("image_base64") or "").strip()
        if not image_base64:
            continue
        content.append(
            {
                "type": "text",
                "text": (
                    f"关键帧 frame_index={frame.get('frame_index')} "
                    f"timestamp={frame.get('timestamp')}秒"
                ),
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            }
        )
    if len(content) == 1:
        raise ValueError("At least one frame with image_base64 is required.")
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "qwen35-local-video-adapter",
        "backend_url": VLLM_URL,
        "backend_model": MODEL,
        "max_frames": MAX_FRAMES,
    }


@app.post("/video/action-recognition")
async def action_recognition(request_payload: dict[str, Any]) -> dict[str, Any]:
    try:
        chat_payload = build_chat_payload(request_payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(VLLM_URL, headers=headers, json=chat_payload)
        response.raise_for_status()
        data = response.json()
        raw_output = data["choices"][0]["message"]["content"]
        result = extract_json(raw_output)
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail=f"Qwen3.5 video inference failed: {exc}") from exc

    result.setdefault("events", [])
    if not isinstance(result["events"], list):
        raise HTTPException(status_code=502, detail="Qwen3.5 video response field 'events' must be a list")
    result["provider"] = "qwen35_local_video_adapter"
    result["model"] = MODEL
    return result
