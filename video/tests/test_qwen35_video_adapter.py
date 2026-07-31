from __future__ import annotations

import asyncio
import base64

import httpx
import pytest

from ops import qwen35_video_adapter as adapter


def _request_payload() -> dict:
    return {
        "frames": [
            {"frame_index": 0, "timestamp": 0.0, "image_base64": base64.b64encode(b"jpeg").decode("ascii")},
            {"frame_index": 1, "timestamp": 0.5, "image_base64": base64.b64encode(b"jpeg2").decode("ascii")},
        ],
        "labels": ["physical_conflict", "normal_education"],
        "output_schema": {"events": []},
        "task": "video_safety_detection_with_temporal_localization",
    }


def test_build_chat_payload_uses_openai_multimodal_messages() -> None:
    payload = adapter.build_chat_payload(_request_payload())

    assert payload["model"] == adapter.MODEL
    content = payload["messages"][1]["content"]
    images = [item for item in content if item["type"] == "image_url"]
    assert len(images) == 2
    assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert payload["response_format"] == {"type": "json_object"}


def test_action_recognition_returns_events(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"events":[{"category":"physical_conflict","confidence":0.91}]}'
                        }
                    }
                ]
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, headers, json):
            assert url == adapter.VLLM_URL
            assert json["model"] == adapter.MODEL
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda timeout: FakeClient())

    result = asyncio.run(adapter.action_recognition(_request_payload()))

    assert result["events"][0]["category"] == "physical_conflict"
    assert result["provider"] == "qwen35_local_video_adapter"


def test_build_chat_payload_requires_an_image() -> None:
    with pytest.raises(ValueError, match="At least one frame"):
        adapter.build_chat_payload({"frames": [], "task": "video"})
