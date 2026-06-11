from __future__ import annotations

from fastapi.testclient import TestClient

from ops import qwen_judge_adapter


def test_qwen_judge_adapter_returns_hard_case_judgement(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": """
                            ```json
                            {
                              "content_status": "borderline",
                              "privacy_status": "contains_pii",
                              "confidence": 0.77,
                              "rationale": "The record contains contextual PII and should be reviewed.",
                              "recommended_disposition": "P3",
                              "requires_manual_review": true,
                              "final_findings": []
                            }
                            ```
                            """
                        }
                    }
                ]
            }

    class AsyncClient:
        def __init__(self, timeout: int):
            self.timeout = timeout

        async def __aenter__(self) -> "AsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, headers: dict, json: dict) -> Response:
            assert url == qwen_judge_adapter.VLLM_URL
            assert json["model"] == "Qwen3.5-9B"
            assert json["messages"][0]["role"] == "system"
            assert json["messages"][1]["role"] == "user"
            return Response()

    monkeypatch.setattr("httpx.AsyncClient", AsyncClient)
    client = TestClient(qwen_judge_adapter.app)

    response = client.post(
        "/adjudicate",
        json={
            "model": "Qwen3.5-9B",
            "prompt": "{\"task\":\"resolve hard case\"}",
            "temperature": 0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    judgement = payload["judgement"]
    assert payload["provider"] == "qwen_judge"
    assert judgement["content_status"] == "borderline"
    assert judgement["privacy_status"] == "contains_pii"
    assert judgement["confidence"] == 0.77
    assert judgement["recommended_disposition"] == "P3"
    assert judgement["requires_manual_review"] is True


def test_qwen_judge_adapter_normalizes_invalid_labels() -> None:
    judgement = qwen_judge_adapter.normalize_judgement(
        {
            "content_status": "maybe",
            "privacy_status": "unknown",
            "confidence": "high",
            "recommended_disposition": "reject",
            "requires_manual_review": "yes",
            "final_findings": {"not": "a list"},
        }
    )

    assert judgement.content_status == "borderline"
    assert judgement.privacy_status == "borderline"
    assert judgement.confidence == 0.5
    assert judgement.recommended_disposition.value == "P3"
    assert judgement.requires_manual_review is True
    assert judgement.final_findings == []
