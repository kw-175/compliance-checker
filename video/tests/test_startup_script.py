from __future__ import annotations

from pathlib import Path


def test_video_stack_starts_local_qwen_video_adapter() -> None:
    script = Path("scripts/start_video_compliance_stack.sh").read_text(encoding="utf-8")

    assert 'QWEN35_BASE_URL="${QWEN35_BASE_URL:-http://127.0.0.1:8301/v1}"' in script
    assert 'QWEN35_HEALTH_URL="${QWEN35_HEALTH_URL:-http://127.0.0.1:8301/health}"' in script
    assert 'QWEN35_MODEL="${QWEN35_MODEL:-Qwen3.5-9B}"' in script
    assert 'QWEN35_VIDEO_ADAPTER_PORT="${QWEN35_VIDEO_ADAPTER_PORT:-8219}"' in script
    assert 'VIDEO_CLIP_MODERATION_BASE_URL="${VIDEO_CLIP_MODERATION_BASE_URL:-$QWEN35_VIDEO_ADAPTER_URL}"' in script
    assert "ops.qwen35_video_adapter:app" in script
    assert 'VIDEO_CLIP_MODERATION_MAX_FRAMES="${VIDEO_CLIP_MODERATION_MAX_FRAMES:-4}"' in script
