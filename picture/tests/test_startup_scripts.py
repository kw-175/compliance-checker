"""Tests for picture startup script defaults."""
from __future__ import annotations

from pathlib import Path


def test_picture_startup_defaults_visual_safety_to_qwen_sam3_fusion() -> None:
    script = Path("/data/kw/compliance-checker/scripts/start_picture_local_stack.sh").read_text()

    assert 'PICTURE_SAFETY_PROVIDER="${PICTURE_SAFETY_PROVIDER:-qwen_sam3_safety_fusion}"' in script
    assert 'export PICTURE_SAFETY_PROVIDER="$PICTURE_SAFETY_PROVIDER"' in script
    assert "export PICTURE_SAFETY_PROVIDER=qwen35_vl" not in script
