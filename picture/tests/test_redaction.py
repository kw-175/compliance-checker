"""
Tests for redaction operations.

Verifies that all redaction modes (black_box, gaussian_blur, pixelate, solid_fill)
produce valid output images and that overlay rendering works.
"""
# 中文说明：这组测试重点保护 redactor 的输出路径是否可用，
# 而不是对像素结果做严格视觉比对，因此断言主要围绕“文件成功生成”。
from __future__ import annotations

from pathlib import Path

import pytest

from picture.domain.enums import RedactionMode
from picture.domain.models import BBox, RedactionOperation, RegionMask
from picture.providers.redaction.opencv_redactor import OpenCVRedactor

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def redactor() -> OpenCVRedactor:
    return OpenCVRedactor()


@pytest.fixture
def sample_operations() -> list[RedactionOperation]:
    """Sample redaction operations for testing."""
    # 中文说明：这里一次性覆盖四种脱敏模式，便于多模式混合场景测试。
    return [
        RedactionOperation(
            finding_id="f1",
            region=RegionMask(bbox=BBox(x=10, y=10, w=80, h=30), confidence=0.9),
            mode=RedactionMode.BLACK_BOX,
        ),
        RedactionOperation(
            finding_id="f2",
            region=RegionMask(bbox=BBox(x=10, y=50, w=60, h=40), confidence=0.85),
            mode=RedactionMode.GAUSSIAN_BLUR,
        ),
        RedactionOperation(
            finding_id="f3",
            region=RegionMask(bbox=BBox(x=10, y=100, w=70, h=30), confidence=0.8),
            mode=RedactionMode.PIXELATE,
        ),
        RedactionOperation(
            finding_id="f4",
            region=RegionMask(bbox=BBox(x=10, y=140, w=50, h=20), confidence=0.75),
            mode=RedactionMode.SOLID_FILL,
        ),
    ]


class TestRedactionModes:
    """Test individual redaction modes."""

    def test_black_box_redaction(self, redactor: OpenCVRedactor, tmp_path: Path):
        """Black box redaction should produce a valid image."""
        image_path = str(FIXTURES_DIR / "sample_document.png")
        output_path = str(tmp_path / "redacted_bbox.png")

        ops = [
            RedactionOperation(
                finding_id="f1",
                region=RegionMask(bbox=BBox(x=50, y=30, w=200, h=40), confidence=0.9),
                mode=RedactionMode.BLACK_BOX,
            )
        ]

        result = redactor.redact(image_path, ops, output_path)
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_gaussian_blur_redaction(self, redactor: OpenCVRedactor, tmp_path: Path):
        """Gaussian blur redaction should produce a valid image."""
        image_path = str(FIXTURES_DIR / "sample_natural.png")
        output_path = str(tmp_path / "redacted_blur.png")

        ops = [
            RedactionOperation(
                finding_id="f1",
                region=RegionMask(bbox=BBox(x=100, y=100, w=200, h=200), confidence=0.9),
                mode=RedactionMode.GAUSSIAN_BLUR,
            )
        ]

        result = redactor.redact(image_path, ops, output_path)
        assert Path(result).exists()

    def test_pixelate_redaction(self, redactor: OpenCVRedactor, tmp_path: Path):
        """Pixelate redaction should produce a valid image."""
        image_path = str(FIXTURES_DIR / "sample_mixed.png")
        output_path = str(tmp_path / "redacted_pixel.png")

        ops = [
            RedactionOperation(
                finding_id="f1",
                region=RegionMask(bbox=BBox(x=20, y=80, w=150, h=50), confidence=0.9),
                mode=RedactionMode.PIXELATE,
            )
        ]

        result = redactor.redact(image_path, ops, output_path)
        assert Path(result).exists()

    def test_solid_fill_redaction(self, redactor: OpenCVRedactor, tmp_path: Path):
        """Solid fill redaction should produce a valid image."""
        image_path = str(FIXTURES_DIR / "sample_document.png")
        output_path = str(tmp_path / "redacted_fill.png")

        ops = [
            RedactionOperation(
                finding_id="f1",
                region=RegionMask(bbox=BBox(x=50, y=90, w=150, h=30), confidence=0.9),
                mode=RedactionMode.SOLID_FILL,
            )
        ]

        result = redactor.redact(image_path, ops, output_path)
        assert Path(result).exists()

    def test_multiple_redactions(
        self,
        redactor: OpenCVRedactor,
        sample_operations: list,
        tmp_path: Path,
    ):
        """Multiple mixed redaction operations should all be applied."""
        image_path = str(FIXTURES_DIR / "sample_document.png")
        output_path = str(tmp_path / "redacted_multi.png")

        result = redactor.redact(image_path, sample_operations, output_path)
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_no_operations_copies_file(self, redactor: OpenCVRedactor, tmp_path: Path):
        """No redaction operations should result in a copy of the original."""
        image_path = str(FIXTURES_DIR / "sample_document.png")
        output_path = str(tmp_path / "redacted_none.png")

        result = redactor.redact(image_path, [], output_path)
        assert Path(result).exists()


class TestOverlayRendering:
    """Test overlay image rendering."""

    def test_overlay_rendering(
        self,
        redactor: OpenCVRedactor,
        sample_operations: list,
        tmp_path: Path,
    ):
        """Overlay rendering should produce a visualization image."""
        image_path = str(FIXTURES_DIR / "sample_document.png")
        overlay_path = str(tmp_path / "overlay.png")

        result = redactor.render_overlay(image_path, sample_operations, overlay_path)
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0
