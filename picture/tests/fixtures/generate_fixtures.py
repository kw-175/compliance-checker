"""
Generate sample test fixture images for the picture compliance tests.

Run this script to create the fixture images:
    python -m picture.tests.fixtures.generate_fixtures
"""
# 中文说明：这个脚本专门用于生成测试夹具图片。
# 它的目标不是生成真实业务样本，而是构造能稳定触发不同链路的最小输入。
from __future__ import annotations

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent


def generate_fixtures() -> None:
    """Generate sample PNG test fixture images using Pillow."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        # 中文说明：没有 Pillow 时退化为创建最小 PNG，至少保证测试目录结构完整。
        print("Pillow not available. Creating minimal PNG files.")
        _create_minimal_pngs()
        return

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # 中文说明：文档图做成高竖版、低色彩变化，用于触发 document 链路。
    doc = Image.new("RGB", (600, 900), (255, 255, 255))
    draw = ImageDraw.Draw(doc)
    draw.rectangle([50, 30, 550, 70], fill=(0, 0, 0))
    draw.rectangle([50, 90, 400, 110], fill=(80, 80, 80))
    draw.rectangle([50, 130, 350, 150], fill=(100, 100, 100))
    draw.rectangle([50, 170, 500, 190], fill=(60, 60, 60))
    draw.rectangle([50, 210, 450, 230], fill=(90, 90, 90))
    doc.save(str(FIXTURES_DIR / "sample_document.png"))

    # 中文说明：自然图做成天空、草地、太阳等高色彩变化场景，用于触发 natural 链路。
    nat = Image.new("RGB", (800, 600), (135, 206, 235))
    draw = ImageDraw.Draw(nat)
    draw.rectangle([0, 400, 800, 600], fill=(34, 139, 34))
    draw.ellipse([600, 50, 700, 150], fill=(255, 223, 0))
    draw.polygon([(100, 400), (130, 250), (160, 400)], fill=(0, 100, 0))
    draw.polygon([(300, 400), (340, 200), (380, 400)], fill=(0, 120, 0))
    nat.save(str(FIXTURES_DIR / "sample_natural.png"))

    # 中文说明：混合截图同时包含文本块和色块区域，用于触发 mixed 链路。
    mixed = Image.new("RGB", (800, 600), (240, 240, 240))
    draw = ImageDraw.Draw(mixed)
    draw.rectangle([0, 0, 800, 60], fill=(33, 33, 33))
    draw.rectangle([20, 80, 400, 100], fill=(0, 0, 0))
    draw.rectangle([20, 110, 350, 130], fill=(60, 60, 60))
    draw.rectangle([450, 80, 780, 300], fill=(100, 149, 237))
    draw.rectangle([0, 350, 200, 600], fill=(245, 245, 220))
    mixed.save(str(FIXTURES_DIR / "sample_mixed.png"))

    # 中文说明：unsafe 图片本身不需要真的包含违规内容，
    # 文件名命中 mock safety moderator 的规则即可。
    unsafe = Image.new("RGB", (400, 400), (200, 50, 50))
    draw = ImageDraw.Draw(unsafe)
    draw.rectangle([100, 100, 300, 300], fill=(150, 30, 30))
    unsafe.save(str(FIXTURES_DIR / "sample_unsafe_explicit.png"))

    print(f"Generated fixture images in {FIXTURES_DIR}")


def _create_minimal_pngs() -> None:
    """Create minimal valid PNG files without Pillow."""
    import struct
    import zlib

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    def make_png(width: int, height: int, r: int, g: int, b: int) -> bytes:
        """Create a minimal PNG with a solid color."""
        raw_data = b""
        for _ in range(height):
            # 中文说明：每一行的第一个字节是 PNG scanline 的 filter byte。
            raw_data += b"\x00"
            raw_data += bytes([r, g, b]) * width

        compressed = zlib.compress(raw_data)

        def chunk(chunk_type: bytes, data: bytes) -> bytes:
            c = chunk_type + data
            crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            return struct.pack(">I", len(data)) + c + crc

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", compressed)
        png += chunk(b"IEND", b"")
        return png

    (FIXTURES_DIR / "sample_document.png").write_bytes(make_png(100, 150, 255, 255, 255))
    (FIXTURES_DIR / "sample_natural.png").write_bytes(make_png(150, 100, 135, 206, 235))
    (FIXTURES_DIR / "sample_mixed.png").write_bytes(make_png(120, 100, 240, 240, 240))
    (FIXTURES_DIR / "sample_unsafe_explicit.png").write_bytes(make_png(80, 80, 200, 50, 50))
    print(f"Generated minimal fixture PNGs in {FIXTURES_DIR}")


if __name__ == "__main__":
    generate_fixtures()
