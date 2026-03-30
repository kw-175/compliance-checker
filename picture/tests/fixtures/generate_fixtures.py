"""
Generate sample test fixture images for the picture compliance tests.

Run this script to create the fixture images:
    python -m picture.tests.fixtures.generate_fixtures
"""

from __future__ import annotations

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent


def generate_fixtures() -> None:
    """Generate sample PNG test fixture images using Pillow."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow not available. Creating minimal PNG files.")
        _create_minimal_pngs()
        return

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Document image (tall, mostly white, with text-like blocks)
    doc = Image.new("RGB", (600, 900), (255, 255, 255))
    draw = ImageDraw.Draw(doc)
    draw.rectangle([50, 30, 550, 70], fill=(0, 0, 0))
    draw.rectangle([50, 90, 400, 110], fill=(80, 80, 80))
    draw.rectangle([50, 130, 350, 150], fill=(100, 100, 100))
    draw.rectangle([50, 170, 500, 190], fill=(60, 60, 60))
    draw.rectangle([50, 210, 450, 230], fill=(90, 90, 90))
    doc.save(str(FIXTURES_DIR / "sample_document.png"))

    # 2. Natural image (colorful, landscape-like)
    nat = Image.new("RGB", (800, 600), (135, 206, 235))  # sky blue
    draw = ImageDraw.Draw(nat)
    # Ground
    draw.rectangle([0, 400, 800, 600], fill=(34, 139, 34))
    # Sun
    draw.ellipse([600, 50, 700, 150], fill=(255, 223, 0))
    # Trees
    draw.polygon([(100, 400), (130, 250), (160, 400)], fill=(0, 100, 0))
    draw.polygon([(300, 400), (340, 200), (380, 400)], fill=(0, 120, 0))
    nat.save(str(FIXTURES_DIR / "sample_natural.png"))

    # 3. Mixed screenshot (has both text blocks and colorful UI elements)
    mixed = Image.new("RGB", (800, 600), (240, 240, 240))
    draw = ImageDraw.Draw(mixed)
    # Header bar
    draw.rectangle([0, 0, 800, 60], fill=(33, 33, 33))
    # Text blocks
    draw.rectangle([20, 80, 400, 100], fill=(0, 0, 0))
    draw.rectangle([20, 110, 350, 130], fill=(60, 60, 60))
    # Image area
    draw.rectangle([450, 80, 780, 300], fill=(100, 149, 237))
    # Sidebar
    draw.rectangle([0, 350, 200, 600], fill=(245, 245, 220))
    mixed.save(str(FIXTURES_DIR / "sample_mixed.png"))

    # 4. Unsafe image (just has "unsafe" in filename for mock safety moderator)
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
            raw_data += b"\x00"  # filter byte
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
