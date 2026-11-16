"""Generate frontend/favicon.png: a 32x32 'setu' (bridge) mark.

Pure standard library -- zlib + struct are enough to write a PNG, and the
alternative was adding Pillow to requirements.txt for one 32x32 image, which
would not survive the written-justification rule.

The mark is the project's own palette: the paper ground (#f7f6f1) with the
accent green (#1f5a45) arch and piers -- a bridge, which is what 'setu' means.
"""

import math
import struct
import zlib
from pathlib import Path

SIZE = 32
PAPER = (0xF7, 0xF6, 0xF1)
ACCENT = (0x1F, 0x5A, 0x45)


def inside_bridge(x: float, y: float) -> bool:
    """True where the bridge mark should be painted."""
    # Deck: a horizontal band across the middle.
    if 13.0 <= y <= 15.5 and 3.0 <= x <= 29.0:
        return True
    # Arch: the lower half of a circle centred on the deck.
    cx, cy, r = 16.0, 15.0, 9.5
    d = math.hypot(x - cx, y - cy)
    if y >= 15.0 and 7.6 <= d <= 9.6 and 4.0 <= x <= 28.0:
        return True
    # Two piers down to the base.
    if 15.0 <= y <= 27.0 and (4.0 <= x <= 6.2 or 25.8 <= x <= 28.0):
        return True
    # Base line.
    if 25.0 <= y <= 27.0 and 3.0 <= x <= 29.0:
        return True
    # Two uprights above the deck, the cables' suggestion.
    if 6.5 <= y <= 13.0 and (10.0 <= x <= 11.6 or 20.4 <= x <= 22.0):
        return True
    return False


def chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def build() -> bytes:
    rows = bytearray()
    for y in range(SIZE):
        rows.append(0)  # filter type 0 for this scanline
        for x in range(SIZE):
            # 2x2 supersample so the arch does not look like a staircase.
            hits = sum(
                inside_bridge(x + dx, y + dy)
                for dx in (0.25, 0.75)
                for dy in (0.25, 0.75)
            )
            t = hits / 4.0
            rows.extend(
                round(PAPER[i] + (ACCENT[i] - PAPER[i]) * t) for i in range(3)
            )

    header = struct.pack(">IIBBBBB", SIZE, SIZE, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + chunk(b"IEND", b"")
    )


if __name__ == "__main__":
    out = Path(r"E:\MYPROJECTS\NiyamSetu\frontend\favicon.png")
    out.write_bytes(build())
    print(f"wrote {out} ({out.stat().st_size} bytes)")
