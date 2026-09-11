"""アプリアイコンの生成.

Pillow に依存せず、zlib だけで PNG を書く。ホーム画面に置くアイコンは
PNG でないと iOS が受け付けないため、SVG では代用できない。
"""

from __future__ import annotations

import struct
import zlib

# レポートと同じ配色を使う（同じアプリだと一目で分かるように）
INDIGO = (42, 74, 127)
PAPER = (246, 247, 249)
CRIMSON = (178, 52, 64)

# 折れ線の形。0..1 の正規化座標で、右肩上がりだが一度押し戻される。
SPARK = [(0.14, 0.68), (0.30, 0.52), (0.44, 0.61), (0.60, 0.36), (0.76, 0.44), (0.88, 0.22)]


def _png(width: int, height: int, rows: list[list[tuple[int, int, int]]]) -> bytes:
    raw = bytearray()
    for row in rows:
        raw.append(0)  # フィルタ種別 0 (None)
        for r, g, b in row:
            raw += bytes((r, g, b))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0)  # 8bit truecolor
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def _blend(base: tuple, over: tuple, alpha: float) -> tuple[int, int, int]:
    return tuple(int(round(b + (o - b) * alpha)) for b, o in zip(base, over, strict=True))


def _distance_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    cx, cy = ax + t * dx, ay + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def render_icon(size: int) -> bytes:
    """右肩上がりの折れ線 1 本。小さく表示しても形が潰れない."""
    stroke = max(2.0, size * 0.055)
    half = stroke / 2.0
    dot_r = max(3.0, size * 0.085)
    dot_x, dot_y = SPARK[-1][0] * size, SPARK[-1][1] * size
    segments = [
        (SPARK[i][0] * size, SPARK[i][1] * size, SPARK[i + 1][0] * size, SPARK[i + 1][1] * size)
        for i in range(len(SPARK) - 1)
    ]

    rows = []
    for y in range(size):
        row = []
        py = y + 0.5
        for x in range(size):
            px = x + 0.5
            color = INDIGO

            # 折れ線（アンチエイリアスは距離で 1px 分だけ効かせる）
            nearest = min(_distance_to_segment(px, py, *seg) for seg in segments)
            coverage = max(0.0, min(1.0, half + 0.5 - nearest))
            if coverage > 0:
                color = _blend(color, PAPER, coverage)

            # 終点のドット
            d = ((px - dot_x) ** 2 + (py - dot_y) ** 2) ** 0.5
            dot_coverage = max(0.0, min(1.0, dot_r + 0.5 - d))
            if dot_coverage > 0:
                color = _blend(color, CRIMSON, dot_coverage)

            row.append(color)
        rows.append(row)

    return _png(size, size, rows)
