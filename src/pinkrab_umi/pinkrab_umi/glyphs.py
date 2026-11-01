"""Letters cut into printed parts: +, -, X, Y, Z, L, R as strokes.

Seven glyphs are the whole alphabet this project engraves - axis names on the
handheld's wall, face names in the marker solid's pockets, the hand on the
solid's triangles - and every part reads
from one table, so a letter cannot come out one shape on the wall and another
on the solid. Each glyph is a few strokes in a unit box, (u, v) from -0.5 to
0.5; a stroke is a rectangle of the given width between two of its points.
"""
from __future__ import annotations

import math

GLYPHS: dict[str, tuple[tuple[tuple[float, float], tuple[float, float]], ...]] = {
    "+": (((-0.5, 0.0), (0.5, 0.0)), ((0.0, -0.5), (0.0, 0.5))),
    "-": (((-0.5, 0.0), (0.5, 0.0)),),
    "X": (((-0.5, -0.5), (0.5, 0.5)), ((-0.5, 0.5), (0.5, -0.5))),
    "Y": (((-0.5, 0.5), (0.0, 0.0)), ((0.5, 0.5), (0.0, 0.0)), ((0.0, 0.0), (0.0, -0.5))),
    "Z": (((-0.5, 0.5), (0.5, 0.5)), ((0.5, 0.5), (-0.5, -0.5)), ((-0.5, -0.5), (0.5, -0.5))),
    "L": (((-0.5, 0.5), (-0.5, -0.5)), ((-0.5, -0.5), (0.5, -0.5))),
    "R": (((-0.5, -0.5), (-0.5, 0.5)), ((-0.5, 0.5), (0.25, 0.5)), ((0.25, 0.5), (0.5, 0.3)),
          ((0.5, 0.3), (0.5, 0.2)), ((0.5, 0.2), (0.25, 0.0)), ((0.25, 0.0), (-0.5, 0.0)),
          ((0.0, 0.0), (0.5, -0.5))),
}


def text_section(text: str, cx: float, cy: float, height: float, width: float,
                 stroke: float, gap: float, reading: float = 1.0):
    """`text` as a manifold3d CrossSection in a plane, centred on (cx, cy).

    `reading` is which way along x the string runs: +1 for a face read from in
    front of it, -1 for one read from behind, so the letters come out in order
    either way. Glyphs themselves are left as they are - Z, L and R are not
    their own mirror images, and they are only ever read from in front.
    """
    import manifold3d as m

    pitch = width + gap
    x0 = cx - reading * (len(text) - 1) * pitch / 2.0
    out = m.CrossSection()
    for i, char in enumerate(text):
        ox = x0 + reading * i * pitch
        for (u0, v0), (u1, v1) in GLYPHS[char]:
            a = (ox + u0 * width, cy + v0 * height)
            b = (ox + u1 * width, cy + v1 * height)
            dx, dy = b[0] - a[0], b[1] - a[1]
            out = out + m.CrossSection.square([math.hypot(dx, dy), stroke], True) \
                .rotate(math.degrees(math.atan2(dy, dx))) \
                .translate([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2])
    return out
