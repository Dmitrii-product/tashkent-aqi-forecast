"""The site icon (browser tab, bookmarks, home screen): an AQI gauge.

A dark rounded square with a half-ring in the site's AQI category colours (good → unhealthy)
and a white needle in the orange band. One geometry, in a 64-unit box, drawn twice: as SVG
(modern browsers, sharp at any size) and with Pillow at 16x and downscaled (favicon.ico with
16/32/48 px for older browsers and search engines, apple-touch-icon.png 180 px for iOS/Safari).

aqf.site writes the files into output/site/ at every build, so nothing binary is kept in git.

    .venv/bin/python -m aqf.icon   # writes them into output/site/ and prints the sizes
"""
from __future__ import annotations

import io
import math
from pathlib import Path

from PIL import Image, ImageDraw

BG = "#15212c"            # the site's ink colour
NEEDLE = "#ffffff"
BANDS = ("#a8e05f", "#fdd64b", "#ff9b57", "#fe6a69")  # --aqi-1..4 of the dashboard
CX, CY, R, W = 32.0, 43.0, 21.0, 11.0  # ring centre, mid radius, thickness
GAP = 1.5                               # degrees between bands
NEEDLE_DEG, NEEDLE_LEN, NEEDLE_W, HUB = 60.0, 18.0, 5.0, 6.5
CORNER = 14.0

FILES = ("icon.svg", "favicon.ico", "apple-touch-icon.png")
LINKS = ('<link rel="icon" href="/favicon.ico" sizes="32x32">'
         '<link rel="icon" href="/icon.svg" type="image/svg+xml">'
         '<link rel="apple-touch-icon" href="/apple-touch-icon.png">')


def _bands() -> list[tuple[float, float, str]]:
    """(from, to, colour) in degrees counter-clockwise from 3 o'clock, left to right."""
    step = 180 / len(BANDS)
    return [(180 - i * step - (GAP / 2 if i else 0), 180 - (i + 1) * step + (GAP / 2 if i < len(BANDS) - 1 else 0), c)
            for i, c in enumerate(BANDS)]


def _pt(deg: float, r: float) -> tuple[float, float]:
    return CX + r * math.cos(math.radians(deg)), CY - r * math.sin(math.radians(deg))


def svg() -> str:
    f = lambda v: f"{v:.2f}".rstrip("0").rstrip(".")  # noqa: E731
    arcs = "".join(
        f'<path d="M{f(_pt(a, R)[0])} {f(_pt(a, R)[1])}A{f(R)} {f(R)} 0 0 1 {f(_pt(b, R)[0])} {f(_pt(b, R)[1])}" stroke="{c}"/>'
        for a, b, c in _bands())
    tx, ty = _pt(NEEDLE_DEG, NEEDLE_LEN)
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
            f'<rect width="64" height="64" rx="{f(CORNER)}" fill="{BG}"/>'
            f'<g fill="none" stroke-width="{f(W)}">{arcs}</g>'
            f'<line x1="{f(CX)}" y1="{f(CY)}" x2="{f(tx)}" y2="{f(ty)}" stroke="{NEEDLE}" '
            f'stroke-width="{f(NEEDLE_W)}" stroke-linecap="round"/>'
            f'<circle cx="{f(CX)}" cy="{f(CY)}" r="{f(HUB)}" fill="{NEEDLE}"/></svg>')


def png(size: int, corner: bool = True) -> Image.Image:
    """The same drawing at `size` px. corner=False fills the square (iOS rounds it itself)."""
    s = 16 * size / 64
    im = Image.new("RGBA", (16 * size,) * 2, (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, 16 * size - 1, 16 * size - 1), radius=CORNER * s if corner else 0, fill=BG)
    ro = (R + W / 2) * s
    box = (CX * s - ro, CY * s - ro, CX * s + ro, CY * s + ro)
    for a, b, c in _bands():
        d.arc(box, start=360 - a, end=360 - b, fill=c, width=round(W * s))
    tx, ty = _pt(NEEDLE_DEG, NEEDLE_LEN)
    d.line((CX * s, CY * s, tx * s, ty * s), fill=NEEDLE, width=round(NEEDLE_W * s))
    rn = NEEDLE_W / 2 * s
    d.ellipse((tx * s - rn, ty * s - rn, tx * s + rn, ty * s + rn), fill=NEEDLE)
    d.ellipse((CX * s - HUB * s, CY * s - HUB * s, CX * s + HUB * s, CY * s + HUB * s), fill=NEEDLE)
    return im.resize((size, size), Image.LANCZOS)


def files() -> dict[str, bytes]:
    ico = io.BytesIO()
    big = png(48)
    big.save(ico, format="ICO", sizes=[(16, 16), (32, 32), (48, 48)], append_images=[png(16), png(32)])
    touch = io.BytesIO()
    png(180, corner=False).convert("RGB").save(touch, format="PNG", optimize=True)
    return {"icon.svg": svg().encode(), "favicon.ico": ico.getvalue(), "apple-touch-icon.png": touch.getvalue()}


def write(folder: Path) -> dict[str, int]:
    out = {}
    for name, data in files().items():
        (folder / name).write_bytes(data)
        out[name] = len(data)
    return out


if __name__ == "__main__":
    from .site import SITE
    print(write(SITE))
