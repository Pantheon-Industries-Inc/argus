"""Contact sheets: many instants tiled into one image, each tile headed with its exact time.

A tile is one instant: every camera's picture of it stacked top to bottom, under a header with its time and an
optional role (prev / this / next). Tiles run left to right, then top to bottom, in time order. Timestamps drawn on
every tile were worth more than any other layout choice in published WGO-Bench work (Macrodata: F1 0.201 to 0.263
at a twelfth of the cost of separate images), so every tile carries its own. The font is label/fonts' DejaVu Sans
Bold laid out with Pillow's basic engine, so a sheet is byte-identical on every machine.
"""
from __future__ import annotations

import base64
import io

from label.frames import _grid_font

HEADER = 30
GAP = 4
BG = (18, 18, 20)


def sheet(tiles: list[tuple[float, dict[str, bytes]]], cameras: list[str], cols: int, *, roles: list[str] | None = None,
          cell_w: int | None = None, quality: int = 88, png: bool = False) -> bytes:
    """One JPEG sheet. tiles: [(t_s, {camera: jpeg})]. roles, when given, is one word per tile shown after the time.
    cell_w, when given, shrinks every picture to at most that width (never enlarges)."""
    from PIL import Image, ImageDraw

    from label.frames import downscaled
    if not tiles:
        raise ValueError("an empty sheet")
    decoded = [{c: Image.open(io.BytesIO(cams[c])).convert("RGB") for c in cameras if c in cams} for _t, cams in tiles]
    if cell_w:
        decoded = [{c: downscaled(im, cell_w) for c, im in d.items()} for d in decoded]
    w = max((im.width for d in decoded for im in d.values()), default=1)
    cam_h = {c: max((d[c].height for d in decoded if c in d), default=0) for c in cameras}
    cam_h = {c: h for c, h in cam_h.items() if h}
    tile_h = HEADER + sum(h + GAP for h in cam_h.values())
    ncols = min(cols, len(tiles))
    nrows = -(-len(tiles) // ncols)
    g = Image.new("RGB", (ncols * (w + GAP), nrows * (tile_h + GAP)), BG)
    d = ImageDraw.Draw(g)
    font = _grid_font(20)
    for i, ((t, _c), ims) in enumerate(zip(tiles, decoded)):
        x, y = (i % ncols) * (w + GAP), (i // ncols) * (tile_h + GAP)
        label = f"t={t:.2f}s" + (f"  {roles[i]}" if roles else "")
        d.text((x + 6, y + 5), label, fill=(255, 220, 0), font=font)
        yy = y + HEADER
        for c, h in cam_h.items():
            im = ims.get(c)
            if im is not None:
                g.paste(im, (x + (w - im.width) // 2, yy))
            yy += h + GAP
    buf = io.BytesIO()
    if png:
        g.save(buf, format="PNG", optimize=True)
    else:
        g.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def image_part(data: bytes, detail: str = "high") -> dict:
    mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii"),
                                               "detail": detail}}


def sheets(footage, times: list[float], cols: int, rows: int, *, roles: list[str] | None = None,
           cell_w: int | None = None) -> list[tuple[str, bytes]]:
    """The instants as consecutive sheets of cols x rows tiles: [(caption, jpeg)]."""
    cams = footage.cameras
    per = cols * rows
    out = []
    for i in range(0, len(times), per):
        ts = times[i:i + per]
        tiles = [(t, footage.at(t)) for t in ts]
        rs = roles[i:i + per] if roles else None
        caption = f"=== sheet t={ts[0]:.2f}-{ts[-1]:.2f}s, {len(ts)} instants, left to right then top to bottom ==="
        out.append((caption, sheet(tiles, cams, cols, roles=rs, cell_w=cell_w)))
    return out
