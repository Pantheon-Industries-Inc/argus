"""Whether a camera's image is a circle with black corners, the mark of a fisheye lens whose image circle is smaller
than the sensor (FastUMI's gripper cameras). A fisheye whose image circle is wider than the sensor (RealOmni's
gripper cameras, Gen-HumanEgo's and Egocentric-100K's head cameras) fills the frame and is not named by this check.

The check reads the frames the request already decoded, so it costs no extra decode and gives the same answer for
the same episode every time. Each of up to MAX_FRAMES sampled frames, spread evenly over the episode, is shrunk to a
THUMB_W px wide grey image, and every pixel keeps its brightest value over those frames. Outside a lens's image
circle the sensor gets no light, so those pixels stay black in every frame however the camera moves, while a dark
part of a real scene is lit in at least one frame of a moving camera. The pixels that never pass DARK_MAX are then
matched to the outside of an ellipse centred on the image (a circle, once a video's stored shape stretches it),
with the radii that misplace the fewest pixels. The image is circular when

- all four corners never pass DARK_MAX,
- at least MIN_OUTSIDE of the image lies outside the ellipse, and at least MIN_PURITY of that never passes DARK_MAX,
- at most MAX_DARK_INSIDE of the inside never passes DARK_MAX,
- and the band just inside the edge is lit (its median brightest value is at least EDGE_LIT): a lens's image
  circle ends sharply, while a dark scene fades out towards a vignetted edge.

A normal frame fails the first rule (its corners are lit in some frame), a dark scene the last two, and a gripper's
dark fingers along the bottom edge the first (the top corners are lit).
"""
from __future__ import annotations

import numpy as np

DESC = "a fisheye lens; straight lines curve near the edge"
THUMB_W = 64
SOURCE_W = 192          # the narrowest grid cell (label/episode.py CELL_W_STEPS)
MAX_FRAMES = 24
DARK_MAX = 40           # 8-bit luma a never-lit pixel stays under in every frame
MIN_OUTSIDE = 0.02
MIN_PURITY = 0.90
MAX_DARK_INSIDE = 0.10
EDGE_LIT = 80
EDGE_BAND = 0.85        # the band just inside the edge: from this fraction of the radii out to the edge
RADII = np.round(np.arange(0.50, 1.5001, 0.02), 2)   # ellipse radii, as fractions of the half-width and half-height


def thumb(im) -> np.ndarray:
    """A frame as a THUMB_W px wide 8-bit grey array, made from its SOURCE_W px copy (the narrowest grid cell, which
    every request keeps), so a frame kept full size and one kept only at cell widths give the same thumbnail."""
    from PIL import Image
    from label import frames as mf
    if isinstance(im, mf.Shrunk):
        im = im.by_width.get(SOURCE_W) or im.by_width[min(im.by_width)]
    else:
        im = mf.downscaled(im, SOURCE_W)
    g = im.convert("L")
    h = max(2, int(round(g.height * THUMB_W / g.width)))
    return np.asarray(g.resize((THUMB_W, h), Image.BOX), dtype=np.uint8)


def spread(n: int, k: int = MAX_FRAMES) -> list[int]:
    """Up to k indices spread evenly over range(n), first and last included."""
    if n <= k:
        return list(range(n))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, k)})


def measure(bright: np.ndarray) -> dict:
    """The circle test on one camera's brightest-value map (thumb() size): the numbers it decides on, and the
    decision."""
    dark = bright < DARK_MAX
    h, w = dark.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x2 = ((xx - (w - 1) / 2) / (w / 2)) ** 2
    y2 = ((yy - (h - 1) / 2) / (h / 2)) ** 2
    lit = ~dark
    best = None
    for a in RADII:
        for b in RADII:
            e = x2 / a ** 2 + y2 / b ** 2
            out = e > 1
            err = int(np.count_nonzero(out & lit) + np.count_nonzero(~out & dark))
            if best is None or err < best[0]:
                best = (err, float(a), float(b), e)
    _, a, b, e = best
    out = e > 1
    n_out = int(np.count_nonzero(out))
    outside = n_out / out.size
    purity = float(np.count_nonzero(out & dark) / n_out) if n_out else 0.0
    dark_inside = float(np.count_nonzero(~out & dark) / max(1, out.size - n_out))
    band = (e <= 1) & (e > EDGE_BAND ** 2)
    edge = float(np.median(bright[band])) if band.any() else 0.0
    corners = bool(dark[0, 0] and dark[0, -1] and dark[-1, 0] and dark[-1, -1])
    circular = (corners and outside >= MIN_OUTSIDE and purity >= MIN_PURITY and dark_inside <= MAX_DARK_INSIDE
                and edge >= EDGE_LIT)
    return {"circular": bool(circular), "corners_dark": corners, "radii": [a, b], "outside": round(outside, 3),
            "purity": round(purity, 3), "dark_inside": round(dark_inside, 3), "edge": round(edge, 1)}


def brightest(frames: list) -> np.ndarray:
    """Each thumbnail pixel's brightest value over the frames."""
    return np.max(np.stack([thumb(f) for f in frames]), axis=0)


def circular_image(frames_by_k: dict) -> dict:
    """measure() over one camera's decoded frames ({anchor index: frame}), on up to MAX_FRAMES of them spread
    evenly in time."""
    ks = sorted(frames_by_k)
    if not ks:      # a camera recording at none of the sampled instants has no frames to read
        return {"circular": False, "frames": 0}
    return {**measure(brightest([frames_by_k[ks[i]] for i in spread(len(ks))])), "frames": len(spread(len(ks)))}
