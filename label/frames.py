"""The frames the model sees: decoded exactly, then composed into by-timestamp grid images.

Decoding. Some datasets pack many episodes back to back into one mp4 per camera (MolmoAct2: an episode's
frames are k = 0..n-1 at from_timestamp + k/30), and nothing inside the video marks where an episode ends,
so a seek that lands one frame late on the last frame silently returns the next episode's first frame. That
cannot happen here:

- A packed file on a fixed frame grid has one exact integer pts per frame (MolmoAct2: 30 fps on a 1/15360
  time base, pts = k * 512), and only the frame whose pts equals it is kept. Files that keep real capture
  times (ABC-130k, RealOmin, some phone video) pass each frame's exact pts instead.
- k is range-checked against the episode's own frame count before any decode.
- Anything else (a missing frame, a pts off the grid, a window outside the file) raises.

No floating-point seek string is ever built.

Grids. One row per camera, one column per instant, each column headed with its time. The header font is
shipped with the repository (fonts/DejaVuSans-Bold.ttf), so the images are identical on every machine.
"""
from __future__ import annotations

import base64
import io
from fractions import Fraction
from pathlib import Path

GRID_FONT = Path(__file__).resolve().parent / "fonts" / "DejaVuSans-Bold.ttf"

FPS = 30


class FrameError(RuntimeError):
    """A frame could not be decoded exactly (off the grid, out of range, or missing)."""


class DamagedFrame(FrameError):
    """A frame the decoder marks as damaged (its data is cut short or corrupt): part of its picture is made up from
    whatever was decoded before it, so it is no footage of that instant. A file cut off in the middle of a frame gives
    one, and decoded after a seek it comes out smeared."""


def frame_pts_step(time_base: Fraction, fps: float = FPS) -> int:
    step = Fraction(1) / Fraction(fps).limit_denominator(1000) / time_base
    if step.denominator != 1:
        raise FrameError(f"time base {time_base} does not put {fps} fps frames on integer pts")
    return int(step)


# LeRobot v3 stores an episode's offset in a packed video in seconds rounded to the microsecond (218.933334 s for
# frame 6568 at 30 fps), about 2e-5 frames off the grid; a genuinely misplaced offset is a sizeable part of a frame
GRID_TOLERANCE_FRAMES = 1e-3


def base_frame(base_s: float, fps: float = FPS) -> int:
    k = round(base_s * fps)
    if abs(base_s * fps - k) > GRID_TOLERANCE_FRAMES:
        raise FrameError(f"episode offset {base_s} s is not on the {fps} fps frame grid")
    return int(k)


def extract_frames(packed: str | Path, base_s: float, n_frames: int, ks: list[int], pts=None,
                   fps: float = FPS, keep=None, tail_ok: bool = False):
    """Decode episode frames ks (indices into the episode, 0..n_frames-1) from a packed mp4.
    Returns {k: PIL.Image (RGB, native size)}. Raises FrameError unless every k is found at its
    exact pts. `pts` (one integer per episode frame) gives each frame's exact pts for files whose
    frames are not on a fixed grid (ABC-130k keeps real capture times); otherwise frame k sits at
    (base_s * fps + k) * step on the fixed frame grid. keep(k, image), when given, is what is kept of each frame
    as soon as it is decoded (episode.py keeps most frames only at their cell widths). tail_ok leaves out the frames
    after the last one the file has (a camera whose file ends a frame before the episode does) instead of raising;
    a frame missing before that, or a file with none of ks, still raises. A frame the decoder marks as damaged raises
    DamagedFrame, and never passes as the camera's picture."""
    import av
    ks = sorted(set(int(k) for k in ks))
    if not ks:
        return {}
    if ks[0] < 0 or ks[-1] >= n_frames:
        raise FrameError(f"requested frames {ks[0]}..{ks[-1]} outside the episode's 0..{n_frames - 1}")
    if pts is not None and len(pts) != n_frames:
        raise FrameError(f"{len(pts)} pts given for {n_frames} frames ({packed})")
    b0 = base_frame(base_s, fps)
    from prepare import display
    geom = display.geometry(str(packed))
    out = {}
    with av.open(str(packed)) as c:
        s = c.streams.video[0]
        # one decoder thread per stream: many episodes decode at once, so the host's CPU use is
        # bounded by the harness's global gate, not multiplied by the codec's own thread pool
        s.codec_context.thread_count = 1
        step = frame_pts_step(s.time_base, fps) if pts is None else None
        dec, last_k, seek_cost = None, None, []
        for k in ks:
            target = int(pts[k]) if pts is not None else (b0 + k) * step
            # Seek to the keyframe at or before the target, then decode forward to the frame whose
            # pts is EXACTLY the target. When the next frame is closer than a seek has been costing
            # (phone video keeps a keyframe only every ~10 s, MolmoAct2 packs every ~2 frames), keep
            # decoding forward instead. Either way the frame kept is the one with the exact pts.
            forward = dec is not None and seek_cost and k - last_k <= sum(seek_cost) / len(seek_cost)
            if not forward:
                c.seek(target, stream=s, backward=True, any_frame=False)
                dec = c.decode(s)
            n = 0
            for fr in dec:
                n += 1
                if fr.pts is None or fr.pts < target:
                    continue
                if fr.pts != target:
                    raise FrameError(f"frame {k}: expected pts {target}, decoder gave {fr.pts} ({packed})")
                if fr.is_corrupt:
                    raise DamagedFrame(f"frame {k}: the decoder marks it damaged ({packed})")
                im = upright(fr, geom)
                out[k] = keep(k, im) if keep is not None else im
                break
            if not forward:
                seek_cost.append(n)
            last_k = k
    missing = [k for k in ks if k not in out]
    if missing and not (tail_ok and out and min(missing) > max(out)):
        raise FrameError(f"frames {missing[:5]} not decoded from {packed}")
    return out


# the picture a player shows for each mirroring display matrix [a, b, c, d] (signs of its 2x2 part), measured
# against ffmpeg, which the board's clips are made with: pixel for pixel equal for every rotation and mirror
def _mirror_ops():
    from PIL import Image
    T = Image.Transpose
    return {(-1, 0, 0, 1): T.FLIP_LEFT_RIGHT, (1, 0, 0, -1): T.FLIP_TOP_BOTTOM,
            (0, -1, -1, 0): T.TRANSVERSE, (0, 1, 1, 0): T.TRANSPOSE}


def upright(fr, geom: dict | None = None):
    """A decoded frame as a PIL image the way players show it (prepare/display.py says how the file is meant to be
    shown). Pixels that are not square are made square first, on the stored frame, as ffmpeg does. A phone stores
    portrait video as landscape frames with a display rotation; the decoder returns the stored frame, so the
    rotation is applied here, as ffmpeg (and so the board's clips) applies it. A display matrix that also mirrors
    (a front camera) is applied as the matrix says, since its rotation angle alone would turn the picture wrong."""
    from PIL import Image
    im = fr.to_image()
    if geom is not None:
        from prepare import display
        sq = display.square_size(geom)
        if sq != im.size and display.needs_resample(geom):
            im = im.resize(sq, Image.LANCZOS)
        if geom.get("mirror"):
            sign = tuple((v > 0) - (v < 0) for v in geom["matrix"])
            op = _mirror_ops().get(sign)
            if op is not None:
                return im.transpose(op)
    turn = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_270}
    rot = int(round(getattr(fr, "rotation", 0) or 0)) % 360        # degrees counterclockwise
    return im.transpose(turn[rot]) if rot in turn else im


def downscaled(im, width: int):
    """im at most `width` wide, aspect kept, height even (box filter); a narrower frame as it is."""
    from PIL import Image
    if im.width <= width:
        return im
    return im.resize((width, int(round(im.height * width / im.width / 2)) * 2), Image.BOX)


class Shrunk:
    """A decoded frame kept only at the widths it can be sent at (downscaled from the full-size frame, which is then
    let go): a long 4K recording no longer holds every sampled frame at full size (about 25 MB each) in memory.
    width and height are the full-size frame's."""

    def __init__(self, im, widths):
        self.width, self.height = im.size
        self.by_width = {int(w): downscaled(im, int(w)) for w in widths}

    def at(self, width: int):
        if width not in self.by_width:
            raise FrameError(f"frame kept at widths {sorted(self.by_width)}, asked for {width}")
        return self.by_width[width]


def to_jpeg(im, width: int | None = None, quality: int = 90) -> bytes:
    """JPEG bytes, downscaled to at most `width` (aspect kept; box filter). A frame narrower than that is kept at its
    own size: enlarging adds no detail, only a blur the model could read as the camera's."""
    if isinstance(im, Shrunk):
        im = im.at(width)
    if width and im.width > width:
        im = downscaled(im, width)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _grid_font(size: int):
    # the basic layout on every machine: Pillow lays text out with libraqm wherever it finds one (Linux wheels, not
    # macOS ones), which kerns the same font differently, so the same grid would differ in bytes between machines
    from PIL import ImageFont
    return ImageFont.truetype(str(GRID_FONT), size, layout_engine=ImageFont.Layout.BASIC)


def compose_grid(cols: list, cam_labels: list[str], quality: int, gutter: int, header: int) -> bytes:
    """One grid image. cols is a list of (t_s, {camera name: jpeg bytes}) for consecutive instants; cam_labels
    are the rows top to bottom. Returns JPEG bytes."""
    from PIL import Image, ImageDraw
    decoded, cw, ch = {}, None, None
    for ci, (_t, cams) in enumerate(cols):
        for ri, cl in enumerate(cam_labels):
            jpg = cams.get(cl)
            if jpg is None:
                continue
            im = Image.open(io.BytesIO(jpg)).convert("RGB")
            if cw is None:
                cw, ch = im.size
            decoded[(ri, ci)] = im
    if cw is None:
        raise RuntimeError("empty grid block")
    # columns as wide as the widest image: a camera narrower than the cell (never enlarged, to_jpeg) sits centred
    cw = max(im.width for im in decoded.values())
    gap = 4
    ncol, nrow = len(cols), len(cam_labels)
    # each row is as tall as its own camera's cells: cameras of another aspect ratio are cut to the same width, so a
    # taller one laid out at the first camera's height would lose its bottom under the next row
    rh = [max((im.height for (r, _c), im in decoded.items() if r == ri), default=ch) for ri in range(nrow)]
    y0 = [header + sum(h + gap for h in rh[:ri]) for ri in range(nrow)]
    g = Image.new("RGB", (gutter + ncol * (cw + gap), header + sum(h + gap for h in rh)), (18, 18, 20))
    d = ImageDraw.Draw(g)
    for ci, (t_rel, _c) in enumerate(cols):
        d.text((gutter + ci * (cw + gap) + 6, 7), f"t={t_rel:.2f}s", fill=(255, 220, 0), font=_grid_font(22))
    for ri, cl in enumerate(cam_labels):
        d.text((6, y0[ri] + rh[ri] // 2 - 10), cl, fill=(230, 230, 235), font=_grid_font(17))
        for ci in range(ncol):
            im = decoded.get((ri, ci))
            if im is not None:
                g.paste(im, (gutter + ci * (cw + gap) + (cw - im.width) // 2, y0[ri]))
    buf = io.BytesIO()
    g.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def build_content(fixed: str, episode: str, timesteps: list, cam_labels: list[str], grid_cols: int,
                  detail: str, quality: int, gutter: int, header: int) -> tuple[list, int, int]:
    """The request's content parts: the shared instructions, the episode's facts, then its instants packed
    into grids (rows = cameras, columns = time). Returns (content, number of grid images, total JPEG bytes)."""
    content: list = [{"type": "text", "text": fixed}]
    if episode:
        content.append({"type": "text", "text": episode})
    n_grids, total_bytes = 0, 0
    for i in range(0, len(timesteps), grid_cols):
        block = timesteps[i:i + grid_cols]
        cols = [(t_rel, {lbl: jpg for lbl, jpg in imgs}) for t_rel, imgs in block]
        grid_jpg = compose_grid(cols, cam_labels, quality, gutter, header)
        total_bytes += len(grid_jpg)
        b64 = base64.b64encode(grid_jpg).decode("ascii")
        content.append({"type": "text",
                        "text": (f"=== grid t={cols[0][0]:.2f}-{cols[-1][0]:.2f}s "
                                 f"| rows {', '.join(cam_labels)} | columns are "
                                 f"time left to right ===")})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail}})
        n_grids += 1
    return content, n_grids, total_bytes
