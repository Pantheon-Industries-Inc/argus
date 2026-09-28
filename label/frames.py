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
                   fps: float = FPS):
    """Decode episode frames ks (indices into the episode, 0..n_frames-1) from a packed mp4.
    Returns {k: PIL.Image (RGB, native size)}. Raises FrameError unless every k is found at its
    exact pts. `pts` (one integer per episode frame) gives each frame's exact pts for files whose
    frames are not on a fixed grid (ABC-130k keeps real capture times); otherwise frame k sits at
    (base_s * fps + k) * step on the fixed frame grid."""
    import av
    ks = sorted(set(int(k) for k in ks))
    if not ks:
        return {}
    if ks[0] < 0 or ks[-1] >= n_frames:
        raise FrameError(f"requested frames {ks[0]}..{ks[-1]} outside the episode's 0..{n_frames - 1}")
    if pts is not None and len(pts) != n_frames:
        raise FrameError(f"{len(pts)} pts given for {n_frames} frames ({packed})")
    b0 = base_frame(base_s, fps)
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
                out[k] = upright(fr)
                break
            if not forward:
                seek_cost.append(n)
            last_k = k
    missing = [k for k in ks if k not in out]
    if missing:
        raise FrameError(f"frames {missing[:5]} not decoded from {packed}")
    return out


def upright(fr):
    """A decoded frame as a PIL image the way players show it. A phone stores portrait video as landscape
    frames with a display rotation; the decoder returns the stored frame, so the rotation is applied here,
    as ffmpeg (and so the board's clips) applies it."""
    from PIL import Image
    im = fr.to_image()
    turn = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_270}
    rot = int(round(getattr(fr, "rotation", 0) or 0)) % 360        # degrees counterclockwise
    return im.transpose(turn[rot]) if rot in turn else im


def to_jpeg(im, width: int | None = None, quality: int = 90) -> bytes:
    """JPEG bytes, optionally downscaled to `width` (aspect kept; box filter)."""
    from PIL import Image
    if width and im.width != width:
        h = int(round(im.height * width / im.width / 2)) * 2
        im = im.resize((width, h), Image.BOX)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _grid_font(size: int):
    from PIL import ImageFont
    return ImageFont.truetype(str(GRID_FONT), size)


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
    gap = 4
    ncol, nrow = len(cols), len(cam_labels)
    g = Image.new("RGB", (gutter + ncol * (cw + gap), header + nrow * (ch + gap)), (18, 18, 20))
    d = ImageDraw.Draw(g)
    for ci, (t_rel, _c) in enumerate(cols):
        d.text((gutter + ci * (cw + gap) + 6, 7), f"t={t_rel:.2f}s", fill=(255, 220, 0), font=_grid_font(22))
    for ri, cl in enumerate(cam_labels):
        d.text((6, header + ri * (ch + gap) + ch // 2 - 10), cl, fill=(230, 230, 235), font=_grid_font(17))
        for ci in range(ncol):
            im = decoded.get((ri, ci))
            if im is not None:
                g.paste(im, (gutter + ci * (cw + gap), header + ri * (ch + gap)))
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
