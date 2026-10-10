"""Frames kept only at their cell widths (label/frames.py Shrunk) give the same JPEG bytes as full-size frames
downscaled at send time, and detail instants stay full size."""
import shutil
import subprocess
import sys

import pytest

from label import episode
from label import frames as mf

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="no ffmpeg")


def test_cell_width_copies_are_the_bytes_full_frames_would_give(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30", "-t", "6", "-c:v",
                    "libx264", "-pix_fmt", "yuv420p", str(videos / "clip.mp4")], check=True)
    out = tmp_path / "eps"
    subprocess.run([sys.executable, "-m", "prepare", "videos", "prepare", "--root", str(videos), "--rig", "teleop_arms",
                    "--out", str(out)], check=True, capture_output=True)
    ep = episode.load(next(out.glob("episode_*")))
    pl = episode.plan(ep)
    widths = [w for w in episode.CELL_W_STEPS if w <= 448]
    detail = {pl["ks"][0], pl["ks"][-1]}
    full = episode.frames(ep, pl)
    kept = episode.frames(ep, pl, widths=widths, detail_ks=detail)
    for v in full:
        for k in pl["ks"]:
            assert isinstance(kept[v][k], mf.Shrunk) is (k not in detail)
            for w in widths:
                assert mf.to_jpeg(kept[v][k], w) == mf.to_jpeg(full[v][k], w)


def test_a_large_float32_signal_is_read_in_its_own_precision_and_never_copied_whole(tmp_path):
    """A recording's large float32 signal (a tactile skin) is read as stored: building the request peaks within about
    three times the signal's size over the same episode without it, never at a float64 copy of the signal and its
    temporaries (about 13 times before), and its lines in the prompt still say what it does."""
    import json
    import tracemalloc

    import numpy as np

    from test_clip_cameras import _upload
    rep, eps, ep = _upload(tmp_path, {"top": 120})
    n, d = 120, 200_000
    rng = np.random.default_rng(0)
    skin = (3000 + rng.normal(0, 3, (n, d))).astype(np.float32)
    skin[40:70, :60] -= 900
    skin[90:100] = np.nan
    force = np.zeros((n, 1), np.float32)
    force[40:70] = 4.0
    np.savez(ep / "signals.npz", s0=skin, s1=force)
    ctx = json.loads((ep / "context.json").read_text())
    small = [{"name": "fingertip force", "key": "s1", "dims": 1, "names": ["fz"]}]
    big = [{"name": "skin pressure", "key": "s0", "dims": d, "shape": [500, 400]}]

    def peak(signals):
        (ep / "context.json").write_text(json.dumps({**ctx, "signals": signals}))
        tracemalloc.start()
        try:
            req = episode.build_request(ep)
            return tracemalloc.get_traced_memory()[1], req
        finally:
            tracemalloc.stop()
    base, _ = peak(small)
    with_skin, req = peak(small + big)
    assert with_skin - base <= 3 * skin.nbytes, (with_skin - base) / skin.nbytes
    assert "skin pressure (500 x 400 values)" in req["prompt"]
