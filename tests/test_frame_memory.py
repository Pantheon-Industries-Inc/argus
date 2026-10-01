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
