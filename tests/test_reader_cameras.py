"""The reader keeps every camera that works (prepare/formats.py): a camera that cannot be opened, ends early, has
undecodable frames or a short depth stream is flagged on its episode and the rest of the episode is kept; a take of
many cameras is one episode, and every infrared, thermal or mask video is shown with its episode."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prepare import formats as f


def _mp4(path: Path, n: int, shade: int = 60, w: int = 64, h: int = 48) -> Path:
    """n frames of an mpeg4 video at 30 fps, each frame a little brighter."""
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    c = av.open(str(path), "w", format="mov" if path.suffix in (".mp4", ".mov") else None)
    s = c.add_stream("mpeg4", rate=30)
    s.width, s.height, s.pix_fmt = w, h, "yuv420p"
    for k in range(n):
        fr = av.VideoFrame.from_ndarray(np.full((h, w, 3), (shade + 3 * k) % 256, np.uint8), format="rgb24")
        fr.pts = k
        for pkt in s.encode(fr):
            c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()
    return path


def _ctx(out: Path, rep: dict, i: int = 0) -> dict:
    return json.loads((out / rep["episodes"][i]["episode_id"] / "context.json").read_text())


# ---------------------------------------------------------------- the container of a video named for another one

def test_a_matroska_video_named_mp4_is_read_by_its_own_container(tmp_path):
    """A valid Matroska video named .mp4 had been refused, since the extension forced the demuxer. The container is
    read from the file's own first bytes when the named one fails, and a file that is no video container (a concat
    script, which could make ffmpeg read other files) is still refused."""
    import pytest
    mkv = _mp4(tmp_path / "real.mkv", 12)
    named = tmp_path / "cam.mp4"
    named.write_bytes(mkv.read_bytes())
    assert len(f.probe(named)["pts"]) == 12
    script = tmp_path / "list.mp4"
    script.write_text("ffconcat version 1.0\nfile /etc/hosts\n")
    with pytest.raises(Exception):
        f.probe(script)
