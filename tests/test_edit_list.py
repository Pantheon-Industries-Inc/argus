"""A video trimmed without re-encoding (ffmpeg -ss X -c copy, and editors that cut the same way) keeps the frames
before the cut as packets its edit list discards. They are not frames: the reader counts only the frames the file
shows, the model's frames decode, and the board's clip has the same frames."""
import shutil
import subprocess

import pytest

from board import clips
from label import episode
from prepare import formats

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="no ffmpeg")


def test_a_copy_trimmed_video_is_read_as_the_frames_it_shows(tmp_path):
    src, videos = tmp_path / "src.mp4", tmp_path / "videos"
    videos.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30", "-t", "10", "-c:v",
                    "libx264", "-g", "30", "-pix_fmt", "yuv420p", str(src)], check=True)
    trim = videos / "trimmed.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-ss", "1.5", "-i", str(src), "-c", "copy", str(trim)], check=True)
    import av
    with av.open(str(trim)) as c:
        shown = sum(1 for _ in c.decode(c.streams.video[0]))
    with av.open(str(trim)) as c:
        dropped = sum(1 for p in c.demux(c.streams.video[0]) if p.size and p.is_discard)
    assert dropped > 0, "the file must carry an edit list that discards frames, or this test checks nothing"
    assert len(formats.probe(trim)["pts"]) == shown
    out = tmp_path / "eps"
    subprocess.run([__import__("sys").executable, "-m", "prepare", "videos", "prepare", "--root", str(videos), "--rig",
                    "handheld_gripper", "--out", str(out)], check=True, capture_output=True)
    ep = next(out.glob("episode_*"))
    req = episode.build_request(ep)            # raised "expected pts ..., decoder gave 0" on frame 0 before
    assert req["n_images"] > 0
    clip = tmp_path / "clip.mp4"
    clips.extract_one(str(trim), 0.0, shown, clip, "ffmpeg", 1)
    assert clips.clip_frames(clip) == shown
