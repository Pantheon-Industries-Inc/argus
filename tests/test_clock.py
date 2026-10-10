"""Every clip plays on the episode's clock: its first frame at 0 (or its camera's own later start), whatever else the
file holds; a camera that started first never plays early; goal frames and the footage download take the clip's own
frame times at any rate."""
import io
import json
import shutil
import subprocess

import numpy as np
import pytest

from board import clips

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="no ffmpeg")


def _ff(*a):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *a], check=True)


def _pts(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of",
                          "csv=p=0", str(p)], capture_output=True, text=True, check=True).stdout
    return sorted(float(x.strip(",")) for x in out.split() if x.strip(","))


def test_a_clip_starts_at_its_first_video_frame_when_audio_comes_first(tmp_path):
    src = tmp_path / "audiofirst.mp4"
    _ff("-f", "lavfi", "-i", "sine=frequency=440:duration=5", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30",
        "-t", "4", "-filter_complex", "[1:v]setpts=PTS+0.5/TB[v]", "-map", "[v]", "-map", "0:a", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-c:a", "aac", str(src))
    n = len(_pts(src))
    out = tmp_path / "clip.mp4"
    clips.extract_one(str(src), 0.0, n, out, "ffmpeg", 1)
    p = _pts(out)
    assert len(p) == n and abs(p[0]) < 1e-6 and abs(p[1] - 1 / 30) < 2e-3


def test_a_packed_episode_starts_at_0_not_half_a_frame_late(tmp_path):
    src = tmp_path / "packed.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30", "-t", "6", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(src))
    out = tmp_path / "clip.mp4"
    clips.extract_one(str(src), 2.0, 60, out, "ffmpeg", 1)
    p = _pts(out)
    assert len(p) == 60 and abs(p[0]) < 1e-6


def test_a_camera_that_started_first_never_plays_early(tmp_path):
    ep = tmp_path / "episode_x"
    ep.mkdir()
    main = np.arange(30) / 30 + 1.0
    early = np.arange(36) / 30 + 1.0 - 0.2          # started 0.2 s (6 frames) before the main camera
    late = np.arange(24) / 30 + 1.0 + 0.2
    np.savez(ep / "times.npz", exo=main, left=early, right=late)
    (ep / "context.json").write_text(json.dumps({"clock_zero_s": 1.0}))
    sources = {"exo": {"packed": "a"}, "left": {"packed": "b"}, "right": {"packed": "c"}}
    off = clips.start_offsets(ep, sources, 30.0)
    assert off["left"] == (0.0, 6)
    assert abs(off["right"][0] - 0.2) < 1e-9 and off["right"][1] == 0
    # 46 ms early, ABC-130k's largest: the frame more than half a frame early is dropped, the next is within it
    np.savez(ep / "times.npz", exo=main, left=np.arange(31) / 30 + 1.0 - 0.046)
    assert clips.start_offsets(ep, {"exo": {}, "left": {}}, 30.0) == {"left": (0.0, 1)}
    src = tmp_path / "early.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30", "-frames:v", "36", "-c:v", "libx264", "-pix_fmt",
        "yuv420p", str(src))
    out = tmp_path / "clip.mp4"
    clips.extract_one(str(src), 0.0, 36, out, "ffmpeg", 1, skip=6)
    p = _pts(out)
    assert len(p) == 30 and abs(p[0]) < 1e-6


def test_a_goal_frame_at_15_fps_is_the_nearest_frame(tmp_path):
    from board import serve
    from PIL import Image
    src = tmp_path / "f15.mp4"
    # every frame a different brightness, so the frame cut is matched to ffmpeg's own decode of the clip
    _ff("-f", "lavfi", "-i", "color=c=black:s=64x64:r=15:d=2", "-vf", "geq=lum='N*12':cb=128:cr=128", "-c:v", "libx264",
        "-qp", "0", "-pix_fmt", "yuv444p", str(src))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    decoded = np.frombuffer(raw, np.uint8).reshape(-1, 64, 64).mean(axis=(1, 2))
    serve.FFMPEG = shutil.which("ffmpeg")
    # frames at k/15 s: 1.0 is frame 15, 1.03 is nearest 15, 1.05 nearest 16 (a 30 fps grid made it 16, 16, 16)
    for t, want in ((1.0, 15), (1.03, 15), (1.05, 16), (0.0, 0), (1.99, 29)):
        jpg = serve.extract_frame(src, t, 64)
        lum = np.asarray(Image.open(io.BytesIO(jpg)).convert("L"), dtype=float).mean()
        assert int(np.argmin(np.abs(decoded - lum))) == want, (t, want, lum)


def test_the_download_runs_to_the_last_frame_of_a_variable_rate_clip(tmp_path):
    from board import serve
    a, b, src = tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "vfr.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=size=160x90:rate=30", "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(a))
    _ff("-f", "lavfi", "-i", "testsrc2=size=160x90:rate=15", "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(b))
    (tmp_path / "list.txt").write_text(f"file '{a}'\nfile '{b}'\n")
    _ff("-f", "concat", "-safe", "0", "-i", str(tmp_path / "list.txt"), "-c", "copy", str(src))
    serve.FFMPEG = shutil.which("ffmpeg")
    last = _pts(src)[-1]
    assert serve._probe(src)[2] >= last + 1 / 15 - 1e-3


def test_the_static_boards_re_encode_keeps_a_clips_start(tmp_path):
    from board import static
    src, dst = tmp_path / "late.mp4", tmp_path / "out" / "late.mp4"
    # full-range pixels are re-encoded, not copied; the clip starts 1 s in, as a camera that started late does
    _ff("-f", "lavfi", "-i", "testsrc2=size=160x90:rate=30", "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuvj420p",
        "-output_ts_offset", "1.0", str(src))
    r = static.transcode(src, dst, 1)
    assert r["mode"] == "encode"
    assert abs(_pts(dst)[0] - _pts(src)[0]) < 2e-3 and _pts(src)[0] > 0.9
