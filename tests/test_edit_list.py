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


# Two 20-frame clips that ffmpeg 7.0.2 wrote with board/clips.py's recipe from an MPEG-4 Part 2 source (20 fps, the
# FastUMI-100K episodes' format), the second with the recipe's -output_ts_offset 0.5 for a camera that starts late.
# Its encoder gave the packets no duration, so the last packet in decode order lasts 0 s, the edit list ends where the
# latest frame starts, and that frame is discarded: the clip check failed a 480-frame episode at 479 (7.1.5 too).
FIXTURES = __import__("pathlib").Path(__file__).with_name("fixtures")
NO_LENGTHS = {"clip_without_frame_lengths.mp4": 0.0, "clip_without_frame_lengths_late.mp4": 0.5}


def _top_atoms(p):
    import struct
    b, out, i = p.read_bytes(), [], 0
    while i + 8 <= len(b):
        size, kind = struct.unpack(">I4s", b[i:i + 8])
        out.append(kind.decode())
        i += size if size >= 8 else len(b)
    return out


@pytest.mark.parametrize("name", list(NO_LENGTHS))
def test_a_clip_whose_frames_have_no_length_plays_to_its_last_frame(tmp_path, name):
    import av
    import json
    clip = tmp_path / name
    shutil.copy(FIXTURES / name, clip)
    with av.open(str(clip)) as c:
        packets = sum(1 for p in c.demux(c.streams.video[0]) if p.size)
    assert (packets, clips.clip_frames(clip)) == (20, 19), "the fixture must drop its last frame, or this checks nothing"
    assert clips.frame_lengths(clip) is True
    assert clips.clip_frames(clip) == 20
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "frame=best_effort_timestamp_time,duration_time", "-of", "json", str(clip)],
                                  capture_output=True, text=True, check=True).stdout)
    t = [float(f["best_effort_timestamp_time"]) for f in j["frames"]]
    start = NO_LENGTHS[name]
    # every frame at its own time, the late camera still late, the last one lasting a frame like the others
    assert t == pytest.approx([start + i / 20 for i in range(20)], abs=1e-4)
    assert float(j["frames"][-1]["duration_time"]) == pytest.approx(0.05, abs=1e-4)
    atoms = _top_atoms(clip)
    assert atoms.index("moov") < atoms.index("mdat"), "the index stays at the front (faststart)"
    assert clips.frame_lengths(clip) is False          # a clip whose frames all have a length is left alone


def test_the_clip_step_gives_an_encoders_frames_their_length(tmp_path, monkeypatch):
    """extract_one on an ffmpeg that writes such a clip: the episode's 20 frames, not 19 and a failed job."""
    real = subprocess.run

    def ffmpeg_702(cmd, *a, **k):
        if cmd[0] == "ffmpeg-7.0.2":
            shutil.copy(FIXTURES / "clip_without_frame_lengths.mp4", cmd[-1])
            return subprocess.CompletedProcess(cmd, 0, b"", b"")
        return real(cmd, *a, **k)
    monkeypatch.setattr(clips, "source_size", lambda ffmpeg, path: (64, 48, False))
    monkeypatch.setattr(clips.subprocess, "run", ffmpeg_702)
    out = tmp_path / "clips" / "episode_000000.mp4"
    clips.extract_one("src.mp4", 0.0, 20, out, "ffmpeg-7.0.2", 1, fps=20.0)
    assert clips.clip_frames(out) == 20
