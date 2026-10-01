"""How a file is meant to be shown (prepare/display.py) is followed by every copy of its picture: the reader's sizes,
the model's frames and the board's clips. The frames are checked against ffmpeg, which makes the clips."""
import io
import shutil
import subprocess

import numpy as np
import pytest

from board import clips
from label import frames as mf
from prepare import display, formats

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="no ffmpeg")


def _ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def _first_frame(path):
    import av
    with av.open(str(path)) as c:
        return next(c.decode(c.streams.video[0]))


def _ffmpeg_frame(path):
    from PIL import Image
    png = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "image2pipe", "-vcodec",
                          "png", "-"], capture_output=True, check=True).stdout
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=float)


@pytest.fixture(scope="module")
def lossless(tmp_path_factory):
    p = tmp_path_factory.mktemp("display") / "base.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30", "-frames:v", "2", "-c:v", "libx264", "-pix_fmt", "yuv444p",
        "-crf", "0", str(p))
    return p


@pytest.mark.parametrize("rot", [0, 90, 180, 270, -90])
@pytest.mark.parametrize("flip", [False, True])
def test_every_rotation_and_mirror_is_shown_as_a_player_shows_it(lossless, tmp_path, rot, flip):
    f = tmp_path / "v.mp4"
    _ff("-display_rotation", str(rot), *(["-display_hflip"] if flip else []), "-i", str(lossless), "-c", "copy", str(f))
    g = display.geometry(str(f))
    assert g["mirror"] is flip
    im = np.asarray(mf.upright(_first_frame(f), g).convert("RGB"), dtype=float)
    shown = _ffmpeg_frame(f)
    assert im.shape == shown.shape and np.abs(im - shown).mean() < 0.5
    pr = formats.probe(f)
    assert (pr["width"], pr["height"]) == (shown.shape[1], shown.shape[0])
    assert clips.source_size("ffmpeg", str(f))[:2] == (shown.shape[1], shown.shape[0])
    assert pr.get("mirror", False) is flip and "sar" not in pr


def test_a_file_that_is_not_mirrored_decodes_exactly_as_before(lossless, tmp_path):
    for rot in (0, 90, -90, 180):
        f = tmp_path / f"r{rot}.mp4"
        _ff("-display_rotation", str(rot), "-i", str(lossless), "-c", "copy", str(f))
        fr = _first_frame(f)
        assert mf.upright(fr, display.geometry(str(f))).tobytes() == mf.upright(fr).tobytes()


def test_pixels_that_are_not_square_are_made_square_everywhere(tmp_path):
    f = tmp_path / "anamorphic.mp4"
    _ff("-f", "lavfi", "-i", "testsrc2=size=720x480:rate=30", "-frames:v", "3", "-vf", "setsar=4/3", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(f))
    pr = formats.probe(f)
    assert (pr["width"], pr["height"], pr["sar"]) == (960, 480, "4:3")
    assert mf.upright(_first_frame(f), display.geometry(str(f))).size == (960, 480)
    assert clips.source_size("ffmpeg", str(f)) == (960, 480, True)
    out = tmp_path / "clip.mp4"
    clips.extract_one(str(f), 0.0, 3, out, "ffmpeg", 1)
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,sample_aspect_ratio", "-of", "csv=p=0", str(out)],
                       capture_output=True, text=True, check=True).stdout.strip().split(",")
    assert (int(r[0]), int(r[1])) == (960, 480) and r[2] in ("1:1", "N/A")


def test_a_ratio_that_rounds_away_changes_nothing(tmp_path):
    """Egocentric-100K stores 456x256 HEVC at 512:513: the shown width rounds back to 456, so every size, frame and
    clip command is what it was before pixel shape was read at all. (H.264 cannot store 512:513, so the file is HEVC,
    as the dataset's are.)"""
    from fractions import Fraction
    g = {"stored": (456, 256), "sar": Fraction(512, 513), "rotation": 0.0, "matrix": None, "mirror": False}
    assert not display.needs_resample(g) and display.shown_size(g) == display.square_size(g) == (456, 256)
    assert "-vf" not in clips.video_args(456, 256, True, 1, display.needs_resample(g))
    # a ratio that does not round away is resampled
    g["sar"] = Fraction(4, 3)
    assert display.needs_resample(g) and display.shown_size(g) == (608, 256)
