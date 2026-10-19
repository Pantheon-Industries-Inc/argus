"""A native tool whose work grows with its input (a clip encode, a decode of every frame) is bound by its progress,
never by a fixed time (prepare/isolation.py run_while_progressing): a long recording takes as long as it takes, and a
tool that stops making progress is still stopped. A fixed 600 s bound on the clip encode had dropped a 319 s 1080p
HEVC recording, whose single-threaded encode takes about 14 minutes."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from prepare.isolation import run_while_progressing


def child(code):
    return [sys.executable, '-c', code]


def test_a_tool_that_keeps_writing_runs_past_the_stall_bound():
    started = time.monotonic()
    done = run_while_progressing(child('import sys, time\nfor i in range(12):\n    sys.stdout.write("x"); '
                                       'sys.stdout.flush(); time.sleep(0.25)'), stall_s=1, check=True, text=True)
    assert done.stdout == 'x' * 12 and time.monotonic() - started > 2.5


def test_a_tool_that_grows_its_output_file_runs_past_the_stall_bound(tmp_path):
    out = tmp_path / 'clip.mp4'
    code = (f'import time\nwith open({str(out)!r}, "wb") as f:\n    for i in range(12):\n'
            '        f.write(b"x" * 1000); f.flush(); time.sleep(0.25)')
    assert run_while_progressing(child(code), watch=[out], stall_s=1).returncode == 0
    assert out.stat().st_size == 12000


def test_a_silent_tool_is_stopped_and_does_not_outlive_the_call(tmp_path):
    marker = tmp_path / 'still_running'
    code = f'import time\ntime.sleep(3)\nopen({str(marker)!r}, "w").write("x")'
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_while_progressing(child(code), stall_s=1)
    assert time.monotonic() - started < 2.9
    time.sleep(3.2)
    assert not marker.exists()


def test_work_before_the_first_output_has_its_lead_allowance():
    code = 'import sys, time\ntime.sleep(2)\nsys.stdout.write("x"); sys.stdout.flush()'
    assert run_while_progressing(child(code), stall_s=1, lead_s=3, text=True).stdout == 'x'
    with pytest.raises(subprocess.TimeoutExpired):
        run_while_progressing(child(code), stall_s=1)


def test_a_failing_tool_raises_with_its_output_when_checked():
    with pytest.raises(subprocess.CalledProcessError) as raised:
        run_while_progressing(child('import sys\nsys.stderr.write("bad bytes")\nsys.exit(3)'), check=True, text=True)
    assert raised.value.returncode == 3 and raised.value.stderr == 'bad bytes'


def test_the_clip_cut_is_bound_by_progress_with_a_lead_for_the_frames_before_it(tmp_path, monkeypatch):
    """extract_one passes its own temporary clip as the progress to watch, no fixed time, and a lead for every frame
    it may decode before the first one kept (from the keyframe before the seek, and those select skips)."""
    from board import clips
    seen = {}
    monkeypatch.setattr(clips, 'source_size', lambda *args: (16, 16, False))

    def encoder(cmd, **kwargs):
        seen.update(kwargs, out=cmd[-1])
        raise subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(clips, 'run_while_progressing', encoder)
    with pytest.raises(subprocess.CalledProcessError):
        clips.extract_one('source.mp4', 600.0, 300, tmp_path / 'out.mp4', 'ffmpeg', 1, fps=30.0, skip=5)
    assert seen['watch'] == [Path(seen['out'])] and 'timeout' not in seen
    # a file whose index cannot be read is allowed every frame before the seek
    assert seen['lead_s'] == pytest.approx(((600.0 - 0.5 / 30) * 30.0 + 5) / clips.MIN_DECODE_FPS)


def test_the_lead_is_measured_from_the_keyframe_the_seek_decodes_from(tmp_path):
    import av
    import numpy as np
    from board import clips
    path = tmp_path / 'gop.mp4'
    with av.open(str(path), 'w') as out:
        stream = out.add_stream('mpeg4', rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        stream.codec_context.gop_size = 60
        stream.options = {'sc_threshold': '1000000000'}
        for i in range(300):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i % 255, np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    # keyframes every 2 s: a seek to 5.5 s decodes from the one at 4 s
    assert clips.decode_lead_s(str(path), 5.5) == pytest.approx(1.5, abs=0.05)
    assert clips.decode_lead_s(str(path), 0.0) == 0.0
    assert clips.decode_lead_s(str(tmp_path / 'missing.mp4'), 5.5) == 5.5
