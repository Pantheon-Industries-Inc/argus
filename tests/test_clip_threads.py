"""board clips shares its thread budget (--jobs x --clip-threads) among the clips it cuts, so a single long recording
is encoded with every thread the step was given (board/clips.py clip_threads). A 319s 1080p HEVC head camera took
about 14 minutes on the one thread a clip had, while the step's other threads sat idle."""
from __future__ import annotations

import av
import numpy as np

from board import clips


def test_fewer_clips_than_jobs_share_the_budget():
    assert clips.clip_threads(1, 2, 1) == 2
    assert clips.clip_threads(1, 6, 2) == 12
    assert clips.clip_threads(3, 6, 2) == 4
    assert clips.clip_threads(5, 6, 2) == 2          # 12 // 5 rounds down to what each already has
    assert clips.clip_threads(6, 6, 2) == 2
    assert clips.clip_threads(40, 2, 1) == 1
    assert clips.clip_threads(0, 2, 1) == 1


def _movie(path, n=45):
    with av.open(str(path), 'w') as out:
        stream = out.add_stream('mpeg4', rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        for i in range(n):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), (i * 5) % 255, np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


def _times(path):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return [float(f.pts * s.time_base) for f in c.decode(s)]


def test_more_threads_cut_the_same_frames_at_the_same_times(tmp_path):
    src = _movie(tmp_path / 'src.mp4')
    ffmpeg = clips.find_ffmpeg()
    one, four = tmp_path / 'one' / 'clip.mp4', tmp_path / 'four' / 'clip.mp4'
    assert clips.extract_one(str(src), 0.0, 45, one, ffmpeg, 1) is None
    assert clips.extract_one(str(src), 0.0, 45, four, ffmpeg, 4) is None
    assert clips.clip_frames(one) == clips.clip_frames(four) == 45
    assert _times(one) == _times(four)
