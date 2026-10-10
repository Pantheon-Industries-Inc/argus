"""An encoded clip's frame lengths are established before its capture times are assigned."""
import shutil
import subprocess
from pathlib import Path

import av
import numpy as np
import pytest

from board import clips

pytestmark = pytest.mark.skipif(not (shutil.which('ffmpeg') and shutil.which('ffprobe')), reason='no ffmpeg')


@pytest.mark.parametrize('source_frames,wanted', [(1, 1), (20, 20), (12, 20)])
def test_a_readable_camera_keeps_every_encoded_frame_when_timed(tmp_path, source_frames, wanted):
    source, output = tmp_path/'source.mp4', tmp_path/'clip.mp4'
    subprocess.run([clips.find_ffmpeg(), '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=64x48:rate=20',
                    '-frames:v', str(source_frames), '-c:v', 'mpeg4', '-threads', '1', str(source)], check=True)
    with av.open(str(source)) as c:
        c.streams.video[0].codec_context.thread_count = 1
        assert sum(1 for _ in c.decode(video=0)) == source_frames
    result = clips.extract_one(str(source), 0, wanted, output, clips.find_ffmpeg(), 1,
                               fps=20, times=np.arange(wanted)/20)
    assert result == (None if source_frames == wanted else
                      {'clip_frames': source_frames, 'episode_frames': wanted})
    assert clips.clip_frames(output) == source_frames
    with av.open(str(output)) as c:
        c.streams.video[0].codec_context.thread_count = 1
        assert sum(1 for _ in c.decode(video=0)) == source_frames
        assert c.metadata.get('comment') == clips.HALFWAY_TAG


@pytest.mark.parametrize('name,start', [('clip_without_frame_lengths.mp4', 0),
                                       ('clip_without_frame_lengths_late.mp4', 0.5)])
def test_capture_timing_keeps_the_last_native_encoded_frame_without_a_duration(tmp_path, monkeypatch, name, start):
    source = Path(__file__).with_name('fixtures')/name
    with av.open(str(source)) as c:
        packets = [p for p in c.demux(video=0) if p.size]
    assert len(packets) == 20 and sum(not p.is_discard for p in packets) == 19
    assert packets[-1].duration == 0
    with av.open(str(source), options={'ignore_editlist': '1'}) as c:
        c.streams.video[0].codec_context.thread_count = 1
        original_pixels = [f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
    assert len(original_pixels) == 20
    real_run = subprocess.run

    def frozen_encoder(cmd, *args, **kwargs):
        # Only the external encoder is bound to its retained native output. Counts and timing use actual packets.
        if cmd[0] == 'fixture-encoder':
            shutil.copyfile(source, cmd[-1])
            return subprocess.CompletedProcess(cmd, 0, b'', b'')
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(clips.subprocess, 'run', frozen_encoder)
    output = tmp_path/'clip.mp4'
    assert clips.extract_one(str(source), 0, 20, output, 'fixture-encoder', 1,
                             fps=20, offset_s=start, times=start+np.arange(20)/20) is None
    assert clips.clip_frames(output) == 20
    with av.open(str(output)) as c:
        c.streams.video[0].codec_context.thread_count = 1
        pixels = [f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
        assert len(pixels) == 20
        assert all(np.array_equal(a, b) for a, b in zip(original_pixels, pixels))
        assert c.metadata.get('comment') == clips.HALFWAY_TAG
