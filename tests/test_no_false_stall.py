"""A healthy read of a long recording is never taken for a hang (prepare/isolation.py).

The reader's child is stopped when it shows no progress for its stall bound (300s), and the episode is read again with
its cameras alone, which leaves its recorded clocks, decoding check, sensors and notes unread. On the UMI 100h takes
(three 1280x1280 HEVC cameras at 57 Mbit/s) the decoding check of one camera of a take of 14 minutes or more decoded
every picture for over 300s without a beat, so 25 healthy jobs were read with their cameras alone. Every loop over a
video's packets now beats once per packet, a tool run inside the child beats with its own progress, and a step that
cannot beat is allowed a quiet time from the size of its work, while a child stuck inside one unit is still stopped.
Small stall bounds and slowed packets stand in for long footage.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from prepare.isolation import ReaderStopped, run_isolated


@pytest.fixture(autouse=True)
def _in_a_child_process(_uploads_read_in_this_process, monkeypatch):
    monkeypatch.setenv('ARGUS_READER_ISOLATION', '1')        # these tests are of the child process itself


@pytest.fixture
def clip(tmp_path):
    from test_reader_tables import _clip
    path = tmp_path / 'camera.mp4'
    _clip(path, 40)
    return path


def slow_packets(seconds, stuck_at=None):
    """Every packet of a video takes seconds to arrive, as a long HEVC file's packets do in total; with stuck_at, the
    packet of that number never arrives (a decoder that hangs)."""
    from label import presented
    real, count = presented.presents, [0]

    def slow(packet):
        count[0] += 1
        time.sleep(3600 if count[0] == stuck_at else seconds)
        return real(packet)
    presented.presents = slow


def decodes_slowly(path, seconds, stuck_at=None):
    slow_packets(seconds, stuck_at)
    from prepare import formats
    return sorted(formats.decoded_damage(Path(path)))


def probes_slowly(path, seconds):
    slow_packets(seconds)
    from prepare import formats
    return len(formats.probe(Path(path))['pts'])


def quiet_for(seconds, allowance):
    from prepare.isolation import allow_quiet
    with allow_quiet(allowance):
        time.sleep(seconds)
    return 'done'


def runs_a_tool(seconds):
    from prepare.isolation import run_while_progressing
    code = f'import sys, time\nfor i in range({int(seconds * 4)}):\n    sys.stdout.write("x"); sys.stdout.flush(); ' \
           'time.sleep(0.25)'
    return run_while_progressing([sys.executable, '-c', code], stall_s=1, text=True).stdout


def test_a_decoding_check_longer_than_the_stall_bound_is_not_stopped(clip):
    # 40 packets at 0.1s each: 4s of decoding that writes nothing, twice the stall bound
    started = time.monotonic()
    assert run_isolated(decodes_slowly, str(clip), 0.1, stall_s=2) == []
    assert time.monotonic() - started > 4


def test_reading_every_packet_of_a_long_video_is_not_stopped(clip):
    assert run_isolated(probes_slowly, str(clip), 0.1, stall_s=2) == 40


def test_a_decoder_stuck_inside_one_packet_is_still_stopped(clip):
    started = time.monotonic()
    with pytest.raises(ReaderStopped, match='stopped responding'):
        run_isolated(decodes_slowly, str(clip), 0.05, 5, stall_s=2)
    assert time.monotonic() - started < 30


def test_a_tool_that_keeps_working_keeps_its_reader_alive():
    assert run_isolated(runs_a_tool, 4, stall_s=2) == 'x' * 16


def test_a_step_that_cannot_beat_has_the_quiet_time_its_size_allows():
    from prepare.isolation import QUIET_BYTES_PER_S, quiet_allowance
    assert quiet_allowance(30 * QUIET_BYTES_PER_S) == pytest.approx(30)
    assert run_isolated(quiet_for, 4, quiet_allowance(8 * QUIET_BYTES_PER_S), stall_s=1.5) == 'done'
    started = time.monotonic()
    with pytest.raises(ReaderStopped, match=r'no progress for 3\.5s'):
        run_isolated(quiet_for, 3600, 2, stall_s=1.5)
    assert time.monotonic() - started < 30


def test_a_read_again_with_its_cameras_alone_names_everything_it_left_unread(tmp_path, monkeypatch):
    """A folder of three videos with a timestamp sidecar each, shaped like a UMI take. Its first read is stopped as the
    300s bound stopped the long takes; the issue and the board name every part the second read left unread, the
    sidecars among them, and the cameras are still placed against each other by the sidecars."""
    from test_reader_tables import _clip
    from board import build as board_build
    from prepare import formats, isolation
    take = tmp_path / 'upload' / 'fob_take'
    offsets = {'exo': 0.0, 'left_wrist': 0.4, 'right_wrist': -0.25}
    for name, offset in offsets.items():
        _clip(take / f'{name}.mp4', 40)
        np.save(take / f'{name}_timestamp.npy', offset + np.arange(40) / 30)
    real = isolation.run_isolated

    def first_read_stops(fn, *args, mode=None, **kwargs):
        if mode is None:
            raise ReaderStopped('stopped responding (no progress for 300s) and was stopped')
        return real(fn, *args, mode=mode, **kwargs)
    monkeypatch.setattr(isolation, 'run_isolated', first_read_stops)
    report = formats.convert(tmp_path / 'upload', 'handheld_gripper', tmp_path / 'out', 'take', 900)
    ep = tmp_path / 'out' / report['episodes'][0]['episode_id']
    ctx = json.loads((ep / 'context.json').read_text())
    sidecars = "the videos' recorded clocks (exo_timestamp.npy, left_wrist_timestamp.npy and right_wrist_timestamp.npy)"
    left = ctx['read_cameras_only']['left_unread']
    assert sidecars in left and "the episode's note files" in left
    assert "the episode's sensor files and recorded state" in left
    assert {f'the decoding check of the camera {name}' for name in offsets} <= set(left)
    assert not any('depth' in part or 'unshown' in part for part in left)       # parts the take does not have
    assert ctx['read_cameras_only']['why'] == 'its reader stopped responding (no progress for 300s) and was stopped'
    issue = [i['what'] for i in ctx['reader_issues'] if i['kind'] == 'part_unreadable']
    assert len(issue) == 1 and sidecars in issue[0] and 'the decoding check of the camera exo' in issue[0]
    assert 'recorded_container_times' not in ctx
    with np.load(ep / 'times.npz') as times:
        starts = {view: round(float(times[view][0]), 3) for view in ('exo', 'left', 'right')}
    assert starts == {'exo': 0.0, 'left': 0.4, 'right': -0.25}
    notes = board_build.reader_notes(ctx)
    assert notes['cameras_only'] == {'why': ctx['read_cameras_only']['why'], 'left_unread': left}
