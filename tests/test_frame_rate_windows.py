"""A window or a lag given in seconds is counted in frames at the episode's frame rate. A window longer than the frames
it is read over gave a series of the window's length, and a per-frame mask applied to it stopped the whole episode
(ABC-130k episode_0070d777: a 1 s window at a stated 976562.5 fps, 3077 frames). Every windowed series has one value
per frame (label/signals.py window_sum), and a stated rate the frames' own capture times contradict is not used
(prepare/camera_clock.py supported_rate, label/episode.py ep_fps), and is named in the episode's checks."""
from __future__ import annotations

import json

import numpy as np

from label import episode as me
from label import signals as sg
from prepare import abc130k
from prepare import formats as f

STATED = 976562.5     # what ABC-130k's adapter wrote for episode_0070d777: 1e9 / its 1024 ns median step


def _clustered_clock(n: int, fps: float = 30.0) -> np.ndarray:
    """n capture times (s) of a camera at fps that stamps about half its frames 1 us after the frame before, as
    ABC-130k's top camera does: its median step is 1 us, its frames are fps over their span."""
    steps = np.where(np.arange(n - 1) % 2 == 0, 1e-6, 2.0 / fps - 1e-6)
    steps[::11] = 1e-6
    t = np.concatenate([[0.0], np.cumsum(steps)])
    return t * ((n - 1) / fps) / t[-1]


def _follows(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A commanded joint that moves smoothly, and the measured joint that follows it 3 frames later."""
    rng = np.random.default_rng(seed)
    k = np.arange(n) / 30.0
    cmd = sum(rng.uniform(0.2, 0.8) * np.sin(2 * np.pi * k / rng.uniform(4, 9) + rng.uniform(0, 6)) for _ in range(3))
    arm = np.concatenate([np.full(3, cmd[0]), cmd[:-3]]) + rng.normal(0, 0.002, n)
    return cmd, arm


def test_a_window_sum_has_one_value_per_frame_and_is_convolve_up_to_the_frames():
    a = np.random.default_rng(1).normal(size=37)
    for win in (1, 2, 5, 36, 37, 38, 60, 75, 976562):
        got = sg.window_sum(a, win)
        assert got.shape == a.shape, win
        # the frames from win // 2 before each frame to (win - 1) // 2 after it, as np.convolve(mode="same") centres
        want = np.array([a[max(0, i - win // 2):i + (win - 1) // 2 + 1].sum() for i in range(len(a))])
        assert np.allclose(got, want), win
        if win <= len(a):
            assert np.array_equal(got, np.convolve(a, np.ones(win), mode="same")), win


def test_a_pair_read_over_fewer_frames_than_one_window_is_read_and_never_stops():
    """Instants 36 frames apart at 60 fps: the 1 s window (60 frames) is longer than the frames between the first and
    last instant, which made the windowed speed 60 values long against a 36-frame mask."""
    cmd, arm = _follows(40)
    ks = list(range(0, 40, 5))
    assert sg.apart_instants(cmd, arm, ks, 0, 60.0) is not None


def test_the_stated_rate_is_kept_unless_the_frames_own_times_contradict_it():
    from prepare.camera_clock import supported_rate
    n = 3077
    clustered = _clustered_clock(n)
    assert np.median(np.diff(clustered)) < 2e-6
    used, note = supported_rate(STATED, clustered)
    assert abs(used - 30.0) < 0.01 and note["stated_fps"] == STATED and note["used_fps"] == used
    assert note["frames"] == n and "976562" in note["what"]
    # a camera that dropped half its frames, or paused for as long as it recorded, keeps its own stated rate
    dropped = np.sort(np.random.default_rng(2).choice(np.arange(2 * n) / 30.0, n, replace=False))
    paused = np.r_[np.arange(n // 2), np.arange(n - n // 2) + 2 * n] / 30.0
    for t in (np.arange(n) / 30.0, dropped, paused):
        assert supported_rate(30.0, t) == (30.0, None)
    # no stated rate, or no span to measure, changes nothing
    assert supported_rate(None, clustered) == (None, None)
    assert supported_rate(30.0, np.zeros(5)) == (30.0, None)


def test_a_clock_whose_median_step_is_a_microsecond_is_measured_over_its_span():
    clustered = _clustered_clock(3077)
    assert abs(abc130k.measured_fps(np.round(clustered * 1e9)) - 30.0) < 0.01
    assert abs(f.measured_fps(clustered) - 30.0) < 0.01
    assert abc130k.measured_fps(np.arange(300) * 1e9 / 60) == 60.0
    assert f.measured_fps(np.arange(300) / 30.0) == 30.0


def _prepared(tmp_path, n: int = 1800):
    """A prepared episode as ABC-130k's adapter wrote episode_0070d777: real capture times on a clustered clock, a
    stated fps of 976562.5 and two signals that track each other."""
    d = tmp_path / "episode_clustered"
    d.mkdir()
    cmd, arm = _follows(n)
    np.savez(d / "signals.npz", s0=cmd[:, None].astype(np.float32), s1=arm[:, None].astype(np.float32))
    np.savez(d / "times.npz", exo=_clustered_clock(n))
    (d / "sources.json").write_text(json.dumps({"exo": {"n_frames": n}}))
    (d / "context.json").write_text(json.dumps({
        "dataset": "fixture", "profile": "teleop_arms", "state_kind": "none", "fps": STATED, "n_state_frames": n,
        "real_times": "times.npz", "cameras": {"exo": {"width": 64, "height": 36}},
        "signals": [{"name": "left command", "key": "s0", "dims": 1}, {"name": "left joint", "key": "s1", "dims": 1}]}))
    return d


def test_an_episode_whose_stated_rate_its_frames_contradict_reads_its_signals_and_names_the_rate(tmp_path):
    ep = me.load(_prepared(tmp_path))
    pl = me.plan(ep)
    lines, _ = me._signal_readout(ep, pl)
    # the measured joint follows its command, so one of the two is sent and the other is named as tracking it
    assert any("so each is sent once" in line for line in lines)
    assert abs(me.ep_fps(ep) - 30.0) < 0.01
    assert pl["checks"]["frame_rate"]["stated_fps"] == STATED
    assert abs(pl["checks"]["frame_rate"]["used_fps"] - 30.0) < 0.01


def test_an_episode_whose_frames_agree_with_its_stated_rate_names_nothing(tmp_path):
    d = _prepared(tmp_path)
    np.savez(d / "times.npz", exo=np.arange(1800) / 30.0)
    ctx = json.loads((d / "context.json").read_text())
    (d / "context.json").write_text(json.dumps({**ctx, "fps": 30.0}))
    ep = me.load(d)
    assert me.ep_fps(ep) == 30.0 and "frame_rate" not in ep and "frame_rate" not in me.plan(ep)["checks"]
