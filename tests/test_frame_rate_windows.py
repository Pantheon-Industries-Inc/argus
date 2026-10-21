"""A window or a lag given in seconds is counted in frames at the episode's frame rate. A window longer than the frames
it is read over gave a series of the window's length, and a per-frame mask applied to it stopped the whole episode
(ABC-130k episode_0070d777: a 1 s window at a stated 976562.5 fps, 3077 frames). Every windowed series has one value
per frame (label/signals.py window_sum), and a stated rate the frames' own capture times contradict is not used
(prepare/frame_rate.py, the only module that reads a frame rate), and is named in the episode's checks."""
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
    from prepare.frame_rate import supported_rate
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


def _doubled_clock(n: int = 301) -> np.ndarray:
    """n capture times (s) of a camera whose median step is 1/60 s but just under half of whose intervals are doubled,
    as ABC-130k stations stamp frames 1 or 2 intervals apart: 60 fps by the median step, about 40 over the span."""
    steps = np.full(n - 1, 1 / 60)
    steps[1:-2:2] = 2 / 60
    return np.concatenate([[0.0], np.cumsum(steps)])


def _prepared(tmp_path, n: int = 1800, clock=None, fps=STATED, name="episode_clustered"):
    """A prepared episode as ABC-130k's adapter wrote episode_0070d777: real capture times on a clustered clock, a
    stated fps of 976562.5 and two signals that track each other (or the given clock and stated fps)."""
    d = tmp_path / name
    d.mkdir()
    cmd, arm = _follows(n)
    np.savez(d / "signals.npz", s0=cmd[:, None].astype(np.float32), s1=arm[:, None].astype(np.float32))
    np.savez(d / "times.npz", exo=_clustered_clock(n) if clock is None else clock)
    (d / "sources.json").write_text(json.dumps({"exo": {"n_frames": n}}))
    (d / "context.json").write_text(json.dumps({
        "dataset": "fixture", "profile": "teleop_arms", "state_kind": "none", "fps": fps, "n_state_frames": n,
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


def test_every_reader_of_an_already_prepared_episode_reads_its_frames_rate(tmp_path):
    """The episode as prepared, with no duration_s and a stated 976562.5 fps over 1800 frames at 30 fps: every reader
    that takes its length from its frame count says 60 s, not 1.8 ms, with no re-prepare."""
    from board import build
    from gate import score
    from label import pieces
    from label.run import footage_hours
    from prepare.frame_rate import episode_fps, episode_seconds
    d = _prepared(tmp_path)
    ctx = json.loads((d / "context.json").read_text())
    rate, note = episode_fps(ctx, ep_dir=d)
    assert abs(rate - 30.0) < 0.01 and note["stated_fps"] == STATED
    for seconds in (episode_seconds(ctx, ep_dir=d), footage_hours(tmp_path, [d.name]) * 3600,
                    pieces.duration(ctx, d), build.episode_seconds(ctx, d), score.duration_s(d)):
        assert abs(seconds - 60.0) < 0.1, seconds


def test_a_median_step_the_span_contradicts_gives_way_to_the_frames_rate():
    """A rate measured from capture times is the median step's only while the frames over their span agree with it
    (frame_rate.measured_rate). 249 of 3,417 ABC-130k episodes were written at a median rate their span contradicts:
    60 fps on frames that come at 30, and 976562.5 fps on frames 1 us apart in pairs."""
    t = _doubled_clock()
    own = (len(t) - 1) / (t[-1] - t[0])
    assert abs(np.median(np.diff(t)) - 1 / 60) < 1e-12 and 40 < own < 41
    assert abs(abc130k.measured_fps(np.round(t * 1e9)) - own) < 0.01
    assert abs(f.measured_fps(t) - own) < 0.01
    assert abs(abc130k.measured_fps(np.round(_clustered_clock(3077) * 1e9)) - 30.0) < 0.01
    # frames that agree with their median step to within SPAN_AGREEMENT keep its rate exactly, and a pause in the
    # recording is no frame interval: neither changes the rate an adapter writes
    dropped = np.delete(np.arange(3000) / 30.0, np.arange(100, 3000, 100))      # 1% of frames dropped
    paused = np.r_[np.arange(1500), np.arange(1500) + 6000] / 30.0               # recording stopped for 150 s
    for t in (np.arange(300) / 30.0, dropped, paused):
        assert f.measured_fps(t) == 30.0 and abc130k.measured_fps(t * 1e9) == 30.0


def test_every_length_reader_takes_an_old_episodes_length_from_its_capture_span(tmp_path):
    """An episode prepared before adapters wrote duration_s, at the 60 fps its median step gave while its frames come at
    40 fps over their span: no reader divides its frame count by that rate, every one reads the capture span plus one
    frame step, and a stated duration_s is read as written."""
    from board import build
    from gate import score
    from label import pieces
    from label.run import footage_hours
    from prepare.frame_rate import episode_seconds, frames_seconds
    t = _doubled_clock()
    d = _prepared(tmp_path, n=len(t), clock=t, fps=60.0, name="episode_doubled")
    ctx = json.loads((d / "context.json").read_text())
    span = (t[-1] - t[0]) * len(t) / (len(t) - 1)
    assert abs(frames_seconds(ctx, ep_dir=d) - span) > 2.0       # the length counted from the stated rate is wrong
    readers = lambda c: (episode_seconds(c, ep_dir=d), footage_hours(tmp_path, [d.name]) * 3600,  # noqa: E731
                         pieces.duration(c, d), build.episode_seconds(c, d), score.duration_s(d))
    for seconds in readers(ctx):
        assert abs(seconds - span) < 1e-6, seconds
    (d / "context.json").write_text(json.dumps({**ctx, "duration_s": 12.5}))
    for seconds in readers({**ctx, "duration_s": 12.5}):
        assert seconds == 12.5


def test_the_abc_adapter_writes_the_span_rate_and_the_capture_length(tmp_path):
    """The ABC-130k adapter on a recording stamped 1 or 2 frame intervals apart (median 1/60 s): its context states the
    frames' rate over their span and duration_s from the capture times, and the generic reader of the same file reads
    the same length."""
    from prepare.frame_rate import episode_seconds
    from test_abc_generic import capture_mcap, converted, published
    steps = np.ones(20)
    steps[1:18:2] = 2                                  # 9 of 20 intervals doubled
    capture = np.round(np.concatenate([[0.0], np.cumsum(steps)]) * 1e9 / 60).astype(np.int64)
    root = tmp_path / "native"
    root.mkdir()
    capture_mcap(root / "episode.mcap", n=len(capture), capture_ns=capture)
    _, ctx = published(root / "episode.mcap", tmp_path / "old")
    span = (capture[-1] - capture[0]) / 1e9
    assert abs(ctx["fps"] - 20 / span) < 0.01
    # the wrist cameras are stamped 2 and 4 ms after the top camera, so the footage ends 4 ms after the top's last frame
    assert abs(ctx["duration_s"] - (span + 0.004 + 1 / ctx["fps"])) < 0.002
    generic, gctx = converted(root, tmp_path / "generic")
    assert abs(episode_seconds(gctx, ep_dir=generic) - ctx["duration_s"]) < 0.05


def _raw_rate_reads(path) -> list[str]:
    """Each place in a module that reads an "fps" key (a string "fps" that is not a dict literal's key or the target
    of an assignment), counts frames over a rate (n_state_frames then "/"), or reads an episode context's rate as a
    source's own (declared_fps on a context)."""
    import io
    import tokenize
    toks = [t for t in tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
            if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT)]
    out = []
    for i, t in enumerate(toks):
        after = [x.string for x in toks[i + 1:i + 6]]
        if t.type == tokenize.STRING and t.string.strip("rbRB") in ("'fps'", '"fps"'):
            key_of_literal = after[:1] == [":"]
            assigned = after[:1] == ["]"] and after[1:2] == ["="]
            if not key_of_literal and not assigned:
                out.append(f"{path}:{t.start[0]}")
        if t.string in ("n_state_frames", "'n_state_frames'", '"n_state_frames"'):
            rest = [s for s in after if s not in ("]", ")")]
            if rest[:1] == ["/"] or (rest[:1] == ["or"] and "/" in rest[2:4]):
                out.append(f"{path}:{t.start[0]}")
        if t.string == "declared_fps" and after[:1] == ["("] and after[1:2] in (["ctx"], ["context"], ["c"]):
            out.append(f"{path}:{t.start[0]}")
        # a length counted from a frame count at a rate, where the capture times may say otherwise (episode_seconds)
        if t.string == "frames_seconds" and after[:1] == ["("]:
            out.append(f"{path}:{t.start[0]}")
    return out


def test_only_the_frame_rate_module_reads_a_frame_rate():
    """A new read of a stored "fps", or a frame count over a rate, anywhere but prepare/frame_rate.py would read an
    episode at a rate its frames can contradict (episode_0070d777's 976562.5 fps)."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    found = [hit for pkg in ("label", "board", "checks", "compare", "gate", "review", "prepare")
             for p in sorted((root / pkg).rglob("*.py")) if p != root / "prepare" / "frame_rate.py"
             for hit in _raw_rate_reads(p)]
    assert not found, found


def test_the_guard_sees_each_kind_of_raw_read(tmp_path):
    p = tmp_path / "m.py"
    for code in ('x = ctx.get("fps")', "x = ctx['fps']", 'x = ctx["n_state_frames"] / 30',
                 "x = (c.get('n_state_frames') or 0) / rate", "keys = ('fps', 'n')", 'x = declared_fps(ctx)',
                 'x = ctx.get("duration_s") or frames_seconds(ctx, ep_dir=d)'):
        p.write_text(code + "\n")
        assert _raw_rate_reads(p), code
    for code in ('d = {"fps": 30}', "ctx['fps'] = 30", '"""a docstring that says fps"""', 'x = declared_fps(info)'):
        p.write_text(code + "\n")
        assert not _raw_rate_reads(p), code
