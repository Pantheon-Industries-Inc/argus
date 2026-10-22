"""A per-instant row whose readings track a row already sent is sent once, named as tracking it; its range line stays."""
from __future__ import annotations

import re

import numpy as np

from label import episode as me
from label import signals as sg
from tests.test_prompt_blocks import CASES, _pl

N, FPS = 1800, 30


def _path(seed: int) -> np.ndarray:
    """60 s of one joint moving smoothly and irregularly: a few slow waves of random period and phase."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FPS
    return sum(rng.uniform(0.2, 0.8) * np.sin(2 * np.pi * t / rng.uniform(4, 15) + rng.uniform(0, 6)) for _ in range(4))


def _later(x: np.ndarray, frames: int, noise: float = 0.002, seed: int = 9) -> np.ndarray:
    """x reached `frames` later, with a little noise: a measured joint following its command."""
    y = np.concatenate([np.full(frames, x[0]), x[:-frames]])
    return y + np.random.default_rng(seed).normal(0, noise, len(x))


def _case(signals: dict, names: dict | None = None) -> tuple[dict, dict]:
    ep, _ = CASES["teleop_video_only"]()
    ep["state"] = np.zeros((N, 0))
    ep["signals"] = {k: (v if v.ndim == 2 else v[:, None]) for k, v in signals.items()}
    ep["signal_meta"] = {k: ({"names": names[k]} if names and k in names else {}) for k in signals}
    return ep, _pl(N, every=1.0)


def _readout(ep, pl):
    lines, whole = me._signal_readout(ep, pl)
    shown = [l.split(":")[0].strip() for l in lines if l.startswith("    ") and not l.startswith("    at: ")]
    left = [l for l in lines if "so each is sent once" in l]
    return shown, left, whole, lines


def test_a_commanded_arm_and_the_arm_that_follows_it_send_the_following_rows_once():
    """A 7-value command (six joints and a gripper) and the measured arm that follows it 0.1 s later, recorded as six
    joints and a separate gripper: each command value is matched to its measured value column by column, the measured
    rows are kept whatever the names say, and each command row is named as tracking its measured row."""
    cmd = np.stack([_path(s) for s in range(7)], axis=1)
    cmd[:, 6] = (np.sin(2 * np.pi * np.arange(N) / FPS / 9) > 0.3).astype(float)
    arm = np.stack([_later(cmd[:, j], 3, seed=j) for j in range(6)], axis=1)
    grip = _later(cmd[:, 6], 3, seed=6)
    names = {"zz.a": [f"q{j}" for j in range(6)] + ["grip"], "aa.b": [f"q{j}" for j in range(6)]}
    ep, pl = _case({"zz.a": cmd, "aa.b": arm, "aa.b_grip": grip}, names)
    shown, left, whole, _ = _readout(ep, pl)
    assert shown == [f"aa.b q{j}" for j in range(6)] + ["aa.b_grip"]
    assert len(left) == 1
    relations = _relation_lines(_readout(ep, pl)[3])
    assert [r.split(" ≈ ")[0] for r in relations] == [f"zz.a q{j}" for j in range(6)] + ["zz.a grip"]
    assert all(re.search(rf"≈ (0.999|1) x aa.b q{j}, 0.(07|10) s later \(within \d\.\d%\) \(max ", relations[j])
               for j in range(6))
    assert re.search(r"≈ 1 x aa.b_grip, 0.(07|10) s later \(within ", relations[6]) and "(max 1 at " in relations[6]
    assert "zz.a" not in whole and {"aa.b", "aa.b_grip"} <= whole
    # every range line stays
    table = me.build_prompt(ep, pl, cell_w=448, cell_h=252)[1].split("OTHER RECORDED SIGNALS")[1]
    assert "  zz.a (7 values (q0, " in table and "  aa.b (6 values (q0, " in table and "  aa.b_grip (1 value): " in table


def test_two_unrelated_joints_both_keep_their_rows():
    ep, pl = _case({"j.one": _path(1), "j.two": _path(2)})
    shown, left, _, _ = _readout(ep, pl)
    assert shown == ["j.one", "j.two"] and not left


def test_a_sign_flipped_scaled_copy_is_caught():
    """The same joint in degrees and mirrored (-57.3 x + 10) tracks its reading in radians."""
    x = _path(3)
    same, lag = sg.tracking(x, -57.2958 * x + 10.0, list(range(0, N, 30)), FPS)
    assert same and lag == 0
    ep, pl = _case({"rad": x, "deg_mirrored": -57.2958 * x + 10.0})
    shown, left, _, _ = _readout(ep, pl)
    assert len(shown) == 1 and len(left) == 1 and len(_relation_lines(_readout(ep, pl)[3])) == 1


def test_a_pair_that_comes_apart_halfway_keeps_both_rows():
    """A measured joint that follows its command for 30 s and then goes its own way (a tracking fault) is no copy."""
    cmd = _path(4)
    arm = _later(cmd, 3)
    arm[N // 2:] = _path(5)[N // 2:] + arm[N // 2 - 1] - _path(5)[N // 2 - 1]
    ep, pl = _case({"cmd": cmd, "arm": arm})
    shown, left, _, _ = _readout(ep, pl)
    assert sorted(shown) == ["arm", "cmd"] and not left


def test_few_instants_never_track():
    x = _path(6)
    assert sg.tracking(x, x, list(range(0, 30 * (sg.RESOLVED_MIN_INSTANTS - 1), 30)), FPS) == (False, 0)


def _stalled(secs: int, seed: int = 4):
    """A command and the measured joint that follows it 0.1 s later, except that the joint stalls for `secs` seconds
    from 20 s while the command keeps moving (a tracking fault)."""
    cmd = _path(seed)
    arm = _later(cmd, 3)
    s0 = 20 * FPS
    arm[s0:s0 + secs * FPS] = arm[s0]
    return cmd, arm



def _apart_lines(lines):
    """The lines of rows that follow another except where they come apart: (row, partner, [reading times])."""
    out = []
    for l in lines:
        if " ≈ " in l and ", except " in l and "given: " in l:
            row, rest = l.strip().split(" ≈ ", 1)
            other = rest.split(" x ", 1)[1].split(",", 1)[0].split(" (", 1)[0]
            times = [float(p.split()[0]) for p in rest.split("given: ", 1)[1].rstrip(".").split(", ")]
            out.append((row, other, times))
    return out


def _relation_lines(lines):
    """The lines of rows left out whole as following another: the fitted relation of each."""
    return [l.strip() for l in lines if " ≈ " in l and ", except " not in l]


def test_a_two_second_stall_sends_the_command_only_around_the_stall():
    """The joint stalls from 20 to 22 s while its command moves 12% of its range: the joint's row is sent, and the
    command's readings are given only at the instants around the stall, on a line naming the row it tracks."""
    cmd, arm = _stalled(2)
    assert np.ptp(cmd[20 * FPS:22 * FPS]) > 0.1 * np.ptp(cmd)
    shown, left, _, lines = _readout(*_case({"cmd": cmd, "arm": arm}))
    assert shown == ["arm"] and not left
    (row, other, times), = _apart_lines(lines)
    assert (row, other) == ("cmd", "arm")
    assert 20.0 in times and 21.0 in times and min(times) >= 18.0 and max(times) <= 24.0
    assert any(l.strip().startswith("These rows follow a sent row") for l in lines)


def test_a_six_second_stall_sends_the_command_only_around_the_stall():
    cmd, arm = _stalled(6, seed=13)
    shown, _, _, lines = _readout(*_case({"cmd": cmd, "arm": arm}))
    assert shown == ["arm"]
    (row, other, times), = _apart_lines(lines)
    assert (row, other) == ("cmd", "arm")
    assert {21.0, 23.0, 25.0} <= set(times) and min(times) >= 18.0 and max(times) <= 28.0


def _gripper(stop: float | None = None):
    """A gripper commanded open and closed five times in 60 s (1 open, 0 shut, smooth 1 s moves), and the gripper that
    follows it 0.1 s later; with `stop`, the third close is stopped at that opening by what it holds (a grasp)."""
    t = np.arange(N) / FPS
    cmd = np.ones(N)
    for a in (6, 16, 26, 36, 46):
        cmd -= np.clip(t - a, 0, 1) - np.clip(t - a - 5, 0, 1)
    arm = _later(cmd, 3) + np.random.default_rng(2).normal(0, 0.002, N)
    if stop is not None:
        hold = (t >= 26) & (t < 32)
        arm[hold] = np.maximum(arm[hold], stop)
    return cmd, arm


def test_a_grasp_sends_the_command_where_it_closes_past_the_gripper():
    """The third close stops at 0.4 open while the command goes on to shut: the command is read over that close."""
    cmd, arm = _gripper(stop=0.4)
    shown, _, _, lines = _readout(*_case({"grip_cmd": cmd, "grip": arm}))
    assert shown == ["grip"]
    (row, other, times), = _apart_lines(lines)
    assert (row, other) == ("grip_cmd", "grip") and {27.0, 28.0, 29.0, 30.0} <= set(times)
    assert min(times) >= 25.0 and max(times) <= 33.0


def test_a_gripper_that_follows_every_close_is_sent_once():
    cmd, arm = _gripper()
    shown, left, _, lines = _readout(*_case({"grip_cmd": cmd, "grip": arm}))
    assert shown == ["grip"] and len(left) == 1 and not _apart_lines(lines)


def test_an_eight_second_stall_is_read_where_it_stalls():
    """The 8 s stall still tracks its command: the joint is left out except at the instants it stalls (10 of 60)."""
    cmd, arm = _stalled(8)
    shown, left, _, lines = _readout(*_case({"cmd": cmd, "arm": arm}))
    assert shown == ["arm"] and [(r, o, len(t)) for r, o, t in _apart_lines(lines)] == [("cmd", "arm", 10)]


def test_a_ten_second_stall_keeps_both_full_rows():
    """A 10 s stall no longer fits its command (label/signals.py TRACK_LEVEL and TRACK_CHANGE): both rows are sent
    whole, so no row is ever sent as a few readings beside a partner it mostly does not follow."""
    cmd, arm = _stalled(10)
    shown, left, _, lines = _readout(*_case({"cmd": cmd, "arm": arm}))
    assert sorted(shown) == ["arm", "cmd"] and not left and not _apart_lines(lines)


def test_a_clean_copy_gives_no_readings():
    x = _path(3)
    shown, left, _, lines = _readout(*_case({"rad": x, "deg": 57.2958 * x}))
    assert len(shown) == 1 and len(left) == 1 and not _apart_lines(lines)


def test_a_stall_between_instants_is_seen_on_the_frames():
    """A one second stall that starts on an instant and ends before the next one changes no reading at the instants,
    yet the command moved 12% of its range meanwhile: the frames show it, so the instants on both sides are read."""
    cmd, arm = _stalled(1)
    ks = list(range(0, N, FPS))
    assert np.allclose(arm[ks], _later(cmd, 3)[ks])
    same, lag = sg.tracking(cmd, arm, ks, FPS)
    apart = sg.apart_instants(cmd, arm, ks, lag, FPS)
    assert same and {20, 21} <= set(apart) and len(apart) <= 5


def test_grippers_moving_together_and_opposite_give_different_relations():
    """right = left and right = 1 - left are both sent once, but the line says which: the sign and offset of the fit."""
    left, _ = _gripper()
    texts = {}
    for name, right in (("together", left.copy()), ("opposite", 1.0 - left)):
        ep, pl = _case({"left_gripper": left, "right_gripper": right})
        lines, _ = me._signal_readout(ep, pl)
        (line,) = _relation_lines(lines)
        texts[name] = line
    assert texts["together"] != texts["opposite"]
    assert texts["together"].startswith("right_gripper ≈ 1 x left_gripper")
    assert texts["opposite"].startswith("right_gripper ≈ 1 - 1 x left_gripper")


def test_a_mirrored_degree_copy_states_its_scale():
    x = _path(3)
    ep, pl = _case({"rad": x, "deg_mirrored": -57.2958 * x + 10.0})
    lines, _ = me._signal_readout(ep, pl)
    (line,) = _relation_lines(lines)
    assert line.startswith("deg_mirrored ≈ 10 - 57.3 x rad")


def test_a_short_pulse_left_out_keeps_its_time():
    """A 60 ms motor-current pulse on top of noise: too short to lift even the mean of its second above noise, the row is
    left out as unresolved, but when the pulse came stays."""
    rng = np.random.default_rng(4)
    current = 0.05 * rng.normal(size=N)
    current[1260:1262] = 3.21          # 42.00 s to 42.03 s, on the instant at 42 s
    ep, pl = _case({"motor.current": current, "j": _path(1)})
    lines, _ = me._signal_readout(ep, pl)
    left = next(l for l in lines if "does not resolve their readings" in l)
    assert "motor.current (max 3.21 at 42.00 s" in left


def test_what_is_said_about_rows_left_out_is_paid_from_the_readout_budget(monkeypatch):
    """Relations, extreme times, readings where apart and the rows themselves stay within the character budget; only
    the sentence naming what the budget left out is outside it."""
    monkeypatch.setattr(me, "SIGNAL_TABLE_MAX_CHARS", 3000)
    monkeypatch.setattr(me, "SIGNAL_TABLE_MIN_CHARS", 3000)
    cmd, arm = _stalled(2)
    vel = np.random.default_rng(6).normal(size=N)
    joints = {f"j{k}": _path(20 + k) for k in range(6)}
    shown, _, _, lines = _readout(*_case({"cmd": cmd, "arm": arm, "vel": vel, **joints}))
    paid = [l for l in lines if "no more room" not in l]
    room = next(l for l in lines if "no more room" in l)
    # the command's readings where it comes apart are ranked with the rows: sent, or named as left out for room
    assert any(" ≈ " in l for l in paid) or "cmd" in room
    assert any("does not resolve" in l and "vel (max " in l for l in paid)
    # the row the command is read from is sent first, whatever its rank
    assert "arm" in shown
    assert sum(len(l) + 1 for l in paid) <= 3000 + len(lines[0]) + 1


def test_cells_of_one_sensor_are_never_taken_for_copies_of_each_other():
    """Five cells of one fingertip sensor move together, yet each is its own measurement: all five rows are sent."""
    base = _path(5)
    rng = np.random.default_rng(8)
    finger = np.stack([base * (1 + 0.05 * c) + 0.01 * rng.normal(size=N) for c in range(5)], axis=1)
    shown, left, _, lines = _readout(*_case({"finger": finger}, {"finger": [f"cell{c}" for c in range(5)]}))
    assert shown == [f"finger cell{c}" for c in range(5)] and not left and not _relation_lines(lines)


def test_a_stepped_joint_keeps_its_window_means():
    """A slow joint a policy moves in random half-second steps around its path: one reading a second does not stand for
    its second, but the mean over each instant's frames follows the path, so that row is sent, marked."""
    t = np.arange(N) / FPS
    steps = np.repeat(np.random.default_rng(9).choice([-1.0, 1.0], size=N // 16 + 1), 16)[:N]
    joint = np.sin(2 * np.pi * t / 20) + steps
    ep, pl = _case({"joint": joint, "j": _path(1)})
    assert not sg.resolved(joint[:, None], pl["ks"])[0]
    shown, _, _, lines = _readout(ep, pl)
    assert "joint (window mean)" in shown
    assert any("(window mean) gives, at each instant, the mean of the frames" in l for l in lines)


def test_thirteen_instants_keep_every_row_as_it_is():
    """With 13 instants no score is reliable: a noisy row is kept as read, not dropped and not averaged."""
    n = 12 * FPS
    ep, _ = CASES["teleop_video_only"]()
    ep["state"] = np.zeros((n, 0))
    ep["signals"] = {"noisy": np.random.default_rng(10).normal(size=(n, 1)), "j": _path(1)[:n, None]}
    ep["signal_meta"] = {k: {} for k in ep["signals"]}
    pl = _pl(n, every=1.0)
    assert len(pl["ks"]) == 13
    shown, _, _, lines = _readout(ep, pl)
    assert "noisy" in shown and not any("does not resolve" in l for l in lines)


def _stepped(trend: float, seed: int = 9) -> np.ndarray:
    """A joint a policy moves in random half-second steps of 1 around a slow wave of amplitude `trend`."""
    t = np.arange(N) / FPS
    steps = np.repeat(np.random.default_rng(seed).choice([-1.0, 1.0], size=N // 16 + 1), 16)[:N]
    return trend * np.sin(2 * np.pi * t / 20) + steps


def test_window_mean_rows_come_after_resolved_rows_and_by_the_variance_their_means_keep(monkeypatch):
    """With room for one window-mean row, the one whose means keep more of its variance is sent, whatever the order of
    the signals; the other is named with when it peaked, and every resolved row is sent first."""
    low, high = _stepped(0.3), _stepped(1.0, seed=11)
    ep, pl = _case({"a_low": low, "b_high": high, "j": _path(1)})
    assert not sg.resolved(np.stack([low, high], axis=1), pl["ks"]).any()
    kept = sg.window_means(np.stack([low, high], axis=1), pl["ks"])[1]
    assert kept[1] > kept[0]
    room = len(next(l for l in _readout(ep, pl)[3] if l.startswith("    b_high (window mean): ")))
    monkeypatch.setattr(sg, "WINDOW_SHARE", 1.5 * room / me.signal_readout_budget(len(pl["ks"])))
    shown, _, _, lines = _readout(ep, pl)
    assert shown == ["b_high (window mean)", "j"]
    cut = next(l for l in lines if "rows given as window means take at most" in l)
    assert "a_low (max " in cut and " s, min " in cut and "b_high" not in cut


def test_a_line_is_fitted_only_through_readings_that_carry_one():
    """No line through fewer than three finite pairs, or through an x with no spread (all zero, or one value): numpy
    would divide by zero inside the solver and stop the whole episode."""
    x = np.linspace(0, 1, 50)
    assert np.allclose(sg.line_through(x, 2 * x + 1), (2.0, 1.0))
    for bad_x in (np.zeros(50), np.full(50, 0.7), np.r_[np.full(48, np.nan), 1.0, 2.0], np.full(50, np.inf)):
        assert sg.line_through(bad_x, x) is None


def test_a_partner_that_never_moves_outside_where_the_pair_comes_apart_keeps_both_rows(monkeypatch):
    """A command that moves only from 20 s to 40 s and the joint that follows it: when the instants where they come
    apart cover every instant the command moves, the readings left carry no line, so the joint is no copy and both
    rows are sent (episode 151721 of the teleop 100h run stopped here on a zero-spread fit)."""
    t = np.arange(N) / FPS
    cmd = np.where((t > 20) & (t < 40), np.sin(2 * np.pi * (t - 20) / 10), 0.0)
    arm = _later(cmd, 3, noise=0.0)     # still exactly, as the recorded joints were outside the motion
    ep, pl = _case({"cmd": cmd, "arm": arm})
    real = sg.apart_instants

    def moving(x, y, ks, lag, fps):
        ks = np.asarray(ks, dtype=int)
        return [i for i, k in enumerate(ks) if 19 * fps <= k <= 41 * fps]
    monkeypatch.setattr(sg, "apart_instants", moving)
    shown, left, _, lines = _readout(ep, pl)
    assert sorted(shown) == ["arm", "cmd"] and not left and not _apart_lines(lines)
    monkeypatch.setattr(sg, "apart_instants", real)
    _readout(ep, pl)        # and with the real rule the readout is made, whatever it decides


def test_a_row_read_from_a_sent_row_that_is_replaced_is_fitted_again_against_its_new_partner(monkeypatch):
    """Rows q, t and r, in rank order: t follows q and is read from it, then r follows q later and takes q's place. t is
    fitted again against r with the instants where those two come apart: read from r with the instants where t and q came
    apart, its fit had no line (r does not move outside them) and the TypeError stopped a whole teleop episode
    (robot07-teleop 20260827 episode_172301_db02098f)."""
    t_s = np.arange(N) / FPS
    q, t = _path(1), _path(2)
    r = np.where((t_s > 20) & (t_s < 40), _path(3), 0.0)
    ep, _ = _case({"q": q})
    ks = list(range(0, N, FPS))
    rows = [((3.0 - i, 3.0 - i), name, 0, name, name, ["0"] * len(ks)) for i, name in enumerate("qtr")]
    series = {("q", 0): q, ("t", 0): t, ("r", 0): r}
    pairs = {(id(q), id(t)): 3, (id(q), id(r)): -3, (id(r), id(t)): 0}
    moving = [i for i, k in enumerate(ks) if 19 * FPS <= k <= 41 * FPS]
    monkeypatch.setattr(sg, "tracking", lambda x, y, ks, fps: ((id(x), id(y)) in pairs, pairs.get((id(x), id(y)), 0)))
    monkeypatch.setattr(sg, "apart_instants", lambda x, y, ks, lag, fps: moving if (x is q and y is t) else [])
    sent, tracked = me._drop_tracking(ep, rows, series, ks)
    assert [s[3] for s in sent] == ["r"] and sorted(row[3] for row, _, _ in tracked) == ["q", "t"]
    for row, partner, apart in tracked:
        assert partner[3] == "r"
        assert me._relation_words(ep, ks, row, partner, series, apart).startswith(f"{row[4]} ≈ ")
