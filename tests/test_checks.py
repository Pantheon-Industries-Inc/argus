"""The checks stage on synthetic episodes: stream pairing, recorded jumps, gripper channels, the capture checks on a
real decoded video, every label consistency rule, and the sped-up recording scan and apply. No network."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from checks import capture_qc, label_consistency, stream_pairing, timebase
from label import episode as me

REPO = Path(__file__).resolve().parent.parent
T = 120
W, H = 64, 48


def write_video(path: Path, levels) -> None:
    """A 30 fps mp4 of uniform grey frames, one level per frame, encoded losslessly so the decoded grey level
    change between frames is the level change."""
    import av
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=30)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.options = {"qp": "0", "preset": "ultrafast", "g": "15"}
        for k, lv in enumerate(levels):
            f = av.VideoFrame.from_ndarray(np.full((H, W, 3), int(round(lv)), np.uint8), format="rgb24")
            f.pts = k
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def levels_for(speed_cm: np.ndarray, gain: float = 8.0) -> np.ndarray:
    """Grey levels whose change from frame k to k+1 is gain x speed_cm[k], turning back at the ends of the range,
    as the picture of a camera rigidly mounted on an actor moving at that speed would change."""
    out, lv, sign = [100.0], 100.0, 1.0
    for v in speed_cm:
        d = gain * float(v)
        if not 30 <= lv + sign * d <= 225:
            sign = -sign
        lv += sign * d
        out.append(lv)
    return np.array(out)


def write_episode(d: Path, state: np.ndarray, videos: dict, profile: str = "handheld_gripper",
                  kind: str = "ee_pose", **ctx) -> Path:
    """An episode folder as prepare writes it: context.json, sources.json, state.npz and one mp4 per camera."""
    d.mkdir(parents=True)
    src = {}
    for v, levels in videos.items():
        write_video(d / f"{v}.mp4", levels)
        src[v] = {"packed": str(d / f"{v}.mp4"), "base_s": 0.0, "n_frames": len(levels)}
    (d / "sources.json").write_text(json.dumps(src))
    (d / "context.json").write_text(json.dumps({"profile": profile, "state_kind": kind, "fps": 30, **ctx}))
    np.savez(d / "state.npz", state=state)
    return d


def two_grippers(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A handheld two-gripper ee_pose state whose grippers move independently along x, still in rotation, and each
    one's speed in cm per frame."""
    rng = np.random.default_rng(seed)
    vL, vR = rng.uniform(0, 1.0, T - 1), rng.uniform(0, 1.0, T - 1)
    s = np.zeros((T, 14))
    s[1:, 0] = np.cumsum(vL) / 100
    s[1:, 7] = np.cumsum(vR) / 100
    s[:, 6] = np.linspace(0.2, 0.8, T)
    s[:, 13] = np.linspace(0.8, 0.2, T)
    return s, vL, vR


# ---------------------------------------------------------------- stream_pairing

def test_stream_pairing_follows_own_actor(tmp_path):
    s, vL, vR = two_grippers()
    d = write_episode(tmp_path / "episode_000000", s, {"left": levels_for(vL), "right": levels_for(vR)})
    r = stream_pairing.pairing(d)
    assert r["crossed"] is False
    assert r["left_vs_left"] > 0.9 and r["right_vs_right"] > 0.9
    assert abs(r["left_vs_right"]) < 0.4 and abs(r["right_vs_left"]) < 0.4


def test_stream_pairing_swapped_streams(tmp_path):
    s, vL, vR = two_grippers()
    d = write_episode(tmp_path / "episode_000000", s, {"left": levels_for(vR), "right": levels_for(vL)})
    r = stream_pairing.pairing(d)
    assert r["crossed"] is True
    assert r["left_vs_right"] > 0.9 and r["right_vs_left"] > 0.9


def test_stream_pairing_needs_two_mounted_streams(tmp_path):
    s, vL, _ = two_grippers()
    d = write_episode(tmp_path / "episode_000000", s[:, :7], {"right": levels_for(vL)})
    assert stream_pairing.pairing(d) is None


# ---------------------------------------------------------------- recorded_jumps

def _leap_episode(tmp_path: Path, camera_shows_it: bool) -> Path:
    s, vL, vR = two_grippers(1)
    s[60:, 0] += 0.10                          # the left gripper's recorded x leaps 10 cm between frames 59 and 60
    seen = vL.copy()
    if camera_shows_it:
        seen[59] += 10.0
    return write_episode(tmp_path / "episode_000000", s, {"left": levels_for(seen), "right": levels_for(vR)})


def test_recorded_jump_the_camera_does_not_show(tmp_path):
    r = stream_pairing.jumps(_leap_episode(tmp_path, camera_shows_it=False))
    assert r["flagged"] is True
    (ev,) = [e for e in r["events"] if e.get("visual_jump") is False]
    assert ev["actor"] == "left" and ev["camera"] == "left" and ev["unit"] == "cm"
    assert ev["t_s"] == pytest.approx(60 / 30, abs=1e-3) and ev["step"] > 9


def test_recorded_jump_the_camera_shows(tmp_path):
    r = stream_pairing.jumps(_leap_episode(tmp_path, camera_shows_it=True))
    assert r["flagged"] is False
    assert [e["visual_jump"] for e in r["events"] if e["actor"] == "left"] == [True]


def test_no_jump_in_smooth_motion(tmp_path):
    s, vL, vR = two_grippers(2)
    d = write_episode(tmp_path / "episode_000000", s, {"left": levels_for(vL), "right": levels_for(vR)})
    r = stream_pairing.jumps(d)
    assert r["events"] == [] and r["flagged"] is False


# ---------------------------------------------------------------- gripper_channels

def test_flat_gripper_channel(tmp_path):
    s, vL, vR = two_grippers()
    s[:, 6] = 0.5                              # the left gripper reads exactly the same at every frame
    d = write_episode(tmp_path / "episode_000000", s, {"left": levels_for(vL), "right": levels_for(vR)})
    r = stream_pairing.grippers(d)
    assert r["flagged"] is True
    assert r["actors"]["left"]["flat"] is True and r["actors"]["right"]["flat"] is False


def test_moving_gripper_channels(tmp_path):
    s, vL, vR = two_grippers()
    d = write_episode(tmp_path / "episode_000000", s, {"left": levels_for(vL), "right": levels_for(vR)})
    r = stream_pairing.grippers(d)
    assert r["flagged"] is False and not any(a["flat"] for a in r["actors"].values())


def test_video_only_episode_has_no_state_checks(tmp_path):
    d = write_episode(tmp_path / "episode_000000", np.zeros((T, 0)), {"exo": levels_for(np.ones(T - 1))},
                      profile="ego_head", kind="none")
    assert stream_pairing.jumps(d) is None and stream_pairing.grippers(d) is None
    assert stream_pairing.pairing(d) is None


# ---------------------------------------------------------------- capture_qc on a decoded video

def test_capture_qc_frozen_camera_while_the_pose_moves(tmp_path):
    s, vL, vR = two_grippers(3)
    frozen = vL.copy()
    frozen[40:100] = 0.0                       # the left camera's picture stops for 2 s while its gripper moves
    d = write_episode(tmp_path / "episode_000000", s, {"left": levels_for(frozen), "right": levels_for(vR)})
    r = capture_qc.run_episode(d)
    (f,) = [f for f in r["flags"] if f["check"] == "video_frozen_run"]
    assert f["camera"] == "left" and f["t_s"] == pytest.approx(40 / 30, abs=0.02)
    assert r["metrics"]["cameras"]["left"]["decoded"] == T
    listed = {c["check"]: c for c in r["checks"]}
    assert set(listed) == set(capture_qc.CHECKS)
    assert listed["gripper_never_acts"]["status"] == "not_applicable"      # no verified 0-1 gripper unit


def test_capture_qc_live_camera(tmp_path):
    s, vL, vR = two_grippers(3)
    d = write_episode(tmp_path / "episode_000000", s, {"left": levels_for(vL), "right": levels_for(vR)})
    r = capture_qc.run_episode(d)
    assert "video_frozen_run" not in {f["check"] for f in r["flags"]}
    assert r["source"] == capture_qc.SOURCE and r["version"] == capture_qc.VERSION


# ---------------------------------------------------------------- the command line

def test_checks_command_writes_every_key(tmp_path):
    s, vL, vR = two_grippers()
    write_episode(tmp_path / "eps" / "episode_000000", s, {"left": levels_for(vL), "right": levels_for(vR)})
    r = subprocess.run([sys.executable, "-m", "checks", "--jobs", "1", str(tmp_path / "eps")], cwd=REPO,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    ctx = json.loads((tmp_path / "eps" / "episode_000000" / "context.json").read_text())
    assert {"stream_pairing", "recorded_jumps", "gripper_channels", "capture_qc"} <= set(ctx)


def test_checks_help_lists_every_check():
    out = subprocess.run([sys.executable, "-m", "checks", "--help"], cwd=REPO, capture_output=True, text=True,
                         timeout=30).stdout
    for name in ("stream_pairing", "recorded_jumps", "gripper_channels", "capture_qc", "timebase",
                 "label_consistency", "board build"):
        assert name in out, name


# ---------------------------------------------------------------- label_consistency

def rules(label: dict, duration_s: float | None = 10.0) -> list[str]:
    return [f["rule"] for f in label_consistency.check(label, duration_s)]


def test_outcome_vs_alignment():
    assert rules({"completion": {"task_completed": "success"}, "goal_alignment": {"relation": "different"}}) == [
        "outcome_vs_alignment"]
    assert rules({"completion": {"task_completed": "success_then_undone", "goal_reached_at_s": 2, "undone_at_s": 4},
                  "goal_alignment": {"relation": "unrelated"}, "state_changes": [{"t_s": 3}]}) == [
        "outcome_vs_alignment"]
    assert rules({"completion": {"task_completed": "success"}, "goal_alignment": {"relation": "aligned"}}) == []
    assert rules({"completion": {"task_completed": "failure"}, "goal_alignment": {"relation": "different"}}) == []


def test_undone_timing():
    for c in ({"goal_reached_at_s": 5.0}, {"undone_at_s": 5.0}, {"goal_reached_at_s": 5.0, "undone_at_s": 5.0},
              {"goal_reached_at_s": 6.0, "undone_at_s": 4.0}):
        assert rules({"completion": {"task_completed": "success_then_undone", **c}}) == ["undone_timing"], c


def test_nothing_after_goal():
    c = {"task_completed": "success_then_undone", "goal_reached_at_s": 5.0, "undone_at_s": 7.0}
    assert rules({"completion": c, "state_changes": [{"t_s": 2.0}, {"t_s": 5.0}]}) == ["nothing_after_goal"]
    assert rules({"completion": c}) == ["nothing_after_goal"]
    assert rules({"completion": c, "state_changes": [{"t_s": 2.0}, {"t_s": 6.5}]}) == []


def test_time_past_end():
    assert rules({"completion": {"task_completed": "success", "completed_at_s": 30.0}}) == ["time_past_end"]
    assert rules({"completion": {"task_completed": "success", "goal_reached_at_s": 12.0}}) == ["time_past_end"]
    assert rules({"completion": {"task_completed": "success", "completed_at_s": 10.8}}) == []   # within 1 s
    assert rules({"completion": {"task_completed": "success", "completed_at_s": 30.0}}, None) == []


def test_progress_vs_outcome():
    full = [{"start_s": 0, "progress": 0.4}, {"start_s": 5, "progress": 1.0}]
    assert rules({"completion": {"task_completed": "failure"}, "timeline": full}) == ["progress_vs_outcome"]
    assert rules({"completion": {"task_completed": "partial"}, "event_labels": full}) == ["progress_vs_outcome"]
    assert rules({"completion": {"task_completed": "partial"}, "timeline": [{"progress": 0.8}]}) == []
    assert rules({"completion": {"task_completed": "success", "goal_reached_at_s": 5}, "timeline": full}) == []
    assert rules({"completion": {"task_completed": "failure"}, "timeline": [{"progress": None}]}) == []


def test_progress_vs_outcome_for_an_unclear_outcome():
    full = [{"start_s": 0, "progress": 0.4}, {"start_s": 5, "progress": 1.0}]
    assert rules({"completion": {"task_completed": "unclear"}, "timeline": full}) == ["progress_vs_outcome"]


def test_goal_times_differ():
    c = {"task_completed": "success", "completed_at_s": 9.0}
    assert rules({"completion": {**c, "goal_reached_at_s": 0.0}}) == ["goal_times_differ"]
    assert rules({"completion": {**c, "goal_reached_at_s": 9.0}}) == []


def _steps(*rows):
    return [{"start_s": a, "end_s": b, "progress": p, "contribution": c} for a, b, p, c in rows]


def test_progress_past_goal():
    # "place the coffee filter in the dripper": the first filter is seated at the goal frame, the second much later
    two = _steps((0, 4, 0.25, "advancing"), (4, 5, 0.5, "advancing"), (5, 8, 0.75, "advancing"), (8, 9, 1.0, "advancing"))
    c = {"task_completed": "success", "completed_at_s": 5.0, "goal_reached_at_s": 5.0}
    assert rules({"completion": c, "goal_alignment": {"relation": "aligned"}, "timeline": two}) == ["progress_past_goal"]
    # the label already says it does more than asked: nothing to flag
    assert rules({"completion": c, "goal_alignment": {"relation": "narrower"}, "timeline": two}) == []
    assert rules({"completion": c, "goal_alignment": {"relation": "aligned"}, "timeline": two,
                  "data_issues": [{"category": "instruction_mismatch"}]}) == []
    # a step that ends a second after the goal frame is where the step ends, not more work
    near = _steps((0, 4, 0.5, "advancing"), (4, 6, 1.0, "advancing"))
    assert rules({"completion": c, "goal_alignment": {"relation": "aligned"}, "timeline": near}) == []


def test_progress_before_goal():
    early = _steps((0, 2, 1.0, "advancing"), (2, 10, 1.0, "idle"))
    c = {"task_completed": "success", "completed_at_s": 9.0, "goal_reached_at_s": 9.0}
    assert rules({"completion": c, "timeline": early}) == ["progress_before_goal"]
    # a parked arm's idle step at 0 never reads as the level dropping
    parked = _steps((0, 9, 0.0, "idle"), (0, 8, 0.6, "advancing"), (8, 9, 1.0, "advancing"))
    assert rules({"completion": c, "timeline": parked}) == []


def test_consistent_label_has_no_findings():
    label = {"completion": {"task_completed": "success_then_undone", "goal_reached_at_s": 4.0, "undone_at_s": 8.0},
             "goal_alignment": {"relation": "aligned"}, "state_changes": [{"t_s": 4.0}, {"t_s": 8.0}],
             "timeline": [{"progress": 1.0}]}
    assert rules(label) == []


# ---------------------------------------------------------------- timebase

def _recording(seed: int, sped_up: bool, skips: bool, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """(state, action) of one teleop episode stamped at 30 Hz. The leader's arm joints follow smooth random paths
    in real time and the follower trails them by 4/30 s. A sped-up recorder's loop ran at 20 Hz, so each sample is
    1/20 s of real time; with skips, one sample in 12 is dropped (a step that covers two real intervals)."""
    rng = np.random.default_rng(seed)
    dt = 1 / 20 if sped_up else 1 / 30
    steps = np.full(n - 1, dt)
    if skips:
        steps[6::12] = 2 * dt
    t = np.concatenate([[0.0], np.cumsum(steps)]) + 1.0
    fine = np.arange(0.0, t[-1] + 1.0, 1 / 600)               # the real-time paths, sampled at 600 Hz
    walk = np.cumsum(rng.normal(0, 1, (len(fine), 12)), axis=0)
    kernel = np.hanning(181) / np.hanning(181).sum()
    path = np.stack([np.convolve(walk[:, j], kernel, mode="same") for j in range(12)], axis=1) * 0.01

    def joints(at: np.ndarray) -> np.ndarray:
        out = np.zeros((len(at), 14))
        out[:, timebase.JOINTS] = np.stack([np.interp(at, fine, path[:, j]) for j in range(12)], axis=1)
        return out
    return joints(t - 4 / 30), joints(t)


# episode index -> (task, sped up, skips): a real-time run with one noisy short-lag episode (3) and one jittery
# real-time episode (5), then a sped-up run of another task
PLAN = {**{e: ("wipe", False, False) for e in range(7)}, 3: ("wipe", True, True), 5: ("wipe", False, True),
        **{e: ("fold", True, True) for e in range(7, 14)}}


def _raw(tmp_path: Path) -> Path:
    """A MolmoAct2-shaped download folder: meta/ with the episode index, and one data parquet."""
    raw = tmp_path / "raw"
    (raw / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (raw / "data" / "chunk-000").mkdir(parents=True)
    (raw / "meta" / "info.json").write_text("{}")
    pd.DataFrame({"episode_index": list(PLAN), "tasks": [[PLAN[e][0]] for e in PLAN],
                  "data/chunk_index": 0, "data/file_index": 0}).to_parquet(
        raw / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    rows = []
    for e, (_, sped, skips) in PLAN.items():
        state, action = _recording(e, sped, skips)
        rows += [{"observation.state": state[k], "action": action[k], "episode_index": e, "frame_index": k}
                 for k in range(len(state))]
    pd.DataFrame(rows).to_parquet(raw / "data" / "chunk-000" / "file-000.parquet")
    return raw


def test_timebase_measures_lag_and_jitter():
    state, action = _recording(0, sped_up=False, skips=False)
    assert timebase.follower_lag_frames(state, action) == pytest.approx(4.0, abs=0.2)
    state, action = _recording(0, sped_up=True, skips=True)
    assert timebase.follower_lag_frames(state, action) < timebase.SPEDUP_LAG_FRAMES      # 4/30 s at 20 Hz: 2.7
    assert timebase.sample_jitter(state)["skipped_frac"] >= timebase.SPEDUP_JITTER_FRAC
    # measured alone the episode is never flagged: the rule needs the neighbours from the scan
    assert timebase.timebase_check(state, action)["sped_up_recording"] is False
    assert timebase.timebase_check(state, action, neighbour_lag=2.7)["sped_up_recording"] is True


def test_timebase_scan_and_apply(tmp_path, capsys):
    out = tmp_path / "timebase.csv"
    assert timebase.cmd_scan(Namespace(raw=_raw(tmp_path), out=out, jobs=1)) == 0
    x = pd.read_csv(out).set_index("episode_index")
    flagged = set(x.index[x.sped_up_recording.astype(bool)])
    assert flagged == set(range(7, 14))        # not 3 (real-time neighbours), not 5 (a real-time lag)
    assert x.loc[3, "follower_lag_frames"] < timebase.SPEDUP_LAG_FRAMES < x.loc[3, "neighbour_lag_frames"]

    eps = tmp_path / "episodes"
    for e in (3, 8):
        (eps / f"episode_{e:06d}").mkdir(parents=True)
        (eps / f"episode_{e:06d}" / "context.json").write_text(json.dumps({"profile": "teleop_arms"}))
    labels = tmp_path / "run" / "out"
    labels.mkdir(parents=True)
    measured = x.loc[8, ["follower_lag_frames", "skipped_frac", "repeated_frac"]].to_dict()
    (labels / "episode_000008.json").write_text(json.dumps({
        "episode_dir": str(eps / "episode_000008"),
        "dataset_checks": {"timebase": {**measured, "sped_up_recording": False}}}))
    assert timebase.cmd_apply(Namespace(timebase=out, roots=[eps], labels=labels)) == 0
    ctx = json.loads((eps / "episode_000008" / "context.json").read_text())
    assert ctx["timebase_neighbour_lag_frames"] == pytest.approx(x.loc[8, "neighbour_lag_frames"])
    tb = json.loads((labels / "episode_000008.json").read_text())["dataset_checks"]["timebase"]
    assert tb["sped_up_recording"] is True and tb["rule"] == timebase.SPEDUP_RULE


def test_single_arm_teleop_with_action_plans(tmp_path):
    """A one-arm teleop episode (7 values: six joints and a gripper) with leader actions is a layout the reader
    accepts (formats.state_layout), so planning its request must not fail in the sped-up check."""
    rng = np.random.default_rng(0)
    T = 120
    s = np.cumsum(rng.normal(0, 0.01, (T, 7)), axis=0)
    d = write_episode(tmp_path / "episode_000000", s, {"exo": np.full(T, 100.0)}, profile="teleop_arms",
                      kind="joints")
    np.savez(d / "state.npz", state=s, action=np.roll(s, -3, axis=0))
    pl = me.plan(me.load(d))
    assert "timebase" not in pl["checks"]          # the sped-up measure reads two arms, as measure_folder does


def test_single_arm_lerobot_upload_builds_its_request(tmp_path):
    """The same through the reader: a one-arm LeRobot v2.1 dataset (a single 6-joint arm and gripper, 7 values, with
    actions), converted as an upload is, must build its request."""
    from prepare import formats
    from test_prepare import _lerobot_v21
    root = tmp_path / "one_arm"
    _lerobot_v21(root, n=60, dims=7)
    out = tmp_path / "episodes"
    rep = formats.convert(root, "teleop_arms", out, "one_arm", float("inf"), grouping={})
    assert [e["state_kind"] for e in rep["episodes"]] == ["joints"]
    ep = out / rep["episodes"][0]["episode_id"]
    assert np.load(ep / "state.npz")["action"].shape == (60, 7)
    me.build_request(ep)


def _tb_episode(root: Path, name: str, idx: int, folder: str, lag: int) -> None:
    d = root / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(abs(hash((name, idx))) % 2**32)
    a = np.cumsum(rng.normal(0, 0.01, (400, 14)), axis=0)
    np.savez(d / "state.npz", state=np.roll(a, lag, axis=0), action=a)
    (d / "context.json").write_text(json.dumps({"profile": "teleop_arms", "state_kind": "joints",
                                                "task_label": ["stack the cups"],
                                                "source": {"folder": folder, "episode_index": idx}}))


def test_neighbour_lag_is_the_neighbours_median_when_two_datasets_share_indices(tmp_path):
    """An upload of two collections (two LeRobot datasets, or two Galaxea collection folders) has episode indices
    0..4 in each. The neighbour lag of episode 2 of collection B must be the median of its own run (lags 4, 4, 2, 4,
    4 -> 4), not its own lag: the neighbour term exists so one low-lag episode among real-time neighbours is not
    flagged (checks/timebase.py docstring, episode 241)."""
    for i in range(5):
        _tb_episode(tmp_path, f"episode_A_{i:06d}", i, "collection_A", lag=4)
        _tb_episode(tmp_path, f"episode_B_{i:06d}", i, "collection_B", lag=2 if i == 2 else 4)
    assert timebase.measure_folder(tmp_path) == 10
    ctx = json.loads((tmp_path / "episode_B_000002" / "context.json").read_text())
    assert ctx["timebase_neighbours_in_upload"] == 5
    assert abs(ctx["timebase_neighbour_lag_frames"] - 4.0) < 0.3, ctx["timebase_neighbour_lag_frames"]


def test_neighbour_lag_in_one_collection_is_the_neighbours_median(tmp_path):
    """Control: the same five episodes of collection B alone get the median of their run."""
    for i in range(5):
        _tb_episode(tmp_path, f"episode_B_{i:06d}", i, "collection_B", lag=2 if i == 2 else 4)
    timebase.measure_folder(tmp_path)
    ctx = json.loads((tmp_path / "episode_B_000002" / "context.json").read_text())
    assert abs(ctx["timebase_neighbour_lag_frames"] - 4.0) < 0.3


def test_a_one_arm_recorders_folder_builds_its_request(tmp_path):
    """Our own capture stack with one arm (its follower joints and gripper, and its leader's commands): read as one
    episode with 7 values of joint state and the leader as action, which must build its request."""
    from prepare import formats
    from test_formats import recorder_folder
    d = recorder_folder(tmp_path / "upload", n=90)
    for p in list(d.iterdir()):
        if "right" in p.name:
            p.unlink()
    out = tmp_path / "episodes"
    rep = formats.convert(tmp_path / "upload", "teleop_arms", out, "one_arm", float("inf"), grouping={})
    assert [e["state_kind"] for e in rep["episodes"]] == ["joints"]
    ep = out / rep["episodes"][0]["episode_id"]
    z = np.load(ep / "state.npz")
    assert z["state"].shape[1] == 7 and "action" in z.files
    me.build_request(ep)


def test_a_single_arm_is_named_after_its_mounted_camera_not_an_extra_one(tmp_path):
    """One arm (7 values) seen by a scene camera, its wrist camera (left) and a third camera (extra1). The arm is the
    wrist camera's (only left and right count as mounted, a431c8d); capture_qc already takes it so (actor_views), so
    recorded_jumps must compare the arm with that camera and the prompt must name the arm after it."""
    rng = np.random.default_rng(1)
    T = 120
    s = np.cumsum(rng.normal(0, 0.01, (T, 7)), axis=0)
    d = write_episode(tmp_path / "episode_000000", s,
                      {"exo": np.full(T, 100.0), "left": np.full(T, 110.0), "extra1": np.full(T, 120.0)},
                      profile="teleop_arms", kind="joints",
                      cameras={"exo": {"name": "top"}, "left": {"name": "wrist"}, "extra1": {"name": "front"}})
    ep = me.load(d)
    assert me.views(ep) == ["exo", "left", "extra1"]
    from checks import capture_qc
    assert capture_qc.actor_views(ep, me.actors(ep)) == ["left"]
    assert me.actors(ep) == ["wrist"]


def test_a_single_arm_upload_with_a_third_camera_names_the_arm_after_its_wrist_camera(tmp_path):
    """The same through the reader: a one-arm LeRobot dataset (7 values, no action) with a top, a wrist and a front
    camera. The prompt must not tell the model the arm is called after the front camera."""
    from prepare import formats
    from test_prepare import _lerobot_v21, _mp4, W, H
    root = tmp_path / "one_arm"
    _lerobot_v21(root, n=45, dims=7)
    import pandas as pd
    p = root / "data" / "chunk-000" / "episode_000000.parquet"
    df = pd.read_parquet(p).drop(columns=["action"])
    df.to_parquet(p)
    info = json.loads((root / "meta" / "info.json").read_text())
    info["features"].pop("action")
    info["features"]["observation.images.cam_front"] = dict(info["features"]["observation.images.cam_high"])
    (root / "meta" / "info.json").write_text(json.dumps(info))
    _mp4(root / "videos" / "chunk-000" / "observation.images.cam_front" / "episode_000000.mp4", 45, shade=50)
    out = tmp_path / "episodes"
    rep = formats.convert(root, "teleop_arms", out, "one_arm", float("inf"), grouping={})
    ep = out / rep["episodes"][0]["episode_id"]
    e = me.load(ep)
    assert me.views(e) == ["exo", "left", "extra1"] and me.state_kind(e) == "joints"
    wrist = me.cam_name(e, "left")
    prompt = me.build_request(ep)["prompt"]
    assert f'the "arm" field always names the one arm: "{wrist}"' in prompt, [l for l in prompt.splitlines() if '"arm" field' in l]


def test_a_galaxea_episode_with_seven_joint_arms_is_labelled_from_the_video(tmp_path):
    """Two published Galaxea episodes record 7-joint arms, 16 state values, which the checks and still spans cannot
    read as two 7-value arms: the adapter labels such an episode from the video with a note, as the generic reader
    does (formats.state_layout), keeps the arm columns as signals, and its request builds."""
    import pandas as pd
    from prepare import galaxea
    from test_prepare import _mp4
    n = 12
    root = tmp_path / "collection"
    for key in galaxea.VIDEO_KEYS.values():
        _mp4(root / f"{key}.mp4", n, pts=[k * 1024 for k in range(n)])        # the 15 fps grid on 1/15360
    arm = lambda: list(np.cumsum(np.full((n, 7), 0.01), axis=0))
    df = pd.DataFrame({"observation.state.left_arm": arm(), "observation.state.right_arm": arm(),
                       "observation.state.left_gripper": np.linspace(0, 90, n),
                       "observation.state.right_gripper": np.linspace(0, 90, n),
                       "action.left_arm": arm(), "action.right_arm": arm(), "action.left_gripper": np.zeros(n),
                       "action.right_gripper": np.zeros(n), "coarse_task_index": 0, "task_index": 0,
                       "quality_index": 1})
    df.to_parquet(root / "episode.parquet")
    meta = {"folder": "Put_The_Items_Into_The_Storage_Box_20250929_002", "tasks": {0: "put the items away", 1: "good"},
            "info": {"data_path": "episode.parquet", "video_path": "{video_key}.mp4", "chunks_size": 1000}}
    ep = tmp_path / "episode_000004"
    ctx = galaxea.write_episode(meta, {"episode_index": 4}, lambda rel: root / rel, ep, "RogersPyke/Galaxea")
    assert ctx["state_kind"] == "none" and "values per frame" in ctx["state_note"]
    assert {"observation.state.left_arm", "observation.state.right_arm"} <= {s["name"] for s in ctx["signals"]}
    me.build_request(ep)


def test_the_sped_up_lag_is_a_time_at_the_recordings_own_rate():
    """3.4 frames at 30 fps is 113 ms: at 15 fps that delay is 1.7 frames, so a 15 fps recording lagging 3.4 frames
    (227 ms, a normal follower) is not sped up, and one lagging 1.7 is held to the same rule as 3.4 at 30 fps."""
    from checks import timebase as tb
    assert tb.is_sped_up(3.4, 0.03, 0.0, 3.4) is True and tb.rule_text(30) == tb.SPEDUP_RULE
    assert tb.is_sped_up(3.4, 0.03, 0.0, 3.4, fps=15) is False
    assert tb.is_sped_up(1.7, 0.03, 0.0, 1.7, fps=15) is True
    assert "1.7 frames at 15 fps" in tb.rule_text(15)



def test_a_pairing_check_that_crashes_is_recorded_as_errored(tmp_path, monkeypatch):
    """A crash in one of the three checks is that check's result, with the error, never a missing one."""
    def boom(d):
        raise ValueError("boom")
    monkeypatch.setitem(stream_pairing.MODES, "jumps", ("recorded_jumps", boom, "flagged"))
    d, r, err = stream_pairing._safe("jumps", str(tmp_path / "episode_000000"))
    assert err is None and r == {"error": "ValueError: boom", "flagged": False}


def test_a_sensor_check_that_crashes_is_errored_and_the_others_are_kept(tmp_path, monkeypatch):
    from checks import sensors as sc
    from test_board_sensors import _episode
    ep = _episode(tmp_path / "eps")
    ctx = json.loads((ep / "context.json").read_text())
    (ep / "context.json").write_text(json.dumps({**ctx, "state_kind": "none"}))

    def boom(e):
        raise ValueError("boom")
    monkeypatch.setattr(sc, "clock_findings", boom)
    r = sc.run_episode(ep)
    st = {c["check"]: c for c in r["checks"]}
    assert st["clock_offset"]["status"] == "errored" and "ValueError: boom" in st["clock_offset"]["error"]
    assert st["constant"]["status"] in ("fired", "clear") and st["no_reading"]["status"] in ("fired", "clear")


def test_the_page_shows_a_check_that_crashed_as_an_error():
    r = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("errored_checks.js")),
                        str(Path(__file__).resolve().parent.parent / "board" / "serve.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
