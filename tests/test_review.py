"""Reviewing your own data (python -m review) and the pieces of it Data Review shares: the neighbour lag measured
inside a folder, the manifest entry for your own data, what a stitched recording carries onto the board, and the
outcome and severity values the board knows."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from board import build as board_build
from board import rules
from checks import timebase
from test_formats import recorder_folder

REPO = Path(__file__).resolve().parent.parent


def test_unknown_outcomes_and_severities_are_shown_as_unclear():
    d = {"completion": {"task_completed": "mostly_done"}, "_meta": {"task_completed": "success"},
         "data_issues": [{"severity": "critical", "issue": "x"}, {"severity": "low", "issue": "<b>"}],
         "tasks": [{"outcome": "whatever"}], "operator_mistakes": [{"severity": None}]}
    out = board_build.normalize_enums(d)
    assert out["completion"]["task_completed"] == "unclear" and out["_meta"]["task_completed"] == "success"
    assert [i["severity"] for i in out["data_issues"]] == ["unclear", "low"] and out["data_issues"][1]["issue"] == "<b>"
    assert out["tasks"][0]["outcome"] == "whatever" and out["operator_mistakes"][0]["severity"] is None
    known = {"completion": {"task_completed": "success_then_undone"}, "data_issues": [{"severity": "high"}]}
    assert board_build.normalize_enums(known) == known


def test_a_stitched_recording_carries_its_parts_and_the_issues_set_aside_at_our_cuts():
    cut = {"category": "truncated_episode", "issue": "ends mid-task", "excluded_by": "piece_cut", "reason": "our cut"}
    d = {"_excluded": [{"category": "missing_instruction", "issue": "no text", "excluded_by": "no_task_text"}],
         "dataset_checks": {"timebase": {"sped_up_recording": False}}}
    r = {"stitched": {"parts": 2, "cuts_s": [300.0]}, "labels": {"_excluded": [cut]}}
    board_build.carry_pieces(d, r, {"timebase_neighbours_in_upload": 4})
    assert d["_stitched"] == {"parts": 2, "cuts_s": [300.0]}
    assert [x["excluded_by"] for x in d["_excluded"]] == ["no_task_text", "piece_cut"]
    assert d["dataset_checks"]["timebase"]["neighbours_in_upload"] == 4
    plain = {"data_issues": []}
    board_build.carry_pieces(plain, {"labels": {}}, {})
    assert plain == {"data_issues": []}                               # a label that was not stitched is unchanged


def test_own_data_entry_adds_fixed_window_only_for_fixed_length_files():
    e = rules.own_data_entry("mine", "run", "eps", "teleop_arms")
    assert e["rules"] == rules.rules_for("teleop_arms")
    w = rules.own_data_entry("mine", "run", "eps", "ego_head", {"fixed_window_s": 180.0})
    assert w["rules"][:-1] == rules.rules_for("ego_head") and w["rules"][-1]["kind"] == "fixed_window"
    assert w["rules"][-1]["window_s"] == 180.0


def _episode(root: Path, idx: int, lag: int, task: str = "stack", adapter: bool = False) -> None:
    d = root / f"episode_{idx:06d}"
    d.mkdir(parents=True)
    rng = np.random.default_rng(idx)
    a = np.cumsum(rng.normal(0, 0.01, (400, 14)), axis=0)
    s = np.roll(a, lag, axis=0)                      # the follower trails the leader by `lag` frames
    np.savez(d / "state.npz", state=s, action=a)
    ctx = {"profile": "teleop_arms", "state_kind": "joints", "task_label": [task]}
    if adapter:                                      # a dataset's own adapter keeps the index in its source
        ctx["source"] = {"folder": "upload", "episode_index": idx, "adapter": "galaxea"}
    else:
        ctx["episode_index"] = idx
    (d / "context.json").write_text(json.dumps(ctx))


def test_the_neighbour_lag_is_measured_inside_the_folder(tmp_path):
    for i in range(5):
        _episode(tmp_path, i, lag=3)
    _episode(tmp_path, 40, lag=3)                    # alone: measured, but no neighbour lag
    assert timebase.measure_folder(tmp_path) == 5
    ctx = lambda i: json.loads((tmp_path / f"episode_{i:06d}" / "context.json").read_text())
    assert abs(ctx(2)["timebase_neighbour_lag_frames"] - 3.0) < 0.2 and ctx(2)["timebase_neighbours_in_upload"] == 5
    assert "timebase_neighbour_lag_frames" not in ctx(40) and ctx(40)["timebase_neighbours_in_upload"] == 1


def test_the_neighbour_lag_is_measured_when_an_adapter_keeps_the_index_in_its_source(tmp_path):
    for i in range(4):
        _episode(tmp_path, i, lag=3, adapter=True)
    assert timebase.measure_folder(tmp_path) == 4
    ctx = json.loads((tmp_path / "episode_000001" / "context.json").read_text())
    assert abs(ctx["timebase_neighbour_lag_frames"] - 3.0) < 0.2 and ctx["timebase_neighbours_in_upload"] == 4


def test_review_runs_every_stage_on_a_recorders_folder_in_free_mode():
    """python -m review on a capture-stack folder reads it with its arm state, runs the checks and the clips, and
    builds every request without calling the model."""
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "upload"
        recorder_folder(root, n=90)
        job = Path(t) / "job"
        p = subprocess.run([sys.executable, "-m", "review", "--data", str(root), "--rig", "teleop_arms", "--out",
                            str(job), "--dataset", "mine", "--free"], cwd=REPO, capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        rep = json.loads((job / "report.json").read_text())
        assert len(rep["episodes"]) == 1 and rep["episodes"][0]["state_kind"] == "joints"
        dry = list((job / "dry").glob("episode_*.json"))
        assert len(dry) == 1 and json.loads(dry[0].read_text())["dry_run"] is True
        assert list((job / "clips").rglob("*.mp4"))


def _clip_episode(eps: Path, name: str, video: Path) -> None:
    d = eps / name
    d.mkdir(parents=True)
    (d / "sources.json").write_text(json.dumps({"exo": {"packed": str(video), "base_s": 0.0, "n_frames": 30}}))
    (d / "context.json").write_text(json.dumps({"fps": 30}))


def test_a_camera_file_that_does_not_decode_costs_only_its_own_episode(tmp_path):
    from board import clips
    good, bad = tmp_path / "good.mp4", tmp_path / "bad.mp4"
    subprocess.run([clips.find_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30",
                    "-t", "1", "-pix_fmt", "yuv420p", str(good)], check=True)
    bad.write_bytes(good.read_bytes()[:600])                    # a truncated file: no frame decodes
    eps, out = tmp_path / "episodes", tmp_path / "clips"
    _clip_episode(eps, "episode_ok", good)
    _clip_episode(eps, "episode_broken", bad)
    run = lambda: subprocess.run([sys.executable, "-m", "board", "clips", "--episodes", str(eps), "--out", str(out)],
                                 cwd=REPO, capture_output=True, text=True)
    p = run()
    assert p.returncode == 0, p.stdout + p.stderr               # the upload goes on without the broken episode
    assert (out / "episode_ok.mp4").exists() and list(json.loads((out / "failed.json").read_text())) == ["episode_broken"]
    left = clips.set_aside_failed(eps, out)
    assert [f["name"] for f in left] == ["episode_broken"] and "could not be decoded" in left[0]["why"]
    assert sorted(d.name for d in eps.iterdir()) == ["episode_ok"]
    assert (tmp_path / "episodes_unclipped" / "episode_broken").is_dir()
    assert clips.set_aside_failed(eps, out) == []               # moved once, reported once
    rep = {"episodes": [{"name": "ok", "episode_id": "episode_ok", "seconds": 1.0},
                        {"name": "broken", "episode_id": "episode_broken", "seconds": 1.0}], "failed": [], "seconds": 2.0}
    clips.drop_from_report(rep, left)
    assert [e["name"] for e in rep["episodes"]] == ["ok"] and rep["seconds"] == 1.0
    assert rep["failed"] == [{"name": "broken", "why": left[0]["why"]}]
    (eps / "episode_ok").rename(tmp_path / "episode_ok")       # only the broken one left: nothing to show
    shutil.move(str(tmp_path / "episodes_unclipped" / "episode_broken"), str(eps / "episode_broken"))
    assert run().returncode == 1


def test_an_episode_left_off_the_board_is_explained_with_the_boards_camera_names(tmp_path):
    """Only an episode none of whose cameras decodes is left out, so its reason says that once for every camera, and a
    single camera is named as the board names it (never exo or extra1)."""
    from board import clips
    ep = tmp_path / "episode_x"
    ep.mkdir()
    views = ["exo", "left", "right", "extra1", "extra2"]
    (ep / "sources.json").write_text(json.dumps({v: {} for v in views}))
    ctx = {"profile": "teleop_arms", "cameras": {"extra1": {"name": "cam_low"}, "extra2": {"name": "cam_side"}}}
    (ep / "context.json").write_text(json.dumps(ctx))
    assert clips.failed_reason({v: "Command '...' returned non-zero exit status 183." for v in views}, ep) == \
        "every camera's video could not be decoded, so this episode was left out"
    assert clips.failed_reason({"extra1": "boom"}, ep) == "the cam_low camera video could not be decoded, so this " \
                                                          "episode was left out"
    (ep / "context.json").write_text(json.dumps({"profile": "handheld_gripper"}))
    assert clips.failed_reason({"left": "boom"}, ep) == "the left gripper camera video could not be decoded, so this episode was left out"


def _video(path: Path, frames: int) -> None:
    from board import clips
    subprocess.run([clips.find_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=160x120:rate=30", "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path)],
                   check=True)


def _two_camera_upload(tmp_path: Path, top: int = 30, wrist: int = 30) -> tuple[dict, Path, Path]:
    """A video upload of one episode filmed by a top camera and a left wrist camera, read as an upload is (the
    report, the episodes folder, the episode)."""
    from prepare import formats
    up = tmp_path / "up"
    up.mkdir()
    _video(up / "top.mp4", top)
    _video(up / "wrist_left.mp4", wrist)
    eps = tmp_path / "episodes"
    rep = formats.convert(up, "teleop_arms", eps, "mine", float("inf"), grouping={})
    return rep, eps, eps / rep["episodes"][0]["episode_id"]


def _clips(eps: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "board", "clips", "--episodes", str(eps), "--out", str(out)],
                          cwd=REPO, capture_output=True, text=True)


def _board_episode(tmp_path: Path, eps: Path, ep: Path) -> dict:
    """The episode's file on a board built from a labelling run over it."""
    from test_board import _output
    run = tmp_path / "run"
    (run / "out").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": "r1", "code": "abc1234", "kind": "upload", "status": "done"}))
    (run / "out" / f"{ep.name}.json").write_text(json.dumps(_output(ep.name)))
    board = tmp_path / "board"
    board.mkdir()
    (board / "manifest.json").write_text(json.dumps({"board": "b", "datasets": [
        {"dataset": "mine", "run": str(run), "episodes": str(eps), "rules": rules.rules_for("teleop_arms")}]}))
    board_build.build(board)
    return json.loads((board / "qa" / f"{ep.name}.json").read_text())


def test_a_camera_clip_a_frame_short_keeps_the_episode_and_is_a_data_issue(tmp_path):
    """A camera whose clip comes out shorter than the episode (an upload's 63 s episode had 1890 frames in every
    camera's clip where the episode has 1891, and lost the episode) keeps its episode and its clip as cut. The
    mismatch is recorded in context.json, so it survives every rebuild, raises its data issue on the board, goes into
    the job's report notes, and the episode is labelled with that camera's cell empty past its last frame."""
    from board import clips
    from board import serve
    from label import episode as me
    rep, eps, ep = _two_camera_upload(tmp_path)
    _video(Path(json.loads((ep / "sources.json").read_text())["left"]["packed"]), 29)    # ends a frame early
    out = tmp_path / "clips"
    p = _clips(eps, out)
    assert p.returncode == 0, p.stdout + p.stderr
    assert clips.clip_frames(out / f"{ep.name}.mp4") == 30
    assert clips.clip_frames(out / "wrist_left" / f"{ep.name}.mp4") == 29
    assert json.loads((out / "failed.json").read_text()) == {}
    assert clips.set_aside_failed(eps, out) == [] and ep.is_dir()
    ctx = json.loads((ep / "context.json").read_text())
    what = "The left wrist camera video has 29 frames where the episode has 30."
    assert ctx["reader_issues"] == [{"kind": "clip_frame_count", "camera": "left", "what": what}]
    assert set(json.loads((ep / "sources.json").read_text())) == {"exo", "left"}
    d = _board_episode(tmp_path, eps, ep)
    assert d["dataset_checks"]["reader_issues"] == [{**ctx["reader_issues"][0], "family": "clip-frames"}]
    assert "clip-frames" in serve._families(d)["families"]
    clips.note_camera_problems(rep, eps)
    assert rep["notes"] == ["episode_1: the left wrist camera video has 29 frames where the episode has 30."]
    assert rep["episodes"][0]["cameras"] == {"exo": "top", "left": "left"}
    # labelling decodes the files themselves: the left camera has no frame at the last instant, so its cell is empty
    req = me.build_request(ep)
    texts = [c["text"] for c in req["content"] if c.get("type") == "text"]
    last = [t for t in texts if t.startswith("=== detail view, last frame")]
    assert len(last) == 1 and "| camera top ===" in last[0], last
    assert ("Left's video ends before the episode does, so it has no frame at 0.97 s. Its cells there are empty"
            in req["prompt"])
    # a rerun keeps the clip as cut and the record as it was
    assert _clips(eps, out).returncode == 0
    assert json.loads((ep / "context.json").read_text())["reader_issues"] == ctx["reader_issues"]


def test_a_camera_that_does_not_decode_is_taken_out_and_the_episode_kept(tmp_path):
    """The main camera's file is not a video: the episode keeps its wrist camera for labelling and the board, the main
    camera is out of sources.json and the context's cameras (so nothing tries to decode it), the wrist camera, which
    was paired to the main camera by time, is the main camera now, and the camera is recorded as not decodable."""
    from board import clips
    from board import serve
    from label import episode as me
    rep, eps, ep = _two_camera_upload(tmp_path, top=30, wrist=33)
    assert json.loads((ep / "sources.json").read_text())["left"].get("kmap"), "the wrist must be paired by time"
    Path(json.loads((ep / "sources.json").read_text())["exo"]["packed"]).write_bytes(b"not a video at all" * 50)
    ctx = json.loads((ep / "context.json").read_text())
    ctx["reader_issues"] = [{"kind": "already", "what": "an entry another step wrote is kept"}]
    (ep / "context.json").write_text(json.dumps(ctx))
    out = tmp_path / "clips"
    p = _clips(eps, out)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (out / "wrist_left" / f"{ep.name}.mp4").exists() and not (out / f"{ep.name}.mp4").exists()
    assert clips.set_aside_failed(eps, out) == [] and ep.is_dir()
    src = json.loads((ep / "sources.json").read_text())
    ctx = json.loads((ep / "context.json").read_text())
    assert list(src) == ["left"] and "kmap" not in src["left"] and list(ctx["cameras"]) == ["left"]
    what = "The main camera video could not be decoded, so this episode is shown and labelled without it."
    assert ctx["reader_issues"] == [{"kind": "already", "what": "an entry another step wrote is kept"},
                                    {"kind": "camera_not_decodable", "camera": "exo", "what": what}]
    d = _board_episode(tmp_path, eps, ep)
    fams = serve._families(d)["families"]
    assert "camera-undecodable" in fams and "d:Already" in fams        # a kind no family names is a data issue too
    clips.note_camera_problems(rep, eps)
    assert rep["notes"] == ["episode_1: the main camera video could not be decoded, so this episode is shown and "
                            "labelled without it."]
    assert rep["episodes"][0]["cameras"] == {"left": "left"}
    req = me.build_request(ep)
    assert req["views"] == ["left"] and req["cam_labels"] == ["left"]
    assert "There is exactly 1 camera" in req["prompt"] and "\n- top:" not in req["prompt"]
    assert "\n- left:" in req["prompt"]


def test_an_episode_none_of_whose_cameras_decodes_is_set_aside_with_its_reason(tmp_path):
    from board import clips
    rep, eps, ep = _two_camera_upload(tmp_path)
    for v, s in json.loads((ep / "sources.json").read_text()).items():
        Path(s["packed"]).write_bytes(b"not a video at all" * 50)
    out = tmp_path / "clips"
    assert _clips(eps, out).returncode == 1                   # no episode came out whole
    left = clips.set_aside_failed(eps, out)
    assert left == [{"name": ep.name, "why": "every camera's video could not be decoded, so this episode was left out"}]
    assert not ep.exists() and (tmp_path / "episodes_unclipped" / ep.name).is_dir()
    clips.drop_from_report(rep, left)
    assert rep["episodes"] == [] and rep["failed"] == [{"name": "episode_1", "why": left[0]["why"]}]


def test_review_says_why_when_no_episode_can_be_put_on_the_board(tmp_path, monkeypatch):
    """board clips exits 1 when no episode came out whole: review says which camera file did not decode, the
    upload's own reason, instead of reporting the clips step as a crash."""
    import pytest
    import review.__main__ as rv
    root = tmp_path / "upload"
    recorder_folder(root, n=30)
    real = rv.run_step

    def fake(job, step, cmd, env, ok_codes=(0,)):
        if step == "clips":
            assert 1 in ok_codes
            return 1
        return real(job, step, cmd, env, ok_codes)
    monkeypatch.setattr(rv, "run_step", fake)
    from board import clips as board_clips
    monkeypatch.setattr(board_clips, "set_aside_failed",
                        lambda eps, out: [{"name": p.name, "why": "the exo camera video could not be decoded"}
                                          for p in sorted(Path(eps).iterdir()) if p.name.startswith("episode_")])
    monkeypatch.setattr(board_clips, "drop_from_report", lambda rep, left: rep.update(episodes=[]))
    monkeypatch.setattr(sys, "argv", ["python -m review", "--data", str(root), "--rig", "teleop_arms", "--out",
                                      str(tmp_path / "job"), "--dataset", "mine", "--free"])
    with pytest.raises(SystemExit, match="no episode could be put on the board: the exo camera video could not be"):
        rv.main()
