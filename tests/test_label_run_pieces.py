"""python -m label cuts a recording one request cannot hold into parts and stitches their labels, as python -m review
does (label/run.py plan_pieces, stitch); a slice with no long recording runs as before."""
import json

from label import pieces, policy
from label import run as R


def episode(root, name, seconds):
    d = root / name
    d.mkdir(parents=True)
    (d / "context.json").write_text(json.dumps({"profile": "teleop_arms", "duration_s": seconds, "fps": 30}))
    return d


def test_run_policy_reads_the_harness_flags():
    got = R.run_policy(["--", "--model", "openai/gpt-6.1-sol", "--sample-fps", "1", "--cell-w", "224"])
    assert got["model"] == "openai/gpt-6.1-sol" and got["sample_fps"] == 1.0 and got["reasoning"] is None


def test_a_slice_of_short_recordings_is_left_as_it_is(tmp_path, monkeypatch):
    slice_dir = tmp_path / "slice"
    episode(slice_dir, "episode_a", 40)
    before = (slice_dir / "episode_a" / "context.json").read_text()
    monkeypatch.setattr(pieces, "write_units", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no cut")))
    assert R.plan_pieces(tmp_path / "run", slice_dir, ["--", "--model", "openai/gpt-6.1-sol"]) == {}
    assert (slice_dir / "episode_a" / "context.json").read_text() == before


def test_a_long_recording_is_cut_to_the_limit_of_the_model_that_labels_it(tmp_path, monkeypatch):
    slice_dir = tmp_path / "slice"
    episode(slice_dir, "episode_a", 40)
    episode(slice_dir, "episode_b", 604)
    calls = {}
    monkeypatch.setattr(pieces, "write_units", lambda run, eps: calls.setdefault("cut", (run, eps)) and
                        {"episode_b": ["episode_b__p1", "episode_b__p2"]})
    args = ["--", "--model", "openai/gpt-6.1-sol", "--sample-fps", "1"]
    got = R.plan_pieces(tmp_path / "run", slice_dir, args)
    assert got == {"episode_b": ["episode_b__p1", "episode_b__p2"]} and calls["cut"] == (tmp_path / "run", slice_dir)
    saved = json.loads((slice_dir / "episode_b" / "context.json").read_text())["annotation_policy"]
    assert saved["model"] == "openai/gpt-6.1-sol" and saved["part_max_s"] <= 450


def test_the_harness_labels_the_units_into_parts_when_the_run_cut(tmp_path):
    cmd = R.harness_cmd(tmp_path / "slice", tmp_path / "run", "full", 5.0, 4, [], split=True)
    assert cmd[cmd.index("--episodes-root") + 1] == str((tmp_path / "run" / "units").resolve())
    assert cmd[cmd.index("--out-dir") + 1] == str((tmp_path / "run" / "parts").resolve())
    plain = R.harness_cmd(tmp_path / "slice", tmp_path / "run", "full", 5.0, 4, [])
    assert plain[plain.index("--out-dir") + 1] == str((tmp_path / "run" / "out").resolve())
    assert R.harness_out(tmp_path, {"pieces": {"e": ["p"]}}).name == "parts" and R.harness_out(tmp_path, {}).name == "out"


def test_stitch_reads_the_parts_and_records_a_failure(tmp_path, monkeypatch):
    seen = {}

    def fake(job, eps, long_eps, out, src=None):
        seen.update(job=job, out=out, src=src, long_eps=long_eps)
        return {"stitched": 1, "incomplete": [], "unlabelled": []}

    monkeypatch.setattr(pieces, "stitch_run", fake)
    info = {"pieces": {"episode_b": ["episode_b__p1"]}}
    R.stitch(tmp_path, info, tmp_path / "slice")
    assert seen == {"job": tmp_path, "out": tmp_path / "out", "src": tmp_path / "parts",
                    "long_eps": {"episode_b": ["episode_b__p1"]}} and info["stitched"]["stitched"] == 1
    monkeypatch.setattr(pieces, "stitch_run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("changed input")))
    R.stitch(tmp_path, info, tmp_path / "slice")
    assert "changed input" in info["stitch_error"]
    R.stitch(tmp_path, {}, tmp_path / "slice")       # nothing cut: nothing to stitch


def test_the_limit_follows_the_policy(tmp_path):
    ctx = {"profile": "teleop_arms"}
    assert policy.resolve(ctx, model="openai/gpt-6.1-sol", sample_fps=1.0)["part_max_s"] <= 450
