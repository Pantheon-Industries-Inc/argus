"""A signal keeps its identity, its source file and field, wherever it is read, joined or renamed.

Three ways the same class of loss could still happen after the first fixes: an episode prepared before the
channel-plane rule still held H x W x 3 pads, MCAP files pooled every channel of one name whatever their times, and
a note written before a rename kept naming the signal by its old name.
"""
import json
import tempfile
from pathlib import Path

import numpy as np

from board import sensors as board_sensors
from label import episode as me
from label import sensor_evidence as se
from prepare import formats as f
from test_formats import _clip, _json_mcap
from test_image_channel_planes import N, prepared
from test_signal_name_collisions import GLOVES, glove_log


def as_prepared_before_planes(ep: Path) -> None:
    """Rewrite a prepared episode's image signals the way a preparation before the channel-plane rule stored them:
    one descriptor per image column, its rows height x width x channels, with no image_channels."""
    ctx = json.loads((ep / "context.json").read_text())
    with np.load(ep / "signals.npz") as z:
        arrays = {k: z[k] for k in z.files}
    signals, columns = [], {}
    for m in ctx["signals"]:
        channels = m.pop("image_channels", None)
        if channels is None:
            signals.append(m)
            continue
        column = m["source"].removeprefix("image column ")
        columns.setdefault(column, {"meta": m, "planes": []})["planes"].append((channels["channels"], arrays.pop(m["key"])))
    for column, entry in columns.items():
        h, w = entry["meta"]["shape"]
        c = max(i for chans, _ in entry["planes"] for i in chans) + 1
        cube = np.zeros((len(entry["planes"][0][1]), h, w, c))
        for chans, rows in entry["planes"]:
            for i in chans:
                cube[..., i] = rows.reshape(len(rows), h, w)
        key = entry["meta"]["key"]
        arrays[key] = cube.reshape(len(cube), -1)
        signals.append({**entry["meta"], "name": column, "shape": [h, w, c], "dims": h * w * c})
    ctx["signals"] = signals
    ctx["reader_issues"] = ctx.get("reader_issues", []) + [
        {"kind": "signal_bad_cells", "signal": "observation.tactiles.left_tip", "what": "observation.tactiles.left_tip "
         "has 1 of 30 images that could not be decoded"}]
    np.savez(ep / "signals.npz", **arrays)
    (ep / "context.json").write_text(json.dumps(ctx))


def test_an_episode_prepared_before_the_plane_rule_reads_as_planes_for_labelling_and_the_board():
    with tempfile.TemporaryDirectory() as directory:
        ep, grey, tip = prepared(Path(directory))
        as_prepared_before_planes(ep)
        stored = {m["name"]: m.get("shape") for m in json.loads((ep / "context.json").read_text())["signals"]}
        assert stored["observation.tactiles.left_pad"] == [12, 8, 3]
        loaded = me.load(ep)
        pad = loaded["signals"]["observation.tactiles.left_pad"]
        assert pad.shape == (N, 96) and np.array_equal(pad.reshape(N, 12, 8), grey)
        assert np.array_equal(loaded["signals"]["observation.tactiles.left_tip channels 0,1"].reshape(N, 3, 3),
                              tip[..., 0])
        assert np.array_equal(loaded["signals"]["observation.tactiles.left_tip channel 2"].reshape(N, 3, 3), tip[..., 2])
        notes = [i["signal"] for i in loaded["context"]["reader_issues"] if i["kind"] == "signal_bad_cells"
                 and "left_tip" in i["signal"]]
        assert sorted(notes) == ["observation.tactiles.left_tip channel 2", "observation.tactiles.left_tip channels 0,1"]
        doc = se.build(loaded, {"n": N, "ks": [0, 10, 20, N - 1]})
        sensor = next(s for s in doc["sensors"] if s["name"] == "observation.tactiles.left_pad")
        assert sensor["array_evidence"]["shape"] == [12, 8]
        board = board_sensors.episode_doc(ep)
        names = {s["name"] for s in board["signals"]}
        assert {"observation.tactiles.left_pad", "observation.tactiles.left_tip channel 2"} <= names, names


def test_mcap_files_of_one_channel_join_when_their_times_do_not_overlap_and_stay_apart_when_they_do(tmp_path):
    t0 = 1_790_000_000.0
    q = t0 + np.arange(120) / 30

    def piece(path, start, stop, scale):
        _json_mcap(path, {"/wrench": [(s, {"force": [scale * s, 1.0, 2.0]})
                                      for s in np.arange(start, stop, 1 / 60)]}, t0)
    piece(tmp_path / "left.mcap", 0.0, 4.0, 1.0)
    piece(tmp_path / "right.mcap", 0.0, 4.0, -3.0)
    apart = f.mcap_signals([tmp_path / "left.mcap", tmp_path / "right.mcap"], q)
    forces = sorted(k for k in apart if "/wrench" in k and "variation" not in k)
    assert forces == ["left /wrench force", "right /wrench force"], list(apart)
    assert apart[forces[0]][60, 0] > 0 > apart[forces[1]][60, 0]
    said = [i for i in apart.issues if i["kind"] == "signal_name_collision"]
    assert any("left.mcap" in i["what"] and "right.mcap" in i["what"] and "overlap" in i["what"] for i in said), said

    piece(tmp_path / "part_a.mcap", 0.0, 2.0, 1.0)
    piece(tmp_path / "part_b.mcap", 2.05, 4.0, 1.0)
    joined = f.mcap_signals([tmp_path / "part_a.mcap", tmp_path / "part_b.mcap"], q)
    forces = [k for k in joined if k.startswith("/wrench") and "variation" not in k]
    assert len(forces) == 1, list(joined)
    rows = joined[forces[0]]
    assert np.isfinite(rows[:58]).all() and np.isfinite(rows[64:118]).all()   # both pieces, around their 50ms gap
    assert joined.meta[forces[0]]["joined_from"] == ["part_a.mcap", "part_b.mcap"]
    assert not [i for i in joined.issues if i["kind"] == "signal_name_collision"]


def test_a_note_written_before_a_rename_follows_the_signal_it_is_about(tmp_path):
    # two tables whose names both give imu: the first one's placement note is written before the second one arrives
    # and renames it
    import pandas as pd
    n = 60
    pd.DataFrame({"x": np.arange(n) * 0.5}).to_csv(tmp_path / "imu_1.csv", index=False)
    pd.DataFrame({"x": np.arange(n) * -2.0}).to_csv(tmp_path / "imu_2.csv", index=False)
    extra = {}
    out = f.table_signals([tmp_path / "imu_1.csv", tmp_path / "imu_2.csv"], None,
                          {"pts": np.arange(n, dtype=np.int64), "time_base": 1 / 30.0}, extra)
    notes = [i for i in extra.get("reader_issues", []) + out.issues if i["kind"] == "signal_alignment_assumed"]
    assert sorted(i["signal"] for i in notes) == ["imu_1 imu", "imu_2 imu"], notes
    assert all(i["signal_source"] == i["signal"].split()[0] + ".csv" and i["signal_field"] == "imu" for i in notes)

    root = tmp_path / "upload"
    root.mkdir()
    _clip(root / "head.mp4", 30)
    glove_log(root / "first.hdf5", 30, 1, 1_790_000_000.0)
    glove_log(root / "second.hdf5", 30, 2, 1_790_000_500.0)
    rep = f.convert(root, "ego_head", tmp_path / "eps", "test", 900)
    ctx = json.loads((tmp_path / "eps" / rep["episodes"][0]["episode_id"] / "context.json").read_text())
    names = {m["name"] for m in ctx["signals"]}
    for file in ("first", "second"):
        for glove in GLOVES:
            name = f"{file} {glove}"
            assert name in names, names
            notes = [i for i in ctx["reader_issues"] if i["kind"] == "signal_alignment_assumed" and i["signal"] == name]
            assert len(notes) == 1 and notes[0]["what"].startswith(name), notes
            assert notes[0]["signal_source"] == f"{file}.hdf5" and notes[0]["signal_field"] == glove
    assert not [i for i in ctx["reader_issues"] if i.get("signal") in GLOVES], "a note still names a renamed signal"


def test_an_arms_joint_stream_follows_the_same_rule_across_mcap_files(tmp_path):
    t0 = 1_790_000_000.0
    q = t0 + np.arange(120) / 30

    def arm(path, start, stop, scale):
        _json_mcap(path, {"/yam_left/joint_state": [
            (s, {"name": [f"joint{i}" for i in range(6)], "position": [scale * s + i for i in range(6)]})
            for s in np.arange(start, stop, 1 / 60)]}, t0)
    arm(tmp_path / "left.mcap", 0.0, 4.0, 1.0)
    arm(tmp_path / "right.mcap", 0.0, 4.0, -3.0)
    apart = f.mcap_joint_streams([tmp_path / "left.mcap", tmp_path / "right.mcap"], q)
    assert sorted(apart) == ["left /yam_left/joint_state", "right /yam_left/joint_state"], list(apart)
    notes = []
    apart = f.mcap_joint_streams([tmp_path / "left.mcap", tmp_path / "right.mcap"], q, notes=notes)
    assert {s["source"] for s in apart.values()} == {"left.mcap", "right.mcap"}
    assert apart["right /yam_left/joint_state"]["pos"][60, 0] < 0 < apart["left /yam_left/joint_state"]["pos"][60, 0]
    assert len(notes) == 1 and "overlap" in notes[0]["what"] and notes[0]["kind"] == "signal_name_collision"

    arm(tmp_path / "part_a.mcap", 0.0, 2.0, 1.0)
    arm(tmp_path / "part_b.mcap", 2.05, 4.0, 1.0)
    notes = []
    joined = f.mcap_joint_streams([tmp_path / "part_a.mcap", tmp_path / "part_b.mcap"], q, notes=notes)
    assert list(joined) == ["/yam_left/joint_state"] and not notes
    stream = joined["/yam_left/joint_state"]
    assert stream["joined_from"] == ["part_a.mcap", "part_b.mcap"]
    assert stream["t"][0] == t0 and stream["t"][-1] > t0 + 3.9 and (np.diff(stream["t"]) > 0).all()

    # a state is never read from one of two overlapping recordings of the arm
    said, extra = [], {}
    f.mcap_joint_streams([tmp_path / "left.mcap", tmp_path / "right.mcap"], q, notes=said)
    blocked = f.collided_arms(extra, said)
    assert blocked is not None and blocked.why == "layout" and "left.mcap" in blocked
    assert [i["kind"] for i in extra["reader_issues"]] == ["signal_name_collision"]
