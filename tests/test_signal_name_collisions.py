"""Two sources that record a field of the same name both reach the episode; neither replaces the other.

Two real ActionSense HDF5 logs define the same left and right glove arrays. Generic preparation merged signals by
name and the later file replaced the earlier one, so renaming the files changed which recording supplied both gloves.
These tests rebuild that shape: two glove logs beside one video, sharing no clock with it, with the same array paths
and different readings.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import formats as f
from test_formats import _clip

GLOVES = ("tactile-glove-left/tactile_data/data", "tactile-glove-right/tactile_data/data")


def glove_log(path: Path, n: int, seed: int, t0: float) -> dict:
    """One wearable log in the ActionSense layout: each glove a group with its 32 x 32 readings and their clock."""
    import h5py
    rng = np.random.default_rng(seed)
    arrays = {}
    with h5py.File(path, "w") as h:
        for name in GLOVES:
            arrays[name] = rng.integers(500, 700, (n, 32, 32)).astype(np.float64)
            h[name] = arrays[name]
            h[name.rsplit("/", 1)[0] + "/time_s"] = t0 + np.arange(n) / 30
    return arrays


def prepared_gloves(tmp_path: Path, first: str, second: str) -> tuple[dict, list]:
    """Prepare a video beside two glove logs named first and second; returns {source file: {glove: readings}} and the
    collision issues."""
    root = tmp_path / f"upload_{first}"
    root.mkdir()
    _clip(root / "head.mp4", 30)
    sources = {first: glove_log(root / f"{first}.hdf5", 30, 1, 1_790_000_000.0),
               second: glove_log(root / f"{second}.hdf5", 30, 2, 1_790_000_500.0)}
    rep = f.convert(root, "ego_head", tmp_path / f"eps_{first}", "test", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 1, rep
    ep = tmp_path / f"eps_{first}" / rep["episodes"][0]["episode_id"]
    ctx = json.loads((ep / "context.json").read_text())
    arrays = np.load(ep / "signals.npz")
    found = {}
    for meta in ctx["signals"]:
        a = arrays[meta["key"]].reshape(30, 32, 32)
        for file, gloves in sources.items():
            for glove, readings in gloves.items():
                if np.array_equal(a, readings):
                    assert glove in meta["name"] and file in meta["name"], (meta["name"], file, glove)
                    found.setdefault(file, {})[glove] = meta["name"]
    collisions = [i for i in ctx.get("reader_issues") or [] if i["kind"] == "signal_name_collision"]
    return found, collisions


def test_two_logs_with_the_same_glove_names_both_reach_the_episode_in_either_file_order(tmp_path):
    for first, second in (("activities", "calibration"), ("zz_activities", "aa_calibration")):
        found, collisions = prepared_gloves(tmp_path, first, second)
        assert set(found) == {first, second}, found
        assert all(set(gloves) == set(GLOVES) for gloves in found.values()), found
        assert len({name for gloves in found.values() for name in gloves.values()}) == 4, found
        for glove in GLOVES:
            said = [i["what"] for i in collisions if i["what"].startswith(glove + " ")]
            assert len(said) == 1 and first in said[0] and second in said[0], collisions


def test_a_merge_never_replaces_a_held_signal_with_another_sources_readings():
    held, more = f.Signals(), f.Signals()
    held.add("glove", np.zeros((5, 4)))
    held.add("same", np.full((5, 2), 3.0))
    held.origins = {"glove": "left_log.hdf5", "same": "left_log.hdf5"}
    more.add("glove", np.ones((5, 4)))
    more.add("same", np.full((5, 2), 3.0))
    more.origins = {"glove": "right_log.hdf5", "same": "right_log.hdf5"}
    f.merge_signals(held, more)
    assert set(held) == {"left_log glove", "right_log glove", "same"}
    assert (held["left_log glove"] == 0).all() and (held["right_log glove"] == 1).all()
    assert [i["kind"] for i in held.issues] == ["signal_name_collision"]
    # with the held source unknown, it keeps its name and the second still arrives, under its file's name
    plain, more = f.Signals(), f.Signals()
    plain.add("imu", np.zeros((5, 3)))
    more.add("imu", np.ones((5, 3)))
    more.origins = {"imu": "wrist.mcap"}
    f.merge_signals(plain, more)
    assert set(plain) == {"imu", "wrist imu"} and len(plain.issues) == 1


def test_two_tables_whose_names_give_the_same_signal_are_both_kept(tmp_path):
    n = 60
    pd.DataFrame({"x": np.arange(n) * 0.5}).to_csv(tmp_path / "imu_1.csv", index=False)
    pd.DataFrame({"x": np.arange(n) * -2.0}).to_csv(tmp_path / "imu_2.csv", index=False)
    out = f.table_signals([tmp_path / "imu_1.csv", tmp_path / "imu_2.csv"], None,
                          {"pts": np.arange(n, dtype=np.int64), "time_base": 1 / 30.0}, {})
    assert set(out) == {"imu_1 imu", "imu_2 imu"}, set(out)
    assert out["imu_1 imu"][10, 0] == 5.0 and out["imu_2 imu"][10, 0] == -20.0
