"""A tactile pad stored as an RGB image keeps its spatial layout as height x width maps.

The Inspire image variant stores each tactile pad as an H x W x 3 PNG. The full-array comparison accepts one value per
cell, so no pad reached it, and the selected components were the three channels of one cell side by side. These tests
store pads the same way: grey pads whose three channels are equal, and fingertip pads whose channels differ.
"""
import json
import tempfile
from pathlib import Path

import numpy as np

from label import episode as me
from label import sensor_evidence as se
from prepare import formats as f
from test_formats import _lerobot, _png

N = 30


def pads():
    rng = np.random.default_rng(5)
    grey = np.full((N, 12, 8), 40, np.uint8)
    grey[N // 2:, 2:6, 3:5] = np.linspace(60, 200, N - N // 2).astype(np.uint8)[:, None, None]
    grey[:, 0, 0] = rng.integers(0, 30, N)
    tip = rng.integers(0, 255, (N, 3, 3, 3)).astype(np.uint8)
    tip[..., 1] = tip[..., 0]                    # two equal channels and one that differs
    return grey, tip


def prepared(root: Path):
    grey, tip = pads()
    _lerobot(root / "input", {0: {
        "observation.state": list(np.zeros((N, 14))),
        "observation.tactiles.left_pad": [{"bytes": _png(np.repeat(g[..., None], 3, axis=2))} for g in grey],
        "observation.tactiles.left_tip": [{"bytes": _png(t)} for t in tip]}},
        feats={"observation.tactiles.left_pad": {"dtype": "image", "shape": [12, 8, 3]},
               "observation.tactiles.left_tip": {"dtype": "image", "shape": [3, 3, 3]}})
    report = f.convert(root / "input", "teleop_arms", root / "out", "test", 900)
    assert not report["failed"], report
    return root / "out" / report["episodes"][0]["episode_id"], grey, tip


def test_identical_channels_are_one_map_and_differing_channels_are_each_their_own():
    with tempfile.TemporaryDirectory() as directory:
        ep, grey, tip = prepared(Path(directory))
        ctx = json.loads((ep / "context.json").read_text())
        arrays = np.load(ep / "signals.npz")
        signals = {m["name"]: m for m in ctx["signals"] if "tactiles" in m["name"]}
        pad = signals["observation.tactiles.left_pad"]
        assert pad["shape"] == [12, 8] and pad["image_channels"] == {"stored": 3, "channels": [0, 1, 2],
                                                                     "identical": True}
        assert np.array_equal(arrays[pad["key"]][:N].reshape(N, 12, 8), grey)
        both, third = (signals["observation.tactiles.left_tip channels 0,1"],
                       signals["observation.tactiles.left_tip channel 2"])
        assert both["shape"] == third["shape"] == [3, 3]
        assert np.array_equal(arrays[both["key"]][:N].reshape(N, 3, 3), tip[..., 0])
        assert np.array_equal(arrays[third["key"]][:N].reshape(N, 3, 3), tip[..., 2])


def test_a_grey_rgb_pad_reaches_the_whole_array_comparison_with_one_value_per_cell():
    with tempfile.TemporaryDirectory() as directory:
        ep, grey, _ = prepared(Path(directory))
        loaded = me.load(ep)
        doc = se.build(loaded, {"n": N, "ks": [0, 10, 20, N - 1]})
        sensor = next(s for s in doc["sensors"] if s["name"] == "observation.tactiles.left_pad")
        array = sensor.get("array_evidence")
        assert array and array["shape"] == [12, 8] and len(array["frames"][0]["values"]) == 96
        rows = [[float(v) for v in g.ravel()] for g in grey]
        assert all(frame["values"] in rows for frame in array["frames"])
        columns = [c for r in doc["series"] if r["sensor_id"] == sensor["id"] for c in r.get("columns") or []
                   if isinstance(r.get("columns"), list)]
        assert columns and all(0 <= c < 96 for c in columns) and len(set(columns)) == len(columns)
