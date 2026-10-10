"""Independent camera timing keeps every ordered native actor claim."""
import json

import h5py
import numpy as np
import pytest

from label import episode, pieces
from test_container_camera_clocks import convert, hdf_upload


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("known", [False, True])
def test_selected_camera_clock_keeps_native_actor_evidence_and_independent_rows(tmp_path, monkeypatch, tied, known):
    origin = 1_790_000_000_000_000_003
    unique = origin + np.arange(40, dtype=np.int64) * 100_000_000
    raw = np.full(40, origin, np.int64) if tied else unique
    upload = hdf_upload(tmp_path, raw)
    data = np.tile(np.array([.1] * 6 + [.25] + [.9] * 6 + [.75], np.float32), (40, 1))
    names = ([f"right_joint{k}" for k in range(1, 7)] + ["right_gripper"]
             + [f"left_joint{k}" for k in range(1, 7)] + ["left_gripper"] if known
             else [f"joint{k}" for k in range(1, 7)] + ["gripper"])
    if not known:
        names *= 2
    with h5py.File(upload / "recording.h5", "a") as f:
        f.create_dataset("observations/timestamps", data=unique).attrs["units"] = "ns"
        f.create_dataset("observations/qpos", data=data).attrs["names"] = names
        f.create_dataset("sensors/frame_timestamps", data=unique).attrs["units"] = "ns"
        f["sensors/force"] = np.arange(40, dtype=np.float32)
    unit, ctx = convert(upload, tmp_path)
    native = ctx["recorded_container_times"]
    with np.load(unit / native["file"]) as clocks:
        np.testing.assert_array_equal(clocks["timestamps"], raw)
    signal = next(s for s in ctx["signals"] if s["name"] == "sensors/force")
    with np.load(unit / "signals.npz") as arrays:
        np.testing.assert_array_equal(arrays[signal["key"]].ravel(), np.arange(40))
    identities = ctx.get("state_identities")
    assert identities and [i["status"] for i in identities] == (["known"] * 2 if known else ["absent"] * 2)
    assert [i["side"] for i in identities] == (["right", "left"] if known else [None, None])
    if tied:
        assert signal["camera_aligned_by"] == "assumed camera clock"
        assert "assumed presentation" in episode.build_request(unit)["prompt"]
    expected = ["right", "left"] if known else ["recorded arm 1 (side unknown)", "recorded arm 2 (side unknown)"]
    loaded = episode.load(unit)
    if loaded["state"] is not None and ctx["state_kind"] != "none":
        assert episode.actors(loaded) == expected
    else:
        assert ctx["state_why"] == "assumed_clock"
        assert "RECORDED MOTION:" not in episode.build_request(unit)["prompt"]
    monkeypatch.setattr(pieces, "piece_max", lambda _: 2.0)
    for part in pieces.write_pieces(unit, tmp_path / "parts"):
        saved = json.loads((part / "context.json").read_text())
        assert saved["state_identities"] == identities
        part_episode = episode.load(part)
        if part_episode["state"] is not None and saved["state_kind"] != "none":
            assert episode.actors(part_episode) == expected
        else:
            assert saved["state_why"] == "assumed_clock"
