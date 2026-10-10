"""Independent signals and depth follow the selected camera clock in every consumer."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from board import clips
from label import depth, episode, pieces
from prepare import formats
from test_container_camera_clocks import convert, hdf_upload


def clock(pattern):
    if pattern == "unique":
        return np.arange(40, dtype=float) / 10
    return np.repeat(np.arange(4, dtype=float), 10) if pattern == "paired" else np.zeros(40)


def recording(home, colour, *, force=False, depth_clock=None):
    upload = hdf_upload(home, clock(colour))
    with h5py.File(upload / "recording.h5", "a") as f:
        if force:
            f["sensors/frame_timestamps"] = clock("unique")
            f["sensors/force"] = np.arange(40, dtype=np.float32)
        if depth_clock:
            f["depth/timestamps"] = clock(depth_clock)
            f["depth/top"] = np.stack([np.full((64, 64), 1000 + k * 10, np.uint16) for k in range(40)])
    return convert(upload, home)


def board_depth(unit, home, monkeypatch, camera="exo"):
    job = next(j for j in clips.episode_jobs(unit, home / "clips", True) if j[-1] == camera)
    assert clips.extract_one(*job[:4], clips.find_ffmpeg(), 1, *job[4:12]) is None
    selected = []
    picture = depth.picture

    def trace(array, *args, **kwargs):
        selected.append(int(array[0, 0]))
        return picture(array, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(depth, "picture", trace)
        issues = clips.extract_depth(unit, camera, Path(job[3]), home / (unit.name + "_depth.mp4"), 1, 10)
    assert not issues
    return selected


@pytest.mark.parametrize("pattern", ["unique", "paired", "all"])
def test_an_independent_signal_clock_retains_every_value_on_the_selected_camera_clock(tmp_path, pattern):
    ep, ctx = recording(tmp_path, pattern, force=True)
    signal = next(s for s in ctx.get("signals", []) if s["name"] == "sensors/force")
    with np.load(ep / "signals.npz") as z:
        np.testing.assert_array_equal(z[signal["key"]].ravel(), np.arange(40))
    if pattern != "unique":
        assert signal["camera_aligned_by"] == "assumed camera clock"
        assert "sensors/force" in episode.build_request(ep)["prompt"]


@pytest.mark.parametrize("colour,dep", [("unique", "unique"), ("paired", "paired"),
                                       ("paired", "unique"), ("all", "unique"), ("unique", "paired")])
def test_independent_depth_selects_every_original_picture_in_model_and_board(tmp_path, monkeypatch, colour, dep):
    ep, ctx = recording(tmp_path, colour, depth_clock=dep)
    expected = np.load(ep / "depth_kmap_exo.npy")
    np.testing.assert_array_equal(expected, np.arange(40))
    entry = depth.load(ep)["exo"]
    got = depth.decode(entry, entry["pts"], list(range(40)))
    for k in range(40):
        np.testing.assert_array_equal(got[k], np.full((64, 64), 1000 + k * 10, np.uint16))
    np.testing.assert_array_equal(board_depth(ep, tmp_path, monkeypatch), 1000 + expected * 10)


@pytest.mark.parametrize("colour", ["unique", "paired", "all"])
def test_short_and_end_parts_keep_independent_depth_on_the_selected_clock(tmp_path, monkeypatch, colour):
    ep, ctx = recording(tmp_path, colour, depth_clock="unique")
    ctx["extension"] = {"notes": ["Keep this original note"], "unknown": {"value": 17}}
    (ep / "context.json").write_text(json.dumps(ctx))
    monkeypatch.setattr(pieces, "piece_max", lambda _: 2.0)
    parts = pieces.write_pieces(ep, tmp_path / "parts")
    assert len(parts) == 2
    for part in parts:
        context = json.loads((part / "context.json").read_text())
        assert context["extension"] == ctx["extension"]
        expected = np.load(part / "depth_kmap_exo.npy")
        np.testing.assert_array_equal(board_depth(part, tmp_path, monkeypatch), 1000 + expected * 10)
        assert episode.build_request(part)["timesteps"]


@pytest.mark.parametrize("colour", ["unique", "paired", "all"])
def test_trim_keeps_independent_depth_on_the_selected_clock_and_original_native_arrays(tmp_path, monkeypatch, colour):
    ep, ctx = recording(tmp_path, colour, depth_clock="unique")
    native = ep / ctx["recorded_container_times"]["file"]
    before = hashlib.sha256(native.read_bytes()).hexdigest()
    trimmed = formats.trim_episode(ep, 2.0)
    assert hashlib.sha256(native.read_bytes()).hexdigest() == before
    expected = np.load(ep / "depth_kmap_exo.npy")[:trimmed["n_state_frames"]]
    assert len(expected) == 20
    np.testing.assert_array_equal(board_depth(ep, tmp_path, monkeypatch), 1000 + expected * 10)
    assert episode.build_request(ep)["timesteps"]


def test_camera_removal_keeps_independent_unique_depth_on_the_selected_native_clock(tmp_path, monkeypatch):
    upload = tmp_path / "upload"
    upload.mkdir()
    origin, delta = 1_790_000_000_000_000_003, 150_000_003
    with h5py.File(upload / "recording.h5", "w") as f:
        f.attrs["fps"] = 10
        for camera, offset in [("top", 0), ("wrist_left", delta)]:
            path = camera + ("/timestamps" if camera == "top" else "/frame_timestamps")
            ds = f.create_dataset(path, data=origin + np.repeat(np.arange(4, dtype=np.int64), 10) * 1_000_000_000 + offset)
            ds.attrs["units"] = "ns"
            f[camera + "/" + camera + "_images"] = np.stack([np.full((36, 64, 3), k * 5, np.uint8) for k in range(40)])
        ds = f.create_dataset("depth/time", data=origin + np.arange(40, dtype=np.int64) * 100_000_000 + delta)
        ds.attrs["units"] = "ns"
        f["depth/wrist_left"] = np.stack([np.full((64, 64), 1000 + 10 * k, np.uint16) for k in range(40)])
    ep, ctx = convert(upload, tmp_path)
    native = ep / ctx["recorded_container_times"]["file"]
    before = hashlib.sha256(native.read_bytes()).hexdigest()
    with np.load(ep / "depth_times.npz") as z:
        pts = z["depth_left_pts"].copy()
    clips.drop_cameras(ep, ["exo"])
    assert hashlib.sha256(native.read_bytes()).hexdigest() == before
    with np.load(ep / "depth_times.npz") as z:
        np.testing.assert_array_equal(z["depth_left_pts"], pts)
    np.testing.assert_array_equal(np.load(ep / "depth_kmap_left.npy"), np.arange(40))
    np.testing.assert_array_equal(board_depth(ep, tmp_path, monkeypatch, "left"), 1000 + np.arange(40) * 10)
    assert "assumed presentation" in episode.build_request(ep)["prompt"]


def test_unmarked_reanchor_keeps_its_existing_supplied_depth_clock_contract(tmp_path):
    t = np.arange(40, dtype=float) / 10
    (tmp_path / "depth.json").write_text(json.dumps({"left": {"kmap": "depth_kmap_left.npy"}}))
    np.save(tmp_path / "depth_kmap_left.npy", np.arange(40))
    ctx = {"fps": 10, "n_state_frames": 40, "state_kind": "none"}
    sources = {"left": {"n_frames": 40}}
    clips.reanchor(tmp_path, ctx, sources, {"exo": t, "left": t, "depth_left": t}, "exo", "left", "top")
    np.testing.assert_array_equal(np.load(tmp_path / "depth_kmap_left.npy"), np.arange(40))


def test_an_excluded_camera_clock_issue_keeps_its_actual_source_and_omission_in_the_prompt(tmp_path):
    ep, ctx = recording(tmp_path, "all")
    excluded = next(e for e in ctx["unshown_cameras"] if "infrared" in e["name"])
    issue = next(i for i in ctx["reader_issues"] if i.get("camera") == excluded["name"])
    request = episode.build_request(ep)
    text = "\n".join(p["text"] for p in request["content"] if p.get("type") == "text")
    assert excluded["name"] + " is not shown to the model" in text
    assert issue["what"] in text
    assert "infrared" not in request["cam_labels"]
