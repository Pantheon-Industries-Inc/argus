"""Native container capture clocks stay distinct from encoded packet clocks."""
import io
import json

import av
import h5py
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from label import episode
from board import clips, sensors
from prepare import formats


def hdf_upload(tmp_path, raw, *, signal=False):
    upload = tmp_path / "upload"
    upload.mkdir()
    with h5py.File(upload / "recording.h5", "w") as f:
        f.attrs["fps"] = 10
        clock = f.create_dataset("timestamps", data=raw)
        if np.asarray(raw).dtype.kind in "iu":
            clock.attrs["units"] = "ns"
        for offset, camera in enumerate(("top", "wrist_left", "wrist_right", "side", "back", "infrared")):
            f.create_dataset("images/" + camera, data=np.stack([
                np.full((36, 64, 3), 5 * k + 11 * offset, np.uint8) for k in range(len(raw))]))
        if signal:
            f.create_dataset("force", data=np.arange(len(raw), dtype=np.float32))
    return upload


def convert(upload, tmp_path):
    report = formats.convert(upload, "teleop_arms", tmp_path / "episodes", "test", 900)
    assert not report["failed"] and len(report["episodes"]) == 1
    ep = tmp_path / "episodes" / report["episodes"][0]["episode_id"]
    return ep, json.loads((ep / "context.json").read_text())


@pytest.mark.parametrize("integer", [False, True])
def test_hdf_native_tied_clock_is_retained_without_losing_side_camera_frames(tmp_path, integer):
    raw = np.repeat(np.arange(4), 10)
    raw = 1_790_000_000_000_000_003 + raw.astype(np.int64) * 1_000_000_000 if integer else raw.astype(float)
    ep, ctx = convert(hdf_upload(tmp_path, raw), tmp_path)
    assert ctx.get("presentation_times")
    native = ctx["recorded_container_times"]
    with np.load(ep / native["file"]) as clocks:
        np.testing.assert_array_equal(clocks["timestamps"], raw)
        assert clocks["timestamps"].dtype == raw.dtype
    assert native["cameras"]["exo"] == "timestamps"
    assert native["clocks"]["timestamps"]["units"] == ("ns" if integer else None)
    with np.load(ep / "times.npz") as clocks:
        np.testing.assert_allclose(clocks["exo"], np.repeat([0., 1., 2., 3.], 10))
        assert len(clocks["exo_pts"]) == 40 and (np.diff(clocks["exo_pts"]) > 0).all()
    with np.load(ep / ctx["presentation_times"]) as clocks:
        np.testing.assert_allclose(clocks["exo"], np.arange(40) / 10)
    for path in ep.glob("*.mp4"):
        with av.open(str(path)) as container:
            assert sum(1 for _ in container.decode(video=0)) == 40
    unshown = ctx["unshown_cameras"][0]
    assert unshown["n_frames"] == 40 and unshown["fps"] == 10
    assert unshown.get("camera_times")
    assert any(i["kind"] == "camera_timestamp_repeated" and i.get("camera") == "infrared"
               for i in ctx.get("reader_issues", []))
    assert not any(i["kind"] in ("camera_short", "main_camera_short") for i in ctx.get("reader_issues", []))
    request = episode.build_request(ep)
    assert "assumed presentation" in request["prompt"]
    assert "The times are exact: use them, do not invent your own." not in request["prompt"]


def test_hdf_all_tied_clock_uses_its_declared_nominal_cadence(tmp_path):
    raw = np.full(40, 1_790_000_000_000_000_003, dtype=np.int64)
    ep, ctx = convert(hdf_upload(tmp_path, raw), tmp_path)
    assert ctx.get("presentation_times")
    with np.load(ep / "times.npz") as recorded, np.load(ep / ctx["presentation_times"]) as shown:
        np.testing.assert_array_equal(recorded["exo"], np.zeros(40))
        np.testing.assert_allclose(shown["exo"], np.arange(40) / 10)
    assert ctx["camera_clock"]["exo"]["cadence_source"] == "declared nominal cadence"


def test_hdf_native_clock_column_keeps_its_original_array_shape(tmp_path):
    raw = np.repeat(np.arange(4, dtype=float), 10)[:, None]
    ep, ctx = convert(hdf_upload(tmp_path, raw), tmp_path)
    with np.load(ep / ctx["recorded_container_times"]["file"]) as clocks:
        np.testing.assert_array_equal(clocks["timestamps"], raw)


@pytest.mark.parametrize("integer", [False, True])
def test_hdf_unique_native_clock_retains_original_dtype_values_and_units(tmp_path, integer):
    raw = (1_790_000_000_000_000_003 + np.arange(40, dtype=np.int64) * 100_000_000
           if integer else np.arange(40, dtype=np.float32) / 10)
    ep, ctx = convert(hdf_upload(tmp_path, raw), tmp_path)
    assert not ctx.get("presentation_times")
    with np.load(ep / ctx["recorded_container_times"]["file"]) as clocks:
        assert clocks["timestamps"].dtype == raw.dtype
        np.testing.assert_array_equal(clocks["timestamps"], raw)
    assert ctx["recorded_container_times"]["clocks"]["timestamps"]["units"] == ("ns" if integer else None)


@pytest.mark.parametrize("origin,dense", [(1_790_000_000_000_000_003, False), (3_000_000_000_003, True)])
def test_hdf_distinct_native_camera_clocks_keep_exact_relative_integer_offsets(tmp_path, origin, dense):
    upload = tmp_path / "upload"
    upload.mkdir()
    rows = np.repeat(np.arange(4, dtype=np.int64), 10)
    with h5py.File(upload / "recording.h5", "w") as f:
        f.attrs["fps"] = 10
        for camera, offset in [("top", 0), ("wrist_left", 150_000_003)]:
            step = 1_000_000 if dense and camera == "wrist_left" else 1_000_000_000
            clock = f.create_dataset(camera + ("/timestamps" if camera == "top" else "/frame_timestamps"),
                                     data=origin + rows * step + offset)
            clock.attrs["units"] = "ns"
            f.create_dataset(camera + "/" + camera + "_images", data=np.stack([np.full((36, 64, 3), k * 5, np.uint8)
                                                                 for k in range(40)]))
    ep, ctx = convert(upload, tmp_path)
    with np.load(ep / "times.npz") as clocks:
        np.testing.assert_allclose(clocks["left"], rows * (.001 if dense else 1.) + .150000003,
                                   rtol=0, atol=1e-12)


def test_hdf_timed_signal_on_tied_camera_clock_keeps_the_episode_and_all_rows(tmp_path):
    ep, ctx = convert(hdf_upload(tmp_path, np.repeat(np.arange(4, dtype=float), 10), signal=True), tmp_path)
    force = next(s for s in ctx["signals"] if s["name"] == "force")
    assert force["camera_aligned_by"] == "assumed camera clock"
    with np.load(ep / "signals.npz") as signals:
        np.testing.assert_array_equal(signals[force["key"]].ravel(), np.arange(40))


def test_hdf_all_tied_camera_and_signal_clock_retains_every_recorded_value(tmp_path):
    ep, ctx = convert(hdf_upload(tmp_path, np.zeros(40), signal=True), tmp_path)
    force = next(s for s in ctx["signals"] if s["name"] == "force")
    assert force["camera_aligned_by"] == "assumed camera clock"
    with np.load(ep / "signals.npz") as signals:
        np.testing.assert_array_equal(signals[force["key"]].ravel(), np.arange(40))


def test_hdf_board_and_parts_show_every_tied_frame_and_keep_native_clock_provenance(tmp_path, monkeypatch):
    from label import pieces
    raw = 1_790_000_000_000_000_003 + np.repeat(np.arange(4, dtype=np.int64), 10) * 1_000_000_000
    ep, ctx = convert(hdf_upload(tmp_path, raw), tmp_path)
    sources = json.loads((ep / "sources.json").read_text())
    expected = {}
    for view, source in sources.items():
        with av.open(source["packed"]) as c:
            expected[view] = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    decoded = []
    actual_decode = episode._decode_view

    def trace(unit, view, ks, *args, **kwargs):
        got = actual_decode(unit, view, ks, *args, **kwargs)
        mapping = unit["kmap"].get(view)
        for k, image in got.items():
            own = int(mapping[k]) if mapping is not None else k
            np.testing.assert_array_equal(np.asarray(image), expected[view][own])
            decoded.append((view, k))
        return got

    monkeypatch.setattr(episode, "_decode_view", trace)
    request = episode.build_request(ep)
    assert {(v, k) for v, k in decoded} == {(v, k) for v in sources for k in request["plan"]["ks"]}
    np.testing.assert_allclose(sensors.clip_times(ep, ctx, 40), np.arange(40) / 10)
    monkeypatch.setattr(episode, "_decode_view", actual_decode)
    monkeypatch.setattr(pieces, "choose_cuts", lambda *_: [{"frame": 17, "t_s": 1.7}])
    parts = pieces.write_pieces(ep, tmp_path / "parts")
    ranges = [(0, 17), (17, 40)]
    for unit, (lo, hi) in [(ep, (0, 40)), *zip(parts, ranges)]:
        context = json.loads((unit / "context.json").read_text())
        with np.load(unit / context["recorded_container_times"]["file"]) as clocks:
            np.testing.assert_array_equal(clocks["timestamps"], raw)
        jobs = list(clips.episode_jobs(unit, tmp_path / "clips", True))
        assert len(jobs) == 6
        for job in jobs:
            assert clips.extract_one(*job[:4], clips.find_ffmpeg(), 1, *job[4:12]) is None
            with av.open(str(job[0])) as c:
                original = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
            with av.open(str(job[3])) as c:
                shown = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
            assert len(shown) == hi - lo
            for index, picture in enumerate(shown):
                np.testing.assert_array_equal(picture, original[lo + index])


def test_hdf_actual_depth_ties_keep_capture_times_and_pair_every_depth_picture(tmp_path):
    raw = np.repeat(np.arange(4, dtype=float), 10)
    upload = hdf_upload(tmp_path, raw)
    with h5py.File(upload / "recording.h5", "a") as f:
        f.create_dataset("depth/top", data=np.stack([np.full((64, 64), 1000 + k * 10, np.uint16)
                                                    for k in range(40)]))
    ep, ctx = convert(upload, tmp_path)
    assert "exo" in ctx["depth"]
    with np.load(ep / "depth_times.npz") as clocks:
        np.testing.assert_array_equal(clocks["depth_exo"], raw)
        assert len(clocks["depth_exo_pts"]) == 40
    np.testing.assert_array_equal(np.load(ep / "depth_kmap_exo.npy"), np.arange(40))
    from label import depth
    entry = depth.load(ep)["exo"]
    got = depth.decode(entry, entry["pts"], list(range(40)))
    assert set(got) == set(range(40))
    for k, array in got.items():
        np.testing.assert_array_equal(array, np.full((64, 64), 1000 + k * 10, np.uint16))
    assert "Depth is paired using assumed presentation times" in episode.build_request(ep)["prompt"]


def test_hdf_depth_clock_is_qualified_when_colour_capture_clock_is_unique(tmp_path):
    upload = hdf_upload(tmp_path, np.arange(40, dtype=float) / 10)
    tied = np.repeat(np.arange(4, dtype=float), 10)
    with h5py.File(upload / "recording.h5", "a") as f:
        f.create_dataset("depth/timestamps", data=tied)
        f.create_dataset("depth/top", data=np.stack([np.full((64, 64), 1000 + k * 10, np.uint16)
                                                    for k in range(40)]))
    ep, ctx = convert(upload, tmp_path)
    assert ctx.get("depth_camera_clock") and not ctx.get("camera_clock")
    with np.load(ep / "depth_times.npz") as clocks:
        np.testing.assert_array_equal(clocks["depth_exo"], tied)
    np.testing.assert_array_equal(np.load(ep / "depth_kmap_exo.npy"), np.arange(40))


def test_loose_video_native_integer_sidecar_keeps_its_exact_dtype_and_values(tmp_path):
    from test_reader_tables import _clip
    upload = tmp_path / "upload"
    upload.mkdir()
    _clip(upload / "top.mp4", 40)
    raw = 1_790_000_000_000_000_003 + np.arange(40, dtype=np.int64) * 33_333_333
    np.save(upload / "top_timestamp.npy", raw)
    ep, ctx = convert(upload, tmp_path)
    native = ctx["recorded_container_times"]
    with np.load(ep / native["file"]) as clocks:
        np.testing.assert_array_equal(clocks["top_timestamp.npy"], raw)
        assert clocks["top_timestamp.npy"].dtype == np.int64
    assert native["clocks"]["top_timestamp.npy"]["units"] is None


def test_loose_unique_float_sidecar_retains_its_original_dtype_and_values(tmp_path):
    from test_reader_tables import _clip
    upload = tmp_path / "upload"
    upload.mkdir()
    _clip(upload / "top.mp4", 40)
    raw = np.arange(40, dtype=np.float32) / 30
    np.save(upload / "top_timestamp.npy", raw)
    ep, ctx = convert(upload, tmp_path)
    assert not ctx.get("presentation_times")
    with np.load(ep / ctx["recorded_container_times"]["file"]) as clocks:
        assert clocks["top_timestamp.npy"].dtype == raw.dtype
        np.testing.assert_array_equal(clocks["top_timestamp.npy"], raw)


def test_loose_unshown_video_uses_its_supplied_tied_capture_clock(tmp_path):
    from test_reader_tables import _clip
    upload = tmp_path / "upload"
    upload.mkdir()
    for name in ("top", "top_mask"):
        _clip(upload / (name + ".mp4"), 40)
        np.save(upload / (name + "_timestamp.npy"), np.repeat(np.arange(4, dtype=float), 10))
    ep, ctx = convert(upload, tmp_path)
    entry = ctx["unshown_cameras"][0]
    assert entry.get("camera_times") and entry["fps"] == 10
    with np.load(ep / entry["camera_times"]) as clocks:
        np.testing.assert_array_equal(clocks["capture"], np.repeat(np.arange(4, dtype=float), 10))
        np.testing.assert_allclose(clocks["presentation"], np.arange(40) / 10)


def test_parquet_image_tied_clock_does_not_restore_measured_joint_state(tmp_path):
    upload = tmp_path / "upload"
    (upload / "meta").mkdir(parents=True)
    (upload / "data/chunk-000").mkdir(parents=True)
    features = {"observation.images.top": {"dtype": "image", "shape": [72, 96, 3]},
                "observation.state": {"dtype": "float32", "shape": [14]}}
    (upload / "meta/info.json").write_text(json.dumps({"codebase_version": "v2.1", "fps": 10,
        "chunks_size": 1000, "features": features,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"}))
    cells = []
    for k in range(40):
        out = io.BytesIO()
        Image.fromarray(np.full((72, 96, 3), k * 5, np.uint8)).save(out, format="PNG")
        cells.append({"bytes": out.getvalue(), "path": None})
    state = np.arange(40 * 14, dtype=np.float32).reshape(40, 14) / 1000
    pd.DataFrame({"observation.images.top": cells, "timestamp": np.repeat(np.arange(4, dtype=float), 10),
                  "frame_index": np.arange(40), "episode_index": np.zeros(40, int),
                  "observation.state": list(state)}).to_parquet(upload / "data/chunk-000/episode_000000.parquet")
    ep, ctx = convert(upload, tmp_path)
    assert ctx["state_kind"] == "none" and ctx["state_why"] == "assumed_clock"
    assert ctx.get("presentation_times")
    recorded = next(s for s in ctx["signals"] if s["name"] == "recorded state")
    with np.load(ep / "signals.npz") as signals:
        np.testing.assert_array_equal(signals[recorded["key"]], state)
    assert "RECORDED MOTION" not in episode.build_request(ep)["prompt"]


def test_parquet_image_unique_native_column_retains_its_original_dtype_and_values(tmp_path):
    test_parquet_image_tied_clock_does_not_restore_measured_joint_state(tmp_path)
    path = tmp_path / "upload/data/chunk-000/episode_000000.parquet"
    df = pd.read_parquet(path)
    raw = np.arange(40, dtype=np.float32) / 10
    df["timestamp"] = raw
    df.to_parquet(path)
    ep, ctx = convert(tmp_path / "upload", tmp_path / "unique")
    assert not ctx.get("presentation_times") and ctx["state_kind"] == "joints"
    with np.load(ep / ctx["recorded_container_times"]["file"]) as clocks:
        assert clocks["timestamp"].dtype == raw.dtype
        np.testing.assert_array_equal(clocks["timestamp"], raw)
    assert ctx["recorded_container_times"]["clocks"]["timestamp"]["units"] is None
