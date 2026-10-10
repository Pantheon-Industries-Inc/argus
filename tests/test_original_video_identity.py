"""Selected video groups keep the identities and note owners of their original upload."""
import json
import shutil
from pathlib import Path

import pytest

from label import episode
from prepare import formats
from request_identity import assert_same_model_inputs
from test_formats import _clip


@pytest.mark.parametrize("layout", ["flat", "nested", "camera_folders"])
@pytest.mark.parametrize("rig", ["ego_head", "handheld_gripper", "teleop_arms"])
def test_selected_video_take_keeps_the_whole_original_request(tmp_path, layout, rig):
    original = tmp_path / "original"
    original.mkdir()
    paths = []
    for take in [1, 10]:
        for camera in ["top", "wrist_left"]:
            relative = (f"{camera}_ep{take:02d}.mp4" if layout == "flat" else
                        f"outer/{camera}_ep{take:02d}.mp4" if layout == "nested" else
                        f"{camera}/ep{take:02d}.mp4")
            path = original / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            _clip(path, 10)
            paths.append(relative)
    context = {"version": 1, "episodes": [{"kind": "video_alias", "file": path} for path in paths],
               "grouping": {}, "declared_video_seconds": {path: formats._duration(original / path) for path in paths}}
    whole = tmp_path / "whole"
    full = formats.convert(original, rig, whole, "identity", 900)
    chosen = tmp_path / "selected"
    for path in paths:
        if "01" not in Path(path).stem:
            continue
        destination = chosen / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / path, destination)
    subset = tmp_path / "subset"
    report = formats.convert(chosen, rig, subset, "identity", 900, ownership_context=context)
    assert not full["failed"] and not report["failed"] and len(report["episodes"]) == 1
    row = report["episodes"][0]
    original_row = next(item for item in full["episodes"] if item["name"].endswith("ep01"))
    assert row["name"] == original_row["name"]
    assert row["episode_id"] == original_row["episode_id"]
    assert_same_model_inputs(subset / row["episode_id"], whole / row["episode_id"])


def test_selected_same_stem_hdf_keeps_its_original_output_identity(tmp_path):
    import h5py
    import numpy as np

    original = tmp_path / "original"
    original.mkdir()
    for filename, value in [("same.h5", 40), ("same.hdf", 190)]:
        with h5py.File(original / filename, "w") as h:
            h["data/demo_0/obs/front"] = np.full((10, 36, 64, 3), value, dtype=np.uint8)
    context = {"version": 1, "episodes": [
        {"kind": it["kind"], "file": it["file"].relative_to(original).as_posix(),
         "name": it["name"], "group": it["group"]} for it in formats.plan(original)[1]]}
    whole = tmp_path / "whole"
    full = formats.convert(original, "ego_head", whole, "identity", 900)
    chosen = tmp_path / "selected"
    chosen.mkdir()
    shutil.copyfile(original / "same.hdf", chosen / "same.hdf")
    subset = tmp_path / "subset"
    report = formats.convert(chosen, "ego_head", subset, "identity", 900, ownership_context=context)
    row = report["episodes"][0]
    original_row = full["episodes"][1]
    assert row["episode_id"] == original_row["episode_id"]
    assert_same_model_inputs(subset / row["episode_id"], whole / row["episode_id"])


def test_normalized_mcap_names_keep_both_recordings(tmp_path):
    from test_formats import _camera_mcap

    original = tmp_path / "original"
    original.mkdir()
    _camera_mcap(original / "take-1.mcap", ["/front/image/compressed"], n=10)
    _camera_mcap(original / "take_1.mcap", ["/front/image/compressed"], n=20)
    output = tmp_path / "episodes"
    report = formats.convert(original, "ego_head", output, "identity", 900)
    assert not report["failed"] and len(report["episodes"]) == 2
    ids = [row["episode_id"] for row in report["episodes"]]
    assert len(set(ids)) == 2
    assert [json.loads((output / ep / "context.json").read_text())["n_state_frames"] for ep in ids] == [10, 20]


@pytest.mark.parametrize("kind", ["video", "hdf5"])
def test_selected_recording_keeps_original_fixed_window_packaging(tmp_path, kind):
    import h5py
    import numpy as np

    original = tmp_path / "original"
    original.mkdir()
    for take in [1, 2, 3]:
        path = original / f"take_{take}.{'mp4' if kind == 'video' else 'h5'}"
        if kind == "video":
            _clip(path, 900)
        else:
            with h5py.File(path, "w") as h:
                h["obs/front"] = np.full((900, 36, 64, 3), 70, dtype=np.uint8)
    items = formats.plan(original)[1]
    context = {"version": 1, "episodes": [
        {"kind": "video_alias", "file": it["files"][0].name} if kind == "video" else
        {"kind": it["kind"], "file": it["file"].name, "name": it["name"], "group": it["group"],
         "seconds": it["seconds"]} for it in items],
        "declared_video_seconds": {it["files"][0].name: it["seconds"] for it in items} if kind == "video" else {}}
    whole = tmp_path / "whole"
    full = formats.convert(original, "ego_head", whole, "identity", 900)
    chosen = tmp_path / "selected"
    chosen.mkdir()
    filename = "take_1.mp4" if kind == "video" else "take_1.h5"
    shutil.copyfile(original / filename, chosen / filename)
    subset = tmp_path / "subset"
    report = formats.convert(chosen, "ego_head", subset, "identity", 900, ownership_context=context)
    row = report["episodes"][0]
    expected = full["episodes"][0]
    assert report.get("packaging") == full.get("packaging")
    assert_same_model_inputs(subset / row["episode_id"], whole / expected["episode_id"])


def test_subset_cannot_turn_original_unmatched_depth_into_a_paired_model_camera(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    paths = ["capture_rgb.mp4", "capture_rgb.mov", "capture_depth.mp4"]
    for filename in paths:
        _clip(original / filename, 10)
    context = {"version": 1, "episodes": [{"kind": "video_alias", "file": p} for p in paths],
               "declared_video_seconds": {p: formats._duration(original / p) for p in paths}}
    whole = tmp_path / "whole"
    full = formats.convert(original, "ego_head", whole, "identity", 900)
    chosen = tmp_path / "selected"
    chosen.mkdir()
    for filename in ["capture_rgb.mp4", "capture_depth.mp4"]:
        shutil.copyfile(original / filename, chosen / filename)
    subset = tmp_path / "subset"
    report = formats.convert(chosen, "ego_head", subset, "identity", 900, ownership_context=context)
    row = report["episodes"][0]
    expected = full["episodes"][1]
    assert_same_model_inputs(subset / row["episode_id"], whole / expected["episode_id"])



def test_a_video_cannot_receive_a_task_named_for_another_containers_group(tmp_path):
    from test_ownership_context import containers

    original = containers(tmp_path / "original")
    _clip(original / "unrelated_capture.mp4", 23)
    output = tmp_path / "episodes"
    report = formats.convert(original, "ego_head", output, "identity", 900)
    row = next(row for row in report["episodes"] if row["name"] == "unrelated_capture")
    ctx = json.loads((output / row["episode_id"] / "context.json").read_text())
    assert not ctx.get("instruction")
    request = episode.build_request(output / row["episode_id"])
    assert "move the demo zero object" not in request["prompt"]
    owner = next(row for row in report["episodes"] if row["name"] == "a/demo_0")
    assert json.loads((output / owner["episode_id"] / "context.json").read_text())["instruction"] == "move the demo zero object"


@pytest.mark.parametrize("field,value", [
    ("declared_video_seconds", {"absent.mp4": 1}),
    ("declared_video_seconds", {"top_ep01.mp4": True}),
    ("declared_video_seconds", {"top_ep01.mp4": float("nan")}),
    ("declared_video_seconds", {"top_ep01.mp4": -1}),
    ("grouping", {"../outside": "cameras"}),
    ("grouping", {"": "unknown"}),
])
def test_invalid_original_video_declarations_cannot_change_group_identity(tmp_path, field, value):
    original = tmp_path / "original"
    original.mkdir()
    _clip(original / "top_ep01.mp4", 10)
    context = {"version": 1, "episodes": [{"kind": "video_alias", "file": "top_ep01.mp4"}], field: value}
    with pytest.raises(ValueError, match="original video"):
        formats.plan(original, ownership_context=context)


@pytest.mark.parametrize("video", ["capture_depth.mp4", "capture_mask.mp4"])
@pytest.mark.parametrize("original_context", [False, True])
def test_unassigned_unshown_media_never_gives_a_container_its_task(tmp_path, video, original_context):
    from test_ownership_context import containers, descriptor_context

    original = containers(tmp_path / "original")
    (original / "demo0_meta.json").unlink()
    _clip(original / video, 10)
    (original / "colour_only").mkdir()
    _clip(original / "colour_only" / "recording.mp4", 10)
    (original / (Path(video).stem + "_meta.json")).write_text(json.dumps({"task": "the unshown cameras task"}))
    context = descriptor_context(original) if original_context else None
    output = tmp_path / "episodes"
    report = formats.convert(original, "ego_head", output, "identity", 900, ownership_context=context)
    assert len(report["episodes"]) == 5 and not report["failed"]
    for row in report["episodes"]:
        ctx = json.loads((output / row["episode_id"] / "context.json").read_text())
        assert not ctx.get("instruction")
