"""Selected containers retain note ownership from the original upload."""
import copy
import hashlib
import inspect
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from prepare import formats
from request_identity import assert_same_model_inputs
from test_formats import _camera_mcap, _clip


def containers(root):
    root.mkdir()
    for name, groups in [("a.h5", ("demo_0", "demo_1")), ("b.h5", ("first_recording", "second_recording"))]:
        with h5py.File(root / name, "w") as h:
            for group in groups:
                h[f"data/{group}/obs/front"] = np.full((10, 36, 64, 3), 80, dtype=np.uint8)
    (root / "demo0_meta.json").write_text(json.dumps({"task": "move the demo zero object", "note": "demo zero camera loose"}))
    return root


def descriptor_context(root):
    _, items = formats.plan(root)
    rows = [{"kind": it["kind"], "name": it["name"], "file": it["file"].relative_to(root).as_posix(),
             "group": it.get("group", "")} for it in items if it["kind"] in {"hdf5", "mcap"}]
    rows += [{"kind": "video_alias", "file": p.relative_to(root).as_posix()} for p in root.rglob("*")
             if p.is_file() and p.suffix.lower() in formats.VIDEO_EXT and not formats.hidden(p, root)]
    return {"version": 1, "episodes": rows}


@pytest.mark.parametrize("selected", ["a.h5", "b.h5"])
def test_selected_hdf5_tasks_keep_ownership_from_the_original_upload(tmp_path, selected):
    # A missing optional input is an assertion failure rather than a fixture setup exception.
    assert "ownership_context" in inspect.signature(formats.convert).parameters
    original = containers(tmp_path / "original")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in original.iterdir()}
    registry = descriptor_context(original)
    whole = tmp_path / "whole"
    full = formats.convert(original, "ego_head", whole, "ownership", 900)
    chosen = tmp_path / "selected"
    chosen.mkdir()
    for name in (selected, "demo0_meta.json"):
        shutil.copyfile(original / name, chosen / name)
    subset = tmp_path / "subset"
    report = formats.convert(chosen, "ego_head", subset, "ownership", 900, ownership_context=registry)
    assert not report["failed"] and len(report["episodes"]) == 2
    for row in report["episodes"]:
        ep = subset / row["episode_id"]
        ctx = json.loads((ep / "context.json").read_text())
        expected = "move the demo zero object" if row["name"] == "a/demo_0" else None
        assert ctx.get("instruction") == expected
        assert_same_model_inputs(ep, whole / row["episode_id"])
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in original.iterdir()}
    assert registry["episodes"] == descriptor_context(original)["episodes"]
    assert len(full["episodes"]) == 4


@pytest.mark.parametrize("change", [
    "version", "boolean_version", "no_list", "absolute", "parent", "dot", "backslash", "empty_part",
    "wrong_kind", "wrong_suffix", "group_parent", "group_absolute", "group_nontext", "wrong_name", "duplicate",
    "missing_group", "extra_group",
])
def test_stale_or_invalid_ownership_context_cannot_assign_notes(tmp_path, change):
    original = containers(tmp_path / "original")
    context = descriptor_context(original)
    row = context["episodes"][0]
    if change == "version":
        context["version"] = 2
    elif change == "boolean_version":
        context["version"] = True
    elif change == "no_list":
        context["episodes"] = {}
    elif change in {"absolute", "parent", "dot", "backslash", "empty_part"}:
        row["file"] = {"absolute": "/a.h5", "parent": "../a.h5", "dot": "./a.h5",
                       "backslash": "a\\a.h5", "empty_part": "a//a.h5"}[change]
    elif change == "wrong_kind":
        row["kind"] = "unknown"
    elif change == "wrong_suffix":
        row["file"] = "a.txt"
    elif change == "group_parent":
        row["group"] = "data/../demo_0"
    elif change == "group_absolute":
        row["group"] = "/data/demo_0"
    elif change == "group_nontext":
        row["group"] = []
    elif change == "wrong_name":
        row["name"] = "another_episode"
    elif change == "duplicate":
        context["episodes"].append(copy.deepcopy(row))
    elif change == "missing_group":
        context["episodes"].pop(0)
    else:
        context["episodes"].append({**row, "group": "data/demo_2", "name": "a/demo_2"})
    with pytest.raises(ValueError, match="ownership|inspected groups"):
        formats.plan(original, ownership_context=context)


def test_whole_upload_registry_preserves_shared_tasks_and_recorded_instructions(tmp_path):
    original = containers(tmp_path / "original")
    (original / "session_meta.json").write_text(json.dumps({"task": "shared recorded session"}))
    with h5py.File(original / "b.h5", "a") as h:
        h["data/first_recording"].attrs["instruction"] = "instruction inside the recording"
    context = descriptor_context(original)
    base = tmp_path / "base"
    final = tmp_path / "final"
    before = formats.convert(original, "ego_head", base, "ownership", 900)
    after = formats.convert(original, "ego_head", final, "ownership", 900, ownership_context=context)
    assert before == after
    for row in after["episodes"]:
        assert_same_model_inputs(base / row["episode_id"], final / row["episode_id"])
    row = next(row for row in after["episodes"] if row["name"] == "b/first_recording")
    assert json.loads((final / row["episode_id"] / "context.json").read_text())["instruction"] == "instruction inside the recording"


@pytest.mark.parametrize("omitted", ["bag_unique.mcap", "capture_red.mp4", "top_ep1.mp4"])
def test_selected_hdf5_notes_keep_omitted_media_filename_owners(tmp_path, omitted):
    original = containers(tmp_path / "original")
    path = original / omitted
    if path.suffix == ".mcap":
        _camera_mcap(path, ["/front/image/compressed"], n=10)
    else:
        _clip(path, 10)
    note = path.with_name(path.stem + "_meta.json")
    note.write_text(json.dumps({"task": "task of omitted media", "note": "omitted media claim"}))
    context = descriptor_context(original)
    chosen = tmp_path / "selected"
    chosen.mkdir()
    for path in [original / "b.h5", note]:
        shutil.copyfile(path, chosen / path.name)
    whole = tmp_path / "whole"
    formats.convert(original, "ego_head", whole, "ownership", 900)
    subset = tmp_path / "subset"
    report = formats.convert(chosen, "ego_head", subset, "ownership", 900, ownership_context=context)
    assert len(report["episodes"]) == 2 and not report["failed"]
    for row in report["episodes"]:
        assert not json.loads((subset / row["episode_id"] / "context.json").read_text()).get("instruction")
        assert_same_model_inputs(subset / row["episode_id"], whole / row["episode_id"])


@pytest.mark.parametrize("filename", ["session_meta.json", "b_capture_red_meta.json", "b_meta.json"])
def test_subset_context_keeps_shared_and_multiple_owner_tasks(tmp_path, filename):
    original = containers(tmp_path / "original")
    (original / "demo0_meta.json").unlink()
    _clip(original / "capture_red.mp4", 10)
    note = original / filename
    note.write_text(json.dumps({"task": "shared or explicitly multiple owners"}))
    context = descriptor_context(original)
    chosen = tmp_path / "selected"
    chosen.mkdir()
    for path in [original / "b.h5", note]:
        shutil.copyfile(path, chosen / path.name)
    whole = tmp_path / "whole"
    formats.convert(original, "ego_head", whole, "ownership", 900)
    subset = tmp_path / "subset"
    report = formats.convert(chosen, "ego_head", subset, "ownership", 900, ownership_context=context)
    assert len(report["episodes"]) == 2 and not report["failed"]
    for row in report["episodes"]:
        ctx = json.loads((subset / row["episode_id"] / "context.json").read_text())
        assert ctx["instruction"] == "shared or explicitly multiple owners"
        assert_same_model_inputs(subset / row["episode_id"], whole / row["episode_id"])


@pytest.mark.parametrize("video", ["capture_mask.mp4", "capture_depth.mp4"])
def test_unshown_filename_owners_stay_unshown_when_only_hdf5_is_selected(tmp_path, video):
    original = containers(tmp_path / "original")
    (original / "demo0_meta.json").unlink()
    for name in ["capture_red.mp4", video]:
        _clip(original / name, 10)
    note = original / ("b_" + Path(video).stem + "_meta.json")
    note.write_text(json.dumps({"task": "a camera note that is not the HDF task"}))
    context = descriptor_context(original)
    selected = tmp_path / "selected"
    selected.mkdir()
    for path in [original / "b.h5", note]:
        shutil.copyfile(path, selected / path.name)
    full_out, subset_out = tmp_path / "full", tmp_path / "subset"
    formats.convert(original, "ego_head", full_out, "ownership", 900)
    subset = formats.convert(selected, "ego_head", subset_out, "ownership", 900, ownership_context=context)
    for row in subset["episodes"]:
        assert_same_model_inputs(subset_out / row["episode_id"], full_out / row["episode_id"])



def test_original_context_keeps_the_existing_hdf_extension(tmp_path):
    original = containers(tmp_path / "original")
    (original / "b.h5").rename(original / "b.hdf")
    context = descriptor_context(original)
    assert len(formats.plan(original, ownership_context=context)[1]) == 4



def test_hidden_original_filenames_cannot_take_a_shared_task(tmp_path):
    original = containers(tmp_path / "original")
    (original / "demo0_meta.json").unlink()
    (original / "camera_meta.json").write_text(json.dumps({"task": "shared camera session"}))
    context = descriptor_context(original)
    context["episodes"].append({"kind": "video_alias", "file": "._camera.mp4"})
    base, final = tmp_path / "base", tmp_path / "final"
    before = formats.convert(original, "ego_head", base, "ownership", 900)
    after = formats.convert(original, "ego_head", final, "ownership", 900, ownership_context=context)
    assert before == after
    for row in after["episodes"]:
        assert_same_model_inputs(base / row["episode_id"], final / row["episode_id"])
