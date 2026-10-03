"""Root folder identity is recorded separately from relative upload filenames."""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from label import episode
from prepare import formats
from test_adapter_output_identity import original_context
from test_formats import _clip


def test_root_openaoe_identity_survives_a_different_selected_input_folder(tmp_path):
    original = tmp_path / "root_clip_original"
    original.mkdir()
    _clip(original / "raw_video.mp4", 20)
    annotation = original / "ego_annotation" / "ego_action_annotation.json"
    annotation.parent.mkdir()
    annotation.write_text(json.dumps([{"start_ts": 0, "end_ts": 0.2,
                                      "atomic_action": [{"verb": "move", "object": "object"}]}]))
    context = original_context(original)
    context["root_name"] = original.name
    full_dir = tmp_path / "full"
    full = formats.convert(original, "ego_head", full_dir, "identity", 900)
    selected = tmp_path / "selected_upload"
    shutil.copytree(original, selected)
    subset_dir = tmp_path / "subset"
    subset = formats.convert(selected, "ego_head", subset_dir, "identity", 900, ownership_context=context)
    assert not full["failed"] and not subset["failed"]
    expected = full["episodes"][0]["episode_id"]
    assert expected == "episode_root_clip_original"
    assert subset["episodes"][0]["episode_id"] == expected
    assert episode.build_request(subset_dir / expected) == episode.build_request(full_dir / expected)


@pytest.mark.parametrize("suffix", ["tbz", "tbz2", "tar.bz2"])
def test_browser_accepted_bzip_tar_aliases_read_their_actual_hdf_groups(tmp_path, suffix):
    assets = tmp_path / "assets"
    assets.mkdir()
    source = assets / "a.h5"
    with h5py.File(source, "w") as h:
        for group, value in [("demo_0", 40), ("demo_1", 180)]:
            h[f"data/{group}/obs/front"] = np.full((10, 36, 64, 3), value, dtype=np.uint8)
    upload = tmp_path / "upload"
    upload.mkdir()
    archive = upload / ("held." + suffix)
    with tarfile.open(archive, "w:bz2") as t:
        t.add(source, arcname="held/a.h5")
    before = hashlib.sha256(archive.read_bytes()).hexdigest()
    output = tmp_path / "units"
    report = formats.convert(upload, "ego_head", output, "identity", 900)
    assert not report["failed"] and len(report["episodes"]) == 2
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == before
    for row in report["episodes"]:
        assert episode.build_request(output / row["episode_id"])["n_images"] > 0


def test_a_note_named_for_original_root_stays_shared_after_subset_selection(tmp_path):
    from test_ownership_context import containers, descriptor_context

    original = containers(tmp_path / "a")
    (original / "demo0_meta.json").unlink()
    (original / "a.json").write_text(json.dumps({"task": "the original folder task"}))
    context = descriptor_context(original)
    context["root_name"] = original.name
    full_dir = tmp_path / "full"
    full = formats.convert(original, "ego_head", full_dir, "identity", 900)
    selected = tmp_path / "selected"
    selected.mkdir()
    for name in ["b.h5", "a.json"]:
        shutil.copyfile(original / name, selected / name)
    subset_dir = tmp_path / "subset"
    subset = formats.convert(selected, "ego_head", subset_dir, "identity", 900, ownership_context=context)
    assert not full["failed"] and not subset["failed"] and len(subset["episodes"]) == 2
    for row in subset["episodes"]:
        expected = next(r for r in full["episodes"] if r["episode_id"] == row["episode_id"])
        assert expected["instruction"] == "the original folder task"
        assert row["instruction"] == expected["instruction"]
        assert episode.build_request(subset_dir / row["episode_id"]) == episode.build_request(full_dir / row["episode_id"])


@pytest.mark.parametrize("name", [None, "", ".", "..", "other/root", "other\\root", 42])
def test_original_root_name_rejects_values_that_are_not_folder_names(tmp_path, name):
    with pytest.raises(ValueError, match="not a folder name"):
        formats.ownership_root_name(tmp_path, {"root_name": name})


def test_original_root_name_uses_the_real_folder_without_context(tmp_path):
    assert formats.ownership_root_name(tmp_path, None) == tmp_path.name
    assert formats.ownership_root_name(tmp_path, {}) == tmp_path.name


def test_root_openaoe_does_not_guess_identity_when_context_omits_root_name(tmp_path):
    original = tmp_path / "original_clip"
    original.mkdir()
    _clip(original / "raw_video.mp4", 20)
    annotation = original / "ego_annotation" / "ego_action_annotation.json"
    annotation.parent.mkdir()
    annotation.write_text("[]")
    context = original_context(original)
    selected = tmp_path / "selected_upload"
    shutil.copytree(original, selected)
    with pytest.raises(ValueError, match="requires the original upload root name"):
        formats.plan(selected, ownership_context=context)
