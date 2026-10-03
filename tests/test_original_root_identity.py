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


@pytest.mark.parametrize("name", [None, "", ".", "..", "other/root", "other\\root", "\x00", 42])
def test_original_root_name_rejects_values_that_are_not_folder_names(tmp_path, name):
    with pytest.raises(ValueError, match="not a folder name"):
        formats.ownership_root_name(tmp_path, {"root_name": name})


def test_original_root_name_uses_the_real_folder_without_context(tmp_path):
    assert formats.ownership_root_name(tmp_path, None) == tmp_path.name
    assert formats.ownership_root_name(tmp_path, {}) == tmp_path.name


@pytest.mark.parametrize("root_name", ["a", "A"])
def test_specific_group_task_outranks_shared_root_with_a_container_basename(tmp_path, root_name):
    from test_ownership_context import containers, descriptor_context

    original = containers(tmp_path / root_name)
    (original / "a.json").write_text(json.dumps({"task": "shared root task", "note": "original root note"}))
    context = descriptor_context(original)
    context["root_name"] = original.name
    whole_dir = tmp_path / "whole"
    whole = formats.convert(original, "ego_head", whole_dir, "identity", 900)
    selected = tmp_path / "selected"
    selected.mkdir()
    for name in ["a.h5", "a.json", "demo0_meta.json"]:
        shutil.copyfile(original / name, selected / name)
    subset_dir = tmp_path / "subset"
    subset = formats.convert(selected, "ego_head", subset_dir, "identity", 900, ownership_context=context)
    assert not whole["failed"] and not subset["failed"]
    assert len(whole["episodes"]) == 4 and len(subset["episodes"]) == 2
    for row in whole["episodes"]:
        expected = "move the demo zero object" if row["name"] == "a/demo_0" else "shared root task"
        assert row["instruction"] == expected
        ctx = json.loads((whole_dir / row["episode_id"] / "context.json").read_text())
        assert ctx["uploader_notes"]["a.json"]["note"] == "original root note"
    for row in subset["episodes"]:
        name = row["episode_id"]
        assert episode.build_request(subset_dir / name) == episode.build_request(whole_dir / name)


def test_shared_root_task_disagreement_is_not_hidden_by_a_container_alias(tmp_path):
    from test_ownership_context import containers

    original = containers(tmp_path / "a")
    (original / "demo0_meta.json").unlink()
    for filename, task in [("a.json", "first shared task"), ("session_meta.json", "second shared task")]:
        (original / filename).write_text(json.dumps({"task": task}))
    output = tmp_path / "units"
    report = formats.convert(original, "ego_head", output, "identity", 900)
    assert not report["failed"] and len(report["episodes"]) == 4
    for row in report["episodes"]:
        assert not row.get("instruction")
        ctx = json.loads((output / row["episode_id"] / "context.json").read_text())
        assert any(issue["kind"] == "task_files_disagree" for issue in ctx["reader_issues"])
        assert {"a.json", "session_meta.json"} <= ctx["uploader_notes"].keys()


def test_unreadable_shared_root_note_is_retained_with_one_issue_per_episode(tmp_path):
    from test_ownership_context import containers

    original = containers(tmp_path / "a")
    (original / "a.json").write_text('{"task": "incomplete shared task"')
    output = tmp_path / "units"
    report = formats.convert(original, "ego_head", output, "identity", 900)
    assert not report["failed"] and len(report["episodes"]) == 4
    for row in report["episodes"]:
        ctx = json.loads((output / row["episode_id"] / "context.json").read_text())
        issues = [issue for issue in ctx["reader_issues"] if issue["kind"] == "metadata_unreadable"]
        assert len(issues) == 1
        assert ctx["uploader_notes"]["a.json"] == '{"task": "incomplete shared task"'


def test_single_numbered_demo_keeps_its_owned_task_and_shared_claim(tmp_path):
    from test_ownership_context import containers

    original = containers(tmp_path / "a")
    (original / "b.h5").unlink()
    with h5py.File(original / "a.h5", "a") as h:
        del h["data/demo_1"]
    (original / "a.json").write_text(json.dumps({"task": "shared root task"}))
    output = tmp_path / "units"
    report = formats.convert(original, "ego_head", output, "identity", 900)
    assert not report["failed"] and len(report["episodes"]) == 1
    row = report["episodes"][0]
    assert row.get("instruction") == "move the demo zero object"
    ctx = json.loads((output / row["episode_id"] / "context.json").read_text())
    assert ctx["source"]["group"] == "data/demo_0"
    assert {"a.json", "demo0_meta.json"} <= ctx["uploader_notes"].keys()
    assert ctx["uploader_notes"]["a.json"]["task"] == "shared root task"
    assert ctx["uploader_notes"]["demo0_meta.json"]["task"] == ctx["instruction"]


@pytest.mark.parametrize("camera,other_note", [("top", "instruction.txt"), ("recording", "instruction.txt"),
                                             ("top_demo_1", "instruction.txt"), ("top_demo_1", "demo1_meta.json")])
def test_single_video_episode_folder_keeps_its_primary_json_task_after_selection(tmp_path, camera, other_note):
    from test_adapter_output_identity import original_context

    original = tmp_path / "original"
    home = original / "ep2"
    home.mkdir(parents=True)
    _clip(home / (camera + ".mp4"), 10)
    (home / "ep2.json").write_text(json.dumps({"task": "pour the tea", "note": "original episode note"}))
    (home / other_note).write_text(json.dumps({"task": "pick the cup"}) if other_note.endswith(".json")
                                   else "pick the cup")
    context = original_context(original)
    context["root_name"] = original.name
    whole_dir = tmp_path / "whole"
    whole = formats.convert(original, "ego_head", whole_dir, "identity", 900)
    selected = tmp_path / "selected"
    shutil.copytree(original, selected)
    subset_dir = tmp_path / "subset"
    subset = formats.convert(selected, "ego_head", subset_dir, "identity", 900, ownership_context=context)
    assert not whole["failed"] and not subset["failed"]
    assert len(whole["episodes"]) == len(subset["episodes"]) == 1
    name = whole["episodes"][0]["episode_id"]
    assert whole["episodes"][0]["instruction"] == "pour the tea"
    assert subset["episodes"][0]["instruction"] == "pour the tea"
    assert episode.build_request(subset_dir / name) == episode.build_request(whole_dir / name)


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


def test_root_openaoe_rejects_a_nul_in_the_bound_original_folder_name(tmp_path):
    original = tmp_path / "original_clip"
    original.mkdir()
    _clip(original / "raw_video.mp4", 20)
    annotation = original / "ego_annotation" / "ego_action_annotation.json"
    annotation.parent.mkdir()
    annotation.write_text("[]")
    context = original_context(original)
    context["root_name"] = "\x00"
    with pytest.raises(ValueError, match="not a folder name"):
        formats.plan(original, ownership_context=context)


@pytest.mark.parametrize('relative', ['.', '..'])
def test_public_folder_cli_accepts_the_current_or_parent_upload_folder(tmp_path, relative):
    import os
    import subprocess
    import sys
    from test_ownership_context import containers

    original = containers(tmp_path / 'original')
    expected_dir = tmp_path / 'expected'
    expected = formats.convert(original, 'ego_head', expected_dir, 'identity', 900)
    cwd = original
    if relative == '..':
        cwd = original / 'inside'
        cwd.mkdir()
    output = tmp_path / 'relative'
    source = Path(formats.__file__).resolve().parent.parent
    env = dict(os.environ, PYTHONPATH=str(source))
    command = [sys.executable, '-m', 'prepare', 'folder', 'prepare', '--root', relative,
               '--rig', 'ego_head', '--out', str(output), '--dataset', 'identity']
    run = subprocess.run(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    assert not report['failed'] and len(report['episodes']) == len(expected['episodes']) == 4
    for row in report['episodes']:
        name = row['episode_id']
        assert episode.build_request(output / name) == episode.build_request(expected_dir / name)
