"""Upload adapters share the original output identity allocation."""
import json
import shutil
from pathlib import Path

import pytest

from prepare import formats
from request_identity import assert_same_model_inputs
from test_formats import _clip


def original_context(root):
    videos = [p for p in sorted(root.rglob("*")) if p.suffix == ".mp4"]
    return {"version": 1,
            "episodes": [{"kind": "video_alias", "file": p.relative_to(root).as_posix()} for p in videos],
            "declared_video_seconds": {p.relative_to(root).as_posix(): formats._duration(p) for p in videos},
            "original_files": [p.relative_to(root).as_posix() for p in sorted(root.rglob("*")) if p.is_file()]}


def galaxea_upload(root, n):
    import av
    import numpy as np
    import pandas as pd
    from prepare import galaxea

    (root / "meta").mkdir(parents=True)
    features = {key: {"dtype": "float32", "shape": [6] if key.endswith("arm") else [1]}
                for key in galaxea.GALAXEA_COLUMNS}
    features.update({key: {"dtype": "video", "shape": [36, 64, 3]} for key in galaxea.VIDEO_KEYS.values()})
    info = {"codebase_version": "v2.1", "fps": 15, "data_path": "data/episode_{episode_index:06d}.parquet",
            "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4", "features": features}
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": n}) + "\n")
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "move the object"}) + "\n")
    columns = {key: [np.full(6, k / 100) for k in range(n)] for key in
               ["observation.state.left_arm", "observation.state.right_arm", "action.left_arm", "action.right_arm"]}
    columns.update({key: np.arange(n, dtype=float) for key in
                    ["observation.state.left_gripper", "observation.state.right_gripper",
                     "action.left_gripper", "action.right_gripper"]})
    columns.update({key: np.zeros(n, dtype=int) for key in
                    ["coarse_task_index", "task_index", "quality_index", "episode_index"]})
    columns.update({"frame_index": np.arange(n), "timestamp": np.arange(n) / 15})
    (root / "data").mkdir()
    pd.DataFrame(columns).to_parquet(root / "data" / "episode_000000.parquet")
    for key in galaxea.VIDEO_KEYS.values():
        path = root / "videos" / key / "episode_000000.mp4"
        path.parent.mkdir(parents=True)
        with av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate=15)
            stream.width, stream.height, stream.pix_fmt = 64, 36, "yuv420p"
            for k in range(n):
                frame = av.VideoFrame.from_ndarray(np.full((36, 64, 3), k * 9 % 256, np.uint8), format="rgb24")
                frame.pts = k
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)


@pytest.mark.parametrize("first_adapter", [True, False])
def test_selected_openaoe_keeps_original_adapter_output_identity(tmp_path, first_adapter):
    original = tmp_path / "original"
    for name, n in [("first", 10), ("second", 20)]:
        home = original / name / "raw_x_seg_1"
        home.mkdir(parents=True)
        _clip(home / "raw_video.mp4", n)
        if name == "second" or first_adapter:
            annotation = home / "ego_annotation" / "ego_action_annotation.json"
            annotation.parent.mkdir()
            annotation.write_text(json.dumps([{"start_ts": 0, "end_ts": 0.2,
                                                "atomic_action": [{"verb": "move", "object": "object"}]}]))
    context = original_context(original)
    full_dir = tmp_path / "full"
    full = formats.convert(original, "ego_head", full_dir, "identity", 900)
    assert not full["failed"] and len(full["episodes"]) == 2
    selected = tmp_path / "selected"
    shutil.copytree(original / "second", selected / "second")
    subset_dir = tmp_path / "subset"
    subset = formats.convert(selected, "ego_head", subset_dir, "identity", 900, ownership_context=context)
    assert not subset["failed"] and len(subset["episodes"]) == 1
    expected = full["episodes"][1]
    assert expected["episode_id"] == ("episode_raw_x_seg_1_2" if first_adapter else "episode_raw_x_seg_1")
    assert subset["episodes"][0]["episode_id"] == expected["episode_id"]
    assert_same_model_inputs(subset_dir / expected["episode_id"], full_dir / expected["episode_id"])


def test_galaxea_adapter_keeps_separate_normalized_output_names(tmp_path):
    original = tmp_path / "original"
    for name, n in [("collection-1", 10), ("collection_1", 20)]:
        galaxea_upload(original / name, n)
    output = tmp_path / "units"
    report = formats.convert(original, "teleop_arms", output, "identity", 900)
    assert not report["failed"] and len(report["episodes"]) == 2
    identities = [row["episode_id"] for row in report["episodes"]]
    assert len(set(identities)) == 2
    assert [json.loads((output / identity / "context.json").read_text())["n_state_frames"] for identity in identities] == [10, 20]
    for identity in identities:
        assert (output / identity / "state.npz").exists()


@pytest.mark.parametrize("manifest", [None, True, ["../outside.json"], ["/outside.json"],
                                      ["./raw_video.mp4"], ["raw_video.mp4", "raw_video.mp4"],
                                      ["nested\\raw_video.mp4"], ["missing.mp4"]])
def test_invalid_original_adapter_manifest_cannot_reallocate_identity(tmp_path, manifest):
    original = tmp_path / "original"
    original.mkdir()
    _clip(original / "raw_video.mp4", 10)
    context = original_context(original)
    context["original_files"] = manifest
    with pytest.raises(ValueError, match="original.*manifest"):
        formats.plan(original, ownership_context=context)


def test_stale_original_annotation_presence_cannot_change_an_adapter_identity(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    _clip(original / "raw_video.mp4", 10)
    context = original_context(original)
    context["original_files"].append("ego_annotation/ego_action_annotation.json")
    with pytest.raises(ValueError, match="original annotation"):
        formats.plan(original, ownership_context=context)


def test_original_raw_video_identity_requires_its_manifest(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    _clip(original / "raw_video.mp4", 10)
    context = original_context(original)
    del context["original_files"]
    with pytest.raises(ValueError, match="requires.*manifest"):
        formats.plan(original, ownership_context=context)
