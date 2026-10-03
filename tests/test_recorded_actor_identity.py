import json

import h5py
import numpy as np
import pytest

from checks import capture_qc
from label import episode as me
from prepare import formats


@pytest.mark.parametrize("side,cameras,want", [
    ("left", ["top", "wrist_left", "wrist_right"], "left"),
    ("left", ["top"], "left"),
    ("right", ["top", "wrist_left", "wrist_right"], "right"),
    ("right", ["top"], "right"),
    (None, ["top", "wrist_left", "wrist_right"], "recorded arm (side unknown)"),
    (None, ["top"], "recorded arm (side unknown)"),
    (None, ["top", "wrist_right"], "right"),
    ("left", ["top", "wrist_right"], "left"),
])
def test_actual_hdf_state_identity_does_not_follow_camera_order(tmp_path, side, cameras, want):
    upload = tmp_path / "upload"
    upload.mkdir()
    source = upload / "arm.h5"
    values = np.stack([np.full(7, k / 100) for k in range(40)])
    with h5py.File(source, "w") as f:
        f.attrs["fps"] = 10
        f.create_dataset("timestamps", data=np.arange(40) / 10)
        f.create_dataset("observations/" + (side + "/" if side else "") + "qpos", data=values)
        for camera in cameras:
            f.create_dataset("images/" + camera,
                             data=np.stack([np.full((36, 64, 3), k * 5, np.uint8) for k in range(40)]))
    report = formats.convert(upload, "teleop_arms", tmp_path / "units", "test", 900)
    assert not report["failed"] and len(report["episodes"]) == 1
    ep_dir = tmp_path / "units" / report["episodes"][0]["episode_id"]
    ep = me.load(ep_dir)
    np.testing.assert_array_equal(ep["state"], values.astype(np.float32))
    assert me.actors(ep) == [want]
    assert ep["context"]["state_actors"] == [want]
    prompt = me.build_request(ep_dir)["prompt"]
    rows = [line for line in prompt.splitlines() if line.startswith("  ") and "s | " in line]
    assert rows and all("| " + want + " " in row for row in rows)


def single_state(side=None):
    state = np.zeros((40, 7))
    state[10:, 6] = 1
    ctx = {"profile": "teleop_arms", "state_kind": "joints", "fps": 10,
           "cameras": {"exo": {"name": "top"}, "left": {"name": "left"}, "right": {"name": "right"}}}
    if side:
        ctx["source"] = {"format": "hdf5", "state": "observations/" + side + "/qpos"}
    return {"context": ctx, "state": state, "sources": {"exo": {}, "left": {}, "right": {}},
            "times": None, "kmap": {}}


@pytest.mark.parametrize("side,want", [("left", ["exo", "left"]), ("right", ["exo", "right"]),
                                        (None, ["exo", "left", "right"])])
def test_contacts_and_capture_checks_use_the_recorded_arm_mapping(side, want):
    ep = single_state(side)
    pl = {"n": 40, "ks": [0, 15, 39], "contact": [15]}
    assert me.contact_views(ep, pl) == [(15, want)]
    assert capture_qc.actor_views(ep, me.actors(ep)) == ([side] if side else [None])


def test_one_known_mounted_actor_and_two_arm_order_are_preserved():
    ep = single_state()
    ep["sources"].pop("left")
    ep["context"]["cameras"].pop("left")
    ep["context"]["cameras"]["right"]["name"] = "wrist"
    assert me.actors(ep) == ["wrist"]
    ep["state"] = np.zeros((40, 14))
    assert me.actors(ep) == ["left", "right"]


def test_actor_identity_survives_removing_the_mounted_camera(tmp_path):
    ep = single_state("left")
    formats.finish_episode(tmp_path, ep["context"], ep["sources"], state=ep["state"])
    ctx = json.loads((tmp_path / "context.json").read_text())
    ctx["cameras"].pop("left")
    ep["context"] = ctx
    ep["sources"].pop("left")
    assert me.actors(ep) == ["left"]
    assert capture_qc.actor_views(ep, me.actors(ep)) == [None]


@pytest.mark.parametrize("side,topic_names,want", [
    ("left", False, "left"), ("right", False, "right"),
    ("left", True, "left"), ("right", True, "right"),
    (None, False, "recorded arm (side unknown)"),
])
def test_actual_mcap_actor_identity_uses_the_selected_channel(tmp_path, side, topic_names, want):
    from test_reader_cameras import _json_mcap, _jpg_msg
    upload = tmp_path / "upload"
    upload.mkdir()
    topic = "/arm/joint_states" if topic_names or not side else "/" + side + "/joint_states"
    prefix = side + "_" if topic_names else ""
    names = [prefix + f"joint{i}" for i in range(1, 7)] + [prefix + "gripper"]
    _json_mcap(upload / "run.mcap", {
        "/camera/front/image": ("foxglove.CompressedImage", _jpg_msg),
        topic: ("sensor_msgs/msg/JointState", lambda k: {"name": names, "position": [k / 100] * 7}),
    }, n=90)
    report = formats.convert(upload, "teleop_arms", tmp_path / "units", "test", 900)
    assert not report["failed"] and len(report["episodes"]) == 1
    ep_dir = tmp_path / "units" / report["episodes"][0]["episode_id"]
    ep = me.load(ep_dir)
    assert ep["context"]["state_kind"] == "joints"
    assert me.actors(ep) == [want]
    assert ep["context"]["state_actors"] == [want]
    prompt = me.build_request(ep_dir)["prompt"]
    rows = [line for line in prompt.splitlines() if line.startswith("  ") and "s | " in line]
    assert rows and all("| " + want + " " in row for row in rows)


def test_parts_and_reanchor_keep_recorded_actor_identity(tmp_path, monkeypatch):
    from board import clips
    from label import pieces
    from test_camera_timing import recording
    ep_dir = recording(tmp_path, np.arange(90) / 30, np.arange(90) / 30 + 0.01)
    ep = me.load(ep_dir)
    ctx = ep["context"]
    ctx.update(state_kind="joints", source={"format": "hdf5", "state": "left/qpos"})
    state = np.tile(np.arange(90)[:, None] / 100, (1, 7))
    formats.finish_episode(ep_dir, ctx, ep["sources"], state=state, times=ep["times"])
    assert me.actors(me.load(ep_dir)) == ["left"]
    clips.drop_cameras(ep_dir, ["exo"])
    parent = me.load(ep_dir)
    assert me.actors(parent) == ["left"]
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 1.0)
    parts = pieces.write_pieces(ep_dir, tmp_path / "parts")
    assert len(parts) == 3
    for part in parts:
        loaded = me.load(part)
        assert loaded["context"]["state_actors"] == ["left"]
        assert me.actors(loaded) == ["left"]
    np.testing.assert_array_equal(np.concatenate([me.load(part)["state"] for part in parts]), parent["state"])


@pytest.mark.parametrize("images", [False, True])
@pytest.mark.parametrize("side", ["left", "right", None])
def test_actual_lerobot_state_value_names_keep_actor_identity(tmp_path, images, side):
    import pandas as pd
    from test_formats import _lerobot
    from test_reader_cameras import _lerobot_images
    root = tmp_path / "lr"
    values = np.tile(np.arange(30)[:, None] / 100, (1, 7))
    names = [(side + "_" if side else "") + f"joint{i}" for i in range(1, 7)] + [
        (side + "_" if side else "") + "gripper"]
    feature = {"dtype": "float32", "shape": [7], "names": names}
    if images:
        _lerobot_images(root)
        table_path = root / "data/chunk-000/episode_000000.parquet"
        table = pd.read_parquet(table_path)
        table["observation.state"] = list(values)
        table.to_parquet(table_path)
        info_path = root / "meta/info.json"
        info = json.loads(info_path.read_text())
        info["features"]["observation.state"] = feature
        info_path.write_text(json.dumps(info))
    else:
        _lerobot(root, {0: {"observation.state": list(values)}}, {"observation.state": feature})
    report = formats.convert(root, "teleop_arms", tmp_path / "units", "test", 900)
    assert not report["failed"] and len(report["episodes"]) == 1
    ep_dir = tmp_path / "units" / report["episodes"][0]["episode_id"]
    ep = me.load(ep_dir)
    np.testing.assert_array_equal(ep["state"], values.astype(np.float32))
    want = side or "recorded arm (side unknown)"
    assert me.actors(ep) == [want]
    assert ep["context"]["state_actors"] == [want]


def test_legacy_known_mounted_actor_survives_camera_removal(tmp_path):
    from board import clips
    from test_camera_timing import recording
    ep_dir = recording(tmp_path, np.arange(90) / 30, np.arange(90) / 30)
    ep = me.load(ep_dir)
    ctx = ep["context"]
    ctx["state_kind"] = "joints"
    formats.finish_episode(ep_dir, ctx, ep["sources"], state=np.zeros((90, 7)), times=ep["times"])
    ctx.pop("state_actors")
    (ep_dir / "context.json").write_text(json.dumps(ctx))
    assert me.actors(me.load(ep_dir)) == ["left"]
    clips.drop_cameras(ep_dir, ["left"])
    assert me.actors(me.load(ep_dir)) == ["left"]
