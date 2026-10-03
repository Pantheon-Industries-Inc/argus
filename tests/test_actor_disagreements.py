import json

import h5py
import numpy as np
import pytest

from board import build
from board.families import Families
from checks import capture_qc
from label import episode as me
from prepare import formats


UNKNOWN = "recorded arm (side unknown)"
RIGHT_NAMES = [f"right_joint{k}" for k in range(1, 7)] + ["right_gripper"]
LEFT_NAMES = [f"left_joint{k}" for k in range(1, 7)] + ["left_gripper"]
MIXED_NAMES = LEFT_NAMES[:3] + RIGHT_NAMES[3:]


def converted(root, tmp_path):
    report = formats.convert(root, "teleop_arms", tmp_path / "units", "test", 900)
    assert not report["failed"] and len(report["episodes"]) == 1
    return tmp_path / "units" / report["episodes"][0]["episode_id"]


def assert_conflict(ep_dir, source, names):
    ep = me.load(ep_dir)
    assert ep["context"]["state_kind"] == "joints"
    assert me.actors(ep) == [UNKNOWN]
    assert me.actor_views(ep) == capture_qc.actor_views(ep, me.actors(ep)) == [None]
    issues = [i for i in ep["context"].get("reader_issues", []) if i["kind"] == "state_identity_conflict"]
    assert len(issues) == 1
    identity = ep["context"]["state_identity"]
    assert identity["source"] == source and identity["names"] == names
    prompt = me.build_request(ep_dir)["prompt"]
    assert issues[0]["what"] in prompt
    assert source in prompt and all(name in prompt for name in names)
    assert '"arm" field always' not in prompt
    rows = [line for line in prompt.splitlines() if line.startswith("  ") and "s | " in line]
    assert rows and all("| " + UNKNOWN + " " in row for row in rows)
    card = {}
    build.add_context(card, ep["context"], ep_dir)
    fam = Families()
    slug = fam.reader_family("state_identity_conflict")
    assert slug in fam.catalog() and fam.catalog()[slug]["list"] == "data"
    assert slug in fam.classify(card)["counted"]
    return ep


@pytest.mark.parametrize("source,names,cameras", [
    ("observations/left/qpos", RIGHT_NAMES, ["top", "wrist_left"]),
    ("observations/right/qpos", LEFT_NAMES, ["top", "wrist_right"]),
    ("observations/left/qpos", RIGHT_NAMES, ["top", "wrist_left", "wrist_right"]),
    ("observations/qpos", MIXED_NAMES, ["top", "wrist_left"]),
])
def test_native_hdf_conflict_cannot_borrow_a_wrist_identity(tmp_path, source, names, cameras):
    root = tmp_path / "upload"
    root.mkdir()
    values = np.tile(np.arange(40)[:, None] / 100, (1, 7))
    with h5py.File(root / "arm.h5", "w") as f:
        f.attrs["fps"] = 10
        f["timestamps"] = np.arange(40) / 10
        f.create_dataset(source, data=values).attrs["names"] = names
        for camera in cameras:
            f["images/" + camera] = np.stack([np.full((36, 64, 3), k * 5, np.uint8) for k in range(40)])
    ep = assert_conflict(converted(root, tmp_path), source, names)
    np.testing.assert_array_equal(ep["state"], values.astype(np.float32))


@pytest.mark.parametrize("topic,names", [("/left/joint_states", RIGHT_NAMES), ("/right/joint_states", LEFT_NAMES)])
def test_native_mcap_topic_must_agree_with_its_value_names(tmp_path, topic, names):
    from test_reader_cameras import _json_mcap, _jpg_msg
    root = tmp_path / "upload"
    root.mkdir()
    _json_mcap(root / "run.mcap", {
        "/camera/front/image": ("foxglove.CompressedImage", _jpg_msg),
        "/camera/wrist_left/image": ("foxglove.CompressedImage", _jpg_msg),
        topic: ("sensor_msgs/msg/JointState", lambda k: {"name": names, "position": [k / 100] * 7}),
    }, n=90)
    ep = assert_conflict(converted(root, tmp_path), topic, names)
    np.testing.assert_allclose(ep["state"], np.tile(np.arange(90)[:, None] / 100, (1, 7)), atol=1e-6)


@pytest.mark.parametrize("images", [False, True])
def test_lerobot_mixed_side_names_remain_visible_in_the_request(tmp_path, images):
    import pandas as pd
    from test_formats import _lerobot
    from test_reader_cameras import _lerobot_images
    root = tmp_path / "lr"
    values = np.tile(np.arange(30)[:, None] / 100, (1, 7))
    feature = {"dtype": "float32", "shape": [7], "names": MIXED_NAMES}
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
    ep = assert_conflict(converted(root, tmp_path), "observation.state", MIXED_NAMES)
    np.testing.assert_array_equal(ep["state"], values.astype(np.float32))


@pytest.mark.parametrize("mcap", [False, True])
def test_state_beside_videos_preserves_its_native_identity_conflict(tmp_path, monkeypatch, mcap):
    from board import clips
    from label import pieces
    from test_formats import _videos_with_an_hdf5_arm_state
    from test_reader_cameras import _json_mcap
    root = tmp_path / "upload"
    folder = _videos_with_an_hdf5_arm_state(root, n=90)
    source = "/left/joint_states" if mcap else "observations/left/qpos"
    if mcap:
        (folder / "robot.h5").unlink()
        _json_mcap(folder / "robot.mcap", {
            source: ("sensor_msgs/msg/JointState", lambda k: {"name": RIGHT_NAMES, "position": [k / 100] * 7}),
        }, n=90)
    else:
        with h5py.File(folder / "robot.h5", "r+") as f:
            values = f["qpos"][:, :7]
            del f["qpos"]
            del f["action"]
            f.create_dataset(source, data=values).attrs["names"] = RIGHT_NAMES
            f["action"] = values + 0.01
    ep_dir = converted(root, tmp_path)
    parent = assert_conflict(ep_dir, source, RIGHT_NAMES)
    identity = parent["context"]["state_identity"]
    clips.drop_cameras(ep_dir, ["exo"])
    loaded = me.load(ep_dir)
    assert loaded["context"]["state_identity"] == identity
    assert me.actors(loaded) == [UNKNOWN] and me.actor_views(loaded) == [None]
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 1.0)
    parts = pieces.write_pieces(ep_dir, tmp_path / "parts")
    assert len(parts) == 3
    for part in parts:
        loaded = me.load(part)
        assert loaded["context"]["state_identity"] == identity
        assert me.actors(loaded) == [UNKNOWN] and me.actor_views(loaded) == [None]
    np.testing.assert_array_equal(np.concatenate([me.load(part)["state"] for part in parts]),
                                  me.load(ep_dir)["state"])


@pytest.mark.parametrize("state_side,camera_side,want_rule", [
    ("left", "right", False), ("right", "left", False),
    ("left", "left", True), ("right", "right", True),
])
def test_only_a_matching_state_and_camera_constrain_the_output_arm(tmp_path, state_side, camera_side, want_rule):
    root = tmp_path / "upload"
    root.mkdir()
    with h5py.File(root / "arm.h5", "w") as f:
        f.attrs["fps"] = 10
        f["timestamps"] = np.arange(40) / 10
        f["observations/" + state_side + "/qpos"] = np.tile(np.arange(40)[:, None] / 100, (1, 7))
        for camera in ["top", "wrist_" + camera_side]:
            f["images/" + camera] = np.stack([np.full((36, 64, 3), k * 5, np.uint8) for k in range(40)])
    ep_dir = converted(root, tmp_path)
    ep = me.load(ep_dir)
    assert me.actors(ep) == [state_side]
    assert me.actor_views(ep) == ([state_side] if want_rule else [None])
    prompt = me.build_request(ep_dir)["prompt"]
    assert ('"arm" field always' in prompt) == want_rule
    assert f"mounted on the {camera_side.upper()} arm" in prompt
    rows = [line for line in prompt.splitlines() if line.startswith("  ") and "s | " in line]
    assert rows and all("| " + state_side + " " in row for row in rows)


def test_a_camera_name_cannot_map_a_contradicted_actor():
    from test_recorded_actor_identity import single_state
    ep = single_state()
    ep["sources"].pop("right")
    ep["context"]["cameras"].pop("right")
    ep["context"]["cameras"]["left"]["name"] = UNKNOWN
    formats.record_state_identity(ep["context"], "observations/left/qpos", RIGHT_NAMES)
    assert me.actors(ep) == [UNKNOWN]
    assert me.actor_views(ep) == [None]
