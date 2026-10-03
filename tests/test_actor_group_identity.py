import json

import h5py
import numpy as np
import pytest

from checks import capture_qc
from label import episode as me
from prepare import formats


LEFT = [f"left_joint{k}" for k in range(1, 7)] + ["left_gripper"]
RIGHT = [f"right_joint{k}" for k in range(1, 7)] + ["right_gripper"]
MIXED = LEFT[:3] + RIGHT[3:]


def values(n):
    data = np.tile(np.array([.1] * 6 + [.25] + [.9] * 6 + [.75], np.float32), (n, 1))
    data[n // 2:, 6] = .35
    return data


def upload_pair(root, case):
    names = MIXED + LEFT if case == "mixed" else LEFT + LEFT if case == "duplicate" else RIGHT + LEFT
    if case in ("lr_video", "lr_image"):
        import pandas as pd
        from test_formats import _lerobot
        from test_reader_cameras import _lerobot_images
        data = values(30)
        feature = {"dtype": "float32", "shape": [14], "names": names}
        if case == "lr_video":
            _lerobot(root, {0: {"observation.state": list(data)}}, {"observation.state": feature})
        else:
            _lerobot_images(root, wrist=True)
            path = root / "data/chunk-000/episode_000000.parquet"
            table = pd.read_parquet(path)
            table["observation.state"] = list(data)
            table.to_parquet(path)
            path = root / "meta/info.json"
            info = json.loads(path.read_text())
            info["features"]["observation.state"] = feature
            path.write_text(json.dumps(info))
    elif case in ("mcap_conflict", "mcap_agree"):
        from test_reader_cameras import _json_mcap, _jpg_msg
        root.mkdir()
        data = values(90)
        _json_mcap(root / "run.mcap", {
            "/camera/front/image": ("foxglove.CompressedImage", _jpg_msg),
            "/camera/wrist_left/image": ("foxglove.CompressedImage", _jpg_msg),
            "/camera/wrist_right/image": ("foxglove.CompressedImage", _jpg_msg),
            "/left/joint_states": ("sensor_msgs/msg/JointState", lambda k: {
                "name": RIGHT if case == "mcap_conflict" else LEFT, "position": data[k, :7].tolist()}),
            "/right/joint_states": ("sensor_msgs/msg/JointState", lambda k: {
                "name": LEFT if case == "mcap_conflict" else RIGHT, "position": data[k, 7:].tolist()}),
        }, n=90)
    elif case in ("beside_video", "split_conflict"):
        from test_formats import _videos_with_an_hdf5_arm_state
        folder = _videos_with_an_hdf5_arm_state(root, n=90)
        with h5py.File(folder / "robot.h5", "r+") as f:
            data = values(90)
            del f["timestamps"]
            f["timestamps"] = ((1_790_000_000.0 + np.arange(90) / 30) * 1e9).astype(np.int64)
            del f["action"]
            f["action"] = data + .01
            del f["qpos"]
            if case == "split_conflict":
                f.create_dataset("observations/left/qpos", data=data[:, :7]).attrs["names"] = RIGHT
                f.create_dataset("observations/right/qpos", data=data[:, 7:]).attrs["names"] = LEFT
            else:
                f.create_dataset("qpos", data=data).attrs["names"] = names
    else:
        root.mkdir()
        data = values(40)
        with h5py.File(root / "arm.h5", "w") as f:
            f.attrs["fps"] = 10
            f["timestamps"] = np.arange(40) / 10
            if case == "canonical":
                names = LEFT + RIGHT
            f.create_dataset("observations/qpos", data=data).attrs["names"] = names
            cameras = ["top", "wrist_right"] if case == "subset" else ["top", "wrist_left", "wrist_right"]
            for camera in cameras:
                f["images/" + camera] = np.stack([np.full((36, 64, 3), k * 5, np.uint8) for k in range(40)])
    return data


def converted(tmp_path, case):
    root = tmp_path / "upload"
    data = upload_pair(root, case)
    report = formats.convert(root, "teleop_arms", tmp_path / "units", "test", 900)
    assert not report["failed"] and len(report["episodes"]) == 1
    path = tmp_path / "units" / report["episodes"][0]["episode_id"]
    ep = me.load(path)
    np.testing.assert_allclose(ep["state"], data, atol=1e-6)
    return path, ep


@pytest.mark.parametrize("case", ["hdf", "subset", "lr_video", "lr_image", "beside_video"])
def test_native_reversed_groups_keep_their_recorded_order(tmp_path, case):
    path, ep = converted(tmp_path, case)
    assert me.actors(ep) == ["right", "left"]
    mapping = [v if v in me.views(ep) else None for v in ["right", "left"]]
    assert me.actor_views(ep) == capture_qc.actor_views(ep, me.actors(ep)) == mapping
    canonical = capture_qc.canonical_states(ep)
    assert canonical["actors"] == ["right", "left"]
    np.testing.assert_array_equal(canonical["states"][:, [6, 13]], ep["state"][:, [6, 13]])
    prompt = me.build_request(path)["prompt"]
    rows = [line for line in prompt.splitlines() if line.startswith("  ") and "s | " in line]
    assert rows and all(row.index("| right ") < row.index("| left ") for row in rows)
    views = me.contact_views(ep, {"n": len(ep["state"]), "ks": [0, len(ep["state"]) - 1],
                                  "contact": [len(ep["state"]) - 1]})
    want = ["exo", "right"] if "right" in me.views(ep) else list(me.views(ep))
    assert views == [(len(ep["state"]) - 1, want)]


@pytest.mark.parametrize("case,mapping", [("mixed", [None, "left"]), ("split_conflict", [None, None]),
                                        ("mcap_conflict", [None, None])])
def test_native_conflicting_groups_remain_qualified_beside_known_groups(tmp_path, case, mapping):
    path, ep = converted(tmp_path, case)
    issues = [i for i in ep["context"].get("reader_issues", []) if i["kind"] == "state_identity_conflict"]
    assert issues
    assert me.actor_views(ep) == mapping
    assert len(set(me.actors(ep))) == 2
    if case == "mixed":
        assert me.actors(ep)[1] == "left"
    prompt = me.build_request(path)["prompt"]
    assert all(i["what"] in prompt for i in issues)


def test_duplicate_side_claims_keep_both_groups_without_inventing_an_opposite_arm(tmp_path):
    path, ep = converted(tmp_path, "duplicate")
    names = me.actors(ep)
    assert len(names) == len(set(names)) == 2
    assert all("left" in name for name in names) and "right" not in names
    assert me.actor_views(ep) == [None, None]
    from label import state as ms
    rows = ms.recorded_joint_motion(ep["state"], [0, len(ep["state"]) - 1], names)
    assert list(rows[0]["arms"]) == names
    request = me.build_request(path)
    assert "state_identity_issues" in request["blocks"]
    prompt = request["prompt"]
    assert all(name in prompt for name in names)


@pytest.mark.parametrize("case", ["canonical", "mcap_agree"])
def test_native_canonical_group_identity_keeps_its_existing_mapping(tmp_path, case):
    path, ep = converted(tmp_path, case)
    assert me.actors(ep) == ["left", "right"]
    assert me.actor_views(ep) == ["left", "right"]
    assert "state_identity_issues" not in me.build_request(path)["blocks"]


def test_ordered_group_identity_survives_parts_reanchor_and_camera_removal(tmp_path, monkeypatch):
    from board import clips
    from label import pieces
    path, ep = converted(tmp_path, "hdf")
    assert me.actors(ep) == ["right", "left"]
    identities = ep["context"]["state_identities"]
    clips.drop_cameras(path, ["exo"])
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 1.0)
    parts = pieces.write_pieces(path, tmp_path / "parts")
    assert len(parts) == 4
    for part in [path] + parts:
        loaded = me.load(part)
        assert loaded["context"]["state_identities"] == identities
        assert me.actors(loaded) == ["right", "left"]
        assert me.actor_views(loaded) == ["right", "left"]
    np.testing.assert_array_equal(np.concatenate([me.load(part)["state"] for part in parts]), me.load(path)["state"])
    clips.drop_cameras(path, ["right"])
    assert me.actors(me.load(path)) == ["right", "left"]
    assert me.actor_views(me.load(path)) == [None, "left"]


def test_a_known_group_keeps_its_side_beside_an_unnamed_group():
    from test_recorded_actor_identity import single_state
    ep = single_state()
    ep["state"] = values(40)
    names = [f"joint{k}" for k in range(1, 7)] + ["gripper"] + LEFT
    formats.record_state_identity(ep["context"], "qpos", names, 14)
    assert me.actors(ep) == ["recorded arm 1 (side unknown)", "left"]
    assert me.actor_views(ep) == [None, "left"]


def test_two_unnamed_groups_keep_the_existing_layout_convention():
    from test_recorded_actor_identity import single_state
    ep = single_state()
    ep["state"] = values(40)
    formats.record_state_identity(ep["context"], "qpos", None, 14)
    assert me.actors(ep) == ["left", "right"]
    assert me.actor_views(ep) == ["left", "right"]


def test_capture_positional_slots_follow_the_recorded_actor_and_its_wrist(monkeypatch):
    from checks.vendor import public_dataset_adapter_qc as up
    from test_capture_checks import _two_cameras
    feats = _two_cameras(False)
    formats.record_state_identity(feats["ep"]["context"], "state", RIGHT + LEFT, 14)
    calls = []
    original = up._largest_action_video_checks

    def largest(actions, valid, cameras, policy):
        calls.append(list(cameras))
        return original(actions, valid, cameras, policy)

    def correlation(row, policy):
        np.testing.assert_array_equal(row["left_pixel_change_amount"], np.r_[0, feats["cams"]["right"]["pchange"]])
        np.testing.assert_array_equal(row["right_pixel_change_amount"], np.r_[0, feats["cams"]["left"]["pchange"]])
        return {"left": {"status": "matched", "correlation": .123},
                "right": {"status": "matched", "correlation": .789}}, []

    monkeypatch.setattr(up, "_largest_action_video_checks", largest)
    monkeypatch.setattr(up, "visual_action_correlation_checks", correlation)
    assessed = capture_qc.assess(feats)
    assert calls == [["right"], ["left"]]
    assert assessed["checks"]["pixel_action_corr_mismatch"]["metrics"] == {
        "right": {"status": "matched", "correlation": .123},
        "left": {"status": "matched", "correlation": .789}}
    assert list(assessed["actors"]) == ["right", "left"]


def test_duplicate_groups_do_not_overwrite_capture_metrics():
    from test_capture_checks import _two_cameras
    feats = _two_cameras(False)
    formats.record_state_identity(feats["ep"]["context"], "state", LEFT + LEFT, 14)
    names = me.actors(feats["ep"])
    assert names == ["left (recorded group 1)", "left (recorded group 2)"]
    assessed = capture_qc.assess(feats)
    assert list(assessed["actors"]) == names


def test_stream_pairing_compares_each_wrist_with_its_proven_actor(tmp_path):
    from checks import stream_pairing
    from test_checks import two_grippers, levels_for, write_episode
    state, first, second = two_grippers()
    ctx = {}
    formats.record_state_identity(ctx, "state", RIGHT + LEFT, 14)
    path = write_episode(tmp_path / "episode", state,
                         {"right": levels_for(first), "left": levels_for(second)}, **ctx)
    result = stream_pairing.pairing(path)
    assert result["crossed"] is False
    assert result["right_vs_right"] > .9 and result["left_vs_left"] > .9


@pytest.mark.parametrize("names", [MIXED + LEFT, LEFT + LEFT])
def test_stream_pairing_does_not_guess_an_unresolved_or_duplicate_wrist(tmp_path, names):
    from checks import stream_pairing
    from test_checks import two_grippers, levels_for, write_episode
    state, first, second = two_grippers()
    ctx = {}
    formats.record_state_identity(ctx, "state", names, 14)
    path = write_episode(tmp_path / "episode", state,
                         {"left": levels_for(first), "right": levels_for(second)}, **ctx)
    result = stream_pairing.pairing(path)
    assert "not_assessed" in result and "crossed" not in result


def test_recorded_jumps_cannot_map_a_conflict_through_a_camera_display_name(tmp_path):
    from checks import stream_pairing
    from test_checks import two_grippers, levels_for, write_episode
    state, first, _ = two_grippers(1)
    state[60:, 0] += .1
    ctx = {"cameras": {"left": {"name": "recorded gripper (side unknown)"}}}
    formats.record_state_identity(ctx, "left/state", RIGHT)
    path = write_episode(tmp_path / "episode", state[:, :7], {"left": levels_for(first)}, **ctx)
    result = stream_pairing.jumps(path)
    assert result["events"] and all(event["camera"] is None for event in result["events"])
