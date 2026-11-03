import json

import h5py
import numpy as np
import pytest

from checks import capture_qc, stream_pairing
from label import episode as me
from prepare import formats
from test_actor_group_identity import upload_pair, values


UNKNOWN = ["recorded arm 1 (side unknown)", "recorded arm 2 (side unknown)"]
GENERIC = [f"joint{k}_rad" for k in range(1, 7)] + ["gripper"]


def unnamed_upload(root, case, names):
    data = upload_pair(root, case)
    if case in ("lr_video", "lr_image"):
        path = root / "meta/info.json"
        info = json.loads(path.read_text())
        info["features"]["observation.state"]["names"] = names
        path.write_text(json.dumps(info))
    else:
        raw = next(root.rglob("robot.h5")) if case == "beside_video" else root / "arm.h5"
        key = "qpos" if case == "beside_video" else "observations/qpos"
        with h5py.File(raw, "r+") as f:
            del f[key].attrs["names"]
            if names is not None:
                f[key].attrs["names"] = names
    return data


@pytest.mark.parametrize("case", ["hdf", "beside_video", "lr_video", "lr_image"])
@pytest.mark.parametrize("names", [None, GENERIC + GENERIC])
def test_fresh_native_unnamed_groups_have_no_guessed_side_or_wrist(tmp_path, case, names):
    root = tmp_path / "upload"
    data = unnamed_upload(root, case, names)
    report = formats.convert(root, "teleop_arms", tmp_path / "units", "test", 900)
    assert not report["failed"] and len(report["episodes"]) == 1
    path = tmp_path / "units" / report["episodes"][0]["episode_id"]
    ep = me.load(path)
    if names is None:
        assert ep["context"]["state_kind"] == "none"
        signals = ep["context"]["signals"]
        original = next(row for row in signals if row["dims"] == 14 and "action" not in row["name"])
        with np.load(path / "signals.npz") as arrays:
            np.testing.assert_array_equal(arrays[original["key"]], data)
        assert "state_contract" not in ep["context"]
        return
    np.testing.assert_array_equal(ep["state"], data)
    assert me.actors(ep) == UNKNOWN
    assert ep["context"]["state_actors"] == UNKNOWN
    assert me.actor_views(ep) == [None, None]
    assert capture_qc.canonical_states(ep)["actors"] == UNKNOWN
    paired = stream_pairing.pairing(path)
    assert paired is None or "not_assessed" in paired
    request = me.build_request(path)
    assert "state_identity_issues" in request["blocks"]
    assert all(name in request["prompt"] for name in UNKNOWN)
    assert ep["context"]["state_identity_note"] in request["prompt"]
    assert not any(i["kind"] == "state_identity_conflict" for i in ep["context"].get("reader_issues", []))


def test_fresh_unnamed_groups_ignore_an_unproven_saved_guess():
    from test_recorded_actor_identity import single_state
    ep = single_state()
    ep["state"] = values(40)
    ep["context"]["state_actors"] = ["left", "right"]
    formats.record_state_identity(ep["context"], "qpos", GENERIC + GENERIC, 14)
    assert me.actors(ep) == UNKNOWN
    assert me.actor_views(ep) == [None, None]


def test_explicit_ordered_source_contract_preserves_a_saved_mapping():
    from test_recorded_actor_identity import single_state
    ep = single_state()
    ep["state"] = values(40)
    ep["context"].update(state_actor_contract={
        "actors": ["right", "left"], "source": "recording schema maps group 1 to right and group 2 to left"})
    formats.record_state_identity(ep["context"], "observation.state", GENERIC + GENERIC, 14)
    assert me.actors(ep) == ["right", "left"]
    assert me.actor_views(ep) == ["right", "left"]
    assert "state_identity_note" not in ep["context"]


def test_old_saved_actor_order_remains_compatible_without_native_identity_metadata():
    from test_recorded_actor_identity import single_state
    ep = single_state()
    ep["state"] = values(40)
    ep["context"]["state_actors"] = ["right", "left"]
    assert me.actors(ep) == ["right", "left"]
    assert me.actor_views(ep) == ["right", "left"]


def test_unnamed_group_identity_survives_parts_reanchor_and_camera_removal(tmp_path, monkeypatch):
    from board import clips
    from label import pieces
    root = tmp_path / "upload"
    unnamed_upload(root, "hdf", GENERIC + GENERIC)
    report = formats.convert(root, "teleop_arms", tmp_path / "units", "test", 900)
    path = tmp_path / "units" / report["episodes"][0]["episode_id"]
    identities = me.load(path)["context"]["state_identities"]
    clips.drop_cameras(path, ["exo"])
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 1.0)
    parts = pieces.write_pieces(path, tmp_path / "parts")
    assert len(parts) == 4
    for part in [path] + parts:
        ep = me.load(part)
        assert ep["context"]["state_identities"] == identities
        assert me.actors(ep) == UNKNOWN
        assert me.actor_views(ep) == [None, None]
    np.testing.assert_array_equal(np.concatenate([me.load(part)["state"] for part in parts]), me.load(path)["state"])
    clips.drop_cameras(path, ["right"])
    assert me.actors(me.load(path)) == UNKNOWN
    assert me.actor_views(me.load(path)) == [None, None]


def test_sided_source_columns_preserve_explicit_adapter_order(tmp_path):
    from prepare import galaxea
    from test_prepare import _galaxea_folder
    root = tmp_path / "native"
    _galaxea_folder(root)
    meta = galaxea.meta_from(lambda rel: root / rel, "native")
    ep = meta["episodes"][0]
    path = tmp_path / "episode"
    ctx = galaxea.write_episode(meta, ep, lambda rel: root / rel, path, "test")
    loaded = me.load(path)
    assert [i["side"] for i in ctx["state_identities"]] == ["left", "right"]
    assert [i["source"] for i in ctx["state_identities"]] == [
        "observation.state.left_arm", "observation.state.right_arm"]
    assert me.actors(loaded) == ["left", "right"]
    assert me.actor_views(loaded) == ["left", "right"]


@pytest.mark.parametrize("adapter", ["packed", "individual"])
@pytest.mark.parametrize("known", [False, True])
def test_direct_state_writers_retain_native_identity_before_discarding_metadata(tmp_path, monkeypatch, adapter, known):
    import pandas as pd
    from prepare import fastumi, molmo
    from test_actor_group_identity import LEFT, RIGHT
    from test_prepare import _mp4
    raw, out, n = tmp_path / "raw", tmp_path / "out", 12
    data = values(n)
    names = RIGHT + LEFT if known else GENERIC + GENERIC
    feature = {"dtype": "float32", "shape": [14], "names": names}
    if adapter == "packed":
        path = molmo.data_path(raw, 0, 0)
        path.parent.mkdir(parents=True)
        pd.DataFrame({"observation.state": list(data), "action": list(data),
                      "frame_index": range(n), "episode_index": [0] * n}).to_parquet(path)
        row = {"episode_index": 0, "data/chunk_index": 0, "data/file_index": 0, "tasks": ["move"]}
        for cam in molmo.VKEYS:
            _mp4(molmo.packed_path(raw, cam, 0, 0), n)
            row.update({f"videos/observation.images.{cam}/chunk_index": 0,
                        f"videos/observation.images.{cam}/file_index": 0,
                        f"videos/observation.images.{cam}/from_timestamp": 0,
                        f"videos/observation.images.{cam}/to_timestamp": n / 30})
        molmo.prepare_episode(row, raw, out, "move", False,
                              {"fps": 30, "features": {"observation.state": feature}})
        path = out / "episode_000000"
    else:
        root = raw / "dual_arm/move"
        (root / "meta").mkdir(parents=True)
        keys = list(fastumi.CAMS)[:2]
        info = {"fps": 30, "chunks_size": 1000,
                "data_path": "data/episode_{episode_index:06d}.parquet",
                "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {"observation.state": feature, **{key: {"dtype": "video"} for key in keys}}}
        (root / "meta/info.json").write_text(json.dumps(info))
        (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": n, "tasks": ["move"]}))
        (root / "data").mkdir()
        target = np.concatenate([data[1:], data[-1:]])
        pd.DataFrame({"observation.state": list(data), "action": list(target),
                      "frame_index": range(n)}).to_parquet(root / "data/episode_000000.parquet")
        for key in keys:
            _mp4(root / f"videos/{key}/episode_000000.mp4", n)
        monkeypatch.setattr(fastumi, "_dl", lambda rel, folder: folder / rel)
        monkeypatch.setattr(fastumi.hub, "ls", lambda *args: pytest.fail("no recorded task codes need a manifest"))
        fastumi.prepare_one("dual_arm/move/0", raw, out, False)
        path = out / fastumi.episode_dir_name("dual_arm/move/0")
    ep = me.load(path)
    np.testing.assert_array_equal(ep["state"], data)
    if adapter == "packed":
        from prepare.state_contract import action_semantics
        assert action_semantics(ep["context"])["role"] == "command"
        assert "timebase" in me.plan(ep)["checks"]
        old_proof = {'version': 1, 'producer': 'prepare.molmo', 'kind': 'joints', 'native_contract': 'molmo'}
        assert action_semantics({'state_kind': 'joints', 'state_contract': old_proof})['role'] == 'command'
        old_proof = {'version': 1, 'producer': 'prepare.habit', 'kind': 'ee_pose', 'native_contract': 'habit'}
        assert action_semantics({'state_kind': 'ee_pose', 'state_contract': old_proof})['role'] == 'command'
        assert action_semantics({'state_kind': 'ee_pose', 'source': {'adapter': 'habit'}})['role'] == 'command'
        from prepare.state_contract import native_contract
        conflicting = native_contract('habit', 'ee_pose', {'action_semantics': {'version': 1, 'role': 'target'}})
        assert action_semantics({'state_kind': 'ee_pose', 'state_contract': conflicting})['role'] == 'action/target_unknown'
        old_proof['producer'] = 'unverified producer'
        assert action_semantics({'state_kind': 'ee_pose', 'state_contract': old_proof})['role'] == 'action/target_unknown'
    if adapter == "individual":
        from label import evidence_access, dictionary
        semantics = ep["context"]["action_semantics"]
        assert semantics["role"] == "target"
        assert semantics["relationship"] == {"kind": "next_observation", "field": "state",
                                              "row_offset": 1, "terminal": "repeat_last", "scope": "source_recording"}
        assert ep["action"].tobytes() == target.tobytes()
        a = evidence_access.Access(ep, {})
        field = next(f for f in a.inventory() if f["name"] == "action")
        assert field["descriptor"]["recorded_role"] == "target"
        assert field["descriptor"]["action_semantics"] == semantics
        # its episodes.jsonl row holds only LeRobot's own columns (index, length, task text), which the prompt
        # already carries, so like the recording without it, only a changed target opts it in
        assert not evidence_access.needs_inspection(ep, {}, 'all')
        assert ep['context']['recorded_metadata'][str(root / 'meta/episodes.jsonl')][0]['fields']['tasks'] == ['move']
        observed_only = {**ep, 'context': {k: v for k, v in ep['context'].items() if k != 'recorded_metadata'}}
        assert not evidence_access.needs_inspection(observed_only, {}, 'all')
        changed_target = ep['action'].copy()
        changed_target[0, 0] += 1
        assert evidence_access.needs_inspection({**observed_only, 'action': changed_target}, {}, 'all')
        assert evidence_access.needs_inspection({**observed_only, 'action': ep['action'][:-1]}, {}, 'all')
        assert evidence_access.needs_inspection({**ep, 'context': {**ep['context'], 'state_unaligned': True}}, {}, 'all')
        inventory = dictionary.inventory([path])
        field = next(f for f in inventory["fields"] if f["kind"] == "action")
        assert field["action_semantics"] == semantics
        state_field = next(f for f in inventory['fields'] if f['kind'] == 'state')
        assert 'recorded_role' not in state_field and 'action_semantics' not in state_field
        from prepare.state_contract import action_semantics
        old_proof = {'version': 1, 'producer': 'prepare.fastumi', 'kind': 'ee_pose', 'native_contract': 'fastumi'}
        assert action_semantics({'state_kind': 'ee_pose', 'state_contract': old_proof}) == semantics
        assert "next observed state" in me.build_prompt(ep, me.plan(ep), cell_w=320, cell_h=240)[1]
        assert 'action_semantics' in json.dumps(dictionary._request(inventory))
        info['features']['action'] = {'action_semantics': {'version': 1, 'role': 'command'}}
        (root / 'meta/info.json').write_text(json.dumps(info))
        fastumi.prepare_one('dual_arm/move/0', raw, out, True)
        conflict = me.load(path)
        assert action_semantics(conflict['context'])['role'] == 'action/target_unknown'
        assert evidence_access.needs_inspection(conflict, {}, 'all')
        assert conflict['action'].tobytes() == target.tobytes()
    want = ["right", "left"] if known else UNKNOWN
    if not known and adapter == "individual":
        want = ["recorded gripper 1 (side unknown)", "recorded gripper 2 (side unknown)"]
    assert me.actors(ep) == want
    assert me.actor_views(ep) == (["right", "left"] if known else [None, None])
    assert [i["source"] for i in ep["context"]["state_identities"]] == ["observation.state"] * 2
