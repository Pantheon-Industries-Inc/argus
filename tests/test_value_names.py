"""Every array a reader keeps carries the names its recording or a rig description beside it gives its values, and an
array nothing names is marked unnamed (prepare/value_names.py, attached in prepare/formats.py write_signals).

The YAM fixture follows a robots_realtime episode folder: one MCAP per node (yam_left with a joint_state of joint_pos,
joint_vel and gripper_pos as JSON, yam_leader_left with a seven value joint_pos), each camera's video with its frame
capture times, session_meta.json, and the station.yaml of schema pantheon.station/v1, which names the arms and their
parts but no joint."""
import json
from pathlib import Path

import numpy as np

from prepare import formats as f
from prepare import value_names as vn

T0 = 1_790_000_000.0
STATION = """\
schema: pantheon.station/v1
station:
  name: gem0
  machine_id: 0
  role: leader_follower
  revision: 1
  updated_at: '2026-08-22T22:32:38Z'
  updated_by: someone@example.com
  note: fixture
  digest: sha256:0000
dimensions:
  exo_cam_height_mm: null
  leader_separation_mm: null
hardware:
  gem:
    kind: gem-computer
    hostname: pantheon-gem0
  exo_cam:
    kind: realsense-d405
    serial: ''
  arms:
  - name: right
    leader: {kind: gello-dynamixel, ftdi_serial: X, pico_serial: Y, trigger_firmware: null}
    follower: {kind: yam, serial: null, can_serial: Z, can_channel: can_follower_r}
    wrist_cam: {kind: realsense-d405, serial: null}
    gripper: {kind: custom-yam-gripper, instance: 1, full_travel: 542}
  - name: left
    leader: {kind: gello-dynamixel, ftdi_serial: X2, pico_serial: Y2, trigger_firmware: null}
    follower: {kind: yam, serial: null, can_serial: Z2, can_channel: can_follower_l}
    wrist_cam: {kind: realsense-d405, serial: null}
    gripper: {kind: custom-yam-gripper, instance: 1, full_travel: 520}
"""
SESSION = {"prompt": "Stack the cups.", "nodes": [
    {"name": "yam_leader_left", "published_topics": ["joint_pos", "record", "health"],
     "subscribed_topics": ["yam_left/joint_state"]},
    {"name": "yam_left", "published_topics": ["joint_state", "health"], "subscribed_topics": [], "robot_model": "yam"},
    {"name": "hud", "published_topics": [], "subscribed_topics": ["exo_cam/rgb", "yam_left/joint_state",
                                                                  "yam_leader_left/joint_pos"]}]}
JOINTS6 = ["waist", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3"]


def _clip(path: Path, n: int) -> None:
    import av
    c = av.open(str(path), "w")
    s = c.add_stream("mpeg4", rate=30)
    s.width, s.height, s.pix_fmt = 64, 36, "yuv420p"
    for k in range(n):
        fr = av.VideoFrame.from_ndarray(np.full((36, 64, 3), k * 9 % 256, np.uint8), format="rgb24")
        fr.pts = k
        for pkt in s.encode(fr):
            c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()


def _node_mcap(path: Path, topic: str, row, *, channel_metadata=None, schema=b'{"type": "object"}', records=(),
               start=T0 - 0.1):
    """One robots_realtime node's MCAP: JSON messages on topic at 100 Hz on the recorder clock from start, row(k)
    each."""
    from mcap.writer import Writer
    with open(path, "wb") as fh:
        w = Writer(fh)
        w.start()
        sid = w.register_schema(name=topic, encoding="jsonschema", data=schema)
        cid = w.register_channel(topic=topic, message_encoding="json", schema_id=sid,
                                 metadata=channel_metadata or {})
        for k in range(int(0.8 * 100)):
            ns = int((start + k / 100) * 1e9)
            w.add_message(cid, log_time=ns, publish_time=ns, data=json.dumps(row(k)).encode())
        for name, data in records:
            w.add_metadata(name, data)
        w.finish()


def _follower(k):
    return {"joint_pos": [0.3 * np.sin(0.05 * k + j) for j in range(6)],
            "joint_vel": [0.2 * np.cos(0.05 * k + j) for j in range(6)], "gripper_pos": [0.5 + 0.4 * np.sin(0.1 * k)]}


def _leader(k):
    return {"joint_pos": [0.3 * np.sin(0.05 * k + j) for j in range(6)] + [0.5 + 0.4 * np.sin(0.1 * k)]}


def _yam_episode(root: Path, n: int = 18, **left) -> Path:
    d = root / "episode_125625_df6f654c"
    d.mkdir(parents=True)
    _clip(d / "exo_cam-images-rgb.mp4", n)
    np.save(d / "exo_cam-rgb-timestamp.npy", T0 + np.arange(n) / 30)
    _node_mcap(d / "yam_left.mcap", "/yam_left/joint_state", _follower, **left)
    _node_mcap(d / "yam_right.mcap", "/yam_right/joint_state", _follower)
    _node_mcap(d / "yam_leader_left.mcap", "/yam_leader_left/joint_pos", _leader)
    (d / "station.yaml").write_text(STATION)
    (d / "session_meta.json").write_text(json.dumps(SESSION))
    return d


def _read(tmp_path, root):
    from label import episode as me
    report = f.convert(root, "teleop_arms", tmp_path / "episodes", "yam", 900)
    assert not report["failed"], report
    folder = tmp_path / "episodes" / report["episodes"][0]["episode_id"]
    ctx = json.loads((folder / "context.json").read_text())
    request = me.build_request(folder, inspect_evidence=False)
    prompt = "\n".join(c.get("text", "") for c in request["content"] if isinstance(c, dict))
    return report, {s["name"]: s for s in ctx["signals"]}, prompt, ctx


def test_a_yam_recording_whose_station_names_no_joint_keeps_its_arrays_marked_unnamed(tmp_path):
    """The YAM joints reached the model as "(6 values)", with nothing saying that no name was ever found and the
    station.yaml beside them listed as a file no reader opens. The station names the arms (right, left) and their
    parts, never a joint, so every array of two or more values stays as recorded and is marked unnamed, with the
    descriptions searched, and no name list of the station or the session (arms, topics) is put on any array."""
    root = tmp_path / "upload"
    _yam_episode(root)
    report, signals, prompt, _ = _read(tmp_path, root)
    pos = signals["/yam_left/joint_state joint_pos"]
    assert pos["dims"] == 6 and pos.get("unnamed") is True and "names" not in pos
    assert pos["names_searched"] == ["session_meta.json", "station.yaml"]
    assert signals["/yam_leader_left/joint_pos joint_pos"]["unnamed"] is True
    assert "unnamed" not in signals["/yam_left/joint_state gripper_pos"]          # one value is named by its field
    assert not any(s.get("names_source") for s in signals.values())
    assert "/yam_left/joint_state joint_pos (6 unnamed values):" in prompt
    assert "/yam_leader_left/joint_pos joint_pos (7 unnamed values):" in prompt
    assert not any("station.yaml" in m for m in report["missing"]), report["missing"]


def test_a_rig_description_beside_the_recording_names_exactly_the_arrays_it_names(tmp_path):
    """A description beside the recording that names a field's values gives that field its names, and the readout
    says where they came from. It names only the field it is written for: the left arm's joint_pos, never the right
    arm's or the left arm's joint_vel of the same width."""
    root = tmp_path / "upload"
    d = _yam_episode(root)
    (d / "rig.yaml").write_text("/yam_left/joint_state:\n  joint_pos_names: [" + ", ".join(JOINTS6) + "]\n"
                                "yam_leader_left/joint_pos/joint_pos:\n  names: [j1, j2, j3, j4, j5, j6, grip]\n")
    _, signals, prompt, _ = _read(tmp_path, root)
    left = signals["/yam_left/joint_state joint_pos"]
    assert left["names"] == JOINTS6 and left["names_source"] == "episode_125625_df6f654c/rig.yaml"
    assert "unnamed" not in left
    assert signals["/yam_leader_left/joint_pos joint_pos"]["names"] == ["j1", "j2", "j3", "j4", "j5", "j6", "grip"]
    assert signals["/yam_right/joint_state joint_pos"]["unnamed"] is True
    assert signals["/yam_left/joint_state joint_vel"]["unnamed"] is True
    assert ("/yam_left/joint_state joint_pos (6 values (waist, shoulder, elbow, wrist_1, wrist_2, wrist_3), names "
            "from episode_125625_df6f654c/rig.yaml)") in prompt


def test_an_mcap_names_its_values_in_channel_metadata_a_metadata_record_or_its_json_schema(tmp_path):
    """An MCAP can name a channel's values outside its messages: in the channel's metadata, in a metadata record, and
    in the channel's JSON schema (prefixItems titles). Each names the field it is for."""
    root = tmp_path / "upload"
    schema = {"type": "object", "properties": {"joint_vel": {"type": "array", "prefixItems": [
        {"title": f"v{j}"} for j in range(6)]}}}
    _yam_episode(root, channel_metadata={"joint_pos_names": json.dumps(JOINTS6)},
                 schema=json.dumps(schema).encode(),
                 records=[("robot", {"/yam_left/joint_state/joint_vel.names": json.dumps([f"v{j}" for j in range(6)])})])
    _, signals, _, _ = _read(tmp_path, root)
    assert signals["/yam_left/joint_state joint_pos"]["names"] == JOINTS6
    assert signals["/yam_left/joint_state joint_pos"]["names_source"] == "yam_left.mcap channel metadata"
    vel = signals["/yam_left/joint_state joint_vel"]
    assert vel["names"] == [f"v{j}" for j in range(6)]
    assert "yam_left.mcap schema /yam_left/joint_state" in vel["names_source"]
    assert signals["/yam_right/joint_state joint_vel"]["unnamed"] is True


# What robots_realtime's recorder (PR 188) declares on each channel's metadata, exactly as it writes it
JOINTS = json.dumps(["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"])
FOLLOWER_METADATA = {
    "joint_pos_names": JOINTS, "joint_pos_unit": "rad",
    "gripper_pos_names": '["gripper"]', "gripper_pos_unit": "normalized",
    "joint_vel_names": JOINTS, "joint_vel_unit": "rad/s",
    "gripper_vel_names": '["gripper"]', "gripper_vel_unit": "normalized/s",
    "joint_eff_names": JOINTS, "joint_eff_unit": "N·m",
    "gripper_eff_names": '["gripper"]', "gripper_eff_unit": "N·m"}
LEADER_METADATA = {
    "joint_pos_names": json.dumps(["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]),
    "joint_pos_units": json.dumps(["rad"] * 6 + ["normalized"])}


def _recorded_follower(scale=1.0, phase=0.0):
    def row(k):
        return {"joint_pos": [scale * 0.3 * np.sin(0.05 * k + j + phase) for j in range(6)],
                "joint_vel": [0.2 * np.cos(0.05 * k + j) for j in range(6)],
                "joint_eff": [0.1 * np.cos(0.03 * k + j) for j in range(6)],
                "gripper_pos": [0.5 + 0.4 * np.sin(0.1 * k)], "gripper_vel": [0.0], "gripper_eff": [0.0]}
    return row


def _declared_yam_episode(root: Path, scale=1.0, declare=True, n=18) -> Path:
    d = root / "episode_125625_df6f654c"
    d.mkdir(parents=True)
    _clip(d / "exo_cam-images-rgb.mp4", n)
    np.save(d / "exo_cam-rgb-timestamp.npy", T0 + np.arange(n) / 30)
    for side, phase in (("left", 0.0), ("right", 1.0)):
        _node_mcap(d / f"yam_{side}.mcap", f"/yam_{side}/joint_state", _recorded_follower(scale, phase),
                   channel_metadata=FOLLOWER_METADATA if declare else None)
    _node_mcap(d / "yam_leader_left.mcap", "/yam_leader_left/joint_pos", _leader,
               channel_metadata=LEADER_METADATA if declare else None)
    (d / "station.yaml").write_text(STATION)
    (d / "session_meta.json").write_text(json.dumps(SESSION))
    return d


def test_an_arm_whose_channel_declares_its_joint_names_and_radians_is_read_as_the_arm_state(tmp_path):
    """robots_realtime records bare JSON joint lists and declares each field's names and unit on the channel's
    metadata. The arm state reader took a row's names only from the message and its units only from a channel-wide
    unit key, so these arms stayed signals and the episode had no arm state. Names and units the channel declares for
    the fields that form the row now make it the state, in radians, and the prompt shows the recorded arm motion."""
    root = tmp_path / "upload"
    _declared_yam_episode(root)
    _, signals, prompt, ctx = _read(tmp_path, root)
    assert ctx["state_kind"] == "joints", ctx.get("state_note")
    assert "/yam_left/joint_state joint_pos" not in signals and "/yam_right/joint_state joint_pos" not in signals
    assert signals["/yam_left/joint_state joint_vel"]["names_source"] == "yam_left.mcap channel metadata"
    assert "RECORDED MOTION, from the dataset's state" in prompt
    assert not any(i.get("kind") == f.STATE_UNIT_CONTRADICTED for i in ctx.get("reader_issues") or [])


def test_declared_radians_the_readings_contradict_never_make_an_arm_state(tmp_path):
    """A channel that declares radians for joints whose readings can only be degrees is not trusted: the arm is kept
    as recorded, flagged, and never read as the state."""
    root = tmp_path / "upload"
    _declared_yam_episode(root, scale=300.0)
    _, _, _, ctx = _read(tmp_path, root)
    assert ctx["state_kind"] == "none"
    assert any(i.get("kind") == f.STATE_UNIT_CONTRADICTED for i in ctx.get("reader_issues") or [])


def test_an_arm_channel_that_declares_nothing_stays_as_before(tmp_path):
    root = tmp_path / "upload"
    _declared_yam_episode(root, declare=False)
    _, signals, _, ctx = _read(tmp_path, root)
    assert ctx["state_kind"] == "none"
    assert signals["/yam_left/joint_state joint_pos"]["unnamed"] is True


def test_a_channel_declaration_gives_a_row_names_and_units_field_by_field():
    names, units = vn.declared_row(FOLLOWER_METADATA, [("joint_pos", 6), ("gripper_pos", 1)])
    assert names == [f"joint{j}" for j in range(1, 7)] + ["gripper"]
    assert units == ["rad"] * 6 + ["normalized"]
    assert vn.declared_row(LEADER_METADATA, [("joint_pos", 7)]) == (
        [f"joint{j}" for j in range(1, 7)] + ["gripper"], ["rad"] * 6 + ["normalized"])
    # a field the channel declares nothing for leaves the whole row undeclared
    assert vn.declared_row({"joint_pos_names": JOINTS}, [("joint_pos", 6), ("gripper_pos", 1)]) == (None, None)
    assert vn.declared_row({}, [("joint_pos", 6)]) == (None, None)


def test_hdf5_attributes_name_arrays_through_the_step_every_reader_passes(tmp_path):
    """An HDF5 file names an array's values in the dataset's own attributes (a list, a JSON list in a string, a comma
    separated string) or in a group's (qpos_names on /observations). write_signals, which every reader's arrays pass
    through, attaches them."""
    import h5py
    n = 10
    path = tmp_path / "rec.h5"
    with h5py.File(path, "w") as h:
        h["observations/qpos"] = np.zeros((n, 3))
        h["observations/qvel"] = np.zeros((n, 3))
        h["observations/effort"] = np.zeros((n, 2))
        h["observations"].attrs["qpos_names"] = ["a", "b", "c"]
        h["observations/qvel"].attrs["names"] = '["va", "vb", "vc"]'
        h["observations/effort"].attrs["joint_names"] = "ea, eb"
    with h5py.File(path, "r") as h:
        claims = vn.h5_claims(h, ["/observations/qpos", "/observations/qvel"])
        assert vn.dataset_names(h["observations/qvel"], 3) == ["va", "vb", "vc"]
        assert vn.dataset_names(h["observations/effort"], 2) == ["ea", "eb"]
        assert vn.dataset_names(h["observations/qpos"], 3) is None
    sig = f.Signals()
    sig.add("observations/qpos", np.random.rand(n, 3), source="HDF5 dataset /observations/qpos")
    sig.add("observations/qvel", np.random.rand(n, 3), names=["va", "vb", "vc"])
    sig.add("other/qpos", np.random.rand(n, 3))
    sig.claims += claims
    ep = tmp_path / "ep"
    ep.mkdir()
    ctx = {"n_state_frames": n, "fps": 30.0}
    f.write_signals(ep, ctx, sig)
    got = {s["name"]: s for s in ctx["signals"]}
    assert got["observations/qpos"]["names"] == ["a", "b", "c"]
    assert got["observations/qpos"]["names_source"] == "HDF5 attributes of /observations"
    assert got["observations/qvel"]["names"] == ["va", "vb", "vc"] and "names_source" not in got["observations/qvel"]
    assert got["other/qpos"]["unnamed"] is True and "names_searched" not in got["other/qpos"]


def test_a_lerobot_info_json_names_a_column_its_reader_left_unnamed():
    """meta/info.json's features name their values; as a claim it names the column of that key wherever a reader
    left it unnamed."""
    info = {"features": {"observation.state": {"dtype": "float32", "shape": [3], "names": {"motors": ["x", "y", "z"]}},
                         "observation.images.top": {"dtype": "video", "shape": [4, 4, 3],
                                                    "names": ["height", "width", "channels"]}}}
    metas = {"observation.state": {}, "observation.effort": {}}
    issues = vn.resolve(metas, {"observation.state": 3, "observation.effort": 3},
                        vn.claims_in(info, "meta/info.json"))
    assert not issues
    assert metas["observation.state"] == {"names": ["x", "y", "z"], "names_source": "meta/info.json",
                                          "units_undeclared": True}
    assert metas["observation.effort"] == {"units_undeclared": True}


def test_claims_that_disagree_name_nothing_and_say_so():
    """Two declarations of the same rank and match that name a field differently name it with neither, and the
    disagreement is a data issue; a more specific claim or a nearer one wins without one."""
    a = vn.claims_in({"joint_pos": ["a1", "a2"]}, "one.yaml")
    b = vn.claims_in({"joint_pos": ["b1", "b2"]}, "two.yaml")
    metas = {"/arm/state joint_pos": {}}
    issues = vn.resolve(metas, {"/arm/state joint_pos": 2}, a + b)
    assert "names" not in metas["/arm/state joint_pos"] and issues[0]["kind"] == "value_names_conflict"
    specific = vn.claims_in({"arm/state": {"joint_pos": ["s1", "s2"]}}, "three.yaml")
    metas = {"/arm/state joint_pos": {}}
    assert not vn.resolve(metas, {"/arm/state joint_pos": 2}, a + b + specific)
    assert metas["/arm/state joint_pos"]["names"] == ["s1", "s2"]
    near = [{**c, "rank": 1} for c in a]
    metas = {"/arm/state joint_pos": {}}
    assert not vn.resolve(metas, {"/arm/state joint_pos": 2}, near + b)
    assert metas["/arm/state joint_pos"]["names"] == ["a1", "a2"]


def test_only_a_list_of_distinct_names_names_values():
    """A units list ("rad" six times), numbers, one name and a plain string name nothing; a comma separated string
    names values only under a key that says names."""
    assert vn.parse_names(["rad"] * 6) is None
    assert vn.parse_names([0.1, 0.2]) is None
    assert vn.parse_names(["only"]) is None
    assert vn.parse_names("a, b") is None and vn.parse_names("a, b", text=True) == ["a", "b"]
    assert vn.parse_names(np.array([b"x", b"y"])) == ["x", "y"]
    assert vn.parse_names([{"name": "j1"}, {"name": "j2"}]) == ["j1", "j2"]
    assert vn.claims_in({"cameras": {"intrinsics": [[1.0, 0.0], [0.0, 1.0]]}}, "s") == []


def test_the_readout_and_the_board_say_whether_values_are_named():
    from board.sensors import signal_doc
    from label import signals as sg
    a = np.stack([np.linspace(0, 1, 20), np.linspace(1, 0, 20)], axis=1)
    assert sg.describe("arm q", a, unnamed=True).startswith("  arm q (2 unnamed values):")
    assert sg.describe("arm q", a, names=["a", "b"], names_source="rig.yaml").startswith(
        "  arm q (2 values (a, b), names from rig.yaml):")
    assert sg.describe("arm q", a).startswith("  arm q (2 values):")
    t = np.arange(20) / 30
    doc = signal_doc({"name": "arm q", "unnamed": True}, a, t, 1)
    assert doc["unnamed"] is True
    doc = signal_doc({"name": "arm q", "names": ["a", "b"], "names_source": "rig.yaml"}, a, t, 1)
    assert doc["names"] == ["a", "b"] and doc["names_source"] == "rig.yaml"


def _arm(phase):
    return lambda k: {"joint_pos": [0.3 * np.sin(0.05 * k + j + phase) for j in range(3)]}


def _two_files(tmp_path, meta_a, meta_b, start_b=T0 - 0.1, description=None):
    """Two MCAP files of one upload that each record /arm/state (b's at start_b), read the way every reader reads
    several files (mcap_signals, then write_signals), with description the text of a rig.yaml beside them."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    _node_mcap(tmp_path / "a.mcap", "/arm/state", _arm(0.0), channel_metadata=meta_a)
    _node_mcap(tmp_path / "b.mcap", "/arm/state", _arm(1.0), channel_metadata=meta_b, start=start_b)
    q = T0 + np.arange(18) / 30
    if start_b > T0:
        q = T0 + np.arange(60) / 30
    sig = f.mcap_signals([tmp_path / "a.mcap", tmp_path / "b.mcap"], q)
    ep = tmp_path / "ep"
    ep.mkdir()
    ctx = {"n_state_frames": len(q), "fps": 30.0}
    files = []
    if description is not None:
        (tmp_path / "rig.yaml").write_text(description)
        files = [tmp_path / "rig.yaml"]
    with vn.describing(files, tmp_path):
        f.write_signals(ep, ctx, sig)
    return {s["name"]: s for s in ctx["signals"]}, ctx


def test_a_file_that_declares_nothing_never_takes_another_files_names_or_units(tmp_path):
    """Two files of one upload record a channel of the same name, two recordings kept apart. Claims were matched by
    the field's name alone, so b.mcap's undeclared arm took a.mcap's names and unit. A declaration in a file names
    only what that file recorded."""
    got, _ = _two_files(tmp_path, {"joint_pos_names": '["a1", "a2", "a3"]', "joint_pos_unit": "rad"}, None)
    a, b = got["a /arm/state joint_pos"], got["b /arm/state joint_pos"]
    assert a["names"] == ["a1", "a2", "a3"] and a["names_source"] == "a.mcap channel metadata" and a["unit"] == "rad"
    assert b.get("unnamed") is True and "names" not in b and "unit" not in b and b.get("units_undeclared") is True


def test_two_files_that_name_one_channel_differently_each_keep_their_own_names(tmp_path):
    """a.mcap and b.mcap name their own /arm/state differently. The two declarations were one conflict that named
    neither arm; each file settles its own arm, and no conflict is raised."""
    got, ctx = _two_files(tmp_path, {"joint_pos_names": '["a1", "a2", "a3"]', "joint_pos_unit": "rad"},
                          {"joint_pos_names": '["b1", "b2", "b3"]', "joint_pos_unit": "deg"})
    a, b = got["a /arm/state joint_pos"], got["b /arm/state joint_pos"]
    assert (a["names"], a["unit"], a["names_source"]) == (["a1", "a2", "a3"], "rad", "a.mcap channel metadata")
    assert (b["names"], b["unit"], b["names_source"]) == (["b1", "b2", "b3"], "deg", "b.mcap channel metadata")
    assert not [i for i in ctx.get("reader_issues") or [] if i["kind"] in ("value_names_conflict",
                                                                            "value_units_conflict")]


def test_a_description_beside_the_files_and_the_pieces_of_one_recording_still_name_every_file(tmp_path):
    """A rig description beside the recording is tied to no file, so it names the arm of every file that declares
    nothing for it; and pieces of one recording split across files, joined into one array, take what either piece's
    file declares."""
    got, _ = _two_files(tmp_path, None, None, description="arm/state:\n  joint_pos_names: [r1, r2, r3]\n")
    for name in ("a /arm/state joint_pos", "b /arm/state joint_pos"):
        assert got[name]["names"] == ["r1", "r2", "r3"] and got[name]["names_source"] == "rig.yaml"
    got, _ = _two_files(tmp_path / "pieces", None, {"joint_pos_names": '["b1", "b2", "b3"]'}, start_b=T0 + 0.8)
    joined = got["/arm/state joint_pos"]
    assert joined["names"] == ["b1", "b2", "b3"] and joined["names_source"] == "b.mcap channel metadata"


def test_a_claim_on_a_whole_channel_names_that_files_channel_when_another_file_records_one_of_the_same_name(tmp_path):
    """A name list on a.mcap's /arm/state as a whole names that channel's one array of its width. With b.mcap also
    recording /arm/state, the topic held two such arrays and the claim named neither; it names a.mcap's."""
    got, _ = _two_files(tmp_path, {"names": '["c1", "c2", "c3"]'}, None)
    assert got["a /arm/state joint_pos"]["names"] == ["c1", "c2", "c3"]
    assert got["b /arm/state joint_pos"].get("unnamed") is True


def test_hdf5_attributes_name_only_the_arrays_of_their_own_file(tmp_path):
    """Two HDF5 files that each name /observations/qpos in their group's attributes: each file's arrays take that
    file's names, and an array whose file is not known is matched against every claim, as before."""
    import h5py
    claims = []
    for stem, names in (("a", ["a1", "a2"]), ("b", ["b1", "b2"])):
        with h5py.File(tmp_path / f"{stem}.h5", "w") as h:
            h["observations/qpos"] = np.zeros((4, 2))
            h["observations"].attrs["qpos_names"] = names
        with h5py.File(tmp_path / f"{stem}.h5", "r") as h:
            claims += vn.h5_claims(h, ["/observations/qpos"])
    assert {c["origin"] for c in claims} == {"a.h5", "b.h5"}
    metas = {"a observations/qpos": {}, "b observations/qpos": {}, "observations/qpos": {}}
    widths = dict.fromkeys(metas, 2)
    issues = vn.resolve(metas, widths, claims, {"a observations/qpos": "a.h5", "b observations/qpos": "b.h5"})
    assert metas["a observations/qpos"]["names"] == ["a1", "a2"]
    assert metas["b observations/qpos"]["names"] == ["b1", "b2"]
    assert "names" not in metas["observations/qpos"]
    assert [i["signal"] for i in issues if i["kind"] == "value_names_conflict"] == ["observations/qpos"]


def _h5_upload(tmp_path, files: dict) -> tuple[dict, Path]:
    """Prepare an upload of HDF5 files ({file name: fill(h5py.File)}), one of which holds the camera, as one episode;
    returns its context and folder."""
    import h5py
    root = tmp_path / "upload"
    root.mkdir()
    for name, fill in files.items():
        with h5py.File(root / name, "w") as h:
            fill(h)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 1, rep
    ep = tmp_path / "eps" / rep["episodes"][0]["episode_id"]
    return json.loads((ep / "context.json").read_text()), ep


H5_T = T0 + np.arange(40) / 10


def _h5_camera(h, arrays: dict, attrs: dict | None = None) -> None:
    """An HDF5 episode file: a camera on its clock, and arrays {path: values} on a clock of their group."""
    h.attrs["fps"] = 10
    h.create_dataset("timestamps", data=H5_T).attrs["units"] = "s"
    h["images/top"] = np.stack([np.full((36, 64, 3), 5 * k, np.uint8) for k in range(len(H5_T))])
    _h5_sensor(h, arrays, attrs)


def _h5_sensor(h, arrays: dict, attrs: dict | None = None) -> None:
    """Arrays {path: values}, each on a clock of its group (or the file's), and attributes {group: {key: value}}."""
    for path, values in arrays.items():
        group = path.rsplit("/", 1)[0] if "/" in path else ""
        clock = f"{group}/timestamps" if group else "timestamps"
        if clock not in h:
            h.create_dataset(clock, data=H5_T).attrs["units"] = "s"
        h[path] = values
    for group, kv in (attrs or {}).items():
        for k, v in kv.items():
            h[group].attrs[k] = v


def test_an_hdf5_files_attributes_never_name_the_arrays_of_another_file_of_the_upload(tmp_path):
    """An episode file whose sensors/pressure names nothing, beside a sensor file whose /sensors attributes name its own
    pressure: the episode's array had taken the sensor file's names, since the HDF5 reader recorded no file for its
    arrays. Each file's attributes now name only that file's arrays, and two files that name the array differently
    each keep their own names, with no conflict."""
    rows = np.arange(120, dtype=np.float32).reshape(40, 3)
    theirs = {"/sensors": {"pressure_names": ["a1", "a2", "a3"]}}
    for ours, want in ((None, None), ({"/sensors": {"pressure_names": ["b1", "b2", "b3"]}}, ["b1", "b2", "b3"])):
        case = tmp_path / ("ours" if ours else "none")
        case.mkdir()
        ctx, _ = _h5_upload(case, {"b.h5": lambda h: _h5_camera(h, {"sensors/pressure": rows}, ours),
                                   "a.h5": lambda h: _h5_sensor(h, {"sensors/pressure": rows + 1000}, theirs)})
        got = {s["name"]: s for s in ctx["signals"]}
        assert set(got) == {"sensors/pressure", "pressure"}, set(got)
        assert got["pressure"]["names"] == ["a1", "a2", "a3"], got["pressure"]
        assert got["sensors/pressure"].get("names") == want, got["sensors/pressure"]
        assert (got["sensors/pressure"].get("unnamed") is True) == (want is None), got["sensors/pressure"]
        assert not [i for i in ctx.get("reader_issues") or [] if i["kind"] == "value_names_conflict"], ctx


def test_two_hdf5_files_of_one_array_name_are_told_apart_by_their_files(tmp_path):
    """The episode file and a sensor file both record pressure, with other readings: both are kept, each under its
    file's name, and the note names both files. Each takes only its own file's names, matched by the name it was read
    under: the episode file's array had kept the bare name and taken the sensor file's names (or lost both to a
    conflict), been called "the first read", and the sensor file's array, renamed, had matched no names at all."""
    rows = np.arange(120, dtype=np.float32).reshape(40, 3)
    for ours in (None, ["b1", "b2", "b3"]):
        case = tmp_path / ("ours" if ours else "none")
        case.mkdir()
        ctx, _ = _h5_upload(case, {
            "b.h5": lambda h: _h5_camera(h, {"pressure": rows}, {"/": {"pressure_names": ours}} if ours else None),
            "a.h5": lambda h: _h5_sensor(h, {"sensors/pressure": rows + 1000},
                                         {"/sensors": {"pressure_names": ["a1", "a2", "a3"]}})})
        got = {s["name"]: s for s in ctx["signals"]}
        assert set(got) == {"b pressure", "a pressure"}, set(got)
        assert got["a pressure"].get("names") == ["a1", "a2", "a3"], got["a pressure"]
        assert got["b pressure"].get("names") == ours and (got["b pressure"].get("unnamed") is True) == (not ours), got
        issues = ctx["reader_issues"]
        notes = [i for i in issues if i["kind"] == "signal_name_collision"]
        assert len(notes) == 1 and "(b.h5 and a.h5)" in notes[0]["what"], notes
        assert notes[0]["signal"] == "a pressure" and notes[0]["signal_source"] == "a.h5", notes
        assert not [i for i in issues if i["kind"] == "value_names_conflict"], issues


def test_the_episode_files_own_state_stays_first_beside_another_files_array_of_its_name(tmp_path):
    """Two HDF5 files that record observations/qpos on one clock: the camera file's array is the state as before, now
    named with its file, and the other stays a signal. When the file names say the sides (left.h5, right.h5), the two
    are the two arms of one state, left first, as the same two files beside videos are read; the camera file's array
    had been read as the whole state, since its name carried no side."""
    joints = [f"joint{i}_rad" for i in range(1, 7)] + ["gripper"]
    th = np.arange(40) / 10
    mine = np.stack([0.3 * np.sin(th + j) for j in range(7)], axis=1)
    other = np.stack([0.3 * np.cos(th + j) for j in range(7)], axis=1)

    def arm(values, more=None):
        # another group beside observations keeps the group in the array's name (h5_streams leaves off only a group
        # that holds everything), so the two files' arrays have one name
        return {"observations/qpos": values, **(more or {})}, {"/observations/qpos": {"names": joints}}
    for ep_file, other_file, state, kept in (
            ("b.h5", "a.h5", "b observations/qpos", {"a observations/qpos"}),
            ("left.h5", "right.h5", "left observations/qpos and right observations/qpos", set())):
        case = tmp_path / ep_file
        case.mkdir()
        ctx, ep = _h5_upload(case, {ep_file: lambda h: _h5_camera(h, *arm(mine)),
                                    other_file: lambda h: _h5_sensor(h, *arm(other, {"extra/v": th[:, None]}))})
        assert ctx["state_kind"] == "joints" and ctx["source"]["state"] == state, ctx.get("state_note")
        z = np.load(ep / "state.npz")
        assert np.allclose(z["state"][:, :7], mine, atol=1e-6)
        if not kept:
            assert np.allclose(z["state"][:, 7:], other, atol=1e-6)
        assert {s["name"] for s in ctx.get("signals") or [] if "qpos" in s["name"]} == kept, ctx.get("signals")


def test_a_single_hdf5_file_reads_its_names_units_and_issues_as_before(tmp_path):
    """One HDF5 file, whose arrays all carry its name as their origin: its group's attributes still name and give the
    units of its arrays, a dataset's own names still win, and the issues are the same as before."""
    rows = np.arange(120, dtype=np.float64).reshape(40, 3)
    rows[5, 1] = np.inf
    ctx, _ = _h5_upload(tmp_path, {"rec.h5": lambda h: _h5_camera(
        h, {"sensors/pressure": rows, "sensors/force": np.ones((40, 2))},
        {"/sensors": {"pressure_names": ["p1", "p2", "p3"], "pressure_unit": "kPa", "force_names": "f1, f2"}})})
    got = {s["name"]: s for s in ctx["signals"]}
    assert got["sensors/pressure"]["names"] == ["p1", "p2", "p3"] and got["sensors/pressure"]["unit"] == "kPa", got
    assert got["sensors/force"]["names"] == ["f1", "f2"] and got["sensors/force"].get("units_undeclared") is True, got
    assert [(i["kind"], i.get("signal")) for i in ctx.get("reader_issues") or []] == [
        ("signal_not_finite", "sensors/pressure")], ctx.get("reader_issues")
    assert not any("signal_source" in i for i in ctx.get("reader_issues") or []), ctx.get("reader_issues")
