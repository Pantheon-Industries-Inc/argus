"""The reader for your own data (prepare/formats.py), on names and synthetic files: which camera fills which view,
names and paths, fixed-length packaging, packed LeRobot v3 placement, the frame writer, an MCAP's task text and
timed steps, and a recorder's folder of videos with its arm state in MCAP files. Data Review runs these cases too, against its browser pre-flight (its upload/test_formats.py)."""
from __future__ import annotations

import tempfile
from fractions import Fraction
from pathlib import Path

from prepare import formats as f

# (camera names, rig) -> expected {view: name}; view "exo" is the scene (or head) camera
CASES = [
    (["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_low",
      "observation.images.cam_right_wrist"], "teleop_arms",
     {"exo": "observation.images.cam_high", "left": "observation.images.cam_left_wrist", "right": "observation.images.cam_right_wrist",
      "extra1": "observation.images.cam_low"}),
    (["top", "wrist_left", "wrist_right"], "teleop_arms", {"exo": "top", "left": "wrist_left", "right": "wrist_right"}),
    (["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"], "teleop_arms",
     {"exo": "exterior_image_1_left", "left": "wrist_image_left", "extra1": "exterior_image_2_left"}),
    (["/zed/left/image/compressed", "/zed/right/image/compressed"], "teleop_arms", {"exo": "/zed/left/image/compressed"}),
    (["overhead_left", "leftCam", "rightCam"], "teleop_arms", {"exo": "overhead_left", "left": "leftCam", "right": "rightCam"}),
    (["bright_cam", "upright_view"], "teleop_arms", {"exo": "bright_cam", "extra1": "upright_view"}),
    # a fourth and fifth camera are sent too, the other eye of a stereo camera is not, and at most three extras
    (["front", "back", "side_a", "side_b", "side_c", "/zed/front/left", "/zed/front/right"], "teleop_arms",
     {"exo": "/zed/front/left", "extra1": "front", "extra2": "back", "extra3": "side_a", "extra4": "side_b"}),
    (["observation.images.left_camera_rgb_image", "observation.images.right_camera_rgb_image"], "handheld_gripper",
     {"left": "observation.images.left_camera_rgb_image", "right": "observation.images.right_camera_rgb_image"}),
    (["observation.images.camera_rgb_image"], "handheld_gripper", {"right": "observation.images.camera_rgb_image"}),
    (["/robot0/wrist_l", "/robot0/wrist_r", "/scene"], "teleop_arms", {"exo": "/scene", "left": "/robot0/wrist_l", "right": "/robot0/wrist_r"}),
    (["observation.images.top", "observation.images.top_depth", "observation.images.wrist_left"], "teleop_arms",
     {"exo": "observation.images.top", "left": "observation.images.wrist_left"}),
    # a view named for touch is its own view, never the scene or a gripper camera, and a pair of them are two sensors
    (["external_cam", "wrist_cam", "tactile_left", "tactile_right"], "teleop_arms",
     {"exo": "external_cam", "extra1": "wrist_cam", "extra2": "tactile_left", "extra3": "tactile_right"}),
]


EGO = [  # head rig: one camera, the one with a depth stream beside it
    (["/robot0/sensor/camera0/compressed", "/robot0/sensor/camera2/compressed", "/robot0/sensor/camera3/compressed"],
     ["/robot0/sensor/camera2/depth", "/robot0/sensor/camera2/compressed", "/robot0/sensor/camera0/compressed",
      "/robot0/sensor/camera3/compressed"], "/robot0/sensor/camera2/compressed"),
    (["/rgb", "/cam_a/color"], ["/rgb", "/cam_a/color", "/depth"], "/cam_a/color"),
]


def py_views(names, rig, all_names=None):
    return f.pick_cameras(names, rig, all_names or names)[0]


def test_each_camera_fills_the_view_its_name_says():
    for names, rig, want in CASES:
        assert py_views(names, rig) == want, (names, rig)
    for names, all_names, want in EGO:
        assert py_views(names, "ego_head", all_names) == {"exo": want}, names


def test_names_paths_packaging_and_packed_placement():
    bad = 0
    # names never collide, paths never leave the upload
    with tempfile.TemporaryDirectory() as t:
        out = Path(t)
        a = f.unique_dir(out, f.episode_name("run-1"))
        a.mkdir()
        b = f.unique_dir(out, f.episode_name("run_1"))
        if a == b:
            bad += 1
            print("episode names collide")
        for p in (out / "../x.mp4", Path("/etc/passwd")):
            try:
                f.inside(out, p)
                bad += 1
                print(f"inside() let {p} through")
            except ValueError:
                pass
        f.inside(out, out / "videos" / "a.mp4")
    for n, side in (("cam_left_wrist", "left"), ("wrist_r", "right"), ("bright_cam", None), ("upright", None)):
        if f.side_of(n) != side:
            bad += 1
            print(f"side_of({n}) = {f.side_of(n)}, want {side}")
    # fixed-length packaging: many files of one length; mixed lengths are not packaging
    if f.fixed_window([180.0, 179.2, 180.4, 95.0]) != 180.0 or f.fixed_window([60, 95, 180, 33]) is not None \
            or f.fixed_window([180, 180]) is not None:
        bad += 1
        print("fixed_window misjudged")
    # packed v3 without episode metadata: placed only when exactly one offset fits every file boundary
    counts = {"a": 300, "b": 250}
    real = f._frame_count
    f._frame_count = lambda p: counts[p]
    try:
        # the data holds episodes 10..15; file a holds 11..13 (100+120+80), file b 14..15 (130+120)
        lens = [(10, 90), (11, 100), (12, 120), (13, 80), (14, 130), (15, 120)]
        got = f._place_episodes(["a", "b"], lens)
        want = {11: ("a", 0), 12: ("a", 100), 13: ("a", 220), 14: ("b", 0), 15: ("b", 130)}
        if got != want:
            bad += 1
            print(f"packed placement: got {got}")
        counts["a"] = 200                       # no run of consecutive episodes fills file a exactly
        if f._place_episodes(["a", "b"], lens) != {} and 11 in f._place_episodes(["a", "b"], lens):
            bad += 1
            print("packed placement accepted an impossible layout")
        counts.update(a=100)                    # one file, several single episodes of that length: ambiguous
        if f._place_episodes(["a"], [(1, 100), (2, 100), (3, 100)]) != {}:
            bad += 1
            print("packed placement accepted an ambiguous layout")
    finally:
        f._frame_count = real
    assert bad == 0


def frame_writer_checks() -> int:
    """An H.264 camera copied from an MCAP keeps every frame it wrote: each packet's duration is the step to the next
    frame's time (the last one repeats the step before it), so the mp4's edit list reaches the end of the last frame.
    The raw stream's own duration (0, or a guess at 25 fps) ended a MicroAGI recording's edit list on a whole
    millisecond exactly where its last frame starts: it decoded one frame short of its times, and the board's clip
    then failed its frame count, and with it the whole job."""
    try:
        import av
        import numpy as np
    except ImportError:
        print("frame writer checks skipped: PyAV is not installed")
        return 0
    bad = 0
    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = 64, 48, "yuv420p"
    enc.time_base = Fraction(1, 30)
    enc.options = {"preset": "ultrafast", "bframes": "0", "g": "30"}
    n = 61
    packets = []
    for i in range(n):
        fr = av.VideoFrame.from_ndarray(np.full((48, 64, 3), (i * 4) % 256, np.uint8), format="rgb24")
        fr.pts = i
        packets += [bytes(p) for p in enc.encode(fr)]
    packets += [bytes(p) for p in enc.encode()]
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "exo.mp4"
        w = f.FrameWriter(out, "h264")
        for i, pkt in enumerate(packets):
            w.add(i / 30, pkt)
        wrote = w.close()
        with av.open(str(out)) as c:
            st = c.streams.video[0]
            durs = [int(p.duration * p.time_base * f.TIME_BASE_DEN) for p in c.demux(st) if p.size]
        with av.open(str(out)) as c:
            decoded = sum(1 for _ in c.decode(video=0))
        steps = [b - a for a, b in zip(w.pts, w.pts[1:])]
        if not (wrote == decoded == n):
            bad += 1
            print(f"frame writer: {n} frames in, {wrote} written, {decoded} decode")
        if durs != steps + steps[-1:]:
            bad += 1
            print(f"frame writer: packet durations are not the steps between frame times (last {durs[-2:]}, "
                  f"steps {steps[-2:]})")
    return bad


def test_the_frame_writer_keeps_every_frame():
    assert frame_writer_checks() == 0


def mcap_task_text_checks() -> int:
    """An MCAP's task text is the topic named for the task, never the first of its steps. MicroAGI's fragments carry
    /task (a title for the fragment), /task/subtask (each step, repeated while it lasts, first on the clock) and
    /task/health (a heartbeat): the first-arrived text, the first step, had been the instruction of every fragment."""
    bad = 0
    s = 1_000_000_000
    rec = lambda x: f'title: "{x}"\n'
    steps = [(0.0, "Pick up the pipe and the elbow"), (0.7, "Press both onto the welding machine"),
             (6.5, "Press both onto the welding machine"), (18.7, "Remove both from the machine"),
             (20.2, "Press the pipe into the elbow"), (25.7, "Turn the assembly to align it with the frame")]
    texts, counts = {}, {}                        # as convert_mcap_generic reads them, message by message
    for t, x in steps:
        f.add_text(texts, counts, "/task/subtask", s + int(t * 1e9), rec(x))
    f.add_text(texts, counts, "/task", s, rec("The person heats a PVC pipe and an elbow, joins them and aligns the assembly"))
    for i in range(f.TEXT_MSGS_MAX + 1):
        f.add_text(texts, counts, "/task/health", s + i * 33_333_333, f"timestamp {{ nanos: {i} }}\nvalid: true\n")
    sub = texts["/task/subtask"]
    if len(sub) != 6:
        bad += 1
        print(f"MCAP steps: {len(sub)} of 6 step messages kept (a repeated step was merged)")
    instr, notes = f.mcap_task_texts(texts, {**counts, "/task/health": 930}, s)
    if instr != "The person heats a PVC pipe and an elbow, joins them and aligns the assembly":
        bad += 1
        print(f"MCAP task text: instruction {instr!r}, want the /task title")
    if notes.get("/task/subtask") != ["0.0 s: Pick up the pipe and the elbow", "0.7 s: Press both onto the welding machine",
                                      "6.5 s: Press both onto the welding machine", "18.7 s: Remove both from the machine",
                                      "20.2 s: Press the pipe into the elbow", "25.7 s: Turn the assembly to align it with the frame"]:
        bad += 1
        print(f"MCAP task text: the steps did not reach the notes as a timeline: {notes.get('/task/subtask')}")
    if not str(notes.get("/task/health", "")).endswith("(the first of 930 messages)") or "task record" not in notes:
        bad += 1
        print(f"MCAP task text: heartbeat or task record wrong: {notes.get('/task/health')!r}, {sorted(notes)}")
    instr, notes = f.mcap_task_texts({"/task/subtask": sub}, {"/task/subtask": 6}, s)
    if instr is not None or len(notes.get("/task/subtask") or []) != 6:
        bad += 1
        print(f"MCAP task text: steps alone gave the instruction {instr!r}")
    instr, _ = f.mcap_task_texts({"/language_instruction": [(s, "put the cup in the sink")]}, {}, s)
    if instr != "put the cup in the sink":
        bad += 1
        print(f"MCAP task text: a plain instruction topic gave {instr!r}")
    # a head camera's steps as the dataset's timed subtasks, every message as sent: each until the next begins, the
    # last until the end; the heartbeat under the same task topic is not a step
    st, subs = f.mcap_step_subtasks(texts, s, 31.0)
    want = [{"t0": 0.0, "t1": 0.7, "label": "Pick up the pipe and the elbow"},
            {"t0": 0.7, "t1": 6.5, "label": "Press both onto the welding machine"},
            {"t0": 6.5, "t1": 18.7, "label": "Press both onto the welding machine"},
            {"t0": 18.7, "t1": 20.2, "label": "Remove both from the machine"},
            {"t0": 20.2, "t1": 25.7, "label": "Press the pipe into the elbow"},
            {"t0": 25.7, "t1": 31.0, "label": "Turn the assembly to align it with the frame"}]
    if st != "/task/subtask" or subs != want:
        bad += 1
        print(f"MCAP steps as subtasks: {st} {subs}")
    if f.mcap_step_subtasks({"/task/health": texts["/task/health"][:5]}, s, 31.0) != (None, []):
        bad += 1
        print("MCAP steps as subtasks: a heartbeat topic was read as steps")
    # a long fragment's steps (one per 4.5 s window, more than TEXT_MSGS_MAX) are all kept, not cut to a heartbeat
    long_t, long_c = {}, {}
    for i in range(f.TEXT_MSGS_MAX + 15):
        f.add_text(long_t, long_c, "/task/subtask", s + int(i * 4.5e9), rec("Place a yellow anchor into the container"))
    st, subs = f.mcap_step_subtasks(long_t, s, (f.TEXT_MSGS_MAX + 15) * 4.5)
    _, notes = f.mcap_task_texts(long_t, long_c, s)
    if len(subs) != f.TEXT_MSGS_MAX + 15 or subs[1]["t0"] != 4.5 or len(notes.get("/task/subtask") or []) != f.TEXT_MSGS_MAX + 15:
        bad += 1
        print(f"MCAP steps: {len(subs)} subtasks and {len(notes.get('/task/subtask') or [])} note lines of {f.TEXT_MSGS_MAX + 15} steps")
    # a task topic whose text changes is the instruction whole, as a timeline, with no text chosen over another:
    # MicroAGI's placeholder at the start of a fragment, the title sent at the same moment, a later title
    instr, notes = f.mcap_task_texts({"/task": [(s, "open the drawer"), (s + 4 * 10**9, "close the drawer")]}, {"/task": 2}, s)
    if instr != "0.0 s: open the drawer; 4.0 s: close the drawer" or "/task" in notes:
        bad += 1
        print(f"MCAP task text: a changing task topic gave {instr!r} and notes {sorted(notes)}")
    idle = {"/task": [(s, rec("The agent is idle with no activity.")), (s + 1000, rec("A person minces ginger and garlic")),
                      (s + 220 * 10**9, rec("A person slices potato pieces"))]}
    instr, notes = f.mcap_task_texts(idle, {"/task": 3}, s)
    if instr != ("0.0 s: The agent is idle with no activity.; 0.0 s: A person minces ginger and garlic; "
                 "220.0 s: A person slices potato pieces"):
        bad += 1
        print(f"MCAP task text: a placeholder then titles gave {instr!r}")
    return bad


def test_an_mcap_task_is_its_task_topic_and_its_steps_are_timed():
    assert mcap_task_text_checks() == 0


def test_a_gen_humanego_recording_goes_to_its_adapter():
    """A DAS-Ego headset MCAP (its forward camera and its annotation) is read by prepare/genhumanego.py, which gives its
    goal and timed steps; the same cameras without the annotation are an MCAP of cameras only."""
    from prepare import genhumanego as gh
    cams = [f"/robot0/sensor/camera{i}/compressed" for i in range(6)]
    hands = ["/robot0/handtracking/left", "/robot0/handtracking/right"]
    assert f.mcap_layout(cams + hands + [gh.ANNOTATION_TOPIC]) == "genhumanego"
    assert f.mcap_layout(cams + hands) == "generic"


def test_every_adapter_says_whether_it_reads_uploads():
    """An adapter added to prepare/ is found by the reader without editing it, and cannot be forgotten: it declares
    UPLOAD, a kind of upload it reads (with recognizes and convert_upload) or None."""
    import importlib
    import pkgutil
    import prepare
    for m in pkgutil.iter_modules(prepare.__path__):
        if m.name in f.NOT_ADAPTERS:
            continue
        mod = importlib.import_module(f"prepare.{m.name}")
        assert hasattr(mod, "UPLOAD"), f"prepare/{m.name}.py declares no UPLOAD"
        if mod.UPLOAD is not None:
            assert callable(getattr(mod, "recognizes", None)) and callable(getattr(mod, "convert_upload", None)), m.name
    names = [m.__name__.rsplit(".", 1)[-1] for m in f.upload_adapters("mcap")]
    assert names == ["abc130k", "genhumanego", "realomin"], names


def test_each_mcap_adapter_recognizes_its_own_layout_and_no_other():
    from prepare import abc130k, genhumanego, realomin
    layouts = {"abc130k": [abc130k.TOP_TOPICS[0], *abc130k.VIEW_TOPIC.values(), *abc130k.ARM],
               "realomin": [*realomin.CAMERA_TOPICS, *realomin.POSE_TOPICS],
               "genhumanego": [genhumanego.CAMERA_TOPIC, genhumanego.ANNOTATION_TOPIC]}
    for name, topics in layouts.items():
        assert f.mcap_layout(topics) == name, name
    assert f.mcap_layout(["/camera/color/0/image", "/task", "/task/subtask"]) == "generic"


def test_a_lerobot_upload_goes_to_the_adapter_of_its_dataset():
    """HABIT and Galaxea are recognized by the columns only they ship (their real feature lists, trimmed); any other
    LeRobot dataset, a bimanual YAM one included, is read generically."""
    habit = ["action", "episode_index", "frame_index", "human_role_subtask_index", "is_error_segment",
             "is_high_jerk_segment", "is_intervention_segment", "low_level_task_index", "observation.state",
             "observation.images.exo_view", "observation.images.left_wrist_view"]
    galaxea = ["action.left_arm", "action.left_gripper", "coarse_task_index", "episode_index", "frame_index",
               "observation.images.head_rgb", "observation.state.left_arm", "observation.state.left_gripper",
               "observation.state.right_arm", "observation.state.right_gripper", "quality_index", "task_index"]
    yam = ["action", "observation.state", "observation.images.top", "observation.images.left", "episode_index"]
    pick = lambda feats: next((m.__name__.rsplit(".", 1)[-1] for m in f.upload_adapters("lerobot")
                               if m.recognizes({"features": {k: {} for k in feats}})), "generic")
    assert [pick(habit), pick(galaxea), pick(yam)] == ["habit", "galaxea", "generic"]
    assert [m.__name__.rsplit(".", 1)[-1] for m in f.upload_adapters("lerobot")] == ["galaxea", "habit"]


def test_a_video_folder_in_a_datasets_own_layout_goes_to_its_adapter():
    """OpenAoE's clip folder (raw_video.mp4 with ego_annotation/ego_action_annotation.json beside it) is read by its
    adapter, with its action segments; a video folder without that annotation, or with another file, stays generic."""
    with tempfile.TemporaryDirectory() as t:
        _video_folder_adapter(Path(t))


def _video_folder_adapter(tmp_path: Path):
    from prepare import openaoe
    clip = tmp_path / "raw_x_seg_1"
    (clip / "ego_annotation").mkdir(parents=True)
    (clip / "raw_video.mp4").write_bytes(b"")
    (clip / "ego_annotation" / "ego_action_annotation.json").write_text(
        '[{"start_ts": "0.00", "end_ts": "3.00", "atomic_action": [{"verb": "align", "object": "fabric", "hand": "both"}]}]')
    item = {"dir": clip, "files": [clip / "raw_video.mp4"], "name": "raw_x_seg_1"}
    assert [m.__name__.rsplit(".", 1)[-1] for m in f.upload_adapters("video")] == ["openaoe"]
    assert openaoe.recognizes(item)
    assert openaoe.clip_extra(clip, clip.name)["annotation_subtasks"] == [
        {"t0": 0.0, "t1": 3.0, "label": "align fabric (both hands)", "ok": True}]
    assert not openaoe.recognizes({**item, "files": [clip / "other.mp4"]})
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "raw_video.mp4").write_bytes(b"")
    assert not openaoe.recognizes({"dir": plain, "files": [plain / "raw_video.mp4"], "name": "plain"})
    assert openaoe.recognizes({"dir": None, "files": [clip / "raw_video.mp4"], "name": "raw_video"})   # the folder uploaded itself


def _clip(path: Path, n: int) -> None:
    import av
    import numpy as np
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


def recorder_folder(root: Path, n: int = 12, t0: float = 1_790_000_000.0, third_arm: bool = False) -> Path:
    """One episode as a capture stack records it: each camera's colour and depth video with a file of its frames'
    capture times (two frames stamped alike), each arm's joints (follower, joint_pos and gripper_pos) and commands
    (leader, seven values) as JSON MCAP channels beside a health channel, and a metadata file naming the task.
    third_arm adds a third arm that carries the scene camera (yam_camera, six joints and no gripper, driven by
    camera_leader), with the scene camera named arm_cam. Data Review's upload/test_formats.py reads the same folder
    with the page's reader."""
    import json
    import numpy as np
    from mcap.writer import Writer
    d = root / "episode_000001_ab12"
    d.mkdir(parents=True)
    for k, cam in enumerate(("arm_cam" if third_arm else "exo_cam", "left_wrist_cam", "right_wrist_cam")):
        for kind in ("rgb", "depth"):
            _clip(d / f"{cam}-images-{kind}.mp4", n)
            ts = t0 + 0.01 * k + np.arange(n) / 30
            ts[3] = ts[2]
            np.save(d / f"{cam}-{kind}-timestamp.npy", ts)
    arms = [(f"yam_{'leader_' if leader else ''}{side}", leader, True)
            for side in ("left", "right") for leader in (False, True)]
    if third_arm:
        arms += [("yam_camera", False, False), ("camera_leader", True, False)]
    for name, leader, gripper in arms:
        with open(d / f"{name}.mcap", "wb") as fh:
            w = Writer(fh)
            w.start()
            sid = w.register_schema(name=name, encoding="jsonschema", data=b"{}")
            ch = w.register_channel(topic=f"/{name}/{'joint_pos' if leader else 'joint_state'}", message_encoding="json",
                                    schema_id=sid)
            hc = w.register_channel(topic=f"/{name}/health", message_encoding="json", schema_id=sid)
            for i in range(n * 4):
                t = int((t0 - 0.05 + i / 120) * 1e9)
                g = [0.5 + 0.4 * float(np.sin(i / 20))] if gripper else []
                msg = {"joint_pos": [0.1 * i / n] * 6 + g} if leader else \
                    {"joint_pos": [0.1 * i / n] * 6, "joint_vel": [0.0] * 6, **({"gripper_pos": g} if g else {})}
                w.add_message(ch, log_time=t, publish_time=t, data=json.dumps(msg).encode())
                w.add_message(hc, log_time=t, publish_time=t, data=b'{"ok": true, "reason": ""}')
            w.finish()
    (d / "session_meta.json").write_text(json.dumps({"prompt": "Pick up the cube", "nodes": [{"name": "exo_cam"}]}))
    return d


def test_a_recorders_folder_is_one_episode_with_its_arm_state():
    """MCAP files with no camera beside videos are the episode's recorded state, never episodes of their own: a
    capture folder of three cameras (each also with a depth video) and four arm MCAPs had been read as four
    episodes of arm data with the footage left out. It is one episode of the three colour videos, placed on the
    recorder's clock by their frame times, with the followers' joints as its state, the leaders' as its action, and
    the task from its metadata."""
    import json
    import numpy as np
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "upload"
        recorder_folder(root)
        det, items = f.plan(root)
        assert det["format"] == "video" and len(items) == 1
        assert sorted(Path(p).name for p in items[0]["files"]) == [
            "exo_cam-images-rgb.mp4", "left_wrist_cam-images-rgb.mp4", "right_wrist_cam-images-rgb.mp4"]
        assert len(items[0]["state"]) == 4
        # each camera's depth video goes with it (depth_videos), never a camera of its own and never dropped
        assert any("3 depth videos were read with the colour camera they belong to" in u for u in det["used"])
        assert len(items[0]["depth"]) == 3
        rep = f.convert(root, "teleop_arms", Path(t) / "eps", "test", 900)
        assert not rep["failed"] and len(rep["episodes"]) == 1
        ep = Path(t) / "eps" / rep["episodes"][0]["episode_id"]
        ctx = json.loads((ep / "context.json").read_text())
        assert ctx["state_kind"] == "joints" and ctx["instruction"] == "Pick up the cube" and not ctx.get("state_note")
        assert ctx["cameras"]["exo"]["name"] == "exo_cam"
        assert ctx["source"]["unused_cameras"] == []
        z = np.load(ep / "state.npz")
        assert z["state"].shape == (12, 14) and z["action"].shape == (12, 14)
        tm = np.load(ep / "times.npz")
        assert abs(float(tm["left"][0]) - 0.01) < 1e-6 and abs(float(tm["right"][0]) - 0.02) < 1e-6


def test_a_third_arm_that_carries_the_scene_camera_is_neither_working_arm():
    """A rig with a third arm that carries the scene camera (yam_camera, driven by camera_leader, beside yam_left and
    yam_right) had its whole arm state dropped, because the third arm's topic names no side. The left and right arms
    are the state and their leaders the action, the third arm is named in the episode's notes, and the scene camera,
    whose name says it is on an arm, is described as on that arm rather than as a fixed camera."""
    import json
    import numpy as np
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "upload"
        recorder_folder(root, third_arm=True)
        rep = f.convert(root, "teleop_arms", Path(t) / "eps", "test", 900)
        assert not rep["failed"] and len(rep["episodes"]) == 1
        ep = Path(t) / "eps" / rep["episodes"][0]["episode_id"]
        ctx = json.loads((ep / "context.json").read_text())
        assert ctx["state_kind"] == "joints"
        assert ctx["state_note"] == ("The recording has a third arm (/yam_camera/joint_state) beside the left and right "
                                     "arms; it is read as neither working arm.")
        assert ctx["cameras"]["exo"]["name"] == "arm_cam"
        assert ctx["cameras"]["exo"]["desc"].startswith("a camera on a third arm, which is not either working arm "
                                                        "(its joints are recorded as /yam_camera/joint_state)")
        z = np.load(ep / "state.npz")
        assert z["state"].shape == (12, 14) and z["action"].shape == (12, 14)


def test_joint_state_reads_only_the_layout_the_checks_read():
    import numpy as np
    q = np.linspace(0.0, 1.0, 11)
    arm = lambda dims: {"t": np.linspace(-0.1, 1.1, 30), "pos": np.ones((30, dims))}
    state, action, note = f.joint_state({"/left/joint_state": arm(7), "/right/joint_state": arm(7),
                                         "/leader_left/joint_pos": arm(7)}, q)
    assert state.shape == (11, 14) and action is None and note is None      # a command for one arm only is no action
    assert f.joint_state({"/arm/joint_state": arm(8)}, q)[2].startswith("Labelled from the cameras, because the recorded arms have 8")
    assert "does not cover" in f.joint_state({"/arm/joint_state": {"t": np.linspace(5, 6, 30), "pos": np.ones((30, 7))}}, q)[2]
    assert f.joint_state({"/left/joint_state": arm(7), "/arm/joint_state": arm(7)}, q)[2].endswith("which arm is which.")
    assert f.joint_state({}, q) == (None, None, None)
    # beside a left and a right arm, an arm that names no side is a third arm: the two sided arms are the state
    three = {"/left/joint_state": arm(7), "/right/joint_state": arm(7), "/camera/joint_state": arm(6),
             "/camera_leader/joint_pos": arm(6)}
    state, action, note = f.joint_state(three, q)
    assert state.shape == (11, 14) and action is None and note is None
    assert f.third_arms(three) == ["/camera/joint_state"]
    assert f.third_arms({"/left/joint_state": arm(7), "/arm/joint_state": arm(7)}) == []
    assert f.third_arms({"/left/joint_state": arm(7), "/right/joint_state": arm(7)}) == []


def test_an_accented_name_keeps_its_letters_in_the_episode_id():
    assert f.episode_name("Día 1 – cocina/toma 1 瓶子 🍶") == "episode_Dia_1_cocina_toma_1"
    assert f.episode_name("Überprüfung_Greifer-3") == "episode_Uberprufung_Greifer_3"
    assert f.episode_name("run-1") == "episode_run_1" and f.episode_name("瓶子") == "episode_0"



def test_state_that_covers_only_part_of_the_footage_is_not_held_flat_into_a_still_span():
    """Arms recorded moving over 0 to 6 s of 10 s footage: the state must not claim they stood still over the rest
    (np.interp holds the last sample flat), so the episode is labelled from the cameras with a note instead."""
    import numpy as np
    from label import state as ls
    q = np.arange(300) / 30.0
    t = np.arange(180) / 30.0
    moving = np.stack([np.sin(t + j) for j in range(7)], axis=1)
    streams = {"/left/joint_state": {"t": t, "pos": moving}, "/right/joint_state": {"t": t, "pos": moving}}
    state, _, note = f.joint_state(streams, q)
    assert state is None and "does not cover the footage" in note
    # a stream that spans the footage is read as before
    t = np.arange(301) / 30.0 - 0.01
    full = np.stack([np.sin(t + j) for j in range(7)], axis=1)
    state, _, note = f.joint_state({k: {"t": t, "pos": full} for k in streams}, q)
    assert state is not None and note is None
    assert not ls.still_spans(state, fps=30.0, kind="joints", grip_range=None)


def _frame_times_read_relative_millisecond_stamps_as_milliseconds(tmp_path):
    """A recorder that stamps each frame in ms from the start of the recording (0, 33.3, 66.7, ...), not since 1970:
    the unit comes from the frame step, so a 3 s clip stays 3 s."""
    import numpy as np
    (tmp_path / "exo_cam-images-rgb.mp4").write_bytes(b"")
    np.save(tmp_path / "exo_cam-rgb-timestamp.npy", np.arange(90) * (1000 / 30))
    t = f.frame_times(tmp_path / "exo_cam-images-rgb.mp4", 90)
    assert abs(float(np.median(np.diff(t))) - 1 / 30) < 1e-6
    np.save(tmp_path / "exo_cam-rgb-timestamp.npy", 1_790_000_000_000_000_000 + np.arange(90) * (1e9 / 30))
    t = f.frame_times(tmp_path / "exo_cam-images-rgb.mp4", 90)          # stamps since 1970 in ns, as before
    assert abs(float(np.median(np.diff(t))) - 1 / 30) < 1e-6 and t[0] > 1.7e9


def _an_mcap_keeps_every_other_number_it_records_as_a_signal(tmp_path):
    """The recorder's arm channels also carry joint_vel, and a gripper IMU runs beside them: both reach the episode as
    signals under their own names, while the joints and gripper already read as the state, a 0.5 Hz status report and
    a channel that stops before the footage ends do not."""
    import json
    import numpy as np
    from mcap.writer import Writer
    root = tmp_path / "upload"
    d = recorder_folder(root, n=60)          # 2 s of footage
    t0 = 1_790_000_000.0
    with open(d / "imu.mcap", "wb") as fh:
        w = Writer(fh)
        w.start()
        sid = w.register_schema(name="imu", encoding="jsonschema", data=b"{}")
        imu = w.register_channel(topic="/gripper/imu", message_encoding="json", schema_id=sid)
        slow = w.register_channel(topic="/system/cpu", message_encoding="json", schema_id=sid)
        half = w.register_channel(topic="/gripper/force", message_encoding="json", schema_id=sid)
        for i in range(330):
            t = int((t0 - 0.02 + i / 150) * 1e9)
            w.add_message(imu, log_time=t, publish_time=t, data=json.dumps(
                {"header": {"stamp": t}, "angular_velocity": {"x": 0.1 * i, "y": 0.0, "z": -0.1 * i}}).encode())
            if i < 150:                               # stops 1 s before the footage ends
                w.add_message(half, log_time=t, publish_time=t, data=json.dumps({"wrench": [1.0 * i, 0, 0]}).encode())
        w.add_message(slow, log_time=int(t0 * 1e9), publish_time=int(t0 * 1e9), data=b'{"cpu_percent": 3.0}')
        w.finish()
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 1
    ep = tmp_path / "eps" / rep["episodes"][0]["episode_id"]
    ctx = json.loads((ep / "context.json").read_text())
    names = [s["name"] for s in ctx["signals"]]
    assert "/gripper/imu angular_velocity" in names and "/yam_left/joint_state joint_vel" in names, names
    assert not any("joint_pos" in n or "gripper_pos" in n for n in names if "leader" not in n), names
    assert not any(n.startswith(("/system/cpu", "/gripper/force")) for n in names), names
    z = np.load(ep / "signals.npz")
    imu = z[next(s["key"] for s in ctx["signals"] if s["name"] == "/gripper/imu angular_velocity")]
    assert imu.shape == (ctx["n_state_frames"], 3) and imu[-1, 0] > imu[0, 0]


def test_frame_times_read_relative_millisecond_stamps_as_milliseconds():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _frame_times_read_relative_millisecond_stamps_as_milliseconds(Path(t))


def test_an_mcap_keeps_every_other_number_it_records_as_a_signal():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _an_mcap_keeps_every_other_number_it_records_as_a_signal(Path(t))


EPOCH = 1_788_210_000.0          # a moment in 2026, in seconds from the epoch


def test_a_fast_epoch_clock_is_read_in_its_own_unit():
    """A 1 kHz clock in nanoseconds steps by 1e6, which the step alone reads as microseconds; its size (1.79e18, which
    only nanoseconds put in this century) settles the unit."""
    import numpy as np
    for unit, scale in (("ns", 1e9), ("us", 1e6), ("ms", 1e3)):
        s = f._seconds((EPOCH + np.arange(50) / 1000.0) * scale)
        assert abs(s[0] - EPOCH) < 1 and abs((s[1] - s[0]) - 1e-3) < 1e-6, unit


def test_a_clock_from_boot_keeps_the_steps_reading():
    import numpy as np
    assert abs(f._seconds(np.arange(50) * 33_333_333.0)[1] - 1 / 30) < 1e-6       # 30 fps in ns from boot
    assert abs(f._seconds(np.arange(50) / 30.0)[1] - 1 / 30) < 1e-9               # already seconds


def test_a_clock_from_boot_in_nanoseconds_is_not_taken_for_milliseconds_from_the_epoch():
    """OpenTouch's HDF5 clock holds 2.6e12 ns after 43 minutes up, which is also a date in 2002 read as milliseconds;
    the step settles it (33 ms of nanoseconds would be 9 hours of milliseconds)."""
    import numpy as np
    s = f._seconds(2.6e12 + np.arange(50) * 33_333_333.0)
    assert abs(s[0] - 2600) < 1 and abs((s[1] - s[0]) - 1 / 30) < 1e-6


def test_a_slow_clock_from_days_of_uptime_keeps_the_steps_reading():
    """2e15 ns (23 days up) at 200 Hz is 2e9 read as microseconds, but that clock would step 5 s; a stream read as an
    epoch clock steps between 10 us and 1 s. The same holds for 2e12 us."""
    import numpy as np
    for start, step, unit in ((2e15, 5e6, "ns"), (2e12, 5e3, "us")):
        s = f._seconds(start + np.arange(50) * step)
        assert abs(s[0] - 2e6) < 1 and abs((s[1] - s[0]) - 5e-3) < 1e-6, unit


def test_a_nan_first_sample_does_not_hide_an_epoch_clock():
    import numpy as np
    t = (EPOCH + np.arange(50) / 1000.0) * 1e9
    t[0] = np.nan
    s = f._seconds(t)
    assert abs(s[1] - (EPOCH + 0.001)) < 1e-3 and abs((s[2] - s[1]) - 1e-3) < 1e-6


def test_a_leading_zero_sentinel_does_not_hide_an_epoch_clock():
    """A recorder that writes 0 before its first stamp: the clock's size is its median, not its first value."""
    import numpy as np
    t = (EPOCH + np.arange(50) / 1000.0) * 1e6
    t[0] = 0
    assert f._seconds_scale(t) == 1e-6


def test_a_single_stamp_is_read_by_its_size():
    import numpy as np
    assert abs(f._seconds(np.array([EPOCH * 1e9]))[0] - EPOCH) < 1


def test_an_empty_clock_is_empty():
    import numpy as np
    assert len(f._seconds(np.array([]))) == 0


def test_clocks_from_zero_keep_their_own_units():
    """Two clocks that both start at 0 (a camera in seconds from the recording's start beside an IMU in nanoseconds
    from it) do not say they share a unit, so each is read by its own step."""
    import numpy as np
    out = f._clocks_in_seconds({"cam": np.arange(40) / 30.0, "imu": np.arange(400) * 5e6})
    assert abs(np.median(np.diff(out["cam"])) - 1 / 30) < 1e-9
    assert abs(np.median(np.diff(out["imu"])) - 5e-3) < 1e-9


def test_clocks_from_the_same_moment_far_from_zero_share_the_slowest_ones_unit():
    """A 1 kHz pad and a 30 fps camera, both in nanoseconds from 43 minutes of uptime: the pad's step of 1e6 alone
    reads as microseconds, so it takes the camera's unit."""
    import numpy as np
    cam = 2_600e9 + np.arange(40) * 33_333_333.0
    out = f._clocks_in_seconds({"cam": cam, "pad": cam[0] - 1e7 + np.arange(1400) * 1e6})
    assert abs(np.median(np.diff(out["pad"])) - 1e-3) < 1e-9


def test_a_steadily_rising_reading_stays_a_signal_unless_its_name_says_time():
    """A base driving forward at a steady speed rises by a steady step, as a clock does; only a time's name (timestamp,
    t_ns) makes a column a clock."""
    import numpy as np, pandas as pd
    n = 60
    df = pd.DataFrame({"base.odom_x": list(np.arange(n) * 0.01), "sensor_timestamp": list(1e9 + np.arange(n) * 3.3e7),
                       "gripper": list(np.r_[np.zeros(30), np.ones(30)])})
    out = f.recorded_signals(df, set(), n)
    assert "base.odom_x" in out and "gripper" in out
    assert "sensor_timestamp" not in out and "sensor_timestamp" in out.clocks


def _a_table_keeps_a_steadily_rising_column_unless_its_name_says_time(tmp_path):
    """The same rule for a CSV table beside the videos: odom_x rises by a steady step and stays a value, while the
    time column is the one that places the table and is not shown."""
    import numpy as np, pandas as pd
    n = 60
    pd.DataFrame({"time_s": np.arange(n) / 30.0, "odom_x": np.arange(n) * 0.01,
                  "grip": np.r_[np.zeros(30), np.ones(30)]}).to_csv(tmp_path / "traj.csv", index=False)
    pts = np.arange(n, dtype=np.int64)
    out = f.table_signals([tmp_path / "traj.csv"], None, {"pts": pts, "time_base": 1 / 30.0}, {})
    assert out.meta["traj"]["names"] == ["odom_x", "grip"]


def test_a_table_keeps_a_steadily_rising_column_unless_its_name_says_time():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_table_keeps_a_steadily_rising_column_unless_its_name_says_time(Path(t))


CLOCK_NAMES = ["timestamp", "sensor_timestamp", "sensorTimestamp", "observation.timestamp", "/hdas/imu.header.stamp",
               "t_ns", "ros_time", "capture_time_ns", "header.stamp.sec", "stamp_ns", "stamp_nsec", "header.t_usec",
               "time_msec", "ts_nanos", "timestampUtc", "epoch_ns", "time_nsecs", "stamp.nanosec", "TimeStamp"]
READING_NAMES = ["left_arm", "t_joint_3", "base.odom_x", "tact", "hat", "header.seq", "timeline_label", "stamped_force"]


def test_a_name_whose_last_word_is_a_time_word_or_a_unit_after_one_says_time():
    """A name's words are read as tokens() splits them (camelCase and separators), so sensorTimestamp, stamp.nanosec
    and timestampUtc say time as plainly as timestamp_ns does."""
    for name in CLOCK_NAMES:
        assert f.is_time_name(name), name


def test_a_name_that_only_contains_a_time_word_is_a_reading():
    for name in READING_NAMES:
        assert not f.is_time_name(name), name


def test_a_named_clock_needs_both_a_time_name_and_a_clocks_rise():
    import numpy as np
    rising = 1.79e18 + np.arange(60) * 3.3e7
    assert f.is_named_clock("sensorTimestamp", rising)
    assert not f.is_named_clock("base.odom_x", rising)
    assert not f.is_named_clock("sensorTimestamp", rising[::-1])


def test_a_camel_case_clock_column_is_a_clock_not_a_signal():
    import numpy as np, pandas as pd
    n = 60
    df = pd.DataFrame({"sensorTimestamp": list(1.79e18 + np.arange(n) * 3.3e7),
                       "gripper": list(np.r_[np.zeros(30), np.ones(30)])})
    out = f.recorded_signals(df, set(), n)
    assert "sensorTimestamp" in out.clocks and "sensorTimestamp" not in out


def _a_tables_camel_case_clock_places_it_and_is_not_a_value(tmp_path):
    import numpy as np, pandas as pd
    n = 60
    pd.DataFrame({"sensorTimestamp": 1.79e18 + np.arange(n) * 3.3e7,
                  "grip": np.r_[np.zeros(30), np.ones(30)]}).to_csv(tmp_path / "traj.csv", index=False)
    pts = np.arange(n, dtype=np.int64)
    out = f.table_signals([tmp_path / "traj.csv"], None, {"pts": pts, "time_base": 1 / 30.0}, {})
    assert out.meta["traj"]["names"] == ["grip"]


def test_a_tables_camel_case_clock_places_it_and_is_not_a_value():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_tables_camel_case_clock_places_it_and_is_not_a_value(Path(t))


COUNTED_OUT = "counts rows one by one, so it is bookkeeping"


def test_a_column_that_counts_rows_one_by_one_is_a_counter():
    import numpy as np
    assert f.is_counter(np.arange(60))
    assert f.is_counter(1000 + np.r_[np.arange(30), np.arange(29, 59)])     # a row stamped twice is still counting
    assert not f.is_counter(np.arange(60) * 2)                               # steps of 2 are a reading's
    assert not f.is_counter(np.arange(60) + 0.5)                             # not whole numbers


def test_a_sequence_number_is_left_out_of_the_signals_as_bookkeeping_with_the_reason():
    import numpy as np, pandas as pd
    n = 60
    df = pd.DataFrame({"seq": list(np.arange(n)), "frame_id": list(500 + np.arange(n)),
                       "encoder_ticks": list(np.cumsum(np.r_[np.arange(30) % 3, np.zeros(30)]) * 7),
                       "gripper": list(np.r_[np.zeros(30), np.ones(30)])})
    out = f.recorded_signals(df, set(), n)
    assert "seq" not in out and "seq" not in out.clocks and "frame_id" not in out
    assert ("seq", COUNTED_OUT) in out.left_out and ("frame_id", COUNTED_OUT) in out.left_out
    assert "encoder_ticks" in out and "gripper" in out


def _a_tables_sequence_number_is_left_out_as_bookkeeping(tmp_path):
    import numpy as np, pandas as pd
    n = 60
    pd.DataFrame({"time_s": np.arange(n) / 30.0, "seq": np.arange(n),
                  "grip": np.r_[np.zeros(30), np.ones(30)]}).to_csv(tmp_path / "traj.csv", index=False)
    pts = np.arange(n, dtype=np.int64)
    out = f.table_signals([tmp_path / "traj.csv"], None, {"pts": pts, "time_base": 1 / 30.0}, {})
    assert out.meta["traj"]["names"] == ["grip"]
    assert ("seq in traj.csv", COUNTED_OUT) in out.left_out


def test_a_tables_sequence_number_is_left_out_as_bookkeeping():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_tables_sequence_number_is_left_out_as_bookkeeping(Path(t))


def test_a_reading_missing_at_some_frames_is_kept_with_nan_there():
    """A column with a few empty or infinite cells is still a reading; the cells become NaN instead of the whole
    column being dropped."""
    import numpy as np, pandas as pd
    n = 60
    force = np.linspace(0, 1, n)
    force[[5, 6]] = np.nan
    force[9] = np.inf
    ft = [np.array([1.0, 2.0, 3.0]) for _ in range(n)]
    ft[20] = np.array([1.0, np.nan, 3.0])
    out = f.recorded_signals(pd.DataFrame({"force": list(force), "ft": ft}), set(), n)
    assert "force" in out and np.isnan(out["force"][[5, 6, 9], 0]).all() and np.isfinite(out["force"][10, 0])
    assert "ft" in out and np.isnan(out["ft"][20, 1])


def test_a_reading_missing_at_most_frames_is_left_out_with_the_reason():
    import numpy as np, pandas as pd
    n = 60
    force = np.linspace(0, 1, n)
    force[:40] = np.nan
    out = f.recorded_signals(pd.DataFrame({"force": list(force)}), set(), n)
    assert "force" not in out and ("force", "no reading at most frames") in out.left_out


def test_a_clock_with_a_missing_stamp_is_still_a_clock():
    import numpy as np, pandas as pd
    n = 60
    t = 1.79e18 + np.arange(n) * 3.3e7
    t[[5, 40]] = np.nan
    out = f.recorded_signals(pd.DataFrame({"recv_time": list(t)}), set(), n)
    assert "recv_time" in out.clocks and "recv_time" not in out


def test_a_depth_stream_goes_with_the_camera_whose_name_has_all_its_camera_words():
    """HDF5 dataset paths and LeRobot feature keys are paired by one rule: the depth's words, less the words for what
    a file holds (depth, images), are all in the camera's name."""
    hdf5 = {"exo": "observations/images/cam_high", "left": "observations/images/cam_left_wrist"}
    depth = "observations/depth/cam_left_wrist"
    assert f.depth_camera(depth, hdf5, "exo") == ("left", depth)
    lerobot = {"exo": "observation.images.cam_high", "left": "observation.images.cam_left_wrist"}
    assert f.depth_camera("observation.depth.cam_high", lerobot, "exo") == ("exo", "observation.depth.cam_high")


def test_a_depth_stream_with_no_camera_of_its_own_goes_with_the_scene_camera_and_says_why():
    view, source = f.depth_camera("observations/depth/zed", {"exo": "top", "left": "wrist_left"}, "exo")
    assert view == "exo" and source.startswith("observations/depth/zed (") and "scene camera" in source


def _mcap_layout_context_with(convert_upload):
    """mcap_layout_context through a made-up layout whose reader is convert_upload: import_module returns a module
    already in sys.modules, so only that layout is faked."""
    import sys, types
    sys.modules["prepare.fakelayout"] = types.SimpleNamespace(convert_upload=convert_upload)
    try:
        return f.mcap_layout_context({"name": "a.mcap"}, "fakelayout", Path("episode_a"), "ds")
    finally:
        del sys.modules["prepare.fakelayout"]


def test_an_mcap_layout_readers_source_notes_are_kept():
    """The format and file are added to the layout reader's source instead of replacing it."""
    ctx = _mcap_layout_context_with(lambda item, ep: {"source": {"unused_signals": ["/imu (not read yet)"],
                                                                 "invalid_ranges": [[0, 1]]}})
    assert ctx["source"]["unused_signals"] == ["/imu (not read yet)"]
    assert ctx["source"]["invalid_ranges"] == [[0, 1]]
    assert ctx["source"]["format"] == "mcap (fakelayout layout)" and ctx["source"]["file"] == "a.mcap"
    assert ctx["dataset"] == "ds"


def test_an_mcap_layout_reader_that_returns_no_context_is_an_error():
    try:
        _mcap_layout_context_with(lambda item, ep: None)
    except ValueError as e:
        assert str(e) == "prepare.fakelayout returned no context"
    else:
        raise AssertionError("no error for a reader that returned no context")
