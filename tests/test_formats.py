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


def _videos_with_an_hdf5_arm_state(root: Path, n: int = 12, t0: float = 1_790_000_000.0) -> Path:
    """One episode of two cameras (each with a file of its frames' capture times) and a robot.h5 beside them holding
    two arms of six joints and a gripper as qpos, with the commands as action, at 100 Hz on the same recorder clock."""
    import h5py
    import numpy as np
    d = root / "episode_000001"
    d.mkdir(parents=True)
    for k, cam in enumerate(("exo_cam", "left_wrist_cam")):
        _clip(d / f"{cam}-images-rgb.mp4", n)
        np.save(d / f"{cam}-rgb-timestamp.npy", t0 + 0.01 * k + np.arange(n) / 30)
    t = t0 - 0.1 + np.arange(int((n / 30 + 0.2) * 100)) / 100
    q = np.stack([0.3 * np.sin(t - t0 + j) for j in range(14)], axis=1)
    q[:, 6] = q[:, 13] = (t - t0 > 0.2).astype(float)
    with h5py.File(d / "robot.h5", "w") as h:
        h["timestamps"] = (t * 1e9).astype(np.int64)
        h["qpos"] = q
        h["action"] = q + 0.01
    return d


def test_an_hdf5_arm_state_beside_videos_is_the_episodes_state():
    """A robot.h5 of two arms' qpos beside the videos had every number read as a signal, since the video path never
    asked whether one of its arrays is the state, so the episode was labelled as if no arm state was recorded. Its
    qpos is the state and its action the action, by the same rule as an HDF5 episode's (h5_state), also when a
    second HDF5 file beside it puts each file's name before its arrays' names."""
    import json
    import h5py
    import numpy as np
    for glove in (False, True):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "upload"
            d = _videos_with_an_hdf5_arm_state(root)
            if glove:
                with h5py.File(d / "glove.h5", "w") as h:
                    tg = 1_790_000_000.0 - 0.1 + np.arange(60) / 100
                    h["timestamps"] = (tg * 1e9).astype(np.int64)
                    h["pressure"] = np.abs(np.sin(tg))[:, None].repeat(4, axis=1)
            det, items = f.plan(root)
            assert det["format"] == "video" and len(items) == 1 and len(items[0]["state"]) == 1 + glove
            rep = f.convert(root, "teleop_arms", Path(t) / "eps", "test", 900)
            assert not rep["failed"] and len(rep["episodes"]) == 1
            ep = Path(t) / "eps" / rep["episodes"][0]["episode_id"]
            ctx = json.loads((ep / "context.json").read_text())
            assert ctx["state_kind"] == "joints" and not ctx.get("state_note"), ctx.get("state_note")
            prefix = "robot " if glove else ""
            assert ctx["source"]["state"] == f"{prefix}qpos"
            names = {s["name"] for s in ctx.get("signals") or []}
            assert f"{prefix}qpos" not in names and f"{prefix}action" not in names, names
            assert ("glove pressure" in names) == glove, names
            z = np.load(ep / "state.npz")
            assert z["state"].shape == (12, 14) and z["action"].shape == (12, 14)
            assert np.allclose(z["action"], z["state"] + 0.01, atol=1e-4)


def test_names_that_name_a_position_a_velocity_or_an_effort_never_read_as_joints():
    """state_layout read names it could not parse by the width rule, so a pose in metres with units after its axes
    (x_m, roll_rad), a camelCase pose (eefPosX), a cartesian position numbered 0 to 5, UMI's eef position on an arm
    rig, and joint velocities or efforts all read as six joints and a gripper in radians. Each name is read as words
    (separators and camelCase, a trailing unit dropped): an axis as the last word makes a pose, a velocity or an
    effort is not a state the checks read, and a pose without axes the rule reads is not joints. HABIT's
    position_0 to position_13, a position word alone with its index, keeps the width rule."""
    seven = lambda fmt, last: [fmt.format(i) for i in range(6)] + [last]
    umi = [f"robot0_eef_pos_{i}" for i in range(3)] + [f"robot0_eef_rot_axis_angle_{i}" for i in range(3)] + [
        "robot0_gripper_width"]
    cases = [
        ("teleop_arms", ["x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad", "gripper_width"], "ee_pose"),
        ("teleop_arms", ["ee_x_m", "ee_y_m", "ee_z_m", "ee_roll_deg", "ee_pitch_deg", "ee_yaw_deg", "gripper_mm"],
         "ee_pose"),
        ("teleop_arms", ["eefPosX", "eefPosY", "eefPosZ", "eefRoll", "eefPitch", "eefYaw", "gripperWidth"], "ee_pose"),
        ("teleop_arms", seven("cartesian_position_{}", "gripper_position"), "none"),
        ("teleop_arms", umi, "none"),
        ("handheld_gripper", umi, "ee_pose"),            # a pose on a rig whose state is a pose, as before
        ("teleop_arms", ["pos_0", "pos_1", "pos_2", "rot_6d_0", "rot_6d_1", "rot_6d_2", "gripper"], "none"),
        ("teleop_arms", [f"joint{i}_vel" for i in range(1, 7)] + ["gripper_vel"], "none"),
        ("teleop_arms", [f"joint{i}_effort" for i in range(1, 7)] + ["gripper_effort"], "none"),
        ("handheld_gripper", [f"joint{i}_velocity" for i in range(1, 7)] + ["gripper"], "none"),
        ("teleop_arms", ["force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z", "gripper"], "none"),
        ("teleop_arms", [f"jointTorque{i}" for i in range(1, 7)] + ["gripper"], "none"),
        # a frame word counts only before a number: joint names with rotation, rot, tool, flange or effector in them
        # stay joints, and a joint word does not save a numbered pose (wrist_rot_0)
        ("teleop_arms", ["base_rotation", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "wrist_rotation",
                         "gripper"], "joints"),
        ("teleop_arms", ["rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll", "wrist_yaw", "jaw"], "joints"),
        ("teleop_arms", ["torso_rot", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3", "gripper"], "joints"),
        ("teleop_arms", ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "tool_roll", "gripper"],
         "joints"),
        ("teleop_arms", ["base", "shoulder", "elbow", "wrist1", "wrist2", "flange", "gripper"], "joints"),
        ("teleop_arms", [f"arm_l_joint{i}" for i in range(1, 7)] + ["end_effector"], "joints"),
        ("teleop_arms", [f"idx2{i}_arm_l_joint{i}" for i in range(1, 7)] + ["left_effector_position"], "joints"),
        ("teleop_arms", ["wrist_pos_0", "wrist_pos_1", "wrist_pos_2", "wrist_rot_0", "wrist_rot_1", "wrist_rot_2",
                         "gripper"], "none"),
        ("teleop_arms", [f"actual_TCP_pose_{i}" for i in range(6)] + ["gripper_position"], "none"),
        # a plural position word cancels a quantity word as the singular does (current, the present position)
        ("teleop_arms", [f"current_joint_positions_{i}" for i in range(6)] + ["gripper"], "joints"),
        # names that say neither keep the width rule
        ("teleop_arms", [f"position_{i}" for i in range(14)], "joints"),
        ("teleop_arms", [f"motor_{i}" for i in range(7)], "joints"),
        ("teleop_arms", [f"j{i}_deg" for i in range(1, 7)] + ["gripper_pct"], "joints"),
        ("teleop_arms", [f"left_joint_{i}.pos" for i in range(6)] + ["left_gripper.pos"], "joints"),
    ]
    for rig, names, want in cases:
        kind, note = f.state_layout(len(names), rig, names)
        assert kind == want, (rig, names, kind, note)
        assert (note is None) == (want != "none"), (names, note)
    note = f.state_layout(7, "teleop_arms", [f"joint{i}_vel" for i in range(1, 7)] + ["gripper_vel"])[1]
    assert "joint1_vel" in note and "not a position" in note, note
    note = f.state_layout(7, "teleop_arms", seven("cartesian_position_{}", "gripper_position"))[1]
    assert "cartesian_position_0" in note and "without the axes" in note, note
    assert f.state_words("eefPosX") == ["eef", "pos", "x"] and f.state_words("roll_rad") == ["roll"]
    assert f.state_words("left_wrist.pos-x") == ["left", "wrist", "pos", "x"] and f.state_words("m") == ["m"]


def _two_arms_in_their_own_files_or_messages(tmp_path):
    """Two arms recorded apart, where only the value names or the file names say which arm is which: one
    /joint_states with the left and the right arm in messages of their own, and a left.h5 and a right.h5 beside the
    videos, each with one arm's qpos and action. Each side is that side's arm, through the same path two topics take,
    so the state is both arms, left first; one arm had been read as the whole state."""
    import json
    import h5py
    import numpy as np
    t0 = 1_790_000_000.0
    names = [f"joint{i}" for i in range(1, 7)] + ["gripper"]
    tt = np.arange(-0.1, 3.1, 0.01)
    q = np.arange(0, 3, 1 / 30)
    left = np.stack([np.sin(tt + j) for j in range(7)], axis=1)
    right = np.stack([np.cos(tt + j) for j in range(7)], axis=1)
    path = tmp_path / "two_arms.mcap"
    _json_mcap(path, {"/joint_states": [(s, {"name": [f"left_{n}" for n in names], "position": left[k].tolist()})
                                        for k, s in enumerate(tt)]
                      + [(s + 0.002, {"name": [f"right_{n}" for n in names], "position": right[k].tolist()})
                         for k, s in enumerate(tt)]}, t0)
    st = f.mcap_joint_streams([path], t0 + q)
    state, _, note = f.joint_state(st, t0 + q)
    ref = np.concatenate([np.stack([np.interp(q, tt, a[:, j]) for j in range(7)], axis=1) for a in (left, right)],
                         axis=1)
    assert note is None and state.shape == (len(q), 14) and np.abs(state - ref).max() < 5e-3
    sig = f.mcap_signals([path], t0 + q, f.state_fields(st, state, None))
    assert not list(sig) and not sig.left_out, (list(sig), sig.left_out)
    root = tmp_path / "upload"
    d = _videos_with_an_hdf5_arm_state(root)
    (d / "robot.h5").unlink()
    th = t0 - 0.1 + np.arange(60) / 100
    for side, k in (("left", 0), ("right", 7)):
        a = np.stack([0.3 * np.sin(th - t0 + k + j) for j in range(7)], axis=1)
        with h5py.File(d / f"{side}.h5", "w") as h:
            h["timestamps"] = (th * 1e9).astype(np.int64)
            h["qpos"] = a
            h["action"] = a + 0.01
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ep = tmp_path / "eps" / rep["episodes"][0]["episode_id"]
    ctx = json.loads((ep / "context.json").read_text())
    assert ctx["state_kind"] == "joints" and ctx["source"]["state"] == "left qpos and right qpos", ctx
    z = np.load(ep / "state.npz")
    assert z["state"].shape == (12, 14) and z["action"].shape == (12, 14)
    assert np.allclose(z["action"], z["state"] + 0.01, atol=1e-4)
    kept = {x["name"] for x in ctx.get("signals") or []}
    assert not {"left qpos", "right qpos", "left action", "right action"} & kept, kept


def _a_side_from_value_names_only_completes_a_pair(tmp_path):
    """A side read from value names had replaced "only" on any arm whose topic names no side, so one arm named
    right_* beside a base's wheels, one named r_* beside a gripper channel, a sideless arm beside left and right legs,
    and a follower named left_* beside a sideless leader lost their state or action. A side from value names is used
    only when it completes one left and one right arm of six joints and a gripper; otherwise the topics' sides stand."""
    import numpy as np
    t0 = 1_790_000_000.0
    joints = [f"joint{i}" for i in range(1, 7)]
    tt = np.arange(0, 10.0, 0.01)
    q = t0 + np.arange(0, 10, 1 / 30)
    arm = lambda s, k=0: [float(np.sin(s + j + k)) for j in range(7)]
    cases = {
        "wheels": {"/joint_states": [(s, {"name": [f"right_{n}" for n in joints] + ["right_gripper"],
                                          "position": arm(s)}) for s in tt],
                   "/base/joint_states": [(s, {"name": ["left_wheel", "right_wheel"], "position": [s, -s]})
                                          for s in tt]},
        "leader": {"/follower/joint_states": [(s, {"name": [f"left_{n}" for n in joints] + ["left_gripper"],
                                                   "position": arm(s)}) for s in tt],
                   "/leader/joint_states": [(s, {"name": joints + ["gripper"], "position": arm(s, 0.1)}) for s in tt]},
        "gripper": {"/arm/joint_states": [(s, {"name": [f"r_{n}" for n in joints] + ["r_gripper"],
                                               "position": arm(s)}) for s in tt],
                    "/gripper/joint_states": [(s, {"name": ["finger_joint"], "position": [0.04]}) for s in tt]},
        "legs": {"/arm/joint_states": [(s, {"name": joints + ["gripper"], "position": arm(s)}) for s in tt],
                 "/joint_states": [(s, {"name": [f"left_leg_{i}" for i in range(6)], "position": [0.1] * 6})
                                   for s in tt]
                 + [(s + 0.002, {"name": [f"right_leg_{i}" for i in range(6)], "position": [0.2] * 6}) for s in tt]},
    }
    ref = np.stack([np.interp(q - t0, tt, [arm(s)[j] for s in tt]) for j in range(7)], axis=1)
    for label, chans in cases.items():
        path = tmp_path / f"{label}.mcap"
        _json_mcap(path, chans, t0)
        st = f.mcap_joint_streams([path], q)
        state, action, note = f.joint_state(st, q)
        assert note is None and state.shape == (len(q), 7), (label, note)
        assert np.abs(state - ref).max() < 1e-3, label
        assert (action is not None) == (label == "leader"), label


def _a_sided_set_pairs_only_when_its_names_are_an_arm(tmp_path):
    """A right arm and a left hand of seven joints on one /joint_states had paired as two arms, the hand read as the
    left arm. A set's side from its value names counts only when state_layout reads its names as six joints and a
    gripper, so the right arm is the only arm."""
    import numpy as np
    t0 = 1_790_000_000.0
    tt = np.arange(0, 10.0, 0.01)
    q = t0 + np.arange(0, 10, 1 / 30)
    arm = [f"right_joint{i}" for i in range(1, 7)] + ["right_gripper"]
    hand = [f"left_hand_joint{i}" for i in range(1, 8)]
    path = tmp_path / "hand.mcap"
    _json_mcap(path, {"/joint_states": [(s, {"name": arm, "position": [float(np.sin(s + j)) for j in range(7)]})
                                        for s in tt]
                      + [(s + 0.002, {"name": hand, "position": [0.1] * 7}) for s in tt]}, t0)
    st = f.mcap_joint_streams([path], q)
    state, _, note = f.joint_state(st, q)
    assert note is None and state.shape == (len(q), 7), note
    assert np.abs(state[:, 0] - np.sin(q - t0)).max() < 1e-3


def test_a_sided_set_pairs_only_when_its_names_are_an_arm():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_sided_set_pairs_only_when_its_names_are_an_arm(Path(t))


def test_a_side_from_value_names_only_completes_a_pair():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_side_from_value_names_only_completes_a_pair(Path(t))


def _one_side_of_two_hdf5_arms_that_fails_reads_neither(tmp_path):
    """A left.h5 and a right.h5 beside videos whose right qpos cannot be read (an infinity in every reading, or nine
    values) had the left arm read alone as a one arm state with no note. As joint_state leaves both MCAP arms unread
    when one side's channel is not the layout, neither arm is read, and the note names the side that failed and why.
    One infinity alone no longer fails a side: that reading is missing and filled like any short gap (h5_state)."""
    import json
    import h5py
    import numpy as np
    t0 = 1_790_000_000.0
    th = t0 - 0.1 + np.arange(60) / 100
    def upload(case, files):
        root = tmp_path / case / "upload"
        d = _videos_with_an_hdf5_arm_state(root)
        (d / "robot.h5").unlink()
        for stem, array, n in files:
            a = np.stack([0.3 * np.sin(th - t0 + j) for j in range(n)], axis=1)
            if case in ("inf", "one inf") and stem == "right":
                a[20 if case == "one inf" else slice(None), 2] = np.inf
            with h5py.File(d / f"{stem}.h5", "w") as h:
                h["timestamps"] = (th * 1e9).astype(np.int64)
                h[array] = a
        rep = f.convert(root, "teleop_arms", tmp_path / case / "eps", "test", 900)
        ep = tmp_path / case / "eps" / rep["episodes"][0]["episode_id"]
        return json.loads((ep / "context.json").read_text()), ep
    for case, files, failed, kept in (("inf", [("left", "qpos", 7), ("right", "qpos", 7)], "right qpos", "left qpos"),
                                      ("wide", [("left", "qpos", 7), ("right", "qpos", 9)], "right qpos", "left qpos"),
                                      ("six", [("left", "qpos", 6), ("right", "qpos", 7)], "left qpos", "right qpos")):
        ctx, _ = upload(case, files)
        assert ctx["state_kind"] == "none" and failed in ctx["state_note"], (case, ctx.get("state_note"))
        assert kept in {x["name"] for x in ctx.get("signals") or []}
    ctx, ep = upload("one inf", [("left", "qpos", 7), ("right", "qpos", 7)])
    assert ctx["state_kind"] == "joints" and np.load(ep / "state.npz")["state"].shape == (12, 14), ctx.get("state_note")
    assert [i["signal"] for i in _issues(ctx, "signal_not_finite")] == ["right qpos"], ctx.get("reader_issues")
    # an array of the other side named as the state but not the arm's own array (a glove's state) never cancels it
    ctx, ep = upload("glove", [("right_arm", "qpos", 7), ("left_glove", "state", 5)])
    assert ctx["state_kind"] == "joints" and ctx["source"]["state"] == "right_arm qpos", ctx.get("state_note")
    assert np.load(ep / "state.npz")["state"].shape == (12, 7)


def test_one_side_of_two_hdf5_arms_that_fails_reads_neither():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _one_side_of_two_hdf5_arms_that_fails_reads_neither(Path(t))


def test_two_arms_in_their_own_files_or_messages():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _two_arms_in_their_own_files_or_messages(Path(t))


def test_a_value_name_with_no_words_says_nothing_and_never_raises():
    """A value name that splits into no words ("" or "_") raised IndexError in state_layout when every seventh name
    is a gripper. Such a name says neither joints, a gripper nor a pose, so the other names and the width rule
    decide."""
    j = [f"j{i}" for i in range(1, 6)]
    assert f.state_words("") == [] and f.state_words("_") == []
    for blank in ("", "_", "__", " "):
        assert f.state_layout(7, "teleop_arms", [blank] + j + ["gripper"]) == ("joints", None)
        assert f.state_layout(7, "teleop_arms", j + [blank, "gripper"]) == ("joints", None)
        assert f.state_layout(7, "teleop_arms", [blank] * 7) == ("joints", None)
        assert f.state_layout(7, "handheld_gripper", [blank] * 7) == ("ee_pose", None)
        assert f.state_layout(14, "teleop_arms", ([blank] + j + ["gripper"]) * 2) == ("joints", None)
        # the other names still decide: a velocity named beside it is still not a position
        kind, note = f.state_layout(7, "teleop_arms", [blank] + j + ["gripper_vel"])
        assert kind == "none" and "gripper_vel" in note
        # a group that names a position for some values but is not a full pose is never read by width
        for rig in ("teleop_arms", "handheld_gripper"):
            kind, note = f.state_layout(7, rig, ["x", "y", "z", "roll", "pitch", blank, "gripper"])
            assert kind == "none" and "position for some values" in note, (rig, kind, note)


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


def test_an_arm_channel_that_names_seven_joints_and_no_gripper_is_not_read_as_six_and_a_gripper():
    """sensor_msgs/JointState names its values. Seven joints named and no gripper is a Franka arm, not six joints and
    a gripper; nothing is claimed as state, so every joint field stays a signal. The same channels unnamed keep the
    width rule."""
    import numpy as np
    from prepare import formats
    q = np.arange(0, 3, 1 / 30)
    t = np.arange(-0.1, 3.1, 0.01)
    pos = np.stack([np.sin(t + j) for j in range(7)], axis=1)
    named = {f"/{s}/joint_states": {"t": t, "pos": pos, "names": [f"fr3_{s}_joint{j}" for j in range(1, 8)]}
             for s in ("left", "right")}
    state, action, note = formats.joint_state(named, q)
    assert state is None and action is None and "7 joints and no gripper" in note
    assert formats.state_fields(named, state, action) == {}
    unnamed = {k: {"t": v["t"], "pos": v["pos"]} for k, v in named.items()}
    state, _, note = formats.joint_state(unnamed, q)
    assert state.shape == (len(q), 14) and note is None
    msg = {"name": ["j1", "j2"], "position": [0.0, 0.1], "gripper_pos": [0.5]}
    assert formats._joint_names(msg, 3) == ["j1", "j2", "gripper"] and formats._joint_names(msg, 2) == ["j1", "j2"]
    assert formats._joint_names({"name": [], "position": [0.0]}, 1) is None

    class Repeated:
        """A Protobuf repeated field: a sequence, but not a list."""
        def __init__(self, xs):
            self.xs = list(xs)

        def __len__(self):
            return len(self.xs)

        def __iter__(self):
            return iter(self.xs)

    class Msg:
        """A decoded ROS message: its fields are slots."""
        __slots__ = ("name", "position")

        def __init__(self):
            self.name = Repeated(["a", "b"])
            self.position = Repeated([0.0, 1.0])
    assert formats._joint_names(Msg(), 2) == ["a", "b"] and formats._joint_names({"name": "ab"}, 2) is None


def _an_mcap_joint_channel_keeps_the_names_its_messages_give(tmp_path):
    """The names reach joint_state from the file itself: a JointState style JSON channel naming seven Franka joints,
    and one naming six joints beside its gripper reading, are read with their names, and the Franka arm is left as
    signals."""
    import json
    import numpy as np
    from mcap.writer import Writer
    from prepare import formats
    t0 = 1_790_000_000.0
    path = tmp_path / "arms.mcap"
    with open(path, "wb") as fh:
        w = Writer(fh)
        w.start()
        sid = w.register_schema(name="joints", encoding="jsonschema", data=b"{}")
        franka = w.register_channel(topic="/left/joint_states", message_encoding="json", schema_id=sid)
        yam = w.register_channel(topic="/right/joint_state", message_encoding="json", schema_id=sid)
        health = w.register_channel(topic="/left/health", message_encoding="json", schema_id=sid)
        for i in range(330):
            ts = int((t0 - 0.1 + i / 100) * 1e9)
            w.add_message(franka, log_time=ts, publish_time=ts, data=json.dumps(
                {"name": [f"fr3_left_joint{j}" for j in range(1, 8)], "position": [0.01 * i + j for j in range(7)],
                 "velocity": [0.0] * 7}).encode())
            w.add_message(yam, log_time=ts, publish_time=ts, data=json.dumps(
                {"name": [f"joint{j}" for j in range(1, 7)], "position": [0.01 * i] * 6,
                 "gripper_pos": [0.5]}).encode())
            w.add_message(health, log_time=ts, publish_time=ts, data=b'{"ok": true}')
        w.finish()
    streams = formats.mcap_joint_streams([path])
    assert sorted(streams) == ["/left/joint_states", "/right/joint_state"]
    assert streams["/left/joint_states"]["names"] == [f"fr3_left_joint{j}" for j in range(1, 8)]
    assert streams["/right/joint_state"]["names"] == [f"joint{j}" for j in range(1, 7)] + ["gripper"]
    assert streams["/right/joint_state"]["pos"].shape == (330, 7)
    state, action, note = formats.joint_state(streams, t0 + np.arange(90) / 30)
    assert state is None and action is None and "7 joints and no gripper" in note


def test_an_mcap_joint_channel_keeps_the_names_its_messages_give():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _an_mcap_joint_channel_keeps_the_names_its_messages_give(Path(t))


def _json_mcap(path: Path, chans: dict, t0: float) -> None:
    """chans {topic: [(seconds after t0, message dict)]} as JSON channels of one MCAP, in time order."""
    import json
    from mcap.writer import Writer
    with open(path, "wb") as fh:
        w = Writer(fh)
        w.start()
        sid = w.register_schema(name="sensor_msgs/msg/JointState", encoding="jsonschema", data=b"{}")
        ids = {t: w.register_channel(topic=t, message_encoding="json", schema_id=sid) for t in chans}
        for s, t, m in sorted((s, t, m) for t, ms in chans.items() for s, m in ms):
            ns = int((t0 + s) * 1e9)
            w.add_message(ids[t], log_time=ns, publish_time=ns, data=json.dumps(m).encode())
        w.finish()


def _a_joint_states_rows_follow_their_own_names(tmp_path):
    """A JointState merged from several drivers lists the same joints in another order on every other message; its
    rows had been appended by position under the first message's names, so joints were swapped (0.9 rad off) in the
    state and in the signals. Each message's values are put in its name set's first order, by name."""
    import numpy as np
    t0 = 1_790_000_000.0
    names = [f"joint{i}" for i in range(1, 7)] + ["gripper"]
    tt = np.arange(-0.1, 3.1, 0.01)
    truth = np.stack([0.1 * (j + 1) * np.sin(tt + j) for j in range(6)] + [0.04 + 0.02 * np.sin(tt)], axis=1)
    order = [3, 0, 6, 1, 5, 2, 4]
    msgs = [(s, {"name": [names[i] for i in (order if k % 2 else range(7))],
                 "position": [float(truth[k, i]) for i in (order if k % 2 else range(7))],
                 "velocity": [float(10 * truth[k, i]) for i in (order if k % 2 else range(7))]})
            for k, s in enumerate(tt)]
    path = tmp_path / "reorder.mcap"
    _json_mcap(path, {"/left_arm/joint_states": msgs}, t0)
    st = f.mcap_joint_streams([path])
    assert list(st) == ["/left_arm/joint_states"] and st["/left_arm/joint_states"]["names"] == names
    assert np.abs(st["/left_arm/joint_states"]["pos"] - truth).max() < 1e-9
    q = np.arange(0, 3, 1 / 30)
    state, _, note = f.joint_state(st, t0 + q)
    ref = np.stack([np.interp(q, tt, truth[:, j]) for j in range(7)], axis=1)
    assert note is None and np.abs(state - ref).max() < 1e-6
    sig = f.mcap_signals([path], t0 + q)
    for field, scale in (("position", 1), ("velocity", 10)):
        name = f"/left_arm/joint_states {field}"
        assert sig.meta[name]["names"] == names, sig.meta[name]
        assert np.abs(sig[name] - scale * ref).max() < 0.05 * scale, field
    # a detector whose labels change from message to message names no value that is one reading over time
    objects = ["cup", "bowl", "spoon", "fork", "plate", "lid", "box", "bag", "can", "jar"]
    path = tmp_path / "detections.mcap"
    _json_mcap(path, {"/detections": [(s, {"labels": [objects[k % 10], objects[(k + 3) % 10]], "scores": [0.9, 0.8]})
                                      for k, s in enumerate(tt)]}, t0)
    sig = f.mcap_signals([path], t0 + q)
    assert not sig and sig.left_out == [("/detections scores", "its messages name its values in more than 8 "
                                                               "different ways, so no value is one reading over time")]


def _a_message_without_names_reads_with_the_named_set_of_its_width(tmp_path):
    """A JointState whose name list is empty for its first second (a driver before it starts), or on every tenth
    message, had its unnamed rows made a group of their own, which sorted first by its label and lost the state or
    built it from the unnamed tenth. A message without names joins the only named set of its width, as the reader
    read it before name sets. The arm is the named set that fits the layout with the most readings, never the first
    label. An unnamed field of varying width is one signal per width, labelled by it, so a message of another width
    than the first is never dropped."""
    import numpy as np
    t0 = 1_790_000_000.0
    names = [f"joint{i}" for i in range(1, 7)] + ["gripper"]
    tt = np.arange(0, 10.0, 0.01)
    q = np.arange(0, 10, 1 / 30)
    truth = np.stack([np.sin(tt + j) for j in range(7)], axis=1)
    ref = np.stack([np.interp(q, tt, truth[:, j]) for j in range(7)], axis=1)
    for label, unnamed in (("early", tt < 1.0), ("tenth", np.arange(len(tt)) % 10 == 0)):
        path = tmp_path / f"{label}.mcap"
        _json_mcap(path, {"/left_arm/joint_states": [(s, {"name": [] if unnamed[k] else names,
                                                          "position": truth[k].tolist()})
                                                     for k, s in enumerate(tt)]}, t0)
        st = f.mcap_joint_streams([path])
        assert list(st) == ["/left_arm/joint_states"] and st["/left_arm/joint_states"]["pos"].shape == (1000, 7)
        state, _, note = f.joint_state(st, t0 + q)
        assert note is None and np.abs(state - ref).max() < 1e-6, (label, note)
        sig = f.mcap_signals([path], t0 + q, f.state_fields(st, state, None))
        assert not list(sig) and not sig.left_out, (label, list(sig), sig.left_out)
    # two named sets that both fit the layout: the one with the most readings is the arm, whatever its label
    path = tmp_path / "two_sets.mcap"
    other = [f"a{i}" for i in range(1, 7)] + ["gripper"]
    _json_mcap(path, {"/left_arm/joint_states": [(s, {"name": names, "position": truth[k].tolist()})
                                                 for k, s in enumerate(tt)]
                      + [(s + 0.001, {"name": other, "position": [0.0] * 7}) for s in tt[::10]]}, t0)
    st = f.mcap_joint_streams([path])
    state, _, note = f.joint_state(st, t0 + q)
    assert note is None and np.abs(state - ref).max() < 1e-6
    # an unnamed field of varying width is one signal per width, labelled by it
    path = tmp_path / "widths.mcap"
    _json_mcap(path, {"/contacts": [(s, {"pressures": [1.0 + np.sin(s)] * (5 if k % 7 == 3 else 3)})
                                    for k, s in enumerate(tt)]}, t0)
    sig = f.mcap_signals([path], t0 + q)
    assert list(sig) == ["/contacts pressures (3 values)", "/contacts pressures (5 values)"] and not sig.left_out
    assert sig["/contacts pressures (3 values)"].shape == (300, 3) and sig["/contacts pressures (5 values)"].shape == (
        300, 5)
    path = tmp_path / "widths_rare.mcap"
    _json_mcap(path, {"/contacts": [(s, {"pressures": [1.0 + np.sin(s)] * (9 if k % 333 == 0 else 10)})
                                    for k, s in enumerate(tt)]}, t0)
    sig = f.mcap_signals([path], t0 + q)
    assert list(sig) == ["/contacts pressures (9 values)", "/contacts pressures (10 values)"], sig.left_out
    assert sig.meta["/contacts pressures (9 values)"]["rate_hz"] < 1 and not sig.left_out


def test_a_message_without_names_reads_with_the_named_set_of_its_width():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_message_without_names_reads_with_the_named_set_of_its_width(Path(t))


def test_a_field_whose_names_change_every_message_is_grouped_in_linear_time():
    """Each row was compared with every name set seen, so a detector's 7200 messages of new labels took 1.6 s where
    the reader took 0.06 s. Name sets are looked up by set, and a field stops being grouped once it passes
    NAME_SETS_MAX: 20000 messages of changing labels take well under a second."""
    import time
    groups = f.NameSets()
    start = time.perf_counter()
    for k in range(20000):
        f.name_group(groups, [f"obj{k}", f"obj{k + 1}"], [0.9, 0.8])
    assert time.perf_counter() - start < 0.5
    assert groups.overflow and len(groups) == f.NAME_SETS_MAX
    assert f.name_group(groups, ["obj0", "obj1"], [0.9, 0.8]) == (0, [0.9, 0.8])     # a set already seen still reads
    assert f.name_group(groups, ["obj1", "obj0"], [0.9, 0.8]) == (0, [0.8, 0.9])
    assert f.name_group(groups, ["new", "set"], [0.9, 0.8])[0] is None
    # only named sets count toward the limit: a field of eight unnamed widths still takes its first named set
    groups = f.NameSets()
    for n in range(1, 9):
        f.name_group(groups, None, [0.0] * n)
    assert f.name_group(groups, ["a", "b"], [1.0, 2.0]) == (8, [1.0, 2.0]) and not groups.overflow


def test_a_joint_states_rows_follow_their_own_names():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_joint_states_rows_follow_their_own_names(Path(t))


def _a_gripper_in_its_own_joint_states_messages_is_never_lost(tmp_path):
    """/joint_states that carries an arm's six joints and its gripper in messages of their own had every gripper
    message skipped (rows of another width), so the gripper was neither state, signal nor named as left out. A name
    set whose every name is a gripper's joins the channel's one other name set as its gripper, placed at the arm's
    message times, when it covers them; otherwise each name set is its own stream and its own signal. A name set
    beside the arm's that is not a gripper (wheels) stays a signal when the arm is read as the state."""
    import numpy as np
    t0 = 1_790_000_000.0
    joints = [f"joint{i}" for i in range(1, 7)]
    tt = np.arange(-0.1, 3.1, 0.01)
    arm = np.stack([0.2 * np.sin(tt + j) for j in range(6)], axis=1)
    grip = 0.04 + 0.03 * np.sin(2 * tt)
    q = t0 + np.arange(0, 3, 1 / 30)

    def split(until):
        msgs = [(s, {"name": joints, "position": arm[k].tolist()}) for k, s in enumerate(tt)]
        return msgs + [(s + 0.003, {"name": ["finger_joint"], "position": [float(grip[k])]})
                       for k, s in enumerate(tt) if s < until]
    path = tmp_path / "split.mcap"
    _json_mcap(path, {"/joint_states": split(99)}, t0)
    st = f.mcap_joint_streams([path])
    assert list(st) == ["/joint_states"], list(st)
    s = st["/joint_states"]
    assert s["names"] == joints + ["finger_joint"] and s["pos"].shape == (len(tt), 7)
    assert np.abs(s["pos"][:, 6] - grip).max() < 0.002                # the gripper at the arm's message times
    state, _, note = f.joint_state(st, q)
    assert note is None and state.shape == (len(q), 7)
    sig = f.mcap_signals([path], q, f.state_fields(st, state, None))
    assert not [k for k in sig if "position" in k] and not [x for x, _ in sig.left_out if "position" in x]
    # a gripper that stops halfway does not cover the arm: each name set is its own stream and signal, the gripper's
    # NaN after its last message
    path = tmp_path / "half.mcap"
    _json_mcap(path, {"/joint_states": split(1.5)}, t0)
    st = f.mcap_joint_streams([path])
    assert sorted(st) == ["/joint_states (finger_joint)", "/joint_states (joint1, joint2 and 4 more)"], sorted(st)
    state, _, note = f.joint_state(st, q)
    assert state is None and "6 values per frame" in note, note
    sig = f.mcap_signals([path], q, f.state_fields(st, state, None))
    assert sig.meta["/joint_states position (joint1, joint2 and 4 more)"]["names"] == joints
    finger = sig["/joint_states position (finger_joint)"]
    assert np.isnan(finger[q - t0 > 1.6]).all() and np.isfinite(finger[q - t0 < 1.4]).all(), sig.left_out
    # the arm with its gripper is the state; wheels on the same channel stay a signal
    path = tmp_path / "wheels.mcap"
    named = joints + ["gripper"]
    wheels = ["left_wheel", "right_wheel"]
    _json_mcap(path, {"/joint_states": [(s, {"name": named, "position": arm[k].tolist() + [float(grip[k])]})
                                        for k, s in enumerate(tt)]
                      + [(s + 0.003, {"name": wheels, "position": [float(s), float(-s)]}) for s in tt]}, t0)
    st = f.mcap_joint_streams([path])
    state, _, note = f.joint_state(st, q)
    assert note is None and state.shape == (len(q), 7)
    assert np.abs(state[:, 0] - np.interp(q - t0, tt, arm[:, 0])).max() < 1e-6
    sig = f.mcap_signals([path], q, f.state_fields(st, state, None))
    assert list(sig) == ["/joint_states position (left_wheel, right_wheel)"], list(sig)
    assert sig.meta["/joint_states position (left_wheel, right_wheel)"]["names"] == wheels
    # a gripper of two fingers joins as one value, as _joint_row takes a gripper field's first value, and its cover is
    # judged against the footage: an arm channel that runs on past the footage does not keep it apart
    path = tmp_path / "fingers.mcap"
    long = np.arange(-2.0, 5.0, 0.01)
    fingers = ["finger_left", "finger_right"]
    _json_mcap(path, {"/joint_states": [(s, {"name": joints, "position": [0.2 * np.sin(s + j) for j in range(6)]})
                                        for s in long]
                      + [(s + 0.003, {"name": fingers, "position": [0.04 + 0.03 * np.sin(2 * s), 0.0]})
                         for s in tt]}, t0)
    st = f.mcap_joint_streams([path], q)
    assert list(st) == ["/joint_states"], list(st)
    assert st["/joint_states"]["names"] == joints + ["finger_left"] and st["/joint_states"]["pos"].shape[1] == 7
    state, _, note = f.joint_state(st, q)
    assert note is None and np.abs(state[:, 6] - (0.04 + 0.03 * np.sin(2 * (q - t0)))).max() < 0.002


def test_a_gripper_in_its_own_joint_states_messages_is_never_lost():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_gripper_in_its_own_joint_states_messages_is_never_lost(Path(t))


def test_a_list_of_names_beside_a_numeric_array_names_its_values():
    """A JointState's name gives one name per value of position, velocity and effort beside it, in a dict (JSON) and
    in a decoded ROS message with slots. Names with another count, names under another parent, a list with repeated
    entries (units), a field _numbers leaves out, two lists that could both name the array, and a field that already
    names its values give none of their own."""
    msg = {"header": {"stamp": {"sec": 1, "nanosec": 0}, "frame_id": "base"}, "name": ["j1", "j2", "j3"],
           "position": [0.0, 0.1, 0.2], "velocity": [1.0, 1.1, 1.2], "effort": [2.0, 2.1, 2.2]}
    nums = f._numbers(msg)
    for k in ("position", "velocity", "effort"):
        assert nums[k][2] == ["j1", "j2", "j3"], (k, nums[k])
    assert f._numbers({**msg, "name": ["j1", "j2"]})["position"][2] is None
    assert f._numbers({**msg, "name": []})["position"][2] is None
    nested = f._numbers({"names": ["a", "b", "c"], "arm": {"position": [0.0, 0.1, 0.2]}})
    assert nested["arm.position"][2] is None
    nested = f._numbers({"arm": {"names": ["a", "b", "c"]}, "position": [0.0, 0.1, 0.2]})
    assert nested["position"][2] is None
    two = f._numbers({"name": ["a", "b", "c"], "frames": ["x", "y", "z"], "position": [0.0, 0.1, 0.2]})
    assert two["position"][2] is None
    units = f._numbers({"units": ["rad", "rad", "rad"], "position": [0.0, 0.1, 0.2]})
    assert units["position"][2] is None
    one = f._numbers({"name": ["a", "b", "c"], "units": ["rad", "rad", "rad"], "position": [0.0, 0.1, 0.2]})
    assert one["position"][2] == ["a", "b", "c"]
    for skipped in ("stamp", "timestamp", "frame_index", "_names"):          # fields _numbers leaves out
        assert f._numbers({skipped: ["a", "b"], "position": [0.0, 1.0]})["position"][2] is None, skipped
    own =f._numbers({"name": ["a", "b"], "x": 1.0, "y": 2.0, "points": [{"u": 1.0}, {"u": 2.0}]})
    assert own[""][2] == ["x", "y"] and own["points"][2] == ["0.u", "1.u"]

    class Repeated:
        """A Protobuf repeated field: a sequence, but not a list."""
        def __init__(self, xs):
            self.xs = list(xs)

        def __len__(self):
            return len(self.xs)

        def __iter__(self):
            return iter(self.xs)

    class JointState:
        """A decoded ROS message: its fields are slots."""
        __slots__ = ("name", "position")

        def __init__(self):
            self.name = Repeated(["a", "b"])
            self.position = Repeated([0.0, 1.0])
    assert f._numbers(JointState())["position"][2] == ["a", "b"]


def _a_joint_channel_that_is_not_the_state_keeps_its_joint_names_as_signals(tmp_path):
    """A JointState channel read as signals, not as the state (the Franka of seven joints and no gripper): its
    position and velocity carry the joint names, and the rows at each instant are labelled by them, not [0] to [6]."""
    import json
    import numpy as np
    from mcap.writer import Writer
    from label import signals as sg
    t0 = 1_790_000_000.0
    path = tmp_path / "arm.mcap"
    joints = [f"fr3_left_joint{j}" for j in range(1, 8)]
    with open(path, "wb") as fh:
        w = Writer(fh)
        w.start()
        sid = w.register_schema(name="sensor_msgs/msg/JointState", encoding="jsonschema", data=b"{}")
        ch = w.register_channel(topic="/left/joint_states", message_encoding="json", schema_id=sid)
        for i in range(330):
            ts = int((t0 - 0.1 + i / 100) * 1e9)
            w.add_message(ch, log_time=ts, publish_time=ts, data=json.dumps(
                {"header": {"stamp": {"sec": ts // 10**9, "nanosec": ts % 10**9}, "frame_id": ""}, "name": joints,
                 "position": [0.01 * i + j for j in range(7)], "velocity": [0.02 * i * (j + 1) for j in range(7)],
                 "effort": []}).encode())
        w.finish()
    q = t0 + np.arange(90) / 30
    sig = f.mcap_signals([path], q)
    for field in ("position", "velocity"):
        name = f"/left/joint_states {field}"
        assert sig.meta[name]["names"] == joints, (name, sig.meta[name])
        rows = sg.summary_rows(name, sig[name], [0, 45, 89], names=sig.meta[name]["names"])
        assert [r for r, _ in rows] == [f"{name} {j}" for j in joints], rows


def test_a_joint_channel_that_is_not_the_state_keeps_its_joint_names_as_signals():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_joint_channel_that_is_not_the_state_keeps_its_joint_names_as_signals(Path(t))


JOINT_STATE_ROS2 = """std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""


def _a_ros2_joint_state_keeps_its_joint_names_as_signals(tmp_path):
    """The same names through a real decoder: a sensor_msgs/msg/JointState written in ROS 2 CDR and decoded by
    mcap_ros2, whose messages are slots objects with name as a list of strings."""
    import numpy as np
    from mcap_ros2.writer import Writer
    t0 = 1_790_000_000.0
    path = tmp_path / "arm.mcap"
    joints = [f"fr3_left_joint{j}" for j in range(1, 8)]
    with open(path, "wb") as fh:
        w = Writer(fh)
        schema = w.register_msgdef("sensor_msgs/msg/JointState", JOINT_STATE_ROS2)
        for i in range(330):
            ts = int((t0 - 0.1 + i / 100) * 1e9)
            w.write_message("/left/joint_states", schema, {
                "header": {"stamp": {"sec": ts // 10**9, "nanosec": ts % 10**9}, "frame_id": "base"}, "name": joints,
                "position": [0.01 * i + j for j in range(7)], "velocity": [0.02 * i * (j + 1) for j in range(7)],
                "effort": []}, log_time=ts, publish_time=ts)
        w.finish()
    sig = f.mcap_signals([path], t0 + np.arange(90) / 30)
    for field in ("position", "velocity"):
        assert sig.meta[f"/left/joint_states {field}"]["names"] == joints, sig.meta


def test_a_ros2_joint_state_keeps_its_joint_names_as_signals():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    try:
        import mcap_ros2.writer  # noqa: F401
    except ImportError:
        print("ROS 2 joint names check skipped: mcap_ros2 is not installed")
        return
    with tempfile.TemporaryDirectory() as t:
        _a_ros2_joint_state_keeps_its_joint_names_as_signals(Path(t))


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


def test_a_long_gap_in_an_arms_readings_is_not_drawn_as_motion():
    """An arm channel whose recorder stopped for 2 s had the gap filled by a straight line, which the prompt showed as
    recorded motion and no still span could see. A gap longer than STATE_EDGE_SLACK_S leaves the state unread with a
    note naming the channel and the gap's time; a gap of three frames is filled as before."""
    import numpy as np
    q = np.arange(300) / 30.0
    t = np.arange(-0.05, 10.05, 0.01)
    pos = np.stack([np.sin(t + j) for j in range(7)], axis=1)
    gap = (t < 4.0) | (t > 6.0)
    streams = {"/left/joint_state": {"t": t, "pos": pos}, "/right/joint_state": {"t": t[gap], "pos": pos[gap]}}
    state, action, note = f.joint_state(streams, q)
    assert state is None and action is None
    assert note == ("Labelled from the cameras, because the recorded arm state /right/joint_state has no reading from "
                    "4.0 s to 6.0 s, a gap longer than the 0.5 s the reader fills."), note
    short = (t < 4.0) | (t > 4.1)
    state, _, note = f.joint_state({**streams, "/right/joint_state": {"t": t[short], "pos": pos[short]}}, q)
    assert note is None and state.shape == (300, 14)
    assert np.abs(state[:, 7:] - np.stack([np.interp(q, t, pos[:, j]) for j in range(7)], axis=1)).max() < 5e-3
    # a command channel with a long gap is no action, while the state is read
    lead = {"/leader_left/joint_pos": {"t": t[gap], "pos": pos[gap]}, "/leader_right/joint_pos": {"t": t, "pos": pos}}
    state, action, note = f.joint_state({"/left/joint_state": streams["/left/joint_state"],
                                         "/right/joint_state": {"t": t, "pos": pos}, **lead}, q)
    assert state is not None and action is None and note is None
    # a gap is measured inside the footage: one latched message 3 s before the first frame, then readings from 0.2 s
    late = np.concatenate([[-3.0], np.arange(0.2, 10.05, 0.01)])
    rows = np.stack([np.sin(late + j) for j in range(7)], axis=1)
    state, _, note = f.joint_state({"/left/joint_state": {"t": late, "pos": rows},
                                    "/right/joint_state": {"t": t, "pos": pos}}, q)
    assert note is None and state.shape == (300, 14), note
    # a gap is a stop only when it is also longer than STATE_STOP_STEPS of the stream's own steps: an arm logged at
    # 1.5 Hz is read as before, while a 30 Hz recorder that stops for 2 s is still not drawn as motion
    slow = np.arange(0, 10.01, 0.66)
    rows = np.stack([np.sin(slow + j) for j in range(7)], axis=1)
    state, _, note = f.joint_state({"/left_arm/joint_states": {"t": slow, "pos": rows}}, q)
    assert note is None and state.shape == (300, 7), note
    hz30 = np.arange(-0.05, 10.05, 1 / 30)
    hz30 = hz30[(hz30 < 4.0) | (hz30 > 6.0)]
    rows = np.stack([np.sin(hz30 + j) for j in range(7)], axis=1)
    state, _, note = f.joint_state({"/left_arm/joint_states": {"t": hz30, "pos": rows}}, q)
    assert state is None and "has no reading from 4.0 s to 6.0 s" in note, note
    for first, want in ((0.3, None), (0.8, (0.0, 0.8))):
        tt = np.concatenate([[-3.0], np.arange(first, 10.0, 0.01)])
        assert f.fill_rows(q, tt, np.zeros((len(tt), 1)))[1] == want, first


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
    signals under their own names, while the joints and gripper already read as the state and a status report sent
    once do not. A channel that stops before the footage ends is kept, NaN after its last message, with a data issue
    giving the span it misses."""
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
    assert not any(n.startswith("/system/cpu") for n in names) and "/gripper/force wrench" in names, names
    part = _issues(ctx, "signal_partial_span")
    assert [i["signal"] for i in part] == ["/gripper/force wrench"] and abs(part[0]["t0_s"] - 1.0) < 0.1, part
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


BOOT_NS = 2_600e9                # 43 minutes of uptime, in nanoseconds


def _step_s(a):
    import numpy as np
    return float(np.median(np.diff(a)))


def test_a_fast_pad_in_the_cameras_unit_is_read_by_the_span_it_shares_with_the_camera():
    """A 1 kHz pad and a 30 fps camera, both in nanoseconds from boot: the pad's step of 1e6 alone reads as
    microseconds, which would make its 1.4 s span 1400 s."""
    import numpy as np
    cam = BOOT_NS + np.arange(40) * 33_333_333.0
    out = f._clocks_in_seconds({"cam": cam, "pad": cam[0] - 1e7 + np.arange(1400) * 1e6}, reference="cam")
    assert abs(_step_s(out["pad"]) - 1e-3) < 1e-9 and abs(_step_s(out["cam"]) - 1 / 30) < 1e-9


def test_a_pad_in_microseconds_beside_a_camera_in_nanoseconds_is_read_in_microseconds():
    """Each driver stamps in its own unit: a pad in microseconds from boot steps 1e3, which alone reads as
    milliseconds."""
    import numpy as np
    cam = BOOT_NS + np.arange(40) * 33_333_333.0
    out = f._clocks_in_seconds({"cam": cam, "pad": BOOT_NS / 1e3 + np.arange(1400) * 1e3}, reference="cam")
    assert abs(_step_s(out["pad"]) - 1e-3) < 1e-9


def test_a_camera_in_seconds_and_an_imu_in_nanoseconds_both_from_zero_keep_their_units():
    import numpy as np
    out = f._clocks_in_seconds({"cam": np.arange(40) / 30.0, "imu": np.arange(400) * 5e6})
    assert abs(_step_s(out["cam"]) - 1 / 30) < 1e-9 and abs(_step_s(out["imu"]) - 5e-3) < 1e-9


def test_a_pad_log_longer_than_the_clip_is_read_where_its_range_overlaps_the_cameras():
    """A 1 kHz pad in nanoseconds from boot that starts 3 s before a 3 s camera clip and ends 3 s after it: its span is
    three times the camera's, but only nanoseconds put its range over the camera's."""
    import numpy as np
    cam = BOOT_NS + np.arange(90) * 33_333_333.0
    pad = BOOT_NS - 3e9 + np.arange(9000) * 1e6
    out = f._clocks_in_seconds({"cam": cam, "pad": pad}, reference="cam")
    assert abs(_step_s(out["pad"]) - 1e-3) < 1e-9 and abs((out["pad"][0] - out["cam"][0]) + 3) < 1e-6


def test_a_pad_that_records_part_of_the_episode_is_read_where_its_range_overlaps_the_cameras():
    """A 1 kHz pad in nanoseconds from boot that ran for 40% of the clip, from 0.5 s in: its step alone reads as
    microseconds, and no unit puts its span within a factor of 2 of the camera's."""
    import numpy as np
    cam = BOOT_NS + np.arange(90) * 33_333_333.0
    pad = BOOT_NS + 5e8 + np.arange(1200) * 1e6
    out = f._clocks_in_seconds({"cam": cam, "pad": pad}, reference="cam")
    assert abs(_step_s(out["pad"]) - 1e-3) < 1e-9 and abs((out["pad"][0] - out["cam"][0]) - 0.5) < 1e-6


def test_a_clock_from_zero_whose_span_fits_no_unit_keeps_its_own_reading():
    """Beside a camera in seconds from the recording's start, a 100 Hz sensor in nanoseconds from that start that ran
    for a tenth of the clip: its start says nothing about its unit, so it is read as it would be alone."""
    import numpy as np
    short = np.arange(100) * 1e7
    out = f._clocks_in_seconds({"cam": np.arange(300) / 30.0, "short": short}, reference="cam")
    assert np.array_equal(out["short"], f._seconds(short))


def test_without_a_camera_clock_an_ambiguous_largest_step_clock_is_no_reference():
    """A 1 kHz pad in nanoseconds from boot has the largest step, but its size settles no unit (its step alone reads
    as microseconds), so a 100 Hz encoder in milliseconds is not matched to it and keeps its own 10 ms step."""
    import numpy as np
    raw = {"pad": BOOT_NS + np.arange(30000) * 1e6, "encoder": BOOT_NS / 1e6 + np.arange(3000) * 10.0}
    out = f._clocks_in_seconds(raw)
    assert abs(_step_s(out["encoder"]) - 1e-2) < 1e-9


def test_a_reference_clock_that_never_steps_leaves_the_other_clocks_alone():
    """A camera clock stuck at one value says nothing about its unit, so nothing is matched to it."""
    import numpy as np
    pad = BOOT_NS + np.arange(1000) * 1e6
    out = f._clocks_in_seconds({"cam": np.full(50, BOOT_NS), "pad": pad}, reference="cam")
    assert np.array_equal(out["pad"], f._seconds(pad))


def test_sparse_event_stamps_from_zero_keep_their_own_reading():
    """Three event stamps from the recording's start over 30 ms, in nanoseconds, beside a 30 s camera from that start in
    seconds: a few stamps are not a sampled stream, so their span is not matched to the camera's."""
    import numpy as np
    out = f._clocks_in_seconds({"cam": np.arange(900) / 30.0, "events": np.array([0, 1.5e7, 3e7])}, reference="cam")
    assert abs(_step_s(out["events"]) - 0.015) < 1e-9


def test_a_leading_zero_before_a_pads_first_stamp_does_not_hide_its_unit():
    """A recorder that writes 0 before a 1 kHz pad's first stamp in nanoseconds from boot: the clock's position is read
    from its median and its range from its 1st and 99th percentiles, so the 0 neither puts it near zero nor stretches
    its range over every unit."""
    import numpy as np
    cam = BOOT_NS + np.arange(90) * 33_333_333.0
    pad = BOOT_NS + np.arange(3000) * 1e6
    pad[0] = 0
    out = f._clocks_in_seconds({"cam": cam, "pad": pad}, reference="cam")
    assert abs(_step_s(out["pad"][1:]) - 1e-3) < 1e-9


def test_a_single_clock_is_read_as_it_would_be_alone():
    import numpy as np
    cam = BOOT_NS + np.arange(40) * 33_333_333.0
    assert np.array_equal(f._clocks_in_seconds({"cam": cam})["cam"], f._seconds(cam))


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


def test_a_pressure_map_with_one_dead_cell_is_a_reading_at_every_frame():
    """A row has a reading when any of its values is finite, as checks/sensors.py counts it, so a 16 x 16 map whose
    last cell never reads is kept, NaN in that cell."""
    import numpy as np, pandas as pd
    n = 60
    cells = [np.r_[np.full(255, 3000.0), np.nan] for _ in range(n)]
    out = f.recorded_signals(pd.DataFrame({"pad": cells}), set(), n)
    assert "pad" in out and np.isnan(out["pad"][:, 255]).all() and (out["pad"][:, :255] == 3000).all()


def test_a_reading_missing_at_most_frames_is_kept_and_one_never_read_is_left_out_with_the_reason():
    import numpy as np, pandas as pd
    n = 60
    force = np.linspace(0, 1, n)
    force[:40] = np.nan
    out = f.recorded_signals(pd.DataFrame({"force": list(force), "dead": [np.nan] * n}), set(), n)
    assert "force" in out and np.isnan(out["force"][:40]).all() and np.isfinite(out["force"][40:]).all()
    assert "dead" not in out and ("dead", "no reading at any frame") in out.left_out


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


LEROBOT_CAMERAS = {"exo": "observation.images.cam_high", "left": "observation.images.cam_left_wrist",
                   "right": "observation.images.cam_right_wrist"}


def test_a_depth_stream_named_only_by_the_datasets_prefix_goes_with_the_scene_camera_and_says_why():
    """observation.depth has only the word every camera key shares (observation), which names no camera."""
    view, source = f.depth_camera("observation.depth", LEROBOT_CAMERAS, "exo")
    assert view == "exo" and "scene camera" in source


def test_a_depth_stream_whose_words_fit_two_cameras_goes_with_the_scene_camera_and_says_why():
    """observation.depth.wrist fits cam_left_wrist and cam_right_wrist alike, so it is not paired with either."""
    view, source = f.depth_camera("observation.depth.wrist", LEROBOT_CAMERAS, "exo")
    assert view == "exo" and "scene camera" in source


def test_the_one_camera_of_an_episode_is_paired_by_its_own_words():
    assert f.depth_camera("cam_high_depth", {"exo": "cam_high"}, "exo") == ("exo", "cam_high_depth")


def test_a_depth_stream_with_no_camera_of_its_own_goes_with_the_scene_camera_and_says_why():
    view, source = f.depth_camera("observations/depth/zed", {"exo": "top", "left": "wrist_left"}, "exo")
    assert view == "exo" and source.startswith("observations/depth/zed (") and "scene camera" in source


def test_a_fast_signals_variation_is_kept_only_for_a_signal_named_for_touch():
    """A pose's jitter within frames is not where a slip shows, however its numbers rest and rise."""
    import numpy as np
    a = np.r_[np.zeros(30), np.linspace(0, 0.3, 10), np.full(20, 0.3)][:, None]
    var = np.r_[np.zeros(35), np.full(25, 0.1)][:, None]
    assert f._variation_matters("left_pressure", a, var)
    assert not f._variation_matters("observation.state.torso", a, var)


def test_a_name_says_touch_by_whole_words_alone_plural_or_numbered():
    for name in ("tactiles", "tactile_left_raw", "right_pressure", "pressure_topic", "fsr0", "left_contact",
                 "right_contact", "finger_force", "wrench.force"):
        assert f.names_touch(name), name
    for name in ("digital_cam", "reinforcement_reward", "observation.state.torso", "gripper_effort", "odom.position",
                 "contactless_id"):
        assert not f.names_touch(name), name


def test_a_hand_poses_finger_digits_are_not_touch():
    """digit names a DIGIT tactile camera, but in a hand pose it names a finger, so signals do not use the camera
    brands' words."""
    for name in ("hand.digits", "digit_1_tip", "left_hand.digit2.pos"):
        assert not f.names_touch(name), name


def test_a_command_is_never_touch():
    """A commanded force is not a measured one."""
    for name in ("action.gripper_force", "gripper_force_cmd", "target_pressure", "desired_contact"):
        assert not f.names_touch(name), name


def test_tactile_camera_names_still_say_they_sense_touch():
    for name in ("tactile_left_heatmap", "gelsight_left", "digit_0", "xense_right", "tactile_left"):
        assert f.is_sensing(name), name
    assert not f.is_sensing("digital_cam")


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


# ---------------------------------------------------------------- no drop: keep what is usable, flag what is wrong

def _issues(ctx: dict, kind: str | None = None) -> list[dict]:
    return [i for i in ctx.get("reader_issues") or [] if kind is None or i["kind"] == kind]


def _episode_ctx(out: Path, rep: dict, name_part: str) -> dict:
    import json
    e = next(e for e in rep["episodes"] if name_part in e["name"])
    return json.loads((out / e["episode_id"] / "context.json").read_text())


def _a_mixed_upload_reads_every_format_it_holds(tmp_path):
    """An upload of plain videos, an HDF5 file with a camera and an HDF5 glove file with none had been read as HDF5
    alone: the videos were never mentioned. Every format is read, the glove file goes with the camera episode of its
    folder, and a file no reader opens is named. The camera file has no clock and the glove's counts from its own
    start, so the glove is placed from both starts: its arrays are signals marked as aligned by an assumed start,
    each a data issue, and its qpos is never read as the arm state."""
    import h5py
    import numpy as np
    root = tmp_path / "upload"
    (root / "a").mkdir(parents=True)
    _clip(root / "a" / "take1.mp4", 10)
    _clip(root / "a" / "take2.mp4", 10)
    with h5py.File(root / "b.h5", "w") as h:
        h["cam"] = np.zeros((10, 96, 96, 3), np.uint8)
    with h5py.File(root / "glove.h5", "w") as h:
        h["pressure"] = np.random.default_rng(0).random((10, 16, 16))
        h["qpos"] = np.stack([0.3 * np.sin(np.arange(10) / 3 + j) for j in range(14)], axis=1)
        h["time"] = np.arange(10) / 30
    (root / "calib.bin").write_bytes(b"x")
    det, items = f.plan(root)
    assert sorted(it["kind"] for it in items) == ["hdf5", "video", "video"], items
    h5 = next(it for it in items if it["kind"] == "hdf5")
    assert [Path(p).name for p in h5["state"]] == ["glove.h5"]
    assert any("calib.bin" in m for m in det["missing"]), det["missing"]
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 3, rep
    ctx = _episode_ctx(tmp_path / "eps", rep, "b")
    sig = {s["name"]: s for s in ctx["signals"]}
    assert {"pressure", "qpos"} <= set(sig), ctx.get("signals")
    assert ctx["state_kind"] == "none" and "state" not in ctx["source"], (ctx["state_kind"], ctx["source"])
    assert sig["qpos"].get("aligned_by") == "assumed start" and sig["pressure"].get("aligned_by") == "assumed start"
    assert {i["signal"] for i in _issues(ctx, "signal_alignment_assumed")} == {"pressure", "qpos"}, ctx["reader_issues"]


def test_a_mixed_upload_reads_every_format_it_holds():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_mixed_upload_reads_every_format_it_holds(Path(t))


def _glove_h5(path: Path, n: int, t0: float = 0.0) -> None:
    import h5py
    import numpy as np
    with h5py.File(path, "w") as h:
        h["pressure"] = np.random.default_rng(1).random((n, 4, 4))
        h["time"] = t0 + np.arange(n) / 30


def _sensor_files_beside_videos_are_read_without_capture_times_and_by_take(tmp_path):
    """Sensor files beside videos had been read only when the folder held one episode and the videos carried capture
    times. Without capture times they are placed from both starts, as a table is, with a data issue saying the
    alignment assumes a common start (the arm state is never read that way, its channels stay signals); in a folder of
    several episodes a sensor file goes with the episode its name gives the take of, and one whose name gives none
    is listed with the reason."""
    import numpy as np
    root = tmp_path / "upload"
    one = root / "one"
    one.mkdir(parents=True)
    _clip(one / "top.mp4", 30)
    _glove_h5(one / "glove.h5", 30, t0=1_790_000_000.0)
    _json_mcap(one / "imu.mcap", {"/imu": [(k / 60, {"accel": [0.1 * k, 0.0, 9.8]}) for k in range(60)]},
               1_790_000_000.0)
    two = root / "two"
    two.mkdir()
    _clip(two / "run1.mp4", 30)
    _clip(two / "run2.mp4", 30)
    _glove_h5(two / "glove_run2.h5", 30)
    _glove_h5(two / "mat.h5", 30)
    det, items = f.plan(root)
    by = {it["name"]: it for it in items}
    assert [p.name for p in by["two/run2"]["state"]] == ["glove_run2.h5"] and not by["two/run1"]["state"]
    assert [p.name for p in by["two/run1"]["state_shared"]] == ["mat.h5"]
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["failed"], rep
    ctx = _episode_ctx(tmp_path / "eps", rep, "one/top")
    names = [s["name"] for s in ctx.get("signals") or []]
    assert any("pressure" in nm for nm in names) and any("/imu" in nm for nm in names), names
    issues = _issues(ctx, "signal_alignment_assumed")
    assert len(issues) == 2 and all("common start" in i["what"] for i in issues), ctx.get("reader_issues")
    run2 = _episode_ctx(tmp_path / "eps", rep, "two/run2")
    assert any("pressure" in s["name"] for s in run2["signals"]), run2.get("signals")
    run1 = _episode_ctx(tmp_path / "eps", rep, "two/run1")
    assert any(u.startswith("mat.h5") for u in run1["source"]["unused_signals"]), run1["source"]
    assert np.load(tmp_path / "eps" / run2["episode_id"] / "signals.npz")["s0"].shape[0] == 30


def test_sensor_files_beside_videos_are_read_without_capture_times_and_by_take():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _sensor_files_beside_videos_are_read_without_capture_times_and_by_take(Path(t))


def _one_bad_value_never_drops_a_signal(tmp_path):
    """A signal had been dropped whole for one inf among its readings, for starting or ending inside the footage, or
    for reading at fewer than half the frames. Each is kept, NaN where it has no finite reading, and each gap or bad
    value range is a data issue on the episode."""
    import json
    import h5py
    import numpy as np
    import pandas as pd
    root = tmp_path / "upload"
    root.mkdir()
    n = 60
    with h5py.File(root / "ep.h5", "w") as h:
        h["cam"] = np.zeros((n, 64, 64, 3), np.uint8)
        h["time"] = 1_790_000_000.0 + np.arange(n) / 30
        force = np.random.default_rng(2).random((n, 3))
        force[5, 0] = np.inf
        h["force"] = force
        late = h.create_group("late")
        late["time"] = 1_790_000_001.0 + np.arange(30) / 30            # from 1 s of the 2 s footage
        late["temp"] = np.linspace(20, 21, 30)
    rep = f.convert(root, "ego_head", tmp_path / "eps", "test", 900)
    assert not rep["failed"], rep
    ctx = _episode_ctx(tmp_path / "eps", rep, "ep")
    names = [s["name"] for s in ctx["signals"]]
    assert "force" in names and "late/temp" in names, (names, ctx["source"])
    bad = _issues(ctx, "signal_not_finite")
    assert [i["signal"] for i in bad] == ["force"] and abs(bad[0]["t0_s"] - 5 / 30) < 1e-3, ctx["reader_issues"]
    part = _issues(ctx, "signal_partial_span")
    assert [i["signal"] for i in part] == ["late/temp"] and part[0]["t0_s"] == 0.0, ctx["reader_issues"]
    assert abs(part[0]["t1_s"] - 1.0) < 0.1, part
    z = np.load(tmp_path / "eps" / ctx["episode_id"] / "signals.npz")
    temp = z[next(s["key"] for s in ctx["signals"] if s["name"] == "late/temp")]
    assert np.isnan(temp[:25]).all() and np.isfinite(temp[35:]).all()
    frc = z[next(s["key"] for s in ctx["signals"] if s["name"] == "force")]
    assert np.isnan(frc[5, 0]) and np.isfinite(frc[5, 1:]).all() and np.isfinite(np.delete(frc, 5, 0)).all()
    # an MCAP channel with an inf, and one that stops half way, are kept the same way
    q = 1_790_000_000.0 + np.arange(n) / 30
    _json_mcap(tmp_path / "s.mcap", {
        "/inf": [(k / 30, {"v": [float("inf") if k == 10 else float(np.sin(k / 5))]}) for k in range(n)],
        "/half": [(k / 30, {"v": [float(np.sin(k / 5))]}) for k in range(n // 2)]}, 1_790_000_000.0)
    sig = f.mcap_signals([tmp_path / "s.mcap"], q)
    assert "/inf v" in sig and "/half v" in sig, sig.left_out
    assert np.isnan(sig["/inf v"][10, 0]) and np.isnan(sig["/half v"][-5:]).all()
    assert [i["signal"] for i in sig.issues if i["kind"] == "signal_not_finite"] == ["/inf v"]
    # a table column with readings at a third of its rows is kept
    col = [np.nan] * n
    col[::3] = [1.0] * len(col[::3])
    got = f.recorded_signals(pd.DataFrame({"glove": col}), set(), n)
    assert "glove" in got and not got.left_out
    assert json.dumps(ctx)


def test_one_bad_value_never_drops_a_signal():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _one_bad_value_never_drops_a_signal(Path(t))


def _png(a) -> bytes:
    import io
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="PNG")
    return b.getvalue()


def _lerobot(root: Path, episodes: dict, feats: dict | None = None, n_video: int = 30) -> None:
    """A LeRobot v2.1 folder of one scene camera (n_video frames, none when 0) and, per episode index, its data table's
    columns (episodes {index: {column: cells}}), with frame_index, episode_index and timestamp added."""
    import json
    import numpy as np
    import pandas as pd
    info = {"codebase_version": "v2.1", "fps": 30, "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {"observation.images.cam_high": {"dtype": "video", "shape": [36, 64, 3]},
                         "observation.state": {"dtype": "float32", "shape": [14]}, **(feats or {})}}
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "data" / "chunk-000").mkdir(parents=True)
    for e, cols in episodes.items():
        m = len(next(iter(cols.values())))
        table = pd.DataFrame({**cols, "frame_index": np.arange(m), "episode_index": np.full(m, e),
                              "timestamp": np.arange(m) / 30})
        table.to_parquet(root / "data" / "chunk-000" / f"episode_{e:06d}.parquet")
        if n_video:
            d = root / "videos" / "chunk-000" / "observation.images.cam_high"
            d.mkdir(parents=True, exist_ok=True)
            _clip(d / f"episode_{e:06d}.mp4", n_video)


def _a_lerobot_episode_keeps_every_usable_value(tmp_path):
    """A LeRobot episode had lost a whole signal for one empty or odd sized cell or one undecodable pad image, every
    signal when its table was shorter than the video, and its state for one NaN frame (the episode was then told the
    dataset records no observation.state). Each is kept: a bad cell or image is NaN at its frame, a short table is NaN
    past its last row, a state with a short gap is filled and read, and each is a data issue."""
    import json
    import numpy as np
    rng = np.random.default_rng(3)
    root = tmp_path / "upload"
    state = np.cumsum(rng.normal(0, 0.01, (30, 14)), axis=0)
    state[5] = np.nan
    glove = [rng.random(16) for _ in range(30)]
    glove[10], glove[11] = None, rng.random(15)
    pad = [{"bytes": _png((rng.random((8, 8)) * 255).astype(np.uint8))} for _ in range(30)]
    pad[7] = {"bytes": b"not a png"}
    short = np.cumsum(rng.normal(0, 0.01, (10, 14)), axis=0)
    _lerobot(root, {0: {"observation.state": list(state), "glove": glove, "tactile.pad": pad},
                    1: {"observation.state": list(short), "glove": [rng.random(16) for _ in range(10)],
                        "tactile.pad": pad[:7]  + pad[8:11]}},
             feats={"tactile.pad": {"dtype": "image", "shape": [8, 8, 1]}})
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 2, rep
    ctx0 = _episode_ctx(tmp_path / "eps", rep, "000000")
    ep0 = tmp_path / "eps" / ctx0["episode_id"]
    assert ctx0["state_kind"] == "joints", ctx0.get("state_note")
    st = np.load(ep0 / "state.npz")["state"]
    assert st.shape == (30, 14) and np.isfinite(st).all()
    assert _issues(ctx0, "state_filled"), ctx0.get("reader_issues")
    z = np.load(ep0 / "signals.npz")
    sig = {s["name"]: z[s["key"]] for s in ctx0["signals"]}
    assert np.isnan(sig["glove"][[10, 11]]).all() and np.isfinite(np.delete(sig["glove"], [10, 11], 0)).all()
    assert np.isnan(sig["tactile.pad"][7]).all()
    assert np.isfinite(np.delete(sig["tactile.pad"], 7, 0)).all()
    assert {i["signal"] for i in _issues(ctx0, "signal_bad_cells")} == {"glove", "tactile.pad"}
    ctx1 = _episode_ctx(tmp_path / "eps", rep, "000001")
    ep1 = tmp_path / "eps" / ctx1["episode_id"]
    assert ctx1["state_kind"] == "none" and "records no observation.state" not in (ctx1.get("state_note") or "")
    assert "observation.state" in (ctx1.get("state_note") or ""), ctx1.get("state_note")
    z = np.load(ep1 / "signals.npz")
    sig = {s["name"]: z[s["key"]] for s in ctx1["signals"]}
    assert sig["observation.state"].shape == (30, 14) and np.isnan(sig["observation.state"][10:]).all()
    assert sig["glove"].shape == (30, 16) and np.isfinite(sig["glove"][:10]).all()
    assert _issues(ctx1, "table_short"), ctx1.get("reader_issues")
    assert json.dumps(ctx1)


def test_a_lerobot_episode_keeps_every_usable_value():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_lerobot_episode_keeps_every_usable_value(Path(t))


def _jpeg(shade: int, w: int = 64, h: int = 48) -> bytes:
    import io
    import numpy as np
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(np.full((h, w, 3), shade, np.uint8)).save(b, format="JPEG")
    return b.getvalue()


def _camera_mcap(path: Path, topics: list[str], n: int = 20, t0: float = 1_790_000_000.0) -> None:
    import base64
    import json
    from mcap.writer import Writer
    with open(path, "wb") as fh:
        w = Writer(fh)
        w.start()
        sid = w.register_schema(name="foxglove.CompressedImage", encoding="jsonschema", data=b"{}")
        ch = {t: w.register_channel(topic=t, message_encoding="json", schema_id=sid) for t in topics}
        for k in range(n):
            ns = int((t0 + k / 30) * 1e9)
            for t in topics:
                w.add_message(ch[t], log_time=ns, publish_time=ns, data=json.dumps(
                    {"format": "jpeg", "data": base64.b64encode(_jpeg(k * 10 % 256)).decode()}).encode())
        w.finish()


def _every_camera_that_holds_a_picture_is_read(tmp_path):
    """Cameras had been lost to rules that match too much or too little: a name with seg, conf or vis_ inside a word
    (segway_cam) was taken for a mask or depth view, an MCAP whose only camera is infrared failed, an HDF5 camera
    stored channel first, as floats or under 64 pixels was not a camera, and one LeRobot camera whose images in the
    data file do not decode failed the episode. Each is now read, and a camera that cannot be read is listed and is a
    data issue."""
    import json
    import h5py
    import numpy as np
    assert not f.not_rgb("/segway_cam/image") and not f.not_rgb("conference_room") and not f.not_rgb("visual_top")
    assert f.not_rgb("/camera/depth/image") and f.not_rgb("/thermal/image") and f.not_rgb("ir_cam")
    # a RealSense infra1 stream stays a camera the model is shown, as it was before whole words
    assert not f.not_rgb("/camera/infra1/image_rect_raw") and f.not_rgb("hand_mask")
    assert set(f.pick_cameras(["/segway_cam/image", "/top/image"], "teleop_arms")[0].values()) == {
        "/segway_cam/image", "/top/image"}
    root = tmp_path / "mcap"
    root.mkdir()
    _camera_mcap(root / "ir_only.mcap", ["/ir/image/compressed"])
    rep = f.convert(root, "ego_head", tmp_path / "eps_mcap", "test", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 1, rep
    ctx = _episode_ctx(tmp_path / "eps_mcap", rep, "ir_only")
    assert [i["camera"] for i in _issues(ctx, "camera_not_colour")] == ["/ir/image/compressed"], ctx
    root = tmp_path / "h5"
    root.mkdir()
    n = 12
    with h5py.File(root / "ep.h5", "w") as h:
        h["front"] = np.random.default_rng(0).integers(0, 255, (n, 3, 48, 48)).astype(np.uint8)
        h["wrist_left"] = np.random.default_rng(1).random((n, 40, 40, 3)).astype(np.float32)
        h["wrist_right"] = np.random.default_rng(2).integers(0, 255, (n, 32, 32, 3)).astype(np.uint8)
    assert f.h5_has_camera(root / "ep.h5")
    rep = f.convert(root, "teleop_arms", tmp_path / "eps_h5", "test", 900)
    assert not rep["failed"], rep
    ctx = _episode_ctx(tmp_path / "eps_h5", rep, "ep")
    assert {c["key"] for c in ctx["cameras"].values()} == {"front", "wrist_left", "wrist_right"}, ctx["cameras"]
    root = tmp_path / "lerobot"
    good = [{"bytes": _jpeg(k * 8, 96, 72)} for k in range(10)]
    bad = [{"bytes": b"not an image"} for _ in range(10)]
    _lerobot(root, {0: {"observation.state": [np.zeros(14)] * 10, "observation.images.top": good,
                        "observation.images.wrist_left": bad}},
             feats={"observation.images.top": {"dtype": "image", "shape": [72, 96, 3]},
                    "observation.images.wrist_left": {"dtype": "image", "shape": [72, 96, 3]},
                    "observation.images.cam_high": {"dtype": "image", "shape": [72, 96, 3]}}, n_video=0)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps_lr", "test", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 1, rep
    ctx = _episode_ctx(tmp_path / "eps_lr", rep, "000000")
    assert [c["key"] for c in ctx["cameras"].values()] == ["observation.images.top"], ctx["cameras"]
    assert [i["camera"] for i in _issues(ctx, "camera_not_decodable")] == ["observation.images.wrist_left"]
    assert any(u.startswith("observation.images.wrist_left") for u in ctx["source"]["unused_cameras"])
    assert json.dumps(ctx)


def test_every_camera_that_holds_a_picture_is_read():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _every_camera_that_holds_a_picture_is_read(Path(t))


def _lerobot_v3(root: Path, cams: dict, data: dict | None) -> None:
    """A LeRobot v3.0 folder without meta/episodes: one packed file per camera ({key: frames}) and, when given, one
    packed data file of these columns."""
    import json
    import pandas as pd
    feats = {k: {"dtype": "video", "shape": [36, 64, 3]} for k in cams}
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 30, "features": {
        **feats, "observation.state": {"dtype": "float32", "shape": [3]}}}))
    for k, n in cams.items():
        d = root / "videos" / k / "chunk-000"
        d.mkdir(parents=True)
        _clip(d / "file-000.mp4", n)
    if data is not None:
        (root / "data" / "chunk-000").mkdir(parents=True)
        pd.DataFrame(data).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")


def _lerobot_v3_without_its_episode_list_keeps_every_camera_it_can(tmp_path):
    """Without meta/episodes, an episode found in the data had been kept only when every camera's packed video placed
    it, a camera packed differently was dropped from whole recordings without a word on the episode, and a whole
    recording lost its data file's signals. An episode is kept with the cameras that place it, the others listed on
    it with the reason, and a recording keeps the signals of a data file with one row per frame."""
    import numpy as np
    rng = np.random.default_rng(4)
    top, wrist = "observation.images.top", "observation.images.wrist_left"
    lengths = [10, 12, 8]
    root = tmp_path / "placed"
    _lerobot_v3(root, {top: 30, wrist: 25}, {
        "episode_index": np.repeat([0, 1, 2], lengths), "frame_index": np.concatenate([np.arange(m) for m in lengths]),
        "timestamp": np.concatenate([np.arange(m) / 30 for m in lengths]),
        "observation.state": list(rng.random((30, 3)))})
    rep = f.convert(root, "teleop_arms", tmp_path / "eps_placed", "test", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 3, rep
    for e in rep["episodes"]:
        ctx = _episode_ctx(tmp_path / "eps_placed", rep, e["name"])
        assert [c["key"] for c in ctx["cameras"].values()] == [top], ctx["cameras"]
        assert any(u.startswith(wrist) and "lined up" in u for u in ctx["source"]["unused_cameras"]), ctx["source"]
        assert [i["camera"] for i in _issues(ctx, "camera_not_aligned")] == [wrist], ctx.get("reader_issues")
    root = tmp_path / "whole"
    _lerobot_v3(root, {top: 30, wrist: 25}, {"observation.state": list(rng.random((30, 3))),
                                              "timestamp": np.arange(30) / 30})
    rep = f.convert(root, "teleop_arms", tmp_path / "eps_whole", "test", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 1, rep
    ctx = _episode_ctx(tmp_path / "eps_whole", rep, "file-000")
    assert ctx.get("unsplit") and "observation.state" in [s["name"] for s in ctx.get("signals") or []], ctx
    assert any(u.startswith(wrist) for u in ctx["source"]["unused_cameras"]), ctx["source"]
    assert [i["camera"] for i in _issues(ctx, "camera_not_aligned")] == [wrist], ctx.get("reader_issues")


def test_lerobot_v3_without_its_episode_list_keeps_every_camera_it_can():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _lerobot_v3_without_its_episode_list_keeps_every_camera_it_can(Path(t))


def _a_lerobot_depth_stream_is_never_silently_left_out(tmp_path):
    """A LeRobot depth video that would not open, was stored in colour or had no frame in the episode's window had been
    skipped without a word. A depth video stored as an ordinary picture is shown as a camera of its own, as the video
    reader shows one, and one that cannot be read is listed with the reason and is a data issue."""
    import numpy as np
    root = tmp_path / "upload"
    depth = {"dtype": "video", "shape": [36, 64, 3], "info": {"video.is_depth_map": True}}
    _lerobot(root, {0: {"observation.state": [np.zeros(14)] * 30}},
             feats={"observation.images.top_depth": depth, "observation.depth.top": depth})
    d = root / "videos" / "chunk-000"
    (d / "observation.images.top_depth").mkdir()
    _clip(d / "observation.images.top_depth" / "episode_000000.mp4", 30)       # 8-bit colour, not distances
    (d / "observation.depth.top").mkdir()
    (d / "observation.depth.top" / "episode_000000.mp4").write_bytes(b"not a video")
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 1, rep
    ctx = _episode_ctx(tmp_path / "eps", rep, "000000")
    shown = {c["key"]: c for c in ctx["cameras"].values()}
    assert "ordinary picture" in (shown.get("observation.images.top_depth") or {}).get("desc", ""), shown
    assert any(u.startswith("observation.depth.top (") for u in ctx["source"]["unused_cameras"]), ctx["source"]
    assert [i["camera"] for i in _issues(ctx, "depth_not_read")] == ["observation.depth.top"], ctx.get("reader_issues")


def test_a_lerobot_depth_stream_is_never_silently_left_out():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_lerobot_depth_stream_is_never_silently_left_out(Path(t))


def _a_table_over_the_size_limit_is_read(tmp_path):
    """A CSV table over TABLE_MAX_BYTES beside an episode's videos had been ignored, as signals and as notes. It is read
    in chunks, and past TABLE_STREAM_MAX_ROWS rows every so many rows are kept (still finer than the frames), which is
    a data issue; its rows that hold text are read as notes."""
    import numpy as np
    root = tmp_path / "upload"
    root.mkdir()
    _clip(root / "top.mp4", 30)
    t = np.arange(3000) / 1000
    lines = ["time,force_x,force_y,note"] + [f"{x:.4f},{np.sin(x):.5f},{np.cos(x):.5f},{'grasp' if k == 7 else ''}"
                                             for k, x in enumerate(t)]
    (root / "traj.csv").write_text("\n".join(lines) + "\n")
    saved = f.TABLE_MAX_BYTES, f.TABLE_STREAM_MAX_ROWS
    f.TABLE_MAX_BYTES, f.TABLE_STREAM_MAX_ROWS = 1000, 400
    try:
        rep = f.convert(root, "ego_head", tmp_path / "eps", "test", 900)
        tables = dict(f.annotation_tables(root))
    finally:
        f.TABLE_MAX_BYTES, f.TABLE_STREAM_MAX_ROWS = saved
    assert not rep["failed"], rep
    ctx = _episode_ctx(tmp_path / "eps", rep, "top")
    sig = {s["name"]: s for s in ctx.get("signals") or []}
    assert "traj" in sig and sig["traj"]["names"] == ["force_x", "force_y"], ctx.get("signals")
    assert _issues(ctx, "table_downsampled"), ctx.get("reader_issues")
    assert [r["note"] for r in tables.get("traj.csv") or []] == ["grasp"], tables


def test_a_table_over_the_size_limit_is_read():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_table_over_the_size_limit_is_read(Path(t))


def _a_signal_is_never_dropped_for_its_width_its_rate_or_its_size(tmp_path):
    """Messages of another width than a field's first had been dropped (counted only when the field was left out
    anyway), a channel slower than 1 Hz was left out as a setting, and an array wider than SIGNAL_MAX_VALUES was left
    out. Each width is now its own signal, a slow channel is kept with its rate, a single message is listed with its
    values, and a wide array is kept as a map."""
    import h5py
    import numpy as np
    import pandas as pd
    t0 = 1_790_000_000.0
    q = t0 + np.arange(150) / 30                                 # 5 s of footage
    _json_mcap(tmp_path / "s.mcap", {
        "/mixed": [(k / 30, {"v": [float(np.sin(k / 9))] * (5 if k % 10 == 0 else 3)}) for k in range(150)],
        "/slow": [(s, {"battery": 12.0 - 0.1 * s}) for s in (0.2, 2.2, 4.2)],
        "/once": [(1.0, {"gain": 3.5})]}, t0)
    sig = f.mcap_signals([tmp_path / "s.mcap"], q)
    assert "/mixed v (3 values)" in sig and "/mixed v (5 values)" in sig, (list(sig), sig.left_out)
    assert "/slow" in sig and sig.meta["/slow"]["rate_hz"] < 1, (list(sig), sig.left_out)
    assert any(k.startswith("/once") and "3.5" in k + why for k, why in sig.left_out), sig.left_out
    n = 6
    wide = f.recorded_signals(pd.DataFrame({"cloud": [np.arange(f.SIGNAL_MAX_VALUES + 4.0)] * n}), set(), n)
    assert wide["cloud"].shape == (n, f.SIGNAL_MAX_VALUES + 4) and not wide.left_out
    with h5py.File(tmp_path / "w.h5", "w") as h:
        h["cloud"] = np.zeros((n, f.SIGNAL_MAX_VALUES + 4), np.float32)
        assert f.h5_kind("cloud", h["cloud"]) == "signal"


def _an_episode_longer_than_its_header_is_trimmed_to_the_cap(tmp_path):
    """An episode whose measured length passed the footage cap, though its header said it fit, had been deleted. Its
    first minutes up to the cap are labelled, as the first episode's are."""
    root = tmp_path / "upload"
    root.mkdir()
    _clip(root / "a.mp4", 90)
    _clip(root / "b.mp4", 90)
    real = f._duration
    f._duration = lambda p: 0.3 if Path(p).name == "b.mp4" else real(p)
    try:
        rep = f.convert(root, "ego_head", tmp_path / "eps", "test", 4.5)
    finally:
        f._duration = real
    assert [e["name"] for e in rep["episodes"]] == ["a", "b"], rep
    assert abs(rep["episodes"][1]["seconds"] - 1.5) < 0.1 and abs(rep["seconds"] - 4.5) < 0.1, rep["episodes"]
    assert any(u.startswith("b is ") for u in rep["used"]), rep["used"]


def test_a_signal_is_never_dropped_for_its_width_its_rate_or_its_size():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_signal_is_never_dropped_for_its_width_its_rate_or_its_size(Path(t))


def test_an_episode_longer_than_its_header_is_trimmed_to_the_cap():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _an_episode_longer_than_its_header_is_trimmed_to_the_cap(Path(t))


def _every_camera_reaches_the_board_even_when_the_model_is_not_shown_it(tmp_path):
    """Cameras the model is not shown (the second eye of a stereo camera, an infrared video, every camera but one on a
    head rig, more extra cameras than the model takes) had no footage on the board at all. Each is written to the
    episode's context.json unshown_cameras with its file, its frames and the reason, so the board can play it named as
    not shown to the model; an MCAP or HDF5 camera is written to a video of its own for it."""
    import json
    root = tmp_path / "videos" / "take1"
    root.mkdir(parents=True)
    for cam in ("zed_left", "zed_right", "zed_ir"):
        _clip(root / f"{cam}.mp4", 20)
    rep = f.convert(tmp_path / "videos", "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 1, rep
    ctx = _episode_ctx(tmp_path / "eps", rep, "take1")
    un = {u["name"]: u for u in ctx.get("unshown_cameras") or []}
    assert set(un) == {"zed_right", "zed_ir"}, ctx.get("unshown_cameras")
    assert "stereo" in un["zed_right"]["why"] and "colour" in un["zed_ir"]["why"], un
    assert all(Path(u["packed"]).exists() and u["n_frames"] == 20 for u in un.values()), un
    root = tmp_path / "mcap"
    root.mkdir()
    _camera_mcap(root / "head.mcap", ["/cam_a/image/compressed", "/cam_b/image/compressed"])
    rep = f.convert(root, "ego_head", tmp_path / "eps_mcap", "test", 900)
    assert not rep["failed"], rep
    ctx = _episode_ctx(tmp_path / "eps_mcap", rep, "head")
    un = ctx.get("unshown_cameras") or []
    assert len(un) == 1 and "one head camera" in un[0]["why"], un
    assert Path(un[0]["packed"]).exists() and un[0]["n_frames"] == 20 and abs(un[0]["start_s"]) < 1e-6, un
    assert json.dumps(ctx)


def test_every_camera_reaches_the_board_even_when_the_model_is_not_shown_it():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _every_camera_reaches_the_board_even_when_the_model_is_not_shown_it(Path(t))


def _an_upload_or_episode_with_no_camera_names_every_file_and_why(tmp_path):
    """The board shows an episode beside its footage, so an episode with no camera cannot be shown yet. A sensor only
    upload had been refused with a sentence that named none of its files, and a LeRobot episode with a data file but
    no video was dropped without a word. The refusal names every file and why, and the report names each such
    episode's data file."""
    import numpy as np
    root = tmp_path / "sensors"
    root.mkdir()
    _glove_h5(root / "glove.h5", 30)
    _json_mcap(root / "imu.mcap", {"/imu": [(k / 30, {"accel": [0.0, 0.0, 9.8]}) for k in range(30)]},
               1_790_000_000.0)
    (root / "notes.txt").write_text("left glove")
    try:
        f.detect(root)
    except ValueError as e:
        msg = str(e)
    else:
        raise AssertionError("a sensor only upload was accepted")
    assert all(name in msg for name in ("glove.h5", "imu.mcap", "notes.txt")), msg
    assert "no camera" in msg and "video" in msg, msg
    root = tmp_path / "lerobot"
    _lerobot(root, {0: {"observation.state": [np.zeros(14)] * 30}, 1: {"observation.state": [np.zeros(14)] * 30}})
    (root / "videos" / "chunk-000" / "observation.images.cam_high" / "episode_000001.mp4").unlink()
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert len(rep["episodes"]) == 1, rep
    assert any("episode_000001.parquet" in m and "no video" in m for m in rep["missing"]), rep["missing"]


def test_an_upload_or_episode_with_no_camera_names_every_file_and_why():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _an_upload_or_episode_with_no_camera_names_every_file_and_why(Path(t))


def _a_sensor_file_in_a_folder_of_several_episodes_of_any_format_joins_none_by_guess(tmp_path):
    """A folder with run.mp4, cam.h5 and glove.h5 had the glove joined to both episodes, since each format counted
    only its own episodes, and the HDF5 episode read its qpos as the arm state on an assumed start. A folder's
    episodes are counted across every format: the glove names no take, neither episode has a recorder clock to
    place it by, so it is listed on each with the reason and read as neither's state."""
    import h5py
    import numpy as np
    root = tmp_path / "upload"
    (root / "a").mkdir(parents=True)
    _clip(root / "a" / "run.mp4", 30)
    with h5py.File(root / "a" / "cam.h5", "w") as h:
        h["cam"] = np.random.default_rng(0).integers(0, 255, (30, 96, 96, 3)).astype(np.uint8)
    with h5py.File(root / "a" / "glove.h5", "w") as h:
        th = np.arange(30) / 30
        h["time"] = th
        h["qpos"] = np.stack([0.3 * np.sin(th * 3 + j) for j in range(14)], axis=1)
    det, items = f.plan(root)
    assert all(not it["state"] and [Path(p).name for p in it["state_shared"]] == ["glove.h5"] for it in items), items
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 2, rep
    for e in rep["episodes"]:
        ctx = _episode_ctx(tmp_path / "eps", rep, e["name"])
        assert ctx["state_kind"] == "none" and not ctx.get("signals"), (e["name"], ctx.get("signals"))
        assert any(u.startswith("glove.h5") for u in ctx["source"]["unused_signals"]), ctx["source"]
    assert any("glove.h5" in m and "listed" in m for m in rep["used"] + rep["missing"]), rep


def _a_shared_sensor_file_is_placed_by_clock_on_every_episode_of_any_format(tmp_path):
    """Two HDF5 camera episodes and a glove file whose name gives no take, in one folder: the glove had been
    silently unread while the report said it was placed. Both episodes have a recorder clock, so the glove is placed
    by its clock on each, and the report says it was placed on both."""
    import h5py
    import numpy as np
    root = tmp_path / "upload"
    root.mkdir()
    for k in (1, 2):
        with h5py.File(root / f"run{k}.h5", "w") as h:
            h["cam"] = np.random.default_rng(k).integers(0, 255, (30, 96, 96, 3)).astype(np.uint8)
            h["timestamps"] = 1_790_000_000.0 + 10 * k + np.arange(30) / 30
    with h5py.File(root / "glove.h5", "w") as h:
        h["time"] = 1_790_000_000.0 + np.arange(900) / 30
        h["pressure"] = np.arange(900, dtype=np.float64)[:, None] * np.ones((1, 4))
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 2, rep
    for k in (1, 2):
        ctx = _episode_ctx(tmp_path / "eps", rep, f"run{k}")
        z = np.load(tmp_path / "eps" / ctx["episode_id"] / "signals.npz")
        s = next(s for s in ctx.get("signals") or [] if s["name"] == "pressure")
        assert abs(float(z[s["key"]][0, 0]) - 300 * k) < 1.5, (k, z[s["key"]][:3, 0])
        assert not s.get("aligned_by"), s
    assert any("glove.h5" in u and "placed by its own clock on 2 episodes" in u for u in rep["used"]), rep["used"]


def _every_mcap_camera_is_named_and_the_models_choice_is_kept(tmp_path):
    """The whole word rule for cameras that are not colour took a RealSense infra1 stream for infrared and dropped it
    before cameras were chosen, so a camera the model had been shown was nowhere; a thermal or mask channel was
    never listed at all. infra1 is shown as before, and a thermal camera the model is not shown is listed among the
    unused cameras and written for the board."""
    root = tmp_path / "upload"
    root.mkdir()
    _camera_mcap(root / "rs.mcap", ["/realsense/color/image_raw/compressed", "/realsense/infra1/image_rect_raw",
                                    "/thermal/image/compressed"])
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["failed"], rep
    ctx = _episode_ctx(tmp_path / "eps", rep, "rs")
    keys = {c["key"] for c in ctx["cameras"].values()}
    assert "/realsense/infra1/image_rect_raw" in keys and "/thermal/image/compressed" not in keys, keys
    assert "/thermal/image/compressed" in ctx["source"]["unused_cameras"], ctx["source"]
    assert [u["name"] for u in ctx.get("unshown_cameras") or []] == ["/thermal/image/compressed"], ctx.get(
        "unshown_cameras")


def test_a_sensor_file_in_a_folder_of_several_episodes_of_any_format_joins_none_by_guess():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_sensor_file_in_a_folder_of_several_episodes_of_any_format_joins_none_by_guess(Path(t))


def test_a_shared_sensor_file_is_placed_by_clock_on_every_episode_of_any_format():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_shared_sensor_file_is_placed_by_clock_on_every_episode_of_any_format(Path(t))


def test_every_mcap_camera_is_named_and_the_models_choice_is_kept():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _every_mcap_camera_is_named_and_the_models_choice_is_kept(Path(t))


def test_a_streamed_table_keeps_its_epoch_times_and_rows_sit_on_their_frame_index():
    """A table read in chunks had every column cast to float32, which collapsed epoch seconds at 100 Hz into a few
    distinct times, and a table whose frame_index starts at 3 was moved to start at frame 0. A time column stays
    float64, and a row sits at the frame its frame_index gives."""
    import numpy as np
    import pandas as pd
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "traj.csv"
        n = 3000
        ts = 1_790_000_000.0 + np.arange(n) / 100.0
        pd.DataFrame({"timestamp": ts, "x": np.sin(np.arange(n) / 50)}).to_csv(p, index=False)
        saved = f.TABLE_MAX_BYTES
        f.TABLE_MAX_BYTES = 10
        try:
            big, _, _ = f.read_number_table(p)
        finally:
            f.TABLE_MAX_BYTES = saved
    assert big["timestamp"].dtype == np.float64 and big["timestamp"].nunique() == n
    df = pd.DataFrame({"frame_index": np.arange(3, 10), "force": np.arange(3, 10, dtype=float)})
    out = f.recorded_signals(df, set(), 10)
    assert np.flatnonzero(np.isfinite(out["force"][:, 0])).tolist() == list(range(3, 10))
    assert out["force"][3, 0] == 3.0
    st = np.stack([np.arange(3, 10, dtype=float)] * 14, axis=1)
    state, _, kind, note, _ = f.state_on_frames(df, st, None, 10, 30.0, "joints", None)
    assert kind == "joints" and state[3, 0] == 3.0 and state[9, 0] == 9.0, (kind, note)


def _a_table_placed_from_both_starts_is_marked_signal_by_signal(tmp_path):
    """A table beside videos with no capture times is placed from both starts. It had no data issue, a second table
    overwrote the first's note, and nothing told the model the alignment was assumed. Each such signal is a data
    issue naming it, carries aligned_by in its meta, and its line in the prompt says so."""
    import numpy as np
    from label import episode as me
    root = tmp_path / "upload"
    root.mkdir()
    _clip(root / "top.mp4", 30)
    for name in ("traj", "grip"):
        t = np.arange(50) / 25
        (root / f"{name}.csv").write_text("time,v\n" + "\n".join(f"{x:.3f},{np.sin(x + len(name)):.4f}"
                                                                  for x in t) + "\n")
    rep = f.convert(root, "ego_head", tmp_path / "eps", "test", 900)
    assert not rep["failed"], rep
    ctx = _episode_ctx(tmp_path / "eps", rep, "top")
    sig = {s["name"]: s for s in ctx["signals"]}
    assert set(sig) == {"traj", "grip"} and all(s.get("aligned_by") == "assumed start" for s in sig.values()), sig
    assert {i["signal"] for i in _issues(ctx, "signal_alignment_assumed")} == {"traj", "grip"}, ctx["reader_issues"]
    ep = me.load(tmp_path / "eps" / ctx["episode_id"])
    text = me.build_prompt(ep, me.plan(ep), cell_w=448, cell_h=252)[1]
    lines = [x for x in str(text).splitlines() if x.strip().startswith(("traj (", "grip ("))]
    assert len(lines) == 2 and all("both starts" in x for x in lines), lines


def test_a_table_placed_from_both_starts_is_marked_signal_by_signal():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_table_placed_from_both_starts_is_marked_signal_by_signal(Path(t))


def _lerobot_image_cameras_the_model_is_not_shown_reach_the_board(tmp_path):
    """A LeRobot episode whose cameras are images in its data file wrote no clip for a camera the model is not shown
    (a mask). It is written to a video of its own and listed in unshown_cameras."""
    import numpy as np
    root = tmp_path / "upload"
    good = [{"bytes": _jpeg(k * 8, 96, 72)} for k in range(10)]
    feat = {"dtype": "image", "shape": [72, 96, 3]}
    _lerobot(root, {0: {"observation.state": [np.zeros(14)] * 10, "observation.images.top": good,
                        "observation.images.top_mask": good}},
             feats={"observation.images.top": feat, "observation.images.top_mask": feat,
                    "observation.images.cam_high": feat}, n_video=0)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["failed"], rep
    ctx = _episode_ctx(tmp_path / "eps", rep, "000000")
    un = ctx.get("unshown_cameras") or []
    assert [u["name"] for u in un] == ["observation.images.top_mask"] and un[0]["n_frames"] == 10, un
    assert Path(un[0]["packed"]).exists()


def test_lerobot_image_cameras_the_model_is_not_shown_reach_the_board():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _lerobot_image_cameras_the_model_is_not_shown_reach_the_board(Path(t))


def test_messages_of_another_width_on_an_arm_channel_read_as_the_state_are_listed():
    """An arm channel read as the state whose messages sometimes carry another number of values had those messages
    dropped without a word, since the field read as the state is not a signal. They are listed with the reason."""
    import numpy as np
    with tempfile.TemporaryDirectory() as t:
        t0 = 1_790_000_000.0
        tt = np.arange(-0.1, 3.1, 0.01)
        extra = lambda k: [0.0, 0.0] if k % 50 == 7 else []
        msgs = lambda: [(s, {"joint_pos": [0.1 * np.sin(s + j) for j in range(7)] + extra(k)})
                        for k, s in enumerate(tt)]
        path = Path(t) / "arms.mcap"
        _json_mcap(path, {"/left/joint_state": msgs(), "/right/joint_state": msgs()}, t0)
        q = t0 + np.arange(0, 3, 1 / 30)
        st = f.mcap_joint_streams([path], q)
        state, action, note = f.joint_state(st, q)
        assert state is not None and note is None
        left = f.joint_left_out(st, state, action)
        assert sorted(k for k, _ in left) == ["/left/joint_state joint_pos", "/right/joint_state joint_pos"], left
        assert all("9 values" in why for _, why in left), left


def _signals_over_the_episode_budget_are_kept_as_summaries(tmp_path):
    """A wide array was read as float64 several times over (a 480 MB signal took 5.9 GB). Arrays are read in their
    own float32, and the signals of an episode past SIGNAL_EPISODE_BYTES keep their widest as a summary per frame
    (the lowest, mean and highest value), a data issue, so an episode's signals fit in memory."""
    import h5py
    import numpy as np
    root = tmp_path / "upload"
    root.mkdir()
    n = 40
    rng = np.random.default_rng(5)
    with h5py.File(root / "ep.h5", "w") as h:
        h["cam"] = rng.integers(0, 255, (n, 64, 64, 3)).astype(np.uint8)
        h["timestamps"] = 1_790_000_000.0 + np.arange(n) / 30
        h["cloud"] = rng.random((n, 6000)).astype(np.float32)
        h["force"] = rng.random((n, 3))
    saved = f.SIGNAL_EPISODE_BYTES
    f.SIGNAL_EPISODE_BYTES = 100_000
    try:
        rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    finally:
        f.SIGNAL_EPISODE_BYTES = saved
    assert not rep["failed"], rep
    ctx = _episode_ctx(tmp_path / "eps", rep, "ep")
    sig = {s["name"]: s for s in ctx["signals"]}
    assert sig["force"]["dims"] == 3 and sig["cloud"]["dims"] == 3 and sig["cloud"]["summary_of"] == 6000, sig
    assert sig["cloud"]["names"] == ["lowest", "mean", "highest"]
    assert [i["signal"] for i in _issues(ctx, "signal_summarised")] == ["cloud"], ctx.get("reader_issues")
    z = np.load(tmp_path / "eps" / ctx["episode_id"] / "signals.npz")
    c = z[sig["cloud"]["key"]]
    assert c.shape == (n, 3) and (c[:, 0] <= c[:, 1]).all() and (c[:, 1] <= c[:, 2]).all()


def test_signals_over_the_episode_budget_are_kept_as_summaries():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _signals_over_the_episode_budget_are_kept_as_summaries(Path(t))


def test_a_short_gap_at_the_start_of_a_signal_is_reported():
    import numpy as np
    a = np.ones((60, 2))
    a[:3] = np.nan
    got = f.signal_gaps("force", a, np.arange(60) / 30)
    assert got and got[0]["t0_s"] == 0.0 and "3 of its 60 frames" in got[0]["what"], got
    assert "0.07 s" in got[0]["what"], got


def test_a_short_channel_whose_name_says_time_is_a_clock_not_a_reading():
    """A slow sensor's own stamp (/d360_0/ght_topic ts, at 1.7 Hz) has too few messages to be judged a counter, and
    was shown to the model as a reading. A value whose name says time and that rises is a clock however few its
    messages."""
    import numpy as np
    import pandas as pd
    with tempfile.TemporaryDirectory() as t:
        t0 = 1_790_000_000.0
        path = Path(t) / "ght.mcap"
        _json_mcap(path, {"/ght": [(s, {"ts": 6.48e6 + 600 * k, "sensor_ts": 4.14e8 + 600 * k,
                                         "temperature": 72.0 + 0.1 * k}) for k, s in enumerate(np.arange(0, 5, 0.6))]},
                   t0)
        sig = f.mcap_signals([path], t0 + np.arange(0, 5, 1 / 30))
    assert list(sig) == ["/ght"] and sig.meta["/ght"]["names"] == ["temperature"], (list(sig), sig.meta)
    out = f.recorded_signals(pd.DataFrame({"recv_time": [1.79e9 + k for k in range(8)],
                                           "v": np.arange(8.0)}), set(), 8)
    assert "recv_time" in out.clocks and "recv_time" not in out


def test_a_float_picture_is_judged_on_more_than_its_first_frame():
    """A float array whose first frame lies within 0 to 1 but whose later frames reach 1000 is not a picture."""
    import h5py
    import numpy as np
    with tempfile.TemporaryDirectory() as t:
        with h5py.File(Path(t) / "x.h5", "w") as h:
            a = np.random.default_rng(0).random((20, 32, 32, 3)).astype(np.float32)
            a[5:] *= 1000
            h["flow"] = a
            b = np.random.default_rng(1).random((20, 32, 32, 3)).astype(np.float32)
            b[0] = 0.0
            h["cam"] = b
            assert f.h5_kind("flow", h["flow"]) != "camera"
            assert f.h5_kind("cam", h["cam"]) == "camera" and f.picture_scale(h["cam"]) == 255.0


def _trimming_an_episode_drops_the_issues_past_its_new_end(tmp_path):
    root = tmp_path / "upload"
    root.mkdir()
    _clip(root / "a.mp4", 90)
    rep = f.convert(root, "ego_head", tmp_path / "eps", "test", 900)
    ep = tmp_path / "eps" / rep["episodes"][0]["episode_id"]
    import json
    ctx = json.loads((ep / "context.json").read_text())
    ctx["reader_issues"] = [{"kind": "signal_gap", "what": "early", "t0_s": 0.2, "t1_s": 2.5},
                            {"kind": "signal_gap", "what": "late", "t0_s": 2.0, "t1_s": 2.5},
                            {"kind": "table_short", "what": "untimed"}]
    (ep / "context.json").write_text(json.dumps(ctx))
    ctx = f.trim_episode(ep, 1.5)
    assert [i["what"] for i in ctx["reader_issues"]] == ["early", "untimed"], ctx["reader_issues"]
    assert ctx["reader_issues"][0]["t1_s"] <= 1.5


def test_trimming_an_episode_drops_the_issues_past_its_new_end():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _trimming_an_episode_drops_the_issues_past_its_new_end(Path(t))


def _unread_files_are_exactly_those_no_reader_opened(tmp_path):
    """A notes file no reader opens (a YAML file, a text file beside no episode) had been left off the list of files
    not read, by its extension alone, while a video's notes beside it are read."""
    root = tmp_path / "upload"
    (root / "a").mkdir(parents=True)
    _clip(root / "a" / "take.mp4", 10)
    (root / "a" / "take.txt").write_text("pick up the cube")
    (root / "a" / "calib.yaml").write_text("k: 1")
    (root / "misc").mkdir()
    (root / "misc" / "readme.txt").write_text("hello")
    det, _ = f.plan(root)
    line = next((m for m in det["missing"] if "no reader opens" in m), "")
    assert "a/calib.yaml" in line and "misc/readme.txt" in line and "take.txt" not in line, det["missing"]


def test_unread_files_are_exactly_those_no_reader_opened():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _unread_files_are_exactly_those_no_reader_opened(Path(t))


def _every_note_file_of_an_episode_is_read_and_named(tmp_path):
    """An episode folder with two cameras and a note beside each (top.txt, wrist.md) had neither note read, and one
    with annotations.json, notes.txt and a recorder's operator.json had only annotations.json read, while the list of
    files no reader opens named none of them. Every note file is read, each under its file name, a camera's own
    note is never taken for the episode's task, and a .json of the folder that gives neither the task nor a depth
    scale is named as not read."""
    import json
    root = tmp_path / "upload"
    ep1 = root / "ep1"
    ep1.mkdir(parents=True)
    _clip(ep1 / "top.mp4", 10)
    _clip(ep1 / "wrist.mp4", 10)
    (ep1 / "top.txt").write_text("the operator dropped the cup at 0.5 s")
    (ep1 / "wrist.md").write_text("wrist camera loose")
    ep2 = root / "ep2"
    ep2.mkdir()
    _clip(ep2 / "top.mp4", 10)
    _clip(ep2 / "wrist.mp4", 10)
    (ep2 / "annotations.json").write_text(json.dumps({"task": "pick the cup"}))
    (ep2 / "notes.txt").write_text("wrist camera came loose at the end")
    (ep2 / "operator.json").write_text(json.dumps({"operator_comment": "second attempt"}))
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = {}
    for e in rep["episodes"]:
        c = json.loads((tmp_path / "eps" / e["episode_id"] / "context.json").read_text())
        ctx[c["source"]["upload"]] = c
    assert ctx["ep1"]["uploader_notes"] == {"top.txt": "the operator dropped the cup at 0.5 s",
                                            "wrist.md": "wrist camera loose"}, ctx["ep1"]
    assert "instruction" not in ctx["ep1"]
    assert ctx["ep2"]["instruction"] == "pick the cup"
    assert ctx["ep2"]["uploader_notes"] == {"annotations.json": {"task": "pick the cup"},
                                            "notes.txt": "wrist camera came loose at the end"}, ctx["ep2"]
    line = next((m for m in rep["missing"] if "no reader opens" in m), "")
    assert "ep2/operator.json" in line, rep["missing"]
    assert not any(x in line for x in ("top.txt", "wrist.md", "annotations.json", "notes.txt")), line


def test_every_note_file_of_an_episode_is_read_and_named():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _every_note_file_of_an_episode_is_read_and_named(Path(t))


def _one_note_file_is_read_as_it_was_sent(tmp_path):
    """One note file stays its own text, never put under its file name (a note of one line is the task, as before),
    and a recorder's metadata that gives the task is read."""
    import json
    root = tmp_path / "upload"
    ep1 = root / "ep1"
    ep1.mkdir(parents=True)
    _clip(ep1 / "top.mp4", 10)
    _clip(ep1 / "wrist.mp4", 10)
    (ep1 / "notes.txt").write_text("wrist camera came loose\nat the end")
    (ep1 / "session_meta.json").write_text(json.dumps({"prompt": "Pick up the cube"}))
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = json.loads((tmp_path / "eps" / rep["episodes"][0]["episode_id"] / "context.json").read_text())
    assert ctx["uploader_notes"] == "wrist camera came loose\nat the end"
    assert ctx["uploader_annotation"] == "wrist camera came loose\nat the end"
    assert ctx["instruction"] == "Pick up the cube"
    assert not any("no reader opens" in m for m in rep["missing"]), rep["missing"]


def test_one_note_file_is_read_as_it_was_sent():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _one_note_file_is_read_as_it_was_sent(Path(t))


def _notes_case(tmp_path, files: dict, cams=("top", "wrist")):
    """The context of one episode folder ep1 with these cameras and note files ({name: text or JSON object})."""
    import json
    root = tmp_path / "upload"
    ep1 = root / "ep1"
    ep1.mkdir(parents=True)
    for c in cams:
        _clip(ep1 / f"{c}.mp4", 10)
    for k, v in files.items():
        (ep1 / k).write_text(v if isinstance(v, str) else json.dumps(v))
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    return json.loads((tmp_path / "eps" / rep["episodes"][0]["episode_id"] / "context.json").read_text()), rep


def _a_cameras_own_json_never_gives_the_task(tmp_path):
    """top.json beside top.mp4 and wrist.mp4 is about the top camera: its task key had been taken for the episode's
    task by the search of the folder's other .json files. It stays that camera's note."""
    import json
    ctx, _ = _notes_case(tmp_path, {"top.json": {"task": "calibrate the top camera"}})
    assert "instruction" not in ctx, ctx.get("instruction")
    assert "calibrate the top camera" in json.dumps(ctx["uploader_notes"])


def test_a_cameras_own_json_never_gives_the_task():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_cameras_own_json_never_gives_the_task(Path(t))


def _only_a_task_file_gives_the_task(tmp_path):
    """notes.txt of one line ("camera bumped at 3 s") had become the episode's task, the task text the uploader sent,
    and was dropped from the notes. A plain text file gives the task only when it is named for a task (instruction.txt,
    task.txt) or for the episode (ep1.txt, or the own .txt of an episode's only video); every other one is a note under
    its name. A JSON gives the task only through its task keys."""
    ctx, _ = _notes_case(tmp_path / "b", {"meta.json": {"operator": "A"}, "notes.txt": "camera bumped at 3 s"})
    assert "instruction" not in ctx
    assert ctx["uploader_notes"] == {"meta.json": {"operator": "A"}, "notes.txt": "camera bumped at 3 s"}
    ctx, _ = _notes_case(tmp_path / "n", {"notes.txt": "camera bumped at 3 s"})
    assert "instruction" not in ctx and ctx["uploader_notes"] == "camera bumped at 3 s"
    ctx, _ = _notes_case(tmp_path / "m", {"annotations.json": "pick the cup"})
    assert "instruction" not in ctx and ctx["uploader_notes"] == "pick the cup"
    ctx, _ = _notes_case(tmp_path / "f", {"instruction.txt": "pick the cup", "notes.txt": "dropped it once"})
    assert ctx["instruction"] == "pick the cup" and ctx["uploader_notes"] == {"notes.txt": "dropped it once"}
    ctx, _ = _notes_case(tmp_path / "e", {"ep1.txt": "pick the cup"})
    assert ctx["instruction"] == "pick the cup" and "uploader_notes" not in ctx
    ctx, _ = _notes_case(tmp_path / "g", {"top.txt": "pick the cup"}, cams=("top",))
    assert ctx["instruction"] == "pick the cup" and "uploader_notes" not in ctx


def test_only_a_task_file_gives_the_task():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _only_a_task_file_gives_the_task(Path(t))


def _a_cameras_own_note_is_given_under_its_file_name(tmp_path):
    """top.txt alone beside top.mp4 and wrist.mp4 had been given as bare text, so neither the model nor the board knew
    it is about the top camera. It is given under its file name; a single note of the episode stays bare."""
    ctx, _ = _notes_case(tmp_path / "a", {"top.txt": "operator bumped the top camera at 3 s"})
    assert ctx["uploader_notes"] == {"top.txt": "operator bumped the top camera at 3 s"}
    assert ctx["uploader_annotation"] == '{"top.txt": "operator bumped the top camera at 3 s"}'
    ctx, _ = _notes_case(tmp_path / "n", {"notes.txt": "camera bumped\nat 3 s"})
    assert ctx["uploader_notes"] == "camera bumped\nat 3 s"


def test_a_cameras_own_note_is_given_under_its_file_name():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_cameras_own_note_is_given_under_its_file_name(Path(t))


def _upload_notes(tmp_path, videos, files: dict):
    """({upload name: context}, report) of an upload of these videos and note files ({path: text, bytes or JSON})."""
    import json
    root = tmp_path / "upload"
    for v in videos:
        (root / v).parent.mkdir(parents=True, exist_ok=True)
        _clip(root / v, 10)
    for k, v in files.items():
        (root / k).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(v, bytes):
            (root / k).write_bytes(v)
        else:
            (root / k).write_text(v if isinstance(v, str) else json.dumps(v))
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = {}
    for e in rep["episodes"]:
        c = json.loads((tmp_path / "eps" / e["episode_id"] / "context.json").read_text())
        ctx[c["source"]["upload"]] = c
    return ctx, rep


def _unread_line(rep) -> str:
    return next((m for m in rep["missing"] if "no reader opens" in m), "")


def _another_episodes_note_never_gives_an_episode_its_task(tmp_path):
    """Two takes in one folder (top_ep1, wrist_ep1, top_ep2, wrist_ep2) and a note of the second take (ep2.json, or its
    top camera's top_ep2.json) had given the first take that task, through the search of the folder's .json files;
    so had ep2.json at the root of camera folders (top/ep1.mp4, top/ep2.mp4). A folder's .json gives an episode its task
    only for a file named for no video and no episode of the folder, which the folder's episodes share."""
    takes = ["d/top_ep1.mp4", "d/wrist_ep1.mp4", "d/top_ep2.mp4", "d/wrist_ep2.mp4"]
    ctx, _ = _upload_notes(tmp_path / "a", takes, {"d/ep2.json": {"task": "pour the tea"}})
    assert "instruction" not in ctx["d/ep1"] and ctx["d/ep2"]["instruction"] == "pour the tea"
    ctx, _ = _upload_notes(tmp_path / "b", takes, {"d/top_ep2.json": {"task": "pour the tea"}})
    assert "instruction" not in ctx["d/ep1"] and "instruction" not in ctx["d/ep2"]
    assert ctx["d/ep2"]["uploader_notes"] == {"top_ep2.json": {"task": "pour the tea"}}
    folders = ["top/ep1.mp4", "wrist/ep1.mp4", "top/ep2.mp4", "wrist/ep2.mp4"]
    ctx, _ = _upload_notes(tmp_path / "c", folders, {"ep2.json": {"task": "pour the tea"}})
    assert "instruction" not in ctx["ep1"] and ctx["ep2"]["instruction"] == "pour the tea"
    # a recorder's file named for no video and no episode of the folder is shared by the folder: every take's task
    ctx, rep = _upload_notes(tmp_path / "d", takes, {"d/session_meta.json": {"prompt": "pour the tea"}})
    assert [c.get("instruction") for c in ctx.values()] == ["pour the tea", "pour the tea"], ctx
    assert "session_meta.json" not in _unread_line(rep), rep["missing"]
    # in a folder of one episode, the file of a video that is not one of its cameras (an infrared video) is not its task
    ctx, rep = _upload_notes(tmp_path / "e", ["ep1/top.mp4", "ep1/wrist.mp4", "ep1/top_ir.mp4"],
                             {"ep1/top_ir.json": {"task": "calibrate the infrared camera"}})
    assert "instruction" not in ctx["ep1"] and "ep1/top_ir.json" in _unread_line(rep)


def test_another_episodes_note_never_gives_an_episode_its_task():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _another_episodes_note_never_gives_an_episode_its_task(Path(t))


def _a_note_is_found_by_its_name_in_any_case(tmp_path):
    """Instruction.TXT and top.TXT were read on a filesystem that does not tell case apart and also listed as not read,
    and on one that does were never read; ep2.JSON gave its take the task and was listed as not read. A note's name is
    compared casefolded, so it is found, read and not listed on every filesystem."""
    ctx, rep = _upload_notes(tmp_path / "a", ["ep1/top.mp4", "ep2/top.mp4"],
                             {"ep1/Instruction.TXT": "pick the cup", "ep2/top.TXT": "pour the tea"})
    assert ctx["ep1/top"]["instruction"] == "pick the cup" and ctx["ep2/top"]["instruction"] == "pour the tea", ctx
    assert not _unread_line(rep), rep["missing"]
    takes = ["d/top_ep1.mp4", "d/wrist_ep1.mp4", "d/top_ep2.mp4", "d/wrist_ep2.mp4"]
    ctx, rep = _upload_notes(tmp_path / "b", takes, {"d/ep2.JSON": {"task": "pour the tea"}})
    assert "instruction" not in ctx["d/ep1"] and ctx["d/ep2"]["instruction"] == "pour the tea", ctx
    assert not _unread_line(rep), rep["missing"]


def test_a_note_is_found_by_its_name_in_any_case():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_note_is_found_by_its_name_in_any_case(Path(t))


def _a_note_named_for_an_episode_in_a_subfolder_never_gives_the_folders_episode_its_task(tmp_path):
    """ep1/top.mp4 is the one episode of ep1, and ep1/ep2/top.mp4 an episode of its own folder. ep1/ep2.json, the note
    of the episode in the subfolder ep2, had given ep1/top its task through the search of ep1's .json files, as had
    s/ep1/ep2_meta.json; so had ep1/ep2.json beside the cameras of the episode folder ep1 holding the episode ep1/ep2.
    A file named for a subfolder that holds an episode is about that episode, never the folder's own."""
    lone = ["ep1/top.mp4", "ep1/ep2/top.mp4"]
    ctx, rep = _upload_notes(tmp_path / "a", lone, {"ep1/ep2.json": {"task": "pour the tea"}})
    assert not any("instruction" in c for c in ctx.values()), ctx
    assert "ep1/ep2.json" in _unread_line(rep), rep["missing"]
    ctx, _ = _upload_notes(tmp_path / "b", ["s/ep1/top.mp4", "s/ep1/ep2/top.mp4"],
                           {"s/ep1/ep2_meta.json": {"task": "pour the tea"}})
    assert not any("instruction" in c for c in ctx.values()), ctx
    nested = ["ep1/top.mp4", "ep1/wrist.mp4", "ep1/ep2/top.mp4", "ep1/ep2/wrist.mp4"]
    ctx, _ = _upload_notes(tmp_path / "c", nested, {"ep1/ep2.json": {"task": "pour the tea"}})
    assert not any("instruction" in c for c in ctx.values()), ctx
    # a recorder's file that names no episode stays the folder's
    ctx, _ = _upload_notes(tmp_path / "d", nested, {"ep1/session_meta.json": {"prompt": "pick the cup"}})
    assert ctx["ep1"]["instruction"] == "pick the cup" and "instruction" not in ctx["ep1/ep2"]


def test_a_note_named_for_an_episode_in_a_subfolder_never_gives_the_folders_episode_its_task():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_note_named_for_an_episode_in_a_subfolder_never_gives_the_folders_episode_its_task(Path(t))


def _a_folder_json_named_for_an_episode_by_its_words_never_gives_another_episode_its_task(tmp_path):
    """A folder's .json had counted as named for an episode only when its whole name was the episode's, in the same
    case: ep1_meta.json gave the take ep2 its task, ep1_session.json gave ep10, ep2_meta.json gave ep1 though ep1.json
    was there, EP2.json and Top_ep2.json gave ep1, and ep-2.json gave the take spelt ep_1, each said to be the task the
    uploader sent with that episode. A file is named for what its words (casefolded, a number apart from the letters
    beside it) hold: a video's name, a camera's, an episode's or a subfolder's that holds an episode. It gives the task
    only to the one episode it is named for, and a file named for nothing is shared by every episode of its folder."""
    takes = ["d/top_ep1.mp4", "d/wrist_ep1.mp4", "d/top_ep2.mp4", "d/wrist_ep2.mp4"]
    tens = ["d/top_ep1.mp4", "d/wrist_ep1.mp4", "d/top_ep10.mp4", "d/wrist_ep10.mp4"]
    ctx, _ = _upload_notes(tmp_path / "a", tens, {"d/ep1_session.json": {"task": "pick the cup"}})
    assert ctx["d/ep1"]["instruction"] == "pick the cup" and "instruction" not in ctx["d/ep10"], ctx
    ctx, _ = _upload_notes(tmp_path / "b", takes, {"d/ep1_meta.json": {"task": "pick the cup"}})
    assert ctx["d/ep1"]["instruction"] == "pick the cup" and "instruction" not in ctx["d/ep2"], ctx
    ctx, _ = _upload_notes(tmp_path / "c", takes, {"d/ep2_meta.json": {"task": "pour the tea"},
                                                   "d/ep1.json": {"operator": "A"}})
    assert "instruction" not in ctx["d/ep1"] and ctx["d/ep2"]["instruction"] == "pour the tea", ctx
    for name in ("EP2.json", "ep_2.json", "Top_ep2.json", "topEp2Meta.json"):
        ctx, _ = _upload_notes(tmp_path / name, takes, {f"d/{name}": {"task": "pour the tea"}})
        assert "instruction" not in ctx["d/ep1"], (name, ctx["d/ep1"])
    dashed = ["d/top_ep-1.mp4", "d/wrist_ep-1.mp4", "d/top_ep-2.mp4", "d/wrist_ep-2.mp4"]
    ctx, _ = _upload_notes(tmp_path / "e", dashed, {"d/ep-2.json": {"task": "pour the tea"}})
    assert "instruction" not in ctx["d/ep_1"] and ctx["d/ep_2"]["instruction"] == "pour the tea", ctx
    # top.json in a folder of takes is about the top camera of every take: no take's task, and named as not read
    ctx, rep = _upload_notes(tmp_path / "f", takes, {"d/top.json": {"task": "calibrate the top camera"}})
    assert not any("instruction" in c for c in ctx.values()), ctx
    assert "d/top.json" in _unread_line(rep), rep["missing"]
    # a recorder's file named for nothing stays every take's
    ctx, _ = _upload_notes(tmp_path / "g", tens, {"d/session_meta.json": {"prompt": "pour the tea"}})
    assert [c.get("instruction") for c in ctx.values()] == ["pour the tea", "pour the tea"], ctx


def test_a_folder_json_named_for_an_episode_by_its_words_never_gives_another_episode_its_task():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_folder_json_named_for_an_episode_by_its_words_never_gives_another_episode_its_task(Path(t))


def _folder_json_files_that_name_different_tasks_give_none_and_are_named(tmp_path):
    """Two recorder files of one folder that named different tasks gave every episode the first one by name order,
    and the other was listed as not read. Neither is the task then: both are given as notes under their file names and
    the disagreement is a data issue of the episode. Files that name the same task give it and are both read."""
    takes = ["d/top_ep1.mp4", "d/wrist_ep1.mp4", "d/top_ep2.mp4", "d/wrist_ep2.mp4"]
    ctx, rep = _upload_notes(tmp_path / "a", takes, {"d/a_session.json": {"prompt": "pick the cup"},
                                                     "d/b_session.json": {"prompt": "pour the tea"}})
    for c in ctx.values():
        assert "instruction" not in c, c
        assert c["uploader_notes"] == {"a_session.json": {"prompt": "pick the cup"},
                                       "b_session.json": {"prompt": "pour the tea"}}, c
        (issue,) = [i for i in c.get("reader_issues") or [] if i["kind"] == "task_files_disagree"]
        assert "a_session.json" in issue["what"] and "b_session.json" in issue["what"], issue
    assert not _unread_line(rep), rep["missing"]
    ctx, rep = _upload_notes(tmp_path / "b", ["ep1/top.mp4", "ep1/wrist.mp4"],
                             {"ep1/a.json": {"prompt": "pick the cup"}, "ep1/b.json": {"task": "pick the cup"}})
    assert ctx["ep1"]["instruction"] == "pick the cup" and "uploader_notes" not in ctx["ep1"], ctx
    assert not ctx["ep1"].get("reader_issues") and not _unread_line(rep), rep["missing"]


def test_folder_json_files_that_name_different_tasks_give_none_and_are_named():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _folder_json_files_that_name_different_tasks_give_none_and_are_named(Path(t))


def _a_folder_json_is_named_for_the_longest_name_it_holds_and_a_date_names_no_take(tmp_path):
    """top_ep1_meta.json beside the takes ep1 and ep2 holds top, the camera of every take, inside top_ep1, the video
    of ep1 alone, and was read as named for every take, so it gave ep1 no task; 2024_01_02_session.json beside the
    takes 01 and 02 was read as named for both, so the session's task reached neither. A name inside a longer name the
    file holds counts as the longer one, and a bare number beside other numbers (a date or a time) is no take's name."""
    takes = ["d/top_ep1.mp4", "d/wrist_ep1.mp4", "d/top_ep2.mp4", "d/wrist_ep2.mp4"]
    ctx, _ = _upload_notes(tmp_path / "a", takes, {"d/top_ep1_meta.json": {"task": "pick the cup"}})
    assert ctx["d/ep1"]["instruction"] == "pick the cup" and "instruction" not in ctx["d/ep2"], ctx
    numbered = ["d/top_01.mp4", "d/wrist_01.mp4", "d/top_02.mp4", "d/wrist_02.mp4"]
    ctx, _ = _upload_notes(tmp_path / "b", numbered, {"d/2024_01_02_session.json": {"prompt": "pick the cup"}})
    assert [c.get("instruction") for c in ctx.values()] == ["pick the cup", "pick the cup"], ctx
    ctx, _ = _upload_notes(tmp_path / "c", numbered, {"d/session_02.json": {"prompt": "pour the tea"}})
    assert "instruction" not in ctx["d/01"] and ctx["d/02"]["instruction"] == "pour the tea", ctx


def test_a_folder_json_is_named_for_the_longest_name_it_holds_and_a_date_names_no_take():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_folder_json_is_named_for_the_longest_name_it_holds_and_a_date_names_no_take(Path(t))


def _a_hidden_file_names_nothing(tmp_path):
    """A hidden file the upload carries (._session.mp4, a resource fork a Mac writes beside a file) added its name to
    the folder's, so session_meta.json read as named for it and gave no task. Hidden files are never read for names,
    as the upload page never sees them."""
    ctx, _ = _upload_notes(tmp_path, ["ep1/top.mp4", "ep1/wrist.mp4"],
                           {"ep1/._session.mp4": b"\x00\x05\x16\x07",
                            "ep1/session_meta.json": {"prompt": "pick the cup"}})
    assert ctx["ep1"]["instruction"] == "pick the cup", ctx


def test_a_hidden_file_names_nothing():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_hidden_file_names_nothing(Path(t))


def _absent_line(rep) -> str:
    return next((m for m in rep["missing"] if "not in the upload" in m), "")


def _a_folder_json_whose_words_name_no_episode_stays_shared(tmp_path):
    """A recorder's file had been read as named for an episode by a word that names none: metadata_v2.json beside the
    takes 1 and 2 gave only take 2 its task (v2 split into v and 2), robot1_config.json and rs2_settings.json likewise;
    session_01.json inside the folder session_01 of takes 01 and 02 counted as named for take 01; scene_description.json
    beside takes whose cameras are all named scene counted as named for every take; session_meta.json beside a
    subfolder session holding a lone clip counted as named for that clip. Each names no episode, so every take of the
    folder shares it. A number is split from the letters before it only after a take word (ep1, take3), a file named
    as its own folder is the folder's, a camera every take has names a file only as its whole name (scene.json), and a
    subfolder named in words alone names a file only as its whole name."""
    two = ["d/top_1.mp4", "d/wrist_1.mp4", "d/top_2.mp4", "d/wrist_2.mp4"]
    for name in ("metadata_v2.json", "robot1_config.json", "rs2_settings.json"):
        ctx, _ = _upload_notes(tmp_path / name, two, {f"d/{name}": {"task": "pick the cup"}})
        assert [c.get("instruction") for c in ctx.values()] == ["pick the cup"] * 2, (name, ctx)
    folder = [f"session_01/{c}_{i}.mp4" for i in ("01", "02") for c in ("top", "wrist")]
    ctx, _ = _upload_notes(tmp_path / "a", folder, {"session_01/session_01.json": {"task": "pick the cup"}})
    assert [c.get("instruction") for c in ctx.values()] == ["pick the cup"] * 2, ctx
    scene = ["d/scene_1.mp4", "d/wrist_1.mp4", "d/scene_2.mp4", "d/wrist_2.mp4"]
    ctx, _ = _upload_notes(tmp_path / "b", scene, {"d/scene_description.json": {"task": "pick the cup"}})
    assert [c.get("instruction") for c in ctx.values()] == ["pick the cup"] * 2, ctx
    ctx, rep = _upload_notes(tmp_path / "c", scene, {"d/scene.json": {"task": "calibrate the scene camera"}})
    assert not any("instruction" in c for c in ctx.values()) and "d/scene.json" in _unread_line(rep), ctx
    takes = ["d/top_ep1.mp4", "d/wrist_ep1.mp4", "d/top_ep2.mp4", "d/wrist_ep2.mp4"]
    ctx, _ = _upload_notes(tmp_path / "d", takes + ["d/session/clip.mp4"],
                           {"d/session_meta.json": {"prompt": "pour the tea"}})
    assert [ctx[k].get("instruction") for k in ("d/ep1", "d/ep2")] == ["pour the tea"] * 2, ctx
    ctx, _ = _upload_notes(tmp_path / "e", ["ep1/top.mp4", "ep1/wrist.mp4", "ep1/raw/full.mp4"],
                           {"ep1/raw_meta.json": {"task": "pour the tea"}})
    assert ctx["ep1"]["instruction"] == "pour the tea", ctx


def test_a_folder_json_whose_words_name_no_episode_stays_shared():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_folder_json_whose_words_name_no_episode_stays_shared(Path(t))


def _a_take_number_is_read_by_its_value(tmp_path):
    """ep1.json beside the takes ep01 and ep02 named no take, so both were given its task. A number is read by its
    value, so ep1 is the take ep01."""
    takes = ["d/top_ep01.mp4", "d/wrist_ep01.mp4", "d/top_ep02.mp4", "d/wrist_ep02.mp4"]
    ctx, _ = _upload_notes(tmp_path, takes, {"d/ep1.json": {"task": "pick the cup"}})
    assert ctx["d/ep01"]["instruction"] == "pick the cup" and "instruction" not in ctx["d/ep02"], ctx


def test_a_take_number_is_read_by_its_value():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_take_number_is_read_by_its_value(Path(t))


def _a_take_word_is_learned_from_separated_or_glued_episode_names(tmp_path):
    """A take word before a separate number had not been learned, so demo1_meta.json was shared beside demo_1
    and demo_2. Both spellings identify the numbered take, including a take that was not uploaded."""
    for word in ("demo", "seq", "sequence", "clip", "part", "seg", "segment", "shot", "rec", "recording",
                 "session", "chunk"):
        for sep in ("_", "-", "", "."):
            videos = [f"d/{c}_{word}{sep}{i}.mp4" for i in (1, 2) for c in ("top", "wrist")]
            for meta_sep in ("", "_"):
                name = f"{word}{meta_sep}1_meta.json"
                ctx, _ = _upload_notes(tmp_path / word / (sep or "glued") / (meta_sep or "glued"), videos,
                                       {f"d/{name}": {"task": "pick the cup"}})
                take = word + ("_" if sep else "")
                assert ctx[f"d/{take}1"]["instruction"] == "pick the cup", ctx
                assert "instruction" not in ctx[f"d/{take}2"], ctx
            absent = f"{word}3_meta.json"
            ctx, rep = _upload_notes(tmp_path / word / (sep or "glued") / "absent", videos,
                                     {f"d/{absent}": {"task": "stack the blocks"}})
            assert not any("instruction" in c for c in ctx.values()), ctx
            assert f"d/{absent}" in _absent_line(rep), rep["missing"]


def test_a_take_word_is_learned_from_separated_or_glued_episode_names():
    with tempfile.TemporaryDirectory() as t:
        _a_take_word_is_learned_from_separated_or_glued_episode_names(Path(t))


def _a_folder_json_named_for_a_take_not_in_the_upload_gives_no_task(tmp_path):
    """ep3.json, ep3_meta.json or take3.json beside the takes ep1 and ep2, 3.json beside the takes 1 and 2, and
    ep2.json in the folder of the one episode ep1 named no episode there, so every take was given the task of a take
    that was not uploaded. A take word and a number no episode of the folder has (a bare number of the takes' digit
    count, not beside another number) name an absent take: the file gives no episode its task and is named as not read
    because its take is not in the upload."""
    takes = ["d/top_ep1.mp4", "d/wrist_ep1.mp4", "d/top_ep2.mp4", "d/wrist_ep2.mp4"]
    for name in ("ep3.json", "ep3_meta.json", "take3.json"):
        ctx, rep = _upload_notes(tmp_path / name, takes, {f"d/{name}": {"task": "stack the blocks"}})
        assert not any("instruction" in c for c in ctx.values()), (name, ctx)
        assert f"d/{name}" in _absent_line(rep) and f"d/{name}" not in _unread_line(rep), rep["missing"]
    two = ["d/top_1.mp4", "d/wrist_1.mp4", "d/top_2.mp4", "d/wrist_2.mp4"]
    ctx, rep = _upload_notes(tmp_path / "bare", two, {"d/3.json": {"task": "stack the blocks"}})
    assert not any("instruction" in c for c in ctx.values()) and "d/3.json" in _absent_line(rep), ctx
    ctx, rep = _upload_notes(tmp_path / "one", ["ep1/top.mp4", "ep1/wrist.mp4"],
                             {"ep1/ep2.json": {"task": "stack the blocks"}})
    assert "instruction" not in ctx["ep1"] and "ep1/ep2.json" in _absent_line(rep), ctx
    # a date beside numbered takes names no take and stays shared
    ctx, _ = _upload_notes(tmp_path / "date", two, {"d/session_2024_10_03.json": {"task": "stack the blocks"}})
    assert [c.get("instruction") for c in ctx.values()] == ["stack the blocks"] * 2, ctx


def test_a_folder_json_named_for_a_take_not_in_the_upload_gives_no_task():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_folder_json_named_for_a_take_not_in_the_upload_gives_no_task(Path(t))


def _a_folder_json_named_for_its_episodes_outranks_a_shared_one(tmp_path):
    """ep1_meta.json and session_meta.json naming different tasks had given ep1 no task, though the file named for ep1
    is about it alone, as live read it. A file named for this episode outranks a shared one, and only files of one
    rank can disagree; tasks that differ only in case or spacing agree; a file named for several takes
    (ep1_ep2_summary.json) gives each of them its task."""
    takes = ["d/top_ep1.mp4", "d/wrist_ep1.mp4", "d/top_ep2.mp4", "d/wrist_ep2.mp4"]
    ctx, _ = _upload_notes(tmp_path / "a", takes, {"d/ep1_meta.json": {"task": "pick the red cup"},
                                                   "d/session_meta.json": {"prompt": "pick a cup"}})
    assert ctx["d/ep1"]["instruction"] == "pick the red cup" and ctx["d/ep2"]["instruction"] == "pick a cup", ctx
    assert not any(c.get("reader_issues") for c in ctx.values()), ctx
    ctx, _ = _upload_notes(tmp_path / "b", ["ep1/top.mp4", "ep1/wrist.mp4"],
                           {"ep1/a.json": {"task": "pick  the cup"},
                            "ep1/session_meta.json": {"prompt": "Pick the cup"}})
    assert ctx["ep1"]["instruction"] == "pick  the cup" and not ctx["ep1"].get("reader_issues"), ctx
    ctx, _ = _upload_notes(tmp_path / "c", takes, {"d/ep1_ep2_summary.json": {"task": "pick the cup"}})
    assert [c.get("instruction") for c in ctx.values()] == ["pick the cup"] * 2, ctx


def test_a_folder_json_named_for_its_episodes_outranks_a_shared_one():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_folder_json_named_for_its_episodes_outranks_a_shared_one(Path(t))


def _the_notes_of_a_folder_holding_one_video_are_read(tmp_path):
    """A video alone in its folder (ep1/top.mp4) is an episode of its own, and the notes of its folder (instruction.txt,
    ep1.txt named for the folder, a recorder's session_meta.json) had never been read, so its task was lost. When a
    folder's videos make one episode, that folder's notes are read as an episode folder's are; videos beside other
    videos that are episodes of their own share no folder notes."""
    ctx, rep = _upload_notes(tmp_path / "a", ["ep1/top.mp4", "ep2/top.mp4", "ep3/top.mp4", "ep4/top.mp4"], {
        "ep1/instruction.txt": "pick the cup", "ep2/ep2.txt": "pour the tea",
        "ep3/session_meta.json": {"prompt": "stack the blocks"}, "ep4/notes.txt": "camera bumped\nat 3 s"})
    assert ctx["ep1/top"]["instruction"] == "pick the cup" and "uploader_notes" not in ctx["ep1/top"]
    assert ctx["ep2/top"]["instruction"] == "pour the tea"
    assert ctx["ep3/top"]["instruction"] == "stack the blocks"
    assert "instruction" not in ctx["ep4/top"] and ctx["ep4/top"]["uploader_notes"] == "camera bumped\nat 3 s"
    assert not _unread_line(rep), rep["missing"]
    ctx, rep = _upload_notes(tmp_path / "b", ["clips/IMG_0001.mp4", "clips/IMG_0002.mp4"],
                             {"clips/instruction.txt": "pick the cup"})
    assert not any("instruction" in c for c in ctx.values()) and "clips/instruction.txt" in _unread_line(rep)


def test_the_notes_of_a_folder_holding_one_video_are_read():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _the_notes_of_a_folder_holding_one_video_are_read(Path(t))


def _the_task_comes_from_the_first_source_that_gives_one(tmp_path):
    """ep1.txt of one line ("camera bumped at 3 s") had been taken for the task over instruction.txt beside it. The
    task comes from a JSON note's task keys, then instruction.txt or task.txt, then the text file named for the
    episode, then a video's own .txt; a lower source is a note under its file name, and one that only repeats the task
    is not given again."""
    ctx, _ = _notes_case(tmp_path / "a", {"ep1.txt": "camera bumped at 3 s", "instruction.txt": "pick the cup"})
    assert ctx["instruction"] == "pick the cup" and ctx["uploader_notes"] == {"ep1.txt": "camera bumped at 3 s"}
    ctx, _ = _notes_case(tmp_path / "b", {"annotations.json": {"task": "pick the cup"}, "task.txt": "pour the tea"})
    assert ctx["instruction"] == "pick the cup"
    assert ctx["uploader_notes"] == {"annotations.json": {"task": "pick the cup"}, "task.txt": "pour the tea"}
    ctx, _ = _upload_notes(tmp_path / "c", ["ep1/top.mp4"], {"ep1/top.txt": "the lens was dirty",
                                                             "ep1/ep1.txt": "pick the cup"})
    assert ctx["ep1/top"]["instruction"] == "pick the cup"
    assert ctx["ep1/top"]["uploader_notes"] == {"top.txt": "the lens was dirty"}
    ctx, _ = _notes_case(tmp_path / "d", {"instruction.txt": "pick the cup", "task.txt": "pick the cup"})
    assert ctx["instruction"] == "pick the cup" and "uploader_notes" not in ctx, ctx.get("uploader_notes")


def test_the_task_comes_from_the_first_source_that_gives_one():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _the_task_comes_from_the_first_source_that_gives_one(Path(t))


def _a_camera_folders_own_note_is_that_cameras(tmp_path):
    """In camera folders (top/ep1.mp4, wrist/ep1.mp4) the top camera's ep1.txt or ep1.json had become the episode's task
    while the wrist camera's said something else. A camera's own note is a note of that camera, named with its folder;
    it is the task only when every camera's own note gives the same text."""
    cams = ["top/ep1.mp4", "wrist/ep1.mp4"]
    ctx, _ = _upload_notes(tmp_path / "a", cams, {"top/ep1.txt": "the top camera was loose",
                                                  "wrist/ep1.txt": "pick the cup"})
    assert "instruction" not in ctx["ep1"]
    assert ctx["ep1"]["uploader_notes"] == {"top/ep1.txt": "the top camera was loose", "wrist/ep1.txt": "pick the cup"}
    ctx, _ = _upload_notes(tmp_path / "b", cams, {"top/ep1.json": {"task": "calibrate the top camera"}})
    assert "instruction" not in ctx["ep1"]
    assert ctx["ep1"]["uploader_notes"] == {"top/ep1.json": {"task": "calibrate the top camera"}}
    ctx, _ = _upload_notes(tmp_path / "c", cams, {"top/ep1.txt": "pick the cup", "wrist/ep1.txt": "pick the cup"})
    assert ctx["ep1"]["instruction"] == "pick the cup" and "uploader_notes" not in ctx["ep1"]
    # the episode's own note beside the camera folders shares the camera note's file name, and keeps its own: each is
    # named by its path from the episode's folder, never by the name of the folder the upload sits in
    ctx, _ = _upload_notes(tmp_path / "d", cams, {"ep1.json": {"operator": "A"}, "top/ep1.json": {"exposure": 3}})
    assert ctx["ep1"]["uploader_notes"] == {"ep1.json": {"operator": "A"}, "top/ep1.json": {"exposure": 3}}


def test_a_camera_folders_own_note_is_that_cameras():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_camera_folders_own_note_is_that_cameras(Path(t))


def _a_note_is_read_without_its_byte_order_mark(tmp_path):
    """instruction.txt saved with a byte order mark had given a task starting with it."""
    ctx, _ = _notes_case(tmp_path, {"instruction.txt": "\ufeffpick the cup\r\n"})
    assert ctx["instruction"] == "pick the cup"


def test_a_note_is_read_without_its_byte_order_mark():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_note_is_read_without_its_byte_order_mark(Path(t))


def _a_json_lines_file_with_no_row_is_named_as_not_read(tmp_path):
    """A JSON Lines file none of whose lines parses was passed over by the table reader and named nowhere."""
    _, rep = _upload_notes(tmp_path, ["clips/IMG_0001.mp4", "clips/IMG_0002.mp4"],
                           {"clips/labels.jsonl": "pick the cup\npour the tea\n"})
    assert "clips/labels.jsonl" in _unread_line(rep), rep["missing"]


def test_a_json_lines_file_with_no_row_is_named_as_not_read():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_json_lines_file_with_no_row_is_named_as_not_read(Path(t))


def test_notes_cut_for_the_prompt_name_the_files_left_out():
    notes = {"a.txt": "x" * (f.ANNOTATION_MAX_CHARS - 20), "b.txt": "y" * 100, "c.txt": "z"}
    txt = f.annotation_text(notes, files=True)
    assert txt.endswith(" [truncated, the end of b.txt and all of c.txt left out]"), txt[-120:]
    assert f.annotation_text("x" * (f.ANNOTATION_MAX_CHARS + 5)).endswith("x [truncated]")
    # a file whose notes hold another file's name as a key is never taken for where that file starts
    nested = {"a.json": {"b.txt": "y" * 10, "pad": "x" * f.ANNOTATION_MAX_CHARS}, "b.txt": "z"}
    txt = f.annotation_text(nested, files=True)
    assert txt.endswith(" [truncated, the end of a.json and all of b.txt left out]"), txt[-120:]


def _the_no_camera_refusal_accounts_for_every_file(tmp_path):
    root = tmp_path / "sensors"
    root.mkdir()
    for k in range(5):
        _glove_h5(root / f"glove{k}.h5", 10)
    (root / "a.bin").write_bytes(b"x")
    saved = f.NO_CAMERA_FILES_MAX
    f.NO_CAMERA_FILES_MAX = 2
    try:
        f.detect(root)
    except ValueError as e:
        msg = str(e)
    finally:
        f.NO_CAMERA_FILES_MAX = saved
    assert "6 files" in msg and "5 HDF5 files" in msg and "1 other file" in msg, msg


def test_the_no_camera_refusal_accounts_for_every_file():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _the_no_camera_refusal_accounts_for_every_file(Path(t))


def _an_mcap_beside_videos_of_the_same_length_is_noted_as_a_possible_duplicate(tmp_path):
    root = tmp_path / "upload"
    root.mkdir()
    _camera_mcap(root / "rec.mcap", ["/cam/image/compressed"], n=30)
    _clip(root / "rec_cam.mp4", 30)
    det, items = f.plan(root)
    assert len(items) == 2
    assert any("rec.mcap" in u and "same footage" in u for u in det["used"]), det["used"]


def test_an_mcap_beside_videos_of_the_same_length_is_noted_as_a_possible_duplicate():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _an_mcap_beside_videos_of_the_same_length_is_noted_as_a_possible_duplicate(Path(t))


def _a_folder_read_by_a_dataset_adapter_is_not_called_unread(tmp_path):
    """OpenAoE's clip folder holds its annotation and a video_info.json, which its adapter reads; they are not files
    no reader opened."""
    clip = tmp_path / "upload" / "raw_x_seg_1"
    (clip / "ego_annotation").mkdir(parents=True)
    _clip(clip / "raw_video.mp4", 10)
    action = '{"verb": "align", "object": "fabric", "hand": "both"}'
    (clip / "ego_annotation" / "ego_action_annotation.json").write_text(
        '[{"start_ts": "0.00", "end_ts": "0.30", "atomic_action": [' + action + ']}]')
    (clip / "video_info.json").write_text("{}")
    det, _ = f.plan(tmp_path / "upload")
    assert not any("no reader opens" in m for m in det["missing"]), det["missing"]


def test_a_folder_read_by_a_dataset_adapter_is_not_called_unread():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _a_folder_read_by_a_dataset_adapter_is_not_called_unread(Path(t))


def _an_own_sensor_file_recorded_outside_the_footage_is_listed_not_placed(tmp_path):
    """An HDF5 episode and its own glove file are both on epoch clocks, the glove recorded an hour earlier. It had been
    placed from both starts, as if the two shared no clock; both clocks are real and the glove covers none of the
    footage, so it is listed as recorded outside it and never placed."""
    import h5py
    import numpy as np
    root = tmp_path / "a"
    root.mkdir()
    with h5py.File(root / "run1.h5", "w") as h:
        h["cam"] = np.random.default_rng(1).integers(0, 255, (60, 96, 96, 3)).astype(np.uint8)
        h["timestamps"] = 1_790_003_600.0 + np.arange(60) / 30
    with h5py.File(root / "glove.h5", "w") as h:
        h["time"] = 1_790_000_000.0 + np.arange(60) / 30
        h["pressure"] = np.random.default_rng(2).random((60, 4, 4))
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = _episode_ctx(tmp_path / "eps", rep, "run1")
    assert not ctx.get("signals") and not _issues(ctx, "signal_alignment_assumed"), ctx.get("signals")
    assert ctx["source"]["sensor_files"]["glove.h5"].startswith("not placed: recorded from"), ctx["source"]
    assert any(u.startswith("glove.h5 (recorded from") and "outside the footage" in u
               for u in ctx["source"]["unused_signals"]), ctx["source"]


def test_an_own_sensor_file_recorded_outside_the_footage_is_listed_not_placed():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _an_own_sensor_file_recorded_outside_the_footage_is_listed_not_placed(Path(t))


def _the_episode_budget_holds_while_signals_are_read(tmp_path):
    """The budget on an episode's signals had been applied only after every signal was in memory. It holds as they are
    read: an array that would take the running total past SIGNAL_EPISODE_BYTES is kept as its summary when it is
    added, an HDF5 array that would is summarised a block of rows at a time and never read whole, and signals merged
    from another reader count toward the same total."""
    import h5py
    import numpy as np
    n = 40
    rng = np.random.default_rng(6)
    saved = f.SIGNAL_EPISODE_BYTES, f.float_rows
    seen = []
    f.SIGNAL_EPISODE_BYTES = 400_000                       # one 40 x 2000 float32 array (320 kB) fits, two do not
    f.float_rows = lambda a: seen.append(np.shape(a)) or saved[1](a)
    try:
        out = f.Signals()
        out.add("a", rng.random((n, 2000)).astype(np.float32))
        out.add("b", rng.random((n, 2000)).astype(np.float32))
        assert out["a"].shape == (n, 2000) and out["b"].shape == (n, 3) and out.meta["b"]["summary_of"] == 2000
        assert [i["signal"] for i in out.issues] == ["b"]
        more = f.Signals()
        more.add("c", rng.random((n, 2000)).astype(np.float32))
        f.merge_signals(out, more)
        assert out["c"].shape == (n, 3) and out.meta["c"]["names"] == f.SUMMARY_NAMES
        with h5py.File(tmp_path / "ep.h5", "w") as h:
            h["timestamps"] = 1_790_000_000.0 + np.arange(n) / 30
            h["cloud_a"] = rng.random((n, 2000)).astype(np.float32)
            h["cloud_b"] = rng.random((n, 2000)).astype(np.float32)
        with h5py.File(tmp_path / "ep.h5", "r") as h:
            st = f.h5_streams(h, "")
            got = f.h5_signals(h, st, st["clock"]["timestamps"], None, n)
    finally:
        f.SIGNAL_EPISODE_BYTES, f.float_rows = saved
    assert got["cloud_a"].shape == (n, 2000) and got["cloud_b"].shape == (n, 3), {k: v.shape for k, v in got.items()}
    assert (n, 2000) in seen and seen.count((n, 2000)) == 1 + 2, seen   # cloud_b is never read whole


def test_the_episode_budget_holds_while_signals_are_read():
    # no pytest fixture: Data Review runs this file's tests as plain functions (upload/test_formats.py)
    with tempfile.TemporaryDirectory() as t:
        _the_episode_budget_holds_while_signals_are_read(Path(t))


def test_a_signal_removed_or_replaced_gives_its_budget_back():
    """A signal taken out of the signals (h5_state reads it as the state) or replaced under its name had stayed in the
    running total, so a later signal could be summarised for room no longer used."""
    import numpy as np
    saved = f.SIGNAL_EPISODE_BYTES
    f.SIGNAL_EPISODE_BYTES = 400_000                       # one 40 x 2000 float32 array (320 kB) fits, two do not
    try:
        out = f.Signals()
        wide = lambda: np.random.default_rng(7).random((40, 2000)).astype(np.float32)
        out.add("qpos", wide())
        out.pop("qpos")
        out.add("a", wide())
        out.add("a", wide())                               # replaced under its name: counted once
        del out["a"]
        out.add("b", wide())
    finally:
        f.SIGNAL_EPISODE_BYTES = saved
    assert out["b"].shape == (40, 2000) and not out.issues and out.bytes == 40 * 2000 * 4, (out.issues, out.bytes)
