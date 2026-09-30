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
     {"exo": "/zed/front/left", "extra1": "front", "extra2": "back", "extra3": "side_a"}),
    (["observation.images.left_camera_rgb_image", "observation.images.right_camera_rgb_image"], "handheld_gripper",
     {"left": "observation.images.left_camera_rgb_image", "right": "observation.images.right_camera_rgb_image"}),
    (["observation.images.camera_rgb_image"], "handheld_gripper", {"right": "observation.images.camera_rgb_image"}),
    (["/robot0/wrist_l", "/robot0/wrist_r", "/scene"], "teleop_arms", {"exo": "/scene", "left": "/robot0/wrist_l", "right": "/robot0/wrist_r"}),
    (["observation.images.top", "observation.images.top_depth", "observation.images.wrist_left"], "teleop_arms",
     {"exo": "observation.images.top", "left": "observation.images.wrist_left"}),
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
        assert any("3 depth or infrared videos were left out" in u for u in det["used"])
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
