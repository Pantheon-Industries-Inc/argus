"""Touch, depth and the other signals a recording keeps, on synthetic inputs: the reader keeps every array with its
shape and names and says why when it leaves one out (prepare/formats.py), HDF5 files are read by what each array is,
depth goes with its colour camera, fast sensors are summarised per frame, contacts are found from touch signals
(label/contacts.py) and checked against the model's answers (checks/contacts.py), and an episode without any of this
is labelled exactly as before."""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from checks import contacts as cc
from label import contacts as lc
from label import depth as dp
from label import episode as me
from label import signals as sg
from prepare import formats

av = pytest.importorskip("av")
h5py = pytest.importorskip("h5py")


def _jpeg(shade: int) -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.full((96, 128, 3), shade % 256, np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def _press(n: int, on: tuple[int, int], rest: float = 3072.0, palm: bool = False) -> np.ndarray:
    """A 16 x 16 glove map that rests at its untouched reading and falls where it is pressed, frames on[0]..on[1]: a
    fingertip's patch, or with palm the whole map (a palm pressed flat)."""
    a = np.full((n, 16, 16), rest, np.float32)
    a += np.random.default_rng(0).normal(0, 2, a.shape).astype(np.float32)
    if palm:
        a[on[0]:on[1]] -= 2000.0
    else:
        a[on[0]:on[1], 3:8, 5:10] = 900.0
    return a


def _hdf5(path: Path, demos: int = 2, n: int = 40, press: tuple[int, int] | None = (12, 26),
          palm: bool = False) -> None:
    """Demos of JPEG frames, a clock, a right glove's pressure map pressed over frames press (never, when None; the
    whole map with palm) and hand landmarks."""
    with h5py.File(path, "w") as f:
        f.attrs["hand_mode"] = "right"
        f.create_dataset("calibration/rgb/T_device_camera", data=np.eye(4))
        for d in range(demos):
            g = f.create_group(f"data/demo_{d:02d}")
            dt = h5py.special_dtype(vlen=np.dtype("uint8"))
            imgs = g.create_dataset("rgb_images_jpeg", (n,), dtype=dt)
            for i in range(n):
                imgs[i] = np.frombuffer(_jpeg(i * 5), np.uint8)
            g.create_dataset("timestamps", data=(np.arange(n) * 33_333_333 + 10**12).astype(np.int64))
            g.create_dataset("right_pressure", data=_press(n, press or (0, 0), palm=palm))
            g.create_dataset("right_hand_landmarks", data=np.random.default_rng(d).normal(0, 0.1, (n, 21, 3)))
            if d == 0:
                flags = np.array([(0, 1)], dtype=[("low_light", "u1"), ("hand_out_of_frame", "u1")])
                g.create_dataset("labels", data=flags)


def test_an_hdf5_file_of_demos_is_one_episode_per_demo_with_its_pressure_map_and_landmarks(tmp_path):
    """Demo groups with one layout are episodes (one with an extra array still is); JPEG bytes are the camera, a rising
    ns array its clock, a 16 x 16 map and 21 x 3 landmarks signals with their shapes, a 4 x 4 calibration a note."""
    root = tmp_path / "up"
    root.mkdir()
    _hdf5(root / "kitchen_p1.hdf5")
    rep = formats.convert(root, "ego_head", tmp_path / "eps", "touchset", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 2
    ep = tmp_path / "eps" / rep["episodes"][0]["episode_id"]
    ctx = json.loads((ep / "context.json").read_text())
    sig = {s["name"]: s for s in ctx["signals"]}
    assert sig["right_pressure"]["shape"] == [16, 16] and sig["right_hand_landmarks"]["shape"] == [21, 3]
    assert abs(ctx["fps"] - 30.0) < 0.1
    assert "calibration/rgb/T_device_camera" in ctx["uploader_annotation"]
    # the pressure map is touch: one contact on the right hand, timed from the frames where it is pressed
    (c,) = ctx["contacts"]
    assert c["hand"] == "right" and c["signals"] == ["right_pressure"]
    assert abs(c["start_s"] - 12 / 30) < 0.05 and abs(c["end_s"] - 25 / 30) < 0.05
    assert c["regions"]["right_pressure"]["rows"] == [3, 7] and c["regions"]["right_pressure"]["columns"] == [5, 9]


def _h5_rate(d: Path, attrs: dict, root_attrs: dict | None = None):
    """h5_fps of demo_0 in a file whose data group holds attrs and whose root holds root_attrs."""
    with h5py.File(d / "rate.hdf5", "w") as f:
        g = f.create_group("data")
        for k, v in attrs.items():
            g.attrs[k] = v
        for k, v in (root_attrs or {}).items():
            f.attrs[k] = v
        g.create_group("demo_0")
        return formats.h5_fps(f, "data/demo_0")


def test_an_hdf5_rate_inside_a_json_attribute_times_the_frames(tmp_path):
    """robomimic keeps its control rate in the data group's env_args, a JSON string ({"env_kwargs": {"control_freq":
    20}}); the demo is timed at that rate, not the 30 fps default."""
    env_args = json.dumps({"env_name": "Lift", "env_kwargs": {"control_freq": 20, "horizon": 400}})
    assert _h5_rate(tmp_path, {"env_args": env_args}) == 20.0


def test_an_attribute_that_is_not_json_or_is_malformed_json_is_passed_over(tmp_path):
    attrs = {"note": "pick up the cup", "blob": np.bytes_(b"{not json"), "empty": "",
             "env_args": json.dumps({"env_kwargs": {"control_freq": 20}})}
    assert _h5_rate(tmp_path, attrs) == 20.0


def test_json_held_in_bytes_or_in_a_one_element_string_array_is_read(tmp_path):
    assert _h5_rate(tmp_path, {"env_args": np.bytes_(json.dumps({"fps": 12}).encode())}) == 12.0
    one = np.array([json.dumps({"fps": 30})], dtype=h5py.string_dtype())
    assert _h5_rate(tmp_path, {"env_args": one}) == 30.0


def test_a_rate_inside_a_json_list_counts(tmp_path):
    assert _h5_rate(tmp_path, {"cameras": json.dumps([{"name": "agentview"}, {"fps": 25}])}) == 25.0


def test_a_rate_outside_1_to_1000_is_not_a_rate(tmp_path):
    assert _h5_rate(tmp_path, {"env_args": json.dumps({"env_kwargs": {"control_freq": 0.5}})}) is None
    assert _h5_rate(tmp_path, {"fps": 5000}) is None


def test_a_bool_is_not_a_rate(tmp_path):
    assert _h5_rate(tmp_path, {"fps": np.bool_(True)}) is None
    assert _h5_rate(tmp_path, {"env_args": json.dumps({"fps": True})}) is None
    assert _h5_rate(tmp_path, {"env_args": json.dumps({"frame_rate": True, "fps": 12})}) == 12.0


def test_the_nearest_groups_frame_rate_wins(tmp_path):
    assert _h5_rate(tmp_path, {"fps": 15}, root_attrs={"fps": 30}) == 15.0


def test_a_direct_frame_rate_beats_one_inside_json_on_the_same_group(tmp_path):
    """Attributes are read in name order, so env_args sorts before fps; the group's own fps (15) still times the frames
    over the control_freq (20) inside its env_args."""
    assert _h5_rate(tmp_path, {"env_args": json.dumps({"env_kwargs": {"control_freq": 20}}), "fps": 15}) == 15.0


def test_a_direct_frame_rate_on_the_root_beats_a_frame_rate_inside_a_groups_json(tmp_path):
    attrs = {"env_args": json.dumps({"env_kwargs": {"control_freq": 20}})}
    assert _h5_rate(tmp_path, attrs, root_attrs={"fps": 30}) == 30.0


def test_a_generic_rate_name_inside_json_never_counts(tmp_path):
    """A configuration holds its sensors' rates under generic names, so {"sensors": {"imu": {"rate": 200}}} does not
    time the frames."""
    assert _h5_rate(tmp_path, {"sensor_config": json.dumps({"sensors": {"imu": {"rate": 200}}})}) is None
    assert _h5_rate(tmp_path, {"sensor_config": json.dumps({"audio": {"hz": 48}})}) is None


def test_a_frame_rate_name_inside_json_counts_beside_a_sensors_generic_one(tmp_path):
    both = json.dumps({"imu": {"rate": 200}, "env_kwargs": {"control_freq": 20}})
    assert _h5_rate(tmp_path, {"env_args": both}) == 20.0


def test_a_direct_generic_rate_counts_only_when_no_frame_rate_is_stated_anywhere(tmp_path):
    assert _h5_rate(tmp_path, {"rate": 25}) == 25.0
    assert _h5_rate(tmp_path, {"hz": 200, "env_args": json.dumps({"fps": 30})}) == 30.0
    assert _h5_rate(tmp_path, {"rate": 200}, root_attrs={"fps": 30}) == 30.0


def test_the_first_frame_rate_in_document_order_wins_inside_json(tmp_path):
    attrs = {"env_args": json.dumps({"camera": {"fps": 15}, "env_kwargs": {"control_freq": 20}})}
    assert _h5_rate(tmp_path, attrs) == 15.0


def test_per_camera_frame_rates_count_when_they_agree(tmp_path):
    assert _h5_rate(tmp_path, {"meta": json.dumps({"fps": {"cam_high": 30, "cam_wrist": 30}})}) == 30.0
    assert _h5_rate(tmp_path, {"meta": json.dumps({"fps": {"cam_high": 30, "cam_wrist": 15}})}) is None


def test_json_nested_deeper_than_the_parser_allows_is_passed_over(tmp_path):
    """json.loads stops at about 1000 levels and raises RecursionError, which h5_fps passes over."""
    assert _h5_rate(tmp_path, {"env_args": "[" * 100000}) is None


def test_json_the_parser_accepts_is_searched_at_any_depth(tmp_path):
    """The search through parsed JSON is a loop, with no depth limit of its own."""
    assert _h5_rate(tmp_path, {"env_args": '{"a":' * 900 + '{"fps": 9}' + "}" * 900}) == 9.0


def test_a_robomimic_style_demo_without_timestamps_is_timed_at_its_json_rate(tmp_path):
    """The rate reaches the episode: 40 frames at the 20 Hz the data group's env_args states last 2.0 s, not 1.33 s
    (robomimic's 84 x 84 agentview images, two or more demos, actions beside them, no timestamps)."""
    root = tmp_path / "up"
    root.mkdir()
    with h5py.File(root / "rm.hdf5", "w") as f:
        g = f.create_group("data")
        g.attrs["env_args"] = json.dumps({"env_kwargs": {"control_freq": 20}})
        for i in range(2):
            d = g.create_group(f"demo_{i}")
            d.create_dataset("obs/agentview_image",
                             data=np.random.default_rng(i).integers(0, 255, (40, 84, 84, 3), dtype=np.uint8))
            d.create_dataset("actions", data=np.zeros((40, 7), np.float32))
    rep = formats.convert(root, "teleop_arms", tmp_path / "eps", "rm", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 2
    ctx = json.loads((tmp_path / "eps" / rep["episodes"][0]["episode_id"] / "context.json").read_text())
    assert abs(ctx["fps"] - 20.0) < 0.1 and abs(ctx["duration_s"] - 2.0) < 0.1


def test_a_fast_pad_on_the_cameras_clock_from_boot_is_read_in_the_cameras_unit(tmp_path):
    """One recorder stamps a 30 fps camera and a 1 kHz pad in nanoseconds from boot (43 minutes up). The pad's step of
    1e6 alone reads as microseconds, which would put it 1000 times too far along the clock and leave it out; the
    file's clocks start together, so the pad is read in the camera's unit and kept at its real rate."""
    root = tmp_path / "up"
    root.mkdir()
    boot_ns = 2_600 * 10**9
    n_cam, n_pad = 40, 1400
    with h5py.File(root / "glove.hdf5", "w") as f:
        imgs = f.create_dataset("rgb_images_jpeg", (n_cam,), dtype=h5py.special_dtype(vlen=np.dtype("uint8")))
        for i in range(n_cam):
            imgs[i] = np.frombuffer(_jpeg(i * 5), np.uint8)
        f.create_dataset("timestamps", data=(boot_ns + np.arange(n_cam) * 33_333_333).astype(np.int64))
        f.create_dataset("pad_timestamps", data=(boot_ns - 10**7 + np.arange(n_pad) * 1_000_000).astype(np.int64))
        f.create_dataset("pad_pressure", data=np.random.default_rng(0).normal(0, 1, (n_pad, 4)))
    rep = formats.convert(root, "handheld_gripper", tmp_path / "eps", "glove", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 1
    ctx = json.loads((tmp_path / "eps" / rep["episodes"][0]["episode_id"] / "context.json").read_text())
    pad = {s["name"]: s for s in ctx.get("signals", [])}.get("pad_pressure")
    assert pad is not None, ctx["source"].get("unused_signals")
    assert abs(pad["rate_hz"] - 1000) < 10


def test_a_pad_log_longer_than_the_clip_is_kept_at_its_real_rate(tmp_path):
    """A 1 kHz pad log that starts 3 s before a 3 s camera clip and ends 3 s after it, both in nanoseconds from boot: a
    sensor log longer than the clip is common, and the pad is placed on the camera's frames, not left out."""
    root = tmp_path / "up"
    root.mkdir()
    boot_ns = 2_600 * 10**9
    n_cam, n_pad = 90, 9000
    with h5py.File(root / "glove.hdf5", "w") as f:
        imgs = f.create_dataset("rgb_images_jpeg", (n_cam,), dtype=h5py.special_dtype(vlen=np.dtype("uint8")))
        for i in range(n_cam):
            imgs[i] = np.frombuffer(_jpeg(i * 5), np.uint8)
        f.create_dataset("timestamps", data=(boot_ns + np.arange(n_cam) * 33_333_333).astype(np.int64))
        f.create_dataset("pad_timestamps", data=(boot_ns - 3 * 10**9 + np.arange(n_pad) * 1_000_000).astype(np.int64))
        f.create_dataset("pad_pressure", data=np.random.default_rng(0).normal(0, 1, (n_pad, 4)))
    rep = formats.convert(root, "handheld_gripper", tmp_path / "eps", "glove", 900)
    ctx = json.loads((tmp_path / "eps" / rep["episodes"][0]["episode_id"] / "context.json").read_text())
    pad = {s["name"]: s for s in ctx.get("signals", [])}.get("pad_pressure")
    assert pad is not None, ctx["source"].get("unused_signals")
    assert abs(pad["rate_hz"] - 1000) < 10


def test_the_notes_an_upload_sends_reach_the_board_as_sent_with_their_times_on_the_episode(tmp_path):
    """A table row that names a demo goes to the board under its own column names, beside the notes read from the file,
    and a value on the recorder's clock (ns here) carries its moment in the episode; a count stays a count."""
    from board.build import uploader_groups
    root = tmp_path / "up"
    root.mkdir()
    _hdf5(root / "kitchen_p1.hdf5")
    start_ns = 10**12
    (root / "notes.csv").write_text("clip_id,object,onset_idx,onset_ts\n"
                                    f"kitchen_p1::demo_00,cup,12,{start_ns + 400_000_000}\n"
                                    f"kitchen_p1::demo_01,bowl,3,{start_ns + 100_000_000}\n")
    rep = formats.convert(root, "ego_head", tmp_path / "eps", "touchset", 900)
    ctx = json.loads((tmp_path / "eps" / rep["episodes"][0]["episode_id"] / "context.json").read_text())
    assert abs(ctx["clock_start_s"] - 1000.0) < 1e-6
    rows, notes = uploader_groups(ctx["uploader_notes"], ctx["clock_start_s"], ctx["duration_s"])
    assert rows["title"] == "Row of notes.csv" and notes["title"] == "Notes in the files"
    got = {it["name"]: it for it in rows["items"]}
    assert got["object"]["value"] == "cup" and "t" not in got["onset_idx"]
    assert abs(got["onset_ts"]["t"] - 0.4) < 1e-6
    assert any(it["name"].endswith("calibration/rgb/T_device_camera") for it in notes["items"])


def test_an_episode_with_a_contact_shows_it_and_one_without_is_labelled_as_before(tmp_path):
    """The contact's picture follows the detail views and the prompt says what it is and what to return; the touch
    signal's timing is given only there. A recording with no touch signal gets no word about contacts."""
    root = tmp_path / "up"
    root.mkdir()
    _hdf5(root / "kitchen_p1.hdf5", demos=2)
    rep = formats.convert(root, "ego_head", tmp_path / "eps", "touchset", 900)
    r = me.build_request(tmp_path / "eps" / rep["episodes"][0]["episode_id"])
    assert r["contact_views"]["shown"] == ["c1"] and "CONTACTS:" in r["prompt"]
    assert '"contacts": [{"id"' in r["prompt"] and "contacts_missing" in r["prompt"]
    assert any(c.get("type") == "text" and c["text"].startswith("=== contact c1") for c in r["content"])
    table = r["prompt"].split("OTHER RECORDED SIGNALS")[1].split("BETWEEN INSTANTS")[0]
    assert "right_pressure (16 x 16 values)" in table and "right_pressure total" not in table
    strips = r["contact_views"]["strips"]["c1"]
    assert len(strips["begin"]) == 5 and len(strips["end"]) == 3
    assert "contacts" in r["blocks"] and {"contacts", "contacts_missing"} <= set(r["schema_fields"])
    # the same glove never pressed: no contact, no word of contacts and no contact fields asked for
    (tmp_path / "up2").mkdir()
    _hdf5(tmp_path / "up2" / "kitchen_p2.hdf5", demos=1, press=None)
    rep = formats.convert(tmp_path / "up2", "ego_head", tmp_path / "eps2", "touchset", 900)
    r = me.build_request(tmp_path / "eps2" / rep["episodes"][0]["episode_id"])
    assert "contacts" not in r["blocks"] and not {"contacts", "contacts_missing"} & set(r["schema_fields"])
    assert "CONTACTS:" not in r["prompt"] and "contact_views" not in r


def test_one_request_judges_each_signal_touch_at_most_once(tmp_path, monkeypatch):
    """Touch is judged once per plan (label/episode.py plan, pl["touch"]): a 450 s episode with a 16 x 16 glove and 60
    contacts spent about two minutes per request when every presence test and every contact judged it again."""
    root = tmp_path / "up"
    root.mkdir()
    _hdf5(root / "kitchen_p1.hdf5", demos=1)
    rep = formats.convert(root, "ego_head", tmp_path / "eps", "touchset", 900)
    calls = {}
    judge = sg.is_touch

    def counted(name, *a, **k):
        calls[name] = calls.get(name, 0) + 1
        return judge(name, *a, **k)
    monkeypatch.setattr(sg, "is_touch", counted)
    ep = tmp_path / "eps" / rep["episodes"][0]["episode_id"]
    r = me.build_request(ep)
    assert r["contact_views"]["shown"] == ["c1"]
    assert calls and max(calls.values()) == 1, calls
    # a context prepared before contacts were measured: they are found with the plan's verdicts, not judged again
    ctx = json.loads((ep / "context.json").read_text())
    del ctx["contacts"]
    (ep / "context.json").write_text(json.dumps(ctx))
    calls.clear()
    r = me.build_request(ep)
    assert r["contact_views"]["shown"] == ["c1"] and [c["id"] for c in r["contacts"]] == ["c1"]
    assert calls and max(calls.values()) == 1, calls


def test_a_contact_touching_at_the_first_frame_is_told_and_asked_only_for_the_strip_its_picture_has(tmp_path):
    """A press from the first frame has no begin of its own in the clip, so its picture has no begin strip
    (contact_image): the prompt says so and asks only for last_touch_frame. Beside a contact with both strips, the
    begin strip and first_touch_frame are said to be for that contact only."""
    root = tmp_path / "up"
    root.mkdir()
    _hdf5(root / "kitchen_p1.hdf5", demos=1, press=(0, 20))
    rep = formats.convert(root, "ego_head", tmp_path / "eps", "touchset", 900)
    r = me.build_request(tmp_path / "eps" / rep["episodes"][0]["episode_id"])
    (c,) = r["contacts"]
    assert c["from_start"] and not c["to_end"] and list(r["contact_views"]["strips"]["c1"]) == ["end"]
    assert ("each contact above has one picture: three frames around the time the signal says the touch ends, "
            "numbered 1 to 3, and the moment it is strongest") in r["prompt"]
    assert "first_touch_frame" not in r["prompt"] and "begin strip" not in r["prompt"]
    assert '"last_touch_frame": <1-3, the last frame of the end strip' in r["prompt"]
    ep = me.load(tmp_path / "eps" / rep["episodes"][0]["episode_id"])
    pl = me.plan(ep)
    later = {**c, "id": "c2", "from_start": False, "start_s": 0.9, "end_s": 1.1, "peak_s": 1.0}
    ep["contacts"] = ep["contacts_shown"] = [c, later]
    block = me.contacts_block(ep, pl)
    assert ("five frames around the time the signal says the touch begins, numbered 1 to 5 (not for c1, already "
            "touching at the first frame), three around the time it says the touch ends, numbered 1 to 3, and the "
            "moment it is strongest") in block
    assert ('"first_touch_frame": <for c2 only, 1-5, the first frame of the begin strip in which the hand is touching, '
            'or null>') in block
    assert '"last_touch_frame": <1-3, the last frame of the end strip' in block


def test_a_depth_video_beside_its_colour_video_goes_with_that_camera(tmp_path):
    d = tmp_path / "up" / "ep1"
    d.mkdir(parents=True)
    n = 20
    for cam in ("exo_cam", "wrist_cam_left"):
        c = av.open(str(d / f"{cam}-images-rgb.mp4"), "w")
        s = c.add_stream("mpeg4", rate=30)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for k in range(n):
            fr = av.VideoFrame.from_ndarray(np.full((48, 64, 3), k * 9, np.uint8), format="rgb24")
            fr.pts = k
            for p in s.encode(fr):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
        c.close()
    dw = formats.DepthWriter(d / "exo_cam-images-depth.mkv")
    for k in range(n):
        dw.add(k / 30, np.full((48, 64), 1000 + 10 * k, np.uint16), None)
    dw.close()
    rep = formats.convert(tmp_path / "up", "teleop_arms", tmp_path / "eps", "rig", 900)
    assert any("1 depth video was read with the colour camera it belongs to" in u for u in rep["used"])
    ep = me.load(tmp_path / "eps" / rep["episodes"][0]["episode_id"])
    assert list(ep["depth"]) == ["exo"] and ep["depth"]["exo"].get("range")        # measured across the upload
    got = dp.at_anchor(ep, ep["depth"], "exo", [0, 10])
    assert int(got[10][0, 0]) == 1100


def test_depth_colours_are_one_scale_and_black_is_only_no_reading():
    near, far = dp.picture(np.array([[300]], np.uint16), {"scale_m": 0.001}), dp.picture(np.array([[3000]], np.uint16),
                                                                                          {"scale_m": 0.001})
    assert near.getpixel((0, 0))[0] > near.getpixel((0, 0))[2] and far.getpixel((0, 0))[2] > far.getpixel((0, 0))[0]
    assert dp.picture(np.array([[0]], np.uint16), {"scale_m": 0.001}).getpixel((0, 0)) == (0, 0, 0)
    assert dp.picture(np.array([[65535]], np.uint16), {"range": [0, 100]}).getpixel((0, 0)) == (0, 0, 0)
    assert sum(dp.picture(np.array([[100]], np.uint16), {"range": [0, 100]}).getpixel((0, 0))) > 40   # far is not black


def test_a_sensor_faster_than_the_camera_is_summarised_per_frame():
    q = np.arange(30) / 30.0
    t = np.arange(900) / 900.0
    v = np.where((t > 0.5) & (t < 0.6), np.sin(t * 2000), 0.0)[:, None]
    a, var, gaps = formats.place_on_frames(t, v, q)
    assert a.shape == (30, 1) and var is not None and gaps == 0
    assert var[16, 0] > 0.3 and var[5, 0] == 0.0                 # the vibration between frames is kept


def test_counters_are_bookkeeping_and_a_multiplexed_topic_is_split():
    r = {"v": [[i, 1 + (i % 3 == 0), 0.1 * i] for i in range(40)], "names": ["seq", "type", "x"], "d": 3}
    assert formats._counters(r) == [0]
    r2 = {"v": [[float(i % 2 + 1), float(i)] for i in range(40)], "names": ["type", "x"], "d": 2}
    assert formats._multiplexed(r2) == 0


def test_a_table_row_that_names_an_episode_becomes_its_notes():
    tables = [("ann/kitchen_p1_merged.csv", [{"clip_id": "kitchen_p1::demo_01", "object_name": "mug", "action": "lift"},
                                             {"clip_id": "kitchen_p1::demo_02", "object_name": "tongs"}])]
    item = {"kind": "hdf5", "name": "kitchen_p1/demo_01", "file": Path("kitchen_p1.hdf5"), "group": "data/demo_01"}
    rows = formats.table_rows_for(tables, item)
    assert len(rows) == 1 and rows[0]["object_name"] == "mug"


def test_tactile_images_are_views_of_their_own_never_the_scene_or_gripper_camera():
    vm, unused = formats.assign_views(["external_cam", "wrist_cam", "tactile_left", "tactile_right"], "teleop_arms")
    assert vm["exo"] == "external_cam" and "tactile_left" in vm.values() and "tactile_right" in vm.values()
    assert vm.get("left") != "tactile_left" and not unused


def test_contacts_are_checked_against_the_frames():
    contacts = [{"id": f"c{i}", "hand": "right", "start_s": 1.0 + i, "end_s": 1.5 + i} for i in range(3)]
    strips = {f"c{i}": {"begin": [0.7 + i, 0.85 + i, 1.0 + i, 1.15 + i, 1.3 + i], "end": [1.35 + i, 1.5 + i, 1.65 + i]}
              for i in range(3)}
    seen = [{"id": f"c{i}", "touch_seen": "yes", "first_touch_frame": 5, "last_touch_frame": 3, "hand": "left"}
            for i in range(3)]
    res = cc.check({"contacts": seen, "contacts_missing": [{"t_s": 9.0, "hand": "left", "object": "cup"}]}, contacts,
                   strips, 30)
    kinds = {n["check"] for n in res["notes"]}
    assert {"clock_offset", "hand_mismatch", "contact_missing"} <= kinds
    assert res["offset_ms"]["median"] > 100                 # touch shows about 0.22 s after the signal says it begins


def test_a_signal_that_swings_both_ways_or_a_switching_setting_is_not_touch():
    n = 200
    vel = np.sin(np.linspace(0, 20, n))[:, None] * (np.linspace(0, 20, n) % 4 < 1)[:, None]
    assert not sg.touch_like(vel)
    setting = np.tile(np.array([[1.0, 0.0, 320.0]]), (n, 1))
    setting[150:] = [1.0, 0.0, 321.0]
    assert not sg.touch_like(setting)
    t = np.arange(n) / 30.0
    press = _press(n, (60, 120)).reshape(n, -1).astype(np.float64)
    (c,) = lc.find({"left_glove_pressure": press}, {"left_glove_pressure": {"shape": [16, 16]}}, t)
    assert c["hand"] == "left" and abs(c["start_s"] - 2.0) < 0.05


def test_a_position_that_leaves_its_level_one_way_or_a_pad_that_only_rounds_is_not_touch():
    """A hand's position that holds one level and then rises still moves past that level, so it is not bounded at
    rest; an idle pad read in whole numbers flickers by one step, which is rounding; a contact flag moves by one step
    when it closes, and still is touch."""
    n, rng = 200, np.random.default_rng(1)
    pos = np.full((n, 3), 1.1) + rng.normal(0, 0.002, (n, 3))
    pos[:60, 1] -= 0.05                      # the hand starts a little lower, then holds
    pos[120:, 1] += np.linspace(0, 0.1, 80)  # and rises
    rest = list(pos[60:120].mean(axis=0))
    assert not sg.touch_like(pos, rest, 0.7)
    pad = np.zeros((n, 96))
    pad[:, 5] = rng.random(n) < 0.3          # one cell flips between 0 and 1
    pad[:, 40] = 3 + (rng.random(n) < 0.5)   # one sits at 3 and flips to 4
    pad[0, 41] = 7
    assert sg.rounding_only(pad) and not sg.touch_like(pad)
    pressed = pad.copy()
    pressed[80:120, 10:30] = 40
    assert not sg.rounding_only(pressed) and sg.touch_like(pressed)
    flag = np.zeros((n, 1))
    flag[70:110] = 1
    assert sg.touch_like(flag)
    (c,) = lc.find({"right_contact": flag}, {}, np.arange(n) / 30.0)
    assert c["hand"] == "right"


def test_a_lerobot_feature_keys_dots_are_part_of_its_camera_words():
    """A LeRobot key has dots that are not an extension (observation.images.cam_high), so its camera words are cam and
    high, matching its depth key, while a video's extension is taken off."""
    words = formats.camera_words
    assert words("observation.images.cam_high") == words("observation.depth.cam_high")
    assert words("observation.images.cam_high") != words("observation.images.cam_left_wrist")
    assert words("exo_cam-images-rgb.mp4") == words("exo_cam-images-depth.mkv")


def test_a_depth_entry_creates_the_episode_folder_it_writes_into(tmp_path):
    """A LeRobot episode writes its depth before finish_episode makes the episode's folder."""
    ep = tmp_path / "not_yet" / "episode_000000"
    t = np.arange(4) / 30.0
    formats.depth_entry(ep, "top", tmp_path / "d.mkv", t, t, np.arange(4), 0.001, "observation.depth.cam_high")
    assert (ep / "depth_kmap_top.npy").exists()


def _lerobot_depth_of(tmp_path, *keys: str) -> tuple[dict, list]:
    """(lerobot_depth's entries, the cameras it left out) for one 16-bit depth video under each key, beside a scene and
    a left wrist camera."""
    videos = {}
    for key in keys:
        videos[key] = str(tmp_path / f"{key}.mkv")
        dw = formats.DepthWriter(Path(videos[key]))
        for k in range(6):
            dw.add(k / 30, np.full((48, 64), 1000 + k, np.uint16), None)
        assert dw.close()
    vmap = {"exo": "observation.images.cam_high", "left": "observation.images.cam_left_wrist"}
    unused = []
    dep, _ = formats.lerobot_depth(tmp_path / "ep", {"depth_cams": list(keys), "features": {k: {} for k in keys}},
                                   {"videos": videos}, vmap, 30.0, 6, unused)
    return dep, unused


def test_lerobot_depth_goes_with_the_camera_whose_name_has_its_words(tmp_path):
    """The rule the HDF5 reader uses (depth_camera): left_wrist's words are all in cam_left_wrist."""
    dep, _ = _lerobot_depth_of(tmp_path, "observation.depth.left_wrist")
    assert list(dep) == ["left"] and dep["left"]["source"] == "observation.depth.left_wrist"


def test_lerobot_depth_with_no_camera_of_its_own_goes_with_the_scene_camera_and_says_why(tmp_path):
    dep, _ = _lerobot_depth_of(tmp_path, "observation.depth.zed")
    assert list(dep) == ["exo"] and "scene camera" in dep["exo"]["source"]


def test_a_second_lerobot_depth_stream_for_one_camera_is_listed_as_left_out(tmp_path):
    """As the HDF5 reader lists it among the unused cameras, never dropped without a word."""
    dep, unused = _lerobot_depth_of(tmp_path, "observation.depth.cam_left_wrist", "observation.depth.left_wrist")
    assert dep["left"]["source"] == "observation.depth.cam_left_wrist"
    assert unused == ["observation.depth.left_wrist (depth with no camera of its own)"]


HOLD_THEN_RISE = np.r_[np.zeros(30), np.linspace(0, 0.3, 10), np.full(20, 0.3)][:, None]   # rests, then moves one way


def test_a_torso_joint_that_holds_and_then_rises_gives_no_contact():
    """A humanoid's torso joint that holds still and then moves one way rests and rises like a pressure pad; only its
    name says it is not touch."""
    t = np.arange(60) / 30.0
    assert lc.find({"observation.state.torso": HOLD_THEN_RISE}, {}, t) == []


def test_the_same_numbers_under_a_touch_name_give_a_contact():
    t = np.arange(60) / 30.0
    (c,) = lc.find({"left_pressure": HOLD_THEN_RISE}, {}, t)
    assert c["hand"] == "left" and c["signals"] == ["left_pressure"]


def test_actions_poses_odometry_and_gripper_effort_give_no_contacts():
    t = np.arange(60) / 30.0
    for name in ("action.delta_ee", "observation.ee_pose", "odom.position", "gripper_effort"):
        assert lc.find({name: HOLD_THEN_RISE}, {}, t) == [], name


def test_a_contact_flag_named_for_contact_gives_a_contact():
    t = np.arange(60) / 30.0
    flag = np.r_[np.zeros(20), np.ones(15), np.zeros(25)][:, None]
    (c,) = lc.find({"right_contact": flag}, {}, t)
    assert c["hand"] == "right"


def test_a_torso_joint_stays_in_the_per_instant_readout_and_a_pressure_does_not():
    """A touch signal's timing is given once as the episode's contacts, so its rows leave the readout; a joint's
    stay."""
    ep = {"signals": {"observation.state.torso": HOLD_THEN_RISE, "left_pressure": HOLD_THEN_RISE},
          "signal_meta": {}, "context": {"fps": 30}, "times": None}
    table = me._signals_table(ep, {"n": 60, "spans": [], "ks": [0, 20, 40, 59]})
    readout = table.split("    at: ")[1]
    assert "observation.state.torso:" in readout and "left_pressure:" not in readout


def _ramps(n: int, d: int) -> np.ndarray:
    """d values over n frames, value i rising from i to i + 1."""
    return np.arange(d)[None, :] + np.linspace(0, 1, n)[:, None]


def test_a_14_value_signal_gives_each_values_range_under_its_name():
    """habit's action_tcd has 14 values; each one's range is a claim the model can check, as main gave it."""
    names = [f"j{i}" for i in range(14)]
    line = sg.describe("action_tcd", _ramps(30, 14), names=names)
    assert "(14 values (" + ", ".join(names) + "))" in line
    assert line.count(" to ") == 14 and "13 to 14" in line and "values from" not in line


def test_a_26_value_signal_without_names_gives_each_values_range_by_position():
    line = sg.describe("observation.state", _ramps(30, 26))
    assert line.count(" to ") == 26 and line.split(": ", 1)[1].startswith("0 to 1, 1 to 2")


def test_a_9_value_constant_gives_its_values_in_the_same_at_every_frame_line():
    ep = {"signals": {"camera_info K": np.tile(np.arange(1.0, 10.0), (30, 1))}, "signal_meta": {},
          "context": {"fps": 30}, "times": None}
    table = me._signals_table(ep, {"n": 30, "spans": [], "ks": [0, 29]})
    assert "The same at every frame: camera_info K [1, 2, 3, 4, 5, 6, 7, 8, 9]" in table


def test_a_256_value_map_keeps_its_pooled_range():
    line = sg.describe("right_pressure", _ramps(30, 256), shape=[16, 16])
    assert "values from 0 to 256" in line


def test_loading_an_episode_keeps_each_signals_source_and_companion(tmp_path):
    """contacts.find groups hand-less signals by source and skips a fast sensor's variation companion; both read the
    meta load() returns."""
    ep = tmp_path / "episode_000000"
    ep.mkdir()
    n = 10
    np.savez(ep / "signals.npz", s0=np.zeros((n, 1)), s1=np.zeros((n, 1)))
    (ep / "sources.json").write_text(json.dumps({"right": {"n_frames": n, "base_s": 0.0}}))
    (ep / "context.json").write_text(json.dumps({
        "episode_id": ep.name, "profile": "handheld_gripper", "fps": 30, "n_state_frames": n, "state_kind": "none",
        "cameras": {}, "signals": [
            {"name": "pad", "key": "s0", "shape": [1], "source": "mcap channel /pad"},
            {"name": "pad variation within each frame", "key": "s1", "shape": [1], "variation_of": "pad"}]}))
    meta = me.load(ep)["signal_meta"]
    assert meta["pad"]["source"] == "mcap channel /pad"
    assert meta["pad variation within each frame"]["variation_of"] == "pad"


def test_loading_an_episode_keeps_every_field_of_a_signal_but_its_name_and_key(tmp_path):
    """A field a reader adds later reaches the checks without load() having to list it."""
    ep = tmp_path / "episode_000000"
    ep.mkdir()
    np.savez(ep / "signals.npz", s0=np.zeros((10, 1)))
    (ep / "sources.json").write_text(json.dumps({"right": {"n_frames": 10, "base_s": 0.0}}))
    (ep / "context.json").write_text(json.dumps({
        "episode_id": ep.name, "profile": "handheld_gripper", "fps": 30, "n_state_frames": 10, "state_kind": "none",
        "cameras": {}, "signals": [{"name": "pad", "key": "s0", "dims": 1, "gaps": 3, "units": "kPa"}]}))
    assert me.load(ep)["signal_meta"]["pad"] == {"dims": 1, "gaps": 3, "units": "kPa"}


def _h5_rig(path: Path, arrays: dict, n: int = 40) -> None:
    """An HDF5 episode of one 20 fps scene camera on a ns clock from boot, with these arrays beside it."""
    with h5py.File(path, "w") as f:
        f.create_dataset("observations/timestamps", data=(10**12 + np.arange(n) * 50_000_000).astype(np.int64))
        f.create_dataset("observations/images/cam_high",
                         data=np.random.default_rng(0).integers(0, 255, (n, 96, 128, 3), dtype=np.uint8))
        for k, v in arrays.items():
            f.create_dataset(k, data=v)


def test_an_hdf5_state_is_read_by_the_same_rule_as_a_lerobot_one(tmp_path):
    """ALOHA keeps two arms of six joints and a gripper in observations/qpos and the commands in action: the state
    and action of the episode, no longer signals. Its velocities stay a signal."""
    n = 40
    t = np.arange(n) / 20.0
    q = np.stack([0.3 * np.sin(t + j) for j in range(14)], axis=1)
    q[:, 6] = q[:, 13] = (t > 1).astype(float)
    root = tmp_path / "up"
    root.mkdir()
    _h5_rig(root / "episode_0.hdf5", {"observations/qpos": q, "observations/qvel": np.gradient(q, axis=0),
                                      "action": q + 0.01})
    rep = formats.convert(root, "teleop_arms", tmp_path / "eps", "aloha", 900)
    assert not rep["failed"]
    ep = tmp_path / "eps" / rep["episodes"][0]["episode_id"]
    ctx = json.loads((ep / "context.json").read_text())
    assert ctx["state_kind"] == "joints" and ctx["source"]["state"] == "observations/qpos"
    z = np.load(ep / "state.npz")
    assert z["state"].shape == (n, 14) and z["action"].shape == (n, 14)
    assert np.allclose(z["action"], q + 0.01, atol=1e-5)
    names = {s["name"] for s in ctx.get("signals") or []}
    assert "observations/qvel" in names and "observations/qpos" not in names and "action" not in names
    assert "RECORDED MOTION" in me.build_request(ep)["prompt"]


def test_an_hdf5_array_named_for_joint_positions_names_every_value_a_joint(tmp_path):
    """DROID keeps a Franka's seven joints in robot_state/joint_positions and its gripper on its own; the array's name
    says every value is a joint, so seven of them stay a signal, with the reason as the state note."""
    n = 40
    t = np.arange(n) / 20.0
    root = tmp_path / "up"
    root.mkdir()
    joints = np.stack([np.sin(t + j) for j in range(7)], axis=1)
    _h5_rig(root / "droid.hdf5", {"observations/robot_state/joint_positions": joints,
                                  "observations/robot_state/gripper_position": np.linspace(0, 1, n)[:, None]})
    rep = formats.convert(root, "teleop_arms", tmp_path / "eps", "droid", 900)
    ctx = json.loads((tmp_path / "eps" / rep["episodes"][0]["episode_id"] / "context.json").read_text())
    assert ctx["state_kind"] == "none" and "7 joints and no gripper" in ctx["state_note"]
    assert any(s["name"].endswith("robot_state/joint_positions") for s in ctx["signals"])


def test_the_note_on_an_unnamed_joint_positions_array_says_its_name_makes_every_value_a_joint(tmp_path):
    """An unnamed joint_positions array had joint 1 to joint N made up as its value names, and the note said the
    recorded state's value names give 14 joints although the file names no value. The note says what is true, that
    the array's name says every value is a joint, and no made up name reaches it."""
    n = 40
    t = np.arange(n) / 20.0
    for dims in (7, 14):
        root = tmp_path / f"up{dims}"
        root.mkdir()
        joints = np.stack([np.sin(t + j) for j in range(dims)], axis=1)
        _h5_rig(root / "droid.hdf5", {"observations/robot_state/joint_positions": joints})
        rep = formats.convert(root, "teleop_arms", tmp_path / f"eps{dims}", "droid", 900)
        ctx = json.loads((tmp_path / f"eps{dims}" / rep["episodes"][0]["episode_id"] / "context.json").read_text())
        assert ctx["state_kind"] == "none"
        assert ctx["state_note"] == (
            f"Labelled from the video: the array's name says every value is a joint, so its {dims} values are {dims} "
            "joints and no gripper, and our checks read six joints and a gripper per arm. The recorded state is the "
            "HDF5 array robot_state/joint_positions."), ctx["state_note"]
        assert "value names" not in ctx["state_note"] and "joint 1" not in json.dumps(ctx)


def _aloha(n: int = 40) -> np.ndarray:
    """Two arms of six joints and a gripper over n frames at 20 fps, each gripper closing after 1 s."""
    t = np.arange(n) / 20.0
    q = np.stack([0.3 * np.sin(t + j) for j in range(14)], axis=1)
    q[:, 6] = q[:, 13] = (t > 1).astype(float)
    return q


def _convert(tmp_path: Path, arrays: dict) -> tuple[dict, Path]:
    root = tmp_path / "up"
    root.mkdir(parents=True)
    _h5_rig(root / "episode_0.hdf5", arrays)
    rep = formats.convert(root, "teleop_arms", tmp_path / "eps", "aloha", 900)
    assert not rep["failed"]
    ep = tmp_path / "eps" / rep["episodes"][0]["episode_id"]
    return json.loads((ep / "context.json").read_text()), ep


@pytest.mark.parametrize("hz, late_s", [(100, 0.0), (100, 0.06), (100, 0.3), (20, 0.2)])
def test_an_hdf5_state_on_its_own_clock_is_placed_as_an_mcap_arm_is(tmp_path, hz, late_s):
    """An arm logged on its own clock that starts a little after the camera leaves the first frames with no reading;
    they hold its first reading, as joint_state places an MCAP arm's channel that starts within STATE_EDGE_SLACK_S."""
    n, cam_ns = 40, 10**12
    m = int(round(2.0 * hz))
    t = np.arange(m) / hz + late_s
    s = np.stack([0.3 * np.sin(t + j) for j in range(14)], axis=1)
    s[:, 6] = s[:, 13] = (t > 1).astype(float)
    ctx, ep = _convert(tmp_path, {"observations/arm/timestamps": (cam_ns + t * 1e9).astype(np.int64),
                                  "observations/arm/qpos": s})
    # every array sits under observations/, which h5_streams leaves off the names
    assert ctx["state_kind"] == "joints" and ctx["source"]["state"] == "arm/qpos"
    z = np.load(ep / "state.npz")["state"]
    assert z.shape == (n, 14) and np.isfinite(z).all()
    # the first frames hold the first frame's reading (at 100 Hz, the mean of the samples in its interval)
    assert np.allclose(z[0], s[0], atol=0.02)


def test_an_hdf5_state_that_does_not_cover_the_footage_is_named_in_the_note(tmp_path):
    """An arm that starts 0.7 s after the camera is past what an MCAP arm may miss: the episode is labelled from the
    video, and the note says which array it is."""
    hz = 100
    t = np.arange(200) / hz + 0.7
    s = np.stack([0.3 * np.sin(t + j) for j in range(14)], axis=1)
    ctx, _ = _convert(tmp_path, {"observations/arm/timestamps": (10**12 + t * 1e9).astype(np.int64),
                                 "observations/arm/qpos": s})
    assert ctx["state_kind"] == "none" and "recorded state arm/qpos" in ctx["state_note"]


def test_each_array_named_as_the_state_is_tried_and_the_first_one_laid_out_is_read(tmp_path):
    """A shorter name is tried first, but seven joints (joint_pos) or 45 simulator values (states) are not the layout,
    so ALOHA's qpos beside them is the state; alone, either is named in the note that says why it was not read."""
    q = _aloha()
    seven = np.stack([np.sin(np.arange(40) / 20.0 + j) for j in range(7)], axis=1)
    sim = np.random.default_rng(1).normal(size=(40, 45))
    for i, other in enumerate([{"joint_pos": seven}, {"states": sim}]):
        ctx, _ = _convert(tmp_path / f"with_{i}", {"observations/qpos": q, **other})
        assert ctx["state_kind"] == "joints" and ctx["source"]["state"] == "observations/qpos"
        assert set(other) <= {x["name"] for x in ctx["signals"]}
    ctx, _ = _convert(tmp_path / "alone", {"states": sim})
    assert ctx["state_kind"] == "none" and "45 values" in ctx["state_note"] and "array states" in ctx["state_note"]


def test_an_array_under_an_action_group_is_never_the_state(tmp_path):
    """DROID keeps the commanded joints in action/joint_position: a command, not the arm's state. An action/qpos of
    two arms of six joints and a gripper, the layout the checks read, is a command all the same."""
    for i, name in enumerate(["action/joint_position", "action/qpos"]):
        ctx, _ = _convert(tmp_path / str(i), {name: _aloha()})
        assert ctx["state_kind"] == "none" and "state_note" not in ctx
        assert name in {x["name"] for x in ctx["signals"]}


def test_a_state_with_values_that_are_not_numbers_says_so(tmp_path):
    """A qpos holding an infinity is left out of the signals for it, and the note says that, not that it could not be
    placed on the frames."""
    q = _aloha()
    q[5, 2] = np.inf
    ctx, _ = _convert(tmp_path, {"observations/qpos": q})
    assert ctx["state_kind"] == "none"
    # every array sits under observations/, which h5_streams leaves off the names
    assert "recorded state qpos has values that are not all finite numbers" in ctx["state_note"]


def test_a_long_gap_in_an_hdf5_state_is_not_drawn_as_motion(tmp_path):
    """A qpos with no reading for 1 s (a recorder that stopped) had the gap filled by a straight line and shown as
    recorded motion. A gap longer than STATE_EDGE_SLACK_S leaves the state unread, and the note names the array and
    the gap's time; the array stays a signal with no reading there. A gap of three frames is filled as before."""
    q = _aloha()
    q[11:30] = np.nan
    ctx, _ = _convert(tmp_path / "long", {"observations/qpos": q})
    assert ctx["state_kind"] == "none"
    assert ctx["state_note"] == ("Labelled from the video, because the recorded state qpos has no reading from 0.5 s "
                                 "to 1.5 s, a gap longer than the 0.5 s the reader fills."), ctx["state_note"]
    assert "qpos" in {s["name"] for s in ctx["signals"]}
    q = _aloha()
    q[20:23] = np.nan
    ctx, ep = _convert(tmp_path / "short", {"observations/qpos": q})
    assert ctx["state_kind"] == "joints" and not ctx.get("state_note")
    z = np.load(ep / "state.npz")["state"]
    assert np.isfinite(z).all() and np.allclose(z[21], (q[19] + q[23]) / 2, atol=1e-5)


def test_an_action_of_another_width_stays_a_signal(tmp_path):
    """The action goes with the state only when it has the state's shape; a 7 value action beside 14 values of state
    is something else, and stays a signal."""
    ctx, ep = _convert(tmp_path, {"observations/qpos": _aloha(), "action": np.zeros((40, 7))})
    assert ctx["state_kind"] == "joints" and "action" not in np.load(ep / "state.npz")
    assert "action" in {x["name"] for x in ctx["signals"]}
