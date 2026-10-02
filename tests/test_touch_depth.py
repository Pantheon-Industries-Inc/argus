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


def _press(n: int, on: tuple[int, int], rest: float = 3072.0) -> np.ndarray:
    """A 16 x 16 glove map that rests at its untouched reading and falls where it is pressed, frames on[0]..on[1]."""
    a = np.full((n, 16, 16), rest, np.float32)
    a += np.random.default_rng(0).normal(0, 2, a.shape).astype(np.float32)
    a[on[0]:on[1], 3:8, 5:10] = 900.0
    return a


def _hdf5(path: Path, demos: int = 2, n: int = 40) -> None:
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
            g.create_dataset("right_pressure", data=_press(n, (12, 26)))
            g.create_dataset("right_hand_landmarks", data=np.random.default_rng(d).normal(0, 0.1, (n, 21, 3)))
            if d == 0:
                g.create_dataset("labels", data=np.array([(0, 1)], dtype=[("low_light", "u1"), ("hand_out_of_frame", "u1")]))


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
    time the frames, while a frame rate's name beside it does."""
    assert _h5_rate(tmp_path, {"sensor_config": json.dumps({"sensors": {"imu": {"rate": 200}}})}) is None
    assert _h5_rate(tmp_path, {"sensor_config": json.dumps({"audio": {"hz": 48}})}) is None
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


def test_json_nested_deeper_than_the_parser_allows_is_passed_over_and_deep_json_is_searched(tmp_path):
    """json.loads stops at about 1000 levels; the search through what it parsed has no depth limit of its own."""
    assert _h5_rate(tmp_path, {"env_args": "[" * 100000}) is None
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
    (c,) = lc.find({"left_glove": press}, {"left_glove": {"shape": [16, 16]}}, t)
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


def test_a_lerobot_feature_key_keeps_its_camera_words_and_depth_writes_its_own_folder(tmp_path):
    """A LeRobot key has dots that are not an extension (observation.images.cam_high), so its camera words are cam and
    high, matching its depth key; and the depth key map is written even before the episode folder exists."""
    assert formats.camera_words("observation.images.cam_high") == formats.camera_words("observation.depth.cam_high")
    assert formats.camera_words("observation.images.cam_high") != formats.camera_words("observation.images.cam_left_wrist")
    assert formats.camera_words("exo_cam-images-rgb.mp4") == formats.camera_words("exo_cam-images-depth.mkv")
    ep = tmp_path / "not_yet" / "episode_000000"
    n = 4
    t = np.arange(n) / 30.0
    e, _ = formats.depth_entry(ep, "top", tmp_path / "d.mkv", t, t, np.arange(n), 0.001, "observation.depth.cam_high")
    assert (ep / "depth_kmap_top.npy").exists()


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
