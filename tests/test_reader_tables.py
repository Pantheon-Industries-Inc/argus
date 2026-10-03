"""The reader's tables, archives and sensor files under the No drop rule (prepare/formats.py): a sensor MCAP cut short
keeps every message before the cut, a damaged archive loses only its damaged member, a table is read whatever its
delimiter or a stray text cell, its placement on the frames says when it is assumed, a slow table keeps its unit and
its rate, and a bad cell is flagged where it is. Small synthetic files only."""
from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pytest

from prepare import formats as f

T0 = 1_790_000_000.0


def _clip(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _anchor(n: int) -> dict:
    return {"pts": np.arange(n, dtype=np.int64), "time_base": 1 / 30.0}


def _issues(x, kind: str) -> list[dict]:
    got = x.get("reader_issues", []) if isinstance(x, dict) and not isinstance(x, f.Signals) else x.issues
    return [i for i in got if i["kind"] == kind]


# ---------------------------------------------------------------- a sensor MCAP cut short

def _arm_mcap(path: Path, seconds: float = 4.0, hz: float = 100.0, topic: str = "/yam_left/joint_state",
              t0: float = T0, compressed: bool = True) -> None:
    from mcap.writer import CompressionType, Writer
    with open(path, "wb") as fh:
        w = Writer(fh, chunk_size=2048, **({} if compressed else {"compression": CompressionType.NONE}))
        w.start()
        sid = w.register_schema(name="arm", encoding="jsonschema", data=b"{}")
        ch = w.register_channel(topic=topic, message_encoding="json", schema_id=sid)
        for i in range(int(seconds * hz)):
            t = int((t0 + i / hz) * 1e9)
            msg = {"joint_pos": [0.01 * i] * 6, "gripper_pos": [0.5]}
            w.add_message(ch, log_time=t, publish_time=t, data=json.dumps(msg).encode())
        w.finish()


def test_a_sensor_mcap_cut_short_is_placed_by_the_messages_before_the_cut_and_flagged(tmp_path):
    p = tmp_path / "yam_left.mcap"
    _arm_mcap(p)
    b = p.read_bytes()
    p.write_bytes(b[: len(b) // 2])
    t = f.sensor_times(p)
    assert t is not None and 0.5 < t[-1] - t[0] < 3.5
    q = T0 + np.arange(120) / 30.0
    by_clock, assumed, unplaced = f.split_sensors({"state": [p], "state_shared": []}, q, True)
    assert by_clock == [p] and not unplaced
    extra, sig = {}, f.Signals()
    f.note_sensors(extra, sig, by_clock, assumed, unplaced, q=q)
    cut = _issues(sig, "sensor_file_cut")
    assert len(cut) == 1 and "yam_left.mcap" in cut[0]["what"]
    assert cut[0]["t0_s"] == pytest.approx(0.0, abs=0.05) and cut[0]["t1_s"] == pytest.approx(t[-1] - T0, abs=0.05)


def test_a_whole_sensor_mcap_is_not_flagged_as_cut(tmp_path):
    p = tmp_path / "yam_left.mcap"
    _arm_mcap(p)
    q = T0 + np.arange(120) / 30.0
    by_clock, assumed, unplaced = f.split_sensors({"state": [p], "state_shared": []}, q, True)
    sig = f.Signals()
    f.note_sensors({}, sig, by_clock, assumed, unplaced, q=q)
    assert by_clock == [p] and not _issues(sig, "sensor_file_cut")


def test_a_sensor_file_that_cannot_be_opened_is_not_said_to_have_no_time(tmp_path):
    p = tmp_path / "glove.h5"
    p.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\0" * 64)
    _, _, unplaced = f.split_sensors({"state": [p], "state_shared": []}, T0 + np.arange(30) / 30.0, True)
    assert unplaced and "could not be opened" in unplaced[0][1]


def test_a_cut_sensor_files_span_reads_the_same_in_its_numbers_and_its_words(tmp_path):
    p = tmp_path / "yam_left.mcap"
    _arm_mcap(p)
    b = p.read_bytes()
    p.write_bytes(b[: len(b) // 2])
    q = T0 + 0.007 + np.arange(120) / 30.0          # the footage starts 7 ms after the file's first message
    by_clock, assumed, unplaced = f.split_sensors({"state": [p], "state_shared": []}, q, True)
    sig = f.Signals()
    f.note_sensors({}, sig, by_clock, assumed, unplaced, q=q)
    cut = _issues(sig, "sensor_file_cut")[0]
    assert cut["t0_s"] == 0.0
    assert f"cover {cut['t0_s']:.1f} s to {cut['t1_s']:.1f} s" in cut["what"]


def test_a_cut_sensor_file_is_scanned_once(tmp_path, monkeypatch):
    p = tmp_path / "yam_left.mcap"
    _arm_mcap(p)
    b = p.read_bytes()
    p.write_bytes(b[: len(b) // 2])
    scans, scan = [], f._mcap_stream
    monkeypatch.setattr(f, "_mcap_stream", lambda path, topics=None: scans.append(path) or scan(path, topics))
    q = T0 + np.arange(120) / 30.0
    by_clock, assumed, unplaced = f.split_sensors({"state": [p], "state_shared": []}, q, True)
    f.note_sensors({}, f.Signals(), by_clock, assumed, unplaced, q=q)
    assert len(scans) == 1


def _two_arms(folder: Path, cut_left: float) -> list[Path]:
    """yam_left.mcap and yam_right.mcap, 4 s of each arm's six joints and gripper at 100 Hz, the left file cut to
    cut_left of its bytes."""
    out = []
    for side in ("left", "right"):
        p = folder / f"yam_{side}.mcap"
        _arm_mcap(p, topic=f"/yam_{side}/joint_state")
        out.append(p)
    b = out[0].read_bytes()
    out[0].write_bytes(b[: int(len(b) * cut_left)])
    return out


def test_an_arm_in_a_cut_sensor_file_is_read_up_to_its_cut(tmp_path):
    q = T0 + np.arange(60) / 30.0                   # 2 s of footage, inside what the cut left file still holds
    streams = f.mcap_joint_streams(_two_arms(tmp_path, 0.75), q)
    assert {"/yam_left/joint_state", "/yam_right/joint_state"} <= set(streams)
    state, _, note = f.joint_state(streams, q)
    assert state is not None and state.shape == (60, 14) and note is None


def test_an_arm_cut_short_inside_the_footage_is_named_in_the_state_note(tmp_path):
    q = T0 + np.arange(120) / 30.0                  # 4 s of footage, the left arm's file cut about half way
    state, _, note = f.joint_state(f.mcap_joint_streams(_two_arms(tmp_path, 0.5), q), q)
    assert state is None and "/yam_left/joint_state" in note and "does not cover" in note


def _recorder_with(tmp_path, name: str, third_arm: bool = False, cut: bool = True) -> dict:
    """recorder_folder's episode with the sensor MCAP name cut before its first message (written as a whole file of
    no message when not cut), converted; its context.json, and its folder under "_ep"."""
    from mcap.writer import Writer
    from test_formats import recorder_folder
    d = recorder_folder(tmp_path / "upload", third_arm=third_arm)
    p = d / name
    if cut:
        p.write_bytes((p if p.exists() else d / "yam_left.mcap").read_bytes()[:64])
    else:
        with open(p, "wb") as fh:
            w = Writer(fh)
            w.start()
            w.finish()
    rep = f.convert(tmp_path / "upload", "teleop_arms", tmp_path / "out" / "eps", "t", 900)
    ep = tmp_path / "out" / "eps" / rep["episodes"][0]["episode_id"]
    return json.loads((ep / "context.json").read_text()) | {"_ep": ep}


def test_an_arm_file_that_cannot_be_read_leaves_the_episode_with_no_arm_state(tmp_path):
    """A follower arm's file cut before its first message had left a state of the other arm alone, so the model was
    told one arm works on a two arm rig."""
    from label import episode as me
    ctx = _recorder_with(tmp_path, "yam_left.mcap")
    assert ctx["state_kind"] == "none" and not (ctx["_ep"] / "state.npz").exists()
    assert "yam_left.mcap is cut short before its first message" in ctx["state_note"]
    assert "What it records is unknown, so the other sensor files are not read as the arm state" in ctx["state_note"]
    assert ctx["state_why"] == "unreadable"
    assert "RECORDED STATE: no arm state" in me.build_request(ctx["_ep"])["prompt"]


@pytest.mark.parametrize("name", ["gelsight_pad.mcap", "yam_leader_left.mcap"])
def test_a_sensor_file_that_cannot_be_read_is_named_and_never_called_an_arm(tmp_path, name):
    ctx = _recorder_with(tmp_path, name)
    assert ctx["state_kind"] == "none" and name in ctx["state_note"] and ctx["state_why"] == "unreadable"
    assert f"{name} is cut short before its first message, so nothing in it could be read. What it records is " \
           "unknown" in ctx["state_note"]


def test_a_whole_sensor_file_with_no_message_takes_nothing_from_the_arm_state(tmp_path):
    ctx = _recorder_with(tmp_path, "health_log.mcap", cut=False)
    assert ctx["state_kind"] == "joints" and "health_log.mcap" not in (ctx.get("state_note") or "")


def test_a_third_arm_file_that_cannot_be_read_puts_no_camera_on_a_third_arm(tmp_path):
    ctx = _recorder_with(tmp_path, "yam_camera.mcap", third_arm=True)
    assert ctx["state_kind"] == "none" and "yam_camera.mcap" in ctx["state_note"]
    assert "third arm" not in ctx["state_note"] and "third arm" not in (ctx["cameras"]["exo"].get("desc") or "")


def test_a_sensor_file_copied_over_at_the_same_size_and_time_is_read_again(tmp_path):
    import os
    import shutil
    p, later = tmp_path / "yam_left.mcap", tmp_path / "later.mcap"
    _arm_mcap(p, compressed=False)          # uncompressed, so both files are the same size
    _arm_mcap(later, t0=T0 + 100.0, compressed=False)
    st = os.stat(p)
    assert st.st_size == os.stat(later).st_size
    assert f.sensor_times(p)[0] - T0 == pytest.approx(0.0, abs=0.01)
    shutil.copyfile(later, p)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert f.sensor_times(p)[0] - T0 == pytest.approx(100.0, abs=0.01)


def test_an_indexed_mcap_damaged_inside_is_flagged_with_the_span_read(tmp_path):
    p = tmp_path / "yam_left.mcap"
    _arm_mcap(p)
    b = bytearray(p.read_bytes())
    b[len(b) // 2:len(b) // 2 + 400] = b"\xff" * 400
    p.write_bytes(bytes(b))
    assert not f.sensor_cut(p)
    out = f.mcap_signals([p], T0 + np.arange(120) / 30.0)
    hit = _issues(out, "mcap_file_damaged")
    assert len(hit) == 1 and "yam_left.mcap" in hit[0]["what"]
    assert hit[0]["t0_s"] == 0.0 and hit[0]["t1_s"] == pytest.approx(119 / 30)
    assert not hit[0]["footage_complete"] and hit[0]["unreadable_spans_s"]
    assert f"{hit[0]['t1_s']:.1f} s of the footage" in hit[0]["what"]


def _damage_chunk_after(p: Path, seconds: float) -> None:
    """Bytes overwritten inside the first chunk of an MCAP file whose messages start past seconds after T0, its
    summary at its end left whole, so every message before that chunk still reads."""
    from mcap.reader import make_reader
    with open(p, "rb") as fh:
        chunks = make_reader(fh).get_summary().chunk_indexes
    at = min(c.chunk_start_offset for c in chunks if c.message_start_time > (T0 + seconds) * 1e9)
    b = bytearray(p.read_bytes())
    b[at + 40:at + 240] = b"\xff" * 200
    p.write_bytes(bytes(b))


def test_file_damage_past_the_footage_stays_visible_without_claiming_missing_footage(tmp_path):
    p = tmp_path / "yam_left.mcap"
    _arm_mcap(p)
    _damage_chunk_after(p, 3.0)                     # the footage ends at 2 s
    q = T0 + np.arange(60) / 30.0
    (issue,) = _issues(f.mcap_signals([p], q), "mcap_file_damaged")
    assert issue["footage_complete"] and "cover the whole footage" in issue["what"]
    lost = []
    streams = f.mcap_joint_streams([p], q, lost)
    assert lost == [] and "/yam_left/joint_state" in streams


def test_a_damaged_mcaps_span_is_given_within_the_footage(tmp_path):
    p = tmp_path / "yam_left.mcap"
    _arm_mcap(p, t0=T0 + 1.0)                       # the file starts 1 s into the footage
    _damage_chunk_after(p, 3.0)
    q = T0 + np.arange(60) / 30.0                   # 2 s of footage
    (hit,) = _issues(f.mcap_signals([p], q), "mcap_file_damaged")
    assert hit["t0_s"] == pytest.approx(1.0, abs=0.02) and hit["t1_s"] == pytest.approx(q[-1] - q[0])
    assert f"cover 1.0 s to {q[-1] - q[0]:.1f} s of the footage" in hit["what"]


def test_a_cut_sensor_files_span_is_given_within_the_footage(tmp_path):
    p = tmp_path / "yam_left.mcap"
    _arm_mcap(p, t0=T0 + 1.0)
    b = p.read_bytes()
    p.write_bytes(b[: int(len(b) * 0.9)])
    q = T0 + np.arange(60) / 30.0
    sig = f.Signals()
    f.note_sensors({}, sig, [p], [], [], q=q)
    (cut,) = _issues(sig, "sensor_file_cut")
    assert cut["t1_s"] == pytest.approx(q[-1] - q[0])


def _recorder(tmp_path, change) -> dict:
    """recorder_folder's episode of 60 frames, its files changed by change(folder), converted; its context.json and
    its folder under "_ep"."""
    from test_formats import recorder_folder
    d = recorder_folder(tmp_path / "upload", n=60)
    change(d)
    rep = f.convert(tmp_path / "upload", "teleop_arms", tmp_path / "out" / "eps", "t", 900)
    ep = tmp_path / "out" / "eps" / rep["episodes"][0]["episode_id"]
    return json.loads((ep / "context.json").read_text()) | {"_ep": ep}


def _damage_whole_chunk(d: Path, name: str = "yam_left.mcap") -> None:
    """Bytes overwritten inside the file's one chunk, its summary at its end left whole: no message reads."""
    p = d / name
    b = bytearray(p.read_bytes())
    b[len(b) // 3:len(b) // 3 + 200] = b"\xff" * 200
    p.write_bytes(bytes(b))


def _signal_names(ctx: dict) -> set:
    return {s["name"] for s in ctx.get("signals") or []}


def test_an_arm_file_damaged_inside_before_any_message_leaves_no_arm_state(tmp_path):
    """A follower's file whose summary is whole but whose one chunk does not read had left a state of the other arm
    alone, its file still named as a source of the state."""
    ctx = _recorder(tmp_path, _damage_whole_chunk)
    assert ctx["state_kind"] == "none" and not (ctx["_ep"] / "state.npz").exists()
    assert ctx["state_why"] == "unreadable" and "state" not in ctx["source"]
    assert "yam_left.mcap is damaged inside" in ctx["state_note"]
    assert "declared channels" in ctx["state_note"] and "/yam_left/joint_state" in ctx["state_note"]
    assert "/yam_right/joint_state joint_pos" in _signal_names(ctx)


def _on_its_own_clock(d: Path, name: str, topic: str, gripper: bool = True) -> None:
    """recorder_folder's sensor file name written again with log times that count from 0, as a recorder without a
    wall clock writes them: an arm's six joints and its gripper (none when not gripper) on topic."""
    from mcap.writer import Writer
    with open(d / name, "wb") as fh:
        w = Writer(fh)
        w.start()
        sid = w.register_schema(name=name, encoding="jsonschema", data=b"{}")
        ch = w.register_channel(topic=topic, message_encoding="json", schema_id=sid)
        for i in range(240):
            t = int(i / 120 * 1e9) + 1
            msg = {"joint_pos": [0.1 * i / 60] * 6, **({"gripper_pos": [0.5]} if gripper else {})}
            w.add_message(ch, log_time=t, publish_time=t, data=json.dumps(msg).encode())
        w.finish()


def test_an_arm_file_on_a_clock_of_its_own_beside_arms_on_the_footages_clock_leaves_no_arm_state(tmp_path):
    """A follower's file whose log times count from 0, beside files on the footage's clock, is placed from both
    starts and had left a state of the other arm alone."""
    ctx = _recorder(tmp_path, lambda d: _on_its_own_clock(d, "yam_left.mcap", "/yam_left/joint_state"))
    assert ctx["state_kind"] == "none" and ctx["state_why"] == "assumed_clock"
    assert "yam_left.mcap records an arm (/yam_left/joint_state) on a clock the footage does not share" \
        in ctx["state_note"]
    assert {"/yam_right/joint_state joint_pos", "/yam_left/joint_state joint_pos"} <= _signal_names(ctx)


def test_commands_on_a_clock_of_their_own_leave_the_arms_state_without_an_action(tmp_path):
    """A leader's commands give the action, never the state, so on a clock of their own they cost only the action,
    as commands that do not cover the footage do."""
    ctx = _recorder(tmp_path, lambda d: _on_its_own_clock(d, "yam_leader_left.mcap", "/yam_leader_left/joint_pos"))
    z = np.load(ctx["_ep"] / "state.npz")
    assert ctx["state_kind"] == "joints" and z["state"].shape == (60, 14) and "action" not in z.files


def test_a_third_arm_on_a_clock_of_its_own_leaves_the_working_arms_state(tmp_path):
    from test_formats import recorder_folder
    d = recorder_folder(tmp_path / "upload", n=60, third_arm=True)
    _on_its_own_clock(d, "yam_camera.mcap", "/yam_camera/joint_state", gripper=False)
    rep = f.convert(tmp_path / "upload", "teleop_arms", tmp_path / "out" / "eps", "t", 900)
    ctx = json.loads((tmp_path / "out" / "eps" / rep["episodes"][0]["episode_id"] / "context.json").read_text())
    assert ctx["state_kind"] == "joints" and "state_why" not in ctx


def _follower_in_chunks(d: Path, keep: float = 1.0) -> None:
    """recorder_folder's yam_left.mcap written again in small chunks, its joint and health messages sharing their log
    times as the recorder writes them, then cut to keep of its bytes."""
    from mcap.writer import CompressionType, Writer
    p = d / "yam_left.mcap"
    with open(p, "wb") as fh:
        w = Writer(fh, chunk_size=2048, compression=CompressionType.NONE)
        w.start()
        sid = w.register_schema(name="yam_left", encoding="jsonschema", data=b"{}")
        ch = w.register_channel(topic="/yam_left/joint_state", message_encoding="json", schema_id=sid)
        hc = w.register_channel(topic="/yam_left/health", message_encoding="json", schema_id=sid)
        for i in range(240):
            t = int((1_790_000_000.0 - 0.05 + i / 120) * 1e9)
            msg = {"joint_pos": [0.1 * i / 60] * 6, "joint_vel": [0.0] * 6, "gripper_pos": [0.5]}
            w.add_message(ch, log_time=t, publish_time=t, data=json.dumps(msg).encode())
            w.add_message(hc, log_time=t, publish_time=t, data=b'{"ok": true}')
        w.finish()
    p.write_bytes(p.read_bytes()[: int(p.stat().st_size * keep)])


def test_an_arm_file_whose_channels_share_their_stamps_is_placed_by_its_clock(tmp_path):
    """The follower's joint and health messages share their log times, so half of its steps are 0 and its median
    step was 0: cut short, the file looked clockless, was placed from both starts and left a state of the other arm
    alone. It is placed by its clock and its arm read up to the cut, which here is past the footage."""
    ctx = _recorder(tmp_path, lambda d: _follower_in_chunks(d, keep=0.97))
    assert ctx["source"]["sensor_files"]["yam_left.mcap"] == "placed by its own clock"
    assert ctx["state_kind"] == "joints" and np.load(ctx["_ep"] / "state.npz")["state"].shape == (60, 14)
    assert [i["kind"] for i in ctx["reader_issues"]] == ["sensor_file_cut"]


def test_an_arm_file_cut_inside_the_footage_gives_its_short_span_as_why(tmp_path):
    ctx = _recorder(tmp_path, lambda d: _follower_in_chunks(d, keep=0.5))
    assert ctx["source"]["sensor_files"]["yam_left.mcap"] == "placed by its own clock"
    assert ctx["state_kind"] == "none" and ctx["state_why"] == "short"
    assert "/yam_left/joint_state has readings from 0.0 s" in ctx["state_note"]


def test_a_clock_with_repeated_stamps_steps_by_its_distinct_times():
    t = np.repeat(T0 + np.arange(100) / 100.0, 2)   # two channels written at each instant
    assert f._clock_facts(t)[1] == pytest.approx(0.01) and f.recorder_clock(t)


def test_the_note_on_an_unreadable_sensor_file_names_the_other_files_only_when_there_are_some():
    bad = [("yam_left.mcap", "is cut short before its first message, so nothing in it could be read")]
    alone = f.unread_sensors_note(bad, None)
    assert alone == ("Labelled from the cameras, because yam_left.mcap is cut short before its first message, so "
                     "nothing in it could be read. What it records is unknown.")
    assert alone.why == "unreadable"
    assert f.unread_sensors_note(bad, "the other sensor files").endswith(
        "What it records is unknown, so the other sensor files are not read as the arm state, and their channels "
        "are given as signals.")
    assert f.unread_sensors_note([], "the other sensor files") is None


def test_a_gap_inside_a_stream_is_given_with_the_limit_it_broke():
    q = np.arange(120) / 30.0
    t = np.concatenate([np.arange(0, 1.0, 0.25), np.arange(3.0, 4.01, 0.25)])     # 4 Hz, a 2 s hole
    rows, gap = f.fill_rows(q, t, np.ones((len(t), 7)))
    assert rows is None and gap[2] == pytest.approx(0.75)                       # three steps of 0.25 s
    assert "a gap longer than the 0.75 s the reader fills" in f.gap_words(gap, 0.0)


# ---------------------------------------------------------------- a damaged archive

_RNG = np.random.default_rng(0)
MEMBERS = {f"take/a_camera{k}.mp4": _RNG.bytes(20000) for k in (1, 2, 3)}     # incompressible, so a cut lands in the last


def _zip_one_encrypted(path: Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for k, v in MEMBERS.items():
            z.writestr(k, v)
    b = bytearray(path.read_bytes())
    # mark the last member encrypted in the central directory, as zip -P writes it
    cd = b.rfind(b"PK\x01\x02")
    b[cd + 8] |= 1
    path.write_bytes(bytes(b))


def _tar_cut(path: Path, mode: str) -> None:
    with tarfile.open(path, mode) as tf:
        for k, v in MEMBERS.items():
            ti = tarfile.TarInfo(k)
            ti.size = len(v)
            tf.addfile(ti, io.BytesIO(v))
    b = path.read_bytes()
    path.write_bytes(b[: int(len(b) * 0.8)])


@pytest.mark.parametrize("kind", ["zip", "tar", "tar.gz"])
def test_a_damaged_archive_keeps_its_good_members_and_names_the_bad_one(tmp_path, kind):
    a = tmp_path / f"upload.{kind}"
    if kind == "zip":
        _zip_one_encrypted(a)
    else:
        _tar_cut(a, "w" if kind == "tar" else "w:gz")
    root, notes = f.open_archives(a, tmp_path / "unpacked")
    got = sorted(p.relative_to(root).as_posix() for p in f.files_under(root))
    assert got == ["take/a_camera1.mp4", "take/a_camera2.mp4"]
    assert (root / "take" / "a_camera1.mp4").read_bytes() == MEMBERS["take/a_camera1.mp4"]
    words = " ".join(notes)
    assert "a_camera3.mp4" in words
    assert ("password" in words) if kind == "zip" else ("cut short" in words)


def test_a_refused_upload_carries_what_the_archive_says(tmp_path):
    a = tmp_path / "upload.zip"
    with zipfile.ZipFile(a, "w", zipfile.ZIP_STORED) as z:
        z.writestr("take/a_camera1.mp4", b"A" * 1000)
    b = bytearray(a.read_bytes())
    b[b.rfind(b"PK\x01\x02") + 8] |= 1
    a.write_bytes(bytes(b))
    with pytest.raises(ValueError) as e:
        f.convert(a, "handheld_gripper", tmp_path / "out" / "eps", "t", 900)
    assert "password" in str(e.value) and "a_camera1.mp4" in str(e.value)
    assert "it is empty" not in str(e.value)


# ---------------------------------------------------------------- tables read, or named

def test_one_text_cell_keeps_its_column_and_is_flagged(tmp_path):
    n = 60
    df = pd.DataFrame({"x": np.arange(n) * 0.1, "y": np.ones(n)}).astype(object)
    df.loc[10, "x"] = "ERR"
    df.to_csv(tmp_path / "traj.csv", index=False)
    out = f.table_signals([tmp_path / "traj.csv"], None, _anchor(n), {})
    assert out.meta["traj"]["names"] == ["x", "y"]
    assert np.isnan(out["traj"][10, 0]) and out["traj"][11, 0] == pytest.approx(1.1)
    bad = _issues(out, "signal_bad_cells")
    assert len(bad) == 1 and "x" in bad[0]["what"] and "1 of 60" in bad[0]["what"]


def test_a_text_column_is_still_not_a_signal(tmp_path):
    n = 60
    pd.DataFrame({"note": ["pick"] * n, "y": np.arange(n) * 0.5}).to_csv(tmp_path / "traj.csv", index=False)
    out = f.table_signals([tmp_path / "traj.csv"], None, _anchor(n), {})
    assert out.meta["traj"]["names"] == ["y"] and not _issues(out, "signal_bad_cells")


@pytest.mark.parametrize("sep", [";", "\t", "|"])
def test_a_table_is_read_whatever_its_delimiter(tmp_path, sep):
    n = 60
    pd.DataFrame({"x": np.arange(n) * 0.5, "y": np.ones(n)}).to_csv(tmp_path / "traj.csv", index=False, sep=sep)
    out = f.table_signals([tmp_path / "traj.csv"], None, _anchor(n), {})
    assert out.meta["traj"]["names"] == ["x", "y"]


def test_a_tsv_whose_header_names_hold_commas_is_split_on_its_tabs(tmp_path):
    n = 50
    p = tmp_path / "traj.tsv"
    p.write_text("time\tforce (N, x)\tforce (N, y)\n"
                 + "".join(f"{i / 30:.4f}\t{i * 0.5}\t{i * 1.5}\n" for i in range(n)))
    assert f.table_separator(p) == "\t"
    out = f.table_signals([p], None, _anchor(n), {})
    assert out.meta["traj"]["names"] == ["force (N, x)", "force (N, y)"]


@pytest.mark.parametrize("head,row", [
    ('time,"force [N; x; y]"', "{t},{x}"),         # separators inside a quoted name
    ("time,a|b|c", "{t},{x}"),                     # a separator in the header alone, not in the rows
])
def test_a_table_is_split_on_the_separator_that_gives_every_line_the_same_fields(tmp_path, head, row):
    p = tmp_path / "traj.csv"
    p.write_text(head + "\n" + "".join(row.format(t=f"{i / 30:.4f}", x=i * 0.5) + "\n" for i in range(50)))
    assert f.table_separator(p) == ","
    assert f.table_signals([p], None, _anchor(50), {})["traj"][10, 0] == 5.0


def test_a_semicolon_table_with_decimal_commas_reads_as_numbers(tmp_path):
    n = 50
    p = tmp_path / "traj.csv"
    p.write_text("time;force\n" + "".join(f"{i / 30:.3f}".replace(".", ",") + f";{i},5\n" for i in range(n)))
    out = f.table_signals([p], None, _anchor(n), {})
    assert out.meta["traj"]["names"] == ["force"] and out["traj"][10, 0] == 10.5
    assert not f._table_has_text(p)                 # numbers, never notes


def test_a_separator_at_the_end_of_every_line_is_no_column(tmp_path):
    n = 50
    p = tmp_path / "traj.csv"
    p.write_text("time,a,b,\n" + "".join(f"{i / 30:.4f},{i * 0.5},{i * 1.5},\n" for i in range(n)))
    out = f.table_signals([p], None, _anchor(n), {})
    assert out.meta["traj"]["names"] == ["a", "b"] and not _issues(out, "signal_bad_cells")
    assert not [name for name, _ in out.left_out if "Unnamed" in name]


def test_a_column_of_codes_written_as_numbers_and_words_is_not_a_signal(tmp_path):
    n = 60
    pd.DataFrame({"time": np.arange(n) / 30, "phase": ["1" if i % 3 else "grasp" for i in range(n)],
                  "force": np.arange(n) * 0.5}).to_csv(tmp_path / "traj.csv", index=False)
    out = f.table_signals([tmp_path / "traj.csv"], None, _anchor(n), {})
    assert out.meta["traj"]["names"] == ["force"] and not _issues(out, "signal_bad_cells")
    # named, never dropped without a word
    assert dict(out.left_out)["phase in traj.csv"] == "20 of its 60 filled cells are not numbers, so it is not read " \
                                                      "as a signal"


@pytest.mark.parametrize("mark", ["-", "ERR"])
def test_a_reading_with_one_repeated_mark_where_it_dropped_out_is_read_as_numbers(tmp_path, mark):
    n = 90
    x = np.sin(np.arange(n) / 10)
    p = tmp_path / "ft.csv"
    p.write_text("time,force,torque\n" + "".join(f"{i / 30:.4f},{mark if i % 7 == 0 else f'{x[i]:.4f}'},"
                                                f"{x[i] * 2:.4f}\n" for i in range(n)))
    out = f.table_signals([p], None, _anchor(n), {})
    assert out.meta["ft"]["names"] == ["force", "torque"] and np.isnan(out["ft"][7, 0])
    bad = _issues(out, "signal_bad_cells")
    assert len(bad) == 1 and "force has 13 of 90" in bad[0]["what"]


def test_a_semicolon_table_with_thousands_dots_reads_as_numbers(tmp_path):
    n = 50
    p = tmp_path / "traj.csv"
    p.write_text("time;force\n" + "".join(f"{i / 30:.4f}".replace(".", ",") + f";1.{200 + 3 * i:03d},5\n"
                                         for i in range(n)))
    assert f.table_format(p) == (";", ",")
    out = f.table_signals([p], None, _anchor(n), {})
    assert out["traj"][10, 0] == 1230.5 and not f._table_has_text(p)


def test_dot_decimals_beside_decimal_commas_keep_their_decimal_point(tmp_path):
    """A semicolon table whose time is written with decimal commas and whose force has three places after a dot had
    its force read as thousands, 1000 times too large."""
    n = 60
    p = tmp_path / "traj.csv"
    p.write_text("time;force;count\n" + "".join(f"{i / 30:.3f}".replace(".", ",") + f";{0.5 + i / 1000:.3f};"
                                               f"1.{100 + i:03d},5\n" for i in range(n)))
    out = f.table_signals([p], None, _anchor(n), {})
    assert out["traj"][10, 0] == pytest.approx(0.51)            # 0.510: a leading 0 is never a thousands group
    assert out["traj"][10, 1] == 1110.5                           # 1.110,5 proves its own column's dots


def test_thousands_dots_are_read_only_in_a_column_whose_cells_prove_them(tmp_path):
    n = 60
    p = tmp_path / "traj.csv"
    p.write_text("time;gain;count\n" + "".join(f"{i / 30:.3f}".replace(".", ",") + f";1.{200 + i:03d};"
                                               f"1.{200 + i:03d}.000\n" for i in range(n)))
    out = f.table_signals([p], None, _anchor(n), {})
    assert out["traj"][10, 0] == pytest.approx(1.21)              # 1.210 alone could be a decimal point
    assert out["traj"][10, 1] == 1_210_000


def test_a_comma_table_of_grouped_thousands_is_a_signal_and_not_notes(tmp_path):
    """Grouped cells ("1.100.000") were numbers to the notes reader and text to the table reader, so the table was
    neither notes nor a signal."""
    n = 60
    p = tmp_path / "traj.csv"
    p.write_text("time,count\n" + "".join(f'{i / 30:.4f},"1.{100 + i:03d}.000"\n' for i in range(n)))
    out = f.table_signals([p], None, _anchor(n), {})
    assert out.meta["traj"]["names"] == ["count"] and out["traj"][10, 0] == 1_110_000
    assert not f._table_has_text(p) and not out.left_out


def test_a_text_cell_in_a_decimal_comma_column_leaves_its_other_cells_numbers(tmp_path):
    n = 60
    p = tmp_path / "traj.csv"
    p.write_text("time;force\n" + "".join(f"{i / 30:.3f}".replace(".", ",") + (";ERR\n" if i == 7 else f";{i},5\n")
                                         for i in range(n)))
    out = f.table_signals([p], None, _anchor(n), {})
    assert out["traj"][10, 0] == 10.5 and np.isnan(out["traj"][7, 0])
    assert len(_issues(out, "signal_bad_cells")) == 1


@pytest.mark.parametrize("words", [["ERR"], ["ERR", "NaN?"]])
def test_a_column_of_exactly_nine_tenths_numbers_is_a_signal(tmp_path, words):
    n = 60
    p = tmp_path / "traj.csv"
    grip = [words[i // 10 % len(words)] if i % 10 == 5 else i % 2 for i in range(n)]     # 6 words in 60 cells
    p.write_text("time,grip,force\n" + "".join(f"{i / 30:.4f},{grip[i]},{np.sin(i / 9):.4f}\n" for i in range(n)))
    out = f.table_signals([p], None, _anchor(n), {})
    assert out.meta["traj"]["names"] == ["grip", "force"] and not out.left_out
    assert "grip has 6 of 60" in _issues(out, "signal_bad_cells")[0]["what"]


def test_a_table_is_judged_on_its_first_lines_read_once(tmp_path, monkeypatch):
    import builtins
    p = tmp_path / "traj.csv"
    p.write_text("time;force\n" + "".join(f"{i};{i * 0.5}\n" for i in range(50)))
    opened, real_open = [], builtins.open
    monkeypatch.setattr(builtins, "open", lambda file, *a, **k: opened.append(file) or real_open(file, *a, **k))
    f.table_format(p)
    assert opened.count(p) == 1


def test_a_table_beside_capture_times_that_count_from_zero_is_not_recorded_timing(tmp_path):
    """Capture times from the recording's start and a time column that counts frames both start at 0, so their ranges
    overlap whatever they are: the table had been read as 1 Hz recorded timing."""
    n = 240
    rel = np.arange(n) / 30.0
    x = np.sin(np.arange(n) / 20)
    pd.DataFrame({"time": np.arange(n), "force": x}).to_csv(tmp_path / "traj.csv", index=False)
    out = f.table_signals([tmp_path / "traj.csv"], rel, _anchor(n), {})
    assert out.meta["traj"]["aligned_by"] == f.ALIGNED_ROWS and "rate_hz" not in out.meta["traj"]
    assert out["traj"][120, 0] == pytest.approx(x[120], abs=1e-6)
    pd.DataFrame({"timestamp": np.arange(400) * 20, "force": np.sin(np.arange(400) / 30)}).to_csv(
        tmp_path / "fast.csv", index=False)
    assert f.table_signals([tmp_path / "fast.csv"], rel, _anchor(n), {}).meta["fast"]["aligned_by"] \
        == f.ALIGNED_ASSUMED


def test_a_one_row_table_is_named_with_why(tmp_path):
    pd.DataFrame({"x": [1.5], "y": [2.0]}).to_csv(tmp_path / "calib.csv", index=False)
    out = f.table_signals([tmp_path / "calib.csv"], None, _anchor(60), {})
    assert "calib" not in out
    assert any(name == "calib.csv" and "one row" in why for name, why in out.left_out)


def test_a_table_of_numbers_no_episode_takes_is_named(tmp_path):
    _clip(tmp_path / "videos" / "top.mp4", 30)
    (tmp_path / "tables").mkdir()
    pd.DataFrame({"t": np.arange(100) / 10.0, "force": np.sin(np.arange(100))}).to_csv(
        tmp_path / "tables" / "force.csv", index=False)
    pd.DataFrame({"episode": ["top"], "task": ["pick the cube"]}).to_csv(tmp_path / "tables" / "notes.csv",
                                                                          index=False)
    det, items = f.plan(tmp_path)
    words = " ".join(det["missing"])
    assert "tables/force.csv" in words and "tables/notes.csv" not in words


# ---------------------------------------------------------------- a table's placement says when it is assumed

def test_a_table_with_one_row_per_frame_is_placed_one_row_per_frame_whatever_its_times_span(tmp_path):
    """A table with as many rows as the video has frames was written one row per frame, also when its own time column
    spans another length, as a recorder that stamps each frame's row on the wall clock while the video is written at
    its nominal rate does: placed by its times, its readings drift from the motion they record. Each row sits on its
    frame, marked so, and a data issue says that by its own times the video plays fast."""
    n = 240
    pd.DataFrame({"timestamp": 1756534813.0 + np.arange(n) * 0.043, "x": np.arange(n) * 0.5}).to_csv(
        tmp_path / "traj.csv", index=False)
    extra = {}
    out = f.table_signals([tmp_path / "traj.csv"], None, _anchor(n), extra)
    assert out.meta["traj"]["aligned_by"] == f.ALIGNED_ROWS and "rate_hz" not in out.meta["traj"]
    assert out["traj"][120, 0] == 60.0                              # row 120 on frame 120
    assert len(_issues(extra, "signal_alignment_assumed")) == 1
    span = _issues(extra, "table_span_differs")
    assert len(span) == 1 and span[0]["signal"] == "traj"
    assert "10.3 s" in span[0]["what"] and "8.0 s" in span[0]["what"] and "plays fast" in span[0]["what"]


def test_a_time_column_that_counts_frames_is_placed_one_row_per_frame(tmp_path):
    n = 240
    pd.DataFrame({"time": np.arange(n), "force": np.sin(np.arange(n) / 20)}).to_csv(tmp_path / "traj.csv", index=False)
    out = f.table_signals([tmp_path / "traj.csv"], None, _anchor(n), {})
    assert out.meta["traj"]["aligned_by"] == f.ALIGNED_ROWS and "rate_hz" not in out.meta["traj"]
    assert out["traj"][120, 0] == pytest.approx(np.sin(6.0), abs=1e-6)
    assert np.isfinite(out["traj"][:, 0]).all()


def test_a_table_with_one_row_per_frame_whose_times_agree_has_no_span_issue(tmp_path):
    n = 240
    pd.DataFrame({"timestamp_ms": np.arange(n) * 1000 / 30, "x": np.arange(n) * 0.5}).to_csv(
        tmp_path / "traj.csv", index=False)
    extra = {}
    out = f.table_signals([tmp_path / "traj.csv"], None, _anchor(n), extra)
    assert out.meta["traj"]["aligned_by"] == f.ALIGNED_ROWS and out["traj"][120, 0] == 60.0
    assert not _issues(extra, "table_span_differs")


def test_a_table_with_no_time_column_and_one_row_per_frame_is_marked_as_placed_row_by_row(tmp_path):
    n = 60
    pd.DataFrame({"x": np.arange(n) * 0.5, "y": np.ones(n)}).to_csv(tmp_path / "traj.csv", index=False)
    extra = {}
    out = f.table_signals([tmp_path / "traj.csv"], None, _anchor(n), extra)
    assert out.meta["traj"]["aligned_by"] == f.ALIGNED_ROWS
    assert out["traj"][30, 0] == 15.0
    got = _issues(extra, "signal_alignment_assumed")
    assert len(got) == 1 and "one row per frame" in got[0]["what"]


def test_the_prompt_says_how_an_assumed_table_was_placed():
    from label import signals as sg
    a = np.arange(60, dtype=float)[:, None] * 0.5
    rows = sg.describe("traj", a, aligned_by=f.ALIGNED_ROWS)
    assert "one row per frame" in rows and "both starts" not in rows
    # a table placed row by row may have a time column of its own, so the line gives the reason that holds for both
    assert "as many rows as the video has frames" in rows and "no time of its own" not in rows
    assert "both starts" in sg.describe("traj", a, aligned_by=f.ALIGNED_ASSUMED)


def test_a_table_on_the_capture_clock_is_recorded_timing(tmp_path):
    n = 60
    real = T0 + np.arange(n) / 30.0
    pd.DataFrame({"timestamp": real, "x": np.arange(n) * 0.5}).to_csv(tmp_path / "traj.csv", index=False)
    extra = {}
    out = f.table_signals([tmp_path / "traj.csv"], real, _anchor(n), extra)
    assert "aligned_by" not in out.meta["traj"] and not extra.get("reader_issues")


# ---------------------------------------------------------------- a slow table keeps its unit

def test_a_table_under_1_hz_on_the_capture_clock_is_read_in_seconds(tmp_path):
    n = 240
    real = T0 + np.arange(n) / 30.0
    pd.DataFrame({"timestamp": T0 + np.arange(5) * 2.0, "x": np.arange(5) * 1.0}).to_csv(
        tmp_path / "traj.csv", index=False)
    extra = {}
    out = f.table_signals([tmp_path / "traj.csv"], real, _anchor(n), extra)
    assert "aligned_by" not in out.meta["traj"]
    assert out["traj"][60, 0] == 1.0 and out["traj"][180, 0] == 3.0
    assert np.isfinite(out["traj"][:, 0]).sum() > 200


def test_a_table_under_1_hz_with_no_capture_times_is_read_in_the_unit_that_spans_the_footage(tmp_path):
    n = 240
    pd.DataFrame({"timestamp": T0 + np.arange(5) * 2.0, "x": np.arange(5) * 1.0}).to_csv(
        tmp_path / "traj.csv", index=False)
    out = f.table_signals([tmp_path / "traj.csv"], None, _anchor(n), {})
    assert out["traj"][60, 0] == 1.0 and out["traj"][180, 0] == 3.0


# ---------------------------------------------------------------- bad cells flagged where they are

def test_bad_cells_of_a_table_are_missing_and_flagged_value_by_value(tmp_path):
    n = 60
    df = pd.DataFrame({"x": np.arange(n) * 0.5, "y": np.arange(n) * 2.0, "g": np.ones(n), "q": np.nan})
    df.loc[10:20, "x"] = np.nan
    df.loc[30, "y"] = np.inf
    df.loc[31, "y"] = -np.inf
    df["g"] = df["g"].astype(object)
    df.loc[40:45, "g"] = ""
    df.to_csv(tmp_path / "traj.csv", index=False)
    out = f.table_signals([tmp_path / "traj.csv"], None, _anchor(n), {})
    a = out["traj"]
    assert not np.isinf(a).any()
    assert out.meta["traj"]["names"] == ["x", "y", "g"]
    bad = {i["what"].split(" has ")[0]: i for i in _issues(out, "signal_bad_cells")}
    assert set(bad) == {"traj x", "traj y", "traj g", "traj q"}
    assert "11 of 60" in bad["traj x"]["what"] and "2 of 60" in bad["traj y"]["what"]
    assert "no reading" in bad["traj q"]["what"]
    assert any(name == "q in traj.csv" for name, _ in out.left_out)


def test_an_inf_is_never_written_to_the_signals(tmp_path):
    n = 30
    ctx = {"n_state_frames": n, "fps": 30.0}
    a = np.ones((n, 2))
    a[5, 1] = np.inf
    f.write_signals(tmp_path, ctx, {"s": a})
    with np.load(tmp_path / "signals.npz") as z:
        assert not np.isinf(z["s0"]).any() and np.isnan(z["s0"][5, 1])
    assert _issues(ctx, "signal_not_finite")


def test_an_inf_in_a_lerobot_column_is_flagged():
    n = 30
    col = [[1.0, 2.0]] * n
    col[7] = [1.0, float("inf")]
    out = f.recorded_signals(pd.DataFrame({"force": col}), set(), n)
    assert not np.isinf(out["force"]).any()
    bad = [i for i in out.issues if i.get("signal") == "force"]
    assert bad and "1 of 30" in bad[0]["what"]


# ---------------------------------------------------------------- a sparse table keeps its rate

def test_a_sparse_table_records_its_rate(tmp_path):
    n = 240
    real = T0 + np.arange(n) / 30.0
    pd.DataFrame({"timestamp": T0 + np.arange(10) * 0.8, "x": np.arange(10) * 1.0}).to_csv(
        tmp_path / "traj.csv", index=False)
    out = f.table_signals([tmp_path / "traj.csv"], real, _anchor(n), {})
    assert out.meta["traj"]["rate_hz"] == pytest.approx(1.25, abs=0.01)


# ---------------------------------------------------------------- the minutes limit skips, then keeps trying

def test_an_episode_past_the_minutes_left_is_skipped_and_later_ones_still_tried(tmp_path):
    for name, frames in (("ep1", 30), ("ep2", 90), ("ep3", 30)):
        _clip(tmp_path / "up" / name / "top.mp4", frames)
    rep = f.convert(tmp_path / "up", "teleop_arms", tmp_path / "out" / "eps", "t", 2.5)
    assert [e["name"] for e in rep["episodes"]] == ["ep1/top", "ep3/top"]
    assert [s["name"] for s in rep["skipped"]] == ["ep2/top"] and "minutes" in rep["skipped"][0]["why"]


def test_once_the_minutes_are_used_up_no_later_episode_is_converted(tmp_path, monkeypatch):
    """As the upload page counts them (read.js chooseEpisodes), the minutes are used up once what is taken comes within
    a second of the limit: a later episode is listed, never converted and then deleted."""
    for name in ("ep1", "ep2", "ep3"):
        _clip(tmp_path / "up" / name / "top.mp4", 30)
    calls, convert_video = [], f.convert_video
    monkeypatch.setattr(f, "convert_video", lambda it, *a: calls.append(it["name"]) or convert_video(it, *a))
    rep = f.convert(tmp_path / "up", "teleop_arms", tmp_path / "out" / "eps", "t", 3.0)
    assert [e["name"] for e in rep["episodes"]] == ["ep1/top", "ep2/top"] == calls
    assert [s["name"] for s in rep["skipped"]] == ["ep3/top"]


def test_episodes_under_the_minutes_are_all_taken(tmp_path):
    for name in ("ep1", "ep2"):
        _clip(tmp_path / "up" / name / "top.mp4", 30)
    rep = f.convert(tmp_path / "up", "teleop_arms", tmp_path / "out" / "eps", "t", 900)
    assert len(rep["episodes"]) == 2 and not rep["skipped"]


# ---------------------------------------------------------------- a recorder's own logs, calibration and start up

def _ros_mcap(path: Path, lead_s: float = 0.25) -> None:
    from mcap.writer import Writer
    with open(path, "wb") as fh:
        w = Writer(fh)
        w.start()
        chans = {}
        for topic, schema in (("/rosout", "rcl_interfaces/msg/Log"), ("/diag", "diagnostic_msgs/msg/DiagnosticArray"),
                              ("/cam/camera_info", "sensor_msgs/msg/CameraInfo"), ("/imu", "sensor_msgs/msg/Imu")):
            sid = w.register_schema(name=schema, encoding="jsonschema", data=b"{}")
            chans[topic] = w.register_channel(topic=topic, message_encoding="json", schema_id=sid)
        for i in range(120):
            s = i / 30.0
            ns = int((T0 + s) * 1e9)
            w.add_message(chans["/imu"], log_time=ns, publish_time=ns,
                          data=json.dumps({"x": float(np.sin(i / 7)), "y": float(np.cos(i / 5))}).encode())
            w.add_message(chans["/rosout"], log_time=ns, publish_time=ns,
                          data=json.dumps({"level": 20 + i % 3, "line": 100 + i, "msg": "tick"}).encode())
            w.add_message(chans["/diag"], log_time=ns, publish_time=ns,
                          data=json.dumps({"status": [{"level": i % 2, "values": []}]}).encode())
            if s >= lead_s:
                w.add_message(chans["/cam/camera_info"], log_time=ns, publish_time=ns,
                              data=json.dumps({"height": 480, "width": 640, "k": [600.0, 0, 320, 0, 600, 240, 0, 0, 1]})
                              .encode())
        w.finish()


def test_ros_logs_and_diagnostics_are_bookkeeping_named_not_signals(tmp_path):
    p = tmp_path / "rec.mcap"
    _ros_mcap(p)
    q = T0 + np.arange(120) / 30.0
    out = f.mcap_signals([p], q)
    assert not [k for k in out if k.startswith(("/rosout", "/diag"))]
    assert any(k.startswith("/imu") for k in out) and any(k.startswith("/cam/camera_info") for k in out)
    left = dict(out.left_out)
    assert "log" in left["/rosout"] and "diagnostics" in left["/diag"]


def test_camera_info_keeps_its_values_and_gets_no_gap_issue(tmp_path):
    p = tmp_path / "rec.mcap"
    _ros_mcap(p, lead_s=1.0)
    q = T0 + np.arange(120) / 30.0
    out = f.mcap_signals([p], q)
    ctx = {"n_state_frames": 120, "fps": 30.0}
    f.write_signals(tmp_path, ctx, out, t=q)
    info = [s for s in ctx["signals"] if s["name"].startswith("/cam/camera_info")]
    assert info
    assert not [i for i in ctx.get("reader_issues", []) if str(i.get("signal", "")).startswith("/cam/camera_info")]


def test_a_short_start_up_lead_is_not_a_gap_but_a_real_gap_is():
    t = np.arange(120) / 30.0
    a = np.ones((120, 1))
    a[:7] = np.nan                                  # first reading at 0.23 s: the recorder starting up
    assert f.signal_gaps("s", a, t) == []
    a[60:75] = np.nan                               # half a second with no reading in the middle
    got = f.signal_gaps("s", a, t)
    assert [i["kind"] for i in got] == ["signal_gap"] and "15 of its 120" in got[0]["what"]
    b = np.ones((120, 1))
    b[-5:] = np.nan                                 # the last reading 0.17 s before the end
    assert f.signal_gaps("s", b, t) == []


def test_an_arm_state_is_never_held_over_a_lead_the_signals_call_missing():
    """A 1 s episode whose arms start 0.4 s late: the signals flag the lead, so the state must not hold the first
    reading over it as recorded stillness. One edge slack (edge_slack) serves both."""
    q = 100.0 + np.arange(30) / 30.0
    t = 100.4 + np.arange(60) / 100.0
    pos = np.column_stack([np.linspace(0, 1, 60)] * 6 + [np.full(60, 0.5)])
    streams = {f"/yam_{s}/joint_state": {"t": t, "pos": pos, "names": None} for s in ("left", "right")}
    state, _, note = f.joint_state(streams, q)
    assert state is None and "/yam_left/joint_state has readings from 0.4 s" in note
    rows, gap = f.fill_rows(q, t, pos)
    assert rows is None and "a gap longer than the 0.1 s the reader fills" in f.gap_words(gap, q[0])
    a = np.full((30, 1), np.nan)
    a[12:] = 1.0
    assert [i["kind"] for i in f.signal_gaps("arm", a, q - q[0])] == ["signal_partial_span"]
    # a minute of footage keeps half a second, as before
    assert f.edge_slack(60.0) == f.EDGE_SLACK_S == 0.5


def test_a_short_episode_missing_a_large_share_at_an_edge_gets_its_issue():
    t = np.arange(30) / 30.0                        # a 1 s episode
    a = np.ones((30, 1))
    a[:12] = np.nan                                 # no reading over its first 0.4 s, 40 percent of it
    assert [i["kind"] for i in f.signal_gaps("s", a, t)] == ["signal_partial_span"]
    b = np.ones((30, 1))
    b[-12:] = np.nan
    assert [i["kind"] for i in f.signal_gaps("s", b, t)] == ["signal_partial_span"]


# ---------------------------------------------------------------- a LeRobot dataset's other files are named

def test_a_readme_in_a_lerobot_root_is_named_as_not_read(tmp_path):
    root = tmp_path / "ds"
    info = {"codebase_version": "v2.1", "fps": 30, "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {"observation.images.top": {"dtype": "video", "shape": [36, 64, 3]},
                         "observation.state": {"dtype": "float32", "shape": [2]}}}
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "data" / "chunk-000").mkdir(parents=True)
    pd.DataFrame({"observation.state": [np.zeros(2)] * 30, "frame_index": np.arange(30),
                  "episode_index": np.zeros(30, int), "timestamp": np.arange(30) / 30}).to_parquet(
        root / "data" / "chunk-000" / "episode_000000.parquet")
    _clip(root / "videos" / "chunk-000" / "observation.images.top" / "episode_000000.mp4", 30)
    (root / "README.md").write_text("# a dataset card\n")
    det, items = f.plan(tmp_path)
    words = " ".join(det["missing"])
    assert "ds/README.md" in words
    assert "info.json" not in words and "episode_000000" not in words


# ---------------------------------------------------------------- why an episode has no arm state

def test_the_reasons_for_no_arm_state_are_one_closed_set():
    assert set(f.STATE_WHY) == {"layout", "not_recorded", "unreadable", "short", "assumed_clock"}
    with pytest.raises(ValueError):
        f.StateNote("Labelled from the cameras.", "unknown")


def test_an_episode_with_its_arm_state_gives_no_reason_for_none(tmp_path):
    ctx = _recorder(tmp_path, lambda d: None)
    assert ctx["state_kind"] == "joints" and "state_why" not in ctx and "state_note" not in ctx


def test_arms_on_a_clock_none_of_the_footage_shares_give_the_assumed_clock_as_why(tmp_path):
    def relative(d):
        for p in d.glob("*-timestamp.npy"):
            np.save(p, np.load(p) - np.load(p)[0])
    ctx = _recorder(tmp_path, relative)
    assert ctx["state_why"] == "assumed_clock"
    assert ctx["state_note"] == ("Labelled from the cameras, because the sensor files share no clock with the videos "
                                 "to place a recorded arm state against; their channels are kept as signals.")


def test_the_arm_channels_note_says_why_there_is_no_state():
    q = np.arange(60) / 30.0
    arm = lambda d, t=q: {"t": t, "pos": np.ones((len(t), d))}
    assert f.joint_state({"/arm/joint_state": arm(8)}, q)[2].why == "layout"
    assert f.joint_state({"/left/joint_state": arm(7), "/arm/joint_state": arm(7)}, q)[2].why == "layout"
    assert f.joint_state({"/arm/joint_state": arm(7, q[30:])}, q)[2].why == "short"
    gap = np.concatenate([q[:10], q[50:]])
    assert f.joint_state({"/arm/joint_state": arm(7, gap)}, q)[2].why == "short"


def _mcap_ctx(tmp_path, write) -> dict:
    root = tmp_path / "upload"
    root.mkdir()
    write(root / "rec.mcap")
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "t", 900)
    return json.loads((tmp_path / "eps" / rep["episodes"][0]["episode_id"] / "context.json").read_text())


def test_an_mcap_recording_with_no_robot_state_says_it_records_none(tmp_path):
    from test_formats import _camera_mcap
    ctx = _mcap_ctx(tmp_path, lambda p: _camera_mcap(p, ["/cam/image/compressed"]))
    assert ctx["state_note"] == "Labelled from the cameras, because the file records no robot state."
    assert ctx["state_why"] == "not_recorded"


def test_an_mcap_recording_whose_motion_our_checks_do_not_read_gives_the_layout_as_why(tmp_path):
    import base64
    from mcap.writer import Writer
    from test_formats import _jpeg

    def write(p):
        with open(p, "wb") as fh:
            w = Writer(fh)
            w.start()
            img = w.register_schema(name="foxglove.CompressedImage", encoding="jsonschema", data=b"{}")
            pose = w.register_schema(name="foxglove.PoseInFrame", encoding="jsonschema", data=b"{}")
            cam = w.register_channel(topic="/cam/image/compressed", message_encoding="json", schema_id=img)
            hand = w.register_channel(topic="/hand/pose", message_encoding="json", schema_id=pose)
            for k in range(20):
                ns = int((T0 + k / 30) * 1e9)
                w.add_message(cam, log_time=ns, publish_time=ns, data=json.dumps(
                    {"format": "jpeg", "data": base64.b64encode(_jpeg(k * 10)).decode()}).encode())
                if k == 0:                          # one message, a setting rather than a signal
                    w.add_message(hand, log_time=ns, publish_time=ns, data=json.dumps({"x": 0.01}).encode())
            w.finish()
    ctx = _mcap_ctx(tmp_path, write)
    assert "did not run on this file's motion channels" in ctx["state_note"] and ctx["state_why"] == "layout"


def _lerobot_ctx(tmp_path, cols: dict, data: bytes | None = None) -> dict:
    from test_formats import _lerobot
    root = tmp_path / "ds"
    _lerobot(root, {0: cols})
    if data is not None:
        (root / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(data)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "t", 900)
    return json.loads((tmp_path / "eps" / rep["episodes"][0]["episode_id"] / "context.json").read_text())


def test_a_lerobot_episode_says_why_it_has_no_state(tmp_path):
    rng = np.random.default_rng(1)
    none = _lerobot_ctx(tmp_path / "a", {"glove": [rng.random(4) for _ in range(30)]})
    assert none["state_note"].endswith("the dataset records no observation.state.")
    assert none["state_why"] == "not_recorded"
    wide = _lerobot_ctx(tmp_path / "b", {"observation.state": [rng.random(5) for _ in range(30)]})
    assert "5 values per frame" in wide["state_note"] and wide["state_why"] == "layout"
    short = _lerobot_ctx(tmp_path / "c", {"observation.state": [rng.random(14) for _ in range(10)]})
    assert "observation.state has no reading from" in short["state_note"] and short["state_why"] == "short"
    bad = _lerobot_ctx(tmp_path / "d", {"observation.state": [rng.random(14) for _ in range(30)]}, data=b"not parquet")
    assert "could not be opened" in bad["state_note"] and bad["state_why"] == "unreadable"


def test_an_hdf5_episode_says_why_it_has_no_state(tmp_path):
    from test_touch_depth import _convert
    t = np.arange(40) / 20.0
    ctx, _ = _convert(tmp_path, {"observations/robot_state/joint_positions":
                                 np.stack([np.sin(t + j) for j in range(7)], axis=1)})
    assert "7 joints and no gripper" in ctx["state_note"] and ctx["state_why"] == "layout"


def test_sensor_file_and_signal_span_issues_are_named_data_families():
    from board.families import Families
    fam = Families()
    slugs = {k: fam.reader_family(k) for k in ("mcap_file_damaged", "sensor_file_cut", "signal_partial_span")}
    assert slugs["mcap_file_damaged"] == slugs["sensor_file_cut"] and not slugs["sensor_file_cut"].startswith("d:")
    assert not slugs["signal_partial_span"].startswith("d:")
    assert {fam.catalog()[s]["list"] for s in slugs.values()} == {"data"}


@pytest.mark.parametrize("topic,blocked", [("/yam_left/joint_state", True),
                                          ("/yam_leader_left/joint_pos", False),
                                          ("/yam_camera/joint_state", False)])
def test_an_outside_follower_blocks_state_but_commands_and_a_third_arm_do_not(tmp_path, topic, blocked):
    name = topic.split("/")[1] + ".mcap"
    ctx = _recorder(tmp_path, lambda d: _arm_mcap(d / name, seconds=2, t0=T0 + 3600, topic=topic))
    assert (ctx["state_kind"] == "none") == blocked
    if blocked:
        assert ctx["state_why"] == "short" and name in ctx["state_note"]
        assert "outside" in ctx["state_note"] and "/yam_right/joint_state joint_pos" in _signal_names(ctx)
        assert _issues(ctx, "sensor_outside_footage")
    elif "leader" in topic:
        with np.load(ctx["_ep"] / "state.npz") as z:
            assert z["state"].shape == (60, 14) and "action" not in z.files


def test_a_damaged_command_file_costs_only_the_action_and_names_its_channels(tmp_path):
    ctx = _recorder(tmp_path, lambda d: _damage_whole_chunk(d, "yam_leader_left.mcap"))
    assert ctx["state_kind"] == "joints" and "state_note" not in ctx
    with np.load(ctx["_ep"] / "state.npz") as z:
        assert z["state"].shape == (60, 14) and "action" not in z.files


def test_a_damaged_follower_names_its_declared_channels_without_calling_them_unknown(tmp_path):
    ctx = _recorder(tmp_path, _damage_whole_chunk)
    assert "/yam_left/joint_state" in ctx["state_note"]
    assert "What it records is unknown" not in ctx["state_note"]


@pytest.mark.parametrize("chunked", [False, True])
def test_late_grouping_proof_keeps_every_readable_cell_consistently(tmp_path, monkeypatch, chunked):
    p = tmp_path / "traj.csv"
    p.write_text("time;force\n0,0;1.234\n0,1;2.345\n0,2;12.345.678\n0,3;23.456.789\n")
    if chunked:
        monkeypatch.setattr(f, "TABLE_MAX_BYTES", 1)
        monkeypatch.setattr(f, "TABLE_CHUNK_ROWS", 2)
    df, text, _, _ = f.read_number_table(p)
    assert text == [] and df["force"].tolist() == [1234., 2345., 12345678., 23456789.]


@pytest.mark.parametrize("chunked", [False, True])
def test_decimal_evidence_prevents_grouping_proof_from_flipping_decimal_cells(tmp_path, monkeypatch, chunked):
    p = tmp_path / "traj.csv"
    p.write_text("time;force\n0,0;1.000\n0,1;1.286\n0,2;1.5\n0,3;1.234.567\n")
    if chunked:
        monkeypatch.setattr(f, "TABLE_MAX_BYTES", 1)
        monkeypatch.setattr(f, "TABLE_CHUNK_ROWS", 2)
    df, _, _, _ = f.read_number_table(p)
    assert df["force"].tolist() == pytest.approx([1., 1.286, 1.5, 1234567.])
    out = f.table_signals([p], None, _anchor(4), {})
    assert any("force" in i["what"] and "decimal" in i["what"] for i in _issues(out, "table_number_ambiguous"))


def test_single_dot_groups_in_a_decimal_comma_table_name_the_inferred_reading(tmp_path):
    p = tmp_path / "traj.csv"
    p.write_text("time;force\n0,0;12.500\n0,1;13.500\n0,2;14.500\n")
    df, _, _, _ = f.read_number_table(p)
    assert df["force"].tolist() == [12.5, 13.5, 14.5]
    out = f.table_signals([p], None, _anchor(3), {})
    assert _issues(out, "table_number_ambiguous")


@pytest.mark.parametrize("separator,suffix", [("\t", ".tsv"), (",", ".csv")])
def test_decimal_commas_in_quoted_csv_and_tsv_are_numbers_to_notes_and_signals(tmp_path, separator, suffix):
    import csv
    p = tmp_path / ("traj" + suffix)
    with p.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter=separator)
        w.writerows([["time", "force"], ["0,0", "0,5"], ["0,1", "1,5"], ["0,2", "2,5"]])
    df, text, _, _ = f.read_number_table(p)
    assert df["force"].tolist() == [0.5, 1.5, 2.5] and text == []
    assert not f._table_has_text(p)


def test_a_numeric_column_starting_after_the_first_chunk_is_kept(tmp_path, monkeypatch):
    p = tmp_path / "traj.csv"
    p.write_text("time,force\n0,ERR\n1,ERR\n" + "".join(f"{i},{i / 3}\n" for i in range(2, 22)))
    monkeypatch.setattr(f, "TABLE_MAX_BYTES", 1)
    monkeypatch.setattr(f, "TABLE_CHUNK_ROWS", 2)
    df, text, _, _ = f.read_number_table(p)
    assert "force" in df and np.isnan(df["force"].iloc[:2]).all() and text == []


@pytest.mark.parametrize("step", [0.1, 1.0])
def test_coarse_row_clocks_keep_signals_without_claiming_precise_state_or_false_gaps(tmp_path, step):
    import h5py
    def change(d):
        for p in d.glob("*.mcap"):
            p.unlink()
        t = T0 + np.arange(66) / 30
        with h5py.File(d / "robot.h5", "w") as h:
            h["qpos"] = np.tile(np.arange(66)[:, None], (1, 14)).astype(float)
            h["timestamp"] = T0 + np.floor((t - T0 + 1e-5) / step) * step
    ctx = _recorder(tmp_path, change)
    assert ctx["state_kind"] == "none" and ctx["state_why"] == "assumed_clock"
    assert "coarse" in ctx["state_note"] and _issues(ctx, "signal_clock_coarse")
    assert "qpos" in _signal_names(ctx)
    assert not _issues(ctx, "signal_gap") and not _issues(ctx, "signal_partial_span")
    assert all(s.get("aligned_by") == "coarse clock" for s in ctx["signals"] if "qpos" in s["name"])


def test_a_coarse_table_clock_names_its_uncertainty_without_false_gaps(tmp_path):
    p = tmp_path / "traj.csv"
    pd.DataFrame({"timestamp": T0 + np.floor(np.arange(66) / 30), "x": np.sin(np.arange(66))}).to_csv(p, index=False)
    out = f.table_signals([p], T0 + np.arange(60) / 30, _anchor(60), {})
    assert out.meta["traj"]["aligned_by"] == "coarse clock" and _issues(out, "signal_clock_coarse")
    assert np.isfinite(out["traj"]).all()


@pytest.mark.parametrize("images", [False, True])
@pytest.mark.parametrize("column,reason", [("observation.joint_positions", "layout"),
                                           ("action.joint_positions", "not_recorded"),
                                           ("force", "not_recorded")])
def test_missing_lerobot_state_names_recorded_observation_motion_truthfully(tmp_path, images, column, reason):
    from test_formats import _lerobot, _jpeg
    root = tmp_path / "ds"
    cols = {column: np.tile(np.linspace(0, 1, 30)[:, None], (1, 6)).tolist()}
    feats = {column: {"dtype": "float32", "shape": [6]}}
    if images:
        cols["observation.images.top"] = [{"bytes": _jpeg(k * 8, 96, 72)} for k in range(30)]
        feats["observation.images.top"] = {"dtype": "image", "shape": [72, 96, 3]}
    _lerobot(root, {0: cols}, feats=feats, n_video=0 if images else 30)
    ip = root / "meta" / "info.json"
    info = json.loads(ip.read_text())
    info["features"].pop("observation.state")
    ip.write_text(json.dumps(info))
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "t", 900)
    ctx = json.loads((tmp_path / "eps" / rep["episodes"][0]["episode_id"] / "context.json").read_text())
    assert ctx["state_kind"] == "none" and ctx["state_why"] == reason
    assert "0 values per frame" not in ctx["state_note"]
    if reason == "layout":
        assert column in ctx["state_note"] and column in _signal_names(ctx)
    else:
        assert "records no observation.state" in ctx["state_note"]


@pytest.mark.parametrize("indexed", [False, True])
def test_a_crc_mismatch_keeps_readable_messages_and_names_damage(tmp_path, indexed):
    import struct
    from mcap.reader import make_reader
    p = tmp_path / "arm.mcap"
    _arm_mcap(p, seconds=2, compressed=False)
    with p.open("rb") as fh:
        chunk = make_reader(fh).get_summary().chunk_indexes[1]
    b = bytearray(p.read_bytes())
    # Chunk header has opcode and length, then three uint64 fields before its CRC.
    at = chunk.chunk_start_offset + 9 + 24
    crc = struct.unpack_from("<I", b, at)[0]
    struct.pack_into("<I", b, at, crc ^ 1)
    if not indexed:
        b = b[:chunk.chunk_start_offset + chunk.chunk_length]
    p.write_bytes(b)
    with p.open("rb") as fh:
        damaged = []
        msgs = list(f.mcap_messages(fh, p, ["/yam_left/joint_state"], damaged))
    assert damaged and len(msgs) > 10
    if indexed:
        assert len(msgs) == 200 and msgs[-1][2].log_time > int((T0 + 1.9) * 1e9)
    out = f.mcap_signals([p], T0 + np.arange(30) / 30)
    assert _issues(out, "mcap_file_damaged") and list(out)


def test_a_bad_middle_chunk_preserves_messages_in_later_chunks(tmp_path):
    p = tmp_path / "arm.mcap"
    _arm_mcap(p, seconds=4)
    _damage_chunk_after(p, 1.0)
    with p.open("rb") as fh:
        damage = []
        msgs = list(f.mcap_messages(fh, p, ["/yam_left/joint_state"], damage))
    assert damage and msgs[-1][2].log_time > int((T0 + 3.9) * 1e9)
    assert len(msgs) > 300


def test_a_cut_summary_names_file_damage_but_keeps_complete_footage(tmp_path):
    from mcap.reader import make_reader
    p = tmp_path / "arm.mcap"
    _arm_mcap(p, seconds=2)
    with p.open("rb") as fh:
        last = make_reader(fh).get_summary().chunk_indexes[-1]
    p.write_bytes(p.read_bytes()[:last.chunk_start_offset + last.chunk_length])
    out = f.Signals()
    f.note_sensors({}, out, [p], [], [], T0 + np.arange(30) / 30)
    (issue,) = _issues(out, "sensor_file_cut")
    assert issue["footage_complete"] and "cover the whole footage" in issue["what"]



def test_new_reader_issues_have_explicit_data_or_handling_families():
    from board.families import Families
    fam = Families()
    for kind, category in [("sensor_outside_footage", "data"), ("signal_clock_coarse", "data"),
                           ("table_number_ambiguous", "handling")]:
        slug = fam.reader_family(kind)
        assert not slug.startswith("d:") and fam.catalog()[slug]["list"] == category



def test_camera_crc_damage_keeps_all_decodable_frames_and_flags_the_recording(tmp_path):
    import struct
    from mcap.reader import make_reader
    from test_formats import _camera_mcap
    root = tmp_path / "up"
    root.mkdir()
    p = root / "rec.mcap"
    _camera_mcap(p, ["/cam/image/compressed"])
    with p.open("rb") as fh:
        chunk = make_reader(fh).get_summary().chunk_indexes[0]
    b = bytearray(p.read_bytes())
    at = chunk.chunk_start_offset + 9 + 24
    struct.pack_into("<I", b, at, struct.unpack_from("<I", b, at)[0] ^ 1)
    p.write_bytes(b)
    rep = f.convert(root, "ego_head", tmp_path / "out", "t", 900)
    ep = tmp_path / "out" / rep["episodes"][0]["episode_id"]
    ctx = json.loads((ep / "context.json").read_text())
    assert ctx["n_state_frames"] == 20 and _issues(ctx, "mcap_file_damaged")



def test_crc_recovery_keeps_indexed_camera_messages_on_their_recorded_instants(tmp_path):
    import struct
    from mcap.reader import make_reader
    from mcap.writer import Writer
    from test_formats import _camera_mcap
    p = tmp_path / "source.mcap"
    _camera_mcap(p, ["/cam/image/compressed"])
    with p.open("rb") as fh:
        rows = list(make_reader(fh).iter_messages())
    root = tmp_path / "up"
    root.mkdir()
    p = root / "rec.mcap"
    with p.open("wb") as fh:
        w = Writer(fh, chunk_size=2000)
        w.start()
        schema, channel, _ = rows[0]
        sid = w.register_schema(name=schema.name, encoding=schema.encoding, data=schema.data)
        cid = w.register_channel(topic=channel.topic, message_encoding=channel.message_encoding, schema_id=sid)
        for _, _, msg in reversed(rows):
            w.add_message(cid, log_time=msg.log_time, publish_time=msg.publish_time, data=msg.data)
        w.finish()
    with p.open("rb") as fh:
        chunk = make_reader(fh).get_summary().chunk_indexes[-1]
    b = bytearray(p.read_bytes())
    at = chunk.chunk_start_offset + 9 + 24
    struct.pack_into("<I", b, at, struct.unpack_from("<I", b, at)[0] ^ 1)
    p.write_bytes(b)
    rep = f.convert(root, "ego_head", tmp_path / "out", "t", 900)
    ep = tmp_path / "out" / rep["episodes"][0]["episode_id"]
    with np.load(ep / "times.npz") as z:
        assert z["exo"] == pytest.approx(np.arange(20) / 30, abs=1e-6)


@pytest.mark.parametrize("damaged", [False, True])
def test_indexed_message_order_keeps_equal_stamp_messages_across_overlapping_chunks(tmp_path, damaged):
    import struct
    from mcap.reader import make_reader
    from mcap.writer import CompressionType, Writer
    p = tmp_path / "rows.mcap"
    with p.open("wb") as fh:
        w = Writer(fh, chunk_size=160, compression=CompressionType.NONE)
        w.start()
        sid = w.register_schema(name="row", encoding="jsonschema", data=b"{}")
        cid = w.register_channel(topic="/rows", message_encoding="json", schema_id=sid)
        for t, value in [(0.3, 0), (0.1, 1), (0.2, 2), (0.1, 3), (0., 4), (0.4, 5), (0.2, 6), (0.2, 6)]:
            ns = int((T0 + t) * 1e9)
            w.add_message(cid, log_time=ns, publish_time=ns, data=json.dumps({"value": value}).encode())
        w.finish()
    if damaged:
        with p.open("rb") as fh:
            chunk = make_reader(fh).get_summary().chunk_indexes[-1]
        b = bytearray(p.read_bytes())
        at = chunk.chunk_start_offset + 9 + 24
        struct.pack_into("<I", b, at, struct.unpack_from("<I", b, at)[0] ^ 1)
        p.write_bytes(b)
    with p.open("rb") as fh:
        damage = []
        rows = list(f.mcap_messages(fh, p, ["/rows"], damage))
    assert [json.loads(m.data)["value"] for _, _, m in rows] == [4, 1, 3, 2, 6, 6, 0, 5]
    assert [m.log_time for _, _, m in rows] == sorted(m.log_time for _, _, m in rows)
    assert bool(damage) == damaged


@pytest.mark.parametrize("damaged", [False, True])
def test_mcap_frames_sharing_stamps_keep_raw_capture_times_separate_from_mux_pts(tmp_path, damaged):
    import struct
    from mcap.reader import make_reader
    from mcap.writer import Writer
    from test_formats import _camera_mcap
    source = tmp_path / "source.mcap"
    _camera_mcap(source, ["/cam/image/compressed"])
    with source.open("rb") as fh:
        rows = list(make_reader(fh).iter_messages())
    root = tmp_path / "up"
    root.mkdir()
    p = root / "rec.mcap"
    with p.open("wb") as fh:
        w = Writer(fh, chunk_size=2000)
        w.start()
        schema, channel, _ = rows[0]
        sid = w.register_schema(name=schema.name, encoding=schema.encoding, data=schema.data)
        cid = w.register_channel(topic=channel.topic, message_encoding=channel.message_encoding, schema_id=sid)
        for k, (_, _, msg) in enumerate(rows):
            ns = int((T0 + (k // 2) / 30) * 1e9)
            w.add_message(cid, log_time=ns, publish_time=ns, data=msg.data)
        w.finish()
    if damaged:
        with p.open("rb") as fh:
            chunk = make_reader(fh).get_summary().chunk_indexes[-1]
        b = bytearray(p.read_bytes())
        at = chunk.chunk_start_offset + 9 + 24
        struct.pack_into("<I", b, at, struct.unpack_from("<I", b, at)[0] ^ 1)
        p.write_bytes(b)
    rep = f.convert(root, "ego_head", tmp_path / "out", "t", 900)
    ep = tmp_path / "out" / rep["episodes"][0]["episode_id"]
    ctx = json.loads((ep / "context.json").read_text())
    with np.load(ep / "times.npz") as z:
        assert z["exo"] == pytest.approx(np.repeat(np.arange(10) / 30, 2), abs=1e-6)
        assert len(z["exo"]) == len(z["exo_pts"]) == 20 and (np.diff(z["exo_pts"]) > 0).all()
        assert f.nearest(z["exo"], z["exo"]).tolist() == [0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12,
                                                         14, 14, 16, 16, 18, 18]
    assert _issues(ctx, "camera_timestamp_repeated")


def test_multiplexed_device_clocks_keep_their_sensor_names_when_left_out(tmp_path):
    from mcap.writer import Writer
    p = tmp_path / 'imu.mcap'
    with p.open('wb') as fh:
        w = Writer(fh)
        w.start()
        schema = w.register_schema(name='imu', encoding='jsonschema', data=b'{}')
        ch = w.register_channel(topic='/imu', message_encoding='json', schema_id=schema)
        for i in range(80):
            t = int((T0 + i / 40) * 1e9)
            w.add_message(ch, t, json.dumps({'type': i % 2 + 1, 'ts': i // 2,
                                           'x': float(np.sin(i))}).encode(), publish_time=t)
        w.finish()
    out = f.mcap_signals([p], T0 + np.arange(60) / 30)
    assert {n for n, _ in out.left_out} == {'/imu (type 1) ts', '/imu (type 2) ts'}
    assert not _issues(out, 'signal_clock_coarse')


@pytest.mark.parametrize('step,leader', [(0.1, False), (1.0, False), (1.0, True)])
def test_coarse_mcap_row_times_need_assumed_alignment_without_false_gaps(tmp_path, step, leader):
    from mcap.writer import Writer
    def change(d):
        p = d / ('leader_left.mcap' if leader else 'yam_left.mcap')
        with p.open('wb') as fh:
            w = Writer(fh)
            w.start()
            schema = w.register_schema(name='arm', encoding='jsonschema', data=b'{}')
            ch = w.register_channel(topic='/leader_left/joint_state' if leader else '/yam_left/joint_state',
                                    message_encoding='json', schema_id=schema)
            for i in range(66):
                t = int((T0 + np.floor(i / 30 / step + 1e-5) * step) * 1e9)
                w.add_message(ch, t, json.dumps({'joint_pos': [float(np.sin(i))] * 6,
                                                'gripper_pos': [0.5]}).encode(), publish_time=t)
            w.finish()
    ctx = _recorder(tmp_path, change)
    if leader:
        assert ctx['state_kind'] == 'joints'
        assert not (ctx['_ep'] / 'action.npy').exists()
    else:
        assert ctx['state_kind'] == 'none' and ctx['state_why'] == 'assumed_clock'
        assert 'coarse' in ctx['state_note']
    assert _issues(ctx, 'signal_clock_coarse')
    assert not _issues(ctx, 'signal_gap') and not _issues(ctx, 'signal_partial_span')
    coarse = [s for s in ctx['signals'] if s.get('aligned_by') == 'coarse clock']
    assert any('joint_pos' in s['name'] for s in coarse)


def test_unplaced_arm_ownership_is_a_layout_reason_rather_than_missing_time():
    note = f.state_blockers([], [], 'the other file',
                            [('arm.mcap', 'several episodes share its folder and its name gives none of their takes',
                              ['/arm/joint_state'])])
    assert note.why == 'layout' and 'several episodes' in str(note)


@pytest.mark.parametrize('source', ['csv', 'h5', 'mcap'])
def test_whole_coarse_signals_prompt_names_the_within_stamp_assumption(tmp_path, source):
    from label import episode as me
    from mcap.writer import Writer
    def change(d):
        t = T0 + np.floor(np.arange(66) / 30)
        vals = np.sin(np.arange(66))
        if source == 'csv':
            pd.DataFrame({'timestamp': t, 'force': vals}).to_csv(d / 'pressure.csv', index=False)
        elif source == 'h5':
            import h5py
            with h5py.File(d / 'pressure.h5', 'w') as h:
                h['timestamp'] = t
                h['force'] = vals
        else:
            with (d / 'pressure.mcap').open('wb') as fh:
                w = Writer(fh)
                w.start()
                schema = w.register_schema(name='pressure', encoding='jsonschema', data=b'{}')
                ch = w.register_channel(topic='/pressure', message_encoding='json', schema_id=schema)
                for stamp, value in zip(t, vals):
                    ns = int(stamp * 1e9)
                    w.add_message(ch, ns, json.dumps({'force': value}).encode(), publish_time=ns)
                w.finish()
    ctx = _recorder(tmp_path, change)
    req = me.build_request(ctx['_ep'])
    text = '\n'.join(c['text'] for c in req['content'] if c['type'] == 'text')
    names = [s['name'] for s in ctx['signals'] if s.get('aligned_by') == 'coarse clock']
    lines = [line for line in text.splitlines() if any(line.startswith('  ' + n + ' (') for n in names)]
    assert lines and all('tied readings placed within each stamp interval as an assumption' in line for line in lines)
    assert all('no clock is shared' not in line for line in lines)


def test_existing_signal_alignment_descriptions_stay_exact():
    from label import signals as sg
    a = np.arange(3)[:, None]
    assert sg.describe('force', a, aligned_by='row per frame') == (
        '  force (1 value, placed one row per frame as it has as many rows as the video has frames): 0 to 2')
    assert sg.describe('force', a, aligned_by='assumed start') == (
        '  force (1 value, placed from both starts as no clock is shared): 0 to 2')


@pytest.mark.parametrize("step", [None, 0.1, 1.0])
def test_hdf5_commands_with_coarse_stamps_stay_signals_beside_precise_state(tmp_path, step):
    import h5py
    from label import episode as me
    def change(d):
        for p in d.glob("*.mcap"):
            p.unlink()
        with h5py.File(d / "robot.h5", "w") as h:
            h["observations/qpos"] = np.tile(np.sin(np.arange(66))[:, None], (1, 14))
            h["observations/timestamp"] = T0 + np.arange(66) / 30
            h["commands/action"] = np.tile(np.cos(np.arange(66))[:, None], (1, 14))
            h["commands/timestamp"] = T0 + (np.arange(66) / 30 if step is None else
                                          np.floor(np.arange(66) / 30 / step + 1e-5) * step)
    ctx = _recorder(tmp_path, change)
    ep = ctx["_ep"]
    assert ctx["state_kind"] == "joints"
    with np.load(ep / "state.npz") as z:
        assert z["state"].shape == (60, 14)
        assert ("action" in z.files) == (step is None)
    req = me.build_request(ep)
    assert ("timebase" in req["plan"]["checks"]) == (step is None)
    if step is not None:
        command = next(s for s in ctx["signals"] if s["name"] == "commands/action")
        assert command["aligned_by"] == "coarse clock"
        with np.load(ep / "signals.npz") as z:
            assert z[command["key"]].shape == (60, 14)
            assert np.isfinite(z[command["key"]]).all()
        text = "\n".join(c["text"] for c in req["content"] if c["type"] == "text")
        line = next(line for line in text.splitlines() if line.startswith("  commands/action ("))
        assert "tied readings placed within each stamp interval as an assumption" in line
    else:
        assert not any(s["name"] == "commands/action" for s in ctx.get("signals", []))


@pytest.mark.parametrize("source", ["h5", "mcap"])
@pytest.mark.parametrize("coarse", [False, True])
def test_variation_signals_keep_their_parents_timing_assumption(tmp_path, source, coarse):
    from label import episode as me
    def change(d):
        t = T0 + (np.arange(660) // 300 if coarse else np.arange(660) / 300)
        vals = np.where(np.arange(660) < 450, 0., np.abs(np.sin(np.arange(660) * 0.9)) *
                        (np.arange(660) % 11 + 1))
        if source == "h5":
            import h5py
            with h5py.File(d / "pressure.h5", "w") as h:
                h["timestamp"] = t
                h["force"] = vals
        else:
            from mcap.writer import Writer
            with (d / "pressure.mcap").open("wb") as fh:
                w = Writer(fh)
                w.start()
                schema = w.register_schema(name="pressure", encoding="jsonschema", data=b"{}")
                ch = w.register_channel(topic="/left/tactile", message_encoding="json", schema_id=schema)
                for stamp, value in zip(t, vals):
                    ns = int(stamp * 1e9)
                    w.add_message(ch, ns, json.dumps({"force": float(value)}).encode(), publish_time=ns)
                w.finish()
    ctx = _recorder(tmp_path, change)
    companion = next(s for s in ctx["signals"] if s.get("variation_of"))
    parent = next(s for s in ctx["signals"] if s["name"] == companion["variation_of"])
    assert companion.get("aligned_by") == parent.get("aligned_by")
    assert (companion.get("aligned_by") == "coarse clock") == coarse
    with np.load(ctx["_ep"] / "signals.npz") as z:
        assert z[companion["key"]].shape == (60, 1) and z[companion["key"]].max() > 0
    req = me.build_request(ctx["_ep"])
    text = "\n".join(c["text"] for c in req["content"] if c["type"] == "text")
    line = next(line for line in text.splitlines() if line.startswith("  " + companion["name"] + " ("))
    assert ("tied readings placed within each stamp interval as an assumption" in line) == coarse
