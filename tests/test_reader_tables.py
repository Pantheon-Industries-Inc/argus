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

def _arm_mcap(path: Path, seconds: float = 4.0, hz: float = 100.0) -> None:
    from mcap.writer import Writer
    with open(path, "wb") as fh:
        w = Writer(fh, chunk_size=2048)
        w.start()
        sid = w.register_schema(name="arm", encoding="jsonschema", data=b"{}")
        ch = w.register_channel(topic="/yam_left/joint_state", message_encoding="json", schema_id=sid)
        for i in range(int(seconds * hz)):
            t = int((T0 + i / hz) * 1e9)
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

def test_a_table_with_one_row_per_frame_is_placed_by_its_own_times(tmp_path):
    n = 240
    pd.DataFrame({"timestamp": 1756534813.0 + np.arange(n) / 10.0, "x": np.arange(n) * 0.5}).to_csv(
        tmp_path / "traj.csv", index=False)
    extra = {}
    out = f.table_signals([tmp_path / "traj.csv"], None, _anchor(n), extra)
    assert out.meta["traj"]["aligned_by"] == f.ALIGNED_ASSUMED
    assert out["traj"][120, 0] == pytest.approx(20.0)              # 4 s in, the row its own clock puts at 4 s
    assert _issues(extra, "signal_alignment_assumed")
    span = _issues(extra, "table_span_differs")
    assert len(span) == 1 and "23.9 s" in span[0]["what"] and "8.0 s" in span[0]["what"]


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


# ---------------------------------------------------------------- a sparse table keeps its rate

def test_a_sparse_table_records_its_rate(tmp_path):
    n = 240
    real = T0 + np.arange(n) / 30.0
    pd.DataFrame({"timestamp": T0 + np.arange(10) * 0.8, "x": np.arange(10) * 1.0}).to_csv(
        tmp_path / "traj.csv", index=False)
    out = f.table_signals([tmp_path / "traj.csv"], real, _anchor(n), {})
    assert out.meta["traj"]["rate_hz"] == pytest.approx(1.25, abs=0.01)
