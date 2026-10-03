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


