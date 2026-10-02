"""The board's touch view: each contact's strength in the sensors file (board/sensors.py), the recording's contacts
joined with the model's answers in the episode file (board/build.py add_contacts), and the Touch lane and contact card
the page draws from them (tests/touch_lane.js)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from board import build as board_build
from board import sensors
from label import contacts as lc


def _episode(root: Path, n: int = 300, fps: float = 30.0) -> Path:
    """A prepared episode whose right glove's 4 x 4 pressure map (with a real sensor's noise) is pressed twice, beside a
    joint velocity that is no touch, and whose contacts prepare found (label/contacts.py)."""
    ep = root / "episode_000000"
    ep.mkdir(parents=True)
    rng = np.random.default_rng(0)
    pmap = (3072.0 + rng.normal(0, 2, (n, 16))).astype(np.float32)
    pmap[60:90, 5] -= 1000.0
    pmap[200:240, 10] -= 1500.0
    vel = rng.normal(0, 1, (n, 2)).astype(np.float32)
    np.savez(ep / "signals.npz", s0=pmap, s1=vel)
    t = np.arange(n) / fps
    np.savez(ep / "times.npz", exo=t, exo_pts=np.arange(n) * 512)
    metas = [{"name": "right_pressure", "key": "s0", "dims": 16, "shape": [4, 4]},
             {"name": "joint velocity", "key": "s1", "dims": 2}]
    found = lc.find({"right_pressure": pmap.astype(np.float64)}, {"right_pressure": {"shape": [4, 4]}}, t)
    ctx = {"dataset": "you/your-own-dataset", "profile": "teleop_arms", "fps": fps, "n_state_frames": n,
           "signals": metas, "contacts": found}
    (ep / "context.json").write_text(json.dumps(ctx))
    (ep / "sources.json").write_text(json.dumps({"exo": {"packed": str(ep / "exo.mp4"), "base_s": 0.0,
                                                          "n_frames": n}}))
    return ep


def test_a_contacts_signals_keep_its_strength_in_the_sensors_file(tmp_path):
    ep = _episode(tmp_path)
    ctx = json.loads((ep / "context.json").read_text())
    assert [c["hand"] for c in ctx["contacts"]] == ["right", "right"]
    doc = sensors.episode_doc(ep)
    pmap, vel = doc["signals"]
    assert "strength" in pmap and "strength" not in vel        # only a signal a contact is timed by
    got = sensors.dequantize(pmap["strength"], doc["n"])[:, 0]
    a = np.load(ep / "signals.npz")["s0"].astype(np.float64)
    want = lc._strength(a, {})[::doc["stride"]]
    assert np.max(np.abs(got - want)) <= pmap["strength"]["step"][0] / 2 + 1e-9
    # the curve peaks inside each contact, as high as the contact's own peak strength
    for c in ctx["contacts"]:
        inside = [i for i in range(doc["n"]) if c["start_s"] <= i * doc["stride"] / 30.0 <= c["end_s"]]
        assert max(got[inside]) == pytest.approx(c["peak_strength"], rel=0.05)


def _run(runs: Path, ep: Path, contacts: list) -> Path:
    run = runs / "r1"
    (run / "out").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": "r1", "code": "abc1234", "kind": "full", "status": "done",
                                              "slice": str(ep.parent)}))
    first = contacts[0]
    labels = {"task_summary": "press the pad", "timeline": [{"start_s": 0.0, "end_s": 9.0, "action": "press"}],
              "completion": {"task_completed": "success", "completed_at_s": 9.0},
              "contacts": [{"id": first["id"], "touch_seen": "no", "first_touch_frame": None, "last_touch_frame": None,
                            "hand": "right", "object": "nothing visible", "grip": None, "action": None,
                            "slip": "unclear", "notes": None}],
              "contacts_missing": [{"t_s": 5.0, "hand": "left", "object": "cup"}]}
    strips = {first["id"]: {"begin": [first["start_s"] + o for o in (-0.3, -0.15, 0.0, 0.15, 0.3)],
                            "end": [first["end_s"] + o for o in (-0.15, 0.0, 0.15)]}}
    (run / "out" / f"{ep.name}.json").write_text(json.dumps({
        "episode_dir": str(ep), "parse_ok": True, "model": "some/model", "labels": labels,
        "contact_views": {"shown": [first["id"]], "strips": strips}}))
    return run


def test_board_build_joins_each_contact_with_what_the_model_saw(tmp_path):
    ep = _episode(tmp_path / "eps")
    recorded = json.loads((ep / "context.json").read_text())["contacts"]
    board = tmp_path / "board"
    board.mkdir()
    (board / "manifest.json").write_text(json.dumps({"board": "b", "datasets": [
        {"dataset": "mine", "run": str(_run(tmp_path / "runs", ep, recorded)), "episodes": str(ep.parent)}]}))
    board_build.build(board)
    d = json.loads((board / "qa" / "episode_000000.json").read_text())
    c1, c2 = d["contacts"]
    assert c1["shown"] is True and c1["seen"]["touch_seen"] == "no" and c2["shown"] is False and "seen" not in c2
    assert {k: c1[k] for k in ("start_s", "end_s", "peak_s", "regions")} == \
        {k: recorded[0][k] for k in ("start_s", "end_s", "peak_s", "regions")}
    assert d["contacts_missing"] == [{"t_s": 5.0, "hand": "left", "object": "cup"}]
    assert "contacts_model" not in d and "contact_views" not in d
    cc = d["dataset_checks"]["contact_checks"]
    assert cc["contacts"] == 2 and cc["checked"] == 1
    assert {n["check"] for n in cc["notes"]} == {"touch_not_seen", "contact_missing"}
    # the strength the Touch lane draws is in the episode's sensors file
    sn = json.loads((board / "sensors" / "episode_000000.json").read_text())
    assert "strength" in sn["signals"][0]


@pytest.mark.skipif(not shutil.which("node"), reason="no node")
def test_the_touch_lane_and_contact_card_say_what_they_must():
    """tests/touch_lane.js on the page's touch block: a bar per hand, each contact styled by what the model found, the
    lane's counts, the grasps with no contact on their hand's row, the card's times and sentences, the strength
    curve."""
    here = Path(__file__).resolve().parent
    r = subprocess.run([shutil.which("node"), str(here / "touch_lane.js"), str(here.parent / "board" / "serve.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
