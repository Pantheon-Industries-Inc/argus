"""What the request states about a recording's signals is what the recording establishes: a touch signal on a part
that never grasps is no hand contact, where a contact's signals sit carries what says so, and a value with no clock in
common with the cameras is never shown at a frame instant.

The fixtures take the shapes of the audited ActionSense head-camera recording (S05, Argus 4b98cca): two 32 x 32 glove
maps that rest near 555 and rise where pressed, Xsens foot contact points (four 0 or 1 values), and a Myo armband's
acceleration, every one of them placed on the footage from both starts because no clock is shared."""
from __future__ import annotations

import re

import numpy as np
import pytest

from label import contacts as lc
from label import episode as me

N = 1533
FOOT = "xsens-foot-contacts/foot_contact_points/data"
GLOVE = "tactile-glove-left/tactile_data/data"
MYO = "myo-left/acceleration_g/data"
IMU = "imu/linear_acceleration"


def _glove(on=(180, 240)):
    a = 555.0 + np.random.default_rng(0).normal(0, 1, (N, 1024))
    a[on[0]:on[1], 246] += 130.0
    a[on[0]:on[1], 278] += 100.0
    return a


# the audited recording's four foot contact points over its 1533 frames, as runs of (first frame, value): three of them
# down until near the end, the fourth down only there, with the half readings a placement between two samples gives
FOOT_RUNS = ([(0, 1.0), (1359, 0.0), (1388, 1.0), (1452, 0.0), (1476, 1.0)],
             [(0, 1.0), (1350, 0.0), (1390, 1.0), (1394, 0.5), (1395, 0.0), (1450, 1.0), (1454, 0.5), (1455, 1 / 3),
              (1456, 0.0), (1501, 1.0)],
             [(0, 1.0), (1393, 0.5), (1394, 0.0), (1448, 0.5), (1449, 1.0), (1481, 0.0), (1510, 0.5), (1511, 1.0)],
             [(0, 0.0), (1448, 0.5), (1450, 1.0), (1453, 0.5), (1454, 0.0), (1483, 1.0), (1484, 0.0), (1516, 0.5),
              (1517, 1.0)])


def _foot():
    a = np.zeros((N, 4))
    for c, runs in enumerate(FOOT_RUNS):
        for (k, v), (k1, _) in zip(runs, runs[1:] + [(N, None)]):
            a[k:k1, c] = v
    return a


def _steps():
    a = np.full((N, 3), 0.95)
    a[200:330] = [-0.07, 0.3, 0.6]
    return a


SIGNALS = {GLOVE: _glove(), FOOT: _foot()}
META = {GLOVE: {"shape": [32, 32], "swing": 130.0, "rest": [555.0] * 1024, "aligned_by": "assumed start"},
        FOOT: {"aligned_by": "assumed start"}}
T = np.arange(N) / 30.0


def test_a_touch_signal_on_a_part_that_never_grasps_times_no_contact_and_instruments_no_hand():
    found = lc.find(SIGNALS, META, T)
    assert found and all(c["signals"] == [GLOVE] for c in found)
    assert lc.coverage(META) == {"left": [GLOVE]}
    assert all(c["place"] == {"part": "hand", "basis": "name", "side_basis": "name"} for c in found)


@pytest.mark.parametrize("name, meta, grasps, part, basis", [
    (FOOT, {}, False, "foot", "name"),
    ("insole/right_heel_pressure", {}, False, "heel", "name"),
    ("left_leg_robot/gripper_pressure", {}, True, "gripper", "name"),
    (GLOVE, {}, True, "glove", "name"),
    ("observation.tactile_left_raw", {}, True, None, None),
    # the recording's own description decides before the name, and a person's dictionary edit counts as declared
    ("pad_pressure", {"description": "pressure insole under the left foot"}, False, "foot", "declared"),
    ("pad_pressure", {"dictionary": {"role": "pressure", "meaning": "worn on the right shoe",
                                     "provenance": "human"}}, False, "shoe", "declared"),
    # a machine reading speaks only where the name says nothing
    ("pad_pressure", {"dictionary": {"role": "pressure", "meaning": "a fingertip pad",
                                     "provenance": "machine"}}, True, "fingertip", "interpreted"),
    (GLOVE, {"dictionary": {"role": "contact_state", "meaning": "foot contact", "provenance": "machine"}},
     True, "glove", "name"),
    # a touch sensor's own parts named like a body's are no part of a body: its contacts stay
    ("observation.tactile_left_raw", {"description": "DIGIT sensor head, gel contact face"}, True, None, None),
    ("observation.tactile_left_raw", {"dictionary": {"role": "tactile_image", "meaning": "GelSight sensor head",
                                                     "provenance": "machine"}}, True, None, None),
])
def test_where_a_touch_signal_sits_comes_from_the_strongest_evidence_and_says_which(name, meta, grasps, part, basis):
    from label import signal_class as sc
    place = sc.signal_place(name, meta)
    assert (place["grasps"], place["part"], place["basis"]) == (grasps, part, basis)


def test_contacts_a_context_prepared_before_held_for_a_foot_are_not_read_as_a_hands():
    ep = {"context": {"contacts": [{"id": "c1", "hand": None, "signals": [FOOT], "start_s": 2.0, "peak_s": 3.0,
                                    "end_s": 5.0, "aligned_by": "assumed start"}]},
          "signals": dict(SIGNALS), "signal_meta": {k: dict(v) for k, v in META.items()}}
    assert lc.of_episode(ep) == []


def _ego(signals: dict, meta: dict):
    ep = {"context": {"dataset": "you/head", "fps": 30, "profile": "ego_head", "state_kind": "none",
                      "cameras": {"exo": {"width": 1280, "height": 720}}},
          "state": np.zeros((N, 0)), "sources": {"exo": {}}, "signals": signals, "signal_meta": meta}
    ks = list(range(0, N, 15))
    return ep, {"n": N, "ks": ks, "spans": [], "state_usable": True, "contact": []}


def test_an_unaligned_signal_is_never_given_at_a_frame_instant():
    """The Myo armband shares no clock with the head camera: its range is given under a line saying its timing is
    unknown, and it has no row at the instants, no time of its extremes and no relation to another row. A signal on
    the camera's clock is read at the instants as before."""
    signals = {MYO: _steps(), IMU: _steps()}
    meta = {MYO: {"aligned_by": "assumed start"}, IMU: {}}
    ep, pl = _ego(signals, meta)
    table = me.build_prompt(ep, pl, cell_w=256, cell_h=144)[1].split("OTHER RECORDED SIGNALS")[1]
    table = table.split("BETWEEN INSTANTS")[0]
    head, instants = table.split("at every instant you receive")
    assert f"    {IMU} [0]: " in instants and MYO not in instants
    unplaced = head.split("their timing relative to the video is unknown")[1]
    assert unplaced.lstrip(". ").count(MYO) == 1 and "placed from both starts" not in table
    assert me.untimed_signals(ep) == {MYO}


def test_a_contact_of_an_unaligned_touch_signal_is_counted_never_timed_on_the_frames():
    ep, pl = _ego({GLOVE: _glove()}, {GLOVE: dict(META[GLOVE])})
    pl["touch"] = frozenset({GLOVE})
    ep["contacts"] = lc.find(ep["signals"], ep["signal_meta"], T)
    ep["contacts_shown"] = me.chosen_contacts(ep, ep["contacts"])
    assert ep["contacts"] and ep["contacts_shown"] == []
    block = me.contacts_block(ep, pl)
    assert "share no clock with the cameras" in block and '"contacts": []' in block
    assert "s to " not in block and '"contacts_missing": [], because the touch signals share no clock' in block


@pytest.mark.parametrize("meta, context, grade", [
    ({}, {}, "recorded"),
    ({"aligned_by": "recorder arrival"}, {}, "recorder arrival"),
    ({}, {"source": {"presentation_clock": {"basis": "mcap_log_time"}}}, "recorder arrival"),
    ({"aligned_by": "coarse clock"}, {}, "placed on frames"),
    ({"aligned_by": "row per frame"}, {}, "placed on frames"),
    ({"camera_aligned_by": "assumed camera clock"}, {}, "placed on frames"),
    ({"clock_problem": "backwards steps"}, {}, "placed on frames"),
    ({}, {"cameras": {"exo": {}}, "camera_clock": {"exo": {"clock_problem": "nonfinite"}}}, "placed on frames"),
    ({"aligned_by": "assumed start"}, {}, "untimed"),
    ({"aligned_by": "a placement from the future"}, {}, "untimed"),
    ({}, {"state_unaligned": "the state's camera was taken out"}, "untimed"),
])
def test_one_rule_grades_how_a_values_time_on_the_video_is_established(meta, context, grade):
    from prepare import signal_alignment as al
    got = al.video_timing(meta, context)
    assert got["grade"] == grade
    assert got["timed"] == (grade != "untimed") and got["verified"] == (grade == "recorded")


# ---- the invariant every request holds, applied here and to every setup of tests/test_setup_matrix.py ----------------

CONTACT_LINE = re.compile(r"^  (c\d+): from (.+?) \((.+?)\), ")
PLACE_SAID = ("from its signal", "the recording's own description", "the field dictionary's reading",
              "the recording does not say which hand or gripper")


def untruths(ctx: dict, prompt: str) -> list[str]:
    """What a request states about its recording that the recording does not establish, as one line each; empty when
    it holds. A signal with no clock in common with the cameras (prepare/signal_alignment.py video_timing) appears in
    no per-instant row, no still-span change, no time of an extreme and no contact line; a contact is timed only by
    signals on a part that grasps (label/signal_class.py signal_place) and says where its side and part come from; the
    hands the contacts block lists hold no signal on a part that never grasps."""
    from label import signal_class as sc
    from label.dictionary_context import field_interpretation
    from prepare import signal_alignment as al
    meta = {s["name"]: {**s, "dictionary": field_interpretation(ctx, s["name"])}
            for s in ctx.get("signals") or [] if isinstance(s, dict) and s.get("name")}
    untimed = {name for name, m in meta.items() if not al.timed(m, ctx)}
    out = []
    lines = prompt.splitlines()
    readout = prompt.split("at every instant you receive")[1].split("BETWEEN INSTANTS")[0] \
        if "at every instant you receive" in prompt else ""
    still = [ln for ln in lines if re.match(r"^    \d+\.\d\d-\d+\.\d\ds: ", ln)]
    extremes = [ln for ln in lines if re.search(r"\((max|min) [^()]* at \d+\.\d\d s", ln)]
    for name in untimed:
        if f"    {name} " in readout or f"    {name}:" in readout:
            out.append(f"{name} shares no clock with the cameras and has a row at the sampled instants")
        if any(name + " " in ln for ln in still):
            out.append(f"{name} shares no clock with the cameras and is given over a still span")
        if any(name in ln for ln in extremes):
            out.append(f"{name} shares no clock with the cameras and is given the time of an extreme")
    for ln in lines:
        m = CONTACT_LINE.match(ln)
        if not m:
            continue
        cid, signals, place = m.groups()
        names = [n for n in meta if n in signals]
        if any(n in untimed for n in names):
            out.append(f"{cid} is timed on the frames by a signal that shares no clock with the cameras")
        if any(not sc.grasps(n, meta[n]) for n in names):
            out.append(f"{cid} is a contact of a signal on a part that never grasps")
        if not any(said in place for said in PLACE_SAID):
            out.append(f"{cid} names a hand or side without saying what establishes it: {place}")
    header = "\n".join(ln for ln in lines if ln.startswith("CONTACTS: "))
    out += [f"the contacts block lists {n} as a hand's, which is on a part that never grasps"
            for n in meta if n in header and not sc.grasps(n, meta[n])]
    return out


def test_the_invariant_reads_zero_on_the_audited_recording_and_catches_each_untruth():
    ctx = {"signals": [{"name": GLOVE, **META[GLOVE]}, {"name": FOOT, **META[FOOT]}, {"name": MYO, **META[FOOT]}]}
    ep, pl = _ego({MYO: _steps(), GLOVE: _glove(), FOOT: _foot()},
                  {MYO: {"aligned_by": "assumed start"}, GLOVE: dict(META[GLOVE]), FOOT: dict(META[FOOT])})
    ep["context"]["contacts"] = [{"id": "c1", "hand": None, "signals": [FOOT], "start_s": 2.0, "peak_s": 3.0,
                                  "end_s": 5.0, "aligned_by": "assumed start"}]
    pl["touch"] = frozenset({GLOVE, FOOT})
    ep["contacts"] = me.touch_contacts(ep, pl, lc.of_episode(ep, pl["touch"]))
    ep["contacts_shown"] = me.chosen_contacts(ep, ep["contacts"])
    prompt = "".join(me.build_prompt(ep, pl, cell_w=256, cell_h=144))
    assert untruths(ctx, prompt) == []
    # each kind of untruth the main branch told about this recording is caught
    told = prompt + ("\n  Each signal that changes, at every instant you receive:\n    at: 0.00 0.50\n"
                     f"    {MYO} [0]: 0.95 0.95\nBETWEEN INSTANTS\n")
    told += (f"\nCONTACTS: instrumented hands and their recorded touch signals: left: {GLOVE}, {FOOT}.\n"
             f"  c1: from {FOOT} (hand not named by the recording), 45.52 s to 50.91 s, largest response at 50.01 s\n"
             f"  c2: from {GLOVE} (left hand), 1.00 s to 2.00 s, largest response at 1.50 s\n")
    found = untruths(ctx, told)
    assert any(MYO in x and "sampled instants" in x for x in found)
    assert any(x.startswith("c1 is a contact of a signal on a part that never grasps") for x in found)
    assert any(x.startswith("c1 is timed on the frames") for x in found)
    assert any(x.startswith("c2 names a hand or side without saying") for x in found)
    assert any("lists " + FOOT in x for x in found)


def test_the_inspection_inventory_says_which_fields_take_seconds(tmp_path):
    """The audited run's first selection round asked the left glove and the gaze confidence for a 0 to 51.69 s window;
    both share no clock with the cameras, so both requests failed and the round bought nothing. The inventory now
    says each numeric field takes source rows only, a refused window names the rows to ask instead, and a field on the
    camera's clock is still read by seconds."""
    from label import evidence_access as ea
    ctx = {"episode_id": "episode_a", "fps": 10, "state_kind": "none"}
    signals = {GLOVE: _glove()[:4, :8], IMU: _steps()[:4]}
    a = ea.Access({"dir": tmp_path, "context": ctx, "sources": {}, "state": np.zeros((4, 0)), "action": None,
                   "times": None, "signals": signals,
                   "signal_meta": {GLOVE: {"aligned_by": "assumed start"}, IMU: {}}, "depth": {}})
    fields = {f["name"]: f for f in a.inventory() if f["kind"] == "numeric"}
    assert fields[GLOVE]["select_by"].startswith("start_row/end_row or rows only")
    assert fields[IMU]["select_by"].startswith("start_s/end_s")
    with pytest.raises(ValueError, match="start_row and end_row, or rows"):
        a.inspect({"field_id": fields[GLOVE]["id"], "mode": "extrema", "start_s": 0, "end_s": 0.3})
    rows = a.inspect({"field_id": fields[GLOVE]["id"], "mode": "extrema", "start_row": 0, "end_row": 4})
    assert "times_s" not in rows
    assert a.inspect({"field_id": fields[IMU]["id"], "start_s": 0, "end_s": 0.3})["times_s"] == pytest.approx(
        [0, .1, .2, .3])
