"""The recording's contacts checked against what the model saw at them, after labelling.

The touch signals time each contact exactly (label/contacts.py). The model is shown frames around each contact's begin
and end (label/episode.py contact_image) and says, per contact, whether the frames show the hand touching, the first
frame of the begin strip that shows touch and the last of the end strip, and which hand; and it lists the moments a
hand clearly takes hold of something that no contact covers. The two sources are compared here:

  clock_offset       where touch first shows in the frames against where the signal says it begins (and the same at
                     the end), over every contact whose strip brackets it: the median is how far the touch sensor's
                     clock runs from the cameras'. Reported when at least MIN_CONTACTS contacts agree on it and it is
                     more than OFFSET_FRAMES camera frames
  touch_not_seen     contacts the signal records but the frames show no touch at (a sensor that fires on its own, a
                     contact on the other hand, a clock far off)
  hand_mismatch      contacts the model sees on the other hand than the signal's name says; when every contact with a
                     hand disagrees, the gloves look swapped
  contact_missing    moments the model sees a hand take hold of or press something that no recorded contact covers (a
                     sensor that missed it, or one that was off)

The result is context-free numbers and sentences, written by `python -m board build` into the episode's
dataset_checks["contact_checks"]. Like checks/sensors.py they are notes, not counted issues, until the datasets they
fire on are checked on the frames.
"""
from __future__ import annotations

import numpy as np

MIN_CONTACTS = 3
OFFSET_FRAMES = 1.0


def _between(strip: list, f, first: bool) -> float | None:
    """The time touch begins (first) or ends between two frames of a strip, from the frame number the model gave: the
    middle of the gap before the first frame that shows touch, or after the last one. None when the strip does not
    bracket it (touch already showing at frame 1, or still at frame 5) or no frame was given."""
    if not isinstance(f, int) or not strip or not 1 <= f <= len(strip):
        return None
    i = f - 1
    if first:
        return None if i == 0 else (strip[i - 1] + strip[i]) / 2
    return None if i == len(strip) - 1 else (strip[i] + strip[i + 1]) / 2


def check(labels: dict, contacts: list[dict], strips: dict, fps: float) -> dict | None:
    """dataset_checks["contact_checks"] for one episode: the model's contacts (labels["contacts"], by id), the
    recording's (contacts), the strip times each contact picture was built with ({id: {"begin": [...], "end": [...]}})
    and the camera's frame rate. None when the episode has no recorded contacts."""
    if not contacts:
        return None
    seen = {c.get("id"): c for c in (labels.get("contacts") or []) if isinstance(c, dict)}
    by_id = {c["id"]: c for c in contacts}
    notes, offsets = [], []
    for cid, rec in by_id.items():
        m = seen.get(cid)
        if not m:
            continue
        st = strips.get(cid) or {}
        b = _between(st.get("begin") or [], m.get("first_touch_frame"), True)
        e = _between(st.get("end") or [], m.get("last_touch_frame"), False)
        if b is not None:
            offsets.append(("begin", cid, b - rec["start_s"]))
        if e is not None:
            offsets.append(("end", cid, e - rec["end_s"]))
    shown = [cid for cid in by_id if cid in seen]
    frame_ms = 1000.0 / float(fps or 30.0)
    out = {"contacts": len(by_id), "checked": len(shown), "notes": []}
    if offsets:
        ms = np.array([o for _, _, o in offsets]) * 1000.0
        out["offset_ms"] = {"median": round(float(np.median(ms)), 1), "spread": round(float(np.std(ms)), 1),
                            "n": len(ms)}
        if len({cid for _, cid, _ in offsets}) >= MIN_CONTACTS and abs(float(np.median(ms))) > OFFSET_FRAMES * frame_ms:
            lead = "after" if np.median(ms) > 0 else "before"
            notes.append({"check": "clock_offset", "evidence": (
                f"the frames show touch begin and end about {abs(float(np.median(ms))):.0f} ms {lead} the touch "
                f"signal says (median of {len(ms)} measurements over {len({c for _, c, _ in offsets})} contacts, "
                f"spread {float(np.std(ms)):.0f} ms), more than a camera frame of {frame_ms:.0f} ms")})
    not_seen = [cid for cid in shown if str(seen[cid].get("touch_seen")).lower() == "no"]
    if not_seen:
        notes.append({"check": "touch_not_seen", "evidence": (
            f"{len(not_seen)} of the {len(shown)} contacts checked show no touch in the frames "
            f"({', '.join(not_seen)})")})
    pairs = [(cid, by_id[cid].get("hand"), str(seen[cid].get("hand") or "").lower()) for cid in shown]
    known = [(cid, h, m) for cid, h, m in pairs if h in ("left", "right") and m in ("left", "right")]
    wrong = [(cid, h, m) for cid, h, m in known if h != m]
    if wrong:
        swapped = len(wrong) == len(known) and len(known) >= 2
        notes.append({"check": "hand_mismatch", "evidence": (
            ("every contact with a hand is seen on the other hand than its signal's name says "
             f"({', '.join(c for c, _, _ in wrong)}): the left and right sensors look swapped") if swapped else
            f"{len(wrong)} of {len(known)} contacts are seen on the other hand than the signal's name says "
            f"({', '.join(f'{c} recorded {h}, seen {m}' for c, h, m in wrong)})")})
    missing = [x for x in (labels.get("contacts_missing") or []) if isinstance(x, dict)]
    if missing:
        notes.append({"check": "contact_missing", "evidence": (
            f"{len(missing)} moment{'s' if len(missing) != 1 else ''} where a hand takes hold of or presses something "
            "with no recorded contact: " + "; ".join(
                f"{float(x.get('t_s') or 0):.1f} s, {x.get('hand') or 'a hand'}, {x.get('object') or 'an object'}"
                for x in missing[:6]))})
    out["notes"] = notes
    return out
