"""Contacts: the moments a recording's touch signals say a hand or a gripper is touching something.

A touch signal is one whose name says touch and whose numbers behave like touch (label/signals.py is_touch): a glove's
pressure map, a fingertip's force, a pad's reading. Its name also says which hand it is on, read the way a camera's
side is (prepare/formats.py side_of: right_pressure is on the right hand).

A contact is a span in which one hand's touch signals are away from rest (label/signals.py active_spans), timed from
every recorded frame. Spans of the same hand's signals that overlap are one contact (the pads of one hand touching
together); signals whose names give no hand join only signals from the same source (one MCAP channel, one column). A
signal that never rests in the episode (a pad pressed throughout) sets no span of its own when another signal of the
same hand does: it would make the whole episode one contact and hide the grasps the other signals time. It still
counts in the strength and the regions. A sensor faster than the camera's variation within each frame (prepare's
companion signal) is part of its signal, never a contact of its own. For each contact:

  id            c1, c2, ... in time order
  hand          "left", "right", or null when the signals' names do not say
  signals       the touch signals active in it
  start_s, peak_s, end_s
                when it begins, when it is strongest and when it ends, on the episode's clock; from_start / to_end
                say the contact was already on at the first frame or still on at the last, so its start or end is the
                recording's, not the touch's
  regions       where on the sensor it bears at its peak: for a map, the rows and columns of its active cells and how
                many there are; for a hand of several named signals, which of them are active
  dips_s        the times its strength fell below half of its peak and came back during the contact (a regrasp or a
                slip shows this way; the frames say which)
  aligned_by    "assumed start" when a signal it is timed by was placed on the footage from both starts, as it shares
                no clock with the cameras (prepare/formats.py mark_assumed): its times are then the placement's, not
                recorded times, and the prompt, the checks and the board say so. Absent for a contact of signals on a
                recorded clock (mark_aligned)

Strength is summed over the contact's signals as each signal's activity over its swing (the upload's, when prepare
measured it), so signals in different units add up.
"""
from __future__ import annotations

import numpy as np

from label import signals as sg

DIP_BELOW = 0.5      # a dip is a fall under this share of the contact's peak
VARIATION_SUFFIX = " variation within each frame"   # prepare/formats.py's name for a fast signal's companion
DIP_BACK = 0.8       # that comes back above this share before the contact ends


def _side(name: str) -> str | None:
    from prepare.formats import side_of
    return side_of(name)


def _strength(a: np.ndarray, m: dict) -> np.ndarray:
    act = sg.activity(a, m.get("rest"), m.get("swing"))
    swing = sg.swing_of(a, m.get("rest"), m.get("swing"))
    return np.nan_to_num(act / swing) if swing > 0 else np.zeros(len(a))


def find(signals: dict, meta: dict, t: np.ndarray, verdicts=None) -> list[dict]:
    """The contacts of one episode from its signals ({name: (n, values) array}), their meta (shape, rest, swing) and
    the anchor frames' times t. verdicts, the names of the signals that measure touch when the caller has judged them
    (label/episode.py plan()["touch"], label/pieces.py write_pieces), are used instead of judging each signal again:
    one judgement of a 16 x 16 glove takes about a second."""
    n = len(t)
    touch = {}
    for name, a in signals.items():
        a = np.asarray(a[:n])          # as stored: label/signals.py reads it in float64 pieces
        m = meta.get(name) or {}
        if m.get("variation_of") or name.endswith(VARIATION_SUFFIX):
            continue
        if len(a) == n and (name in verdicts if verdicts is not None
                            else sg.is_touch(name, a, m.get("rest"), m.get("swing"))):
            touch[name] = (a, m)
    if not touch:
        return []
    groups: dict = {}
    for name, (a, m) in touch.items():
        hand = _side(name)
        groups.setdefault(hand if hand else ("source", m.get("source") or name), []).append(name)
    by_hand: dict = {}
    for key, names in groups.items():
        timed = {nm: sg.active_spans(touch[nm][0], t, touch[nm][1].get("rest"), touch[nm][1].get("swing"))
                 for nm in names
                 if sg.rests_and_rises(touch[nm][0], touch[nm][1].get("rest"), touch[nm][1].get("swing"))}
        timed = {nm: sp for nm, sp in timed.items() if sp}
        for nm in names:
            if timed:
                spans = timed.get(nm, [])        # a signal on through the episode sets no span beside timed ones
            else:
                spans = [(float(t[0]), float(t[-1]))] if np.isfinite(touch[nm][0]).any() else []
            for s0, s1 in spans:
                by_hand.setdefault(key, []).append([s0, s1, {nm}])
        if timed:
            # the signals on throughout still count in each contact's strength and regions
            always = {nm for nm in names if nm not in timed}
            for sp in by_hand.get(key, []):
                sp[2] |= always
    contacts = []
    for key, spans in by_hand.items():
        hand = key if isinstance(key, str) else None
        spans.sort(key=lambda x: x[0])
        merged = []
        for s0, s1, names in spans:
            if merged and s0 <= merged[-1][1] + sg.MERGE_GAP_S:
                merged[-1][1] = max(merged[-1][1], s1)
                merged[-1][2] |= names
            else:
                merged.append([s0, s1, set(names)])
        for s0, s1, names in merged:
            k0, k1 = int(np.searchsorted(t, s0 - 1e-9)), int(np.searchsorted(t, s1 + 1e-9)) - 1
            k1 = max(k0, min(k1, n - 1))
            strength = sum(_strength(touch[nm][0], touch[nm][1]) for nm in names)[k0:k1 + 1]
            kp = k0 + int(np.argmax(strength)) if len(strength) else k0
            peak = float(strength.max()) if len(strength) else 0.0
            dips, low = [], False
            for i, x in enumerate(strength):
                if not low and x < DIP_BELOW * peak and i > 0 and strength[:i].max() >= DIP_BACK * peak:
                    low, at = True, i
                elif low and x >= DIP_BACK * peak:
                    dips.append(round(float(t[k0 + at]), 3))
                    low = False
            contacts.append({"hand": hand, "signals": sorted(names), "start_s": round(float(t[k0]), 3),
                             "peak_s": round(float(t[kp]), 3), "end_s": round(float(t[k1]), 3),
                             "from_start": k0 == 0, "to_end": k1 == n - 1, "peak_strength": round(peak, 3),
                             "regions": _regions(touch, names, kp), "dips_s": dips})
    contacts.sort(key=lambda c: (c["start_s"], str(c["hand"])))
    for i, c in enumerate(contacts):
        c["id"] = f"c{i + 1}"
    return mark_aligned([{"id": c.pop("id"), **c} for c in contacts], meta)


def mark_aligned(contacts: list[dict], meta: dict) -> list[dict]:
    """contacts, each timed by a signal placed from both starts (its meta's "aligned_by", prepare/formats.py
    mark_assumed) carrying that aligned_by, so its times are never read as recorded ones. A contact found before
    contacts carried it (a context.json prepared earlier) gets it here from its signals' meta; a contact of signals on
    a recorded clock is returned as it was."""
    out = []
    for c in contacts:
        by = next((m["aligned_by"] for nm in c.get("signals") or [] if (m := meta.get(nm) or {}).get("aligned_by")),
                  None)
        out.append({**c, "aligned_by": by} if by and not c.get("aligned_by") else c)
    return out


def _regions(touch: dict, names: set, k: int) -> dict:
    """Where the contact bears at frame k: per map, its active cells' rows, columns and count; with several named
    signals, which are active."""
    out = {}
    active = []
    for nm in sorted(names):
        a, m = touch[nm]
        act = sg.active_at(a, k, m.get("rest"), m.get("swing"))
        if act.any():
            active.append(nm)
        shape = m.get("shape")
        if shape and len(shape) == 2 and act.any():
            rows, cols = np.divmod(np.flatnonzero(act), int(shape[1]))
            out[nm] = {"cells": int(act.sum()), "rows": [int(rows.min()), int(rows.max())],
                       "columns": [int(cols.min()), int(cols.max())], "of": [int(shape[0]), int(shape[1])]}
    if len(names) > 1:
        out["active_signals"] = active
    return out


def of_episode(ep: dict, verdicts=None) -> list[dict]:
    """The episode's contacts: the ones prepare wrote (context["contacts"], found with the upload's scales), else found
    now from its signals, with the caller's touch verdicts when it has them (find)."""
    if "contacts" in ep["context"]:
        return mark_aligned(ep["context"]["contacts"] or [], ep.get("signal_meta") or {})
    from label import episode as me
    sig = ep.get("signals") or {}
    if not sig:
        return []
    n = len(next(iter(sig.values())))
    return find(sig, ep.get("signal_meta") or {}, np.array([me.frame_time(ep, k) for k in range(n)]), verdicts)
