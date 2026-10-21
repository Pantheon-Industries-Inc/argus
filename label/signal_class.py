"""Where a recorded touch signal sits: on which part of the body or the robot, and on which side.

One function (signal_place) makes the decision for every reader of it (the contacts and the hands they cover,
label/contacts.py; the touch verdicts and the prompt's contacts, label/episode.py; the checks; the board), from the
evidence the recording carries, strongest first, and says which evidence decided it (basis):

- declared: what the source itself records about the field (its description, sensor type or sensor semantics,
  prepare/formats.py SIGNAL_METADATA_KEYS) or what a person set in the field dictionary.
- name: a word of the field's own recorded name (prepare/formats.py word_tokens), read the way a camera's side is.
- interpreted: what the machine field dictionary wrote for it (label/dictionary.py), a reading of the same names and
  metadata, not a declaration, used only when the name says nothing.

A part is read as one of three kinds. A hand part (a hand, a glove, a palm, a thumb) and an end-effector part (a
gripper, its jaw or claw, a finger, which a robot gripper has too) can carry a touch that grasps. Any other part of the
body (a foot, a toe, a heel, a shoe, a leg, a hip, the head) never does: a touch signal placed there is no hand or
gripper contact, whatever its name says about touch. When the hand or end-effector words and another part's words are
both present, the hand or end effector decides (a left_leg_robot/gripper_pressure is on the gripper). Nothing found
leaves the part unknown, and the side, when only the name gives it, is said as the name's.
"""
from __future__ import annotations

HAND_PARTS = ("hand", "glove", "palm", "thumb", "knuckle")
END_EFFECTOR_PARTS = ("gripper", "jaw", "claw", "finger", "fingertip", "fingerpad")
# the rest of a body: parts that never grasp. An arm, a forearm, an elbow and a shoulder are left out, since a gripper
# or a hand sits at their end and a robot names its arms (left_arm/pressure). So are the words a touch sensor's own
# parts share with a body's (a sensor head, a gel's contact face, a trunk cable): read as a part of the body they would
# take a gripper's tactile sensor for one that never grasps, and an unknown part keeps its contacts
OTHER_PARTS = ("foot", "feet", "toe", "heel", "sole", "insole", "shoe", "ankle", "leg", "knee", "thigh", "shin",
               "hip", "pelvis", "torso", "chest", "spine", "neck")
BASES = ("declared", "name", "interpreted")


def _words(text) -> list[str]:
    from prepare.formats import word_tokens
    if isinstance(text, dict):
        return [w for v in text.values() for w in _words(v)]
    if isinstance(text, (list, tuple)):
        return [w for v in text for w in _words(v)]
    return word_tokens(text) if isinstance(text, str) else []


def _part(words: list[str]) -> tuple[str, str] | None:
    """(kind, word): the hand or end-effector part a list of words names, else any other body part, else None. Of two
    parts of one kind the later word is taken, as a path names the most specific last (insole/right_heel: the heel)."""
    def said(parts):
        for w in reversed(words):
            for p in parts:
                if w == p or w == p + "s" or (p.endswith("s") and w == p[:-1]):
                    return p
        return None
    for kind, parts in (("hand", HAND_PARTS), ("end_effector", END_EFFECTOR_PARTS), ("other", OTHER_PARTS)):
        p = said(parts)
        if p:
            return kind, p
    return None


def _evidence(name: str, meta: dict) -> list[tuple[str, list[str]]]:
    """[(basis, words)] in the order they decide."""
    meta = meta or {}
    entry = meta.get("dictionary") or {}
    declared = [meta.get(k) for k in ("description", "sensor_type", "sensor_semantics")]
    human = entry.get("provenance") == "human"
    said = [entry.get("role"), entry.get("meaning")]
    return [("declared", _words(declared) + (_words(said) if human else [])),
            ("name", _words(name)),
            ("interpreted", [] if human else _words(said))]


def signal_place(name: str, meta: dict | None = None) -> dict:
    """{part, kind, basis, side, side_basis, grasps}: where a touch signal sits. part is the word that says it (glove,
    finger, foot) and kind hand, end_effector, other or None; basis says which evidence named it (BASES). side is left
    or right when the name says it (prepare/formats.py recorded_state_side, the one side rule), side_basis then
    "name". grasps is False only for a part that never grasps (kind other): such a signal times no hand or gripper
    contact and instruments no hand."""
    from prepare.formats import recorded_state_side
    meta = meta or {}
    part = None
    for basis, words in _evidence(name, meta):
        found = _part(words)
        if found:
            part = (*found, basis)
            break
    side = recorded_state_side(name)
    return {"part": part[1] if part else None, "kind": part[0] if part else None,
            "basis": part[2] if part else None, "side": side, "side_basis": "name" if side else None,
            "grasps": not part or part[0] != "other"}


def grasps(name: str, meta: dict | None = None) -> bool:
    """Whether a touch signal may time a hand or gripper contact (signal_place)."""
    return signal_place(name, meta)["grasps"]


def contact_place(names, meta: dict | None = None) -> dict:
    """{part, basis, side_basis}: where a contact's signals sit. part is "hand" when any of them is on a hand part (a
    glove's map, a palm and fingertips), else the end-effector word one of them names (gripper, finger), else None;
    basis is what named it, side_basis "name" when a signal's name gives the contact's side."""
    meta = meta or {}
    places = [signal_place(nm, meta.get(nm)) for nm in names or []]
    chosen = (next((p for p in places if p["kind"] == "hand"), None)
              or next((p for p in places if p["kind"] == "end_effector"), None))
    return {"part": ("hand" if chosen["kind"] == "hand" else chosen["part"]) if chosen else None,
            "basis": chosen["basis"] if chosen else None,
            "side_basis": "name" if any(p["side"] for p in places) else None}


def place_words(side: str | None, place: dict | None, signals: int = 1) -> str:
    """Where a contact's signals sit and what says so, as the prompt and the board say it: "left hand, from its
    signal's name", "left side, from its signal's name; the recording does not say which hand or gripper", "the
    recording does not say which hand or gripper"."""
    place = place or {}
    part = place.get("part")
    if not side and not part:
        return "the recording does not say which hand or gripper"
    many = signals > 1
    source = {"name": "its signals' names" if many else "its signal's name",
              "declared": "the recording's own description of " + ("them" if many else "it"),
              "interpreted": "the field dictionary's reading of " + ("them" if many else "it")}
    # a contact's side is read from its signals' names (label/contacts.py), so a contact saved before it carried its
    # place still says where its side came from
    bases = dict.fromkeys(b for b in ((place.get("side_basis") or "name") if side else None, place.get("basis")) if b)
    said = " ".join(x for x in (side, part or "side") if x)
    return (said + (", from " + " and ".join(source[b] for b in bases) if bases else "")
            + ("" if part else "; the recording does not say which hand or gripper"))
