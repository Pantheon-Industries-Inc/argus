"""What each pass asks, and the convention that says what one event is.

A convention is the one thing that differs between a benchmark and a customer: where an event starts and ends and
how coarse it is. WGO-Bench (VideoDB revision) counts one completed manipulation of one object, reach to release,
as one event; the SentientX Label spec splits a pick from its place and names the hand. The passes are the same.
"""
from __future__ import annotations

CONVENTIONS = {
    "wgo": """\
What one event is (follow this exactly):
- One event is one completed manipulation of one object: from the reach toward it until the hand or gripper lets
  go of it after the last thing done with it. Picking an object up, carrying it, pouring from it and putting it
  down is ONE event ("pick up the test tube with red liquid, pour it into the beaker and place the tube in the
  cup"). Do not split an event into reach, grasp, carry, pour and place.
- Grip adjustments, re-grasps and short pauses stay inside their event.
- A new event starts when a different object is handled, or when the same object is picked up again after it was
  let go. When something else happens while an object is held (wiping a surface while holding a cup), the pick-up,
  the other action and the put-down are separate events.
- Idle time, walking and camera motion with no manipulation are not events: leave them uncovered.
- Events never overlap. A failed attempt that is retried is its own event.
{density}""",
    "sx_label": """\
What one event is (follow this exactly):
- One event is one skill applied by one or both hands or grippers: pick, place, carry (a real distance only),
  hand over, pour, insert, open, close, push, pull, wipe, fold, press, twist and the like. A short pick-up-and-put-
  down is TWO events (pick, then place), never a carry.
- An event starts when the hand or gripper starts moving toward its target and ends when the skill is complete
  (released, set down, closed).
- Idle time is not an event. Events of one hand never overlap; the two hands may act at once, as two events.
{density}""",
}

FIND = """\
You are annotating a recording of manipulation. The images are contact sheets: instants sampled every {step:.2f} s
across the whole recording, read left to right then top to bottom, each tile headed with its time in seconds.
{cameras}
{instruction}

List every event in the recording, in time order, with its start and end time in seconds and a short label.

{convention}

Boundaries may fall between the sampled instants: give them to 0.05 s where the pictures support it. Every event you
miss now is lost: later steps can move boundaries and rename events but never add one. Look for short events.

Answer with JSON only: {{"events": [{{"start_s": 0.0, "end_s": 0.0, "label": "..."}}]}}
The recording lasts {duration:.2f} s."""

REFINE = """\
You are checking the start and end times of events already found in a recording of manipulation.
{cameras}
{instruction}

First comes an overview of the whole recording, one instant every {overview_step:.0f} s. Then, for each boundary
below, a close-up sheet at 8 instants per second spanning {window:.1f} s either side of the proposed time. Every tile
is headed with its time in seconds.

{convention}

Boundaries to check:
{boundaries}

For each boundary give the time it really falls at, to 0.05 s, judged on the close-up. A start is the first instant
the event's movement begins; an end is the first instant the event is complete. Keep the proposed time when the
close-up does not show a better one. A boundary may move at most {window:.1f} s.

Answer with JSON only: {{"boundaries": [{{"id": "e3.start", "t_s": 0.0}}]}} with one entry per boundary listed."""

NAME = """\
You are naming events in a recording of manipulation. Each event's start and end are fixed; do not change them and
do not add or drop events.
{cameras}
{instruction}

For each event you get one atlas: up to five instants from the previous event (prev), up to five from this event
(this) and up to five from the next event (next), each tile headed with its time and role. Name only what happens
during "this"; use prev and next to tell where an object came from and where it ends up.

{convention}
{vocabulary}
Events to name:
{events}

Answer with JSON only: {{"events": [{{"id": "e3", "label": "...", "skill": "...", "parts": ["..."],
"objects": ["..."], "destination": "... or null"}}]}} with one entry per event listed. The label is one short
self-contained phrase in the imperative ("pick up the blue mug from the shelf and place it in the sink"), naming
the object by colour, material or position when that tells it apart from similar objects."""


DENSITY = {
    "teleop_arms": "- This is robot video: a robot usually handles one to four objects per recording, so expect one to four "
                   "events, each lasting several seconds or more.",
    "handheld_gripper": "- This is handheld-gripper video: expect one event per object handled.",
    "ego_head": "- This is first-person human video: people act fast and often, so expect many events, often one to four "
                "seconds each (a minute of video commonly holds 10 to 20). List every one, including quick pick-ups, "
                "put-downs, slides and adjustments of other objects.",
}


def convention_text(convention: str, rig: str) -> str:
    return CONVENTIONS[convention].format(density=DENSITY.get(rig, ""))


def instruction_line(instruction: str | None) -> str:
    if not instruction:
        return "No task instruction was given: infer what the person or robot is doing from the pictures."
    return (f"The task instruction given with the recording is: \"{instruction}\". It is a claim: the recording may "
            "do more, less or something else. Label what actually happens.")


def cameras_line(cameras: list[str], rig: str) -> str:
    what = {"ego_head": "a camera worn on a person's head", "teleop_arms": "a camera watching robot arms",
            "handheld_gripper": "cameras on handheld grippers"}.get(rig, "the recording's cameras")
    if len(cameras) == 1:
        return f"The footage is from {what}."
    return f"The footage is from {what}; each tile stacks the cameras top to bottom: {', '.join(cameras)}."


def vocabulary_line(skills: list[str] | None, parts: list[str] | None) -> str:
    if not skills:
        return ('Give "skill" as one verb, "parts" as the hands or grippers that act (left, right or both) and '
                '"objects" as the objects acted on.\n')
    return (f'"skill" must be one of: {", ".join(skills)}. "parts" must be from: {", ".join(parts or [])}. '
            '"objects" names the objects acted on.\n')
