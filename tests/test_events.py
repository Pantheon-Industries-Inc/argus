"""The event passes and the WGO-Bench scorer, offline: no model call, no download."""
from __future__ import annotations

import json

from bench import wgo
from events import passes
from events.footage import Footage
from events.model import Reply, parse_json


def test_match_in_order_is_one_to_one_and_order_preserving():
    pred = [(0, 5), (5, 10), (10, 20)]
    gold = [(0, 4.8), (9, 20)]
    assert wgo.match_in_order(pred, gold, 0.5) == [(0, 0), (2, 1)]
    # one prediction spanning everything matches at most one gold segment
    assert wgo.match_in_order([(0, 20)], [(0, 10), (10, 20)], 0.4) == [(0, 0)]
    # order: a later prediction cannot match an earlier gold segment once a later one matched
    assert len(wgo.match_in_order([(10, 20), (0, 10)], [(0, 10), (10, 20)], 0.9)) == 1


def test_gold_scored_against_itself_is_perfect_timing():
    gold = [(0.0, 3.0), (3.0, 7.5), (8.0, 12.0)]
    assert wgo.match_in_order(gold, gold, 0.75) == [(0, 0), (1, 1), (2, 2)]


def test_snapping_moves_only_the_outer_boundaries():
    assert wgo.snapped([(1.0, 4.0), (4.0, 9.0)], [(0.0, 4.0), (4.0, 10.0)]) == [(0.0, 4.0), (4.0, 10.0)]
    assert wgo.snapped([], [(0, 1)]) == []


def test_clean_orders_clips_and_splits_overlaps():
    evs = [passes.Event(5, 9, "b"), passes.Event(0, 6, "a"), passes.Event(9, 9.05, "tiny"), passes.Event(-1, 30, "")]
    out = passes.clean(evs, 20.0)
    assert [(e.label, e.start_s, e.end_s) for e in out] == [("a", 0, 5.5), ("b", 5.5, 9)]


def test_clean_per_part_splits_only_events_sharing_a_part():
    E = passes.Event
    evs = [E(0, 6, "pick cup", parts=["right_gripper"]), E(1, 5, "pick lid", parts=["left_gripper"]),
           E(4, 9, "fold cloth", parts=["left_gripper", "right_gripper"]), E(2, 3, "unnamed")]
    out = passes.clean(evs, 20.0, per_part=True)
    got = [(e.label, e.start_s, e.end_s) for e in out]
    # the two hands overlap freely; the both-hands fold splits against each hand's last event; an event whose
    # parts are not known yet is left as found
    assert got == [("pick cup", 0, 5.0), ("pick lid", 1, 4.5), ("unnamed", 2, 3), ("fold cloth", 5.0, 9)]


def test_parse_json_reads_fenced_and_chatty_replies():
    assert parse_json('```json\n{"events": []}\n```') == {"events": []}
    assert parse_json('Here you go: {"a": 1} hope it helps') == {"a": 1}
    assert parse_json("no json") is None


def _footage(duration=4.0):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 36), (40, 40, 40)).save(buf, format="JPEG")
    n = int(duration * 8)
    return Footage(duration_s=duration, cells={"camera": {k: buf.getvalue() for k in range(n + 1)}})


def test_three_passes_with_a_scripted_model():
    fp = _footage()
    replies = {
        "find": '{"events": [{"start_s": 0.4, "end_s": 1.9, "label": "pick cup"}, {"start_s": 2.0, "end_s": 3.5, "label": "place cup"}]}',
        "refine": '{"boundaries": [{"id": "e0.start", "t_s": 0.5}, {"id": "e1.end", "t_s": 9.0}]}',
        "name": '{"events": [{"id": "e0", "label": "pick up the red cup", "skill": "pick", "parts": ["right"], '
                '"objects": ["red cup"], "destination": null}]}',
    }
    seen = []

    def call(content, purpose):
        seen.append((purpose, sum(1 for c in content if c["type"] == "image_url")))
        return Reply(text=replies[purpose], cost=0.001)

    res = passes.annotate(fp, passes.Settings(), call)
    # e1 came back with no name, so it is asked once more on its own
    assert [p for p, _ in seen] == ["find", "refine", "name", "name"]
    e0, e1 = res.events
    assert (e0.start_s, e0.label, e0.objects, e0.destination) == (0.5, "pick up the red cup", ["red cup"], None)
    # a boundary moves at most 1.5 s from where it was proposed, and is clipped to the recording
    assert e1.end_s == 4.0
    assert e1.label == "place cup" and "name kept e1: no answer" in res.notes
    assert json.loads(json.dumps(res.to_dict()))["cost_usd"] == 0.004


def test_naming_asks_again_only_for_events_left_without_a_known_skill():
    fp = _footage()
    find = '{"events": [{"start_s": 0.4, "end_s": 1.9, "label": "pick cup"}, {"start_s": 2.0, "end_s": 3.5, "label": "place cup"}]}'
    names = iter([
        '{"events": [{"id": "e0", "label": "pick up the cup", "skill": "pick", "parts": ["right"]}, '
        '{"id": "e1", "label": "put the cup down", "skill": "juggle"}]}',
        '{"events": [{"id": "e1", "label": "put the cup down", "skill": "place", "parts": ["right"]}]}',
    ])
    asked = []

    def call(content, purpose):
        if purpose != "name":
            return Reply(text=find if purpose == "find" else '{"boundaries": []}')
        text = next(c["text"] for c in content if c["type"] == "text")
        asked.append([i for i in ("e0:", "e1:") if f"- {i}" in text])
        return Reply(text=next(names))

    res = passes.annotate(fp, passes.Settings(skills=["pick", "place"]), call)
    assert asked == [["e0:", "e1:"], ["e1:"]]
    assert [e.skill for e in res.events] == ["pick", "place"]


def test_times_are_grid_instants_including_both_ends():
    fp = _footage(4.0)
    assert fp.times(4)[:3] == [0.0, 0.25, 0.5] and fp.times(4)[-1] == 4.0
    assert fp.times(8, 1.0, 1.5) == [1.0, 1.125, 1.25, 1.375, 1.5]


def test_summary_pass_is_strict_about_outcome():
    import pytest
    fp = _footage()
    replies = {"find": '{"events": [{"start_s": 0.5, "end_s": 2.0, "label": "pick cup"}]}',
               "summary": '{"task_summary": "put the cup in the bowl", "matches_given": null, "outcome": "success", '
                          '"goal_reached_at_s": 3.0, "reason": "cup inside bowl at the end", "scene": {"setting": "table", '
                          '"objects": [{"name": "cup", "location": "left"}]}, "issues": [{"kind": "operator_mistake", '
                          '"category": "drop", "severity": "bogus", "t_s": 1.0, "note": "cup slips"}, {"kind": "x"}]}'}
    res = passes.annotate(fp, passes.Settings(passes=1, summary=True), lambda c, p: Reply(text=replies[p]))
    assert res.summary["outcome"] == "success" and res.summary["issues"] == [
        {"kind": "operator_mistake", "category": "drop", "severity": "medium", "t_s": 1.0, "note": "cup slips"}]
    replies["summary"] = '{"task_summary": "x", "outcome": "maybe"}'
    with pytest.raises(passes.PassError):
        passes.annotate(fp, passes.Settings(passes=1, summary=True), lambda c, p: Reply(text=replies[p]))


def test_split_pass_divides_long_events_only_inside_them():
    fp = _footage(10.0)
    replies = {"find": '{"events": [{"start_s": 0.0, "end_s": 6.0, "label": "tidy"}, {"start_s": 6.5, "end_s": 7.5, "label": "tap"}]}',
               "split": '{"events": [{"id": "e0", "parts": [{"start_s": 0.0, "end_s": 2.5, "label": "pick up cup"}, '
                        '{"start_s": 2.5, "end_s": 6.0, "label": "wipe table"}, {"start_s": 8.0, "end_s": 9.0, "label": "outside"}]}]}'}
    res = passes.annotate(fp, passes.Settings(passes=1, split=True), lambda c, p: Reply(text=replies[p]))
    assert [(e.start_s, e.end_s, e.label) for e in res.events] == [
        (0.0, 2.5, "pick up cup"), (2.5, 6.0, "wipe table"), (6.5, 7.5, "tap")]


def test_refine_asks_fewer_boundaries_per_request_when_the_images_do_not_fit(monkeypatch):
    fp = _footage(8.0)
    events = [passes.Event(i * 1.0, i * 1.0 + 0.8, f"e{i}") for i in range(4)]
    asked = []

    def fitting(build, cell_w, budget=passes.IMAGE_BUDGET_BYTES):
        images = build(160)
        if len({c.split()[3] for c, _ in images if "close-up" in c}) > 2:  # more than two boundaries never fit
            raise passes.PassError("too big")
        return images

    def call(content, purpose):
        asked.append(len({c["text"].split()[3] for c in content if c["type"] == "text" and c["text"].startswith("=== close-up")}))
        return Reply(text='{"boundaries": []}')

    monkeypatch.setattr(passes, "fitting", fitting)
    passes.refine(fp, passes.Settings(), call, [], [], events)
    assert asked == [2, 2, 2, 2]


def _four_camera_footage(moving: str, duration=4.0):
    import io
    from PIL import Image

    def jpeg(shade):
        buf = io.BytesIO()
        Image.new("RGB", (64, 36), (shade, shade, shade)).save(buf, format="JPEG")
        return buf.getvalue()

    n = int(duration * 8)
    still = jpeg(40)
    cams = ["head_left", "head_right", "wrist_left", "wrist_right"]
    cells = {c: {k: (jpeg(40 + (k % 2) * 150) if c == moving else still) for k in range(n + 1)} for c in cams}
    return Footage(duration_s=duration, cells=cells, sizes={c: (64, 36) for c in cams})


def test_naming_shows_the_heads_and_the_acting_wrist():
    chosen = passes.NameViews(fixed=["head_left", "head_right"], choose=["wrist_left", "wrist_right"],
                              by_part={"left_gripper": "wrist_left", "right_gripper": "wrist_right"})
    settings = passes.Settings(convention="sx_label", name_views=chosen)
    fp = _four_camera_footage("wrist_right")
    e = passes.Event(0.5, 2.5, "pick")
    assert passes.views(fp, settings, e) == ["head_left", "head_right", "wrist_right"]
    # parts that name one wrist decide without looking at motion
    e.parts = ["left_gripper"]
    assert passes.views(fp, settings, e) == ["head_left", "head_right", "wrist_left"]
    # two cameras or no choice configured: every camera, as before
    assert passes.views(fp, passes.Settings(), e) == fp.cameras
    line, images = passes._atlas(fp, [passes.Event(0.5, 2.5, "pick")], 0, settings)
    assert "each tile stacks head_left, head_right, wrist_right" in images[0][0]
