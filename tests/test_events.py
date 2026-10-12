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
    assert [p for p, _ in seen] == ["find", "refine", "name"]
    e0, e1 = res.events
    assert (e0.start_s, e0.label, e0.objects, e0.destination) == (0.5, "pick up the red cup", ["red cup"], None)
    # a boundary moves at most 1.5 s from where it was proposed, and is clipped to the recording
    assert e1.end_s == 4.0
    assert e1.label == "place cup" and "name kept e1: no answer" in res.notes
    assert json.loads(json.dumps(res.to_dict()))["cost_usd"] == 0.003


def test_times_are_grid_instants_including_both_ends():
    fp = _footage(4.0)
    assert fp.times(4)[:3] == [0.0, 0.25, 0.5] and fp.times(4)[-1] == 4.0
    assert fp.times(8, 1.0, 1.5) == [1.0, 1.125, 1.25, 1.375, 1.5]
