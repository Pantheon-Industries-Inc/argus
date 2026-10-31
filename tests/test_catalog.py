"""label/catalog.py: reading tasks from results, the reply parser, the merge rules, the coarse list's assign rounds, the
levels, and a whole catalog built against a scripted model, then grown from it as a base. Offline: no model is called."""
import json
import re

import pytest

from label import catalog as C


def result(tasks, rig="handheld_gripper", timeline=()):
    return {"annotation_policy": {"rig": rig}, "labels": {"tasks": tasks, "timeline": list(timeline)}}


def task(a, b, text, objects, outcome="success"):
    return {"start_s": a, "end_s": b, "task": text, "objects": objects, "outcome": outcome}


def test_tasks_come_from_the_task_list_with_the_timeline_as_context_only():
    rec = C.tasks_of(result([task(0, 10, "Fold the towel.", ["towel"]), task(5, 5, "empty span", [])],
                            timeline=[{"start_s": 1, "end_s": 2, "action": "Grasp towel", "contribution": "advancing"},
                                      {"start_s": 2, "end_s": 3, "action": "pause", "contribution": "idle"}]), "r1")
    assert [t["task"] for t in rec["tasks"]] == ["Fold the towel."] and rec["tasks"][0]["acts"] == ["grasp towel"]
    assert rec["rig"] == "handheld_gripper"


def test_results_are_read_from_a_folder_or_json_lines(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(result([task(0, 5, "Wipe the table.", ["cloth"])])))
    (tmp_path / "failed_b.json").write_text(json.dumps(result([task(0, 5, "x", [])])))
    assert [r["name"] for r in C.read_results(tmp_path)] == ["a"]
    lines = tmp_path / "lines.jsonl"
    lines.write_text("\n".join(json.dumps(x) for x in [{"take": "t1", "label": result([task(0, 5, "Wipe.", [])])},
                                                       {"take": "t1", "label": result([])}, result([], rig="ego_head")]))
    recs = C.read_results(lines)
    assert [r["name"] for r in recs] == ["t1", "2"] and len(recs[0]["tasks"]) == 1 and recs[1]["rig"] == "ego_head"


def test_every_text_is_in_exactly_one_call():
    ids = [f"t{i}" for i in range(250)] + ["t3"]
    flat = [x for b in C.batches(ids, 100) for x in b]
    assert sorted(flat) == sorted(set(ids)) and len(flat) == len(set(flat))


def test_one_malformed_row_loses_only_itself():
    rows = C.parse_rows('[{"id": 1, "task": "fold towel"},\n {"id": 2, "task": "stack "cups""},\n {"id": 3, "task": "pour rice"}]')
    assert [r["id"] for r in rows] == [1, 3]
    assert [r[0] for r in C.parse_rows('[0, "pick and place", "rubber duck"]\n[1, "slide", card]\n[2, "fold", "towel"]')] == [0, 2]
    assert C.parse_rows("[[0, 5], [1, null]]") == [[0, 5], [1, None]]
    assert C.parse_rows('[3, "fold", "towel"]') == [[3, "fold", "towel"]]


@pytest.mark.parametrize("a,b", [("sort by color", "sort"), ("tube", "toothpaste tube"), ("card", "playing card"), ("turn", "turn over")])
def test_a_qualifier_or_a_more_general_name_never_merges(a, b):
    assert not C.merge_allowed(a, b)


@pytest.mark.parametrize("a,b", [("toy duck", "rubber duck"), ("put down", "pick and place"), ("duck toy", "toy duck"), ("screws", "screw")])
def test_wording_variants_may_merge(a, b):
    assert C.merge_allowed(a, b)


def test_the_same_words_in_another_order_or_number_are_one_name():
    got = C.reorder_merges({"toy duck": 5, "duck toy": 2, "screws": 2, "screw": 7, "glasses": 4, "glass": 6, "rubber duck": 9})
    assert got == {"duck toy": "toy duck", "screws": "screw"}


def test_candidates_are_more_used_names_sharing_a_word():
    names = ["rubber duck", "towel", "toy duck", "hand towel", "marble"]
    got = C.candidates(names)
    assert got == {2: [0], 3: [1]} and 4 not in got


def test_a_failed_assign_call_is_asked_again_not_counted_as_no_task():
    todo = ["fold towel", "stack cups", "pour rice", "wipe table"]
    assigned, no_task, again = C.assign_round(todo, [range(0, 2), range(2, 4)], [None, [[2, 7], [3, None]]], {7})
    assert assigned == {"pour rice": 7} and no_task == {"wipe table"} and again == ["fold towel", "stack cups"]


def test_a_small_leftover_name_never_founds_a_coarse_task_on_a_base():
    hours = {"big": 2.0, "tiny": 0.01}
    assert C.add_candidates(list(hours), hours, base=True) == ["big"]
    assert C.add_candidates(list(hours), hours, base=False) == ["big", "tiny"]


def test_different_objects_under_one_coarse_task_are_different_specific_tasks():
    t = lambda name, a, b, obj, outcome="success": {"name": name, "a": a, "b": b, "outcome": outcome, "coarse": "place object",
                                                    "family": "handling", "verb": "pick and place", "obj": obj}
    out = {x["task"]: x for x in C.levels([t("r1", 0, 3600, "rubber duck"), t("r2", 0, 1800, "stuffed animal", "failure"),
                                           t("r1", 4000, 5800, "rubber duck"), {**t("r3", 0, 60, "x"), "coarse": None, "verb": None}])}
    c = out["place object"]
    assert (c["hours"], c["tasks"], c["recordings"]) == (2.0, 3, 2)
    fine = {f["task"]: f for f in c["specific"]}
    assert set(fine) == {"pick and place rubber duck", "pick and place stuffed animal"}
    assert fine["pick and place stuffed animal"]["success_hours"] == 0.0 and out[None]["specific"] == []


def test_the_prompts_name_the_rig():
    ego = C.prompts(C.setup_of([{"rig": "ego_head"}]))["specific"]
    assert "head camera" in ego and "pass between\n  hands" in ego
    assert "handheld gripper" in C.prompts(C.setup_of([{"rig": "handheld_gripper"}]))["coarse"]


# ---------------------------------------------------------------- a whole catalog against a scripted model

class Script:
    """A model that names a task from its sentence ("Place the toy duck." -> coarse "place toy duck", verb "pick and
    place", object its first object), maps each coarse name onto a task of the same name, merges nothing on its own and
    passes every check. fail_first makes the first assign call fail."""

    def __init__(self, fail_first=False):
        self.calls, self.fail = collections_counter(), fail_first
        self.model, self.reasoning, self.problems = "scripted", "low", {}
        self.budget = type("B", (), {"spent": 0.0})()

    def __call__(self, text, tokens, model=None):
        kind = ("coarse" if "give its coarse task" in text else "specific" if "give its verb and its object" in text
                else "list" if "Write the final list" in text else "assign" if "Map each name" in text
                else "add" if "Add the fewest new tasks" in text else "judge_merge" if "merged two names" in text
                else "judge" if "Judge both" in text else "other")
        self.calls[kind] += 1
        items = [json.loads(l) for l in text.splitlines() if l.startswith('{"id"')]
        if kind == "coarse":
            return [{"id": x["id"], "task": " ".join(re.findall(r"[a-z]+", x["text"].lower())[:3])} for x in items]
        if kind == "specific":
            return [[x["id"], "pick and place", (x["objects"] or ["thing"])[0].lower()] for x in items]
        if kind == "list":
            names = [l.split("\t")[0] for l in text.split("Names (name<TAB>hours):\n")[1].splitlines() if l.strip()]
            return [{"id": i + 1, "family": "handling", "task": n, "definition": n} for i, n in enumerate(names)]
        if kind == "assign":
            if self.fail and self.calls["assign"] == 1: return None
            tasks = {l.split("\t")[2]: int(l.split("\t")[0]) for l in text.split("Names:")[0].splitlines() if l[:1].isdigit()}
            names = [l.split("\t", 1) for l in text.split("Names:\n")[1].splitlines() if l.strip()]
            return [[int(j), tasks.get(n)] for j, n in names]
        if kind == "judge": return [{"coarse_correct": True, "specific_correct": True}]
        if kind == "judge_merge": return [{"same_task": True}]
        return None

    def many(self, texts, tokens, concurrency, model=None):
        return [self(t, tokens, model) for t in texts]


def collections_counter():
    import collections
    return collections.Counter()


def recs():
    return [C.tasks_of(result([task(0, 600, "Place the toy duck.", ["toy duck"]), task(600, 900, "Fold the towel.", ["towel"])]), "r1"),
            C.tasks_of(result([task(0, 300, "Place the toy duck.", ["duck toy"]), task(300, 1200, "Fold the towel.", ["towels"])]), "r2")]


def test_a_whole_catalog_merges_word_order_and_survives_a_failed_call():
    ask = Script(fail_first=True)
    out = C.build(recs(), ask, check=10)
    s, levels = out["summary"], {x["task"]: x for x in out["levels"]}
    assert s["coarse_tasks"] == 2 and s["tasks_with_no_coarse_task"] == 0
    duck = levels["place the toy"]
    assert [f["task"] for f in duck["specific"]] == ["pick and place toy duck"] and duck["tasks"] == 2
    assert [f["task"] for f in levels["fold the towel"]["specific"]] == ["pick and place towel"]
    assert ask.calls["assign"] >= 2 and out["check"]["merges"]["correct"] == out["check"]["merges"]["judged"] > 0


def test_a_catalog_on_a_base_names_nothing_again_and_keeps_its_tasks(tmp_path):
    first = C.build(recs(), Script())
    ask = Script()
    more = recs() + [C.tasks_of(result([task(0, 60, "Place the toy duck.", ["toy duck"])]), "r3")]
    second = C.build(more, ask, base=first["vocab"])
    assert ask.calls["coarse"] == ask.calls["specific"] == ask.calls["list"] == ask.calls["assign"] == 0
    assert [x["task"] for x in second["vocab"]["coarse_tasks"]] == [x["task"] for x in first["vocab"]["coarse_tasks"]]
    assert {x["task"]: x["recordings"] for x in second["levels"]}["place the toy"] == 3
    C.write(second, tmp_path)
    md = (tmp_path / "catalog.md").read_text()
    assert "2 coarse tasks" in md and (tmp_path / "catalog_specific.csv").read_text().splitlines()[0].startswith("coarse task,task family")
