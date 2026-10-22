"""Episode and subgoal quality scores (board/scores.py): measured from the labels, compared within a task group."""
from board import scores


def _record(prompt, goal_s, steps, subgoals=(), mistakes=(), outcome="success"):
    return {"dataset": "ds", "episode_prompt": prompt, "duration_s": steps[-1][1],
            "event_labels": [{"t_s": a, "end_s": b, "contribution": c, "progress": p} for a, b, c, p in steps],
            "key_events": [{"t_s": t, "kind": "subgoal_complete", "outcome": "success", "label": label}
                           for t, label in subgoals],
            "operator_mistakes": list(mistakes),
            "completion": {"task_completed": outcome, "completed_at_s": goal_s, "success_predicate": "done"}}


def _board(n_peers=6, peer_goal=10.0):
    peers = {f"p{i}.json": _record("open the left door then the right door", peer_goal,
                                   [(0, peer_goal / 2, "advancing", 0.5), (peer_goal / 2, peer_goal, "advancing", 1.0)],
                                   [(peer_goal / 2, "left door open")]) for i in range(n_peers)}
    return peers


def test_a_clean_demo_at_the_typical_time_scores_full_and_a_slow_one_loses_speed_only():
    recs = _board()
    recs["slow.json"] = _record("open the left door, then open the right door", 20.0,
                                [(0, 10, "advancing", 0.5), (10, 20, "advancing", 1.0)], [(10, "left door open")])
    s = scores.score_board(recs)
    assert s["p0.json"]["efficiency"] == 10.0 and s["p0.json"]["cleanliness"] == 10.0
    slow = s["slow.json"]
    assert slow["peers"] == 6 and slow["efficiency"] == 5.0          # twice the median time
    assert slow["cleanliness"] == 10.0 and slow["outcome"] == 10.0
    assert slow["inputs"]["peer_median_time_to_goal_s"] == 10.0


def test_mistakes_and_wasted_time_lower_cleanliness_and_a_recovered_mistake_costs_half():
    recs = _board()
    recs["messy.json"] = _record("open the left door then the right door", 10.0,
                                 [(0, 4, "advancing", 0.3), (4, 6, "wasteful", 0.3), (6, 10, "advancing", 1.0)],
                                 [(5, "left door open")],
                                 [{"issue": "dropped the handle", "severity": "high", "recovered": True, "t_s": 4}])
    s = scores.score_board(recs)["messy.json"]
    assert s["inputs"]["wasteful_share"] == 0.2
    assert s["cleanliness"] == round(10 * 0.8 - 3.0 * 0.5, 1)


def test_too_few_peers_leaves_the_relative_scores_empty_never_guessed():
    recs = _board(n_peers=3)
    s = scores.score_board(recs)["p0.json"]
    assert s["efficiency"] is None and s["cleanliness"] == 10.0 and s["execution"] == 10.0


def test_subgoals_split_the_work_and_a_goal_never_reached_is_its_own_stretch():
    recs = _board()
    recs["fail.json"] = _record("open the left door then the right door", None,
                                [(0, 5, "advancing", 0.5), (5, 12, "advancing", 0.7)], [(5, "left door open")],
                                outcome="failure")
    s = scores.score_board(recs)
    ok = s["p0.json"]["subgoals"]
    assert [(st["start_s"], st["end_s"], st["goal"]) for st in ok] == [(0, 5.0, False), (5.0, 10.0, True)]
    assert ok[0]["peers"] == 5 and ok[0]["execution"] == 10.0
    failed = s["fail.json"]
    assert failed["subgoals"][-1]["goal"] and not failed["subgoals"][-1]["reached"]
    assert failed["outcome"] == 4.0 and failed["efficiency"] is None   # failure is capped and has no time to goal


def test_a_paraphrase_joins_its_task_group_and_another_task_does_not():
    recs = _board()
    recs["para.json"] = _record("open the left door, then open the right door", 10.0,
                                [(0, 10, "advancing", 1.0)])
    recs["other.json"] = _record("stack the red cube on the blue cube", 10.0, [(0, 10, "advancing", 1.0)])
    g = scores.task_groups(recs)
    assert g["para.json"] == g["p0.json"] and g["other.json"] != g["p0.json"]


def test_failed_attempts_belong_to_the_subgoal_they_precede_and_cost_only_when_no_mistake_explains_them():
    recs = _board()
    rec = _record("open the left door then the right door", None,
                  [(0, 5, "advancing", 0.5), (5, 12, "wasteful", 0.5)], [(5, "left door open")], outcome="failure")
    rec["key_events"] += [{"t_s": 3.0, "kind": "grasp", "outcome": "failure", "label": "handle slips"},
                          {"t_s": 8.0, "kind": "grasp", "outcome": "failure", "label": "right handle missed"},
                          {"t_s": 10.0, "kind": "pickup", "outcome": "success", "label": "handle held"}]
    rec["operator_mistakes"] = [{"issue": "slipped grasp", "severity": "high", "recovered": True, "t_s": 3.0}]
    recs["tries.json"] = rec
    first, last = scores.score_board(recs)["tries.json"]["subgoals"]
    assert [k["label"] for k in first["failed_attempts"]] == ["handle slips"]
    assert first["cleanliness"] == round(10.0 - 3.0 * 0.5, 1)         # the mistake explains the failed grasp
    assert [k["label"] for k in last["failed_attempts"]] == ["right handle missed"]
    assert last["goal"] and not last["reached"]
    assert last["cleanliness"] == 0.0                                 # all wasteful, then one unexplained failure


def test_with_subtasks_each_subgoal_stretch_starts_in_its_own_subtask_and_the_gap_between_them_is_not_charged():
    rec = _record("wash the bowl, then peel the carrot", None,
                  [(0, 10, "advancing", 1.0), (10, 30, "idle", 0.0), (30, 40, "advancing", 0.5), (40, 50, "advancing", 1.0)],
                  [(10, "bowl rinsed"), (40, "carrot half peeled")])
    rec["completion"] = {}
    rec["tasks"] = [{"start_s": 0, "end_s": 10, "task": "wash the bowl", "outcome": "success", "completed_at_s": 10},
                    {"start_s": 30, "end_s": 50, "task": "peel the carrot", "outcome": "success", "completed_at_s": 50}]
    subs = scores.score_board({"ego.json": rec})["ego.json"]["subgoals"]
    assert [(st["start_s"], st["end_s"], st["label"], st["goal"]) for st in subs] == [
        (0, 10, "bowl rinsed", False), (30, 40, "carrot half peeled", False), (40, 50, "peel the carrot", True)]
    assert all(st["steady"] == 10.0 for st in subs)                   # the 20s between the two never counts


def test_with_subtasks_the_outcome_is_the_mean_of_the_settled_subtasks_and_an_unclear_one_counts_for_nothing():
    rec = _record("wash the bowl, peel the carrot, wipe the board", None,
                  [(0, 10, "advancing", 1.0), (12, 20, "advancing", 0.4), (22, 25, "advancing", 0.2)],
                  [(10, "bowl rinsed")])
    rec["completion"] = {}
    rec["tasks"] = [{"start_s": 0, "end_s": 10, "task": "wash the bowl", "outcome": "success", "completed_at_s": 10},
                    {"start_s": 12, "end_s": 20, "task": "peel the carrot", "outcome": "partial", "completed_at_s": None},
                    {"start_s": 22, "end_s": 25, "task": "wipe the board", "outcome": "unclear", "completed_at_s": None}]
    s = scores.score_board({"ego.json": rec})["ego.json"]
    assert s["outcome"] == round(10 * (1.0 + 0.4) / 2, 1)
    assert (s["inputs"]["tasks_total"], s["inputs"]["tasks_judged"], s["inputs"]["tasks_done"]) == (3, 2, 1)
    assert [(st["task"], st["goal"], st["reached"]) for st in s["subgoals"]] == [(1, False, True), (2, True, False),
                                                                                (3, True, False)]


def test_a_record_with_nothing_scoreable_or_malformed_steps_gets_empty_scores_never_an_error():
    s = scores.score_board({"empty.json": {}, "odd.json": {"event_labels": ["not a step", None], "key_events": [None],
                                                           "tasks": "none"}})
    assert s["empty.json"]["subgoals"] == [] and s["empty.json"]["outcome"] is None
    assert s["odd.json"]["subgoals"] == [] and s["odd.json"]["execution"] is None      # nothing measured, no score


def test_a_time_stored_as_text_or_not_finite_is_skipped_never_compared():
    rec = _record("open the left door then the right door", 10.0,
                  [(0, 5, "advancing", 0.5), (5, 10, "advancing", 1.0)], [(5, "left door open")])
    rec["key_events"] += [{"t_s": "7", "kind": "grasp", "outcome": "failure"},
                          {"t_s": float("nan"), "kind": "grasp", "outcome": "failure"}]
    subs = scores.score_board({"r.json": rec})["r.json"]["subgoals"]
    assert [st["failed_attempts"] for st in subs] == [[], []]


def test_a_partial_result_as_a_board_record_stores_it_is_scored_as_partial_not_failure():
    rec = _record("open the left door then the right door", None,
                  [(0, 5, "advancing", 0.5), (5, 9, "advancing", 0.85)], [(5, "left door open")], outcome="failure")
    rec["completion"]["failure_kind"] = "partial"
    s = scores.score_board({"p.json": rec})["p.json"]
    assert s["outcome"] == 8.5 and s["inputs"]["outcome"] == "partial"


def test_progress_no_step_carries_is_unknown_so_a_failure_has_no_outcome_score_never_zero():
    rec = _record("open the left door then the right door", None,
                  [(0, 5, "advancing", None), (5, 9, "advancing", None)], outcome="failure")
    s = scores.score_board({"parts.json": rec})["parts.json"]
    assert s["outcome"] is None and s["inputs"]["progress_reached"] is None


def test_a_subgoal_just_past_one_subtask_belongs_to_the_next_one_only():
    rec = _record("wash the bowl, then peel the carrot", None,
                  [(0, 10, "advancing", 1.0), (10, 20, "advancing", 1.0)], [(10.3, "carrot rinsed"), (20, "carrot peeled")])
    rec["completion"] = {}
    rec["tasks"] = [{"start_s": 0, "end_s": 10, "task": "wash the bowl", "outcome": "success", "completed_at_s": 10},
                    {"start_s": 10, "end_s": 20, "task": "peel the carrot", "outcome": "success", "completed_at_s": 20}]
    subs = scores.score_board({"ego.json": rec})["ego.json"]["subgoals"]
    assert [(st["task"], st["label"]) for st in subs] == [(1, "wash the bowl"), (2, "carrot rinsed"), (2, "carrot peeled")]


def test_the_peer_median_left_one_out_equals_the_median_of_the_other_demonstrations():
    import random
    import statistics
    rng = random.Random(7)
    recs = {}
    for i in range(40):
        goal = round(rng.uniform(5, 30), 1)
        recs[f"e{i}.json"] = _record("stack the cups", goal, [(0, goal / 2, "advancing", 0.5), (goal / 2, goal, "advancing", 1.0)])
    s = scores.score_board(recs)
    for f, d in recs.items():
        others = [r["completion"]["completed_at_s"] for g, r in recs.items() if g != f]
        assert s[f]["inputs"]["peer_median_time_to_goal_s"] == statistics.median(others)


def test_a_board_of_two_thousand_demonstrations_of_one_task_scores_in_seconds():
    import time
    recs = {f"e{i}.json": _record("stack the cups", 10.0 + i % 7, [(0, 5, "advancing", 0.5), (5, 10 + i % 7, "advancing", 1.0)],
                                  [(5, "first cup stacked")]) for i in range(2000)}
    t = time.time()
    scores.score_board(recs)
    assert time.time() - t < 5


def _given(rec, instruction, relation="aligned", rig="teleop_arms"):
    return {**rec, "_meta": {"given_prompt": instruction}, "goal_alignment": {"relation": relation}, "_rig": rig}


def test_the_task_is_the_given_instruction_so_one_recording_labelled_twice_is_one_task_whatever_the_summary():
    a = _given(_record("Arrange the blocks into a row", 10.0, [(0, 10, "advancing", 1.0)]), "spell CORL with the blocks")
    b = _given(_record("Line up four lettered cubes", 10.0, [(0, 10, "advancing", 1.0)]), "spell CORL with the blocks")
    g = scores.task_groups({"a.json": a, "b.json": b})
    assert g["a.json"] == g["b.json"]


def test_a_mislabelled_episode_is_grouped_by_what_it_shows_not_by_its_instruction():
    shoes = _given(_record("Pair the shoes on the rack", 10.0, [(0, 10, "advancing", 1.0)]), "organize scattered shoes")
    shirt = _given(_record("Fold the gray shirt", 10.0, [(0, 10, "advancing", 1.0)]), "organize scattered shoes",
                   relation="different")
    g = scores.task_groups({"shoes.json": shoes, "shirt.json": shirt})
    assert g["shoes.json"] != g["shirt.json"]


def test_without_peers_of_its_task_a_subgoal_is_compared_with_similar_subgoals_on_the_same_setup():
    recs = {}
    for i in range(6):
        recs[f"o{i}.json"] = _given(_record(f"task {i}", 10.0, [(0, 4, "advancing", 0.5), (4, 10, "advancing", 1.0)],
                                            [(4, "red block added to the row")]), f"build tower number {i}")
    recs["me.json"] = _given(_record("lone task", 20.0, [(0, 8, "advancing", 0.5), (8, 20, "advancing", 1.0)],
                                     [(8, "blue block added to the row")]), "a task nobody else did")
    st = scores.score_board(recs)["me.json"]["subgoals"][0]
    assert st["peer_basis"] == "similar" and st["peers"] == 6 and st["peer_median_s"] == 4.0
    assert st["efficiency"] == 5.0                                    # twice the median
    assert scores.score_board(recs)["me.json"]["peer_basis"] is None   # an episode's speed needs its own task


def test_an_episode_with_no_timeline_has_no_execution_score_never_a_perfect_one():
    rec = _record("open the door", 10.0, [(0, 10, "advancing", 1.0)])
    rec["event_labels"] = []
    s = scores.score_board({"empty.json": rec})["empty.json"]
    assert s["execution"] is None and s["cleanliness"] is None


def test_a_subgoal_inside_overlapping_subtasks_belongs_to_the_innermost_and_is_scored_once():
    rec = _record("tidy the bench", None, [(0, 33.5, "advancing", 1.0)], [(23.0, "screwdriver put away")])
    rec["completion"] = {}
    rec["tasks"] = [{"start_s": 0, "end_s": 33.5, "task": "tidy the bench", "outcome": "success", "completed_at_s": 33.5},
                    {"start_s": 14.5, "end_s": 23.0, "task": "put the screwdriver away", "outcome": "success",
                     "completed_at_s": 23.0}]
    subs = scores.score_board({"ego.json": rec})["ego.json"]["subgoals"]
    assert sum(st["end_s"] == 23.0 for st in subs) == 1 and [st["task"] for st in subs if st["end_s"] == 23.0] == [2]


def test_nothing_after_a_single_goal_is_scored_as_a_subgoal_of_it():
    rec = _record("set the table", 20.0, [(0, 20, "advancing", 1.0), (20, 30, "advancing", 1.0)],
                  [(10, "plate set"), (20, "table set"), (27, "plate returned to the counter")])
    rec["completion"].update({"task_completed": "success_then_undone", "completed_at_s": None, "goal_reached_at_s": 20.0})
    subs = scores.score_board({"undo.json": rec})["undo.json"]["subgoals"]
    assert all(st["end_s"] <= 20.0 for st in subs)


def test_fields_of_the_wrong_type_are_read_as_absent_never_a_crash():
    rec = _record("open the door", 10.0, [(0, 10, "advancing", 1.0)])
    rec.update({"tasks": 3, "completion": ["x"], "goal_alignment": ["y"], "_meta": "z"})
    assert scores.score_board({"odd.json": rec})["odd.json"]["version"] == scores.VERSION


def test_a_goal_reached_then_undone_reads_as_reached_in_its_goal_row():
    rec = _record("set the table", 20.0, [(0, 10, "advancing", 0.5), (10, 25, "advancing", 1.0)], [(10, "plate set")])
    rec["completion"].update({"task_completed": "success_then_undone", "completed_at_s": None, "goal_reached_at_s": 20.0})
    goal = scores.score_board({"undo.json": rec})["undo.json"]["subgoals"][-1]
    assert goal["goal"] and goal["reached"] and goal["end_s"] == 20.0


def test_instructions_that_recur_word_for_word_are_separate_tasks_even_from_one_template():
    recs = {}
    for i in range(3):
        recs[f"oven{i}.json"] = _given(_record("bake", 10.0, [(0, 10, "advancing", 1.0)]),
                                       "Open the electric oven, put the bread inside and close it")
        recs[f"micro{i}.json"] = _given(_record("heat", 10.0, [(0, 10, "advancing", 1.0)]),
                                        "Open the microwave oven, put the bread inside and close it")
    recs["para.json"] = _given(_record("bake", 10.0, [(0, 10, "advancing", 1.0)]),
                               "Open the electric oven, put bread inside, close it")
    g = scores.task_groups(recs)
    assert g["oven0.json"] == g["oven2.json"] != g["micro0.json"] == g["micro2.json"]
    assert g["para.json"] == g["oven0.json"]                          # a one-off paraphrase finds its task


def test_an_empty_task_key_never_joins_anything_and_other_alphabets_keep_their_words():
    recs = {name: _given(_record("", 10.0, [(0, 10, "advancing", 1.0)]), text) for name, text in
            (("cup.json", "把杯子放到盘子上"), ("micro.json", "打开微波炉"), ("blank.json", ""), ("stop.json", "the a an"))}
    for r in recs.values():
        r["episode_prompt"] = ""
    g = scores.task_groups(recs)
    assert len(set(g.values())) == 4


def test_similar_subgoals_are_compared_only_within_the_same_dataset():
    recs = {}
    for i in range(6):
        other = _given(_record(f"t{i}", 10.0, [(0, 4, "advancing", 0.5), (4, 10, "advancing", 1.0)],
                               [(4, "red block added to the row")]), f"build tower {i}")
        other["dataset"] = "another_robot"
        recs[f"o{i}.json"] = other
    recs["me.json"] = _given(_record("lone", 20.0, [(0, 8, "advancing", 0.5), (8, 20, "advancing", 1.0)],
                                     [(8, "blue block added to the row")]), "a task nobody else did")
    st = scores.score_board(recs)["me.json"]["subgoals"][0]
    assert st["peer_basis"] is None and st["efficiency"] is None


def test_unreached_subgoals_never_enter_a_similar_pool_and_a_task_basis_with_enough_peers_is_kept():
    recs = {}
    for i in range(6):
        recs[f"same{i}.json"] = _given(_record("t", 10.0, [(0, 5, "advancing", 0.5), (5, 10, "advancing", 1.0)],
                                               [(5, "cup placed")]), "place the cup then the plate")
    for i in range(6):
        recs[f"other{i}.json"] = _given(_record("t", 30.0, [(0, 25, "advancing", 0.5), (25, 30, "advancing", 1.0)],
                                                [(25, "cup placed")]), f"other task {i}")
    miss = _record("t", 30.0, [(0, 25, "advancing", 0.5), (25, 30, "advancing", 1.0)], [(25, "cup placed")])
    miss["key_events"][0]["outcome"] = "failure"                     # a "cup placed" that was never reached
    recs["miss.json"] = _given(miss, "a third task")
    st = scores.score_board(recs)["same0.json"]["subgoals"][0]
    assert st["peer_basis"] == "task" and st["peers"] == 5 and st["peer_median_s"] == 5.0
    pools = scores._similar_pools(recs, {f: scores.subgoals(d, scores.facts(d)) for f, d in recs.items()})
    assert all(len(p) == 11 for p in pools.values())                # 12 reached "cup placed", itself left out


def test_a_subtask_whose_steps_carry_no_progress_is_left_out_of_the_outcome_unless_it_succeeded():
    rec = _record("two jobs", None, [(0, 10, "advancing", None), (10, 20, "advancing", None)])
    rec["completion"] = {}
    rec["tasks"] = [{"start_s": 0, "end_s": 10, "task": "a", "outcome": "success", "completed_at_s": 10},
                    {"start_s": 10, "end_s": 20, "task": "b", "outcome": "failure", "completed_at_s": None}]
    s = scores.score_board({"e.json": rec})["e.json"]
    assert s["outcome"] == 10.0 and s["inputs"]["tasks_judged"] == 1


def test_a_subgoal_within_half_a_second_past_its_subtask_belongs_to_it_when_the_next_has_not_begun():
    rec = _record("two jobs", None, [(0, 10, "advancing", 1.0), (12, 20, "advancing", 1.0)], [(10.3, "bowl rinsed")])
    rec["completion"] = {}
    rec["tasks"] = [{"start_s": 0, "end_s": 10, "task": "wash", "outcome": "success", "completed_at_s": 10.3},
                    {"start_s": 12, "end_s": 20, "task": "peel", "outcome": "success", "completed_at_s": 20}]
    subs = scores.score_board({"e.json": rec})["e.json"]["subgoals"]
    assert [(st["task"], st["label"]) for st in subs if not st["goal"]] == [(1, "bowl rinsed")]


def test_similar_pools_stay_fast_when_every_label_shares_one_common_word():
    import time
    recs = {}
    for i in range(1500):
        recs[f"e{i}.json"] = _given(_record(f"t{i}", 10.0, [(0, 4, "advancing", 0.5), (4, 10, "advancing", 1.0)],
                                            [(4, f"left gripper placed part{i % 400} variant{i % 37}")]), f"task {i}")
    t = time.time()
    scores.score_board(recs)
    assert time.time() - t < 10



def test_only_mistakes_the_board_counts_cost_points_a_recovery_halves_one_and_a_duplicate_costs_once():
    rec = _record("open the door", 10.0, [(0, 10, "advancing", 1.0)])
    rec["operator_mistakes"] = [
        {"issue": "fumbled the latch", "severity": "low", "t_s": 2.0},                       # minor: no points
        {"issue": "paused", "t_s": 3.0},                                                     # unrated: no points
        {"issue": "dropped the handle", "severity": "high", "category": "dropped_object", "t_s": 5.0},
        {"issue": "dropped the handle", "severity": "high", "category": "dropped_object", "t_s": 5.0}]  # listed twice
    rec["recovery"] = [{"failure_t_s": 5.5, "failure": "handle dropped", "recovered": True, "recovered_at_s": 7.0}]
    s = scores.score_board({"m.json": rec})["m.json"]
    assert [(p["severity"], p["recovered"], p["points"]) for p in s["inputs"]["penalties"]] == [("high", True, 1.5)]
    assert s["cleanliness"] == 8.5


def test_two_different_mistakes_at_one_time_or_without_a_time_are_each_charged():
    for t in (5.0, None):
        rec = _record("set the table", 10.0, [(0, 10, "advancing", 1.0)])
        rec["operator_mistakes"] = [
            {"issue": "dropped the cup", "severity": "high", "category": "dropped_object", "t_s": t},
            {"issue": "dropped the plate", "severity": "high", "category": "dropped_object", "t_s": t}]
        assert scores.score_board({"m.json": rec})["m.json"]["cleanliness"] == 4.0


def test_a_mistake_listed_twice_explains_one_failed_attempt_not_two():
    rec = _record("open the door", 10.0, [(0, 10, "advancing", 1.0)])
    rec["operator_mistakes"] = [{"issue": "missed the handle", "severity": "high", "category": "missed_grasp",
                                 "t_s": 5.0}] * 2
    rec["key_events"] = [{"t_s": 4.0, "kind": "grasp", "outcome": "failure", "label": "grasp handle"},
                         {"t_s": 6.0, "kind": "grasp", "outcome": "success", "label": "handle held"},
                         {"t_s": 8.0, "kind": "grasp", "outcome": "failure", "label": "grasp handle"}]
    s = scores.score_board({"m.json": rec})["m.json"]
    assert s["inputs"]["unexplained_failed_attempts"] == scores.FAILED_ATTEMPT_POINTS and s["cleanliness"] == 6.0


def test_a_recording_with_a_part_that_has_no_labels_gets_no_whole_recording_score():
    # 185622: part 1 of 2 got no reply, and the labelled half scored 10 for the whole recording
    rec = _record("sort the blocks", None, [(372, 680, "idle", None)])
    rec["_stitched"] = {"parts": 2, "cuts_s": [371.7], "missing": [{"part": 1, "t0_s": 0.0, "t1_s": 371.7,
                                                                     "why": "the model gave no reply"}]}
    s = scores.score_board({"m.json": rec})["m.json"]
    assert s["execution"] is None and s["outcome"] is None and s["cleanliness"] is None
    assert s["inputs"]["unlabelled_parts"] == [{"part": 1, "t0_s": 0.0, "t1_s": 371.7}]


def test_failed_tries_inside_one_mistakes_span_are_one_incident_and_any_other_failure_counts_alone():
    # a sticker peel retried ten times inside the one mistake the labels describe is one incident, which that mistake
    # explains; the failures outside every mistake span, and one with no time, count one each, as on main
    rec = _record("peel the stickers", 30.0, [(0, 30, "advancing", 1.0)])
    rec["operator_mistakes"] = [{"issue": "struggles to peel the first sticker", "severity": "low",
                                 "category": "slow_progress", "t_s": 0.5, "end_s": 11.0}]
    rec["key_events"] = ([{"t_s": 1.0 + i, "kind": "peel", "outcome": "failure", "label": "sticker stays"}
                          for i in range(10)]
                         + [{"t_s": 14.0, "kind": "peel", "outcome": "failure", "label": "next sticker stays"},
                            {"t_s": 15.0, "kind": "peel", "outcome": "failure", "label": "next sticker stays"},
                            {"kind": "peel", "outcome": "failure", "label": "a try with no time"}])
    s = scores.score_board({"p.json": rec})["p.json"]
    assert s["inputs"]["unexplained_failed_attempts"] == 3 * scores.FAILED_ATTEMPT_POINTS
    assert s["cleanliness"] == 7.0
    # two unrelated failures with no mistake around them, a grasp at 1s and a collision at 500s, are two incidents
    rec = _record("tidy the kitchen", 600.0, [(0, 600, "advancing", 1.0)])
    rec["key_events"] = [{"t_s": 1.0, "kind": "grasp", "outcome": "failure", "label": "left hand misses the cup"},
                         {"t_s": 500.0, "kind": "contact", "outcome": "failure", "label": "right hand hits the drawer"}]
    assert scores.score_board({"k.json": rec})["k.json"]["inputs"]["unexplained_failed_attempts"] == 2.0


def test_a_staged_mistake_is_charged_like_the_board_counts_it_and_one_without_text_is_not_read():
    rec = _record("open the door", 10.0, [(0, 10, "advancing", 1.0)])
    rec["operator_mistakes"] = [
        {"issue": "dropped the handle", "severity": "high", "category": "dropped_object", "t_s": 5.0,
         "deliberate": "yes"},
        {"issue": "", "severity": "high", "category": "dropped_object", "t_s": 7.0},
        {"severity": "high", "category": "dropped_object", "t_s": 8.0}]
    s = scores.score_board({"m.json": rec})["m.json"]
    assert [p["mistake"] for p in s["inputs"]["penalties"]] == ["dropped the handle"] and s["cleanliness"] == 7.0


def test_a_goal_stretch_that_was_not_reached_has_no_speed_even_with_peers_of_its_task():
    recs = _board()
    part = _record("open the left door then the right door", None,
                   [(0, 5, "advancing", 0.5), (5, 8, "advancing", 0.7)], [(5, "left door open")], outcome="partial")
    recs["part.json"] = part
    last = scores.score_board(recs)["part.json"]["subgoals"][-1]
    assert last["goal"] and not last["reached"]
    assert last["efficiency"] is None and last["peer_basis"] is None and last["peer_median_s"] is None


def test_values_of_the_wrong_type_inside_fields_never_stop_the_board_and_scores_stay_between_0_and_10():
    bad = _record("open the door", 10.0, [(0, 10, "advancing", -3.0)], outcome="failure")
    bad["completion"]["task_completed"] = ["failure"]
    bad["operator_mistakes"] = [{"issue": "x", "severity": {"level": "high"}, "t_s": 10 ** 400}]
    bad["dataset"], bad["_rig"] = ["ds"], {"rig": 1}
    neg = _record("open the door", 10.0, [(0, 10, "advancing", -2.0)], outcome="failure")
    s = scores.score_board({"bad.json": bad, "neg.json": neg, "junk.json": "not a record"})
    for x in s.values():
        for k in ("execution", "outcome", "cleanliness", "efficiency", "fluency"):
            assert x[k] is None or 0 <= x[k] <= 10
    assert s["neg.json"]["outcome"] == 0.0


def test_a_peer_whose_goal_comes_before_its_first_step_is_left_out_never_a_negative_median():
    recs = _board()
    for i in range(6):
        r = _record("open the left door then the right door", 10.0, [(12, 14, "advancing", 0.5), (14, 16, "advancing", 1.0)])
        recs[f"backwards{i}.json"] = r
    s = scores.score_board(recs)
    assert all(x["efficiency"] is None or 0 <= x["efficiency"] <= 10 for x in s.values())
    assert s["p0.json"]["inputs"]["peer_median_time_to_goal_s"] == 10.0


def test_similar_subgoals_never_include_the_episodes_own_other_subgoals():
    recs = {}
    units = [(i * 5.0 + 5, "unit fitted and set aside") for i in range(10)]
    recs["me.json"] = _given(_record("factory", 50.0, [(0, 50, "advancing", 1.0)], units), "fit the units")
    st = scores.score_board(recs)["me.json"]["subgoals"][0]
    assert st["peer_basis"] is None and st["peers"] == 0


def test_labels_that_all_share_their_common_words_stay_fast_to_compare():
    import time
    recs = {}
    for i in range(3000):
        recs[f"e{i}.json"] = _given(_record(f"t{i}", 10.0, [(0, 4, "advancing", 0.5), (4, 10, "advancing", 1.0)],
                                            [(4, f"grasp widget alpha{i} beta{i * 7919}")]), f"task {i}")
    t = time.time()
    scores.score_board(recs)
    assert time.time() - t < 10


def test_a_stitched_recordings_failed_task_is_judged_from_its_steps_progress_in_their_own_part():
    # the UMI 900h stitched takes all scored 10: the stitch keeps no recording-wide progress, so every failed or partial
    # task went unjudged and only the successes counted; a step's progress toward its own part's sub-task is the
    # progress of the task it belongs to
    rec = _record("", None, [(0, 10, "advancing", None), (12, 20, "advancing", None), (22, 25, "advancing", None)])
    for step, p in zip(rec["event_labels"], (1.0, 0.4, 0.2)):
        step["progress_in_part"] = p
    rec["completion"] = {}
    rec["_stitched"] = {"parts": 2, "cuts_s": [11.0]}
    rec["tasks"] = [{"start_s": 0, "end_s": 10, "task": "wash the bowl", "outcome": "success", "completed_at_s": 10},
                    {"start_s": 12, "end_s": 20, "task": "peel the carrot", "outcome": "failure", "completed_at_s": None},
                    {"start_s": 22, "end_s": 25, "task": "wipe the board", "outcome": "partial", "completed_at_s": None}]
    s = scores.score_board({"umi.json": rec})["umi.json"]
    assert (s["inputs"]["tasks_total"], s["inputs"]["tasks_judged"], s["inputs"]["tasks_done"]) == (3, 3, 1)
    assert s["outcome"] == round(10 * (1.0 + 0.4 + 0.2) / 3, 1)
