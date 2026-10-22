"""Episode and subgoal quality scores, computed from the labels a board already holds, never asked of the model.

A model reads one episode at a time, so a number it writes for one episode is not comparable with the number it writes
for another. These scores are computed instead: each one is built from fields the annotation already records (the
timeline's advancing, wasteful and idle steps, the key events, the operator mistakes and recoveries, the completion and
the goal alignment), and the ones that only mean something next to other demonstrations of the same task (how long the
goal took, how busy the work was) are compared within the episode's task group on the board.

Four dimensions, each 0 to 10, each with the numbers it came from (inputs), so the board can say why:

| Dimension | Question | Built from | Compared |
|---|---|---|---|
| outcome | How much of the labelled goal was achieved | completion, the progress reached | absolute |
| cleanliness | How free the work is of mistakes and wasted motion | wasteful time, the operator mistakes the board counts (by severity, half when recovered), failed attempts no mistake explains (tries in a row with no success between them count once) | absolute |
| efficiency | How quickly the goal was reached, for this task | time from the first action to the goal | within the task group |
| fluency | How steadily the work went | hesitation inside the work (absolute), and actions per unit of progress | hesitation absolute, actions within the task group |

execution, the quality of what the operator actually did, is cleanliness, efficiency and fluency weighted 0.75, 0.10 and
0.15 (EXECUTION_WEIGHTS), over the ones that exist, for an episode and for each subgoal alike; outcome is kept apart, so
a clean demonstration of a mislabelled task reads as a good execution with relation "different", never as a bad one.
Speed is 10 at or below the median of its peers, which means typical, not fast. A dimension with too few peers is None,
never a guess, and every score stays between 0 and 10 whatever the labels hold. One malformed record is scored as an
empty one; it never stops the board.
"""
from __future__ import annotations

import bisect
import collections
import math
import re
import statistics

from board.families import Families, union_seconds as _union
from board.label_projection import (progress_toward_verdict, severity as _severity,
                                    verdict_or_tasks as _verdict_or_tasks)
from checks.results import stored_number


def _num(x):
    """A stored number the scores can compare as it is: a finite int or float (stored_number), never text, true or
    false, or an integer too large for a float."""
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return None
    try:
        return stored_number(x)
    except OverflowError:
        return None


VERSION = 2
MIN_PEERS = 5                     # a relative dimension needs this many other demonstrations of the same task
SEVERITY_POINTS = {"low": 0.5, "medium": 1.5, "high": 3.0}
FAILED_ATTEMPT_POINTS = 1.0       # a failed incident (_incidents), beyond the mistake that may describe it
RECOVERY_WINDOW_S = 1.5           # a recovery belongs to a mistake when its failure is this close to the mistake's time
OUTCOME_CAPS = {"success": 1.0, "success_then_undone": 0.7, "partial": 0.9, "failure": 0.4}
EXECUTION_WEIGHTS = {"cleanliness": 0.75, "efficiency": 0.10, "fluency": 0.15}
# the record fields the scores read, so a board build can hold every episode at once without its whole record
FIELDS = ("dataset", "episode_prompt", "task_label", "event_labels", "key_events", "operator_mistakes", "recovery",
          "completion", "goal_alignment", "tasks", "_meta", "_rig", "_stitched", "progress_scope")
SIMILAR = 0.5                     # two subgoal labels are the same kind of subgoal when they share this share of words
SAME_TASK = 0.6                   # a one-off wording joins a task whose instruction shares this share of its words
STOP = {"the", "a", "an", "and", "then", "with", "its", "it", "of", "on", "to", "into", "in", "from", "onto", "by", "at",
        "for", "both", "each", "one", "that", "this", "is", "are", "be", "so", "as", "up", "down", "out", "back", "over"}
_FAMILIES = None


def _families() -> Families:
    """The board's own issue rules (board/families.py), read once: which operator mistakes count."""
    global _FAMILIES
    if _FAMILIES is None:
        _FAMILIES = Families()
    return _FAMILIES


def _clamp(v):
    return None if v is None else round(min(10.0, max(0.0, v)), 1)


def _subtract(spans, cut):
    """The parts of spans not covered by any span in cut."""
    out = []
    for a, b in spans:
        pieces = [(a, b)]
        for c, d in cut:
            pieces = [p for x, y in pieces for p in ((x, min(y, c)), (max(x, d), y)) if p[1] > p[0]]
        out += pieces
    return out


def _verdict(x: dict) -> str | None:
    """A completion's or a task's outcome word, with a partial result read back from the record's failure_kind (a
    board record stores partial as failure with failure_kind "partial", board/to_board.py convert). A value that is not
    a word is no verdict."""
    word = x.get("task_completed", x.get("outcome"))
    if not isinstance(word, str):
        return None
    return "partial" if word == "failure" and x.get("failure_kind") == "partial" else word


def _dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _list(x) -> list:
    return x if isinstance(x, list) else []


def _str(x) -> str | None:
    return x if isinstance(x, str) else None


def _subtasks(d: dict) -> list:
    """The tasks scored one by one, by the rule every outcome reader shares (board/label_projection.py
    verdict_or_tasks): none for a recording whose completion, derived from its one task, gives a verdict."""
    return _verdict_or_tasks(d)[1]


def _steps(d: dict) -> list:
    """The timeline steps with usable times, in time order."""
    return sorted((e for e in _list(d.get("event_labels")) if isinstance(e, dict) and _num(e.get("t_s")) is not None
                   and _num(e.get("end_s")) is not None), key=lambda e: e["t_s"])


def _mistakes(d: dict) -> tuple[list, list]:
    """(the operator mistakes the board counts, each with recovered read from the mistake or from the recovery list,
    the times of the mistakes of any rating, which explain failed attempts). Like the board, only a mistake with issue
    text is read, and a staged one (deliberate) is charged like any other. A mistake the board shows as minor or unrated
    costs nothing; its wasted motion is already in the wasteful share. The same mistake listed twice (same text,
    category, rating and time) is read once, both to charge it and to explain a failed attempt; mistakes without a time
    are never taken for one another."""
    rows, seen = [], set()
    for m in _list(d.get("operator_mistakes")):
        if not isinstance(m, dict) or not m.get("issue"):
            continue
        sev, t = _severity(m), _num(m.get("t_s"))
        key = (str(m.get("issue")), _str(m.get("category")), sev, t)
        if t is not None and key in seen:
            continue
        seen.add(key)
        rows.append((m, sev, t))
    recovered_at = [_num(r.get("failure_t_s")) for r in _list(d.get("recovery")) if isinstance(r, dict)
                    and (r.get("recovered") is True or _num(r.get("recovered_at_s")) is not None)]
    recovered_at = [t for t in recovered_at if t is not None]
    counted = []
    for m, sev, t in rows:
        try:
            counts = sev is not None and _families().counts("operator_mistakes", {**m, "severity": sev})
        except (TypeError, ValueError, AttributeError):
            counts = False
        if not counts:
            continue
        recovered = m.get("recovered") is True or (t is not None and any(abs(r - t) <= RECOVERY_WINDOW_S
                                                                        for r in recovered_at))
        counted.append({"issue": _str(m.get("issue")), "severity": sev, "t_s": t, "recovered": recovered})
    return counted, [t for _, _, t in rows]


def _points(m: dict) -> float:
    return SEVERITY_POINTS[m["severity"]] * (0.5 if m["recovered"] else 1.0)


def facts(d: dict) -> dict:
    """What one episode's labels measure, before any comparison: its work window, its time split, its attempts."""
    d = d if isinstance(d, dict) else {}
    steps = _steps(d)
    spans = {c: [(e["t_s"], e["end_s"]) for e in steps if e.get("contribution") == c]
             for c in ("advancing", "wasteful", "idle")}
    work = spans["advancing"] + spans["wasteful"]
    start = min((a for a, _ in work), default=None)
    comp = _dict(d.get("completion"))
    goal = _num(comp.get("completed_at_s"))
    if goal is None:
        goal = _num(comp.get("goal_reached_at_s"))
    end = goal if goal is not None else max((b for _, b in work), default=None)
    active = _union(work)
    wasteful = _union(spans["wasteful"])
    # hesitation: stretches inside the work window where no arm advances or wastes, ie every arm waits
    window = [(start, end)] if start is not None and end is not None and end > start else []
    hesitation = _union(_subtract(window, work)) if window else 0.0
    # progress toward this goal, from the steps that carry it, read within 0 to 1; a recording labelled in parts keeps
    # only each part's local progress (label/pieces.py), which says nothing about the whole goal, so there it is
    # unknown, never 0
    progress = [min(1.0, max(0.0, p)) for p in (_num(e.get("progress")) for e in steps) if p is not None]
    reached = max(progress) if progress else None
    # a head-camera clip judged on its annotated goal has progress per sub-task (progress_scope), none toward the goal
    # its verdict is about, so its outcome is never scaled by it
    goal_progress = reached if progress_toward_verdict(d) else None
    mistakes, explaining = _mistakes(d)
    any_mistakes = len(explaining)
    incident = _incidents(d)
    # a recording with its own verdict is one goal, whatever tasks it lists (a recording of one task takes that task's
    # verdict, board/label_projection.py recording_completion); only without one is each subtask judged on its own,
    # and progress restarts at each subtask, so a subtask's progress is the furthest its own steps got
    tasks = []
    for t in _subtasks(d):
        if not isinstance(t, dict) or _num(t.get("start_s")) is None or _num(t.get("end_s")) is None:
            continue
        # in a recording labelled in parts a step keeps its progress toward its own part's sub-task, which is this
        # task's progress; without it every failed or partial task went unjudged and every stitched score was 10
        own = [min(1.0, max(0.0, p)) for e in steps if t["start_s"] <= e["t_s"] <= t["end_s"]
               for p in [_num(e.get("progress")) if _num(e.get("progress")) is not None
                         else _num(e.get("progress_in_part"))] if p is not None]
        tasks.append({"outcome": _verdict(t), "progress": max(own) if own else None})
    return {"tasks": tasks, "measured": bool(steps), "start_s": start, "goal_s": goal, "end_s": end, "active_s": active,
            "wasteful_s": wasteful, "hesitation_s": hesitation, "window_s": (end - start) if window else None,
            "actions": sum(1 for e in steps if e.get("contribution") == "advancing"), "progress_reached": reached,
            "goal_progress": goal_progress,
            "outcome": _verdict(comp) if comp else None, "mistakes": mistakes, "any_mistakes": any_mistakes,
            "failed_attempts": len(set(incident.values())),
            "relation": _str(_dict(d.get("goal_alignment")).get("relation"))}


def _incidents(d: dict) -> dict:
    """{id(failed key event): its incident}. The labels already say which failed tries share an aim: the prompt makes
    one operator mistake "the failed tries at one aim and the motion between them", with t_s and end_s spanning the
    incident. So the failed tries inside one mistake's span are one incident (a peel retried ten times inside one
    mistake costs one failed attempt, not ten), and a failed try outside every mistake span, or with no time, is an
    incident of its own, never grouped by a guess about its aim from its kind or timing."""
    spans = sorted((t, e) for m in _list(d.get("operator_mistakes")) if isinstance(m, dict) and m.get("issue")
                   for t, e in [(_num(m.get("t_s")), _num(m.get("end_s")))] if t is not None and e is not None and e >= t)
    out, count = {}, len(spans)
    for k in _list(d.get("key_events")):
        if not isinstance(k, dict) or k.get("outcome") != "failure":
            continue
        t = _num(k.get("t_s"))
        within = next((i for i, (a, b) in enumerate(spans) if t is not None and a <= t <= b), None)
        if within is None:
            within, count = count, count + 1
        out[id(k)] = within
    return out


def _content_words(text) -> frozenset:
    """The content words of a text, in any alphabet (a Chinese instruction keeps its words)."""
    return frozenset(w for w in re.findall(r"\w+", text.lower()) if w not in STOP) if isinstance(text, str) else frozenset()


def task_source(d: dict) -> tuple[frozenset, bool]:
    """(content words of the task an episode demonstrates, whether they are the given instruction). The task is the
    instruction the episode was given, so the same recording labelled twice is always the same task, and the model's
    own summary when there is no usable instruction or the work done is a different task from it (a mislabelled
    episode is compared with what it actually shows)."""
    d = d if isinstance(d, dict) else {}
    given = _dict(d.get("_meta")).get("given_prompt")
    mislabelled = _dict(d.get("goal_alignment")).get("relation") in ("different", "unrelated")
    words = frozenset() if mislabelled else _content_words(given)
    return (words, True) if words else (_content_words(d.get("episode_prompt")), False)


def task_key(d: dict) -> frozenset:
    return task_source(d)[0]


class _Near:
    """Word sets of one scope, found again by any later set sharing at least `share` of its words (Jaccard). A set is
    indexed only under its rarest words (two sets that share that much must share one of each other's rarest few), and
    each entry keeps the set's size and the word's place in it, so a whole run of entries that cannot reach `share`
    (too long or short, or the shared word too late in both) is skipped without a look. The answer is exact."""

    def __init__(self, share: float, all_sets):
        self.share, self.freq, self.index = share, collections.Counter(), {}
        for words in all_sets:
            self.freq.update(words)

    def _ordered(self, words):
        return sorted(words, key=lambda w: (self.freq[w], w))

    def _prefix_len(self, n):
        return n - math.ceil(self.share * n) + 1

    def add(self, words, value):
        ordered = self._ordered(words)
        for pos, w in enumerate(ordered[:self._prefix_len(len(ordered))]):
            self.index.setdefault(w, {}).setdefault((len(ordered), pos), []).append((words, value))

    def near(self, words):
        """[(Jaccard, other words, value)] for every added set within share of words."""
        t, out, seen = self.share, [], set()
        ordered = self._ordered(words)
        n = len(ordered)
        for pa, w in enumerate(ordered[:self._prefix_len(n)]):
            for (size, pb), entries in self.index.get(w, {}).items():
                if size < t * n - 1e-9 or size > n / t + 1e-9:
                    continue
                need = math.ceil(t / (1 + t) * (n + size) - 1e-9)
                if 1 + min(n - pa - 1, size - pb - 1) < need:
                    continue
                for other, value in entries:
                    if (other, value) in seen:
                        continue
                    seen.add((other, value))
                    j = len(words & other) / len(words | other)
                    if j >= t:
                        out.append((j, other, value))
        return out


def task_groups(records: dict) -> dict:
    """{file: group id}: demonstrations of one task, within one dataset. An instruction given word for word to two or
    more episodes is a task of its own, so a dataset's templated instructions ("put the bread in the toaster oven",
    "... in the microwave") never join each other. A wording seen once, or a model summary, joins the group whose words
    it mostly shares (Jaccard of content words at least SAME_TASK), so a paraphrase finds its task. An episode with no
    words at all is a group of its own."""
    keyed = {}
    for f, d in sorted(records.items()):
        d = d if isinstance(d, dict) else {}
        keyed[f] = (_str(d.get("dataset")), *task_source(d))
    count = collections.Counter((ds, key) for ds, key, given in keyed.values() if key and given)
    near = {}
    for ds, key, _ in keyed.values():
        near.setdefault(ds, []).append(key)
    near = {ds: _Near(SAME_TASK, [k for k in keys if k]) for ds, keys in near.items()}
    groups, exact, next_id = {}, {}, 0
    # the recurring instructions first, each a group, so nothing later can merge two of them
    for f, (ds, key, given) in keyed.items():
        if key and given and count[(ds, key)] >= 2:
            if (ds, key) not in exact:
                exact[(ds, key)] = next_id
                near[ds].add(key, next_id)
                next_id += 1
            groups[f] = exact[(ds, key)]
    for f, (ds, key, given) in keyed.items():
        if f in groups:
            continue
        if not key:
            groups[f], next_id = next_id, next_id + 1
            continue
        if (ds, key) in exact:
            groups[f] = exact[(ds, key)]
            continue
        best = max(near[ds].near(key), key=lambda m: (m[0], -m[2]), default=None)
        if best is not None:
            groups[f] = best[2]
        else:
            groups[f] = next_id
            near[ds].add(key, next_id)
            next_id += 1
        exact[(ds, key)] = groups[f]
    return groups


class _Peers:
    """The sorted values of a group's demonstrations with some values (the episode's own) left out, so the median of
    its peers costs a few binary searches rather than a pass over the group."""

    def __init__(self, ordered, without=None):
        self.ordered = ordered
        wanted = collections.Counter([] if without is None else without if isinstance(without, (list, tuple))
                                     else [without])
        # only values the group holds can be left out, each at most as often as it occurs
        self.without = sorted(v for v, n in wanted.items()
                              for _ in range(min(n, bisect.bisect_right(ordered, v) - bisect.bisect_left(ordered, v))))

    def __len__(self):
        return len(self.ordered) - len(self.without)

    def _at(self, k):
        """The k-th smallest value (from 0) of the group with the left-out values removed."""
        lo, hi = 0, len(self.ordered) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            v = self.ordered[mid]
            kept = bisect.bisect_right(self.ordered, v) - bisect.bisect_right(self.without, v)
            if kept >= k + 1:
                hi = mid
            else:
                lo = mid + 1
        return self.ordered[lo]

    def median(self):
        n = len(self)
        return self._at(n // 2) if n % 2 else (self._at(n // 2 - 1) + self._at(n // 2)) / 2


def _ratio_score(value, peers):
    """0 to 10 for a lower-is-better measure against its peers' median: the median or better is 10, twice the median
    is 5, and so on (10 x median / value). A rank would put half of any group below 5 even when every demonstration
    is fine; against the median, typical work scores full and only work slower or busier than typical loses points.
    None with too few peers or no positive value to compare. peers is a _Peers or a plain list."""
    peers = peers if isinstance(peers, _Peers) else _Peers(sorted(peers))
    if value is None or value <= 0 or len(peers) < MIN_PEERS:
        return None
    median = peers.median()
    if median <= 0:
        return None
    return 10.0 if value <= median else _clamp(10 * median / value)


def _outcome(f: dict):
    if f["tasks"]:
        # several subtasks: the mean of the subtasks the footage settles; an unclear one (eg cut off by the end of the
        # clip) counts for nothing either way
        judged = [t for t in f["tasks"] if t["outcome"] in OUTCOME_CAPS
                  and (t["outcome"] == "success" or t["progress"] is not None)]
        values = [1.0 if t["outcome"] == "success" else min(OUTCOME_CAPS[t["outcome"]], t["progress"]) for t in judged]
        return (_clamp(10 * statistics.mean(values)) if values else None,
                {"outcome": None, "tasks_total": len(f["tasks"]), "tasks_judged": len(judged),
                 "tasks_done": sum(1 for t in judged if t["outcome"] == "success")})
    # the progress toward the verdict's goal, never a sub-task's (facts, goal_progress)
    cap, reached = OUTCOME_CAPS.get(f["outcome"]), f["goal_progress"]
    if cap is None or (f["outcome"] != "success" and reached is None):
        return None, {"outcome": f["outcome"], "progress_reached": reached}
    value = 1.0 if f["outcome"] == "success" else min(cap, reached)
    return _clamp(10 * value), {"outcome": f["outcome"], "progress_reached": reached}


def _cleanliness(f: dict):
    if not f["measured"]:
        return None, {"wasteful_share": None, "penalties": [], "unexplained_failed_attempts": 0}
    penalties = [{"mistake": m["issue"], "severity": m["severity"], "recovered": m["recovered"], "points": _points(m)}
                 for m in f["mistakes"]]
    attempts = FAILED_ATTEMPT_POINTS * max(0, f["failed_attempts"] - f["any_mistakes"])
    share = f["wasteful_s"] / f["active_s"] if f["active_s"] else 0.0
    score = 10 * (1 - share) - sum(p["points"] for p in penalties) - attempts
    return _clamp(score), {"wasteful_share": round(share, 3), "penalties": penalties,
                           "unexplained_failed_attempts": attempts}


def _weighted(parts: dict):
    """execution: the EXECUTION_WEIGHTS mean of the dimensions that exist, None when none does."""
    have = {k: v for k, v in parts.items() if v is not None}
    weight = sum(EXECUTION_WEIGHTS[k] for k in have)
    return _clamp(sum(EXECUTION_WEIGHTS[k] * v for k, v in have.items()) / weight) if weight else None


def subgoals(d: dict, f: dict) -> list:
    """The stretches of work that end in the episode's subgoals: from the start of the work (or the previous subgoal)
    to each key event tagged subgoal_complete, and a last stretch to the goal, or to the end of the work when the goal
    is never reached ("reached": False). The model records a subgoal only once it is achieved, so a subgoal's own
    outcome is always success and is not scored; how cleanly, steadily and quickly its stretch went is. A subgoal is a
    lasting change the goal depends on, so the failed attempts between two subgoals are attempts at the later one. Each
    carries its duration, wasteful share, hesitation, and the mistakes and failed attempts inside it."""
    d = d if isinstance(d, dict) else {}
    events = sorted((k for k in _list(d.get("key_events")) if isinstance(k, dict) and k.get("kind") == "subgoal_complete"
                     and _num(k.get("t_s")) is not None), key=lambda k: k["t_s"])
    if f["start_s"] is None or not events:
        return []
    steps = _steps(d)
    starts = [e["t_s"] for e in steps]
    longest = max((e["end_s"] - e["t_s"] for e in steps), default=0.0)
    failed_all = sorted((k for k in _list(d.get("key_events")) if isinstance(k, dict) and k.get("outcome") == "failure"
                         and _num(k.get("t_s")) is not None), key=lambda k: k["t_s"])
    failed_t = [k["t_s"] for k in failed_all]
    incident = _incidents(d)
    mistakes = sorted((m for m in f["mistakes"] if m["t_s"] is not None), key=lambda m: m["t_s"])
    mistake_t = [m["t_s"] for m in mistakes]
    any_mistake_t = sorted(t for t in _mistakes(d)[1] if t is not None)
    # a recording with no verdict of its own (several tasks, human ego, sessions) has no single goal: each subtask is
    # scored on its own, from its start through its subgoals to its own end-state, and the time between subtasks
    # belongs to none of them
    tasks = sorted((t for t in _subtasks(d) if isinstance(t, dict) and _num(t.get("start_s")) is not None
                    and _num(t.get("end_s")) is not None), key=lambda t: t["start_s"])
    spans = ([(t["start_s"], _num(t.get("completed_at_s")) if _num(t.get("completed_at_s")) is not None else t["end_s"],
               _str(t.get("task")), _verdict(t), t["end_s"], i + 1) for i, t in enumerate(tasks)] if tasks
             else [(f["start_s"], f["end_s"], _str(_dict(d.get("completion")).get("success_predicate")), f["outcome"],
                    f["goal_s"], None)])

    def owner(t):
        holding = [(task["end_s"] - task["start_s"], i) for i, task in enumerate(tasks) if task["start_s"] < t <= task["end_s"]]
        if holding:
            return min(holding)[1] + 1
        for i, task in enumerate(tasks):
            nxt = tasks[i + 1]["start_s"] if i + 1 < len(tasks) else None
            if task["end_s"] < t <= task["end_s"] + 0.5 and (nxt is None or t < nxt):
                return i + 1
        return None
    owned = {id(k): owner(k["t_s"]) for k in events} if tasks else {}
    bounds = []
    for a, end, goal_label, goal_outcome, last, task in spans:
        for k in events:
            if k["t_s"] > a and ((last is None or k["t_s"] <= last + 0.5) if not tasks else owned[id(k)] == task):
                bounds.append((a, k["t_s"], _str(k.get("label")), _str(k.get("outcome")), False, task))
                a = k["t_s"]
        if end is not None and end > a + 0.5:
            bounds.append((a, end, goal_label, goal_outcome, True, task))
    out = []
    for a, b, label, outcome, goal, task in bounds:
        near = steps[bisect.bisect_left(starts, a - longest):bisect.bisect_left(starts, b)]
        clip = lambda spans: [(max(a, x), min(b, y)) for x, y in spans if min(b, y) > max(a, x)]
        work = clip([(e["t_s"], e["end_s"]) for e in near if e.get("contribution") in ("advancing", "wasteful")])
        waste = clip([(e["t_s"], e["end_s"]) for e in near if e.get("contribution") == "wasteful"])
        active = _union(work)
        share = _union(waste) / active if active else 0.0
        hes = _union(_subtract([(a, b)], work)) / (b - a)
        ms = mistakes[bisect.bisect_left(mistake_t, a):bisect.bisect_left(mistake_t, b)]
        failed = failed_all[bisect.bisect_left(failed_t, a):bisect.bisect_left(failed_t, b)]
        explained = bisect.bisect_left(any_mistake_t, b) - bisect.bisect_left(any_mistake_t, a)
        tried = len({incident[id(k)] for k in failed})
        clean = 10 * (1 - share) - sum(_points(m) for m in ms) - FAILED_ATTEMPT_POINTS * max(0, tried - explained)
        out.append({"start_s": a, "end_s": b, "label": label, "outcome": outcome, "goal": goal, "task": task,
                    "reached": outcome in ("success", "success_then_undone"),
                    "duration_s": round(b - a, 2),
                    "cleanliness": _clamp(clean), "steady": _clamp(10 * (1 - 2 * hes)),
                    "wasteful_share": round(share, 3), "mistakes": [m["issue"] for m in ms],
                    "penalties": [{"mistake": m["issue"], "severity": m["severity"], "recovered": m["recovered"],
                                   "t_s": m["t_s"], "points": _points(m)} for m in ms],
                    "failed_attempts": [{"t_s": k["t_s"], "label": _str(k.get("label")), "kind": _str(k.get("kind"))}
                                        for k in failed]})
    return out


def _words(label) -> frozenset:
    return frozenset(w for w in _content_words(label) if len(w) > 2 and not w.isdigit())


def _scope(d: dict):
    """(dataset, recording setup) of an episode, None without a setup: one dataset is one kind of robot and camera."""
    rig, ds = _str(d.get("_rig")), _str(d.get("dataset"))
    return (ds, rig) if rig else None


def _similar_pools(records: dict, subs_of: dict) -> dict:
    """{(file, subgoal index): _Peers} of the durations of reached subgoals of the same kind in other episodes of the
    same dataset on the same recording setup: labels sharing at least SIMILAR of their content words. Durations from
    another dataset are never compared, and neither are the episode's own other subgoals, so one operator's repeated
    units are not their own peers."""
    rows = {}                                   # (scope, words) -> [(file, duration)]
    for f, subs in subs_of.items():
        where = _scope(records[f])
        if where is None:
            continue
        for st in subs:
            words = _words(st["label"])
            if words and not st["goal"] and st["reached"] and st["duration_s"] > 0:
                rows.setdefault((where, words), []).append((f, st["duration_s"]))
    by_scope = collections.defaultdict(list)
    for where, words in rows:
        by_scope[where].append(words)
    near = {}
    for where, sets in by_scope.items():
        index = _Near(SIMILAR, sets)
        for words in sets:
            index.add(words, words)
        near[where] = index
    pools = {}
    for where, words in rows:
        kin = [other for _, other, _ in near[where].near(words)]
        members = [m for other in kin for m in rows[(where, other)]]
        own = collections.defaultdict(list)
        for f, dur in members:
            own[f].append(dur)
        pools[(where, words)] = (sorted(dur for _, dur in members), own)
    out = {}
    for f, subs in subs_of.items():
        where = _scope(records[f])
        for i, st in enumerate(subs):
            key = (where, _words(st["label"]))
            if key in pools and not st["goal"] and st["reached"]:
                ordered, own = pools[key]
                out[(f, i)] = _Peers(ordered, own.get(f, []))
    return out


def _safe(fn, *args, fallback):
    """fn(*args), or fallback when the record is too malformed to read, so one record never stops the board."""
    try:
        return fn(*args)
    except (TypeError, ValueError, AttributeError, KeyError, OverflowError, RecursionError):
        return fallback


def score_board(records: dict) -> dict:
    """{file: scores} for every episode of a board: each record's own measures, and the relative dimensions ranked
    within its task group among the episodes that reached the goal."""
    records = {f: d if isinstance(d, dict) else {} for f, d in records.items()}
    empty = facts({})
    fs = {f: _safe(facts, d, fallback=empty) for f, d in records.items()}
    unread = {f for f, d in records.items() if fs[f] is empty and d}
    groups = task_groups(records)
    took_of = {f: (x["goal_s"] - x["start_s"]) if x["goal_s"] is not None and x["start_s"] is not None
               and x["goal_s"] > x["start_s"] else None for f, x in fs.items()}
    pp_of = {f: x["actions"] / x["progress_reached"] if x["progress_reached"] else None for f, x in fs.items()}
    subs_of = {f: _safe(subgoals, d, fs[f], fallback=[]) for f, d in records.items()}
    similar = _similar_pools(records, subs_of)
    # the successful demonstrations of each group, as sorted values, built once: time to goal, actions per progress,
    # and each subgoal's duration among those that split the task into the same number of subgoals
    count, took_in, pp_in, sub_in = {}, {}, {}, {}
    for f, x in fs.items():
        if x["outcome"] != "success":
            continue
        g = groups[f]
        count[g] = count.get(g, 0) + 1
        if took_of[f] is not None:
            took_in.setdefault(g, []).append(took_of[f])
        if pp_of[f] is not None:
            pp_in.setdefault(g, []).append(pp_of[f])
        for i, st in enumerate(subs_of[f]):
            if st["duration_s"] > 0:
                sub_in.setdefault((g, len(subs_of[f]), i), []).append(st["duration_s"])
    for table in (took_in, pp_in, sub_in):
        for values in table.values():
            values.sort()

    def peers_of(table, key, f, own):
        """The group's values with this episode's own left out when it is one of them (a successful episode)."""
        return _Peers(table.get(key, []), own if fs[f]["outcome"] == "success" and own is not None else None)

    out = {}
    for f in records:
        x = fs[f]
        g = groups[f]
        peers = count.get(g, 0) - (x["outcome"] == "success")
        outcome, o_in = _outcome(x)
        clean, c_in = _cleanliness(x)
        took = took_of[f]
        peer_took = peers_of(took_in, g, f, took)
        efficiency = _ratio_score(took, peer_took) if x["outcome"] == "success" else None
        hes = x["hesitation_s"] / x["window_s"] if x["window_s"] else None
        per_progress = pp_of[f]
        peer_pp = peers_of(pp_in, g, f, per_progress)
        # hesitation is absolute (half the work spent with every arm waiting scores 0); busyness is against peers
        steady = _clamp(10 * (1 - 2 * hes)) if hes is not None else None
        busy = _ratio_score(per_progress, peer_pp)
        parts = [s for s in (steady, busy) if s is not None]
        fluency = _clamp(statistics.mean(parts)) if parts else None
        dims = {"cleanliness": clean, "efficiency": efficiency, "fluency": fluency}
        subs = subs_of[f]
        # a subgoal's speed is judged against the same subgoal of peers that split the task the same way, and without
        # enough of them against the same kind of subgoal elsewhere in the dataset; a stretch that never reached its
        # end has no execution score
        for i, st in enumerate(subs):
            same = peers_of(sub_in, (g, len(subs), i), f, st["duration_s"])
            basis = "task"
            if len(same) < MIN_PEERS and (f, i) in similar:
                same, basis = similar[(f, i)], "similar"
            enough = st["reached"] and len(same) >= MIN_PEERS
            st["efficiency"] = _ratio_score(st["duration_s"], same) if enough else None
            st["peers"] = len(same)
            st["peer_basis"] = basis if enough else None
            st["peer_median_s"] = same.median() if enough else None
            st["execution"] = (_weighted({"cleanliness": st["cleanliness"], "efficiency": st["efficiency"],
                                          "fluency": st["steady"]}) if st["reached"] else None)
        # a recording labelled in parts with a part that has no labels: the labelled parts say nothing of the rest, so
        # it gets no whole-recording execution or outcome, only its subgoals' and the parts that are missing
        missing = [g for g in _list(_dict(records[f].get("_stitched")).get("missing")) if isinstance(g, dict)]
        if missing:
            dims = {key: None for key in dims}
            outcome = None
            o_in = {**o_in, "unlabelled_parts": [{key: g.get(key) for key in ("part", "t0_s", "t1_s")} for g in missing]}
        out[f] = {"version": VERSION, "execution": _weighted(dims), "outcome": outcome, **dims, "relation": x["relation"],
                  "weights": dict(EXECUTION_WEIGHTS),
                  "subgoals": subs,
                  "peers": peers, "peer_basis": "task" if efficiency is not None else None, "group": g,
                  "inputs": {**o_in, **c_in, "time_to_goal_s": took, "peer_median_time_to_goal_s":
                             peer_took.median() if len(peer_took) else None,
                             "hesitation_share": round(hes, 3) if hes is not None else None,
                             "steadiness": steady, "busyness": busy,
                             "actions_per_progress": round(per_progress, 2) if per_progress else None},
                  **({"unread": True} if f in unread else {})}
    return out
