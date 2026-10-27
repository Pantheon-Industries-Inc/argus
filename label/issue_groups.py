"""Group the issues of a long recording that report one condition, after its parts are stitched.

A long recording is labelled in parts (label/pieces.py), and each part reports its own data issues and operator
mistakes. A condition of the whole take (the wrist cameras' sides swapped, the exo camera facing the ceiling) is then
reported once by every part, a condition that comes and goes (a cable that crosses the right camera at 959s and again
at 1073s) once for each time it is seen, and one event seen by both parts around a cut once by each. In 12,903 data
issues of about 2,150 labelled UMI takes, 2,536 of 5,891 (take, category) groups hold more than one issue, but most of
those are separate events (failed grasps at 83s, 96s and 104s are three grasps), so a category alone never groups
anything.

Which issues are asked about. Within one list of one recording, issues of one category are linked when
- one of them reads as a condition rather than an event (condition_signal): it has no time, it spans at least half of
  the part that reported it (WIDE_FRACTION), or its sentence uses words of persistence or recurrence
  (CONDITION_WORDS: again, throughout, persistently, intermittently, ...). Such an issue is linked to every other issue
  of its category, since the condition may be what each of them reports.
- they come from different parts and their spans overlap or lie within NEAR_S of each other: one event that both parts
  saw around a cut, which the stitch keeps from both (it keeps one copy of a step or key event in a shared window,
  label/pieces.py window_copies, but each part's issues as they came).
Each connected set of two or more is a candidate. Ordinal words (another, a second, further) and words of one event's
own retries (repeatedly, remains) are not signals: in the issues above they mark separate events far more often than
one condition.

Who decides. A model reads only the candidate's issues as JSON and answers which of them describe one condition or one
event (prompt); every issue it puts in no group stays as it is, and a reply that cannot be read, a failed call or no
model key leaves every issue as it is. The call goes through the labelling client (label/harness.py call_model_once,
keys from get_keys), with the recording's labelling model at low reasoning unless asked otherwise; like every Argus
call it sends no temperature, so each model runs with its provider's own sampling. Each answer is kept under the
fingerprint of what was asked (the issues, the model, the reasoning and the prompt's version), so a stitch run again
never pays for it twice.

What a group becomes (grouped_issue). One issue that keeps every member verbatim under occurrences, in time order,
with the model's one sentence as its issue, the members' category, the highest severity of the members, the span from
the earliest member time to the latest, parts naming every part that reported it, and an evidence text that joins
each member's evidence under its time. Any other field the members share (recovered, deliberate) is kept. The model
writes only the sentence; nothing a part reported is dropped or reworded. A grouped issue counts once wherever issues
are counted, at its severity.

The check after grouping (verify). Every original issue appears in exactly one place, on its own or as an
occurrence, unchanged; every grouped issue holds at least two occurrences of its category and carries their highest
severity and their span. A group that fails is kept as its members, and a list that fails as a whole is kept as it
came, each with the reason under the record's fallbacks.

The record. result["issue_groups"] says what was asked, what was grouped, every fallback and its reason, and the
calls' cost; the cost is also an invoice of kind issue_groups in result["auxiliary_cost_history"], so
label/harness.py episode_cost counts it. ARGUS_ISSUE_GROUPS=0 turns the step off (label/pieces.py stitch_run);
ARGUS_ISSUE_GROUPS_CAP_USD bounds what one stitch run may spend on it (CAP_USD by default).

    python -m label.issue_groups --label RESULT.json|DIR --out OUT [--dry-run] [--cap 5]
    python -m label.issue_groups --dump ISSUES.jsonl --out OUT [--takes 150] [--seed 0] [--dry-run] [--cap 5]

The first runs over stitched results (a job's run_final/out); the second over issue rows ({"dataset", "take",
"seconds", "issue"}, one per line) grouped by take. Neither touches the input: OUT gets results.jsonl (one line per
recording: its issues before and after, and its record), cache.json (the answers, reused by a later run with the
same OUT) and summary.json.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VERSION = 1
PROMPT_VERSION = 2
LISTS = ("data_issues", "operator_mistakes")
ENV = "ARGUS_ISSUE_GROUPS"
CAP_ENV = "ARGUS_ISSUE_GROUPS_CAP_USD"
# whether the step runs when ARGUS_ISSUE_GROUPS is unset
DEFAULT_ON = True
CAP_USD = 5.0
# the part limit (label/policy.py part_max_s at a rig's default sampling): the window a part of unknown length covers
PART_MAX_S = 450.0
# an issue spanning at least this share of the part that reported it reads as a condition of that part
WIDE_FRACTION = 0.5
# two parts' issues this close in time can be one event seen by both around a cut (label/pieces.py OVERLAP_S)
NEAR_S = 15.0
# words that say a condition persists or comes back, read in the issue's own sentence (not its evidence, which
# describes one event's moments)
CONDITION_WORDS = re.compile(
    r"\b(again|still|persist\w*|throughout|entire(?:ly)?|the whole (?:take|recording|episode|window|part)"
    r"|continu\w*|recurr\w*|intermittent\w*|for the rest|all along|once more)\b", re.I)
# a candidate larger than this is not asked about: its issues stay as they are, and the record says why
MAX_CANDIDATE = 24
REASONING = "low"
ANSWER_TOKENS = 1500
SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}
TIMEOUT_S = 600


# ---------------------------------------------------------------- numbers and identity

def _num(x) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)) and math.isfinite(x):
        return float(x)
    return None


def canonical(item) -> str:
    return json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _key4(item: dict) -> str:
    """The four fields every original issue must keep where it ends up: its times, its evidence and its part."""
    return canonical([item.get("t_s"), item.get("end_s"), item.get("evidence"), item.get("part")])


def _rank(item: dict) -> int:
    value = item.get("severity")
    return SEVERITY_RANK.get(value.strip().lower(), 0) if isinstance(value, str) else 0


def _parts_of(item: dict) -> list:
    parts = item.get("parts") if isinstance(item.get("parts"), list) else [item.get("part")]
    return [p for p in parts if p is not None]


# ---------------------------------------------------------------- which issues are asked about

def part_lengths(result: dict) -> dict:
    """{part index: seconds the part covers} from a stitched result's config.pieces."""
    out = {}
    for piece in (result.get("config") or {}).get("pieces") or []:
        if not isinstance(piece, dict):
            continue
        t0, t1 = _num(piece.get("t0_s")), _num(piece.get("t1_s"))
        if piece.get("part") is not None and t0 is not None and t1 is not None and t1 > t0:
            out[piece["part"]] = t1 - t0
    return out


def condition_signal(item: dict, part_s: float) -> str | None:
    """Why an issue reads as a condition rather than one event ("untimed", "wide", "wording"), or None."""
    t, end = _num(item.get("t_s")), _num(item.get("end_s"))
    if t is None or end is None:
        return "untimed"
    if part_s > 0 and end - t >= WIDE_FRACTION * part_s:
        return "wide"
    if isinstance(item.get("issue"), str) and CONDITION_WORDS.search(item["issue"]):
        return "wording"
    return None


def _near(a: dict, b: dict) -> bool:
    """Two issues of different parts whose spans overlap or lie within NEAR_S of each other."""
    pa, pb = set(_parts_of(a)), set(_parts_of(b))
    if not pa or not pb or pa & pb:
        return False
    a0, b0 = _num(a.get("t_s")), _num(b.get("t_s"))
    if a0 is None or b0 is None:
        return False
    a1, b1 = _num(a.get("end_s")), _num(b.get("end_s"))
    a1 = a1 if a1 is not None and a1 >= a0 else a0
    b1 = b1 if b1 is not None and b1 >= b0 else b0
    return max(a0, b0) - min(a1, b1) <= NEAR_S


def candidates(items: list, part_s: dict, default_s: float) -> list[dict]:
    """[{"category", "rows": [index into items], "signals": {index: why}}] for one list of issues: each connected set
    of two or more issues of one category, linked as the module docstring says. An issue already grouped (it has
    occurrences) is never asked about again."""
    by_category = collections.defaultdict(list)
    for n, item in enumerate(items):
        if (isinstance(item, dict) and isinstance(item.get("issue"), str) and item["issue"].strip()
                and isinstance(item.get("category"), str) and not isinstance(item.get("occurrences"), list)):
            by_category[item["category"]].append(n)
    out = []
    for category, rows in by_category.items():
        if len(rows) < 2:
            continue
        signals = {}
        for n in rows:
            lengths = [part_s.get(p) for p in _parts_of(items[n]) if part_s.get(p)]
            why = condition_signal(items[n], min(lengths) if lengths else default_s)
            if why:
                signals[n] = why
        parent = {n: n for n in rows}

        def find(n):
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n

        def link(a, b):
            parent[find(a)] = find(b)
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if a in signals or b in signals or _near(items[a], items[b]):
                    link(a, b)
        sets = collections.defaultdict(list)
        for n in rows:
            sets[find(n)].append(n)
        for members in sets.values():
            if len(members) >= 2:
                out.append({"category": category, "rows": sorted(members),
                            "signals": {n: signals[n] for n in members if n in signals}})
    return sorted(out, key=lambda c: c["rows"][0])


# ---------------------------------------------------------------- the request

def shown(item: dict, n: int) -> dict:
    return {"id": n, "part": item.get("part"), "t_s": item.get("t_s"), "end_s": item.get("end_s"),
            "severity": item.get("severity"), "issue": item.get("issue"), "evidence": item.get("evidence")}


def prompt(category: str, members: list, recording_s: float | None = None, n_parts: int | None = None) -> str:
    about = ("A long robot-learning recording" + (f" of {recording_s:.0f}s" if recording_s else "")
             + (f", labelled in {n_parts} parts" if n_parts and n_parts > 1 else ", labelled in parts"))
    return (
        f"ISSUE GROUPING. {about}, each part by its own request that saw only its own stretch of footage. The parts "
        f'reported the issues below, all of category "{category}", with times in seconds on the recording\'s clock '
        "(null where the part gave no time). Each part reports what it sees, so one condition can be reported "
        "several times: a condition of the whole recording by every part, a condition that comes and goes each time "
        "it is seen, and one event seen by two parts around the point where they meet by both of them.\n\n"
        "Decide which of these issues describe the same underlying condition or the same single event.\n"
        "- Group issues that report one state of the recording, its cameras, its setup or its scene that persists or "
        "keeps coming back with the same cause (the same camera, cable, lighting, side assignment, person or "
        "arrangement), so a reviewer would judge it once for the whole recording.\n"
        "- Group two reports of one event: the same objects and action at overlapping or adjacent times, reported by "
        "different parts.\n"
        "- Keep apart issues that are separate events, each with its own moment: separate attempts, grasps, drops, "
        "spills, collisions, interventions or pauses, even of the same kind, on the same object or by the same "
        "hand. Several events of one kind are not one condition.\n"
        "- When unsure, keep issues apart.\n"
        "- The sentence of a group covers everything each of its issues reports. When one issue summarizes several "
        "events (several pauses, several throws) and another reports one of them, the sentence describes the several "
        "events, never only the one.\n\n"
        "Answer with one JSON object and nothing else:\n"
        '{"groups": [{"ids": [<the ids of two or more issues that are one condition or one event>], "issue": "<one '
        'plain sentence describing that condition or event for the whole recording, without times>", "why": "<a '
        'short reason>"}]}\n'
        'List only groups of two or more issues; an issue in no group stays as it is, and "groups" is [] when every '
        "issue is separate.\n\n"
        "Issues:\n" + json.dumps([shown(m, n) for n, m in enumerate(members)], ensure_ascii=False, indent=1))


def fingerprint(members: list, model: str, reasoning: str) -> str:
    return hashlib.sha256(canonical({"version": VERSION, "prompt": PROMPT_VERSION, "model": model,
                                     "reasoning": reasoning, "issues": members}).encode()).hexdigest()


def read_answer(text: str, n: int) -> list[dict]:
    """[{"ids", "issue", "why"}] for each group of two or more from the reply; ValueError when the reply is not that
    strict JSON (ids that are not distinct whole numbers below n, an id in two groups, a group with no sentence)."""
    fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text or "", re.S)
    try:
        answer = json.loads(fenced.group(1) if fenced else text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"the reply is not JSON: {error}") from error
    groups = answer.get("groups") if isinstance(answer, dict) else None
    if not isinstance(groups, list):
        raise ValueError('the reply has no "groups" list')
    seen, out = set(), []
    for g in groups:
        ids = g.get("ids") if isinstance(g, dict) else None
        if not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < n for i in ids):
            raise ValueError(f"a group's ids are not ids of the issues asked about: {ids!r}"[:200])
        if len(set(ids)) != len(ids) or seen & set(ids):
            raise ValueError("an issue is named twice")
        seen |= set(ids)
        if len(ids) < 2:
            continue
        sentence = g.get("issue")
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError("a group has no sentence")
        out.append({"ids": sorted(ids), "issue": " ".join(sentence.split())[:600],
                    "why": str(g.get("why") or "")[:400]})
    return out


# ---------------------------------------------------------------- what a group becomes

def _when(item: dict) -> str:
    t, end = _num(item.get("t_s")), _num(item.get("end_s"))
    span = "untimed" if t is None else f"{t:g}s" if end is None or end == t else f"{t:g}-{end:g}s"
    parts = _parts_of(item)
    return span + (f" (part {', '.join(str(p) for p in parts)})" if parts else "")


def _span(members: list) -> tuple[float | None, float | None]:
    starts = [t for m in members if (t := _num(m.get("t_s"))) is not None]
    ends = [t for m in members for t in (_num(m.get("end_s")), _num(m.get("t_s"))) if t is not None]
    return (min(starts) if starts else None), (max(ends) if ends else None)


OWN_FIELDS = ("issue", "category", "severity", "t_s", "end_s", "evidence", "part", "parts", "occurrences", "grouped")


def grouped_issue(members: list, sentence: str, why: str, key: str) -> dict:
    """One issue for a group: every member verbatim under occurrences, in time order (module docstring)."""
    occurrences = sorted((copy.deepcopy(m) for m in members),
                         key=lambda m: (_num(m.get("t_s")) is None, _num(m.get("t_s")) or 0.0,
                                        _num(m.get("end_s")) or 0.0))
    t, end = _span(members)
    top = max(members, key=_rank)
    parts = sorted({p for m in members for p in _parts_of(m)}, key=lambda p: (str(type(p)), p))
    shared = {}
    for field in {k for m in members for k in m} - set(OWN_FIELDS):
        values = [m.get(field, KeyError) for m in members]
        if all(canonical(v) == canonical(values[0]) for v in values if v is not KeyError) and KeyError not in values:
            shared[field] = copy.deepcopy(values[0])
    return {"issue": sentence, "category": members[0].get("category"), "severity": top.get("severity"),
            "t_s": t, "end_s": end,
            "evidence": " ".join(f"{_when(m)}: {m.get('evidence') or ''}".strip() for m in occurrences),
            "part": parts[0] if parts else None, "parts": parts, **shared,
            "occurrences": occurrences,
            "grouped": {"by": "label/issue_groups.py", "version": VERSION, "why": why, "fingerprint": key}}


def verify(before: list, after: list) -> str | None:
    """Why a grouped list is not the original list regrouped, or None when it is: every original issue appears
    exactly once, on its own or as an occurrence, unchanged (its times, evidence and part above all), and every
    grouped issue holds two or more occurrences of its own category, with their highest severity and their span."""
    flat = []
    for item in after:
        if isinstance(item, dict) and "occurrences" in item and "grouped" in item:
            occ = item["occurrences"]
            if not isinstance(occ, list) or len(occ) < 2:
                return "a grouped issue holds fewer than two occurrences"
            if any(not isinstance(m, dict) or m.get("category") != item.get("category") for m in occ):
                return "a grouped issue holds an occurrence of another category"
            if _rank(item) != max(_rank(m) for m in occ):
                return "a grouped issue's severity is not the highest of its occurrences"
            if (item.get("t_s"), item.get("end_s")) != _span(occ):
                return "a grouped issue's span is not its occurrences' span"
            if not isinstance(item.get("issue"), str) or not item["issue"].strip():
                return "a grouped issue has no sentence"
            flat += occ
        else:
            flat.append(item)
    if collections.Counter(map(_key4, (x for x in before if isinstance(x, dict)))) != collections.Counter(
            map(_key4, (x for x in flat if isinstance(x, dict)))):
        return "an original issue's times, evidence or part is missing or appears more than once"
    if collections.Counter(map(canonical, before)) != collections.Counter(map(canonical, flat)):
        return "an original issue is missing, changed or appears more than once"
    return None


# ---------------------------------------------------------------- grouping one list

def group_list(items: list, ask, part_s: dict, default_s: float, *, list_name: str = "data_issues",
               context: dict | None = None) -> tuple[list, dict]:
    """(the list with its groups, the record of what was done) for one list of issues. ask(category, members,
    context) gives ({"groups": [...]} or {"problem": why}, the call's record) for one candidate (model_asker)."""
    record = {"asked": 0, "grouped": [], "kept_apart": 0, "fallbacks": [], "calls": []}
    replace, used = {}, set()
    for cand in candidates(items, part_s, default_s):
        members = [items[n] for n in cand["rows"]]
        if len(members) > MAX_CANDIDATE:
            record["fallbacks"].append({"list": list_name, "category": cand["category"], "issues": len(members),
                                        "reason": f"more than {MAX_CANDIDATE} issues in one candidate; not asked"})
            continue
        record["asked"] += 1
        answer, call = ask(cand["category"], members, context or {})
        if call:
            record["calls"].append(call)
        if "problem" in answer:
            record["fallbacks"].append({"list": list_name, "category": cand["category"], "issues": len(members),
                                        "reason": answer["problem"]})
            continue
        for g in answer["groups"]:
            group = [members[i] for i in g["ids"]]
            merged = grouped_issue(group, g["issue"], g["why"], answer.get("fingerprint") or "")
            problem = verify(group, [merged])
            if problem:
                record["fallbacks"].append({"list": list_name, "category": cand["category"], "issues": len(group),
                                            "reason": problem})
                continue
            rows = [cand["rows"][i] for i in g["ids"]]
            replace[rows[0]] = merged
            used.update(rows[1:])
            record["grouped"].append({"list": list_name, "category": cand["category"], "issues": len(group),
                                      "signals": sorted({cand["signals"][n] for n in rows if n in cand["signals"]})})
        record["kept_apart"] += len(members) - sum(len(g["ids"]) for g in answer["groups"])
    out = [replace.get(n, item) for n, item in enumerate(items) if n not in used]
    problem = verify(items, out)
    if problem:
        record["fallbacks"].append({"list": list_name, "category": None, "issues": len(items),
                                    "reason": f"the whole list is kept as it came: {problem}"})
        record["grouped"] = []
        return list(items), record
    return _in_time_order(out), record


def _in_time_order(items: list) -> list:
    """The list in time order as label/pieces.py chronological sorts it: by start, ties by end, untimed last."""
    def key(item):
        a = _num(item.get("t_s")) if isinstance(item, dict) else None
        b = _num(item.get("end_s")) if isinstance(item, dict) else None
        return (a is None, a if a is not None else 0.0, b is None, b if b is not None else 0.0)
    return sorted(items, key=key)


# ---------------------------------------------------------------- the model

class Budget:
    """What the calls may spend, held at each call's bound until its reply says what it cost (label/seam.py)."""

    def __init__(self, cap_usd: float):
        self.cap, self.spent, self.lock = float(cap_usd), 0.0, threading.Lock()

    def reserve(self, bound: float) -> bool:
        with self.lock:
            if self.spent + bound > self.cap:
                return False
            self.spent += bound
            return True

    def settle(self, bound: float, cost: float) -> None:
        with self.lock:
            self.spent += cost - bound


class Cache:
    """The answers by fingerprint, in memory and, given a path, in a JSON file written after every new answer."""

    def __init__(self, path: Path | None = None):
        self.path, self.lock = (Path(path) if path else None), threading.Lock()
        self.answers = {}
        if self.path and self.path.is_file():
            self.answers = (json.loads(self.path.read_text()) or {}).get("answers") or {}

    def get(self, key: str):
        with self.lock:
            return copy.deepcopy(self.answers.get(key))

    def put(self, key: str, value: dict) -> None:
        from label.atomic import write_atomic
        with self.lock:
            self.answers[key] = copy.deepcopy(value)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                write_atomic(self.path, {"version": VERSION, "answers": self.answers}, indent=None)


def settings(result: dict | None = None, model: str | None = None, reasoning: str | None = None) -> dict:
    """The recording's labelling model unless asked otherwise, at REASONING, and the output limit."""
    from label import policy
    result = result or {}
    chosen = result.get("annotation_policy") or {}
    model = model or result.get("model") or chosen.get("model") or policy.ASTRA
    reasoning = reasoning or REASONING
    return {"model": model, "reasoning": reasoning, "max_tokens": policy.output_limit(ANSWER_TOKENS, reasoning)}


def model_asker(chosen: dict, key: str | None, cache: Cache, budget: Budget, *, timeout: int = TIMEOUT_S,
                dry_run: bool = False):
    """ask() for group_list through the labelling client. A cached answer is used as it was paid for; a call that
    provably cost nothing is not kept, so a later run asks again; any other answer is kept, read or not."""
    from label import harness
    from label.errors import DispatchError

    def ask(category: str, members: list, context: dict):
        fp = fingerprint(members, chosen["model"], chosen["reasoning"])
        saved = cache.get(fp)
        if saved is not None:
            return _answer(saved, len(members), fp), {**(saved.get("request") or {}), "cached": True}
        content = [{"type": "text", "text": prompt(category, members, context.get("recording_s"),
                                                   context.get("parts"))}]
        bound = harness.final_cost_bound(content, chosen["max_tokens"], model=chosen["model"],
                                         require_known_pricing=False)
        if dry_run:
            return {"problem": "dry run: not asked"}, {"dry_run": True, "bound_usd": bound}
        if key is None:
            return {"problem": "no model key: set OPENROUTER_API_KEYS or OPENAI_API_KEY"}, None
        if not budget.reserve(bound):
            return {"problem": f"the ${budget.cap:g} cap for issue grouping is reached; not asked"}, None
        request = {"model": chosen["model"], "reasoning": chosen["reasoning"], "max_tokens": chosen["max_tokens"],
                   "reserved_usd": bound}
        try:
            resp = harness.call_model_once(content, chosen["model"], chosen["reasoning"], key,
                                           chosen["max_tokens"], timeout)
        except DispatchError as error:
            unpaid = bool(getattr(error, "verified_unpaid", False))
            budget.settle(bound, 0.0 if unpaid else bound)
            request.update(dispatch="failed", cost_usd=0.0 if unpaid else bound,
                           cost_is_conservative_estimate=not unpaid)
            saved = {"text": None, "problem": f"the request failed: {error}"[:300], "request": request}
            if not unpaid:
                cache.put(fp, saved)
            return _answer(saved, len(members), fp), request
        usage = resp.get("usage") or {}
        cost, source = harness._verified_final_cost(usage, chosen["model"])
        budget.settle(bound, bound if cost is None else cost)
        choice = (resp.get("choices") or [{}])[0]
        request.update(dispatch="received", usage=usage, cost_usd=bound if cost is None else cost,
                       cost_source=source, cost_is_conservative_estimate=cost is None,
                       generation_id=resp.get("id"), provider=resp.get("provider"),
                       finish_reason=choice.get("finish_reason"))
        saved = {"text": (choice.get("message") or {}).get("content") or "", "request": request}
        cache.put(fp, saved)
        return _answer(saved, len(members), fp), request
    return ask


def _answer(saved: dict, n: int, fp: str) -> dict:
    if saved.get("problem"):
        return {"problem": saved["problem"]}
    if (saved.get("request") or {}).get("finish_reason") == "length":
        return {"problem": "the reply was cut off at the output limit"}
    try:
        return {"groups": read_answer(saved.get("text") or "", n), "fingerprint": fp}
    except ValueError as error:
        return {"problem": f"the reply could not be read: {error}"[:300]}


# ---------------------------------------------------------------- one result

def enabled() -> bool:
    value = os.environ.get(ENV)
    return DEFAULT_ON if value is None or not value.strip() else value.strip().lower() not in ("0", "false", "off", "no")


def cap_usd() -> float:
    try:
        value = float(os.environ.get(CAP_ENV) or CAP_USD)
    except ValueError:
        return CAP_USD
    return value if math.isfinite(value) and value >= 0 else CAP_USD


def recording_seconds(result: dict) -> float | None:
    ends = [_num(p.get("t1_s")) for p in (result.get("config") or {}).get("pieces") or [] if isinstance(p, dict)]
    ends = [e for e in ends if e is not None]
    return max(ends) if ends else None


def apply(result: dict, ask, *, default_s: float | None = None) -> dict:
    """Group the issues of one stitched result in place and return its record (result["issue_groups"]). A result
    already grouped is left as it is. The calls' cost is an invoice of kind issue_groups in its
    auxiliary_cost_history."""
    if isinstance(result.get("issue_groups"), dict):
        return result["issue_groups"]
    labels = result.get("labels")
    if not isinstance(labels, dict):
        return {}
    lengths = part_lengths(result)
    seconds = recording_seconds(result)
    default = default_s or min(seconds or PART_MAX_S, PART_MAX_S)
    context = {"recording_s": seconds, "parts": len(lengths) or None}
    record = {"version": VERSION, "asked": 0, "grouped": [], "kept_apart": 0, "fallbacks": [], "calls": []}
    before = {}
    for name in LISTS:
        if not isinstance(labels.get(name), list):
            continue
        before[name] = len(labels[name])
        labels[name], one = group_list(labels[name], ask, lengths, default, list_name=name, context=context)
        for k in ("asked", "kept_apart"):
            record[k] += one[k]
        for k in ("grouped", "fallbacks", "calls"):
            record[k] += one[k]
    if not record["asked"] and not record["fallbacks"]:
        return {}
    record["issues"] = {name: {"before": n, "after": len(labels[name])} for name, n in before.items()}
    calls = [c for c in record["calls"] if c]
    record["cost_usd"] = math.fsum(float(c.get("cost_usd") or 0) for c in calls)
    record["models"] = sorted({c["model"] for c in calls if c.get("model")})
    result["issue_groups"] = record
    if record["cost_usd"] > 0:
        result.setdefault("auxiliary_cost_history", []).append({
            "kind": "issue_groups", "cost_usd": record["cost_usd"], "previously_billed_usd": 0.0,
            "cost_is_conservative_estimate": any(c.get("cost_is_conservative_estimate") for c in calls)})
    return record


def apply_in_stitch(result: dict, cache_path: Path, budget: Budget) -> dict:
    """The step as label/pieces.py stitch_run runs it on each stitched recording: the labelling client, the answers
    kept in cache_path. With no model key (a stitch of replies labelled elsewhere) nothing is asked and the result is
    left as it is; python -m label.issue_groups --label groups it later."""
    from label import harness
    keys = harness.get_keys()
    if not keys:
        return {}
    ask = model_asker(settings(result), keys[0], Cache(cache_path), budget)
    return apply(result, ask)


# ---------------------------------------------------------------- offline

def read_dump(path: Path) -> dict:
    """{take: {"dataset", "seconds", "issues": [issue, ...]}} from issue rows, one per line, in file order."""
    takes = {}
    with open(path) as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            take = takes.setdefault(row["take"], {"dataset": row.get("dataset"), "seconds": row.get("seconds"),
                                                  "issues": []})
            take["issues"].append(row["issue"])
    return takes


def dump_result(take: dict) -> dict:
    """A take of the dump as a result apply() reads: its issues as data_issues and its length as one window, since
    the dump does not say where its parts were cut."""
    seconds = _num(take.get("seconds"))
    parts = sorted({i.get("part") for i in take["issues"] if i.get("part") is not None})
    return {"labels": {"data_issues": copy.deepcopy(take["issues"])},
            "_dump": {"dataset": take.get("dataset"), "seconds": seconds, "parts": len(parts)}}


def stratified(takes: dict, n: int, seed: int) -> list[str]:
    """n takes of the dump labelled in parts: an even share from each dataset, half of it from takes with at least one
    candidate, and then one take for each category with candidates that the draw left out."""
    rng = random.Random(seed)
    parted = {k: v for k, v in takes.items() if any(i.get("part") is not None for i in v["issues"])}
    by = collections.defaultdict(lambda: ([], []))
    cand_cats = {}
    for name in sorted(parted):
        take = parted[name]
        found = candidates(take["issues"], {}, min(_num(take.get("seconds")) or PART_MAX_S, PART_MAX_S))
        cand_cats[name] = {c["category"] for c in found}
        by[take.get("dataset")][0 if found else 1].append(name)
    datasets = sorted(by)
    share = n // max(1, len(datasets))
    chosen = []
    for d in datasets:
        with_c, without = by[d]
        k = min(len(with_c), (share + 1) // 2)
        chosen += rng.sample(with_c, k)
        chosen += rng.sample(without, min(len(without), share - k))
    covered = set().union(*(cand_cats[t] for t in chosen)) if chosen else set()
    for cat in sorted(set().union(*cand_cats.values()) - covered):
        pool = sorted(t for t, cs in cand_cats.items() if cat in cs and t not in chosen)
        if pool:
            chosen.append(rng.choice(pool))
    return chosen


def main(argv: list[str] | None = None) -> int:
    from label.atomic import write_atomic
    ap = argparse.ArgumentParser(prog="python -m label.issue_groups", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--label", type=Path, help="a stitched result or a folder of them")
    src.add_argument("--dump", type=Path, help="issue rows, one JSON object per line, grouped by take")
    ap.add_argument("--out", type=Path, required=True, help="the folder to write (never the input)")
    ap.add_argument("--takes", type=int, default=150, help="with --dump: takes in the stratified sample (0: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=float, default=5.0, help="most the calls may spend, in USD (default 5)")
    ap.add_argument("--model", default=None, help="default: the recording's labelling model (Astra for --dump)")
    ap.add_argument("--reasoning", default=None, help=f"default: {REASONING}")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true", help="count and price the candidates; ask nothing")
    a = ap.parse_args(argv)
    from label import harness
    a.out.mkdir(parents=True, exist_ok=True)
    work = []
    if a.dump:
        takes = read_dump(a.dump)
        names = stratified(takes, a.takes, a.seed) if a.takes else sorted(takes)
        work = [(name, dump_result(takes[name])) for name in names]
    else:
        paths = sorted(a.label.glob("*.json")) if a.label.is_dir() else [a.label]
        work = [(p.stem, json.loads(p.read_text())) for p in paths
                if not p.name.startswith(("failed_", "noreply_"))]
    keys = harness.get_keys()
    if not keys and not a.dry_run:
        raise SystemExit("no model key: set OPENROUTER_API_KEYS or OPENAI_API_KEY")
    cache, budget = Cache(a.out / "cache.json"), Budget(a.cap)
    lock, rows = threading.Lock(), {}

    def one(job):
        name, result = job
        before = {k: copy.deepcopy(result["labels"].get(k)) for k in LISTS
                  if isinstance((result.get("labels") or {}).get(k), list)}
        default = _num((result.get("_dump") or {}).get("seconds"))
        chosen = settings(result, a.model, a.reasoning)
        ask = model_asker(chosen, keys[0] if keys else None, cache, budget, dry_run=a.dry_run)
        record = apply(result, ask, default_s=min(default, PART_MAX_S) if default else None)
        row = {"name": name, **(result.get("_dump") or {}), "before": before,
               "after": {k: result["labels"][k] for k in before}, "record": record}
        with lock:
            rows[name] = row
        return row

    with ThreadPoolExecutor(max(1, a.concurrency)) as pool:
        list(pool.map(one, work))
    ordered = [rows[name] for name, _ in work]
    with open(a.out / "results.jsonl", "w") as stream:
        for row in ordered:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(ordered, budget)
    write_atomic(a.out / "summary.json", summary, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


def summarize(rows: list, budget: Budget | None = None) -> dict:
    before = sum(len(v) for r in rows for v in r["before"].values())
    after = sum(len(v) for r in rows for v in r["after"].values())
    merges, absorbed, fallbacks, reasons = collections.Counter(), collections.Counter(), 0, collections.Counter()
    asked, cost, bound = 0, 0.0, 0.0
    for r in rows:
        rec = r.get("record") or {}
        asked += rec.get("asked", 0)
        for g in rec.get("grouped") or []:
            merges[g["category"]] += 1
            absorbed[g["category"]] += g["issues"]
        for f in rec.get("fallbacks") or []:
            fallbacks += 1
            reasons[f["reason"][:80]] += 1
        for c in rec.get("calls") or []:
            if c and not c.get("cached"):
                cost += float(c.get("cost_usd") or 0)
            bound += float((c or {}).get("bound_usd") or 0)
    return {"recordings": len(rows), "issues_before": before, "issues_after": after, "candidates_asked": asked,
            "groups_by_category": dict(merges.most_common()), "issues_grouped_by_category": dict(absorbed.most_common()),
            "fallbacks": fallbacks, "fallback_reasons": dict(reasons), "cost_usd_new_calls": round(cost, 4),
            **({"dry_run_bound_usd": round(bound, 4)} if bound else {}),
            **({"spent_usd": round(budget.spent, 4)} if budget else {})}


if __name__ == "__main__":
    raise SystemExit(main())
