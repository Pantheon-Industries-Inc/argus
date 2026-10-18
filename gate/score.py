"""Score gate runs: per rig, the cases met, the parse rate, the label consistency flags and the cost per footage hour.

    python -m gate score RUN [RUN ...] [--json FILE]

RUN is a run folder of `python -m gate label` (or of `python -m label` over a gate folder). Every number comes from
the run folders and the episodes they name; nothing is repaired. Per rig:

- cases: each episode with a fact in gate/cases.json is judged against its label (judge()). A reply that did not
  parse, and an episode of the rig's gate folder with no output at all (the harness failed before the model call),
  count as failures.
- parse: replies that parsed, of episodes asked.
- flags: episodes whose label contradicts itself (checks/label_consistency.py).
- cost: over the cost-sample episodes (role cost:<dataset>), per footage hour, billed and cold. The billed cost can
  be lower than a real run's: a request identical to one sent shortly before (the same episode labelled again) is
  served from the provider's prompt cache. In a real run each episode is sent once, so only the rig's shared
  instructions are ever read from the cache. The cold cost reprices every cache read beyond them (the smallest
  nonzero read of the rig in these runs) at the cache-write price, which is the cost of a first send.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
from pathlib import Path

import numpy as np

from checks import label_consistency
from label.harness import PRICE_IN, episode_cost, annotation_record, reply_truncated

HERE = Path(__file__).resolve().parent
RIGS = ("teleop", "handheld", "ego")
SEV = {"low": 1, "medium": 2, "high": 3}
# the provider's prompt cache: a read costs a tenth of the input price, a write 1.25 times it
CACHE_READ, CACHE_WRITE = 0.1 * PRICE_IN, 1.25 * PRICE_IN
def literal_pattern(term: str, *, plural=False):
    """Literal words/phrases, with an explicit optional regular s plural for held objects."""
    suffix = 's?' if plural and not term.lower().endswith('s') else ''
    return re.compile(r'(?<!\w)' + re.escape(term) + suffix + r'(?!\w)', re.I)


def load_cases() -> dict:
    return {k: v for k, v in json.loads((HERE / "cases.json").read_text()).items() if not k.startswith("_")}


def load_selection() -> dict:
    return json.loads((HERE / "selection.json").read_text())["rigs"]


def case_for(name: str, cases: dict) -> dict | None:
    for k, v in cases.items():
        if fnmatch.fnmatch(name, k):
            return v
    return None


def duration_s(ep_dir: Path) -> float:
    """The episode's length: its recorded duration, else the span of its real frame times, else frames / fps."""
    c = json.loads((Path(ep_dir) / "context.json").read_text())
    if c.get("duration_s"):
        return float(c["duration_s"])
    if c.get("real_times"):
        z = np.load(Path(ep_dir) / c["real_times"])
        t = z[next(v for v in ("exo", "left", "right") if v in z.files)]
        return float(t[-1] - t[0])
    return (c.get("n_state_frames") or 0) / float(c.get("fps") or 30)


def texts_handled(labels: dict) -> list[str]:
    """What the label says is handled: the action, object and subgoal text of every timeline step, key events,
    head-camera tasks and the task summary."""
    out = []
    for s in labels.get("timeline") or []:
        if isinstance(s, dict):
            out += [str(s[k]) for k in ("action", "object", "subgoal", "label", "description") if s.get(k)]
    for s in labels.get("key_events") or []:
        if isinstance(s, dict):
            out += [str(s[k]) for k in ("label", "event", "description", "note") if s.get(k)]
    for t in labels.get("tasks") or []:
        if isinstance(t, dict):
            out += [str(t[k]) for k in ("task", "name", "description") if t.get(k)]
    if labels.get("task_summary"):
        out.append(str(labels["task_summary"]))
    return out


def _matches(items, patterns) -> list[dict]:
    return [i for i in items or [] if isinstance(i, dict)
            and any(fnmatch.fnmatch(str(i.get("category")), p) for p in patterns)]


# every kind of fact a case can state; a case with any other key is refused, so no fact is silently left unjudged
FACTS = {"outcome", "not_outcome", "align", "issue", "sev", "mistake", "named", "max_issue_sev", "held", "why"}


def judge(case: dict, labels: dict) -> tuple[bool, list[str]]:
    """(the label meets every fact of the case, why not)."""
    from board.label_projection import effective_outcome, project
    labels, _ = project(labels)
    unknown = set(case) - FACTS
    if unknown:
        raise ValueError(f"gate case states facts the scorer does not judge: {sorted(unknown)}")
    ok, why = [], []
    comp = labels.get("completion") or {}
    outcome = effective_outcome(comp.get("task_completed"))
    if labels.get("tasks") and not outcome:
        outcome = "tasks:" + ",".join(sorted({str(effective_outcome(t.get("outcome"))) for t in labels["tasks"]}))
    if "outcome" in case:
        ok.append(outcome in case["outcome"])
        ok[-1] or why.append(f"outcome {outcome}")
    if "not_outcome" in case:
        ok.append(outcome not in case["not_outcome"])
        ok[-1] or why.append(f"outcome {outcome}")
    if "align" in case:
        rel = (labels.get("goal_alignment") or {}).get("relation")
        ok.append(rel in case["align"])
        ok[-1] or why.append(f"alignment {rel}")
    if "issue" in case:
        matches = _matches([item for item in labels.get('data_issues') or [] if item.get('issue')], case['issue'])
        sevs = [i.get('severity') for i in matches]
        meets = bool(matches)
        if 'sev' in case:
            meets = any(SEV.get(severity, 0) >= SEV[case['sev']] for severity in sevs)
        ok.append(meets)
        meets or why.append('issue missing' if not matches else f'issue at {sevs}')
    if "mistake" in case:
        ok.append(bool(_matches([item for item in labels.get('operator_mistakes') or [] if item.get('issue')], case["mistake"])))
        ok[-1] or why.append("mistake missing")
    if "named" in case:
        text = " ".join(texts_handled(labels)).lower()
        groups = [[term] if isinstance(term, str) else term for term in case["named"]]
        if not all(isinstance(group, list) and group and all(isinstance(w, str) and w for w in group)
                   for group in groups):
            raise ValueError('named facts require words, phrases or explicit alternatives')
        miss = [' / '.join(group) for group in groups if not any(literal_pattern(w).search(text) for w in group)]
        ok.append(not miss)
        miss and why.append(f"never names {miss}")
    if "max_issue_sev" in case:
        issues = [item for item in labels.get('data_issues') or [] if isinstance(item, dict) and item.get('issue')]
        unrated = [(item.get('category'), item.get('severity')) for item in issues if item.get('severity') not in SEV]
        over = [(item.get('category'), item.get('severity')) for item in issues
                if item.get('severity') in SEV and SEV[item['severity']] > SEV[case['max_issue_sev']]]
        ok.append(not unrated and not over)
        unrated and why.append(f'unrated issues prevent verifying severity bound: {unrated}')
        over and why.append(f"issues above {case['max_issue_sev']}: {over}")
    if "held" in case:
        texts = texts_handled(labels)
        wrong, right = case['held']['not_named'], case['held']['named']
        wrong_pattern, right_pattern = literal_pattern(wrong, plural=True), literal_pattern(right, plural=True)
        # The prompt defines timeline.object as the object acted on. Other prose
        # can quote instructions or deny a claim; its meaning requires review.
        objects = [step['object'] for step in labels.get('timeline') or []
                   if isinstance(step, dict) and isinstance(step.get('object'), str)]
        claimed = [text for text in objects if wrong_pattern.search(text)]
        references = [text for text in texts if wrong_pattern.search(text)]
        named = any(right_pattern.search(text) for text in objects)
        ok.append(not references and named)
        if claimed:
            why.append(f'claims {wrong} in timeline.object: {claimed[0][:80]}')
        elif references:
            why.append(f'needs review: {wrong} appears in prose; handling cannot be determined: {references[0][:80]}')
        if not named:
            if any(right_pattern.search(text) for text in texts):
                why.append(f'needs review: no structured timeline.object claim for {right}')
            else:
                why.append(f'never names a {right}')
    return all(ok), why


def score(runs: list[Path]) -> dict:
    cases, sel = load_cases(), load_selection()
    role = {x["name"]: (rig, x["role"]) for rig, xs in sel.items() for x in xs}
    per = {}

    def rig_stats(rig):
        return per.setdefault(rig, {"cases": 0, "met": 0, "asked": 0, "parsed": 0, "flags": 0, "cost": {},
                                    "cache_reads": [], "failures": [], "flagged": [], "unresolved": 0})
    seen = set()
    selected_runs = []
    owners = {}
    for run in runs:
        out = Path(run) / 'out'
        selected = []
        names = {file.stem.removeprefix('failed_').removeprefix('noreply_')
                 for file in out.glob('*.json') if not file.stem.startswith('stale_')}
        for name in sorted(names & role.keys()):
            chosen = annotation_record(out, name + '.json')
            if chosen is None:
                continue
            file, record = chosen
            if name in owners:
                raise ValueError(f'Duplicate gate episode {name}: {owners[name]} and {file}; score the runs separately')
            owners[name] = file
            selected.append((name, file, record))
        selected_runs.append((Path(run), selected))
    for run, selected in selected_runs:
        # a setup whose run wrote no output at all still scores, every episode as "no output"
        rj = Path(run) / "run.json"
        ds = str(json.loads(rj.read_text()).get("dataset") or "") if rj.exists() else ""
        if ds.startswith("gate_") and ds[5:] in sel:
            rig_stats(ds[5:])
        out = Path(run) / "out"
        for name, f, d in selected:
            seen.add(name)
            rig, r = role[name]
            p = rig_stats(rig)
            p["asked"] += 1
            sec = duration_s(Path(d["episode_dir"]))
            if r.startswith("cost:"):
                c = p["cost"].setdefault(r[5:], [0.0, 0.0, 0])
                c[0] += episode_cost(d)
                c[1] += sec
                c[2] += 1
                p["cache_reads"].append(int((d.get("usage") or {}).get("cached_tokens") or 0))
            if not d.get("parse_ok") or reply_truncated(d):
                p["failures"].append(f"{name}: did not parse" if "labels" in d else f"{name}: reply cut off")
                if case_for(name, cases):
                    p["cases"] += 1
                continue
            p["parsed"] += 1
            labels = d.get("labels") or {}
            flags = [x["rule"] for x in label_consistency.check(labels, sec)]
            if flags:
                p["flags"] += 1
                p["flagged"].append(f"{name}: {', '.join(flags)}")
            case = case_for(name, cases)
            if case:
                met, why = judge(case, labels)
                p["unresolved"] += any(reason.startswith("needs review:") for reason in why)
                p["cases"] += 1
                p["met"] += met
                met or p["failures"].append(f"{name}: {'; '.join(why)}")
    # an episode of a scored rig with no output at all failed before the model answered
    for rig in list(per):
        for x in sel[rig]:
            if x["name"] not in seen:
                p = per[rig]
                p["asked"] += 1
                p["failures"].append(f"{x['name']}: no output (the harness failed before the model call)")
                if case_for(x["name"], cases):
                    p["cases"] += 1
    for p in per.values():
        billed = sum(c[0] for c in p["cost"].values())
        hours = sum(c[1] for c in p["cost"].values()) / 3600
        reads = [x for x in p["cache_reads"] if x > 0]
        prefix = min(reads) if reads else 0
        cold = billed + sum(max(0, x - prefix) for x in p["cache_reads"]) * (CACHE_WRITE - CACHE_READ)
        p["cost_sample"] = {"episodes": sum(c[2] for c in p["cost"].values()), "hours": round(hours, 3),
                            "billed_usd": round(billed, 2), "cold_usd": round(cold, 2),
                            "billed_per_hour": round(billed / hours, 1) if hours else None,
                            "cold_per_hour": round(cold / hours, 1) if hours else None,
                            "shared_prefix_tokens": prefix}
        p["cost_by_dataset"] = {ds: {"episodes": c[2], "minutes": round(c[1] / 60, 1), "billed_usd": round(c[0], 2),
                                     "billed_per_hour": round(c[0] / (c[1] / 3600), 1) if c[1] else None}
                                for ds, c in sorted(p["cost"].items())}
        del p["cost"], p["cache_reads"]
    return per


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m gate score", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+", help="run folders")
    ap.add_argument("--json", type=Path, default=None, help="also write the full result here")
    a = ap.parse_args()
    per = score(a.runs)
    for rig in RIGS:
        if rig not in per:
            continue
        p, c = per[rig], per[rig]["cost_sample"]
        print(f"{rig:9s} cases {p['met']}/{p['cases']}  parse {p['parsed']}/{p['asked']}  flags {p['flags']}  "
              f"needs review {p['unresolved']}  cost sample {c['episodes']} episodes, {c['hours'] * 60:.1f} min: ${c['billed_per_hour']}/h billed, "
              f"${c['cold_per_hour']}/h cold")
        for ds, x in p["cost_by_dataset"].items():
            print(f"    {ds:15s} {x['episodes']:3d} episodes {x['minutes']:6.1f} min  ${x['billed_per_hour']}/h billed")
        for x in p["failures"]:
            print(f"    FAIL {x}")
        for x in p["flagged"]:
            print(f"    FLAG {x}")
    if a.json:
        a.json.write_text(json.dumps(per, indent=1))
    return 0
