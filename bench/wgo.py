"""WGO-Bench: run the event passes on its 100 clips and score them the way the benchmark's publishers do.

    python -m bench.wgo extract  [--parquet P] [--macrodata-gold J]   # videos to data/raw/wgo, gold to data/bench/wgo
    python -m bench.wgo split                                          # configs/bench/wgo_split.json, once
    python -m bench.wgo run   --model M [--split dev|test|all] [--passes 3] [--cap USD] [--jobs N] [--name NAME]
    python -m bench.wgo score RUN [--gold videodb|macrodata] [--judge MODEL] [--langfuse]

Two references score the same predictions:

- videodb (VideoDB/WGO-Bench, revised 2026-09-23): 534 segments. One-to-one matching in order at IoU >= 0.5 (and
  0.75 strict); where eligible the first predicted start and last predicted end are aligned to the reference's;
  a text-only judge (VideoDB: GPT-5.5) decides whether a matched prediction's label describes the reference action
  given the task. Counts are pooled over episodes.
- macrodata (macrodata/WGO-Bench): 743 segments, IoU >= 0.75, the same snapping, judge Gemini 3.5 Flash.

Every score is reported per source (DROID, Galaxea, HomER) before the pooled figure, snapped and unsnapped. The
split is fixed by a hash of each id, stratified by source, 30 development and 70 test episodes: prompts and passes
are tuned on development only. WGO-Bench is CC BY-NC-SA 4.0: it is an evaluation input and never training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "wgo"
GOLD = ROOT / "data" / "bench" / "wgo"
RUNS = ROOT / "data" / "runs" / "wgo"
SPLIT = ROOT / "configs" / "bench" / "wgo_split.json"
CACHE = Path(os.environ.get("WGO_CACHE", Path.home() / ".cache" / "sx" / "wgo-bench"))
DEV_SHARE = 0.3

RIG = {"homer": "ego_head", "robointer": "teleop_arms", "galaxea": "teleop_arms"}
SOURCE_NAME = {"homer": "HomER", "robointer": "DROID", "galaxea": "Galaxea"}
JUDGES = {"videodb": "openai/gpt-5.5", "macrodata": "google/gemini-3.5-flash"}
IOU = {"videodb": 0.5, "macrodata": 0.75}


def source(eid: str) -> str:
    return eid.split("_", 1)[0]


# ── extract ──────────────────────────────────────────────────────────────────

def extract(parquet: Path, macrodata_gold: Path | None) -> None:
    import pyarrow.parquet as pq
    RAW.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    f = pq.ParquetFile(parquet)
    gold = {}
    for batch in f.iter_batches(batch_size=4, columns=["id", "video", "instruction", "segments"]):
        for r in batch.to_pylist():
            p = RAW / f"{r['id']}.mp4"
            if not p.exists():
                p.write_bytes(r["video"])
            gold[r["id"]] = {"instruction": r["instruction"], "segments": r["segments"]}
    (GOLD / "videodb.json").write_text(json.dumps(gold, indent=1))
    if macrodata_gold:
        md = json.loads(Path(macrodata_gold).read_text())
        rows = md if isinstance(md, list) else md.get("episodes", [])
        (GOLD / "macrodata.json").write_text(json.dumps(
            {r["id"]: {"instruction": r["instruction"], "segments": r["segments"]} for r in rows}, indent=1))
    print(f"{len(gold)} episodes, {sum(len(g['segments']) for g in gold.values())} segments")


def split() -> None:
    gold = json.loads((GOLD / "videodb.json").read_text())
    out = {"about": "WGO-Bench development/test split: within each source, ids ordered by sha256(id), the first 30% "
                    "are development. Fixed once; tune only on dev.", "dev": [], "test": []}
    by_src: dict[str, list[str]] = {}
    for eid in gold:
        by_src.setdefault(source(eid), []).append(eid)
    for ids in by_src.values():
        ids.sort(key=lambda e: hashlib.sha256(e.encode()).hexdigest())
        n = round(len(ids) * DEV_SHARE)
        out["dev"] += ids[:n]
        out["test"] += ids[n:]
    SPLIT.parent.mkdir(parents=True, exist_ok=True)
    if SPLIT.exists():
        raise SystemExit(f"{SPLIT} exists; the split is fixed once")
    SPLIT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"dev {len(out['dev'])}, test {len(out['test'])}")


def episodes(which: str) -> list[str]:
    s = json.loads(SPLIT.read_text())
    return s["dev"] + s["test"] if which == "all" else s[which]


# ── run ──────────────────────────────────────────────────────────────────────

def _commit() -> str:
    try:
        sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", "events", "bench"]) != 0
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(model: str, which: str, passes: int, cap: float, jobs: int, name: str | None, reasoning: str,
        convention: str, ids: list[str] | None, name_model: str | None = None, name_reasoning: str | None = None) -> Path:
    from events import footage as fo
    from events.model import Spend, SpendCapReached, openrouter
    from events.passes import Settings, annotate
    gold = json.loads((GOLD / "videodb.json").read_text())
    picks = ids or episodes(which)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    slug = name or f"{model.split('/')[-1]}_{reasoning}_p{passes}_{which}"
    out = RUNS / f"{stamp}_{slug}_{_commit()}"
    (out / "out").mkdir(parents=True, exist_ok=True)
    spend = Spend(cap)
    call = openrouter(model, reasoning=reasoning, spend=spend)
    name_call = openrouter(name_model, reasoning=name_reasoning or reasoning, spend=spend) if name_model else None
    meta = {"model": model, "reasoning": reasoning, "name_model": name_model, "name_reasoning": name_reasoning, "passes": passes, "split": which, "convention": convention,
            "commit": _commit(), "cap_usd": cap, "episodes": picks, "started": stamp}
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    lock = threading.Lock()

    def one(eid: str) -> tuple[str, str]:
        dest = out / "out" / f"{eid}.json"
        t0 = time.monotonic()
        try:
            fp = fo.decode([fo.Camera("camera", RAW / f"{eid}.mp4")])
            settings = Settings(convention=convention, rig=RIG[source(eid)], instruction=gold[eid]["instruction"],
                                passes=passes)
            res = annotate(fp, settings, call, name_call)
            doc = {"id": eid, "duration_s": fp.duration_s, **res.to_dict(), "seconds": round(time.monotonic() - t0, 1)}
            dest.write_text(json.dumps(doc, indent=1))
            return eid, f"ok {len(res.events)} events ${doc['cost_usd']:.4f}"
        except SpendCapReached as e:
            return eid, f"skipped: {e}"
        except Exception as e:  # one episode's failure is recorded, never the batch's
            dest.with_suffix(".error.txt").write_text(f"{type(e).__name__}: {e}")
            return eid, f"FAILED {type(e).__name__}: {str(e)[:200]}"

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for fut in as_completed([pool.submit(one, e) for e in picks]):
            eid, msg = fut.result()
            with lock:
                print(f"{eid}: {msg}  (spent ${spend.spent:.4f})", flush=True)
    meta["spent_usd"] = round(spend.spent, 6)
    (out / "run.json").write_text(json.dumps(meta, indent=1))
    print(out)
    return out


# ── score ────────────────────────────────────────────────────────────────────

def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match_in_order(pred: list[tuple[float, float]], gold: list[tuple[float, float]], thr: float) -> list[tuple[int, int]]:
    """One-to-one, order-preserving pairs with IoU >= thr: the most pairs, then the highest total IoU (dynamic
    programming over both sequences, so a prediction spanning many gold segments can match at most one)."""
    n, m = len(pred), len(gold)
    best = [[(0, 0.0)] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            cands = [best[i + 1][j], best[i][j + 1]]
            v = iou(pred[i], gold[j])
            if v >= thr:
                c, s = best[i + 1][j + 1]
                cands.append((c + 1, s + v))
            best[i][j] = max(cands)
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        v = iou(pred[i], gold[j])
        if v >= thr and best[i][j] == (best[i + 1][j + 1][0] + 1, best[i + 1][j + 1][1] + v):
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif best[i][j] == best[i + 1][j]:
            i += 1
        else:
            j += 1
    return pairs


def snapped(pred: list[tuple[float, float]], gold: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The first predicted start and last predicted end aligned to the reference's, when that keeps both valid."""
    if not pred or not gold:
        return pred
    p = list(pred)
    if gold[0][0] < p[0][1]:
        p[0] = (gold[0][0], p[0][1])
    if gold[-1][1] > p[-1][0]:
        p[-1] = (p[-1][0], gold[-1][1])
    return p


JUDGE_PROMPT = """\
You judge whether a predicted action label describes the same action as a reference label, for one segment of a
manipulation video. You do not see the video. Accept when the predicted label names the same action on the same
object (and the same source or destination when the reference names one); wording, extra detail and minor phrasing
differences do not matter, and timing is not your concern. Reject when the action, the object or the destination
differs, or when the prediction covers only a small part of what the reference describes.

Task: {task}
Reference: {ref}
Prediction: {pred}

Answer with JSON only: {{"match": true}} or {{"match": false}}"""


class Judge:
    def __init__(self, model: str, cache: Path):
        from events.model import openrouter
        self.model = model
        self.cache_path = cache
        self.cache = json.loads(cache.read_text()) if cache.exists() else {}
        self.call = openrouter(model, reasoning="low", max_tokens=2000)
        self.lock = threading.Lock()
        self.cost = 0.0

    def __call__(self, task: str, ref: str, pred: str) -> bool:
        from events.model import parse_json
        key = hashlib.sha256(json.dumps([self.model, task, ref, pred]).encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        for _ in range(3):
            reply = self.call([{"type": "text", "text": JUDGE_PROMPT.format(task=task, ref=ref, pred=pred)}], "judge")
            doc = parse_json(reply.text)
            if isinstance(doc, dict) and isinstance(doc.get("match"), bool):
                with self.lock:
                    self.cost += reply.cost or 0.0
                    self.cache[key] = doc["match"]
                    self.cache_path.write_text(json.dumps(self.cache))
                return doc["match"]
        raise RuntimeError(f"judge gave no verdict for {ref!r} / {pred!r}")


DURATION_BINS = ("<2s", "2-4s", "4-8s", "8-16s", ">=16s")


def duration_bin(d: float) -> str:
    return "<2s" if d < 2 else "2-4s" if d < 4 else "4-8s" if d < 8 else "8-16s" if d < 16 else ">=16s"


def _f1(tp: int, n_pred: int, n_gold: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def score(run_dir: Path, ref: str, judge_model: str | None, jobs: int) -> dict:
    gold = json.loads((GOLD / f"{ref}.json").read_text())
    meta = json.loads((run_dir / "run.json").read_text())
    judge = Judge(judge_model or JUDGES[ref], GOLD / "judge_cache.json")
    thr_main = IOU[ref]
    per_ep, tasks = {}, []
    for eid in meta["episodes"]:
        f = run_dir / "out" / f"{eid}.json"
        if not f.exists():
            per_ep[eid] = None
            continue
        doc = json.loads(f.read_text())
        pred = [(e["start_s"], e["end_s"]) for e in doc["events"]]
        labels = [e["label"] for e in doc["events"]]
        g = gold[eid]
        gs = [(s["start_sec"], s["end_sec"]) for s in g["segments"]]
        rec = {"n_pred": len(pred), "n_gold": len(gs), "cost_usd": doc.get("cost_usd", 0.0),
               "duration_s": doc.get("duration_s", 0.0)}
        for snap in (True, False):
            p = snapped(pred, gs) if snap else pred
            for thr in sorted({thr_main, 0.75}):
                key = f"{'snap' if snap else 'raw'}@{thr}"
                pairs = match_in_order(p, gs, thr)
                rec[key] = {"pairs": pairs}
                for i, j in pairs:
                    tasks.append((g["instruction"], g["segments"][j]["subtask"], labels[i]))
        rec["labels"], rec["gold_labels"], rec["instruction"] = labels, [s["subtask"] for s in g["segments"]], g["instruction"]
        per_ep[eid] = rec
    verdicts = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(judge, *t): t for t in set(tasks)}
        for fut in as_completed(futs):
            verdicts[futs[fut]] = fut.result()

    def tally(eids: list[str], key: str) -> dict:
        tp = matched = n_pred = n_gold = 0
        for eid in eids:
            rec = per_ep.get(eid)
            n_gold += len(gold[eid]["segments"])
            if rec is None:
                continue
            n_pred += rec["n_pred"]
            for i, j in rec[key]["pairs"]:
                matched += 1
                tp += verdicts[(rec["instruction"], rec["gold_labels"][j], rec["labels"][i])]
        out = {"segment": _f1(matched, n_pred, n_gold), "semantic": _f1(tp, n_pred, n_gold),
               "label_accuracy": round(tp / matched, 4) if matched else 0.0,
               "matched": matched, "accepted": tp, "predicted": n_pred, "gold": n_gold}
        if len(eids) > 1:
            per = [tally([e], key) for e in eids]
            out["macro"] = {"semantic_f1": round(sum(p["semantic"]["f1"] for p in per) / len(per), 4),
                            "segment_f1": round(sum(p["segment"]["f1"] for p in per) / len(per), 4)}
            bins: dict[str, list[int]] = {b: [0, 0] for b in DURATION_BINS}
            for eid in eids:
                rec = per_ep.get(eid)
                hit = {j for _i, j in rec[key]["pairs"]} if rec else set()
                for j, seg in enumerate(gold[eid]["segments"]):
                    b = duration_bin(seg["end_sec"] - seg["start_sec"])
                    bins[b][1] += 1
                    bins[b][0] += j in hit
            out["temporal_recall_by_duration"] = {b: {"recall": round(m / n, 4) if n else None, "matched": m, "gold": n}
                                                  for b, (m, n) in bins.items()}
        return out

    by_src: dict[str, list[str]] = {}
    for eid in meta["episodes"]:
        by_src.setdefault(SOURCE_NAME[source(eid)], []).append(eid)
    keys = [k for k in next((r for r in per_ep.values() if r), {}) if "@" in k]
    report = {"run": run_dir.name, "reference": ref, "judge": judge.model, "model": meta["model"],
              "passes": meta["passes"], "split": meta["split"], "episodes": len(meta["episodes"]),
              "missing": [e for e, r in per_ep.items() if r is None],
              "cost_usd": round(sum(r["cost_usd"] for r in per_ep.values() if r), 4),
              "footage_h": round(sum(r["duration_s"] for r in per_ep.values() if r) / 3600, 4),
              "judge_cost_usd": round(judge.cost, 4), "scores": {}}
    report["usd_per_footage_h"] = round(report["cost_usd"] / report["footage_h"], 3) if report["footage_h"] else None
    for key in keys:
        report["scores"][key] = {s: tally(ids, key) for s, ids in sorted(by_src.items())}
        report["scores"][key]["all"] = tally(meta["episodes"], key)
    report["per_episode"] = {e: (None if r is None else tally([e], f"snap@{thr_main}")) for e, r in per_ep.items()}
    (run_dir / f"score_{ref}.json").write_text(json.dumps(report, indent=1))
    return report


def summary(report: dict) -> str:
    lines = [f"{report['run']}  ref={report['reference']}  model={report['model']}  passes={report['passes']}  "
             f"split={report['split']}  cost=${report['cost_usd']}  (${report['usd_per_footage_h']}/footage h)"]
    for key, by in report["scores"].items():
        lines.append(f"  {key}:")
        for s, v in by.items():
            macro = v.get("macro") or {}
            lines.append(f"    {s:8s} semF1 {v['semantic']['f1']*100:5.1f}  segF1 {v['segment']['f1']*100:5.1f}  "
                         f"labelAcc {v['label_accuracy']*100:5.1f}  macro sem/seg {macro.get('semantic_f1', 0)*100:5.1f}/"
                         f"{macro.get('segment_f1', 0)*100:5.1f}  ({v['accepted']}/{v['matched']} of {v['gold']} gold, "
                         f"{v['predicted']} pred)")
        short = by["all"].get("temporal_recall_by_duration", {})
        if short:
            lines.append("    recall by gold duration: " + "  ".join(
                f"{b} {x['matched']}/{x['gold']}" for b, x in short.items()))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--parquet", type=Path, default=CACHE / "videodb-annotations.parquet")
    p.add_argument("--macrodata-gold", type=Path, default=None)
    sub.add_parser("split")
    p = sub.add_parser("run")
    p.add_argument("--model", required=True)
    p.add_argument("--reasoning", default="low")
    p.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--name-model", default=None, help="name the events with this model (default: --model)")
    p.add_argument("--name-reasoning", default=None)
    p.add_argument("--convention", default="wgo")
    p.add_argument("--cap", type=float, required=True, help="stop starting calls once this many dollars are spent")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--name", default=None)
    p = sub.add_parser("score")
    p.add_argument("run", type=Path)
    p.add_argument("--gold", default="videodb", choices=["videodb", "macrodata"])
    p.add_argument("--judge", default=None)
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--langfuse", action="store_true", help="also record the run as a Langfuse dataset run with scores")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.parquet, a.macrodata_gold)
    elif a.cmd == "split":
        split()
    elif a.cmd == "run":
        run(a.model, a.split, a.passes, a.cap, a.jobs, a.name, a.reasoning, a.convention, a.ids, a.name_model,
            a.name_reasoning)
    else:
        report = score(a.run, a.gold, a.judge, a.jobs)
        print(summary(report))
        if a.langfuse:
            from bench import langfuse
            langfuse.record(a.run, report, json.loads((GOLD / f"{a.gold}.json").read_text()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
