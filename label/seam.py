"""Ask whether the activity on the two sides of a request cut is one task, for long recordings labelled in parts that
share no window at their cuts (every recording labelled before windows existed).

A long recording is labelled in parts, and a task under way when one part ends is seen by the next part as a task
that starts at the cut. Where exactly one task of the part before the cut is still under way at its end and exactly
one task of the next part starts there (label/pieces.py seam_candidates), the two are usually one task cut in two, but
not always: in 40 such UMI pairs about 6 were a new task that began right at the cut. The stitch never decides this
from what the parts wrote (their notes, task text or object names). This check asks once per candidate cut, with the
frames on both sides of it, and label/pieces.py join_across_cuts joins the halves only on a verdict of same_task true.

    python -m review seam --label LABEL --footage FOOTAGE --out SEAMS.json [--dry-run] [--cap 5]

LABEL is a stitched result (a long recording's JSON in a job's final out/ folder), a folder of them, or an export's
JSON lines (.jsonl or .jsonl.gz, one recording per row with the result under "label", named by its take, episode_uid
or episode, ROW_NAMES). FOOTAGE is the recording's prepared
episode folder or a folder of its camera videos named by view (exo.mp4, left.mp4, right.mp4); for many recordings, a
folder that holds one such folder per recording, at the recording's name (the label file's name, or the export row's;
a teleop episode robot/day/episode is a nested folder). SEAMS.json is {"seams": {recording: {cut: verdict}}, ...}; stitch_run reads it as JOB/seams.json, and --apply
writes LABEL's recordings joined by it. --dry-run builds no images and calls nothing: it counts the requests and
prices them from the labels' own camera layout.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from label.atomic import write_atomic

WINDOW_S = 15.0         # seconds of footage shown on each side of the cut
FPS = 2.0               # frames per second shown
CELL_W = 256            # width of each camera's frame in the grid, in pixels
GRID_COLS = 4           # instants per grid image, as the labelling requests pack them (label/harness.py GRID_COLS)
QUALITY = 85
ANSWER_TOKENS = 600     # the answer is one short JSON object; reasoning gets its own allowance (label/policy.py)
SEAMS_VERSION = 1


# ---------------------------------------------------------------- which cuts to ask about

def part_tasks(labels: dict) -> list:
    """A stitched label's tasks as its parts gave them: a joined task is its halves again, and a task marked as
    continuing past a cut (label/pieces.py mark_cut_tasks) has its part's own outcome back."""
    out = []
    for task in labels.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        for half in task.get("joined_from") or [task]:
            half = copy.deepcopy(half)
            if half.get("outcome") == "continues_past_cut":
                cut = (half.get("cut_continuation") or {}).get("t_s")
                half["outcome"] = half.pop("outcome_in_part", None)
                half["completed_at_s"] = half.pop("completed_at_s_in_part", None)
                half.pop("cut_continuation", None)
                suffix = f" Still under way at the internal request cut at {cut:g}s." if cut is not None else ""
                note = str(half.get("note") or "")
                half["note"] = note[:-len(suffix)] if suffix and note.endswith(suffix) else note
            half.pop("cut_seams", None)
            out.append(half)
    return out


def candidates(result: dict) -> list[dict]:
    """[{"cut_s", "left", "right"}] for every cut of a stitched result where one task may continue across it."""
    from label import pieces
    stitched = result.get("stitched") or {}
    if stitched.get("windows"):
        return []           # parts that share a window at their cuts are joined by time (join_across_cuts)
    tasks = part_tasks((result.get("labels") or {}) if isinstance(result.get("labels"), dict) else {})
    return [{"cut_s": cut, "left": left, "right": right}
            for cut, left, right in pieces.seam_candidates(tasks, [float(c) for c in stitched.get("cuts_s") or []])]


# ---------------------------------------------------------------- the request

def instants(cut: float, end_s: float | None) -> list[float]:
    """The times shown: every 1 / FPS seconds from WINDOW_S before the cut to WINDOW_S after it, within the
    recording."""
    n = int(round(WINDOW_S * FPS))
    times = [round(cut + k / FPS, 3) for k in range(-n, n + 1)]
    return [t for t in times if t >= 0 and (end_s is None or t <= end_s)]


def _task_lines(task: dict) -> str:
    from label.episode import number
    objects = [o.get("name") if isinstance(o, dict) else o for o in task.get("objects") or []]
    span = f"{number(task.get('start_s')):.1f}s to {number(task.get('end_s')):.1f}s"
    lines = [f'  task: "{task.get("task") or ""}"', f"  span: {span} on the recording's clock"]
    if objects:
        lines.append("  objects: " + ", ".join(str(o) for o in objects))
    if task.get("success_predicate"):
        lines.append(f'  its goal as that part wrote it: "{task["success_predicate"]}"')
    return "\n".join(lines)


def prompt(candidate: dict, times: list[float], cam_labels: list[str]) -> str:
    cut = candidate["cut_s"]
    before = max((t for t in times if t < cut), default=None)
    after = min((t for t in times if t >= cut), default=None)
    where = (f"the cut lies between the columns at {before:.1f}s and {after:.1f}s" if before is not None
             and after is not None else f"the cut is at {cut:.1f}s")
    return (
        "SEAM CHECK. A long recording was labelled in parts, each part by its own request, and two parts meet at a "
        f"cut at {cut:.2f}s on the recording's clock. The cut falls where the request's length limit put it, not "
        "where the work changes.\n\n"
        f"The images show the cameras the labelling used ({', '.join(cam_labels)}, one row each), {FPS:g} frames per "
        f"second from {times[0]:.1f}s to {times[-1]:.1f}s; each column is captioned with its time on the recording's "
        f"clock, and {where}.\n\n"
        "The part before the cut ended with this task still under way:\n" + _task_lines(candidate["left"]) + "\n\n"
        "The part after the cut began with this task:\n" + _task_lines(candidate["right"]) + "\n\n"
        "Each description was written by a request that saw only its own side of the cut, so the two can name one "
        "task differently, and a new task can also begin right at the cut. If you answer true, the two become one "
        "task: the goal the part before the cut wrote, judged by the outcome the part after the cut gave. So the "
        "question is whether the work on both sides is aimed at the same end state, not whether the hands keep "
        "moving: the hands nearly always keep moving across a cut.\n"
        "- true: the work after the cut carries on toward the same end state as the work before it (the same build, "
        "fill, sort or arrangement, perhaps described more narrowly as what was left).\n"
        "- false: the end states differ, even with no pause between them (building, then taking it apart; laying "
        "pieces out, then assembling something new from them; one object finished and set aside, then another "
        "begun), or the frames show one task ending and another starting near the cut.\n"
        "- unclear: the frames do not show which.\n\n"
        "Answer with one JSON object and nothing else:\n"
        '{"same_task": true | false | "unclear", "evidence": "<the end state the work aims at on each side of the '
        'cut, as the frames show it, in one or two sentences>"}')


def settings(result: dict, model: str | None = None, reasoning: str | None = None) -> dict:
    """The model and reasoning the labels were made with, unless asked otherwise, and the output limit."""
    from label import policy
    chosen = result.get("annotation_policy") or {}
    model = model or result.get("model") or chosen.get("model") or policy.ASTRA
    reasoning = reasoning or (result.get("reasoning_effort") if model == result.get("model") else None) \
        or chosen.get("reasoning") or "medium"
    return {"model": model, "reasoning": reasoning, "max_tokens": policy.output_limit(ANSWER_TOKENS, reasoning)}


def cameras(result: dict) -> tuple[list[str], list[str], float]:
    """(views, their row labels, cell height / width) as the labels' own requests showed them."""
    cfg = result.get("config") or {}
    views = list(cfg.get("views") or [])
    labels = list(cfg.get("cam_labels") or views)
    cell = cfg.get("cell") or [1, 1]
    aspect = float(cell[1]) / float(cell[0]) if len(cell) == 2 and float(cell[0]) > 0 else 1.0
    return views, labels if len(labels) == len(views) else views, aspect


def grid_size(n_cols: int, n_views: int, aspect: float) -> tuple[int, int]:
    """A grid image's pixel size (label/frames.py compose_grid) for n_cols instants of n_views cameras."""
    from label.episode import GRID_GUTTER, GRID_HEADER
    cell_h = int(round(CELL_W * aspect / 2)) * 2
    return GRID_GUTTER + n_cols * (CELL_W + 4), GRID_HEADER + n_views * (cell_h + 4)


def estimate(result: dict, candidate: dict, model: str, max_tokens: int) -> dict:
    """What one request holds and costs, from the labels' camera layout, without decoding a frame: its images, its
    input tokens (the text at four bytes a token, each image by label/harness.py estimate_image_tokens) and its cost,
    from the input alone plus a 200-token answer, up to the input plus the whole output allowance."""
    from label.harness import OPENAI_PRICES, LONG_PROMPT_TOKENS, estimate_image_tokens
    views, labels, aspect = cameras(result)
    times = instants(candidate["cut_s"], _recording_end(result))
    grids = [len(times[i:i + GRID_COLS]) for i in range(0, len(times), GRID_COLS)]
    image_tokens = sum(estimate_image_tokens(*grid_size(n, len(views), aspect)) for n in grids)
    text = prompt(candidate, times, labels)
    text_tokens = (len(text.encode()) + sum(len(_caption(times[i:i + GRID_COLS], labels).encode())
                                            for i in range(0, len(times), GRID_COLS))) / 4
    tiers = OPENAI_PRICES.get(model.split("/", 1)[-1]) if model.startswith("openai/") else None
    tokens = text_tokens + image_tokens
    out = {"instants": len(times), "views": views, "images": len(grids), "input_tokens": round(tokens)}
    if tiers is not None:
        p_in, _, p_out = tiers[tokens >= LONG_PROMPT_TOKENS]
        out.update(low_usd=tokens * p_in + 200 * p_out, high_usd=tokens * p_in + max_tokens * p_out)
    return out


def _caption(block: list[float], labels: list[str]) -> str:
    return f"=== grid t={block[0]:.2f}-{block[-1]:.2f}s | rows {', '.join(labels)} | columns are time left to right ==="


def _recording_end(result: dict) -> float | None:
    pieces = (result.get("config") or {}).get("pieces") or []
    ends = [float(p["t1_s"]) for p in pieces if isinstance(p, dict) and p.get("t1_s") is not None]
    return max(ends) if ends else None


# ---------------------------------------------------------------- frames

def _prepared_frames(ep_dir: Path, views: list[str], times: list[float]) -> dict:
    """{view: {time: image}} from a prepared episode: each time at the anchor camera's nearest frame, every camera at
    the frame the episode pairs with it (label/episode.py _decode_view)."""
    import numpy as np
    from label import episode as me
    ep = me.load(ep_dir)
    n = int(ep["sources"][me.anchor(ep)]["n_frames"])
    clock = np.array([me.frame_time(ep, k) for k in range(n)])
    ks = [int(np.argmin(np.abs(clock - t))) for t in times]
    out = {}
    for v in views:
        if v not in ep["sources"]:
            continue
        got = me._decode_view(ep, v, sorted(set(ks)))
        out[v] = {t: got[k] for t, k in zip(times, ks) if k in got}
    return out


def _video_frames(path: Path, times: list[float], start_s: float = 0.0) -> dict:
    """{time: image} from one camera's video: the last frame shown at or before each time. start_s is when the video's
    first frame falls on the recording's clock (an export's sync start_s), so recording time t is video time
    t - start_s."""
    import av
    from label import frames as mf
    from prepare import display
    geom = display.geometry(str(path))
    out = {}
    times = sorted(times)
    if not times:
        return out
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        first = times[0] - start_s - 1.0
        container.seek(int(first / stream.time_base) if first > 0 else 0, stream=stream, backward=True)
        decoded = container.decode(stream)
        shown, ahead = None, None       # the last frame at or before the time asked, and the next one decoded
        for t in times:
            at = t - start_s
            while True:
                if ahead is None:
                    ahead = next(decoded, None)
                    if ahead is None:
                        break
                    if ahead.time is None:
                        ahead = None
                        continue
                if ahead.time > at + 1e-3:
                    break
                shown, ahead = ahead, None
            if shown is not None:
                out[t] = mf.upright(shown, geom)
    return out


def frames_for(footage: Path, views: list[str], times: list[float], sync: dict | None = None) -> dict:
    """{view: {time: image}} from a prepared episode folder or a folder of camera videos named by view."""
    footage = Path(footage)
    if (footage / "context.json").is_file() and (footage / "sources.json").is_file():
        return _prepared_frames(footage, views, times)
    out = {}
    for v in views:
        files = sorted(footage.glob(f"{v}.mp4")) or sorted(p for p in footage.glob("*.mp4") if p.stem.endswith(v))
        if files:
            out[v] = _video_frames(files[0], times, float((sync or {}).get(v) or 0.0))
    return out


def build_request(result: dict, candidate: dict, footage: Path, sync: dict | None = None) -> tuple[list, dict]:
    """(the request's content parts, what it holds): the prompt, then the instants packed into grids as the labelling
    requests pack them (label/frames.py build_content), rows the cameras, columns the time."""
    from label import frames as mf
    from label.episode import GRID_GUTTER, GRID_HEADER
    from label.harness import DETAIL
    views, labels, _ = cameras(result)
    times = instants(candidate["cut_s"], _recording_end(result))
    images = frames_for(footage, views, times, sync)
    shown = [(v, lab) for v, lab in zip(views, labels) if images.get(v)]
    if not shown:
        raise RuntimeError(f"no camera of {', '.join(views)} found in {footage}")
    steps = [(t, [(lab, mf.to_jpeg(images[v][t], CELL_W, QUALITY)) for v, lab in shown if t in images[v]])
             for t in times]
    steps = [step for step in steps if step[1]]
    content, n_grids, n_bytes = mf.build_content(prompt(candidate, [t for t, _ in steps], [lab for _, lab in shown]),
                                                 "", steps, [lab for _, lab in shown], GRID_COLS, DETAIL, QUALITY,
                                                 gutter=GRID_GUTTER, header=GRID_HEADER)
    return content, {"instants": len(steps), "views": [v for v, _ in shown], "window_s": [steps[0][0], steps[-1][0]],
                     "images": n_grids, "image_bytes": n_bytes, "cell_w": CELL_W, "fps": FPS}


# ---------------------------------------------------------------- the answer

def read_answer(text: str) -> dict:
    """{"same_task", "evidence"} from the reply, or "unclear" with why when the reply is not that strict JSON, so a
    reply that cannot be read never joins two tasks."""
    try:
        fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text or "", re.S)
        answer = json.loads(fenced.group(1) if fenced else text)
        same = answer.get("same_task") if isinstance(answer, dict) else None
        if not (type(same) is bool or same == "unclear") or not isinstance(answer.get("evidence"), str):
            raise ValueError('the reply is not {"same_task": true | false | "unclear", "evidence": text}')
        return {"same_task": answer["same_task"], "evidence": answer["evidence"]}
    except (ValueError, TypeError) as error:
        return {"same_task": "unclear", "evidence": "", "reply_problem": f"{type(error).__name__}: {error}"[:300],
                "reply": str(text)[:1000]}


def ask(content: list, chosen: dict, key: str, timeout: int) -> tuple[dict, dict]:
    """(verdict, the call's record) for one request, sent once (label/harness.py call_model_once: a call that may have
    been billed is never sent again; one that provably cost nothing is retried there)."""
    from label import harness
    from label.errors import DispatchError
    try:
        resp = harness.call_model_once(content, chosen["model"], chosen["reasoning"], key, chosen["max_tokens"],
                                       timeout)
    except DispatchError as error:
        return ({"same_task": "unclear", "evidence": "", "reply_problem": f"the request failed: {error}"[:300]},
                {"dispatch": "failed", "verified_unpaid": bool(getattr(error, "verified_unpaid", False))})
    usage = resp.get("usage") or {}
    cost, source = harness._verified_final_cost(usage, chosen["model"])
    choice = (resp.get("choices") or [{}])[0]
    text = ((choice.get("message") or {}).get("content")) or ""
    record = {"dispatch": "received", "usage": usage, "cost_usd": cost, "cost_source": source,
              "generation_id": resp.get("id"), "provider": resp.get("provider"),
              "finish_reason": choice.get("finish_reason")}
    return read_answer(text), record


# ---------------------------------------------------------------- inputs and outputs

# the field that names a recording in an export's rows: a UMI take, a LeRobot episode_uid, a teleop episode
# (robot/day/episode). A verdict is stored under the name, so a row without one is refused, never asked
ROW_NAMES = ("take", "episode_uid", "episode")


def read_labels(path: Path):
    """[(recording name, stitched result, camera sync or None)] from one result, a folder of them, or export rows."""
    path = Path(path)
    if path.is_dir():
        for p in sorted(path.glob("*.json")):
            if not p.name.startswith(("failed_", "noreply_")):
                yield from read_labels(p)
        return
    if path.name.endswith((".jsonl", ".jsonl.gz")):
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt") as stream:
            for n, line in enumerate(stream, 1):
                if line.strip():
                    row = json.loads(line)
                    name = next((row[k] for k in ROW_NAMES if isinstance(row.get(k), str) and row[k].strip()), None)
                    if name is None:
                        raise SystemExit(f"{path} line {n}: the row has no name ({', '.join(ROW_NAMES)})")
                    yield name, row.get("label") or {}, (row.get("sync") or {}).get("start_s")
        return
    yield path.stem, json.loads(path.read_text()), None


def _footage_of(footage: Path, name: str) -> Path:
    footage = Path(footage)
    return footage / name if (footage / name).is_dir() else footage


def rejoin(result: dict, seams: dict) -> dict:
    """A stitched result with its tasks joined again by seam verdicts ({cut: verdict}): its parts' tasks recovered
    (part_tasks), joined (join_across_cuts) and marked (mark_cut_tasks) as the stitch does, its scene objects joined
    again by the new tasks, and a completion derived from its tasks (a tasks-format recording with no instruction and
    no missing part, not head camera) derived again."""
    from board.label_projection import recording_completion
    from label import pieces
    out = copy.deepcopy(result)
    labels, stitched = out.get("labels") or {}, out.get("stitched") or {}
    if stitched.get("windows"):
        raise ValueError("parts that share a window at their cuts are joined by time when they are stitched")
    cuts = [float(c) for c in stitched.get("cuts_s") or []]
    labels["tasks"] = pieces.mark_cut_tasks(pieces.join_across_cuts(part_tasks(labels), cuts, seams), cuts, labels)
    scene = labels.get("scene") or {}
    if isinstance(scene.get("objects"), list):
        flat = []
        for obj in scene["objects"]:
            for record in [{k: v for k, v in obj.items() if k != "continues"}, *(obj.get("continues") or [])]:
                flat.append(((record.get("source_piece") or {}).get("index"), record))
        flat.sort(key=lambda item: item[0] if isinstance(item[0], int) else 0)
        scene["objects"] = pieces.join_scene_objects(flat, labels["tasks"])
    completion = labels.get("completion") or {}
    rig = (out.get("annotation_policy") or {}).get("rig")
    if out.get("label_schema") == "tasks" and completion.get("basis") in ("one_task", "tasks") \
            and rig != "ego_head" and not stitched.get("missing"):
        labels["completion"] = recording_completion(labels["tasks"])
    stitched["seams"] = copy.deepcopy(seams)
    return out


def main(argv: list[str] | None = None) -> int:
    from label import harness
    from label.pieces import at_cut
    ap = argparse.ArgumentParser(prog="python -m review seam", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", type=Path, required=True, help="a stitched result, a folder of them, or export rows")
    ap.add_argument("--footage", type=Path, default=None,
                    help="the recording's prepared folder or camera videos, or a folder of one such folder per "
                         "recording")
    ap.add_argument("--out", type=Path, required=True, help="the seams file to write (JOB/seams.json for stitch_run)")
    ap.add_argument("--dry-run", action="store_true", help="count and price the requests; build and send nothing")
    ap.add_argument("--cap", type=float, default=5.0, help="most the checks may spend, in USD (default 5)")
    ap.add_argument("--model", default=None, help="default: the model the labels were made with")
    ap.add_argument("--reasoning", default=None, help="default: the reasoning effort the labels were made with")
    ap.add_argument("--concurrency", type=int, default=8, help="requests in flight (default 8)")
    ap.add_argument("--timeout", type=int, default=600, help="seconds a request's connection may carry no bytes")
    ap.add_argument("--apply", type=Path, default=None, metavar="DIR",
                    help="also write each recording of --label joined by the seams into DIR")
    a = ap.parse_args(argv)
    if not math.isfinite(a.cap) or a.cap <= 0:
        ap.error("--cap must be finite and positive")
    saved = json.loads(a.out.read_text()) if a.out.is_file() else {}
    seams = saved.get("seams") or {}
    work, recordings, totals = [], 0, {"low_usd": 0.0, "high_usd": 0.0, "images": 0, "input_tokens": 0}
    unpriced = set()
    named = set()
    for name, result, sync in read_labels(a.label):
        if name in named:
            # two recordings under one name would share one recording's verdicts
            ap.error(f"{name} names two recordings in --label")
        named.add(name)
        if not (result.get("stitched") or {}).get("cuts_s"):
            continue
        recordings += 1
        chosen = settings(result, a.model, a.reasoning)
        for candidate in candidates(result):
            if at_cut(seams.get(name), candidate["cut_s"]) is not None:
                continue        # asked before: never paid for twice
            guess = estimate(result, candidate, chosen["model"], chosen["max_tokens"])
            for key in ("images", "input_tokens"):
                totals[key] += guess[key]
            if "low_usd" in guess:
                totals["low_usd"] += guess["low_usd"]
                totals["high_usd"] += guess["high_usd"]
            else:
                unpriced.add(chosen["model"])
            work.append((name, result, sync, candidate, chosen))
    models = sorted({f"{c['model']} at {c['reasoning']} reasoning" for *_, c in work})
    print(f"{recordings} recordings cut into parts; seam checks to ask: {len(work)}"
          + (f" ({', '.join(models)})" if models else ""), flush=True)
    print(f"about {totals['images']} images and {totals['input_tokens']} input tokens; estimated "
          f"${totals['low_usd']:.2f} to ${totals['high_usd']:.2f}"
          + (f" (no list price for {', '.join(sorted(unpriced))})" if unpriced else ""), flush=True)
    if a.dry_run:
        return 0
    if a.footage is None:
        ap.error("--footage is required unless --dry-run")
    keys = harness.get_keys()
    if not keys and work:
        raise SystemExit("no model key: set OPENROUTER_API_KEYS or OPENAI_API_KEY")
    lock, spent = threading.Lock(), {"usd": float(saved.get("cost_usd") or 0.0)}

    def save():
        write_atomic(a.out, {"version": SEAMS_VERSION, "seams": seams, "cost_usd": spent["usd"],
                             "window_s": WINDOW_S, "fps": FPS, "cell_w": CELL_W})

    def one(job):
        name, result, sync, candidate, chosen = job
        try:
            content, holds = build_request(result, candidate, _footage_of(a.footage, name), sync)
        except (OSError, RuntimeError, ValueError) as error:
            # footage that cannot be read gives no verdict, so the cut stays apart and a later run can ask again
            print(f"{name} at {candidate['cut_s']:g}s: no request, {type(error).__name__}: {error}"[:400], flush=True)
            return None
        bound = harness.final_cost_bound(content, chosen["max_tokens"], model=chosen["model"])
        with lock:
            if spent["usd"] + bound > a.cap:
                return False
            spent["usd"] += bound        # held until the reply says what it cost
        verdict, call = ask(content, chosen, keys[0], a.timeout)
        cost = call.get("cost_usd")
        with lock:
            # a call whose cost is not known stays held at its bound, as the labelling harness keeps a reservation
            spent["usd"] += (float(cost) - bound) if cost is not None else (
                -bound if call.get("verified_unpaid") else 0.0)
            if call.get("verified_unpaid"):
                save()
                return None         # it cost nothing, so a later run asks again
            # a verdict is kept once paid for (or possibly paid for), so a later run never sends it again
            seams.setdefault(name, {})[str(round(candidate["cut_s"], 3))] = {
                **verdict, "request": {"model": chosen["model"], "reasoning": chosen["reasoning"],
                                       "max_tokens": chosen["max_tokens"], "reserved_usd": bound, **holds, **call}}
            save()
        return True

    with ThreadPoolExecutor(max(1, a.concurrency)) as pool:
        asked = list(pool.map(one, work))
    if False in asked:
        print(f"stopped at the ${a.cap:g} cap: {asked.count(False)} checks not asked", flush=True)
    if None in asked:
        print(f"{asked.count(None)} checks gave no verdict (no footage, or a request that failed unpaid)", flush=True)
    save()
    print(f"{asked.count(True)} checks asked; ${spent['usd']:.2f} spent in all; {a.out}", flush=True)
    if a.apply is not None:
        a.apply.mkdir(parents=True, exist_ok=True)
        for name, result, _ in read_labels(a.label):
            if (result.get("stitched") or {}).get("cuts_s"):
                (a.apply / f"{name}.json").parent.mkdir(parents=True, exist_ok=True)
                write_atomic(a.apply / f"{name}.json", rejoin(result, seams.get(name) or {}))
    return 0 if all(x is True for x in asked) else 1
