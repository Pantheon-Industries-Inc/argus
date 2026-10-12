"""Three passes over one episode: find the events, settle their boundaries, then name them.

After Midcentury's "Optimizing Argus" (2026-10-05), which separates what Argus decides in one call (what happened,
when, and what to call it):

1. find   every event and its coarse times, from 4 fps contact sheets of the whole recording;
2. refine each boundary, from the whole recording at 1 fps plus 8 fps close-ups within +-1.5 s of it; only times
          move, and by at most 1.5 s;
3. name   each event, its times fixed, from up to five instants each of the previous, this and next event, up to
          eight events per request.

Everything reads one decoded footage (events/footage.py), so the passes cost one decode. The model is called through
a Caller (events/model.py), so the same passes run against OpenRouter on a laptop and through the platform's metered
effect door in production. Every reply is parsed strictly: a reply that is not the asked-for JSON fails the episode,
except in pass 2 and 3, where an unusable answer keeps the previous pass's value for that event and is recorded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from events import prompts
from events.footage import Footage
from events.model import Caller, Reply, parse_json
from events.sheets import image_part, sheets

FIND_FPS = 4
FIND_COLS, FIND_ROWS = 4, 3
FIND_CELL_W = 448
MAX_FIND_IMAGES = 120  # past this the find pass samples more sparsely rather than refusing a long recording
REFINE_WINDOW_S = 1.5
REFINE_COLS = 5
REFINE_CELL_W = 320
OVERVIEW_FPS = 1
OVERVIEW_COLS, OVERVIEW_ROWS = 8, 6
OVERVIEW_CELL_W = 224
NAME_PER_ROLE = 5
NAME_BATCH = 8
NAME_ATLAS_MAX_PX = 2400 * 2400  # an atlas past this splits into parts; its pictures are never shrunk
MIN_EVENT_S = 0.1


# A request's images, as bytes before base64: OpenRouter refuses more than 30 MB of downloaded image content, and
# base64 adds a third, so the passes keep each request under this and step their cells down to fit.
IMAGE_BUDGET_BYTES = 18 * 1024 * 1024
CELL_STEPS = (448, 384, 320, 256, 224, 192, 160)


def fitting(build, cell_w: int, budget: int = IMAGE_BUDGET_BYTES):
    """build(cell_w) -> [(caption, bytes)], at the widest cell width at or under cell_w whose images fit the budget."""
    for w in [cell_w] + [c for c in CELL_STEPS if c < cell_w]:
        images = build(w)
        if sum(len(b) for _c, b in images) <= budget:
            return images
    raise PassError(f"the request's images exceed {budget} bytes even at {CELL_STEPS[-1]} px cells")


class PassError(RuntimeError):
    """A pass could not produce a usable answer."""


@dataclass
class Event:
    start_s: float
    end_s: float
    label: str
    skill: str | None = None
    parts: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    destination: str | None = None


@dataclass
class Settings:
    convention: str = "wgo"
    rig: str = "ego_head"
    instruction: str | None = None
    skills: list[str] | None = None
    parts: list[str] | None = None
    passes: int = 3
    split: bool = False  # after finding, look inside each longer event for several (first-person video)
    summary: bool = False  # a fourth call judging the whole episode: task, outcome, scene and issues


@dataclass
class Result:
    events: list[Event]
    calls: list[dict]
    notes: list[str]
    summary: dict | None = None

    def to_dict(self) -> dict:
        return {"events": [asdict(e) for e in self.events], "summary": self.summary, "calls": self.calls,
                "notes": self.notes,
                "cost_usd": round(sum(c.get("cost_usd") or 0.0 for c in self.calls), 6)}


def annotate(footage: Footage, settings: Settings, call: Caller, name_call: Caller | None = None) -> Result:
    """name_call, when given, names the events with another (usually stronger) model: finding and timing events
    take most of the images, naming takes few, so a stronger namer costs little."""
    calls: list[dict] = []
    notes: list[str] = []
    events = find(footage, settings, call, calls)
    if settings.split and events:
        events = split(footage, settings, call, calls, notes, events)
    if settings.passes >= 2 and events:
        events = refine(footage, settings, call, calls, notes, events)
    if settings.passes >= 3 and events:
        events = name(footage, settings, name_call or call, calls, notes, events)
        if _per_part(settings):
            events = clean(events, footage.duration_s, per_part=True)
    summary = summarize(footage, settings, name_call or call, calls, events) if settings.summary else None
    return Result(events, calls, notes, summary)


OUTCOMES = ("success", "partial", "failure", "unclear")
SEVERITIES = ("low", "medium", "high")
ISSUE_KINDS = ("operator_mistake", "data_issue")


def summarize(footage: Footage, settings: Settings, call: Caller, calls: list, events: list[Event]) -> dict:
    """The episode as a whole: what task was done, whether it succeeded, the scene and what is wrong with it.
    Strict: a reply missing the outcome fails the episode rather than inventing one."""
    from events.sheets import sheet
    overview = sheets(footage, footage.times(OVERVIEW_FPS), OVERVIEW_COLS, OVERVIEW_ROWS, cell_w=OVERVIEW_CELL_W)
    ends = [("=== first instant ===", sheet([(0.0, footage.at(0.0))], footage.cameras, 1, png=True)),
            ("=== last instant ===", sheet([(footage.duration_s, footage.at(footage.duration_s))], footage.cameras, 1,
                                          png=True))]
    listing = "\n".join(f"- {e.start_s:.2f}-{e.end_s:.2f} s: {e.label}" for e in events) or "- none found"
    text = prompts.SUMMARY.format(overview_step=1 / OVERVIEW_FPS, events=listing, **_header(settings, footage))
    reply = call(_content(text, overview + ends), "summary")
    _record(calls, "summary", reply, len(overview) + len(ends))
    doc = parse_json(reply.text)
    if not isinstance(doc, dict) or doc.get("outcome") not in OUTCOMES or not str(doc.get("task_summary") or "").strip():
        raise PassError("summary: the reply has no task and outcome")
    issues = []
    for r in doc.get("issues") or []:
        if not isinstance(r, dict) or r.get("kind") not in ISSUE_KINDS or not str(r.get("note") or "").strip():
            continue
        t = r.get("t_s")
        issues.append({"kind": r["kind"], "category": str(r.get("category") or r["kind"]).strip(),
                       "severity": r.get("severity") if r.get("severity") in SEVERITIES else "medium",
                       "t_s": float(t) if isinstance(t, (int, float)) else None, "note": str(r["note"]).strip()})
    scene = doc.get("scene") if isinstance(doc.get("scene"), dict) else {}
    goal = doc.get("goal_reached_at_s")
    return {"task_summary": str(doc["task_summary"]).strip(),
            "matches_given": doc.get("matches_given") if isinstance(doc.get("matches_given"), bool) else None,
            "outcome": doc["outcome"], "goal_reached_at_s": float(goal) if isinstance(goal, (int, float)) else None,
            "reason": str(doc.get("reason") or "").strip(),
            "scene": {"setting": str(scene.get("setting") or "").strip(),
                      "objects": [{"name": str(o.get("name")), "location": str(o.get("location") or "")}
                                  for o in scene.get("objects") or [] if isinstance(o, dict) and o.get("name")]},
            "issues": issues}


def _header(settings: Settings, footage: Footage) -> dict:
    return {"cameras": prompts.cameras_line(footage.cameras, settings.rig),
            "instruction": prompts.instruction_line(settings.instruction),
            "convention": prompts.convention_text(settings.convention, settings.rig)}


def _record(calls: list, purpose: str, reply: Reply, images: int) -> None:
    calls.append({"pass": purpose, "images": images, "cost_usd": reply.cost, "seconds": round(reply.seconds, 2),
                  "usage": reply.usage, "served": reply.served})


def _content(text: str, images: list[tuple[str, bytes]]) -> list:
    content: list = [{"type": "text", "text": text}]
    for caption, jpeg in images:
        content.append({"type": "text", "text": caption})
        content.append(image_part(jpeg))
    return content


def clean(events: list[Event], duration: float, per_part: bool = False) -> list[Event]:
    """Order by start, clip to the recording, drop empty events and resolve overlaps at the midpoint of the overlap.

    per_part: events may overlap unless they share a body part (the sx_label convention, where the two hands may
    act at once), so only events sharing a part are split, and an event whose parts are not known yet is left as
    found. Otherwise every overlap is split (one actor at a time, as WGO-Bench counts events)."""
    out: list[Event] = []
    for e in sorted(events, key=lambda e: (e.start_s, e.end_s)):
        e.start_s, e.end_s = max(0.0, min(e.start_s, duration)), max(0.0, min(e.end_s, duration))
        if e.end_s - e.start_s < MIN_EVENT_S or not e.label.strip():
            continue
        if per_part:
            last = {}
            for k in out:
                for p in k.parts:
                    last[p] = k
            rivals = list({id(k): k for p in e.parts if (k := last.get(p)) is not None}.values())
        else:
            rivals = out[-1:]
        start = e.start_s
        for prev in rivals:
            if start < prev.end_s:
                prev.end_s = max(prev.start_s + MIN_EVENT_S, (start + prev.end_s) / 2)
                e.start_s = max(e.start_s, prev.end_s)
        if e.end_s - e.start_s < MIN_EVENT_S:
            continue
        out.append(e)
    return sorted(out, key=lambda e: (e.start_s, e.end_s))


def _per_part(settings: Settings) -> bool:
    """Whether the convention lets two parts act at once (sx_label: the two hands)."""
    return settings.convention == "sx_label"


def find(footage: Footage, settings: Settings, call: Caller, calls: list) -> list[Event]:
    fps = FIND_FPS
    while len(footage.times(fps)) > MAX_FIND_IMAGES * FIND_COLS * FIND_ROWS and fps > 1:
        fps //= 2
    times = footage.times(fps)
    images = fitting(lambda w: sheets(footage, times, FIND_COLS, FIND_ROWS, cell_w=w), FIND_CELL_W)
    text = prompts.FIND.format(step=1 / fps, duration=footage.duration_s, **_header(settings, footage))
    rows = None
    for _attempt in range(2):  # one retry: a find pass that does not parse leaves the episode with nothing
        reply = call(_content(text, images), "find")
        _record(calls, "find", reply, len(images))
        doc = parse_json(reply.text)
        rows = doc.get("events") if isinstance(doc, dict) else None
        if isinstance(rows, list):
            break
        calls[-1]["unparsed"] = reply.text[-2000:]
    if not isinstance(rows, list):
        raise PassError("find: the reply has no events list")
    events = []
    for r in rows:
        try:
            events.append(Event(float(r["start_s"]), float(r["end_s"]), str(r["label"]).strip()))
        except (KeyError, TypeError, ValueError):
            continue
    return clean(events, footage.duration_s, per_part=_per_part(settings))


SPLIT_MIN_S = 3.0
SPLIT_BATCH = 6
SPLIT_FPS = 4
SPLIT_CELL_W = 320


def split(footage: Footage, settings: Settings, call: Caller, calls: list, notes: list,
          events: list[Event]) -> list[Event]:
    """Every event of at least SPLIT_MIN_S is shown again at 4 fps and may come back as several. Only times inside
    the event are accepted; an answer outside it, or none, keeps the event as found."""
    long_ = [i for i, e in enumerate(events) if e.end_s - e.start_s >= SPLIT_MIN_S]
    out: dict[int, list[Event]] = {}
    for b0 in range(0, len(long_), SPLIT_BATCH):
        idx = long_[b0:b0 + SPLIT_BATCH]
        lines = [f"- e{i}: {events[i].start_s:.2f}-{events[i].end_s:.2f} s, found as \"{events[i].label}\"" for i in idx]

        def build(w: int, idx=idx) -> list:
            imgs = []
            for i in idx:
                ts = footage.times(SPLIT_FPS, events[i].start_s, events[i].end_s)
                for caption, jpeg in sheets(footage, ts, 4, 4, cell_w=w):
                    imgs.append((f"=== e{i} ===\n" + caption, jpeg))
            return imgs
        images = fitting(build, SPLIT_CELL_W)
        text = prompts.SPLIT.format(events="\n".join(lines), **_header(settings, footage))
        reply = call(_content(text, images), "split")
        _record(calls, "split", reply, len(images))
        doc = parse_json(reply.text)
        for r in (doc.get("events") if isinstance(doc, dict) else None) or []:
            if not isinstance(r, dict) or not isinstance(r.get("id"), str) or not r["id"][1:].isdigit():
                continue
            i = int(r["id"][1:])
            if i not in idx:
                continue
            e = events[i]
            parts = []
            for p in r.get("parts") or []:
                try:
                    a, b, lab = float(p["start_s"]), float(p["end_s"]), str(p["label"]).strip()
                except (KeyError, TypeError, ValueError):
                    continue
                if e.start_s - 0.5 <= a < b <= e.end_s + 0.5 and lab:
                    parts.append(Event(max(a, e.start_s), min(b, e.end_s), lab))
            if len(parts) > 1:
                out[i] = parts
                notes.append(f"split e{i} into {len(parts)}")
    merged = []
    for i, e in enumerate(events):
        merged.extend(out.get(i, [e]))
    return clean(merged, footage.duration_s, per_part=_per_part(settings))


def refine(footage: Footage, settings: Settings, call: Caller, calls: list, notes: list,
           events: list[Event]) -> list[Event]:
    overview = sheets(footage, footage.times(OVERVIEW_FPS), OVERVIEW_COLS, OVERVIEW_ROWS, cell_w=OVERVIEW_CELL_W)
    for b0 in range(0, len(events), NAME_BATCH):
        batch = list(enumerate(events[b0:b0 + NAME_BATCH], start=b0))
        ids, lines = [], []
        for i, e in batch:
            for edge, t in (("start", e.start_s), ("end", e.end_s)):
                bid = f"e{i}.{edge}"
                ids.append((bid, i, edge, t))
                lines.append(f"- {bid}: the {edge} of \"{e.label}\" ({e.start_s:.2f}-{e.end_s:.2f} s), proposed at {t:.2f} s")

        def closeups(w: int, ids=ids) -> list:
            out = []
            for bid, _i, _edge, t in ids:
                ts = footage.times(8, t - REFINE_WINDOW_S, t + REFINE_WINDOW_S)
                for caption, jpeg in sheets(footage, ts, REFINE_COLS, 5, cell_w=w):
                    out.append((f"=== close-up for {bid} ===\n" + caption, jpeg))
            return out
        images = fitting(lambda w: overview + closeups(w), REFINE_CELL_W)
        text = prompts.REFINE.format(overview_step=1 / OVERVIEW_FPS, window=REFINE_WINDOW_S,
                                     boundaries="\n".join(lines), **_header(settings, footage))
        reply = call(_content(text, images), "refine")
        _record(calls, "refine", reply, len(images))
        doc = parse_json(reply.text)
        got = {}
        for r in (doc.get("boundaries") if isinstance(doc, dict) else None) or []:
            try:
                got[str(r["id"])] = float(r["t_s"])
            except (KeyError, TypeError, ValueError):
                continue
        for bid, i, edge, t in ids:
            if bid not in got:
                notes.append(f"refine kept {bid}: no answer")
                continue
            new = min(max(got[bid], t - REFINE_WINDOW_S), t + REFINE_WINDOW_S)
            setattr(events[i], f"{edge}_s", new)
    for e in events:
        if e.end_s < e.start_s:
            e.start_s, e.end_s = e.end_s, e.start_s
    return clean(events, footage.duration_s, per_part=_per_part(settings))


def _instants(e: Event | None, n: int, ends: bool = False) -> list[float]:
    """n instants spread over the event: centred in n equal parts, or, with ends, from its first to its last instant
    (the event being named shows where it starts and how it is left)."""
    if e is None:
        return []
    span = e.end_s - e.start_s
    if n == 1:
        return [(e.start_s + e.end_s) / 2]
    if ends:
        return [e.start_s + span * j / (n - 1) for j in range(n)]
    return [e.start_s + span * (j + 0.5) / n for j in range(n)]


def _atlas_parts(tiles: list, roles: list, footage: Footage) -> list[tuple[list, list]]:
    """The atlas's tiles in as few chronological parts as keep each part under NAME_ATLAS_MAX_PX: an oversized atlas
    splits rather than shrinking its pictures."""
    from events.footage import CELL_MAX_W
    w, h = max((s for s in footage.sizes.values()), default=(CELL_MAX_W, CELL_MAX_W * 9 // 16))
    w, h = (CELL_MAX_W, round(h * CELL_MAX_W / w)) if w > CELL_MAX_W else (w, h)
    tile_px = (w + 4) * (h * max(1, len(footage.cameras)) + 34)
    per = max(NAME_PER_ROLE, (NAME_ATLAS_MAX_PX // max(1, tile_px)) // NAME_PER_ROLE * NAME_PER_ROLE)
    return [(tiles[i:i + per], roles[i:i + per]) for i in range(0, len(tiles), per)]


def _atlas(footage: Footage, events: list[Event], i: int) -> tuple[str, list[tuple[str, bytes]]]:
    """Event i's request line and its atlas images: prev, this and next instants, each capture once."""
    from events.sheets import sheet
    e = events[i]
    prev = events[i - 1] if i > 0 else None
    nxt = events[i + 1] if i + 1 < len(events) else None
    tiles, roles, seen = [], [], set()
    for role, ev in (("prev", prev), ("this", e), ("next", nxt)):
        for t in _instants(ev, NAME_PER_ROLE, ends=role == "this"):
            k = footage.index(t)
            if k in seen:  # a capture shared by two roles appears once, under the earlier role
                continue
            seen.add(k)
            tiles.append((k / 8, footage.at(t)))
            roles.append(role)
    mapping = "; ".join(f"{r}: " + ", ".join(f"{t:.2f}s" for (t, _c), rr in zip(tiles, roles) if rr == r)
                        for r in ("prev", "this", "next") if r in roles)
    images = []
    for part, (tt, rr) in enumerate(_atlas_parts(tiles, roles, footage)):
        caption = (f"=== atlas for e{i}{f' (part {part + 1})' if part else ''}: {e.start_s:.2f}-{e.end_s:.2f} s; "
                   f"tiles by role: {mapping} ===")
        png = sheet(tt, footage.cameras, NAME_PER_ROLE, roles=rr, png=True)
        # lossless unless one event's atlas alone would crowd the request's image budget
        images.append((caption, png if len(png) <= IMAGE_BUDGET_BYTES // NAME_BATCH
                       else sheet(tt, footage.cameras, NAME_PER_ROLE, roles=rr, quality=95)))
    return f"- e{i}: {e.start_s:.2f}-{e.end_s:.2f} s (found as \"{e.label}\")", images


def name(footage: Footage, settings: Settings, call: Caller, calls: list, notes: list,
         events: list[Event]) -> list[Event]:
    """Name every event, at most NAME_BATCH per request and within the request's image budget."""
    built = [_atlas(footage, events, i) for i in range(len(events))]
    batches, cur, size = [], [], 0
    for i, (_line, images) in enumerate(built):
        n = sum(len(b) for _c, b in images)
        if cur and (len(cur) == NAME_BATCH or size + n > IMAGE_BUDGET_BYTES):
            batches.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += n
    if cur:
        batches.append(cur)
    for idx in batches:
        batch = [(i, events[i]) for i in idx]
        lines = [built[i][0] for i in idx]
        atlases = [img for i in idx for img in built[i][1]]
        text = prompts.NAME.format(events="\n".join(lines),
                                   vocabulary=prompts.vocabulary_line(settings.skills, settings.parts),
                                   **_header(settings, footage))
        reply = call(_content(text, atlases), "name")
        _record(calls, "name", reply, len(atlases))
        doc = parse_json(reply.text)
        got = {}
        for r in (doc.get("events") if isinstance(doc, dict) else None) or []:
            if isinstance(r, dict) and isinstance(r.get("id"), str):
                got[r["id"]] = r
        for i, e in batch:
            r = got.get(f"e{i}")
            if r is None or not str(r.get("label") or "").strip():
                notes.append(f"name kept e{i}: no answer")
                continue
            e.label = str(r["label"]).strip()
            e.skill = (str(r["skill"]).strip() or None) if r.get("skill") else None
            e.parts = [str(p) for p in r.get("parts") or [] if str(p).strip()]
            e.objects = [str(o) for o in r.get("objects") or [] if str(o).strip()]
            d = r.get("destination")
            e.destination = str(d).strip() if d and str(d).strip().lower() not in ("null", "none") else None
    return events
