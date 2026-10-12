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
NAME_ATLAS_MAX_PX = 1800 * 1800  # an atlas past this splits into parts; its pictures are never shrunk
MIN_EVENT_S = 0.1


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


@dataclass
class Result:
    events: list[Event]
    calls: list[dict]
    notes: list[str]

    def to_dict(self) -> dict:
        return {"events": [asdict(e) for e in self.events], "calls": self.calls, "notes": self.notes,
                "cost_usd": round(sum(c.get("cost_usd") or 0.0 for c in self.calls), 6)}


def annotate(footage: Footage, settings: Settings, call: Caller, name_call: Caller | None = None) -> Result:
    """name_call, when given, names the events with another (usually stronger) model: finding and timing events
    take most of the images, naming takes few, so a stronger namer costs little."""
    calls: list[dict] = []
    notes: list[str] = []
    events = find(footage, settings, call, calls)
    if settings.passes >= 2 and events:
        events = refine(footage, settings, call, calls, notes, events)
    if settings.passes >= 3 and events:
        events = name(footage, settings, name_call or call, calls, notes, events)
    return Result(events, calls, notes)


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


def clean(events: list[Event], duration: float) -> list[Event]:
    """Order, clip to the recording, drop empty events and resolve overlaps at the midpoint of the overlap."""
    out = []
    for e in sorted(events, key=lambda e: (e.start_s, e.end_s)):
        e.start_s, e.end_s = max(0.0, min(e.start_s, duration)), max(0.0, min(e.end_s, duration))
        if e.end_s - e.start_s < MIN_EVENT_S or not e.label.strip():
            continue
        if out and e.start_s < out[-1].end_s:
            mid = (e.start_s + out[-1].end_s) / 2
            out[-1].end_s = max(out[-1].start_s + MIN_EVENT_S, mid)
            e.start_s = out[-1].end_s
            if e.end_s - e.start_s < MIN_EVENT_S:
                continue
        out.append(e)
    return out


def find(footage: Footage, settings: Settings, call: Caller, calls: list) -> list[Event]:
    fps = FIND_FPS
    while len(footage.times(fps)) > MAX_FIND_IMAGES * FIND_COLS * FIND_ROWS and fps > 1:
        fps //= 2
    times = footage.times(fps)
    images = sheets(footage, times, FIND_COLS, FIND_ROWS, cell_w=FIND_CELL_W)
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
    return clean(events, footage.duration_s)


def refine(footage: Footage, settings: Settings, call: Caller, calls: list, notes: list,
           events: list[Event]) -> list[Event]:
    overview = sheets(footage, footage.times(OVERVIEW_FPS), OVERVIEW_COLS, OVERVIEW_ROWS, cell_w=OVERVIEW_CELL_W)
    for b0 in range(0, len(events), NAME_BATCH):
        batch = list(enumerate(events[b0:b0 + NAME_BATCH], start=b0))
        ids, lines, closeups = [], [], []
        for i, e in batch:
            for edge, t in (("start", e.start_s), ("end", e.end_s)):
                bid = f"e{i}.{edge}"
                ids.append((bid, i, edge, t))
                lines.append(f"- {bid}: the {edge} of \"{e.label}\" ({e.start_s:.2f}-{e.end_s:.2f} s), proposed at {t:.2f} s")
                ts = footage.times(8, t - REFINE_WINDOW_S, t + REFINE_WINDOW_S)
                for caption, jpeg in sheets(footage, ts, REFINE_COLS, 5, cell_w=REFINE_CELL_W):
                    closeups.append((f"=== close-up for {bid} ===\n" + caption, jpeg))
        text = prompts.REFINE.format(overview_step=1 / OVERVIEW_FPS, window=REFINE_WINDOW_S,
                                     boundaries="\n".join(lines), **_header(settings, footage))
        reply = call(_content(text, overview + closeups), "refine")
        _record(calls, "refine", reply, len(overview) + len(closeups))
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
    return clean(events, footage.duration_s)


def _instants(e: Event | None, n: int) -> list[float]:
    if e is None:
        return []
    if n == 1:
        return [(e.start_s + e.end_s) / 2]
    span = e.end_s - e.start_s
    return [e.start_s + span * (j + 0.5) / n for j in range(n)]


def _atlas_parts(tiles: list, roles: list, footage: Footage) -> list[tuple[list, list]]:
    """The atlas's tiles in as few chronological parts as keep each part under NAME_ATLAS_MAX_PX: an oversized atlas
    splits rather than shrinking its pictures."""
    w, h = max((s for s in footage.sizes.values()), default=(448, 252))
    w, h = min(w, 448), min(h, round(h * 448 / w)) if w > 448 else h
    tile_px = (w + 4) * (h * max(1, len(footage.cameras)) + 34)
    per = max(NAME_PER_ROLE, (NAME_ATLAS_MAX_PX // max(1, tile_px)) // NAME_PER_ROLE * NAME_PER_ROLE)
    return [(tiles[i:i + per], roles[i:i + per]) for i in range(0, len(tiles), per)]


def name(footage: Footage, settings: Settings, call: Caller, calls: list, notes: list,
         events: list[Event]) -> list[Event]:
    from events.sheets import sheet
    for b0 in range(0, len(events), NAME_BATCH):
        batch = list(enumerate(events[b0:b0 + NAME_BATCH], start=b0))
        lines, atlases = [], []
        for i, e in batch:
            prev = events[i - 1] if i > 0 else None
            nxt = events[i + 1] if i + 1 < len(events) else None
            tiles, roles, seen = [], [], set()
            for role, ev in (("prev", prev), ("this", e), ("next", nxt)):
                for t in _instants(ev, NAME_PER_ROLE):
                    k = footage.index(t)
                    if k in seen:  # a capture shared by two roles appears once, under the earlier role
                        continue
                    seen.add(k)
                    tiles.append((k / 8, footage.at(t)))
                    roles.append(role)
            lines.append(f"- e{i}: {e.start_s:.2f}-{e.end_s:.2f} s (found as \"{e.label}\")")
            mapping = "; ".join(f"{r}: " + ", ".join(f"{t:.2f}s" for (t, _c), rr in zip(tiles, roles) if rr == r)
                                for r in ("prev", "this", "next") if r in roles)
            for part, (tt, rr) in enumerate(_atlas_parts(tiles, roles, footage)):
                atlases.append((f"=== atlas for e{i}{f' (part {part + 1})' if part else ''}: "
                                f"{e.start_s:.2f}-{e.end_s:.2f} s; tiles by role: {mapping} ===",
                                sheet(tt, footage.cameras, NAME_PER_ROLE, roles=rr, png=True)))
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
