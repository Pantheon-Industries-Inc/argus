"""One episode's request: which frames, decoded exactly, and a prompt that states only facts we are sure of.

An episode is a sidecar folder written by a preparer (prepare/): context.json (the dataset's facts: rig, state
kind, fps, cameras and what each one is, the instruction or annotation), sources.json (per camera, the video
file, the episode's offset in it and its frame count), state.npz (recorded state and action, absent for
video-only rigs) and, where the dataset keeps real capture times, times.npz and a kmap per camera paired to the
anchor camera by time. Nothing about the data is changed: the model sees the episode as the dataset ships it.

What the model is told about the episode, and where each fact comes from:
- dataset, robot type, fps, camera names and resolution: the dataset's own metadata (context.json).
- what each camera is: checked by eye on the dataset's frames at prepare time, stated as the dataset's
  camera identity, and the model is told to verify it against the pixels.
- the instruction or annotation: the dataset's per-episode text, as a claim to check.
- still spans: from the recorded state, true by construction (state.still_spans), as the recording's claim.
- recorded motion between consecutive instants: from the recorded state, as a claim to check.
- sampling: exactly what state.sample_frames did.
Nothing is said about field of view, lens, lighting, object identities, or what a gripper reading implies.

Layout (fixed, not flags):
- Grid cells. A teleop episode's width is chosen from its task text alone (label/route.py): a task that needs
  fine detail (lettering, numbers, symbols or a display, which face of an object is up, small objects of
  similar shape) keeps every camera at 448 px, because at narrower cells a wrist camera's oblique view cannot
  show which face is up and the model reads a change of viewing direction as a change of state; every other
  teleop episode is sent at 224 px. Handheld cells are 320 px: a gripper camera has the held object in every
  frame, and at 256 px or less a thin utensil held upright was misread. Head-camera cells are 256 px. An
  episode whose grids would pass the request's image-size cap (a long ABC-130k episode at 448 px) is sent at
  the largest of CELL_W_STEPS that fits, instead of being refused.
- Contact detail views, on teleop episodes sent at narrow cells: at the sampled instants just after the
  recorded gripper value changed sharply, the scene camera and the acting arm's own camera are sent again at
  detail size, because that is where what is held, and how it is left, is decided.
- After the grids, the episode's first and last instant are sent again larger (all cameras stacked, at most
  DETAIL_MAX_W wide each), because completion is judged on the end state and small detail (lettering, a
  display, fine alignment) only reads at native resolution.
"""
from __future__ import annotations

import base64
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from checks import timebase
from label import frames as mf
from label import prompts
from label import state as ms

FPS = 30
VIEW_ORDER = ("exo", "left", "right")          # harness view keys with a role: the scene camera, the two mounted ones
MOUNTED = ("left", "right")                     # the cameras mounted on the left and right arm or gripper
EXTRA_VIEW = re.compile(r"extra(\d+)")          # any other camera the recording has (extra1, extra2, ...)
CAM_NAME = {"exo": "top", "left": "left", "right": "right"}   # dataset camera names
GRID_CELL_W_BY_RIG = {"teleop_arms": 448, "handheld_gripper": 320, "ego_head": 256}
CELL_W_STEPS = (448, 384, 320, 288, 256, 224, 192)   # widths tried, largest first, when a request is too big
DETAIL_VIEW_BYTES_MAX = 2_000_000   # room kept under the cap for the two full-resolution first and last frames
# The provider caps a request's total image size at 50 MB. Its counted size is not the raw JPEG bytes: on a
# measured request 34 MB of JPEG counted as 60.14 MB (about 1.77x, base64 on the wire plus overhead), so the
# inflated size is bounded with a margin.
IMAGE_LIMIT_BYTES = 50 * 1024 * 1024
IMAGE_SIZE_INFLATION = 1.85
# The first and last instant are also sent larger, for small detail (lettering, a display, fine
# alignment). Image tokens scale with pixel area, so each camera is capped at this width: 768 px is above
# MolmoAct2's native 640 px (where lettering and displays read fine), while native 1920 px frames would
# cost about 6x as much for little gain.
DETAIL_MAX_W = 768
# Resolution routing (label/route.py): the (narrow, wide) cell width of a routed rig. A task that needs fine detail
# gets the wide cells, every other task the narrow ones plus contact detail views.
ROUTE_WIDTHS = {"teleop_arms": (224, 448)}
CONTACT_BELOW_W = 384   # contact detail views only for episodes sent at narrower cells than this
# Contact detail views: the sampled instants just after the recorded gripper value changed sharply (a close or an
# open between two instants), where what is held, or how it is left, is decided. The value only chooses where to
# look closer; what is held is still read from the frames. One view per CONTACT_EVERY_S of footage at most, never
# more than CONTACT_MAX, and none for a flat channel, video only or head cameras. Teleop only: a handheld episode is
# one or two gripper cameras already sent at 320 px, which settles what is held at lower cost.
CONTACT_RIGS = ("teleop_arms",)
CONTACT_EVERY_S = 4.0
CONTACT_MAX = 8
CONTACT_MIN_CHANGE = 0.25    # of the channel's own range over the episode, between consecutive instants
CONTACT_MIN_GAP_S = 2.0
PAIRED_SPAN_SLACK_S = 0.1   # a paired camera is shown at an instant up to this far outside its own first and last frame
GRID_GUTTER = 84
GRID_HEADER = 30


def is_episode_dir(ep_dir: Path) -> bool:
    return (Path(ep_dir) / "context.json").exists() and (Path(ep_dir) / "sources.json").exists()


def load(ep_dir: Path) -> dict:
    ep_dir = Path(ep_dir)
    ctx = json.loads((ep_dir / "context.json").read_text())
    src = json.loads((ep_dir / "sources.json").read_text())
    for v, d in src.items():
        if "n_frames" not in d:
            raise RuntimeError(f"{ep_dir}: sources.json has no n_frames for {v}; prepare the episode again")
    if ctx.get("state_kind") == "none":
        # video-only rigs (a head camera on a person): no recorded state; the state array is empty
        # columns over the anchor camera's frames so frame counts and times work unchanged
        first = order_views(src)[0]
        state, action = np.zeros((int(src[first]["n_frames"]), 0)), None
    else:
        z = np.load(ep_dir / "state.npz")
        if "state" not in z.files:
            raise RuntimeError(f"{ep_dir}: state.npz has no state array; prepare the episode again")
        state, action = z["state"], (z["action"] if "action" in z.files else None)
    ep = {"dir": ep_dir, "context": ctx, "sources": src, "state": state,
          "action": action, "times": None, "kmap": {}, "signals": {}}
    if ctx.get("signals"):
        # the recording's other per-frame numbers, under the dataset's names (prepare/formats.py recorded_signals)
        z = np.load(ep_dir / "signals.npz")
        ep["signals"] = {s["name"]: z[s["key"]] for s in ctx["signals"]}
    if ctx.get("real_times"):
        # datasets with real per-frame capture times (ABC-130k, RealOmin): every time shown uses them, and
        # each camera's frames are decoded by their exact pts
        t = np.load(ep_dir / ctx["real_times"])
        ep["times"] = {k: t[k] for k in t.files}
    for v, d in src.items():
        if d.get("kmap"):
            ep["kmap"][v] = np.load(ep_dir / d["kmap"])
    return ep


def ep_fps(ep: dict) -> float:
    return float(ep["context"].get("fps") or FPS)


def order_views(keys) -> list[str]:
    """View keys in row order: the scene camera, the left and right mounted ones, then the extra cameras by number."""
    keys = list(keys)
    extra = sorted((k for k in keys if EXTRA_VIEW.fullmatch(str(k))), key=lambda k: int(EXTRA_VIEW.fullmatch(k)[1]))
    return [v for v in VIEW_ORDER if v in keys] + extra


def views(ep: dict) -> list[str]:
    """The episode's cameras in row order (top, then left, then right gripper, then any other cameras)."""
    have = ep.get("sources") or ep["context"].get("cameras") or dict.fromkeys(VIEW_ORDER)
    return order_views(have)


def anchor(ep: dict) -> str:
    return views(ep)[0]


def cam_name(ep: dict, v: str) -> str:
    return ((ep["context"].get("cameras") or {}).get(v) or {}).get("name") or CAM_NAME.get(v, v)


def frame_time(ep: dict, k: int) -> float:
    """Seconds from the episode's start for anchor frame k: the real capture time when the dataset
    has one, otherwise k / fps (MolmoAct2's and FastUMI's timestamps are exactly frame_index / fps)."""
    if ep.get("times") is not None:
        return float(ep["times"][anchor(ep)][k])
    return k / ep_fps(ep)


def describe_spans(ep: dict, spans) -> list[dict]:
    last = len(ep["state"]) - 1
    return [{"start_s": round(frame_time(ep, a), 2), "end_s": round(frame_time(ep, min(b, last)), 2)} for a, b in spans]


# Rigs this harness knows. The episode's context.json declares one as "profile"; nothing is assumed from the
# dataset name or the camera set.
RIGS = ("teleop_arms", "handheld_gripper", "ego_head")
STATE_KINDS = ("joints", "ee_pose", "none")
# One sampled instant every N s for the whole episode, still spans included (a still span is exactly
# where a stopped recording would hide). Teleop arms move slowly and are seen by up to three cameras, so one
# instant every 1.5 s is enough. Handheld demonstrations are short and fast, so denser. Egocentric footage has one
# low-resolution head camera and the hands are the whole point, so it is sampled twice as densely again.
SAMPLE_EVERY_S = {"teleop_arms": 1.5, "handheld_gripper": 1.0, "ego_head": 0.5}


def rig(ep: dict) -> str:
    p = ep["context"].get("profile")
    if p not in RIGS:
        raise RuntimeError(f"{ep.get('dir', '?')}: context.json must declare profile as one of {RIGS}, got {p!r}")
    return p


def state_kind(ep: dict) -> str:
    k = ep["context"].get("state_kind")
    if k not in STATE_KINDS:
        raise RuntimeError(f"{ep.get('dir', '?')}: context.json must declare state_kind as one of {STATE_KINDS}, "
                           f"got {k!r}")
    return k


def plan(ep: dict) -> dict:
    """Frames to send plus the deterministic checks we report ourselves."""
    T = int(len(ep["state"]))
    r, kind, fps = rig(ep), state_kind(ep), ep_fps(ep)
    windows = {v: int(ep["sources"][v]["n_frames"]) for v in views(ep)}
    # the state follows the anchor camera's frames; a camera paired to the anchor by real time (kmap) has
    # its own frame count and is matched through the map, so only unpaired cameras must equal the state
    a = anchor(ep)
    paired = {v for v in windows if v != a and v in ep["kmap"] and len(ep["kmap"][v]) >= windows[a]}
    checks = {"state_frames": T, "camera_frames": windows,
              "camera_windows_match_state": all(n == T for v, n in windows.items() if v not in paired)}
    if ep.get("action") is not None and r == "teleop_arms" and kind == "joints" and ep["state"].shape[1] == 14:
        # sped-up recording (the rig's loop ran below the rate its samples are stamped at): a report
        # field computed from the leader/follower joint lag, not a claim made to the model. It reads the 12 arm
        # joints of two arms (timebase.JOINTS), as measure_folder does, so a one-arm recording is not measured
        checks["timebase"] = timebase.timebase_check(ep["state"], ep["action"],
                                               ep["context"].get("timebase_neighbour_lag_frames"))
    if kind != "none" and checks["camera_windows_match_state"]:
        spans = ms.still_spans(ep["state"], fps=fps, kind=kind, grip_range=ms.gripper_full_range(ep["context"]))
        n = T
    else:
        # video only, or a dataset defect (the cameras do not cover the same frames as the state): label the
        # video as shipped over the anchor frames every camera not paired to it by time also has, and make no
        # state claims (the defect is reported in checks)
        spans = []
        n = min([windows[a]] + [w for v, w in windows.items() if v != a and v not in paired])
    if ep["context"].get("stream_checks"):
        checks["streams"] = ep["context"]["stream_checks"].get("streams")
    if ep["context"].get("stream_pairing"):
        # checks/stream_pairing.py: whether each mounted stream follows its own actor's recorded motion (a
        # report field, never a claim made to the model)
        checks["stream_pairing"] = ep["context"]["stream_pairing"]
    every = SAMPLE_EVERY_S[r]
    ks = ms.sample_frames(n, spans, fps=fps, moving_every_s=every, still_every_s=every)
    pl = {"n": n, "ks": ks, "spans": spans, "checks": checks,
          "state_usable": checks["camera_windows_match_state"]}
    pl["contact"] = contact_instants(ep, pl)
    return pl


def contact_instants(ep: dict, pl: dict) -> list[int]:
    """Sampled instants just after the recorded gripper value changed sharply, where an object is usually picked
    up or put down: the largest changes first, at least CONTACT_MIN_GAP_S apart, never the first or last instant
    (already sent in detail). Empty for head cameras, video only, a state that does not line up with the cameras,
    and a flat channel (it carries no timing)."""
    if rig(ep) not in CONTACT_RIGS or state_kind(ep) == "none" or not pl["state_usable"]:
        return []
    ks = pl["ks"]
    st = np.asarray(ep["state"][:pl["n"]], dtype=np.float64)
    cands = []
    for g in range(st.shape[1] // 7):
        v = st[:, 7 * g + 6]
        rng = float(np.nanmax(v) - np.nanmin(v)) if np.isfinite(v).any() else 0.0
        if not rng > 1e-6:
            continue
        for a, b in zip(ks, ks[1:]):
            d = abs(float(v[b] - v[a])) / rng
            if d >= CONTACT_MIN_CHANGE and b != ks[-1]:
                cands.append((d, b))
    n_max = min(CONTACT_MAX, max(1, int(round(frame_time(ep, ks[-1]) / CONTACT_EVERY_S))))
    gap = CONTACT_MIN_GAP_S * ep_fps(ep)
    chosen = []
    for d, k in sorted(cands, key=lambda x: (-x[0], x[1])):
        if len(chosen) >= n_max:
            break
        if all(abs(k - c) >= gap for c in chosen):
            chosen.append(k)
    return sorted(chosen)


def contact_views(ep: dict, pl: dict) -> list[tuple[int, list[str]]]:
    """[(k, views)]: each contact instant with the scene camera (it shows how the object is left) and the mounted
    cameras of the arms whose gripper value changed sharply into it (a wrist camera shows what its own gripper
    holds)."""
    mounted = [v for v in views(ep) if v in MOUNTED]
    if not mounted:
        return [(k, list(views(ep))) for k in pl["contact"]]
    st = np.asarray(ep["state"][:pl["n"]], dtype=np.float64)
    ks = pl["ks"]
    out = []
    for k in pl["contact"]:
        i = ks.index(k)
        a = ks[i - 1] if i > 0 else k
        who = []
        for g in range(st.shape[1] // 7):
            v = st[:, 7 * g + 6]
            rng = float(np.nanmax(v) - np.nanmin(v))
            if rng > 1e-6 and abs(float(v[k] - v[a])) / rng >= CONTACT_MIN_CHANGE:
                who.append(g)
        vs = [mounted[g] for g in who if g < len(mounted)] if len(mounted) > 1 else mounted
        out.append((k, (["exo"] if "exo" in views(ep) else []) + (vs or mounted)))
    return out


def actors(ep: dict) -> list[str]:
    """Names of the arms or grippers in state order (7 values each): left then right, or the one.
    On a person (ego), the actors are their own two hands."""
    if rig(ep) == "ego_head":
        return ["left", "right"]
    if state_kind(ep) == "none":
        # video only: the actors are the mounted cameras' own, or both when no single mounted camera names one
        mounted = [v for v in views(ep) if v in MOUNTED]
        return [mounted[0]] if len(mounted) == 1 else ["left", "right"]
    if ep["state"].shape[1] == 14:
        return ["left", "right"]
    # the one arm or gripper is named by its own mounted camera, never by an extra camera that sorts after it
    mounted = [v for v in views(ep) if v in MOUNTED]
    return [cam_name(ep, (mounted or views(ep)[:1])[-1])]


def _decode_view(ep: dict, v: str, ks: list[int], gate=None):
    """Frames for anchor indices ks. A camera paired to the anchor by real time (kmap) is decoded at
    its own frame nearest each anchor frame; results are keyed by the anchor index."""
    s = ep["sources"][v]
    km = ep["kmap"].get(v)
    own = [int(km[k]) for k in ks] if km is not None else list(ks)
    pts = ep["times"].get(f"{v}_pts") if ep.get("times") is not None else None

    def run():
        return mf.extract_frames(s["packed"], s["base_s"], int(s["n_frames"]), own, pts=pts, fps=ep_fps(ep))
    if gate is not None:
        with gate:
            got = run()
    else:
        got = run()
    return {k: got[j] for k, j in zip(ks, own)}


def frames(ep: dict, pl: dict, gate=None) -> dict:
    """{view: {k: PIL image}} for every planned k, decoded exactly (raises otherwise)."""
    vs = views(ep)
    with ThreadPoolExecutor(max_workers=len(vs)) as ex:
        futs = {v: ex.submit(_decode_view, ep, v, pl["ks"], gate) for v in vs}
        return {v: f.result() for v, f in futs.items()}


def recording_at(ep: dict, v: str, k: int) -> bool:
    """Whether camera v was recording at anchor instant k (within PAIRED_SPAN_SLACK_S of its own first and last frame).
    A camera paired to the anchor by real time (kmap) that started later or stopped earlier was not: its nearest frame
    there is its first or last, taken at another time, so it is not shown under this instant's time."""
    km = (ep.get("kmap") or {}).get(v)
    t = ep["times"] if ep.get("times") is not None else None
    if km is None or t is None or v not in t:
        return True
    # only the camera's own span counts: inside it, a dropped frame leaves the nearest frame a few hundredths of a
    # second away, which is still the view at that moment; and a camera whose first frame comes a frame or two after
    # the anchor's (ABC-130k's wrists, 0.034 s) is shown as usual
    tk = frame_time(ep, k)
    return float(t[v][0]) - PAIRED_SPAN_SLACK_S <= tk <= float(t[v][-1]) + PAIRED_SPAN_SLACK_S


def timesteps(ep: dict, pl: dict, imgs: dict, cell_w: int, quality: int = 90):
    """[(t_s, [(camera_name, jpeg bytes), ...]), ...] in time order, cameras in a fixed order; a camera with no
    frame at an instant (recording_at) is left out there, so its grid cell stays empty."""
    vs = order_views(imgs)
    out = []
    for k in pl["ks"]:
        out.append((frame_time(ep, k), [(cam_name(ep, v), mf.to_jpeg(imgs[v][k], cell_w, quality)) for v in vs
                                        if recording_at(ep, v, k)]))
    return out


def detail_size(w: int, h: int) -> tuple[int, int]:
    if w <= DETAIL_MAX_W:
        return w, h
    return DETAIL_MAX_W, int(round(h * DETAIL_MAX_W / w / 2)) * 2


def fullres_stack(ep: dict, imgs: dict, k: int, label: str, t_s: float, only: list[str] | None = None) -> bytes:
    """All cameras (or `only` these) at instant k, at detail size (native, capped at DETAIL_MAX_W wide), stacked
    top to bottom with a name strip."""
    from PIL import Image, ImageDraw
    vs = [v for v in order_views(imgs) if only is None or v in only]
    ims = [imgs[v][k] for v in vs]
    ims = [im if im.width <= DETAIL_MAX_W else im.resize(detail_size(im.width, im.height), Image.LANCZOS)
           for im in ims]
    w = max(i.width for i in ims)
    strip = 26
    g = Image.new("RGB", (w, sum(i.height + strip for i in ims)), (18, 18, 20))
    d = ImageDraw.Draw(g)
    y = 0
    for v, im in zip(vs, ims):
        d.text((6, y + 5), f"{cam_name(ep, v)}   {label}   t={t_s:.2f}s", fill=(255, 220, 0))
        g.paste(im, (0, y + strip))
        y += im.height + strip
    buf = io.BytesIO()
    g.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _rig_nouns(r: str) -> dict:
    if r == "ego_head":
        return {"actor": "hand", "an_actor": "a hand", "actors": "hands", "gripper_of": "hand",
                "who": "a person wears a camera on their head and works with their own two hands"}
    if r == "teleop_arms":
        return {"actor": "arm", "an_actor": "an arm", "actors": "arms", "gripper_of": "arm's gripper",
                "who": "a person teleoperates the robot arms"}
    return {"actor": "gripper", "an_actor": "a gripper", "actors": "grippers", "gripper_of": "handheld gripper",
            "who": "a person holds handheld grippers and does the task with them"}


def _camera_line(ep: dict, v: str) -> str:
    """One camera's facts. A description written at prep time (verified for that dataset) wins; the
    fallback says only what the camera's slot implies: on the left/right gripper, or not on one."""
    cam = (ep["context"].get("cameras") or {}).get(v) or {}
    if cam.get("desc"):
        return f"- {cam_name(ep, v)}: {cam['desc'].rstrip('.')}."
    n = _rig_nouns(rig(ep))
    if v == "exo" and rig(ep) == "ego_head":
        return f"- {cam_name(ep, v)}: the camera worn on the person's head."
    if v == "exo":
        return (f"- {cam_name(ep, v)}: a camera that is not mounted on any {n['actor']}. Use it for the "
                "scene layout, object locations and where things end up.")
    if v not in MOUNTED:
        return (f"- {cam_name(ep, v)}: another camera the recording has; the dataset does not say where it is "
                "mounted, so read that from its frames.")
    side = "" if len(views(ep)) == 1 or v not in ("left", "right") else f"{v.upper()} "
    return (f"- {cam_name(ep, v)}: the camera mounted on the {side}{n['gripper_of']}. Its own gripper's fingers sit in "
            "the same place in every frame, usually along the bottom edge, and change only by opening and closing; "
            "the rest of the image moves whenever the gripper moves.")


def camera_desc(ep: dict) -> str:
    vs, r = views(ep), rig(ep)
    n = _rig_nouns(r)
    cams = ep["context"].get("cameras") or {}
    res = sorted({f"{c.get('width')}x{c.get('height')}" for c in cams.values() if c.get("width")})
    rec = f" (recording {', '.join(res)} at {ep_fps(ep):g} fps)" if res else ""
    count = "There is exactly 1 camera; every grid row and every image strip is labelled with its name" \
        if len(vs) == 1 else (f"There are exactly {len(vs)}; every grid row and every image strip is labelled "
                              "with one of these names")
    s = (f"Cameras in this episode, as named in the dataset{rec}. {count}:\n"
         + "\n".join(_camera_line(ep, v) for v in vs) + "\n")
    if not any(v not in MOUNTED for v in vs):
        mounted = [v for v in vs if v in MOUNTED]
        s += (f"No camera in this episode is off the {n['actor'] if len(mounted) == 1 else n['actors']}: the "
              f"whole scene is seen only through {'that camera' if len(mounted) == 1 else 'those cameras'}, so "
              "reconstruct the layout and where things end up from what it shows.\n")
    if r == "ego_head":
        s += ("This camera description is dataset metadata, not guaranteed truth: check it against the pixels. "
              "If the stream contradicts it (a view that stays fixed instead of moving with the person's head, "
              "a black, frozen or corrupted stream), describe what the view actually is and record it as a data "
              "issue. Where the camera is worn and where it points are read from the frames: a head camera turns "
              "and tilts with the head, so it shows wherever the person looks, and a camera that is really on the "
              "chest or held in the hand, or a mount that slips and tilts the view partway through, is worth "
              "recording.")
    else:
        s += ("These camera identities are dataset metadata, not guaranteed truth: check them against the "
              "pixels. If a stream's content contradicts its name (a mounted camera that shows a fixed view "
              f"or the reverse, two {n['actor']} streams swapped or identical, a black or frozen stream), "
              "describe what the view actually is and record it as a data issue.")
    if r != "ego_head" and any(v in MOUNTED for v in vs):
        s += (f" A mounted camera turns with its {n['actor']}, so where it looks changes through the episode: "
              "sometimes down onto the work, sometimes along or across it. Work out its direction at each instant "
              "from the frame itself (the perspective of the table or floor, which faces of an object are in view, "
              f"where walls or the room appear) and from where that {n['actor']} is in the other views at the same "
              "instant, and read heights, contacts and which face of an object is up from that geometry, never from "
              "an assumed direction. The "
              "same object seen from another direction shows other faces although nothing about it changed: a face "
              "that fills a view looking along the table is a side, and only a view looking down on an object shows "
              "its top. So an object changed state only when views from comparable directions, or the camera that "
              "is not mounted, show the change, never because a mounted camera now sees it from elsewhere.")
    if "left" in vs and "right" in vs:
        s += (f" In the output, \"left\", \"right\" and \"both\" name the streams: an action is \"left\" when "
              "the left stream's own gripper makes the contact, \"right\" likewise, "
              f"\"both\" when the two act together. The other {n['actor']} often appears inside a view, and an "
              "object lying between open fingers is not yet held, so neither is a contact of that camera's own. "
              "This naming is bookkeeping only; it does not settle whether the names are right. Whether each "
              "stream really sits on the side its name says is a separate question for the pixels: where the "
              f"other {n['actor']} and the scene appear in it once you have worked out from the frame how that camera "
              "is turned at that instant, and which recorded motion its view follows.")
    elif r == "ego_head":
        s += (" In the output, \"left\", \"right\" and \"both\" name the person's own left and right hands, "
              "as seen from their head. Which hand is which follows the person's body (the forearm it belongs to, "
              "the thumb side), not which half of the image it is in, because hands cross the midline and reach "
              "across.")
    elif len([v for v in vs if v in MOUNTED]) == 1:
        s += f" In the output, the \"arm\" field always names the one {n['actor']}: \"{actors(ep)[0]}\"."
    return s


def _detail_desc(native: tuple) -> str:
    try:
        w, h = int(native[0]), int(native[1])
    except (TypeError, ValueError):
        return f"up to {DETAIL_MAX_W} px wide"
    dw, dh = detail_size(w, h)
    return f"the full {w}x{h}" if (dw, dh) == (w, h) else f"{dw}x{dh} (the recording is {w}x{h})"


def sampling_desc(ep: dict, pl: dict, cell_w: int, cell_h: int, native: tuple) -> str:
    r, kind = rig(ep), state_kind(ep)
    n = _rig_nouns(r)
    names = ", ".join(cam_name(ep, v) for v in views(ep))
    every = SAMPLE_EVERY_S[r]
    s = (
        f"\nFRAMES. You receive the episode as grid images: ROWS are the cameras ({names}, top to "
        "bottom), COLUMNS are instants left to right, and each column is headed with its exact time in "
        "seconds from the episode's first frame. Read each grid left to right and the grids in order. "
        "The times are exact: use them, do not invent your own. Each grid cell is the camera frame "
        f"downscaled to {_cell_sizes(ep, cell_w, cell_h)}. After the grids, the episode's first and last instant are "
        f"repeated larger, at {_detail_desc(native)}; use them for the start and end state and any "
        "small detail (lettering, a display, fine alignment)."
        + (" So are the instants just after a gripper's recorded value changes sharply, where something is "
           "usually picked up or put down, from the scene camera and that gripper's own camera ("
           + ", ".join(f"{frame_time(ep, k):.2f}" for k in pl.get("contact") or []) + " s): use them to read what "
           "is held and how it is left. The value only chose where to look closer; what is held is read from "
           "the frames." if pl.get("contact") else "")
        + "\n"
        f"Which instants you get: one every {every:g} s for the whole episode, plus its first and last "
        "frame." + _coverage_note(ep, pl))
    sig = _signals_table(ep, pl)
    if kind == "none":
        what = ("no hand, head or device tracking" if r == "ego_head" else "no robot or gripper state")
        return s + ((f"\nRECORDED STATE: none; this dataset records {what}, so the video is all there is."
                     if not sig else f"\nRECORDED STATE: no {n['actor']} state in the layout our checks read.")
                    + sig + BETWEEN_INSTANTS)
    if not pl.get("state_usable", True):
        return s + ("\nRECORDED STATE: not given. This episode's cameras do not cover the same frames "
                    "as its recorded state, so the state cannot be aligned to the video." + BETWEEN_INSTANTS)
    src = ("joint encoders" if kind == "joints" else
           "recorded end-effector poses" if r == "teleop_arms" else "tracked gripper poses")
    if pl["spans"]:
        sp = ", ".join(f"{d['start_s']:.2f}-{d['end_s']:.2f}s" for d in describe_spans(ep, pl["spans"]))
        s += (f"\nRECORDED STILL SPANS, from the dataset's {src}: {sp}. Over each span the recording says "
              f"no {n['actor']} moved and none opened or closed. This is the recording's claim, not a "
              f"fact: check it. {'An' if n['actor'][0] in 'aeiou' else 'A'} {n['actor']} that is really still "
              "shows a steady view in its own camera"
              + (" unless something that carries it moves, which the other recorded signals below may show; the "
                 f"claim covers only the {n['actors']}" if sig else "")
              + ". If the views show motion during a span, the recording is "
              f"wrong there. If the views hold steady and the scene still changes, the {n['actors']} did not do it: "
              "say what you see.")
    else:
        s += f"\nRECORDED STILL SPANS, from the dataset's {src}: none."
    return s + _motion_table(ep, pl) + sig + BETWEEN_INSTANTS


def _cell_sizes(ep: dict, cell_w: int, cell_h: int) -> str:
    """The grid cell size: every camera is cut to cell_w wide with its own aspect kept (label/frames.py to_jpeg), so
    cameras of different aspect get cells of different height, and each is named then."""
    cams = ep["context"].get("cameras") or {}
    sizes = {}
    for v in views(ep):
        c = cams.get(v) or {}
        try:
            w, h = int(c["width"]), int(c["height"])
        except (KeyError, TypeError, ValueError):
            return f"{cell_w}x{cell_h}"
        sizes[cam_name(ep, v)] = int(round(h * cell_w / w / 2)) * 2
    if len(set(sizes.values())) <= 1:
        return f"{cell_w}x{cell_h}"
    return f"{cell_w} px wide (" + ", ".join(f"{n} {cell_w}x{h}" for n, h in sizes.items()) + ")"


def _coverage_note(ep: dict, pl: dict) -> str:
    """A camera that has no frame at some instants (recording_at): when it records, so its empty cells are read as
    what they are."""
    gaps = []
    for v in views(ep):
        if all(recording_at(ep, v, k) for k in pl["ks"]):
            continue
        t = ep["times"][v]
        gaps.append(f"{cam_name(ep, v)} has frames only from {float(t[0]):.2f} s to {float(t[-1]):.2f} s")
    if not gaps:
        return ""
    s = "; ".join(gaps)
    return (" " + s[0].upper() + s[1:] + ", so its cells are empty at the instants outside that time, and it is "
            "left out of a detail view there.")


def _num(x: float) -> str:
    return f"{float(x):.3g}"


def _signals_table(ep: dict, pl: dict) -> str:
    """The recording's other per-frame numbers (ep["signals"], under the dataset's own names): every one listed once
    with the range each of its values takes over the episode, and over each recorded still span how much each one
    changed. The still span is the claim they bear on (a mobile base can drive while the arms are still), so their
    values are spent there, not repeated at every instant. They are shown, not interpreted: the model reads what each
    is from its name and the robot's description."""
    sig = ep.get("signals") or {}
    if not sig:
        return ""
    n = pl["n"]
    arrs = {k: np.asarray(a[:n], dtype=np.float64) for k, a in sig.items()}
    lines = []
    for name, a in arrs.items():
        d = a.shape[1]
        head = f"  {name} ({d} value{'s' if d > 1 else ''})"
        if not len(a):
            continue
        lo, hi = a.min(axis=0), a.max(axis=0)
        if (hi == lo).all():
            lines.append(f"{head}: " + (_num(lo[0]) if d == 1 else "[" + ", ".join(_num(x) for x in lo) + "]")
                         + " throughout")
        else:
            lines.append(f"{head}: " + ", ".join(_num(l) if l == h else f"{_num(l)} to {_num(h)}" for l, h in zip(lo, hi)))
    if pl["spans"]:
        lines.append("  Over each recorded still span, the largest change of any one value of each signal (a signal "
                     "that did not change is left out):")
        for a0, b0 in pl["spans"]:
            ch = [f"{name} {_num(c)}" for name, a in arrs.items()
                  if (c := float((a[a0:b0 + 1].max(axis=0) - a[a0:b0 + 1].min(axis=0)).max())) > 0]
            lines.append(f"    {frame_time(ep, a0):.2f}-{frame_time(ep, min(b0, n - 1)):.2f}s: "
                         + ("; ".join(ch) if ch else "none changed"))
    return ("\nOTHER RECORDED SIGNALS: every other number the dataset records per frame, under the dataset's own "
            "name, with the range each of its values takes over the episode (one that never changes is given as its "
            "value). They are not interpreted for you: read what each is from its name and the robot's description "
            "above. Like the rest of the recording they are claims to check against the video; a camera carried by "
            "something they show moving (a mobile base, a torso) moves with it.\n" + "\n".join(lines))


BETWEEN_INSTANTS = (
    "\nBETWEEN INSTANTS: reconstruct, do not smooth. A brief event can fall between two "
    "instants. When the scene in the frames differs between consecutive instants (something held "
    "is released, moved or gone; a contact is made or broken), an event happened in that "
    "interval: place it there and say in its notes that it is inferred. Do not fill an unseen interval with the "
    "expected, competent version of the task; a change that does not fit smooth progress may "
    "be a slip, drop, knock or failed grasp, and should be weighed against the frames before "
    "and after, the other side, and where the object ends up. Where the evidence does not "
    "settle it, say so in a note instead of defaulting to the charitable "
    "reading, and never invent an event when consecutive instants are consistent. An event you "
    "infer between instants but cannot confirm from the frames around it goes in the timeline as "
    "inferred; it is not by itself an operator mistake or a data issue.")


def _motion_table(ep: dict, pl: dict) -> str:
    """The recorded motion between consecutive sampled instants, as a claim to check against the
    cameras. Mounted cameras are rigid on their gripper, so a real move shows in that camera."""
    names, kind, n = actors(ep), state_kind(ep), _rig_nouns(rig(ep))
    st = ep["state"][:pl["n"]]
    rows = []
    if kind == "ee_pose":
        for r in ms.recorded_motion(st, pl["ks"], names):
            parts = [f"{a} {g['move_cm']:.1f}, {g['max_step_cm']:.1f}, {g['turn_deg']:.0f}, "
                     f"{g['open_a']:.2f}>{g['open_b']:.2f}" for a, g in r["grippers"].items()]
            rows.append(f"  {frame_time(ep, r['a']):.2f}-{frame_time(ep, r['b']):.2f}s | " + " | ".join(parts))
        what = ("Each row gives, per gripper: moved (cm, the straight-line distance between the recorded "
                "positions at the two instants), largest single-frame step (cm, the biggest recorded jump "
                "between two consecutive frames inside the interval), turned (deg, the recorded rotation "
                "between the two instants), and the opening at the two instants written start>end")
        clear = ("a view that clearly shifts or turns over an interval where the recorded move is near 0 cm "
                 "AND the recorded turn is near 0 deg (the pose stopped updating); a recorded single-frame "
                 "step of several cm with no jump in the view at that moment")
    else:
        for r in ms.recorded_joint_motion(st, pl["ks"], names):
            parts = [f"{a} {g['max_deg']:.1f}, {g['max_step_deg']:.1f}, {g['grip_a']:.2f}>{g['grip_b']:.2f}"
                     for a, g in r["arms"].items()]
            rows.append(f"  {frame_time(ep, r['a']):.2f}-{frame_time(ep, r['b']):.2f}s | " + " | ".join(parts))
        what = ("Each row gives, per arm: joints moved up to (deg, the largest change of any one joint "
                "between the two instants), largest single-frame step (deg, the biggest change of any joint "
                "between two consecutive frames inside the interval), and the gripper value at the two "
                "instants written start>end. Joint angles tell you whether and when an arm moved, not where "
                "its gripper is")
        clear = ("an arm that clearly moves in the video over an interval where every joint is recorded as "
                 "near 0 deg (the recording stopped updating), or the reverse; a single-frame joint step "
                 "of many degrees with no matching jump in the video")
    step_ms = 1000.0 / ep_fps(ep)
    grip = (ep["context"].get("gripper_value") or
            "the dataset's own number (units and direction not documented)").rstrip(".")
    return (
        f"\nRECORDED MOTION, from the dataset's state, one row per interval between consecutive instants "
        f"you receive ({step_ms:.0f} ms per recorded frame). {what}; the gripper/opening value is "
        f"{grip}. This is the recording's claim, not a "
        f"fact: check it against the video. A camera mounted on {n['an_actor']} shifts when that "
        f"{n['actor']} moves. The gripper value is the recorded jaw position; whether anything is held is "
        "read from the frames, never from the value. A disagreement between the recording and the video is "
        f"a data issue only when it is clear: {clear}; or fingers that clearly open or close while the value "
        "stays flat, or a value that swings from open to shut while the fingers stay still. Do not judge "
        "whether a recorded motion is too small or too large from how much a view changes: apparent motion "
        "depends on the lens, the distance to the scene and the direction of travel. Likewise never compare "
        "how open the fingers look with the gripper number: its scale is not a picture of how wide the "
        "fingers look, so only the timing of a change can be compared with the video.\n"
        + "\n".join(rows))


def ego_annotation_block(ctx: dict) -> str:
    goal = (ctx.get("instruction") or "").strip()
    subs = ctx.get("annotation_subtasks") or []
    if not goal and not subs:
        return ("\nTHE DATASET'S ANNOTATION FOR THIS EPISODE: none; the dataset ships no task description for this "
                "clip. Infer the activities from the footage alone and leave goal_alignment out.\n")
    lines = [f"  {x['t0']:.1f}-{x['t1']:.1f}s  {x['label']}" + ("" if x.get("ok", True) else "  (marked unsuccessful)")
             for x in subs]
    return ("\nTHE DATASET'S ANNOTATION FOR THIS EPISODE (claims to check, see ABOUT THE DATASET'S ANNOTATION above):\n"
            + (f"  goal: \"{goal}\"\n" if goal else "")
            + ("  subtasks, with the times the dataset gives:\n" + "\n".join(lines) + "\n" if lines else "")
            + (f"  about these annotations: {ctx['annotation_note'].strip()}\n" if ctx.get("annotation_note") else ""))


def build_prompt(ep: dict, pl: dict, *, cell_w: int, cell_h: int, example_dir=None) -> tuple[str, str]:
    """(fixed, episode): the shared instructions (output schema, what the episode is, the data contract),
    identical for every episode of the dataset, then the facts about THIS episode (rig, cameras, frames,
    recorded state, instruction)."""
    ctx = ep["context"]
    r = rig(ep)
    n = _rig_nouns(r)
    robot = ctx.get("robot_type")
    what = f" ({robot})" if robot else ""
    k = len(actors(ep))
    who = n["who"] if r in ("teleop_arms", "ego_head") else (
        # one camera says nothing about how many grippers the rig has: a two-gripper rig's upload can carry one
        # gripper's footage, and its other gripper then appears in that camera, held in the other hand
        "a person does the task with one or two handheld grippers, and this recording has one gripper's camera"
        if k == 1
        else f"a person holds {k} handheld grippers, one per hand, and does the task with them")
    kind_of = ("one clip of first-person human video from the {d} dataset, collected to train robots and world "
               "models" if r == "ego_head" else "one episode of a robot-learning demonstration from the {d} dataset{w}")
    intro = (
        f"You are labelling {kind_of.format(d=ctx.get('dataset'), w=what)}: {who}. You see the episode exactly as "
        "the dataset ships it, from its first recorded frame to its last; nothing was trimmed, cleaned or edited.\n"
        + (f"How the dataset cuts its recordings into episodes: {ctx['collection_note'].strip()}\n"
           if ctx.get("collection_note") else "")
        + "\n" + camera_desc(ep) + "\n")
    cams = ctx.get("cameras") or {}
    c0 = cams.get(anchor(ep), {})
    native = (c0.get("width") or "native", c0.get("height") or "resolution")
    given = (ctx.get("instruction") or "").strip()
    label = "; ".join(ctx.get("task_label") or [])
    given_block = ""
    if r == "ego_head":
        given_block = ego_annotation_block(ctx)
    elif given:
        # the rules for using the instruction are the same for every episode and live in the cached
        # instructions (instruction_rules); only the instruction itself belongs to the episode
        given_block = ("\nTHE TASK FOR THIS EPISODE WAS GIVEN TO YOU as the dataset's per-episode instruction:\n"
                       f"  \"{given}\"\nHow to use it is set out under ABOUT THE EPISODE'S INSTRUCTION above.\n")
        if ctx.get("instruction_note"):
            given_block += ctx["instruction_note"].strip() + "\n"
        elif label:
            given_block += (f"The dataset's coarse task label for this episode is \"{label}\"; the "
                            "instruction above is the dataset's per-episode annotation of it, and "
                            "the outcome is graded against it.\n")
    if ctx.get("uploader_annotation"):
        # notes the person who uploaded the episode sent with it (a note file beside a video, an annotation
        # channel in an MCAP), in whatever form they came
        given_block += ("\nTHE UPLOADER'S OWN NOTES FOR THIS EPISODE, as sent. They are claims to check against the "
                        "video, not ground truth; where the video contradicts them, record it as a data issue:\n"
                        + ctx["uploader_annotation"].strip() + "\n")
    return (prompts.fixed_instructions(r, has_instruction=bool(given)) + prompts.example_block(r, example_dir),
            EPISODE_HEADER + intro + sampling_desc(ep, pl, cell_w, cell_h, native) + "\n" + given_block)


EPISODE_HEADER = "\n\nTHE EPISODE TO LABEL.\n\n"


def build_request(ep_dir: Path, *, detail: str = "high", gate=None, cell_w: int | None = None,
                  max_cell_w: int | None = None, grid_cols: int = 4, grid_quality: int = 80,
                  example_dir=None) -> dict:
    """Everything the harness sends for one episode (content parts), and what it records about it. cell_w fixes
    the cell width; max_cell_w (the routed width, label/route.py) replaces the rig's default widest cell."""
    ep = load(ep_dir)
    pl = plan(ep)
    imgs = frames(ep, pl, gate)
    any_img = next(iter(imgs.values()))[pl["ks"][0]]
    if len(views(ep)) == 1:
        # one camera: a grid row holds 6 instants (still under 2048 px wide), halving the per-image overhead
        grid_cols = max(grid_cols, 6)
    cam_labels = [cam_name(ep, v) for v in order_views(imgs)]
    # An explicit cell width is used as given. The rig's default is the largest width whose grids fit the
    # request's image-size cap: a long episode at 448 px can exceed it, and is then sent at the next step
    # down rather than refused. Episodes that fit are unchanged.
    widths = [cell_w] if cell_w else [w for w in CELL_W_STEPS if w <= (max_cell_w or GRID_CELL_W_BY_RIG[rig(ep)])]
    if max(widths) >= CONTACT_BELOW_W:
        pl["contact"] = []   # wide cells already show contact in detail
    contact = [(k, vs, fullres_stack(ep, imgs, k, "just after a sharp gripper change", frame_time(ep, k), vs))
               for k, vs in ((k, [v for v in vs if recording_at(ep, v, k)]) for k, vs in contact_views(ep, pl)) if vs]
    budget = (IMAGE_LIMIT_BYTES / IMAGE_SIZE_INFLATION - DETAIL_VIEW_BYTES_MAX
              - sum(len(j) for _, _, j in contact))
    for cell_w in widths:
        steps = timesteps(ep, pl, imgs, cell_w)
        cell_h = int(round(any_img.height * cell_w / any_img.width / 2)) * 2
        fixed, episode = build_prompt(ep, pl, cell_w=cell_w, cell_h=cell_h, example_dir=example_dir)
        content, n_grids, grid_bytes = mf.build_content(fixed, episode, steps, cam_labels, grid_cols, detail,
                                                        grid_quality, gutter=GRID_GUTTER, header=GRID_HEADER)
        if grid_bytes <= budget:
            break
    prompt = fixed + episode
    extra_bytes = 0
    # the first frame, the contact views in time order, then the last frame
    views_sent = []
    for k, name in ((pl["ks"][0], "first frame"), (pl["ks"][-1], "last frame")):
        here = [v for v in order_views(imgs) if recording_at(ep, v, k)]
        views_sent.append((k, f"{name} of the episode",
                           cam_labels if len(here) == len(imgs) else [cam_name(ep, v) for v in here],
                           fullres_stack(ep, imgs, k, name, frame_time(ep, k), None if len(here) == len(imgs) else here)))
    views_sent[1:1] = [(k, "just after a sharp change of the recorded gripper value",
                        [cam_name(ep, v) for v in order_views(vs)], jpg) for k, vs, jpg in contact]
    for k, what, names, jpg in views_sent:
        extra_bytes += len(jpg)
        # the first and last views name every camera; a contact view names only its own cameras
        cams = (f"cameras {', '.join(names)} stacked top to bottom" if names is cam_labels or len(names) > 1
                else f"camera {names[0]}")
        content.append({"type": "text", "text": f"=== detail view, {what}, t={frame_time(ep, k):.2f}s | {cams} ==="})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii"),
            "detail": detail}})
    return {"content": content, "prompt": prompt, "plan": pl, "n_grids": n_grids,
            "n_images": n_grids + len(views_sent), "image_bytes": grid_bytes + extra_bytes,
            "contact_s": [round(frame_time(ep, k), 3) for k in pl["contact"]],
            "given_prompt": (ep["context"].get("instruction") or "").strip() or None,
            "task_label": ep["context"].get("task_label"), "cam_labels": cam_labels,
            "cell": [cell_w, cell_h], "timesteps": [round(frame_time(ep, k), 3) for k in pl["ks"]],
            "still_spans": describe_spans(ep, pl["spans"]), "views": views(ep),
            "sampling": f"{rig(ep)}-every-{SAMPLE_EVERY_S[rig(ep)]:g}s"}
