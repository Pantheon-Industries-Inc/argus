"""Static build of the board, for a CDN with no server behind it.

The page is board/serve.py's own template rendered with a static data source (render_index), and the rail
records are serve.py's own rail_records, so the static board cannot drift from the served one. Everything the
page loads is a plain file:

  <out>/<build_id>/index.html            the page (relative bases: data/ and ../media/)
  <out>/<build_id>/index.public.html     the same page with absolute bases, when --public-base is given
  <out>/<build_id>/data/index.json       every episode's dataset, file and length (tabs, totals, search)
  <out>/<build_id>/data/lists/<ds>.json  one dataset's rail records, each with its _media and _frames paths
  <out>/<build_id>/data/ep/<file>        each episode file, byte for byte as the board stores it
  <out>/<build_id>/data/compare/...      other models' labels when the board has them (board/build.py compare/):
                                         index.json, metrics.json, <key>/<file> and lists/<key>.json (one model's
                                         rail records, for "Labels by"), as the live board serves them
  <out>/<build_id>/data/hands/<file>     the hand pose of each head-camera episode when the board has it
                                         (board/build.py hands/), timed against the web copy of its clip
  <out>/<build_id>/data/keypoints/index.json  the head-camera episodes with a hand keypoints download, each with the
                                         path of its file under media/k/
  <out>/<build_id>/BUILD.json            provenance, counts, and which media files are still missing
  <out>/media/v/<cam>/<eid>.<hash>.mp4   web copies of the clips (board/clips.py's own files, stream-copied)
  <out>/media/f/<cam>/<eid>/<ms>.<hash>.jpg  goal frames, the same JPEGs the live /api/frame returns
  <out>/media/k/<eid>.<hash>.json        the hand keypoint downloads, byte for byte as board/build.py wrote them
                                         (hand_keypoints/: the dataset video's own pixels and frame times, so no
                                         re-timing), named by their content like the other media, so a new build
                                         does not upload them again

Media names carry a hash of the source clip's size and mtime and of the encode settings, so a media file
never changes under its name (cache it forever), a re-cut clip gets a new name, and builds share media.
The site step writes these names whether or not the media exists yet, so a build made before the
transcode finishes is already complete once it does.

Subcommands (all resumable):
  measure  probe every clip the board shows, sample-encode a few per dataset and camera, and project the
           transcode's output size and wall time
  media    extract goal frames and transcode clips; skips anything already done
  site     write a build's page and data
  status   how much of a build's media exists

  python -m board static measure --board BOARD --clips CLIPS
  python -m board static media --board BOARD --clips CLIPS [--limit-per-dataset 50] [--dataset NAME]
  python -m board static site --board BOARD --clips CLIPS [--public-base https://example.org/board/] [--title T]

Every subcommand needs ffmpeg and ffprobe. Goal frames are cut by board/serve.py's extract_frame, so
BOARD_FFMPEG_CONCURRENCY (default 4) also bounds how many are cut at once. OUT (--out) defaults to BOARD/static.
The page's header shows --title (default "Data Board") and the board's name from BOARD/manifest.json. Test a
build with any static server that answers byte ranges, e.g. `npx http-server BOARD/static -p 8991`, then open
http://localhost:8991/<build_id>/.
board/publish.sh uploads a build with rclone (to an S3-compatible bucket, for example).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

from board import clips as bc
from board import hands as hands_overlay
from board import serve as sa

HERE_DIR = Path(__file__).resolve().parent
TITLE = "Data Board"

FFMPEG = sa.FFMPEG
FFPROBE = (str(Path(FFMPEG).with_name("ffprobe")) if FFMPEG and Path(FFMPEG).with_name("ffprobe").exists()
           else shutil.which("ffprobe"))
# the clips' recipe is board/clips.py's; its name is in every media file's name, so a new recipe gets new names
ENC_TAG = bc.ENC_TAG
FRAME_TAG = "frame-640-v1"   # serve.extract_frame at w=640


# ---------------------------------------------------------------- what the page shows

clip_path = sa.clip_path   # the source clip board/serve.py's /api/video serves for a camera value


def media_key(cam: str) -> str:
    return cam if cam in ("left", "right") else "exo"


def shown_cams(d: dict) -> list:
    """Camera values the page's renderEp requests from /api/video, main camera first."""
    views = d.get("camera_views") or ["exo", "left", "right"]
    main = "exo" if "exo" in views else views[0]
    cams = [main]
    if d.get("_rig") != "ego_head":      # a head camera is shown alone
        for v in views:
            if v != main and v != "exo":
                c = "left" if v == "left" else "right"
                if c not in cams:
                    cams.append(c)
    return cams


def goal_times(d: dict) -> list:
    """Times the page shows a goal frame for (renderEp's goal-frame panels)."""
    comp = d.get("completion") or {}
    tasks = [t for t in (d.get("tasks") or []) if t and (t.get("task") or t.get("start_s") is not None)]
    out = []
    if tasks:
        out = [t.get("completed_at_s") for t in tasks if t.get("completed_at_s") is not None]
    else:
        if comp.get("completed_at_s") is not None:
            out.append(comp["completed_at_s"])
        undone = str(comp.get("task_completed") or "").lower() == "success_then_undone"
        if undone and comp.get("goal_reached_at_s") is not None:
            out.append(comp["goal_reached_at_s"])
    res = []
    for t in out:
        try:
            res.append(float(t))
        except (TypeError, ValueError):
            pass
    return res


def ms_key(t: float) -> int:
    return math.floor(t * 1000 + 0.5)   # JavaScript's Math.round(t * 1000)


def src_sig(p: Path) -> str:
    try:
        st = p.stat()
        return f"{st.st_size}-{st.st_mtime_ns}"
    except OSError:
        return "missing"


def _h(s: str, n: int) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def q(s: str) -> str:
    return urllib.parse.quote(s, safe="")


def video_rel(eid: str, key: str, sig: str) -> str:
    return f"v/{key}/{q(eid)}.{_h(ENC_TAG + '|' + sig, 10)}.mp4"


def frame_rel(eid: str, key: str, ms: int, sig: str) -> str:
    return f"f/{key}/{q(eid)}/{ms}.{_h(FRAME_TAG + '|' + sig, 8)}.jpg"


def compare_dir(qa: Path, given: Path | None = None) -> Path | None:
    """The board's other models' labels (board/build.py compare/), when it has them."""
    c = given if given is not None else qa.parent / "compare"
    return c if (c / "index.json").exists() else None


def hands_dir(qa: Path, given: Path | None = None) -> Path | None:
    """The board's hand pose files (board/build.py hands/), when it has them."""
    h = given if given is not None else qa.parent / "hands"
    return h if h.is_dir() and any(h.glob("episode_*.json")) else None


def keypoints_dir(qa: Path, given: Path | None = None) -> Path | None:
    """The board's hand keypoint downloads (board/build.py hand_keypoints/), when it has them."""
    k = given if given is not None else qa.parent / "hand_keypoints"
    return k if (k / "index.json").exists() else None


def plan(qa: Path, clips: Path, compare: Path | None = None):
    """Every listed episode with its rail record, dataset, rank in its dataset's rail, and media. The goal frames
    include those of other models' labels (compare/), which the page shows when a comparison is chosen."""
    recs = sa.rail_records(qa)
    cmp_keys = [p for p in compare.iterdir() if p.is_dir()] if compare is not None else []
    rank = Counter()
    eps = []
    for rec in recs:
        ds = rec.get("dataset") or ""
        d = json.loads((qa / rec["file"]).read_text())
        eid = (d.get("_meta") or {}).get("episode_id") or ""
        media, frames = {}, {}
        if eid:
            cams = shown_cams(d)
            for cam in cams:
                src = clip_path(clips, eid, cam)
                key = media_key(cam)
                media[key] = {"src": src, "rel": video_rel(eid, key, src_sig(src)), "main": cam == cams[0]}
                # the camera's first frame, the video's poster: the player shows footage from the first paint
                frames[f"{key}|0"] = {"src": src, "t": 0.0, "rel": frame_rel(eid, key, 0, src_sig(src))}
            main = cams[0]
            msrc = clip_path(clips, eid, main)
            times = goal_times(d)
            for kd in cmp_keys:
                cf = kd / rec["file"]
                if cf.exists():
                    times += goal_times(json.loads(cf.read_text()))
            for t in times:
                k = f"{media_key(main)}|{ms_key(t)}"
                frames[k] = {"src": msrc, "t": t,
                             "rel": frame_rel(eid, media_key(main), ms_key(t), src_sig(msrc))}
        eps.append({"rec": rec, "ds": ds, "rank": rank[ds], "eid": eid, "media": media, "frames": frames})
        rank[ds] += 1
    return eps


# ---------------------------------------------------------------- ffmpeg

def probe(p: Path) -> dict:
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries",
                        "stream=codec_type,codec_name,width,height,pix_fmt,nb_frames,avg_frame_rate,bit_rate"
                        ":format=duration,size",
                        "-of", "json", str(p)], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe {p}: {r.stderr.strip()[:200]}")
    j = json.loads(r.stdout or "{}")
    v = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), {})
    fmt = j.get("format", {})
    return {"codec": v.get("codec_name"), "w": int(v.get("width") or 0), "h": int(v.get("height") or 0),
            "pix_fmt": v.get("pix_fmt"), "frames": int(v.get("nb_frames") or 0),
            "fps": v.get("avg_frame_rate"), "dur": float(fmt.get("duration") or 0),
            "v_bps": int(v.get("bit_rate") or 0),
            "bytes": int(fmt.get("size") or p.stat().st_size),
            "audio": any(s.get("codec_type") == "audio" for s in j.get("streams", []))}


def transcode(src: Path, dst: Path, threads: int, main: bool = True) -> dict:
    """Web copy of src at dst: H.264 with the index at the front (faststart), every source timestamp kept (the
    page syncs its cameras by time).

    The board's clips (board/clips.py) already are that, so they are stream-copied into a faststart mp4 and never
    re-encoded: the static board plays the same pictures at the same times as the live one (-copyts keeps a clip
    that starts after 0, a camera that started recording late, where it is), and a second lossy pass would only
    cost quality. A clip made some other way (not H.264 4:2:0, or larger than the recipe allows) is encoded with
    board/clips.py's recipe, as the page's main camera or a side one (main)."""
    t0 = time.time()
    sp = probe(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name("." + dst.name + ".part.mp4")
    if _compliant(sp):
        mode = "copy"
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-copyts", "-i", str(src),
               "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-movflags", "+faststart", str(part)]
    else:
        mode = "encode"
        amap = ["-map", "0:a:0"] if sp["audio"] else []
        acodec = ["-c:a", "aac", "-b:a", "96k"] if sp["audio"] else []
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(src),
               "-map", "0:v:0", *amap, "-fps_mode", "passthrough",
               *bc.video_args(sp["w"], sp["h"], main, threads), *acodec, str(part)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg {mode} {src}: {r.stderr.strip()[-300:]}")
    op = probe(part)
    warn = []
    if sp["frames"] and op["frames"] and sp["frames"] != op["frames"]:
        warn.append(f"frames {sp['frames']}->{op['frames']}")
    if abs(sp["dur"] - op["dur"]) > 0.1:
        warn.append(f"duration {sp['dur']:.3f}->{op['dur']:.3f}")
    os.replace(part, dst)
    return {"src": str(src), "dst": str(dst), "mode": mode, "src_bytes": sp["bytes"],
            "out_bytes": op["bytes"], "dur": sp["dur"], "w": sp["w"], "out_w": op["w"],
            "secs": round(time.time() - t0, 2), "warn": warn}


# ---------------------------------------------------------------- measure

def _compliant(pr: dict) -> bool:
    """H.264 4:2:0 within the recipe's largest size: what the clip builder writes, so it is copied, not re-encoded."""
    return (pr["codec"] == "h264" and pr["pix_fmt"] == "yuv420p"
            and pr["w"] <= bc.MAIN_BOX[0] and pr["h"] <= bc.MAIN_BOX[1])


def cmd_measure(a):
    """Probe every clip, then sample-encode the ones transcode() would encode and sample-copy the rest,
    and project output size and wall time from what the samples measured."""
    eps = plan(a.qa, a.clips, compare_dir(a.qa, a.compare))
    pairs = []   # (ds, key, rank, src, main)
    for e in eps:
        for key, m in e["media"].items():
            pairs.append((e["ds"], key, e["rank"], m["src"], m["main"]))
    print(f"{len(eps)} episodes, {len(pairs)} clips shown by the page; probing...", flush=True)
    t0 = time.time()
    probes = {}
    with cf.ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(probe, p[3]): p for p in pairs if p[3].exists()}
        for f in cf.as_completed(futs):
            try:
                probes[futs[f][3]] = f.result()
            except Exception as err:   # noqa: BLE001
                print("probe failed:", err, flush=True)
    print(f"probed {len(probes)} clips in {time.time() - t0:.0f}s", flush=True)
    groups = defaultdict(list)
    for p in pairs:
        mode = "no-source" if p[3] not in probes else "copy" if _compliant(probes[p[3]]) else "encode"
        groups[(p[0], p[1], mode)].append(p)
    # samples: spread evenly through each group's rail order
    samples = []
    for g, ps in sorted(groups.items()):
        ok = [p for p in ps if p[3] in probes]
        n = min(a.samples, len(ok))   # the no-source group has nothing to sample
        for i in range(n):
            samples.append((g, ok[int(i * (len(ok) - 1) / max(1, n - 1))]))
    mdir = a.out / "_measure" / ENC_TAG
    print(f"sampling {len(samples)} clips ({a.jobs} at a time, {a.threads} threads each)",
          flush=True)
    res = defaultdict(list)
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(transcode, p[3], mdir / g[0] / g[1] / p[3].name, a.threads, p[4]): (g, p[3])
                for g, p in samples}
        for f in cf.as_completed(futs):
            g, src = futs[f]
            try:
                r = f.result()
                res[g].append(r)
                print(f"  {g[0]}/{g[1]} {src.name}: {r['mode']} {r['w']}->{r['out_w']} wide, {r['dur']:.0f}s video in "
                      f"{r['secs']:.1f}s, {r['src_bytes']/1e6:.1f} -> {r['out_bytes']/1e6:.1f} MB "
                      f"{' '.join(r['warn'])}", flush=True)
            except Exception as err:   # noqa: BLE001
                print("  sample failed:", err, flush=True)
    tot = defaultdict(float)
    rows = {}
    print(f"\n{'dataset/cam/mode':34} {'clips':>6} {'miss':>5} {'hours':>7} {'src GB':>7} {'out GB':>7} "
          f"{'job h':>6}  resolutions")
    for g, ps in sorted(groups.items()):
        pr = [(p, probes[p[3]]) for p in ps if p[3] in probes]
        hours = sum(x["dur"] for _, x in pr) / 3600
        src_b = sum(x["bytes"] for _, x in pr)
        rs = res.get(g) or []
        vid = sum(r["dur"] for r in rs) or 1
        out_per_src = (sum(r["out_bytes"] for r in rs) / max(1, sum(r["src_bytes"] for r in rs))) if rs else 1.0
        sec_per_vid = sum(r["secs"] for r in rs) / vid if rs else 0.0
        out_b = src_b * out_per_src
        job_s = hours * 3600 * sec_per_vid
        f_pr = [(p, x) for p, x in pr if p[2] < a.first]
        f_out = sum(x["bytes"] for _, x in f_pr) * out_per_src
        f_job = sum(x["dur"] for _, x in f_pr) * sec_per_vid
        rows[f"{g[0]}/{g[1]}/{g[2]}"] = {"clips": len(ps), "missing": len(ps) - len(pr), "hours": round(hours, 2),
                                         "src_gb": round(src_b / 1e9, 3), "out_gb": round(out_b / 1e9, 3),
                                         "job_hours": round(job_s / 3600, 3), "out_per_src": round(out_per_src, 3),
                                         "res": Counter(f"{x['w']}x{x['h']}" for _, x in pr).most_common(3)}
        for k, v in (("hours", hours), ("src_gb", src_b / 1e9), ("out_gb", out_b / 1e9), ("job_hours", job_s / 3600),
                     ("first_out_gb", f_out / 1e9), ("first_job_hours", f_job / 3600), ("clips", len(ps)),
                     ("missing", len(ps) - len(pr)), (g[2] + "_clips", len(pr))):
            tot[k] += v
        print(f"{g[0] + '/' + g[1] + '/' + g[2]:34} {len(ps):6d} {len(ps) - len(pr):5d} {hours:7.1f} {src_b/1e9:7.2f} "
              f"{out_b/1e9:7.2f} {job_s/3600:6.2f}  {rows[f'{g[0]}/{g[1]}/{g[2]}']['res']}")
    wall_h = tot["job_hours"] / a.jobs
    f_wall_h = tot["first_job_hours"] / a.jobs
    print(f"\ntotal: {int(tot['clips'])} clips ({int(tot['missing'])} with no source clip, "
          f"{int(tot['copy_clips'])} copied, "
          f"{int(tot['encode_clips'])} encoded), {tot['hours']:.1f} h of video, source {tot['src_gb']:.1f} GB, "
          f"projected output {tot['out_gb']:.1f} GB, projected wall time {wall_h:.2f} h at {a.jobs} jobs")
    print(f"first {a.first} episodes per dataset: projected {tot['first_out_gb']:.1f} GB, {f_wall_h:.2f} h")
    report = {"measured_at": dt.datetime.now().isoformat(timespec="seconds"), "enc": ENC_TAG,
              "jobs": a.jobs, "threads": a.threads, "groups": rows,
              "total": {**{k: round(v, 3) for k, v in tot.items()}, "wall_hours": round(wall_h, 2),
                        "first_n_wall_hours": round(f_wall_h, 2), "first_n": a.first},
              "samples": {"/".join(g): rs for g, rs in res.items()}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "measure.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {a.out / 'measure.json'}")


# ---------------------------------------------------------------- media

def cmd_media(a):
    eps = plan(a.qa, a.clips, compare_dir(a.qa, a.compare))
    sel = [e for e in eps if not a.limit_per_dataset or e["rank"] < a.limit_per_dataset]
    if a.dataset:
        sel = [e for e in sel if e["ds"] in a.dataset]
    sel.sort(key=lambda e: (e["rank"], e["ds"]))   # every dataset's first episodes first
    media_root = a.out / "media"
    media_root.mkdir(parents=True, exist_ok=True)
    logf = open(media_root / "media_log.jsonl", "a")
    lock = threading.Lock()

    # goal frames first: cheap, and the same bytes the live /api/frame returns
    fjobs = [(f["src"], f["t"], media_root / f["rel"]) for e in sel for f in e["frames"].values()]
    fjobs = [j for j in fjobs if not j[2].exists() and j[0].exists()]
    print(f"frames: {len(fjobs)} to extract", flush=True)

    def do_frame(j):
        jpg = sa.extract_frame(j[0], j[1], 640)
        if not jpg:
            return False
        j[2].parent.mkdir(parents=True, exist_ok=True)
        tmp = j[2].with_name("." + j[2].name + ".part")
        tmp.write_bytes(jpg)
        os.replace(tmp, j[2])
        return True
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        nf = sum(1 for ok in ex.map(do_frame, fjobs) if ok)
    print(f"frames: extracted {nf} of {len(fjobs)}", flush=True)

    vjobs = []
    missing_src = 0
    for e in sel:
        for key, m in e["media"].items():
            dst = media_root / m["rel"]
            if dst.exists():
                continue
            if not m["src"].exists():
                missing_src += 1
                continue
            vjobs.append((e, key, m["src"], dst))
    total = len(vjobs)
    print(f"videos: {total} to transcode ({missing_src} have no source clip), {a.jobs} jobs x {a.threads} threads",
          flush=True)
    done = {"n": 0, "vid_s": 0.0, "src": 0, "out": 0, "fail": 0}
    t0 = time.time()
    prog = media_root / "progress.json"

    def do_video(j):
        e, key, src, dst = j
        try:
            r = transcode(src, dst, a.threads, e["media"][key]["main"])
        except Exception as err:   # noqa: BLE001
            with lock:
                done["fail"] += 1
                logf.write(json.dumps({"src": str(src), "error": str(err)[:400], "at": time.time()}) + "\n")
                logf.flush()
                print(f"FAIL {src}: {str(err)[:200]}", flush=True)
            return
        with lock:
            done["n"] += 1; done["vid_s"] += r["dur"]; done["src"] += r["src_bytes"]; done["out"] += r["out_bytes"]
            r.update({"ds": e["ds"], "rank": e["rank"], "at": time.time()})
            logf.write(json.dumps(r) + "\n"); logf.flush()
            el = time.time() - t0
            left = total - done["n"] - done["fail"]
            eta = el / max(1, done["n"] + done["fail"]) * left
            print(f"[{done['n'] + done['fail']}/{total}] {e['ds']}#{e['rank']} {key} {r['mode']} {r['dur']:.0f}s "
                  f"{r['src_bytes']/1e6:.1f}->{r['out_bytes']/1e6:.1f}MB {r['secs']:.0f}s {' '.join(r['warn'])} "
                  f"| {done['vid_s']/max(1, el):.1f}x rt, eta {eta/60:.0f} min", flush=True)
            prog.write_text(json.dumps({"total": total, "done": done["n"], "failed": done["fail"],
                                        "video_hours_done": round(done["vid_s"] / 3600, 2),
                                        "src_gb": round(done["src"] / 1e9, 2), "out_gb": round(done["out"] / 1e9, 2),
                                        "elapsed_min": round(el / 60, 1), "eta_min": round(eta / 60, 1),
                                        "updated": dt.datetime.now().isoformat(timespec="seconds")}))
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(do_video, vjobs))
    print(f"videos: {done['n']} done, {done['fail']} failed, {done['vid_s']/3600:.2f} h of video, "
          f"{done['src']/1e9:.2f} GB -> {done['out']/1e9:.2f} GB in {(time.time()-t0)/60:.1f} min", flush=True)
    logf.close()
    return 1 if done["fail"] else 0


# ---------------------------------------------------------------- site

def _git_commit() -> str:
    r = subprocess.run(["git", "-C", str(HERE_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def build_id_for(qa: Path, files: list, compare: Path | None = None, hands: Path | None = None,
                 keypoints: Path | None = None) -> str:
    h = hashlib.sha1()
    if keypoints is not None:
        h.update(hashlib.sha1((keypoints / "index.json").read_bytes()).digest())
    for name in ("index.json", "metrics.json"):
        if compare is not None and (compare / name).exists():
            h.update(hashlib.sha1((compare / name).read_bytes()).digest())
    if hands is not None:
        for f in files:
            hp = hands / f
            if hp.exists():
                st = hp.stat()
                h.update(f"hands|{f}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    newest = 0
    for f in files:
        st = (qa / f).stat()
        newest = max(newest, st.st_mtime)
        h.update(f"{f}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    h.update(hashlib.sha1(sa.INDEX_HTML.encode()).digest())
    h.update(ENC_TAG.encode() + FRAME_TAG.encode())
    return f"{dt.datetime.fromtimestamp(newest):%Y%m%d}-{h.hexdigest()[:8]}"


def media_status(eps, media_root: Path):
    st = {"videos": 0, "videos_present": 0, "videos_no_source": 0, "frames": 0, "frames_present": 0}
    missing = []
    for e in eps:
        for m in e["media"].values():
            st["videos"] += 1
            if (media_root / m["rel"]).exists():
                st["videos_present"] += 1
            elif not m["src"].exists():
                st["videos_no_source"] += 1
            else:
                missing.append(m["rel"])
        for f in e["frames"].values():
            st["frames"] += 1
            st["frames_present"] += (media_root / f["rel"]).exists()
    return st, missing


def cmd_site(a):
    compare = compare_dir(a.qa, a.compare)
    eps = plan(a.qa, a.clips, compare)
    files = [e["rec"]["file"] for e in eps]
    hands = hands_dir(a.qa, a.hands)
    keypoints = keypoints_dir(a.qa, a.keypoints)
    bid = a.build_id or build_id_for(a.qa, files, compare, hands, keypoints)
    out = a.out / bid
    if out.exists() and not a.force:
        print(f"{out} exists (a build id names one set of inputs); pass --force to rewrite it")
        return 1
    stage = a.out / f".{bid}.staging"
    if stage.exists():
        stage.rename(a.out / f".{bid}.staging.old.{int(time.time())}")   # never rm; set aside
    (stage / "data/lists").mkdir(parents=True)
    (stage / "data/ep").mkdir(parents=True)
    datasets = []
    by_ds = defaultdict(list)
    idx_eps = []
    for e in eps:
        rec = dict(e["rec"])
        rec["_media"] = {k: m["rel"] for k, m in e["media"].items()}
        # only frames that exist, so the page never asks for one that is not there
        rec["_frames"] = {k: f["rel"] for k, f in e["frames"].items() if (a.out / "media" / f["rel"]).exists()}
        if e["ds"] not in datasets:
            datasets.append(e["ds"])
        by_ds[e["ds"]].append(rec)
        row = [datasets.index(e["ds"]), rec["file"], rec.get("duration_s")]
        if rec["episode_id"] != rec["file"][:-5]:
            row.append(rec["episode_id"])
        idx_eps.append(row)
        # the page's view of the episode: each issue carries the family it is counted under
        view = sa.episode_view(json.loads((a.qa / rec["file"]).read_text()))
        (stage / "data/ep" / rec["file"]).write_text(json.dumps(view))
    for ds, recs in by_ds.items():
        (stage / "data/lists" / f"{ds}.json").write_text(json.dumps(recs, separators=(",", ":")))
    (stage / "data/index.json").write_text(json.dumps({"build": bid, "datasets": datasets, "eps": idx_eps},
                                                      separators=(",", ":")))
    n_cmp = 0
    if compare is not None:
        # other models' labels: the same files the live board's /api/compare serves, each episode as the page shows it
        (stage / "data/compare").mkdir()
        for name in ("index.json", "metrics.json"):
            shutil.copyfile(compare / name, stage / "data/compare" / name)
        listed = set(files)
        (stage / "data/compare/lists").mkdir()
        for kd in sorted(p for p in compare.iterdir() if p.is_dir()):
            (stage / "data/compare" / kd.name).mkdir()
            for f in kd.glob("*.json"):
                if f.name in listed:
                    view = sa.episode_view(json.loads(f.read_text()))
                    (stage / "data/compare" / kd.name / f.name).write_text(json.dumps(view))
                    n_cmp += 1
            # the model's rail records, the same the live board serves at /api/compare/list
            recs = [r for r in sa.rail_records(kd) if r["file"] in listed]
            (stage / "data/compare/lists" / f"{kd.name}.json").write_text(json.dumps(recs, separators=(",", ":")))
    hands_res = None
    if hands is not None:
        # the hand pose files, each re-timed against the web copy the static page plays (the copy keeps the clip's
        # timestamps, and retime refuses a copy whose frames or size differ); an episode whose web copy is not made
        # yet gets none until the next build
        (stage / "data/hands").mkdir()
        todo = [(e, a.out / "media" / e["media"]["exo"]["rel"]) for e in eps
                if (hands / e["rec"]["file"]).exists() and "exo" in e["media"]]

        def one(job):
            e, mp4 = job
            f = e["rec"]["file"]
            if not mp4.exists():
                return f, "no web copy of the clip yet"
            try:
                doc = hands_overlay.retime(json.loads((hands / f).read_text()), mp4)
            except (ValueError, RuntimeError) as err:
                return f, str(err)[:200]
            (stage / "data/hands" / f).write_text(json.dumps(doc, separators=(",", ":")))
            return f, None
        with cf.ThreadPoolExecutor(8) as ex:
            res = list(ex.map(one, todo))
        hands_res = {"dir": str(hands), "files": sum(1 for _, err in res if err is None),
                     "skipped": {f: err for f, err in res if err is not None}}
    kp_res = None
    if keypoints is not None:
        # the downloads, byte for byte: they are in the dataset video's own pixels and times, not the web copy's.
        # They go under media/ with a name from their content, so publishing a new build uploads only the ones
        # that changed
        (stage / "data/keypoints").mkdir()
        kidx = json.loads((keypoints / "index.json").read_text())
        listed = set(files)
        kidx["files"] = {f: v for f, v in kidx["files"].items() if f in listed and (keypoints / f).exists()}
        (a.out / "media" / "k").mkdir(parents=True, exist_ok=True)
        for f, v in kidx["files"].items():
            body = (keypoints / f).read_bytes()
            rel = f"k/{q(f[:-5])}.{hashlib.sha1(body).hexdigest()[:10]}.json"
            dst = a.out / "media" / rel
            if not dst.exists():
                part = dst.with_name(dst.name + ".part")
                part.write_bytes(body)
                part.rename(dst)
            v["path"] = rel
        (stage / "data/keypoints/index.json").write_text(json.dumps(kidx, separators=(",", ":")))
        kp_res = {"dir": str(keypoints), "files": len(kidx["files"]),
                  "bytes": sum(v["bytes"] for v in kidx["files"].values())}
    # the page asks for other models' labels, hand pose files and keypoint downloads only when the build has them
    # (no request that can only fail)
    has = {"compare": compare is not None, "hands": bool(hands_res and hands_res["files"]),
           "keypoints": bool(kp_res and kp_res["files"])}
    name = sa.board_name(a.board)
    (stage / "index.html").write_text(sa.render_index(
        a.title, {"mode": "static", "data": "data/", "media": "../media/", **has}, name))
    if a.public_base:
        base = a.public_base.rstrip("/") + "/"
        (stage / "index.public.html").write_text(sa.render_index(
            a.title, {"mode": "static", "data": f"{base}{bid}/data/", "media": f"{base}media/", **has}, name))
    st, missing = media_status(eps, a.out / "media")
    build = {"build_id": bid, "built_at": dt.datetime.now().isoformat(timespec="seconds"), "code": _git_commit(),
             "qa": str(a.qa), "clips": str(a.clips), "enc": ENC_TAG, "public_base": a.public_base,
             "episodes": len(eps), "datasets": {ds: len(r) for ds, r in by_ds.items()},
             "compare": {"dir": str(compare), "files": n_cmp} if compare is not None else None,
             "hands": hands_res, "keypoints": kp_res,
             "media": st, "missing_media": missing}
    (stage / "BUILD.json").write_text(json.dumps(build, indent=1))
    if out.exists():
        out.rename(a.out / f".{bid}.old.{int(time.time())}")
    stage.rename(out)
    (a.out / "LATEST").write_text(bid + "\n")
    print(json.dumps({k: v for k, v in build.items() if k != "missing_media"}, indent=1))
    print(f"wrote {out}; {len(missing)} media files not yet made")
    return 0


def cmd_status(a):
    eps = plan(a.qa, a.clips, compare_dir(a.qa, a.compare))
    st, missing = media_status(eps, a.out / "media")
    per = Counter()
    miss = set(missing)
    for e in eps:
        for m in e["media"].values():
            if m["rel"] in miss:
                per[e["ds"]] += 1
    print(json.dumps({"media": st, "missing_by_dataset": per}, indent=1))
    p = a.out / "media/progress.json"
    if p.exists():
        print("progress:", p.read_text())
    return 0


def main():
    ap = argparse.ArgumentParser(prog="python -m board static", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["measure", "media", "site", "status"])
    ap.add_argument("--board", type=Path, required=True, help="a board folder (its qa/ holds the episode files)")
    ap.add_argument("--clips", type=Path, required=True, help="the clips folder board clips wrote")
    ap.add_argument("--compare", type=Path, default=None, help="other models' labels (default BOARD/compare)")
    ap.add_argument("--hands", type=Path, default=None, help="hand pose files (default BOARD/hands)")
    ap.add_argument("--keypoints", type=Path, default=None,
                    help="hand keypoint downloads (default BOARD/hand_keypoints)")
    ap.add_argument("--out", type=Path, default=None, help="output root (default BOARD/static)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel ffmpeg processes (default 4)")
    ap.add_argument("--threads", type=int, default=4, help="threads per ffmpeg (default 4)")
    ap.add_argument("--limit-per-dataset", type=int, default=0, help="media: only each dataset's first N rail episodes")
    ap.add_argument("--dataset", action="append", help="media: only this dataset (repeatable)")
    ap.add_argument("--samples", type=int, default=3, help="measure: sample encodes per dataset and camera")
    ap.add_argument("--first", type=int, default=50, help="measure: also project each dataset's first N episodes")
    ap.add_argument("--public-base", help="site: absolute URL of the bucket root, for index.public.html")
    ap.add_argument("--build-id", help="site: override the content-derived build id")
    ap.add_argument("--force", action="store_true", help="site: rewrite an existing build id")
    ap.add_argument("--title", default=TITLE, help=f"site: the page title, in the header and the browser tab "
                                                   f"(default {TITLE!r})")
    a = ap.parse_args()
    a.qa = a.board / "qa"
    a.out = a.out or a.board / "static"
    if not FFMPEG or not FFPROBE:
        sys.exit("ffmpeg/ffprobe not found")
    return {"measure": cmd_measure, "media": cmd_media, "site": cmd_site, "status": cmd_status}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
