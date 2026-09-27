"""Prepare builddotai/Egocentric-100K clips as episode sidecars (rig ego_head, no recorded state).

    python -m prepare egocentric100k sample --out EPISODES --hours 5.5 --seed 41 [--list LIST] [--raw RAW] [--jobs N]
    python -m prepare egocentric100k prepare --episodes configs/slices/egocentric100k.txt --out EPISODES [--raw RAW]
        [--jobs N] [--force]

An episode list has one "<shard> <clip file>" per line (factory004/worker025/part268.tar
factory_004_worker_025_0045.mp4). The dataset is 2M three-minute head-camera clips (456x256 H.265, 30 fps) packed
into tar shards of about 1 GB, one folder per factory and worker. Downloading shards to sample clips would pull
about 75 clips per clip kept, so this reads the tar headers with HTTP range requests and downloads only the clip
and its metadata, into RAW/<shard without .tar>/<clip file> and <clip>.json. It writes EPISODES/episode_<clip>/
with context.json (no instruction, the dataset ships none; the clip's metadata under source.clip), sources.json
(pointing at the downloaded clip) and, only when the clip's frames are not on the exact 30 fps grid, times.npz
with their real times.

`sample` takes one clip per worker, workers round robin across factories in a seeded random order (so every
factory is represented before any factory gets a second worker), a random shard of that worker and a random
clip among its first 60, prepares them as it draws until the clip durations reach --hours, and writes the list
of what it prepared to LIST (default EPISODES.txt). The shard listing is cached in RAW/shards.json. The dataset
asks you to accept its terms on Hugging Face, so HF_TOKEN must be set.
"""
from __future__ import annotations

import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np

from prepare import cli
from prepare import hub

REPO = "builddotai/Egocentric-100K"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
COLLECTION_NOTE = ("the dataset is continuous head-camera footage of a worker's shift, split into consecutive "
                   "clips of about 3 minutes wherever the clock falls; a clip is a window of ongoing work, not a "
                   "task episode, and the dataset ships no task annotation for it.")
CAMERA_DESC = ("the fisheye camera worn on the worker's head, looking forward and down at their hands and the "
               "work in front of them")


def _get(url: str, rng: tuple[int, int] | None = None) -> tuple[bytes, dict]:
    return hub.get(url, rng, timeout=120)


def list_shards(cache: Path) -> list[dict]:
    if cache.exists():
        return json.loads(cache.read_text())
    out, url = [], API + "?recursive=true&expand=false"
    while url:
        body, headers = _get(url)
        out += [x for x in json.loads(body) if x["type"] == "file" and x["path"].endswith(".tar")]
        link = headers.get("Link") or headers.get("link") or ""
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
        print(f"listed {len(out)} shards", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def tar_members(path: str, max_members: int, until: str | None = None) -> list[tuple[str, int, int]]:
    """(name, size, data offset) for the shard's first members, reading 512-byte headers only; with `until`,
    up to and including that member and the one after it (a clip's .json follows its .mp4)."""
    url, off, out = hub.resolve(REPO, path), 0, []
    while len(out) < max_members:
        if until and len(out) >= 2 and out[-2][0] == until:
            break
        h, _ = _get(url, (off, off + 511))
        name = h[:100].rstrip(b"\0").decode(errors="replace")
        if not name or len(h) < 512:
            break
        size = int(h[124:136].rstrip(b"\0 ").decode() or "0", 8)
        out.append((name, size, off + 512))
        off += 512 + (size + 511) // 512 * 512
    return out


def episode_dir_name(member: str) -> str:
    """factory_004_worker_025_0045.mp4 -> episode_factory_004_worker_025_0045"""
    return f"episode_{member[:-4]}"


def clip_files(raw: Path, shard: str, member: str) -> tuple[Path, Path]:
    d = raw / shard.removesuffix(".tar")
    return d / member, d / (member[:-4] + ".json")


def download(shard: str, raw: Path, member: str | None = None, seed: int | None = None) -> str | None:
    """One clip of the shard and its metadata into RAW; returns the clip's member name. With member, that clip
    (skipped when already in RAW); with seed, a random clip among the shard's first 60."""
    if member and clip_files(raw, shard, member)[0].exists():
        return member
    if member:
        members = tar_members(shard, 10**6, until=member)
        clips = [m for m in members if m[0] == member]
    else:
        rnd = random.Random(seed)
        members = tar_members(shard, 2 * rnd.randint(1, 60))
        clips = [m for m in members if m[0].endswith(".mp4")]
    if not clips:
        return None
    name, size, data_off = clips[-1]
    meta = next((m for m in members if m[0] == name[:-4] + ".json"), None)
    mp4, js = clip_files(raw, shard, name)
    if not mp4.exists():
        url = hub.resolve(REPO, shard)
        clip_meta = json.loads(_get(url, (meta[2], meta[2] + meta[1] - 1))[0]) if meta else {}
        mp4.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(clip_meta, indent=1))
        part = mp4.with_suffix(".part")
        part.write_bytes(_get(url, (data_off, data_off + size - 1))[0])
        part.replace(mp4)
    return name


def prepare_clip(shard: str, member: str, raw: Path, out_root: Path) -> dict:
    """The episode folder of one downloaded clip; returns its context."""
    import av
    mp4, js = clip_files(raw, shard, member)
    clip_meta = json.loads(js.read_text()) if js.exists() else {}
    ep = out_root / episode_dir_name(member)
    ep.mkdir(parents=True, exist_ok=True)
    with av.open(str(mp4)) as c:
        st = c.streams.video[0]
        pts = sorted(p.pts for p in c.demux(st) if p.pts is not None)
        tb, w, h = st.time_base, st.codec_context.width, st.codec_context.height
    n = len(pts)
    fps = float(clip_meta.get("fps") or 30.0)
    ctx = {"dataset": REPO, "profile": "ego_head", "state_kind": "none", "episode_id": ep.name,
           "robot_type": None, "fps": fps, "n_state_frames": n, "instruction": None,
           "task_label": [f"{clip_meta.get('factory_id', '')} {clip_meta.get('worker_id', '')}".strip()],
           "cameras": {"exo": {"name": "head", "width": w, "height": h, "desc": CAMERA_DESC}},
           "source": {"shard": shard, "member": member, "clip": clip_meta},
           "collection_note": COLLECTION_NOTE}
    step = Fraction(1) / Fraction(fps).limit_denominator(1000) / tb
    grid = step.denominator == 1 and bool(pts) and pts[0] == 0 and all(p == k * int(step) for k, p in enumerate(pts))
    if not grid:
        t = np.asarray([float(p * tb) for p in pts]) - float(pts[0] * tb)
        np.savez(ep / "times.npz", exo=t, exo_pts=np.asarray(pts, dtype=np.int64))
        ctx["real_times"] = "times.npz"
    src = {"exo": {"packed": str(mp4.resolve()), "base_s": 0.0, "n_frames": n}}
    (ep / "sources.json").write_text(json.dumps(src, indent=1))
    (ep / "context.json").write_text(json.dumps(ctx, indent=1))
    return ctx


def prepare_one(shard: str, member: str, raw: Path, out_root: Path, force: bool = False) -> str:
    if not force and (out_root / episode_dir_name(member) / "context.json").exists():
        return "skip"
    if download(shard, raw, member=member) is None:
        raise FileNotFoundError(f"{member} is not in {shard}")
    prepare_clip(shard, member, raw, out_root)
    return "ok"


def drawn(shard: str, seed: int, raw: Path, out_root: Path) -> dict | None:
    """Download and prepare one random clip of the shard; its context, or None when that fails (printed)."""
    try:
        member = download(shard, raw, seed=seed)
        if member is None:
            return None
        prepare_one(shard, member, raw, out_root)
        return json.loads((out_root / episode_dir_name(member) / "context.json").read_text())
    except Exception as e:  # a drawn clip that cannot be read is a finding about the data; listed, not hidden
        print(f"FAILED {shard}: {type(e).__name__}: {e}"[:400], file=sys.stderr, flush=True)
        return None


def cmd_sample(a) -> int:
    a.out.mkdir(parents=True, exist_ok=True)
    shards = list_shards(a.raw / "shards.json")
    by_worker: dict[tuple[str, str], list[str]] = {}
    for x in shards:
        f, w = x["path"].split("/")[:2]
        by_worker.setdefault((f, w), []).append(x["path"])
    by_factory: dict[str, list[str]] = {}
    for f, w in by_worker:
        by_factory.setdefault(f, []).append(w)
    rnd = random.Random(a.seed)
    for ws in by_factory.values():
        rnd.shuffle(ws)
    factories = sorted(by_factory)
    rnd.shuffle(factories)
    order = []                                   # round robin: one worker from every factory, then again
    for i in range(max(len(v) for v in by_factory.values())):
        order += [(f, by_factory[f][i]) for f in factories if i < len(by_factory[f])]
    print(f"{len(shards)} shards, {len(by_worker)} workers, {len(factories)} factories", flush=True)
    got, secs, k = [], 0.0, 0
    with ThreadPoolExecutor(a.jobs) as ex:
        while secs < a.hours * 3600 and k < len(order):
            batch = order[k:k + a.jobs]
            k += len(batch)
            jobs = [(rnd.choice(by_worker[fw]), rnd.randrange(1 << 30)) for fw in batch]
            for ctx in ex.map(lambda j: drawn(j[0], j[1], a.raw, a.out), jobs):
                if ctx:
                    got.append(ctx)
                    secs += ctx["n_state_frames"] / ctx["fps"]
            print(f"clips {len(got)} hours {secs / 3600:.2f}", flush=True)
    lst = a.list or Path(f"{a.out}.txt")
    lst.write_text("\n".join(sorted(f"{c['source']['shard']} {c['source']['member']}" for c in got)) + "\n")
    print(json.dumps({"clips": len(got), "hours": round(secs / 3600, 2), "list": str(lst),
                      "factories": len({c["source"]["clip"].get("factory_id") for c in got})}), flush=True)
    return 0


def main() -> int:
    ap, sub = cli.parser("egocentric100k", __doc__)
    s = sub.add_parser("sample", help="draw and prepare clips until about --hours, and write their list")
    s.add_argument("--out", type=Path, required=True, help="where the episode folders are written")
    s.add_argument("--list", type=Path, default=None, help="the episode list to write (default OUT.txt)")
    s.add_argument("--raw", type=Path, default=Path("data/raw/egocentric100k"), help="where downloaded files are kept")
    s.add_argument("--hours", type=float, default=5.5)
    s.add_argument("--jobs", type=int, default=8, help="clips prepared at once (default 8)")
    s.add_argument("--seed", type=int, default=41)
    cli.add_prepare(sub, "egocentric100k", "one '<shard> <clip file>' per line")
    a = ap.parse_args()
    if a.cmd == "sample":
        return cmd_sample(a)
    items = [line.split() for line in cli.read_list(a.episodes)]
    a.out.mkdir(parents=True, exist_ok=True)
    return cli.run(items, lambda it: prepare_one(it[0], it[1], a.raw, a.out, a.force), a.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
