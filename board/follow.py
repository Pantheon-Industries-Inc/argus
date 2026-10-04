"""Add each label to a board as soon as its run writes it.

    python -m board follow BOARD [--every 30] [--once]

`board build` writes a board from finished runs in one go. During a long run, follow adds every new label to
BOARD/qa within --every seconds of the run writing it, as the same file a build writes (board/build.py
board_label), so the home page and the episode list show it while the run goes on. Run it beside `board serve` and
stop it when the runs are done; a final `board build` then applies what follow leaves out (reruns, comparisons and
hand pose).

Every five minutes it also tags the kinds of object not tagged yet as rigid or deformable (board/materials.py, about
a cent for a whole run; it needs the labelling keys in the environment and skips the step without them).

A manifest entry's "run" must name the run folder itself while it labels: RUNS/<dataset>/latest means the newest
finished run. An entry may also name "clips", the clips folder of its review job: follow hard-links each new
episode's clips into BOARD/clips, so one board serves the clips of several jobs.

Each label is written to a temporary name and moved into place, so the server never reads half a file. A run
output that is still being written is read again on the next pass.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from board import materials
from board.build import _path, board_label, resolve_run

MATERIALS_EVERY_S = 300


def link_clips(src: Path, dst: Path, eid: str) -> int:
    """Hard-links the episode's clips from src into dst at the same place (the scene camera at the top, the others
    in their folders: board/clips.py clip_path), copying where the two are on different disks."""
    n = 0
    for p in [src / f"{eid}.mp4", *src.glob(f"*/{eid}.mp4")]:
        if not p.is_file():
            continue
        q = dst / p.relative_to(src)
        if q.exists():
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(p, q)
        except OSError:
            shutil.copy2(p, q)
        n += 1
    return n


def follow_once(board: Path, seen: dict) -> int:
    """One pass over every run the manifest names: each run output not seen before (or changed since) becomes the
    board's episode file. seen maps an output file to the (mtime, size) it was read at. Returns the labels added."""
    manifest = json.loads((board / "manifest.json").read_text())
    here = board.resolve()
    qa = board / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    added = 0
    for entry in manifest.get("datasets", []):
        try:
            run = resolve_run(_path(entry["run"], here))
        except SystemExit:
            continue                        # latest with no finished run yet
        out = run / "out"
        if not out.is_dir():
            continue
        eps = _path(entry["episodes"], here)
        pre = entry.get("file_prefix") or ""
        info = None
        for f in sorted(out.glob("episode_*.json")):
            try:
                st = f.stat()
            except OSError:
                continue
            if seen.get(str(f)) == (st.st_mtime_ns, st.st_size):
                continue
            try:
                r = json.loads(f.read_text())
            except ValueError:
                continue                    # still being written: read on the next pass
            seen[str(f)] = (st.st_mtime_ns, st.st_size)
            if r.get("dry_run") or r.get("parse_ok") is False:
                continue
            name = Path(r.get("episode_dir", f.stem)).name
            fname = (name.replace("episode_", f"episode_{pre}", 1) if pre else name) + ".json"
            try:
                if (qa / fname).stat().st_mtime_ns >= st.st_mtime_ns:
                    continue                # already on the board (a restarted follow does not rewrite it)
            except OSError:
                pass
            info = info or json.loads((run / "run.json").read_text())
            d, _ = board_label(entry, manifest, fname, name, f, r, info, eps)
            tmp = qa / f".{fname}.tmp"
            tmp.write_text(json.dumps(d))
            os.replace(tmp, qa / fname)
            if entry.get("clips"):
                link_clips(_path(entry["clips"], here), board / "clips", Path(fname).stem)
            added += 1
    return added


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m board follow", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("board", type=Path, help="the board folder, with manifest.json")
    ap.add_argument("--every", type=float, default=30.0, help="seconds between passes (default 30)")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    a = ap.parse_args()
    seen: dict = {}
    tagged_at = 0.0
    while True:
        n = follow_once(a.board, seen)
        if n:
            print(f"{time.strftime('%H:%M:%S')} added {n} label{'s' if n != 1 else ''} "
                  f"({len(list((a.board / 'qa').glob('*.json')))} on the board)", flush=True)
        # new kinds of object get their rigid or deformable tag every few minutes (board/materials.py)
        if a.once or time.time() - tagged_at > MATERIALS_EVERY_S:
            k, usd = materials.tag(a.board)
            tagged_at = time.time()
            if k:
                print(f"{time.strftime('%H:%M:%S')} tagged {k} kinds of object for ${usd:.4f}", flush=True)
        if a.once:
            return 0
        time.sleep(a.every)


if __name__ == "__main__":
    raise SystemExit(main())
