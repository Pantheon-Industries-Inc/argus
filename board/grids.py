"""What the model saw, for every episode of a board: each label's grid images, rebuilt from its prepared episode at the
settings the label recorded and checked against the SHA-1 of each image sent (label/grids.py), with no model call.

    python -m board grids BOARD [--jobs 4] [--dataset NAME]

For every episode file the board's manifest builds (board/build.py entry_labels: the run output each label comes
from, reruns included), the grids that output was sent are rebuilt from the prepared episode the manifest names and
written to BOARD/grids/<file stem>/ as grid_01.jpg, grid_02.jpg, ... and grids.json (each grid's times, size, bytes and
SHA-1, how they were checked, and the label's instants and cell size). It is resumable: an episode whose grids.json
already matches its label is skipped. An episode that cannot be rebuilt exactly (its footage is gone, or its label was
made before grid hashes were recorded) is reported and left without grids. Such a label can only be rebuilt by the
code that made it, into the same layout with "check": "code" (label/grids.py index).

The board shows them under "What the model saw" on each episode (board/serve.py /api/grids) and board/static.py
publishes them as media named by their content. They are never part of the labels or their downloads. A grid folder is
used only while it matches the label the board shows (matches): a later build with another label of the episode shows
no grids for it until this step runs again.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
from pathlib import Path

CHECKS = ("sha1", "code")     # how a grid folder was checked against what was sent (label/grids.py index)


def grid_dir(grids: Path, file: str) -> Path:
    return Path(grids) / Path(file).stem


def matches(idx: dict, d: dict) -> bool:
    """Whether a grids.json was rebuilt from the label a board file d shows and checked against what was sent: the same
    instants and, when the file records it (board/to_board.py grid_cell), the same cell size."""
    idx = idx or {}
    lab = idx.get("label") or {}
    if not idx.get("grids") or idx.get("check") not in CHECKS:
        return False
    if lab.get("timesteps_s") is None or lab.get("timesteps_s") != d.get("timesteps_s"):
        return False
    return d.get("grid_cell") is None or list(lab.get("cell") or []) == list(d["grid_cell"])


def read(grids: Path | None, file: str, d: dict) -> dict | None:
    """The grids.json of board file `file` when it matches the label d (matches) and every image it lists is there,
    else None."""
    if grids is None:
        return None
    p = grid_dir(grids, file) / "grids.json"
    try:
        idx = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if not matches(idx, d):
        return None
    if not all((p.parent / g["file"]).is_file() for g in idx["grids"]):
        return None
    return idx


def view(idx: dict, src) -> dict:
    """What the page needs to show a grid folder: {"cell", "grids": [{"t0_s", "t1_s", "width", "height", "bytes",
    "part"?, "src"}]}. src(i, entry) gives each image's address (the served board's /api/grid, a static build's media
    file)."""
    return {"cell": (idx.get("label") or {}).get("cell"),
            "grids": [{**{k: g[k] for k in ("t0_s", "t1_s", "width", "height", "bytes", "part") if k in g},
                       "src": src(i, g)} for i, g in enumerate(idx["grids"])]}


def build(board: Path, jobs: int = 4, datasets: list | None = None, quiet: bool = False) -> dict:
    """Rebuild every missing or stale grid folder of the board (the module docstring). Returns {rebuilt, kept,
    failed: {file: why}, bytes}."""
    from board import build as bb
    from board.to_board import convert
    from label import grids as lg
    board = Path(board)
    here = board.resolve()
    manifest = json.loads((board / "manifest.json").read_text())
    out = board / "grids"
    todo, kept = [], 0
    for entry in manifest.get("datasets", []):
        if datasets and entry["dataset"] not in datasets:
            continue
        eps = bb._path(entry["episodes"], here)
        for fname, (name, _src, r, _run) in sorted(bb.entry_labels(entry, here)[1].items()):
            d = convert(r, entry["dataset"])
            try:
                idx = json.loads((grid_dir(out, fname) / "grids.json").read_text())
            except (OSError, ValueError):
                idx = None
            if idx is not None and matches(idx, d):
                kept += 1
                continue
            todo.append((fname, r, eps / name))
    say = (lambda *a, **k: None) if quiet else print
    say(f"grids: {kept} episodes up to date, {len(todo)} to rebuild", flush=True)
    res = {"rebuilt": 0, "kept": kept, "failed": {}, "bytes": 0}

    def one(job):
        fname, r, ep = job
        try:
            gs = lg.rebuild(r, ep if not (r.get("config") or {}).get("pieces") else None)
        except Exception as e:      # one episode that cannot be rebuilt never stops the others
            return fname, None, f"{type(e).__name__}: {e}"[:300]
        return fname, lg.write(grid_dir(out, fname), r, gs), None
    with cf.ThreadPoolExecutor(max(1, jobs)) as ex:
        for i, (fname, idx, err) in enumerate(ex.map(one, todo), 1):
            if err:
                res["failed"][fname] = err
                say(f"grids: {fname} not rebuilt: {err}", file=sys.stderr, flush=True)
            else:
                res["rebuilt"] += 1
                res["bytes"] += sum(g["bytes"] for g in idx["grids"])
            if i % 50 == 0 or i == len(todo):
                say(f"grids: [{i}/{len(todo)}] {res['rebuilt']} rebuilt, {len(res['failed'])} failed, "
                    f"{res['bytes'] / 1e6:.1f} MB", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m board grids", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("board", type=Path, help="the board folder, with manifest.json")
    ap.add_argument("--jobs", type=int, default=4, help="episodes rebuilt at once (default 4)")
    ap.add_argument("--dataset", action="append", help="only this dataset (repeatable)")
    a = ap.parse_args()
    res = build(a.board, a.jobs, a.dataset)
    print(json.dumps({k: (len(v) if k == "failed" else v) for k, v in res.items()}, indent=1))
    return 1 if res["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
