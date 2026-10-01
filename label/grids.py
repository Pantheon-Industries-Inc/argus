"""What the model saw: a label's grid images, rebuilt exactly from its episode and the settings the label recorded.

Every label records how its grids were built (label/harness.py, in the output's config): the instants sent
(timesteps_s, seconds from the episode's first frame, to the millisecond), the cell size (cell, [width, height]), the
cameras in row order (views, cam_labels), the instants per grid (grid_cols) and each grid image's SHA-1 (grid_sha1).
Everything else that shapes a grid is fixed in the code: the JPEG qualities, the gutter and the header
(label/episode.py) and the font, which ships in label/fonts/ and is laid out with Pillow's basic engine
(label/frames.py). rebuild() builds the episode's request again at the recorded cell width (label/episode.py
build_request: the same instants, frames and grids), takes its grid images, and checks the instants, the cell size and
every image's SHA-1 against the label, so a grid is only ever given out byte for byte as it was sent. No model is
called.

A label made before grid_sha1 was recorded (2026-10-01) cannot be checked this way and is refused: its grids may have
been built by older code (a different font engine, row height or frame reader). Such a label can only be rebuilt by
the code that made it.

A recording labelled in parts (label/pieces.py) was sent one request per part. Its grids are each part's, rebuilt from
the part's own folder at the settings the stitched label keeps for it (config.pieces[i].grids). Each grid's times are
then given on the recording's clock, while the image is headed with the part's own times, as it was sent.

    python -m label.grids EPISODE LABEL.json --out DIR       DIR/grid_01.jpg, ... and DIR/grids.json
    python -m label.grids --trace RUN/out --out TRACE.zip     the labelling trace, every call with its grids
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

from label import episode as me
from label import frames as mf


class GridError(RuntimeError):
    """A label's grids cannot be rebuilt exactly (no recorded settings, or an episode that no longer matches them)."""


def settings(record: dict) -> dict:
    """What a label recorded about its grids: {cell, grid_cols, timesteps_s, views, cam_labels, grid_sha1}. Raises
    GridError when the label records no cell size, instants or grid hashes."""
    cfg = record.get("config") if "config" in record else record
    cfg = cfg or {}
    missing = [k for k in ("cell", "timesteps_s", "grid_sha1") if not cfg.get(k)]
    if missing:
        raise GridError(f"the label does not record its grids' {' or '.join(missing)}")
    return {"cell": [int(cfg["cell"][0]), int(cfg["cell"][1])], "grid_cols": int(cfg.get("grid_cols") or 4),
            "timesteps_s": [float(t) for t in cfg["timesteps_s"]],
            "views": list(cfg["views"]) if cfg.get("views") else None,
            "cam_labels": list(cfg["cam_labels"]) if cfg.get("cam_labels") else None,
            "grid_sha1": list(cfg["grid_sha1"])}


def _rebuild_one(ep_dir: Path, s: dict, gate=None) -> list[dict]:
    # the request itself, built again at the label's cell width: the instants are planned and the frames decoded and
    # composed by the same code, so no frame is looked up by its time (two frames can share a time to the
    # millisecond). The cell width alone decides the grids: the detail views a fixed width may add or drop are
    # separate images, and a frame kept full size is cut to the cell exactly as one kept at the cell's width
    req = me.build_request(Path(ep_dir), gate=gate, cell_w=s["cell"][0], grid_cols=s["grid_cols"])
    if s["views"] is not None and req["views"] != s["views"]:
        raise GridError(f"{ep_dir}: cameras {req['views']}, the label was sent {s['views']}")
    if s["cam_labels"] is not None and req["cam_labels"] != s["cam_labels"]:
        raise GridError(f"{ep_dir}: camera names {req['cam_labels']}, the label was sent {s['cam_labels']}")
    if [round(float(t), 3) for t in req["timesteps"]] != [round(t, 3) for t in s["timesteps_s"]]:
        raise GridError(f"{ep_dir}: the episode's instants are not the ones the label was sent")
    if list(req["cell"]) != s["cell"]:
        raise GridError(f"{ep_dir}: cell {req['cell']}, the label was sent {s['cell']}")
    ts, cols = req["timesteps"], req["grid_cols"]
    out = [{"t0_s": round(float(ts[i * cols]), 3), "t1_s": round(float(ts[min(len(ts), (i + 1) * cols) - 1]), 3),
            "jpeg": jpg} for i, jpg in enumerate(mf.grid_jpegs(req["content"]))]
    got = [hashlib.sha1(g["jpeg"]).hexdigest() for g in out]
    if got != s["grid_sha1"]:
        bad = next((i for i, (a, b) in enumerate(zip(got, s["grid_sha1"])) if a != b), min(len(got), len(s["grid_sha1"])))
        raise GridError(f"{ep_dir}: the rebuilt grids are not the ones sent ({len(got)} rebuilt, "
                        f"{len(s['grid_sha1'])} sent, first difference at grid {bad + 1})")
    return out


def rebuild(record: dict, ep_dir: Path | None = None, gate=None) -> list[dict]:
    """The grid images a label was sent, in order: [{t0_s, t1_s, width, height, jpeg}, ...] (t0_s and t1_s are the
    times of the grid's first and last column). ep_dir is the prepared episode (default: the label's episode_dir). A
    recording labelled in parts gives each part's grids with "part" (1, 2, ...) and its times on the recording's
    clock. Raises GridError unless every image matches the SHA-1 the label recorded for it."""
    from PIL import Image
    cfg = record.get("config") or {}
    if cfg.get("pieces"):
        out = []
        for p in cfg["pieces"]:
            if not p.get("grids"):
                raise GridError("the stitched label was made before each part kept its grid settings")
            if not p.get("episode_dir") or not Path(p["episode_dir"]).is_dir():
                raise GridError(f"part {p.get('part')}: its folder {p.get('episode_dir')} is not here")
            for g in _rebuild_one(Path(p["episode_dir"]), settings(p["grids"]), gate):
                out.append({**g, "part": int(p["part"]), "t0_s": round(g["t0_s"] + float(p["t0_s"]), 3),
                            "t1_s": round(g["t1_s"] + float(p["t0_s"]), 3)})
    else:
        s = settings(record)
        ep_dir = Path(ep_dir or record.get("episode_dir") or "")
        if not me.is_episode_dir(ep_dir):
            raise GridError(f"{ep_dir} is not a prepared episode")
        out = _rebuild_one(ep_dir, s, gate)
    for g in out:
        g["width"], g["height"] = Image.open(io.BytesIO(g["jpeg"])).size
    return out


def index(record: dict, grids: list[dict], check: str = "sha1") -> dict:
    """What grids.json says about a set of rebuilt grids: the label's instants and cell size (which the board matches
    against the label it shows, board/grids.py), how the grids were checked against what was sent ("sha1": each
    image's SHA-1 equals the one the label recorded; "code": rebuilt by the code that made the label, with its
    instants, cell size and image count checked), and each grid's times, size, bytes and SHA-1, under
    the file name write() gives it."""
    cfg = record.get("config") or {}
    s = {"cell": cfg.get("cell"), "timesteps_s": cfg.get("timesteps_s"),
         **({"parts": len(cfg["pieces"])} if cfg.get("pieces") else {})}
    return {"label": s, "check": check,
            "grids": [{"file": f"grid_{i:02d}.jpg", "t0_s": g["t0_s"], "t1_s": g["t1_s"],
                       "width": g["width"], "height": g["height"], "bytes": len(g["jpeg"]),
                       "sha1": hashlib.sha1(g["jpeg"]).hexdigest(),
                       **({"part": g["part"]} if "part" in g else {})}
                      for i, g in enumerate(grids, 1)]}


def write(out_dir: Path, record: dict, grids: list[dict], check: str = "sha1") -> dict:
    """The grids as out_dir/grid_01.jpg, ... and out_dir/grids.json (index); grids.json is written last, so a folder
    with one holds every grid it lists."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = index(record, grids, check)
    for g, e in zip(grids, idx["grids"]):
        (out_dir / e["file"]).write_bytes(g["jpeg"])
    tmp = out_dir / ".grids.json.tmp"
    tmp.write_text(json.dumps(idx, indent=1))
    tmp.replace(out_dir / "grids.json")
    return idx


# where a job's files sit on the server that labelled it; the trace leaves these out
SERVER_PATH_KEYS = ("episode_dir", "example_dir")


def zip_trace(calls_dir: Path, out: Path, arc: str) -> dict:
    """The labelling trace: every call the harness made, as it kept it (calls_dir is a run's out/, one file per call;
    a long recording has one per part), without the paths on the labelling server, and each call's grids rebuilt
    beside it as <arc>/<call>/grid_01.jpg, ... with <arc>/<call>/grids.json. A call whose grids cannot be rebuilt
    exactly (its episode folder is gone, or it was labelled before grid hashes were recorded) gets a grids.json that
    says why instead. Returns {calls, grids, missing}."""
    import zipfile
    calls = sorted(Path(calls_dir).glob("*.json")) if Path(calls_dir).exists() else []
    res = {"calls": 0, "grids": 0, "missing": {}}
    if not calls:
        return res
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in calls:
            d = json.loads(p.read_text())
            try:
                grids = rebuild(d)
                idx = index(d, grids)
            except (GridError, mf.FrameError, OSError, KeyError, ValueError, RuntimeError) as e:
                grids, idx = [], {"error": f"not rebuilt: {e}"[:400]}
                res["missing"][p.name] = idx["error"]
            for k in SERVER_PATH_KEYS:
                d.pop(k, None)
            for part in (d.get("config") or {}).get("pieces") or []:
                part.pop("episode_dir", None)
            z.writestr(f"{arc}/{p.name}", json.dumps(d, indent=1))
            for g, e in zip(grids, idx.get("grids") or []):
                # JPEG bytes do not compress: stored as they are
                z.writestr(zipfile.ZipInfo(f"{arc}/{p.stem}/{e['file']}"), g["jpeg"], compress_type=zipfile.ZIP_STORED)
            z.writestr(f"{arc}/{p.stem}/grids.json", json.dumps(idx, indent=1))
            res["calls"] += 1
            res["grids"] += len(grids)
    tmp.replace(out)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m label.grids", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", nargs="?", type=Path, help="a prepared episode folder")
    ap.add_argument("label", nargs="?", type=Path, help="its label (a harness output, out/<episode>.json)")
    ap.add_argument("--trace", type=Path, help="a run's out/ folder: write the labelling trace with every call's grids")
    ap.add_argument("--out", type=Path, required=True, help="the folder for the grids, or the trace's zip file")
    a = ap.parse_args(argv)
    if a.trace:
        res = zip_trace(a.trace, a.out, a.out.stem)
        print(json.dumps(res, indent=1))
        return 0 if res["calls"] else 1
    if not a.episode or not a.label:
        ap.error("give EPISODE and LABEL, or --trace")
    record = json.loads(a.label.read_text())
    idx = write(a.out, record, rebuild(record, a.episode))
    print(f"{len(idx['grids'])} grids ({sum(g['bytes'] for g in idx['grids']):,} bytes) in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
