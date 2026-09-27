"""The gate: the regression suite the harness is held to, per rig, on frame-verified cases and a cost sample.

    python -m gate prepare [--episodes data/episodes]
    python -m gate label --kind full --cap 30 [--rigs teleop,handheld,ego] [--runs data/runs]
    python -m gate score RUN [RUN ...] [--json FILE]

gate/selection.json names the episodes of each rig, each by its line in its dataset's episode list, and gate/cases.json
what the label of each case must say. `prepare` prepares and checks every selected episode with its dataset's
adapter (python -m prepare, then python -m checks) into EPISODES/<dataset>/gate, and links each rig's episodes into
EPISODES/gate/<rig>/ under their gate names; an episode labelled several times gets one link per sample. Four of the
datasets need HF_TOKEN (see the README). `label` starts one labelling run per rig (python -m label, dataset
gate_<rig>), all at once, each with its own --cap. `score` prints each rig's cases met, parse rate, consistency
flags and cost per footage hour (gate/score.py).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from gate import score

REPO = Path(__file__).resolve().parent.parent


def cmd_prepare(a) -> int:
    sel = score.load_selection()
    lines: dict[str, dict[str, str]] = {}
    for xs in sel.values():
        for x in xs:
            lines.setdefault(x["dataset"], {})[x["line"]] = x["episode"]
    for ds, eps in sorted(lines.items()):
        out = a.episodes / ds / "gate"
        lst = out.with_suffix(".txt")
        lst.parent.mkdir(parents=True, exist_ok=True)
        lst.write_text(f"# {ds} episodes of the gate (gate/selection.json)\n" + "".join(f"{l}\n" for l in eps))
        for step in (["prepare", ds, "prepare", "--episodes", str(lst), "--out", str(out)], ["checks", str(out)]):
            print(f"== python -m {' '.join(step)}", flush=True)
            rc = subprocess.run([sys.executable, "-m", *step], cwd=REPO).returncode
            if rc:
                return rc
    for rig, xs in sel.items():
        dest = a.episodes / "gate" / rig
        dest.mkdir(parents=True, exist_ok=True)
        for x in xs:
            src = (a.episodes / x["dataset"] / "gate" / x["episode"]).resolve()
            if not (src / "context.json").exists():
                raise SystemExit(f"{src} was not prepared")
            link = dest / x["name"]
            if link.is_symlink() and link.resolve() != src:
                link.unlink()
            if not link.exists():
                link.symlink_to(src)
        print(f"{dest}: {len(xs)} episodes")
    return 0


def cmd_label(a) -> int:
    rigs = a.rigs.split(",")
    unknown = [r for r in rigs if r not in score.RIGS]
    if unknown:
        raise SystemExit(f"unknown rigs {unknown}; the gate has {list(score.RIGS)}")
    if a.kind != "dry" and a.cap <= 0:
        raise SystemExit("a paid run needs --cap (USD per rig)")
    procs = {}
    for rig in rigs:
        folder = a.episodes / "gate" / rig
        if not folder.is_dir():
            raise SystemExit(f"no {folder}; run python -m gate prepare first")
        cmd = [sys.executable, "-m", "label", "--dataset", f"gate_{rig}", "--episodes", str(folder), "--kind", a.kind,
               "--cap", str(a.cap), "--runs", str(a.runs), "--concurrency", str(a.concurrency), "--note", "gate"]
        print(f"starting gate_{rig}", flush=True)
        procs[rig] = subprocess.Popen(cmd, cwd=REPO)
    rc = {rig: p.wait() for rig, p in procs.items()}
    print("runs are under " + ", ".join(str(a.runs / f"gate_{r}") for r in rigs) + "; score them with python -m gate score")
    return 0 if all(v == 0 for v in rc.values()) else 1


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        return score.main()
    ap = argparse.ArgumentParser(prog="python -m gate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="prepare and check the gate's episodes and link them per rig")
    p.add_argument("--episodes", type=Path, default=REPO / "data" / "episodes", help="where prepared episodes go")
    p = sub.add_parser("label", help="label each rig's gate episodes, one run per rig")
    p.add_argument("--episodes", type=Path, default=REPO / "data" / "episodes", help="where prepared episodes are")
    p.add_argument("--rigs", default=",".join(score.RIGS), help="comma list of rigs (default all three)")
    p.add_argument("--kind", default="full", choices=["dry", "smoke", "full"], help="as for python -m label")
    p.add_argument("--cap", type=float, default=0.0, help="spend cap per rig's run, USD (required unless dry)")
    p.add_argument("--runs", type=Path, default=REPO / "data" / "runs", help="where run folders go")
    p.add_argument("--concurrency", type=int, default=16, help="episodes in flight per rig")
    sub.add_parser("score", help="score gate runs (python -m gate score --help)")
    a = ap.parse_args()
    return {"prepare": cmd_prepare, "label": cmd_label}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
