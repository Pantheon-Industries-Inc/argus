"""Run several models over one set of episodes through the same harness, with and without in-context learning from one
reference trace.

    python -m compare prepare --selection configs/compare/main.json
    python -m compare label --selection configs/compare/main.json --kind full --cap 100
    python -m compare label --selection configs/compare/third.json --with-example --kind full --cap 15
    python -m compare board --entries data/runs/compare/main.json data/runs/compare/third_ex.json \
        --out data/boards/compare

A selection names the episodes to compare. configs/compare/main.json is a seeded, task-diverse draw of about one
hour per rig, 193 episodes over the nine datasets; each entry carries the episode's line in its dataset's episode
list. configs/compare/third.json is a seeded third of it (65 episodes) for the in-context runs,
named by the episodes' folder names in main.

`prepare` writes each dataset's lines to EPISODES/<dataset>/compare_<selection>.txt, prepares exactly those
episodes into EPISODES/<dataset>/compare_<selection> with that dataset's adapter (python -m prepare) and runs the
deterministic checks on them (python -m checks). Four of the datasets need HF_TOKEN (see the README).

`label` links the selected episodes into one folder, EPISODES/compare/<selection>, and starts one labelling run per
model (label/run.py) with the model id and settings of configs/models.json (the top-level reasoning effort unless
the model's entry names its own), all at once, each in its own run folder under RUNS/compare/ named
<time>_<kind>_<commit>_<model key>[_ex]. With --with-example the models of the "with_example" list are also shown
configs/examples/example_<rig>.json, one complete annotation of a different episode of the same rig. A paid kind needs --cap, the spend cap of each model's run. It then writes
RUNS/compare/<selection>[_ex].json: the runs as entries of a board manifest's "comparisons" list, the run of
models.json's "reference" model marked as the reference. A teleop episode's cell width is a sampled routing answer
(label/route.py), so two runs can send an episode different frames; --routes gives every run the reference run's
answers instead (configs/compare/routes_main.json holds those of the published comparison), and refuses a routed
episode it holds no answer for.

`board` writes a board manifest, BOARD/manifest.json, from the entries files: one board dataset, "compare", whose
labels are the reference run's, and every other run as a comparison. Build it with `python -m board build BOARD`:
the board shows the reference model's labels, its "Labels by" control switches to any other model's, and the
comparison view measures every model against the board's own labels; `python -m compare.metrics BOARD` prints
the same numbers. To compare models on a board of your own, add the other entries to its manifest's
"comparisons" instead: the board's labels are then the reference.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def link_name(ds: str, episode: str) -> str:
    """The episode's name in a mixed folder. HABIT's episode names repeat MolmoAct2's (both are episode_<index>), so
    HABIT's carry the dataset's prefix, episode_habit_000686, as on the board."""
    return "episode_habit_" + episode[len("episode_"):] if ds == "habit" else episode


def prepared_dir(episodes: Path, ds: str, name: str) -> Path:
    """Where `prepare` puts a dataset's episodes of the selection."""
    return episodes / ds / f"compare_{name}"


def selected(selection: Path, episodes: Path) -> list[tuple[str, Path]]:
    """(name in the comparison folder, prepared episode folder) for every episode of the selection."""
    sel, name = json.loads(selection.read_text()), selection.stem
    if "rigs" in sel:
        return [(link_name(ds, e["episode"]), prepared_dir(episodes, ds, name) / e["episode"])
                for dss in sel["rigs"].values() for ds, v in dss.items() for e in v["episodes"]]
    # a subset of another selection in the same folder, by the names used in that selection's comparison folder
    parent = dict(selected(selection.with_name(f"{sel['subset_of']}.json"), episodes))
    return [(ep, parent[ep]) for eps in sel["episodes"].values() for ep in eps]


def cmd_prepare(a) -> int:
    sel, name = json.loads(a.selection.read_text()), a.selection.stem
    if "rigs" not in sel:
        raise SystemExit(f"{a.selection} is a subset of {sel['subset_of']}; prepare that selection instead")
    for dss in sel["rigs"].values():
        for ds, v in dss.items():
            out = prepared_dir(a.episodes, ds, name)
            lst = out.with_suffix(".txt")
            lst.parent.mkdir(parents=True, exist_ok=True)
            lst.write_text(f"# {ds} episodes of the comparison selection {a.selection.name}\n"
                           + "".join(e["line"] + "\n" for e in v["episodes"]))
            for step in (["prepare", ds, "prepare", "--episodes", str(lst), "--out", str(out)], ["checks", str(out)]):
                print(f"== python -m {' '.join(step)}", flush=True)
                rc = subprocess.run([sys.executable, "-m", *step], cwd=REPO).returncode
                if rc:
                    return rc
    return 0


def link_slice(selection: Path, episodes: Path) -> Path:
    """EPISODES/compare/<selection>: one link per selected episode to its prepared folder."""
    pairs = selected(selection, episodes)
    missing = [str(src) for _, src in pairs if not (src / "context.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} selected episodes are not prepared, e.g. {missing[:3]}; "
                         f"run python -m compare prepare first")
    dest = episodes / "compare" / selection.stem
    dest.mkdir(parents=True, exist_ok=True)
    for link, src in pairs:
        p = dest / link
        if p.is_symlink() and p.resolve() != src.resolve():
            raise SystemExit(f"{p} points elsewhere ({os.readlink(p)}); choose another selection name")
        if not p.exists():
            p.symlink_to(src.resolve())
    return dest


def unrouted(slice_dir: Path, answers: dict) -> list[str]:
    """The episodes of a routed rig (label/episode.py ROUTE_WIDTHS) whose task text has no answer in answers."""
    from label import episode as me
    from label import route
    miss = []
    for d in sorted(p for p in slice_dir.iterdir() if (p / "context.json").exists()):
        ep = me.load(d)
        if me.rig(ep) in me.ROUTE_WIDTHS and route.route_text(ep) not in answers:
            miss.append(d.name)
    return miss


def cmd_label(a) -> int:
    name = a.selection.stem
    cfg = json.loads((REPO / "configs" / "models.json").read_text())
    names = a.models.split(",") if a.models else (cfg["with_example"] if a.with_example else list(cfg["models"]))
    unknown = [n for n in names if n not in cfg["models"]]
    if unknown:
        raise SystemExit(f"unknown models {unknown}; configs/models.json has {list(cfg['models'])}")
    if a.kind != "dry" and a.cap <= 0:
        raise SystemExit("a paid run needs --cap (USD per model run)")
    slice_dir = link_slice(a.selection, a.episodes)
    seeds = []
    if a.routes:
        unheld = unrouted(slice_dir, json.loads(a.routes.read_text()))
        if unheld:
            raise SystemExit(f"{a.routes} holds no routing answer for {len(unheld)} routed episodes, e.g. "
                             f"{unheld[:3]}; they would route on their own and may see other frames")
        seeds = ["--route-seeds", str(a.routes.resolve())]
    runs = a.runs / "compare"
    before = {p for p in runs.glob("*") if p.is_dir()} if runs.exists() else set()
    procs = {}
    for n in names:
        key = n + ("_ex" if a.with_example else "")
        m = cfg["models"][n]
        cmd = [sys.executable, "-m", "label", "--dataset", "compare", "--episodes", str(slice_dir), "--kind", a.kind,
               "--cap", str(a.cap), "--runs", str(a.runs), "--concurrency", str(a.concurrency), "--label", key,
               "--note", f"model comparison on {name}: {m['model']}"
                         + (", in-context learning with a reference trace" if a.with_example else ""),
               "--", "--model", m["model"], "--reasoning", m.get("reasoning", cfg["reasoning"]),
               "--max-tokens", str(cfg["max_tokens"])]
        if a.with_example:
            cmd += ["--example-dir", str(REPO / "configs" / "examples")]
        cmd += seeds
        print(f"starting {key}: {m['model']}", flush=True)
        procs[key] = subprocess.Popen(cmd, cwd=REPO)
    rc = {key: p.wait() for key, p in procs.items()}
    after = sorted(p for p in runs.glob("*") if p.is_dir() and p not in before)
    ref = cfg["models"][cfg["reference"]]["name"]
    icl = f", in-context learning with {'an' if ref[:1].lower() in 'aeiou' else 'a'} {ref} trace"
    entries = []
    for key in procs:
        mine = [p for p in after if p.name.endswith("_" + key)]
        if not mine:
            print(f"{key}: no run folder (exit {rc[key]})", file=sys.stderr)
            continue
        base = key.removesuffix("_ex")
        e = {"key": key, "name": cfg["models"][base]["name"] + (icl if a.with_example else ""),
             "run": str(mine[-1].resolve()), "episodes": str(slice_dir.resolve())}
        if base == cfg["reference"] and not a.with_example:
            e["reference"] = True
        if a.with_example:
            e.update(example=True, base=base)
        entries.append(e)
    index = runs / f"{name}{'_ex' if a.with_example else ''}.json"
    index.write_text(json.dumps(entries, indent=1))
    print(f"wrote {index} ({len(entries)} comparison entries)")
    return 0 if all(v == 0 for v in rc.values()) else 1


def cmd_board(a) -> int:
    from board.rules import rules_for
    entries = [e for f in a.entries for e in json.loads(f.read_text())]
    refs = [e for e in entries if e.get("reference")]
    if len(refs) != 1:
        raise SystemExit(f"the entries name {len(refs)} reference runs; exactly one is needed (the run of "
                         f"configs/models.json's reference model without the example)")
    others = [e for e in entries if not e.get("reference")]
    manifest = {"board": a.out.name,
                "datasets": [{"dataset": "compare", "run": refs[0]["run"], "episodes": refs[0]["episodes"],
                              "rules": rules_for(None)}],
                "comparisons": others}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {a.out / 'manifest.json'} ({len(others)} comparison runs beside the board's own labels); "
          f"now python -m board build {a.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m compare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c, what in (("prepare", "prepare and check the selection's episodes, dataset by dataset"),
                    ("label", "label the selection with every model, one run each")):
        p = sub.add_parser(c, help=what, description=what)
        p.add_argument("--selection", type=Path, required=True, help="configs/compare/<name>.json")
        p.add_argument("--episodes", type=Path, default=REPO / "data" / "episodes", help="where prepared episodes go")
    p = sub.choices["label"]
    p.add_argument("--models", default=None, help="comma list of configs/models.json names (default: all, or the "
                                                  "with_example list with --with-example)")
    p.add_argument("--with-example", action="store_true", help="show every model one complete example annotation")
    p.add_argument("--kind", default="full", choices=["dry", "smoke", "full"], help="as for python -m label")
    p.add_argument("--cap", type=float, default=0.0, help="spend cap per model run, USD (required unless dry)")
    p.add_argument("--runs", type=Path, default=REPO / "data" / "runs", help="where run folders go")
    p.add_argument("--concurrency", type=int, default=8, help="episodes in flight per model")
    p.add_argument("--routes", type=Path, default=None,
                   help="configs/compare/routes_<selection>.json: the reference run's routing answer for each task "
                        "text, so every model is sent the frames the reference was (label/route.py)")
    p = sub.add_parser("board", help="write a board manifest for the comparison runs",
                       description="write a board manifest for the comparison runs")
    p.add_argument("--entries", type=Path, nargs="+", required=True, help="RUNS/compare/<selection>[_ex].json files")
    p.add_argument("--out", type=Path, required=True, help="the board folder")
    a = ap.parse_args()
    return {"prepare": cmd_prepare, "label": cmd_label, "board": cmd_board}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
