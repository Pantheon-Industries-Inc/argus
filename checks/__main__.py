"""Run the per-episode deterministic checks on prepared episodes, before labelling.

    python -m checks EPISODES [EPISODES ...] [--jobs 8] [--force]

EPISODES are folders of prepared episode_* folders. Each check writes its result into the episode's context.json
under its own key, and `python -m board build` copies each result into the episode's dataset_checks on the board.
None of it goes into the prompt. This runs, in order:

  stream_pairing     do the mounted camera streams follow their own arm's or gripper's recorded motion
                     (python -m checks.stream_pairing)
  recorded_jumps     single-frame jumps in the recorded pose that the actor's own camera does not show
                     (python -m checks.stream_pairing --jumps)
  gripper_channels   recorded gripper openings that never change over the episode
                     (python -m checks.stream_pairing --grippers)
  capture_qc         the capture checks of public-dataset-adapter (clocks, exposure, frozen or duplicated
                     frames, motion the video does not show), with the per-rig calibration in capture_qc.py
                     (python -m checks.capture_qc)

Two checks do not run here:

  timebase           sped-up recordings (MolmoAct2). Its rule compares neighbouring episodes, so it scans the
                     whole dataset on its own (python -m checks.timebase scan, then apply); labelling reports it.
  label_consistency  annotations that contradict themselves. It reads stored labels, so it runs inside
                     `python -m board build` on every episode.
"""
import argparse
import subprocess
import sys

ap = argparse.ArgumentParser(prog="python -m checks", description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("roots", nargs="+", metavar="EPISODES", help="folders of prepared episode_* folders")
ap.add_argument("--jobs", type=int, default=8, help="episodes checked in parallel")
ap.add_argument("--force", action="store_true", help="recompute episodes that already have a result")
a = ap.parse_args()
extra = ["--jobs", str(a.jobs)] + (["--force"] if a.force else [])
steps = [["checks.stream_pairing"], ["checks.stream_pairing", "--jumps"], ["checks.stream_pairing", "--grippers"],
         ["checks.capture_qc"]]
for step in steps:
    print(f"== {' '.join(step)}", flush=True)
    rc = subprocess.run([sys.executable, "-m", *step, *extra, *a.roots]).returncode
    if rc:
        raise SystemExit(rc)
