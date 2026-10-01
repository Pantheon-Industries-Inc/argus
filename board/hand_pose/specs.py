"""One hand pose spec per prepared head-camera episode: its clip and the camera model the model's input is built from.

    python -m board.hand_pose.specs EPISODES [EPISODES ...] --out specs.json

  Egocentric-100K  the worker's own fisheye calibration (OpenCV fisheye / Kannala-Brandt, scaled to 456x256),
                   <factory>/<worker>/intrinsics.json on the Hub (a gated dataset: HF_TOKEN), resampled to a
                   832x480 pinhole with a 120 degree horizontal field of view
  Gen-HumanEgo     the headset camera's Double Sphere calibration from the episode's MCAP (meta.json calib2,
                   written by prepare/genhumanego.py), resampled to an 832x672 pinhole with a 90 degree field of
                   view pitched down 30 degrees, since the wearer's hands sit at the bottom of that fisheye (the
                   earlier 110 degree view drew the hands smaller; on 9 episodes the 90 degree view put 3 points more
                   of the drawn keypoints inside an independent detector's hand boxes, 14 points more on frames with
                   another person's hand in view, and drew a hand on 96% of the detector's hands against 94%)
  OpenAoE          phone video; its metadata's intrinsics are not self-consistent, so no camera is passed and the
                   K-free checkpoint reads one camera for the whole clip (it affects only the 3D decode, not the
                   2D keypoints the board draws)
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from prepare import hub

EGO100K = "builddotai/Egocentric-100K"


def spec(ep: Path) -> dict | None:
    ctx = json.loads((ep / "context.json").read_text())
    if ctx.get("profile") != "ego_head":
        return None
    src = Path(json.loads((ep / "sources.json").read_text())["exo"]["packed"])
    base = {"ds": _short(ctx["dataset"]), "ep": ep.name, "src": str(src), "board_path": ep.name,
            "size": src.stat().st_size}
    if ctx["dataset"] == EGO100K:
        worker = "/".join(ctx["source"]["shard"].split("/")[:2])
        k = json.loads(hub.fetch(EGO100K, f"{worker}/intrinsics.json"))
        if k["model"] != "fisheye" or (k["image_width"], k["image_height"]) != (456, 256):
            raise ValueError(f"{ep.name}: unexpected calibration {k}")
        cam = {"model": "kb", "convention": "opencv",
               **{x: k[x] for x in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")},
               "source": f"{EGO100K} {worker}/intrinsics.json"}
        return {**base, "camera": cam, "pinhole": {"w": 832, "h": 480, "hfov_deg": 120}}
    if ctx["dataset"] == "genrobot2025/Gen-HumanEgo":
        meta = json.loads((src.parent / "meta.json").read_text())
        c = meta.get("calib2")
        if isinstance(c, str):
            c = ast.literal_eval(c)
        if not c or c.get("model") != "ds" or (c["w"], c["h"]) != (1600, 1300):
            raise ValueError(f"{ep.name}: no Double Sphere calibration for camera2 in {src.parent / 'meta.json'}")
        fx, fy, cx, cy, xi, al = c["D"]
        cam = {"model": "ds", "convention": "opencv", "fx": fx, "fy": fy, "cx": cx, "cy": cy, "xi": xi, "alpha": al,
               "source": "meta.json calib2 (camera2, double sphere)"}
        return {**base, "camera": cam, "pinhole": {"w": 832, "h": 672, "hfov_deg": 90, "pitch_deg": 30}}
    return {**base, "camera": {"model": "unknown_pinhole"}}


def _short(dataset: str) -> str:
    return {EGO100K: "egocentric100k", "genrobot2025/Gen-HumanEgo": "genhumanego",
            "inclusionAI/OpenAoE-2000h": "openaoe"}.get(dataset, dataset.split("/")[-1].lower())


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m board.hand_pose.specs", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path, help="folders of prepared episode_* folders")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    specs = [s for root in a.roots for d in sorted(root.glob("episode_*")) if (d / "context.json").exists()
             for s in [spec(d)] if s]
    a.out.write_text(json.dumps(specs, indent=1))
    print(f"{len(specs)} head-camera episodes -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
