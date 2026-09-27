"""Prepare episodes as sidecar folders the checks and the labelling harness read.

    python -m prepare <adapter> prepare --episodes LIST --out FOLDER [--raw RAW] [--jobs N] [--force]
    python -m prepare <adapter> sample ...          (the adapters that have a sampler)
    python -m prepare <adapter> --help              (that adapter's inputs, outputs and options)

Each public-dataset adapter downloads exactly the episodes in LIST (one per line, in the adapter's own form; the
lists in configs/slices and configs/quickstart are examples) into RAW, default data/raw/<adapter>, and writes one
folder per episode into FOLDER. lerobot and videos read your own data from --root instead and download nothing.

Adapters:
"""
import importlib
import sys

ADAPTERS = {
    "molmo": "allenai/MolmoAct2-BimanualYAM-Dataset (teleop, packed LeRobot v3)",
    "abc130k": "XDOF/ABC-130k (teleop, one MCAP per episode)",
    "galaxea": "RogersPyke/Galaxea-Open-World-Dataset_10K_20260123 (teleop, LeRobot v2.1 per collection)",
    "habit": "configinc/HABIT (teleop with a person in the task, LeRobot v2.0)",
    "fastumi": "IPEC-COMMUNITY/FastUMI_100k_lerobot (handheld grippers, LeRobot v2.1)",
    "realomin": "genrobot2025/10Kh-RealOmin-OpenData (handheld grippers, one MCAP per episode)",
    "egocentric100k": "builddotai/Egocentric-100K (head camera, 3-minute clips in tar shards)",
    "genhumanego": "genrobot2025/Gen-HumanEgo (head camera, one MCAP per episode)",
    "openaoe": "inclusionAI/OpenAoE-2000h (head-worn phone, one folder per clip)",
    "lerobot": "your own LeRobot dataset on disk (v2.0, v2.1 or v3.0; teleop, handheld or head camera)",
    "videos": "your own folder of video files (one episode per file, video only, any rig)",
}

if len(sys.argv) < 2 or sys.argv[1] not in ADAPTERS:
    print(__doc__ + "".join(f"  {k:<15} {v}\n" for k, v in ADAPTERS.items()))
    raise SystemExit(0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") else 2)
name = sys.argv.pop(1)
sys.argv[0] = f"python -m prepare {name}"
raise SystemExit(importlib.import_module(f"prepare.{name}").main())
