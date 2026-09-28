"""Prepare inclusionAI/OpenAoE-2000h clips (crowd-sourced phone footage worn at the head) as episode sidecars
(rig ego_head, no recorded state).

    python -m prepare openaoe prepare --episodes configs/slices/openaoe.txt --out EPISODES [--raw RAW] [--jobs N]
        [--force]

The list has one clip folder name per line (raw_<recording>_seg_<n>). Per clip this downloads three files into
RAW/<clip>/: raw_video.mp4, video_info.json (device and camera parameters) and
ego_annotation/ego_action_annotation.json, the dataset's auto-generated atomic-action segments (time span, verb,
object, hand, description, confidence). It writes EPISODES/episode_<clip>/ with context.json, sources.json
(pointing at the downloaded mp4), times.npz (the frames' real times from their pts) and instruction.txt. The
segments are the dataset's claims about the clip, so they go to the model as the annotation to check
(context["annotation_subtasks"]), as Gen-HumanEgo's timed subtasks do. The dataset ships no task instruction.

A clip is one numbered segment of a longer recording; the collection note says so, so a segment that starts or
ends mid-activity is read as packaging. The dataset needs no token.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from prepare import cli
from prepare import hub
from prepare import formats

# Data Review and python -m prepare folder hand an upload to an adapter that recognizes it (prepare/formats.py
# upload_adapters): a clip folder in this dataset's own layout is read here, with its action segments and device
UPLOAD = "video"

REPO = "inclusionAI/OpenAoE-2000h"
COLLECTION_NOTE = ("crowd contributors record their own activities on a phone worn at the head; each clip is one "
                   "numbered segment of a longer recording, so a clip can start or end in the middle of an "
                   "activity, which is how the dataset is packaged")
ANNOTATION_NOTE = ("these action segments were generated automatically on a whole-second grid, many as fixed "
                   "5-second windows, so a boundary a second or two from the visible change is that resolution. A "
                   "wrong action, object or hand, or segments that stop following what the hands do, is what would "
                   "mislead someone training on them")
CLIP_FILES = ("raw_video.mp4", "video_info.json", "ego_annotation/ego_action_annotation.json")


def subtasks(ann: list) -> list[dict]:
    out = []
    for seg in ann:
        acts = seg.get("atomic_action") or []
        label = "; ".join(" ".join(x for x in (a.get("verb"), a.get("object")) if x)
                          + (f" ({a['hand']} hand)" if a.get("hand") else "") for a in acts)
        label = label or seg.get("scene") or "segment"
        out.append({"t0": float(seg["start_ts"]), "t1": float(seg["end_ts"]), "label": label, "ok": True})
    return out


def download(clip: str, raw: Path) -> Path:
    for f in CLIP_FILES:
        if not (raw / clip / f).exists():
            try:
                hub.download(REPO, f"{clip}/{f}", raw)
            except Exception as e:
                if f == "raw_video.mp4":
                    raise
                print(f"{clip}: no {f} ({type(e).__name__})", file=sys.stderr)
    return raw / clip


def clip_extra(d: Path, clip: str) -> dict:
    """The clip's context beyond its video: the action segments as the annotation to check, and its device."""
    ann_p = d / "ego_annotation" / "ego_action_annotation.json"
    ann = json.loads(ann_p.read_text()) if ann_p.exists() else []
    info = json.loads((d / "video_info.json").read_text()) if (d / "video_info.json").exists() else {}
    dev = info.get("deviceInfo") or {}
    return {"task_label": [clip], "instruction": None, "annotation_subtasks": subtasks(ann),
            "collection_note": COLLECTION_NOTE, "annotation_note": ANNOTATION_NOTE,
            "source": {"clip": clip, "device": " ".join(x for x in (dev.get("brand"), dev.get("model")) if x),
                       "resolution": (info.get("cameraParams") or {}).get("resolution"),
                       "annotation_segments": len(ann)}}


def prepare_clip(d: Path, clip: str, ep: Path) -> None:
    formats.video_views_episode(ep, {"exo": ("raw_video", d / "raw_video.mp4")}, "ego_head", REPO, clip_extra(d, clip))


def recognizes(item: dict) -> bool:
    """A video upload in this dataset's clip layout: one raw_video.mp4 with its ego_annotation beside it."""
    fs = item.get("files") or []
    return (len(fs) == 1 and Path(fs[0]).name == "raw_video.mp4"
            and (Path(fs[0]).parent / "ego_annotation" / "ego_action_annotation.json").exists())


def convert_upload(item: dict, rig: str, out: Path, dataset: str) -> dict:
    d = Path(item["files"][0]).parent          # the clip folder, named raw_<recording>_seg_<n>
    extra = clip_extra(d, d.name)
    extra["source"]["upload"] = item["name"]
    if rig != "ego_head":
        extra["rig_note"] = f"the upload was marked {rig}; this dataset's clips are from a head-worn phone"
    ep = formats.unique_dir(out, formats.episode_name(d.name))
    return formats.video_views_episode(ep, {"exo": ("raw_video", d / "raw_video.mp4")}, "ego_head", dataset, extra)


def prepare_one(clip: str, ep: Path, raw: Path, force: bool) -> str:
    if not force and (ep / "context.json").exists():
        return "skip"
    prepare_clip(download(clip, raw), clip, ep)
    return "ok"


def main() -> int:
    ap, sub = cli.parser("openaoe", __doc__)
    cli.add_prepare(sub, "openaoe", "one clip folder name per line")
    a = ap.parse_args()
    clips = cli.read_list(a.episodes)
    a.out.mkdir(parents=True, exist_ok=True)
    dirs = dict(zip(clips, formats.episode_dirs(a.out, clips)))
    return cli.run(clips, lambda c: prepare_one(c, dirs[c], a.raw, a.force), a.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
