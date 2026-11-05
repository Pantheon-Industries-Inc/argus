"""board clips --decode-only and review --board-clips off: no viewing copy is encoded, and every camera problem a cut finds
is found and recorded the same, since it decides which cameras the model is shown."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from board import clips
from prepare import formats
from test_clip_cameras import _clips, _damaged, _video

REPO = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="no ffmpeg")


def _upload(root: Path, damage: dict) -> tuple[Path, Path]:
    """One episode of three cameras read as an upload is; damage is {file stem: the packets whose NAL lengths are broken}."""
    up = root / "up"
    up.mkdir(parents=True)
    for stem in ("top", "wrist_left", "wrist_right"):
        if stem in damage:
            _damaged(up / f"{stem}.mp4", 60, damage[stem])
        else:
            _video(up / f"{stem}.mp4", 60)
    eps = root / "episodes"
    rep = formats.convert(up, "teleop_arms", eps, "mine", float("inf"), grouping={})
    return eps, eps / rep["episodes"][0]["episode_id"]


def _found(ep: Path) -> tuple:
    """What a cut records about the cameras, with the job's own paths taken out."""
    ctx = json.loads((ep / "context.json").read_text())
    issues = sorted((x["kind"], x.get("camera"), x.get("what"), x.get("clip_frames"), x.get("episode_frames"))
                    for x in ctx.get("reader_issues") or [])
    return issues, sorted(json.loads((ep / "sources.json").read_text()))


@pytest.mark.parametrize("damage, cameras, kinds", [
    ({}, ["exo", "left", "right"], set()),
    ({"wrist_left": range(60)}, ["exo", "right"], {"camera_not_decodable"}),
    ({"top": range(60)}, ["left", "right"], {"camera_not_decodable"}),
    ({"wrist_right": range(40, 60)}, ["exo", "left", "right"], {"camera_short"}),
])
def test_decoding_finds_what_a_cut_finds_and_writes_no_clip(tmp_path, damage, cameras, kinds):
    made = {}
    for name, extra in (("cut", ()), ("decode", ("--decode-only",))):
        eps, ep = _upload(tmp_path / name, damage)
        r = _clips(eps, tmp_path / name / "clips", *extra)
        assert r.returncode == 0, r.stderr
        made[name] = _found(ep)
        videos = list((tmp_path / name / "clips").rglob("*.mp4"))
        assert (not videos) if extra else videos
        assert not list((tmp_path / name / "clips").rglob("*.progress"))
    assert made["decode"] == made["cut"]
    assert made["decode"][1] == cameras
    assert kinds <= {x[0] for x in made["decode"][0]}


def test_an_episode_with_no_camera_that_decodes_is_left_out_the_same(tmp_path):
    out = {}
    for name, extra in (("cut", ()), ("decode", ("--decode-only",))):
        eps, _ = _upload(tmp_path / name, {s: range(60) for s in ("top", "wrist_left", "wrist_right")})
        r = _clips(eps, tmp_path / name / "clips", *extra)
        assert r.returncode == 1
        failed = json.loads((tmp_path / name / "clips" / clips.FAILED).read_text())
        out[name] = sorted((k, sorted(v)) for k, v in failed.items())
        assert [f["name"] for f in clips.set_aside_failed(eps, tmp_path / name / "clips")] == [k for k, _ in out[name]]
    assert out["cut"] == out["decode"] and out["decode"]


@pytest.mark.parametrize("damaged", [None, "wrist_left"])
def test_a_review_without_board_clips_builds_the_same_requests(tmp_path, damaged):
    """With a dead camera too: off still probes every camera, or the request would keep that camera as placeholders."""
    jobs = {}
    up = tmp_path / "up"
    up.mkdir()
    for stem in ("top", "wrist_left", "wrist_right"):
        if stem == damaged:
            _damaged(up / f"{stem}.mp4", 60, range(60))
        else:
            _video(up / f"{stem}.mp4", 60)
    for name, extra in (("on", ()), ("off", ("--board-clips", "off"))):
        job = tmp_path / name
        r = subprocess.run([sys.executable, "-m", "review", "--data", str(up), "--rig", "teleop_arms", "--dataset", "mine",
                            "--out", str(job), "--free", "--field-dictionary", "off", *extra],
                           cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        jobs[name] = job
    assert list((jobs["on"] / "clips").rglob("*.mp4")) and not list((jobs["off"] / "clips").rglob("*.mp4"))
    sent = {}
    for name, job in jobs.items():
        record = json.loads(next((job / "dry").glob("episode_*.json")).read_text())
        sent[name] = ([i["sha256"] for i in record["config"]["request_images"]["images"]], record["prompt_text"],
                      record["config"]["views"])
    assert sent["on"] == sent["off"] and sent["off"][0]
    if damaged:
        assert "left" not in sent["off"][2]


def test_a_camera_that_ends_with_no_frame_read_is_an_error_not_a_short_camera(tmp_path, monkeypatch):
    """ffmpeg can exit 0 having decoded nothing, and only the count then shows the camera is dead."""
    _video(tmp_path / "a.mp4", 10)

    def nothing(cmd, watch, lead_s, check):
        watch[0].write_text("frame=0\nprogress=end\n")
    monkeypatch.setattr(clips, "run_while_progressing", nothing)
    with pytest.raises(RuntimeError, match="no frame"):
        clips.decode_one(str(tmp_path / "a.mp4"), 0.0, 10, tmp_path / "out" / "a.mp4", "ffmpeg", 1)
