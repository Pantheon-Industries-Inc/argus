"""Requests and displayed footage keep the same samples and clock."""
import json
import re
import subprocess
from pathlib import Path

import av
import numpy as np
import pytest

from board import clips, sensors, serve
from board.hands import probe_pts
from label import episode
from prepare import formats


def video(path, n):
    with av.open(str(path), "w") as dst:
        stream = dst.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        stream.options = {"bf": "0", "g": "1"}
        for k in range(n):
            image = np.full((64, 96, 3), k * 2, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                dst.mux(packet)
        for packet in stream.encode():
            dst.mux(packet)


def recording(tmp_path, main, side=None):
    files, times = {}, {"exo": np.asarray(main)}
    if side is not None:
        times["left"] = np.asarray(side)
    for view, ts in times.items():
        path = tmp_path / (view + ".mp4")
        video(path, len(ts))
        files[view] = (view, path)
    ep = tmp_path / "episode_a"
    formats.video_views_episode(ep, files, "teleop_arms", "probe", {}, real=times,
                                signals={"force": np.arange(len(main))[:, None].astype(float)})
    with np.load(ep / "times.npz") as z:
        saved = {k: z[k] for k in z.files}
    np.savez(ep / "times.npz", **{**saved, **times})
    source = json.loads((ep / "sources.json").read_text())
    for view, ts in times.items():
        if view != "exo":
            source[view]["kmap"] = view + "_kmap.npy"
            np.save(ep / source[view]["kmap"], formats.nearest(ts, times["exo"]))
    (ep / "sources.json").write_text(json.dumps(source))
    return ep


@pytest.mark.parametrize("missing", [[0, 45], [0, 45, 89]])
def test_unavailable_planned_samples_keep_usable_footage_at_its_real_time(tmp_path, missing):
    ep = recording(tmp_path, np.arange(90) / 30)
    context = json.loads((ep / "context.json").read_text())
    context["placeholder_frames"] = {"exo": [[k, k] for k in missing]}
    (ep / "context.json").write_text(json.dumps(context))
    request = episode.build_request(ep)
    assert request["plan"]["ks"] and not set(missing) & set(request["plan"]["ks"])
    assert "No camera could be decoded at the planned instants" in request["prompt"]
    assert "0.00 s" in request["prompt"] and "1.50 s" in request["prompt"]
    details = [p["text"] for p in request["content"] if p.get("type") == "text" and "detail view" in p["text"]]
    assert "first available frame" in details[-2]
    assert "first frame of the episode" not in details[-2]


def test_a_paired_camera_keeps_its_existing_span_sentence():
    main, side = np.arange(90) / 30, 0.32 + np.arange(80) / 30
    ep = {"context": {"fps": 30}, "sources": {"exo": {}, "left": {}},
          "times": {"exo": main, "left": side}, "kmap": {"left": formats.nearest(side, main)}}
    assert episode._coverage_note(ep, {"ks": [0, 45, 89]}) == (
        " Left has frames only from 0.32 s to 2.95 s, so its cells are empty at the instants outside that time, "
        "and it is left out of a detail view there.")

