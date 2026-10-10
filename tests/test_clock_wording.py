"""Request headings identify the same zero as their sampled capture times."""
import hashlib
import json

import numpy as np
import pytest

from label import episode
from test_camera_timing import recording


@pytest.mark.parametrize("first", [0.5, -0.5])
def test_request_names_the_episode_clock_when_the_first_capture_is_not_zero(tmp_path, first):
    ep = recording(tmp_path, first + np.arange(90) / 30)
    ctx = json.loads((ep / "context.json").read_text())
    ctx["clock_zero_s"] = 0.0
    (ep / "context.json").write_text(json.dumps(ctx))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ep.iterdir() if p.is_file()}
    request = episode.build_request(ep)
    heading = request["prompt"].split("FRAMES. ", 1)[1].split("Which instants", 1)[0]
    assert "seconds from the episode clock's zero" in heading
    assert f"The first recorded frame is at {first:.2f} s on this clock." in heading
    assert "seconds from the episode's first frame" not in heading
    assert request["timesteps"][0] == pytest.approx(max(first, 0.0))
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ep.iterdir() if p.is_file()}


def test_request_keeps_the_legacy_heading_when_the_first_capture_is_zero(tmp_path):
    ep = recording(tmp_path, 0.5 + np.arange(90) / 30)
    ctx = json.loads((ep / "context.json").read_text())
    ctx["clock_zero_s"] = 0.5
    (ep / "context.json").write_text(json.dumps(ctx))
    request = episode.build_request(ep)
    heading = request["prompt"].split("FRAMES. ", 1)[1].split("Which instants", 1)[0]
    assert "seconds from the episode's first frame" in heading
    assert "The times are exact: use them, do not invent your own." in heading
    assert "The first recorded frame is at" not in heading
    assert request["timesteps"][0] == 0.0
