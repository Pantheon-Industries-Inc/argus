"""Exact recorded chunk counts retain usable LeRobot paths."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from label import episode
from prepare import formats
from test_reader_cameras import _lerobot_images


@pytest.mark.parametrize("value", [9007199254740993, 10**100 + 1, np.int64(9007199254740993),
                                  "9007199254740993", "+9007199254740993", " 9007199254740993 ",
                                  "9_007_199_254_740_993"])
def test_positive_chunk_counts_do_not_round_through_float(value):
    assert formats.chunk_size({"chunks_size": value}) == int(value)


@pytest.mark.parametrize("value", [None, True, False, 0, -1, 1.5, float("inf"), float("nan"), "1.5", "1e3"])
def test_invalid_chunk_counts_stay_invalid(value):
    assert formats.chunk_size({"chunks_size": value}) is None


def test_native_image_episode_uses_its_exact_large_chunk_count(tmp_path):
    root = tmp_path / "upload"
    _lerobot_images(root, n=40)
    n = 9007199254740993
    path = root / "data/chunk-000/episode_000000.parquet"
    table = pd.read_parquet(path)
    table["episode_index"] = np.full(40, n, dtype=np.int64)
    table["observation.state"] = [np.asarray(v, dtype=np.float32) for v in table["observation.state"]]
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info.update(chunks_size=n, data_path="data/{episode_chunk:d}/{episode_index:d}.parquet")
    info_path.write_text(json.dumps(info))
    target = root / info["data_path"].format(episode_chunk=n // n, episode_index=n)
    target.parent.mkdir(parents=True)
    table.to_parquet(target)
    path.unlink()
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": n, "length": 40,
                                                         "tasks": ["Keep the actual recording"]}) + "\n")
    original = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*") if p.is_file()}
    report = formats.convert(root, "teleop_arms", tmp_path / "units", "test", 900)
    assert not report["failed"] and len(report["episodes"]) == 1
    unit = tmp_path / "units" / report["episodes"][0]["episode_id"]
    with np.load(unit / "state.npz") as arrays:
        np.testing.assert_array_equal(arrays["state"], np.stack(table["observation.state"]))
    request = episode.build_request(unit)
    assert request["timesteps"] and "Keep the actual recording" in request["prompt"]
    assert original == {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in root.rglob("*") if p.is_file()}
