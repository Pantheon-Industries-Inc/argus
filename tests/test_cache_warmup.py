"""A batch sends the first episode of each prompt variant alone before the rest (label/harness.py cache_warmup), so the
others find the shared instructions cached."""
import json

from label.harness import cache_variant, cache_warmup


def episode(root, name, **ctx):
    d = root / name
    d.mkdir()
    (d / "context.json").write_text(json.dumps({"profile": "teleop_arms", **ctx}))
    return d


def test_first_of_each_variant_goes_alone(tmp_path):
    a = episode(tmp_path, "a", instruction="pick the cup")
    b = episode(tmp_path, "b", instruction="pick the bowl")
    c = episode(tmp_path, "c")
    d = episode(tmp_path, "d", instruction="sweep", whole_body=True)
    e = episode(tmp_path, "e", instruction="sweep the floor", whole_body=True)
    assert cache_warmup([a, b, c, d, e], concurrency=8, dry=False) == [a, c, d]


def test_timed_steps_are_their_own_variant(tmp_path):
    a = episode(tmp_path, "a", instruction="clean up")
    b = episode(tmp_path, "b", instruction="clean up",
                annotation_subtasks=[{"t0": 0.0, "t1": 2.0, "label": "move to bed"}], n_state_frames=90)
    assert cache_variant(a) != cache_variant(b)


def test_no_warmup_when_it_cannot_help(tmp_path, monkeypatch):
    a = episode(tmp_path, "a", instruction="x")
    b = episode(tmp_path, "b", instruction="y")
    assert cache_warmup([a, b], concurrency=1, dry=False) == []          # one at a time already
    assert cache_warmup([a, b], concurrency=8, dry=True) == []           # a dry run calls no model
    assert cache_warmup([a], concurrency=8, dry=False) == []             # nothing to wait for it
    c = episode(tmp_path, "c")
    assert cache_warmup([a, c], concurrency=8, dry=False) == []          # every episode its own variant
    monkeypatch.setenv("ARGUS_CACHE_WARMUP", "0")
    assert cache_warmup([a, b], concurrency=8, dry=False) == []


def test_unreadable_context_is_one_variant(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b"
    b.mkdir()
    (b / "context.json").write_text("{not json")
    assert cache_variant(a) is None and cache_variant(b) is None
    assert cache_warmup([a, b], concurrency=8, dry=False) == [a]
