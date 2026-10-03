"""Fallback search work and capture ties stay bounded after damaged stretches."""
import numpy as np
import pytest

from label import episode


@pytest.mark.parametrize("survivors", [[2699], []])
def test_fallback_traversal_skips_known_failures_for_later_missing_samples(monkeypatch, survivors):
    n, planned = 2700, np.linspace(0, 2698, 60).astype(int).tolist()
    ep = {"sources": {"exo": {}}, "context": {}}
    counts = {"clock": 0, "decode": 0, "search": 0}
    def clock(ep, k):
        counts["clock"] += 1
        return k / 30
    def decode(ep, v, ks, gate=None, widths=None, detail_ks=(), failed=None, damaged=None):
        counts["decode"] += len(ks)
        if failed is not None:
            failed.update(k for k in ks if k not in survivors)
        if damaged is not None:
            damaged.update(k for k in ks if k not in survivors)
        return {k: object() for k in ks if k in survivors}
    search = np.searchsorted
    def searchsorted(*args, **kwargs):
        counts["search"] += 1
        return search(*args, **kwargs)
    monkeypatch.setattr(np, "searchsorted", searchsorted)
    for name, fn in [("frame_time", clock), ("_decode_view", decode), ("views", lambda ep: ["exo"]),
                     ("_in_span", lambda ep, v, k: True), ("placeholder_instants", lambda ep, ks: {})]:
        monkeypatch.setattr(episode, name, fn)
    plan = {"n": n, "ks": planned}
    if survivors:
        episode.frames(ep, plan, widths=[128], detail_ks={planned[0], planned[-1]})
        assert plan["ks"] == survivors and len(ep["fallback_instants"]) == len(planned)
    else:
        with pytest.raises(episode.mf.FrameError):
            episode.frames(ep, plan, widths=[128])
    assert counts["search"] < 12 * n + 16 * len(planned)
    assert counts["clock"] == n and counts["decode"] <= n + 2


@pytest.mark.parametrize("times,readable,want", [([0, 1, 2], {0, 2}, 0),
                                                ([2, 1, 0], {0, 2}, 0),
                                                ([0, 0, 1, 1], {0, 3}, 0),
                                                ([1, 1, 1, 1], {2, 3}, 2)])
def test_fallback_keeps_the_earlier_index_at_equal_capture_distance(monkeypatch, times, readable, want):
    missing = 1
    ep = {"sources": {"exo": {}}, "context": {}}
    def decode(ep, v, ks, gate=None, widths=None, detail_ks=(), failed=None, damaged=None):
        if failed is not None:
            failed.update(k for k in ks if k not in readable)
        return {k: object() for k in ks if k in readable}
    for name, fn in [("frame_time", lambda ep, k: times[k]), ("_decode_view", decode),
                     ("views", lambda ep: ["exo"]), ("_in_span", lambda ep, v, k: True),
                     ("placeholder_instants", lambda ep, ks: {})]:
        monkeypatch.setattr(episode, name, fn)
    episode.frames(ep, {"n": len(times), "ks": [missing]})
    assert ep["fallback_instants"][missing] == want
