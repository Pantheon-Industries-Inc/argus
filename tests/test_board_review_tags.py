"""A board can say why each episode is on it: tags a reviewer filters by, each with its reason (manifest review_tags)."""
import collections
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from board.build import build
from test_board_serve import _built_board, _get, _serve

TAGS = {"episode_000000": [{"tag": "tracking/low valid", "why": "valid on 48% of frames"},
                           {"tag": "top view/720", "why": "render size 720 x 720"}],
        "episode_000001": [{"tag": "top view/720", "why": "render size 720 x 720"}]}


def _tag(board, tags):
    (board / "tags.json").write_text(json.dumps(tags))
    manifest = json.loads((board / "manifest.json").read_text())
    (board / "manifest.json").write_text(json.dumps({**manifest, "review_tags": "tags.json"}))
    build(board)


def test_tags_reach_the_episode_and_the_rail_and_an_untagged_episode_has_none(tmp_path, monkeypatch):
    board = _built_board(tmp_path)
    _tag(board, TAGS)
    assert json.loads((board / "qa" / "episode_000000.json").read_text())["review_tags"] == TAGS["episode_000000"]
    httpd, url = _serve(monkeypatch, board, tmp_path / "clips")
    try:
        rail = {r["file"]: r for r in json.loads(_get(url + "/api/episodes")[2])}
        assert rail["episode_000000.json"]["review_tags"] == ["tracking/low valid", "top view/720"]
        assert "review_tags" not in rail["episode_000002.json"]
        ep = json.loads(_get(url + "/api/episode?file=episode_000000.json")[2])
        assert ep["review_tags"] == TAGS["episode_000000"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_tag_for_an_episode_not_on_the_board_stops_the_build(tmp_path):
    board = _built_board(tmp_path)
    with pytest.raises(ValueError, match="episode_nope"):
        _tag(board, {**TAGS, "episode_nope": [{"tag": "x", "why": "y"}]})


@pytest.mark.parametrize("bad", [[{"tag": "", "why": "y"}], [{"tag": "  ", "why": "y"}], [{"why": "y"}], [], "x"])
def test_a_tag_that_says_nothing_stops_the_build(tmp_path, bad):
    board = _built_board(tmp_path)
    with pytest.raises(ValueError, match="episode_000000"):
        _tag(board, {"episode_000000": bad})


@pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
def test_the_page_filters_by_tag_and_shows_why(tmp_path, monkeypatch):
    board = _built_board(tmp_path)
    _tag(board, TAGS)
    httpd, url = _serve(monkeypatch, board, tmp_path / "clips")
    try:
        (tmp_path / "page.html").write_bytes(_get(url + "/")[2])
        r = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("review_tags.js")),
                            str(tmp_path / "page.html"), url + "/"], capture_output=True, text=True, timeout=120)
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert r.returncode == 0, r.stderr or r.stdout
    res = json.loads(r.stdout)
    assert res["rows"] == {"top view/720": 1, "tracking/low valid": 1}, res  # the first dataset holds one tagged episode
    assert res["filtered"] == ["episode_000000.json"]
    assert res["other_tag"] == []
    assert "valid on 48% of frames" in res["why"] and "tracking/low valid" in res["why"]
    assert res["why_untagged"] == ""
    assert "tracking/low valid" in res["card"]
    # what the page itself left on its elements, past the helpers
    assert res["failure"] is None
    assert "valid on 48% of frames" in res["open_why"], "the open episode says why it is on the board"
    assert res["why_after_end"] == "", "a reason must not stay up with no episode open"
    assert any("tracking/low valid" in h for h in res["card_chips"]), "the episode's card carries its tag"
    assert any("tracking/low valid" in h for h in res["rows_shown"]), "the filter has a row for the tag"
    assert "Filter by tag or issue" in res["button"]
    assert "tracking/low valid" in res["button_filtered"], "the filter names the tag it is on"
    assert any("top view/720" in h for h in res["rows_no_issues"]), "tags alone are something to filter by"
    assert "<img" not in res["hostile_card"] and "&lt;img" in res["hostile_card"]
    assert "<img" not in res["hostile_why"] and res["hostile_why"].count("&lt;img") == 2


def test_a_changed_tag_file_reaches_the_board_with_no_rebuild_of_the_label(tmp_path, capsys):
    board = _built_board(tmp_path)
    _tag(board, TAGS)
    capsys.readouterr()
    (board / "tags.json").write_text(json.dumps({"episode_000001": [{"tag": "other", "why": "z"}]}))
    build(board)
    assert "reused" in capsys.readouterr().err, "a changed tag file must not rebuild every label"
    assert "review_tags" not in json.loads((board / "qa" / "episode_000000.json").read_text()), "a removed tag stays gone"
    assert json.loads((board / "qa" / "episode_000001.json").read_text())["review_tags"] == [{"tag": "other", "why": "z"}]


def _memo(tmp_path, monkeypatch, cap=256):
    """serve's episode memo over small files of its own, counting each filter."""
    from board import serve as S
    monkeypatch.setattr(S, "_VIEW_CACHE", collections.OrderedDict())
    monkeypatch.setattr(S, "_VIEW_CACHE_MAX", cap)
    monkeypatch.setattr(S, "DICTIONARY_JOB", None)
    calls = []
    real = S.episode_view
    monkeypatch.setattr(S, "episode_view", lambda d: calls.append(d["episode_prompt"]) or real(d))

    def label(name, prompt, mtime=None):
        p = tmp_path / name
        p.write_text(json.dumps({"episode_prompt": prompt}))
        if mtime:
            os.utime(p, ns=(mtime, mtime))
        return p
    return S, calls, label


def test_the_memo_tells_two_files_and_two_versions_of_one_file_apart(tmp_path, monkeypatch):
    S, calls, label = _memo(tmp_path, monkeypatch)
    t = 1_700_000_000_000_000_000
    a, b = label("a.json", "aaaa", t), label("b.json", "bbbb", t)      # same size and same time, so only the path differs
    assert "aaaa" in S._episode_body(a) and "bbbb" in S._episode_body(b)
    label("a.json", "cccc", t + 1000)                                  # same size, a new time
    assert "cccc" in S._episode_body(a)
    label("a.json", "ddddd", t + 1000)                                 # same time, a new size
    assert "ddddd" in S._episode_body(a)
    assert calls == ["aaaa", "bbbb", "cccc", "ddddd"]


def test_the_memo_drops_the_episode_read_longest_ago_and_stays_within_its_limit(tmp_path, monkeypatch):
    S, calls, label = _memo(tmp_path, monkeypatch, cap=2)
    a, b, c = label("a.json", "a"), label("b.json", "b"), label("c.json", "c")
    S._episode_body(a), S._episode_body(b), S._episode_body(a), S._episode_body(c)
    assert len(S._VIEW_CACHE) == 2
    del calls[:]
    S._episode_body(a)
    assert calls == [], "a was read more recently than b, so a stays"
    S._episode_body(b)
    assert calls == ["b"], "b was read longest ago, so it was the one dropped"


def test_a_board_editing_its_dictionary_filters_every_time(tmp_path, monkeypatch):
    """Its labels change with no change to the file, so a kept view would show the old definitions."""
    S, calls, label = _memo(tmp_path, monkeypatch)
    monkeypatch.setattr(S, "DICTIONARY_JOB", str(tmp_path / "job"))
    a = label("a.json", "a")
    S._episode_body(a), S._episode_body(a)
    assert calls == ["a", "a"] and not S._VIEW_CACHE


def test_an_episode_is_filtered_once_until_its_file_changes(tmp_path, monkeypatch):
    """A long stitched label takes seconds to filter for the page (public_export over a megabyte of text). The second
    open of an unchanged file must not pay that again, and a changed file must not be answered from memory."""
    from board import serve as S
    board = _built_board(tmp_path)
    httpd, url = _serve(monkeypatch, board, tmp_path / "clips")
    calls = []
    real = S.episode_view
    monkeypatch.setattr(S, "episode_view", lambda d: calls.append(1) or real(d))
    try:
        first = _get(url + "/api/episode?file=episode_000000.json")[2]
        assert _get(url + "/api/episode?file=episode_000000.json")[2] == first and len(calls) == 1
        p = board / "qa" / "episode_000000.json"
        d = json.loads(p.read_text())
        d["episode_prompt"] = "changed"
        p.write_text(json.dumps(d))
        assert b"changed" in _get(url + "/api/episode?file=episode_000000.json")[2] and len(calls) == 2
    finally:
        httpd.shutdown()
        httpd.server_close()
