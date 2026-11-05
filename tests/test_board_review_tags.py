"""A board can say why each episode is on it: tags a reviewer filters by, each with its reason (manifest review_tags)."""
import json
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


@pytest.mark.parametrize("bad", [[{"tag": "", "why": "y"}], [{"why": "y"}], "x"])
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
