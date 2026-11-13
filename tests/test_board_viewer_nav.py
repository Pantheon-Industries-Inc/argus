"""The board page under navigation that outruns its requests (tests/viewer_nav.js), against a live board.

A read that fails is never remembered as the episode, a response that arrives after the reader moved on never takes
the view back, and what fills in a drawn episode (its frame grids, its dictionary) lands while that draw is on screen."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from test_board_serve import _built_board, _get, _serve


@pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
def test_navigation_outrunning_requests(tmp_path, monkeypatch):
    board = _built_board(tmp_path)
    httpd, url = _serve(monkeypatch, board, tmp_path / "clips")
    try:
        (tmp_path / "page.html").write_bytes(_get(url + "/")[2])
        r = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("viewer_nav.js")),
                            str(tmp_path / "page.html"), url + "/"], capture_output=True, text=True, timeout=120)
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert r.returncode == 0, r.stderr or r.stdout
    res = json.loads(r.stdout)
    assert not res["first"]["error"] and res["first"]["drawn"] == res["first"]["active"]
    # a 500 with a JSON error body is an error on the page, and is not kept as the episode
    failed = res["failed_read"]
    assert failed["error"] and failed["drawn"] is None and not failed["cached"]
    # opened again once the document reads, the episode is drawn
    again = res["after_repair"]
    assert not again["error"] and again["drawn"] == again["file"] == again["active"] and again["cached"]
    # Back while the comparison was on its way (its metrics, or its fade): the episode stays, as the URL says
    for case in ("back_while_metrics_load", "back_while_view_fades"):
        back = res[case]
        assert not back["compare_view"] and not back["fading"] and back["search"] == back["back_to"], case
        assert back["drawn"] == back["active"], case
    # the same steps without Back do open the comparison, so the cases above are not passing by doing nothing
    assert res["no_back"]["compare_view"] and res["no_back"]["search"].startswith("?view=compare")
    # what fills in a drawn episode (its frame grids, its dictionary rows) belongs to that draw: held while the
    # comparison opened and Back came before it appeared, it still lands on the episode Back kept (the page before
    # this fix dropped the grids and left their button hidden), as it does with no Back at all
    for kind in ("grids", "dictionary"):
        for how in ("back", "no_back"):
            case = res[kind + "_" + how]
            assert case["drawn"] == case["active"] == case["a"] and not case["compare_view"], (kind, how)
            assert case["grids"] == "fp:" + case["a"] and not case["grids_hidden"], (kind, how)
            assert case["dictionary"] == case["a"], (kind, how)
        # and once another episode is drawn in its place, the first one's late response is dropped
        case = res[kind + "_replaced"]
        assert case["drawn"] == case["active"] == case["b"], kind
        assert case["grids"] == "fp:" + case["b"] and case["dictionary"] == case["b"], kind
    # an episode drawn in full, then another whose read fails: every element of the page's markup ends as it does on a
    # page that only ever opened the failing one (the page before this fix kept the first episode's JSON download link)
    gone = res["failed_after_drawn"]
    assert gone["error"] and gone["reference_error"] and gone["drawn"] is None
    assert gone["left"] == [], gone["left"]
    # and the first episode had filled in the header, so the check is not passing on an empty draw
    assert gone["drawn_a"]["drawn"] == gone["a"]
    # Overlay controls now belong to the video pane and clear with its contents.
    assert gone["drawn_a"]["overlay_canvas"] and not gone["overlay_canvas"]
    assert {"current-ep-raw", "current-ep-src", "current-ep-reader", "left-col", "vd", "ms-btn", "dl-json"} <= set(
        gone["drawn_a"]["differs"])
    # random clicks, tabs, labellers, comparisons and Backs, each outrunning randomly late responses: the page always
    # ends showing what its URL says (the page before this fix ended wrong in 8 of these 25), with the drawn episode's
    # own frame grids and dictionary rows (the page that tied them to the navigation lost them in 4 of these 25, and in
    # 9 of 100 seeds)
    assert res["random"]["runs"] == 25 and res["random"]["wrong"] == []
