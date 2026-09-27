"""The board's page and its two data sources: the server's rail records, episode view, page and endpoints
(board/serve.py), and the static build of the same page (board/static.py)."""
from __future__ import annotations

import argparse
import json
import shutil
import socketserver
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from board import serve, static


def _episode(dataset: str, rig: str, **extra) -> dict:
    """A board episode file as board/build.py writes it, with a short dense timeline."""
    d = {"dataset": dataset, "_rig": rig, "episode_prompt": "put the cup on the plate", "duration_s": 12.0,
         "event_labels": [{"t_s": 0.0, "end_s": 4.0, "arm": "left", "verb_class": "reach for the cup",
                           "contribution": "advancing", "progress": 0.3},
                          {"t_s": 4.0, "end_s": 12.0, "arm": "left", "verb_class": "place the cup",
                           "contribution": "advancing", "progress": 1.0}],
         "completion": {"task_completed": "success", "completed_at_s": 11.0},
         "_meta": {"episode_id": "", "model": "openai/gpt-6-astra"}, "_usage": {"est_cost_usd": 0.5}}
    d.update(extra)
    return d


def _board(tmp: Path) -> Path:
    """A board folder: a manifest, three episode files and two files the rail must skip."""
    board = tmp / "board"
    qa = board / "qa"
    qa.mkdir(parents=True)
    (board / "manifest.json").write_text(json.dumps({"board": "trial <one>", "datasets": []}))
    arms = _episode("galaxea", "teleop_arms",
                    data_issues=[{"issue": "the instruction names another object", "category": "instruction_mismatch",
                                  "severity": "medium"},
                                 {"issue": "slight blur", "category": "image_quality", "severity": "low"}],
                    operator_mistakes=[{"issue": "drops the cup", "category": "dropped_object", "severity": "high"}])
    arms["_meta"]["episode_id"] = "episode_000001"
    head = _episode("my_own_rig", "ego_head",
                    tasks=[{"task": "wash a cup", "outcome": "success", "completed_at_s": 5.0},
                           {"task": "dry it", "outcome": "failure"}])
    head["event_labels"][1]["hands_visible"] = False
    head["_meta"]["episode_id"] = "episode_000002"
    grip = _episode("fastumi", "handheld_gripper", camera_views=["left", "right"])
    grip["_meta"]["episode_id"] = "episode_000003"
    for name, d in (("episode_000001", arms), ("episode_000002", head), ("episode_000003", grip)):
        (qa / f"{name}.json").write_text(json.dumps(d))
    (qa / "notes.json").write_text(json.dumps({"about": "not an episode"}))
    (qa / "broken.json").write_text("{")
    return board


def test_rail_records_and_episode_view(tmp_path):
    board = _board(tmp_path)
    recs = {r["episode_id"]: r for r in serve.rail_records(board / "qa")}
    assert set(recs) == {"episode_000001", "episode_000002", "episode_000003"}
    arms = recs["episode_000001"]
    assert (arms["n_issues"], arms["n_minor_issues"], arms["max_severity"]) == (1, 1, "medium")
    assert arms["top_issue"] == "the instruction names another object"
    assert arms["n_mistakes"] == 1 and arms["families"] and arms["hands_hidden_s"] is None
    head = recs["episode_000002"]
    assert (head["n_tasks"], head["n_task_success"], head["dataset"]) == (2, 1, "my_own_rig")
    assert head["hands_hidden_s"] == pytest.approx(8.0)
    # every issue the page shows carries its family and whether it counts, from the one counting rule
    view = serve.episode_view(json.loads((board / "qa" / "episode_000001.json").read_text()))
    assert [i["counted"] for i in view["data_issues"]] == [True, False]
    assert all(i["family"] for i in view["data_issues"] + view["operator_mistakes"])


def test_render_index_fills_every_placeholder():
    page = serve.render_index("Data <Board>", {"mode": "api", "compare": False}, "trial <one>")
    assert "<title>Data &lt;Board&gt;</title>" in page and '<span class="ph-board">trial &lt;one&gt;</span>' in page
    assert "__" not in page.split("<script>")[1].split("const BOARD")[0]
    for marker in ("__PAGE_TITLE__", "__BOARD_NAME__", "__BOARD_CONFIG__", "__TAG_NAMES__", "__FAMILIES__"):
        assert marker not in page
    cfg = json.loads(page.split("const BOARD = ", 1)[1].split(";\n", 1)[0])
    assert cfg["mode"] == "api" and cfg["models"].get("openai/gpt-6-astra") == "Astra"
    assert "Pantheon" not in page and "\u2014" not in page and "\u2013" not in page


def test_page_script_parses(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    js = serve.render_index("Data Board", {"mode": "api"}).split("<script>", 1)[1].split("</script>", 1)[0]
    (tmp_path / "page.js").write_text(js)
    r = subprocess.run([node, "--check", str(tmp_path / "page.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.fixture
def server(tmp_path, monkeypatch):
    board = _board(tmp_path)
    clips = tmp_path / "clips"
    (clips / "wrist_left").mkdir(parents=True)
    (clips / "episode_000001.mp4").write_bytes(bytes(range(100)))
    monkeypatch.setattr(serve, "HERE", (board / "qa").resolve())
    monkeypatch.setattr(serve, "MP4_DIR", clips.resolve())
    monkeypatch.setattr(serve, "COMPARE_DIR", (board / "compare").resolve())
    monkeypatch.setattr(serve, "HANDS_DIR", (board / "hands").resolve())
    monkeypatch.setattr(serve, "BOARD_NAME", serve.board_name(board))
    monkeypatch.setattr(serve, "_LIST_CACHE", {"sig": None, "val": None})
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), serve.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url: str, headers: dict | None = None, data: bytes | None = None):
    req = urllib.request.Request(url, headers=headers or {}, data=data)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_server_endpoints(server):
    code, _, body = _get(server + "/")
    page = body.decode()
    assert code == 200 and "<h1>Data Board</h1>" in page and "trial &lt;one&gt;" in page
    assert '"compare":false' in page           # no comparisons on this board, so the page never asks for them
    code, _, body = _get(server + "/api/episodes")
    assert code == 200 and len(json.loads(body)) == 3
    code, _, body = _get(server + "/api/episode?file=episode_000001.json")
    assert code == 200 and json.loads(body)["data_issues"][0]["counted"] is True
    code, headers, body = _get(server + "/api/episode?file=episode_000001.json&download=1")
    assert "attachment" in headers["Content-Disposition"] and "counted" not in body.decode()   # the file as stored
    assert _get(server + "/api/episode?file=../manifest.json")[0] == 404
    assert _get(server + "/api/compare/index")[0] == 404
    assert _get(server + "/api/hands?file=episode_000002.json")[0] == 404
    code, _, body = _get(server + "/api/export", {"Content-Type": "application/json"},
                         json.dumps({"files": ["episode_000001.json", "episode_000003.json"]}).encode())
    assert code == 200 and [json.loads(x)["dataset"] for x in body.decode().splitlines()] == ["galaxea", "fastumi"]
    assert _get(server + "/api/export", {}, json.dumps({"files": ["../manifest.json"]}).encode())[0] == 404
    # the clips are served with byte ranges, so the player can seek
    code, headers, body = _get(server + "/api/video?id=episode_000001&cam=exo", {"Range": "bytes=10-19"})
    assert code == 206 and body == bytes(range(10, 20)) and headers["Content-Range"] == "bytes 10-19/100"
    assert _get(server + "/api/video?id=episode_000001", {"Range": "bytes=200-"})[0] == 416
    assert _get(server + "/api/video?id=episode_000001&cam=left")[0] == 404
    assert _get(server + "/api/video?id=../board/manifest")[0] == 404


def test_static_site(tmp_path):
    board = _board(tmp_path)
    clips = tmp_path / "clips"
    clips.mkdir()
    eps = {e["eid"]: e for e in static.plan(board / "qa", clips)}
    assert list(eps["episode_000002"]["media"]) == ["exo"]               # a head camera is shown alone
    assert list(eps["episode_000003"]["media"]) == ["left", "right"]     # grippers only: no top camera
    assert set(eps["episode_000001"]["frames"]) >= {"exo|0", "exo|11000"}  # the poster and the goal frame
    a = argparse.Namespace(board=board, qa=board / "qa", clips=clips, compare=None, hands=None, out=tmp_path / "out",
                           build_id="b1", force=False, public_base="https://example.org/board/", title="Data Board")
    assert static.cmd_site(a) == 0
    site = tmp_path / "out" / "b1"
    page = (site / "index.html").read_text()
    assert "<h1>Data Board</h1>" in page and '"mode":"static"' in page and '"compare":false' in page
    assert '"data":"https://example.org/board/b1/data/"' in (site / "index.public.html").read_text()
    index = json.loads((site / "data" / "index.json").read_text())
    assert index["datasets"] == ["galaxea", "my_own_rig", "fastumi"] and len(index["eps"]) == 3
    lst = json.loads((site / "data" / "lists" / "fastumi.json").read_text())
    assert lst[0]["_media"]["left"].startswith("v/left/episode_000003.") and lst[0]["_frames"] == {}
    assert json.loads((site / "data" / "ep" / "episode_000001.json").read_text())["data_issues"][1]["counted"] is False
    build = json.loads((site / "BUILD.json").read_text())
    assert build["episodes"] == 3 and build["media"]["videos_no_source"] == build["media"]["videos"] == 6
