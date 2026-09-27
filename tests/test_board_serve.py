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


def _serve(monkeypatch, board: Path, clips: Path):
    """board/serve.py's server on a free port, as `python -m board serve --board BOARD --clips CLIPS` sets it up."""
    monkeypatch.setattr(serve, "HERE", (board / "qa").resolve())
    monkeypatch.setattr(serve, "MP4_DIR", clips.resolve())
    monkeypatch.setattr(serve, "COMPARE_DIR", (board / "compare").resolve())
    monkeypatch.setattr(serve, "HANDS_DIR", (board / "hands").resolve())
    monkeypatch.setattr(serve, "KEYPOINTS_DIR", (board / "hand_keypoints").resolve())
    monkeypatch.setattr(serve, "BOARD_NAME", serve.board_name(board))
    monkeypatch.setattr(serve, "_LIST_CACHE", {})
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), serve.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


@pytest.fixture
def server(tmp_path, monkeypatch):
    board = _board(tmp_path)
    clips = tmp_path / "clips"
    (clips / "wrist_left").mkdir(parents=True)
    (clips / "episode_000001.mp4").write_bytes(bytes(range(100)))
    httpd, url = _serve(monkeypatch, board, clips)
    yield url
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
    a = argparse.Namespace(board=board, qa=board / "qa", clips=clips, compare=None, hands=None, keypoints=None,
                           out=tmp_path / "out", build_id="b1", force=False, public_base="https://example.org/board/",
                           title="Data Board")
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


# ---------------------------------------------------------------- a board built from runs, with a comparison

def _output(ep_dir: Path, model: str, head: bool, parsed: bool = True) -> dict:
    """One harness output (label/harness.py): a teleop or gripper episode with one outcome, or a head-camera session
    of tasks."""
    labels = {"task_summary": "The left arm lifts the cup.",
              "timeline": [{"start_s": 0.0, "end_s": 6.0, "arm": "left", "action": "lift", "object": "cup",
                            "contribution": "advancing", "progress": 0.5}],
              "key_events": [{"t_s": 5.0, "label": "cup lifted", "kind": "processing_pass_complete",
                              "outcome": "success"}],
              "data_issues": [{"issue": "the image is blurred", "category": "image_quality", "severity": "medium"}],
              **({"tasks": [{"task": "wash a cup", "outcome": "success", "start_s": 0.0, "end_s": 5.0}]} if head
                 else {"completion": {"task_completed": "success", "completed_at_s": 9.0}})}
    return {"episode_dir": str(ep_dir), "parse_ok": parsed, "model": model, "given_prompt": "lift the cup",
            "prompt_mode": "given", "usage": {"est_cost_usd": 0.1, "latency_s": 12.0, "prompt_tokens": 900,
                                              "completion_tokens": 400},
            "config": {"timesteps_s": [0.0, 4.0, 8.0], "views": ["exo"], "cam_labels": ["top"]},
            "labels": labels if parsed else {"_raw": "{", "_parse_error": "JSONDecodeError"}}


def _built_board(tmp: Path) -> Path:
    """Three datasets, one episode each (a public teleop dataset, a public head-camera dataset with hand keypoints,
    and a dataset of your own), built by board/build.py, and one other model over two of the episodes, one of whose
    responses did not parse."""
    board, cmp_slice = tmp / "boards" / "b", tmp / "episodes" / "compare"
    cmp_slice.mkdir(parents=True)
    entries = []
    for i, (ds, rig, hub) in enumerate((("molmo", "teleop_arms", "allenai/MolmoAct2-BimanualYAM-Dataset"),
                                        ("openaoe", "ego_head", "inclusionAI/OpenAoE-2000h"),
                                        ("mine", "handheld_gripper", "you/your-own-dataset"))):
        ep = tmp / "episodes" / ds / f"episode_{i:06d}"
        ep.mkdir(parents=True)
        (ep / "context.json").write_text(json.dumps({"dataset": hub, "profile": rig, "duration_s": 10.0}))
        run = tmp / "runs" / ds / "r1"
        (run / "out").mkdir(parents=True)
        (run / "run.json").write_text(json.dumps({"run_id": "r1", "code": "abc1234", "kind": "full", "status": "done",
                                                  "slice": str(ep.parent)}))
        (run / "out" / f"{ep.name}.json").write_text(json.dumps(_output(ep, "openai/gpt-6-astra", rig == "ego_head")))
        entries.append({"dataset": ds, "run": str(run), "episodes": str(ep.parent)})
        if ds != "mine":
            (cmp_slice / ep.name).symlink_to(ep)
    other = tmp / "runs" / "compare" / "other"
    (other / "out").mkdir(parents=True)
    (other / "run.json").write_text(json.dumps({"run_id": "other", "code": "abc1234", "kind": "full",
                                                "status": "done", "slice": str(cmp_slice)}))
    for name, head, parsed in (("episode_000000", False, True), ("episode_000001", True, False)):
        (other / "out" / f"{name}.json").write_text(json.dumps(_output(cmp_slice / name, "vendor/other", head,
                                                                       parsed)))
    board.mkdir(parents=True)
    (board / "manifest.json").write_text(json.dumps({"board": "built", "datasets": entries, "comparisons": [
        {"key": "other", "name": "Other model", "run": str(other)}]}))
    from board import build
    build.build(board)
    # the head-camera episode's hand keypoints download (board/hands.py build_keypoints writes these)
    (board / "hand_keypoints").mkdir()
    (board / "hand_keypoints" / "episode_000001.json").write_text(json.dumps({"format": "pantheon-hand-keypoints/1"}))
    (board / "hand_keypoints" / "index.json").write_text(json.dumps(
        {"files": {"episode_000001.json": {"frames": 300, "bytes": 38}}}))
    return board


def test_comparison_lists_and_keypoint_downloads_are_served(tmp_path, monkeypatch):
    board = _built_board(tmp_path)
    httpd, url = _serve(monkeypatch, board, tmp_path / "clips")
    try:
        page = _get(url + "/")[2].decode()
        cfg = json.loads(page.split("const BOARD = ", 1)[1].split(";\n", 1)[0])
        assert cfg["compare"] is True and cfg["keypoints"] is True and cfg["hands"] is False
        index = json.loads(_get(url + "/api/compare/index")[2])
        assert index["reference"]["name"] == "Astra" and [m["key"] for m in index["models"]] == ["other"]
        code, _, body = _get(url + "/api/compare/list?key=other")
        recs = {r["file"]: r for r in json.loads(body)}
        assert code == 200 and set(recs) == {"episode_000000.json", "episode_000001.json"}
        assert recs["episode_000001.json"]["cmp_status"] == "unparsed" and recs["episode_000000.json"]["n_issues"] == 1
        for bad in ("../qa", "", ".", "nope"):
            assert _get(url + "/api/compare/list?key=" + bad)[0] == 404
        # the footage line's source travels in the episode, the comparison's copy included
        ep = json.loads(_get(url + "/api/episode?file=episode_000000.json")[2])
        assert ep["dataset_source"]["publisher"] == "Ai2"
        assert "dataset_source" not in json.loads(_get(url + "/api/episode?file=episode_000002.json")[2])
        cep = json.loads(_get(url + "/api/compare/episode?key=other&file=episode_000000.json")[2])
        assert cep["dataset_source"] == ep["dataset_source"]
        assert json.loads(_get(url + "/api/keypoints?file=index.json")[2])["files"]
        code, headers, body = _get(url + "/api/keypoints?file=episode_000001.json&download=1")
        assert code == 200 and 'filename="episode_000001.hand_keypoints.json"' in headers["Content-Disposition"]
        assert json.loads(body)["format"] == "pantheon-hand-keypoints/1"
        for bad in ("episode_000000.json", "../qa/episode_000000.json", ""):
            assert _get(url + "/api/keypoints?file=" + bad)[0] == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def _smoke(page: str, base: str) -> dict:
    """tests/render_smoke.js over a page as served: every episode, every model under Labels by, the comparison
    view."""
    r = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("render_smoke.js")), page, base],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr or r.stdout
    return json.loads(r.stdout)


def _check_smoke(res: dict) -> None:
    assert res["episodes"] == 3 and res["rendered"] >= 5 and res["rendered_comparisons"] >= 2
    assert res["footage_lines"] == 2 and res["keypoint_links"] == 1 and res["compare_view"]
    (lb,) = res["labellers"]
    assert lb["key"] == "other" and lb["episodes"] == 2 and "not Astra&rsquo;s labels" in lb["note"]
    assert "labels by Other model" in lb["band"]


@pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
def test_the_served_page_runs(tmp_path, monkeypatch):
    board = _built_board(tmp_path)
    httpd, url = _serve(monkeypatch, board, tmp_path / "clips")
    try:
        (tmp_path / "page.html").write_bytes(_get(url + "/")[2])
        _check_smoke(_smoke(str(tmp_path / "page.html"), url + "/"))
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
def test_the_static_page_runs(tmp_path):
    import functools
    import http.server
    board = _built_board(tmp_path)
    (tmp_path / "clips").mkdir()
    a = argparse.Namespace(board=board, qa=board / "qa", clips=tmp_path / "clips", compare=None, hands=None,
                           keypoints=None, out=tmp_path / "out", build_id="b1", force=False, public_base=None,
                           title="Data Board")
    assert static.cmd_site(a) == 0
    site = tmp_path / "out" / "b1"
    assert json.loads((site / "data" / "compare" / "lists" / "other.json").read_text())[0]["file"] == "episode_000000.json"
    kidx = json.loads((site / "data" / "keypoints" / "index.json").read_text())
    (path,) = [v["path"] for v in kidx["files"].values()]
    assert path.startswith("k/episode_000001.") and (tmp_path / "out" / "media" / path).exists()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path / "out"))
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        _check_smoke(_smoke(str(site / "index.html"), f"http://127.0.0.1:{httpd.server_address[1]}/b1/"))
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
def test_the_smoke_fails_on_a_page_that_throws(tmp_path, monkeypatch):
    """The smoke is only worth having if it fails: a name used before it exists, in the episode's render, must."""
    board = _built_board(tmp_path)
    httpd, url = _serve(monkeypatch, board, tmp_path / "clips")
    try:
        page = _get(url + "/")[2].decode()
        assert page.count("function renderEp(d, opts) {") == 1
        (tmp_path / "broken.html").write_text(page.replace("function renderEp(d, opts) {",
                                                           "function renderEp(d, opts) { notDefinedAnywhere;"))
        r = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("render_smoke.js")),
                            str(tmp_path / "broken.html"), url + "/"], capture_output=True, text=True, timeout=120)
        assert r.returncode == 1 and "notDefinedAnywhere is not defined" in r.stderr
    finally:
        httpd.shutdown()
        httpd.server_close()
