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


def test_public_label_leaves_out_how_the_label_was_made():
    """What the page, the downloads and the exports serve never names a run, a commit, a replaced label or frame
    verdicts; everything else in the label is kept as it is, and the label file itself is not changed."""
    d = {"dataset": "molmo", "completion": {"task_completed": "failure"}, "_meta": {"model": "m"}, "_rig": "teleop_arms",
         "_run": {"run_id": "r", "code": "abc1234"}, "_replaced_label": {"completion": {}}, "_supplements": [1],
         "_verification": [2], "_withheld_checks": [3], "_carried_verdicts": [4],
         "_compare": {"key": "k", "name": "N", "run_id": "r2", "code": "def5678", "status": "parsed"}}
    before = json.dumps(d, sort_keys=True)
    p = serve.public_label(d)
    assert not set(serve.PRIVATE_KEYS) & set(p)
    assert p["_compare"] == {"key": "k", "name": "N", "status": "parsed"}
    assert p["completion"] == d["completion"] and p["_meta"] == d["_meta"] and p["_rig"] == "teleop_arms"
    assert json.dumps(d, sort_keys=True) == before
    assert not set(serve.PRIVATE_KEYS) & set(serve.episode_view(dict(d)))


def test_the_page_is_given_the_checks_a_rule_withheld_with_the_reason():
    """A check a manifest rule withheld (drop_check) is still on the page, named with the rule's reason; the check's
    stored result stays in the label file."""
    d = {"dataset": "molmo", "_withheld_checks": {"gripper_channels": {"reason": "one-armed tasks",
                                                                       "result": {"flagged": True}}}}
    view = serve.episode_view(d)
    assert view["set_aside_checks"] == [{"check": "gripper_channels", "reason": "one-armed tasks", "status": "fired"}]
    assert "_withheld_checks" not in view
    assert "set_aside_checks" not in serve.episode_view({"_withheld_checks": [3]})
    # a withheld check that stopped with an error says so, never clear
    err = serve.episode_view({"_withheld_checks": {"recorded_jumps": {"reason": "r", "result": {
        "error": "ValueError: boom", "flagged": False}}}})
    assert err["set_aside_checks"] == [{"check": "recorded_jumps", "reason": "r", "status": "errored",
                                        "error": "ValueError: boom"}]


def test_a_withheld_check_reads_as_what_it_found():
    """A withheld check read clear whatever it found: the capture checks and the sensor checks keep their findings in
    a list of checks, and a check that measured nothing says so in not_assessed. Each reads as it came out: an error,
    not assessed with why, or how many of its own checks that ran fired and errored."""
    st = serve.withheld_status
    assert st({"not_assessed": "the right arm reads at too few frames", "flagged": False}) == {
        "status": "not_assessed", "why": "the right arm reads at too few frames"}
    capture = {"checks": [{"status": "fired"}, {"status": "fired"}, {"status": "clear"},
                          {"status": "not_applicable"}, {"status": "errored"}], "not_assessed": {"x": "y"}}
    assert st(capture) == {"status": "errored", "fired": 2, "errored": 1, "of": 4}
    sensors = {"flagged": False, "notes": [{"check": "constant"}], "checks": [{"status": "fired"}, {"status": "clear"},
                                                                             {"status": "na"}]}
    assert st(sensors) == {"status": "fired", "fired": 1, "errored": 0, "of": 2}
    assert st({"checks": [{"status": "na"}]})["status"] == "not_assessed"
    assert st({"checks": [{"status": "clear"}]})["status"] == "clear"
    assert st({"crossed": False, "left_vs_left": 0.9})["status"] == "clear"


def test_a_withheld_check_that_flagged_reads_fired_even_with_no_check_listed():
    """A withheld result that says it flagged, with no check of its own listed or none that ran, read not assessed."""
    st = serve.withheld_status
    assert st({"flagged": True, "checks": []}) == {"status": "fired", "fired": 0, "errored": 0, "of": 0}
    assert st({"flagged": True, "checks": [{"status": "na"}]})["status"] == "fired"
    assert st({"flagged": False, "checks": []})["status"] == "not_assessed"


def test_render_index_fills_every_placeholder():
    page = serve.render_index("Data <Board>", {"mode": "api", "compare": False}, "trial <one>")
    assert "<title>Data &lt;Board&gt;</title>" in page and '<span class="ph-board">trial &lt;one&gt;</span>' in page
    assert "__" not in page.split("<script>")[1].split("const BOARD")[0]
    for marker in ("__PAGE_TITLE__", "__BOARD_NAME__", "__BOARD_CONFIG__", "__TAG_NAMES__", "__FAMILIES__"):
        assert marker not in page
    cfg = json.loads(page.split("const BOARD = ", 1)[1].split(";\n", 1)[0])
    assert cfg["mode"] == "api" and cfg["models"].get("openai/gpt-6-astra") == "Astra"
    assert "Pantheon" not in page and "\u2014" not in page and "\u2013" not in page


def test_a_site_names_the_model_its_own_way():
    """A site that shows the board under its own names passes them in "models"; every other model keeps its name."""
    page = serve.render_index("Data Dashboard", {"mode": "static", "models": {"openai/gpt-6-astra": "the model"}})
    cfg = json.loads(page.split("const BOARD = ", 1)[1].split(";\n", 1)[0])
    assert cfg["models"]["openai/gpt-6-astra"] == "the model"
    assert cfg["models"] == {**json.loads(serve.render_index("x", {"mode": "api"}).split("const BOARD = ", 1)[1]
                                          .split(";\n", 1)[0])["models"], "openai/gpt-6-astra": "the model"}


def test_a_site_header_takes_the_place_of_the_title_bar():
    """A board served as part of a site shows the site's own header (--header): its markup where the title bar was,
    its styles in the page's head, and nothing else of the page changed."""
    header = '<style>.site { height: 76px; } :root { --header-h: 76px; }</style>\n<header class="site">Site</header>'
    plain = serve.render_index("Data Dashboard", {"mode": "api"}, "trial")
    page = serve.render_index("Data Dashboard", {"mode": "api"}, "trial", header)
    head, body = page.split("</head>", 1)
    assert '<header class="site">Site</header>' in body and 'class="page-head"' not in body
    assert "--header-h: 76px" in head and "<style>.site" not in body
    assert page.split("<script>", 1)[1] == plain.split("<script>", 1)[1]


def test_capture_check_families_are_keyed_by_the_check():
    """A capture check that fired is a family of its own under the check's id, so renaming a check never moves its
    episodes to another family; the page's catalog gives it the check's current name."""
    d = _episode("galaxea", "teleop_arms", dataset_checks={"capture_qc": {
        "checks": [{"check": "video_frozen_run", "name": "An older name", "status": "fired", "shown_as": "issue"}],
        "flags": [{"check": "video_frozen_run", "title": "the left camera repeats one frame"}]}})
    fams = serve._families(d)["families"]
    assert "cq:video_frozen_run" in fams and not any(f.startswith("d:An older") for f in fams)
    from checks.capture_qc import NAMES
    assert serve.capture_catalog()["cq:video_frozen_run"] == {"name": NAMES["video_frozen_run"][0], "list": "data",
                                                             "check": True}


def test_frames_are_kept_on_disk(tmp_path, monkeypatch):
    """With BOARD_FRAME_DIR a frame cut once is served from disk after a restart (an empty memory cache), with no
    ffmpeg run."""
    clip = tmp_path / "episode_000001.mp4"
    clip.write_bytes(b"not a video")
    monkeypatch.setattr(serve, "_FRAME_DIR", tmp_path / "frames")
    monkeypatch.setattr(serve, "FFMPEG", "/no/ffmpeg/here")
    monkeypatch.setattr(serve, "_FRAME_CACHE", serve.OrderedDict())
    key = (str(clip), clip.stat().st_size, clip.stat().st_mtime_ns, 1.5, 640)
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / (serve.hashlib.sha1(repr(key).encode()).hexdigest() + ".jpg")).write_bytes(b"jpeg")
    assert serve.extract_frame(clip, 1.5, 640) == b"jpeg"
    assert serve.extract_frame(clip, 2.5, 640) is None          # not on disk, and ffmpeg cannot run


def test_one_connection_serves_several_requests(server):
    """The server keeps a connection open (HTTP/1.1), so a page's labels, posters and videos share one."""
    import http.client
    host, port = server.split("//")[1].split(":")
    c = http.client.HTTPConnection(host, int(port), timeout=10)
    for path in ("/api/episodes", "/api/episode?file=episode_000001.json", "/api/video?id=episode_000001&cam=exo"):
        c.request("GET", path, headers={"Range": "bytes=0-9"} if "video" in path else {})
        r = c.getresponse()
        assert r.status in (200, 206) and r.version == 11
        r.read()
    c.close()


def test_page_script_parses(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    js = serve.render_index("Data Dashboard", {"mode": "api"}).split("<script>", 1)[1].split("</script>", 1)[0]
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
    monkeypatch.setattr(serve, "SENSORS_DIR", (board / "sensors").resolve())
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
    assert code == 200 and "<h1>Data Dashboard</h1>" in page and "trial &lt;one&gt;" in page
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
                           title="Data Dashboard")
    assert static.cmd_site(a) == 0
    site = tmp_path / "out" / "b1"
    page = (site / "index.html").read_text()
    assert "<h1>Data Dashboard</h1>" in page and '"mode":"static"' in page and '"compare":false' in page
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


def _sensors(ep: Path) -> None:
    """The teleop episode's other signals and depth (board/sensors.py reads them): a gripper's effort that rests and
    rises, a 4 x 4 pressure map, a constant flag, and a depth stream on its main camera; and the contact prepare found
    in the effort (label/contacts.py)."""
    import numpy as np
    n = 300
    eff = np.zeros((n, 1), np.float32)
    eff[60:120] = 4.0
    pmap = (3072.0 + np.random.default_rng(0).normal(0, 2, (n, 16))).astype(np.float32)   # a real sensor's noise
    pmap[150:200, 6] -= 3072.0 - 1800.0
    np.savez(ep / "signals.npz", s0=eff, s1=pmap, s2=np.ones((n, 1), np.float32))
    np.save(ep / "depth_kmap_exo.npy", np.arange(n))
    (ep / "depth.json").write_text(json.dumps({"exo": {"packed": str(ep / "depth.mkv"), "base_s": 0.0, "n_frames": n,
                                                        "kmap": "depth_kmap_exo.npy", "scale_m": None}}))
    (ep / "sources.json").write_text(json.dumps({"exo": {"packed": str(ep / "exo.mp4"), "base_s": 0.0,
                                                          "n_frames": n}}))
    ctx = json.loads((ep / "context.json").read_text())
    (ep / "context.json").write_text(json.dumps({**ctx, "fps": 30, "n_state_frames": n, "depth": {"exo": {
        "units": "relative", "scale_m": None}}, "signals": [
        {"name": "gripper effort", "key": "s0", "dims": 1},
        {"name": "pressure", "key": "s1", "dims": 16, "shape": [4, 4]},
        {"name": "health", "key": "s2", "dims": 1, "names": ["ok"]}], "contacts": [
        {"id": "c1", "hand": "right", "signals": ["gripper effort"], "start_s": 2.0, "peak_s": 2.5, "end_s": 3.967,
         "from_start": False, "to_end": False, "peak_strength": 1.0, "regions": {}, "dips_s": []}]}))


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
        if ds == "molmo":
            _sensors(ep)
        run = tmp / "runs" / ds / "r1"
        (run / "out").mkdir(parents=True)
        (run / "run.json").write_text(json.dumps({"run_id": "r1", "code": "abc1234", "kind": "full", "status": "done",
                                                  "slice": str(ep.parent)}))
        # the dataset of your own: its model reply did not parse, so its episode has no labels
        out = _output(ep, "openai/gpt-6-astra", rig == "ego_head", ds != "mine")
        if ds == "molmo":
            # the model's answer for the episode's one contact, and a grasp it saw that no contact covers
            out["labels"]["contacts"] = [{"id": "c1", "touch_seen": "yes", "first_touch_frame": 3,
                                          "last_touch_frame": 2, "hand": "right", "object": "cup", "grip": "pinch",
                                          "action": "lifts it", "slip": "no", "notes": None}]
            out["labels"]["contacts_missing"] = [{"t_s": 7.0, "hand": "left", "object": "lid"}]
            out["contact_views"] = {"shown": ["c1"], "strips": {"c1": {"begin": [1.7, 1.85, 2.0, 2.15, 2.3],
                                                                       "end": [3.817, 3.967, 4.117]}}}
        (run / "out" / f"{ep.name}.json").write_text(json.dumps(out))
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
    assert res["sensors_panels"] == 1          # the teleop episode's signals and depth, drawn under its timeline
    # the episode whose reply did not parse: one line in place of the label sections, and its card says no labels
    assert res["no_label_pages"] == 1 and res["answered_on_failed"] == 0 and res["no_label_cards"] == 1
    assert res["touch_lanes"] == 1 and res["contact_cards"] == 1 and res["contact_checks"] == 1   # and its contact
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
                           title="Data Dashboard")
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


def _page_js() -> str:
    return serve.render_index("t", {"mode": "static", "data": "data/", "media": "M/"}).split("<script>", 1)[1] \
        .split("</script>", 1)[0]


@pytest.mark.skipif(not shutil.which("node"), reason="node is not installed")
def test_static_board_plays_each_extra_camera_its_own_clip(tmp_path):
    """board/static.py (and Data Review's static board) publish extra1..3 under their own _media keys; the page's
    videoSrc must play them, not the main camera's clip."""
    js = _page_js()
    start = js.index("function mediaKey(cam)")
    end = js.index("// the camera's first frame as the video's poster attribute")
    body = js[start:end]
    prog = ("const STATIC = true, BOARD = {media: 'M/'};\n"
            "let _activeFile = 'f.json';\n"
            "const ALL_EPS = [{file: 'f.json', _media: {exo: 'v/exo/e.mp4', left: 'v/left/e.mp4', "
            "right: 'v/right/e.mp4', extra1: 'v/extra1/e.mp4', extra2: 'v/extra2/e.mp4'}}];\n"
            + body +
            "\nconsole.log(JSON.stringify(['exo', 'left', 'right', 'extra1', 'extra2'].map(c => videoSrc('e', c))));")
    (tmp_path / "t.js").write_text(prog)
    r = subprocess.run([shutil.which("node"), str(tmp_path / "t.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == ["M/v/exo/e.mp4", "M/v/left/e.mp4", "M/v/right/e.mp4", "M/v/extra1/e.mp4",
                                    "M/v/extra2/e.mp4"]


def test_an_extra_cameras_download_is_named_after_that_camera(tmp_path, monkeypatch):
    """/api/video?download=1 names the saved file after the camera; an extra camera must not be saved under the
    main camera's name (both would be <episode>_main.mp4 and overwrite each other)."""
    board = _board(tmp_path)
    clips = tmp_path / "clips"
    (clips / "extra1").mkdir(parents=True)
    (clips / "episode_000001.mp4").write_bytes(b"main")
    (clips / "extra1" / "episode_000001.mp4").write_bytes(b"extra")
    httpd, url = _serve(monkeypatch, board, clips)
    try:
        _, h_main, b_main = _get(url + "/api/video?id=episode_000001&cam=exo&download=1")
        _, h_x, b_x = _get(url + "/api/video?id=episode_000001&cam=extra1&download=1")
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert (b_main, b_x) == (b"main", b"extra")
    assert h_main["Content-Disposition"] != h_x["Content-Disposition"], h_x["Content-Disposition"]


def test_a_static_build_made_before_its_frames_gets_a_new_id_once_they_exist(tmp_path):
    """The page lists only the frames on disk, so the frames present are part of the build's id: a build made before
    the media finish is followed by a new build once they have, never refused as the same build."""
    qa = tmp_path / "qa"
    qa.mkdir()
    (qa / "episode_000001.json").write_text("{}")
    before = static.build_id_for(qa, ["episode_000001.json"], frames=[])
    after = static.build_id_for(qa, ["episode_000001.json"], frames=["f/exo/episode_000001.jpg"])
    assert before != after and after == static.build_id_for(qa, ["episode_000001.json"],
                                                            frames=["f/exo/episode_000001.jpg"])


def test_a_part_of_a_long_recording_that_was_undone_counts_like_a_short_episode_that_was(tmp_path):
    """label/pieces.py stitch puts each part's outcome under tasks and leaves completion empty. A short
    episode whose outcome is success_then_undone raises the "undone" data-issue family (families.json completion);
    the same outcome on a part of a long recording raises nothing, and the rail counts that part as not done."""
    from board import to_board
    stitched = {"parse_ok": True, "labels": {"completion": {"task_completed": None},
                                             "tasks": [{"start_s": 0, "end_s": 250, "task": "stack", "outcome": "success"},
                                                       {"start_s": 250, "end_s": 500, "task": "stack",
                                                        "outcome": "success_then_undone", "completed_at_s": 300}]}}
    short = {"parse_ok": True, "labels": {"completion": {"task_completed": "success_then_undone",
                                                         "goal_reached_at_s": 50, "undone_at_s": 60}}}
    recs = {}
    for name, r in (("episode_long", stitched), ("episode_short", short)):
        d = to_board.convert(r, "own")
        d["_meta"]["episode_id"] = name
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(d))
        recs[name] = serve._rail_record(p, d)
    assert "undone" in recs["episode_short"]["families"]
    assert "undone" in recs["episode_long"]["families"]


def test_the_list_is_encoded_once_and_follows_the_files(server):
    """The page asks for the whole list on every load: the server encodes it once per list, gzipped or not, and a
    changed episode file still changes the answer."""
    import gzip as _gzip
    code, headers, body = _get(server + "/api/episodes", {"Accept-Encoding": "gzip"})
    plain = json.dumps(serve.list_episodes()).encode()
    got = _gzip.decompress(body) if headers.get("Content-Encoding") == "gzip" else body
    assert code == 200 and got == plain
    raw, gz = serve.list_json()
    assert serve.list_json()[1] is gz                      # the second request reuses the encoded body
    assert _gzip.decompress(gz) == raw == plain
    f = serve.HERE / "episode_000003.json"
    d = json.loads(f.read_text())
    d["episode_prompt"] = "a different instruction, long enough to change the file's size"
    f.write_text(json.dumps(d))
    code, _, body = _get(server + "/api/episodes")
    assert code == 200 and any(e.get("episode_prompt") == d["episode_prompt"] for e in json.loads(body))



def test_a_long_recording_with_a_part_not_labelled_never_reads_complete_on_its_card(tmp_path):
    d = _episode("mine", "teleop_arms",
                 tasks=[{"task": "a", "outcome": "success"}, {"task": "b", "outcome": "success"}],
                 _stitched={"parts": 3, "cuts_s": [300.0, 600.0], "missing": [{"part": 2, "t0_s": 300.0,
                                                                              "t1_s": 600.0, "why": "x"}]})
    rec = serve._rail_record(tmp_path / "episode_000001.json", d)
    assert rec["parts_missing"] == 1 and rec["parts"] == 3
    assert "parts_missing" not in serve._rail_record(tmp_path / "episode_000001.json", _episode("mine", "teleop_arms"))
    r = subprocess.run([shutil.which("node"), str(Path(__file__).with_name("card_outcome.js")),
                        str(Path(__file__).resolve().parent.parent / "board" / "serve.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_board_file_with_a_number_that_is_not_finite_is_served_as_json_a_browser_reads(server):
    """A board file built before non finite numbers were written as null still loads: the server writes them as null
    in the episode, the list and the export, never as NaN, which the page's JSON.parse rejects."""
    qa = serve.HERE
    d = json.loads((qa / "episode_000001.json").read_text())
    d["dataset_checks"] = {"stream_pairing": {"left_vs_left": float("nan"), "crossed": False}}
    d["event_labels"][0]["t_s"] = float("nan")
    d["duration_s"] = float("inf")
    (qa / "episode_000001.json").write_text(json.dumps(d))

    def strict(body: bytes):
        def no(c):
            raise ValueError(f"{c} is not JSON")
        return json.loads(body, parse_constant=no)
    code, _, body = _get(server + "/api/episode?file=episode_000001.json")
    assert code == 200 and strict(body)["dataset_checks"]["stream_pairing"]["left_vs_left"] is None
    code, _, body = _get(server + "/api/episodes")
    assert code == 200 and len(strict(body)) == 3
    code, _, body = _get(server + "/api/episode?file=episode_000001.json&download=1")
    assert code == 200 and strict(body)["event_labels"][0]["t_s"] is None
    code, _, body = _get(server + "/api/export", {"Content-Type": "application/json"},
                         json.dumps({"files": ["episode_000001.json"]}).encode())
    assert code == 200 and strict(body.decode().splitlines()[0])["duration_s"] is None
