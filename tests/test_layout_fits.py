"""The board page holds its layout rule at every width it supports: no text runs out of its box.

The page checks itself under ?check=layout (board/index.html layoutDefects): once an episode is drawn it lists every
piece of text (or image, video or control) that runs past a box it sits in, every box that clips text without an
ellipsis, and the page scrolling sideways. This test serves a board whose content is larger than any board's so far
(targets of millions of episodes and hours, a long dataset name, long subgoal, key event and timeline labels, many
problems of one cause), opens each episode in headless Chrome at a phone's width and three desktop widths, and asks
for no defects. Without Chrome the test is skipped."""
from __future__ import annotations

import html
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from board import serve
from test_board_serve import _output, _sensors, _serve

WIDTHS = (390, 1200, 1440, 1920)
HEIGHT = 1000

LONG_DATASET = "a_dataset_of_our_own_recorded_across_many_sites_with_a_name_longer_than_any_tab"
LONG_WORD = "pick_up_the_extraordinarily_long_named_object_identifier_0123456789abcdef"


def chrome() -> str | None:
    """Google Chrome or Chromium: on Linux by name (the GitHub ubuntu-24.04 runner ships Google Chrome), on a Mac in
    /Applications."""
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        if found := shutil.which(name):
            return found
    for path in ("/usr/bin/google-chrome", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                 "/Applications/Chromium.app/Contents/MacOS/Chromium"):
        if os.access(path, os.X_OK):
            return path
    return None


def _labels(n_sub: int) -> dict:
    """A model reply far larger than usual: long subgoal and key event labels (one a single unbroken word), long
    timeline actions and objects, and long issues."""
    long = ("lifts the tall glass jar with the red screw lid off the second shelf of the cabinet and carries it across "
            "the counter to the open cardboard box beside the sink")
    events = [{"t_s": 1.0 + i, "label": f"subgoal {i + 1}: {long}", "kind": "subgoal_complete", "outcome": "success"}
              for i in range(n_sub)]
    events += [{"t_s": 1.5, "label": LONG_WORD, "kind": "processing_pass_complete", "outcome": "unclear",
                "note": "the gripper closes on the lid and the jar slips once before it is lifted " * 3},
               {"t_s": 8.0, "label": f"goal reached: {long}", "kind": "goal_reached", "outcome": "success"}]
    return {"task_summary": f"The left arm {long}. " * 4,
            "timeline": [{"start_s": float(i), "end_s": float(i + 1), "arm": "left", "action": f"{long} ({i})",
                          "object": LONG_WORD, "contribution": "advancing", "progress": (i + 1) / 10}
                         for i in range(10)],
            "key_events": events,
            "data_issues": [{"issue": f"the image is blurred while the arm {long}", "category": "image_quality",
                             "severity": "medium"},
                            {"issue": LONG_WORD, "category": "an_unusually_long_category_name_nobody_planned_for",
                             "severity": "low"}],
            "operator_mistakes": [{"issue": f"drops the jar while it {long}", "category": "dropped_object",
                                   "severity": "high", "t_s": 3.0}]}


def _reader_issues(n: int) -> list[dict]:
    """n problems of one cause, one per recorded signal, as prepare records a file placed from both starts."""
    return [{"kind": "signal_gap", "signal": f"left_arm/joint_{i}/an_unusually_long_signal_field_name_{i}",
             "what": f"The left_arm/joint_{i}/an_unusually_long_signal_field_name_{i} signal stops for "
                     f"{12.5 + i} s while the arm {LONG_WORD}.", "t0_s": 2.0 + i / 10}
            for i in range(n)]


def oversized_board(tmp: Path) -> tuple[Path, list[str]]:
    """A board built by board/build.py with content larger than any so far, and the episode files to open: a teleop
    dataset with signals and a head-camera dataset of tasks whose name is longer than any tab, each against a target
    of millions of episodes and hours, each episode many hours long."""
    board = tmp / "boards" / "b"
    entries, files = [], []
    for k, (ds, rig, hub, target) in enumerate((("molmo", "teleop_arms", "allenai/MolmoAct2-BimanualYAM-Dataset",
                                  {"episodes": 9876543, "hours": 987654.3}),
                                 (LONG_DATASET, "ego_head", "you/" + LONG_DATASET,
                                  {"episodes": 12345678, "hours": 1234567.8}))):
        run = tmp / "runs" / ds / "r1"
        (run / "out").mkdir(parents=True)
        (run / "run.json").write_text(json.dumps({"run_id": "r1", "code": "abc1234", "kind": "full",
                                                  "status": "done", "slice": str(tmp / "episodes" / ds)}))
        for i in range(3):
            ep = tmp / "episodes" / ds / f"episode_{3 * k + i:06d}"
            ep.mkdir(parents=True)
            (ep / "context.json").write_text(json.dumps({
                "dataset": hub, "profile": rig, "duration_s": 3600.0 * 123.4,
                "prompt": f"{LONG_WORD} and then put every object back where it was found " * 2,
                "reader_issues": _reader_issues(40)}))
            if rig == "teleop_arms":
                _sensors(ep)
            out = _output(ep, "openai/gpt-6-astra", rig == "ego_head")
            out["labels"].update(_labels(6))
            if rig == "ego_head":
                out["labels"].pop("completion", None)
                out["labels"]["tasks"] = [{"task": f"task {k}: {LONG_WORD} {_labels(0)['task_summary']}",
                                           "outcome": ("success", "failure", "partial")[k % 3],
                                           "start_s": float(k), "end_s": float(k + 1)} for k in range(5)]
            (run / "out" / f"{ep.name}.json").write_text(json.dumps(out))
        entries.append({"dataset": ds, "run": str(run), "episodes": str(tmp / "episodes" / ds), "target": target})
    board.mkdir(parents=True)
    (board / "manifest.json").write_text(json.dumps({
        "board": "an oversized board whose name is longer than any board's name so far, to see that it wraps",
        "datasets": entries}))
    from board import build
    build.build(board)
    by_ds: dict[str, str] = {}
    for p in sorted((board / "qa").glob("episode_*.json")):
        by_ds.setdefault(json.loads(p.read_text()).get("dataset"), p.name)
    files = list(by_ds.values())
    assert len(files) == 2, by_ds
    return board, files


# A desktop browser's window has a smallest width (500px for headless Chrome), so the page is opened in a frame of
# exactly the width asked for, on a page of the same origin that copies the frame's check out to where --dump-dom sees it.
FRAME = """<!doctype html><html><head><meta charset="utf-8"></head><body style="margin:0">
<iframe id="f" src="{src}" style="display:block;border:0;width:{w}px;height:{h}px"></iframe>
<pre id="layout-result"></pre>
<script>
const f = document.getElementById('f'), out = document.getElementById('layout-result');
setInterval(() => {{
  const p = f.contentDocument && f.contentDocument.getElementById('layout-check');
  if (p && p.dataset.state === 'done') {{ out.textContent = p.textContent; out.dataset.state = 'done'; }}
}}, 200);
</script></body></html>"""


class FramedHandler(serve.Handler):
    """board/serve.py's handler, and /layout-frame?src=PATH&w=W&h=H: the page at PATH in a frame W x H px."""

    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path != "/layout-frame":
            return super().do_GET()
        q = dict(urllib.parse.parse_qsl(parsed.query))
        body = FRAME.format(src=html.escape(q["src"]), w=int(q["w"]), h=int(q["h"])).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def layout_check(browser: str, base: str, page: str, width: int, height: int = HEIGHT) -> dict:
    """The page's own layout check (?check=layout) of PAGE (a path with its query) in a frame WIDTH px wide, read from
    the DOM headless Chrome dumps once the page has settled."""
    src = page + ("&" if "?" in page else "?") + "check=layout"
    frame = f"{base}/layout-frame?" + urllib.parse.urlencode({"src": src, "w": width, "h": height})
    with tempfile.TemporaryDirectory() as profile:
        r = subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-sandbox", "--no-first-run",
                            "--hide-scrollbars", "--disable-extensions", f"--user-data-dir={profile}",
                            f"--window-size={max(width, 500) + 40},{height + 200}", "--virtual-time-budget=20000",
                            "--dump-dom", frame], capture_output=True, text=True, timeout=120)
    m = re.search(r'<pre id="layout-result" data-state="done">(.*?)</pre>', r.stdout, re.S)
    assert m, f"no layout check at {width}px: {r.stderr[-2000:]}"
    return json.loads(html.unescape(m.group(1)))


def check_all(browser: str, base: str, files: list[str]) -> list[dict]:
    """The layout check of every file at every width, four at a time."""
    jobs = [(f, w) for f in files for w in WIDTHS]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda j: layout_check(browser, base, f"/?ep={urllib.parse.quote(j[0])}", j[1]),
                                jobs))
    for (f, w), res in zip(jobs, results):
        res["asked"] = {"file": f, "width": w}
    return results


@pytest.mark.skipif(chrome() is None, reason=f"no Google Chrome or Chromium on this {platform.system()} machine")
def test_no_text_runs_out_of_its_box_at_any_width(tmp_path, monkeypatch):
    board, files = oversized_board(tmp_path)
    monkeypatch.setattr(serve, "Handler", FramedHandler)
    httpd, url = _serve(monkeypatch, board, tmp_path / "clips")
    try:
        results = check_all(chrome(), url, files)
    finally:
        httpd.shutdown()
        httpd.server_close()
    for res in results:
        # the check ran on the episode asked for, at the width asked for
        assert (res["episode"], res["width"]) == (res["asked"]["file"], res["asked"]["width"]), res
    bad = [f"{r['width']}px {r['episode']}: {d['kind']} {d['el']} by {d['over']}px ({d['text']!r})"
           for r in results for d in r["defects"]]
    assert not bad, "\n".join(bad)
