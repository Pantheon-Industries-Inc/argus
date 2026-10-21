"""The board page holds its layout rule at every width it supports: no text runs out of its box.

The page checks itself under ?check=layout (board/index.html layoutDefects): once an episode is drawn it lists every
piece of text (or image, video or control) that runs past a box it sits in, every box that clips text without an
ellipsis, and the page scrolling sideways. This test serves a board whose content is larger than any board's so far
(targets of millions of episodes and hours, a long dataset name, long subgoal, key event and timeline labels, many
problems of one cause, an upload's deep notes, long check lists), with short clips of the shapes real cameras have (a
wide scene or head camera, a square gripper camera) so every box over the footage is placed on a frame of a real size,
opens each episode in headless Chrome at a phone's width and three desktop widths, once as it opens and once with every
fold opened (&open=all, where a fold left closed is a defect too), and asks for no defects. Without Chrome or ffmpeg the
test is skipped."""
from __future__ import annotations

import fcntl
import html
import json
import os
import platform
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from board import clips, serve
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


def _uploader_notes() -> dict:
    """An upload's notes far deeper and longer than usual: a table row and notes from its files, nested many levels
    under long field names, with long paths as values."""
    deep = {"recording_session_metadata_with_a_long_name": {"left_arm_configuration": {"joint_calibration_offsets": {
        f"joint_{i}_calibration_offset_in_radians_relative_to_the_factory_zero": {
            "source_file": f"/mnt/shared/uploads/{LONG_DATASET}/sessions/2026-10-20/rig_07/calibration/joint_{i}.yaml",
            "value": 0.0123 * i, "note": f"measured twice; the second reading is the one kept {LONG_WORD}"}
        for i in range(8)}}}}
    return {"table rows": [{"table": "episodes_with_an_unusually_long_table_name.parquet", **deep}], "notes": deep}


def _capture_checks() -> dict:
    """A capture-check result (checks/capture_qc.py) with every check listed and more than five fired, each with long
    evidence, so the fired list and the full list both fold."""
    from checks.capture_qc import NAMES
    names = list(NAMES)
    fired = names[:9]
    return {"checks": [{"check": c, "status": "fired" if c in fired else "clear", "shown_as": "issue",
                        "why": f"{c} {LONG_WORD}"} for c in names],
            "flags": [{"check": c, "evidence": f"/mnt/shared/uploads/{LONG_DATASET}/{c}/{LONG_WORD}.parquet fails "
                                                f"at rows 1 to 400 of the state stream recorded by the left arm"}
                      for c in fired],
            "notes": []}


def _clip(path: Path, size: str) -> None:
    """A short clip of ffmpeg's test picture at SIZE (w x h), encoded as the board's clips are (H.264)."""
    subprocess.run([clips.find_ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30",
                    "-t", "3", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-movflags",
                    "+faststart", str(path)], check=True)


def oversized_board(tmp: Path, clip_dir: Path) -> tuple[Path, list[str]]:
    """A board built by board/build.py with content larger than any so far, and the episode files to open: a teleop
    dataset with signals, a head-camera dataset of tasks whose name is longer than any tab, and a handheld gripper's
    dataset of one camera (whose caption sits in the strip under it), each against a target of millions of episodes
    and hours, each episode many hours long. Much of it lands inside folds: twenty key events, forty problems of one
    cause, an upload's deep notes, and a capture-check list with nine fired. Each episode's camera has a clip in
    CLIP_DIR where board/clips.py writes it: a wide one for the scene and head cameras, a square one for the gripper."""
    board = tmp / "boards" / "b"
    wide, square = tmp / "wide.mp4", tmp / "square.mp4"
    _clip(wide, "640x360")
    _clip(square, "480x480")
    entries, files = [], []
    for k, (ds, rig, hub, target) in enumerate((("molmo", "teleop_arms", "allenai/MolmoAct2-BimanualYAM-Dataset",
                                  {"episodes": 9876543, "hours": 987654.3}),
                                 (LONG_DATASET, "ego_head", "you/" + LONG_DATASET,
                                  {"episodes": 12345678, "hours": 1234567.8}),
                                 ("handheld", "handheld_gripper", "you/a-handheld-gripper-dataset",
                                  {"episodes": 7654321, "hours": 765432.1}))):
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
                "reader_issues": _reader_issues(40), "uploader_notes": _uploader_notes(),
                **({"capture_qc": _capture_checks()} if rig == "teleop_arms" else {})}))
            if rig == "teleop_arms":
                _sensors(ep)
            out = _output(ep, "openai/gpt-6-astra", rig == "ego_head")
            out["labels"].update(_labels(18))
            # a handheld gripper records one camera, mounted on the gripper (the right one, as FastUMI's single one)
            cam = "right" if rig == "handheld_gripper" else "exo"
            out["config"].update(views=[cam], cam_labels=["gripper" if cam == "right" else "top"])
            clip = clips.clip_path(clip_dir, ep.name, cam)
            clip.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(square if cam == "right" else wide, clip)
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
    assert len(files) == 3, by_ds
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


def _devtools(browser: str, args: list[str]):
    """A headless Chrome driven over its DevTools pipe (--remote-debugging-pipe: commands on fd 3, replies on fd 4,
    each a JSON message ended by a NUL), and send(method, params, session) that returns a command's result."""
    def high(fd: int) -> int:
        # above 64, so putting the ends on fds 3 and 4 can never overwrite the other end
        moved = fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 64)
        os.close(fd)
        return moved
    to_chrome_r, to_chrome_w = map(high, os.pipe())
    from_chrome_r, from_chrome_w = map(high, os.pipe())
    # a small launcher puts the pipe ends on the child's fds 3 and 4 and runs Chrome in its place (no preexec_fn:
    # the checks start several browsers from threads at once; and no shell: dash, Ubuntu's sh, takes one-digit fds
    # only). Chrome leads its own process group, so closing it ends every helper it started, even when it never
    # answered.
    place = "import os, sys; os.dup2(int(sys.argv[1]), 3); os.dup2(int(sys.argv[2]), 4); os.execv(sys.argv[3], sys.argv[3:])"
    proc = subprocess.Popen([sys.executable, "-c", place, str(to_chrome_r), str(from_chrome_w), browser, *args,
                             "--remote-debugging-pipe", "--use-mock-keychain", "--password-store=basic"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, pass_fds=(to_chrome_r, from_chrome_w),
                            start_new_session=True)
    os.close(to_chrome_r)
    os.close(from_chrome_w)
    out, buf, ids = os.fdopen(to_chrome_w, "wb", buffering=0), b"", [0]

    def send(method: str, params: dict | None = None, session: str | None = None, deadline: float = 0.0) -> dict:
        nonlocal buf
        ids[0] += 1
        msg = {"id": ids[0], "method": method, "params": params or {}}
        if session:
            msg["sessionId"] = session
        out.write(json.dumps(msg).encode() + b"\0")
        while True:
            while b"\0" not in buf:
                if deadline and time.monotonic() > deadline:
                    raise TimeoutError(method)
                ready, _, _ = select.select([from_chrome_r], [], [], 1.0)
                if ready:
                    chunk = os.read(from_chrome_r, 1 << 16)
                    if not chunk:
                        raise RuntimeError(f"Chrome closed its DevTools pipe: {proc.stderr.read()[-2000:]!r}")
                    buf += chunk
            raw, buf = buf.split(b"\0", 1)
            reply = json.loads(raw)
            if reply.get("id") == ids[0]:
                if "error" in reply:
                    raise RuntimeError(f"{method}: {reply['error']}")
                return reply.get("result") or {}

    def close():
        try:
            send("Browser.close", deadline=time.monotonic() + 5)
        except Exception:  # noqa: BLE001 - a browser that will not close is killed below
            pass
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        out.close()
        os.close(from_chrome_r)
        proc.stderr.close()

    return send, close


def layout_check(browser: str, base: str, page: str, width: int, height: int = HEIGHT, timeout: float = 90.0) -> dict:
    """The page's own layout check (?check=layout) of PAGE (a path with its query) in a frame WIDTH px wide, read from
    the frame page once the check has finished. Chrome runs on real time and is asked over DevTools until the result
    is there: a run on virtual time (--virtual-time-budget with --dump-dom) holds its clock while a fetch is pending,
    and with the clips' videos loading it sometimes never reached its budget, which hung the test."""
    src = page + ("&" if "?" in page else "?") + "check=layout"
    frame = f"{base}/layout-frame?" + urllib.parse.urlencode({"src": src, "w": width, "h": height})
    with tempfile.TemporaryDirectory() as profile:
        send, close = _devtools(browser, ["--headless=new", "--disable-gpu", "--no-sandbox", "--no-first-run",
                                          "--hide-scrollbars", "--disable-extensions", f"--user-data-dir={profile}",
                                          f"--window-size={max(width, 500) + 40},{height + 200}", "about:blank"])
        try:
            # a browser starting on a busy machine gets its own minute before the check's time starts
            target = send("Target.createTarget", {"url": frame}, deadline=time.monotonic() + 60)["targetId"]
            deadline = time.monotonic() + timeout
            session = send("Target.attachToTarget", {"targetId": target, "flatten": True},
                           deadline=deadline)["sessionId"]
            read = ("(() => { const p = document.getElementById('layout-result');"
                    " return p && p.dataset.state === 'done' ? p.textContent : null; })()")
            while True:
                value = send("Runtime.evaluate", {"expression": read, "returnByValue": True}, session,
                             deadline=deadline).get("result", {}).get("value")
                if value:
                    return json.loads(value)
                assert time.monotonic() < deadline, f"no layout check at {width}px of {page} in {timeout:.0f}s"
                time.sleep(0.2)
        finally:
            close()


def check_all(browser: str, base: str, files: list[str]) -> list[dict]:
    """The layout check of every file at every width, as the page opens and with every fold opened, four at a time."""
    jobs = [(f, w, o) for f in files for w in WIDTHS for o in (False, True)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda j: layout_check(
            browser, base, f"/?ep={urllib.parse.quote(j[0])}" + ("&open=all" if j[2] else ""), j[1]), jobs))
    for (f, w, o), res in zip(jobs, results):
        res["asked"] = {"file": f, "width": w, "open": o}
    return results


@pytest.mark.skipif(chrome() is None, reason=f"no Google Chrome or Chromium on this {platform.system()} machine")
@pytest.mark.skipif(clips.tools.ffmpeg_or_none() is None, reason="no ffmpeg to make the clips")
def test_no_text_runs_out_of_its_box_at_any_width(tmp_path, monkeypatch):
    board, files = oversized_board(tmp_path, tmp_path / "clips")
    monkeypatch.setattr(serve, "Handler", FramedHandler)
    httpd, url = _serve(monkeypatch, board, tmp_path / "clips")
    try:
        results = check_all(chrome(), url, files)
    finally:
        httpd.shutdown()
        httpd.server_close()
    for res in results:
        # the check ran on the episode asked for, at the width asked for, with the folds as asked
        assert (res["episode"], res["width"], res["open"]) == tuple(res["asked"].values()), res
    bad = [f"{r['width']}px {r['episode']}{' (every fold open)' if r['open'] else ''}: {d['kind']} {d['el']} by "
           f"{d['over']}px ({d['text']!r})"
           for r in results for d in r["defects"]]
    assert not bad, "\n".join(bad)
