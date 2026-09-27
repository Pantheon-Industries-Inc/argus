"""The labelling harness: exact frames, sampling guarantees, prompt facts, the pinned instructions, keys, batches,
one episode end to end, the dry run, run folders (label/run.py) and re-reading stored replies (label/reparse.py).

The frame tests build a real packed mp4 with two back-to-back episodes whose every frame encodes its own global
frame number, on the 1/15360 time base MolmoAct2 uses, so a one-frame-late decode of an episode's last frame (which
would return the next episode's first frame) is caught by value. git and the harness subprocess are faked in the
run folder tests; nothing calls a model.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from board import to_board
from checks.timebase import JOINTS
from label import episode as me
from label import frames as mf
from label import harness
from label import prompts
from label import reparse as lreparse
from label import run as lrun
from label import state as ms

av = pytest.importorskip("av")
GRIPS = [6, 13]


def _packed_mp4(path, n_frames: int) -> None:
    """mp4 with frame k a flat grey of value (k*7) % 256, pts = k*512 on a 1/15360 time base."""
    from fractions import Fraction
    c = av.open(str(path), "w")
    s = c.add_stream("mpeg4", rate=30)
    s.width, s.height, s.pix_fmt = 64, 36, "yuv420p"
    s.time_base = Fraction(1, 15360)
    s.codec_context.gop_size = 2
    s.codec_context.qmin = s.codec_context.qmax = 1
    for k in range(n_frames):
        fr = av.VideoFrame.from_ndarray(np.full((36, 64, 3), (k * 7) % 256, np.uint8), format="rgb24")
        fr.pts = k * 512
        fr.time_base = s.time_base
        for pkt in s.encode(fr):
            c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()


def _val(im) -> int:
    return int(round(float(np.asarray(im.convert("L")).mean())))


def _skip_unless_grid(p):
    with av.open(str(p)) as c:
        tb = c.streams.video[0].time_base
    try:
        if mf.frame_pts_step(tb) != 512:
            pytest.skip("muxer did not keep the 1/15360 time base")
    except mf.FrameError:
        pytest.skip("muxer did not keep the 1/15360 time base")


# ---------------------------------------------------------------- frames

def test_last_frame_is_the_episodes_own(tmp_path):
    p = tmp_path / "pack.mp4"
    _packed_mp4(p, 60)                     # episode A = frames 0..40, episode B = 41..59
    _skip_unless_grid(p)
    base_a, n_a = 0.0, 41
    base_b, n_b = 41 / 30, 19
    a = mf.extract_frames(p, base_a, n_a, [0, 20, n_a - 1])
    b = mf.extract_frames(p, base_b, n_b, [0, n_b - 1])
    assert abs(_val(a[n_a - 1]) - (40 * 7) % 256) <= 2          # A's last frame, not B's first
    assert abs(_val(b[0]) - (41 * 7) % 256) <= 2
    assert abs(_val(a[20]) - (20 * 7) % 256) <= 2
    with pytest.raises(mf.FrameError):
        mf.extract_frames(p, base_a, n_a, [n_a])                 # one past the end is refused
    with pytest.raises(mf.FrameError):
        mf.extract_frames(p, 0.01, n_a, [0])                     # an off-grid offset is refused


# ---------------------------------------------------------------- sampling

def _state(T, still=()):
    """Moving arms except inside the given (a, b) frame spans, which are perfectly still."""
    t = np.arange(T) / 30.0
    s = np.zeros((T, 14))
    for j in JOINTS:
        s[:, j] = 0.3 * np.sin(t * (1 + j / 10))
    s[:, 6] = s[:, 13] = 0.5 + 0.4 * np.sin(t)
    for a, b in still:
        s[a:b + 1] = s[a]
    return s


def test_no_still_spans_when_always_moving():
    s = _state(900)
    assert ms.still_spans(s) == []
    ks = ms.sample_frames(900, [])
    assert ks[0] == 0 and ks[-1] == 899
    assert max(np.diff(ks)) <= 30


def test_still_span_found_and_sampling_guarantees():
    s = _state(1800, still=[(300, 899)])
    spans = ms.still_spans(s)
    assert len(spans) == 1
    a, b = spans[0]
    assert a <= 301 and b >= 898
    ks = ms.sample_frames(1800, spans)
    assert 0 in ks and 1799 in ks
    assert b + 1 in ks                                  # the instant motion resumes is sent
    for x, y in zip(ks, ks[1:]):
        inside = a <= x and y <= b + 1
        assert y - x <= (150 if inside else 30), (x, y)


def test_dither_within_tolerance_is_still_but_real_motion_is_not():
    s = _state(600, still=[(0, 599)])
    tick = np.radians(0.022)
    rng = np.random.default_rng(0)
    s[:, JOINTS] += rng.integers(-2, 3, size=(600, 12)) * tick     # encoder dither
    s[:, GRIPS] += rng.uniform(-0.003, 0.003, size=(600, 2))
    assert ms.still_spans(s) == [(0, 599)]
    s[300:, 0] += np.radians(1.0)                                   # a 1 degree move
    assert all(not (a < 300 <= b) for a, b in ms.still_spans(s))


def test_short_pause_is_not_a_span():
    assert ms.still_spans(_state(900, still=[(300, 359)])) == []   # a 2 s pause


# ---------------------------------------------------------------- prompt

def _ep():
    return {"context": {"dataset": "allenai/MolmoAct2-BimanualYAM-Dataset", "robot_type": "bi_yam_follower", "fps": 30,
                        "task_label": ["Spell out Ai2"], "instruction": "Spell AI2.", "profile": "teleop_arms",
                        "state_kind": "joints", "cameras": {"exo": {"width": 640, "height": 360}}},
            "state": np.zeros((900, 14)), "sources": {"exo": {}, "left": {}, "right": {}}}


def test_prompt_states_only_what_we_know():
    pl = {"n": 900, "spans": [(300, 449)], "ks": ms.sample_frames(900, [(300, 449)])}
    p = "".join(me.build_prompt(_ep(), pl, cell_w=448, cell_h=252))
    for bad in ("EXO_OVERHEAD", "LEFT_WRIST", "MOTOR FACTS", "straight down", "teleop_artifact", "mcap"):
        assert bad not in p, bad
    assert "10.00-14.97s" in p                          # the exact still span from the state
    assert "RECORDED MOTION" in p and "This is the recording's claim, not a fact" in p
    assert '"Spell AI2."' in p and "Spell out Ai2" in p
    assert "success_then_undone" in p and "goal_reached_at_s" in p and "COST-LEAN" in p
    assert '"data_issues"' in p and '"operator_mistakes"' in p and "TWO KINDS OF PROBLEM" in p


def test_prompt_without_spans_says_so():
    pl = {"n": 900, "spans": [], "ks": ms.sample_frames(900, [])}
    p = "".join(me.build_prompt(_ep(), pl, cell_w=448, cell_h=252))
    assert "RECORDED STILL SPANS, from the dataset's joint encoders: none." in p


def test_collection_note_reaches_the_episode_block_only():
    pl = {"n": 900, "spans": [], "ks": ms.sample_frames(900, [])}
    ep = _ep()
    fixed, episode = me.build_prompt(ep, pl, cell_w=448, cell_h=252)
    assert "How the dataset cuts its recordings" not in fixed + episode
    ep["context"]["collection_note"] = "consecutive 3-minute clips of a shift."
    fixed2, episode2 = me.build_prompt(ep, pl, cell_w=448, cell_h=252)
    assert fixed2 == fixed                              # the cached instructions stay byte-identical
    assert "How the dataset cuts its recordings into episodes: consecutive 3-minute clips" in episode2


def test_no_rig_borrows_another_rigs_hardware():
    ego = prompts.fixed_instructions("ego_head")
    assert "gripper" not in ego.lower() and "teleoperat" not in ego.lower() and "robot arm" not in ego.lower()
    hand = prompts.fixed_instructions("handheld_gripper")
    assert "what the arm does" not in hand


# The shared instructions each rig is sent, pinned so that no prompt text changes by accident. A deliberate prompt
# change updates these in the same commit.
PINNED = {
    ("teleop_arms", True): "21c60b86a51458d0d9f87269b22c01222589f377f6f2cef4ad974eaabf2d3f0d",
    ("teleop_arms", False): "a7b32468abf9070fe2f604490f4814d5a9f3920b0d4518959319538d6ef0a480",
    ("handheld_gripper", True): "9f466deb5aba87ebc0f13d28db28a8b0ac040b5f9727e54cadd6740374ccf7f9",
    ("handheld_gripper", False): "a12c45e99e44203c882ffe779d67ca91b4d63831b1ca35202e111b3f60d6193c",
    ("ego_head", True): "b77250b58b3ef34dab5faf1941f26400a50b45d37b2bb5701f905294e5c58ff7",
    ("ego_head", False): "b77250b58b3ef34dab5faf1941f26400a50b45d37b2bb5701f905294e5c58ff7",
}


@pytest.mark.parametrize("rig,has_instruction", sorted(PINNED))
def test_prompts_are_pinned(rig, has_instruction):
    text = prompts.fixed_instructions(rig, has_instruction=has_instruction)
    assert hashlib.sha256(text.encode()).hexdigest() == PINNED[(rig, has_instruction)]


# ---------------------------------------------------------------- keys and batches

def test_keypool_round_robin_and_retire():
    pool = harness.KeyPool(["sk-or-a", "sk-or-b"])
    assert {pool.take() for _ in range(4)} == {"sk-or-a", "sk-or-b"}
    pool.retire("sk-or-a", "402")
    assert {pool.take() for _ in range(4)} == {"sk-or-b"}
    pool.retire("sk-or-b", "402")
    assert pool.take() is None


def test_only_openrouter_keys_are_used(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEYS", "sk-or-a, sk-proj-b ,sk-or-c")
    assert harness.get_keys() == ["sk-or-a", "sk-or-c"]


def test_key_exhaustion_classification():
    assert harness.is_key_exhausted(402, '{"error":{"message":"Insufficient credits"}}')
    assert harness.is_key_exhausted(429, '{"error":{"code":"insufficient_quota"}}')
    assert not harness.is_key_exhausted(429, '{"error":{"message":"Rate limit reached"}}')
    assert not harness.is_key_exhausted(500, "server error")


def _dirs(tmp_path, n):
    eps = []
    for i in range(n):
        d = tmp_path / f"episode_{i:06d}"
        d.mkdir()
        eps.append(d)
    return eps


def test_batch_finishes_when_a_key_runs_out(tmp_path, monkeypatch):
    calls = {"sk-or-a": 0, "sk-or-b": 0}

    def fake_label(ep, out, *, api_key, **kw):
        calls[api_key] += 1
        if api_key == "sk-or-a" and calls[api_key] > 2:
            raise harness.KeyExhausted("HTTP 402: Insufficient credits")
        out.write_text(json.dumps({"parse_ok": True, "usage": {"est_cost_usd": 0.5}}))
        return {"usage": {"est_cost_usd": 0.5}}

    monkeypatch.setattr(harness, "label_episode", fake_label)
    rc = harness.run_batch(_dirs(tmp_path, 12), tmp_path / "out", keys=["sk-or-a", "sk-or-b"], concurrency=3,
                           force=False)
    assert rc == 0 and len(list((tmp_path / "out").glob("*.json"))) == 12


def test_spend_cap_stops_new_episodes(tmp_path, monkeypatch):
    def fake_label(ep, out, *, api_key, **kw):
        out.write_text(json.dumps({"parse_ok": True, "usage": {"est_cost_usd": 1.0}}))
        return {"usage": {"est_cost_usd": 1.0}}

    monkeypatch.setattr(harness, "label_episode", fake_label)
    harness.run_batch(_dirs(tmp_path, 10), tmp_path / "out", keys=["sk-or-a"], concurrency=1, force=False,
                      max_spend=3.0)
    assert len(list((tmp_path / "out").glob("*.json"))) == 3


# ---------------------------------------------------------------- one episode, end to end, offline

def _packed_episode(tmp_path):
    """A prepared teleop episode, frames 50..169 of three packed mp4s, with recorded joint state."""
    packs = {}
    for v in ("exo", "left", "right"):
        packs[v] = tmp_path / f"{v}.mp4"
        _packed_mp4(packs[v], 200)                     # previous episode 0..49, ours 50..169, next 170..
    _skip_unless_grid(packs["exo"])
    ep = tmp_path / "episode_000007"
    ep.mkdir()
    T = 120
    json.dump({v: {"packed": str(packs[v]), "base_s": 50 / 30, "n_frames": T} for v in packs},
              open(ep / "sources.json", "w"))
    np.savez(ep / "state.npz", state=_state(T), action=_state(T))
    json.dump({"dataset": "allenai/MolmoAct2-BimanualYAM-Dataset", "robot_type": "bi_yam_follower", "fps": 30,
               "task_label": ["Spell out Ai2"], "instruction": "Spell AI2.", "n_state_frames": T,
               "profile": "teleop_arms", "state_kind": "joints", "cameras": {"exo": {"width": 64, "height": 36}}},
              open(ep / "context.json", "w"))
    return ep, T


SERVED = {"provider": "SomeHost", "id": "gen-123", "model": "openai/gpt-6-astra-20260901", "system_fingerprint": "fp_1"}


def test_episode_end_to_end_offline(tmp_path, monkeypatch):
    """A synthetic packed episode through label_episode (model call mocked) and into a board file."""
    ep, T = _packed_episode(tmp_path)
    sent = {}

    def fake_call(content, model, reasoning, api_key, max_tokens, timeout):
        sent["content"] = content
        labels = {"completion": {"task_completed": "success", "completed_at_s": 3.0, "goal_reached_at_s": 3.0},
                  "timeline_columns": harness.TIMELINE_COLUMNS,
                  "timeline": [[0.0, 3.0, "left", "reach", "block", None, None, "advancing", 0.5, None]],
                  "data_issues": []}
        return {"choices": [{"message": {"content": json.dumps(labels)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "cost": 0.01}, **SERVED}

    monkeypatch.setattr(harness, "call_model", fake_call)
    out = tmp_path / "out" / "episode_000007.json"
    r = harness.label_episode(ep, out, model="openai/gpt-6-astra", reasoning="medium", api_key="sk-or-x",
                              max_tokens=64000, timeout=60)
    assert r["parse_ok"] and r["usage"]["est_cost_usd"] == 0.01 and r["usage"]["cost_source"] == "billed"
    assert (r["provider_name"], r["generation_id"], r["model_served"], r["system_fingerprint"]) == \
        ("SomeHost", "gen-123", "openai/gpt-6-astra-20260901", "fp_1")
    assert json.loads(out.read_text())["model_served"] == "openai/gpt-6-astra-20260901"
    assert r["config"]["timesteps_s"][0] == 0.0 and r["config"]["timesteps_s"][-1] == round(119 / 30, 3)
    assert r["config"]["cell"][0] == me.GRID_CELL_W_BY_RIG["teleop_arms"]
    imgs = [c for c in sent["content"] if c["type"] == "image_url"]
    assert len(imgs) == r["config"]["n_image_parts"] == -(-len(r["config"]["timesteps_s"]) // 4) + 2
    # the detail view of the LAST instant is episode frame 119 (global 169), never the next episode's 170
    from PIL import Image
    last = Image.open(io.BytesIO(base64.b64decode(imgs[-1]["image_url"]["url"].split(",", 1)[1])))
    strip = np.asarray(last.convert("L"))[26:26 + 36, :]           # the top camera's row
    assert abs(float(strip.mean()) - (169 * 7) % 256) <= 3
    # a request over the image cap is sent at a smaller cell width instead of being refused
    monkeypatch.setattr(me, "IMAGE_LIMIT_BYTES", (1 + me.DETAIL_VIEW_BYTES_MAX) * me.IMAGE_SIZE_INFLATION)
    small = me.build_request(ep)
    assert small["cell"][0] == min(me.CELL_W_STEPS) < r["config"]["cell"][0]
    board = to_board.convert(json.loads(out.read_text()), "molmo")
    assert board["dataset_checks"]["camera_windows_match_state"] and board["timesteps_s"]
    assert board["event_labels"][0]["verb_class"] == "reach" and board["completion"]["task_completed"] == "success"


def test_request_body_is_what_the_harness_docstring_says(monkeypatch):
    """The body sent: model, reasoning effort, output limit, JSON mode, a cache breakpoint closing the shared
    instructions and the billed usage; no temperature, top_p, seed or provider pin."""
    seen = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        seen["body"], seen["headers"] = json.loads(req.data), dict(req.header_items())
        return Resp(json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode())

    monkeypatch.setattr(harness.urllib.request, "urlopen", fake_urlopen)
    content = [{"type": "text", "text": "x" * 5000}, {"type": "text", "text": "episode"}]
    harness.call_model(content, "openai/gpt-6-astra", "medium", "sk-or-x", 64000, 60)
    b = seen["body"]
    assert set(b) == {"model", "messages", "max_completion_tokens", "reasoning", "response_format", "usage"}
    assert (b["model"], b["max_completion_tokens"]) == ("openai/gpt-6-astra", 64000)
    assert b["reasoning"] == {"effort": "medium"}
    assert b["response_format"] == {"type": "json_object"} and b["usage"] == {"include": True}
    sent = b["messages"][0]["content"]
    assert sent[0]["cache_control"] == {"type": "ephemeral"} and "cache_control" not in sent[1]
    assert "cache_control" not in content[0]                                 # the caller's parts are not changed
    assert seen["headers"]["Authorization"] == "Bearer sk-or-x"


def test_cut_off_reply_is_kept_beside_the_outputs_and_fails(tmp_path, monkeypatch):
    ep, _ = _packed_episode(tmp_path)

    def fake_call(content, model, reasoning, api_key, max_tokens, timeout):
        return {"choices": [{"message": {"content": '{"timeline": [[0.0, 1'}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 64000, "cost": 0.5}, **SERVED}

    monkeypatch.setattr(harness, "call_model", fake_call)
    out = tmp_path / "out" / "episode_000007.json"
    with pytest.raises(RuntimeError, match="truncated"):
        harness.label_episode(ep, out, model="m", reasoning="medium", api_key="sk-or-x", max_tokens=64000, timeout=60)
    assert not out.exists()                                  # a resumed run still counts the episode as not done
    failed = json.loads((out.parent / "failed_episode_000007.json").read_text())
    assert failed["finish_reason"] == "length" and failed["content_tail"] == '{"timeline": [[0.0, 1'
    assert failed["model_served"] == SERVED["model"] and failed["usage"]["cost"] == 0.5


def test_dry_run_from_the_command_line(tmp_path, monkeypatch, capsys):
    """python -m label.harness --episodes-root --out-dir --dry-run: every request built and recorded, no key."""
    _packed_episode(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEYS", raising=False)
    monkeypatch.setattr(harness, "call_model", lambda *a, **k: pytest.fail("a dry run called the model"))
    monkeypatch.setattr(sys, "argv", ["harness", "--episodes-root", str(tmp_path), "--out-dir", str(tmp_path / "dry"),
                                      "--dry-run"])
    assert harness.main() == 0
    r = json.loads((tmp_path / "dry" / "episode_000007.json").read_text())
    assert r["dry_run"] and r["model"] == harness.DEFAULT_MODEL and r["reasoning_effort"] == "medium"
    assert r["request"]["n_images"] == r["config"]["n_image_parts"] and r["request"]["est_input_tokens"] > 0
    assert r["prompt_text"].startswith(prompts.FIXED_HEADER) and "THE EPISODE TO LABEL." in r["prompt_text"]
    assert "done=1 failed=0" in capsys.readouterr().out


def test_timeline_rows_convert_and_malformed_rows_fail():
    rows = {"timeline_columns": harness.TIMELINE_COLUMNS,
            "timeline": [[0.0, 1.5, "left", "reach", "cup", None, None, "advancing", 0.1, None],
                         [1.5, 2.0, "right", "grasp", "cup", None, "in gripper", "advancing", 0.2]]}
    out = harness.normalize_timeline(rows)
    assert "timeline_columns" not in out
    assert out["timeline"][1] == {"start_s": 1.5, "end_s": 2.0, "arm": "right", "action": "grasp", "object": "cup",
                                  "destination": None, "spatial_relation": "in gripper",
                                  "contribution": "advancing", "progress": 0.2}
    null = harness.normalize_timeline({"timeline": [[0.0, 1.0, "left", "reach", "cup", None, None, "idle", None,
                                                      None]]})
    assert null["_schema_violations"] == ["timeline row 0: progress null"]          # counted, the answer kept
    labels, ok = harness.parse_response("```json\n" + json.dumps(rows) + "\n```")
    assert ok and labels["timeline"][0]["action"] == "reach"                           # a fenced reply is read
    labels, ok = harness.parse_response("not json")
    assert not ok and labels["_raw"] == "not json" and labels["_parse_error"]
    bad = harness.normalize_timeline({"timeline": [[0.0, 1.0, "left", "reach", "cup", None, None, "fast", 0.1, None]]})
    assert bad["_schema_violations"] == ["timeline row 0: contribution 'fast'"]     # counted, the answer kept
    with pytest.raises(ValueError):
        harness.normalize_timeline({"timeline": [[0.0, 1.0, "left"]]})             # a shifted row is refused
    with pytest.raises(ValueError):
        harness.normalize_timeline({"timeline": [["0", 1.0, "left", "reach", "cup", None, None, "idle", 0.1, None]]})


# ---------------------------------------------------------------- run folders (label/run.py), git and harness faked


def _slice(tmp_path, n_frames=3600):
    sl = tmp_path / "slice"
    (sl / "episode_a").mkdir(parents=True)
    (sl / "episode_a" / "context.json").write_text(json.dumps({"fps": 30, "n_state_frames": n_frames}))
    return sl


def _fake_harness(monkeypatch, calls, cost=0.5, rc=0):
    """label.run's subprocess.run: records the command and env, writes one output and the harness's last line."""
    def fake_run(cmd, cwd, env, stdout, stderr):
        calls.append({"cmd": cmd, "env": env, "cwd": cwd})
        out = Path(cwd) / "out"
        out.mkdir(exist_ok=True)
        (out / "episode_a.json").write_text(json.dumps({"parse_ok": True, "usage": {"est_cost_usd": cost}}))
        stdout.write(f"done=1 failed=0 skipped=0 total_cost=${cost:.2f} out_dir={out} retired_keys=0\n")
        return subprocess.CompletedProcess(cmd, rc)
    monkeypatch.setattr(lrun.subprocess, "run", fake_run)


def _argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["python -m label", *map(str, args)])


def test_paid_run_refuses_a_dirty_checkout_and_needs_a_cap(tmp_path, monkeypatch):
    sl, runs = _slice(tmp_path), tmp_path / "runs"
    monkeypatch.setattr(lrun, "commit", lambda: ("abc1234", True))
    _argv(monkeypatch, "--dataset", "d", "--episodes", sl, "--kind", "smoke", "--cap", "1", "--runs", runs)
    with pytest.raises(SystemExit, match="uncommitted changes"):
        lrun.main()
    monkeypatch.setattr(lrun, "commit", lambda: ("abc1234", False))
    _argv(monkeypatch, "--dataset", "d", "--episodes", sl, "--kind", "full", "--runs", runs)
    with pytest.raises(SystemExit, match="needs --cap"):
        lrun.main()
    assert not runs.exists()                                                     # nothing was started


def test_run_folder_name_and_record(tmp_path, monkeypatch, capsys):
    sl, runs, calls = _slice(tmp_path), tmp_path / "runs", []
    monkeypatch.setattr(lrun, "commit", lambda: ("abc1234", False))
    monkeypatch.setenv("OPENROUTER_API_KEYS", "sk-or-a,sk-proj-b")
    _fake_harness(monkeypatch, calls)
    _argv(monkeypatch, "--dataset", "d", "--episodes", sl, "--kind", "smoke", "--cap", "2", "--runs", runs,
          "--label", "opus55", "--note", "a note", "--", "--model", "m/x")
    lrun.main()
    (run,) = (runs / "d").iterdir()
    assert re.fullmatch(r"\d{8}-\d{4}_smoke_abc1234_opus55", run.name)
    info = json.loads((run / "run.json").read_text())
    assert (info["run_id"], info["code"], info["code_dirty"], info["note"]) == (run.name, "abc1234", False, "a note")
    assert (info["status"], info["cost_usd"], info["episodes_done"], info["episodes_failed"]) == ("done", 0.5, 1, 0)
    assert info["footage_hours"] == round(120 / 3600, 3) and info["usd_per_hour"] == 15.0      # $0.50 over 2 minutes
    cmd = calls[0]["cmd"]
    assert cmd[1:3] == ["-m", "label.harness"] and cmd[cmd.index("--max-spend") + 1] == "2.00"
    assert cmd[-2:] == ["--model", "m/x"] and "--" not in cmd and "--dry-run" not in cmd
    assert calls[0]["env"]["OPENROUTER_API_KEYS"] == "sk-or-a"                   # only OpenRouter keys are passed
    assert "done=1" in (run / "log.txt").read_text() and '"status": "done"' in capsys.readouterr().out


def test_dry_run_needs_no_cap_no_key_and_no_clean_checkout(tmp_path, monkeypatch):
    sl, runs, calls = _slice(tmp_path), tmp_path / "runs", []
    monkeypatch.setattr(lrun, "commit", lambda: ("abc1234", True))
    monkeypatch.delenv("OPENROUTER_API_KEYS", raising=False)
    _fake_harness(monkeypatch, calls, cost=0.0)
    _argv(monkeypatch, "--dataset", "d", "--episodes", sl, "--kind", "dry", "--runs", runs)
    lrun.main()
    (run,) = (runs / "d").iterdir()
    assert re.fullmatch(r"\d{8}-\d{4}_dry_abc1234", run.name)
    assert "--dry-run" in calls[0]["cmd"] and "--max-spend" not in calls[0]["cmd"]
    assert json.loads((run / "run.json").read_text())["code_dirty"] is True


def _killed_run(tmp_path, spent=0.4, kind="smoke"):
    sl = _slice(tmp_path)
    run = tmp_path / "runs" / "d" / f"20260927-1200_{kind}_abc1234"
    (run / "out").mkdir(parents=True)
    (run / "out" / "episode_a.json").write_text(json.dumps({"parse_ok": True, "usage": {"est_cost_usd": spent}}))
    (run / "run.json").write_text(json.dumps({"run_id": run.name, "kind": kind, "dataset": "d", "slice": str(sl),
                                              "code": "abc1234", "cap_usd": 1.0, "status": "running"}))
    (run / "log.txt").write_text("before the kill\n")
    return sl, run


@pytest.mark.parametrize("change,why", [
    ({"--dataset": "other"}, "dataset is 'd'"),
    ({"--kind": "full"}, "kind is 'smoke'"),
    ({"commit": ("fff0000", False)}, "code is 'abc1234'"),
    ({"commit": ("abc1234", True)}, "uncommitted changes"),
    ({"--why": ""}, "needs --why"),
    ({"slice": "elsewhere"}, "the run's slice is"),
    ({"spent": 1.0}, "already spent"),
])
def test_resume_is_refused_unless_it_is_the_same_run(tmp_path, monkeypatch, change, why):
    sl, run = _killed_run(tmp_path, spent=change.get("spent", 0.4))
    if "slice" in change:
        sl = tmp_path / change["slice"]
        sl.mkdir()
    monkeypatch.setattr(lrun, "commit", lambda: change.get("commit", ("abc1234", False)))
    monkeypatch.setattr(lrun.subprocess, "run", lambda *a, **k: pytest.fail("the harness was started"))
    monkeypatch.setenv("OPENROUTER_API_KEYS", "sk-or-a")
    _argv(monkeypatch, "--dataset", change.get("--dataset", "d"), "--episodes", sl, "--kind",
          change.get("--kind", "smoke"), "--resume", run, "--why", change.get("--why", "the host rebooted"))
    with pytest.raises(SystemExit, match=why):
        lrun.main()


def test_resume_refuses_a_folder_without_outputs(tmp_path, monkeypatch):
    sl, run = _killed_run(tmp_path)
    (run / "out" / "episode_a.json").unlink()
    monkeypatch.setattr(lrun, "commit", lambda: ("abc1234", False))
    _argv(monkeypatch, "--dataset", "d", "--episodes", sl, "--kind", "smoke", "--resume", run, "--why", "x")
    with pytest.raises(SystemExit, match="no outputs"):
        lrun.main()


def test_resume_finishes_the_run_within_what_is_left_of_its_cap(tmp_path, monkeypatch):
    sl, run = _killed_run(tmp_path, spent=0.4)
    calls = []
    monkeypatch.setattr(lrun, "commit", lambda: ("abc1234", False))
    monkeypatch.setenv("OPENROUTER_API_KEYS", "sk-or-a")
    _fake_harness(monkeypatch, calls, cost=0.4)
    _argv(monkeypatch, "--dataset", "d", "--episodes", sl, "--kind", "smoke", "--resume", run, "--why", "rebooted")
    lrun.main()
    cmd = calls[0]["cmd"]
    assert cmd[cmd.index("--max-spend") + 1] == "0.60" and Path(calls[0]["cwd"]) == run.resolve()
    info = json.loads((run / "run.json").read_text())
    assert info["resumes"][0]["why"] == "rebooted" and info["resumes"][0]["spent_before"] == 0.4
    assert info["status"] == "done" and info["run_id"] == run.name
    assert (run / "log.txt").read_text().startswith("before the kill\n")      # the log is appended to


# ---------------------------------------------------------------- re-reading stored replies (label/reparse.py)

def test_reparse_rereads_only_unparsed_replies(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    (run / "out").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": "r"}))
    now_parses = json.dumps({"timeline_columns": harness.TIMELINE_COLUMNS,
                             "timeline": [[0.0, 1.0, "left", "reach", "cup", None, None, "advancing", 0.1, None]]})
    records = {"episode_a": {"parse_ok": False, "labels": {"_raw": now_parses, "_parse_error": "old parser"}},
               "episode_b": {"parse_ok": False, "labels": {"_raw": "not json", "_parse_error": "JSONDecodeError"}},
               "episode_c": {"parse_ok": True, "labels": {"timeline": []}}}
    for name, r in records.items():
        (run / "out" / f"{name}.json").write_text(json.dumps(r))
    monkeypatch.setattr(lreparse, "commit", lambda: ("abc1234", False))
    monkeypatch.setattr(sys, "argv", ["python -m label.reparse", str(run)])
    assert lreparse.main() == 0
    assert json.loads(capsys.readouterr().out) == {"run": "run", "parsed_now": 1, "still_unparsed": 1}
    a = json.loads((run / "out" / "episode_a.json").read_text())
    assert a["parse_ok"] and a["labels"]["timeline"][0]["action"] == "reach"
    assert a["labels"]["_reparsed"] == {"was": "old parser", "code": "abc1234"}
    assert json.loads((run / "out" / "reparsed_originals" / "episode_a.json").read_text()) == records["episode_a"]
    assert json.loads((run / "out" / "episode_b.json").read_text()) == records["episode_b"]      # left as it is
    assert json.loads((run / "out" / "episode_c.json").read_text()) == records["episode_c"]
    assert sorted(p.name for p in (run / "out" / "reparsed_originals").iterdir()) == ["episode_a.json"]
    log = json.loads((run / "run.json").read_text())["reparsed"][0]
    assert log["code"] == "abc1234" and [x["episode"] for x in log["parsed_now"]] == ["episode_a"]
    assert [x["episode"] for x in log["still_unparsed"]] == ["episode_b"]
