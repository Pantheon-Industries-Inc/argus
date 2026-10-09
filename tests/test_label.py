"""The labelling harness: exact frames, sampling guarantees, prompt facts, the pinned instructions, keys, batches,
one episode end to end, the dry run, run folders (label/run.py) and re-reading stored replies (label/reparse.py).

The frame tests build a real packed mp4 with two back-to-back episodes whose every frame encodes its own global
frame number, on the 1/15360 time base MolmoAct2 uses, so a one-frame-late decode of an episode's last frame (which
would return the next episode's first frame) is caught by value. git and the harness subprocess are faked in the
run folder tests; nothing calls a model.
"""
from __future__ import annotations

import base64
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


def test_offsets_stored_to_the_microsecond_are_on_the_grid():
    # LeRobot v3 packed files store each episode's offset rounded to 6 decimals of a second
    for k in (6568, 6977, 8465, 12840, 18869):
        assert mf.base_frame(round(k / 30, 6)) == k
    assert mf.base_frame(282.16666599999996) == 8465
    with pytest.raises(mf.FrameError):
        mf.base_frame(0.01)                                       # a third of a frame off is still refused


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


def test_the_gripper_tolerance_is_a_share_of_its_own_range():
    """A gripper recorded 0 to 100 (Galaxea) or in metres (RealOmni, 0 to 0.10) is judged as one recorded 0 to 1: its
    resting wander stays still, and a real change of 5% of its travel is motion."""
    assert ms.still_tolerance("joints", 7)[6] == pytest.approx(0.01)                  # no range: a 0 to 1 opening
    assert ms.still_tolerance("joints", 14, 100.0)[[6, 13]] == pytest.approx([1.0, 1.0])
    assert ms.still_tolerance("ee_pose", 7, 0.10)[6] == pytest.approx(0.001)
    assert ms.gripper_full_range({"gripper_range": [0, 100]}) == 100.0
    assert ms.gripper_full_range({"gripper_range": [0.103, 0.0]}) == pytest.approx(0.103)
    assert ms.gripper_full_range({}) is None and ms.gripper_full_range({"gripper_range": "x"}) is None
    rng = np.random.default_rng(0)
    # Galaxea: a still pair of arms whose grippers wander by 0.2 of 100, as measured on the dataset's 99th percentile
    s = _state(600, still=[(0, 599)])
    s[:, GRIPS] = 40.0 + rng.uniform(-0.1, 0.1, size=(600, 2))
    assert ms.still_spans(s) == []                                   # the old fixed 0.01 split it
    assert ms.still_spans(s, grip_range=100.0) == [(0, 599)]
    s[300:, 6] += 5.0                                                # a real 5% close
    assert all(not (a < 300 <= b) for a, b in ms.still_spans(s, grip_range=100.0))
    # RealOmni: a still gripper that closes by 5 mm of its 0.10 m travel is not still
    p = np.zeros((600, 7))
    p[:, 6] = 0.06
    p[300:, 6] -= 0.005
    assert ms.still_spans(p, kind="ee_pose") == [(0, 599)]           # the old fixed 0.01 (1 cm) called it still
    assert all(not (a < 300 <= b) for a, b in ms.still_spans(p, kind="ee_pose", grip_range=0.10))


def test_an_upload_gets_its_gripper_range_measured_across_its_episodes(tmp_path):
    """A reader that does not know the gripper's unit gets the range from every episode of the upload together, so
    an episode whose gripper never moves is still judged against the gripper's full travel; a declared range stays."""
    from prepare import formats
    rng = np.random.default_rng(1)
    for i, (lo, hi) in enumerate([(0.0, 100.0), (40.0, 40.0), (5.0, 60.0)]):
        d = tmp_path / f"episode_{i}"
        d.mkdir()
        st = np.zeros((300, 14), np.float32)
        st[:, [6, 13]] = np.linspace(lo, hi, 300)[:, None] + rng.uniform(-0.05, 0.05, size=(300, 2))
        np.savez(d / "state.npz", state=st)
        (d / "context.json").write_text(json.dumps({"state_kind": "joints", **({"gripper_range": [0, 1]} if i == 2
                                                                                else {})}))
    got = formats.measure_gripper_range(tmp_path, ["episode_0", "episode_1", "episode_2", "episode_none"])
    assert got[0] == pytest.approx(0.0, abs=2) and got[1] == pytest.approx(100.0, abs=2)
    ctx = [json.loads((tmp_path / f"episode_{i}" / "context.json").read_text()) for i in range(3)]
    assert ctx[0]["gripper_range"] == ctx[1]["gripper_range"] == got            # the flat one gets the upload's range
    assert "2 episodes" in ctx[1]["gripper_range_note"]
    assert ctx[2]["gripper_range"] == [0, 1] and "gripper_range_note" not in ctx[2]
    assert formats.measure_gripper_range(tmp_path, []) is None


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
    assert "success_then_undone" in p and "goal_reached_at_s" in p and "TIMELINE GRANULARITY" in p and "Segment finely" not in p
    assert "confidence" not in p                         # the timeline has no such column
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


@pytest.mark.parametrize("rig", ["teleop_arms", "ego_head"])
def test_uploader_notes_reach_the_episode_block_as_claims(rig):
    pl = {"n": 900, "spans": [], "ks": ms.sample_frames(900, [])}
    ep = _ep()
    ep["context"]["profile"] = rig
    fixed, episode = me.build_prompt(ep, pl, cell_w=448, cell_h=252)
    assert "UPLOADER'S OWN NOTES" not in fixed + episode
    ep["context"]["uploader_annotation"] = '{"operator": "A", "note": "the second block slipped"}\n'
    fixed2, episode2 = me.build_prompt(ep, pl, cell_w=448, cell_h=252)
    assert fixed2 == fixed                              # the cached instructions stay byte-identical
    assert episode2.endswith("THE UPLOADER'S OWN NOTES FOR THIS EPISODE, as sent. They are claims to check against "
                             "the video, not ground truth; where the video contradicts them, record it as a data "
                             'issue:\n{"operator": "A", "note": "the second block slipped"}\n')


def test_no_rig_borrows_another_rigs_hardware():
    ego = prompts.fixed_instructions("ego_head")
    assert "gripper" not in ego.lower() and "teleoperat" not in ego.lower() and "robot arm" not in ego.lower()
    hand = prompts.fixed_instructions("handheld_gripper")
    assert "what the arm does" not in hand


@pytest.mark.parametrize("rig", ["teleop_arms", "handheld_gripper", "ego_head"])
@pytest.mark.parametrize("has_instruction", [False, True])
def test_recorded_and_video_only_instructions_keep_the_same_output_contract(rig, has_instruction):
    recorded = prompts.fixed_instructions(rig, has_instruction=has_instruction)
    video = prompts.fixed_instructions(rig, has_instruction=has_instruction, recorded=False)
    columns = ["start_s", "end_s", "arm", "action", "object", "destination", "spatial_relation",
               "contribution", "progress"]
    if rig == "ego_head":
        columns += ["hands_visible", "hands_wearing"]
    columns += ["notes"]
    for text in (recorded, video):
        offered = json.loads(re.search(r'"timeline_columns": (\[[^\]]+\])', text).group(1))
        assert offered == columns
        for field in ("scene", "timeline", "key_events", "data_issues", "operator_mistakes", "recovery"):
            assert f'"{field}":' in text
        assert ('"tasks":' in text) is (rig == "ego_head")
        assert ('"completion":' in text) is (rig != "ego_head")
    assert "recorded motion" in recorded
    assert "recorded motion" not in video and "what is recorded" not in video
    assert "state_video_mismatch" not in video
    if rig != "ego_head":
        assert "state_video_mismatch" in recorded


def test_an_episode_without_instruction_gets_the_task_rule_and_head_cameras_never_do():
    for r in ("teleop_arms", "handheld_gripper"):
        without, given = prompts.fixed_instructions(r, has_instruction=False), prompts.fixed_instructions(r)
        assert "ABOUT THE TASK. This episode comes with no instruction" in without
        assert "ABOUT THE EPISODE'S INSTRUCTION" not in without and "ABOUT THE TASK." not in given
    assert "ABOUT THE TASK." not in prompts.fixed_instructions("ego_head", has_instruction=False)


# ---------------------------------------------------------------- resolution routing and contact views

def _gripper_ep(profile="teleop_arms", T=900):
    """Two arms; the left gripper closes at frame 300 and opens at 600, the right one never moves."""
    s = np.zeros((T, 14))
    s[300:600, 6] = 1.0
    return {"context": {"profile": profile, "state_kind": "none" if profile == "ego_head" else "joints", "fps": 30},
            "state": s, "sources": {"exo": {}, "left": {}, "right": {}}, "times": None, "kmap": {}}


def _plan(ep):
    ks = ms.sample_frames(len(ep["state"]), [], fps=30, moving_every_s=1.5, still_every_s=1.5)
    return {"n": len(ep["state"]), "ks": ks, "state_usable": True}


def test_contact_views_follow_sharp_gripper_changes_on_teleop_only():
    ep = _gripper_ep()
    pl = _plan(ep)
    pl["contact"] = me.contact_instants(ep, pl)
    assert pl["contact"] == [315, 630]                  # the first instants after the close and the open
    assert me.contact_views(ep, pl) == [(315, ["exo", "left"]), (630, ["exo", "left"])]
    for other in ("handheld_gripper", "ego_head"):
        assert me.contact_instants(_gripper_ep(other), _plan(_gripper_ep(other))) == []
    flat = _gripper_ep()
    flat["state"][:, 6] = 0.4
    assert me.contact_instants(flat, _plan(flat)) == []
    assert me.contact_instants(ep, dict(pl, state_usable=False)) == []


def test_narrow_teleop_cells_carry_contact_views_and_wide_ones_do_not(tmp_path):
    ep, _ = _packed_episode(tmp_path)
    narrow, wide = me.build_request(ep, max_cell_w=224), me.build_request(ep, max_cell_w=448)
    assert narrow["cell"][0] == 224 and narrow["contact_s"] and wide["cell"][0] == 448 and wide["contact_s"] == []
    captions = [c["text"] for c in narrow["content"] if c["type"] == "text" and c["text"].startswith("=== detail")]
    assert len(captions) == 2 + len(narrow["contact_s"]) == narrow["n_images"] - narrow["n_grids"]
    assert "first frame" in captions[0] and "last frame" in captions[-1]
    assert all("sharp change of the recorded gripper value" in c for c in captions[1:-1])
    assert f"({narrow['contact_s'][0]:.2f} s)" in narrow["prompt"]
    assert "recorded value changes sharply" not in wide["prompt"]


def _route_call(answers):
    calls = []

    def call(content, model, reasoning, api_key, max_tokens, timeout):
        calls.append((content[0]["text"], model, reasoning))
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return {"choices": [{"message": {"content": a}}], "usage": {"cost": 0.0004}}
    return call, calls


def test_routing_reads_the_task_text_and_falls_back_to_wide_cells(tmp_path, monkeypatch):
    from label import route
    monkeypatch.setattr(route, "_CACHE", {})
    ep, _ = _packed_episode(tmp_path)
    call, calls = _route_call(['{"fine_detail": false, "why": "whole objects"}'])
    w, rec = route.route_width(ep, "sk-or-x", call)
    assert (w, rec["fine_detail"], rec["cost_usd"]) == (224, False, 0.0004)
    assert calls[0][1:] == (route.ROUTE_MODEL, "low") and "Instruction: Spell AI2." in calls[0][0]
    # the same task text again: the earlier answer, no call, nothing billed
    w, rec = route.route_width(ep, "sk-or-x", call)
    assert (w, rec["cached"], rec["cost_usd"], len(calls)) == (224, True, 0.0, 1)
    monkeypatch.setattr(route, "_CACHE", {})
    for answer in ('{"fine_detail": true, "why": "lettering"}', "no JSON here", RuntimeError("HTTP 500")):
        call, _ = _route_call([answer])
        assert route.route_width(ep, "sk-or-x", call)[0] == 448
        monkeypatch.setattr(route, "_CACHE", {})
    # a failed routing call is not cached: the next episode with the same task text asks again
    call, calls = _route_call([RuntimeError("HTTP 500"), '{"fine_detail": false, "why": "whole objects"}'])
    assert route.route_width(ep, "sk-or-x", call)[1]["why"].startswith("routing failed")
    assert route.route_width(ep, "sk-or-x", call)[0] == 224 and len(calls) == 2
    monkeypatch.setattr(route, "_CACHE", {})
    assert route.route_width(ep, None, call)[1]["why"] == "no key (dry run)"
    ctx = json.loads((ep / "context.json").read_text())
    ctx.pop("instruction")
    # a plain video's task label is its file name, which is no task text
    (ep / "context.json").write_text(json.dumps({**ctx, "task_label": ["take_03"],
                                                 "source": {"format": "video files", "file": "take_03.mp4"}}))
    no_text = (448, {"routed": True, "fine_detail": None, "why": "no task text", "cell_w": 448})
    assert route.route_width(ep, "sk-or-x", call) == no_text
    ctx.pop("task_label")
    (ep / "context.json").write_text(json.dumps(ctx))
    assert route.route_width(ep, "sk-or-x", call) == no_text
    ctx["profile"] = "handheld_gripper"
    (ep / "context.json").write_text(json.dumps(ctx))
    assert route.route_width(ep, "sk-or-x", call) == (None, {"routed": False})


def test_a_routing_call_is_recorded_and_counted_in_the_episode_cost():
    r = {"usage": {"est_cost_usd": 0.4}, "config": {"resolution_route": {"cost_usd": 0.0004}}}
    assert harness.episode_cost(r) == pytest.approx(0.4004)
    assert harness.episode_cost({"usage": {"est_cost_usd": 0.4}, "config": {}}) == 0.4


def test_a_portrait_phone_frame_is_decoded_upright():
    from PIL import Image

    class Frame:
        rotation = 90

        def to_image(self):
            return Image.new("RGB", (40, 20))
    assert mf.upright(Frame()).size == (20, 40)
    Frame.rotation = 0
    assert mf.upright(Frame()).size == (40, 20)


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


def test_openai_keys_are_used_only_without_openrouter_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-a, sk-or-b")
    monkeypatch.setenv("OPENROUTER_API_KEYS", "sk-or-c")
    assert harness.get_keys() == ["sk-or-c"]
    monkeypatch.setenv("OPENROUTER_API_KEYS", "")
    assert harness.get_keys() == ["sk-proj-a"]                         # an OpenRouter key never goes to OpenAI


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


def test_resume_source_identity_skips_same_input_and_stops_changed_input(tmp_path, monkeypatch):
    ep = tmp_path / 'episode_000000'
    ep.mkdir()
    (ep / 'context.json').write_text(json.dumps({'profile': 'teleop_arms', 'state_kind': 'none'}))
    (ep / 'sources.json').write_text('{}')
    out = tmp_path / 'out'
    out.mkdir()
    calls = []

    def fake_label(path, target, *, api_key, **kw):
        calls.append(path)
        result = {'parse_ok': True, 'input_identity': harness.input_identity(path, model='m', reasoning='medium',
                  max_tokens=100, cell_w=0, example_dir=None)}
        target.write_text(json.dumps(result))
        return result

    monkeypatch.setattr(harness, 'label_episode', fake_label)
    args = dict(keys=['sk-or-test'], concurrency=1, force=False, model='m', reasoning='medium', max_tokens=100,
                timeout=30)
    assert harness.run_batch([ep], out, **args) == 0
    assert harness.run_batch([ep], out, **args) == 0
    assert len(calls) == 1
    (ep / 'sources.json').write_text('{"camera": {}}')
    assert harness.run_batch([ep], out, **args) == 1
    assert len(calls) == 1
    assert json.loads((out / f'stale_{ep.name}.json').read_text())['status'] == 'stale'


def test_resume_legacy_reply_is_marked_unverified_without_a_paid_call(tmp_path, monkeypatch):
    ep = tmp_path / 'episode_000000'
    ep.mkdir()
    (ep / 'context.json').write_text(json.dumps({'profile': 'teleop_arms', 'state_kind': 'none'}))
    out = tmp_path / 'out'
    out.mkdir()
    (out / f'{ep.name}.json').write_text(json.dumps({'parse_ok': True}))
    monkeypatch.setattr(harness, 'label_episode', lambda *args, **kwargs: pytest.fail('unexpected paid call'))
    assert harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=False,
                             model='m', reasoning='medium', max_tokens=100, timeout=30) == 0
    assert json.loads((out / f'stale_{ep.name}.json').read_text())['status'] == 'unverified'


@pytest.mark.parametrize('receipt,body', [('episode_000000.json', '[]'),
                                           ('failed_episode_000000.json', 'null'),
                                           ('noreply_episode_000000.json', '[]')])
def test_resume_preserves_non_object_receipt_without_a_paid_call(tmp_path, monkeypatch, receipt, body):
    ep = tmp_path / 'episode_000000'
    ep.mkdir()
    (ep / 'context.json').write_text(json.dumps({'profile': 'teleop_arms', 'state_kind': 'none'}))
    out = tmp_path / 'out'
    out.mkdir()
    saved = out / receipt
    saved.write_text(body)
    monkeypatch.setattr(harness, 'label_episode', lambda *args, **kwargs: pytest.fail('unexpected paid call'))
    assert harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=False,
                             model='m', reasoning='medium', max_tokens=100, timeout=30) == 1
    marker = json.loads((out / f'stale_{ep.name}.json').read_text())
    assert marker['status'] == 'stale' and receipt in marker['reason']
    assert saved.read_text() == body


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
    assert len(list((tmp_path / "out").glob("episode_*.json"))) == 3
    # every episode the cap stopped says so, for the board (board/to_board.py label_outputs)
    left = sorted((tmp_path / "out").glob("noreply_episode_*.json"))
    assert len(left) == 7 and json.loads(left[0].read_text())["no_reply"] == "spend cap $3.00 reached"


def test_an_episode_that_got_no_reply_says_why_until_it_gets_one(tmp_path, monkeypatch):
    """An episode whose request could not be built, or whose call failed, left no file, so the board never showed it.
    Each gets noreply_<episode>.json with why; a later run that labels it removes it, and a dry run writes none."""
    state = {"fail": True}

    def fake_label(ep, out, *, api_key, **kw):
        if state["fail"] and ep.name.endswith("1"):
            raise ValueError("the request could not be built")
        out.write_text(json.dumps({"parse_ok": True, "usage": {"est_cost_usd": 0.1}}))
        return {"usage": {"est_cost_usd": 0.1}}

    monkeypatch.setattr(harness, "label_episode", fake_label)
    eps = _dirs(tmp_path, 3)
    assert harness.run_batch(eps, tmp_path / "out", keys=["sk-or-a"], concurrency=1, force=False) == 1
    rec = json.loads((tmp_path / "out" / "noreply_episode_000001.json").read_text())
    assert rec["no_reply"] == "ValueError: the request could not be built" and rec["parse_ok"] is False
    assert rec["episode_dir"] == str(eps[1])
    state["fail"] = False
    assert harness.run_batch(eps, tmp_path / "out", keys=["sk-or-a"], concurrency=1, force=False) == 0
    assert not (tmp_path / "out" / "noreply_episode_000001.json").exists()
    state["fail"] = True
    harness.run_batch(eps, tmp_path / "dry", keys=["dry-run"], concurrency=1, force=True, dry_run=True)
    assert not list((tmp_path / "dry").glob("noreply_*"))


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
    (ep / "sources.json").write_text(json.dumps({v: {"packed": str(packs[v]), "base_s": 50 / 30, "n_frames": T}
                                                for v in packs}))
    np.savez(ep / "state.npz", state=_state(T), action=_state(T))
    (ep / "context.json").write_text(json.dumps({
        "dataset": "allenai/MolmoAct2-BimanualYAM-Dataset", "robot_type": "bi_yam_follower", "fps": 30,
        "task_label": ["Spell out Ai2"], "instruction": "Spell AI2.", "n_state_frames": T, "profile": "teleop_arms",
        "state_kind": "joints", "cameras": {"exo": {"width": 64, "height": 36}}}))
    return ep, T


def test_final_timeout_claim_survives_resume_without_second_dispatch(tmp_path, monkeypatch):
    ep, _ = _packed_episode(tmp_path)
    out = tmp_path / 'out'
    calls = []
    def provider(*args, **options):
        calls.append(options.get('attempts'))
        raise TimeoutError('response outcome unknown')
    monkeypatch.setattr(harness, 'call_model', provider)
    args = dict(keys=['sk-or-fixture'], concurrency=1, force=False, model='openai/gpt-6-astra',
                reasoning='medium', max_tokens=2000, timeout=1, cell_w=128, max_spend=20)
    assert harness.run_batch([ep], out, **args) == 1
    receipt = json.loads((out / f'noreply_{ep.name}.json').read_text())
    assert calls == [1]
    assert receipt['final_dispatch_outcome'] == 'unverified'
    assert receipt['final_reserved_usd'] > 0
    assert harness.run_batch([ep], out, **args) == 0
    assert calls == [1]


@pytest.mark.parametrize('usage', [None, {}, {'prompt_tokens': 100},
                                   {'cost': 'unknown', 'prompt_tokens': 100},
                                   {'prompt_tokens': 100, 'completion_tokens': -1}])
def test_successful_reply_without_verifiable_usage_keeps_final_reservation(tmp_path, monkeypatch, usage):
    from label.dictionary_stage import label_spend, label_spend_complete
    out = tmp_path / 'job' / 'run' / 'out' / 'episode_a.json'
    out.parent.mkdir(parents=True)
    calls = []
    def provider(*args, **kwargs):
        calls.append(1)
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}],
                **({'usage': usage} if usage is not None else {})}
    monkeypatch.setattr(harness, 'call_model', provider)
    result = harness._call_and_record(tmp_path, out, [{'type': 'text', 'text': 'test'}], 0, {},
                                      model='openai/gpt-6-astra', reasoning='medium', api_key='fixture',
                                      max_tokens=100, timeout=1)
    assert calls == [1]
    assert result['parse_ok'] is True
    assert result['usage']['est_cost_usd'] is None
    assert result['final_dispatch_outcome'] == 'unverified'
    assert result['final_reserved_usd'] > 0
    assert harness.episode_cost(result) == result['final_reserved_usd']
    assert label_spend(tmp_path / 'job') == result['final_reserved_usd']
    assert label_spend_complete(tmp_path / 'job') is False


def test_complete_token_usage_settles_final_claim(tmp_path, monkeypatch):
    out = tmp_path / 'episode_a.json'
    monkeypatch.setattr(harness, 'call_model', lambda *a, **kw: {
        'usage': {'prompt_tokens': 100, 'completion_tokens': 10},
        'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]})
    result = harness._call_and_record(tmp_path, out, [{'type': 'text', 'text': 'test'}], 0, {},
                                      model='openai/gpt-6-astra', reasoning='medium', api_key='fixture',
                                      max_tokens=100, timeout=1)
    assert result['usage']['est_cost_usd'] > 0
    assert result['usage']['cost_source'] == 'estimate'
    assert 'final_dispatch_outcome' not in result


def test_run_batch_does_not_replay_successful_reply_with_missing_usage(tmp_path, monkeypatch):
    from label.dictionary_stage import label_spend, label_spend_complete
    ep, _ = _packed_episode(tmp_path)
    calls = []
    def provider(*args, **kwargs):
        calls.append(kwargs.get('attempts'))
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]}
    monkeypatch.setattr(harness, 'call_model', provider)
    out = tmp_path / 'job' / 'run' / 'out'
    args = dict(keys=['sk-or-fixture'], concurrency=1, force=False, model='openai/gpt-6-astra',
                reasoning='medium', max_tokens=2000, timeout=1, cell_w=128, max_spend=20)
    assert harness.run_batch([ep], out, **args) == 0
    receipt = json.loads((out / f'{ep.name}.json').read_text())
    assert receipt['parse_ok'] is True
    assert receipt['final_dispatch_outcome'] == 'unverified'
    assert calls == [1]
    assert label_spend(tmp_path / 'job') >= receipt['final_reserved_usd']
    assert label_spend_complete(tmp_path / 'job') is False
    assert harness.run_batch([ep], out, **args) == 0
    assert calls == [1]


def test_final_claim_survives_worker_crash_and_budget_refusal_has_no_dispatch(tmp_path, monkeypatch):
    ep, _ = _packed_episode(tmp_path)
    calls = []
    def crash(*args, **options):
        calls.append(options.get('attempts'))
        raise KeyboardInterrupt('worker stopped after dispatch')
    monkeypatch.setattr(harness, 'call_model', crash)
    args = dict(keys=['sk-or-fixture'], concurrency=1, force=False, model='openai/gpt-6-astra',
                reasoning='medium', max_tokens=2000, timeout=1, cell_w=128)
    refused = tmp_path / 'refused'
    assert harness.run_batch([ep], refused, max_spend=.001, **args) == 0
    assert calls == []
    out = tmp_path / 'crashed'
    with pytest.raises(KeyboardInterrupt):
        harness.run_batch([ep], out, max_spend=20, **args)
    receipt = json.loads((out / f'noreply_{ep.name}.json').read_text())
    assert receipt['final_dispatch_outcome'] == 'claimed'
    assert calls == [1]
    assert harness.run_batch([ep], out, max_spend=20, **args) == 0
    assert calls == [1]


def test_single_episode_cli_refuses_unreadable_no_reply_claim(tmp_path, monkeypatch):
    ep = tmp_path / 'episode_a'
    ep.mkdir()
    target = tmp_path / 'answer.json'
    (tmp_path / 'noreply_answer.json').write_text('damaged claim')
    monkeypatch.setenv('OPENROUTER_API_KEYS', 'sk-or-fixture')
    monkeypatch.setattr(sys, 'argv', ['label.harness', '--episode-dir', str(ep), '--out', str(target)])
    assert harness.main() == 1
    assert not target.exists()


SERVED = {"provider": "SomeHost", "id": "gen-123", "model": "openai/gpt-6-astra-20260901", "system_fingerprint": "fp_1"}


def test_episode_end_to_end_offline(tmp_path, monkeypatch):
    """A synthetic packed episode through label_episode (model call mocked) and into a board file."""
    ep, T = _packed_episode(tmp_path)
    sent = {}

    def fake_call(content, model, reasoning, api_key, max_tokens, timeout, **options):
        sent["content"] = content
        sent['attempts'] = options.get('attempts')
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
    assert sent['attempts'] == 1
    assert r["parse_ok"] and r["usage"]["est_cost_usd"] == 0.01 and r["usage"]["cost_source"] == "billed"
    assert (r["provider_name"], r["generation_id"], r["model_served"], r["system_fingerprint"]) == \
        ("SomeHost", "gen-123", "openai/gpt-6-astra-20260901", "fp_1")
    assert json.loads(out.read_text())["model_served"] == "openai/gpt-6-astra-20260901"
    assert r["config"]["timesteps_s"][0] == 0.0 and r["config"]["timesteps_s"][-1] == round(119 / 30, 3)
    assert r["config"]["cell"][0] == me.GRID_CELL_W_BY_RIG["teleop_arms"]
    assert isinstance(r["config"]["prompt_blocks"], list) and "contacts" not in r["config"]["schema_fields"]
    assert "stream_pairing" in r["config"]["checks_implied"] and "still_spans" not in r["config"]["checks_implied"]
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


def test_an_openai_key_sends_the_request_straight_to_openai(monkeypatch):
    """The same request in OpenAI's fields, no cache breakpoint and no usage field, and the list-price cost
    recorded with cached input at its own price. A model OpenAI does not serve is refused before any call."""
    seen = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    usage = {"prompt_tokens": 100000, "completion_tokens": 2000, "prompt_tokens_details": {"cached_tokens": 40000}}

    def fake_urlopen(req, timeout):
        seen["url"], seen["body"], seen["headers"] = req.full_url, json.loads(req.data), dict(req.header_items())
        return Resp(json.dumps({"choices": [{"message": {"content": "{}"}}], "usage": dict(usage)}).encode())

    monkeypatch.setattr(harness.urllib.request, "urlopen", fake_urlopen)
    content = [{"type": "text", "text": "x" * 5000}, {"type": "text", "text": "episode"}]
    resp = harness.call_model(content, "openai/gpt-6-astra", "medium", "sk-proj-x", 64000, 60)
    b = seen["body"]
    assert seen["url"] == harness.OPENAI_URL and seen["headers"]["Authorization"] == "Bearer sk-proj-x"
    assert set(b) == {"model", "messages", "max_completion_tokens", "reasoning_effort", "response_format"}
    assert (b["model"], b["reasoning_effort"], b["max_completion_tokens"]) == ("gpt-6-astra", "medium", 64000)
    assert all("cache_control" not in c for c in b["messages"][0]["content"])
    assert resp["provider"] == "OpenAI"
    assert resp["usage"]["list_cost"] == pytest.approx(60000 * 1e-5 + 40000 * 1e-6 + 2000 * 5e-5)
    assert harness._cost(resp["usage"]) == resp["usage"]["list_cost"]
    long = dict(usage, prompt_tokens=300000)                            # the long-context prices from 272,000
    assert harness.openai_list_cost("openai/gpt-6-sol", long) == pytest.approx(260000 * 4e-6 + 40000 * 4e-7
                                                                                + 2000 * 1.5e-5)
    with pytest.raises(RuntimeError, match="needs an OpenRouter key"):
        harness.call_model(content, "anthropic/claude-opus-5.5", "medium", "sk-proj-x", 64000, 60)


def test_cut_off_reply_is_kept_beside_the_outputs_and_fails(tmp_path, monkeypatch):
    ep, _ = _packed_episode(tmp_path)

    def fake_call(content, model, reasoning, api_key, max_tokens, timeout, **options):
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
    # what the board shows of an episode with no labels: its checks and the stretches no camera decoded
    assert "dataset_checks" in failed and "decode_failed" in failed and "given_prompt" in failed
    # the cut-off reply was billed: it counts toward the run's cost and the spend cap
    assert harness.episode_cost(failed) == 0.5 and lrun.billed_cost(tmp_path) == 0.5
    with pytest.raises(harness.Truncated) as e:
        harness.label_episode(ep, out, model="m", reasoning="medium", api_key="sk-or-x", max_tokens=64000, timeout=60)
    assert e.value.cost == 0.5


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
    # a shifted row, or one whose time is no number, is left out and counted; the rest of the reply is kept
    short = harness.normalize_timeline({"timeline": [
        [0.0, 1.0, "left"], [0.0, 1.5, "left", "reach", "cup", None, None, "idle", 0.1, None]]})
    assert [s["action"] for s in short["timeline"]] == ["reach"]
    assert short["_dropped"] == [{"field": "timeline", "row": 0, "why": "it has 3 values for 10 columns"}]
    late = harness.normalize_timeline({"timeline": [["late", 1.0, "left", "reach", "cup", None, None, "idle", 0.1,
                                                     None]]})
    assert late["timeline"] == [] and late["_dropped"][0]["why"] == "its start_s is not a number ('late')"


def test_one_bad_row_or_field_never_costs_the_whole_reply():
    """One string time in a timeline row made the whole reply unparsed, and a parsed reply with a field of the wrong
    type (a timeline object, key events as text, a task summary list) crashed the board build or the stitch. A time
    written as text that reads as a number is that number; a row or field of the wrong type is left out of that reply
    and counted in _dropped, and everything else is kept. A reply that keeps to the format is unchanged."""
    good = {"timeline_columns": harness.TIMELINE_COLUMNS,
            "timeline": [[0.0, 1.5, "left", "reach", "cup", None, None, "advancing", 0.1, None],
                         ["1.5", "3.0", "left", "lift", "cup", None, None, "advancing", "0.5", None],
                         ["abc", 4.0, "left", "drop", "cup", None, None, "wasteful", 0.5, None]],
            "task_summary": ["a", "b"], "key_events": [{"t_s": "2.5s", "label": "lifted"}, "goal reached",
                                                        {"t_s": "late", "label": "untimed"}],
            "completion": {"task_completed": "success", "completed_at_s": "3"}, "performance_review": 3,
            "scene": {"objects": ["cup", {"name": "plate"}]}, "data_issues": {"issue": "x"},
            "instruction_variants": ["lift the cup", 7]}
    labels, ok = harness.parse_response(json.dumps(good))
    assert ok
    assert [(s["start_s"], s["end_s"], s["progress"]) for s in labels["timeline"]] == [(0.0, 1.5, 0.1),
                                                                                      (1.5, 3.0, 0.5)]
    assert [(k["t_s"], k["label"]) for k in labels["key_events"]] == [(2.5, "lifted"), (None, "untimed")]
    assert labels["completion"]["completed_at_s"] == 3.0
    assert "task_summary" not in labels and "performance_review" not in labels and "data_issues" not in labels
    assert labels["scene"]["objects"] == [{"name": "plate"}] and labels["instruction_variants"] == ["lift the cup"]
    assert sorted((d["field"], d.get("row", -1)) for d in labels["_dropped"]) == [
        ("data_issues", -1), ("instruction_variants", 1), ("key_events", 1), ("performance_review", -1),
        ("scene.objects", 0), ("task_summary", -1), ("timeline", 2)]
    # each reason names the value in the output format's words, never Python's ("a str", "a int")
    why = {(d["field"], d.get("row", -1)): d["why"] for d in labels["_dropped"]}
    assert why[("performance_review", -1)] == "a number, not text" and why[("key_events", 1)] == "text, not an object"
    assert why[("task_summary", -1)] == "a list, not text" and why[("data_issues", -1)] == "an object, not a list"
    assert "key_events row 1: t_s 'late' is not a time, kept untimed" in labels["_schema_violations"]
    # running it again on its own output changes nothing, and a reply that keeps to the format comes back unchanged
    assert harness.typed_labels(json.loads(json.dumps(labels))) == labels
    clean = {"timeline": [{"start_s": 0.0, "end_s": 1.0, "action": "reach", "progress": 0.1}],
             "key_events": [{"t_s": 1, "label": "x"}], "completion": {"task_completed": "success",
                                                                    "completed_at_s": None},
             "task_summary": "reach", "scene": {"objects": [{"name": "cup"}], "setting": "a table"}}
    assert harness.parse_response(json.dumps(clean)) == (clean, True)
    # a reply that is JSON but not an object is a reply that did not parse
    labels, ok = harness.parse_response("[1, 2]")
    assert not ok and "not an object" in labels["_parse_error"]


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


def test_seeded_routing_answers_are_used_with_no_call(tmp_path, monkeypatch):
    """A comparison run seeded with the reference run's answers sends each task text at the width the reference
    did, with no routing call and nothing billed; an answer that is not true or false is refused."""
    from label import route
    monkeypatch.setattr(route, "_CACHE", {})
    ep, _ = _packed_episode(tmp_path)
    text = route.route_text(me.load(ep))
    assert route.seed({text: {"fine_detail": True, "why": "lettering"}}, "routes_main.json") == 1
    call, calls = _route_call([])
    w, rec = route.route_width(ep, "sk-or-x", call)
    assert (w, rec["fine_detail"], rec["cost_usd"], rec["seeded_from"], calls) == (448, True, 0.0,
                                                                                   "routes_main.json", [])
    with pytest.raises(ValueError, match="not true or false"):
        route.seed({text: {"fine_detail": None}}, "x")


def test_a_camera_taller_than_the_first_keeps_its_whole_frame_in_the_grid():
    """Rows are laid out with the first camera's cell height; a 4:3 wrist camera under a 16:9 scene camera must still
    show its whole frame (here: a white band in its bottom quarter)."""
    import io
    from PIL import Image
    from label import frames as mf
    top = Image.new("RGB", (1280, 720), (200, 0, 0))
    wrist = Image.new("RGB", (640, 480), (0, 200, 0))
    wrist.paste((255, 255, 255), (0, 360, 640, 480))              # bottom quarter white
    cols = [(0.0, {"top": mf.to_jpeg(top, 224), "left": mf.to_jpeg(wrist, 224), "right": mf.to_jpeg(wrist, 224)})]
    g = Image.open(io.BytesIO(mf.compose_grid(cols, ["top", "left", "right"], 95, 84, 30))).convert("RGB")
    white_rows = 0
    for y in range(g.height):
        r, gg, b = g.getpixel((84 + 112, y))
        white_rows += r > 230 and gg > 230 and b > 230
    assert white_rows >= 2 * 40


def test_the_grid_font_lays_text_out_the_same_on_every_machine():
    from PIL import ImageFont
    from label import frames as mf
    assert mf._grid_font(22).layout_engine == ImageFont.Layout.BASIC


def test_one_arm_is_named_by_its_own_camera_not_an_extra_one():
    from label import state as ms
    ep = {"context": {"dataset": "mine", "fps": 30, "profile": "teleop_arms", "state_kind": "joints",
                      "cameras": {"exo": {"name": "top"}, "right": {"name": "wrist"}, "extra1": {"name": "front"}}},
          "state": np.zeros((300, 7)), "sources": {"exo": {}, "right": {}, "extra1": {}}}
    assert me.actors(ep) == ["wrist"]
    pl = {"n": 300, "spans": [], "ks": ms.sample_frames(300, [], moving_every_s=1.5, still_every_s=1.5)}
    assert 'always names the one arm: "wrist"' in me.build_prompt(ep, pl, cell_w=224, cell_h=126)[1]


def test_a_camera_that_starts_late_is_not_shown_before_its_first_frame():
    """RealOmni's right camera can start 2 s after the left: at 0 s and 1 s its nearest frame is its first, taken at
    2.03 s, and must not appear under those times."""
    left = np.arange(0, 6, 1 / 30)
    right = np.arange(2.03, 6, 1 / 30)
    from prepare import formats
    ep = {"context": {"fps": 30, "cameras": {"left": {"name": "left"}, "right": {"name": "right"}}},
          "sources": {"left": {}, "right": {}}, "times": {"left": left, "right": right},
          "kmap": {"right": formats.nearest(right, left)}}
    ks = [0, 30, 60, 90, 120]
    assert [me.recording_at(ep, "right", k) for k in ks] == [False, False, True, True, True]
    assert all(me.recording_at(ep, "left", k) for k in ks)
    assert ("Right has frames only from 2.03 s to 6.00 s"
            in me._coverage_note(ep, {"ks": ks}))


def test_an_instant_a_hair_before_the_episode_start_is_said_as_zero_seconds_never_minus_zero():
    """A recorder's clock can put the main camera's first frame a fraction of a millisecond before the episode's zero
    (-0.0004 s). The prompt says that instant as 0.00 s; "-0.00 s" reads as a time before the episode."""
    left = np.arange(0, 6, 1 / 30) - 0.0004
    right = np.arange(2.03, 6, 1 / 30)
    from prepare import formats
    ep = {"context": {"fps": 30, "cameras": {"left": {"name": "left"}, "right": {"name": "right"}}},
          "sources": {"left": {}, "right": {}}, "times": {"left": left, "right": right},
          "kmap": {"right": formats.nearest(right, left)}, "footage_end": 0}
    note = me._coverage_note(ep, {"ks": [0, 30, 60, 90, 120]})
    assert "Right has frames only from 2.03 s to 6.00 s" in note
    assert "the last frame they have, at 0.00 s." in note
    assert "-0.00" not in note


def test_the_prompt_gives_each_cameras_own_cell_size_when_they_differ():
    """Rexair: a portrait scene camera (480x640) above two 640x480 wrist cameras. The cells are 448x598 and 448x336."""
    from label import state as ms
    cams = {"exo": {"name": "top", "width": 480, "height": 640}, "left": {"name": "left", "width": 640, "height": 480},
            "right": {"name": "right", "width": 640, "height": 480}}
    ep = {"context": {"dataset": "rexair", "fps": 30, "profile": "teleop_arms", "state_kind": "joints", "cameras": cams},
          "state": np.zeros((300, 14)), "sources": {"exo": {}, "left": {}, "right": {}}}
    pl = {"n": 300, "spans": [], "ks": ms.sample_frames(300, [], moving_every_s=1.5, still_every_s=1.5)}
    p = me.build_prompt(ep, pl, cell_w=448, cell_h=598)[1]
    assert "downscaled to 448 px wide (top 448x598, left 448x336, right 448x336)" in p
    for c in cams.values():
        c.update(width=640, height=480)
    assert "downscaled to 448x336." in me.build_prompt(ep, pl, cell_w=448, cell_h=336)[1]


def _scene(rng, i: int, w: int = 640, h: int = 480, level: float = 120.0) -> np.ndarray:
    """A lit scene that moves from frame to frame: noise over a pattern that shifts with i."""
    xx = np.arange(w)[None, :, None]
    return (rng.random((h, w, 3)) * 60 + level + 40 * np.sin(xx / 37 + i)).clip(0, 255)


def _frames(make, n: int = 12) -> dict:
    from PIL import Image
    rng = np.random.default_rng(0)
    return {k * 30: Image.fromarray(make(rng, k).astype(np.uint8)) for k in range(n)}


def _circular(rng, i):
    """A fisheye's image circle inside the sensor: outside it the sensor stays black (with a little noise)."""
    a = _scene(rng, i)
    yy, xx = np.mgrid[0:480, 0:640]
    out = np.hypot(xx - 319.5, yy - 239.5) > 300
    a[out] = rng.integers(0, 8, size=(int(out.sum()), 3))
    return a


def _dark_room(rng, i):
    """A dark scene with one lit patch: most of the image is near black in every frame, the corners included."""
    a = rng.random((480, 640, 3)) * 14
    a[200:280, 280 + i:360 + i] += 150
    return a


def _fingers(rng, i):
    """A gripper camera: its dark fingers fill the bottom edge in every frame, the corners there included."""
    a = _scene(rng, i)
    a[380:] = rng.integers(0, 10, size=(100, 640, 3))
    return a


def test_a_circular_image_names_a_fisheye_lens_and_other_images_do_not():
    from label import lens
    assert lens.circular_image(_frames(_circular))["circular"]
    for make in (_scene, _dark_room, _fingers):
        assert not lens.circular_image(_frames(make))["circular"], make.__name__
    # a frame kept only at cell widths gives the thumbnail its full-size frame gives
    im = _frames(_circular, 1)[0]
    assert np.array_equal(lens.thumb(im), lens.thumb(mf.Shrunk(im, [448, 320, 192])))


def test_the_camera_line_names_the_fisheye_only_where_the_check_fired():
    ep = {"context": {"dataset": "x", "fps": 30, "profile": "handheld_gripper", "state_kind": "none",
                      "cameras": {"left": {"name": "left", "desc": "the camera carried on the LEFT-hand gripper"},
                                  "right": {"name": "right", "desc": "the camera carried on the RIGHT-hand gripper"}}},
          "sources": {"left": {}, "right": {}}, "lens": {"left": {"circular": True}, "right": {"circular": False}}}
    desc = me.camera_desc(ep)
    assert "- left: the camera carried on the LEFT-hand gripper. It has a fisheye lens; straight lines curve near " \
           "the edge.\n" in desc
    assert "- right: the camera carried on the RIGHT-hand gripper.\n" in desc
    assert desc.count("fisheye") == 1
    del ep["lens"]
    assert "fisheye" not in me.camera_desc(ep)


def test_circular_image_with_no_frames_is_not_circular():
    from label import lens
    assert lens.circular_image({}) == {"circular": False, "frames": 0}


def test_a_dataset_step_with_no_end_or_no_time_is_stated_as_it_is():
    """The dataset's timed steps in the prompt: a step with no end time is a moment, one with no time is listed
    without one, and neither stops the prompt from being built."""
    block = me.ego_annotation_block({"annotation_subtasks": [{"t0": 1.0, "t1": 2.5, "label": "reach"},
                                                             {"t0": 3.0, "label": "open"},
                                                             {"t0": None, "label": "wipe"}]})
    assert "  1.0-2.5s  reach" in block and "  3.0s  open" in block and "  no time  wipe" in block
