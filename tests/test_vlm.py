"""The prescribed VLMs (label/vlm.py): Astra keeps its per-rig sampling and part length, every other model is sent 2
instants per second in parts of at most 330 s, and a part cut for one model is refused by a model that needs another
part length."""
from __future__ import annotations

import pytest

from label import episode as me
from label import pieces, vlm


def _ep(rig):
    return {"context": {"profile": rig}}


def test_astra_keeps_its_per_rig_rates_and_every_other_model_gets_2_per_second():
    for rig, astra in (("teleop_arms", 1.5), ("handheld_gripper", 1.0), ("ego_head", 0.5)):
        assert me.every_s(_ep(rig)) == astra                                  # no model given is the default, Astra
        assert me.every_s(_ep(rig), vlm.ASTRA) == astra
        assert me.every_s(_ep(rig), "openai/gpt-6-astra-20260901") == astra   # a dated id it is served as
        assert me.every_s(_ep(rig), vlm.SOL61) == 0.5
        assert me.every_s(_ep(rig), "anthropic/claude-opus-5.5") == 0.5


def test_part_length_follows_the_model():
    for rig in ("teleop_arms", "handheld_gripper", "ego_head"):
        assert pieces.piece_max({"profile": rig}, vlm.ASTRA) == 450.0
        assert pieces.piece_max({"profile": rig}, vlm.SOL61) == 330.0


def test_each_prescribed_model_runs_at_its_own_reasoning_effort():
    assert vlm.reasoning_for(vlm.SOL61, "medium") == "high"
    assert vlm.reasoning_for(vlm.ASTRA, "medium") == "medium"
    assert vlm.reasoning_for("deepseek/deepseek-v4.1-flash", "medium") == "medium"


def test_a_part_cut_for_astra_is_refused_by_a_model_that_needs_shorter_parts(tmp_path, monkeypatch):
    from prepare import formats
    from test_formats import _clip
    (tmp_path / "v").mkdir()
    _clip(tmp_path / "v" / "a.mp4", 90)                      # 3 s at 30 fps
    ep = tmp_path / "eps" / "episode_a"
    formats.video_views_episode(ep, {"exo": ("a", tmp_path / "v" / "a.mp4")}, "ego_head", "mine", {})
    monkeypatch.setitem(pieces.PIECE_MAX_S, "ego_head", 1.6)
    parts = pieces.write_pieces(ep, tmp_path / "pieces")    # cut for Astra
    assert me.load(parts[0])["context"]["piece"]["max_s"] == 1.6
    me.build_request(parts[0])                               # Astra labels its own parts
    with pytest.raises(ValueError, match="cut the recording again"):
        me.build_request(parts[0], model=vlm.SOL61)


def test_a_request_is_sampled_and_recorded_at_its_models_rate(tmp_path, monkeypatch):
    from prepare import formats
    from test_formats import _clip
    (tmp_path / "v").mkdir()
    _clip(tmp_path / "v" / "a.mp4", 90)
    ep = tmp_path / "eps" / "episode_a"
    formats.video_views_episode(ep, {"exo": ("a", tmp_path / "v" / "a.mp4")}, "ego_head", "mine", {})
    monkeypatch.setitem(me.SAMPLE_EVERY_S, "ego_head", 1.0)
    astra, sol = me.build_request(ep), me.build_request(ep, model=vlm.SOL61)
    assert astra["sampling"] == "ego_head-every-1s" and sol["sampling"] == "ego_head-every-0.5s"
    assert len(sol["timesteps"]) > len(astra["timesteps"])
