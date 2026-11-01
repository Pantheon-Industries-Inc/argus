"""A robot that moves its whole body (context["whole_body"]) gets teleop_arms with WHOLE_BODY_WORDING, in every
teleop variant, and every other request keeps main's words exactly."""
import itertools

import pytest

from label import prompts

VARIANTS = list(itertools.product((True, False), (True, False), (True, False)))


@pytest.mark.parametrize("has_instruction,recorded,has_steps", VARIANTS)
def test_every_teleop_variant_takes_every_change(has_instruction, recorded, has_steps):
    kw = dict(has_instruction=has_instruction, recorded=recorded, has_steps=has_steps)
    base = prompts.fixed_instructions("teleop_arms", **kw)
    body = prompts.fixed_instructions("teleop_arms", whole_body=True, **kw)
    for old, new in prompts.WHOLE_BODY_WORDING:
        assert base.count(old) == 1 and body.count(new) == 1
    assert '"body"' in body and "robot-arm activity" not in body and "the arm is not engaged" not in body
    assert "a person reaching in or moving objects" not in body


@pytest.mark.parametrize("rig", ["handheld_gripper", "ego_head"])
def test_other_rigs_ignore_the_flag(rig):
    assert prompts.fixed_instructions(rig, whole_body=True) == prompts.fixed_instructions(rig)


@pytest.mark.parametrize("rig", ["teleop_arms", "handheld_gripper", "ego_head"])
@pytest.mark.parametrize("has_instruction,recorded,has_steps", VARIANTS)
def test_without_the_flag_every_prompt_is_unchanged(rig, has_instruction, recorded, has_steps):
    kw = dict(has_instruction=has_instruction, recorded=recorded, has_steps=has_steps)
    assert prompts.fixed_instructions(rig, **kw) == prompts.fixed_instructions_for_rig(rig, **kw)


def test_body_is_an_allowed_arm():
    from board.label_projection import allowed_arms
    assert "body" in allowed_arms()
