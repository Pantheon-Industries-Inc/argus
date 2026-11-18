"""Every rig's prompt asks for colours judged from the clearest view and lets look-alike objects share a description
(label/prompts.py; the case is ABC-130k episode_ce55ec68, whose black jar cap the labels called dark green)."""
import pytest

from label import prompts

RIGS = ("teleop_arms", "handheld_gripper", "ego_head")


def flat(text):
    return " ".join(text.split())


@pytest.mark.parametrize("rig", RIGS)
@pytest.mark.parametrize("given", [True, False])
def test_every_rig_judges_colour_from_the_clearest_view_and_never_invents_a_difference(rig, given):
    text = flat(prompts.fixed_instructions(rig, has_instruction=given, recorded=True, has_steps=False))
    assert "When you name an object's colour, be sure of it: name it from the clearest, closest view of it you have" in text
    assert "a small or dark object in a low-resolution cell can look tinted" in text
    assert "Refer to each one consistently, in whatever way works" in text
    assert "never give it a property it does not have just to tell it apart" in text
    assert text.count("Some objects look alike or are identical.") == 1
