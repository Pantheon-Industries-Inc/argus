"""Every rig's timeline note asks for the direction of motion in the actor's own frame, never a camera's
(label/prompts.py direction_note)."""
import pytest

from label import prompts

FRAMES = {"teleop_arms": "the robot's own frame", "handheld_gripper": "the frame of the person holding the gripper",
          "ego_head": "the wearer's own frame"}


def flat(text):
    return " ".join(text.split())


@pytest.mark.parametrize("rig", list(FRAMES))
@pytest.mark.parametrize("given", [True, False])
def test_direction_is_named_in_the_actors_frame_and_only_when_plain(rig, given):
    text = flat(prompts.fixed_instructions(rig, has_instruction=given, recorded=True, has_steps=False))
    assert text.count("Name the direction of motion when the frames show it plainly.") == 1
    assert f"say which way in {FRAMES[rig]} (left or right, forward or back, up or down), never in a camera's frame" in text
    assert "Leave it out when it is not plain; do not work it out." in text
