"""The shared instructions of a video only robot episode, and of a robot dataset of sessions."""
from label import prompts


def test_video_only_instructions_say_nothing_of_a_recorded_motion():
    for rig in ("teleop_arms", "handheld_gripper"):
        recorded = prompts.fixed_instructions(rig, has_instruction=True, recorded=True)
        video = prompts.fixed_instructions(rig, has_instruction=True, recorded=False)
        assert "recorded motion" in recorded
        assert "recorded motion" not in video
        assert "state_video_mismatch" not in video
        # the recorded variant is the pinned text, unchanged
        assert recorded == prompts.fixed_instructions(rig, has_instruction=True)


def test_a_dataset_of_sessions_is_labelled_with_tasks():
    plain = prompts.fixed_instructions("handheld_gripper", has_instruction=False, recorded=False)
    sess = prompts.fixed_instructions("handheld_gripper", has_instruction=False, recorded=False, sessions=True)
    assert prompts.NO_INSTRUCTION_RULES in plain and prompts.NO_INSTRUCTION_RULES not in sess
    assert '"tasks": [' in sess and "ABOUT THE TASKS" in sess
    assert "grippers move on" in sess
    teleop = prompts.fixed_instructions("teleop_arms", has_instruction=False, sessions=True)
    assert "arms move on" in teleop and "grippers move on" not in teleop
    # the head-camera rig keeps its own schema whatever the flags
    assert prompts.fixed_instructions("ego_head", sessions=True) == prompts.fixed_instructions("ego_head")
