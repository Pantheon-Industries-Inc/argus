"""python -m prepare lerobot --whole-body / --robot write the robot's body and description into context.json."""
import json

from prepare import lerobot


def test_describe_writes_body_and_robot(tmp_path):
    (tmp_path / "context.json").write_text(json.dumps({"robot_type": "unitree_go2", "profile": "teleop_arms"}))
    lerobot.describe(tmp_path, "legged_no_arms", "Unitree Go2 quadruped")
    ctx = json.loads((tmp_path / "context.json").read_text())
    assert ctx["whole_body"] == "legged_no_arms" and ctx["robot_type"] == "Unitree Go2 quadruped"
    lerobot.describe(tmp_path, "humanoid", None)
    assert json.loads((tmp_path / "context.json").read_text())["whole_body"] is True


def test_describe_without_options_leaves_context_alone(tmp_path):
    (tmp_path / "context.json").write_text('{"robot_type": "x"}')
    lerobot.describe(tmp_path, None, None)
    assert (tmp_path / "context.json").read_text() == '{"robot_type": "x"}'
