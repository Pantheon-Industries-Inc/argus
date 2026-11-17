"""A new policy leaves IMU rows and depth pictures out of the request and names each stream that never changes once
(label/policy.py IMU, DEPTH_PICTURES); --imu on and --depth-pictures on, and a policy saved before the settings, give
the request as it was before them."""
import json

import numpy as np
import pytest

from label import episode as me
from label import policy
from label import signals as sg
from test_depth_inspection_scope import depth_episode
from test_prompt_blocks import CASES

# the twelve IMU streams of an Ego recording (1x onlincam onl9997): accelerometer and gyroscope, and their spread over
# each frame, of the head and both wrists
EGO_IMU = [f"{where}.{what}{std}" for where in ("ego", "left_wrist", "right_wrist")
           for what in ("accel_mps2", "gyro_rads") for std in ("", "_std")]


def _ego(**settings):
    """A head camera recording with the Ego IMU streams and one stream that is not an IMU, under a policy resolved
    with these settings (none: a new policy)."""
    ep, pl = CASES["ego_plain"]()
    n = len(ep["state"])
    t = np.arange(n) / 30.0
    rng = np.random.default_rng(0)
    ep["signals"] = {name: np.stack([np.sin(t + j) + 0.1 * rng.normal(size=n) for j in range(3)], axis=1)
                     for name in EGO_IMU}
    ep["signals"]["glove.pressure_kpa"] = (2 + np.sin(0.5 * t))[:, None]
    ep["signals"]["recorder.gain"] = np.full((n, 1), 4.0)
    ep["signal_meta"] = {name: {"names": ["x", "y", "z"]} for name in EGO_IMU}
    ep["signal_meta"].update({"glove.pressure_kpa": {}, "recorder.gain": {}})
    ep["context"]["annotation_policy"] = policy.resolve(ep["context"], **settings)
    return ep, pl


def _signals_text(ep, pl):
    return me.build_prompt(ep, pl, cell_w=448, cell_h=252)[1]


def test_a_new_policy_names_the_ego_imu_streams_and_gives_none_of_their_rows():
    episode = _signals_text(*_ego())
    for name in EGO_IMU:
        assert f"    {name} x: " not in episode and f"  {name} (" not in episode
    assert "IMU streams also recorded, their readings not given here: " + ", ".join(EGO_IMU) + "." in episode
    assert "    glove.pressure_kpa: " in episode          # the other streams keep their rows


def test_imu_on_gives_the_ego_imu_rows_as_before():
    on = _signals_text(*_ego(imu="on", depth_pictures="on"))
    legacy_ep, pl = _ego()
    legacy_ep["context"]["annotation_policy"] = {k: v for k, v in legacy_ep["context"]["annotation_policy"].items()
                                                 if k not in ("imu", "depth_pictures")}
    assert on == _signals_text(legacy_ep, pl)
    for name in EGO_IMU:
        assert f"    {name} x: " in on
    assert "IMU streams also recorded" not in on


def test_a_constant_stream_is_named_in_one_list_under_a_new_policy_and_keeps_its_value_line_otherwise():
    new = _signals_text(*_ego())
    assert "Also recorded, each the same wherever it reads (values not given): recorder.gain." in new
    assert "The same at every frame: recorder.gain 4" not in new
    old = _signals_text(*_ego(imu="on", depth_pictures="on"))
    assert "The same at every frame: recorder.gain 4" in old and "values not given" not in old


def test_imu_off_alone_or_depth_pictures_off_alone_names_the_constant_streams():
    for settings in ({"imu": "off", "depth_pictures": "on"}, {"imu": "on", "depth_pictures": "off"}):
        assert "(values not given): recorder.gain." in _signals_text(*_ego(**settings))


@pytest.mark.parametrize("name, source", [
    ("ego.accel_mps2", None), ("left_wrist.gyro_rads_std", None), ("observation.base.imu", None),
    ("observation.state.chassis.imu", None), ("/robot0/sensor/imu angular_velocity", "MCAP channel /robot0/sensor/imu"),
    ("/camera/camera/accel/sample orientation_covariance", "MCAP channel /camera/camera/accel/sample"),
    ("/camera/camera/gyro/sample linear_acceleration", "MCAP channel /camera/camera/gyro/sample"),
    ("headIMU", None), ("imu0", None), ("/gripper_left/imu angular_velocity", "MCAP channel /gripper_left/imu"),
    ("accelerometer", None), ("wrist_gyroscope", None),
    # an IMU by its message type, whatever its topic is called
    ("/livox/data linear_acceleration", "MCAP channel /livox/data"), ("/livox/data orientation", "MCAP channel /livox/data"),
])
def test_these_streams_are_imu(name, source):
    assert sg.is_imu(name, source, imu_topics={"/livox/data"})


@pytest.mark.parametrize("name, source", [
    ("joint_accel", None), ("observation.joint_acceleration", None), ("qacc", None), ("joint_vel", None),
    ("/yam_left/joint_state joint_vel", "MCAP channel /yam_left/joint_state"), ("gripper_accel", None),
    ("ee_accel", None), ("eef_acceleration", None), ("base_velocity", None), ("observation.base_vel", None),
    ("/cmd_vel linear", "MCAP channel /cmd_vel"), ("/cmd_vel angular", "MCAP channel /cmd_vel"),
    ("action.accel_cmd", None), ("observation.state.chassis", None), ("/robot0/sensor/magnetic_encoder", None),
    ("observation.base.odom", None), ("left_gripper_pos", None), ("observation.state.left_ee_pose", None),
    ("/tf base_link translation", "MCAP channel /tf"), ("angular_velocity", None), ("orientation", None),
    ("/livox/lidar orientation", "MCAP channel /livox/lidar"),
])
def test_these_streams_are_not_imu(name, source):
    assert not sg.is_imu(name, source, imu_topics={"/livox/data"})


def test_the_imu_topics_are_read_from_the_mcap_message_types():
    ctx = {"mcap_field_inventory": [{"topic": "/camera/camera/accel/sample", "schema": "sensor_msgs/msg/Imu"},
                                    {"topic": "/livox/imu", "schema": "sensor_msgs/Imu"},
                                    {"topic": "/camera/camera/infra1/camera_info", "schema": "sensor_msgs/msg/CameraInfo"},
                                    {"topic": "/tf", "schema": None}]}
    assert sg.imu_topics(ctx) == {"/camera/camera/accel/sample", "/livox/imu"}


def test_a_new_policy_sends_no_depth_picture_and_keeps_the_depth_note(tmp_path):
    home = depth_episode(tmp_path)
    off = me.build_request(home, inspect_evidence=True)
    assert "depth_s" not in off and "depth" in off["blocks"]
    assert "\nDEPTH: " in off["prompt"] and "No depth picture is sent with this request." in off["prompt"]
    assert not any(p["type"] == "text" and p["text"].startswith("=== depth") for p in off["content"])
    on = me.build_request(home, inspect_evidence=True, depth_pictures="on")
    assert on["depth_s"] and on["n_images"] == off["n_images"] + len(on["depth_s"])
    assert "Detail views are followed by available recorded depth" in on["prompt"]
    assert off["evidence_access"] is None and on["evidence_access"] is None   # depth never buys an inspection


def test_a_policy_saved_before_the_settings_gives_the_request_it_was_paid_for(tmp_path):
    home = depth_episode(tmp_path)
    path = home / "context.json"
    ctx = json.loads(path.read_text())
    saved = policy.resolve(ctx)
    ctx["annotation_policy"] = {k: v for k, v in saved.items() if k not in ("imu", "depth_pictures")}
    path.write_text(json.dumps(ctx))
    legacy = me.build_request(home, inspect_evidence=True)
    assert policy.imu_mode(legacy["annotation_policy"]) == "on"
    assert policy.depth_pictures_mode(legacy["annotation_policy"]) == "on"
    assert legacy["depth_s"]
    on = me.build_request(home, inspect_evidence=True, imu="on", depth_pictures="on")
    assert legacy["prompt"] == on["prompt"] and legacy["n_images"] == on["n_images"]


def test_a_new_policy_saves_both_settings_off_and_a_policy_with_both_on_saves_neither():
    ctx = {"profile": "ego_head"}
    new = policy.resolve(ctx)
    assert new["imu"] == "off" and new["depth_pictures"] == "off"
    both = policy.resolve(ctx, imu="on", depth_pictures="on")
    assert "imu" not in both and "depth_pictures" not in both
    assert policy.resolve({**ctx, "annotation_policy": new})["imu"] == "off"     # a resume keeps what was saved
    with pytest.raises(ValueError):
        policy.resolve(ctx, imu="maybe")
