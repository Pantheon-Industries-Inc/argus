"""The signals setting of the annotation policy (label/policy.py): one table of what each stream kind gets on every path
to the model, and the auto, all and off modes over it."""
import argparse

import pytest

from label import policy
from prepare.signal_kind import KINDS

pytestmark = pytest.mark.signals_policy


def test_every_kind_has_a_treatment_on_every_path():
    assert set(policy.SIGNAL_TREATMENT) == set(KINDS)
    for kind, t in policy.SIGNAL_TREATMENT.items():
        assert t["prompt"] in policy.PROMPT_TREATMENTS, kind
        assert type(t["inspection"]) is bool and type(t["sensor_evidence"]) is bool, kind


def test_the_approved_defaults():
    t = policy.SIGNAL_TREATMENT
    assert {k for k, v in t.items() if v["prompt"] == "rows"} == {"joints", "gripper", "pose", "touch", "force_torque",
                                                                     "base"}
    assert {k for k, v in t.items() if v["prompt"] == "one_line"} == {"imu", "velocity", "effort", "unknown"}
    assert {k for k, v in t.items() if v["prompt"] == "drop"} == {"constant", "bookkeeping"}
    assert t["depth"]["prompt"] == "note"
    # force/torque is sent once, as rows, not again as sensor evidence series; touch is both
    assert not t["force_torque"]["sensor_evidence"] and t["touch"]["sensor_evidence"]
    # what leaves the rows stays fetchable through inspection; a constant or bookkeeping stream has nothing to fetch
    assert all(v["inspection"] for k, v in t.items() if k not in ("constant", "bookkeeping"))
    assert not t["constant"]["inspection"] and not t["bookkeeping"]["inspection"]


def test_all_treats_every_kind_as_today_and_off_lists_every_stream():
    for kind in KINDS:
        assert policy.signal_treatment(kind, "all") == policy.TODAY
        assert policy.signal_treatment(kind, "off")["prompt"] == "drop"
        assert policy.signal_treatment(kind, "auto") == policy.SIGNAL_TREATMENT[kind]
    with pytest.raises(ValueError):
        policy.signal_treatment("joints", "some")


def test_a_new_policy_writes_auto_and_a_saved_one_without_the_setting_keeps_all():
    new = policy.resolve({"profile": "teleop_arms"})
    assert new["signals"] == "auto" and policy.signals_mode(new) == "auto"
    saved = {k: v for k, v in new.items() if k != "signals"}
    resumed = policy.resolve({"profile": "teleop_arms", "annotation_policy": saved})
    assert "signals" not in resumed and policy.signals_mode(resumed) == "all"
    # a mode asked for wins over the saved one, and all is written as the absence of the key, as inspection's legacy is
    assert policy.resolve({"profile": "teleop_arms", "annotation_policy": new}, signals="off")["signals"] == "off"
    assert "signals" not in policy.resolve({"profile": "teleop_arms"}, signals="all")
    with pytest.raises(ValueError):
        policy.resolve({"profile": "teleop_arms"}, signals="some")


def test_the_cli_flag_is_shaped_like_inspection():
    ap = argparse.ArgumentParser()
    policy.add_arguments(ap)
    assert ap.parse_args([]).signals is None
    assert ap.parse_args(["--signals", "off"]).signals == "off"
    with pytest.raises(SystemExit):
        ap.parse_args(["--signals", "some"])
