"""Resolution routing is off by default (label/route.py enabled): a teleop episode gets the narrow cells with no call,
and ARGUS_RESOLUTION_ROUTE=on (--resolution-route on) routes it by its task text as before."""
import pytest

from label import route


def teleop_episode(monkeypatch, instruction="Read the label on the jar."):
    ep = {"context": {"profile": "teleop_arms", "instruction": instruction}}
    monkeypatch.setattr(route.me, "load", lambda ep_dir: ep)
    return "episode_a"


def refuse_call(*args, **kwargs):
    raise AssertionError("routing is off, so no model call is made")


def test_off_by_default_sends_narrow_cells_with_no_call(monkeypatch):
    monkeypatch.delenv("ARGUS_RESOLUTION_ROUTE", raising=False)
    width, record = route.route_width(teleop_episode(monkeypatch), "sk-test", refuse_call)
    low, _ = route.me.ROUTE_WIDTHS["teleop_arms"]
    assert width == low and record == {"routed": False, "why": "resolution routing off", "cell_w": low}


@pytest.mark.parametrize("value", ["off", "OFF", "", "yes"])
def test_anything_but_on_is_off(monkeypatch, value):
    monkeypatch.setenv("ARGUS_RESOLUTION_ROUTE", value)
    assert route.enabled() is False


def test_on_routes_by_the_task_text(monkeypatch):
    monkeypatch.setenv("ARGUS_RESOLUTION_ROUTE", "on")
    called = []
    monkeypatch.setattr(route, "_paid_route", lambda *a, **k: called.append(1) or {"fine_detail": True, "why": "lettering"})
    width, record = route.route_width(teleop_episode(monkeypatch), "sk-test", refuse_call)
    _, high = route.me.ROUTE_WIDTHS["teleop_arms"]
    assert called and width == high and record["routed"] is True


def test_other_rigs_are_never_routed(monkeypatch):
    monkeypatch.setenv("ARGUS_RESOLUTION_ROUTE", "on")
    monkeypatch.setattr(route.me, "load", lambda ep_dir: {"context": {"profile": "ego_head"}})
    assert route.route_width("episode_a", "sk-test", refuse_call) == (None, {"routed": False})
