"""The rules every public board gets from `board static site` (board/public_board.py). Before them a public board was
patched by hand after the build: a CSS line to hide the dataset tabs and the repeated board name, a script that emptied
the data issues and their counts, put the summary on each card and moved chosen episodes first. A split episode showed
the summary of the whole recording it came from, and an empty summary went out as an empty card."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from board import static

SUMMARY = "(1) Return the loose keycaps to the rows. (2) Build a short row on the lid."
SUBTASK = "Arrange selected letter keycaps in a straight row near the center of the laptop lid."


def _label(eid: str, summary: str = SUMMARY, **extra) -> dict:
    return {"dataset": "DiverseUMI", "_rig": "handheld_gripper", "duration_s": 107.1, "episode_prompt": summary,
            "camera_views": ["exo"], "_meta": {"episode_id": eid},
            "tasks": [{"start_s": 0.0, "end_s": 107.1, "task": SUBTASK, "outcome": "success"}],
            "data_issues": [{"issue": "The operator's chin is in view at the frame edge.", "category": "face_visible",
                             "severity": "medium", "t_s": 3.0}],
            "operator_mistakes": [{"issue": "Several empty pickups of one cap.", "category": "failed_grasp",
                                   "severity": "low", "t_s": 29.0}],
            "dataset_checks": {"reader_issues": [
                {"kind": "signal_gap", "what": "left_pose_world has no reading for 2 s", "family": "signal-gap"},
                {"kind": "state_partial", "what": "state covers 60% of the frames"}]},
            "reader_notes": {"state_note": "Kinematic checks unavailable: the state has a gap."}, **extra}


def _board(tmp_path: Path, labels: dict) -> SimpleNamespace:
    board = tmp_path / "board"
    qa = board / "qa"
    qa.mkdir(parents=True)
    clips = tmp_path / "clips"
    clips.mkdir()
    for eid, label in labels.items():
        (qa / f"{eid}.json").write_text(json.dumps(label))
        (clips / f"{eid}.mp4").write_bytes(b"rgb " + eid.encode())
    return SimpleNamespace(qa=qa, board=board, clips=clips, compare=None, hands=None, keypoints=None, sensors=None,
                           dictionary_job=None, out=board / "static", build_id=None, force=False,
                           title="Pantheon DiverseUMI", public_base="https://cdn.example.org/b/", header=None,
                           publish_uploader_values=False, public_header=True, customer=True, audit_deny=None,
                           split_episodes=None, episode_order=None)


def _ids(tmp_path: Path, name: str, ids: list) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(ids))
    return p


def _site(args) -> Path:
    assert static.cmd_site(args) == 0
    return args.out / (args.out / "LATEST").read_text().strip()


def _list(site: Path) -> list:
    idx = json.loads((site / "data/index.json").read_text())
    return json.loads((site / "data/lists" / idx["lists"][0]).read_text())


def test_a_split_episode_card_shows_its_subtask_prompt(tmp_path):
    # the 10K's fob_47b204-d525af99-1657-16_p2 is one subtask of a longer recording: its card showed the whole
    # recording's two-part summary, and now shows the subtask it holds
    args = _board(tmp_path, {"episode_x_p2": _label("episode_x_p2"), "episode_y": _label("episode_y")})
    args.split_episodes = _ids(tmp_path, "split.json", ["episode_x_p2"])
    site = _site(args)
    split = json.loads((site / "data/ep/episode_x_p2.json").read_text())
    whole = json.loads((site / "data/ep/episode_y.json").read_text())
    assert split["episode_prompt"] == SUBTASK and whole["episode_prompt"] == SUMMARY
    assert {r["episode_id"]: r["episode_prompt"] for r in _list(site)} == {"episode_x_p2": SUBTASK, "episode_y": SUMMARY}


def test_several_subtasks_are_numbered_in_order():
    from board.public_board import subtask_prompt
    view = {"tasks": [{"task": "Open the case."}, {"task": "Lift the board."}, {"task": "Open the case."}]}
    assert subtask_prompt(view) == "(1) Open the case. (2) Lift the board."


def test_an_unsplit_recording_card_shows_the_session_summary_and_never_an_empty_one(tmp_path):
    # the hair dryer teardown fob_47b204-63430517-1809-14 is a whole recording: its card is the session summary, and a
    # public board with no summary for a card stops instead of publishing an empty card
    site = _site(_board(tmp_path / "a", {"episode_y": _label("episode_y")}))
    assert json.loads((site / "data/ep/episode_y.json").read_text())["episode_prompt"] == SUMMARY
    args = _board(tmp_path / "b", {"episode_y": _label("episode_y", summary="")})
    with pytest.raises(ValueError, match="no empty card"):
        static.cmd_site(args)
    args = _board(tmp_path / "c", {"episode_x_p2": _label("episode_x_p2", tasks=[])})
    args.split_episodes = _ids(tmp_path, "split.json", ["episode_x_p2"])
    with pytest.raises(ValueError, match="no empty card"):
        static.cmd_site(args)


def test_a_customer_board_keeps_no_data_issue_trace(tmp_path):
    site = _site(_board(tmp_path, {"episode_y": _label("episode_y")}))
    d = json.loads((site / "data/ep/episode_y.json").read_text())
    assert d["data_issues"] == [] and "state_note" not in d["reader_notes"]
    # a reader issue the page counts as a data issue goes; one it shows as a limit of how we read the recording stays
    assert [x["kind"] for x in d["dataset_checks"]["reader_issues"]] == ["signal_gap"]
    assert d["operator_mistakes"] and d["operator_mistakes"][0]["category"] == "failed_grasp"
    (row,) = _list(site)
    assert (row["n_issues"], row["n_minor_issues"], row["n_unrated_issues"], row["max_severity"], row["top_issue"]) \
        == (0, 0, 0, None, None)
    assert not [f for k in ("families", "minor_families", "unrated_families") for f in row[k] if f.startswith("d:")]
    text = "".join(p.read_text() for p in (site / "data").rglob("*.json"))
    assert "face_visible" not in text and "Kinematic checks unavailable" not in text and '"d:' not in text


def test_the_customer_audit_fails_the_build(tmp_path):
    args = _board(tmp_path / "a", {"episode_y": _label("episode_y", objects=[{"name": "rack in /vol/cuts/abc"}])})
    with pytest.raises(ValueError, match="must not"):
        static.cmd_site(args)
    args = _board(tmp_path / "b", {"episode_y": _label("episode_y", scene="filmed on hades7")})
    args.audit_deny = [r"hades\d*"]
    with pytest.raises(ValueError, match="must not"):
        static.cmd_site(args)
    assert not list((tmp_path / "b/board/static").glob("*-*"))      # nothing written under a build id
    # a flag word left in a field no rule empties still stops the build
    args = _board(tmp_path / "c", {"episode_y": _label("episode_y", performance_review="flagged face_visible at 3 s")})
    with pytest.raises(ValueError, match="face_visible"):
        static.cmd_site(args)


def test_the_public_header_is_one_title_with_no_tabs_or_repeated_name(tmp_path):
    site = _site(_board(tmp_path, {"episode_y": _label("episode_y")}))
    for page in ("index.html", "index.public.html"):
        html = (site / page).read_text()
        assert html.count('<header class="page-head"><h1>Pantheon DiverseUMI</h1></header>') == 1
        assert "<style>#coverage,#rd-name{display:none!important}</style></head>" in html
    args = _board(tmp_path / "x", {"episode_y": _label("episode_y")})
    args.header = tmp_path / "h.html"
    args.header.write_text("<header>site</header>")
    with pytest.raises(ValueError, match="leave out --header"):
        static.cmd_site(args)


def test_the_episode_order_reaches_the_index_the_lists_and_the_rail(tmp_path):
    names = ["episode_a", "episode_b", "episode_c", "episode_d"]
    args = _board(tmp_path, {n: _label(n) for n in names})
    args.episode_order = _ids(tmp_path, "order.json", ["episode_c", "episode_a"])
    site = _site(args)
    idx = json.loads((site / "data/index.json").read_text())
    assert [row[1] for row in idx["eps"]] == ["episode_c.json", "episode_a.json", "episode_b.json", "episode_d.json"]
    assert [r["episode_id"] for r in _list(site)] == ["episode_c", "episode_a", "episode_b", "episode_d"]
    args.episode_order = _ids(tmp_path, "typo.json", ["episode_z"])
    with pytest.raises(ValueError, match="does not have"):
        static.cmd_site(args)


def test_a_board_without_the_options_is_unchanged(tmp_path):
    args = _board(tmp_path, {"episode_y": _label("episode_y", summary="")})
    args.public_header = args.customer = False
    site = _site(args)
    d = json.loads((site / "data/ep/episode_y.json").read_text())
    assert d["episode_prompt"] == "" and d["data_issues"] and d["reader_notes"]["state_note"]
    assert "#coverage,#rd-name" not in (site / "index.html").read_text()
