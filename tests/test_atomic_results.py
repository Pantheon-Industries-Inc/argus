"""Readers never see a result file while its JSON is being written."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from board.to_board import dumps
from checks import timebase
from label import atomic, reparse, pieces, run


def test_an_atomic_board_write_keeps_normalization_and_the_old_file_on_serialization_error(tmp_path):
    p = tmp_path / "BUILT.json"
    p.write_text('{"old":true}')
    atomic.write_atomic(p, {"value": float("nan")}, serializer=dumps, indent=None)
    assert p.read_text() == '{"value": null}'
    with pytest.raises(TypeError):
        atomic.write_atomic(p, {"value": object()}, serializer=dumps)
    assert p.read_text() == '{"value": null}'


def test_timebase_apply_obeys_the_shared_atomic_write_policy(tmp_path, monkeypatch):
    ep = tmp_path / "episode_000000"
    ep.mkdir()
    p = ep / "context.json"
    p.write_text('{"old":true}')
    tb = tmp_path / "timebase.csv"
    tb.write_text("episode_index,neighbour_lag_frames\n0,4\n")

    def refused(*args, **kwargs):
        raise OSError("atomic write refused")
    monkeypatch.setattr(timebase, "write_atomic", refused)
    with pytest.raises(OSError, match="atomic write refused"):
        timebase.cmd_apply(SimpleNamespace(roots=[tmp_path], timebase=tb, labels=None))
    assert p.read_text() == '{"old":true}'


def _deny_direct_json(monkeypatch, names):
    original = Path.write_text

    def write(p, *args, **kwargs):
        if p.name in names:
            raise OSError("the result was written directly")
        return original(p, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", write)


def test_run_metadata_is_replaced_after_the_dry_run(tmp_path, monkeypatch):
    from test_label import _slice, _fake_harness
    sl = _slice(tmp_path)
    monkeypatch.setattr(run, "commit", lambda: ("abc1234", False))
    _fake_harness(monkeypatch, [], cost=0)
    _deny_direct_json(monkeypatch, {"run.json"})
    import sys
    monkeypatch.setattr(sys, "argv", ["label", "--dataset", "d", "--episodes", str(sl), "--kind", "dry",
                                     "--runs", str(tmp_path / "runs")])
    run.main()
    meta = json.loads(next((tmp_path / "runs").glob("d/*/run.json")).read_text())
    assert meta["status"] == "done" and meta["episodes_done"] == 1


def test_reparse_metadata_is_replaced_after_reading_stored_replies(tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    (tmp_path / "run.json").write_text('{"run_id":"r"}')
    monkeypatch.setattr(reparse, "commit", lambda: ("abc1234", False))
    _deny_direct_json(monkeypatch, {"run.json"})
    reparse.reparse(tmp_path)
    assert json.loads((tmp_path / "run.json").read_text())["reparsed"][0]["parsed_now"] == []


def test_reparse_obeys_the_shared_atomic_write_policy(tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    p = tmp_path / "run.json"
    p.write_text('{"run_id":"r"}')
    monkeypatch.setattr(reparse, "commit", lambda: ("abc1234", False))

    def refused(*args, **kwargs):
        raise OSError("atomic write refused")
    monkeypatch.setattr(reparse, "write_atomic", refused, raising=False)
    with pytest.raises(OSError, match="atomic write refused"):
        reparse.reparse(tmp_path)
    assert p.read_text() == '{"run_id":"r"}'


def test_board_metadata_is_replaced_with_board_json_normalization(tmp_path, monkeypatch):
    from test_board_sensors import _episode, _board
    from board import build
    ep = _episode(tmp_path / "eps")
    board = _board(tmp_path, ep)
    _deny_direct_json(monkeypatch, {"BUILT.json"})
    out = build.build(board)
    assert json.loads((board / "BUILT.json").read_text()) == out


def test_piece_sources_and_stitched_results_are_replaced_without_encoding_instruction_text(tmp_path, monkeypatch):
    from test_pieces import _three_parts, _part_result
    job, src, parts = _three_parts(tmp_path, monkeypatch)
    _deny_direct_json(monkeypatch, {"sources.json", "depth.json"})
    again = pieces.write_pieces(src, job / "pieces_again")
    assert again and (again[0] / "instruction.txt").read_bytes() == b"\n"
    for p in parts:
        (job / "run" / "out" / f"{p.name}.json").write_text(json.dumps(_part_result(p, "reach")))
    _deny_direct_json(monkeypatch, {f"{src.name}.json"})
    final = tmp_path / "final"
    pieces.stitch_run(job, src.parent, {src.name: [p.name for p in parts]}, final)
    assert json.loads((final / f"{src.name}.json").read_text())["parse_ok"]
