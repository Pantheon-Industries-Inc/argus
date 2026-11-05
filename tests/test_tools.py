"""prepare/tools.py: the one place the repo finds ffmpeg and ffprobe. board/hands.py read $FFPROBE and exited without
it, board/serve.py and board/static.py took the ffprobe beside the server's ffmpeg and fell back to PATH, and
board/clips.py looked beside the ffmpeg it was handed, so one machine could probe a clip with one install and cut it
with another."""
from __future__ import annotations

from pathlib import Path

import pytest

from prepare import tools


def _program(folder: Path, name: str) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_text("#!/bin/sh\nexit 0\n")
    p.chmod(0o755)
    return str(p)


@pytest.fixture
def bare(tmp_path, monkeypatch):
    """A machine with neither tool anywhere: nothing on PATH, no variables, no install places."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("FFMPEG", raising=False)
    monkeypatch.delenv("FFPROBE", raising=False)
    monkeypatch.setattr(tools, "INSTALL_DIRS", ())
    return tmp_path


def test_ffprobe_is_the_one_beside_the_ffmpeg_in_use_before_the_one_on_path(bare, monkeypatch):
    ffmpeg = _program(bare / "pinned", "ffmpeg")
    probe = _program(bare / "pinned", "ffprobe")
    _program(bare / "path", "ffprobe")
    monkeypatch.setenv("PATH", str(bare / "path"))
    monkeypatch.setenv("FFMPEG", ffmpeg)
    assert (tools.ffmpeg(), tools.ffprobe()) == (ffmpeg, probe)
    monkeypatch.setenv("FFPROBE", str(bare / "path" / "ffprobe"))
    assert tools.ffprobe() == str(bare / "path" / "ffprobe")      # $FFPROBE before anything else


def test_a_tool_on_path_or_in_an_install_place_is_found(bare, monkeypatch):
    _program(bare / "path", "ffmpeg")
    monkeypatch.setenv("PATH", str(bare / "path"))
    assert tools.ffmpeg() == str(bare / "path" / "ffmpeg")
    monkeypatch.setattr(tools, "INSTALL_DIRS", (str(bare / "brew"),))
    probe = _program(bare / "brew", "ffprobe")
    assert tools.ffprobe() == probe          # a server started outside a login shell still finds it


def test_a_missing_tool_is_a_clear_error_and_never_a_silent_fallback(bare, monkeypatch):
    with pytest.raises(tools.Missing, match="ffprobe not found: put it on PATH or set FFPROBE"):
        tools.ffprobe()
    with pytest.raises(tools.Missing, match="ffmpeg not found: put it on PATH or set FFMPEG"):
        tools.ffmpeg()
    assert tools.ffmpeg_or_none() is None and tools.ffprobe_or_none() is None
    _program(bare / "path", "ffprobe")
    monkeypatch.setenv("PATH", str(bare / "path"))
    monkeypatch.setenv("FFPROBE", str(bare / "no" / "ffprobe"))
    with pytest.raises(tools.Missing, match=r"\$FFPROBE is .*, which is not a program"):
        tools.ffprobe()
    assert isinstance(tools.Missing("ffprobe", "x"), FileNotFoundError)    # the machine's, board/clips.py machine_stopped


def _ffmpeg_saying(folder: Path, version_line: str) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / "ffmpeg"
    p.write_text(f"#!/bin/sh\necho '{version_line}'\necho 'built with gcc 12'\n")
    p.chmod(0o755)
    return str(p)


# what real installs print: Debian 12, a plain release, a tagged build, Ubuntu 24.04, Homebrew, and a git build
@pytest.mark.parametrize("line,found,too_old", [
    ("ffmpeg version 5.1.6-0+deb12u1 Copyright (c) 2000-2024 the FFmpeg developers", (5, 1), True),
    ("ffmpeg version 6.0 Copyright (c) 2000-2023 the FFmpeg developers", (6, 0), True),
    ("ffmpeg version n6.0.1 Copyright (c) 2000-2023 the FFmpeg developers", (6, 0), True),
    ("ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers", (4, 4), True),
    ("ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023 the FFmpeg developers", (6, 1), False),
    ("ffmpeg version 8.1.2 Copyright (c) 2000-2026 the FFmpeg developers", (8, 1), False),
    ("ffmpeg version 10.0 Copyright (c) 2000-2030 the FFmpeg developers", (10, 0), False),
    ("ffmpeg version N-113007-g8d24a28d06 Copyright (c) 2000-2023 the FFmpeg developers", None, False),
])
def test_clip_cutting_refuses_an_ffmpeg_too_old_for_demux_time_base(tmp_path, line, found, too_old):
    path = _ffmpeg_saying(tmp_path, line)
    assert tools.ffmpeg_version(path) == found
    if too_old:
        with pytest.raises(tools.TooOld) as stopped:
            tools.ffmpeg_for_clips(path)
        said = stopped.value.strerror
        assert path in said and f"ffmpeg {found[0]}.{found[1]} " in said and "6.1 or newer" in said
    else:
        assert tools.ffmpeg_for_clips(path) == path


def test_an_ffmpeg_upgraded_in_place_is_read_again(tmp_path):
    path = _ffmpeg_saying(tmp_path, "ffmpeg version 5.1.6 Copyright")
    assert tools.ffmpeg_version(path) == (5, 1)
    _ffmpeg_saying(tmp_path, "ffmpeg version 7.1.1 Copyright")
    assert tools.ffmpeg_version(path) == (7, 1)


def test_too_old_is_the_machines_fault_and_finding_ffmpeg_does_not_check_it(bare, monkeypatch):
    from board.clips import machine_stopped
    old = _ffmpeg_saying(bare / "old", "ffmpeg version 5.1.6 Copyright")
    monkeypatch.setenv("FFMPEG", old)
    with pytest.raises(tools.Missing):
        tools.ffmpeg_for_clips()
    assert machine_stopped(tools.TooOld(old, (5, 1)))
    # frames and probes work on 5.1, and neither passes the option
    assert tools.ffmpeg() == old and tools.ffmpeg_or_none() == old


def test_each_command_that_cuts_clips_stops_on_a_too_old_ffmpeg(bare, monkeypatch, capsys):
    """Each entry point reaches the check. board clips and board static pass -enc_time_base demux, and python -m review
    runs board clips as a later step, so it must stop before its earlier steps."""
    import sys
    from board import clips, static
    import review.__main__ as review
    old = _ffmpeg_saying(bare / "old", "ffmpeg version 5.1.6-0+deb12u1 Copyright")
    monkeypatch.setenv("FFMPEG", old)
    monkeypatch.setenv("FFPROBE", _program(bare / "old", "ffprobe"))
    monkeypatch.setattr(static, "FFMPEG", old)
    (bare / "board" / "qa").mkdir(parents=True)
    runs = {
        "clips": (clips.main, ["clips", "--episodes", str(bare / "eps"), "--out", str(bare / "out")]),
        "static": (static.main, ["static", "media", "--board", str(bare / "board"), "--clips", str(bare / "clips")]),
        "review": (review.main, ["review", "--data", str(bare), "--rig", "teleop_arms", "--out", str(bare / "job")]),
    }
    for name, (main, argv) in runs.items():
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as stopped:
            main()
        assert "6.1 or newer" in str(stopped.value.code), name
    assert not (bare / "job").exists()      # review stopped before it made anything


def test_stated_minimum_matches_the_checked_one_and_the_option_needing_it_is_still_passed():
    from board import clips
    root = Path(__file__).resolve().parents[1]
    need = ".".join(map(str, tools.FFMPEG_MINIMUM))
    assert f"{need} or newer" in (root / "README.md").read_text()
    args = clips.video_args(640, 480, True, 1)
    assert args[args.index("-enc_time_base") + 1] == "demux"


def test_every_caller_runs_the_same_ffprobe():
    """The callers that probe a clip all take it from prepare/tools.py, with no lookup of their own left."""
    root = Path(__file__).resolve().parents[1]
    own = []
    for folder in ("board", "checks", "label", "prepare"):
        for p in (root / folder).rglob("*.py"):
            if p.name == "tools.py" or "hand_pose" in p.parts or "vendor" in p.parts:
                continue
            text = p.read_text()
            for needle in ('which("ffprobe")', "which('ffprobe')", 'which("ffmpeg")', "which('ffmpeg')",
                           '"FFPROBE"', '"FFMPEG"', "['ffprobe'", "['ffmpeg'", '["ffprobe"', '["ffmpeg"',
                           "\n        'ffprobe',", "\n        'ffmpeg',"):
                if needle in text:
                    own.append(f"{p.relative_to(root)}: {needle.strip()}")
    assert not own, own
