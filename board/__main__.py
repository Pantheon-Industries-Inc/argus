"""The board: build it from runs, cut its clips, serve it, or write it as static files.

    python -m board build BOARD                                   BOARD/manifest.json -> BOARD/qa, BOARD/BUILT.json
    python -m board follow BOARD                                  each new label of a run in progress -> BOARD/qa
    python -m board materials BOARD                               each kind of object tagged rigid or deformable
    python -m board verbs BOARD                                   the main verb of each task sentence
    python -m board clips --episodes EPISODES --out CLIPS         browser clips of every camera
    python -m board serve --board BOARD --clips CLIPS             the board at http://localhost:8896
    python -m board static site --board BOARD --clips CLIPS       the same board as plain files for a CDN
    python -m board to_board --in-dir RUN/out --out-dir OUT --dataset NAME   one run's outputs as board files
    python -m board hands build|verify ...                        the hand pose overlay files (board build runs it)

Each command takes --help.
"""
import importlib
import sys

COMMANDS = {"build": "board.build", "follow": "board.follow", "materials": "board.materials", "verbs": "board.verbs",
            "clips": "board.clips", "serve": "board.serve", "static": "board.static",
            "to_board": "board.to_board", "hands": "board.hands"}
if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
    print(__doc__)
    raise SystemExit(0 if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") else 2)
cmd = sys.argv.pop(1)
sys.argv[0] = f"python -m board {cmd}"
raise SystemExit(importlib.import_module(COMMANDS[cmd]).main())
