"""Camera slots from what each camera records, once its frames are written.

A reader first slots cameras by name (prepare/formats.py assign_views). Once their frames are written, the same camera
decision the labeller uses (label/camera_class.py camera_class) reads each camera with the recording's other channels
and its own pictures. A camera classed as a tactile sensor's own image (a DIGIT 360 image beside its pressure topic)
never fills a scene or gripper slot, so it never becomes the anchor whose clock times everything else. The anchor is
then chosen from the cameras that record the scene, by the evidence that already decides a camera's clock: a camera
whose recorded capture times need no assumed placement (prepare/camera_clock.py presentation_clock) comes first, then
the one with more frames, then the names' order. A recording whose only cameras are tactile keeps them, and says so.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_FRAMES = 8
ONLY_TACTILE = 'only_tactile_cameras'


def sample_frames(path, pts=None, count: int = SAMPLE_FRAMES) -> dict:
    """{frame index: picture} for up to count frames spread over a written camera file, or {} when it cannot be read."""
    from label import frames as mf, lens
    try:
        if pts is None:
            from prepare.formats import probe
            pts = probe(Path(path))['pts']
        n = len(pts)
        if not n:
            return {}
        return mf.extract_frames(path, 0.0, n, lens.spread(n, count), pts=pts)
    except Exception:  # noqa: BLE001 - an unreadable file only leaves the camera's class undecided
        return {}


def classes(vmap: dict, paths: dict, signal_names, unshown_names=(), pts=None) -> dict:
    """{view: camera_class decision} for the cameras vmap {view: name}, written at paths {view: file}, beside the
    recording's other channels signal_names and its cameras not shown unshown_names."""
    from label.camera_class import camera_class
    ctx = {'cameras': {v: {'key': name, 'name': name} for v, name in vmap.items()},
           'signals': [{'name': str(name)} for name in signal_names],
           'unshown_cameras': [{'name': str(name)} for name in unshown_names]}
    return {v: camera_class(ctx, v, sample_frames(paths[v], (pts or {}).get(v)) if v in paths else None)
            for v in vmap}


def clock_rank(times) -> int:
    """0 when a camera's recorded capture times order every frame as recorded, 1 when they need an assumed placement
    (ties or backwards steps, prepare/camera_clock.py presentation_clock), 2 when it has none."""
    from prepare.camera_clock import presentation_clock
    if times is None or not len(times):
        return 2
    return 0 if presentation_clock(np.asarray(times, dtype=np.float64))[1] is None else 1


def class_views(vmap: dict, paths: dict, rig: str, signal_names, unshown_names=(), frames: dict | None = None,
                times: dict | None = None, pts: dict | None = None) -> tuple[dict, dict]:
    """(new vmap {view: name}, record). The record holds every camera's class and basis, which view each camera moved
    from, and an only_tactile_cameras note when no camera records the scene. frames {name: frame count} and times
    {name: recorded capture seconds} rank the scene cameras for the anchor slot. A camera never leaves the episode:
    one the name rules leave unslotted is added as a further extra view."""
    from label import episode as me
    from prepare.formats import assign_views
    decided = classes(vmap, paths, signal_names, unshown_names, pts)
    by_name = {vmap[v]: d for v, d in decided.items()}
    record = {'camera_classes': {name: {'class': d['class'], 'known': d['known'], 'basis': d['basis']}
                                 for name, d in by_name.items()}}
    tactile = {name for name, d in by_name.items() if d['tactile']}
    slotted = [vmap[v] for v in me.VIEW_ORDER if v in vmap]
    if not tactile & set(slotted):
        return dict(vmap), record
    if tactile == set(vmap.values()):
        record[ONLY_TACTILE] = sorted(tactile)
        return dict(vmap), record
    names = [vmap[v] for v in me.order_views(vmap)]

    def prefer(name):
        return (clock_rank((times or {}).get(name)), -int((frames or {}).get(name) or 0))
    new, _ = assign_views(names, rig, tactile=tactile, prefer=prefer)
    for name in names:
        if name not in new.values():
            new[f'extra{sum(1 for k in new if k.startswith("extra")) + 1}'] = name
    moved = {old: next(v for v, n in new.items() if n == name) for old, name in vmap.items()}
    record['moved'] = {old: view for old, view in moved.items() if old != view}
    return new, record


def note(extra: dict, record: dict) -> None:
    """Keep the camera decisions with the episode's source facts, and name a recording with only tactile cameras."""
    from prepare.formats import add_issue
    extra.setdefault('source', {})['camera_classes'] = record['camera_classes']
    if record.get('moved'):
        extra['source']['camera_slots_moved'] = record['moved']
    if record.get(ONLY_TACTILE):
        add_issue(extra, ONLY_TACTILE, 'Every camera of this recording is a tactile sensor\'s own image ('
                  + ', '.join(record[ONLY_TACTILE]) + '); none records the scene, so the episode is timed by a '
                  'tactile camera and the visual annotation has no scene view.')
