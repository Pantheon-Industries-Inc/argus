"""How long a head-camera label has the wearer's hands out of frame, as three separate figures: both hands out, only
the left hand out and only the right hand out, in seconds and as a percent of the episode. The model records
left_hand_visible and right_hand_visible on every timeline step (label/prompts.py), and label/harness.py,
label/pieces.py stitch_run and label/reparse.py write the figures into each result as hands_out; label/finalize.py sums
them over a run into run.json. A label without the per-hand fields (other rigs, and Ego labels written before they
existed) gets no hands_out, because its hands_visible only ever said whether both hands were out."""
from __future__ import annotations

import json
from pathlib import Path

KEYS = ('left_hand_visible', 'right_hand_visible')
FIGURES = ('both', 'left_only', 'right_only')


def _side(row: dict, key: str):
    value = row.get(key)
    return value if type(value) is bool else None


def hands_out(labels, episode_s: float | None) -> dict | None:
    """The three figures for one label's timeline, or None when no step records either hand. Each moment counts once
    across overlapping steps (a left-arm row and a right-arm row at the same time). A moment is one-hand only when the
    other hand is seen in it, so a moment whose other hand is unknown (null) counts in no figure. Spans are clipped to
    the episode when its length is known, and the percents are None when it is not."""
    rows = [r for r in (labels or {}).get('timeline') or [] if isinstance(r, dict)] if isinstance(labels, dict) else []
    if not any(key in r for r in rows for key in KEYS):
        return None
    end = float(episode_s) if isinstance(episode_s, (int, float)) and episode_s > 0 else None
    steps = []
    for r in rows:
        try:
            a, b = float(r['start_s']), float(r['end_s'])
        except (KeyError, TypeError, ValueError):
            continue
        a, b = max(a, 0.0), (min(b, end) if end is not None else b)
        if b > a:
            steps.append((a, b, _side(r, KEYS[0]), _side(r, KEYS[1])))
    seconds = dict.fromkeys(FIGURES, 0.0)
    cuts = sorted({t for a, b, _, _ in steps for t in (a, b)})
    for a, b in zip(cuts, cuts[1:]):
        now = [(left, right) for s0, s1, left, right in steps if s0 <= a and s1 >= b]
        left_out, right_out = (any(st[i] is False for st in now) for i in (0, 1))
        left_seen, right_seen = (any(st[i] is True for st in now) for i in (0, 1))
        if left_out and right_out:
            seconds['both'] += b - a
        elif left_out and right_seen:
            seconds['left_only'] += b - a
        elif right_out and left_seen:
            seconds['right_only'] += b - a
    out = {'episode_s': round(end, 3) if end is not None else None}
    for name in FIGURES:
        out[f'{name}_s'] = round(seconds[name], 3)
        out[f'{name}_pct'] = round(100 * seconds[name] / end, 3) if end else None
    return out


def episode_length(ep_dir) -> float | None:
    """The prepared episode's length (prepare/frame_rate.py episode_seconds), None when its context cannot be read."""
    from prepare.frame_rate import episode_seconds
    if not ep_dir:
        return None
    try:
        ctx = json.loads((Path(ep_dir) / 'context.json').read_text())
        return float(episode_seconds(ctx, ep_dir=Path(ep_dir)))
    except (OSError, TypeError, ValueError):
        return None


def attach(result: dict, episode_s: float | None) -> dict:
    """result with hands_out set from its labels, or removed when its labels record no hand."""
    figures = hands_out(result.get('labels'), episode_s)
    if figures is None:
        result.pop('hands_out', None)
    else:
        result['hands_out'] = figures
    return result


def run_totals(records) -> dict | None:
    """The three figures summed over a run's results that carry hands_out, in hours and as a percent of those
    episodes' hours, with how many episodes and hours they cover. None when no result carries them."""
    figures = [r['hands_out'] for r in records if isinstance(r, dict) and isinstance(r.get('hands_out'), dict)
               and isinstance(r['hands_out'].get('episode_s'), (int, float))]
    if not figures:
        return None
    total = sum(f['episode_s'] for f in figures)
    out = {'episodes': len(figures), 'hours': round(total / 3600, 4)}
    for name in FIGURES:
        s = sum(float(f.get(f'{name}_s') or 0) for f in figures)
        out[f'{name}_hours'] = round(s / 3600, 4)
        out[f'{name}_pct'] = round(100 * s / total, 3) if total else None
    return out
