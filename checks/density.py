"""How densely a label annotates its footage, and a paired comparison of two sets of labels on the same footage.

Every metric is per minute of footage or a share of the footage, except the step lengths and the shares of steps. A
split left and right pair adds two items but only one moment, so a drop cannot hide behind per-side steps, and a both
step counts for each side. A rise in LOWER_IS_DENSER counts against density, so a merged step that hides two actions
counts as the loss it is. compare resamples episodes, never frames, and calls a drop only when the whole interval lies
on the side of less density. Run it over labels of the same episodes with the same model and settings.
tests/test_density.py holds the timeline's rules to such a measurement; it has no command line."""
from __future__ import annotations

import hashlib
import math
import random
import re
import statistics

MOMENT_TOLERANCE_S = 0.5          # half the 1 fps sampling period: two starts closer than this are one moment
TEXT_SKIP = {'timeline_columns', 'source_piece', 'part_assessments', 'cut_seams', 'cut_continuation', 'progress_scope'}
LOWER_IS_DENSER = {'median_step_s', 'median_active_step_s'}
# The metrics a change must not lower. Rarer kinds of key event and the other counts of a few per minute move too much
# between two runs of one commit to judge on 30 episodes, so they are measured but not guarded.
GUARDED = ['items_per_min', 'moments_per_min', 'left_items_per_min', 'right_items_per_min', 'coverage_left',
           'coverage_right', 'key_events_per_min', 'objects_per_min', 'issues_nonface_per_min', 'text_chars_per_min',
           'median_step_s', 'median_active_step_s']


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def steps_of(labels: dict) -> list:
    cols = labels.get('timeline_columns') or []
    rows = []
    for x in labels.get('timeline') or []:
        if isinstance(x, list):
            rows.append(dict(zip(cols, x)))
        elif isinstance(x, dict):
            rows.append(x)
    return rows


def _norm(s) -> str:
    return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip() if isinstance(s, str) else ''


def _union(intervals, limit: float) -> float:
    total, end = 0.0, None
    for a, b in sorted(intervals):
        a, b = max(a, 0.0), min(b, limit)
        if b <= a:
            continue
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def _chars(value) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(_chars(v) for v in value)
    if isinstance(value, dict):
        return sum(_chars(v) for k, v in value.items() if k not in TEXT_SKIP)
    return 0


def _occurrences(issue: dict) -> int:
    occurrences = issue.get('occurrences')
    return len(occurrences) if isinstance(occurrences, list) and occurrences else 1


def _is_face(issue: dict) -> bool:
    words = re.split(r'[^a-z0-9]+', re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', str(issue.get('category') or '')).lower())
    return bool({'face', 'faces'}.intersection(words))


def metrics(labels: dict, footage_s: float) -> dict:
    """The density of one label over footage_s seconds of footage."""
    if footage_s <= 0:
        raise ValueError('footage must be positive')
    minutes = footage_s / 60
    rows = steps_of(labels)
    timed = [s for s in rows if _num(s.get('start_s')) is not None and _num(s.get('end_s')) is not None
             and s['end_s'] >= s['start_s']]
    span = lambda s: s['end_s'] - s['start_s']
    arm = lambda s: str(s.get('arm') or '').lower()
    idle = lambda s: str(s.get('contribution') or '').lower() == 'idle'
    side = lambda name: [s for s in timed if arm(s) in (name, 'both')]
    moments, anchor = 0, None
    for t in sorted(s['start_s'] for s in timed):
        if anchor is None or t - anchor > MOMENT_TOLERANCE_S:
            moments += 1
            anchor = t
    lengths = [span(s) for s in timed if span(s) > 0] or [0.0]
    active = [span(s) for s in timed if span(s) > 0 and not idle(s)] or [0.0]
    events = [e for e in labels.get('key_events') or [] if isinstance(e, dict)]
    kind = lambda word: sum(word in str(e.get('kind') or '').lower() for e in events)
    issues = [i for i in labels.get('data_issues') or [] if isinstance(i, dict)]
    scene = labels.get('scene') if isinstance(labels.get('scene'), dict) else {}
    objects = scene.get('objects') if isinstance(scene.get('objects'), list) else []
    return {
        'items_per_min': len(rows) / minutes,
        'moments_per_min': moments / minutes,
        'left_items_per_min': len(side('left')) / minutes,
        'right_items_per_min': len(side('right')) / minutes,
        'both_steps_per_min': sum(arm(s) == 'both' for s in rows) / minutes,
        'coverage_left': _union([(s['start_s'], s['end_s']) for s in side('left')], footage_s) / footage_s,
        'coverage_right': _union([(s['start_s'], s['end_s']) for s in side('right')], footage_s) / footage_s,
        'coverage_any': _union([(s['start_s'], s['end_s']) for s in timed], footage_s) / footage_s,
        'key_events_per_min': len(events) / minutes,
        'contact_per_min': kind('contact') / minutes,
        'grasp_per_min': kind('grasp') / minutes,
        'release_per_min': kind('release') / minutes,
        'subgoals_per_min': kind('subgoal') / minutes,
        'objects_per_min': len(objects) / minutes,
        'issues_per_min': sum(_occurrences(i) for i in issues) / minutes,
        'issues_nonface_per_min': sum(_occurrences(i) for i in issues if not _is_face(i)) / minutes,
        'tasks_per_min': len(labels.get('tasks') or []) / minutes,
        'text_chars_per_min': _chars(labels) / minutes,
        'distinct_actions_per_min': len({_norm(s.get('action')) for s in rows} - {''}) / minutes,
        'shortest_step_s': min(lengths),
        'median_step_s': statistics.median(lengths),
        'median_active_step_s': statistics.median(active),
    }


def compare(before: dict, after: dict, *, n_boot: int = 20000, seed: int = 20261104, level: float = 0.95) -> dict:
    """before and after are {episode: [metrics of each run]} over the same episodes. For each metric: each side's mean over
    episodes, the paired change, an interval from resampling episodes, and a verdict in terms of density."""
    episodes = sorted(before)
    if set(before) != set(after) or not episodes:
        raise ValueError(f'the two sides must hold the same episodes: {sorted(set(before) ^ set(after))[:5]}')
    rnd = random.Random(seed)
    draws = [[rnd.randrange(len(episodes)) for _ in episodes] for _ in range(n_boot)]
    tail = (1 - level) / 2
    out = {}
    for name in before[episodes[0]][0]:
        a = [statistics.fmean(m[name] for m in before[e]) for e in episodes]
        b = [statistics.fmean(m[name] for m in after[e]) for e in episodes]
        changes = sorted(sum(b[i] - a[i] for i in draw) / len(draw) for draw in draws)
        lo, hi = changes[int(tail * n_boot)], changes[min(n_boot - 1, int((1 - tail) * n_boot))]
        worse, better = (lo > 0, hi < 0) if name in LOWER_IS_DENSER else (hi < 0, lo > 0)
        out[name] = {'before': statistics.fmean(a), 'after': statistics.fmean(b), 'lo': lo, 'hi': hi,
                     'verdict': 'drop' if worse else 'rise' if better else 'holds'}
    return out


def drops(result: dict) -> list:
    """The guarded metrics a comparison finds lower."""
    return [name for name in GUARDED if name in result and result[name]['verdict'] == 'drop']


def record_drops(record: dict, *, n_boot: int = 5000) -> list:
    """The guarded metrics a stored measurement finds lower. A record holds each episode's guarded metrics on each side."""
    side = lambda key: {str(i): [row[key]] for i, row in enumerate(record['episodes'])}
    return drops(compare(side('before'), side('after'), n_boot=n_boot))


def prompt_fingerprint(prompts) -> str:
    """Changes when the rules that set the timeline's granularity change, on the rig the measurement ran on."""
    return hashlib.sha256((prompts.LEAN_NOTE + prompts.two_actors_note('gripper')).encode()).hexdigest()
