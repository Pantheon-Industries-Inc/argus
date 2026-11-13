"""Apply source-bound task-purpose reviews to a derived copy of paid annotations.

python -m label.task_scope --input RUN/out --reviews reviews.json --out DERIVED/out

No task is classified by its name, position, duration or outcome here. A review must identify
the source labels and exact task. Human decisions override machine decisions; uncertain and
unreviewed tasks stay in the work list. The original reply and footage are never rewritten.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from board.label_projection import recording_completion
from label.atomic import write_atomic


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def terminal_task(tasks: list, index: int, clip_end_s: float, *, sample_period_s: float) -> bool:
    """A boundary prior for review, never sufficient evidence of capture equipment or purpose."""
    end = tasks[index].get('end_s') if type(index) is int and 0 <= index < len(tasks) else None
    return (index == len(tasks) - 1 and type(end) in (int, float) and math.isfinite(end)
            and sample_period_s > 0 and abs(clip_end_s - end) <= sample_period_s)


def apply_review(result: dict, review: dict, *, rig: str) -> dict:
    if rig != 'ego_head' or (result.get('annotation_policy') or {}).get('rig') != rig:
        raise ValueError('Task-purpose review currently supports human Ego recordings only')
    labels = result.get('labels')
    if not isinstance(labels, dict) or not isinstance(labels.get('tasks'), list):
        raise ValueError('Review requires a saved task list')
    if review.get('version') != 1:
        raise ValueError('Unknown task-purpose review version')
    previous = result.get('task_scope_review') or {}
    if (previous.get('review_sha256') == digest(review)
            and previous.get('projected_result_sha256') == digest({k: v for k, v in result.items()
                                                                  if k != 'task_scope_review'})):
        return copy.deepcopy(result)
    if review.get('result_sha256') != digest(result):
        raise ValueError('Review belongs to a different source result or request')
    if review.get('labels_sha256') != digest(labels):
        raise ValueError('Review belongs to different source labels')
    tasks = labels['tasks']
    effective = {}
    for source in ('machine', 'human'):
        rows = review.get(source, [])
        if not isinstance(rows, list):
            raise ValueError('Review decisions must be lists')
        seen = set()
        for row in rows:
            i = row.get('task_index') if isinstance(row, dict) else None
            if type(i) is not int or not 0 <= i < len(tasks) or i in seen:
                raise ValueError('Invalid or duplicate reviewed task index')
            seen.add(i)
            if row.get('task_sha256') != digest(tasks[i]):
                raise ValueError('Reviewed task identity changed')
            if row.get('kind') not in ('work', 'recording_control', 'uncertain'):
                raise ValueError('Unknown task purpose')
            if row.get('target') not in ('capture_rig', 'task_object', 'unknown'):
                raise ValueError('Unknown reviewed target')
            if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('reason', 'evidence')):
                raise ValueError('A decision requires its reason and evidence')
            if row['kind'] == 'recording_control' and row['target'] != 'capture_rig':
                raise ValueError('Only a confirmed operation of this recording rig can be excluded')
            related = row.get('related', {})
            if not isinstance(related, dict) or set(related) - {'timeline', 'key_events', 'operator_mistakes'}:
                raise ValueError('Unknown related annotation field')
            for field, indices in related.items():
                entries = labels.get(field, [])
                if (not isinstance(indices, list) or not isinstance(entries, list)
                        or any(type(j) is not int or not 0 <= j < len(entries) for j in indices)
                        or len(set(indices)) != len(indices)):
                    raise ValueError('Invalid related annotation index')
                for j in indices:
                    entry = entries[j]
                    if not isinstance(entry, dict):
                        raise ValueError('Related annotation is not an object')
                    a = entry.get('start_s') if field == 'timeline' else entry.get('t_s')
                    b = entry.get('end_s', a)
                    span = (a, b, tasks[i].get('start_s'), tasks[i].get('end_s'))
                    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in span)
                            or not tasks[i]['start_s'] <= a <= b <= tasks[i]['end_s']):
                        raise ValueError('Related annotation exceeds the reviewed capture task span')
            effective[i] = {**copy.deepcopy(row), 'source': source}
    out = copy.deepcopy(result)
    excluded = [i for i, row in effective.items() if row['kind'] == 'recording_control']
    if excluded:
        if not isinstance(out['labels'].get('recording_activities', []), list):
            raise ValueError('Existing recording activities are not a list')
        out['labels']['tasks'] = [t for i, t in enumerate(out['labels']['tasks']) if i not in excluded]
        activities = out['labels'].setdefault('recording_activities', [])
        related_owners = {}
        for i in sorted(excluded):
            task, decision = tasks[i], effective[i]
            source_annotations = {}
            for field, indices in decision.get('related', {}).items():
                for j in indices:
                    if (field, j) in related_owners:
                        raise ValueError('One annotation assigned to several recording activities')
                    related_owners[field, j] = i
                source_annotations[field] = [{'index': j, 'row': copy.deepcopy(labels[field][j])} for j in indices]
            activities.append({'start_s': task.get('start_s'), 'end_s': task.get('end_s'),
                'activity': task.get('task'), 'target': 'capture_rig',
                'reason': decision['reason'], 'evidence': decision['evidence'],
                'source_task': copy.deepcopy(task), 'scope_review': decision, 'source_annotations': source_annotations})
        for field in ('key_events', 'operator_mistakes'):
            if field in out['labels']:
                out['labels'][field] = [row for j, row in enumerate(out['labels'][field])
                                       if (field, j) not in related_owners]
        for j, row in enumerate(out['labels'].get('timeline', [])):
            if ('timeline', j) in related_owners:
                row['activity_scope'] = 'recording_control'
        for source_key, label_key in (('work_summary', 'task_summary'), ('work_review', 'performance_review')):
            if source_key in review:
                if not isinstance(review[source_key], str) or not review[source_key].strip():
                    raise ValueError('Reviewed work text must be nonempty')
                out['labels'][label_key] = review[source_key]
        out['task_scope_version'] = 1
        # Preserve an explicitly supplied goal judgment. Recompute only a verdict derived from the tasks.
        completion = out['labels'].get('completion') or {}
        if not completion or completion.get('basis') in ('tasks', 'one_task'):
            out['labels']['completion'] = recording_completion(out['labels']['tasks'])
    out['task_scope_review'] = {'version': 1, 'source_result_sha256': review['result_sha256'],
        'source_labels_sha256': review['labels_sha256'], 'review_sha256': digest(review),
        'projected_result_sha256': digest(out),
        'tasks_before': len(tasks), 'tasks_after': len(out['labels']['tasks']),
        'decisions': [effective[i] for i in sorted(effective)],
        'reviews': copy.deepcopy({key: review.get(key, []) for key in ('machine', 'human')}),
        'original_text': {k: labels.get(k) for k in ('task_summary', 'performance_review')}}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--reviews', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    source, target = a.input.resolve(), a.out.resolve()
    if source == target or source in target.parents or target in source.parents:
        p.error('Use a separate derived output directory')
    reviews = json.loads(a.reviews.read_text())['episodes']
    files = sorted(source.glob('episode_*.json'))
    if not files or {f.stem for f in files} != set(reviews):
        p.error('Reviews must name the exact source episode set')
    planned = []
    for path in files:
        original = json.loads(path.read_text())
        rig = (original.get('annotation_policy') or {}).get('rig')
        derived = apply_review(original, reviews[path.stem], rig=rig)
        dest = target / path.name
        if dest.exists() and json.loads(dest.read_text()) != derived:
            raise ValueError(f'Existing derived output differs: {path.name}')
        planned.append((dest, derived))
    target.mkdir(parents=True, exist_ok=True)
    for dest, derived in planned:
        if not dest.exists():
            write_atomic(dest, derived, indent=1)
    summary = {'version': 1, 'episodes': len(planned), 'reviews_sha256': digest(reviews),
        'tasks_before': sum(r['task_scope_review']['tasks_before'] for _, r in planned),
        'work_tasks_after': sum(r['task_scope_review']['tasks_after'] for _, r in planned),
        'output_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path, _ in planned}}
    write_atomic(target / 'TASK_SCOPE.json', summary, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'output_sha256'}))


if __name__ == '__main__':
    main()
