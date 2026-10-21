"""A model reply that holds the work any request asks for (label/completeness.py), for fixtures whose subject is
something else: a timeline step over the footage and its one task with its outcome. Like a reply in the tasks format
(label/prompts.py), it writes no completion; the harness derives one from the task. A fixture that sends only a task
summary sends a hollow reply, which is never a finished annotation."""
import json
import re

GRID = re.compile(r'=== grid t=(-?[\d.]+)-(-?[\d.]+)s')
END_S = 3600.0


def span(request=None) -> tuple[float, float]:
    """The footage a request sent, from its grid captions, or 0 to END_S when there is no request to read."""
    if request is None:
        return 0.0, END_S
    data = request.data if hasattr(request, 'data') else request
    times = [float(t) for pair in GRID.findall(data.decode() if isinstance(data, bytes) else str(data)) for t in pair]
    return (min(times), max(times)) if times else (0.0, END_S)


def labels(summary='The hand reaches.', *, request=None, **fields):
    lo, hi = span(request)
    # one step as the parser saves it, which a reply may also send
    return {'task_summary': summary,
            'timeline': [{'start_s': lo, 'end_s': hi, 'arm': 'right', 'action': 'reach', 'contribution': 'advancing',
                          'progress': 1.0}],
            'tasks': [{'start_s': lo, 'end_s': hi, 'task': summary, 'objects': [], 'outcome': 'success'}],
            **fields}


def text(summary='The hand reaches.', **kw) -> str:
    return json.dumps(labels(summary, **kw))
