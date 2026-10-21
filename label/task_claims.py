"""What a recording's task texts claim, by one law for every source (a LeRobot task_index or subtask column, an MCAP
/task or /task/subtask topic, a publisher's timed annotation, an adapter's sub-steps).

A dataset states its task in two ways: an instruction for the whole recording (context["instruction"]) and timed texts
(context["annotation_subtasks"], each {t0, t1, label} with the column or topic it came from). A timed text is only a
timed claim when its source's text changes within the recording. One that holds a single text over the whole recording
is a claim about the whole recording, never a step of it, whatever its source. Within one source, consecutive rows with
the same text are one claim, and an empty text, or a message replaced the moment it was sent, claims nothing. A claim
is read as holding over the whole recording only on established times: a row with no time, or a claim that ends in a
recording whose length is not stated, never is.

claims() applies the law when the context is read, so a context prepared before it is read the same way with no new
preparation; the readers apply it too (prepare/formats.py), so new contexts hold it already. Every reader of the
dataset's task texts goes through claims(): the request (label/episode.py), the parts of a long recording
(label/pieces.py), the board and its export (board/build.py).

- A whole-recording claim is the recording's instruction when it has none.
- One whose text differs from the instruction is kept as the dataset's task for the whole episode, shown in the request
  and on the board as such, never as a timed step.
- One whose text is the instruction's adds nothing.
- Every other timed text is a step, in the order the context lists its rows and numbered in the whole recording
  (step): the number a task judged against it names (label/prompts.py TASK_RULES), in every part of a long recording
  too.
"""
from __future__ import annotations

import copy
import math

# a message replaced within this many seconds of being sent holds no stretch of the recording (MicroAGI sends a
# placeholder and its title at the same moment)
INSTANT_S = 0.1


def _number(x):
    """A finite number, else None (a string, a bool, NaN or an infinity is no time)."""
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def text(row: dict) -> str:
    """A row's text, whitespace normalised; '' for none."""
    value = row.get('label') if 'label' in row else row.get('text')
    return ' '.join(value.split()) if isinstance(value, str) else ''


def has_text(row: dict) -> bool:
    """Whether a row states its claim as a text (a label or text field), whatever that text is."""
    return 'label' in row or 'text' in row


def _same(a: str, b: str) -> bool:
    """One text, whatever its case and spacing."""
    return ' '.join(a.split()).casefold() == ' '.join(b.split()).casefold()


def _merged(rows: list[tuple[int, dict]]) -> list[tuple[int, dict]]:
    """One source's (position, row) pairs, in the order the context lists them, with every non-claim dropped and
    consecutive rows of one text joined: the first row's fields and position, the last row's end, and joined_rows
    when more than one was joined."""
    out = []
    for k, (at, row) in enumerate(rows):
        words = text(row)
        t0, t1 = _number(row.get('t0')), _number(row.get('t1'))
        if not has_text(row):
            # a recorded annotation with no label or text field is kept as it was recorded, never joined
            out.append((at, copy.deepcopy(row)))
            continue
        # an empty text; a message replaced at once (a span of recorded rows, row_start, is never an instant)
        if not words or ('row_start' not in row and t0 is not None and t1 is not None and t1 - t0 < INSTANT_S
                         and k < len(rows) - 1):
            continue
        if out and _same(text(out[-1][1]), words):
            last = out[-1][1]
            if 't1' in row or 't1' in last:
                last['t1'] = row.get('t1')
            if 'row_end' in row:
                last['row_end'] = row['row_end']
            last['joined_rows'] = last.get('joined_rows', 1) + 1
            continue
        out.append((at, copy.deepcopy(row)))
    return out


def _reaches_end(row: dict, end_s: float | None, n_rows) -> bool:
    """Whether a claim holds to the end of the recording, on established times only: its span ends at the last
    recorded row, or it starts at a time and has no end (it holds until the recording ends), or it ends within a
    second (or 5% of the length) of the recording's stated end."""
    if isinstance(row.get('row_end'), int) and isinstance(n_rows, int):
        return row['row_end'] >= n_rows - 1
    t0, t1 = _number(row.get('t0')), _number(row.get('t1'))
    if t0 is None:
        return False
    if t1 is None:
        return 't1' not in row or row.get('t1') is None
    return end_s is not None and t1 >= end_s - max(1.0, 0.05 * end_s)


def _starts(row: dict, end_s: float | None) -> bool:
    """Whether a claim starts with the recording: at its first row, or within a second (or 5% of its length)."""
    if row.get('row_start') == 0:
        return True
    t0 = _number(row.get('t0'))
    return t0 is not None and t0 <= max(1.0, 0.05 * (end_s or 0.0))


def claims(ctx: dict, duration_s: float | None = None) -> dict:
    """{"instruction": the recording's instruction (its own, else its whole-recording claim), "instruction_from":
    "context" | "claim" | None, "whole": [{"text", "column", "source"}] the other whole-recording claims, "steps": the
    timed claims, each a copy of its row with step, its number in the whole recording, "held": the (column, source) of
    every source read as one whole-recording claim, for a reader that keeps its rows apart}.

    A part of a long recording (rows carrying request_window, label/pieces.py) was read by this law on the whole
    recording when it was cut, so its rows are its steps and keep their step numbers."""
    ctx = ctx if isinstance(ctx, dict) else {}
    rows = [x for x in ctx.get('annotation_subtasks') or [] if isinstance(x, dict)]
    # the instruction as the context holds it (only stripped), compared by its words
    instruction = ctx.get('instruction').strip() if isinstance(ctx.get('instruction'), str) else ''
    instruction = instruction or None
    whole = [copy.deepcopy(x) for x in ctx.get('dataset_task_claims') or [] if isinstance(x, dict)]
    if any(isinstance(x.get('request_window'), dict) for x in rows):
        steps = [copy.deepcopy(x) for x in rows if text(x) or not has_text(x)]
        for i, row in enumerate(steps, 1):
            row.setdefault('step', i)
        return {'instruction': instruction, 'instruction_from': 'context' if instruction else None, 'whole': whole,
                'steps': steps, 'held': []}
    # the recording's length as its context states it; a length that is not stated is not guessed from the claims
    end_s = _number(duration_s) if duration_s is not None else _number(ctx.get('duration_s'))
    n_rows = ctx.get('n_state_frames')
    sources = {}
    for at, row in enumerate(rows):
        sources.setdefault((row.get('column'), row.get('source') if row.get('column') else None), []).append((at, row))
    # a LeRobot task_index span the reader already took as the task (prepare/formats.py lerobot_annotations)
    for row in ctx.get('annotation_task_claims') or []:
        if isinstance(row, dict) and text(row):
            whole.append({'text': text(row), 'column': row.get('column'), 'source': row.get('source')})
    kept, held = [], []
    for (column, source), group in sources.items():
        merged = _merged(group)
        if not merged:
            continue
        first, last = merged[0][1], merged[-1][1]
        # it holds over the whole recording when it reaches the end and starts with it, or starts after nothing but
        # rows of its source that claim nothing (a placeholder, an empty text)
        led = merged[0][0] != group[0][0]
        one_text = all(has_text(row) for _, row in merged) and len({text(row).casefold() for _, row in merged}) == 1
        if one_text and _reaches_end(last, end_s, n_rows) and (
                _starts(first, end_s) or led and _number(first.get('t0')) is not None):
            whole.append({'text': text(first), 'column': column, 'source': source})
            held.append((column, source))
            continue
        kept += merged
    steps = [row for _, row in sorted(kept, key=lambda pair: pair[0])]
    for i, row in enumerate(steps, 1):
        row['step'] = i
    found = 'context' if instruction else None
    other = []
    for claim in whole:
        if instruction is None:
            instruction, found = claim['text'], 'claim'
        elif not _same(claim['text'], instruction) and not any(_same(claim['text'], k['text']) for k in other):
            other.append(claim)
    return {'instruction': instruction, 'instruction_from': found, 'whole': other, 'steps': steps, 'held': held}
