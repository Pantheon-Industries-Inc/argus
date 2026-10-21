"""What a recording's task texts claim, by one law for every source (a LeRobot task_index or subtask column, an MCAP
/task or /task/subtask topic, a publisher's timed annotation, an adapter's sub-steps).

A dataset states its task in two ways: an instruction for the whole recording (context["instruction"]) and timed texts
(context["annotation_subtasks"], each {t0, t1, label} with the column or topic it came from). A timed text is only a
timed claim when its source's text changes within the recording. One that holds a single text over the whole recording
is a claim about the whole recording, never a step of it, whatever its source.

Each source keeps its own semantics.
- A message source holds each text until the next one replaces it: an MCAP topic (a column named /...) and a recorded
  text column (rows with row_start and row_end). A message an MCAP topic replaces within 0.1s of sending it claims
  nothing (MicroAGI sends a placeholder and its title at the same moment); a recorded row is a sample of its frame and
  always claims its text.
- An annotation with an explicit start and end is a span, and one with no end, or an end equal to its start, is a
  moment. Both are always kept as they were given, however short.
- An empty or missing text claims nothing, and it is not covered by any text either.
- Identical texts of one source join into one claim only where their spans touch, with no gap between them; moments
  never join.
- A claim holds over the whole recording only when it is one unbroken span from within 1s of the recording's start (or
  its first row) to within 1s of its end (or its last row), on established times: a moment, a row with no time, or a
  claim in a recording whose length is not stated never is. A message source's leading text replaced within 0.1s does
  not break that coverage; any other leading text, or an empty one, does.

claims() applies the law when the context is read, so a context prepared before it is read the same way with no new
preparation; the LeRobot and MCAP readers apply it too (prepare/formats.py), so new contexts hold it already. An
instruction a reader already flattened from several timed texts (main's MCAP reader, "0.0 s: ...; 66.5 s: ...") stays
as it is: such a recording reads its timed prompts once it is prepared again. Every reader of the dataset's task texts
goes through claims(): the request (label/episode.py), the parts of a long recording (label/pieces.py), the board and
its export (board/build.py).

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

# a message source's text replaced within this many seconds of being sent holds no stretch of the recording
INSTANT_S = 0.1
# how close to the recording's start and end a claim must reach to hold over the whole recording
EDGE_S = 1.0
# two spans of one text touch when the second starts within this of the first's end (times are kept to the ms)
TOUCH_S = 0.001


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


def message_source(rows: list[dict], column) -> bool:
    """Whether a source holds each text until the next replaces it: recorded rows (row_start and row_end) or an MCAP
    topic (a column named /...). Any other source gives spans and moments."""
    return (isinstance(column, str) and column.startswith('/')) or all(
        isinstance(row.get('row_start'), int) for row in rows)


def _moment(row: dict) -> bool:
    """A row with a time and no end, or an end equal to its start (a point annotation)."""
    t0, t1 = _number(row.get('t0')), _number(row.get('t1'))
    return t0 is not None and (t1 is None or t1 == t0) and not isinstance(row.get('row_end'), int)


def _instant(rows: list[dict], k: int) -> bool:
    """Whether a message source's row k is replaced by the next row within INSTANT_S of being sent. A recorded row is
    a sample of its frame, held for at least that frame, never a message replaced between frames: at 30 fps a value
    held for two frames lasts 0.067s and is a real step (a 4-frame episode's "pick" then "place")."""
    if k + 1 >= len(rows) or isinstance(rows[k].get('row_start'), int):
        return False
    t0, nxt = _number(rows[k].get('t0')), _number(rows[k + 1].get('t0'))
    return t0 is not None and nxt is not None and nxt - t0 < INSTANT_S


def _touch(a: dict, b: dict) -> bool:
    """Whether span b starts where span a ends: the next recorded row, or a start within TOUCH_S of a's end."""
    if isinstance(a.get('row_end'), int) and isinstance(b.get('row_start'), int):
        return b['row_start'] == a['row_end'] + 1
    a1, b0 = _number(a.get('t1')), _number(b.get('t0'))
    return a1 is not None and b0 is not None and abs(b0 - a1) <= TOUCH_S


def _merged(rows: list[tuple[int, dict]], messages: bool) -> tuple[list[tuple[int, dict]], bool]:
    """(one source's claims, whether everything before its first claim was a message replaced at once). The (position,
    row) pairs keep the order the context lists them; a non-claim is dropped; consecutive spans of one text join where
    they touch, keeping the first row's fields and position, the last row's end, and joined_rows."""
    out, lead_ok = [], True
    plain = [row for _, row in rows]
    for k, (at, row) in enumerate(rows):
        words = text(row)
        if not has_text(row):
            # a recorded annotation with no label or text field is kept as it was recorded, never joined
            out.append((at, copy.deepcopy(row)))
            continue
        if not words or (messages and _instant(plain, k)):
            # an empty text claims nothing and covers nothing; a message replaced at once claims nothing
            if not out and not (messages and _instant(plain, k)):
                lead_ok = False
            continue
        if out and not _moment(row) and not _moment(out[-1][1]) and has_text(out[-1][1]) and _touch(out[-1][1], row) \
                and _same(text(out[-1][1]), words):
            last = out[-1][1]
            if 't1' in row or 't1' in last:
                last['t1'] = row.get('t1')
            if 'row_end' in row:
                last['row_end'] = row['row_end']
            last['joined_rows'] = last.get('joined_rows', 1) + 1
            continue
        out.append((at, copy.deepcopy(row)))
    return out, lead_ok


def _covers(row: dict, start_row: dict, end_s: float | None, n_rows) -> bool:
    """Whether one claim holds from the recording's start (start_row's start: its own row, or a placeholder replaced at
    once before it) to its end, on established times."""
    if _moment(row):
        return False
    if isinstance(start_row.get('row_start'), int) and isinstance(row.get('row_end'), int) and isinstance(n_rows, int):
        return start_row['row_start'] == 0 and row['row_end'] >= n_rows - 1
    t0, t1 = _number(start_row.get('t0')), _number(row.get('t1'))
    return (end_s is not None and t0 is not None and t1 is not None
            and t0 <= EDGE_S and t1 >= end_s - EDGE_S)


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
        messages = message_source([row for _, row in group], column)
        merged, lead_ok = _merged(group, messages)
        if not merged:
            continue
        first = merged[0][1]
        # one unbroken claim of one text, from the start (or a placeholder replaced at once before it) to the end
        start_row = group[0][1] if lead_ok else first
        if len(merged) == 1 and has_text(first) and _covers(first, start_row, end_s, n_rows):
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
