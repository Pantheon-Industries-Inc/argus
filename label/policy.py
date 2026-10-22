"""Resolve annotation settings once for requests, parts, resume and upload callers."""
from __future__ import annotations

import math

VERSION = 1
ASTRA = 'openai/gpt-6-astra'
SOL61 = 'openai/gpt-6.1-sol'
PERIODS = {'teleop_arms': 1.5, 'handheld_gripper': 1.0, 'ego_head': 0.5}
# The task modes a policy saved before the tasks format names. Every request now asks for a list of tasks
# (label/prompts.py fixed_instructions), so a new policy names none. A saved mode, or one a replay passes from a saved
# policy, is kept as it is, so a job labelled then resumes and rebinds its paid replies; it never changes the request.
TASK_MODES = ('episode', 'session')
# OpenAI counts reasoning tokens inside max_completion_tokens, so a call's output limit is
# the answer it writes plus room to reason at its effort. Medium is about twice the most
# reasoning any call used in the October 2026 Astra runs (2,128 tokens, a sensor analysis;
# evidence selection used at most 239), and each effort step doubles it.
REASONING_TOKENS = {'none': 0, 'minimal': 1000, 'low': 2000, 'medium': 4000, 'high': 8000, 'xhigh': 16000}
# The paid calls beside the annotation. Evidence inspection asks the model, before it annotates, which recorded fields
# to read at full rate (label/evidence_access.py discover); it was built for depth and touch sensing, so by default
# (auto) only a recording with depth or a touch sensor (tactile, pressure, contact; a wrist force/torque sensor alone
# does not count) is inspected, all inspects any recording with recorded fields beyond its cameras, and off buys no
# selection. The field dictionary asks once per upload what uncertain fields mean
# (label/dictionary_stage.py), on by default, and is written into the policy only when off.
# A policy saved before inspection could be set inspected every recording with recorded fields, so a saved policy
# without the setting keeps that (LEGACY_INSPECTION, never written) and a job labelled then resumes as it was paid
# for; a new policy writes its mode, auto unless asked otherwise.
INSPECTION_MODES = ('auto', 'all', 'off')
INSPECTION = 'auto'
LEGACY_INSPECTION = 'all'


def inspection_mode(selected: dict) -> str:
    """The inspection mode of a resolved or saved policy."""
    return selected.get('inspection', LEGACY_INSPECTION)


def output_limit(answer_tokens: int, reasoning: str) -> int:
    """max_completion_tokens for a call that writes up to answer_tokens at this reasoning effort."""
    if reasoning not in REASONING_TOKENS:
        raise ValueError(f'unsupported reasoning effort {reasoning!r}')
    return answer_tokens + REASONING_TOKENS[reasoning]


FIELD_DICTIONARY = False


def resolve(ctx: dict, *, model: str | None = None, reasoning: str | None = None,
            sample_fps: float | None = None, task_mode: str | None = None,
            dataset_note: str | None = None, inspection: str | None = None,
            field_dictionary: bool | None = None) -> dict:
    """Source declarations and caller overrides, independent of dataset names or absent task text."""
    saved = ctx.get('annotation_policy') or {}
    if saved and (not isinstance(saved, dict) or saved.get('version') != VERSION):
        raise ValueError('annotation policy version is not supported')
    rig = ctx.get('profile') or 'teleop_arms'
    if rig not in PERIODS:
        raise ValueError(f'unknown recording rig {rig!r}')
    model = model or saved.get('model') or ASTRA
    if not isinstance(model, str) or not model.strip():
        raise ValueError('model must be a nonempty identifier')
    sol = model.split('/')[-1] == SOL61.split('/')[-1]
    prior_reasoning = saved.get('reasoning') if saved.get('model') == model else None
    # Astra defaults to medium reasoning, every other model to high
    astra = model.split('/')[-1] == ASTRA.split('/')[-1]
    reasoning = reasoning or prior_reasoning or ('medium' if astra else 'high')
    if reasoning not in REASONING_TOKENS:
        raise ValueError('unsupported reasoning effort')
    if sample_fps is None:
        sample_fps = saved.get('sample_fps')
    if sample_fps is not None:
        if type(sample_fps) not in (float, int) or not math.isfinite(sample_fps) or sample_fps <= 0:
            raise ValueError('sample FPS must be finite and positive')
        period = 1.0 / sample_fps
    elif saved.get('model') == model and saved.get('sample_every_s') is not None:
        period = saved['sample_every_s']
        if type(period) not in (float, int) or not math.isfinite(period) or period <= 0:
            raise ValueError('saved sampling cadence is invalid')
    else:
        period = 1.0 if sol else PERIODS[rig]
    if sol and period > 1.0:
        raise ValueError('GPT-6.1 Sol requires at least 1 sample per second')
    # a source's declaration (ctx task_mode or sessions, prepare/annotations.py) is provenance only
    mode = task_mode or saved.get('task_mode')
    if mode is not None and mode not in TASK_MODES:
        raise ValueError('task mode must be episode or session')
    note = dataset_note if dataset_note is not None else saved.get('dataset_note', ctx.get('collection_note', ''))
    if not isinstance(note, str):
        raise ValueError('dataset description must be text')
    # Preserve Astra's request budget. Denser sampling reduces the duration allowance proportionally.
    baseline = 1.0 if sol else PERIODS[rig]
    maximum = 450.0 if period >= baseline else 330.0 * min(1.0, period / 0.5)
    if sample_fps is None and saved.get('model') == model and saved.get('part_max_s') is not None:
        previous = saved['part_max_s']
        if type(previous) not in (float, int) or not math.isfinite(previous) or not 0 < previous <= maximum:
            raise ValueError('saved part allowance is incompatible; rebuild with an explicit policy')
        maximum = previous
    inspect = inspection if inspection is not None else inspection_mode(saved) if saved else INSPECTION
    if inspect not in INSPECTION_MODES:
        raise ValueError(f'inspection must be one of {", ".join(INSPECTION_MODES)}')
    # a new policy has no field dictionary unless asked for: arm state alone qualifies an upload for the call, so it
    # ran on nearly every teleop upload. A policy saved without the key was made when the dictionary ran by default,
    # and resumes with it
    dictionary = (field_dictionary if field_dictionary is not None
                  else saved.get('field_dictionary', True) if saved else FIELD_DICTIONARY)
    if type(dictionary) is not bool:
        raise ValueError('field dictionary must be on or off')
    return {'version': VERSION, 'rig': rig, 'model': model, 'reasoning': reasoning,
            'sample_fps': float(sample_fps) if sample_fps is not None else None,
            'sample_every_s': period, 'part_max_s': maximum,
            **({'task_mode': mode} if mode is not None else {}),
            'dataset_note': note,
            **({'inspection': inspect} if inspect != LEGACY_INSPECTION else {}),
            **({'field_dictionary': False} if not dictionary else {})}


def effective(ctx: dict) -> dict:
    return resolve(ctx)


def add_arguments(parser) -> None:
    """Shared optional controls; omitted values inherit source policy or defaults."""
    parser.add_argument('--model', default=None, help='model identifier (default openai/gpt-6-astra)')
    parser.add_argument('--reasoning', default=None, help='reasoning effort (default medium for Astra, high for every other model)')
    parser.add_argument('--sample-fps', type=float, default=None,
                        help='optional sampling override; GPT-6.1 Sol defaults to at least 1 FPS')
    parser.add_argument('--dataset-note', default=None, help='dataset description supplied as recording context')
    parser.add_argument('--inspection', choices=INSPECTION_MODES, default=None,
                        help='paid evidence selection before each annotation: auto (default) for a recording with depth '
                             'or a touch sensor (not a force/torque sensor alone), all for any recording with recorded '
                             'fields beyond its cameras, off for none')


def persist(episodes, **overrides) -> dict:
    """Write resolved policy into prepared sidecars before cuts or dispatch."""
    import json
    from pathlib import Path
    from label.atomic import write_atomic
    results = {}
    for directory in sorted(Path(episodes).glob('episode_*')):
        path = directory / 'context.json'
        if not path.is_file():
            continue
        ctx = json.loads(path.read_text())
        selected = resolve(ctx, **overrides)
        if ctx.get('annotation_policy') != selected:
            ctx['annotation_policy'] = selected
            write_atomic(path, ctx)
        results[directory.name] = selected
    return results


def require_request_window(ctx: dict, seconds: float) -> None:
    """Refuse an oversized request before routing or final dispatch."""
    maximum = effective(ctx)['part_max_s']
    original = (ctx.get('piece') or {}).get('max_s')
    if original is not None:
        if type(original) not in (float, int) or not math.isfinite(original) or original <= 0:
            raise ValueError('saved request-part allowance is invalid')
        maximum = min(maximum, original)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('recording duration is invalid')
    if seconds > maximum + 1e-8:
        raise ValueError(f'Recording window {seconds:g} s exceeds the {maximum:g} s request allowance. '
                         'Prepare request parts with the selected policy before labeling.')
