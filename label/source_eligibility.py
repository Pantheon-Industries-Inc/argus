"""Decide whether retained source members add evidence beyond prepared fields."""
import json


def _leaves(value, pointer=''):
    if isinstance(value, dict):
        if not value:
            yield pointer
        for key, child in value.items():
            token = str(key).replace('~', '~0').replace('/', '~1')
            yield from _leaves(child, pointer + '/' + token)
    elif isinstance(value, list):
        if not value:
            yield pointer
        for index, child in enumerate(value):
            yield from _leaves(child, pointer + '/' + str(index))
    else:
        yield pointer


def _covered(pointer, paths):
    return any(pointer == path or pointer.startswith(path + '/') for path in paths)


def unrepresented_json(value, paths, pointer=''):
    """Return only unknown source leaves for prompt notes, preserving original context elsewhere."""
    if _covered(pointer, paths):
        return None
    if isinstance(value, dict):
        kept = {}
        for key, child in value.items():
            token = str(key).replace('~', '~0').replace('/', '~1')
            selected = unrepresented_json(child, paths, pointer + '/' + token)
            if selected is not None:
                kept[key] = selected
        return kept or None
    if isinstance(value, list):
        kept = [unrepresented_json(child, paths, pointer + '/' + str(i))
                for i, child in enumerate(value)]
        return kept if any(item is not None for item in kept) else None
    return value


def archive_has_unrepresented(archive, coverage):
    """Unknown members remain eligible; coverage is exact source path and member identity."""
    covered_json = coverage.get('json', {})
    for source, value in (archive.get('recorded_metadata') or {}).items():
        paths = covered_json.get(str(source), [])
        if any(not _covered(pointer, paths) for pointer in _leaves(value)):
            return True
    source = archive.get('recorded_hdf5_metadata') or {}
    # A member in another HDF5 file never represents this file's same-named member.
    members = coverage.get('hdf5', {}).get(source.get('source_path'), {})
    for field in source.get('fields') or []:
        obj = field.get('object')
        attribute = field.get('attribute')
        key = json.dumps([obj, attribute], separators=(',', ':'))
        if key not in members:
            return True
        paths = members[key]
        if paths is None:
            continue
        try:
            from prepare.hdf_metadata import read
            value = read(source, field, '')
            # h5_fps unwraps a one-element string array before parsing its JSON.
            while isinstance(value, list) and len(value) == 1:
                value = value[0]
            if isinstance(value, str):
                value = json.loads(value)
            if any(not _covered(pointer, paths) for pointer in _leaves(value)):
                return True
        except (OSError, ValueError, KeyError, TypeError):
            return True
    return False


def needs_extra_evidence(ctx):
    coverage = ctx.get('represented_source_members') or {}
    return any(archive_has_unrepresented(archive, coverage)
               for archive in [ctx, *(ctx.get('recorded_sensor_fields') or [])])


def mark_hdf(coverage, source_path, obj, attribute=None, pointer=None):
    members = coverage.setdefault('hdf5', {}).setdefault(str(source_path), {})
    key = json.dumps([obj, attribute], separators=(',', ':'))
    if pointer is None:
        members[key] = None
    elif key not in members or members[key] is not None:
        paths = members.setdefault(key, [])
        if pointer not in paths:
            paths.append(pointer)


def mark_json(coverage, source, pointer):
    paths = coverage.setdefault('json', {}).setdefault(str(source), [])
    if pointer not in paths:
        paths.append(pointer)
