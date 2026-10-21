"""Bind upload interpretations to exact prepared fields without changing recorded context."""
from __future__ import annotations

import copy
import re
from pathlib import Path

from label.dictionary import effective, _digest


def _owner(context, episode_id=None):
    return (episode_id or (context.get("piece") or {}).get("of") or context.get("episode_id")
            or (context.get("data_dictionary") or {}).get("episode_id"))


def _pointer_key(value):
    return value.replace("~1", "/").replace("~0", "~")


def _bound(context, field, binding):
    path = binding.get("context_path")
    if field.get('kind') == 'native':
        if not isinstance(path, str) or not re.fullmatch(r'table_field_inventory/\d+', path):
            return False
        index = int(path.split('/')[1])
        fields = context.get('table_field_inventory') or []
        if index >= len(fields):
            return False
        native = fields[index]
        return (native.get('name') == field.get('name') and native.get('array') == binding.get('key')
                and context.get('recorded_table_fields') == binding.get('file')
                and native.get('dtype') == field.get('dtype') and native.get('source_shape') == field.get('shape')
                and _digest(native.get('descriptor') or {}) == binding.get('source_descriptor_sha256'))
    if field.get("kind") == "signal":
        if not isinstance(path, str) or not re.fullmatch(r"signals/\d+", path):
            return False
        index = int(path.split("/")[1])
        signals = context.get("signals") or []
        if index >= len(signals):
            return False
        meta = signals[index]
        return (meta.get("name") == field.get("name") and meta.get("key") == binding.get("key")
                and (meta.get("file") or "signals.npz") == binding.get("file"))
    if field.get("kind") in ("state", "action"):
        kind = field["kind"]
        name = (context.get("source") or {}).get(kind) or kind
        return path == kind and field.get("name") == name and binding.get("file") == "state.npz" and binding.get("key") == kind
    if field.get("kind") == "camera":
        if isinstance(path, str) and path.startswith("cameras/"):
            camera = (context.get("cameras") or {}).get(_pointer_key(path.split("/", 1)[1]))
            return isinstance(camera, dict) and camera.get('name') == field.get('name')
        if isinstance(path, str) and re.fullmatch(r'unshown_cameras/\d+', path):
            index = int(path.split('/')[1])
            cameras = context.get('unshown_cameras') or []
            return index < len(cameras) and isinstance(cameras[index], dict) and cameras[index].get('name') == field.get('name')
        return False
    if not isinstance(path, str):
        return False
    value = context
    try:
        for part in path.split("/"):
            value = value[int(part)] if isinstance(value, list) else value[_pointer_key(part)]
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    return True


def _public_source(value):
    if isinstance(value, str):
        return re.sub(r'(?<![\w])/(?:Users|home|data|app|private|tmp|var)/[^\s<>"\']+',
                      lambda match: Path(match.group()).name, value)
    if isinstance(value, list):
        return [_public_source(v) for v in value]
    if isinstance(value, dict):
        return {key: _public_source(v) for key, v in value.items()}
    return value


def apply_context(context: dict, record: dict, overrides: dict | None = None, *, episode_id=None) -> dict:
    """Copy context and attach only fields whose owner, prepared path, key and name agree."""
    resolved = effective(record, overrides)
    owner = _owner(context, episode_id)
    result = copy.deepcopy(context)
    fields, limitations = [], list(resolved.get("limitations") or [])
    for field in resolved.get("inventory", {}).get("fields", []):
        bindings = [binding for binding in field.get("bindings", []) if binding.get("episode") == owner]
        if not bindings:
            continue
        matched = [binding for binding in bindings if _bound(context, field, binding)]
        if not matched:
            limitations.append(f"dictionary field {field['id']} has no exact prepared binding for {owner}")
            continue
        selected = {key: copy.deepcopy(field[key]) for key in
                    ("id", "name", "kind", "shape", "dtype", "names", "names_source", "units_source", "units_contradicted", "rate_hz", "source", "summary", "limitations", 'unit', 'units', 'response_direction', 'calibration', 'coordinate_frame', 'sensor_type', 'description', 'action_semantics', 'recorded_role') if key in field}
        selected.update(episodes=[owner], bindings=copy.deepcopy(matched))
        selected["source"] = _public_source(selected.get("source"))
        fields.append(selected)
    ids = {field["id"] for field in fields}
    result["data_dictionary"] = {"schema": record.get("schema"), "inventory_digest": record.get("inventory_digest"),
        "episode_id": owner, "status": resolved.get("status"), "fields": fields,
        "entries": {ident: copy.deepcopy(value) for ident, value in resolved.get("entries", {}).items() if ident in ids},
        "machine_entries": {ident: copy.deepcopy(value) for ident, value in record.get("entries", {}).items() if ident in ids},
        "override_revision": resolved.get("override_revision", 0),
        "override_history": copy.deepcopy(resolved.get("override_history") or []),
        "override_limitations": _public_source(resolved.get("override_limitations") or []),
        "limitations": _public_source(limitations),
        "missing_fields": [ident for ident in resolved.get("missing_fields", []) if ident in ids],
        **{key: [ident for ident in record[key] if ident in ids]
           for key in ('request_field_ids', 'deferred_fields') if key in record}}
    return result


def native_field_name(context: dict, key: str, kind: str) -> str:
    """Recorded name for a canonical prepared array or camera key."""
    if kind in ("state", "action"):
        return (context.get("source") or {}).get(kind) or kind
    if kind == "camera":
        return ((context.get("cameras") or {}).get(key) or {}).get("name") or key
    return key


def prepared_interpretation(context: dict, key: str, kind: str) -> dict:
    """Resolve native meaning while retaining the exact prepared camera binding."""
    path = "cameras/" + key.replace("~", "~0").replace("/", "~1") if kind == "camera" else key
    return field_interpretation(context, native_field_name(context, key, kind), kind, context_path=path)


def field_interpretation(context: dict, name: str, kind="signal", *, context_path=None, source_identity=None) -> dict:
    """One exact bound field, or no interpretation when recorded identities are ambiguous."""
    from label.content_identity import same_recorded
    overlay = context.get("data_dictionary") or {}
    owner = _owner(context)
    if owner != overlay.get("episode_id"):
        return {}
    if kind == 'native' and (not source_identity or source_identity.get('missing')):
        return {}
    matched = [field for field in overlay.get("fields", []) if field.get("name") == name and field.get("kind") == kind
               and any(binding.get("episode") == owner and _bound(context, field, binding)
                       and (context_path is None or binding.get("context_path") == context_path)
                       and (kind != 'native' or same_recorded(binding.get('source_identity'), source_identity))
                       for binding in field.get("bindings", []))]
    if len(matched) != 1:
        return {}
    return copy.deepcopy((overlay.get("entries") or {}).get(matched[0]["id"]) or {})


SCENE_CONTENT_ROLES = frozenset(('scene', 'scene_camera', 'scene_image'))
SENSOR_CONTENT_ROLES = frozenset(('sensor', 'sensor_view', 'tactile_image', 'optical_tactile', 'touch', 'tactile'))
TACTILE_CONTENT_ROLES = SENSOR_CONTENT_ROLES - {'sensor', 'sensor_view'}


def image_role(role) -> str:
    """The role an interpretation gives a camera, checked against what the camera is: images. A role that names a
    numeric touch quantity (label/signals.py TOUCH_QUANTITY_ROLES, eg pressure) says the images show touch, so it reads
    as tactile; a wrench (force_torque) has no image form, so it reads as no role and the camera's own name decides.
    The numeric counterpart is label/signals.py signal_role. Every other role is kept."""
    from label.signals import _role, TOUCH_QUANTITY_ROLES
    role = _role(role)
    if role in SENSOR_CONTENT_ROLES or role not in TOUCH_QUANTITY_ROLES:
        return role
    return '' if role == 'force_torque' else 'tactile'


def camera_role(ctx: dict, view: str, interpretation: dict, meta: dict | None = None) -> str | None:
    """The role an interpretation gives a camera, with what the reader read from the camera's own recorded name taking
    precedence over a model's guess. A machine-interpreted sensor or tactile role is dropped when the reader placed the
    camera on an arm from its recorded name (prepare/formats.py camera_role_provenance, eg right_wrist_cam) and that
    name carries no sensing word (is_sensing, eg gelsight or digit): the dictionary model can fill a gap the reader
    left, or agree with it, but never turn a camera the reader identified into a touch sensor. A role a person set
    always stands."""
    from prepare.formats import camera_role_provenance, is_sensing
    role = image_role(interpretation.get('role'))
    if role not in SENSOR_CONTENT_ROLES or interpretation.get('provenance') != 'machine':
        return role
    meta = meta if meta is not None else (ctx.get('cameras') or {}).get(view) or {}
    # placed from the recording itself: its recorded name or a recorded mount; an adapter's description of a
    # presentation slot is not a placement
    placed = camera_role_provenance(meta, view, ctx.get('profile') or 'teleop_arms') in ('recorded_name', 'recorded')
    said = ' '.join(str(meta.get(key) or '') for key in ('key', 'name', 'desc')) or view
    return None if placed and not is_sensing(said) else role


def camera_content_role(ctx: dict, view: str, entry: dict | None = None) -> dict | None:
    """Resolve explicit content roles independently of mounting and initial machine interpretations."""
    from label.signals import _role
    if entry is None:
        entry = (ctx.get('cameras') or {}).get(view) or {}
    path = 'cameras/' + view.replace('~', '~0').replace('/', '~1')
    if view in (ctx.get('cameras') or {}):
        interpretation = prepared_interpretation(ctx, view, 'camera')
    else:
        index = next((i for i, item in enumerate(ctx.get('unshown_cameras') or []) if item is entry), None)
        path = 'unshown_cameras/' + str(index)
        interpretation = field_interpretation(ctx, entry.get('name'), 'camera', context_path=path)
    overlay = ctx.get('data_dictionary') or {}
    fields = [f for f in overlay.get('fields') or [] if f.get('kind') == 'camera'
              and f.get('name') == entry.get('name') and any(b.get('episode') == _owner(ctx)
              and b.get('context_path') == path and _bound(ctx, f, b) for b in f.get('bindings') or [])]
    if interpretation.get('provenance') == 'human' and 'role' in interpretation and len(fields) == 1:
        ident = fields[0]['id']
        machine = (overlay.get('machine_entries') or {}).get(ident)
        explicit = (not isinstance(machine, dict) or interpretation['role'] != machine.get('role')
                    or any(row.get('field_id') == ident and 'role' in (row.get('after') or {})
                           for row in (overlay.get('override_history') or overlay.get('history') or [])))
        if explicit:
            role = image_role(interpretation['role'])
            return {'role': 'scene' if role in SCENE_CONTENT_ROLES else
                    'tactile' if role in TACTILE_CONTENT_ROLES else
                    'sensor' if role in SENSOR_CONTENT_ROLES else 'other', 'provenance': 'human'}
    # These structured facts declare content. Generic RGB and mount declarations do not.
    for key in ('content_role', 'recorded_role', 'role'):
        role = _role(entry.get(key))
        if role in SCENE_CONTENT_ROLES | SENSOR_CONTENT_ROLES:
            return {'role': 'scene' if role in SCENE_CONTENT_ROLES else
                    'tactile' if role in TACTILE_CONTENT_ROLES else 'sensor', 'provenance': 'recorded'}
        if key in entry and role in ('', 'other', 'unknown', 'unclear'):
            return {'role': 'other', 'provenance': 'recorded'}
    return None
