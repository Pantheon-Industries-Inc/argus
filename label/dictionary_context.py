"""Bind upload interpretations to exact prepared fields without changing recorded context."""
from __future__ import annotations

import copy
import re
from pathlib import Path

from label.dictionary import effective


def _owner(context, episode_id=None):
    return (episode_id or (context.get("piece") or {}).get("of") or context.get("episode_id")
            or (context.get("data_dictionary") or {}).get("episode_id"))


def _bound(context, field, binding):
    path = binding.get("context_path")
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
        return isinstance(path, str) and path.startswith("cameras/") and path.split("/", 1)[1] in (context.get("cameras") or {})
    if not isinstance(path, str):
        return False
    value = context
    try:
        for part in path.split("/"):
            value = value[int(part)] if isinstance(value, list) else value[part]
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
                    ("id", "name", "kind", "shape", "dtype", "names", "rate_hz", "source", "summary", "limitations") if key in field}
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
        "missing_fields": [ident for ident in resolved.get("missing_fields", []) if ident in ids]}
    return result


def field_interpretation(context: dict, name: str, kind="signal") -> dict:
    """One exact bound field, or no interpretation when recorded identities are ambiguous."""
    overlay = context.get("data_dictionary") or {}
    owner = _owner(context)
    if owner != overlay.get("episode_id"):
        return {}
    matched = [field for field in overlay.get("fields", []) if field.get("name") == name and field.get("kind") == kind
               and any(binding.get("episode") == owner and _bound(context, field, binding) for binding in field.get("bindings", []))]
    if len(matched) != 1:
        return {}
    return copy.deepcopy((overlay.get("entries") or {}).get(matched[0]["id"]) or {})
