"""Names for the values of a recorded array, wherever the recording or a rig description shipped with it gives them.

A reader names an array's values when the record itself does: a JointState's name list beside its position
(formats._joint_names), a LeRobot feature's names, an HDF5 dataset's own names attribute. Everything else that can name
them is a declaration somewhere else in the upload, and each format has its own:

- MCAP: the file's metadata records and attachments, each channel's metadata and each channel's JSON schema
  (mcap_claims);
- HDF5: the attributes of the file and of every group above an array (h5_claims);
- LeRobot: meta/info.json, whose features name their values;
- any recording: a YAML or JSON description in its folder or a folder above it, up to the upload's root, such as the
  station.yaml robots_realtime writes beside each episode (description_files, read through describing).

A declaration becomes claims, {"path": the keys that lead to a list of names, "names", "source"}. A list of two or more
distinct non empty strings is a list of names (parse_names): a list, a JSON list in a string, a comma separated
string under a key that says names, a mapping holding one list (LeRobot's {"motors": [...]}), or a list of entries
that each carry a name or a title (a JSON schema's prefixItems). The list names the key it sits under (joint_pos:
[...]), or, under a key that says names (names, joint_names, labels, ...), the key above it ({joint_pos: {names:
[...]}}); a key such as joint_pos_names names joint_pos. Container keys (properties, features, fields, items) are not
part of a path, and an entry of a list that carries a name adds its name (arms: [{name: left, ...}] is arms/left).

write_signals (prepare/formats.py), the step every reader's arrays pass through on their way to context.json, attaches
them (resolve). A claim names an array when its path ends the array's name, or the name ends its path (both split at
/, . and spaces, in any case), and it gives one name per value; the claims that match the most segments win,
the nearer declaration wins a tie, and claims that still disagree name nothing, which is a data issue. A claim on an
MCAP channel as a whole names that channel's one array of its width. Names a reader already attached are kept. An
array of two or more values that nothing names is kept as it is and marked unnamed, so the readout, the board and the
exports say that its values are known only by their order."""
from __future__ import annotations

import contextlib
import contextvars
import json
import re
from pathlib import Path

# keys whose value is the names of the values of what they describe
NAME_KEYS = frozenset({"names", "value_names", "joint_names", "channel_names", "column_names", "field_names",
                       "element_names", "dim_names", "dimension_names", "axis_names", "motor_names", "actuator_names",
                       "labels", "columns", "fields"})
# a key that ends in one of these names the field its first part says (joint_pos_names, joint_pos.names, jointPosNames)
NAME_SUFFIX = re.compile(r"(?:[_. -]|(?<=[a-z0-9]))(?:value_?|joint_?|channel_?|column_?|element_?)?names$", re.I)
# keys that only hold what they describe, never part of a field's identity
CONTAINER_KEYS = frozenset({"properties", "features", "fields", "items", "prefixitems", "definitions", "$defs",
                            "signals", "channels", "topics", "streams", "datasets"})
ENTRY_NAME_KEYS = ("name", "title", "joint", "id")
SEGMENT_SPLIT = re.compile(r"[/.\s]+")
WALK_DEPTH = 12                 # a declaration nested deeper than this is not searched further
WALK_ITEMS = 2000               # entries of one list searched, so a recorded table in a description costs nothing
DESCRIPTION_SUFFIXES = (".yaml", ".yml", ".json")
DESCRIPTION_BYTES = 1_000_000   # as formats.NOTE_JSON_MAX_BYTES: a larger file is data, not a description
DESCRIPTIONS_PER_FOLDER = 64
DESCRIPTION_RANK_FLOOR = 10_000


def segments(name) -> tuple[str, ...]:
    """A name's segments for matching: split at /, . and whitespace, lowercased, empty parts dropped."""
    return tuple(s.lower() for s in SEGMENT_SPLIT.split(str(name)) if s)


def _text(x):
    if isinstance(x, (bytes, bytearray)):
        return bytes(x).decode("utf-8", "replace")
    return x


def parse_names(value, *, text: bool = False) -> list[str] | None:
    """The names a value gives, or None. A list (or a numpy array) of two or more distinct non empty strings, nested
    lists flattened in order; a mapping holding exactly one such list; a list of entries that each carry a distinct
    name or title; a string holding a JSON list; and, when text is set (the value sits under a key that says names), a
    comma separated string. Numbers, repeated entries (a units list of "rad" six times) and empty strings name
    nothing."""
    value = _text(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("["):
            try:
                return parse_names(json.loads(s))
            except (ValueError, RecursionError):
                return None
        if text and "," in s:
            return parse_names([p.strip() for p in s.split(",")])
        return None
    if isinstance(value, dict):
        lists = [v for v in value.values() if isinstance(v, (list, tuple))]
        return parse_names(lists[0]) if len(lists) == 1 and len(value) == 1 else None
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    if all(isinstance(_text(v), dict) for v in value):
        value = [next((v[k] for k in ENTRY_NAME_KEYS if isinstance(v.get(k), str)), None) for v in value]
    flat = []

    def walk(x, depth=0):
        if isinstance(x, (list, tuple)) and depth < 4:
            for y in x:
                walk(y, depth + 1)
        else:
            flat.append(_text(x))
    walk(value)
    if len(flat) < 2 or not all(isinstance(x, str) and x.strip() for x in flat) or len(set(flat)) != len(flat):
        return None
    return [x.strip() for x in flat]


def _field_of(key: str) -> str | None:
    """The field a key like joint_pos_names or jointPosNames names, or None."""
    m = NAME_SUFFIX.search(key)
    if not m or m.start() == 0:
        return None
    return key[:m.start()].rstrip("_.- ") or None


def claims_in(doc, source: str, scope=()) -> list[dict]:
    """Every claim a declaration makes (module docstring), its paths under scope (an MCAP channel's topic, an HDF5
    group's path)."""
    out = []

    def claim(path, names):
        if names:
            out.append({"path": list(path), "names": names, "source": source})

    def walk(x, path, depth):
        x = _text(x)
        if depth > WALK_DEPTH:
            return
        if isinstance(x, dict):
            for k, v in list(x.items())[:WALK_ITEMS]:
                k = str(_text(k))
                low = k.lower()
                v = _text(v)
                if low in NAME_KEYS:
                    named = parse_names(v, text=True)
                    if named:
                        claim(path, named)
                        continue
                field = _field_of(k) if low not in NAME_KEYS else None
                if field is not None:
                    named = parse_names(v, text=True)
                    if named:
                        claim(path + segments(field), named)
                        continue
                here = path if low in CONTAINER_KEYS or low in NAME_KEYS else path + segments(k)
                if isinstance(v, (list, tuple)):
                    claim(here if low in ("items", "prefixitems") else path + segments(k), parse_names(v))
                walk(v, here, depth + 1)
        elif isinstance(x, (list, tuple)):
            for v in list(x)[:WALK_ITEMS]:
                v = _text(v)
                if isinstance(v, dict):
                    entry = next((v[k] for k in ENTRY_NAME_KEYS if isinstance(v.get(k), str) and v[k].strip()), None)
                    walk(v, path + (segments(entry) if entry else ()), depth + 1)
                elif isinstance(v, (list, tuple)):
                    walk(v, path, depth + 1)
    walk(doc, tuple(segments("/".join(scope)) if not isinstance(scope, str) else segments(scope)), 0)
    return out


def decoded(value):
    """A declaration's text value as the structure it holds: JSON, or YAML when it spans lines and has a key; any
    other text stays as it is."""
    value = _text(value)
    if not isinstance(value, str):
        return value
    s = value.strip()
    if s[:1] in "[{":
        try:
            return json.loads(s)
        except (ValueError, RecursionError):
            return value
    if "\n" in s and ":" in s:
        parsed = load_yaml(s)
        if isinstance(parsed, (dict, list)):
            return parsed
    return value


def load_yaml(text: str):
    """YAML's safe reading of text, or None when it is not YAML or PyYAML is not installed (it ships with the locked
    environment, through huggingface-hub and mcap-ros1-support)."""
    try:
        import yaml
    except ImportError:
        return None
    try:
        return yaml.safe_load(text)
    except Exception:  # noqa: BLE001 - yaml on upload bytes raises many types; a file that does not parse names nothing
        return None


def read_description(path: Path):
    """A YAML or JSON description file as the structure it holds, or None."""
    try:
        p = Path(path)
        if p.stat().st_size > DESCRIPTION_BYTES:
            return None
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if p.suffix.lower() == ".json":
        try:
            return json.loads(text)
        except (ValueError, RecursionError):
            return None
    return load_yaml(text)


def description_files(folder: Path, root: Path, extra=()) -> list[str]:
    """The YAML and JSON files directly in folder and in each folder above it up to root (inclusive), nearest first,
    at most DESCRIPTIONS_PER_FOLDER of each folder, and extra (a LeRobot dataset's meta/info.json). Hidden files are
    left out, as every reader leaves them out."""
    folder, root = Path(folder), Path(root)
    chain = [folder]
    if folder != root and root in folder.parents:
        for up in folder.parents:
            chain.append(up)
            if up == root:
                break
    out = []
    for d in chain:
        try:
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in DESCRIPTION_SUFFIXES
                           and not p.name.startswith(".") and p.is_file())
        except OSError:
            continue
        out += [str(p) for p in files[:DESCRIPTIONS_PER_FOLDER]]
    out += [str(p) for p in extra if Path(p).is_file()]
    return list(dict.fromkeys(out))


_DESCRIPTIONS: contextvars.ContextVar = contextvars.ContextVar("argus_rig_descriptions", default=None)


@contextlib.contextmanager
def describing(paths, root=None):
    """While one episode is read (formats.convert_item), the description files that may name its arrays' values."""
    token = _DESCRIPTIONS.set({"paths": [str(p) for p in paths or ()], "root": str(root) if root else None,
                               "claims": None})
    try:
        yield
    finally:
        _DESCRIPTIONS.reset(token)


def description_claims() -> tuple[list[dict], list[str]]:
    """(the claims of the episode being read's description files, the files searched), each file read once. A file's
    claims carry its rank, how far its folder is from the recording, so a nearer description wins a tie."""
    state = _DESCRIPTIONS.get()
    if not state:
        return [], []
    if state["claims"] is None:
        claims, root = [], Path(state["root"]) if state["root"] else None
        for path in state["paths"]:
            doc = read_description(Path(path))
            if not isinstance(doc, (dict, list)):
                continue
            label = Path(path).relative_to(root).as_posix() if root and root in Path(path).parents else Path(path).name
            # below every claim the recording makes itself (rank 0), and nearer the recording above farther
            rank = len(Path(path).parents) - DESCRIPTION_RANK_FLOOR
            claims += [{**c, "rank": rank} for c in claims_in(doc, label)]
        state["claims"] = claims
    return state["claims"], [Path(p).name for p in state["paths"]]


def mcap_claims(path: Path) -> list[dict]:
    """The claims an MCAP file makes about its channels' values: its metadata records (each value read as JSON or YAML
    when it holds one), its YAML and JSON attachments, each channel's metadata (under its topic) and each JSON schema
    (under its channel's topic). A file whose summary cannot be read makes none; what its messages name is the
    reader's (formats._joint_names, mcap_signals)."""
    from mcap.reader import make_reader
    out, name = [], Path(path).name
    try:
        with open(path, "rb") as fh:
            reader = make_reader(fh)
            summary = reader.get_summary()
            if summary is None:
                return out
            for ch in summary.channels.values():
                meta = {k: decoded(v) for k, v in (ch.metadata or {}).items()}
                if meta:
                    out += [{**c, "channel": c["path"] == list(segments(ch.topic))}
                            for c in claims_in(meta, f"{name} channel metadata", (ch.topic,))]
                schema = summary.schemas.get(ch.schema_id)
                if schema is not None and schema.encoding == "jsonschema" and schema.data:
                    try:
                        doc = json.loads(bytes(schema.data).decode("utf-8", "replace"))
                    except (ValueError, RecursionError):
                        doc = None
                    if isinstance(doc, dict):
                        out += [{**c, "channel": c["path"] == list(segments(ch.topic))}
                                for c in claims_in(doc, f"{name} schema {schema.name}", (ch.topic,))]
            if summary.metadata_indexes:
                for record in reader.iter_metadata():
                    doc = {k: decoded(v) for k, v in dict(record.metadata).items()}
                    # under the record's name (a record named for a topic) and as a description of its own
                    out += claims_in({record.name: doc}, f"{name} metadata {record.name}")
                    out += claims_in(doc, f"{name} metadata {record.name}")
            if summary.attachment_indexes:
                for att in reader.iter_attachments():
                    if Path(att.name).suffix.lower() in DESCRIPTION_SUFFIXES and len(att.data) <= DESCRIPTION_BYTES:
                        doc = decoded(bytes(att.data).decode("utf-8", "replace"))
                        if isinstance(doc, (dict, list)):
                            out += claims_in(doc, f"{name} attachment {att.name}")
    except Exception:  # noqa: BLE001 - the mcap reader on upload bytes raises many types; the claims before stay
        pass
    return out


def h5_claims(f, paths) -> list[dict]:
    """The claims an HDF5 file's attributes make: the file's own and those of every group above each of paths (the
    arrays read as signals), each under its group's path. A dataset's own names attribute is read with the dataset
    (formats.h5_signals)."""
    from prepare.hdf_metadata import attribute_values
    groups = {"/"}
    for p in paths:
        parts = [x for x in str(p).split("/") if x]
        groups |= {"/" + "/".join(parts[:i]) for i in range(1, len(parts))}
    out = []
    for g in sorted(groups):
        try:
            obj = f[g]
        except Exception:  # noqa: BLE001 - h5py on upload bytes raises many types
            continue
        attrs = {k: decoded(v) for k, v in attribute_values(obj)}
        if attrs:
            out += claims_in(attrs, f"HDF5 attributes of {g}", (g,))
    return out


def dataset_names(ds, width: int) -> list[str] | None:
    """The names an HDF5 dataset's own attributes give its width values: the first key that says names and gives one
    name per value."""
    from prepare.hdf_metadata import attribute_values
    for key, value in attribute_values(ds):
        if str(key).lower() in NAME_KEYS:
            value = decoded(value)
            names = parse_names(value, text=True)
            if names and len(names) == width:
                return names
            # the dataset's own list, as it was read before declarations: one string per value, a single value's
            # one name and a repeated name included
            if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
                value = value.tolist()
            own = value if isinstance(value, (list, tuple)) else [value] if width == 1 else None
            if own is not None:
                flat = [str(_text(x)) for x in own]
                if len(flat) == width and all(isinstance(_text(x), str) for x in own):
                    return flat
    return None


UNIT_KEY = re.compile(r"^(?P<field>.+?)[_. ](?P<key>units?)$", re.I)


def _declared_units(value, width: int) -> list[str] | None:
    """The units a declaration gives width values: one unit for every value ("rad"), or one per value (a list, or a
    string holding a JSON list); None otherwise."""
    value = decoded(value)
    if isinstance(value, str) and value.strip():
        return [value.strip()] * width
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == width and all(
            isinstance(_text(u), str) and _text(u).strip() for u in value):
        return [_text(u).strip() for u in value]
    return None


def field_declaration(doc, field: str, width: int) -> tuple[list[str] | None, list[str] | None]:
    """(names, units) a mapping such as an MCAP channel's metadata declares for one field of width values: its names
    as every other claim is read (claims_in, so {field}_names, {field}.names or {field}: {names: [...]}), a single
    value's one name included, and its units from {field}_unit or {field}_units (or {field}: {unit: ...}). Either is
    None where the mapping declares none, or not one per value."""
    if not isinstance(doc, dict):
        return None, None
    doc = {str(_text(k)): decoded(v) for k, v in doc.items()}
    want = segments(field)
    names = None
    for c in claims_in(doc, ""):
        if tuple(c["path"]) == want and len(c["names"]) == width:
            names = c["names"]
            break
    if names is None and width == 1:
        for k, v in doc.items():
            if _field_of(k) is not None and segments(_field_of(k)) == want:
                one = v if isinstance(v, (list, tuple)) else [v] if isinstance(v, str) else None
                if one and len(one) == 1 and isinstance(_text(one[0]), str) and _text(one[0]).strip():
                    names = [_text(one[0]).strip()]
    units = None
    for k, v in doc.items():
        m = UNIT_KEY.match(k)
        if m and segments(m.group("field")) == want:
            units = _declared_units(v, width)
            if units:
                break
    nested = doc.get(field)
    if units is None and isinstance(nested, dict):
        units = next((u for u in (_declared_units(nested.get(k), width) for k in ("unit", "units")
                                  if nested.get(k) is not None) if u), None)
    return names, units


def declared_row(doc, fields) -> tuple[list[str] | None, list[str] | None]:
    """(names, units) a mapping declares for a row read from several fields in order, fields [(field, width)] (an
    arm's joint vector and then its gripper reading): each joined from every field's own declaration, and None unless
    every field declares one per value."""
    parts = [field_declaration(doc, field, width) for field, width in fields]
    names = [x for n, _ in parts for x in n] if parts and all(n for n, _ in parts) else None
    units = [x for _, u in parts for x in u] if parts and all(u for _, u in parts) else None
    return names, units


def _matches(path, name_segs) -> int:
    """How many segments a claim's path and an array's name share when one ends with the other, else 0: a path
    (yam_left, joint_state, joint_pos) never names /yam_right/joint_state joint_pos, whose last two segments it
    shares."""
    k = 0
    for a, b in zip(reversed(path), reversed(name_segs)):
        if a != b:
            break
        k += 1
    return k if k == min(len(path), len(name_segs)) else 0


def _channel_topic(meta: dict) -> str | None:
    source = str(meta.get("source") or "")
    return source[len("MCAP channel "):] if source.startswith("MCAP channel ") else None


def resolve(metas: dict, widths: dict, claims: list[dict]) -> list[dict]:
    """Attach declared names to every array of metas ({name: its meta}, widths {name: values per frame}) that has none:
    names and names_source (where the names were declared). Returns the issues (add_issue's arguments) for arrays that
    claims name differently."""
    issues = []
    if not claims:
        return issues
    field_claims = [c for c in claims if not c.get("channel")]
    by_topic: dict[str, list[str]] = {}
    for name, m in metas.items():
        topic = _channel_topic(m)
        if topic:
            by_topic.setdefault(topic, []).append(name)
    for name, m in metas.items():
        width = widths.get(name) or 0
        if width < 2 or (m.get("names") and len(m["names"]) == width):
            continue
        segs = segments(name)
        found = [(c, _matches(tuple(c["path"]), segs)) for c in field_claims if len(c["names"]) == width]
        found = [(c, k) for c, k in found if k > 0]
        if not found:
            topic = _channel_topic(m)
            same = [n for n in by_topic.get(topic, []) if widths.get(n) == width] if topic else []
            if topic and same == [name]:
                found = [(c, 1) for c in claims if c.get("channel") and len(c["names"]) == width
                         and tuple(c["path"]) == segments(topic)]
        if not found:
            continue
        best = max(k for _, k in found)
        found = [c for c, k in found if k == best]
        near = max(c.get("rank", 0) for c in found)
        found = [c for c in found if c.get("rank", 0) == near]
        lists = {tuple(c["names"]) for c in found}
        if len(lists) > 1:
            sources = sorted({c["source"] for c in found})
            issues.append({"kind": "value_names_conflict", "signal": name,
                           "what": f"{name}: {' and '.join(sources)} name its {width} values differently, so its "
                                   "values keep their order without names"})
            continue
        m["names"] = list(found[0]["names"])
        m["names_source"] = ", ".join(sorted({c["source"] for c in found}))
    return issues
