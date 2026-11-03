"""ROS 2 bags stored as SQLite (.db3), read by the MCAP reader.

A .db3 bag records the same topics, message types and CDR payloads as an MCAP ROS 2 bag; only the container differs.
So each bag is copied, message for message and byte for byte, into one MCAP file in a staged copy of the upload, and
the generic reader reads that file as it reads any MCAP (its cameras, its depth and infrared left out, its numbers as
signals, its text channels). Nothing is decoded or re-encoded here.

A bag is the .db3 files its metadata.yaml lists (relative_file_paths), or without one the .db3 files of one folder
that share a name up to the split index (rosbag2_2025_07_30-09_54_44_0.db3, _1.db3, ...), read in split order as one
recording. Each message type's definition comes from the bag's own message_definitions table (written since Jazzy),
else from the standard ROS 2 definitions rosbags ships. A type with neither (a vendor's own message) keeps its
channel without a schema, so the MCAP reader names it as unreadable, and the report names the type. A compressed bag, or a file that does not open as a bag, is left out and named.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

SPLIT = re.compile(r"^(?P<bag>.+)_(?P<index>\d+)\.db3$", re.I)


def _metadata(path: Path) -> dict:
    from ruamel.yaml import YAML
    data = YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    info = (data or {}).get("rosbag2_bagfile_information") if isinstance(data, dict) else None
    if not isinstance(info, dict):
        raise ValueError("it holds no rosbag2_bagfile_information")
    return info


def find_bags(root: Path, files: list[Path]) -> tuple[list[dict], list[str]]:
    """([{"name", "folder", "files", "metadata"}], what could not be read) for every .db3 bag among files."""
    db3s = [p for p in files if p.suffix.lower() == ".db3"]
    rel = lambda p: p.relative_to(root).as_posix()
    taken, bags, notes = set(), [], []
    for meta in sorted(p for p in files if p.name == "metadata.yaml"):
        try:
            info = _metadata(meta)
        except Exception as error:  # noqa: BLE001 - yaml on upload bytes; named, and its .db3 files are read by name
            notes.append(f"{rel(meta)} could not be read as a ROS 2 bag's metadata ({error}).")
            continue
        if info.get("storage_identifier") not in (None, "sqlite3"):
            continue
        listed = [meta.parent / str(f) for f in info.get("relative_file_paths") or []]
        if info.get("compression_format") or info.get("compression_mode") not in (None, "", "NONE"):
            notes.append(f"The ROS 2 bag of {rel(meta)} is compressed "
                         f"({info.get('compression_format')}, {info.get('compression_mode')}), which is not read.")
            taken.update(listed)
            taken.add(meta)
            continue
        found = [p for p in listed if p in db3s]
        if not found:
            continue
        lost = [p.name for p in listed if p not in db3s]
        if lost:
            notes.append(f"{rel(meta)} lists {', '.join(lost)}, which the upload does not hold; the bag was read "
                         "without them.")
        bags.append({"name": SPLIT.match(found[0].name).group("bag") if SPLIT.match(found[0].name) else found[0].stem,
                     "folder": meta.parent, "files": found, "metadata": meta})
        taken.update(found)
    groups: dict[tuple[Path, str], list[tuple[int, Path]]] = {}
    for p in db3s:
        if p in taken:
            continue
        m = SPLIT.match(p.name)
        key, index = (p.parent, m.group("bag")) if m else (p.parent, p.stem), int(m.group("index")) if m else 0
        groups.setdefault(key, []).append((index, p))
    for (folder, name), parts in sorted(groups.items()):
        bags.append({"name": name, "folder": folder, "files": [p for _, p in sorted(parts)], "metadata": None})
    return bags, notes


def _connect(path: Path) -> sqlite3.Connection:
    # immutable, so a bag on a read-only mount opens without a lock file; a bag with an uncheckpointed write-ahead log
    # beside it is opened read-only instead, so the messages still in the log are read
    wal = path.with_name(path.name + "-wal").exists()
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro{'' if wal else '&immutable=1'}", uri=True)


def _definitions(con: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = con.execute("SELECT topic_type, encoding, encoded_message_definition FROM message_definitions").fetchall()
    except sqlite3.Error:
        return {}
    return {t: d for t, enc, d in rows if enc == "ros2msg" and d}


def _standard_definition(name: str) -> str | None:
    from rosbags.typesys import Stores, get_typestore
    store = _standard_definition.__dict__.setdefault("store", get_typestore(Stores.ROS2_HUMBLE))
    if name not in store.fielddefs:
        return None
    return store.generate_msgdef(name, ros_version=2)[0]


def write_mcap(bag: dict, target: Path) -> list[str]:
    """Copy every message of the bag's .db3 files into one MCAP at target; returns what could not be copied."""
    from mcap.writer import CompressionType, Writer
    notes, schemas, channels, unknown = [], {}, {}, {}
    with open(target, "wb") as fh:
        writer = Writer(fh, compression=CompressionType.NONE)
        writer.start(profile="ros2", library="argus prepare.ros2_bag")
        for path in bag["files"]:
            try:
                con = _connect(path)
                defs = _definitions(con)
                topics = {}
                for tid, name, type_, fmt in con.execute(
                        "SELECT id, name, type, serialization_format FROM topics"):
                    if (name, type_) not in channels:
                        if type_ not in schemas:
                            text = defs.get(type_) or _standard_definition(type_)
                            # schema 0 is MCAP's "no schema"; an empty definition would decode as a message with no
                            # fields and lose the channel without a word
                            schemas[type_] = writer.register_schema(type_, "ros2msg", text.encode()) if text else 0
                        if not schemas[type_]:
                            unknown.setdefault(type_, []).append(name)
                        channels[name, type_] = writer.register_channel(name, fmt or "cdr", schemas[type_])
                    topics[tid] = channels[name, type_]
                n = 0
                for tid, stamp, data in con.execute("SELECT topic_id, timestamp, data FROM messages ORDER BY id"):
                    if tid in topics:
                        writer.add_message(topics[tid], log_time=stamp, data=bytes(data), publish_time=stamp,
                                           sequence=n)
                        n += 1
                con.close()
            except sqlite3.Error as error:
                notes.append(f"{path.name} stopped reading as a ROS 2 bag ({error}); the messages before that were "
                             "kept.")
        writer.finish()
    notes += [f"No definition of {type_} is recorded in the bag or known as a standard ROS 2 type, so "
              f"{', '.join(topics)} could not be decoded." for type_, topics in sorted(unknown.items())]
    return notes


def stage(root: Path, dest: Path) -> tuple[Path, list[str]]:
    """The upload's root, with each .db3 bag in it read as one MCAP: a fresh tree under dest linking every other file
    and holding the bags' MCAP files in their folders, and what was done, in words. An upload with no .db3 file is
    returned as it is."""
    from prepare.formats import files_under
    root = Path(root)
    if not root.is_dir():
        return root, []
    files = files_under(root)
    bags, notes = find_bags(root, files)
    if not bags:
        return root, notes
    import tempfile
    dest.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix="ros2-bags-", dir=dest)) / root.name
    staged.mkdir()
    consumed = {p for b in bags for p in b["files"]} | {b["metadata"] for b in bags if b["metadata"]}
    for source in files:
        if source in consumed:
            continue
        target = staged / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve())
    for bag in bags:
        target = staged / bag["folder"].relative_to(root) / f"{bag['name']}.mcap"
        target.parent.mkdir(parents=True, exist_ok=True)
        lost = write_mcap(bag, target)
        names = [p.relative_to(root).as_posix() for p in bag["files"]]
        notes.append(f"Read the ROS 2 bag {bag['name']} ({', '.join(names)}"
                     + (f", described by {bag['metadata'].relative_to(root).as_posix()}" if bag["metadata"] else "")
                     + ") as one recording." + "".join(f" {n}" for n in lost))
    return staged, notes
