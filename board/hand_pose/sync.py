"""Copying the hand pose outputs from the Modal output volume to the machine, by content (modal_app.py pull).

A local file is kept only when it holds the bytes of the remote one. Each episode's outputs on the volume carry
digests.json, the SHA-256 of each file computed where it was written (modal_app.py run), and a local file is compared
with it through label/content_identity.py. Outputs written before digests.json existed are read whole and compared
with the local file's bytes, so a remote file rewritten under the same size is still copied. A file is replaced only
once its new bytes are complete and match.

This module needs only the standard library, so pull runs from the Modal client's own environment.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

OUTPUTS = ("hands2d.json", "raw.npz", "run.json")
DIGESTS = "digests.json"


def digests(out_dir: Path) -> dict:
    """{file name: SHA-256} of an episode's outputs, as written beside them on the volume."""
    result = {}
    for name in OUTPUTS:
        path = Path(out_dir) / name
        if path.is_file():
            h = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    h.update(chunk)
            result[name] = h.hexdigest()
    return result


def _read_all(read_file, path: str) -> bytes:
    return b"".join(read_file(path))


def _download(read_file, path: str, dst: Path, expected: str | None) -> bool:
    """Write the remote file at `path` to a temporary file beside dst and put it in place unless dst already holds the
    same bytes. Returns whether dst was replaced. A download that does not match the remote digest is refused."""
    from label import content_identity
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dst.name}.", dir=dst.parent)
    try:
        h = hashlib.sha256()
        with os.fdopen(fd, "wb") as out:
            for chunk in read_file(path):
                h.update(chunk)
                out.write(chunk)
        got = h.hexdigest()
        if expected is not None and got != expected:
            raise ValueError(f"{path}: downloaded bytes do not match the digest written beside them on the volume")
        try:
            if dst.is_file() and content_identity.digest(dst) == got:
                return False
        except (OSError, ValueError):
            pass
        os.replace(tmp, dst)
        tmp = None
        return True
    finally:
        if tmp is not None:
            Path(tmp).unlink(missing_ok=True)


def pull_outputs(root: Path, entries, read_file) -> dict:
    """Copy each episode's outputs under `root` and write root/index.json, which board/hands.py reads.

    entries: (volume path, size) of every file on the output volume; read_file(volume path) yields the file's bytes.
    Returns {"index": ..., "copied": [...], "kept": [...]} with the local paths, relative to root, that were copied
    and those already holding the remote bytes."""
    from label import content_identity
    root = Path(root)
    files = {}
    for path, size in entries:
        parts = path.lstrip("/").split("/")
        if len(parts) == 3 and parts[2] in (*OUTPUTS, DIGESTS):
            files.setdefault((parts[0], parts[1]), {})[parts[2]] = (path, size)
    index, copied, kept = {}, [], []
    for (ds, ep), names in sorted(files.items()):
        known = {}
        if DIGESTS in names:
            try:
                known = json.loads(_read_all(read_file, names[DIGESTS][0]))
            except ValueError:
                known = {}
            known = known if isinstance(known, dict) else {}
        for name in OUTPUTS:
            if name not in names:
                continue
            path, size = names[name]
            dst = root / ds / ep / name
            expected = known.get(name) if isinstance(known.get(name), str) else None
            if expected is not None:
                # the size only spares a read when it already differs; equal sizes are compared by content
                try:
                    same = dst.stat().st_size == size and content_identity.digest(dst) == expected
                except (OSError, ValueError):
                    same = False
                if same:
                    kept.append(f"{ds}/{ep}/{name}")
                    continue
            (copied if _download(read_file, path, dst, expected) else kept).append(f"{ds}/{ep}/{name}")
        if "hands2d.json" in names:
            index[f"{ds}/{ep}"] = {"hands2d": f"{ds}/{ep}/hands2d.json", "raw": f"{ds}/{ep}/raw.npz"}
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.json").write_text(json.dumps(dict(sorted(index.items())), indent=1))
    return {"index": index, "copied": copied, "kept": kept}
