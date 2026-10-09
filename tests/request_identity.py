"""Compare relocated episodes while checking their own source proofs."""
import copy
import hashlib
import json
from pathlib import Path

from label import episode, sensor_evidence
from label.evidence_access import source_proof


def _file_identity(path, ep_dir):
    path = Path(path)
    if not path.is_absolute():
        path = ep_dir / path
    return {"name": path.name, "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _context_identity(ctx, ep_dir, *, cli_stamp=False):
    ctx = copy.deepcopy(ctx)
    if cli_stamp:
        from prepare.cli import commit
        assert ctx["source"].pop("adapter") == "folder"
        assert ctx["source"].pop("adapter_commit") == commit()
    hdf = ctx.get("recorded_hdf5_metadata") or {}
    if hdf.get("source_path"):
        path = Path(hdf["source_path"])
        assert hdf.pop("source_mtime_ns") == path.stat().st_mtime_ns
        assert hdf["source_size"] == path.stat().st_size
        hdf["source_path"] = _file_identity(path, ep_dir)
    members = (ctx.get("represented_source_members") or {}).get("hdf5")
    if members is not None:
        relocated = {json.dumps(_file_identity(path, ep_dir), sort_keys=True): value
                     for path, value in members.items()}
        assert len(relocated) == len(members)
        ctx["represented_source_members"]["hdf5"] = relocated
    for camera in ctx.get("unshown_cameras") or []:
        if camera.get("packed"):
            camera["packed"] = _file_identity(camera["packed"], ep_dir)
    return ctx


def _manifest_identity(ep_dir):
    manifest = json.loads((ep_dir / "sources.json").read_text())
    for camera in manifest.values():
        for key in ("packed", "kmap"):
            if camera.get(key):
                camera[key] = _file_identity(camera[key], ep_dir)
    return manifest


def assert_same_model_inputs(left, right, *, cli_stamp=False):
    """Keep exact model inputs, clocks, ownership and exclusions across relocation."""
    identities = []
    for index, ep_dir in enumerate((Path(left), Path(right))):
        ctx = json.loads((ep_dir / "context.json").read_text())
        request = episode.build_request(ep_dir)
        evidence = request["sensor_evidence"]
        assert request["plan"]["sensor_evidence"] == evidence
        assert evidence["source_proof"] == source_proof(ctx, ep_dir)
        assert sensor_evidence.compatible(evidence, ctx, ep_dir)
        assert evidence["clock_digest"] == sensor_evidence._clock_digest(ctx)
        # JSON hashes include relocated paths; compare their decoded contents below.
        # All other source bytes, including readings and clocks, must match exactly.
        proofs = {row["name"]: row["sha256"] for row in evidence["source_proof"]}
        assert len(proofs) == len(evidence["source_proof"])
        proofs["prepared:context.json"] = _context_identity(
            ctx, ep_dir, cli_stamp=cli_stamp and index == 0)
        proofs["prepared:sources.json"] = _manifest_identity(ep_dir)
        normalized = copy.deepcopy(request)
        normalized["sensor_evidence"]["source_proof"] = proofs
        normalized["plan"]["sensor_evidence"]["source_proof"] = proofs
        identities.append(normalized)
    assert identities[0] == identities[1]
