"""Preparation projections must preserve the meaning of the readings they retain."""
import json

import numpy as np
import pytest

from label import episode, evidence_access, sensor_evidence
from prepare import formats


@pytest.mark.parametrize("unit_key", ["unit", "units"])
def test_mixed_process_record_preserves_sensor_component_units(tmp_path, unit_key):
    recorded = np.array([[1024, 2, 21], [2048, 7, 22], [1536, 4, 23]], dtype=np.float32)
    signals = formats.Signals()
    signals.add("/worker/pressure", recorded, names=["rss_kb", "pressure", "temperature"],
                metadata={unit_key: ["KiB", "kPa", "degC"]})
    ctx = {"state_kind": "none", "n_state_frames": 3, "fps": 10,
           "cameras": {"exo": {"name": "scene"}}}
    formats.write_signals(tmp_path, ctx, signals)
    (tmp_path / "context.json").write_text(json.dumps(ctx))
    (tmp_path / "sources.json").write_text(json.dumps({"exo": {"n_frames": 3}}))
    loaded = episode.load(tmp_path)
    access = evidence_access.Access(loaded)
    field = next(f for f in access.inventory() if f["kind"] == "numeric")
    inspection = access.inspect({"field_id": field["id"], "columns": [1, 0]})
    assert inspection["component_names"] == ["temperature", "pressure"]
    assert inspection["component_units"] == ["degC", "kPa"]
    assert inspection["values"] == [[21, 2], [22, 7], [23, 4]]
    evidence = sensor_evidence.build(loaded, {"n": 3})
    assert [(s["label"], s["unit"]) for s in evidence["series"]] == [
        ("pressure", "kPa"), ("temperature", "degC")]
    with np.load(tmp_path / "signals.npz") as saved:
        np.testing.assert_array_equal(saved["s0"], recorded)
        np.testing.assert_array_equal(saved["s0_readings"], recorded[:, 1:])


def test_summary_uses_only_finite_readings_in_each_source_row():
    recorded = np.array([[2, np.inf, 8, np.nan], [-np.inf, -6, 0, np.nan],
                         [np.nan, np.inf, -np.inf, np.nan]], dtype=np.float64)
    summary = formats.summarise_rows(recorded)
    np.testing.assert_allclose(summary, [[2, 5, 8], [-6, -3, 0], [np.nan] * 3], equal_nan=True)


def test_summary_keeps_wide_hdf_reads_within_a_small_working_set(tmp_path, monkeypatch):
    import h5py
    selected_bytes = []
    original = h5py.Dataset.__getitem__

    def read(dataset, selection):
        values = original(dataset, selection)
        selected_bytes.append(values.nbytes)
        return values

    with h5py.File(tmp_path / "wide_map.h5", "w") as source:
        recorded = source.create_dataset("pressure", shape=(5, 1024, 512), dtype="float32", fillvalue=2)
        monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
        summary = formats.summarise_rows(recorded)
    np.testing.assert_array_equal(summary, np.full((5, 3), 2))
    assert max(selected_bytes) <= 8 * 1024 * 1024


@pytest.mark.parametrize("order", ["C", "F"])
def test_native_inspection_reads_storage_order_without_changing_values(tmp_path, order):
    recorded = np.array([[[2**60 + 3, -5], [9, 4]], [[2**60 + 7, -8], [0, 17]],
                         [[2**60 + 1, -1], [6, 3]]], dtype=np.int64, order=order)
    path = tmp_path / "native.npz"
    np.savez_compressed(path, recorded=recorded)
    selected = evidence_access.array_rows(path, "recorded", [2, 0], [3, 0, 2])
    np.testing.assert_array_equal(selected, [[3, 2**60 + 1, 6], [4, 2**60 + 3, 9]])
    _, extrema = evidence_access.extrema_rows({"path": path, "key": "recorded"}, np.arange(3), [0, 3])
    assert extrema[0]["minimum"] == {"row": 2, "value": 2**60 + 1}
    assert extrema[0]["maximum"] == {"row": 1, "value": 2**60 + 7}
    assert extrema[1]["maximum"] == {"row": 1, "value": 17}


@pytest.mark.parametrize("syntax, first", [("proto2", None), ("proto3", False)])
def test_protobuf_native_presence_distinguishes_absence_from_default(tmp_path, syntax, first):
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    from mcap_protobuf.writer import Writer
    from prepare.mcap_fields import native_fields
    descriptor = descriptor_pb2.FileDescriptorProto(name="presence.proto", package="presence", syntax=syntax)
    record = descriptor.message_type.add(name="Contact")
    record.field.add(name="contact", number=1, type=8, label=1)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(descriptor)
    contact = message_factory.GetMessageClass(pool.FindMessageTypeByName("presence.Contact"))
    messages = [contact(), contact(contact=True), contact(contact=False)]
    source = tmp_path / "source.mcap"
    with Writer(str(source)) as writer:
        for index, message in enumerate(messages):
            writer.write_message("/contact", message, log_time=10**18 + index * 10**8,
                                 publish_time=10**18 + index * 10**8)
    ctx = {"state_kind": "none", "fps": 10, "cameras": {"exo": {"name": "scene"}}}
    formats.retain_mcap_fields(source, tmp_path, ctx)
    loaded = {"dir": tmp_path, "context": ctx, "state": np.zeros((3, 0)), "signals": {},
              "sources": {"exo": {"n_frames": 3}}}
    access = evidence_access.Access(loaded)
    field = next(f for f in access.inventory() if f["kind"] == "native" and
                 f["name"].split(" (", 1)[0] == "/contact contact")
    inspection = access.inspect({"field_id": field["id"]})
    assert inspection["values"] == [[first], [True], [False]]
    assert native_fields(messages[0])["contact"]["present"] is (syntax == "proto3")


@pytest.mark.parametrize("part, changed", [(False, "context"), (False, "camera"),
                                          (True, "context"), (True, "camera"), (True, "parent_context")])
def test_stitch_refuses_changed_proven_inputs_before_publishing(tmp_path, part, changed):
    from label import harness, pieces
    job = tmp_path / "job"
    source = tmp_path / "episodes" / "episode_source"
    source.mkdir(parents=True)
    target = job / "pieces" / "episode_source__p01" if part else source
    target.mkdir(parents=True, exist_ok=True)
    media = tmp_path / "camera.bin"
    media.write_bytes(b"original camera")
    context = {"state_kind": "none", "n_state_frames": 3, "fps": 10, "profile": "teleop_arms",
               "cameras": {"exo": {"name": "scene"}}}
    if part:
        context["piece"] = {"of": source.name, "index": 1, "count": 1, "t0_s": 0, "t1_s": .3}
    (target / "context.json").write_text(json.dumps(context))
    (target / "sources.json").write_text(json.dumps({"exo": {"n_frames": 3, "packed": str(media)}}))
    if part:
        (source / "context.json").write_text(json.dumps({k: v for k, v in context.items() if k != "piece"}))
        (source / "sources.json").write_text((target / "sources.json").read_text())
        parent_context = json.loads((source / "context.json").read_text())
        (target.parent / ("." + source.name + ".source_proof.json")).write_text(
            json.dumps(evidence_access.source_proof(parent_context, source)))
    identity = harness.input_identity(target, model="saved-model", reasoning="high", max_tokens=100,
                                      cell_w=0, example_dir=None)
    run_out = job / "run" / "out"
    run_out.mkdir(parents=True)
    saved = run_out / (target.name + ".json")
    saved.write_text(json.dumps({"parse_ok": True, "input_identity": identity,
                                "labels": {"timeline": [{"action": "original source claim"}]},
                                "usage": {"est_cost_usd": .5}}))
    original = saved.read_bytes()
    if changed == "context":
        (target / "context.json").write_text(json.dumps({**context, "instruction": "changed task"}))
    elif changed == "parent_context":
        (source / "context.json").write_text(json.dumps({**parent_context, "instruction": "changed full task"}))
    else:
        media.write_bytes(b"modified camera")
    final = tmp_path / "final"
    long_eps = {source.name: [target.name]} if part else {}
    with pytest.raises(RuntimeError, match="changed or missing input"):
        pieces.stitch_run(job, source.parent, long_eps, final)
    assert saved.read_bytes() == original
    assert not (final / (source.name + ".json")).exists()


def test_stitch_preserves_current_proven_and_legacy_saved_short_results(tmp_path):
    from label import harness, pieces
    source = tmp_path / "episodes" / "episode_source"
    source.mkdir(parents=True)
    (source / "context.json").write_text(json.dumps({"state_kind": "none", "fps": 10}))
    (source / "sources.json").write_text(json.dumps({"exo": {"n_frames": 3}}))
    identity = harness.input_identity(source, model="saved-model", reasoning="high", max_tokens=100,
                                      cell_w=0, example_dir=None)
    run_out = tmp_path / "job" / "run" / "out"
    run_out.mkdir(parents=True)
    saved = run_out / (source.name + ".json")
    record = {"parse_ok": True, "labels": {"task_summary": "saved historical claim"}, "input_identity": identity}
    saved.write_text(json.dumps(record))
    final = tmp_path / "final"
    pieces.stitch_run(tmp_path / "job", source.parent, {}, final)
    assert (final / saved.name).read_bytes() == saved.read_bytes()
    record.pop("input_identity")
    saved.write_text(json.dumps(record))
    (source / "context.json").write_text(json.dumps({"state_kind": "none", "instruction": "changed"}))
    pieces.stitch_run(tmp_path / "job", source.parent, {}, final)
    assert (final / saved.name).read_bytes() == saved.read_bytes()


@pytest.mark.parametrize("native_names", [False, True])
def test_inspection_preserves_exact_reviewed_state_action_interpretations(tmp_path, native_names):
    from label import dictionary
    from label.dictionary_context import apply_context, field_interpretation
    ep_dir = tmp_path / "episode_native"
    ep_dir.mkdir()
    source = {"state": "/observations/qpos", "action": "/action"} if native_names else {}
    ctx = {"episode_id": ep_dir.name, "profile": "teleop_arms", "state_kind": "joints", "fps": 30,
           "source": source, "cameras": {"exo": {"name": "scene"}}}
    values = np.arange(21, dtype=np.float64).reshape(3, 7)
    np.savez(ep_dir / "state.npz", state=values, action=values + 1)
    (ep_dir / "context.json").write_text(json.dumps(ctx))
    (ep_dir / "sources.json").write_text(json.dumps({"exo": {"n_frames": 3}}))
    inventory = dictionary.inventory([ep_dir])
    fields = {field["kind"]: field for field in inventory["fields"] if field["kind"] in ("state", "action")}
    record = {"schema": 1, "inventory_digest": inventory["digest"], "inventory": inventory, "status": "success",
              "entries": {field["id"]: {"meaning": "Machine guess", "role": "unknown", "provenance": "machine"}
                          for field in fields.values()}}
    edits = {field["id"]: {"meaning": "Reviewed " + kind, "role": "joint_state" if kind == "state" else "command",
                            "layout": [{"start": 0, "count": 7, "name": "recorded arm"}]}
             for kind, field in fields.items()}
    loaded = episode.load(ep_dir)
    loaded["context"] = apply_context(ctx, record, edits)
    access = evidence_access.Access(loaded)
    canonical = evidence_access.Access({**loaded, "context": {**ctx, "source": {}}})
    canonical_ids = {row["name"]: row["id"] for row in canonical.inventory() if row["kind"] == "numeric"}
    for field in (row for row in access.inventory() if row["kind"] == "numeric"):
        kind = field["name"]
        expected = field_interpretation(loaded["context"], source.get(kind, kind), kind)
        assert expected["provenance"] == "human"
        assert field["id"] == canonical_ids[kind]
        assert field["descriptor"]["interpretation"] == expected
        assert field["descriptor"]["source"] == source.get(kind, kind)
        inspected = access.inspect({"field_id": field["id"], "columns": [6, 0]})
        assert inspected["values"] == (values + (kind == "action"))[:, [6, 0]].tolist()
    assert all(entry["meaning"] == "Machine guess" for entry in record["entries"].values())
    changed = evidence_access.Access({**loaded, "context": {**loaded["context"],
        "source": {"state": "/different/qpos", "action": "/different/command"}}})
    assert all(not field["descriptor"]["interpretation"] for field in changed.inventory()
               if field["kind"] == "numeric")
