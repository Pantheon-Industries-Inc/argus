"""Record a scored WGO-Bench run in Langfuse: the benchmark as a dataset, the run as a dataset run.

Each episode is a dataset item (input: id, instruction, source; expected output: the reference segments). Each
episode of a run is one trace named `wgo-bench.episode`, holding one generation per model call (model, usage,
billed cost, pass), linked to its item under the run's name, and scored `wgo_semantic_f1`, `wgo_segment_f1`,
`wgo_label_accuracy` and `cost_usd`. Images are never sent: a trace records the instants each call showed.

Keys: LANGFUSE_HOST (or LANGFUSE_BASE_URL), LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY. In sx, `just langfuse` sets
them for the laptop project; the bench writes there, never into production's project.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

NS = uuid.UUID("6f1b2a0e-9d7c-4c55-9a51-1f3c2b7d8e90")


def _client():
    host = (os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL") or "").rstrip("/")
    pk = os.environ.get("LANGFUSE_LAPTOP_PUBLIC_KEY") or os.environ.get("LANGFUSE_PUBLIC_KEY")
    sk = os.environ.get("LANGFUSE_LAPTOP_SECRET_KEY") or os.environ.get("LANGFUSE_SECRET_KEY")
    if not (host and pk and sk):
        raise SystemExit("set LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (in sx: just langfuse ...)")
    auth = "Basic " + base64.b64encode(f"{pk}:{sk}".encode()).decode()

    def post(path: str, body: dict) -> dict:
        req = urllib.request.Request(host + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": auth, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode()
        return json.loads(text) if text else {}
    return post


def _attr(k: str, v) -> dict:
    if isinstance(v, bool):
        return {"key": k, "value": {"boolValue": v}}
    if isinstance(v, int):
        return {"key": k, "value": {"intValue": str(v)}}
    if isinstance(v, list):
        return {"key": k, "value": {"arrayValue": {"values": [{"stringValue": str(x)} for x in v]}}}
    return {"key": k, "value": {"stringValue": str(v)}}


def _span(trace_id: str, span_id: str, parent: str | None, name: str, start, end, attrs: dict) -> dict:
    span = {"traceId": trace_id, "spanId": span_id, "name": name, "kind": 1,
            "startTimeUnixNano": str(int(start.timestamp() * 1e9)), "endTimeUnixNano": str(int(end.timestamp() * 1e9)),
            "attributes": [_attr(k, v) for k, v in attrs.items()]}
    if parent:
        span["parentSpanId"] = parent
    return span


def _id(*parts: str) -> str:
    return str(uuid.uuid5(NS, "/".join(parts)))


def record(run_dir: Path, report: dict, gold: dict, dataset: str | None = None) -> None:
    post = _client()
    ref = report["reference"]
    dataset = dataset or f"wgo-bench-{ref}"
    post("/api/public/datasets", {"name": dataset, "description": f"WGO-Bench ({ref} reference), evaluation only "
                                  "(CC BY-NC-SA 4.0). Items are episode ids; expected output is the reference.",
                                  "metadata": {"reference": ref}})
    run_name = f"{report['run']}"
    meta = json.loads((run_dir / "run.json").read_text())
    batch = []
    now = datetime.now(timezone.utc)
    for eid, stats in report["per_episode"].items():
        item_id = _id(dataset, eid)
        g = gold[eid]
        post("/api/public/dataset-items", {"datasetName": dataset, "id": item_id,
                                           "input": {"id": eid, "instruction": g["instruction"]},
                                           "expectedOutput": {"segments": g["segments"]},
                                           "metadata": {"source": eid.split("_", 1)[0]}})
        f = run_dir / "out" / f"{eid}.json"
        if stats is None or not f.exists():
            continue
        doc = json.loads(f.read_text())
        trace_id = uuid.UUID(_id(run_name, eid)).hex
        root = trace_id[:16]
        t0 = now
        spans = [_span(trace_id, root, None, "wgo-bench.episode", now, now + timedelta(seconds=doc.get("seconds") or 0), {
            "langfuse.trace.name": "wgo-bench.episode",
            "langfuse.trace.input": json.dumps({"id": eid, "instruction": g["instruction"]}),
            "langfuse.trace.output": json.dumps({"events": doc["events"]}),
            "langfuse.trace.tags": ["wgo-bench", ref, meta["model"]],
            "langfuse.trace.metadata.model": meta["model"], "langfuse.trace.metadata.passes": str(meta["passes"]),
            "langfuse.trace.metadata.commit": meta["commit"], "langfuse.trace.metadata.reasoning": meta["reasoning"],
            "langfuse.trace.metadata.duration_s": str(doc["duration_s"])})]
        for n, c in enumerate(doc["calls"]):
            u = c.get("usage") or {}
            end = t0 + timedelta(seconds=c.get("seconds") or 0)
            spans.append(_span(trace_id, uuid.UUID(_id(trace_id, str(n))).hex[:16], root, f"pass.{c['pass']}", t0, end, {
                "langfuse.observation.type": "generation", "gen_ai.request.model": meta["model"],
                "langfuse.observation.model.name": meta["model"],
                "gen_ai.usage.input_tokens": int(u.get("prompt_tokens") or 0),
                "gen_ai.usage.output_tokens": int(u.get("completion_tokens") or 0),
                "langfuse.observation.cost_details": json.dumps({"total": c.get("cost_usd") or 0.0}),
                "langfuse.observation.metadata.images": str(c.get("images")),
                "langfuse.observation.metadata.provider": str((c.get("served") or {}).get("provider_name"))}))
            t0 = end
        post("/api/public/otel/v1/traces", {"resourceSpans": [{"resource": {"attributes": [
            _attr("service.name", "argus-bench")]}, "scopeSpans": [{"scope": {"name": "argus.bench"}, "spans": spans}]}]})
        for name, value in (("wgo_semantic_f1", stats["semantic"]["f1"]), ("wgo_segment_f1", stats["segment"]["f1"]),
                            ("wgo_label_accuracy", stats["label_accuracy"]), ("cost_usd", doc.get("cost_usd") or 0.0)):
            batch.append({"id": _id(trace_id, "score", name), "type": "score-create", "timestamp": now.isoformat(),
                          "body": {"id": _id(trace_id, "score", name), "traceId": trace_id, "name": name,
                                   "value": float(value), "dataType": "NUMERIC"}})
        if len(batch) > 200:
            post("/api/public/ingestion", {"batch": batch})
            batch = []
        post("/api/public/dataset-run-items", {"runName": run_name, "datasetItemId": item_id, "traceId": trace_id,
                                               "runDescription": f"{meta['model']} passes={meta['passes']} "
                                                                 f"reasoning={meta['reasoning']} commit={meta['commit']}",
                                               "metadata": {k: report[k] for k in ("model", "passes", "split")}})
    if batch:
        post("/api/public/ingestion", {"batch": batch})
    print(f"Langfuse: dataset {dataset}, run {run_name}")
