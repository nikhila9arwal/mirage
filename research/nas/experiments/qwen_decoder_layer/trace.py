"""Decoder-stack trace: logical slot progress and actual worker execution.

The preserved top-k validator checks branch joins and descriptor ownership. This
exporter adds attention-head-safe tracks: each DataId and each MoE branch has its
own lane, so parallel heads never become ambiguous overlapping Perfetto slices.
Logical W13/W2 lifetimes include waiting; only worker lanes measure execution.
"""
import csv
import importlib.util
import json
from pathlib import Path


# Reuse the already exercised validator from the isolated baseline. Keeping it
# unchanged makes this patch about integrating the full layer, not rewriting
# the branch-completion protocol or its checks.
BASELINE = (Path(__file__).resolve().parents[1] / "baseline" / "baseline" /
            "topk" / "topk_trace.py")
_spec = importlib.util.spec_from_file_location("decoder_topk_trace_checks", BASELINE)
_checks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_checks)


def export_trace(buffer, graph, mapping, contexts, out, expected_experts=None, layers=None):
    """Save raw evidence, validate top-8 work, then export Perfetto.

    expected_experts, when provided, is {layer_id: experts[context][branch]}. It should
    describe the GPU's own captured routing ranks, not replacement reference
    routes. The independent numerical checker compares those routes with HF.
    """
    # Names and branch operation IDs come from the actual native graph. The
    # embedding/readout have no layer; every decoder layer must be present exactly once
    # in the set, even when its numeric operation IDs exceed a 32-bit mask.
    graph_layers = {row["layer"] for row in mapping if row["layer"] is not None}
    inferred_layers = max(graph_layers) + 1
    assert graph_layers == set(range(inferred_layers)), "Missing decoder layer"
    if layers is None:
        layers = inferred_layers
    assert layers == inferred_layers and 1 <= layers <= 48
    readout_stages = ('readout_norm', 'readout_logits', 'predicted_tokens')
    readout = any(row['operator'] == 'predicted_tokens' for row in mapping)
    if readout:
        for slot, description in enumerate(graph['request_slots']):
            stages = {name: [row['data'] for row in mapping
                             if row['slot'] == slot and row['operator'] == name]
                      for name in readout_stages}
            assert all(len(ids) == 1 for ids in stages.values()), 'Missing slot readout stage'
            assert description['terminal_data_ids'] == stages['predicted_tokens'], \
                'Slot must complete after its argmax'
    out = Path(out)
    raw = buffer.cpu()
    count, overflow = int(raw[0, 0]), int(raw[0, 1])
    assert 0 < count < raw.shape[0] and overflow == 0, (
        "Empty or overflowing trace", count, overflow)
    records = raw[1:count + 1].tolist()
    with (out / "trace.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp_ns", "kind", "epoch", "task_or_slot",
                         "data", "worker", "extra", "sm", "block", "thread",
                         "reserved", "published"])
        writer.writerows(records)

    report, details = _checks.validate_records(
        records, graph, mapping, contexts, top_k=8, layers=layers, dense=False,
        expected_experts=expected_experts)
    info = {row["data"]: row for row in mapping}
    assert len(info) == len(mapping) == len(graph["all_data"])
    assert all(row["slot"] == graph["data_slot_ids"][d]
               for d, row in info.items())
    assert all(graph["data_slot_ids"][edge["src_data_id"]] ==
               graph["data_slot_ids"][edge["dst_data_id"]]
               for edge in graph["data_edges"]), "Cross-slot graph edge"
    for (epoch, data), start in details["starts"].items():
        end = details["ends"][epoch, data]
        assert start[3] == end[3] == info[data]["task"]

    origin = min(record[0] for record in records)
    events = [dict(ph="M", name="process_name", pid=1,
                   args=dict(name=f"Qwen {layers} decoder layers: attention + top-8 MoE" +
                             (' + readout' if readout else '')))]
    tracks = {}
    checked_intervals = []

    def track(key, name):
        # Allocate IDs rather than reserving numeric ranges that might overlap
        # when the number of slots, DataIds, or branches changes.
        if key not in tracks:
            tracks[key] = len(tracks) + 1
            events.append(dict(ph="M", name="thread_name", pid=1,
                               tid=tracks[key], args=dict(name=name)))
        return tracks[key]

    def span(tid, name, begin, end, **args):
        assert begin <= end
        # Check nesting with integer nanoseconds. Adjacent slices can acquire
        # tiny floating-point overlaps after conversion to JSON microseconds.
        checked_intervals.append(dict(ph="X", pid=1, tid=tid, name=name,
                                      ts=begin - origin, dur=end - begin))
        events.append(dict(ph="X", pid=1, tid=tid, name=name,
                           ts=(begin - origin) / 1000,
                           dur=(end - begin) / 1000, args=args))

    timeline = []
    for (slot, epoch), admission in sorted(details["loaded"].items()):
        saved = details["saved"][slot, epoch]
        context = admission[6]
        ids = graph["request_slots"][slot]["data_ids"]
        # All added readout tasks are ordinary tasks. Their completion must
        # precede the save, just like every decoder task in this slot epoch.
        first = min(details["starts"][epoch, d][0]
                    for d in ids if (epoch, d) in details["starts"])
        done = max(details["logical"][epoch, d][1] for d in ids)
        tid = track(("slot", slot), f"Slot {slot}: context lifetime")
        span(tid, f"context {context} / epoch {epoch}", admission[0], saved[0],
             context=context, slot=slot, epoch=epoch)
        span(tid, f"{layers} decoder layers" + (' + readout' if readout else ''),
             first, done, context=context, epoch=epoch)
        timeline.append(dict(context=context, slot=slot, epoch=epoch,
                             start_us=(first - origin) / 1000,
                             done_us=(done - origin) / 1000,
                             saved_us=(saved[0] - origin) / 1000))
        for data in ids:
            row = info[data]
            begin, end = details["logical"][epoch, data]
            tid = track(("data", data), f"Slot {slot}: {row['name']} / D{data}")
            span(tid, f"context {context} / epoch {epoch}", admission[0], saved[0])
            span(tid, row["name"], begin, end, context=context, epoch=epoch,
                 data=data, task=row["task"], resident=row["resident"])

    # Ordinary kernels and physical batch descriptors share worker lanes.
    # Every descriptor is drawn once, regardless of how many logical rows it owns.
    for (epoch, data), start in details["starts"].items():
        end = details["ends"][epoch, data]
        worker = start[5]
        tid = track(("worker", worker), f"Worker {worker}: actual execution")
        span(tid, info[data]["name"], start[0], end[0], epoch=epoch, data=data,
             slot=info[data]["slot"], worker=worker)
    batch_for_branch = {}
    for batch in details["batches"]:
        worker = batch["worker"]
        tid = track(("worker", worker), f"Worker {worker}: actual execution")
        operation = "W13" if batch["operation"] % 4 == 0 else "W2"
        span(tid, f"layer {batch['operation'] // 4} / {operation} batch {batch['sequence']}",
             batch["begin_ns"], batch["end_ns"], **batch)
        for member in batch["members"]:
            key = member["epoch"], member["data"], member["branch"]
            assert key not in batch_for_branch
            batch_for_branch[key] = batch

    for branch in details["branches"]:
        data, epoch, rank = branch["data"], branch["epoch"], branch["branch"]
        batch = batch_for_branch[epoch, data, rank]
        tid = track(("branch", data, rank),
                    f"Slot {branch['slot']}: {info[data]['name']} branch {rank}")
        span(tid, f"epoch {epoch} / expert {branch['expert']}",
             branch["ready_ns"], branch["join_ns"], **branch)
        # A branch's exact instructions are not separately timed. Show the
        # physical batch envelope, not a fabricated per-expert kernel duration.
        phases = [
            ("pending", branch["ready_ns"], branch["dispatch_ns"]),
            ("queued", branch["dispatch_ns"], batch["begin_ns"]),
            ("physical batch executing", batch["begin_ns"], batch["end_ns"]),
            ("completion awaiting controller", batch["end_ns"], branch["acquired_ns"]),
            ("waiting for other branches", branch["acquired_ns"], branch["join_ns"]),
        ]
        for name, begin, end in phases:
            if begin != end:
                span(tid, name, begin, end, context=branch["context"],
                     epoch=epoch, data=data, branch=rank, batch=batch["sequence"])

    report.update(layers=layers, readout_enabled=readout,
                  readout_compute_executions=contexts * len(readout_stages) if readout else 0,
                  trace_records=count, trace_capacity=raw.shape[0] - 1,
                  trace_overflow=overflow,
                  ordinary_compute_executions=len(details["starts"]),
                  dependencies_verified=True, independent_slots=True,
                  perfetto_tracks_validated=_checks.validate_tracks(checked_intervals))
    events.sort(key=lambda event: (event.get("ts", -1), -event.get("dur", 0)))
    (out / "trace.perfetto.json").write_text(json.dumps(
        dict(traceEvents=events, displayTimeUnit="ms")) + "\n")
    timeline.sort(key=lambda row: row["context"])
    for name, value in [("timeline.json", timeline),
                        ("branches.json", details["branches"]),
                        ("joins.json", details["joins"]),
                        ("batches.json", details["batches"]),
                        ("trace_checks.json", report)]:
        (out / name).write_text(json.dumps(value, indent=2) + "\n")
    return report
