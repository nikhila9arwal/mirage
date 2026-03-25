"""Trace parsing utilities extracted from display_task_graph_timeline.py."""

from typing import Tuple

from . import _parse_trace_name, SCHEDULER_TYPES


# ---------------------------------------------------------------------------
# Parse the Perfetto trace
# ---------------------------------------------------------------------------
def parse_perfetto_trace(path: str):
    from perfetto.trace_processor import TraceProcessor
    tp = TraceProcessor(file_path=path)

    # All slices
    slices = []
    for row in tp.query("SELECT name, ts, dur, track_id FROM slice ORDER BY ts"):
        slices.append((row.name, int(row.ts), int(row.dur), int(row.track_id)))

    # Track hierarchy: build mapping from slice track_id -> block name.
    # tg4perfetto creates: block_N (id=X) -> group_0 (id=X+1) -> group_0 (id=X+2)
    # Slices live on the innermost track (X+2).
    all_tracks = {}
    for row in tp.query("SELECT id, name, parent_id FROM track ORDER BY id"):
        all_tracks[int(row.id)] = (row.name, row.parent_id)

    # Map slice track_id -> human-readable block name
    track_names = {}
    for tid, (name, _) in all_tracks.items():
        if name.startswith("block_"):
            # The slice track is at tid+2 (block -> group -> group/track)
            if (tid + 2) in all_tracks:
                track_names[tid + 2] = name
            elif (tid + 1) in all_tracks:
                track_names[tid + 1] = name
    # Fallback for any unmapped slice tracks
    slice_tids = set(s[3] for s in slices)
    for tid in slice_tids:
        if tid not in track_names:
            track_names[tid] = all_tracks.get(tid, (f"track_{tid}",))[0]

    return slices, track_names


def filter_trace_slices_to_graph(slices, dag_groups, data_nodes):
    """Keep only trace slices that correspond to graph-backed stage/data nodes."""
    allowed_stage_keys = {
        (group["task_type_name"], int(group["trace_event_no"])) for group in dag_groups
    }
    allowed_data_keys = {
        (
            node["task_type_name"],
            int(node["trace_event_no"]),
            int(node["data_id"]),
        )
        for node in data_nodes
    }

    filtered = []
    dropped = 0
    for slice_entry in slices:
        name, ts, dur, track_id = slice_entry
        task_type_name, event_no, data_id = _parse_trace_name(name)
        if task_type_name in SCHEDULER_TYPES:
            filtered.append(slice_entry)
            continue
        if data_id >= 0:
            if (task_type_name, event_no, data_id) in allowed_data_keys:
                filtered.append(slice_entry)
            else:
                dropped += 1
            continue
        if (task_type_name, event_no) in allowed_stage_keys:
            filtered.append(slice_entry)
        else:
            dropped += 1
    return filtered, dropped
