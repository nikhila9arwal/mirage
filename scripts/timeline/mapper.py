"""Trace-to-graph mapping and data overlap row construction."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from . import (
    _color_for,
    _data_trace_name,
    _parse_trace_name,
    _short,
    _stage_trace_name,
    SCHEDULER_TYPES,
)


def _topo_sort(dag_groups, adjacency):
    """Return group-ids in topological order."""
    in_deg: Dict[int, int] = defaultdict(int)
    for succs in adjacency.values():
        for s in succs:
            in_deg[s] += 1
    queue = [g["group_id"] for g in dag_groups
             if in_deg.get(g["group_id"], 0) == 0]
    order = []
    visited: set = set()
    while queue:
        gid = queue.pop(0)
        if gid in visited:
            continue
        visited.add(gid)
        order.append(gid)
        for succ in adjacency.get(gid, []):
            in_deg[succ] -= 1
            if in_deg[succ] <= 0 and succ not in visited:
                queue.append(succ)
    return order


def map_trace_to_graph(slices, dag_groups, adjacency, reverse_adj, global_start):
    """Map trace task-groups to DAG groups and compute timing / dependency info.

    Returns
    -------
    group_deps : dict
        ``trace_name -> {p:[pred_names], s:[succ_names], rt:ready_time,
        ad:avg_dur, gs:group_start, ge:group_end}`` (all offsets from
        global_start, in nanoseconds).  Serialised directly into the HTML.
    group_timing : dict
        ``dag_group_id -> {start_time, end_time, ready_time, avg_dur}``
        (absolute nanosecond timestamps).  Used for critical path DP.
    dag_to_trace : dict
        ``dag_group_id -> (task_type_name, event_no)``
    """
    # ---- Step 1: Aggregate raw slices into per-(type, event_no) groups ----
    trace_agg: Dict[Tuple[str, int], list] = defaultdict(list)
    for name, ts, dur, track_id in slices:
        tts, eno, _ = _parse_trace_name(name)
        if tts in SCHEDULER_TYPES:
            continue
        trace_agg[(tts, eno)].append((ts, dur, track_id))

    trace_timing: Dict[Tuple[str, int], dict] = {}
    for (tts, eno), records in trace_agg.items():
        min_start = min(r[0] for r in records)
        max_end = max(r[0] + r[1] for r in records)
        durs = [r[1] for r in records]
        entry = {
            "eno": eno,
            "min_start": min_start,
            "max_end": max_end,
            "avg_dur": sum(durs) / len(durs),
        }
        trace_timing[(tts, eno)] = entry

    exact_trace_to_dag: Dict[Tuple[str, int], int] = {}
    for g in dag_groups:
        exact_trace_to_dag[(g["task_type_name"], g["trace_event_no"])] = g["group_id"]

    unknown_trace_keys = sorted(k for k in trace_timing if k not in exact_trace_to_dag)
    if unknown_trace_keys:
        preview = ", ".join(f"{tts}_{eno}" for tts, eno in unknown_trace_keys[:10])
        raise ValueError(
            "Trace contains task groups absent from the DAG mapping: "
            f"{preview}"
        )

    trace_to_dag: Dict[Tuple[str, int], int] = {
        trace_key: exact_trace_to_dag[trace_key] for trace_key in trace_timing
    }
    dag_to_trace: Dict[int, Tuple[str, int]] = {
        gid: trace_key for trace_key, gid in trace_to_dag.items()
    }

    group_timing: Dict[int, dict] = {}
    for (tts, eno), gid in trace_to_dag.items():
        tg = trace_timing[(tts, eno)]
        group_timing[gid] = {
            "start_time": tg["min_start"],
            "end_time": tg["max_end"],
            "avg_dur": tg["avg_dur"],
        }

    # Compute ready_time for group_timing via DAG predecessor end-times.
    for gid in group_timing:
        preds = reverse_adj.get(gid, [])
        pred_ends = [group_timing[p]["end_time"]
                     for p in preds if p in group_timing]
        if pred_ends:
            group_timing[gid]["ready_time"] = max(pred_ends)
        else:
            group_timing[gid]["ready_time"] = group_timing[gid]["start_time"]

    group_deps: Dict[str, dict] = {}
    for gid, trace_key in dag_to_trace.items():
        tts, eno = trace_key
        tg = trace_timing[trace_key]
        trace_name = f"{tts}_{eno}"
        my_start = tg["min_start"]
        my_end = tg["max_end"]

        pred_names: List[str] = []
        pred_ends: List[int] = []
        seen_p: set = set()
        for pred_gid in reverse_adj.get(gid, []):
            pred_trace = dag_to_trace.get(pred_gid)
            if pred_trace is None:
                continue
            pname = f"{pred_trace[0]}_{pred_trace[1]}"
            if pname in seen_p:
                continue
            pred_names.append(pname)
            pred_ends.append(trace_timing[pred_trace]["max_end"])
            seen_p.add(pname)

        succ_names: List[str] = []
        seen_s: set = set()
        for succ_gid in adjacency.get(gid, []):
            succ_trace = dag_to_trace.get(succ_gid)
            if succ_trace is None:
                continue
            sname = f"{succ_trace[0]}_{succ_trace[1]}"
            if sname in seen_s:
                continue
            succ_names.append(sname)
            seen_s.add(sname)

        ready_time = max(pred_ends) if pred_ends else my_start
        group_deps[trace_name] = {
            "p": pred_names,
            "s": succ_names,
            "rt": ready_time - global_start,
            "ad": tg["avg_dur"],
            "gs": my_start - global_start,
            "ge": my_end - global_start,
        }

    return group_deps, group_timing, dag_to_trace


def map_data_trace_to_graph(slices, data_nodes, adjacency, reverse_adj, global_start):
    """Map data-aware trace slices onto the schema-v2 per-data DAG."""
    if not data_nodes:
        return {}, {}, {}

    trace_agg: Dict[Tuple[str, int, int], list] = defaultdict(list)
    for name, ts, dur, track_id in slices:
        task_type_name, event_no, data_id = _parse_trace_name(name)
        if task_type_name in SCHEDULER_TYPES or data_id < 0:
            continue
        trace_agg[(task_type_name, event_no, data_id)].append((ts, dur, track_id))

    if not trace_agg:
        return {}, {}, {}

    trace_timing: Dict[Tuple[str, int, int], dict] = {}
    for trace_key, records in trace_agg.items():
        min_start = min(r[0] for r in records)
        max_end = max(r[0] + r[1] for r in records)
        durs = [r[1] for r in records]
        trace_timing[trace_key] = {
            "min_start": min_start,
            "max_end": max_end,
            "avg_dur": sum(durs) / len(durs),
        }

    exact_trace_to_data: Dict[Tuple[str, int, int], int] = {}
    for node in data_nodes:
        exact_trace_to_data[(
            node["task_type_name"],
            node["trace_event_no"],
            node["data_id"],
        )] = node["data_id"]

    unknown_trace_keys = sorted(k for k in trace_timing if k not in exact_trace_to_data)
    if unknown_trace_keys:
        preview = ", ".join(
            _data_trace_name(task_type, event_no, data_id)
            for task_type, event_no, data_id in unknown_trace_keys[:10]
        )
        raise ValueError(
            "Trace contains data-aware task instances absent from the DAG mapping: "
            f"{preview}"
        )

    trace_to_data: Dict[Tuple[str, int, int], int] = {
        trace_key: exact_trace_to_data[trace_key] for trace_key in trace_timing
    }
    data_to_trace: Dict[int, Tuple[str, int, int]] = {
        data_id: trace_key for trace_key, data_id in trace_to_data.items()
    }

    data_timing: Dict[int, dict] = {}
    for trace_key, data_id in trace_to_data.items():
        timing = trace_timing[trace_key]
        data_timing[data_id] = {
            "start_time": timing["min_start"],
            "end_time": timing["max_end"],
            "avg_dur": timing["avg_dur"],
        }

    for data_id, timing in data_timing.items():
        preds = reverse_adj.get(data_id, [])
        pred_ends = [
            data_timing[pred_id]["end_time"]
            for pred_id in preds
            if pred_id in data_timing
        ]
        timing["ready_time"] = max(pred_ends) if pred_ends else timing["start_time"]

    data_deps: Dict[str, dict] = {}
    for data_id, trace_key in data_to_trace.items():
        task_type_name, event_no, _ = trace_key
        trace_name = _data_trace_name(task_type_name, event_no, data_id)
        timing = data_timing[data_id]
        pred_names = [
            _data_trace_name(data_to_trace[pred_id][0], data_to_trace[pred_id][1], pred_id)
            for pred_id in reverse_adj.get(data_id, [])
            if pred_id in data_to_trace
        ]
        succ_names = [
            _data_trace_name(data_to_trace[succ_id][0], data_to_trace[succ_id][1], succ_id)
            for succ_id in adjacency.get(data_id, [])
            if succ_id in data_to_trace
        ]
        data_deps[trace_name] = {
            "p": pred_names,
            "s": succ_names,
            "rt": timing["ready_time"] - global_start,
            "ad": timing["avg_dur"],
            "gs": timing["start_time"] - global_start,
            "ge": timing["end_time"] - global_start,
            "data_id": data_id,
            "resident_task_id": data_nodes[data_id]["resident_task_id"],
        }

    return data_deps, data_timing, data_to_trace


def build_data_overlap_rows(data_nodes, data_timing, stage_order, global_start):
    """Build rows for the Data Overlap tab."""
    if not data_timing:
        return []

    rows_by_group: Dict[int, list] = defaultdict(list)
    row_meta: Dict[int, dict] = {}
    for data_id, timing in data_timing.items():
        node = data_nodes[data_id]
        group_id = int(node["trace_event_no"])
        row_meta[group_id] = {
            "task_type_name": node["task_type_name"],
            "trace_event_no": group_id,
        }
        rows_by_group[group_id].append({
            "start": timing["start_time"] - global_start,
            "end": timing["end_time"] - global_start,
            "name": node["trace_name"],
            "blocks": 1,
            "avg_dur": timing["end_time"] - timing["start_time"],
            "color": _color_for(node["task_type_name"]),
            "data_id": data_id,
            "execution_kind": int(node.get("execution_kind", 0)),
        })

    order_lookup = {group_id: idx for idx, group_id in enumerate(stage_order)}
    ordered_group_ids = sorted(
        rows_by_group,
        key=lambda group_id: (
            order_lookup.get(group_id, len(order_lookup)),
            rows_by_group[group_id][0]["start"],
            group_id,
        ),
    )

    rows = []
    for group_id in ordered_group_ids:
        bars = sorted(rows_by_group[group_id], key=lambda bar: (bar["start"], bar["name"]))
        meta = row_meta[group_id]
        rows.append({
            "label": f"{_short(meta['task_type_name'])}_{group_id} ({len(bars)} data)",
            "bars": bars,
        })
    return rows
