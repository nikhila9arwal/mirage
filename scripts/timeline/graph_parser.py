"""Graph parsing utilities extracted from display_task_graph_timeline.py.

Pure structural parsing of Mirage task-graph JSON files — no trace or
visualisation logic lives here.
"""

import json
from collections import Counter, defaultdict
from typing import Dict, List

from . import (
    _TRACE_NAME_MAP,
    _data_trace_name,
    SCHEDULER_TYPES,
)


# ---------------------------------------------------------------------------
# Parse the task graph JSON  (for structural info / stats)
# ---------------------------------------------------------------------------
def parse_task_graph(path: str):
    with open(path) as f:
        data = json.load(f)
    schema_version = int(data.get("schema_version", 1))
    resident_tasks = data.get("resident_tasks") or []
    all_data = data.get("all_data") or []
    control_events = data.get("control_events") or data.get("all_events") or []
    if schema_version >= 2 and resident_tasks and all_data:
        return {
            "schema_version": schema_version,
            "events": control_events,
            "legacy_events": data.get("all_events") or [],
            "tasks": all_data,
            "legacy_tasks": data.get("all_tasks") or [],
            "resident_tasks": resident_tasks,
            "all_data": all_data,
            "data_edges": data.get("data_edges") or [],
            "first_data_ids": data.get("first_data_ids") or [],
            "graph_task_count": len(resident_tasks),
            "graph_data_count": len(all_data),
        }
    return {
        "schema_version": schema_version,
        "events": data["all_events"],
        "legacy_events": data["all_events"],
        "tasks": data["all_tasks"],
        "legacy_tasks": data["all_tasks"],
        "resident_tasks": [],
        "all_data": [],
        "data_edges": [],
        "first_data_ids": [],
        "graph_task_count": len(data["all_tasks"]),
        "graph_data_count": 0,
    }


def get_index_from_id(eid: int) -> int:
    return eid & 0xFFFFFFFF


BASE_EVENT = 0xFFFFFFFE
INVALID_PROFILER_GROUP_ID = 0xFFFFFFFF

UNPROFILED_TASK_TYPES = {0, 200, 201, 202, 203}


def _task_profiler_group_id(task):
    profiler_group_id = task.get("profiler_group_id")
    if profiler_group_id is None:
        return None
    profiler_group_id = int(profiler_group_id)
    if profiler_group_id == INVALID_PROFILER_GROUP_ID:
        return None
    return profiler_group_id


def build_stage_sequence(graph_data):
    """Return an ordered list of (event_idx, [task_dict, ...]) describing the
    pipeline in dependency order."""
    if graph_data["schema_version"] >= 2 and graph_data["resident_tasks"]:
        resident_tasks = graph_data["resident_tasks"]
        all_data = graph_data["all_data"]
        data_edges = graph_data["data_edges"]
        adjacency = defaultdict(set)
        reverse_adj = defaultdict(set)

        for edge in data_edges:
            src_data = int(edge["src_data_id"])
            dst_data = int(edge["dst_data_id"])
            src_resident = int(all_data[src_data]["resident_task_id"])
            dst_resident = int(all_data[dst_data]["resident_task_id"])
            if src_resident == dst_resident:
                continue
            adjacency[src_resident].add(dst_resident)
            reverse_adj[dst_resident].add(src_resident)

        depth = {}
        queue = [resident_id for resident_id in range(len(resident_tasks))
                 if not reverse_adj.get(resident_id)]
        while queue:
            resident_id = queue.pop(0)
            parent_depth = max((depth[pred] for pred in reverse_adj.get(resident_id, [])),
                               default=-1)
            depth[resident_id] = parent_depth + 1
            for succ in sorted(adjacency.get(resident_id, [])):
                if succ in depth:
                    continue
                if all(pred in depth for pred in reverse_adj.get(succ, [])):
                    queue.append(succ)

        layers = defaultdict(list)
        for resident_id, resident_task in enumerate(resident_tasks):
            layers[depth.get(resident_id, 0)].append({
                "task_type": resident_task["task_type"],
                "resident_task_id": resident_id,
                "task_count": resident_task.get("total_data_count", 0),
                "execution_kind": int(resident_task.get("execution_kind", 0)),
            })

        return [(layer_idx, layers[layer_idx]) for layer_idx in sorted(layers)]

    events = graph_data["events"]
    tasks = graph_data["tasks"]
    stages = defaultdict(list)
    for t in tasks:
        dep_ev = get_index_from_id(t["dependent_event"])
        stages[dep_ev].append(t)

    visited = set()
    ordered = []

    def visit(ev_idx):
        if ev_idx in visited or ev_idx == BASE_EVENT:
            return
        visited.add(ev_idx)
        stage_tasks = stages.get(ev_idx, [])
        if stage_tasks:
            ordered.append((ev_idx, stage_tasks))
        next_evs = set()
        for t in stage_tasks:
            trig = get_index_from_id(t["trigger_event"])
            if trig != BASE_EVENT:
                next_evs.add(trig)
        for nev in sorted(next_evs):
            visit(nev)

    root_tasks = stages.get(BASE_EVENT, [])
    if root_tasks:
        ordered.append((BASE_EVENT, root_tasks))
        for t in root_tasks:
            trig = get_index_from_id(t["trigger_event"])
            if trig != BASE_EVENT:
                visit(trig)
    else:
        visit(0)
    return ordered


def detect_layer_pattern(stage_seq):
    """Try to detect a repeating layer pattern from the task-type sequence.
    Returns the period (number of stages per layer) or 0 if none found."""
    type_seq = []
    for _, stage_tasks in stage_seq:
        types = frozenset(t["task_type"] for t in stage_tasks)
        type_seq.append(types)

    # Skip the first few stages (prologue: BEGIN, EMBEDDING, etc.)
    # and look for a repeating period among the main body.
    n = len(type_seq)
    for period in range(3, min(100, n // 2)):
        # Check if type_seq[start:start+period] repeats
        start = 0
        # Find a plausible start by looking for the first occurrence of
        # a common type set
        ok = True
        matches = 0
        for i in range(start + period, min(n, start + period * 5)):
            if type_seq[i] == type_seq[start + (i - start) % period]:
                matches += 1
            else:
                ok = False
                break
        if ok and matches >= period * 2:
            return period
    return 0


def build_dependency_dag(graph_data):
    """Build a dependency DAG from the task graph JSON.

    Uses the exact ``profiler_group_id`` embedded in the task graph JSON.
    Tasks without a profiler group id are rejected unless they are explicitly
    unprofiled runtime sentinels such as ``TASK_TERMINATE``.

    Returns
    -------
    dag_groups : list of dicts
        Each dict has: group_id, dep_event, trig_event, task_type,
        task_type_name, task_count, trace_event_no.
    adjacency : dict[group_id -> [successor group_ids]]
    reverse_adj : dict[group_id -> [predecessor group_ids]]
    """
    if graph_data["schema_version"] >= 2 and graph_data["resident_tasks"]:
        resident_tasks = graph_data["resident_tasks"]
        all_data = graph_data["all_data"]
        data_edges = graph_data["data_edges"]
        data_indices_by_resident = defaultdict(list)
        for data_idx, data_desc in enumerate(all_data):
            data_indices_by_resident[int(data_desc["resident_task_id"])].append(data_idx)

        dag_groups_by_id = {}
        resident_to_gid = {}
        for resident_id, resident_task in enumerate(resident_tasks):
            task_type = resident_task["task_type"]
            profiler_group_id = resident_task.get("profiler_group_id")
            if profiler_group_id is None or int(profiler_group_id) == INVALID_PROFILER_GROUP_ID:
                raise ValueError(
                    "Resident task graph is missing profiler_group_id for "
                    f"resident_task_id={resident_id}"
                )
            gid = int(profiler_group_id)
            resident_to_gid[resident_id] = gid
            dag_groups_by_id[gid] = {
                "group_id": gid,
                "dep_event": -1,
                "trig_event": -1,
                "task_type": task_type,
                "task_type_name": _TRACE_NAME_MAP.get(task_type, f"TASK_{task_type}"),
                "task_count": len(data_indices_by_resident.get(resident_id, [])),
                "task_indices": list(data_indices_by_resident.get(resident_id, [])),
                "trace_event_no": gid,
                "fan_in": 0,
                "fan_out": 0,
                "resident_task_id": resident_id,
                "execution_kind": int(resident_task.get("execution_kind", 0)),
            }

        adjacency_sets: Dict[int, set] = defaultdict(set)
        reverse_sets: Dict[int, set] = defaultdict(set)
        for edge in data_edges:
            src_data = int(edge["src_data_id"])
            dst_data = int(edge["dst_data_id"])
            src_resident = int(all_data[src_data]["resident_task_id"])
            dst_resident = int(all_data[dst_data]["resident_task_id"])
            if src_resident == dst_resident:
                continue
            src_gid = resident_to_gid[src_resident]
            dst_gid = resident_to_gid[dst_resident]
            adjacency_sets[src_gid].add(dst_gid)
            reverse_sets[dst_gid].add(src_gid)

        dag_groups = [dag_groups_by_id[gid] for gid in sorted(dag_groups_by_id)]
        adjacency = {gid: sorted(succs) for gid, succs in adjacency_sets.items()}
        reverse_adj = {gid: sorted(preds) for gid, preds in reverse_sets.items()}
        for group in dag_groups:
            gid = group["group_id"]
            group["fan_in"] = len(reverse_adj.get(gid, []))
            group["fan_out"] = len(adjacency.get(gid, []))
        return dag_groups, adjacency, reverse_adj

    tasks = graph_data["tasks"]
    exact_groups = defaultdict(list)
    missing_profiler_ids = Counter()

    for i, t in enumerate(tasks):
        task_type = t["task_type"]
        if task_type in UNPROFILED_TASK_TYPES:
            continue
        profiler_group_id = _task_profiler_group_id(t)
        if profiler_group_id is not None:
            exact_groups[profiler_group_id].append(i)
            continue
        missing_profiler_ids[_TRACE_NAME_MAP.get(task_type, f"TASK_{task_type}")] += 1

    if missing_profiler_ids:
        summary = ", ".join(
            f"{task_type} ({count})"
            for task_type, count in sorted(missing_profiler_ids.items())
        )
        raise ValueError(
            "Task graph is missing profiler_group_id for profiled task types: "
            f"{summary}"
        )

    dag_groups_by_id = {}
    dep_event_to_gids = defaultdict(list)

    for gid in sorted(exact_groups):
        task_indices = exact_groups[gid]
        first_task = tasks[task_indices[0]]
        dep = get_index_from_id(first_task["dependent_event"])
        tt = first_task["task_type"]
        trig = get_index_from_id(first_task["trigger_event"])
        for task_idx in task_indices[1:]:
            task = tasks[task_idx]
            assert get_index_from_id(task["dependent_event"]) == dep
            assert task["task_type"] == tt
            assert get_index_from_id(task["trigger_event"]) == trig
        tname = _TRACE_NAME_MAP.get(tt, f"TASK_{tt}")
        dag_groups_by_id[gid] = {
            "group_id": gid,
            "dep_event": dep,
            "trig_event": trig,
            "task_type": tt,
            "task_type_name": tname,
            "task_count": len(task_indices),
            "task_indices": task_indices,
            "trace_event_no": gid,
        }
        dep_event_to_gids[dep].append(gid)

    dag_groups = [dag_groups_by_id[gid] for gid in sorted(dag_groups_by_id)]

    adjacency: Dict[int, List[int]] = defaultdict(list)
    reverse_adj: Dict[int, List[int]] = defaultdict(list)

    for g in dag_groups:
        trig = g["trig_event"]
        if trig == BASE_EVENT:
            continue
        for succ_gid in dep_event_to_gids.get(trig, []):
            adjacency[g["group_id"]].append(succ_gid)
            reverse_adj[succ_gid].append(g["group_id"])

    return dag_groups, dict(adjacency), dict(reverse_adj)


def build_data_dependency_dag(graph_data):
    """Build the per-data dependency DAG for schema-v2 resident graphs."""
    if graph_data["schema_version"] < 2 or not graph_data["resident_tasks"]:
        return [], {}, {}

    resident_tasks = graph_data["resident_tasks"]
    all_data = graph_data["all_data"]
    data_edges = graph_data["data_edges"]

    data_nodes = []
    for data_id, data_desc in enumerate(all_data):
        resident_task_id = int(data_desc["resident_task_id"])
        resident_task = resident_tasks[resident_task_id]
        task_type = resident_task["task_type"]
        profiler_group_id = resident_task.get("profiler_group_id")
        if profiler_group_id is None or int(profiler_group_id) == INVALID_PROFILER_GROUP_ID:
            raise ValueError(
                "Resident task graph is missing profiler_group_id for "
                f"resident_task_id={resident_task_id}"
            )
        trace_event_no = int(profiler_group_id)
        task_type_name = _TRACE_NAME_MAP.get(task_type, f"TASK_{task_type}")
        data_nodes.append({
            "data_id": data_id,
            "resident_task_id": resident_task_id,
            "task_type": task_type,
            "task_type_name": task_type_name,
            "trace_event_no": trace_event_no,
            "trace_name": _data_trace_name(task_type_name, trace_event_no, data_id),
            "initial_predecessor_count": int(
                data_desc.get("initial_predecessor_count", 0)
            ),
            "execution_kind": int(resident_task.get("execution_kind", 0)),
        })

    adjacency_sets: Dict[int, set] = defaultdict(set)
    reverse_sets: Dict[int, set] = defaultdict(set)
    for edge in data_edges:
        src_data = int(edge["src_data_id"])
        dst_data = int(edge["dst_data_id"])
        if src_data == dst_data:
            continue
        adjacency_sets[src_data].add(dst_data)
        reverse_sets[dst_data].add(src_data)

    adjacency = {gid: sorted(succs) for gid, succs in adjacency_sets.items()}
    reverse_adj = {gid: sorted(preds) for gid, preds in reverse_sets.items()}
    return data_nodes, adjacency, reverse_adj
