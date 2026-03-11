#!/usr/bin/env python3
"""Combine a static task graph JSON with a Perfetto execution trace to produce
an interactive HTML Gantt / pipeline timeline diagram.

Usage:
    python scripts/display_task_graph_timeline.py task_graph_0.json mirage_0.perfetto-trace [-o output.html]

Outputs an interactive HTML timeline with three views:
  1. Pipeline Phases  -- trace events grouped into repeating transformer layers
  2. By Task Type     -- one row per task type, bars per invocation
  3. Per Block        -- one row per GPU block (SM), with dependency highlighting,
                        straggler detection, ready-time markers, and schedule metrics

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

Data sources
------------
  task_graph_*.json
      Static DAG exported by Mirage at graph-build time.  Each "event" is a
      GPU synchronisation point.  Each "task" has:
        dependent_event  -- the event this task WAITS for before it can run
        trigger_event    -- the event this task FIRES when it completes
      Tasks with the same (task_type, dependent_event, trigger_event) form a
      "task group" -- all blocks run one task from this group in parallel.

  *.perfetto-trace
      Perfetto binary trace captured at runtime (via tg4perfetto).
      Each slice is named "TASK_TYPE_N" where:
        TASK_TYPE  -- the task type string matching _TRACE_NAME_MAP
        N          -- a PER-BLOCK sequential counter (0, 1, 2, ...)
                     This is NOT a DAG event index!  It simply numbers the
                     invocations for that block in time order.
      Slices on different block tracks running the same N represent the same
      logical task group executing across all GPU blocks simultaneously.

Key computed data structures
----------------------------
  dag_groups     List of DAG task groups, each with task_type, dep/trig event,
                 and task_count.  Groups are nodes in the dependency graph.

  adjacency /    Forward and reverse adjacency dicts (group_id -> [group_ids])
  reverse_adj    representing the DAG dependency edges.

  group_timing   dict[dag_group_id -> {start, end, ready_time}].
                 Populated via the Nth-match approach (see below).
                 Used for: block utilization sums, critical path approximation.

  group_deps     dict[trace_name -> {p, s, rt, ad, gs, ge}]
                 Populated via temporal proximity matching (see below).
                 Used for: interactive dependency highlighting in the browser,
                 queue wait computation in the schedule metrics.

================================================================================
MAPPING CHALLENGE: PARALLEL EXECUTION CHAINS
================================================================================

Naive approach -- Nth positional match
  Sort DAG groups of type T by (dep_event, trig_event) to approximate pipeline
  execution order; sort trace groups by min_start; pair by position (0th with
  0th, 1st with 1st, etc.).

  Problem: Mirage can run MULTIPLE PARALLEL COMPUTATION CHAINS simultaneously.
  For example, speculative decoding runs a draft chain and a verify chain at
  the same time, both executing RMS_NORM → LINEAR → ATTN → ... sequences.
  This means the trace may have ~2x as many invocations of each task type as
  the DAG has groups for that type.  Temporal interleaving of invocations from
  different chains breaks the Nth-match: you end up pairing task 0 from chain A
  with task 1 from chain B, producing nonsensical predecessor relationships
  (e.g. a "predecessor" that actually ends AFTER the task started).

  The Nth-match is still used for group_timing / critical path because it gives
  a reasonable structural approximation for those aggregate metrics.

Temporal proximity matching (used for group_deps)
  Instead of positional matching, this approach:
    1. Extracts TYPE-LEVEL predecessor/successor structure from the DAG into
       type_preds / type_succs dicts (which task types depend on which others).
    2. Builds per-type sorted timing lists for binary search.
    3. For each specific trace invocation (task_type T, event_no N):
         predecessor = for each predecessor type P, find the P invocation
                       whose max_end is the LATEST one before this T's min_start.
                       (binary search: bisect_right on sorted max_end list)
         successor   = for each successor type S, find the S invocation with
                       the EARLIEST min_start after this T's max_end.
                       (binary search: bisect_left on sorted min_start list)

WHY IS THE MOST RECENT PREDECESSOR THE RIGHT ONE?
  In Mirage's execution model, task group B "waits for" event E which is
  "fired by" task group A (when all A tasks across all blocks complete).
  So the causal relationship is: event E fires → B becomes eligible → B runs.

  For a specific B invocation (trace_name B_N):
    - The predecessor A that "released" B is the A invocation that fired event E.
    - E fires when the LAST block finishes its A task -- i.e. at max_end(A_M).
    - Therefore: max_end(A_M) < min_start(B_N), and nothing ran between them.
    - Among all A invocations that ended before B_N started, the LATEST-ending
      one is the actual bottleneck -- if B was gated by some earlier A, there
      would be a large unexplained gap where B could have started but didn't.
      A well-functioning scheduler starts B as soon as E fires, so the gap
      between pred max_end and B min_start should be near-zero (just scheduling
      latency), not 100s of microseconds.

  Limitation: when parallel chains run in tight lock-step, invocations from
  different chains can interleave within a few microseconds.  In that case,
  chain A's A_0 (ending at t=99) may be picked as chain B's B_3's predecessor
  instead of chain B's A_3 (ending at t=98).  In practice this produces small
  queue-wait estimation errors but not the gross errors (>100 us off) that the
  Nth-match caused.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

from whatif_model import (
    analyze_bubbles,
    build_group_models,
    build_locality_opportunities,
    build_trace_groups,
    compute_edge_locality,
    rank_stationary_candidates,
    short_name as whatif_short_name,
    simulate_policy,
)

# ---------------------------------------------------------------------------
# Task type name / colour mappings (mirrors runtime_header.h)
# ---------------------------------------------------------------------------
_TRACE_NAME_MAP: Dict[int, str] = {
    10: "TASK_BEGIN_TASK_GRAPH",
    101: "TASK_EMBEDDING",
    102: "TASK_RMS_NORM_LINEAR",
    103: "TASK_ATTENTION_1",
    104: "TASK_ATTENTION_2",
    105: "TASK_SILU_MUL_LINEAR",
    106: "TASK_ALLREDUCE",
    107: "TASK_REDUCE",
    108: "TASK_LINEAR_WITH_RESIDUAL",
    109: "TASK_ARGMAX",
    110: "TASK_ARGMAX_PARTIAL",
    111: "TASK_ARGMAX_REDUCE",
    112: "TASK_FIND_NGRAM_PARTIAL",
    113: "TASK_FIND_NGRAM_GLOBAL",
    114: "TASK_TARGET_VERIFY_GREEDY",
    115: "TASK_SINGLE_BATCH_EXTEND_ATTENTION",
    116: "TASK_PAGED_ATTENTION_1",
    117: "TASK_PAGED_ATTENTION_2",
    118: "TASK_SILU_MUL",
    119: "TASK_RMS_NORM",
    120: "TASK_LINEAR",
    121: "TASK_IDENTITY",
    151: "TASK_LINEAR_WITH_RESIDUAL_HOPPER",
    152: "TASK_LINEAR_HOPPER",
    153: "TASK_PAGED_ATTENTION_HOPPER",
    154: "TASK_RMS_NORM_HOPPER",
    155: "TASK_LINEAR_SWAPAB_HOPPER",
    156: "TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER",
    157: "TASK_LINEAR_CUTLASS_HOPPER",
    158: "TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER",
    159: "TASK_SILU_MUL_HOPPER",
    160: "TASK_EMBEDDING_HOPPER",
    161: "TASK_MOE_W13_LINEAR_SM90",
    162: "TASK_MOE_W2_LINEAR_SM90",
    163: "TASK_SPLITK_LINEAR_SWAPAB_HOPPER",
    164: "TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER",
    200: "TASK_SCHD_TASKS",
    201: "TASK_SCHD_EVENTS",
    202: "TASK_GET_EVENT",
    203: "TASK_GET_NEXT_TASK",
    251: "TASK_SPLITK_LINEAR_SM100",
    252: "TASK_LINEAR_WITH_RESIDUAL_SM100",
    253: "TASK_LINEAR_SM100",
    254: "TASK_MOE_W13_LINEAR_SM100",
    255: "TASK_MOE_W2_LINEAR_SM100",
    257: "TASK_ATTN_SM100",
    258: "TASK_ARGMAX_REDUCE_SM100",
    259: "TASK_ARGMAX_PARTIAL_SM100",
    260: "TASK_MOE_TOPK_SOFTMAX_SM100",
    261: "TASK_MOE_MUL_SUM_ADD_SM100",
    262: "TASK_TENSOR_INIT",
    263: "TASK_PAGED_ATTENTION_SPLIT_KV_SM100",
    264: "TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100",
    265: "TASK_SAMPLING_SM100",
    301: "TASK_NVSHMEM_ALLGATHER_STRIDED_PUT",
    302: "TASK_NVSHMEM_TILE_ALLREDUCE",
}

# Short display names
_SHORT_NAMES: Dict[str, str] = {
    "TASK_BEGIN_TASK_GRAPH": "BEGIN",
    "TASK_EMBEDDING": "EMBED",
    "TASK_RMS_NORM_HOPPER": "RMS_NORM",
    "TASK_LINEAR_SWAPAB_HOPPER": "LINEAR",
    "TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER": "LINEAR+RES",
    "TASK_PAGED_ATTENTION_HOPPER": "ATTN",
    "TASK_SILU_MUL": "SILU_MUL",
    "TASK_ARGMAX_PARTIAL_SM100": "ARGMAX_P",
    "TASK_ARGMAX_REDUCE": "ARGMAX_R",
    "TASK_ARGMAX_REDUCE_SM100": "ARGMAX_R",
    "TASK_SCHD_TASKS": "SCHD_T",
    "TASK_SCHD_EVENTS": "SCHD_E",
    "TASK_GET_EVENT": "GET_EV",
    "TASK_GET_NEXT_TASK": "GET_T",
}

# Colours per task-type category (Tableau 10-ish)
TASK_COLORS: Dict[str, str] = {
    "EMBEDDING":  "#4e79a7",
    "RMS_NORM":   "#f28e2b",
    "ATTENTION":  "#e15759",
    "PAGED_ATT":  "#e15759",
    "LINEAR":     "#59a14f",
    "SILU_MUL":   "#76b7b2",
    "ARGMAX":     "#edc948",
    "REDUCE":     "#b07aa1",
    "ALLREDUCE":  "#ff9da7",
    "MOE":        "#9c755f",
    "SAMPLING":   "#bab0ac",
    "BEGIN":      "#555555",
}

SCHEDULER_TYPES = {"TASK_SCHD_TASKS", "TASK_SCHD_EVENTS", "TASK_GET_EVENT", "TASK_GET_NEXT_TASK"}


def _color_for(name: str) -> str:
    upper = name.upper()
    for prefix, color in TASK_COLORS.items():
        if prefix in upper:
            return color
    return "#cccccc"


def _short(name: str) -> str:
    return _SHORT_NAMES.get(name, name.replace("TASK_", ""))


# ---------------------------------------------------------------------------
# Parse the task graph JSON  (for structural info / stats)
# ---------------------------------------------------------------------------
def parse_task_graph(path: str):
    with open(path) as f:
        data = json.load(f)
    return data["all_events"], data["all_tasks"]


def get_index_from_id(eid: int) -> int:
    return eid & 0xFFFFFFFF


BASE_EVENT = 0xFFFFFFFE


def build_stage_sequence(events, tasks):
    """Return an ordered list of (event_idx, [task_dict, ...]) describing the
    pipeline in dependency order."""
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


def _parse_trace_name(name: str) -> Tuple[str, int]:
    """Split a Perfetto slice name into (task_type_string, event_no).

    Example: 'TASK_RMS_NORM_HOPPER_42' -> ('TASK_RMS_NORM_HOPPER', 42)

    The trailing integer is a per-block sequential invocation counter assigned
    by tg4perfetto -- it is NOT related to dep_event or trig_event in the DAG.
    """
    m = re.match(r"^(.+)_(\d+)$", name)
    if m:
        return m.group(1), int(m.group(2))
    return name, 0


# ---------------------------------------------------------------------------
# Dependency DAG, trace-to-graph mapping, and schedule metrics
# ---------------------------------------------------------------------------
_TRACE_ID_MAP: Dict[str, int] = {v: k for k, v in _TRACE_NAME_MAP.items()}


def build_dependency_dag(events, tasks):
    """Build a dependency DAG from the task graph JSON.

    A "task group" is the set of all tasks sharing the same
    (task_type, dependent_event, trigger_event) triple.  At runtime, every
    GPU block executes exactly one task from each group in parallel.

    The DAG edges are: group A -> group B  iff  A.trigger_event == B.dependent_event.
    This means B cannot start until all blocks have finished A (the trigger event fires).

    Parameters
    ----------
    events : list
        Raw event list from the task graph JSON (used only for length reporting).
    tasks : list
        Raw task list, each with keys: task_type, dependent_event, trigger_event.

    Returns
    -------
    dag_groups : list of dicts
        Each dict has: group_id, dep_event, trig_event, task_type,
        task_type_name, task_count.
    adjacency : dict[group_id -> [successor group_ids]]
    reverse_adj : dict[group_id -> [predecessor group_ids]]
    """
    group_key_to_tasks = defaultdict(list)
    for i, t in enumerate(tasks):
        dep = get_index_from_id(t["dependent_event"])
        tt = t["task_type"]
        trig = get_index_from_id(t["trigger_event"])
        group_key_to_tasks[(dep, tt, trig)].append(i)

    dag_groups = []
    dep_event_to_gids = defaultdict(list)

    for (dep, tt, trig), task_indices in group_key_to_tasks.items():
        gid = len(dag_groups)
        tname = _TRACE_NAME_MAP.get(tt, f"TASK_{tt}")
        dag_groups.append({
            "group_id": gid,
            "dep_event": dep,
            "trig_event": trig,
            "task_type": tt,
            "task_type_name": tname,
            "task_count": len(task_indices),
            "task_indices": task_indices,
        })
        dep_event_to_gids[dep].append(gid)

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


def map_trace_to_graph(slices, track_names, dag_groups, adjacency,
                       reverse_adj, global_start):
    """Map trace task-groups to DAG groups and compute timing / dependency info.

    This function runs two complementary mapping passes:

    Pass 1 -- Nth positional match (-> group_timing, dag_to_trace)
        For each task type T, sort DAG groups by (dep_event, trig_event) and
        trace groups by min_start, then pair by position.  This gives a
        reasonable structural mapping for block utilization and critical path
        approximation.  It can be wrong for workloads with parallel execution
        chains (see module docstring), but the aggregate metrics it feeds are
        robust to occasional mismatches.

    Pass 2 -- Temporal proximity match (-> group_deps)
        For each specific trace invocation (type T, event_no N):
          - Predecessor: for each predecessor TYPE P (from DAG type-level edges),
            find the specific P invocation whose max_end is the latest one that
            still precedes this T_N's min_start.  This is the P invocation that
            causally released T_N -- any P ending later hadn't finished yet
            when T_N started, so it couldn't have been T_N's actual blocker.
          - Successor: earliest-starting S invocation after this T_N's max_end.
          - Ready time: max(predecessor max_end values), i.e. the moment the last
            required dependency completed.  Queue wait = min_start - ready_time.

        Covers ALL trace invocations regardless of whether they have a DAG match.
        See module docstring for why "most recent predecessor" is the right choice
        and what the limitations are with tight parallel chains.

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
    # Each group spans from the earliest block start to the latest block end.
    # avg_dur is the mean per-block task duration (not wall span).
    trace_agg: Dict[Tuple[str, int], list] = defaultdict(list)
    for name, ts, dur, track_id in slices:
        tts, eno = _parse_trace_name(name)
        if tts in SCHEDULER_TYPES:
            continue
        trace_agg[(tts, eno)].append((ts, dur, track_id))

    by_type_trace: Dict[str, list] = defaultdict(list)
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
        by_type_trace[tts].append(entry)
        trace_timing[(tts, eno)] = entry
    # Sort each type's invocations by wall-clock start time.
    for tts in by_type_trace:
        by_type_trace[tts].sort(key=lambda x: x["min_start"])

    # ---- Step 2 (Pass 1): Nth positional match for group_timing ----
    # Sort DAG groups of each type by (dep_event, trig_event).
    # dep_event is a pipeline ordering proxy: smaller dep_event indices appear
    # earlier in the dependency chain and therefore run earlier.  This is an
    # approximation -- it breaks with parallel chains -- but is good enough for
    # the aggregate metrics (block utilization, critical path) that use it.
    by_type_dag: Dict[str, list] = defaultdict(list)
    for g in dag_groups:
        by_type_dag[g["task_type_name"]].append(g["group_id"])

    def _dag_order_key(gid):
        g = dag_groups[gid]
        dep = g["dep_event"]
        trig = g["trig_event"]
        # BASE_EVENT (0xFFFFFFFE) means "no event"; treat as 0 for sorting.
        dep_k = 0 if dep == BASE_EVENT else dep
        trig_k = 0 if trig == BASE_EVENT else trig
        return (dep_k, trig_k)

    for ttype in by_type_dag:
        by_type_dag[ttype].sort(key=_dag_order_key)

    # Pair Nth trace invocation <-> Nth DAG group (by execution order).
    trace_to_dag: Dict[Tuple[str, int], int] = {}
    dag_to_trace: Dict[int, Tuple[str, int]] = {}
    for ttype, trace_list in by_type_trace.items():
        dag_list = by_type_dag.get(ttype, [])
        for i, tg in enumerate(trace_list):
            if i < len(dag_list):
                trace_to_dag[(ttype, tg["eno"])] = dag_list[i]
                dag_to_trace[dag_list[i]] = (ttype, tg["eno"])

    # Populate group_timing from the Nth-matched trace invocations.
    group_timing: Dict[int, dict] = {}
    for (tts, eno), gid in trace_to_dag.items():
        tg = trace_timing[(tts, eno)]
        group_timing[gid] = {
            "start_time": tg["min_start"],
            "end_time": tg["max_end"],
            "avg_dur": tg["avg_dur"],
        }

    # Compute ready_time for group_timing via DAG predecessor end-times.
    # (Used only by the critical path DP in compute_schedule_metrics.)
    for gid in group_timing:
        preds = reverse_adj.get(gid, [])
        pred_ends = [group_timing[p]["end_time"]
                     for p in preds if p in group_timing]
        if pred_ends:
            group_timing[gid]["ready_time"] = max(pred_ends)
        else:
            group_timing[gid]["ready_time"] = group_timing[gid]["start_time"]

    # ---- Step 3 (Pass 2): Temporal proximity match for group_deps ----
    # Extract type-level predecessor/successor structure from the DAG.
    # We use the DAG only for WHICH TYPES depend on which other types;
    # the specific INSTANCE pairing is done by temporal proximity below.
    type_preds: Dict[str, set] = defaultdict(set)  # task type -> {predecessor types}
    type_succs: Dict[str, set] = defaultdict(set)  # task type -> {successor types}
    for gid in range(len(dag_groups)):
        g_type = dag_groups[gid]["task_type_name"]
        for sgid in adjacency.get(gid, []):
            sg_type = dag_groups[sgid]["task_type_name"]
            type_succs[g_type].add(sg_type)
            type_preds[sg_type].add(g_type)

    # Pre-sort per-type timing lists to enable O(log N) binary search.
    # by_type_max_end[P]   = [(max_end, eno), ...] sorted ascending by max_end
    # by_type_min_start[S] = [(min_start, eno), ...] sorted ascending by min_start
    by_type_max_end: Dict[str, list] = defaultdict(list)
    by_type_min_start: Dict[str, list] = defaultdict(list)
    for (tts, eno), tg in trace_timing.items():
        by_type_max_end[tts].append((tg["max_end"], eno))
        by_type_min_start[tts].append((tg["min_start"], eno))
    for tts in by_type_max_end:
        by_type_max_end[tts].sort()
        by_type_min_start[tts].sort()

    # Build group_deps for EVERY trace invocation using temporal proximity.
    # We iterate over all (tts, eno) in trace_timing (not just mapped ones),
    # so every bar in the Per Block view gets dependency / ready-time info.
    import bisect
    group_deps: Dict[str, dict] = {}
    for (tts, eno), tg in trace_timing.items():
        trace_name = f"{tts}_{eno}"
        my_start = tg["min_start"]
        my_end = tg["max_end"]

        # --- Find predecessor instance for each predecessor TYPE ---
        # bisect_right(ends, my_start) gives the insertion point after all
        # values <= my_start.  Subtract 1 to get the latest-ending predecessor
        # that completed at or before this task started.  This is the causal
        # predecessor: the one whose completion was the last required condition
        # before this task became eligible.
        pred_names: List[str] = []
        seen_p: set = set()
        for pred_type in sorted(type_preds.get(tts, [])):
            ends = [x[0] for x in by_type_max_end[pred_type]]
            idx = bisect.bisect_right(ends, my_start) - 1
            if idx >= 0:
                _, peno = by_type_max_end[pred_type][idx]
                pname = f"{pred_type}_{peno}"
                if pname not in seen_p:
                    pred_names.append(pname)
                    seen_p.add(pname)

        # --- Find successor instance for each successor TYPE ---
        # bisect_left(starts, my_end) gives the first position with value >=
        # my_end -- i.e. the earliest-starting successor after this task ends.
        succ_names: List[str] = []
        seen_s: set = set()
        for succ_type in sorted(type_succs.get(tts, [])):
            starts = [x[0] for x in by_type_min_start[succ_type]]
            idx = bisect.bisect_left(starts, my_end)
            if idx < len(starts):
                _, seno = by_type_min_start[succ_type][idx]
                sname = f"{succ_type}_{seno}"
                if sname not in seen_s:
                    succ_names.append(sname)
                    seen_s.add(sname)

        # Ready time = when all matched predecessors had finished.
        # Initialize to 0 (not my_start) so we can correctly detect the case
        # where a predecessor ends before my_start (any >0 value is a real pred).
        ready_time = 0
        for pname in pred_names:
            last_u = pname.rfind("_")
            ptup = (pname[:last_u], int(pname[last_u + 1:]))
            ptg = trace_timing.get(ptup)
            if ptg:
                ready_time = max(ready_time, ptg["max_end"])
        if ready_time == 0:
            ready_time = my_start  # no predecessors: task was ready at its own start

        # gs/ge/rt are stored as offsets from global_start (nanoseconds) so
        # JavaScript can compute queue_wait = gs - rt without knowing global_start.
        group_deps[trace_name] = {
            "p": pred_names,
            "s": succ_names,
            "rt": ready_time - global_start,
            "ad": tg["avg_dur"],
            "gs": my_start - global_start,
            "ge": my_end - global_start,
        }

    return group_deps, group_timing, dag_to_trace


def compute_schedule_metrics(slices, track_names, group_timing, dag_groups,
                             adjacency, reverse_adj, dag_to_trace,
                             global_start, total_dur, group_deps):
    """Compute schedule quality metrics.

    Parameters
    ----------
    slices : list of (name, ts, dur, track_id)
        Raw Perfetto slices -- used for block utilization (sum of raw durations).
    group_timing : dict[dag_group_id -> {start_time, end_time, avg_dur, ready_time}]
        Timing from the Nth positional match.  Used for the critical path DP
        (which is a DAG-structural computation and tolerates approximate mapping).
    group_deps : dict[trace_name -> {gs, ge, rt, ad, p, s}]
        Timing and dependency info from temporal proximity matching.  Used for
        queue wait computation because temporal proximity gives accurate
        ready_time values even with parallel execution chains.

    Returns
    -------
    dict with keys: block_util, top_waits, critical_path, summary.
    """
    # ---- Block utilization ----
    block_busy: Dict[int, int] = defaultdict(int)
    block_names: Dict[int, str] = {}
    for name, ts, dur, track_id in slices:
        tts, _ = _parse_trace_name(name)
        if tts in SCHEDULER_TYPES:
            continue
        block_busy[track_id] += dur
        if track_id not in block_names:
            block_names[track_id] = track_names.get(track_id, f"track_{track_id}")

    def _bsort(tid):
        m2 = re.match(r"block_(\d+)", block_names.get(tid, ""))
        return int(m2.group(1)) if m2 else tid

    block_util = []
    for tid in sorted(block_busy, key=_bsort):
        busy = block_busy[tid]
        util = busy / total_dur * 100 if total_dur > 0 else 0
        block_util.append({
            "name": block_names[tid],
            "util": round(util, 1),
            "busy_us": round(busy / 1e3, 1),
            "idle_us": round((total_dur - busy) / 1e3, 1),
        })

    # ---- Per-invocation queue wait (from temporal proximity data) ----
    # We use group_deps (temporal proximity match) rather than group_timing
    # (Nth positional match) because group_timing.ready_time is unreliable
    # for workloads with parallel execution chains.  group_deps.rt gives the
    # actual completion time of the predecessor that released each invocation.
    #
    # queue_wait = gs - rt
    #   gs = group min_start offset from global_start (when the first block started)
    #   rt = ready_time offset (when the last required predecessor finished)
    # A large queue_wait means the scheduler was slow to assign the task after
    # its dependencies were satisfied -- a scheduling efficiency gap.
    group_metrics = []
    for trace_name, gd in group_deps.items():
        last_u = trace_name.rfind("_")
        tts = trace_name[:last_u]
        if tts in SCHEDULER_TYPES:
            continue
        try:
            eno = int(trace_name[last_u + 1:])
        except ValueError:
            continue
        gs = gd["gs"]   # group start offset (ns from global_start)
        rt = gd["rt"]   # ready time offset  (ns from global_start)
        ge = gd["ge"]   # group end offset    (ns from global_start)
        queue_wait = max(0, gs - rt)
        spread = ge - gs
        avg_dur = gd["ad"]
        group_metrics.append({
            "name": f"{_short(tts)}_{eno}",
            "trace_key": trace_name,
            "queue_wait_us": round(queue_wait / 1e3, 1),
            "avg_dur_us": round(avg_dur / 1e3, 1),
            "spread_us": round(spread / 1e3, 1),
        })
    top_waits = sorted(group_metrics, key=lambda x: -x["queue_wait_us"])[:20]

    # ---- Critical path (longest path by actual wall time) ----
    longest: Dict[int, float] = {}
    pred_on_path: Dict[int, int] = {}

    def dp(gid):
        if gid in longest:
            return longest[gid]
        timing = group_timing.get(gid)
        if not timing:
            longest[gid] = 0
            pred_on_path[gid] = None
            return 0
        dur = timing["end_time"] - timing["start_time"]
        preds = reverse_adj.get(gid, [])
        best_pred, best_val = None, 0
        for p in preds:
            pv = dp(p)
            if pv > best_val:
                best_val = pv
                best_pred = p
        longest[gid] = best_val + dur
        pred_on_path[gid] = best_pred
        return longest[gid]

    for gid in group_timing:
        dp(gid)

    critical_path = []
    if longest:
        end_gid = max(longest, key=longest.get)
        g = end_gid
        while g is not None:
            ti = dag_to_trace.get(g)
            if ti and g in group_timing:
                dur = group_timing[g]["end_time"] - group_timing[g]["start_time"]
                critical_path.append({
                    "name": f"{_short(ti[0])}_{ti[1]}",
                    "dur_us": round(dur / 1e3, 1),
                })
            g = pred_on_path.get(g)
        critical_path.reverse()

    # ---- Summary ----
    utils = [b["util"] for b in block_util]
    waits = [gm["queue_wait_us"] for gm in group_metrics if gm["queue_wait_us"] > 0]
    cp_dur = max(longest.values()) if longest else 0

    summary = {
        "avg_util": round(sum(utils) / len(utils), 1) if utils else 0,
        "min_util": round(min(utils), 1) if utils else 0,
        "max_util": round(max(utils), 1) if utils else 0,
        "critical_path_us": round(cp_dur / 1e3, 1),
        "total_dur_us": round(total_dur / 1e3, 1),
        "num_groups": len(group_metrics),
        "avg_queue_wait_us": round(sum(waits) / len(waits), 1) if waits else 0,
        "max_queue_wait_us": round(max(waits), 1) if waits else 0,
        "p50_wait_us": round(sorted(waits)[len(waits) // 2], 1) if waits else 0,
        "p90_wait_us": round(sorted(waits)[int(len(waits) * 0.9)], 1) if waits else 0,
    }

    return {
        "block_util": block_util,
        "top_waits": top_waits,
        "critical_path": critical_path,
        "summary": summary,
    }


def _generate_sched_html(metrics):
    """Return HTML string for the Schedule Analysis section."""
    s = metrics["summary"]
    h = '<div class="sched-grid">\n'

    # Summary card
    h += '<div class="sched-card"><h4>Summary</h4>'
    h += '<p style="font-size:11px;color:#888;margin:0 0 6px">Key metrics for overall schedule quality.</p><table>\n'
    h += f'<tr><td>Avg Block Utilization</td><td><b>{s["avg_util"]}%</b></td><td style="color:#888;font-size:11px">% of wall time each block is busy (higher = better)</td></tr>\n'
    h += f'<tr><td>Min / Max Utilization</td><td>{s["min_util"]}% / {s["max_util"]}%</td><td style="color:#888;font-size:11px">Spread indicates load imbalance across blocks</td></tr>\n'
    h += f'<tr><td>Critical Path</td><td>{s["critical_path_us"]:.1f} us</td><td style="color:#888;font-size:11px">Longest dependency chain; lower bound on execution time</td></tr>\n'
    h += f'<tr><td>Total Wall Time</td><td>{s["total_dur_us"]:.1f} us</td><td style="color:#888;font-size:11px">Actual end-to-end duration</td></tr>\n'
    h += f'<tr><td>Task Groups</td><td>{s["num_groups"]}</td><td style="color:#888;font-size:11px">Distinct (task_type, invocation) groups</td></tr>\n'
    h += f'<tr><td>Avg Queue Wait</td><td>{s["avg_queue_wait_us"]:.1f} us</td><td style="color:#888;font-size:11px">Mean time tasks wait after deps satisfied before running</td></tr>\n'
    h += f'<tr><td>Max Queue Wait</td><td>{s["max_queue_wait_us"]:.1f} us</td><td style="color:#888;font-size:11px">Worst-case scheduling delay</td></tr>\n'
    h += f'<tr><td>p50 / p90 Queue Wait</td><td>{s["p50_wait_us"]:.1f} / {s["p90_wait_us"]:.1f} us</td><td style="color:#888;font-size:11px">Median and 90th percentile wait</td></tr>\n'
    h += '</table></div>\n'

    # Block utilization card
    h += '<div class="sched-card"><h4>Block Utilization</h4>'
    h += '<p style="font-size:11px;color:#888;margin:0 0 6px">Fraction of wall time each GPU block spent executing tasks vs idle. Green &gt;80%, yellow 50-80%, red &lt;50%.</p>'
    h += '<div class="util-chart">\n'
    for b in metrics["block_util"][:40]:
        color = "#59a14f" if b["util"] > 80 else "#edc948" if b["util"] > 50 else "#e15759"
        h += (f'<div class="util-row">'
              f'<span class="util-label">{b["name"]}</span>'
              f'<div class="util-track"><div class="util-fill" '
              f'style="width:{b["util"]}%;background:{color}">'
              f'{b["util"]}%</div></div></div>\n')
    h += '</div></div>\n'

    # Top queue waits card
    h += '<div class="sched-card"><h4>Top Queue Waits</h4>'
    h += '<p style="font-size:11px;color:#888;margin:0 0 6px">Task groups with the longest delay between becoming ready (all deps satisfied) and actually starting. <b>Click a row</b> to jump to and highlight that group in the Per Block view.</p>'
    h += '<table><tr><th>Group</th><th>Wait</th><th>Avg Dur</th><th>Spread</th></tr>\n'
    for w in metrics["top_waits"][:15]:
        tk = w.get("trace_key", "")
        row_attr = f' onclick="showGroupFromWait(\'{tk}\')" title="Click to highlight in Per Block view"' if tk else ""
        h += (f'<tr{row_attr}><td>{w["name"]}</td><td>{w["queue_wait_us"]:.1f} us</td>'
              f'<td>{w["avg_dur_us"]:.1f} us</td>'
              f'<td>{w["spread_us"]:.1f} us</td></tr>\n')
    h += '</table></div>\n'

    # Critical path card
    h += '<div class="sched-card"><h4>Critical Path</h4>'
    h += '<p style="font-size:11px;color:#888;margin:0 0 6px">Longest chain of dependent task groups by wall-clock time. This is the theoretical minimum execution time &mdash; no amount of parallelism can shorten it.</p>'
    h += '<div class="cp-flow">\n'
    for i, step in enumerate(metrics["critical_path"][:30]):
        if i > 0:
            h += '<span class="cp-arrow">&#8594;</span>'
        h += f'<span class="cp-node">{step["name"]}<br><small>{step["dur_us"]:.1f} us</small></span>'
    if len(metrics["critical_path"]) > 30:
        h += f'<span class="cp-arrow">&#8594;</span><span class="cp-node">... +{len(metrics["critical_path"])-30} more</span>'
    h += '</div></div>\n'

    h += '</div>\n'
    return h


# ---------------------------------------------------------------------------
# Build the three views
# ---------------------------------------------------------------------------
def build_views(slices, track_names, stage_seq):
    """Return (pipeline_rows, tasktype_rows, block_rows, global_start, total_dur, type_stats)."""

    # ---- Aggregate trace by (task_type_str, event_no) ----
    agg = defaultdict(list)
    for name, ts, dur, track_id in slices:
        tts, eno = _parse_trace_name(name)
        agg[(tts, eno)].append((ts, dur, track_id))

    entries = []  # List of dicts, sorted by start time
    for (tts, eno), records in agg.items():
        starts = [r[0] for r in records]
        ends = [r[0] + r[1] for r in records]
        durs = [r[1] for r in records]
        entries.append({
            "type": tts,
            "event_no": eno,
            "start": min(starts),
            "end": max(ends),
            "blocks": len(records),
            "avg_dur": sum(durs) / len(durs),
        })
    entries.sort(key=lambda e: e["start"])

    if not entries:
        return [], [], [], 0, 1, {}

    global_start = entries[0]["start"]
    global_end = max(e["end"] for e in entries)
    total_dur = max(1, global_end - global_start)

    # Filter out scheduler entries for the main views
    compute_entries = [e for e in entries if e["type"] not in SCHEDULER_TYPES]

    # ---- Type stats ----
    type_stats = defaultdict(lambda: {"count": 0, "total_dur": 0, "sum_avg": 0})
    for e in compute_entries:
        ts = type_stats[e["type"]]
        ts["count"] += 1
        ts["total_dur"] += e["end"] - e["start"]
        ts["sum_avg"] += e["avg_dur"]

    # ---- 1. Pipeline Phases view ----
    # Detect the layer pattern from the task graph, then group trace entries
    # into phases.  If no clear pattern, fall back to temporal bucketing.
    period = detect_layer_pattern(stage_seq) if stage_seq else 0

    pipeline_rows = _build_pipeline_rows(compute_entries, global_start, stage_seq, period)

    # ---- 2. By Task Type view ----
    type_groups = defaultdict(list)
    for e in compute_entries:
        type_groups[e["type"]].append(e)
    sorted_types = sorted(type_groups.keys(),
                          key=lambda t: type_groups[t][0]["start"])
    tasktype_rows = []
    for tts in sorted_types:
        bars = []
        for e in type_groups[tts]:
            bars.append(_bar(e, global_start, tts))
        tasktype_rows.append({"label": _short(tts), "bars": bars})

    # ---- 3. Per-Block view ----
    block_slices = defaultdict(list)
    for name, ts, dur, track_id in slices:
        tts, eno = _parse_trace_name(name)
        if tts in SCHEDULER_TYPES:
            continue
        block_slices[track_id].append((name, ts, dur, tts))

    # Sort blocks by block number (extract from "block_N")
    def _block_sort_key(tid):
        name = track_names.get(tid, "")
        m = re.match(r"block_(\d+)", name)
        return int(m.group(1)) if m else tid

    block_rows = []
    for tid in sorted(block_slices.keys(), key=_block_sort_key):
        tname = track_names.get(tid, f"track_{tid}")
        sorted_sl = sorted(block_slices[tid], key=lambda s: s[1])
        bars = []
        for name, ts, dur, tts in sorted_sl:
            bars.append({
                "start": ts - global_start,
                "end": ts + dur - global_start,
                "name": name,
                "blocks": 1,
                "avg_dur": dur,
                "color": _color_for(tts),
            })
        block_rows.append({"label": tname, "bars": bars})

    return pipeline_rows, tasktype_rows, block_rows, global_start, total_dur, dict(type_stats)


def _bar(entry, global_start, tts_hint=""):
    return {
        "start": entry["start"] - global_start,
        "end": entry["end"] - global_start,
        "name": f'{entry["type"]}_{entry["event_no"]}',
        "blocks": entry["blocks"],
        "avg_dur": entry["avg_dur"],
        "color": _color_for(tts_hint or entry["type"]),
    }


def _build_pipeline_rows(compute_entries, global_start, stage_seq, period):
    """Build pipeline rows by detecting repeating phases.

    Uses the task graph to identify which task type marks the start of each
    pipeline layer (typically RMS_NORM).  When that type re-appears in the
    trace with a new event_no, a new phase row begins.
    """
    if not compute_entries:
        return []

    # ---------- Identify the layer-boundary task type ----------
    # Use the task graph: find the first *repeating* compute stage type.
    # In a typical LLM pipeline this is RMS_NORM (appears at the start of
    # each transformer layer).
    layer_start_type = None
    if stage_seq:
        type_counts = Counter()
        for _, stasks in stage_seq:
            for t in stasks:
                tt = t["task_type"]
                if tt in (0, 200, 201, 202, 203):
                    continue
                type_counts[tt] += 1
        # Pick the repeating type that appears earliest in the sequence
        # (skip the very first stage which may be prologue).
        seen_once = set()
        for _, stasks in stage_seq:
            stage_types = set(t["task_type"] for t in stasks
                              if t["task_type"] not in (0, 200, 201, 202, 203))
            for tt in stage_types:
                if tt in seen_once and type_counts[tt] > 5:
                    tname = _TRACE_NAME_MAP.get(tt)
                    if tname:
                        layer_start_type = tname
                        break
                seen_once.add(tt)
            if layer_start_type:
                break

    # Fallback heuristic
    if not layer_start_type:
        type_freq = Counter(e["type"] for e in compute_entries)
        for e in compute_entries:
            if "RMS_NORM" in e["type"] and type_freq[e["type"]] > 5:
                layer_start_type = e["type"]
                break

    if not layer_start_type:
        # No repeating pattern found — fall back to temporal chunking
        n_phases = min(80, max(10, len(compute_entries) // 8))
        chunk = max(1, len(compute_entries) // n_phases)
        rows = []
        for i in range(0, len(compute_entries), chunk):
            phase_entries = compute_entries[i:i + chunk]
            label = f"Phase {i // chunk}"
            bars = [_bar(e, global_start) for e in phase_entries]
            rows.append({"label": label, "bars": bars})
        return rows

    # ---------- Split trace entries into phases ----------
    # A new phase starts each time we see `layer_start_type` with a LATER
    # start time than the previous occurrence (handles overlapping entries).
    phases: List[List] = []
    current_phase: List = []
    last_split_eno = -1

    for e in compute_entries:
        if (e["type"] == layer_start_type
                and e["event_no"] != last_split_eno
                and current_phase):
            phases.append(current_phase)
            current_phase = []
            last_split_eno = e["event_no"]
        current_phase.append(e)
    if current_phase:
        phases.append(current_phase)

    rows = []
    for idx, phase_entries in enumerate(phases):
        types_here = Counter(e["type"] for e in phase_entries)
        types_str = " + ".join(_short(t) for t, _ in types_here.most_common(3))
        label = f"L{idx}: {types_str}"
        bars = [_bar(e, global_start) for e in phase_entries]
        rows.append({"label": label, "bars": bars})
    return rows


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
def generate_html(pipeline_rows, tasktype_rows, block_rows,
                  global_start, total_dur, type_stats,
                  graph_info, group_deps, sched_metrics, output_path: str):
    num_events, num_tasks = graph_info

    # Build legend from actually-used colors
    legend_items = []
    seen_colors = set()
    for rows in [tasktype_rows]:
        for row in rows:
            for bar in row.get("bars", []):
                c = bar["color"]
                if c not in seen_colors:
                    seen_colors.add(c)
                    tts, _ = _parse_trace_name(bar["name"])
                    legend_items.append({"name": _short(tts), "color": c})

    # Stats table rows
    stats_rows_html = ""
    total_dur_sum = sum(v["total_dur"] for v in type_stats.values())
    for tts, st in sorted(type_stats.items(), key=lambda x: -x[1]["total_dur"]):
        avg = st["total_dur"] / st["count"] if st["count"] else 0
        pct = st["total_dur"] / total_dur_sum * 100 if total_dur_sum else 0
        stats_rows_html += (
            f'<tr><td>{_short(tts)}</td><td>{st["count"]}</td>'
            f'<td>{st["total_dur"]/1e6:.3f} ms</td>'
            f'<td>{avg/1e3:.1f} us</td>'
            f'<td>{pct:.1f}%</td></tr>\n'
        )

    sched_html = _generate_sched_html(sched_metrics) if sched_metrics else ""

    html = _HTML_TEMPLATE.format(
        num_events=num_events,
        num_tasks=num_tasks,
        total_dur_ms=total_dur / 1e6,
        stats_rows=stats_rows_html,
        sched_analysis=sched_html,
        js_global_start=global_start,
        js_total_dur=total_dur,
        js_pipeline=json.dumps(pipeline_rows),
        js_tasktype=json.dumps(tasktype_rows),
        js_blocks=json.dumps(block_rows),
        js_legend=json.dumps(legend_items),
        js_group_deps=json.dumps(group_deps or {}),
    )

    with open(output_path, "w") as f:
        f.write(html)
    print(f"  -> {output_path}")


# ---------------------------------------------------------------------------
# Static SVG fallback
# ---------------------------------------------------------------------------
def generate_svg(tasktype_rows, global_start, total_dur, output_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if not tasktype_rows:
        print("  No data for SVG.")
        return

    fig_height = max(4, len(tasktype_rows) * 0.6)
    fig, ax = plt.subplots(figsize=(24, fig_height))

    y_labels = []
    for y, row in enumerate(tasktype_rows):
        y_labels.append(row["label"])
        for bar in row["bars"]:
            start_ms = bar["start"] / 1e6
            dur_ms = (bar["end"] - bar["start"]) / 1e6
            ax.barh(y, dur_ms, left=start_ms, height=0.7,
                    color=bar["color"], edgecolor="none", alpha=0.85)

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)")
    ax.set_title("Mirage Execution Timeline – By Task Type")
    ax.grid(axis="x", alpha=0.3)

    seen = set()
    patches = []
    for row in tasktype_rows:
        for bar in row["bars"]:
            c = bar["color"]
            if c not in seen:
                seen.add(c)
                tts, _ = _parse_trace_name(bar["name"])
                patches.append(mpatches.Patch(color=c, label=_short(tts)))
    ax.legend(handles=patches, loc="upper right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  -> {output_path}")


# ---------------------------------------------------------------------------
# What-if HTML
# ---------------------------------------------------------------------------
def _ensure_bar_colors(rows):
    colored_rows = []
    for row in rows:
        bars = []
        for bar in row.get("bars", []):
            task_type, _ = _parse_trace_name(bar["name"])
            bars.append({
                "start": bar["start"],
                "end": bar["end"],
                "name": bar["name"],
                "blocks": bar.get("blocks", 1),
                "avg_dur": bar.get("avg_dur", bar["end"] - bar["start"]),
                "color": bar.get("color", _color_for(task_type)),
            })
        colored_rows.append({"label": row["label"], "bars": bars})
    return colored_rows


def _metric_card(title: str, value: str, hint: str) -> str:
    return (
        '<div class="wf-card">'
        f'<div class="wf-card-title">{title}</div>'
        f'<div class="wf-card-value">{value}</div>'
        f'<div class="wf-card-hint">{hint}</div>'
        '</div>'
    )


def _summary_grid(cards: List[str]) -> str:
    return '<div class="wf-card-grid">' + "".join(cards) + "</div>"


def _render_table(title: str, subtitle: str, headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        rows_html = '<tr><td colspan="{0}">No data.</td></tr>'.format(len(headers))
    else:
        rows_html = "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
            for row in rows
        )
    head_html = "".join(f"<th>{header}</th>" for header in headers)
    return (
        '<div class="wf-table-card">'
        f"<h4>{title}</h4>"
        f'<p class="wf-subtle">{subtitle}</p>'
        "<table>"
        f"<tr>{head_html}</tr>"
        f"{rows_html}"
        "</table>"
        "</div>"
    )


def _baseline_same_block_ratio(locality_rows):
    total = sum(item["overlap_bytes"] for item in locality_rows if item["overlap_bytes"] > 0)
    same = sum(
        item["overlap_bytes"] * item["same_block_ratio"]
        for item in locality_rows if item["overlap_bytes"] > 0
    )
    return same / total if total else 0.0


def _whatif_legend(rows_per_tab):
    seen = {}
    for rows in rows_per_tab:
        for row in rows:
            for bar in row.get("bars", []):
                task_type, _ = _parse_trace_name(bar["name"])
                seen.setdefault(bar["color"], whatif_short_name(task_type))
    return [{"color": color, "name": name} for color, name in seen.items()]


def generate_whatif_html(whatif_data: dict, output_path: str):
    legend = _whatif_legend([
        whatif_data["baseline_rows"],
        whatif_data["affinity_rows"],
    ])
    html = _WHATIF_HTML_TEMPLATE.format(
        total_dur_ms=whatif_data["baseline_wall_us"] / 1000.0,
        js_baseline=json.dumps(whatif_data["baseline_rows"]),
        js_affinity=json.dumps(whatif_data["affinity_rows"]),
        js_legend=json.dumps(legend),
        js_baseline_dur=whatif_data["baseline_wall_ns"],
        js_affinity_dur=whatif_data["affinity_wall_ns"],
        baseline_cards=whatif_data["baseline_cards"],
        baseline_tables=whatif_data["baseline_tables"],
        affinity_cards=whatif_data["affinity_cards"],
        affinity_tables=whatif_data["affinity_tables"],
    )
    with open(output_path, "w") as output_file:
        output_file.write(html)
    print(f"  -> {output_path}")


def build_whatif_report(tasks,
                        events,
                        slices,
                        track_names,
                        dag_groups,
                        adjacency,
                        reverse_adj,
                        group_deps,
                        dag_to_trace,
                        global_start,
                        total_dur,
                        block_rows,
                        sched_metrics,
                        locality_cap,
                        l2_factor,
                        sm_tau_scale,
                        l2_tau_scale,
                        bandwidth_scale,
                        top_candidates,
                        split_factors,
                        target_types):
    trace_to_dag = {trace: gid for gid, trace in dag_to_trace.items()}
    trace_groups = build_trace_groups(
        slices, track_names, group_deps, global_start, SCHEDULER_TYPES)
    edge_locality = compute_edge_locality(dag_groups, tasks, adjacency)
    locality_rows = build_locality_opportunities(
        trace_groups, trace_to_dag, edge_locality, locality_cap, l2_factor)
    bubble_rows = analyze_bubbles(trace_groups)
    stationary_rows = rank_stationary_candidates(locality_rows)
    num_blocks = len(block_rows)
    group_models = build_group_models(
        tasks,
        events,
        dag_groups,
        adjacency,
        reverse_adj,
        trace_groups,
        dag_to_trace,
        edge_locality,
        locality_rows,
    )
    analytical_baseline = simulate_policy(
        group_models,
        edge_locality,
        num_blocks,
        policy="baseline",
        locality_cap=locality_cap,
        l2_factor=l2_factor,
        sm_tau_scale=sm_tau_scale,
        l2_tau_scale=l2_tau_scale,
        bandwidth_scale=bandwidth_scale,
    )

    affinity_sim = simulate_policy(
        group_models,
        edge_locality,
        num_blocks,
        policy="affinity",
        locality_cap=locality_cap,
        l2_factor=l2_factor,
        sm_tau_scale=sm_tau_scale,
        l2_tau_scale=l2_tau_scale,
        bandwidth_scale=bandwidth_scale,
    )
    analytical_wall_us = analytical_baseline["wall_time_ns"] / 1e3

    baseline_summary = sched_metrics["summary"]
    baseline_same_block_ratio = _baseline_same_block_ratio(locality_rows)
    baseline_cards = _summary_grid([
        _metric_card("Observed wall time", f"{total_dur / 1e3:.1f} us", "Current trace end-to-end duration."),
        _metric_card("MPK-like wall time", f"{analytical_wall_us:.1f} us", "FIFO ready queue plus round-robin placement with resource and memory constraints."),
        _metric_card("MPK-like avg util", f"{analytical_baseline['avg_util']:.1f}%", "Predicted SM utilization under the FIFO / round-robin baseline."),
        _metric_card("MPK-like avg queue wait", f"{analytical_baseline['avg_queue_wait_us']:.1f} us", "Ready-to-start delay in the FIFO / round-robin baseline."),
        _metric_card("Trace avg queue wait", f"{baseline_summary['avg_queue_wait_us']:.1f} us", "Observed delay after deps are satisfied."),
        _metric_card("Same-block reuse ratio", f"{baseline_same_block_ratio * 100:.1f}%", "Bytes with overlap that already stay on the same block."),
        _metric_card("Baseline delta", f"{analytical_wall_us - total_dur / 1e3:+.1f} us", "MPK-like analytical wall time minus observed trace wall time."),
    ])
    baseline_tables = (
        '<div class="wf-table-grid">'
        + _render_table(
            "Bubble hot spots",
            "Largest idle gaps before a task starts on a block.",
            ["Group", "Total gap", "Max gap", "Blocks"],
            [[row["name"], f"{row['total_gap_us']:.1f} us", f"{row['max_gap_us']:.1f} us", str(row["affected_blocks"])]
             for row in bubble_rows[:12]],
        )
        + _render_table(
            "Locality opportunities",
            "Producer-consumer edges ranked by overlap, time decay, and current same-block placement.",
            ["Edge", "Eff reuse", "Overlap", "Reuse delay", "Same-block"],
            [[f"{row['pred_type']} -> {row['succ_type']}",
              f"{row['effective_reuse']:.2f}",
              f"{row['overlap_bytes'] / 1024:.1f} KiB",
              f"{row['reuse_delay_us']:.1f} us",
              f"{row['same_block_ratio'] * 100:.0f}%"]
             for row in locality_rows[:12]],
        )
        + _render_table(
            "Stationary-SM candidates",
            "Short chains that look promising for keeping work fixed per SM and moving data through it.",
            ["Chain", "Overlap score", "Reuse delay", "Risk", "Recommendation"],
            [[row["chain"], f"{row['overlap_score']:.2f}", f"{row['reuse_delay_us']:.1f} us",
              row["resource_risk"], row["recommendation"]]
             for row in stationary_rows[:10]],
        )
        + _render_table(
            "Baseline bottlenecks",
            "Groups that dominate wait or span in the FIFO / round-robin baseline schedule.",
            ["Group", "Max wait", "Avg wait", "Span"],
            [[row["name"], f"{row['max_wait_us']:.1f} us", f"{row['avg_wait_us']:.1f} us", f"{row['span_us']:.1f} us"]
             for row in analytical_baseline["bottlenecks"][:12]],
        )
        + "</div>"
    )

    affinity_wall_us = affinity_sim["wall_time_ns"] / 1e3
    affinity_delta_us = affinity_wall_us - analytical_wall_us
    affinity_delta_wait_us = affinity_sim["avg_queue_wait_us"] - analytical_baseline["avg_queue_wait_us"]
    affinity_cards = _summary_grid([
        _metric_card("Predicted wall time", f"{affinity_wall_us:.1f} us", "Analytical schedule with locality-aware placement."),
        _metric_card("Predicted avg util", f"{affinity_sim['avg_util']:.1f}%", "Predicted SM utilization under affinity."),
        _metric_card("Predicted avg queue wait", f"{affinity_sim['avg_queue_wait_us']:.1f} us", "Average ready-to-start delay under affinity."),
        _metric_card("Predicted max queue wait", f"{affinity_sim['max_queue_wait_us']:.1f} us", "Worst bubble under affinity."),
        _metric_card("Same-block / L2 reuse", f"{affinity_sim['same_block_reuse_ratio'] * 100:.1f}% / {affinity_sim['l2_reuse_ratio'] * 100:.1f}%", "Reuse bytes that hit same-SM hot state or warm L2."),
        _metric_card("Delta vs MPK-like baseline", f"{affinity_delta_us:+.1f} us / {affinity_delta_wait_us:+.1f} us", "Wall-time delta and avg queue-wait delta."),
    ])
    affinity_tables = (
        '<div class="wf-table-grid">'
        + _render_table(
            "Top bottleneck groups",
            "Groups with the largest predicted queue wait or span under affinity.",
            ["Group", "Max wait", "Avg wait", "Span"],
            [[row["name"], f"{row['max_wait_us']:.1f} us", f"{row['avg_wait_us']:.1f} us", f"{row['span_us']:.1f} us"]
             for row in affinity_sim["bottlenecks"][:12]],
        )
        + _render_table(
            "Highest-value locality edges",
            "Edges most likely to benefit when successor placement follows the producer block.",
            ["Edge", "Gain same", "Gain L2", "Risk"],
            [[f"{row['pred_type']} -> {row['succ_type']}",
              f"{row['predicted_gain_same'] * 100:.1f}%",
              f"{row['predicted_gain_diff'] * 100:.1f}%",
              row["resource_risk"]]
             for row in locality_rows[:12]],
        )
        + "</div>"
    )

    return {
        "baseline_rows": _ensure_bar_colors(analytical_baseline["block_rows"]),
        "affinity_rows": _ensure_bar_colors(affinity_sim["block_rows"]),
        "baseline_wall_us": analytical_wall_us,
        "baseline_wall_ns": analytical_baseline["wall_time_ns"],
        "affinity_wall_ns": affinity_sim["wall_time_ns"],
        "baseline_cards": baseline_cards,
        "baseline_tables": baseline_tables,
        "affinity_cards": affinity_cards,
        "affinity_tables": affinity_tables,
    }


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Mirage Task Graph Execution Timeline</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;margin:0;padding:20px;background:#1a1a2e;color:#e0e0e0}}
h1{{color:#e0e0e0;margin-bottom:5px}}
.sub{{color:#888;margin-bottom:16px;font-size:14px}}
.ctrls{{margin-bottom:12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
.ctrls button{{background:#16213e;color:#e0e0e0;border:1px solid #0f3460;padding:6px 14px;border-radius:4px;cursor:pointer;font-size:13px}}
.ctrls button:hover{{background:#0f3460}}
.ctrls button.active{{background:#e94560;border-color:#e94560}}
.tab{{display:none}}.tab.active{{display:block}}
.tc{{overflow:auto;max-height:82vh;border:1px solid #0f3460;border-radius:6px;background:#16213e}}
.tl{{position:relative;min-width:100%}}
.row{{display:flex;align-items:center;border-bottom:1px solid #1a1a2e;min-height:26px}}
.row:hover{{background:#1a1a3e}}
.rl{{min-width:220px;max-width:220px;padding:2px 8px;font-size:11px;font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;background:#16213e;position:sticky;left:0;z-index:2;border-right:1px solid #0f3460}}
.rb{{position:relative;flex:1;height:26px}}
.bar{{position:absolute;height:20px;top:3px;border-radius:3px;opacity:.85;cursor:pointer;min-width:1px}}
.bar:hover{{opacity:1;outline:2px solid #fff;z-index:10}}
.leg{{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:14px;padding:10px;background:#16213e;border-radius:6px}}
.li{{display:flex;align-items:center;gap:5px;font-size:12px}}
.lc{{width:14px;height:14px;border-radius:3px;flex-shrink:0}}
.ta{{display:flex;position:sticky;top:0;z-index:3;background:#16213e;border-bottom:2px solid #0f3460}}
.ta .rl{{font-weight:bold;font-size:12px}}
.tk{{position:absolute;top:0;font-size:10px;color:#888;transform:translateX(-50%);white-space:nowrap}}
.st{{margin-top:15px;padding:10px;background:#16213e;border-radius:6px;font-size:13px}}
.st table{{border-collapse:collapse;width:100%}}
.st th,.st td{{padding:4px 10px;text-align:left;border-bottom:1px solid #1a1a2e}}
.st th{{color:#e94560}}
.zoom-ctrl{{display:flex;gap:6px;align-items:center;margin-left:auto}}
.zoom-ctrl button{{padding:4px 10px;font-size:14px}}
.zoom-ctrl span{{font-size:12px;color:#888}}
/* straggler glow */
.bar.straggler{{box-shadow:0 0 6px 2px rgba(233,69,96,.7);border:1px solid #e94560;z-index:4}}
/* ready-time marker (shown on click) */
.ready-mk{{position:absolute;top:0;width:2px;height:20px;z-index:3;pointer-events:none}}
/* dependency highlights */
.bar.dep-hl{{outline:2px solid #fff;z-index:9;opacity:1!important}}
.bar.dep-pred{{outline:2px dashed #e94560;z-index:9;opacity:1!important}}
.bar.dep-succ{{outline:2px dashed #76b7b2;z-index:9;opacity:1!important}}
/* dim unfocused bars */
.bar.dimmed{{opacity:0.08!important;box-shadow:none!important}}
/* schedule analysis */
.sa{{margin-top:15px;padding:10px;background:#16213e;border-radius:6px;font-size:13px}}
.sa h3{{color:#e0e0e0;margin:0 0 10px}}
.sa table{{border-collapse:collapse;width:100%}}
.sa th,.sa td{{padding:4px 10px;text-align:left;border-bottom:1px solid #1a1a2e}}
.sa th{{color:#e94560}}
.sched-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
@media(max-width:1200px){{.sched-grid{{grid-template-columns:1fr}}}}
.sched-card{{background:#1a1a2e;border:1px solid #0f3460;border-radius:6px;padding:10px}}
.sched-card h4{{margin:0 0 8px;color:#e94560;font-size:13px}}
.sched-card table{{width:100%;border-collapse:collapse;font-size:12px}}
.sched-card td{{padding:3px 6px;border-bottom:1px solid #16213e}}
.sched-card tr[onclick]{{cursor:pointer}}
.sched-card tr[onclick]:hover td{{background:#16213e}}
.util-chart{{max-height:300px;overflow-y:auto}}
.util-row{{display:flex;align-items:center;gap:6px;margin-bottom:2px;font-size:11px;font-family:monospace}}
.util-label{{min-width:70px}}
.util-track{{flex:1;background:#0f3460;border-radius:3px;height:14px;overflow:hidden}}
.util-fill{{height:100%;border-radius:3px;font-size:10px;line-height:14px;padding-left:4px;color:#fff;white-space:nowrap}}
.cp-flow{{display:flex;flex-wrap:wrap;align-items:center;gap:2px;font-size:11px;font-family:monospace;max-height:200px;overflow-y:auto}}
.cp-node{{background:#0f3460;padding:3px 6px;border-radius:4px;text-align:center}}
.cp-arrow{{color:#e94560;font-size:14px}}
/* help panel */
.help-toggle{{background:#0f3460;color:#76b7b2;border:1px solid #0f3460;padding:4px 10px;border-radius:4px;cursor:pointer;font-size:12px;margin-left:8px}}
.help-toggle:hover{{background:#1a1a3e}}
.help-panel{{display:none;background:#0f3460;border:1px solid #1a1a2e;border-radius:6px;padding:12px 16px;margin:8px 0;font-size:12px;line-height:1.6}}
.help-panel.show{{display:block}}
.help-panel h4{{margin:0 0 6px;color:#76b7b2;font-size:13px}}
.help-panel dl{{margin:4px 0;display:grid;grid-template-columns:auto 1fr;gap:2px 12px}}
.help-panel dt{{font-weight:bold;white-space:nowrap}}
.help-panel dd{{margin:0;color:#bbb}}
.help-panel .swatch{{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:middle;margin-right:4px}}
</style>
</head>
<body>
<h1>Mirage Task Graph Execution Timeline</h1>
<div class="sub">
{num_events} events | {num_tasks} tasks | Total: {total_dur_ms:.2f} ms
</div>
<div class="leg" id="leg"></div>
<div class="ctrls">
<button class="active" onclick="sw('pip',this)">Pipeline Phases</button>
<button onclick="sw('tt',this)">By Task Type</button>
<button onclick="sw('blk',this)">Per Block</button>
<button class="help-toggle" onclick="document.getElementById('blk-help').classList.toggle('show')">? Help</button>
<div class="zoom-ctrl">
<button onclick="zoom(-1)">-</button>
<span id="zl">100%</span>
<button onclick="zoom(1)">+</button>
<button onclick="zoom(0)">Fit</button>
</div>
</div>
<div id="tt-tip"></div>
<div id="pip" class="tab active"><div class="tc"><div class="tl" id="tl-pip"></div></div></div>
<div id="tt" class="tab"><div class="tc"><div class="tl" id="tl-tt"></div></div></div>
<div id="blk" class="tab">
<div class="help-panel" id="blk-help">
<h4>Per-Block View Guide</h4>
<p>Each row is one GPU block (SM). Bars show individual task executions on that block over time.</p>
<dl>
<dt>Colored bars</dt><dd>Task executions. Color = task type (see legend above). Hover for timing details. <b>Click</b> to highlight the task group and its dependencies.</dd>
<dt><span class="swatch" style="border:2px solid #e94560;box-shadow:0 0 4px #e94560"></span> Glowing red border</dt><dd><b>Straggler</b> &mdash; this task took &gt;1.5&times; the group average duration across all blocks.</dd>
<dt>Click a bar &rarr; highlights</dt><dd><span style="color:#fff"><b>White outline</b></span> = all instances of the clicked task group (same task across all blocks). <span style="color:#e94560"><b>Red dashed outline</b></span> = predecessor groups (must complete first). <span style="color:#76b7b2"><b>Teal dashed outline</b></span> = successor groups (waiting on this). All other bars are faded out. Click background to dismiss.</dd>
<dt><span class="swatch" style="background:#76b7b2"></span> Teal line inside bar</dt><dd><b>Ready-time marker</b> (appears after clicking) &mdash; the moment all dependencies were satisfied. Gap between this marker and the bar&apos;s left edge = <b>queue wait time</b> (scheduler delay after task became eligible).</dd>
</dl>
</div>
<div class="tc" id="blk-tc"><div class="tl" id="tl-blk"></div></div></div>
<div class="st">
<h3>Task Type Summary</h3>
<table>
<tr><th>Task Type</th><th>Invocations</th><th>Total Duration</th><th>Avg Duration</th><th>%</th></tr>
{stats_rows}
</table>
</div>
<div class="sa">
<h3>Schedule Analysis</h3>
<p style="font-size:12px;color:#888;margin:0 0 10px">How well the task scheduler utilized the GPU blocks. Covers block utilization, queue wait times, straggler effects, and the critical dependency path.</p>
{sched_analysis}
</div>
<script>
const G={js_global_start},D={js_total_dur};
const pipR={js_pipeline};
const ttR={js_tasktype};
const blkR={js_blocks};
const legI={js_legend};
const GD={js_group_deps};

let bw=3000;
let barMap={{}};

const legEl=document.getElementById('leg');
legI.forEach(i=>{{const d=document.createElement('div');d.className='li';d.innerHTML=`<div class="lc" style="background:${{i.color}}"></div>${{i.name}}`;legEl.appendChild(d)}});

/* ---- generic render (pipeline / tasktype views) ---- */
function render(cid,rows){{
  const c=document.getElementById(cid);c.innerHTML='';
  const ar=document.createElement('div');ar.className='ta';
  const al=document.createElement('div');al.className='rl';al.textContent='Time \\u2192';ar.appendChild(al);
  const ab=document.createElement('div');ab.className='rb';ab.style.minWidth=bw+'px';
  const nt=20;
  for(let i=0;i<=nt;i++){{const t=document.createElement('div');t.className='tk';t.style.left=(i/nt*bw)+'px';t.textContent=((i/nt*D)/1e6).toFixed(2)+' ms';ab.appendChild(t)}}
  ar.appendChild(ab);c.appendChild(ar);
  rows.forEach(row=>{{
    const rd=document.createElement('div');rd.className='row';
    const l=document.createElement('div');l.className='rl';l.textContent=row.label;l.title=row.label;rd.appendChild(l);
    const bd=document.createElement('div');bd.className='rb';bd.style.minWidth=bw+'px';
    row.bars.forEach(b=>{{
      const be=document.createElement('div');be.className='bar';
      const left=b.start/D*bw,w=Math.max(1,(b.end-b.start)/D*bw);
      be.style.left=left+'px';be.style.width=w+'px';be.style.background=b.color;
      be.addEventListener('mouseenter',e=>{{
        const tp=document.getElementById('tt-tip');
        const d=((b.end-b.start)/1e3).toFixed(1),a=(b.avg_dur/1e3).toFixed(1);
        tp.innerHTML=`<b>${{b.name}}</b>\\nWall: ${{d}} us  Avg/blk: ${{a}} us  Blocks: ${{b.blocks}}`;
        tp.style.display='block';tp.style.left=(e.clientX+12)+'px';tp.style.top=(e.clientY-30)+'px'}});
      be.addEventListener('mousemove',e=>{{const tp=document.getElementById('tt-tip');tp.style.left=(e.clientX+12)+'px';tp.style.top=(e.clientY-30)+'px'}});
      be.addEventListener('mouseleave',()=>{{document.getElementById('tt-tip').style.display='none'}});
      bd.appendChild(be)}});
    rd.appendChild(bd);c.appendChild(rd)}})}}

/* ---- enhanced block render with stragglers ---- */
function renderBlk(cid,rows){{
  const c=document.getElementById(cid);c.innerHTML='';
  barMap={{}};

  /* time axis */
  const ar=document.createElement('div');ar.className='ta';
  const al=document.createElement('div');al.className='rl';al.textContent='Time \\u2192 (click bar to highlight deps)';ar.appendChild(al);
  const ab=document.createElement('div');ab.className='rb';ab.style.minWidth=bw+'px';
  const nt=20;
  for(let i=0;i<=nt;i++){{const t=document.createElement('div');t.className='tk';t.style.left=(i/nt*bw)+'px';t.textContent=((i/nt*D)/1e6).toFixed(2)+' ms';ab.appendChild(t)}}
  ar.appendChild(ab);c.appendChild(ar);

  rows.forEach((row,ri)=>{{
    const rd=document.createElement('div');rd.className='row';
    const l=document.createElement('div');l.className='rl';l.textContent=row.label;l.title=row.label;rd.appendChild(l);
    const bd=document.createElement('div');bd.className='rb';bd.style.minWidth=bw+'px';

    /* task bars */
    row.bars.forEach(b=>{{
      const be=document.createElement('div');be.className='bar';
      const left=b.start/D*bw,w=Math.max(1,(b.end-b.start)/D*bw);
      be.style.left=left+'px';be.style.width=w+'px';be.style.background=b.color;
      const gd=GD[b.name];

      /* straggler detection */
      if(gd&&gd.ad>0&&b.avg_dur>gd.ad*1.5)be.classList.add('straggler');

      /* register in barMap */
      if(!barMap[b.name])barMap[b.name]=[];
      barMap[b.name].push({{el:be,ri:ri,s:b.start,e:b.end}});

      /* click -> show dependency arrows */
      be.addEventListener('click',e=>{{e.stopPropagation();showDeps(b.name)}});

      /* tooltip */
      be.addEventListener('mouseenter',e=>{{
        const tp=document.getElementById('tt-tip');
        let txt='<b>'+b.name+'</b>';
        txt+='\\nDuration: '+((b.end-b.start)/1e3).toFixed(1)+' us (this block)';
        if(gd){{
          txt+='\\nGroup avg: '+(gd.ad/1e3).toFixed(1)+' us (across all blocks)';
          const qw=(b.start-gd.rt)/1e3;
          if(qw>0)txt+='\\nQueue wait: '+qw.toFixed(1)+' us (ready\\u2192start delay)';
          if(b.avg_dur>gd.ad*1.5&&gd.ad>0)txt+='\\n\\u26a0 STRAGGLER: '+(b.avg_dur/gd.ad).toFixed(1)+'x group avg';
          txt+='\\n\\nClick bar to highlight dependencies:';
          txt+='\\n  Predecessors: '+(gd.p.length?gd.p.length+' groups':'none');
          txt+='\\n  Successors: '+(gd.s.length?gd.s.length+' groups':'none');
        }}
        tp.innerHTML=txt;tp.style.display='block';
        tp.style.left=(e.clientX+12)+'px';tp.style.top=(e.clientY-30)+'px'}});
      be.addEventListener('mousemove',e=>{{const tp=document.getElementById('tt-tip');tp.style.left=(e.clientX+12)+'px';tp.style.top=(e.clientY-30)+'px'}});
      be.addEventListener('mouseleave',()=>{{document.getElementById('tt-tip').style.display='none'}});
      bd.appendChild(be);
    }});

    rd.appendChild(bd);c.appendChild(rd);
  }});

  /* click background to clear highlights */
  c.addEventListener('click',()=>{{clearDeps()}});
}}

/* ---- dependency highlight logic ---- */
function showDeps(gid){{
  clearDeps();
  const gd=GD[gid];if(!gd)return;

  /* Show all DAG-defined predecessors and successors that have trace data */
  const validP=gd.p.filter(pid=>GD[pid]&&barMap[pid]);
  const validS=gd.s.filter(sid=>GD[sid]&&barMap[sid]);

  /* Build focus set (own group + valid deps) */
  const focus=new Set([gid]);
  validP.forEach(pid=>focus.add(pid));
  validS.forEach(sid=>focus.add(sid));

  /* Dim everything not in focus */
  Object.keys(barMap).forEach(nm=>{{
    if(!focus.has(nm))barMap[nm].forEach(bi=>bi.el.classList.add('dimmed'));
  }});

  /* Highlight own group + add ready-time marker (teal) */
  (barMap[gid]||[]).forEach(bi=>{{
    bi.el.classList.add('dep-hl');
    if(gd.rt>=0){{
      const barL=bi.s/D*bw,rdyL=gd.rt/D*bw;
      if(rdyL<barL-1){{
        const rm=document.createElement('div');rm.className='ready-mk';
        rm.style.background='#76b7b2';rm.style.left=(rdyL-barL)+'px';
        bi.el.appendChild(rm);
      }}
    }}
  }});

  /* Highlight predecessors (dashed red outline) */
  validP.forEach(pid=>{{(barMap[pid]||[]).forEach(bi=>bi.el.classList.add('dep-pred'));}});
  /* Highlight successors (dashed teal outline) */
  validS.forEach(sid=>{{(barMap[sid]||[]).forEach(bi=>bi.el.classList.add('dep-succ'));}});
}}

function clearDeps(){{
  document.querySelectorAll('.dep-hl,.dep-pred,.dep-succ,.dimmed').forEach(e=>{{
    e.classList.remove('dep-hl','dep-pred','dep-succ','dimmed');
  }});
  document.querySelectorAll('.ready-mk').forEach(e=>e.remove());
}}

function showGroupFromWait(tk){{
  if(!tk)return;
  if(!document.getElementById('blk').classList.contains('active')){{
    document.querySelectorAll('.ctrls button').forEach(b=>{{
      if(b.textContent.includes('Per Block'))sw('blk',b);
    }});
  }}
  setTimeout(()=>{{
    showDeps(tk);
    const first=document.querySelector('.dep-hl');
    if(first)first.scrollIntoView({{block:'center',behavior:'smooth'}});
  }},80);
}}

/* ---- tab switching ---- */
function sw(id,btn){{
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.ctrls button').forEach(b=>b.classList.remove('active'));
  document.getElementById(id).classList.add('active');btn.classList.add('active');
  const tlId='tl-'+id;
  if(!document.getElementById(tlId).hasChildNodes()){{
    if(id==='pip')render(tlId,pipR);
    else if(id==='tt')render(tlId,ttR);
    else renderBlk(tlId,blkR);
  }}
}}

/* ---- zoom ---- */
function zoom(dir){{
  if(dir===0)bw=3000;
  else if(dir>0)bw=Math.min(60000,bw*1.5);
  else bw=Math.max(500,bw/1.5);
  document.getElementById('zl').textContent=Math.round(bw/30)+'%';
  document.querySelectorAll('.tl').forEach(el=>el.innerHTML='');
  const active=document.querySelector('.tab.active');
  if(active){{const id=active.id;
    const tlId='tl-'+id;
    if(id==='pip')render(tlId,pipR);
    else if(id==='tt')render(tlId,ttR);
    else renderBlk(tlId,blkR);
  }}
}}

/* tooltip */
document.getElementById('tt-tip').style.cssText='display:none;position:fixed;background:#0f3460;color:#e0e0e0;padding:8px 12px;border-radius:6px;font-size:12px;font-family:monospace;z-index:1000;pointer-events:none;white-space:pre-line;box-shadow:0 4px 12px rgba(0,0,0,.5);max-width:400px';

/* initial render */
render('tl-pip',pipR);
</script>
</body>
</html>"""

_WHATIF_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Mirage What-If Timeline</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;margin:0;padding:20px;background:#1a1a2e;color:#e0e0e0}}
h1{{margin:0 0 4px}}
.sub{{color:#9aa4bf;margin-bottom:12px;font-size:13px}}
.note{{margin-bottom:16px;padding:10px 12px;background:#16213e;border:1px solid #0f3460;border-radius:6px;font-size:12px;color:#b8c0d4;line-height:1.5}}
.ctrls{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px}}
.ctrls button{{background:#16213e;color:#e0e0e0;border:1px solid #0f3460;padding:6px 14px;border-radius:4px;cursor:pointer;font-size:13px}}
.ctrls button.active{{background:#e94560;border-color:#e94560}}
.leg{{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:14px;padding:10px;background:#16213e;border-radius:6px}}
.li{{display:flex;align-items:center;gap:5px;font-size:12px}}
.lc{{width:14px;height:14px;border-radius:3px;flex-shrink:0}}
.tab{{display:none}}.tab.active{{display:block}}
.wf-card-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-bottom:14px}}
.wf-card{{background:#16213e;border:1px solid #0f3460;border-radius:6px;padding:10px}}
.wf-card-title{{font-size:11px;text-transform:uppercase;color:#9aa4bf;margin-bottom:4px}}
.wf-card-value{{font-size:22px;font-weight:700;color:#f7f7f7;margin-bottom:4px}}
.wf-card-hint{{font-size:11px;color:#9aa4bf;line-height:1.4}}
.wf-table-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;margin-bottom:14px}}
.wf-table-card{{background:#16213e;border:1px solid #0f3460;border-radius:6px;padding:10px}}
.wf-table-card h4{{margin:0 0 4px;color:#e94560}}
.wf-subtle{{font-size:11px;color:#9aa4bf;margin:0 0 8px;line-height:1.4}}
.wf-table-card table{{width:100%;border-collapse:collapse;font-size:12px}}
.wf-table-card th,.wf-table-card td{{padding:4px 6px;text-align:left;border-bottom:1px solid #1a1a2e;vertical-align:top}}
.wf-table-card th{{color:#76b7b2}}
.tc{{overflow:auto;max-height:72vh;border:1px solid #0f3460;border-radius:6px;background:#16213e}}
.tl{{position:relative;min-width:100%}}
.row{{display:flex;align-items:center;border-bottom:1px solid #1a1a2e;min-height:26px}}
.rl{{min-width:220px;max-width:220px;padding:2px 8px;font-size:11px;font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;background:#16213e;position:sticky;left:0;z-index:2;border-right:1px solid #0f3460}}
.rb{{position:relative;flex:1;height:26px}}
.bar{{position:absolute;height:20px;top:3px;border-radius:3px;opacity:.88;cursor:pointer;min-width:1px}}
.bar:hover{{opacity:1;outline:1px solid #fff}}
.ta{{display:flex;position:sticky;top:0;z-index:3;background:#16213e;border-bottom:2px solid #0f3460}}
.ta .rl{{font-weight:bold;font-size:12px}}
.tk{{position:absolute;top:0;font-size:10px;color:#888;transform:translateX(-50%);white-space:nowrap}}
.zoom-ctrl{{display:flex;gap:6px;align-items:center;margin-left:auto}}
.zoom-ctrl button{{padding:4px 10px;font-size:14px}}
.zoom-ctrl span{{font-size:12px;color:#888}}
</style>
</head>
<body>
<h1>Mirage What-If Timeline</h1>
<div class="sub">Trace-calibrated brainstorming view | MPK-like baseline wall time: {total_dur_ms:.2f} ms</div>
<div class="note">This view is intentionally first-order. Locality uses overlap size, reuse delay, and same-block vs cross-block placement. Register and shared-memory pressure are only treated as coarse feasibility warnings.</div>
<div class="leg" id="leg"></div>
<div class="ctrls">
<button class="active" onclick="sw('baseline',this)">Baseline</button>
<button onclick="sw('affinity',this)">Affinity Scheduling</button>
<div class="zoom-ctrl">
<button onclick="zoom(-1)">-</button>
<span id="zl">100%</span>
<button onclick="zoom(1)">+</button>
<button onclick="zoom(0)">Fit</button>
</div>
</div>
<div id="tip"></div>
<div id="baseline" class="tab active">
{baseline_cards}
{baseline_tables}
<div class="tc"><div class="tl" id="tl-baseline"></div></div>
</div>
<div id="affinity" class="tab">
{affinity_cards}
{affinity_tables}
<div class="tc"><div class="tl" id="tl-affinity"></div></div>
</div>
<script>
const tabs={{baseline:{{rows:{js_baseline},dur:{js_baseline_dur}}},affinity:{{rows:{js_affinity},dur:{js_affinity_dur}}}}};
const legend={js_legend};
let bw=3000;

const legEl=document.getElementById('leg');
legend.forEach(item=>{{const d=document.createElement('div');d.className='li';d.innerHTML=`<div class="lc" style="background:${{item.color}}"></div>${{item.name}}`;legEl.appendChild(d);}});

function renderTimeline(id,rows,totalDur){{
  const c=document.getElementById('tl-'+id);c.innerHTML='';
  const axis=document.createElement('div');axis.className='ta';
  const axisLabel=document.createElement('div');axisLabel.className='rl';axisLabel.textContent='Time →';axis.appendChild(axisLabel);
  const axisBody=document.createElement('div');axisBody.className='rb';axisBody.style.minWidth=bw+'px';
  for(let i=0;i<=20;i++){{const tick=document.createElement('div');tick.className='tk';tick.style.left=(i/20*bw)+'px';tick.textContent=((i/20*totalDur)/1e6).toFixed(2)+' ms';axisBody.appendChild(tick);}}
  axis.appendChild(axisBody);c.appendChild(axis);
  rows.forEach(row=>{{
    const rowEl=document.createElement('div');rowEl.className='row';
    const label=document.createElement('div');label.className='rl';label.textContent=row.label;label.title=row.label;rowEl.appendChild(label);
    const barsEl=document.createElement('div');barsEl.className='rb';barsEl.style.minWidth=bw+'px';
    row.bars.forEach(bar=>{{
      const barEl=document.createElement('div');barEl.className='bar';
      const left=bar.start/totalDur*bw;
      const width=Math.max(1,(bar.end-bar.start)/totalDur*bw);
      barEl.style.left=left+'px';barEl.style.width=width+'px';barEl.style.background=bar.color;
      barEl.addEventListener('mouseenter',evt=>{{
        const tip=document.getElementById('tip');
        tip.innerHTML=`<b>${{bar.name}}</b><br>Duration: ${{(((bar.end-bar.start)/1e3)).toFixed(1)}} us`;
        tip.style.display='block';
        tip.style.left=(evt.clientX+12)+'px';
        tip.style.top=(evt.clientY-28)+'px';
      }});
      barEl.addEventListener('mousemove',evt=>{{const tip=document.getElementById('tip');tip.style.left=(evt.clientX+12)+'px';tip.style.top=(evt.clientY-28)+'px';}});
      barEl.addEventListener('mouseleave',()=>{{document.getElementById('tip').style.display='none';}});
      barsEl.appendChild(barEl);
    }});
    rowEl.appendChild(barsEl);c.appendChild(rowEl);
  }});
}}

function sw(id,btn){{
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.ctrls button').forEach(b=>b.classList.remove('active'));
  document.getElementById(id).classList.add('active');btn.classList.add('active');
  renderTimeline(id,tabs[id].rows,tabs[id].dur);
}}

function zoom(dir){{
  if(dir===0)bw=3000;
  else if(dir>0)bw=Math.min(60000,bw*1.5);
  else bw=Math.max(500,bw/1.5);
  document.getElementById('zl').textContent=Math.round(bw/30)+'%';
  const active=document.querySelector('.tab.active').id;
  renderTimeline(active,tabs[active].rows,tabs[active].dur);
}}

document.getElementById('tip').style.cssText='display:none;position:fixed;background:#0f3460;color:#e0e0e0;padding:8px 12px;border-radius:6px;font-size:12px;font-family:monospace;z-index:1000;pointer-events:none;box-shadow:0 4px 12px rgba(0,0,0,.5)';
renderTimeline('baseline',tabs.baseline.rows,tabs.baseline.dur);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Combine task graph + Perfetto trace into a Gantt timeline"
    )
    parser.add_argument("task_graph", help="Path to task_graph JSON file")
    parser.add_argument("trace", help="Path to .perfetto-trace file")
    parser.add_argument("-o", "--output", "--out", default=None,
                        help="Output HTML path (default: <base>_timeline.html)")
    parser.add_argument("--svg", default=None,
                        help="Also generate a static SVG/PNG at this path")
    parser.add_argument("--whatif", action="store_true",
                        help="Generate the lightweight what-if HTML view")
    parser.add_argument("--whatif-locality-cap", type=float, default=0.25,
                        help="Cap for first-order locality gain (default: 0.25)")
    parser.add_argument("--whatif-l2-factor", "--whatif-cross-block-factor",
                        dest="whatif_l2_factor", type=float, default=0.35,
                        help="Reuse factor when successor lands on warm L2 instead of same-SM state (default: 0.35)")
    parser.add_argument("--whatif-sm-tau-scale", type=float, default=1.0,
                        help="Scale factor for same-SM hot reuse decay windows (default: 1.0)")
    parser.add_argument("--whatif-l2-tau-scale", type=float, default=1.0,
                        help="Scale factor for warm-L2 reuse decay windows (default: 1.0)")
    parser.add_argument("--whatif-bandwidth-scale", type=float, default=1.0,
                        help="Scale factor for the analytical bandwidth-pressure threshold (default: 1.0)")
    parser.add_argument("--whatif-top-candidates", type=int, default=12,
                        help="Reserved for future concurrency candidate ranking; unused in the current 2-tab view (default: 12)")
    parser.add_argument("--whatif-split-factors", default="2,4",
                        help="Reserved for future finer-task replay; unused in the current 2-tab view (default: 2,4)")
    parser.add_argument("--whatif-target-types",
                        default="TASK_SILU_MUL,TASK_ARGMAX_PARTIAL_SM100",
                        help="Reserved for future finer-task analysis; unused in the current 2-tab view")
    args = parser.parse_args()

    if args.output is None:
        suffix = "_whatif.html" if args.whatif else "_timeline.html"
        args.output = os.path.splitext(args.task_graph)[0] + suffix

    print("Loading task graph …")
    events, tasks = parse_task_graph(args.task_graph)
    print(f"  {len(events)} events, {len(tasks)} tasks")

    print("Loading Perfetto trace …")
    slices, track_names = parse_perfetto_trace(args.trace)
    print(f"  {len(slices)} trace slices, {len(track_names)} tracks")

    print("Building pipeline stage sequence …")
    stage_seq = build_stage_sequence(events, tasks)
    print(f"  {len(stage_seq)} stages")
    period = detect_layer_pattern(stage_seq)
    if period:
        print(f"  Detected layer period = {period} stages")

    print("Building dependency DAG …")
    dag_groups, adjacency, reverse_adj = build_dependency_dag(events, tasks)
    print(f"  {len(dag_groups)} task groups, "
          f"{sum(len(v) for v in adjacency.values())} edges")

    print("Building views …")
    pip_rows, tt_rows, blk_rows, g_start, t_dur, t_stats = \
        build_views(slices, track_names, stage_seq)
    print(f"  Pipeline: {len(pip_rows)} rows, TaskType: {len(tt_rows)} rows, "
          f"Blocks: {len(blk_rows)} rows")

    print("Mapping trace to task graph …")
    group_deps, group_timing, dag_to_trace = map_trace_to_graph(
        slices, track_names, dag_groups, adjacency, reverse_adj, g_start)
    print(f"  Mapped {len(group_deps)} trace groups to DAG")

    print("Computing schedule metrics …")
    sched_metrics = compute_schedule_metrics(
        slices, track_names, group_timing, dag_groups,
        adjacency, reverse_adj, dag_to_trace, g_start, t_dur, group_deps)
    sm = sched_metrics["summary"]
    print(f"  Avg utilization: {sm['avg_util']}%, "
          f"critical path: {sm['critical_path_us']:.1f} us, "
          f"avg queue wait: {sm['avg_queue_wait_us']:.1f} us")

    if args.whatif:
        split_factors = [
            int(value.strip()) for value in args.whatif_split_factors.split(",")
            if value.strip()
        ]
        target_types = [
            value.strip() for value in args.whatif_target_types.split(",")
            if value.strip()
        ]
        print("Building what-if report …")
        whatif_data = build_whatif_report(
            tasks,
            events,
            slices,
            track_names,
            dag_groups,
            adjacency,
            reverse_adj,
            group_deps,
            dag_to_trace,
            g_start,
            t_dur,
            blk_rows,
            sched_metrics,
            args.whatif_locality_cap,
            args.whatif_l2_factor,
            args.whatif_sm_tau_scale,
            args.whatif_l2_tau_scale,
            args.whatif_bandwidth_scale,
            args.whatif_top_candidates,
            split_factors,
            target_types,
        )
        print("Writing what-if HTML …")
        generate_whatif_html(whatif_data, args.output)
    else:
        print("Writing HTML …")
        generate_html(pip_rows, tt_rows, blk_rows, g_start, t_dur, t_stats,
                      (len(events), len(tasks)), group_deps, sched_metrics,
                      args.output)

    if args.svg:
        print("Writing SVG …")
        generate_svg(tt_rows, g_start, t_dur, args.svg)

    print("Done!")


if __name__ == "__main__":
    main()
