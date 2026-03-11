from __future__ import annotations

import heapq
import math
import re
from collections import defaultdict
from copy import deepcopy
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List, Optional, Tuple


_TRACE_NAME_RE = re.compile(r"^(.+)_(\d+)$")

_DATA_TYPE_SIZES = {
    930: 1,
    935: 1,
    936: 1,
    940: 2,
    941: 2,
    945: 2,
    946: 2,
    950: 4,
    955: 4,
    956: 4,
    960: 8,
    965: 8,
    966: 8,
}

_RESOURCE_VECTORS = {
    "small": (0.20, 0.10, 0.125),
    "medium": (0.35, 0.25, 0.25),
    "large": (0.60, 0.50, 0.50),
}

_LOCALITY_FALLBACK_THRESHOLD = 0.02


def parse_trace_name(name: str) -> Tuple[str, int]:
    match = _TRACE_NAME_RE.match(name)
    if match is None:
        return name, 0
    return match.group(1), int(match.group(2))


def short_name(name: str) -> str:
    return name.replace("TASK_", "")


def task_memory_weight(task_type: str) -> float:
    upper = task_type.upper()
    if "RMS_NORM" in upper or "SILU_MUL" in upper or "ARGMAX" in upper:
        return 0.7
    if "LINEAR" in upper:
        return 0.4
    if "ATTENTION" in upper or "ATTN" in upper:
        return 0.5
    return 0.3


def task_resource_class(task_type: str) -> str:
    upper = task_type.upper()
    if "ATTENTION" in upper or "ATTN" in upper:
        return "large"
    if "LINEAR" in upper:
        return "medium"
    if "RMS_NORM" in upper or "SILU_MUL" in upper or "ARGMAX" in upper:
        return "small"
    return "medium"


def task_resource_vector(task_type: str) -> Tuple[float, float, float]:
    return _RESOURCE_VECTORS[task_resource_class(task_type)]


def task_memory_demand(task_type: str) -> float:
    upper = task_type.upper()
    if "ATTENTION" in upper or "ATTN" in upper:
        return 0.65
    if "RMS_NORM" in upper or "SILU_MUL" in upper or "ARGMAX" in upper:
        return 0.55
    if "LINEAR" in upper:
        return 0.35
    return 0.30


def task_compute_class(task_type: str) -> str:
    return "memory-heavy" if task_memory_demand(task_type) >= 0.50 else "compute-heavy"


def resource_risk(task_types: Iterable[str]) -> str:
    kinds = [task_resource_class(task_type) for task_type in task_types]
    if kinds.count("large") >= 2:
        return "high"
    if "large" in kinds or kinds.count("medium") >= 2:
        return "medium"
    return "low"


def merge_intervals(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _dtype_size(dtype: int) -> int:
    return _DATA_TYPE_SIZES.get(dtype, 2)


def tensor_span_bytes(tensor_desc: Optional[dict]) -> Optional[Tuple[str, int, int]]:
    if not tensor_desc:
        return None
    base_ptr = tensor_desc.get("base_ptr")
    dims = tensor_desc.get("dims") or []
    strides = tensor_desc.get("strides") or []
    if base_ptr in (None, "nullptr") or not dims:
        return None
    elem_size = _dtype_size(int(tensor_desc.get("data_type", 999)))
    offset = int(tensor_desc.get("offset", 0))
    if strides and len(strides) == len(dims):
        max_elem_offset = 0
        for dim, stride in zip(dims, strides):
            if dim <= 0:
                return None
            max_elem_offset += (dim - 1) * abs(int(stride))
        span_bytes = (max_elem_offset + 1) * elem_size
    else:
        numel = 1
        for dim in dims:
            if dim <= 0:
                return None
            numel *= int(dim)
        span_bytes = numel * elem_size
    return str(base_ptr), offset, offset + max(span_bytes, elem_size)


def _collect_group_regions(task_indices: List[int],
                           tasks: List[dict],
                           field_name: str) -> Tuple[Dict[str, List[Tuple[int, int]]], int]:
    by_base: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for task_idx in task_indices:
        for tensor_desc in tasks[task_idx].get(field_name) or []:
            span = tensor_span_bytes(tensor_desc)
            if span is None:
                continue
            base_ptr, start, end = span
            by_base[base_ptr].append((start, end))
    merged = {base_ptr: merge_intervals(intervals)
              for base_ptr, intervals in by_base.items()}
    total_bytes = sum(end - start
                      for intervals in merged.values()
                      for start, end in intervals)
    return merged, total_bytes


def _interval_overlap(lhs: List[Tuple[int, int]],
                      rhs: List[Tuple[int, int]]) -> int:
    i = 0
    j = 0
    overlap = 0
    while i < len(lhs) and j < len(rhs):
        lhs_start, lhs_end = lhs[i]
        rhs_start, rhs_end = rhs[j]
        overlap += max(0, min(lhs_end, rhs_end) - max(lhs_start, rhs_start))
        if lhs_end <= rhs_end:
            i += 1
        else:
            j += 1
    return overlap


def compute_edge_locality(dag_groups: List[dict],
                          tasks: List[dict],
                          adjacency: Dict[int, List[int]]) -> Dict[Tuple[object, object], dict]:
    input_cache: Dict[object, Tuple[Dict[str, List[Tuple[int, int]]], int]] = {}
    output_cache: Dict[object, Tuple[Dict[str, List[Tuple[int, int]]], int]] = {}
    edge_locality: Dict[Tuple[object, object], dict] = {}

    def get_inputs(group_id: object) -> Tuple[Dict[str, List[Tuple[int, int]]], int]:
        if group_id not in input_cache:
            input_cache[group_id] = _collect_group_regions(
                dag_groups[int(group_id)]["task_indices"], tasks, "inputs")
        return input_cache[group_id]

    def get_outputs(group_id: object) -> Tuple[Dict[str, List[Tuple[int, int]]], int]:
        if group_id not in output_cache:
            output_cache[group_id] = _collect_group_regions(
                dag_groups[int(group_id)]["task_indices"], tasks, "outputs")
        return output_cache[group_id]

    for pred_gid, succ_gids in adjacency.items():
        pred_regions, pred_bytes = get_outputs(pred_gid)
        for succ_gid in succ_gids:
            succ_regions, succ_bytes = get_inputs(succ_gid)
            overlap = 0
            for base_ptr, pred_intervals in pred_regions.items():
                succ_intervals = succ_regions.get(base_ptr)
                if succ_intervals:
                    overlap += _interval_overlap(pred_intervals, succ_intervals)
            edge_locality[(pred_gid, succ_gid)] = {
                "pred_gid": pred_gid,
                "succ_gid": succ_gid,
                "pred_type": dag_groups[pred_gid]["task_type_name"],
                "succ_type": dag_groups[succ_gid]["task_type_name"],
                "overlap_bytes": overlap,
                "consumer_input_bytes": succ_bytes,
                "pred_output_bytes": pred_bytes,
            }
    return edge_locality


def build_trace_groups(slices: List[Tuple[str, int, int, int]],
                       track_names: Dict[int, str],
                       group_deps: Dict[str, dict],
                       global_start: int,
                       scheduler_types: Iterable[str]) -> Dict[str, dict]:
    scheduler_set = set(scheduler_types)
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for name, ts, dur, track_id in slices:
        task_type, _ = parse_trace_name(name)
        if task_type in scheduler_set:
            continue
        track_name = track_names.get(track_id, f"track_{track_id}")
        block_match = re.match(r"block_(\d+)", track_name)
        if block_match is None:
            continue
        block_id = int(block_match.group(1))
        grouped[name].append({
            "block": block_id,
            "start": ts - global_start,
            "end": ts + dur - global_start,
            "dur": dur,
        })

    trace_groups: Dict[str, dict] = {}
    for name, records in grouped.items():
        task_type, event_no = parse_trace_name(name)
        records.sort(key=lambda item: (item["start"], item["block"]))
        ready = group_deps.get(name, {}).get("rt", 0)
        durations = [record["dur"] for record in records]
        cv = 0.0
        if len(durations) > 1:
            mu = mean(durations)
            if mu > 0:
                cv = pstdev(durations) / mu
        trace_groups[name] = {
            "name": name,
            "type": task_type,
            "event_no": event_no,
            "start": min(record["start"] for record in records),
            "end": max(record["end"] for record in records),
            "avg_dur": sum(durations) / len(durations),
            "durations": durations,
            "duration_cv": cv,
            "count": len(records),
            "blocks": [record["block"] for record in records],
            "instances": records,
            "preds": [pred for pred in group_deps.get(name, {}).get("p", [])
                      if pred in grouped],
            "succs": [succ for succ in group_deps.get(name, {}).get("s", [])
                      if succ in grouped],
            "ready": ready,
            "queue_wait": max(0.0, min(record["start"] for record in records) - ready),
        }
    return trace_groups


def median_duration_by_type(trace_groups: Dict[str, dict]) -> Dict[str, float]:
    durations: Dict[str, List[float]] = defaultdict(list)
    for group in trace_groups.values():
        durations[group["type"]].extend(float(duration) for duration in group["durations"])
    return {task_type: median(samples) for task_type, samples in durations.items()}


def effective_reuse_score(overlap_bytes: int,
                          consumer_input_bytes: int,
                          reuse_delay_ns: float,
                          tau_ns: float) -> Tuple[float, float, float]:
    if overlap_bytes <= 0 or consumer_input_bytes <= 0:
        return 0.0, 0.0, 0.0
    reuse_fraction = min(1.0, overlap_bytes / max(1.0, float(consumer_input_bytes)))
    safe_tau = max(1.0, float(tau_ns))
    temporal_decay = math.exp(-max(0.0, reuse_delay_ns) / safe_tau)
    return reuse_fraction, temporal_decay, reuse_fraction * temporal_decay


def build_locality_opportunities(trace_groups: Dict[str, dict],
                                 trace_to_dag: Dict[Tuple[str, int], int],
                                 edge_locality: Dict[Tuple[object, object], dict],
                                 locality_cap: float,
                                 l2_factor: float) -> List[dict]:
    tau_by_type = median_duration_by_type(trace_groups)
    opportunities: List[dict] = []
    for succ_name, succ_group in trace_groups.items():
        succ_key = (succ_group["type"], succ_group["event_no"])
        succ_gid = trace_to_dag.get(succ_key)
        if succ_gid is None:
            continue
        for pred_name in succ_group["preds"]:
            pred_group = trace_groups.get(pred_name)
            if pred_group is None:
                continue
            pred_key = (pred_group["type"], pred_group["event_no"])
            pred_gid = trace_to_dag.get(pred_key)
            if pred_gid is None:
                continue
            static_edge = edge_locality.get((pred_gid, succ_gid))
            if static_edge is None:
                continue
            reuse_delay = max(0.0, succ_group["start"] - pred_group["end"])
            tau_ns = tau_by_type.get(pred_group["type"], pred_group["avg_dur"])
            reuse_fraction, temporal_decay, effective_reuse = effective_reuse_score(
                static_edge["overlap_bytes"],
                static_edge["consumer_input_bytes"],
                reuse_delay,
                tau_ns,
            )
            shared_blocks = len(set(pred_group["blocks"]).intersection(succ_group["blocks"]))
            same_block_ratio = shared_blocks / max(1, succ_group["count"])
            mem_weight = task_memory_weight(succ_group["type"])
            predicted_gain_same = mem_weight * effective_reuse * locality_cap
            predicted_gain_diff = mem_weight * effective_reuse * l2_factor * locality_cap
            opportunities.append({
                "pred_name": pred_name,
                "succ_name": succ_name,
                "pred_gid": pred_gid,
                "succ_gid": succ_gid,
                "pred_type": pred_group["type"],
                "succ_type": succ_group["type"],
                "overlap_bytes": static_edge["overlap_bytes"],
                "consumer_input_bytes": static_edge["consumer_input_bytes"],
                "reuse_delay_ns": reuse_delay,
                "reuse_delay_us": reuse_delay / 1e3,
                "tau_ns": tau_ns,
                "reuse_fraction": reuse_fraction,
                "temporal_decay": temporal_decay,
                "effective_reuse": effective_reuse,
                "same_block_ratio": same_block_ratio,
                "predicted_gain_same": predicted_gain_same,
                "predicted_gain_diff": predicted_gain_diff,
                "resource_risk": resource_risk((pred_group["type"], succ_group["type"])),
            })
    opportunities.sort(key=lambda item: (
        -item["effective_reuse"],
        -item["overlap_bytes"],
        item["reuse_delay_ns"],
        item["succ_name"],
    ))
    return opportunities


def analyze_bubbles(trace_groups: Dict[str, dict]) -> List[dict]:
    per_block: Dict[int, List[dict]] = defaultdict(list)
    for group in trace_groups.values():
        for instance in group["instances"]:
            per_block[instance["block"]].append({
                "name": group["name"],
                "type": group["type"],
                "start": instance["start"],
                "end": instance["end"],
            })

    gaps_by_group: Dict[str, dict] = {}
    for block_id, records in per_block.items():
        records.sort(key=lambda item: item["start"])
        prev_end = 0
        for record in records:
            gap = max(0, record["start"] - prev_end)
            if gap > 0:
                stats = gaps_by_group.setdefault(record["name"], {
                    "name": record["name"],
                    "type": record["type"],
                    "total_gap_ns": 0,
                    "max_gap_ns": 0,
                    "gap_count": 0,
                    "blocks": set(),
                })
                stats["total_gap_ns"] += gap
                stats["max_gap_ns"] = max(stats["max_gap_ns"], gap)
                stats["gap_count"] += 1
                stats["blocks"].add(block_id)
            prev_end = max(prev_end, record["end"])

    rows = []
    for stats in gaps_by_group.values():
        rows.append({
            "name": stats["name"],
            "type": stats["type"],
            "total_gap_us": stats["total_gap_ns"] / 1e3,
            "max_gap_us": stats["max_gap_ns"] / 1e3,
            "avg_gap_us": stats["total_gap_ns"] / max(1, stats["gap_count"]) / 1e3,
            "affected_blocks": len(stats["blocks"]),
            "gap_count": stats["gap_count"],
        })
    rows.sort(key=lambda item: (-item["total_gap_us"], -item["max_gap_us"], item["name"]))
    return rows


def rank_stationary_candidates(locality_opportunities: List[dict]) -> List[dict]:
    aggregated: Dict[Tuple[str, str], dict] = {}
    for edge in locality_opportunities:
        key = (edge["pred_type"], edge["succ_type"])
        entry = aggregated.setdefault(key, {
            "chain": f"{short_name(edge['pred_type'])} -> {short_name(edge['succ_type'])}",
            "overlap_score": 0.0,
            "reuse_delay_us": edge["reuse_delay_us"],
            "effective_reuse": 0.0,
            "resource_risk": edge["resource_risk"],
            "recommendation": "unlikely to matter",
        })
        entry["overlap_score"] = max(entry["overlap_score"], edge["reuse_fraction"])
        entry["effective_reuse"] = max(entry["effective_reuse"], edge["effective_reuse"])
        entry["reuse_delay_us"] = min(entry["reuse_delay_us"], edge["reuse_delay_us"])
        risks = ("low", "medium", "high")
        if risks.index(edge["resource_risk"]) > risks.index(entry["resource_risk"]):
            entry["resource_risk"] = edge["resource_risk"]

    rows = []
    for entry in aggregated.values():
        if entry["effective_reuse"] > 0.20 and entry["resource_risk"] == "low":
            entry["recommendation"] = "good candidate"
        elif entry["effective_reuse"] > 0.10 or entry["resource_risk"] != "high":
            entry["recommendation"] = "possible but risky"
        else:
            entry["recommendation"] = "unlikely to matter"
        rows.append(entry)

    rows.sort(key=lambda item: (-item["effective_reuse"], item["reuse_delay_us"], item["chain"]))
    return rows


def _safe_cv(samples: List[float]) -> float:
    if len(samples) <= 1:
        return 0.0
    mu = mean(samples)
    if mu <= 0:
        return 0.0
    return pstdev(samples) / mu


def _topological_order_models(group_models: Dict[object, dict]) -> List[object]:
    in_degree = {gid: len(group.get("preds", [])) for gid, group in group_models.items()}
    ready = [gid for gid, degree in in_degree.items() if degree == 0]
    ready.sort(key=lambda gid: (str(group_models[gid]["name"]), str(gid)))
    order = []
    while ready:
        gid = ready.pop(0)
        order.append(gid)
        for succ in group_models[gid].get("succs", []):
            if succ not in in_degree:
                continue
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                ready.append(succ)
                ready.sort(key=lambda item: (str(group_models[item]["name"]), str(item)))
    if len(order) != len(group_models):
        missing = [gid for gid in group_models if gid not in set(order)]
        order.extend(sorted(missing, key=str))
    return order


def build_group_models(tasks: List[dict],
                       events: List[dict],
                       dag_groups: List[dict],
                       adjacency: Dict[int, List[int]],
                       reverse_adj: Dict[int, List[int]],
                       trace_groups: Dict[str, dict],
                       dag_to_trace: Dict[int, Tuple[str, int]],
                       edge_locality: Dict[Tuple[object, object], dict],
                       locality_rows: List[dict]) -> Dict[object, dict]:
    trace_by_key = {(group["type"], group["event_no"]): group for group in trace_groups.values()}
    type_duration_samples: Dict[str, List[float]] = defaultdict(list)
    type_qwait_samples: Dict[str, List[float]] = defaultdict(list)
    type_cv_samples: Dict[str, List[float]] = defaultdict(list)

    for group in trace_groups.values():
        type_duration_samples[group["type"]].extend(float(duration) for duration in group["durations"])
        type_qwait_samples[group["type"]].append(float(group["queue_wait"]))
        type_cv_samples[group["type"]].append(float(group.get("duration_cv", 0.0)))

    all_samples = [sample for samples in type_duration_samples.values() for sample in samples]
    global_median = median(all_samples) if all_samples else 10_000.0
    type_median = {task_type: median(samples) for task_type, samples in type_duration_samples.items()}
    type_qwait = {task_type: median(samples) for task_type, samples in type_qwait_samples.items()}
    type_cv = {task_type: median(samples) for task_type, samples in type_cv_samples.items()}

    input_bytes_by_gid: Dict[object, int] = {}
    output_bytes_by_gid: Dict[object, int] = {}
    for group in dag_groups:
        gid = group["group_id"]
        _, input_bytes = _collect_group_regions(group["task_indices"], tasks, "inputs")
        _, output_bytes = _collect_group_regions(group["task_indices"], tasks, "outputs")
        input_bytes_by_gid[gid] = input_bytes
        output_bytes_by_gid[gid] = output_bytes

    incoming_locality: Dict[object, List[dict]] = defaultdict(list)
    for row in locality_rows:
        incoming_locality[row["succ_gid"]].append(row)

    max_fan_in = max((event.get("num_triggers", 0) for event in events), default=1)
    group_models: Dict[object, dict] = {}
    for group in dag_groups:
        gid = group["group_id"]
        task_type = group["task_type_name"]
        trace_key = dag_to_trace.get(gid)
        trace_group = trace_by_key.get(trace_key) if trace_key is not None else None
        base_duration = float(trace_group["avg_dur"]) if trace_group else float(type_median.get(task_type, global_median))
        duration_cv = float(trace_group.get("duration_cv", 0.0)) if trace_group else float(type_cv.get(task_type, 0.0))
        observed_queue_wait = float(trace_group["queue_wait"]) if trace_group else float(type_qwait.get(task_type, 0.0))
        fan_in = 0
        dep_event = int(group["dep_event"])
        if 0 <= dep_event < len(events):
            fan_in = int(events[dep_event].get("num_triggers", 0))
        locality_score = max(
            (row["effective_reuse"] for row in incoming_locality.get(gid, [])),
            default=0.0,
        )
        model = {
            "id": gid,
            "name": trace_group["name"] if trace_group else f"{task_type}_{gid}",
            "type": task_type,
            "count": int(group["task_count"]),
            "preds": list(reverse_adj.get(gid, [])),
            "succs": list(adjacency.get(gid, [])),
            "dep_event": dep_event,
            "trig_event": int(group["trig_event"]),
            "task_indices": list(group["task_indices"]),
            "base_duration_ns": max(1.0, base_duration),
            "duration_cv": max(0.0, duration_cv),
            "observed_queue_wait_ns": max(0.0, observed_queue_wait),
            "resource_class": task_resource_class(task_type),
            "resource_vector": task_resource_vector(task_type),
            "memory_weight": task_memory_weight(task_type),
            "memory_demand": task_memory_demand(task_type),
            "compute_class": task_compute_class(task_type),
            "input_bytes": input_bytes_by_gid.get(gid, 0),
            "output_bytes": output_bytes_by_gid.get(gid, 0),
            "fan_in": fan_in,
            "fan_in_score": fan_in / max(1, max_fan_in),
            "locality_score": locality_score,
            "trace_match": trace_key,
            "trace_count": int(trace_group["count"]) if trace_group else 0,
        }
        group_models[gid] = model

    topo = _topological_order_models(group_models)
    critical_path = {}
    max_cp = 0.0
    for gid in reversed(topo):
        child_cp = max((critical_path[succ] for succ in group_models[gid]["succs"]), default=0.0)
        critical_path[gid] = group_models[gid]["base_duration_ns"] + child_cp
        max_cp = max(max_cp, critical_path[gid])

    max_qwait = max((model["observed_queue_wait_ns"] for model in group_models.values()), default=1.0)
    max_cv = max((model["duration_cv"] for model in group_models.values()), default=1.0)

    for gid, model in group_models.items():
        model["critical_path_ns"] = critical_path[gid]
        model["critical_path_score"] = critical_path[gid] / max(1.0, max_cp)
        model["queue_wait_score"] = model["observed_queue_wait_ns"] / max(1.0, max_qwait)
        model["duration_variability_score"] = model["duration_cv"] / max(1.0, max_cv)

    return group_models


def _make_sm_state(num_sms: int) -> Dict[int, dict]:
    return {
        sm_id: {
            "active": [],
            "used": [0.0, 0.0, 0.0],
            "rows": [],
        }
        for sm_id in range(num_sms)
    }


def _fits(used: Iterable[float], needed: Iterable[float]) -> bool:
    return all((u + n) <= 1.0 + 1e-9 for u, n in zip(used, needed))


def _release_completed(sm_state: dict, current_time: float) -> None:
    if not sm_state["active"]:
        return
    remaining = []
    used = [0.0, 0.0, 0.0]
    for active in sm_state["active"]:
        if active["end"] > current_time + 1e-9:
            remaining.append(active)
            for idx, value in enumerate(active["resource_vector"]):
                used[idx] += value
    sm_state["active"] = remaining
    sm_state["used"] = used


def _earliest_fit_time(sm_state: dict,
                       ready_time: float,
                       resource_vector: Tuple[float, float, float]) -> float:
    events = sorted(sm_state["active"], key=lambda item: item["end"])
    used = [0.0, 0.0, 0.0]
    for active in events:
        if active["end"] > ready_time + 1e-9:
            for idx, value in enumerate(active["resource_vector"]):
                used[idx] += value
    if _fits(used, resource_vector):
        return ready_time
    for active in events:
        if active["end"] < ready_time - 1e-9:
            continue
        for idx, value in enumerate(active["resource_vector"]):
            used[idx] -= value
        if _fits(used, resource_vector):
            return active["end"]
    return ready_time


def _sample_duration(model: dict, instance_idx: int) -> float:
    base = float(model["base_duration_ns"])
    cv = float(model.get("duration_cv", 0.0))
    if cv <= 0.0:
        return max(1.0, base)
    period = (instance_idx % 5) - 2
    return max(1.0, base * (1.0 + period * cv * 0.25))


def _normalize_value(value: float, scale: float) -> float:
    if scale <= 0.0:
        return 0.0
    return max(0.0, value / scale)


def _build_waiter_maps(group_models: Dict[object, dict]) -> Tuple[Dict[object, List[object]], Dict[object, List[Tuple[object, int]]]]:
    full_waiters: Dict[object, List[object]] = defaultdict(list)
    release_waiters: Dict[object, List[Tuple[object, int]]] = defaultdict(list)
    for gid, model in group_models.items():
        for pred in model.get("full_preds", model.get("preds", [])):
            full_waiters[pred].append(gid)
        release_pred = model.get("release_pred")
        threshold = model.get("release_threshold")
        if release_pred is not None and threshold is not None:
            release_waiters[release_pred].append((gid, int(threshold)))
    return full_waiters, release_waiters


def _initialize_group_runtime(group_models: Dict[object, dict]) -> Dict[object, dict]:
    runtime = {}
    for gid, model in group_models.items():
        full_preds = list(model.get("full_preds", model.get("preds", [])))
        runtime[gid] = {
            "remaining": int(model["count"]),
            "scheduled": 0,
            "completed": 0,
            "ready": False,
            "ready_time": None,
            "instances": [],
            "completion_times": [],
            "full_remaining": len(full_preds),
            "full_ready_time": 0.0 if not full_preds else None,
            "release_ready_time": 0.0 if model.get("release_pred") is None else None,
            "end_time": None,
            "threshold_fired": False,
            "release_count_reached": 0,
        }
    return runtime


def _ready_heap_item(gid: object,
                     group_models: Dict[object, dict],
                     runtime: Dict[object, dict]) -> Tuple[float, int, str, object]:
    ready_time = float(runtime[gid]["ready_time"] or 0.0)
    queue_seq = int(runtime[gid].get("queue_seq", 0))
    return (ready_time, queue_seq, str(gid), gid)


def _maybe_mark_ready(gid: object,
                      group_models: Dict[object, dict],
                      runtime: Dict[object, dict],
                      ready_heap: List[Tuple[float, int, str, object]],
                      ready_seq: List[int]) -> None:
    state = runtime[gid]
    if state["ready"] or state["remaining"] <= 0:
        return
    if state["full_ready_time"] is None or state["release_ready_time"] is None:
        return
    state["ready"] = True
    state["ready_time"] = max(state["full_ready_time"], state["release_ready_time"])
    state["queue_seq"] = ready_seq[0]
    ready_seq[0] += 1
    heapq.heappush(ready_heap, _ready_heap_item(gid, group_models, runtime))


def _group_priority(model: dict, ready_time: float) -> Tuple[float, float, float, str]:
    return (
        -float(model["critical_path_score"]),
        ready_time,
        -float(model["queue_wait_score"]),
        str(model["name"]),
    )


def _requeue_group(gid: object,
                   group_models: Dict[object, dict],
                   runtime: Dict[object, dict],
                   ready_heap: List[Tuple[float, int, str, object]],
                   ready_seq: List[int]) -> None:
    if runtime[gid]["remaining"] <= 0:
        return
    runtime[gid]["queue_seq"] = ready_seq[0]
    ready_seq[0] += 1
    heapq.heappush(ready_heap, _ready_heap_item(gid, group_models, runtime))


def _bandwidth_penalty(active_memory_demand: float,
                       candidate_memory_demand: float,
                       num_sms: int,
                       bandwidth_scale: float) -> float:
    threshold = max(1.0, num_sms * max(0.10, 0.35 * bandwidth_scale))
    pressure = active_memory_demand + candidate_memory_demand
    if pressure <= threshold:
        return 1.0
    excess = (pressure - threshold) / threshold
    return 1.0 + excess * max(0.25, bandwidth_scale)


def _estimate_locality_gain(gid: object,
                            sm_id: int,
                            start_time: float,
                            group_models: Dict[object, dict],
                            edge_locality: Dict[Tuple[object, object], dict],
                            completed_groups: Dict[object, dict],
                            sm_cache: Dict[int, Dict[object, float]],
                            l2_cache: Dict[object, float],
                            locality_cap: float,
                            l2_factor: float,
                            sm_tau_scale: float,
                            l2_tau_scale: float) -> Tuple[float, float, float, float]:
    model = group_models[gid]
    total_gain = 0.0
    same_bytes = 0.0
    l2_bytes = 0.0
    total_bytes = 0.0
    for pred in model.get("preds", []):
        edge = edge_locality.get((pred, gid))
        if not edge or edge["overlap_bytes"] <= 0 or edge["consumer_input_bytes"] <= 0:
            continue
        pred_model = group_models.get(pred)
        if pred_model is None:
            continue
        pred_done = completed_groups.get(pred)
        if pred_done is None:
            continue
        total_bytes += edge["overlap_bytes"]
        tau_base = max(1.0, float(pred_model["base_duration_ns"]))
        tau_sm = tau_base * max(0.1, sm_tau_scale)
        tau_l2 = tau_base * 4.0 * max(0.1, l2_tau_scale)
        same_end = sm_cache.get(sm_id, {}).get(pred)
        l2_end = l2_cache.get(pred)
        same_score = 0.0
        l2_score = 0.0
        if same_end is not None:
            _, _, same_score = effective_reuse_score(
                edge["overlap_bytes"], edge["consumer_input_bytes"], max(0.0, start_time - same_end), tau_sm)
        if l2_end is not None:
            _, _, l2_score = effective_reuse_score(
                edge["overlap_bytes"], edge["consumer_input_bytes"], max(0.0, start_time - l2_end), tau_l2)
        if same_score >= l2_score * l2_factor and same_score > 0.0:
            total_gain += model["memory_weight"] * same_score
            same_bytes += edge["overlap_bytes"]
        elif l2_score > 0.0:
            total_gain += model["memory_weight"] * l2_score * l2_factor
            l2_bytes += edge["overlap_bytes"]
    return min(locality_cap, total_gain), same_bytes, l2_bytes, total_bytes


def _best_sm_for_group(gid: object,
                       group_models: Dict[object, dict],
                       runtime: Dict[object, dict],
                       sm_states: Dict[int, dict],
                       current_time: float,
                       policy: str,
                       rr_cursor: List[int],
                       edge_locality: Dict[Tuple[object, object], dict],
                       completed_groups: Dict[object, dict],
                       sm_cache: Dict[int, Dict[object, float]],
                       l2_cache: Dict[object, float],
                       locality_cap: float,
                       l2_factor: float,
                       sm_tau_scale: float,
                       l2_tau_scale: float,
                       bandwidth_scale: float) -> Optional[dict]:
    active_memory = sum(
        active["memory_demand"]
        for sm_state in sm_states.values()
        for active in sm_state["active"]
    )
    state = runtime[gid]
    model = group_models[gid]
    if state["remaining"] <= 0:
        return None
    ready_time = float(state["ready_time"] or 0.0)
    if ready_time > current_time + 1e-9:
        return None
    instance_idx = state["scheduled"]
    base_duration = _sample_duration(model, instance_idx)
    num_sms = len(sm_states)
    search_order = [int((rr_cursor[0] + offset) % num_sms) for offset in range(num_sms)]
    candidates = []
    for rr_rank, sm_id in enumerate(search_order):
        sm_state = sm_states[sm_id]
        _release_completed(sm_state, current_time)
        fit_time = _earliest_fit_time(sm_state, ready_time, model["resource_vector"])
        if fit_time > current_time + 1e-9:
            continue
        locality_gain, same_bytes, l2_bytes, total_bytes = _estimate_locality_gain(
            gid,
            sm_id,
            current_time,
            group_models,
            edge_locality,
            completed_groups,
            sm_cache,
            l2_cache,
            locality_cap,
            l2_factor,
            sm_tau_scale,
            l2_tau_scale,
        )
        penalty = _bandwidth_penalty(active_memory, model["memory_demand"], len(sm_states), bandwidth_scale)
        duration = max(1.0, base_duration * penalty * (1.0 - locality_gain))
        candidates.append({
            "gid": gid,
            "sm_id": sm_id,
            "ready_time": ready_time,
            "duration": duration,
            "locality_gain": locality_gain,
            "same_bytes": same_bytes,
            "l2_bytes": l2_bytes,
            "locality_bytes": total_bytes,
            "rr_rank": rr_rank,
            "finish_time": current_time + duration,
            "occupancy": sum(sm_state["used"]),
        })
    if not candidates:
        return None

    if policy != "affinity":
        chosen = min(candidates, key=lambda item: (item["rr_rank"], item["finish_time"], item["sm_id"]))
    else:
        best_gain = max(item["locality_gain"] for item in candidates)
        if best_gain < _LOCALITY_FALLBACK_THRESHOLD:
            chosen = min(candidates, key=lambda item: (item["rr_rank"], item["finish_time"], item["sm_id"]))
        else:
            chosen = min(
                candidates,
                key=lambda item: (
                    -item["locality_gain"],
                    item["finish_time"],
                    item["occupancy"],
                    item["rr_rank"],
                    item["sm_id"],
                ),
            )

    rr_cursor[0] = (chosen["sm_id"] + 1) % num_sms
    return chosen


def _dispatch_instance(dispatch: dict,
                       group_models: Dict[object, dict],
                       runtime: Dict[object, dict],
                       sm_states: Dict[int, dict],
                       current_time: float,
                       active_heap: List[Tuple[float, int, object, int]],
                       locality_totals: dict) -> None:
    gid = dispatch["gid"]
    sm_id = dispatch["sm_id"]
    model = group_models[gid]
    state = runtime[gid]
    sm_state = sm_states[sm_id]
    start_time = current_time
    end_time = start_time + dispatch["duration"]
    instance_idx = state["scheduled"]
    state["scheduled"] += 1
    state["remaining"] -= 1
    instance = {
        "group_id": gid,
        "instance_idx": instance_idx,
        "block": sm_id,
        "start": start_time,
        "end": end_time,
        "dur": dispatch["duration"],
        "queue_wait": max(0.0, start_time - dispatch["ready_time"]),
    }
    state["instances"].append(instance)
    sm_state["active"].append({
        "gid": gid,
        "end": end_time,
        "resource_vector": model["resource_vector"],
        "memory_demand": model["memory_demand"],
    })
    for idx, value in enumerate(model["resource_vector"]):
        sm_state["used"][idx] += value
    locality_totals["same_bytes"] += dispatch["same_bytes"]
    locality_totals["l2_bytes"] += dispatch["l2_bytes"]
    locality_totals["all_bytes"] += dispatch["locality_bytes"]
    heapq.heappush(active_heap, (end_time, sm_id, gid, instance_idx))


def _pop_next_completion(active_heap: List[Tuple[float, int, object, int]]) -> float:
    return active_heap[0][0]


def _remove_completed(active_heap: List[Tuple[float, int, object, int]],
                      current_time: float) -> List[Tuple[float, int, object, int]]:
    completed = []
    while active_heap and active_heap[0][0] <= current_time + 1e-9:
        completed.append(heapq.heappop(active_heap))
    return completed


def _finalize_completion(completed_items: List[Tuple[float, int, object, int]],
                         group_models: Dict[object, dict],
                         runtime: Dict[object, dict],
                         full_waiters: Dict[object, List[object]],
                         release_waiters: Dict[object, List[Tuple[object, int]]],
                         ready_heap: List[Tuple[float, int, str, object]],
                         ready_seq: List[int],
                         sm_cache: Dict[int, Dict[object, float]],
                         l2_cache: Dict[object, float],
                         completed_groups: Dict[object, dict],
                         current_time: float) -> None:
    by_group: Dict[object, List[Tuple[float, int, object, int]]] = defaultdict(list)
    for item in completed_items:
        by_group[item[2]].append(item)
    for gid, items in by_group.items():
        state = runtime[gid]
        model = group_models[gid]
        for _, sm_id, _, instance_idx in items:
            state["completed"] += 1
            state["completion_times"].append(current_time)
            sm_cache.setdefault(sm_id, {})[gid] = current_time
            l2_cache[gid] = current_time
            if state["completed"] == model["count"]:
                state["end_time"] = current_time
                completed_groups[gid] = {
                    "end_time": current_time,
                    "completion_times": list(state["completion_times"]),
                }
                for succ in full_waiters.get(gid, []):
                    succ_state = runtime[succ]
                    succ_state["full_remaining"] -= 1
                    if succ_state["full_remaining"] <= 0:
                        full_preds = group_models[succ].get("full_preds", group_models[succ].get("preds", []))
                        succ_state["full_ready_time"] = max(
                            (runtime[pred]["end_time"] for pred in full_preds),
                            default=0.0,
                        )
                        _maybe_mark_ready(succ, group_models, runtime, ready_heap, ready_seq)
        for succ, threshold in release_waiters.get(gid, []):
            succ_state = runtime[succ]
            if succ_state["release_ready_time"] is not None:
                continue
            if state["completed"] >= threshold:
                succ_state["release_ready_time"] = current_time
                _maybe_mark_ready(succ, group_models, runtime, ready_heap, ready_seq)


def _simulation_summary(group_models: Dict[object, dict],
                        runtime: Dict[object, dict],
                        sm_states: Dict[int, dict],
                        locality_totals: dict) -> dict:
    wall_time = max((state["end_time"] or 0.0 for state in runtime.values()), default=0.0)
    block_rows = []
    total_queue_wait = 0.0
    max_queue_wait = 0.0
    all_waits = []
    for sm_id, sm_state in sm_states.items():
        bars = []
        for gid, state in runtime.items():
            group = group_models[gid]
            for instance in state["instances"]:
                if instance["block"] != sm_id:
                    continue
                bars.append({
                    "start": instance["start"],
                    "end": instance["end"],
                    "name": group["name"],
                    "avg_dur": instance["dur"],
                    "blocks": 1,
                    "type": group["type"],
                })
                all_waits.append(instance["queue_wait"])
                total_queue_wait += instance["queue_wait"]
                max_queue_wait = max(max_queue_wait, instance["queue_wait"])
        bars.sort(key=lambda item: item["start"])
        block_rows.append({"label": f"block_{sm_id}", "bars": bars})

    bottlenecks = []
    for gid, state in runtime.items():
        waits = [instance["queue_wait"] for instance in state["instances"]]
        if not state["instances"]:
            continue
        bottlenecks.append({
            "name": group_models[gid]["name"],
            "type": group_models[gid]["type"],
            "span_us": ((state["end_time"] or 0.0) - (state["ready_time"] or 0.0)) / 1e3,
            "max_wait_us": max(waits, default=0.0) / 1e3,
            "avg_wait_us": (sum(waits) / max(1, len(waits))) / 1e3,
            "critical_path_score": group_models[gid]["critical_path_score"],
        })
    bottlenecks.sort(key=lambda item: (-item["max_wait_us"], -item["critical_path_score"], item["name"]))

    avg_util = 0.0
    if wall_time > 0.0:
        avg_util = 100.0 * sum(
            sum(bar["end"] - bar["start"] for bar in row["bars"]) / wall_time
            for row in block_rows
        ) / max(1, len(block_rows))

    return {
        "wall_time_ns": wall_time,
        "avg_util": avg_util,
        "avg_queue_wait_us": (total_queue_wait / max(1, len(all_waits))) / 1e3,
        "max_queue_wait_us": max_queue_wait / 1e3,
        "same_block_reuse_ratio": locality_totals["same_bytes"] / max(1.0, locality_totals["all_bytes"]),
        "l2_reuse_ratio": locality_totals["l2_bytes"] / max(1.0, locality_totals["all_bytes"]),
        "block_rows": block_rows,
        "bottlenecks": bottlenecks[:15],
        "groups": {
            gid: {
                "name": group_models[gid]["name"],
                "type": group_models[gid]["type"],
                "start": min((instance["start"] for instance in state["instances"]), default=0.0),
                "end": max((instance["end"] for instance in state["instances"]), default=0.0),
                "count": group_models[gid]["count"],
                "instances": list(state["instances"]),
                "ready_time": state["ready_time"],
            }
            for gid, state in runtime.items()
        },
        "queue_waits": all_waits,
    }


def simulate_policy(group_models: Dict[object, dict],
                    edge_locality: Dict[Tuple[object, object], dict],
                    num_sms: int,
                    policy: str,
                    locality_cap: float,
                    l2_factor: float,
                    sm_tau_scale: float,
                    l2_tau_scale: float,
                    bandwidth_scale: float) -> dict:
    models = deepcopy(group_models)
    sm_states = _make_sm_state(num_sms)
    runtime = _initialize_group_runtime(models)
    full_waiters, release_waiters = _build_waiter_maps(models)
    ready_heap: List[Tuple[float, int, str, object]] = []
    ready_seq = [0]
    completed_groups: Dict[object, dict] = {}
    sm_cache: Dict[int, Dict[object, float]] = defaultdict(dict)
    l2_cache: Dict[object, float] = {}
    locality_totals = {"same_bytes": 0.0, "l2_bytes": 0.0, "all_bytes": 0.0}
    rr_cursor = [0]

    for gid in models:
        _maybe_mark_ready(gid, models, runtime, ready_heap, ready_seq)

    current_time = 0.0
    active_heap: List[Tuple[float, int, object, int]] = []

    while True:
        while True:
            round_progress = False
            blocked = []
            while ready_heap:
                ready_time, _, _, gid = heapq.heappop(ready_heap)
                if runtime[gid]["remaining"] <= 0:
                    continue
                if ready_time > current_time + 1e-9:
                    heapq.heappush(ready_heap, (ready_time, runtime[gid].get("queue_seq", 0), str(gid), gid))
                    break
                dispatch = _best_sm_for_group(
                    gid,
                    models,
                    runtime,
                    sm_states,
                    current_time,
                    policy,
                    rr_cursor,
                    edge_locality,
                    completed_groups,
                    sm_cache,
                    l2_cache,
                    locality_cap,
                    l2_factor,
                    sm_tau_scale,
                    l2_tau_scale,
                    bandwidth_scale,
                )
                if dispatch is None:
                    blocked.append(gid)
                    continue
                round_progress = True
                _dispatch_instance(dispatch, models, runtime, sm_states, current_time, active_heap, locality_totals)
                if runtime[gid]["remaining"] > 0:
                    _requeue_group(gid, models, runtime, ready_heap, ready_seq)
            for gid in blocked:
                _requeue_group(gid, models, runtime, ready_heap, ready_seq)
            if not round_progress:
                break

        if all(runtime[gid]["completed"] >= models[gid]["count"] for gid in models):
            break
        if not active_heap:
            future_ready = [item[0] for item in ready_heap if item[0] > current_time + 1e-9]
            if not future_ready:
                break
            current_time = min(future_ready)
            continue
        current_time = _pop_next_completion(active_heap)
        completed_items = _remove_completed(active_heap, current_time)
        for sm_id, sm_state in sm_states.items():
            _release_completed(sm_state, current_time)
        _finalize_completion(
            completed_items,
            models,
            runtime,
            full_waiters,
            release_waiters,
            ready_heap,
            ready_seq,
            sm_cache,
            l2_cache,
            completed_groups,
            current_time,
        )

    return _simulation_summary(models, runtime, sm_states, locality_totals)


def build_split_candidates(group_models: Dict[object, dict],
                           locality_opportunities: List[dict],
                           edge_locality: Dict[Tuple[object, object], dict],
                           target_types: Iterable[str],
                           top_candidates: int) -> List[dict]:
    target_set = {target_type.strip() for target_type in target_types if target_type.strip()}
    incoming = defaultdict(list)
    for row in locality_opportunities:
        incoming[row["succ_gid"]].append(row)

    rows = []
    for gid, model in group_models.items():
        pred_rows = incoming.get(gid, [])
        if target_set and model["type"] not in target_set and not any(row["pred_type"] in target_set for row in pred_rows):
            continue
        dominant = None
        if pred_rows:
            dominant = max(pred_rows, key=lambda item: (item["effective_reuse"], item["overlap_bytes"]))
        else:
            for pred in model.get("preds", []):
                edge = edge_locality.get((pred, gid))
                if not edge:
                    continue
                overlap_ratio = edge["overlap_bytes"] / max(1.0, edge["consumer_input_bytes"])
                candidate = {
                    "pred_gid": pred,
                    "succ_gid": gid,
                    "pred_type": group_models[pred]["type"],
                    "succ_type": model["type"],
                    "overlap_bytes": edge["overlap_bytes"],
                    "effective_reuse": overlap_ratio,
                    "reuse_fraction": overlap_ratio,
                    "resource_risk": resource_risk((group_models[pred]["type"], model["type"])),
                }
                if dominant is None or (candidate["effective_reuse"], candidate["overlap_bytes"]) > (
                    dominant["effective_reuse"], dominant["overlap_bytes"]):
                    dominant = candidate
        locality_score = dominant["effective_reuse"] if dominant else 0.0
        score = (
            0.35 * model["queue_wait_score"] +
            0.25 * model["critical_path_score"] +
            0.20 * model["fan_in_score"] +
            0.15 * locality_score +
            0.05 * model["duration_variability_score"]
        )
        rows.append({
            "succ_gid": gid,
            "succ_name": model["name"],
            "succ_type": model["type"],
            "pred_gid": dominant["pred_gid"] if dominant else None,
            "pred_type": dominant["pred_type"] if dominant else "",
            "candidate_score": score,
            "normalized_queue_wait": model["queue_wait_score"],
            "critical_path_score": model["critical_path_score"],
            "fan_in_score": model["fan_in_score"],
            "locality_score": locality_score,
            "duration_variability": model["duration_variability_score"],
            "overlap_bytes": dominant["overlap_bytes"] if dominant else 0,
            "resource_risk": dominant["resource_risk"] if dominant else resource_risk((model["type"],)),
        })
    rows.sort(key=lambda item: (-item["candidate_score"], -item["fan_in_score"], item["succ_name"]))
    return rows[:top_candidates]


def _clone_edge(edge: dict, pred_gid: object, succ_gid: object, overlap_scale: float, consumer_scale: float) -> dict:
    clone = dict(edge)
    clone["pred_gid"] = pred_gid
    clone["succ_gid"] = succ_gid
    clone["overlap_bytes"] = int(edge.get("overlap_bytes", 0) * overlap_scale)
    clone["consumer_input_bytes"] = int(max(1, edge.get("consumer_input_bytes", 0) * consumer_scale))
    return clone


def _transform_for_split(group_models: Dict[object, dict],
                         edge_locality: Dict[Tuple[object, object], dict],
                         split_candidates: List[dict],
                         split_factor: int,
                         split_overhead_ns: float,
                         merge_overhead_ns: float) -> Tuple[Dict[object, dict], Dict[Tuple[object, object], dict], List[dict]]:
    models = deepcopy(group_models)
    edges = deepcopy(edge_locality)
    candidate_by_succ = {row["succ_gid"]: row for row in split_candidates if row.get("pred_gid") is not None}
    split_rows = []
    if not candidate_by_succ:
        return models, edges, split_rows

    successors_of: Dict[object, List[object]] = defaultdict(list)
    predecessors_of: Dict[object, List[object]] = defaultdict(list)
    for gid, model in models.items():
        for succ in model.get("succs", []):
            successors_of[gid].append(succ)
        for pred in model.get("preds", []):
            predecessors_of[gid].append(pred)

    for succ_gid, candidate in candidate_by_succ.items():
        if succ_gid not in models:
            continue
        original = models[succ_gid]
        dominant_pred = candidate["pred_gid"]
        partial_ids = [f"{succ_gid}::partial::{idx}" for idx in range(split_factor)]
        merge_id = f"{succ_gid}::merge"
        other_preds = [pred for pred in original.get("preds", []) if pred != dominant_pred]
        original_succs = list(original.get("succs", []))
        original_name = original["name"]
        original_count = original["count"]
        original_output_bytes = max(1, original.get("output_bytes", original.get("input_bytes", 1)))
        for idx, partial_id in enumerate(partial_ids):
            partial = deepcopy(original)
            partial["id"] = partial_id
            partial["name"] = f"{original_name}_PARTIAL{idx}"
            partial["preds"] = list(other_preds)
            partial["full_preds"] = list(other_preds)
            partial["release_pred"] = dominant_pred
            partial["release_fraction"] = (idx + 1) / split_factor
            partial["release_threshold"] = int(math.ceil(group_models[dominant_pred]["count"] * partial["release_fraction"]))
            partial["succs"] = [merge_id]
            partial["count"] = original_count
            partial["base_duration_ns"] = max(1.0, original["base_duration_ns"] / split_factor + split_overhead_ns)
            partial["input_bytes"] = max(1, int(original.get("input_bytes", 0) / split_factor))
            partial["output_bytes"] = max(1, int(original_output_bytes / split_factor))
            partial["memory_demand"] = max(0.05, original["memory_demand"] / split_factor)
            partial["split_parent"] = succ_gid
            partial["split_kind"] = "partial"
            models[partial_id] = partial
            split_rows.append({
                "name": partial["name"],
                "pred_type": candidate["pred_type"],
                "succ_type": original["type"],
                "split_factor": split_factor,
                "resource_risk": candidate["resource_risk"],
            })

        merge = deepcopy(original)
        merge["id"] = merge_id
        merge["name"] = f"{original_name}_MERGE"
        merge["preds"] = list(partial_ids)
        merge["full_preds"] = list(partial_ids)
        merge["release_pred"] = None
        merge["release_fraction"] = None
        merge["release_threshold"] = None
        merge["succs"] = list(original_succs)
        merge["count"] = original_count
        merge["base_duration_ns"] = max(1.0, merge_overhead_ns + original["base_duration_ns"] * 0.10)
        merge["input_bytes"] = max(1, original_output_bytes)
        merge["output_bytes"] = max(1, original_output_bytes)
        merge["memory_demand"] = max(0.10, original["memory_demand"] * 0.50)
        merge["split_parent"] = succ_gid
        merge["split_kind"] = "merge"
        models[merge_id] = merge

        for pred in original.get("preds", []):
            if succ_gid in models[pred]["succs"]:
                models[pred]["succs"] = [merge_id if succ == succ_gid and pred != dominant_pred else succ for succ in models[pred]["succs"]]
                if pred == dominant_pred:
                    models[pred]["succs"] = [succ for succ in models[pred]["succs"] if succ != merge_id]
                    models[pred]["succs"].extend(partial_ids)
            edge = edges.pop((pred, succ_gid), None)
            if edge is None:
                continue
            for partial_id in partial_ids:
                edges[(pred, partial_id)] = _clone_edge(edge, pred, partial_id, 1.0 / split_factor, 1.0 / split_factor)
        for partial_id in partial_ids:
            edges[(partial_id, merge_id)] = {
                "pred_gid": partial_id,
                "succ_gid": merge_id,
                "pred_type": models[partial_id]["type"],
                "succ_type": merge["type"],
                "overlap_bytes": max(1, int(original_output_bytes / split_factor)),
                "consumer_input_bytes": max(1, original_output_bytes),
                "pred_output_bytes": max(1, int(original_output_bytes / split_factor)),
            }
        for succ in original_succs:
            preds = models[succ].get("preds", [])
            models[succ]["preds"] = [merge_id if pred == succ_gid else pred for pred in preds]
            full_preds = models[succ].get("full_preds", preds)
            models[succ]["full_preds"] = [merge_id if pred == succ_gid else pred for pred in full_preds]
            edge = edges.pop((succ_gid, succ), None)
            if edge is not None:
                edges[(merge_id, succ)] = dict(edge, pred_gid=merge_id)
        models.pop(succ_gid, None)

    for gid, model in list(models.items()):
        model["succs"] = list(dict.fromkeys(model.get("succs", [])))
        model["preds"] = [pred for pred in model.get("preds", []) if pred in models]
        model["full_preds"] = [pred for pred in model.get("full_preds", model.get("preds", [])) if pred in models]
        model["succs"] = [succ for succ in model.get("succs", []) if succ in models]

    return models, edges, split_rows


def simulate_finer_tasks(group_models: Dict[object, dict],
                         edge_locality: Dict[Tuple[object, object], dict],
                         split_candidates: List[dict],
                         num_sms: int,
                         split_factors: Iterable[int],
                         split_overhead_ns: float,
                         merge_overhead_ns: float,
                         locality_cap: float,
                         l2_factor: float,
                         sm_tau_scale: float,
                         l2_tau_scale: float,
                         bandwidth_scale: float) -> dict:
    baseline_total = sum(model["base_duration_ns"] * model["count"] for model in group_models.values())
    best_result = None
    for split_factor in split_factors:
        transformed_models, transformed_edges, split_rows = _transform_for_split(
            group_models,
            edge_locality,
            split_candidates,
            int(split_factor),
            split_overhead_ns,
            merge_overhead_ns,
        )
        result = simulate_policy(
            transformed_models,
            transformed_edges,
            num_sms,
            policy="baseline",
            locality_cap=locality_cap,
            l2_factor=l2_factor,
            sm_tau_scale=sm_tau_scale,
            l2_tau_scale=l2_tau_scale,
            bandwidth_scale=bandwidth_scale,
        )
        transformed_total = sum(model["base_duration_ns"] * model["count"] for model in transformed_models.values())
        result["split_candidates"] = split_rows
        result["best_factor"] = int(split_factor)
        result["scheduling_overhead_us"] = max(0.0, transformed_total - baseline_total) / 1e3
        result["baseline_wall_ns"] = 0.0
        result["net_delta_us"] = 0.0
        result["bubble_fill_estimate_us"] = 0.0
        if best_result is None or result["wall_time_ns"] < best_result["wall_time_ns"]:
            best_result = result
    return best_result
