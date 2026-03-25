"""Schedule metrics computation and HTML generation for analysis sections."""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from . import (
    _parse_trace_name,
    _short,
    _stage_trace_name,
    SCHEDULER_TYPES,
)


def compute_schedule_metrics(slices, track_names, group_timing, dag_groups,
                             adjacency, reverse_adj, dag_to_trace,
                             global_start, total_dur, group_deps):
    """Compute schedule quality metrics.

    Parameters
    ----------
    slices : list of (name, ts, dur, track_id)
        Raw Perfetto slices -- used for block utilization (sum of raw durations).
    group_timing : dict[dag_group_id -> {start_time, end_time, avg_dur, ready_time}]
        Timing from exact profiler-group mapping. Used for the critical path DP.
    group_deps : dict[trace_name -> {gs, ge, rt, ad, p, s}]
        Timing and dependency info from exact DAG mapping. Used for queue-wait
        computation and dependency highlighting.

    Returns
    -------
    dict with keys: block_util, top_waits, critical_path, summary.
    """
    # ---- Block utilization ----
    block_busy: Dict[int, int] = defaultdict(int)
    block_names: Dict[int, str] = {}
    for name, ts, dur, track_id in slices:
        tts, _, _ = _parse_trace_name(name)
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

    # ---- Per-invocation queue wait (from dependency-aware group_deps) ----
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
        gs = gd["gs"]
        rt = gd["rt"]
        ge = gd["ge"]
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


def compute_data_schedule_metrics(data_nodes,
                                  data_timing,
                                  adjacency,
                                  reverse_adj,
                                  group_timing):
    """Compute per-data schedule metrics for schema-v2 resident graphs."""
    if not data_timing:
        return None

    data_metrics = []
    longest: Dict[int, float] = {}
    pred_on_path: Dict[int, Optional[int]] = {}

    def dp(data_id: int) -> float:
        if data_id in longest:
            return longest[data_id]
        timing = data_timing.get(data_id)
        if not timing:
            longest[data_id] = 0.0
            pred_on_path[data_id] = None
            return 0.0
        dur = timing["end_time"] - timing["start_time"]
        best_pred = None
        best_val = 0.0
        for pred_id in reverse_adj.get(data_id, []):
            pred_val = dp(pred_id)
            if pred_val > best_val:
                best_val = pred_val
                best_pred = pred_id
        longest[data_id] = best_val + dur
        pred_on_path[data_id] = best_pred
        return longest[data_id]

    for data_id in data_timing:
        dp(data_id)
        timing = data_timing[data_id]
        queue_wait = max(0, timing["start_time"] - timing["ready_time"])
        data_metrics.append({
            "name": data_nodes[data_id]["trace_name"],
            "trace_key": data_nodes[data_id]["trace_name"],
            "data_id": data_id,
            "queue_wait_us": round(queue_wait / 1e3, 1),
            "dur_us": round((timing["end_time"] - timing["start_time"]) / 1e3, 1),
            "resident_task_id": data_nodes[data_id]["resident_task_id"],
        })

    critical_path = []
    if longest:
        end_data_id = max(longest, key=longest.get)
        current = end_data_id
        while current is not None:
            timing = data_timing[current]
            critical_path.append({
                "name": data_nodes[current]["trace_name"],
                "dur_us": round((timing["end_time"] - timing["start_time"]) / 1e3, 1),
            })
            current = pred_on_path.get(current)
        critical_path.reverse()

    overlap_stats: Dict[Tuple[int, int], dict] = {}
    for src_data_id, succ_ids in adjacency.items():
        src_node = data_nodes[src_data_id]
        pred_group_id = int(src_node["trace_event_no"])
        pred_stage = group_timing.get(pred_group_id)
        if pred_stage is None:
            continue
        pred_stage_end = pred_stage["end_time"]
        for dst_data_id in succ_ids:
            if dst_data_id not in data_timing:
                continue
            dst_node = data_nodes[dst_data_id]
            succ_group_id = int(dst_node["trace_event_no"])
            if pred_group_id == succ_group_id:
                continue
            stat = overlap_stats.setdefault(
                (pred_group_id, succ_group_id),
                {
                    "pred_name": _stage_trace_name(
                        src_node["task_type_name"], pred_group_id
                    ),
                    "succ_name": _stage_trace_name(
                        dst_node["task_type_name"], succ_group_id
                    ),
                    "succ_ids": set(),
                    "overlap_ids": set(),
                    "lead_us": [],
                },
            )
            stat["succ_ids"].add(dst_data_id)
            succ_start = data_timing[dst_data_id]["start_time"]
            if succ_start < pred_stage_end:
                stat["overlap_ids"].add(dst_data_id)
                stat["lead_us"].append((pred_stage_end - succ_start) / 1e3)

    overlap_edges = []
    total_overlap = 0
    total_edge_data = 0
    for (pred_group_id, succ_group_id), stat in overlap_stats.items():
        total = len(stat["succ_ids"])
        overlap = len(stat["overlap_ids"])
        if total == 0:
            continue
        total_overlap += overlap
        total_edge_data += total
        overlap_edges.append({
            "pred_name": stat["pred_name"],
            "succ_name": stat["succ_name"],
            "pred_group_id": pred_group_id,
            "succ_group_id": succ_group_id,
            "overlap_ratio": overlap / total,
            "overlap_count": overlap,
            "total_count": total,
            "avg_lead_us": round(sum(stat["lead_us"]) / len(stat["lead_us"]), 1)
            if stat["lead_us"] else 0.0,
            "max_lead_us": round(max(stat["lead_us"]), 1)
            if stat["lead_us"] else 0.0,
        })
    overlap_edges.sort(
        key=lambda item: (
            -item["overlap_ratio"],
            -item["overlap_count"],
            -item["avg_lead_us"],
            item["pred_name"],
            item["succ_name"],
        )
    )

    waits = [item["queue_wait_us"] for item in data_metrics if item["queue_wait_us"] > 0]
    summary = {
        "num_timed_data": len(data_timing),
        "avg_queue_wait_us": round(sum(waits) / len(waits), 1) if waits else 0.0,
        "max_queue_wait_us": round(max(waits), 1) if waits else 0.0,
        "p50_queue_wait_us": round(sorted(waits)[len(waits) // 2], 1) if waits else 0.0,
        "p90_queue_wait_us": round(sorted(waits)[int(len(waits) * 0.9)], 1)
        if waits else 0.0,
        "critical_path_us": round(max(longest.values()) / 1e3, 1) if longest else 0.0,
        "overall_overlap_ratio": (total_overlap / total_edge_data) if total_edge_data else 0.0,
        "edge_pairs_with_overlap": sum(1 for item in overlap_edges if item["overlap_count"] > 0),
        "total_stage_edges": len(overlap_edges),
    }

    top_waits = sorted(data_metrics, key=lambda item: -item["queue_wait_us"])[:20]
    return {
        "summary": summary,
        "top_waits": top_waits,
        "critical_path": critical_path,
        "overlap_edges": overlap_edges,
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


def _generate_data_sched_html(metrics):
    """Return HTML string for the Data-Level Analysis section."""
    if not metrics:
        return ""

    s = metrics["summary"]
    h = '<div class="sched-grid">\n'

    h += '<div class="sched-card"><h4>Data Summary</h4>'
    h += '<p style="font-size:11px;color:#888;margin:0 0 6px">Per-data-item timing derived from the resident schema-v2 DAG and data-aware profiler slices.</p><table>\n'
    h += f'<tr><td>Timed data items</td><td><b>{s["num_timed_data"]}</b></td><td style="color:#888;font-size:11px">Distinct data items that appeared in the trace</td></tr>\n'
    h += f'<tr><td>Data critical path</td><td>{s["critical_path_us"]:.1f} us</td><td style="color:#888;font-size:11px">Longest path through the data DAG using actual observed timings</td></tr>\n'
    h += f'<tr><td>Avg Queue Wait</td><td>{s["avg_queue_wait_us"]:.1f} us</td><td style="color:#888;font-size:11px">Average delay after predecessor data completed</td></tr>\n'
    h += f'<tr><td>Max Queue Wait</td><td>{s["max_queue_wait_us"]:.1f} us</td><td style="color:#888;font-size:11px">Worst data-level ready-to-start gap</td></tr>\n'
    h += f'<tr><td>p50 / p90 Queue Wait</td><td>{s["p50_queue_wait_us"]:.1f} / {s["p90_queue_wait_us"]:.1f} us</td><td style="color:#888;font-size:11px">Median and 90th percentile data queue wait</td></tr>\n'
    h += f'<tr><td>Weighted overlap ratio</td><td>{s["overall_overlap_ratio"] * 100:.1f}%</td><td style="color:#888;font-size:11px">Fraction of downstream data items that started before the upstream stage fully ended</td></tr>\n'
    h += f'<tr><td>Edges with overlap</td><td>{s["edge_pairs_with_overlap"]} / {s["total_stage_edges"]}</td><td style="color:#888;font-size:11px">Resident-stage edges where at least one downstream data item overlapped</td></tr>\n'
    h += '</table></div>\n'

    h += '<div class="sched-card"><h4>Stage-Edge Overlap</h4>'
    h += '<p style="font-size:11px;color:#888;margin:0 0 6px">Top resident-stage edges ranked by how much data-level overlap they exposed.</p>'
    h += '<table><tr><th>Edge</th><th>Overlap</th><th>Count</th><th>Lead</th></tr>\n'
    for row in metrics["overlap_edges"][:15]:
        h += (
            f'<tr><td>{_short(row["pred_name"])} \u2192 {_short(row["succ_name"])}</td>'
            f'<td>{row["overlap_ratio"] * 100:.1f}%</td>'
            f'<td>{row["overlap_count"]}/{row["total_count"]}</td>'
            f'<td>{row["avg_lead_us"]:.1f} us avg</td></tr>\n'
        )
    h += '</table></div>\n'

    h += '<div class="sched-card"><h4>Top Data Queue Waits</h4>'
    h += '<p style="font-size:11px;color:#888;margin:0 0 6px">Data items with the largest gap between becoming ready and actually starting.</p>'
    h += '<table><tr><th>Data</th><th>Wait</th><th>Dur</th></tr>\n'
    for row in metrics["top_waits"][:15]:
        h += (
            f'<tr><td>{row["name"]}</td>'
            f'<td>{row["queue_wait_us"]:.1f} us</td>'
            f'<td>{row["dur_us"]:.1f} us</td></tr>\n'
        )
    h += '</table></div>\n'

    h += '<div class="sched-card"><h4>Data Critical Path</h4>'
    h += '<p style="font-size:11px;color:#888;margin:0 0 6px">Observed bottleneck path through the per-data DAG.</p>'
    h += '<div class="cp-flow">\n'
    for i, step in enumerate(metrics["critical_path"][:24]):
        if i > 0:
            h += '<span class="cp-arrow">&#8594;</span>'
        h += f'<span class="cp-node">{step["name"]}<br><small>{step["dur_us"]:.1f} us</small></span>'
    if len(metrics["critical_path"]) > 24:
        h += f'<span class="cp-arrow">&#8594;</span><span class="cp-node">... +{len(metrics["critical_path"]) - 24} more</span>'
    h += '</div></div>\n'

    h += '</div>\n'
    return h


def compute_streaming_metrics(data_nodes, data_timing, data_adj, data_reverse_adj, dag_groups):
    """Compute streaming-specific metrics.

    Returns a dict with:
    - summary: {prelaunched_count, streaming_count, prelaunched_time_us, streaming_time_us}
    - type_comparison: list of {task_type, prelaunched_avg_us, streaming_avg_us, speedup}
    - boundary_handoff: list of {pred_trace_name, succ_trace_name, handoff_us}
    """
    if not data_nodes or not data_timing:
        return None

    pre_times = []
    str_times = []
    type_pre = {}  # task_type_name -> list of durations
    type_str = {}

    for node in data_nodes:
        timing = data_timing.get(node['data_id'])
        if not timing:
            continue
        dur = (timing['end_time'] - timing['start_time']) / 1e3  # us
        tname = node['task_type_name']
        if node.get('execution_kind', 0) == 1:
            str_times.append(dur)
            type_str.setdefault(tname, []).append(dur)
        else:
            pre_times.append(dur)
            type_pre.setdefault(tname, []).append(dur)

    summary = {
        'prelaunched_count': len(pre_times),
        'streaming_count': len(str_times),
        'prelaunched_time_us': sum(pre_times),
        'streaming_time_us': sum(str_times),
        'prelaunched_avg_us': sum(pre_times) / len(pre_times) if pre_times else 0,
        'streaming_avg_us': sum(str_times) / len(str_times) if str_times else 0,
    }

    type_comparison = []
    for tname in sorted(set(list(type_pre) + list(type_str))):
        if tname in type_pre and tname in type_str:
            pa = sum(type_pre[tname]) / len(type_pre[tname])
            sa = sum(type_str[tname]) / len(type_str[tname])
            type_comparison.append({
                'task_type': tname,
                'prelaunched_avg_us': pa,
                'streaming_avg_us': sa,
                'speedup': pa / sa if sa > 0 else 0,
            })

    # Boundary handoff: edges from prelaunched producer to streaming consumer
    boundary_handoff = []
    node_by_id = {n['data_id']: n for n in data_nodes}
    for pred_id, succs in data_adj.items():
        pred_node = node_by_id.get(pred_id)
        if not pred_node or pred_node.get('execution_kind', 0) != 0:
            continue
        pred_timing = data_timing.get(pred_id)
        if not pred_timing:
            continue
        for succ_id in succs:
            succ_node = node_by_id.get(succ_id)
            if not succ_node or succ_node.get('execution_kind', 0) != 1:
                continue
            succ_timing = data_timing.get(succ_id)
            if not succ_timing:
                continue
            handoff_us = (succ_timing['start_time'] - pred_timing['end_time']) / 1e3
            if handoff_us > 0:
                boundary_handoff.append({
                    'pred_trace_name': pred_node['trace_name'],
                    'succ_trace_name': succ_node['trace_name'],
                    'handoff_us': handoff_us,
                })

    boundary_handoff.sort(key=lambda x: -x['handoff_us'])

    return {
        'summary': summary,
        'type_comparison': type_comparison[:20],
        'boundary_handoff': boundary_handoff[:20],
    }
