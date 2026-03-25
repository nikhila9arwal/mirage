#!/usr/bin/env python3
"""Render a Mirage task-graph JSON and Perfetto trace as an interactive HTML timeline.

Usage:
    python scripts/display_task_graph_timeline.py task_graph_0.json mirage_0.perfetto-trace [-o output.html]

This script requires traces emitted with exact ``profiler_group_id`` values.
It does not support the older per-block counter naming scheme.
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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

# --- Imports from the timeline package ---
from timeline import (
    _TRACE_NAME_MAP,
    _SHORT_NAMES,
    TASK_COLORS,
    SCHEDULER_TYPES,
    STRAGGLER_RATIO,
    _color_for,
    _short,
    _parse_trace_name,
    _stage_trace_name,
    _data_trace_name,
)
from timeline.graph_parser import (
    parse_task_graph,
    get_index_from_id,
    BASE_EVENT,
    INVALID_PROFILER_GROUP_ID,
    UNPROFILED_TASK_TYPES,
    _task_profiler_group_id,
    build_stage_sequence,
    detect_layer_pattern,
    build_dependency_dag,
    build_data_dependency_dag,
)
from timeline.trace_parser import (
    parse_perfetto_trace,
    filter_trace_slices_to_graph,
)
from timeline.mapper import (
    map_trace_to_graph,
    map_data_trace_to_graph,
)
from timeline.metrics import (
    compute_schedule_metrics,
    compute_data_schedule_metrics,
    _generate_sched_html,
    _generate_data_sched_html,
)
from timeline.mode_detector import ExecutionMode, GraphMode, detect_mode
from timeline.views import ViewContext
from timeline.views.pipeline_phases import PipelinePhasesView
from timeline.views.by_task_type import ByTaskTypeView
from timeline.views.data_overlap import DataOverlapView
from timeline.views.per_block import PerBlockView
from timeline.views.streaming_boundary import StreamingBoundaryView
from timeline.html_generator import (
    generate_html as _gen_html,
    generate_whatif_html as _gen_whatif_html,
)

# ---------------------------------------------------------------------------
# View registry — ordered list of all timeline tabs
# ---------------------------------------------------------------------------
VIEW_REGISTRY = [
    PipelinePhasesView(),
    ByTaskTypeView(),
    DataOverlapView(),
    StreamingBoundaryView(),
    PerBlockView(),
]


# ---------------------------------------------------------------------------
# Time-bounds and type-stats helpers (previously inside build_views)
# ---------------------------------------------------------------------------
def _compute_global_bounds(slices):
    """Return (global_start, total_dur) from trace slices."""
    agg = defaultdict(list)
    for name, ts, dur, _track_id in slices:
        tts, eno, _ = _parse_trace_name(name)
        agg[(tts, eno)].append((ts, dur))

    if not agg:
        return 0, 1

    entries = []
    for (_tts, _eno), records in agg.items():
        starts = [r[0] for r in records]
        ends = [r[0] + r[1] for r in records]
        entries.append({"start": min(starts), "end": max(ends)})
    entries.sort(key=lambda e: e["start"])

    global_start = entries[0]["start"]
    global_end = max(e["end"] for e in entries)
    return global_start, max(1, global_end - global_start)


def _compute_type_stats(slices):
    """Return per-task-type stats dict from trace slices."""
    agg = defaultdict(list)
    for name, ts, dur, _track_id in slices:
        tts, eno, _ = _parse_trace_name(name)
        agg[(tts, eno)].append((ts, dur))

    entries = []
    for (tts, eno), records in agg.items():
        starts = [r[0] for r in records]
        ends = [r[0] + r[1] for r in records]
        durs = [r[1] for r in records]
        entries.append({
            "type": tts,
            "start": min(starts),
            "end": max(ends),
            "avg_dur": sum(durs) / len(durs),
        })

    type_stats = defaultdict(lambda: {"count": 0, "total_dur": 0, "sum_avg": 0})
    for e in entries:
        if e["type"] in SCHEDULER_TYPES:
            continue
        st = type_stats[e["type"]]
        st["count"] += 1
        st["total_dur"] += e["end"] - e["start"]
        st["sum_avg"] += e["avg_dur"]
    return dict(type_stats)


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
                tts, _, _ = _parse_trace_name(bar["name"])
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
            task_type, _, _ = _parse_trace_name(bar["name"])
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
                task_type, _, _ = _parse_trace_name(bar["name"])
                seen.setdefault(bar["color"], whatif_short_name(task_type))
    return [{"color": color, "name": name} for color, name in seen.items()]


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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Combine task graph + Perfetto trace into a Gantt timeline"
    )
    parser.add_argument("task_graph", help="Path to task graph JSON file")
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
    graph_data = parse_task_graph(args.task_graph)
    events = graph_data["events"]
    tasks = graph_data["tasks"]
    if graph_data["schema_version"] >= 2 and graph_data["resident_tasks"]:
        print(
            f"  {len(events)} control events, "
            f"{graph_data['graph_task_count']} resident tasks, "
            f"{graph_data['graph_data_count']} data items"
        )
    else:
        print(f"  {len(events)} events, {len(tasks)} tasks")

    print("Loading Perfetto trace …")
    slices, track_names = parse_perfetto_trace(args.trace)
    print(f"  {len(slices)} trace slices, {len(track_names)} tracks")

    print("Building pipeline stage sequence …")
    stage_seq = build_stage_sequence(graph_data)
    print(f"  {len(stage_seq)} stages")
    period = detect_layer_pattern(stage_seq)
    if period:
        print(f"  Detected layer period = {period} stages")

    print("Building dependency DAG …")
    dag_groups, adjacency, reverse_adj = build_dependency_dag(graph_data)
    print(f"  {len(dag_groups)} task groups, "
          f"{sum(len(v) for v in adjacency.values())} edges")

    data_nodes = []
    data_adj = {}
    data_reverse_adj = {}
    if graph_data["schema_version"] >= 2 and graph_data["resident_tasks"]:
        print("Building data dependency DAG …")
        data_nodes, data_adj, data_reverse_adj = build_data_dependency_dag(graph_data)
        print(f"  {len(data_nodes)} data nodes, "
              f"{sum(len(v) for v in data_adj.values())} data edges")

    slices, dropped_slices = filter_trace_slices_to_graph(slices, dag_groups, data_nodes)
    if dropped_slices:
        print(f"  Filtered out {dropped_slices} nested/non-graph trace slices")

    print("Computing time bounds …")
    g_start, t_dur = _compute_global_bounds(slices)
    t_stats = _compute_type_stats(slices)

    print("Mapping trace to task graph …")
    group_deps, group_timing, dag_to_trace = map_trace_to_graph(
        slices, dag_groups, adjacency, reverse_adj, g_start)
    print(f"  Mapped {len(group_deps)} trace groups to DAG")

    data_timing: dict = {}
    data_sched_metrics = None
    if data_nodes:
        print("Mapping trace to data DAG …")
        data_deps, data_timing, data_to_trace = map_data_trace_to_graph(
            slices, data_nodes, data_adj, data_reverse_adj, g_start)
        print(f"  Mapped {len(data_deps)} data-aware trace groups to DAG")
        data_sched_metrics = compute_data_schedule_metrics(
            data_nodes, data_timing, data_adj, data_reverse_adj, group_timing)
        if data_sched_metrics is not None:
            ds = data_sched_metrics["summary"]
            print(
                f"  Data overlap ratio: {ds['overall_overlap_ratio'] * 100:.1f}%, "
                f"data critical path: {ds['critical_path_us']:.1f} us, "
                f"avg data queue wait: {ds['avg_queue_wait_us']:.1f} us"
            )

    print("Computing schedule metrics …")
    sched_metrics = compute_schedule_metrics(
        slices, track_names, group_timing, dag_groups,
        adjacency, reverse_adj, dag_to_trace, g_start, t_dur, group_deps)
    sm = sched_metrics["summary"]
    print(f"  Avg utilization: {sm['avg_util']}%, "
          f"critical path: {sm['critical_path_us']:.1f} us, "
          f"avg queue wait: {sm['avg_queue_wait_us']:.1f} us")

    # --- Detect graph mode ---
    mode = detect_mode(graph_data)
    print(f"  Mode: {mode.execution_mode.value} "
          f"(schema v{mode.schema_version}, "
          f"{mode.num_prelaunched} prelaunched, "
          f"{mode.num_streaming} streaming)")

    # --- Build ViewContext and run registry ---
    ctx = ViewContext(
        mode=mode,
        graph_data=graph_data,
        slices=slices,
        track_names=track_names,
        stage_seq=stage_seq,
        dag_groups=dag_groups,
        adjacency=adjacency,
        reverse_adj=reverse_adj,
        data_nodes=data_nodes,
        data_adj=data_adj,
        data_reverse_adj=data_reverse_adj,
        group_deps=group_deps,
        group_timing=group_timing,
        data_timing=data_timing,
        global_start=g_start,
        total_dur=t_dur,
        sched_metrics=sched_metrics,
        data_sched_metrics=data_sched_metrics,
    )

    print("Building views …")
    view_rows: dict = {}
    for adapter in VIEW_REGISTRY:
        if adapter.is_applicable(mode):
            rows = adapter.build_rows(ctx)
            view_rows[adapter.tab_id] = [r.to_dict() for r in rows]
            print(f"  {adapter.tab_label}: {len(rows)} rows")

    pip_rows = view_rows.get("pip", [])
    tt_rows = view_rows.get("tt", [])
    data_rows = view_rows.get("data", [])
    blk_rows = view_rows.get("blk", [])

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
        _gen_whatif_html(whatif_data, args.output)
    else:
        print("Writing HTML …")
        _gen_html(
            view_rows,
            [a for a in VIEW_REGISTRY if a.is_applicable(mode)],
            g_start, t_dur, t_stats,
            (len(events), graph_data["graph_task_count"]),
            group_deps,
            sched_metrics,
            data_sched_metrics,
            args.output,
        )

    if args.svg:
        print("Writing SVG …")
        generate_svg(tt_rows, g_start, t_dur, args.svg)

    print("Done!")


if __name__ == "__main__":
    main()
