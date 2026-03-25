"""DataOverlapView — per-resident-task rows, each bar is one data item."""
from __future__ import annotations

from typing import List

from timeline.mapper import build_data_overlap_rows, _topo_sort
from timeline.metrics import _generate_data_sched_html
from timeline.mode_detector import GraphMode
from timeline.views import BarData, RowData, ViewAdapter, ViewContext

# Saturated cyan tint applied to streaming bars in the Data Overlap view.
_STREAMING_ACCENT = "#00e5ff"


def _streaming_color(base_color: str) -> str:
    """Blend base_color 60/40 toward cyan for streaming data items.

    Works on 6-digit hex colors only; passes through anything else unchanged.
    """
    try:
        if not base_color.startswith("#") or len(base_color) != 7:
            return base_color
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        # Blend 60% original + 40% cyan (#00e5ff)
        r2 = int(r * 0.6 + 0x00 * 0.4)
        g2 = int(g * 0.6 + 0xe5 * 0.4)
        b2 = int(b * 0.6 + 0xff * 0.4)
        return f"#{r2:02x}{g2:02x}{b2:02x}"
    except (ValueError, IndexError):
        return base_color


class DataOverlapView(ViewAdapter):
    """Shows one row per resident task group; each bar is one data item.

    Only applicable when the graph has a data DAG (resident_data / streaming_data).
    """

    @property
    def tab_id(self) -> str:
        return "data"

    @property
    def tab_label(self) -> str:
        return "Data Overlap"

    def is_applicable(self, mode: GraphMode) -> bool:
        return mode.has_data_dag

    def build_rows(self, ctx: ViewContext) -> List[RowData]:
        if not ctx.data_nodes or not ctx.data_timing:
            return []

        stage_order = _topo_sort(ctx.dag_groups, ctx.adjacency)
        raw_rows = build_data_overlap_rows(
            ctx.data_nodes, ctx.data_timing, stage_order, ctx.global_start
        )

        is_streaming = ctx.mode.is_streaming
        rows = []
        for rd in raw_rows:
            bars = []
            for b in rd["bars"]:
                # For data nodes, look up execution_kind from data_nodes list
                data_id = b.get("data_id", -1)
                exec_kind = 0
                if data_id >= 0 and data_id < len(ctx.data_nodes):
                    exec_kind = ctx.data_nodes[data_id].get("execution_kind", 0)
                # Streaming data items get a slightly lighter/distinct hue via alpha
                color = b.get("color", "#888")
                if is_streaming and exec_kind == 1:
                    color = _streaming_color(color)
                bars.append(BarData(
                    start=b["start"],
                    end=b["end"],
                    name=b["name"],
                    blocks=b.get("blocks", 1),
                    avg_dur=b.get("avg_dur", b["end"] - b["start"]),
                    color=color,
                    data_id=data_id,
                    execution_kind=exec_kind,
                ))
            rows.append(RowData(label=rd["label"], bars=bars))
        return rows

    def build_analysis_html(self, ctx: ViewContext) -> str:
        if not ctx.data_sched_metrics:
            return ""
        data_sched_html = _generate_data_sched_html(ctx.data_sched_metrics)
        if not data_sched_html:
            return ""
        return (
            '<div class="sa">'
            "<h3>Data-Level Analysis</h3>"
            '<p style="font-size:12px;color:#888;margin:0 0 10px">'
            "Per-data timing and overlap metrics. This section is only populated "
            "when the trace carries resident data ids.</p>"
            f"{data_sched_html}</div>"
        )
