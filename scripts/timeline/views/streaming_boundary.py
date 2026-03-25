"""StreamingBoundaryView — visual split between prelaunched and streaming tasks."""
from __future__ import annotations

from collections import defaultdict
from typing import List

from timeline.mode_detector import ExecutionMode, GraphMode
from timeline.views import BarData, RowData, ViewAdapter, ViewContext
from timeline import _short, _color_for


class StreamingBoundaryView(ViewAdapter):
    """Splits resident tasks into prelaunched (execution_kind=0) and streaming
    (execution_kind=1) groups, separated by a visual boundary row.

    Only applicable for streaming_data mode.
    """

    @property
    def tab_id(self) -> str:
        return 'stream'

    @property
    def tab_label(self) -> str:
        return 'Streaming Boundary'

    def is_applicable(self, mode: GraphMode) -> bool:
        return mode.execution_mode == ExecutionMode.STREAMING_DATA

    def build_rows(self, ctx: ViewContext) -> List[RowData]:
        # Group data_nodes by resident_task_id, separated into prelaunched/streaming
        pre_groups: dict = defaultdict(list)   # resident_task_id -> [node]
        str_groups: dict = defaultdict(list)
        for node in ctx.data_nodes:
            rid = node['resident_task_id']
            if node.get('execution_kind', 0) == 1:
                str_groups[rid].append(node)
            else:
                pre_groups[rid].append(node)

        # Topo-order resident tasks by topological sort of dag_groups
        from timeline.mapper import _topo_sort
        stage_order = _topo_sort(ctx.dag_groups, ctx.adjacency)
        gid_order = {gid: idx for idx, gid in enumerate(stage_order)}
        rid_to_gid = {
            group["resident_task_id"]: group["group_id"]
            for group in ctx.dag_groups
            if "resident_task_id" in group
        }

        def _resident_sort_key(rid: int):
            gid = rid_to_gid.get(rid, rid)
            return (gid_order.get(gid, len(gid_order)), gid, rid)

        pre_rids = sorted(pre_groups, key=_resident_sort_key)
        str_rids = sorted(str_groups, key=_resident_sort_key)

        rows = []

        # Prelaunched rows
        for rid in pre_rids:
            nodes_for_rid = sorted(
                pre_groups[rid],
                key=lambda n: ctx.data_timing.get(n['data_id'], {}).get('start_time', 0)
            )
            bars = []
            for node in nodes_for_rid:
                timing = ctx.data_timing.get(node['data_id'])
                if not timing:
                    continue
                bars.append(BarData(
                    start=timing['start_time'] - ctx.global_start,
                    end=timing['end_time'] - ctx.global_start,
                    name=node['trace_name'],
                    avg_dur=timing['end_time'] - timing['start_time'],
                    color=_color_for(node['task_type_name']),
                    data_id=node['data_id'],
                    execution_kind=0,
                ))
            if bars:
                rows.append(RowData(
                    label=f"{_short(nodes_for_rid[0]['task_type_name'])}_{rid} (prelaunched)",
                    bars=bars,
                ))

        # Separator
        rows.append(RowData(label='\u2501\u2501\u2501 STREAMING BOUNDARY \u2501\u2501\u2501', bars=[]))

        # Streaming rows
        for rid in str_rids:
            nodes_for_rid = sorted(
                str_groups[rid],
                key=lambda n: ctx.data_timing.get(n['data_id'], {}).get('start_time', 0)
            )
            bars = []
            for node in nodes_for_rid:
                timing = ctx.data_timing.get(node['data_id'])
                if not timing:
                    continue
                bars.append(BarData(
                    start=timing['start_time'] - ctx.global_start,
                    end=timing['end_time'] - ctx.global_start,
                    name=node['trace_name'],
                    avg_dur=timing['end_time'] - timing['start_time'],
                    color=_color_for(node['task_type_name']),
                    data_id=node['data_id'],
                    execution_kind=1,
                ))
            if bars:
                rows.append(RowData(
                    label=f"{_short(nodes_for_rid[0]['task_type_name'])}_{rid} (streaming)",
                    bars=bars,
                ))

        return rows
