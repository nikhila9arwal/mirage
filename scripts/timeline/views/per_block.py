"""PerBlockView — one row per GPU block, bars are individual task executions."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import List

from timeline import _color_for, SCHEDULER_TYPES, _parse_trace_name, _stage_trace_name
from timeline.mode_detector import GraphMode
from timeline.views import BarData, RowData, ViewAdapter, ViewContext


class PerBlockView(ViewAdapter):
    """One row per GPU block (SM), each bar is a single task slice on that block."""

    @property
    def tab_id(self) -> str:
        return "blk"

    @property
    def tab_label(self) -> str:
        return "Per Block"

    def is_applicable(self, mode: GraphMode) -> bool:
        return True

    def build_rows(self, ctx: ViewContext) -> List[RowData]:
        if not ctx.slices:
            return []

        # Determine global_start from the earliest slice
        global_start = min(ts for _, ts, _, _ in ctx.slices)

        # Group slices by track (block)
        block_slices: dict = defaultdict(list)
        for name, ts, dur, track_id in ctx.slices:
            tts, eno, data_id = _parse_trace_name(name)
            if tts in SCHEDULER_TYPES:
                continue
            block_slices[track_id].append(
                (name, _stage_trace_name(tts, eno), data_id, ts, dur, tts)
            )

        def _block_sort_key(tid):
            name = ctx.track_names.get(tid, "")
            m = re.match(r"block_(\d+)", name)
            return int(m.group(1)) if m else tid

        rows = []
        for tid in sorted(block_slices.keys(), key=_block_sort_key):
            tname = ctx.track_names.get(tid, f"track_{tid}")
            sorted_sl = sorted(block_slices[tid], key=lambda s: s[3])
            bars = [
                BarData(
                    start=ts - global_start,
                    end=ts + dur - global_start,
                    name=name,
                    trace_key=trace_key,
                    data_id=data_id,
                    blocks=1,
                    avg_dur=dur,
                    color=_color_for(tts),
                    execution_kind=(
                        int(ctx.data_nodes[data_id].get("execution_kind", 0))
                        if 0 <= data_id < len(ctx.data_nodes)
                        else 0
                    ),
                )
                for name, trace_key, data_id, ts, dur, tts in sorted_sl
            ]
            rows.append(RowData(label=tname, bars=bars))
        return rows
