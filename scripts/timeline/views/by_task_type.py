"""ByTaskTypeView — one row per task type, sorted by first occurrence."""
from __future__ import annotations

from collections import defaultdict
from typing import List

from timeline import _short, _color_for, SCHEDULER_TYPES, _parse_trace_name
from timeline.mode_detector import GraphMode
from timeline.views import BarData, RowData, ViewAdapter, ViewContext


class ByTaskTypeView(ViewAdapter):
    """One row per distinct task type, bars sorted by start time."""

    @property
    def tab_id(self) -> str:
        return "tt"

    @property
    def tab_label(self) -> str:
        return "By Task Type"

    def is_applicable(self, mode: GraphMode) -> bool:
        return True

    def build_rows(self, ctx: ViewContext) -> List[RowData]:
        # Aggregate slices into per-(task_type, event_no) entries
        agg = defaultdict(list)
        for name, ts, dur, track_id in ctx.slices:
            tts, eno, _ = _parse_trace_name(name)
            agg[(tts, eno)].append((ts, dur))

        entries = []
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
            return []

        global_start = entries[0]["start"]
        compute_entries = [e for e in entries if e["type"] not in SCHEDULER_TYPES]

        # Group by task type
        type_groups: dict = defaultdict(list)
        for e in compute_entries:
            type_groups[e["type"]].append(e)

        sorted_types = sorted(
            type_groups.keys(),
            key=lambda t: type_groups[t][0]["start"],
        )

        rows = []
        for tts in sorted_types:
            bars = [
                BarData(
                    start=e["start"] - global_start,
                    end=e["end"] - global_start,
                    name=f'{e["type"]}_{e["event_no"]}',
                    blocks=e["blocks"],
                    avg_dur=e["avg_dur"],
                    color=_color_for(tts),
                )
                for e in type_groups[tts]
            ]
            rows.append(RowData(label=_short(tts), bars=bars))
        return rows
