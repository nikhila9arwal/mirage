"""PipelinePhasesView — groups trace entries by pipeline layer/phase."""
from __future__ import annotations

from collections import Counter
from typing import List

from timeline import _TRACE_NAME_MAP, _short, _color_for, SCHEDULER_TYPES, _parse_trace_name
from timeline.mode_detector import GraphMode
from timeline.views import BarData, RowData, ViewAdapter, ViewContext


def _make_bar(entry: dict, global_start: float, tts_hint: str = "") -> BarData:
    return BarData(
        start=entry["start"] - global_start,
        end=entry["end"] - global_start,
        name=f'{entry["type"]}_{entry["event_no"]}',
        blocks=entry["blocks"],
        avg_dur=entry["avg_dur"],
        color=_color_for(tts_hint or entry["type"]),
    )


def _build_compute_entries(slices):
    """Aggregate slices into per-(task_type, event_no) entries, excluding scheduler types."""
    from collections import defaultdict
    agg = defaultdict(list)
    for name, ts, dur, track_id in slices:
        tts, eno, _ = _parse_trace_name(name)
        agg[(tts, eno)].append((ts, dur, track_id))

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
    return [e for e in entries if e["type"] not in SCHEDULER_TYPES]


class PipelinePhasesView(ViewAdapter):
    """Groups trace entries into per-pipeline-layer rows.

    Detects the repeating task type that marks the start of each transformer
    layer (typically RMS_NORM) and splits the trace into phase rows.
    """

    @property
    def tab_id(self) -> str:
        return "pip"

    @property
    def tab_label(self) -> str:
        return "Pipeline Phases"

    def is_applicable(self, mode: GraphMode) -> bool:
        return True

    def build_rows(self, ctx: ViewContext) -> List[RowData]:
        compute_entries = _build_compute_entries(ctx.slices)
        if not compute_entries:
            return []

        global_start = compute_entries[0]["start"]
        rows_dicts = _build_pipeline_rows(compute_entries, global_start, ctx.stage_seq)
        result = []
        for rd in rows_dicts:
            bars = [
                BarData(
                    start=b["start"],
                    end=b["end"],
                    name=b["name"],
                    blocks=b.get("blocks", 1),
                    avg_dur=b.get("avg_dur", b["end"] - b["start"]),
                    color=b.get("color", "#888"),
                )
                for b in rd["bars"]
            ]
            result.append(RowData(label=rd["label"], bars=bars))
        return result


def _build_pipeline_rows(compute_entries: list, global_start: float, stage_seq: list) -> list:
    """Build pipeline rows by detecting repeating phases.

    Identifies the task type marking the start of each pipeline layer (typically
    RMS_NORM). When that type re-appears with a new event_no, a new phase row begins.
    """
    if not compute_entries:
        return []

    # ---------- Identify the layer-boundary task type ----------
    layer_start_type = None
    if stage_seq:
        type_counts: Counter = Counter()
        for _, stasks in stage_seq:
            for t in stasks:
                tt = t["task_type"]
                if tt in (0, 200, 201, 202, 203):
                    continue
                type_counts[tt] += 1
        seen_once: set = set()
        for _, stasks in stage_seq:
            stage_types = set(
                t["task_type"]
                for t in stasks
                if t["task_type"] not in (0, 200, 201, 202, 203)
            )
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
        type_freq: Counter = Counter(e["type"] for e in compute_entries)
        for e in compute_entries:
            if "RMS_NORM" in e["type"] and type_freq[e["type"]] > 5:
                layer_start_type = e["type"]
                break

    if not layer_start_type:
        # No repeating pattern — fall back to temporal chunking
        n_phases = min(80, max(10, len(compute_entries) // 8))
        chunk = max(1, len(compute_entries) // n_phases)
        rows = []
        for i in range(0, len(compute_entries), chunk):
            phase_entries = compute_entries[i : i + chunk]
            bars = [_make_bar(e, global_start) for e in phase_entries]
            rows.append({"label": f"Phase {i // chunk}", "bars": [b.__dict__ for b in bars]})
        return rows

    # ---------- Split trace entries into phases ----------
    phases: list = []
    current_phase: list = []
    last_split_eno = -1

    for e in compute_entries:
        if (
            e["type"] == layer_start_type
            and e["event_no"] != last_split_eno
            and current_phase
        ):
            phases.append(current_phase)
            current_phase = []
            last_split_eno = e["event_no"]
        current_phase.append(e)
    if current_phase:
        phases.append(current_phase)

    rows = []
    for idx, phase_entries in enumerate(phases):
        types_here: Counter = Counter(e["type"] for e in phase_entries)
        types_str = " + ".join(_short(t) for t, _ in types_here.most_common(3))
        label = f"L{idx}: {types_str}"
        bars = [_make_bar(e, global_start) for e in phase_entries]
        rows.append({"label": label, "bars": [b.__dict__ for b in bars]})
    return rows
