"""View adapter framework for Mirage task-graph timeline visualization.

All timeline tabs are implemented as ViewAdapter subclasses.
Register them in display_task_graph_timeline.py's VIEW_REGISTRY.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from timeline.mode_detector import GraphMode


@dataclass
class BarData:
    """One horizontal bar in a timeline row."""

    start: float  # nanoseconds relative to global_start
    end: float  # nanoseconds relative to global_start
    name: str  # display / trace-key name
    blocks: int = 1  # number of GPU blocks that contributed
    avg_dur: float = 0.0  # average per-block duration (ns)
    color: str = "#888"
    trace_key: str = ""  # key into group_deps dict ('' == same as name)
    data_id: int = -1  # resident data node id (-1 == not a data bar)
    execution_kind: int = 0  # 0 = prelaunched/compat, 1 = streaming


@dataclass
class RowData:
    """One labeled row containing zero or more bars."""

    label: str
    bars: List[BarData] = field(default_factory=list)

    # ---- serialization helpers used by html_generator ----
    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "bars": [
                {
                    "start": b.start,
                    "end": b.end,
                    "name": b.name,
                    "blocks": b.blocks,
                    "avg_dur": b.avg_dur,
                    "color": b.color,
                    "trace_key": b.trace_key,
                    "data_id": b.data_id,
                    "execution_kind": b.execution_kind,
                }
                for b in self.bars
            ],
        }


@dataclass
class ViewContext:
    """All pre-computed data passed into every ViewAdapter."""

    # --- mode ---
    mode: GraphMode

    # --- raw inputs ---
    graph_data: dict
    slices: list
    track_names: dict

    # --- graph structure ---
    stage_seq: list
    dag_groups: list
    adjacency: dict
    reverse_adj: dict
    data_nodes: list
    data_adj: dict
    data_reverse_adj: dict

    # --- timing maps (output of mapper.py) ---
    group_deps: dict
    group_timing: dict
    data_timing: dict

    # --- global time bounds ---
    global_start: int
    total_dur: int

    # --- metrics ---
    sched_metrics: Optional[dict]
    data_sched_metrics: Optional[dict]


class ViewAdapter(ABC):
    """Abstract base for a single timeline tab.

    Subclasses implement is_applicable() and build_rows().
    Optionally override build_analysis_html() to emit an analysis section.
    """

    @property
    @abstractmethod
    def tab_id(self) -> str:
        """Short identifier used as HTML element id (e.g. 'pip', 'tt')."""
        ...

    @property
    @abstractmethod
    def tab_label(self) -> str:
        """Human-readable button label (e.g. 'Pipeline Phases')."""
        ...

    @abstractmethod
    def is_applicable(self, mode: GraphMode) -> bool:
        """Return True if this view should be shown for the given graph mode."""
        ...

    @abstractmethod
    def build_rows(self, ctx: ViewContext) -> List[RowData]:
        """Build and return the ordered list of timeline rows for this tab."""
        ...

    def build_analysis_html(self, ctx: ViewContext) -> str:
        """Return an optional HTML analysis section rendered below the timeline.

        Returns an empty string by default.
        """
        return ""
