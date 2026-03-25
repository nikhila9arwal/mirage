"""Mode detection for Mirage task-graph timelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet


class ExecutionMode(Enum):
    LEGACY_EVENT = "legacy_event"
    RESIDENT_DATA = "resident_data"
    STREAMING_DATA = "streaming_data"


@dataclass
class GraphMode:
    execution_mode: ExecutionMode
    schema_version: int
    num_prelaunched: int = 0        # resident tasks with execution_kind != 1
    num_streaming: int = 0          # resident tasks with execution_kind == 1
    streaming_task_types: FrozenSet[str] = field(default_factory=frozenset)
    has_data_dag: bool = False

    @property
    def is_streaming(self) -> bool:
        return self.execution_mode == ExecutionMode.STREAMING_DATA

    @property
    def is_resident(self) -> bool:
        return self.execution_mode in (
            ExecutionMode.RESIDENT_DATA, ExecutionMode.STREAMING_DATA
        )


def detect_mode(graph_data: dict) -> "GraphMode":
    """Infer the execution mode and graph properties from a parsed task graph.

    Rules:
    - schema_version < 2  → LEGACY_EVENT
    - schema_version >= 2, no execution_kind=1 tasks → RESIDENT_DATA
    - schema_version >= 2, any execution_kind=1 task  → STREAMING_DATA
    """
    schema_v = int(graph_data.get("schema_version", 1))
    resident_tasks = graph_data.get("resident_tasks") or []
    data_nodes = graph_data.get("all_data") or []

    if schema_v < 2 or not resident_tasks:
        return GraphMode(
            execution_mode=ExecutionMode.LEGACY_EVENT,
            schema_version=schema_v,
            has_data_dag=False,
        )

    num_streaming = sum(
        1 for t in resident_tasks if int(t.get("execution_kind", 0)) == 1
    )
    num_prelaunched = len(resident_tasks) - num_streaming

    streaming_task_types: FrozenSet[str] = frozenset()
    if num_streaming > 0:
        from timeline import _TRACE_NAME_MAP  # avoid circular at module level
        streaming_task_types = frozenset(
            _TRACE_NAME_MAP.get(int(t["task_type"]), str(t["task_type"]))
            for t in resident_tasks
            if int(t.get("execution_kind", 0)) == 1
        )

    exec_mode = (
        ExecutionMode.STREAMING_DATA if num_streaming > 0 else ExecutionMode.RESIDENT_DATA
    )

    return GraphMode(
        execution_mode=exec_mode,
        schema_version=schema_v,
        num_prelaunched=num_prelaunched,
        num_streaming=num_streaming,
        streaming_task_types=streaming_task_types,
        has_data_dag=bool(data_nodes),
    )
