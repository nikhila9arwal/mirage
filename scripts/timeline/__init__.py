"""Shared constants and helpers for Mirage timeline tooling."""

import re
from typing import Dict, Tuple

__all__ = [
    "_TRACE_NAME_MAP",
    "_SHORT_NAMES",
    "TASK_COLORS",
    "SCHEDULER_TYPES",
    "STRAGGLER_RATIO",
    "_color_for",
    "_short",
    "_parse_trace_name",
    "_stage_trace_name",
    "_data_trace_name",
]

# ---------------------------------------------------------------------------
# Task type name / colour mappings (mirrors runtime_header.h)
# ---------------------------------------------------------------------------
_TRACE_NAME_MAP: Dict[int, str] = {
    10: "TASK_BEGIN_TASK_GRAPH",
    101: "TASK_EMBEDDING",
    102: "TASK_RMS_NORM_LINEAR",
    103: "TASK_ATTENTION_1",
    104: "TASK_ATTENTION_2",
    105: "TASK_SILU_MUL_LINEAR",
    106: "TASK_ALLREDUCE",
    107: "TASK_REDUCE",
    108: "TASK_LINEAR_WITH_RESIDUAL",
    109: "TASK_ARGMAX",
    110: "TASK_ARGMAX_PARTIAL",
    111: "TASK_ARGMAX_REDUCE",
    112: "TASK_FIND_NGRAM_PARTIAL",
    113: "TASK_FIND_NGRAM_GLOBAL",
    114: "TASK_TARGET_VERIFY_GREEDY",
    115: "TASK_SINGLE_BATCH_EXTEND_ATTENTION",
    116: "TASK_PAGED_ATTENTION_1",
    117: "TASK_PAGED_ATTENTION_2",
    118: "TASK_SILU_MUL",
    119: "TASK_RMS_NORM",
    120: "TASK_LINEAR",
    121: "TASK_IDENTITY",
    151: "TASK_LINEAR_WITH_RESIDUAL_HOPPER",
    152: "TASK_LINEAR_HOPPER",
    153: "TASK_PAGED_ATTENTION_HOPPER",
    154: "TASK_RMS_NORM_HOPPER",
    155: "TASK_LINEAR_SWAPAB_HOPPER",
    156: "TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER",
    157: "TASK_LINEAR_CUTLASS_HOPPER",
    158: "TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER",
    159: "TASK_SILU_MUL_HOPPER",
    160: "TASK_EMBEDDING_HOPPER",
    161: "TASK_MOE_W13_LINEAR_SM90",
    162: "TASK_MOE_W2_LINEAR_SM90",
    163: "TASK_SPLITK_LINEAR_SWAPAB_HOPPER",
    164: "TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER",
    200: "TASK_SCHD_TASKS",
    201: "TASK_SCHD_EVENTS",
    202: "TASK_GET_EVENT",
    203: "TASK_GET_NEXT_TASK",
    251: "TASK_SPLITK_LINEAR_SM100",
    252: "TASK_LINEAR_WITH_RESIDUAL_SM100",
    253: "TASK_LINEAR_SM100",
    254: "TASK_MOE_W13_LINEAR_SM100",
    255: "TASK_MOE_W2_LINEAR_SM100",
    257: "TASK_ATTN_SM100",
    258: "TASK_ARGMAX_REDUCE_SM100",
    259: "TASK_ARGMAX_PARTIAL_SM100",
    260: "TASK_MOE_TOPK_SOFTMAX_SM100",
    261: "TASK_MOE_MUL_SUM_ADD_SM100",
    262: "TASK_TENSOR_INIT",
    263: "TASK_PAGED_ATTENTION_SPLIT_KV_SM100",
    264: "TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100",
    265: "TASK_SAMPLING_SM100",
    301: "TASK_NVSHMEM_ALLGATHER_STRIDED_PUT",
    302: "TASK_NVSHMEM_TILE_ALLREDUCE",
}

# Short display names
_SHORT_NAMES: Dict[str, str] = {
    "TASK_BEGIN_TASK_GRAPH": "BEGIN",
    "TASK_EMBEDDING": "EMBED",
    "TASK_RMS_NORM_HOPPER": "RMS_NORM",
    "TASK_LINEAR_SWAPAB_HOPPER": "LINEAR",
    "TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER": "LINEAR+RES",
    "TASK_PAGED_ATTENTION_HOPPER": "ATTN",
    "TASK_SILU_MUL": "SILU_MUL",
    "TASK_ARGMAX_PARTIAL_SM100": "ARGMAX_P",
    "TASK_ARGMAX_REDUCE": "ARGMAX_R",
    "TASK_ARGMAX_REDUCE_SM100": "ARGMAX_R",
    "TASK_SCHD_TASKS": "SCHD_T",
    "TASK_SCHD_EVENTS": "SCHD_E",
    "TASK_GET_EVENT": "GET_EV",
    "TASK_GET_NEXT_TASK": "GET_T",
}

# Colours per task-type category (Tableau 10-ish)
TASK_COLORS: Dict[str, str] = {
    "EMBEDDING":  "#4e79a7",
    "RMS_NORM":   "#f28e2b",
    "ATTENTION":  "#e15759",
    "PAGED_ATT":  "#e15759",
    "LINEAR":     "#59a14f",
    "SILU_MUL":   "#76b7b2",
    "ARGMAX":     "#edc948",
    "REDUCE":     "#b07aa1",
    "ALLREDUCE":  "#ff9da7",
    "MOE":        "#9c755f",
    "SAMPLING":   "#bab0ac",
    "BEGIN":      "#555555",
}

SCHEDULER_TYPES = {"TASK_SCHD_TASKS", "TASK_SCHD_EVENTS", "TASK_GET_EVENT", "TASK_GET_NEXT_TASK"}
STRAGGLER_RATIO = 1.3


def _color_for(name: str) -> str:
    upper = name.upper()
    for prefix, color in TASK_COLORS.items():
        if prefix in upper:
            return color
    return "#cccccc"


def _short(name: str) -> str:
    return _SHORT_NAMES.get(name, name.replace("TASK_", ""))


def _parse_trace_name(name: str) -> Tuple[str, int, int]:
    """Split a Perfetto slice name into (task_type_string, event_no, data_id).

    Examples:
      - 'TASK_RMS_NORM_HOPPER_42' -> ('TASK_RMS_NORM_HOPPER', 42, -1)
      - 'TASK_RMS_NORM_HOPPER_42_d1307' -> ('TASK_RMS_NORM_HOPPER', 42, 1307)

    The trailing integer is the DAG-stable profiler_group_id emitted by the
    persistent kernel. Resident traces may also include a ``data_id`` suffix.
    """
    m = re.match(r"^(.+)_(\d+)_d(\d+)$", name)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    m = re.match(r"^(.+)_(\d+)$", name)
    if m:
        return m.group(1), int(m.group(2)), -1
    return name, 0, -1


def _stage_trace_name(task_type: str, event_no: int) -> str:
    return f"{task_type}_{event_no}"


def _data_trace_name(task_type: str, event_no: int, data_id: int) -> str:
    return f"{task_type}_{event_no}_d{data_id}"
