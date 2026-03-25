#!/bin/bash
# Run the toy MoE demo in all 4 modes with profiling, then generate timelines.
#
# Usage:
#   cd /home/nikhilag/mirage
#   bash demo/toy_moe/run_all_modes.sh [output_base_dir]
#
# Requires: MIRAGE_HOME, PYTHONPATH set, mirage importable, perfetto installed.

set -euo pipefail

BASE_DIR="${1:-toy_moe_profiles}"
SCRIPT="demo/toy_moe/demo_toy_moe_hopper.py"
TIMELINE_SCRIPT="scripts/display_task_graph_timeline.py"

mkdir -p "$BASE_DIR"

run_mode() {
    local mode_id="$1"
    shift
    local mode_dir="$BASE_DIR/modes/$mode_id"
    mkdir -p "$mode_dir"

    echo "===== Running: $mode_id ====="
    env "$@" \
        python3 -u "$SCRIPT" \
            --profiling \
            --max-num-batched-tokens 4 \
            --max-num-batched-requests 2 \
            --max-seq-length 8 \
            --output-dir "$mode_dir" \
        2>&1 | tee "$mode_dir/run.log"
    echo ""
}

generate_timeline() {
    local mode_id="$1"
    local mode_dir="$BASE_DIR/modes/$mode_id"
    local tg="$mode_dir/task_graph_rank0.json"
    local trace="$mode_dir/logical_trace.perfetto-trace"
    local html="$BASE_DIR/toy_timeline_${mode_id}.html"

    if [[ ! -f "$tg" ]]; then
        echo "SKIP timeline for $mode_id: no task_graph_rank0.json"
        return
    fi
    if [[ ! -f "$trace" ]]; then
        echo "SKIP timeline for $mode_id: no perfetto trace"
        return
    fi

    echo "Generating timeline: $html"
    python3 "$TIMELINE_SCRIPT" "$tg" "$trace" -o "$html" 2>&1 || echo "WARN: timeline generation failed for $mode_id"
}

# Mode 1: Legacy event (baseline)
run_mode "legacy_event" \
    MIRAGE_TASK_GRAPH_MODE=legacy_event

# Mode 2: Resident hybrid prelaunch
run_mode "resident_hybrid_prelaunch" \
    MIRAGE_TASK_GRAPH_MODE=resident_data \
    MIRAGE_RESIDENT_EXECUTION_MODE=hybrid_prelaunch

# Mode 3: Streaming legacy base
run_mode "streaming_legacy_base" \
    MIRAGE_TASK_GRAPH_MODE=streaming_data

# Mode 4: Streaming hybrid base
run_mode "streaming_hybrid_base" \
    MIRAGE_TASK_GRAPH_MODE=streaming_data \
    MIRAGE_STREAMING_BASE_MODE=hybrid_prelaunch

echo ""
echo "===== All runs complete. Generating timelines... ====="
echo ""

for mode_id in legacy_event resident_hybrid_prelaunch streaming_legacy_base streaming_hybrid_base; do
    generate_timeline "$mode_id"
done

echo ""
echo "===== Done. Artifacts in $BASE_DIR/ ====="
ls -la "$BASE_DIR"/toy_timeline_*.html 2>/dev/null || echo "(no timelines generated)"
