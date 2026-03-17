# Data-Aware Resident Task Graph in Mirage MPK

## What this document is for

This note explains a major change to the Mirage Persistent Kernel (MPK) task-graph runtime.

It is written for a reader who may not know:

- how Mirage/MPK is structured,
- how the old task graph worked,
- why the old model created unnecessary barriers,
- what was changed in the compiler, runtime, and visualization tooling,
- what has and has not been validated yet.

The goal of the change is simple:

1. let independent pieces of work make progress without waiting for unrelated siblings, and
2. keep expensive task state, especially weight-related state, resident across many pieces of data instead of rebuilding a separate task descriptor for every token-sized slice of work.

In short, the old model was "one runtime task object per logical partition". The new model is "one resident task object per operator family, plus many data objects that flow through it".

## Current code layout

Current layout:

- legacy/event path remains in the original files restored to the committed
  baseline:
  - `include/mirage/persistent_kernel/runtime_header.h`
  - `include/mirage/persistent_kernel/persistent_kernel.cuh`
  - `include/mirage/persistent_kernel/tma.cuh`
  - `src/kernel/runtime.cc`
- resident/data path now lives in separate files:
  - `include/mirage/persistent_kernel/resident_runtime_header.h`
  - `include/mirage/persistent_kernel/resident_tma.cuh`
  - `include/mirage/persistent_kernel/resident_persistent_kernel.cuh`
  - `src/kernel/runtime_resident.cc`
- streaming path reuses the resident low-level machinery but has its own
  top-level compiler/runtime entrypoints:
  - `include/mirage/persistent_kernel/streaming_persistent_kernel.cuh`
  - `Graph::generate_streaming_task_graph(...)`

Selection is now above the runtime layer, not inside the old runtime:

- Python/Cython exposes both:
  - `generate_task_graph(...)` for the legacy event path
  - `generate_resident_task_graph(...)` for the resident/data path
  - `generate_streaming_task_graph(...)` for the streaming path
- `PersistentKernel` chooses between them using:
  - constructor arg
    `task_graph_mode="legacy_event" | "resident_data" | "streaming_data"`
  - or environment variable `MIRAGE_TASK_GRAPH_MODE`
  - default mode is `legacy_event`

Within `resident_data`, there are now two resident execution modes that share
the same schema-v2 public graph model:

- `scheduler_dispatch`
  - this is the original resident runtime behavior
  - it remains the default resident execution mode
  - enable explicitly with `MIRAGE_RESIDENT_EXECUTION_MODE=scheduler_dispatch`
- `hybrid_prelaunch`
  - this is the faster resident execution feature added later
  - enable with `MIRAGE_RESIDENT_EXECUTION_MODE=hybrid_prelaunch`

`streaming_data` is a separate compiler/runtime path. It does not use
the old resident scheduler-dispatch runtime for non-streaming work anymore.
Instead, it is now a mixed path with its own base-mode selection:

- default base mode: legacy MPK event semantics
  - non-streaming compatibility tasks behave like the old
    `legacy_event` runtime
  - streaming is layered on top only for whitelisted operators
- optional base mode: hybrid prelaunch
  - enable with `MIRAGE_STREAMING_BASE_MODE=hybrid_prelaunch`
  - for backward compatibility, `MIRAGE_RESIDENT_EXECUTION_MODE=hybrid_prelaunch`
    is also treated as the streaming hybrid-base request when
    `MIRAGE_TASK_GRAPH_MODE=streaming_data`

The streaming execution mode itself:

- keeps the resident/data graph abstraction,
- uses a different resident grouping strategy than `resident_data`,
- keeps non-streaming work on compatibility-task execution,
- uses resident ready queues only for streaming residents,
- currently restricts streaming grouping to a MoE whitelist
  (`TASK_MOE_W13_LINEAR_*`, `TASK_MOE_W2_LINEAR_*`).

So the selection hierarchy is now:

- `MIRAGE_TASK_GRAPH_MODE=legacy_event`
  - use the old event runtime
- `MIRAGE_TASK_GRAPH_MODE=resident_data`
  - use the resident/data compiler path
  - then choose the resident execution strategy with
    `MIRAGE_RESIDENT_EXECUTION_MODE`
- `MIRAGE_TASK_GRAPH_MODE=streaming_data`
  - use the streaming compiler/runtime path
  - default substrate is legacy MPK event behavior
  - optional secondary env var:
    `MIRAGE_STREAMING_BASE_MODE=hybrid_prelaunch`

Current status of `streaming_data` in this checkout:

- default `streaming_data` builds on top of legacy MPK event semantics for
  non-streaming work,
- `MIRAGE_STREAMING_BASE_MODE=hybrid_prelaunch` is the opt-in path that layers
  hybrid-prelaunch behavior underneath the same streaming residents,
- the streamable whitelist is still intentionally narrow:
  - `TASK_MOE_W13_LINEAR_*`
  - `TASK_MOE_W2_LINEAR_*`
- both streaming base modes now complete real single-GPU Qwen Hopper runs on
  the short `max_seq_length=40` validation shape,
- the temporary streaming-specific host/device debug scaffolding used during
  the shutdown investigation has been removed again; the runtime now only keeps
  the real functional fixes,
- the next work is larger-batch validation and broader correctness checking,
  not basic runtime bring-up.

## If you are taking over this work

This section is the shortest path to becoming productive without reading chat
history.

### What is true right now

- `legacy_event` is still the baseline MPK runtime.
- `resident_data` still has two execution modes:
  - `scheduler_dispatch`
  - `hybrid_prelaunch`
- `streaming_data` now defaults to legacy MPK behavior for non-streaming work.
- `streaming_data + MIRAGE_STREAMING_BASE_MODE=hybrid_prelaunch` is the opt-in
  path that combines hybrid-prelaunch behavior with streaming residents.
- The streaming whitelist is intentionally small right now:
  - `TASK_MOE_W13_LINEAR_*`
  - `TASK_MOE_W2_LINEAR_*`
- The default bring-up problem is solved. The remaining work is validation and
  scope expansion:
  - larger-batch runs
  - longer-output correctness checks
  - widening the streaming whitelist only after those pass

### Read these files in this order

If you need to understand the current implementation quickly, read these in
order:

1. this note
2. `src/kernel/runtime_resident.cc`
   - compiler pass and schema-v2 emission
   - search for:
     - `get_streaming_base_execution_mode`
     - `build_data_aware_task_graph`
     - `print_task_graph`
     - `generate_streaming_task_graph`
3. `include/mirage/persistent_kernel/resident_runtime_header.h`
   - runtime enums and descriptor/state layout
   - search for:
     - `ResidentExecutionMode`
     - `StreamingBaseExecutionMode`
     - `ResidentRuntimeConfig`
4. `include/mirage/persistent_kernel/resident_persistent_kernel.cuh`
   - actual runtime behavior
   - search for:
     - `streaming_uses_legacy_base`
     - `prelaunched_task_ready_nonblocking`
     - `trigger_data_event`
     - `release_completed_data_streaming`
     - `execute_worker_streaming`
     - `execute_scheduler_streaming`
5. `include/mirage/persistent_kernel/persistent_kernel.cuh`
   - legacy baseline semantics for comparison
6. if you are debugging tooling/trace behavior:
   - `include/mirage/persistent_kernel/profiler.h`
   - `python/mirage/mpk/profiler_persistent.py`
   - `scripts/display_task_graph_timeline.py`
   - `scripts/whatif_model.py`

### Mode matrix

Use this table as ground truth:

- `MIRAGE_TASK_GRAPH_MODE=legacy_event`
  - old MPK event runtime
- `MIRAGE_TASK_GRAPH_MODE=resident_data`
  - schema-v2 resident/data graph
  - choose one:
    - `MIRAGE_RESIDENT_EXECUTION_MODE=scheduler_dispatch`
    - `MIRAGE_RESIDENT_EXECUTION_MODE=hybrid_prelaunch`
- `MIRAGE_TASK_GRAPH_MODE=streaming_data`
  - schema-v2 streaming graph
  - default base mode is legacy/event semantics for non-streaming work
  - optional:
    - `MIRAGE_STREAMING_BASE_MODE=hybrid_prelaunch`

Important:

- do not assume `streaming_data` means “all work uses the resident/hybrid
  substrate”
- it now means “legacy MPK plus streaming on whitelisted ops” unless
  `MIRAGE_STREAMING_BASE_MODE=hybrid_prelaunch` is set

If you only care about the data-aware family, there are effectively four
runtime combinations today:

1. fine-grain data-aware
   - `MIRAGE_TASK_GRAPH_MODE=resident_data`
   - `MIRAGE_RESIDENT_EXECUTION_MODE=scheduler_dispatch`
2. prelaunched fine-grain data-aware
   - `MIRAGE_TASK_GRAPH_MODE=resident_data`
   - `MIRAGE_RESIDENT_EXECUTION_MODE=hybrid_prelaunch`
3. base streaming
   - `MIRAGE_TASK_GRAPH_MODE=streaming_data`
   - default base mode, which means legacy/event substrate for non-streaming
     work
4. prelaunched streaming
   - `MIRAGE_TASK_GRAPH_MODE=streaming_data`
   - `MIRAGE_STREAMING_BASE_MODE=hybrid_prelaunch`

So yes: if we ignore the old non-data-aware `legacy_event` path, the current
system is easiest to think about as those four data-aware modes.

### Rebuild checklist

Work inside `mirage.sif` from a Slurm allocation. The reconnect pattern is
documented later in this note.

Recommended rebuild order:

1. rebuild the runtime library

```bash
cd /home/nikhilag/mirage
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake -S . -B build
cmake --build build --target mirage_runtime -j4
```

2. then rebuild or relink the Python extension

Preferred path:

```bash
cd /home/nikhilag/mirage
export MIRAGE_HOME=/home/nikhilag/mirage
export PYTHONPATH=/home/nikhilag/mirage/python${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python3 setup.py build_ext --inplace --force
```

Fallback path if `setup.py build_ext --inplace --force` hangs in the Rust or
distutils steps on a specific node:

```bash
cd /home/nikhilag/mirage
EXT=$(python3 - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX"))
PY
)
Z3=$(python3 - <<'PY'
import os, z3
print(os.path.dirname(z3.__file__))
PY
)
g++ -shared -fPIC -std=c++17 -fopenmp -O2 -Wall \
  -DMIRAGE_BACKEND_USE_CUDA -DMIRAGE_FINGERPRINT_USE_CUDA \
  -I/usr/include/python3.10 \
  -Iinclude \
  -Ideps/json/include \
  -Ideps/cutlass/include \
  -Ideps/cutlass/tools/util/include \
  -Ibuild/abstract_subexpr/release \
  -Ibuild/formal_verifier/release \
  -I${Z3}/include \
  -I/usr/local/cuda/targets/x86_64-linux/include \
  python/mirage/_cython/core.cpp \
  -Lbuild \
  -L${Z3}/lib \
  -Lbuild/abstract_subexpr/release \
  -Lbuild/formal_verifier/release \
  -L/usr/local/cuda/lib64 \
  -L/usr/local/cuda/lib64/stubs \
  -lmirage_runtime -lcudadevrt -lcudart_static -lcudart -lcuda \
  -lz3 -lgomp -lrt -labstract_subexpr -lformal_verifier \
  -o python/mirage/core${EXT}.tmp \
  -Wl,-rpath,'$ORIGIN/lib' \
  -Wl,-rpath,'$ORIGIN/../../build/abstract_subexpr/release' \
  -Wl,-rpath,'$ORIGIN/../../build/formal_verifier/release'
mv python/mirage/core${EXT}.tmp python/mirage/core${EXT}
```

3. confirm the updated extension loads

```bash
cd /home/nikhilag/mirage
export PYTHONPATH=/home/nikhilag/mirage/python${PYTHONPATH:+:$PYTHONPATH}
python3 - <<'PY'
import mirage
print("mirage_import_ok")
PY
```

### Validation order

If you have a healthy node/container where `transformers` startup is responsive,
do validation in this order:

1. `legacy_event`
2. `resident_data + scheduler_dispatch`
3. `resident_data + hybrid_prelaunch`
4. `streaming_data`
5. `streaming_data + MIRAGE_STREAMING_BASE_MODE=hybrid_prelaunch`

For the live MoE path, use:

- `demo/qwen3/demo_30B_A3B_hopper.py`

Start with:

- `--max-num-batched-tokens 1`
- `--max-num-batched-requests 1`
- `--max-seq-length 40`

Then rerun with:

- `--max-num-batched-tokens 8`
- `--max-num-batched-requests 4`
- `--max-seq-length 40`

### Validation traps to avoid

- Do not treat small custom-graph `generate_*_task_graph()` probes as a clean
  signal for this work. They still hit a shared `register_mugraph(...)`
  segfault in both legacy and streaming generators.
- Do not treat profiling-mode latency as normal runtime latency. The profiled
  path is much slower and is only useful for timeline/overlap analysis.
- Do not assume a failed live run means the runtime is broken. On the current
  `node-gpu01` allocation, the Python model stack stalls before Mirage
  execution begins.

## Quick background: what Mirage MPK is

Mirage MPK is a compiler/runtime system for executing a model as a persistent GPU megakernel.

At a high level:

1. a Python script builds a high-level graph,
2. Mirage lowers that graph into kernel operators,
3. Mirage generates a task graph plus CUDA runtime code,
4. MPK launches a persistent kernel with worker blocks and scheduler blocks,
5. workers execute tasks while schedulers coordinate progress across the graph.

The main user-facing entry point is `PersistentKernel` in Python. A model author attaches tensors, creates operations such as RMSNorm, attention, linear, MoE, and so on, then calls `compile()` and runs the megakernel.

Internally, the important pieces for this change are:

- legacy path:
  - `src/kernel/runtime.cc`
  - `include/mirage/persistent_kernel/runtime_header.h`
  - `include/mirage/persistent_kernel/persistent_kernel.cuh`
  - `include/mirage/persistent_kernel/tma.cuh`
- resident/data path:
  - `src/kernel/runtime_resident.cc`
  - `include/mirage/persistent_kernel/resident_runtime_header.h`
  - `include/mirage/persistent_kernel/resident_persistent_kernel.cuh`
  - `include/mirage/persistent_kernel/resident_tma.cuh`
- mode-selection glue:
  - `include/mirage/kernel/graph.h`
  - `python/mirage/_cython/CCore.pxd`
  - `python/mirage/_cython/core.pyx`
  - `python/mirage/kernel.py`
  - `python/mirage/mpk/persistent_kernel.py`
- `scripts/display_task_graph.py`
  Static task-graph visualization.
- `scripts/display_task_graph_timeline.py`
  Perfetto timeline visualizer and dependency analyzer.
- `scripts/whatif_model.py`
  Analytical model used by the timeline/what-if tooling.

## How the old task graph worked

Before this change, the compiler generated a flat graph with three main arrays:

- `all_tasks`
- `all_events`
- `first_tasks`

### Old runtime objects

Each logical unit of work was represented as a full task descriptor.

That descriptor contained both:

- static information:
  - task type,
  - kernel variant,
  - profiler group,
  - event wiring,
- dynamic information:
  - input/output tensor pointers,
  - offsets and slices,
  - request/token/expert metadata,
  - TMA descriptors tied to those particular slices.

So if a single operator was partitioned into many token-sized or block-sized pieces, the runtime materialized many separate task descriptors.

### Old dependency model

The old execution model used events as the primary dependency mechanism.

Each task had:

- `dependent_event`: what must fire before the task is allowed to run,
- `trigger_event`: what the task contributes to when it completes.

Schedulers watched event counters. Once all producers of an event had triggered it, the consumers of that event were released.

### Why that created barriers

This event-based approach worked, but it grouped dependencies too coarsely.

If several sibling tasks all contributed to the same event, then every consumer waiting on that event had to wait for all of them, even if a specific consumer only needed data from one matching producer.

That effectively created "phase barriers" inside the compute path.

For a Mixture-of-Experts (MoE) graph, this is exactly the wrong behavior:

- token A may be ready for the next operator,
- token B may still be in flight,
- but if both feed the same event, token A cannot progress until token B also finishes.

The graph was topologically correct, but it prevented overlap that should have been legal.

### Why not just create more tasks?

One obvious fix would be to split tasks more finely so each token or slice had its own dependencies.

That solves the barrier problem, but it makes descriptor residency worse:

- more task objects,
- more repeated weight-related state,
- more repeated TMA descriptor construction,
- more pressure on scheduling queues,
- less benefit from keeping a task "loaded" while many similar data instances flow through it.

So the design requirement was:

- fine-grained dependencies,
- without turning every token into a completely separate resident compute task.

## New mental model: resident tasks plus data

The new runtime model has two levels.

### Resident task

A resident task is the static execution identity of an operator family.

It holds information that does not change across many instances of the same work:

- task type,
- kernel variant,
- number of inputs and outputs,
- maximum allowed parallelism for that resident task,
- total number of data items expected to pass through it,
- profiler group identity.

Think of a resident task as "the code and static execution personality of this stage".

One important nuance in the current codebase is that there are now two
different ways to choose that resident-task identity:

- `resident_data`
  - the original resident compiler path
  - groups fairly coarsely, originally around operator-family execution stages
- `streaming_data`
  - the newer streaming compiler path
  - uses a stricter resident grouping key so the resident unit is closer to the
    original MPK task partition that owns a fixed weight or expert shard
  - the first implementation is intentionally conservative and only applies the
    finer grouping to a whitelist of stream-friendly task types, primarily the
    MoE linears

### Data

A data object is one independent instance of work that flows through a resident task.

It holds the per-instance state that used to be embedded in the old full task descriptor:

- concrete input/output tensor pointers,
- concrete tensor slices and offsets,
- request/expert/KV metadata,
- per-slice TMA descriptors,
- predecessor count,
- event fields still needed by compatibility task implementations.

Think of a data object as "one token/block/expert slice moving through the stage".

### Why this split helps

The split gives two benefits at once:

1. dependency tracking becomes fine-grained because dependencies are attached to data objects instead of being collapsed into shared compute events,
2. residency improves because workers can keep a resident task active while draining many ready data instances from that same task.

## New dependency model

The compute path no longer uses shared event fan-in as the main dependency mechanism.

Instead, it uses explicit edges between data objects.

The new graph representation includes:

- `resident_tasks`
- `all_data`
- `data_edges`
- `first_data_ids`

### `resident_tasks`

One entry per operator family that should stay resident while multiple data instances pass through it.

### `all_data`

One entry per logical work instance from the original flat task graph.

In practice, each old compute task usually becomes:

- one `FullDataDesc` / `DataDesc`,
- associated with one `ResidentTaskDesc`.

### `data_edges`

Explicit producer-data to consumer-data edges.

This is the core change that removes unnecessary barriers.

Now a downstream data item depends only on the upstream data items that actually feed it.

### `first_data_ids`

The data items with zero predecessors. These are the ones activated when a new iteration begins.

## How the compiler now builds the graph

The new graph construction still starts from the old flat task/event graph, because that graph already contains the original partitioning decisions and tensor slices.

The compiler then builds a second, data-aware view.

This logic lives in `build_resident_task_graph(...)` in
`src/kernel/runtime_resident.cc`.

### Step 1: create resident tasks

For each non-input kernel operator:

- find the old tasks associated with that operator,
- create one `ResidentTaskDesc`,
- derive `max_parallelism` from the multiplicity of the original tasks,
- record `total_data_count`.

### Step 2: create data descriptors

For each old logical compute task:

- create a `FullDataDesc`,
- copy per-instance tensor pointers/slices,
- copy task metadata,
- copy old `trigger_event` and `dependent_event`,
- assign the data to its resident task.

### Step 3: refine old event dependencies into data edges

The old graph already tells us which producer stage feeds which consumer stage via events.

The new code refines that stage-level relationship into per-data edges using tensor overlap:

1. gather producer tasks for an event,
2. gather consumer tasks waiting on that event,
3. compare producer outputs against consumer inputs,
4. if tensor regions overlap, create a producer-data to consumer-data edge,
5. if no overlap is found, fall back to dense producer-to-consumer edges so correctness is preserved.

This overlap logic uses byte-interval reasoning over tensor base pointers, dimensions, strides, and data types.

The result is:

- exact predecessor counts when tensor overlap can identify them,
- safe fallback behavior when overlap is ambiguous.

## JSON schema changes

The emitted task-graph JSON now has `schema_version: 2`.

New top-level fields:

- `resident_tasks`
- `all_data`
- `data_edges`
- `first_data_ids`
- `control_events`

Compatibility fields are still emitted:

- `all_tasks`
- `all_events`
- `first_tasks`

This compatibility view exists so older tooling can still open the graph while the ecosystem transitions.

Important detail:

- the resident runtime treats the resident/data DAG as the canonical dependency
  model,
- but the current validated implementation still uses compatibility `all_tasks`
  as the concrete worker execution granule.

## Runtime type changes

The core resident-runtime descriptor changes are in
`include/mirage/persistent_kernel/resident_runtime_header.h`.

The original `include/mirage/persistent_kernel/runtime_header.h` remains the
legacy/event path header.

### New identifiers

- `ResidentTaskId`
- `DataId`

### New descriptor types

- `ResidentTaskDesc`
- `FullDataDesc`
- `DataDesc`
- `DataEdgeDesc`
- `ResolvedTaskDesc`

### Why `ResolvedTaskDesc` exists

Most task implementations in Mirage already expect a `TaskDesc`-style interface:

- `task_desc->input_ptrs[...]`
- `task_desc->output_ptrs[...]`
- `task_desc->task_metadata...`
- for some tasks, `task_desc->trigger_event`

Rewriting every task implementation would have been much more invasive.

Instead, the new generated `_execute_task(...)` accepts:

- `ResidentTaskDesc const *resident_task_desc`
- `DataDesc const *data_desc`

Then it constructs a `ResolvedTaskDesc`, which is layout-compatible with `TaskDesc`, and passes a `TaskDesc`-shaped view to the already-generated task bodies.

This preserves most of the existing task implementation code while still separating static and dynamic state.

### Why `trigger_event` and `dependent_event` were kept in `DataDesc`

Even though compute scheduling no longer depends on events, some generated task variants still read these fields directly, especially multi-GPU/NVSHMEM-related task bodies.

To keep those kernels source-compatible, the data descriptor still stores:

- `trigger_event`
- `dependent_event`

That is a compatibility bridge, not a return to event-based compute scheduling.

## RuntimeConfig changes

The resident path uses `ResidentRuntimeConfig`, defined in
`include/mirage/persistent_kernel/resident_runtime_header.h`.

`ResidentRuntimeConfig` extends the legacy `RuntimeConfig` with
resident-task/data scheduling state rather than replacing the original runtime
config from scratch.

New important fields include:

- `resident_execution_mode`
- `begin_event_index`
- `end_event_index`
- `num_resident_tasks`
- `num_data`
- `num_data_edges`
- `num_first_data_ids`
- `num_control_tasks`
- `num_terminal_data`
- `per_completion_queue_len`
- `resident_tasks`
- `all_data`
- `data_edge_offsets`
- `data_edge_targets`
- `data_initial_predecessor_counts`
- `data_pending_predecessor_counts`
- `task_to_data_id`
- `data_to_task_id`
- `worker_owner_scheduler`
- `completion_queue_last_ready_data_id`
- `completion_queue_next_free_data_id`
- `completion_queues`
- `completed_terminal_data_count`
- `completed_data_this_iteration_count`
- `first_data_ids`
- `current_iteration`

The inherited legacy arrays `all_tasks`, `all_events`, and `first_tasks` are
still present and are still used by the validated worker fast path.

One subtle but important design choice is that the legacy `all_tasks` array is still the actual execution unit on the validated fast path.

That means:

- resident/data objects are the canonical dependency model,
- legacy tasks are the concrete worker-dispatch model.

This is the key architectural compromise that brought the implementation back closer to Mirage's original fast path.

Because both resident execution modes now coexist in one resident runtime,
`ResidentRuntimeConfig` intentionally contains enough state for both:

- the older scheduler-dispatch resident path,
- the newer hybrid-prelaunch resident path.

The slower scheduler-dispatch-only fields are still present so the old resident
path can be preserved for direct comparison from the same checkout.

## Scheduler and worker behavior after the change

The most important behavioral change is in
`include/mirage/persistent_kernel/resident_persistent_kernel.cuh`.

### Old behavior

- schedulers primarily coordinated event counters,
- compute tasks were prelaunched and then blocked on shared events,
- event fan-in determined when an entire consumer stage could proceed.

### New behavior

Control events still exist, but only for iteration-level control:

- begin graph,
- end graph,
- termination.

There are now two resident execution strategies on top of the same schema-v2
graph.

### Resident mode: `scheduler_dispatch`

This is the original resident runtime behavior, preserved as the default
resident execution mode.

It works like this:

1. scheduler 0 handles `END_OF_TASK_GRAPH`, prepares the next batch, resets
   predecessor counts, and enqueues `TASK_BEGIN_TASK_GRAPH`,
2. `TASK_BEGIN_TASK_GRAPH` eventually causes the first ready `data_id`s from
   `first_data_ids` to be launched,
3. workers execute compatibility `TaskDesc`s from `all_tasks`,
4. after a data task finishes, the worker publishes the completed `data_id`
   into a scheduler-owned completion queue,
5. the scheduler drains that completion queue, walks outgoing `data_edges`,
   decrements successor predecessor counts, and maps newly ready `data_id`s
   back to compatibility task ids through `data_to_task_id`,
6. terminal data contributes to end-of-graph completion from the scheduler
   side.

This mode exists so the older resident runtime can still be run without
restoring an old tree.

### Resident mode: `hybrid_prelaunch`

This is the faster resident execution feature added on top of the old resident
path.

It works like this:

1. scheduler 0 handles `END_OF_TASK_GRAPH`, prepares the next batch, resets
   per-data predecessor counts, and enqueues the `BEGIN_TASK_GRAPH` control
   task for the next iteration,
2. when `BEGIN_TASK_GRAPH` fires, the legacy compatibility task range is
   prelaunched to workers exactly like the old event runtime,
3. workers batch-load `TaskDesc` objects from `all_tasks` into shared memory,
   again matching Mirage's original execution style,
4. if a task is a control or compatibility-only task with no data mapping, it
   follows the old event-based behavior,
5. if a task has a `data_id` mapping and its dependency is local/non-NVSHMEM,
   the worker ignores `dependent_event` for readiness and waits on
   `data_pending_predecessor_counts[data_id] == 0`,
6. if a task has a `data_id` mapping but its dependency is remote/NVSHMEM, the
   worker keeps the old event wait path,
7. after executing a local data task, the worker CTA itself walks outgoing
   `data_edges` and decrements successor predecessor counts,
8. terminal data items contribute to end-of-graph completion directly from the
   worker path,
9. schedulers are now only on the control path: iteration transitions,
   control-event handling, and remote/NVSHMEM event launch behavior.

### Important consequence

If data item A is ready and data item B is not, A can move into the downstream resident task immediately.

There is no need to wait for every sibling that used to share the same event.

That is the main source of newly exposed overlap.

The important implementation point is that this overlap is exposed by the
resident/data DAG, while execution still uses Mirage's original batched
`TaskDesc` worker path.

In other words, the current architecture is:

- resident/data DAG for correctness and fine-grained readiness,
- legacy batched task execution as the common execution granule,
- `scheduler_dispatch` as the preserved older resident runtime,
- `hybrid_prelaunch` as the faster resident execution feature.

That high-level design makes sense for Mirage because Mirage was already optimized around batched worker-side execution of prebuilt task descriptors.

## TMA changes

Hopper/Blackwell tasks often need TMA descriptors that depend on the exact tensor slice being accessed.

That means TMA state belongs with the data object, not with the resident task.

To support this, `include/mirage/persistent_kernel/resident_tma.cuh` includes:

- `create_tma_desc_by_data(ResidentTaskDesc const&, FullDataDesc&)`

This helper:

1. constructs a temporary `FullTaskDesc`,
2. copies the resident task's static information,
3. copies the data object's slice-specific tensors and metadata,
4. calls the existing `create_tma_desc_by_task(...)`,
5. copies the resulting TMA pointers back into the data descriptor.

This avoided rewriting all existing TMA creation logic.

## Generated CUDA code changes

The resident CUDA code generator in `src/kernel/runtime_resident.cc` was
updated in several ways.

### New JSON loader

The generated `construct_task_graph(...)` function now loads:

- legacy `all_tasks`,
- control events,
- resident tasks,
- full data descriptors,
- data edges,
- first data ids,
- `task_to_data_id`.

It also creates per-data TMA descriptors where needed.

### New `_init_persistent_kernel(...)` interface

The generated runtime initialization function now takes vectors for:

- `ResidentTaskDesc`
- `FullDataDesc`
- `DataEdgeDesc`
- `DataId` for `first_data_ids`
- `DataId` for `task_to_data_id`

### New `_execute_task(...)` signatures

The generated code now emits two device-side dispatch entry points.

The primary one is the old fast-path shape:

```cpp
void _execute_task(TaskDesc const* task_desc,
                   ResidentRuntimeConfig const &runtime_config)
```

This is what the worker loop uses after batching `TaskDesc`s out of `all_tasks`.

The compatibility wrapper for resident/data resolution is:

```cpp
void _execute_task(ResidentTaskDesc const* resident_task_desc,
                   DataDesc const* data_desc,
                   ResidentRuntimeConfig const &runtime_config)
```

Inside, it resolves those two descriptors into a `TaskDesc`-compatible view and then forwards to the primary `TaskDesc`-based dispatcher.

This matters because it means the refactor no longer forces the steady-state worker path to rebuild resolved descriptors for every single task fetch.

## Tooling changes

The graph/runtime change is only useful if the tooling can show it.

### Static graph visualization

`scripts/display_task_graph.py` now understands schema v2.

For a v2 graph it renders:

- resident tasks as separate nodes,
- data objects as separate nodes,
- explicit `data_edges`,
- control entry/exit points.

This makes it visually obvious that the graph is now "resident stage + many flowing data instances" instead of a flat event-bound task list.

### Profiler and trace export

The resident/data visualization only works if the runtime emits enough
information to distinguish one data item from another in the trace.

The low-level profiler used by MPK lives in:

- `include/mirage/persistent_kernel/profiler.h`
- `python/mirage/mpk/profiler_persistent.py`

At the device level, the profiler writes a stream of 64-bit entries:

- one header entry with `(num_blocks, num_groups)`,
- then one entry per emitted event.

The event tag is still packed into 32 bits:

- bits `[31:19]`: event number
  - in practice, this is the DAG-stable `profiler_group_id` for profiled graph
    tasks,
- bits `[18:11]`: block/group id,
- bits `[10:2]`: task type / event id,
- bits `[1:0]`: event kind.

The event kinds are now:

- `begin`
- `end`
- `instant`
- `metadata`

The important new piece is `metadata`.

For resident compute tasks:

1. the worker emits a normal `begin`,
2. if the task is associated with a real `data_id`, it immediately emits a
   `metadata` event carrying that `data_id`,
3. the worker later emits the normal `end`.

The exporter in `python/mirage/mpk/profiler_persistent.py` reconstructs slices
from those entries:

- legacy or control-style slices still become `TASK_TYPE_<group_id>`,
- resident data-aware slices become `TASK_TYPE_<group_id>_d<data_id>`.

That naming convention is what makes the rest of the tooling possible.

One subtle but important detail is that resident traces can still contain
nested/internal profiler slices from generated task bodies that are not actual
graph nodes. The timeline script now filters those out before DAG mapping so
the reported stage/data metrics are built only from graph-backed tasks.

### Timeline visualization

`scripts/display_task_graph_timeline.py` now understands v2 graphs.

For schema v2 it now builds two views of the same execution:

- a stage-level DAG over `resident_tasks`,
- a data-level DAG over `all_data` and `data_edges`.

The stage-level path is still useful because it keeps the old high-level view:

- one row per resident task / operator stage,
- stage start = earliest observed block start,
- stage end = latest observed block end,
- stage critical path,
- stage-level queue wait.

The data-level path is the new important one.

The script now:

- parses trace names with optional `data_id` suffixes,
- maps `TASK_TYPE_<group_id>_d<data_id>` back to the exact `DataDesc`,
- computes data-ready time from predecessor data completion,
- computes a true data-level critical path,
- computes per-data queue wait,
- computes per-stage-edge overlap ratios,
- renders a `Data Overlap` tab with one bar per data item.

The most important metric it now exposes is the overlap ratio for a resident
stage edge:

- numerator: downstream data items that started before the upstream resident
  stage fully ended,
- denominator: total downstream data items on that edge.

That is the direct signal for whether the resident/data DAG is actually
unlocking overlap.

This is important because the main visible result of the new system is overlap:

- one data instance can enter a downstream resident task before the last sibling data instance has finished upstream.

### What-if model

`scripts/whatif_model.py` was updated so fan-in can come from the resident/data DAG rather than only from legacy event `num_triggers`.

Without that change, the analytical model would keep interpreting the new graph through the lens of the old barrier-heavy event graph.

## Files changed and why

### Legacy runtime files

- `include/mirage/persistent_kernel/runtime_header.h`
- `include/mirage/persistent_kernel/persistent_kernel.cuh`
- `include/mirage/persistent_kernel/tma.cuh`
- `src/kernel/runtime.cc`

These remain the old event-based MPK runtime. The point of the split was to
keep this path available for direct comparison and to avoid mixing the legacy
and resident implementations in the same low-level files.

### `include/mirage/persistent_kernel/resident_runtime_header.h`

This is the resident/data type-system change. It defines:

- resident task/data IDs,
- `ResidentTaskDesc`,
- `FullDataDesc`,
- `DataDesc`,
- `DataEdgeDesc`,
- `ResolvedTaskDesc`,
- `ResidentRuntimeConfig`.

It also keeps `ResolvedTaskDesc` layout-compatible with `TaskDesc`, which is
what lets the resident runtime reuse the existing generated task bodies.

### `include/mirage/persistent_kernel/resident_persistent_kernel.cuh`

This is the resident/data runtime implementation. It contains:

- resident runtime initialization and teardown,
- both resident execution modes,
  - scheduler-owned completion-queue dispatch,
  - hybrid prelaunch with worker-side local dependency release,
- worker-side batched execution over compatibility `TaskDesc`s,
- `task_to_data_id` mapping logic,
- `data_to_task_id` mapping logic,
- end-of-graph detection using terminal data completion,
- kernel launch selection between the two resident execution strategies.

It also now contains the resident-side profiler emission changes:

- data-aware profiler-group selection,
- metadata emission for `data_id`,
- the profiled serving-path completion fix described later in this note.

### `include/mirage/persistent_kernel/profiler.h`

This is the low-level profiler wire-format change.

It now defines the new `metadata` event kind used to carry `data_id` through
the existing profiler buffer without inventing a separate side channel.

### `python/mirage/mpk/profiler_persistent.py`

This is the Perfetto export path for MPK traces.

It now:

- reconstructs slices from begin/end pairs,
- attaches `data_id` from resident metadata events when present,
- emits trace names in the `TASK_TYPE_<group_id>_d<data_id>` format.

### Python/demo profiler buffer allocation

- `python/mirage/mpk/mpk.py`
- `demo/qwen3/demo.py`
- `demo/qwen3/demo_hopper.py`
- `demo/qwen3/demo_30B_A3B.py`
- `demo/qwen3/demo_30B_A3B_hopper.py`
- `demo/qwen3/demo_sampling.py`
- `demo/qwen3/demo_debug.py`
- `demo/qwen3/demo_mpk_wrapper.py`
- `demo/qwen3/demo_chat.py`
- `demo/llama3/demo.py`

These files were updated to allocate a much larger profiler buffer.

That was necessary because the data-aware resident trace emits more profiler
events than the old stage-only trace:

- start,
- metadata carrying `data_id`,
- end.

### `include/mirage/persistent_kernel/resident_tma.cuh`

This adds the resident-path TMA bridge:

- host helpers that build per-data TMA descriptors from resident static state
  plus data-local tensor slices.

This is what allows TMA state to follow the data object without rewriting all
legacy TMA creation logic.

### `src/kernel/runtime_resident.cc`

This is the resident/data compiler and codegen path. It adds:

- resident/data graph construction,
- tensor-overlap-based producer/consumer refinement,
- classification of local compute events versus remote/NVSHMEM events,
- graph validation for predecessor counts / zero-indegree / cycle detection,
- schema-v2 JSON emission and loading,
- compatibility `task_to_data_id` mappings,
- resident-path CUDA generation and wrappers.

### Mode-selection glue

- `include/mirage/kernel/graph.h`
- `python/mirage/_cython/CCore.pxd`
- `python/mirage/_cython/core.pyx`
- `python/mirage/kernel.py`
- `python/mirage/mpk/persistent_kernel.py`

These files expose both generation entry points and let Python select
`legacy_event` or `resident_data` without needing separate checkouts.

### `scripts/display_task_graph.py`

Changed to render schema v2 graphs with explicit resident-task and data nodes.

### `scripts/display_task_graph_timeline.py`

Changed to:

- parse schema v2,
- build stage structure from resident tasks,
- build dependency DAG from `data_edges`,
- map Perfetto trace groups against resident-task profiler groups,
- map data-aware trace groups against exact `data_id`s,
- compute data-level overlap / queue-wait / critical-path metrics,
- filter nested non-graph slices before DAG mapping,
- render the new `Data Overlap` timeline tab.

### `scripts/whatif_model.py`

Changed so analytical fan-in and dependency scoring work with the resident/data
DAG instead of only with legacy event fan-in, while still collapsing trace
names back to stage-level keys when the trace includes `_d<data_id>` suffixes.

## Validation that was performed

The following checks were run during development.

### Python/tooling validation

- `python -m py_compile scripts/display_task_graph.py scripts/display_task_graph_timeline.py scripts/whatif_model.py`

This passed.

### Graph validation

`validate_resident_task_graph(...)` was added on the generator side in
`src/kernel/runtime_resident.cc`.

It throws if the resident/data graph contains:

- an out-of-range edge,
- a self-cycle,
- inconsistent predecessor counts,
- a mismatch between `first_data_ids` and the true zero-indegree data set,
- any cycle that prevents a full topological traversal.

This validator was exercised on real generated MoE graphs during debugging.

### Synthetic schema-v2 validation

Small synthetic in-memory v2 graphs were also used to verify that:

- `build_stage_sequence(...)` works for resident/data graphs,
- `build_dependency_dag(...)` builds the correct resident-task DAG,
- `map_trace_to_graph(...)` maps trace groups onto the resident-task graph,
- `display_task_graph.py` can render a schema-v2 graph to a `.dot` file.

These checks passed.

### Build validation inside `mirage.sif`

The validated environment is the Singularity container:

```bash
singularity exec --nv --fakeroot \
  --bind /home/nikhilag:/home/nikhilag \
  --home /home/nikhilag \
  --writable-tmpfs ~/mirage.sif \
  /bin/sh -lc 'cd /home/nikhilag/mirage && python3 setup.py build_ext --inplace --force'
```

Inside that container:

- the main CMake build completed successfully,
- the Python extension was loadable from inside the container,
- on some nodes, `python3 setup.py build_ext --inplace --force` completed
  normally,
- on `node-gpu01`, that command later hung in Rust/distutils steps, so the
  reliable fallback was:
  - rebuild `mirage_runtime` with CMake first,
  - then relink `python/mirage/core.cpython-310-x86_64-linux-gnu.so` manually
    against the updated static library.

Earlier `z3++.h` failures came from a host-configured build cache, not from this resident/data work. They are not the current blocker.

### Slurm allocation and reconnect workflow

The validation work was done from an existing Slurm GPU allocation and then executed inside `mirage.sif`.

Important operational detail:

- do not assume you can `ssh` directly into the GPU node from this workflow,
- the reliable pattern is to reconnect through the existing allocation with `srun --overlap`.

To look for existing allocations for your user:

```bash
squeue -u $USER -o "%.18i %.9P %.32j %.8T %.10M %.6D %N"
```

Useful follow-up command once you have a job id:

```bash
scontrol show job <jobid>
```

To open an interactive shell inside the existing allocation and inside the Mirage container:

```bash
srun --overlap --jobid <jobid> -w <node> \
  singularity exec --nv --fakeroot \
  --bind /home/nikhilag:/home/nikhilag \
  --home /home/nikhilag \
  --writable-tmpfs /home/nikhilag/mirage.sif \
  /bin/bash
```

To run a one-shot command inside that same environment:

```bash
srun --overlap --jobid <jobid> -w <node> \
  singularity exec --nv --fakeroot \
  --bind /home/nikhilag:/home/nikhilag \
  --home /home/nikhilag \
  --writable-tmpfs /home/nikhilag/mirage.sif \
  /bin/sh -lc 'cd /home/nikhilag/mirage && <command>'
```

This is the environment that was used for the live validation recorded below.

### Live single-GPU Hopper validation

The latest resident-mode layering was revalidated on:

- `demo/qwen3/demo_30B_A3B_hopper.py`
- single GPU
- `--max-num-batched-tokens 1`
- `--max-num-batched-requests 1`
- `--max-seq-length 40`

The important latest artifacts are:

- legacy path:
  - `validation_resident_feature_modes/legacy_single/task_graph_rank0.json`
  - `validation_resident_feature_modes/legacy_single/test_rank0.cu`
  - `validation_resident_feature_modes/legacy_single/run.log`
- resident default (`scheduler_dispatch`):
  - `validation_resident_feature_modes/resident_scheduler_dispatch_single/task_graph_rank0.json`
  - `validation_resident_feature_modes/resident_scheduler_dispatch_single/test_rank0.cu`
  - `validation_resident_feature_modes/resident_scheduler_dispatch_single/run.log`
- resident feature (`hybrid_prelaunch`):
  - `validation_resident_feature_modes/resident_hybrid_prelaunch_single_v4/task_graph_rank0.json`
  - `validation_resident_feature_modes/resident_hybrid_prelaunch_single_v4/test_rank0.cu`
  - `validation_resident_feature_modes/resident_hybrid_prelaunch_single_v4/run.log`

The current layered setup behaved as intended:

- legacy mode still emits the old event runtime
- resident default emits schema v2 and keeps the old slower resident runtime
- resident hybrid emits schema v2 and uses the faster hybrid-prelaunch runtime

The two resident graphs on this shape still emitted the same public resident
graph structure:

- `533` `resident_tasks`
- `21,331` `all_data`
- `128,641` `data_edges`
- `1` `first_data_ids`

Measured latencies on the same validation shape:

- legacy/event path: `5.225 ms/token`
- resident/data `scheduler_dispatch`: `82.036 ms/token`
- resident/data `hybrid_prelaunch`: `5.878 ms/token`

So after adding resident execution modes on top of the old resident runtime:

- the old resident behavior is preserved as the default resident execution mode,
- the faster hybrid-prelaunch resident path is still available in the same
  checkout,
- legacy, resident-default, and resident-hybrid all completed the live Hopper
  run end-to-end from the same tree.

One practical consequence is compile time:

- the resident generated CUDA is now larger because both resident execution
  modes are compiled into the same resident runtime file,
- the live `nvcc` step for resident mode is noticeably slower than it was when
  only one resident runtime existed in the file.

Important caveat:

- these runs validated that all three live paths compile, launch, and return,
- they validated that the visible decoded one-token output matched across
  legacy, resident-default, and resident-hybrid on the tested prompt,
- they did not validate multi-token semantic correctness exhaustively, and they
  did not validate multi-GPU/NVSHMEM model outputs end-to-end.

### Profiled larger-batch resident validation

After the data-aware profiler and timeline changes were added, the resident
hybrid-prelaunch path was revalidated on the earlier larger-batch shape:

- mode:
  - `MIRAGE_TASK_GRAPH_MODE=resident_data`
  - `MIRAGE_RESIDENT_EXECUTION_MODE=hybrid_prelaunch`
- shape:
  - `--max-num-batched-tokens 8`
  - `--max-num-batched-requests 4`
  - `--max-seq-length 40`
- artifacts:
  - `validation_visualization/resident_hybrid_multi_v4/task_graph_rank0.json`
  - `validation_visualization/resident_hybrid_multi_v4/resident_hybrid_multi_v4.perfetto-trace`
  - `validation_visualization/resident_hybrid_multi_v4/resident_hybrid_multi_v4_timeline.html`
  - `validation_visualization/resident_hybrid_multi_v4/resident_hybrid_multi_v4_whatif.html`
  - `validation_visualization/resident_hybrid_multi_v4/run.log`

What this validation proved:

- the profiled resident run returned the expected visible decoded output again,
- `generate length` returned to `1`,
- the data-aware trace mapped cleanly to both the stage DAG and the data DAG,
- the updated timeline script generated HTML successfully,
- the updated what-if script still generated HTML successfully.

Observed profiled latency on that run:

- `1651.482 ms/token`

This number is much slower than the non-profiled resident run and should not be
used as a performance comparison. It mostly reflects heavy profiling overhead
plus the larger trace size.

### Profiled-path bug that was found and fixed

While validating the new profiler/display path, the resident profiled run
initially produced obviously wrong output:

- truncated decoded text,
- `generate length -30`,
- a last token of `0` instead of the expected generated token.

The root cause was not in the exporter or the HTML tooling.

The root cause was a hardcoded profiling-only serving shortcut in both:

- `include/mirage/persistent_kernel/persistent_kernel.cuh`
- `include/mirage/persistent_kernel/resident_persistent_kernel.cuh`

Inside `prepare_next_batch(...)`, the code had:

- `#ifdef MPK_ENABLE_PROFILING`
- `if (true)`

in the request-completion path.

That forced every request to be treated as complete immediately after the first
batch-finalization step, which is why the prompt was truncated and the computed
generated length became negative.

That profiling-only shortcut has now been removed from both the legacy and
resident serving loops.

There was also a second profiling issue:

- the old profiler buffer allocation (`3000 * 128`) was too small once every
  resident compute task emitted an extra metadata event for `data_id`.

The buffer allocations were increased to `20000 * 128` (or the corresponding
scaled value in the Llama demo) to avoid profiler-buffer exhaustion.

### Compiler warning notes from the live run

The current live runs still emit `ptxas` warnings such as:

- `C7520` about `wgmma.mma_async` serialization in `worker_kernel`
- function-scope shared-memory dynamic-initialization warnings

These warnings also appear in the clean baseline run, so they are not sufficient to explain the remaining performance gap by themselves.

### Operational note

The validation runs above were taken on an idle GPU from the active Slurm job
used during this rewrite.

## Known limitations and current status

This section is the handoff state as of March 16, 2026.

### 1. All three execution paths now work end-to-end from one checkout

The current implementation:

- builds,
- preserves the legacy event runtime in the original files,
- keeps the resident/data runtime in separate files,
- preserves the original resident scheduler-dispatch path as the default
  resident execution mode,
- adds the faster hybrid-prelaunch resident path as an opt-in feature mode,
- generates both legacy and schema-v2 resident graphs,
- compiles both generated megakernels,
- runs all three live Hopper single-GPU modes to completion from the same
  checkout.

### 2. Resident execution modes are now directly comparable

Current live numbers on the same validation shape:

- legacy/event path: `5.225 ms/token`
- resident/data `scheduler_dispatch`: `82.036 ms/token`
- resident/data `hybrid_prelaunch`: `5.878 ms/token`

This is the key reason for keeping both resident execution modes:

- `scheduler_dispatch` preserves the old resident runtime for direct A/B
  comparisons,
- `hybrid_prelaunch` preserves the faster resident design that removes the
  scheduler-owned completion queue from the compute hot path.

### 3. Model-output correctness has not been validated yet

The live Hopper runs prove more than they did earlier, but correctness is still
not fully closed out.

What has been checked:

- the visible decoded one-token output matched across legacy,
  resident-default, and resident-hybrid on the validated single-GPU prompt,
- after the profiled-path fix above, the profiled resident larger-batch run
  also returned the expected visible decoded output shape again,
- the earlier resident-only milestone had deeper checks against legacy/Hugging
  Face, but those checks were not rerun after layering both resident execution
  modes into one resident runtime file.

What still needs to be checked:

- deeper token-by-token comparison against Hugging Face for longer generations,
- correctness on larger batches beyond the one-token visible output case,
- correctness on multi-GPU/NVSHMEM paths.

### 4. Multi-GPU/NVSHMEM behavior has not been revalidated end-to-end

Compatibility event fields are still preserved in `DataDesc`, and the resident
runtime now forces the single-kernel path when multiple GPUs are visible so it
does not rely on the known-bad split worker/scheduler launch for NVSHMEM.

However, full cross-GPU validation has not been redone yet in this environment.
The current Slurm allocation used for validation exposed only one GPU to the
container, so there was no way to execute a real multi-GPU model run here.

### 5. Compatibility views are still intentionally present

`all_tasks` and `all_events` are still emitted and are still used by the fast worker path.

This is intentional.

The current mental model should be:

- resident/data graph is the canonical dependency graph,
- compatibility tasks are the canonical execution granule.

### 6. Recommended next steps for the next person

If someone picks this up from here, the highest-value next steps are:

1. run a real multi-GPU/NVSHMEM validation once an allocation exposes more than
   one GPU and a usable sharded model path is available.
2. validate longer-generation token correctness against Hugging Face, not just
   the current one-token visible-output check.
3. rerun end-to-end validation on at least one non-MoE graph so the hybrid
   prelaunch runtime is not only validated on this MoE workload.
4. if the profiler overhead becomes a problem, reduce trace volume or make the
   profiler buffer size configurable instead of hardcoding the larger buffers
   in the demos.

## Running log

This section is meant to be a short rolling handoff, not a full changelog.

### 2026-03-16: streaming default substrate corrected

What changed:

- `streaming_data` no longer treats non-streaming work as resident-wide
  finer-grained prelaunch by default.
- The default streaming substrate is now legacy MPK event behavior.
- `hybrid_prelaunch + streaming` is the explicit opt-in path that combines
  local predecessor-count gating with streaming residents.
- The runtime now carries an explicit streaming base mode in
  `ResidentRuntimeConfig`.
- The generated streaming CUDA now bakes both:
  - `MIRAGE_RESIDENT_EXECUTION_MODE_VALUE=RESIDENT_EXECUTION_STREAMING`
  - `MIRAGE_STREAMING_BASE_EXECUTION_MODE_VALUE=...`
- In legacy-based streaming mode, data tasks still trigger legacy events after
  completion, and streaming `data_edges` are used only to wake streaming
  residents early.
- In hybrid-based streaming mode, the previous local predecessor-count behavior
  remains available for the non-streaming compatibility tasks.

What was validated:

- `mirage_runtime` rebuilt successfully inside `mirage.sif` on Slurm job
  `86948` on `node-gpu01`.
- `python3 setup.py build_ext --inplace --force` completed successfully and the
  Python extension was relinked against the updated static
  `libmirage_runtime.a`.
- `import mirage` succeeds inside the container with the updated extension.
- real end-to-end Qwen3-30B-A3B Hopper runs now complete in:
  - default `streaming_data` legacy-base mode
  - `streaming_data` with `MIRAGE_STREAMING_BASE_MODE=hybrid_prelaunch`

What is still blocked:

- multi-token and larger-batch streaming validation still needs to be rerun
  after the termination fixes.
- output correctness has only been spot-checked on the short
  `max_seq_length=40` case. We still need a broader comparison against
  `resident_data` / legacy Mirage / HF on longer outputs.
- at this point in the log, the temporary
  `MIRAGE_STREAMING_DEBUG_PROGRESS=1` instrumentation was still present.
  It was removed later in the cleanup pass recorded below.

Where we are going next:

1. rerun streaming validation on the earlier larger-batch shapes,
2. compare default legacy-base streaming against
   `MIRAGE_STREAMING_BASE_MODE=hybrid_prelaunch` on those shapes,
3. validate correctness against legacy Mirage / HF on longer outputs,
4. only then widen the streamable-op whitelist beyond `TASK_MOE_W13` and
   `TASK_MOE_W2`.

### 2026-03-16: streaming legacy-base event loader bug found and fixed

Root cause of the streaming kernel hang (`Terminated` after `Finished Launching
Persistent Kernel (Async)` in `validation_streaming_mode/mirage_single_v5_run.log`
and `mirage_single_v6_run.log`):

The generated `construct_task_graph(...)` loader always loaded `control_events`
(3 entries) when that key was present in the JSON, even in streaming legacy-base
mode. But in streaming legacy-base mode, all data items' `trigger_event` and
`dependent_event` fields are indices into the full `all_events` array (1142
entries for the Qwen3-30B-A3B graph). This caused:

- `all_event_counters` and `all_event_num_triggers` to be allocated for only 3
  events,
- every `trigger_data_event` call from a streaming or prelaunched data task to
  access `all_event_counters[N]` with N >> 2, silently corrupting memory,
- the EOG event counter (at index 1141 in the full events) to never reach its
  threshold,
- the kernel spinning forever with the scheduler never seeing the EOG event.

The fix is in `src/kernel/runtime_resident.cc` around line 1232.  The generated
event loader now conditionally chooses `all_events` vs `control_events` at
code-generation time based on `streaming_base_mode`:

- `streaming_data` with legacy base (`needs_full_events = true`): always
  load `all_events` (the full 1142-entry event array).
- `streaming_data` with hybrid base, and `resident_data` modes: keep the
  existing behavior of preferring `control_events` when present.

This is a compile-time (code-generation-time) decision so there is no runtime
overhead.

Second observation: validation artifacts from `validation_streaming_mode/`
were generated by a binary built from commit `7afa884` (before `execution_kind`
was added to the JSON emission in `bb560bd`). As a result, all resident tasks
in those JSONs default to `RESIDENT_TASK_EXECUTION_PRELAUNCHED`, so no
streaming ready queues were populated. Regenerating the task graph with the
current binary (post-`bb560bd`) emits `execution_kind` correctly; these stale
artifacts should not be used to judge the current streaming runtime.

### 2026-03-16 later: rebuild and real-run revalidation after the loader fix

The rebuild and rerun were done inside `mirage.sif` on Slurm job `86948`
(`node-gpu01`).

Rebuild status:

- `cmake --build build -j4` completed successfully.
- `python3 setup.py build_ext --inplace --force` completed successfully.
- `transformers` import on this node/container path was no longer the blocker
  (`from transformers import AutoTokenizer` returned in about 17 seconds).

Real validation command used:

```bash
MIRAGE_TASK_GRAPH_MODE=streaming_data \
python3 -u demo/qwen3/demo_30B_A3B_hopper.py \
  --use-mirage \
  --max-num-batched-tokens 1 \
  --max-num-batched-requests 1 \
  --max-seq-length 40 \
  --output-dir validation_streaming_fixed/legacy_base_single_mirage_v3
```

What this rerun proved:

- the generated graph is current, not stale:
  - `schema_version = 2`
  - `resident_tasks = 9859`
  - `all_data = 21331`
  - `data_edges = 128641`
  - `all_events = 1142`
  - `control_events = 3`
  - `execution_kind` is present on resident tasks
  - execution-kind counts are `{0: 9427, 1: 432}`
- the generated CUDA is using the streaming path:
  - [test_rank0.cu](/home/nikhilag/mirage/validation_streaming_fixed/legacy_base_single_mirage_v3/test_rank0.cu)
    includes `streaming_persistent_kernel.cuh`
  - the generated `construct_task_graph(...)` loader uses `json_task_graph["all_events"]`
    for this legacy-base streaming run

One more bug surfaced during the rerun:

- the streaming TU failed to compile at first because
  `enqueue_prelaunched_task_range(...)` and
  `enqueue_prelaunched_dependent_tasks_partitioned(...)` called
  `enqueue_worker_item(...)` before it had been declared in
  `resident_persistent_kernel.cuh`
- this was fixed by adding a forward declaration for
  `enqueue_worker_item(...)`

Current validation outcome after both fixes:

- the streaming megakernel now compiles successfully
- the run reaches:
  - `Finished megakernel compilation...`
  - `worker kernel & scheduler kernel`
  - `Finished Launching Persistent Kernel (Async)`
- this was enough to prove the loader fix was live, but it did not yet solve
  the remaining post-launch hang

Artifacts from this rerun:

- [run.log](/home/nikhilag/mirage/validation_streaming_fixed/legacy_base_single_mirage_v3/run.log)
- [task_graph_rank0.json](/home/nikhilag/mirage/validation_streaming_fixed/legacy_base_single_mirage_v3/task_graph_rank0.json)
- [test_rank0.cu](/home/nikhilag/mirage/validation_streaming_fixed/legacy_base_single_mirage_v3/test_rank0.cu)

Current conclusion from this stage:

- the original loader bug was real and is fixed
- the stale-JSON `execution_kind` issue is also gone in current artifacts
- the streaming path now gets farther than before: graph generation, NVCC,
  and persistent-kernel launch all succeed on a real run
- there is still a remaining post-launch runtime stall in default
  `streaming_data` with legacy base at this point in the log

Additional narrowing from the next code-inspection pass:

- `BEGIN_TASK_GRAPH` is still launching the full legacy-style compatibility
  task range (`first_task_id = 2`, `last_task_id = 21333`) in the generated
  `all_events` array, so this is not a "nothing got enqueued at launch" bug
- the only `first_data_id` is data `0`, and its resident task is
  `execution_kind = PRELAUNCHED`, so the run does not depend on an initial
  streaming-ready seed to get started
- the first actual prelaunched -> streaming boundary in the real graph is:
  - resident `175` (`task_type = 260`, prelaunched, one data item)
  - into resident `176` (`task_type = 161`, streaming, `24` data items)
  - all first streaming data items `176..199` depend on the single prelaunched
    predecessor data `175`
- that means the remaining stall is more likely in the mixed handoff from a
  prelaunched producer into a streaming resident queue, or in the event-driven
  legacy-base prelaunched path before that boundary, than in the initial
  `first_data_ids` seeding logic

What needed to happen next from that point:

1. Rerun the same legacy-base case with progress instrumentation:

```bash
MIRAGE_TASK_GRAPH_MODE=streaming_data \
MIRAGE_STREAMING_DEBUG_PROGRESS=1 \
python3 -u demo/qwen3/demo_30B_A3B_hopper.py \
  --use-mirage \
  --max-num-batched-tokens 1 \
  --max-num-batched-requests 1 \
  --max-seq-length 40 \
  --output-dir validation_streaming_fixed/legacy_base_single_debug
```

2. Inspect whether `completed_terminal_data_count` /
   `completed_streaming_data_count` advance after launch.

3. Only if default legacy-base streaming returns cleanly: try
   `MIRAGE_STREAMING_BASE_MODE=hybrid_prelaunch`.

4. Only after both bases pass: widen the streaming whitelist beyond
   `TASK_MOE_W13` and `TASK_MOE_W2`.

### 2026-03-16 latest: post-launch hang root causes found and fixed

The next debug pass used host-side progress polling with
`MIRAGE_STREAMING_DEBUG_PROGRESS=1`. The important signal was:

- the run advanced all the way to `iter=39`
- `begin=39` and `end=39`
- then `term=1` appeared, proving scheduler 0 had already decided there was no
  next batch and had called `terminate_schedulers(...)`
- but the process still did not exit cleanly

That narrowed the remaining hang to shutdown, not to graph generation, not to
the first streaming handoff, and not to EOG detection.

#### Fix 1: scheduler termination was using the wrong worker-queue primitive

The first post-launch hang root cause was in
`include/mirage/persistent_kernel/resident_persistent_kernel.cuh`,
inside `execute_scheduler_streaming(...)`.

Streaming mode prelaunches compatibility tasks with
`enqueue_worker_item(...)`, which advances a scheduler-local
`worker_queue_next_free_task_pos[...]` cursor and only publishes readiness via
`worker_queue_last_ready_task_id`.

But the termination branch in `execute_scheduler_streaming(...)` was using
`publish_worker_item(...)` to send `TASK_TERMINATE`. That path uses the global
`worker_queue_next_free_task_id[...]` plus a CAS loop on
`worker_queue_last_ready_task_id[...]`.

In streaming mode, the global `worker_queue_next_free_task_id[...]` stayed near
zero because the normal prelaunch path never touched it, while
`worker_queue_last_ready_task_id[...]` had already advanced into the thousands.
So when the scheduler tried to publish `TASK_TERMINATE`, the CAS loop compared
against `next_pos = 0` while `last_ready` was already about `2848`, and it spun
forever.

The fix was to make the termination branch use the same queue primitive as the
rest of streaming prelaunch:

- replace `publish_worker_item(config, worker, 0)` with
  `enqueue_worker_item(config, worker, &worker_queue_next_free_task_pos[...], 0)`

After this fix:

- `sched_done` became `1`
- but `worker_done` was still `0`

So the scheduler-side deadlock was fixed, but the workers still were not
exiting.

#### Fix 2: streaming workers were not checking the terminate flag

The second root cause was simpler.

I had previously added `streaming_terminate_flag`, but the actual
`execute_worker_streaming(...)` loop was missing the check. The terminate check
existed only in the scheduler-dispatch worker path, not in the streaming worker
path.

So after the scheduler exited, the streaming workers kept running forever
because they never looked at the flag.

The fix was to add the same early-exit check at the top of the
`execute_worker_streaming(...)` outer loop:

```cpp
if (config.streaming_terminate_flag != nullptr &&
    atomicAdd(config.streaming_terminate_flag, 0u) != 0u) {
  return;
}
```

After that change, the full single-token streaming run returned cleanly.

#### Real validation after both fixes

The following real runs were completed inside `mirage.sif` on Slurm job
`86948` (`node-gpu01`) after rebuilding `mirage_runtime` and relinking the
Python extension with `python3 setup.py build_ext --inplace --force`.

Default `streaming_data` legacy-base run:

- command:

```bash
MIRAGE_TASK_GRAPH_MODE=streaming_data \
python3 -u demo/qwen3/demo_30B_A3B_hopper.py \
  --use-mirage \
  --max-num-batched-tokens 1 \
  --max-num-batched-requests 1 \
  --max-seq-length 40 \
  --output-dir validation_streaming_fixed/legacy_base_single_mirage_v4
```

- artifacts:
  - [legacy_base_single_mirage_v4.run.log](/home/nikhilag/mirage/validation_streaming_fixed/legacy_base_single_mirage_v4.run.log)
  - [task_graph_rank0.json](/home/nikhilag/mirage/validation_streaming_fixed/legacy_base_single_mirage_v4/task_graph_rank0.json)
- result:
  - returned cleanly
  - `Prompt length 39, generate length 1`
  - `per-token latency (both prefill and decode): 69.834 ms`

`streaming_data` with hybrid base:

- command:

```bash
MIRAGE_TASK_GRAPH_MODE=streaming_data \
MIRAGE_STREAMING_BASE_MODE=hybrid_prelaunch \
python3 -u demo/qwen3/demo_30B_A3B_hopper.py \
  --use-mirage \
  --max-num-batched-tokens 1 \
  --max-num-batched-requests 1 \
  --max-seq-length 40 \
  --output-dir validation_streaming_fixed/hybrid_base_single_mirage_v1
```

- artifacts:
  - [hybrid_base_single_mirage_v1.run.log](/home/nikhilag/mirage/validation_streaming_fixed/hybrid_base_single_mirage_v1.run.log)
  - [task_graph_rank0.json](/home/nikhilag/mirage/validation_streaming_fixed/hybrid_base_single_mirage_v1/task_graph_rank0.json)
- result:
  - returned cleanly
  - `Prompt length 39, generate length 1`
  - `per-token latency (both prefill and decode): 65.359 ms`

Graph sanity from the successful runs:

- both runs emitted schema-v2 graphs with:
  - `resident_tasks = 9859`
  - `all_data = 21331`
  - `data_edges = 128641`
  - `execution_kind` counts `{0: 9427, 1: 432}`

Output sanity from the successful runs:

- both successful runs produced the same visible one-token output on this short
  `max_seq_length=40` case: the assistant output ended at `<think>`
- this is only a limited correctness check because the run allows exactly one
  generated token

Current state after these fixes:

- default `streaming_data` legacy-base mode is now working end-to-end on a real
  single-GPU Qwen Hopper run
- `streaming_data + MIRAGE_STREAMING_BASE_MODE=hybrid_prelaunch` is also
  working end-to-end on the same shape
- the original streaming hang is fixed
- the next meaningful validation is larger-batch / longer-output behavior, not
  more debugging of the old post-launch stall

### 2026-03-16 cleanup pass: removed ad hoc debug scaffolding and revalidated

After the streaming shutdown hang was understood and fixed, the temporary
debug-only code was removed again so the runtime stayed tight:

- removed the streaming debug-marker buffer from `ResidentRuntimeConfig`
- removed the host-side `MIRAGE_STREAMING_DEBUG_PROGRESS` polling path
- removed one-off device-side marker writes and special-case debug tracking
- kept the real fixes:
  - full-event loading for legacy-base streaming
  - scheduler termination via `enqueue_worker_item(...)`
  - worker termination checks via `streaming_terminate_flag`

Revalidation after that cleanup:

- default legacy-base streaming:
  - [legacy_base_single_mirage_v5.run.log](/home/nikhilag/mirage/validation_streaming_fixed/legacy_base_single_mirage_v5.run.log)
  - `67.918 ms/token`
- hybrid-base streaming:
  - [hybrid_base_single_mirage_v2.run.log](/home/nikhilag/mirage/validation_streaming_fixed/hybrid_base_single_mirage_v2.run.log)
  - `63.886 ms/token`

These runs completed cleanly and produced the same visible one-token output as
the earlier successful streaming validations.

## The conceptual difference in one example

It helps to restate the change in plain language.

### Old model

Suppose an operator produces four independent token slices:

- slice 0
- slice 1
- slice 2
- slice 3

and all four tasks trigger the same event.

The downstream operator waits on that event.

Result:

- if slice 0 is ready early, it still waits for slices 1, 2, and 3.

### New model

The upstream operator is one resident task with four data instances.

The downstream operator is another resident task with four data instances.

The runtime creates edges like:

- upstream data 0 -> downstream data 0
- upstream data 1 -> downstream data 1
- upstream data 2 -> downstream data 2
- upstream data 3 -> downstream data 3

Result:

- when upstream data 0 finishes, downstream data 0 can run immediately.
- data 1, 2, and 3 can remain in flight independently.

That is the entire point of the refactor.

## Summary

The resident-task/data change replaces Mirage MPK's flat compute-task event DAG with a two-level runtime model:

- resident tasks hold static execution identity,
- data objects hold per-instance slices and metadata,
- compute dependencies are explicit `data_edges`,
- control events remain only for iteration-level coordination,
- workers still execute batched legacy `TaskDesc` work for performance.

This removes unnecessary compute barriers while preserving the static/dynamic descriptor split and existing task implementation compatibility.

If you only remember three things, remember these:

1. the old graph was correct but too coarse because events acted like phase barriers,
2. the new graph makes dependencies data-specific, not stage-wide,
3. the current validated runtime keeps the resident/data graph for readiness, but executes batched legacy tasks to stay closer to Mirage's fast path.
