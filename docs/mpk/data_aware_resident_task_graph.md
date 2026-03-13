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

Selection is now above the runtime layer, not inside the old runtime:

- Python/Cython exposes both:
  - `generate_task_graph(...)` for the legacy event path
  - `generate_resident_task_graph(...)` for the resident/data path
- `PersistentKernel` chooses between them using:
  - constructor arg `task_graph_mode="legacy_event" | "resident_data"`
  - or environment variable `MIRAGE_TASK_GRAPH_MODE`
  - default mode is `legacy_event`

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
- `resident_ready_data_head`
- `data_ready_next`
- `resident_active_workers`
- `resident_completed_data`
- `completed_terminal_data_count`
- `first_data_ids`

The inherited legacy arrays `all_tasks`, `all_events`, and `first_tasks` are
still present and are still used by the validated worker fast path.

One subtle but important design choice is that the legacy `all_tasks` array is still the actual execution unit on the validated fast path.

That means:

- resident/data objects are the canonical dependency model,
- legacy tasks are the concrete worker-dispatch model.

This is the key architectural compromise that brought the implementation back closer to Mirage's original fast path.

Some queue-related arrays (`resident_ready_data_head`, `data_ready_next`,
`resident_active_workers`) reflect earlier resident-task activation ideas.
They are not the central mechanism on the currently validated fast path.

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

The compute path that is currently implemented and validated works like this:

1. a new iteration starts,
2. scheduler 0 publishes the begin control event,
3. the begin control event activates all `first_data_ids`,
4. each `first_data_id` is mapped through `data_to_task_id` to the corresponding legacy task position in `all_tasks`,
5. schedulers enqueue those legacy task ids into worker queues using the same per-scheduler local queue-position logic Mirage used before,
6. workers batch-load `TaskDesc` objects from `all_tasks` into shared memory, again matching the original Mirage execution style,
7. if a task is a control or compatibility-only task with no data mapping, it follows the old event-based behavior,
8. if a task has a `data_id` mapping, the worker executes the legacy `TaskDesc` and then publishes the completed `data_id` into a scheduler-owned completion queue,
9. local schedulers drain completion queues, walk outgoing `data_edges`, decrement successor predecessor counts, and enqueue newly ready successor tasks by mapping `succ_data_id -> succ_task_id`,
10. terminal data items contribute to end-of-graph completion,
11. once all terminal data has completed, the end control event is published.

### Important consequence

If data item A is ready and data item B is not, A can move into the downstream resident task immediately.

There is no need to wait for every sibling that used to share the same event.

That is the main source of newly exposed overlap.

The important implementation point is that this overlap is exposed by the
resident/data DAG, while execution still uses Mirage's original batched
`TaskDesc` worker path.

In other words, the current architecture is:

- resident/data DAG for correctness and fine-grained readiness,
- legacy batched task execution for performance,
- scheduler-owned dependency release for newly ready data.

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

### Timeline visualization

`scripts/display_task_graph_timeline.py` now understands v2 graphs.

For schema v2:

- the dependency DAG is built from `resident_tasks` and `data_edges`,
- resident tasks become the grouping unit for trace mapping,
- dependency analysis no longer assumes that event fan-in equals compute fan-in.

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
- worker-side batched execution over compatibility `TaskDesc`s,
- scheduler-owned completion queues,
- `task_to_data_id` / `data_to_task_id` mapping logic,
- successor release by walking `data_edges`,
- end-of-graph detection using terminal data completion.

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
- graph validation for predecessor counts / zero-indegree / cycle detection,
- schema-v2 JSON emission and loading,
- compatibility `task_to_data_id` / `data_to_task_id` mappings,
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
- map Perfetto trace groups against resident-task profiler groups.

### `scripts/whatif_model.py`

Changed so analytical fan-in and dependency scoring work with the resident/data DAG instead of only with legacy event fan-in.

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
- the in-tree Python extension rebuilt successfully,
- the containerized Python runtime was confirmed to load `python/mirage/core.cpython-310-x86_64-linux-gnu.so`.

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

The current split implementation was revalidated on:

- `demo/qwen3/demo_30B_A3B_hopper.py`
- single GPU
- `--max-num-batched-tokens 1`
- `--max-num-batched-requests 1`
- `--max-seq-length 40`

The important post-split artifacts are:

- legacy path:
  - `validation_mode_split_legacy/task_graph_rank0.json`
  - `validation_mode_split_legacy/test_rank0.cu`
  - `validation_mode_split_legacy.log`
- resident/data path:
  - `validation_mode_split_resident/task_graph_rank0.json`
  - `validation_mode_split_resident/test_rank0.cu`
  - `validation_mode_split_resident.log`

The split behaved as intended:

- legacy mode emits the old path:
  - generated CUDA includes `persistent_kernel.cuh`
  - generated JSON is legacy schema (`schema_version` absent, treated as v1)
- resident mode emits the new path:
  - generated CUDA includes `resident_persistent_kernel.cuh`
  - generated JSON has `schema_version = 2`
  - resident graph still has:
    - `533` `resident_tasks`
    - `21,331` `all_data`
    - `128,641` `data_edges`
    - `1` `first_data_ids`

Both modes completed the live Hopper run end-to-end from the same checkout.

Measured latencies on the same validation shape:

- legacy/event path: `5.229 ms/token`
- resident/data path: `81.990 ms/token`

So after the split:

- the coexistence requirement is satisfied,
- the old runtime behavior is preserved in-place,
- the resident/data runtime remains much slower than legacy on this workload.

Important caveat:

- these runs validated that both megakernels compile, launch, and return,
- they did not validate that the generated text/tokens match a reference Hugging Face run or the old Mirage runtime,
- semantic correctness of the actual model output still needs explicit checking.

### Compiler warning notes from the live run

The v21 live run still emits `ptxas` warnings such as:

- `C7520` about `wgmma.mma_async` serialization in `worker_kernel`
- function-scope shared-memory dynamic-initialization warnings

These warnings also appear in the clean baseline run, so they are not sufficient to explain the remaining performance gap by themselves.

### Operational note

At one point it was suspected that a stale GPU job was causing bad runs.

Before the successful v21 rerun, `nvidia-smi` on `node-gpu02` showed:

- `0 MiB` memory in use,
- no running compute processes.

That means the v21 measurement was taken on an idle GPU.

## Known limitations and current status

This section is the handoff state as of March 12, 2026.

### 1. Both paths now work end-to-end from one checkout

The current implementation:

- builds,
- preserves the legacy event runtime in the original files,
- keeps the resident/data runtime in separate files,
- generates both legacy and schema-v2 resident graphs,
- compiles both generated megakernels,
- runs both live Hopper MoE modes to completion from the same checkout.

### 2. The system is still much slower than the original baseline

Current live numbers on the same validation shape:

- legacy/event path: `5.229 ms/token`
- resident/data path: `81.990 ms/token`

So the residual problem is performance, not basic correctness.

The best current hypothesis is that the remaining gap comes from explicit fine-grained dependency bookkeeping:

- scheduler-side draining of completion queues,
- `atomicSub` over `128,641` `data_edges` per iteration,
- mapping every ready successor `data_id` back into a compatibility `TaskId`,
- extra queue traffic compared with the old coarse event-release model.

### 3. Model-output correctness has not been validated yet

The live Hopper run proves that the runtime now completes, but it does not prove that the returned tokens are semantically correct.

What still needs to be checked:

- compare generated token ids against a reference Hugging Face run,
- compare generated token ids against the old Mirage runtime on the same prompt,
- verify that intermediate debug output is not masking silent correctness bugs.

Until that comparison is done, correctness should be treated as partially validated only at the "no crash / returns output" level.

### 4. Multi-GPU/NVSHMEM behavior has not been revalidated end-to-end

Compatibility event fields were preserved in `DataDesc` because some task bodies still read them, but full cross-GPU validation has not yet been redone after the scheduler rewrite.

### 5. Some runtime fields are now transitional

The current validated path no longer uses the original resident-local ready-data stack for compute dispatch.

Some arrays are still present because they still help preserve compatibility.

They can be removed once the current architecture is treated as final.

### 6. Compatibility views are still intentionally present

`all_tasks` and `all_events` are still emitted and are still used by the fast worker path.

This is intentional.

The current mental model should be:

- resident/data graph is the canonical dependency graph,
- compatibility tasks are the canonical execution granule.

### 7. Recommended next steps for the next person

If someone picks this up from here, the highest-value next steps are:

1. measure where the remaining `82 ms/token` is going inside the new scheduler path.
2. focus on completion-queue drainage and successor-release cost, not on graph correctness.
3. validate actual token/output correctness against a reference run before treating the implementation as fully correct.
4. look for ways to coalesce or amortize dependency release for groups of edges that are reproducing the old coarse event behavior without reintroducing unnecessary barriers.
5. remove transitional queue/debug fields only after the performance path is settled.
6. rerun end-to-end validation on at least one non-MoE graph and on multi-GPU once the single-GPU path is fast enough.

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
