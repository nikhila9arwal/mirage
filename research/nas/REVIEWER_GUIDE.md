# Reviewer guide: from resident NAS to asynchronous binned MPK (For a human to read)

This guide is for someone who knows the original NAS runtime and is returning
to the project. The central change is **independent request-slot progression,
with ready projection rows grouped into physical batches at runtime**. The
compiler still supplies logical dependencies. Batching changes how selected
logical work executes; it does not remove those dependencies.

For commands, see [How to run](HOW_TO_RUN.md). For import history and preserved
baselines, see [Provenance](PROVENANCE.md).

## Scope and versions

Reviewed on 2026-11-09 against the `nas-async` working tree, whose HEAD is
`d62d226`. Existing local changes include compact FIFO records and packed-weight
loading; these are not all represented by that HEAD. Check `git status` and
the actual compile flags when comparing a run with a commit.

| Source | What to expect |
|---|---|
| This checkout, `nas-async` | Full 48-layer Qwen3-30B-A3B, top-8, successive prefill chunks, autoregressive decode, EOS/cap stopping, offline replacement, FIFO bins for W13/W2 |
| `feature/dense-prefill-bins-20261108`, `3ac9c8d` | Adds QKV/W_O bins that can mix individual prefill and decode rows; uses the actual rectangular projection shapes |
| Isolated online-serving experiment | Reuses native MPK's pinned transport and engine around the NAS runtime; its adapter is not present on either branch above |

## Native online files already present in Mirage
## do not, by themselves, connect an application to our binned runtime.

## What changed relative to the shared-iteration resident path

The old behavior remains available in this source. The new path is opt-in.

| Concern | Shared-iteration resident path | Asynchronous binned path |
|---|---|---|
| Progress clock | Graph-wide iteration | `slot_iteration[slot]`; one active pass per slot |
| Completion boundary | All graph terminals allow shared END handling | A slot's terminals allow that slot to prepare its next pass |
| Preparation | Shared `prepare_next_batch()` | Slot-local `prepare_slot()` and `advance_slot()` |
| Ordinary ready work | Resident scheduling machinery | Same logical task, queued to a worker |
| Ready W13/W2 work | A static task's CUDA body | Expand into token/branch members; select a physical batch later |
| Completion of a batch | Not a separate entity | Retire a descriptor and account for each captured logical member |
| Request state | Shared batch metadata | Private activations and KV storage per slot; weights shared read-only |

This is still a persistent GPU runtime. Python builds and initializes the graph
and launches it; Python does not relaunch a graph for every token. Independent
progress means different slots can be in different layers, chunk/decode phases,
and epochs. It does not guarantee concurrent execution: worker queues, batching
eligibility, and GPU resources can still delay ready work.

## Suggested reading order

Start with the runtime lifecycle, then follow the compiler and model bindings.
Function names are included because line numbers change during experiments.

| File | Read these parts | Question it answers |
|---|---|---|
| [experimental_moe_runtime.cuh](../../include/mirage/persistent_kernel/experimental_moe_runtime.cuh) | `execute_scheduler_experimental_moe`, `experimental_moe_ready`, `experimental_moe_dispatch`, `experimental_moe_complete_logical` | How does ready work become execution and then successor readiness? |
| [experimental_moe_types.h](../../include/mirage/persistent_kernel/experimental_moe_types.h) | `ExperimentalMoePendingSlot`, `ExperimentalMoeMember`, `ExperimentalMoeBatch`, `ExperimentalMoeState` | Who owns each piece of state? |
| [experimental_moe_fifo.h](../../include/mirage/persistent_kernel/experimental_moe_fifo.h) | `experimental_moe_fifo_append`, `experimental_moe_select_fifo`, `experimental_moe_fifo_commit` | How are members grouped, chosen, and removed? |
| [resident_slot_restart.cuh](../../include/mirage/persistent_kernel/resident_slot_restart.cuh) | `prepare_prefill_request`, `advance_slot`, `dispatch_slot`, `handle_slot_restart_event` | How does one slot advance without resetting others? |
| [resident_persistent_kernel.cuh](../../include/mirage/persistent_kernel/resident_persistent_kernel.cuh) | `report_terminal_data_completion`, slot-state helpers, worker/scheduler dispatch | Where does this path attach to resident NAS? |
| [experimental_moe_host.cuh](../../include/mirage/persistent_kernel/experimental_moe_host.cuh) | `init_experimental_moe` | How are graph families bound to tensors, operators, and device state? |
| [runtime_resident.cc](../../src/kernel/runtime_resident.cc) | `Graph::generate_resident_task_graph`, `build_data_aware_task_graph` | How are native tasks/events converted to resident families and DataId edges? |
| [task_register.cc](../../src/kernel/task_register.cc), [persistent_kernel.py](../../python/mirage/mpk/persistent_kernel.py) | Row/chunk operators, slot attention, task-graph generation and `compile` | How are operator views and CUDA calls generated? |
| [demo.py](experiments/qwen_decoder_layer/demo.py) | `build_layer`, `build_graph`, `assign_slots`, `validate_graph` | What exact model graph does the harness register? |

## Identities that must stay distinct

| Identity | Meaning |
|---|---|
| Regular `TaskId` | Static native task index, including control tasks; an ordinary queue item also carries its epoch |
| `ResidentTaskId` | Operator/subtask family with a common task signature; not a dedicated worker or SM |
| `DataId` | One logical data instance of a family, with predecessor/successor edges; reused across slot epochs |
| Slot | Reusable storage and execution state for one request at a time |
| Request ID | Logical input request; can change when a completed slot is reused |
| Slot epoch | Completed/active graph-pass generation; increases across replacement requests too |
| `slot_token_position` | Position of this pass's first input token in the current sequence/KV cache; advances by the active row count and resets on replacement (example below) |
| `op_id` | Concrete batched operation identity, including its layer and kind; not just W13 versus W2 |
| Token / branch | Row within a chunk / selected expert rank within that token |
| Descriptor index / `batch.sequence` | Reusable physical-batch storage / generation of a publication; neither is a slot epoch |

The host binds operator families in registration order, separately for each
kind, into records indexed as `layer * 4 + kind`. The demo registers complete
layers in order and validates the resulting mapping. Shared weights do not
make two layers the same operation. A physical batch has one `op_id`; members
can come from different slots and epochs, but not different layers' weights.

### Clarification: token position versus epoch

Suppose slot 0 processes request A, whose prompt has eight tokens, using
prefill chunks of four tokens. Positions are zero-based; the table shows each
pass's starting position.

| Request in slot 0 | Epoch | `slot_token_position` | Input consumed and prediction |
|---|---:|---:|---|
| A | 1 | 0 | Prompt tokens 0–3; this intermediate chunk's prediction is ignored |
| A | 2 | 4 | Prompt tokens 4–7; predicts the first generated token |
| A | 3 | 8 | First generated token; predicts the second |
| A | 4 | 9 | Second generated token; suppose it predicts EOS |
| B replaces A | 5 | **0** | New request's prompt tokens 0–3 |

**Epoch** counts graph passes executed by the slot and keeps increasing across
replacement requests. **`slot_token_position`** identifies where this pass's
input belongs in the current sequence/KV cache: it advances by four during
prefill, by one during decode, and resets to zero for a new request. The EOS
prediction is saved but is not fed back through attention before admitting B.

## One controller iteration

`execute_scheduler_experimental_moe()` uses lane 0 of one scheduler CTA.
Simplified, with optional diagnostics omitted:

```text
consume up to completion_budget already-available completion messages
    ordinary item: complete its DataId at the captured epoch
    batch item: account for members; complete any logical joins that reach zero
                retire the physically finished descriptor
consume at most one control message (bootstrap, restart, dispatch, termination)
attempt to dispatch one eligible batch
if nothing progressed: short nanosleep
```

Completion draining does not wait for 16 items to accumulate. It stops at an
empty queue. The code default is one; the checked-in measurement launcher uses
16. Each dispatch attempt publishes **at most one descriptor**. It may fail
because no descriptor is free or no pending bin is eligible. **In other words, each scheduleing pass dispatches at most one bin**. The reason why I did this was because it was the naive approach, we should make it dispatch all eligible bins unless we run out of free descriptors. 

The binned host initializer requires one local scheduler, 1–16 workers, one GPU,
split workers/scheduler, and `resident_data/scheduler_dispatch`. Generic resident
code still contains multi-scheduler and hybrid-prelaunch paths; their presence
does not make this binned controller a multi-scheduler implementation. **Have not tested support for more than 16 workers and/or more than 1 scheduler**. I anticipate needing to change some stuff. 

## Readiness, FIFO selection, and ownership

1. `experimental_moe_ready()` validates the captured epoch. Ordinary or disabled
   dense-bin tasks enter a worker queue directly. Enabled batched tasks instead
   establish `remaining_branches[slot] = active_rows * num_branches` and populate
   `pending[slot]`. 
2. Each member is appended to the FIFO for `key = op_id * 128 + expert`.
   Member storage is flat: `(slot * P + token) * R + branch`. There is one FIFO
   structure containing many per-bin linked lists, not one shared-order queue. The FIFO acts like an glorified set. The data feild _______ denotes the head of each binned linked list. 
3. Selection scans nonempty bins. A bin is eligible at `count >= B` **or** age
   `>= T`. The oldest eligible bin wins; equal timestamps use the smaller key.
   The selector takes its oldest `min(count, 16)` members. B is an eligibility
   threshold, not the physical batch capacity.
4. Dispatch reserves a descriptor before removing pending members. The descriptor
   captures their DataIds, epochs, slot/token/branch identities, routing, and row
   maps. It then clears their pending bits and publishes the worker item.
5. The descriptor owns those dispatched members until physical completion is
   handled. Empty pending bits do **not** mean the logical operation is complete.


## Kernel addressing and completion safety

[fixed_moe_adapter.cuh](../../include/mirage/persistent_kernel/fixed_moe_adapter.cuh)
adapts the Hopper MoE arithmetic to compact selected rows. Let
`t = slot * P + token` and `b = t * R + branch`:

| Projection | Input row | Output row |
|---|---|---|
| W13(gate up) | `t` in `[S*P,H]` | `b` in `[S*P*R,2I]` |
| W2(gate down) | `b` in `[S*P*R,I]` | `b` in `[S*P*R,H]` |
| Dense QKV/W_O on the dense branch | `t` | `t` |

Activation vectors stay in slot allocations. Descriptors carry gather/scatter
indices and stable routing metadata; they do not copy weights. A worker calls
the projection collectively across its CTA. 

Descriptor writes precede worker-queue release publication; the worker acquires
readiness before using them. CTA synchronization and completion publication
make outputs visible before the controller releases logical successors.
`remaining_branches` is single-controller state, so its decrement needs no
worker atomic. `active_data[slot]` is an optional correctness guard against
mixing operations; the join counter performs the completion accounting.

A descriptor retires after its physical completion is handled, even if a
member's slot still awaits branches in another descriptor. Holding it until
the entire logical join finishes could exhaust the pool. `in_flight` includes
queued, executing, and completed-but-not-yet-retired descriptors; it is not a
count of busy workers. Ordinary queued tasks are not included in that counter.

## Slot restart, prefill, and autoregression

```text
slot terminal(s) finish -> COMPLETE + captured-epoch control message
  -> advance_slot()
      prepare_slot(): save accepted prediction, prepare next input/admit request
      reset this slot's predecessor/completion state
      activate slot_iteration + 1
      publish slot-dispatch control message
  -> dispatch_slot(): enqueue roots for scheduler_dispatch
```

`make_slot_restart_plan()` is host initialization(does not run in runtime on GPU): it validates ownership and
dependencies, builds per-slot topological DataId lists and offsets, and checks
finite queue bounds. `dispatch_slot()` and slot reset consume those lists on
the GPU. Cross-slot compute edges and unsafe writable aliases are rejected;
the implementation does not erase dependencies to manufacture independence.

For native prefill, `prepare_slot()` selects `prepare_prefill_request()`; the
older HF-prefix `prepare_decode_request()` path is not used. P is the chunked prefill chunk size. `slot_chunk_rows` is P during a full prompt chunk and 1 during decode.
`slot_token_position` advances by that active count. A slot's chunk progresses
through logical operations together; physical MoE members may finish separately
but join before the next logical operation.

The same CUDA `slot_ids` allocation is written by preparation and read by the
embedding task. During decode, preparation copies the previous argmax token
into it. There is no per-token host feedback. Attention consumes QKV, writes
the current K/V at that slot's positions, and reads its own valid history. Each
layer has private per-slot K/V storage; the generated attention call uses local
metadata rather than shared compacted batch CSR metadata. **This is why we only support a context length of 64 and do not support paged attention**

## Compiler integration to review (Warning! Ethan did only understands this at a surface level, this section is almost fully vibe coded)

The resident entry now calls the newer shared `kn::register_mugraph()` in
[runtime.cc](../../src/kernel/runtime.cc), then converts its native tasks/events
in `build_data_aware_task_graph()`. This is a real compiler path, not a manually
authored JSON graph.

Resident family enumeration follows `graph.operators` and its task maps. Native
TaskIds can interleave between families, so explicit TaskId↔DataId mappings and
global TaskId-indexed serialization matter. Dependency conversion uses the
native events, refines producer matches using tensor overlap when possible,
and retains event producers as a fallback when no overlap matches. It does not
independently infer an arbitrary DAG from unresolved addresses.

The model harness generates the graph, assigns slot ownership from actual
tensor views, calls `set_request_slots()`, regenerates, and validates the mapping
before compilation. Inspect `native_graph.json`, `mapping.json`, and
`graph_validation.json` alongside the source; registration order alone does not
prove which row owns a DataId.

## Boundaries and useful review questions

- **One unfinished batched logical operation per slot.** The compact pending
  representation relies on this. Arbitrary intra-slot parallel batched branches
  would need different ownership state.
- **Fixed small cache/chunk contract.** The checked-in full-model path uses one
  64-position page, P=4, top-8, and full prompt chunks. Online tail handling is
  in the separate experiment. This is not a general paged-cache allocator.
- **Not every operation has a bin.** RMSNorm, routing, attention, SiLU, reduction,
  and readout remain ordinary tasks. Main-branch chunked QKV/W_O do too; the
  dense branch changes those two projections only.

[FIFO replay tests](tests/test_fifo.cc) cover selection, partial drains, reset,
and reuse. The [full-model harness](experiments/native_prefill/run.py) adds native
graph, request lifecycle, cache, and repeated-output checks. Neither CPU selector
tests nor successful CUDA compilation alone establishes end-to-end GPU behavior.
