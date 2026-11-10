# Instructions for coding and review agents

Scope: this repository and its subdirectories, unless a more specific
`AGENTS.md` applies. This file is operational context for agents. Human-facing
explanations belong in `research/nas/REVIEWER_GUIDE.md` and `HOW_TO_RUN.md`.

## Establish the checkout and task scope

- Read `git status --short`, the current branch, and HEAD before editing.
  Preserve user edits and deliberate documentation deletions. When working from
  a newer remote commit, inspect it and prefer a fast-forward; do not overwrite
  local work or force-push.
- For a review request, inspect and report findings without changing runtime,
  compiler, or scheduling behavior. A proposal in the guide or optimization
  notes is not an instruction to implement it.
- Prefer small, readable changes and existing helpers. Comment ownership,
  publication ordering, and reasons for non-obvious constraints. Avoid unrelated
  refactoring or new abstractions for this research prototype.
- Verify which branch contains the feature under review. `nas-async` has the
  offline prefill/decode path; `feature/dense-prefill-bins-20261108` adds dense
  prefill bins. The isolated online adapter is not automatically included just
  because native Mirage online files exist. Do not silently port another branch.
- Historical `/u/elu1` and `/work/hdd` paths are provenance, not portable
  prerequisites. Use repository-relative source paths and explicit asset paths.

## Find the implementation

| Concern | Starting files / symbols |
|---|---|
| Controller and workers | `include/mirage/persistent_kernel/experimental_moe_runtime.cuh`: `execute_scheduler_experimental_moe`, `experimental_moe_ready`, `experimental_moe_dispatch`, `experimental_moe_complete_logical`, `execute_worker_experimental_moe` |
| Bin membership and selection | `experimental_moe_fifo.h`: append/select/commit; `experimental_moe_types.h`: pending records, FIFO, operators, members, descriptors |
| Host bindings and validation | `experimental_moe_host.cuh`: `init_experimental_moe` |
| Per-slot lifecycle | `resident_slot_restart.cuh`: `prepare_prefill_request`, `advance_slot`, `dispatch_slot`, `handle_slot_restart_event`, `make_slot_restart_plan` |
| Resident integration | `resident_persistent_kernel.cuh`, `resident_runtime_header.h` |
| Compact projection adapter | `fixed_moe_adapter.cuh`, `tasks/hopper/moe_linear_swapAB_hopper.cuh` |
| Compiler | `src/kernel/runtime.cc`, `src/kernel/runtime_resident.cc`, `src/kernel/task_register.cc` |
| Python bindings / model | `python/mirage/mpk/persistent_kernel.py`, `research/nas/experiments/qwen_decoder_layer/demo.py` |
| End-to-end execution / timing | `research/nas/experiments/native_prefill/run.py`, `research/nas/measurement/measure.py` |

Bare `.cuh`/`.h` paths in this table are relative to
`include/mirage/persistent_kernel/`. Read the actual caller and compile-time
guards before interpreting a helper: legacy, histogram, FIFO, cohort, and
diagnostic paths coexist.

## Contracts to preserve and review

- The binned path currently requires single-GPU `resident_data`,
  `scheduler_dispatch`, one controller, and 1–16 workers. Check host validation
  rather than assuming generic resident multi-scheduler/hybrid support applies.
- Distinguish static TaskId, logical DataId, resident family, request, slot,
  slot epoch, sequence token position, operation/layer ID, and descriptor
  generation. Never substitute `current_iteration` for a slot epoch.
- Slot epochs increase across request replacement; token position resets for a
  new request. P is storage capacity/stride; active rows are P for full prefill
  and one for decode. Do not schedule stale padded rows.
- There is one unfinished batched logical operation per slot. W13/W2 expand it
  into active-token × branch members. `remaining_branches[slot]` includes pending
  and dispatched members; only zero releases the logical DataId's successors.
  `active_data[slot]` is an optional ownership guard, not the join counter.
- Bins are keyed by concrete `(op_id, expert)`. Do not combine different layers'
  weights merely because their TaskType/kind matches. Dense bins use expert zero
  only on a path that supports their actual dimensions and active rows.
- `fifo.head[key]` is a flat member index; `next[member]` is a member link.
  In the hot-record variant, `active_position[key]` is active-record index + 1,
  with zero meaning absent. It is not the head. Active records are unordered;
  members within each bin are in arrival order.
- FIFO selection chooses the oldest eligible bin, with key as the equal-age
  tie-breaker, then at most 16 members. Eligibility is threshold OR timeout.
  First-eligible early exit, draining dispatches in a loop, companions, and
  cohort gating are policy changes, not equivalent metadata optimizations.
- The controller owns FIFO/pending/join mutations. Workers read published
  descriptors. Preserve release/acquire publication, CTA synchronization,
  captured member epochs, and the descriptor's immutable lifetime.
- Reserve a descriptor before removing pending work. Retire it after its
  physical completion is handled, even if its members' logical joins still
  await other descriptors. `in_flight` is not a busy-worker count.
- Reset only the completed slot after its terminal completion. Retain the
  synthetic END bootstrap distinction. Do not introduce global preparation or
  a cross-slot barrier into independent advancement.
- Preserve the finite invocation queue-capacity proof, slot-private writable
  activations/KV, and shared read-only weights. The offline path does not provide
  general unbounded queue backpressure or a multi-page KV allocator.
- Compiler TaskIds can interleave across families. Preserve explicit mappings
  and initialized tensor descriptors; unresolved pointers cannot establish
  tensor overlap. Slot validation must not discard real cross-slot dependencies.

## Verification workflow

- Select the existing Python environment; verify both `mirage.__file__` and
  `mirage.core.__file__` resolve inside the intended checkout. A virtualenv or
  `PYTHONPATH` alone does not build the extension. Do not install an upstream
  wheel over the fork. Rebuild compiler sources when required; see `INSTALL.md`
  and `research/nas/HOW_TO_RUN.md` for setup and full-model commands.
- For FIFO/runtime metadata or measurement changes, run the existing CPU checks:

  ```bash
  "${NAS_PYTHON:-python3}" -B research/nas/tests/run_checks.py
  ```

  This compiles C++17 replay tests and runs accounting/packed-loader tests;
  `CXX` selects the compiler. It does not execute a model on the GPU.
- When CUDA replay compilation is relevant, use the same command with
  `--compile-gpu` (`NVCC` selects NVCC). Compilation success is not GPU execution.
- For documentation-only edits, check links, formatting, command options, and
  `git diff --check`; do not launch a GPU job or download weights just for prose.
- GPU tests must use an allocation, a timeout, and a new output directory. Record
  branch/commit/diff, imports, GPU, mode, compile flags, workload identity, slots,
  workers, descriptor capacity, B/T, and instrumentation. Keep results in ignored
  directories; do not commit weights, caches, binaries, or raw large traces.
- Do not disable invariants, reduce work, substitute routing, or alter END/join
  semantics merely to make a check pass. If an ablation is requested, isolate it
  and document exactly which checks or behavior changed.
- Full-model checks must separately report graph generation, compilation, GPU
  execution, lifecycle/cache validation, exact preserved-NAS tokens/lengths, and
  HF agreement. Plausible generated text is not a correctness check. Do not claim
  bitwise HF agreement from identical repeated NAS outputs.

## Evidence and reporting

- Cite the current file, function, and relevant line for a review finding. Explain
  its reachable path and consequence; distinguish confirmed behavior from a
  hypothesis. Verify compiler claims against source, not just the reviewer guide.
- Distinguish ready, enqueued, GPU-started, GPU-completed, and controller-retired
  work. Preserve publication and completion timing meanings when adding probes.
- Compare throughput with tracing disabled; measure instrumentation separately.
  Traces can change batch formation. Do not attribute a speedup to metadata alone
  if policy, kernel geometry, routing, or useful-row counts changed too.
- Worker-seconds sum execution over workers. They are not elapsed time or whole-
  GPU utilization. Do not add nested selection/completion scopes twice. State
  whether TTFT includes waiting for slot admission and how TPOT is computed.
- Report checks actually run, failures, and remaining limitations. Do not treat
  archived results from a different checkout as a fresh validation of edits.
