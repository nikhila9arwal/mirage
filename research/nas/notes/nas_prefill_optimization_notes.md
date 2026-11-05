# Potential optimization and simplification points

Date: 2026-11-01  
Code under review: `/u/elu1/mpk_prefill_flat_scan/checkout`

User's code-review notes, organized for later discussion. These are proposed changes and open questions, not confirmed performance findings or approved code changes.

1. **Seen bookkeeping** (original shorthand: “Seem”)

   - In `config` / runtime state.
   - `experimental_moe_clear_seen` is `O(k)`.

2. **Define `members_per_slot` once**

   Consider a shared definition for:

   ```cpp
   constexpr uint32_t members_per_slot =
       EXPERIMENTAL_MOE_CHUNK_ROWS * EXPERIMENTAL_MOE_TOP_K;
   ```

3. **Activation layout: potential CSR / continuous chunks?**

   Review the current mapping:

   ```cpp
   // Compact physical rows gather/scatter independently. Let t=slot*P+token:
   // W13 reads t and writes t*R+branch; W2 reads/writes t*R+branch.
   // Legacy dense batching is P=1 and uses the slot row for both.
   ```

   Could a CSR-style or continuous-chunk representation help?

4. **Simplify `ExperimentalMoeBatch::mask`**

   Could the mask become a single integer if we do not coalesce experts?

5. **Review `ExperimentalMoeState` storage**

   - `ExperimentalBatchOperator` feels wasteful, especially as the layer count grows.
   - Is `ExperimentalMoeBatch batches[EXPERIMENTAL_MOE_POOL]` necessary in its current form?
   - `selection_seen` uses `bool`, which may occupy one byte per entry. What if it used `int32_t` instead?
   - `uint32_t active_data[EXPERIMENTAL_MOE_STATE_SLOTS]` is an optional correctness guard.

6. **Remove unused BF16 ContextIO input handling?**

   If we will never use this path, consider removing it or commenting it out for now:

   ```cpp
   context = get_new_input(
       static_cast<__nv_bfloat16 const *>(config.context_inputs),
       config.num_contexts, config.context_input_width, input_cursor,
       static_cast<__nv_bfloat16 *>(config.context_slot_inputs) +
           size_t(slot) * config.context_input_width);
   ```

7. **Clean up `prepare_slot()`**

   In our current native-prefill mode, the early return through `prepare_prefill_request()` bypasses the older paths below it. Consider commenting out or removing those old blocks if they are no longer needed.

8. **Reduce sparsity in slot input storage**

   ```cpp
   auto *input = static_cast<int64_t *>(config.context_slot_inputs) +
       size_t(slot) * config.prefill_chunk_capacity;
   ```

   Could we introduce some form of “anti-sparseness” around this layout?

9. **Duplicate request/position information**

   ```cpp
   config.request_ids[slot] = request * 64 + position;
   // Example: config.request_ids[1] = 132;
   ```

   `request_ids` and the separately stored request/position may contain duplicate information. Review whether this bookkeeping is necessary.

10. **Call `dispatch_slot()` directly from `advance_slot()`?**

    Current slot-activation flow under review:

    ```text
    advance_slot()
        → publish a dispatch event
        → scheduler reads the event
        → handle_slot_restart_event()
        → dispatch_slot()
    ```

    Could `advance_slot()` call `dispatch_slot()` directly instead of enqueueing and later consuming the control event?

    Would that approach still scale if we introduce multiple schedulers?

11. **Reuse storage for mutually exclusive metadata, as native MPK does with `TaskMetadata`**

    Proposed by the user: consider a tagged union for metadata used by different operations or runtime policies. Inspection below is read-only; no runtime/compiler change is approved or applied here.

    The rule is **share storage only when the values are never needed simultaneously**. Keep the operation/policy discriminator outside the union. A union occupies enough space for its largest member, including alignment; it does not automatically make each array entry as small as its currently active member.

    **We already inherit the native example.** Native defines it at [runtime_header.h:276](/u/elu1/mirage/include/mirage/persistent_kernel/runtime_header.h:276). Our copy contains the same eight-byte union at [runtime_header.h:307](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/runtime_header.h:307). Resident `FullDataDesc`, `DataDesc`, and resolved task descriptors carry it through the [TaskMetadata alias](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/resident_runtime_header.h:54). We should reuse this mechanism for any new mutually exclusive task-specific scalar metadata instead of adding independent fields to every descriptor.

    Concrete places to consider:

    | Location | Possible sharing | Assessment |
    |---|---|---|
    | `ResidentRuntimeConfig::decode_prefix_lengths` and `decode_prompt_lengths` | Both are read-only `int32_t*` length arrays, selected by the preparation policy. | Clearest small opportunity now. One common `request_lengths` pointer would be even simpler than a union. Removes one pointer field, not a request-length allocation. |
    | Imported-prefix preparation fields versus native-prefill preparation fields | Imported mode needs prefix lengths and the KV-binding list; native-prefill mode needs prompt lengths, slot positions, and active chunk rows. | A tagged policy payload can replace separately stored pointer groups. Small config-size improvement; requires mode-aware initialization, validation, and cleanup. |
    | `ExperimentalBatchOperator` | A common header plus a union of operation-specific parameters, particularly when adding more kinds of binned operations. | Useful extension point. The current four projection kinds already share most fields, so a dense/MoE union alone would not shrink today's record. |
    | `TaskDesc` / `DataDesc` / `ResolvedTaskDesc` tensor bindings | A union of typed binding records for RMSNorm, linear, attention, etc., instead of reserving every raw-pointer and TMA-pointer position separately. | Potentially larger benefit, but a much broader compiler/runtime layout change. Requires auditing the largest supported variant; retaining the full generic layout as one union member would retain its size. Defer until profiling justifies it. |

    **Smallest concrete example: request lengths.** Fields are at [resident_runtime_header.h:382](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/resident_runtime_header.h:382). The validator already chooses exactly one length array at [resident_slot_restart.cuh:934](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/resident_slot_restart.cuh:934):

    ```cpp
    // Existing selection:
    lengths = native_prefill ? config.decode_prompt_lengths
                             : config.decode_prefix_lengths;

    // Possible representation; not implemented:
    union RequestLengthMetadata {
      int32_t const *prefix_lengths;
      int32_t const *prompt_lengths;
    } lengths;
    // The existing preparation policy determines the active interpretation.
    // Alternatively, simply keep one pointer named request_lengths.
    ```

    A union here saves eight bytes of pointer storage on the current 64-bit host, before any surrounding layout effects. It does not save the pointed-to length array: the current native-prefill mode already uses only its prompt-length array. Also, the current validator checks `decode_prefix_lengths == nullptr` in native-prefill mode; that test must change if these names become aliases of the same storage. It would be incorrect to overwrite the active pointer by initializing the inactive alias to null afterward.

    **Larger policy grouping, still only a proposal:**

    ```cpp
    // Common admission/output fields remain outside this union.
    union AttentionPreparationMetadata {
      struct {
        int32_t const *prefix_lengths;
        RequestKVBinding *bindings;
        int num_bindings;
      } imported;
      struct {
        int32_t const *prompt_lengths;
        int32_t *token_positions;
        int32_t *active_rows;
      } native_prefill;
    } preparation;
    ```

    The alternatives follow the actual split in [prepare_request_attention():241](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/resident_slot_restart.cuh:241) and the rejection of imported prefixes during native prefill at [resident_slot_restart.cuh:944](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/resident_slot_restart.cuh:944). This tag describes the **preparation policy**, not whether the current pass is prefill or decode: a request that finishes native prefill continues using the native-prefill policy during autoregressive decode. Allocation is already conditional at [resident_persistent_kernel.cuh:3419](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/resident_persistent_kernel.cuh:3419), so the main savings would be config fields, not eliminating large live allocations.

    **Operation records need a realistic expectation of savings.** [ExperimentalBatchOperator:100](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/experimental_moe_types.h:100) already shares `input`, `output`, and `weight_map` across W13, W2, QKV, and W_O. `routing` is MoE-specific, but putting just that pointer into `union { MoEParameters; DenseParameters; }` still reserves the MoE-sized payload in every operator record. This becomes useful if we later add independent fields for other operations that can overlap the MoE-specific fields. Keep common pointers, operation identity, threshold, and timeout outside that union when they are needed together.

    The same caution applies to [ExperimentalMoeBatch:113](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/experimental_moe_types.h:113). A temporary CPU layout probe against the current headers, with `S=16, P=4, R=8, layers=48`, measured:

    | Record/payload | Bytes |
    |---|---:|
    | `ExperimentalBatchOperator` | 88 |
    | `ExperimentalMoePendingSlot` | 152 |
    | `ExperimentalMoeBatch` | 9,200 |
    | All 16 batch descriptors | 147,200 |
    | `ExperimentalMoeState` | 167,200 |
    | Illustrative dense routing/mask payload | 80 |
    | Current-sized MoE routing/mask payload | 8,720 |
    | Union of those dense and MoE payloads | 8,720 |

    Thus `union { DensePayload; MoePayload; }` would **not** shrink the current fixed descriptor pool. Shrinking its large payload needs a compact routing representation or differently sized allocation scheme; that is a separate change. These are host C++ layout measurements, not GPU performance measurements.

    **Fields that should not be overlaid just because their names look operation-specific:**

    - `input_rows` and `output_rows`: W13 simultaneously reads a token row and writes a distinct branch row. W2's equal mappings do not make a union safe for W13.
    - `routing` and `mask`: the existing physical kernel reads both; see [fixed_moe_adapter.cuh:49](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/fixed_moe_adapter.cuh:49).
    - `pending_mask` and `remaining_branches`: undispatched and unfinished work coexist. Dispatched branches leave the mask while still contributing to the completion count.
    - `slot_token_position` / `slot_chunk_rows` and MoE pending metadata: pass metadata remains live while MoE work runs and retires; completion checks explicitly call `request_slot_active_rows()` at [experimental_moe_runtime.cuh:613](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/experimental_moe_runtime.cuh:613).
    - The pending expert table and a batch's members: selected and unselected branches can coexist, and completion/trace code still reads pending metadata until the logical operation finishes.
    - All raw pointers versus all TMA pointers: some kernels use raw activation pointers and a TMA weight descriptor together. A typed operation payload must preserve that combination.

    Any binding-layout change must update compiler-generated accesses and host initialization together. Resident resolution currently copies the generic arrays and checks exact compatibility with `TaskDesc`; see [resident_runtime_header.h:243](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/resident_runtime_header.h:243) and [layout assertions:266](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/resident_runtime_header.h:266). A union changes metadata storage, not the ownership/lifetime of activations, weights, or KV caches.

    Suggested priority: note the small request-policy cleanup now; use tagged operation payloads when adding genuinely different metadata; keep the broader descriptor-layout work behind the TTFT/TPOT bottleneck measurements. No speedup is established by this inspection.

12. **How `config` and `ExperimentalMoeState` occupy GPU memory; what shrinking them can achieve**

    Reading `config` does **not** load the entire `ExperimentalMoeState`, or necessarily the entire `config`. Memory accesses operate on addresses and cache sectors, not whole C++ objects. The allocation arrangement below is confirmed by the source; the exact register and cache behavior has not been measured or checked in generated GPU instructions.

    ```text
    Kernel arguments: config
        |
        | experimental_moe: an 8-byte device pointer
        v
    Device global memory: ExperimentalMoeState
        ├── operators[]
        ├── pending[]
        ├── selection_seen[]
        ├── batches[]
        └── data_to_op_id ──> another device allocation
    ```

    `ResidentRuntimeConfig` contains the state pointer at [resident_runtime_header.h:324](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/resident_runtime_header.h:324). The host separately allocates the state with `cudaMalloc` and copies its initial contents with `cudaMemcpy`; see [experimental_moe_host.cuh:297](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/experimental_moe_host.cuh:297). This initialization copies the complete state, but ordinary field reads during execution do not. The pointed-to lookup arrays, TMA descriptors, queues, activations, and weights are separate allocations.

    This runtime statement creates a reference, not a copy of the structure:

    ```cpp
    ExperimentalMoeState &state = *config.experimental_moe;
    ```

    See [experimental_moe_runtime.cuh:308](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/experimental_moe_runtime.cuh:308). Reading `state.in_flight` conceptually requires only the pointer and the addressed field:

    ```cpp
    // Conceptual operations, not inspected assembly:
    state_pointer = config.experimental_moe; // May already be in a register.
    count = load_u32(state_pointer + offset_of_in_flight);
    ```

    The CPU analogy is useful: dereferencing a field does not fetch every member of the containing object. On modern NVIDIA GPUs, global accesses are serviced in 32-byte transaction units, so a scalar access can fetch neighboring bytes too. Cache hits can avoid HBM traffic; the exact path depends on the generated instruction and cache policy. See the [CUDA Best Practices Guide on coalesced global accesses](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory).

    `config` itself is passed by value to the worker and scheduler kernels; see [resident_persistent_kernel.cuh:2783](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/resident_persistent_kernel.cuh:2783). CUDA uses kernel parameter storage, and the compiler may keep frequently used values in registers. Taking a parameter's address can also lead to a per-thread local copy unless prevented or optimized away. These are compiler decisions we have not inspected here; none automatically copies the allocations referenced by pointer fields. See [CUDA language extensions, including `__grid_constant__`](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html).

    **Potential benefits depend on which bytes are accessed.**

    | Change | Potential benefit |
    |---|---|
    | Make frequently scanned records smaller | Better cache locality and potentially fewer cache sectors accessed. |
    | Put frequently accessed fields together | Less unrelated data fetched with the useful fields. |
    | Reduce reserved but untouched storage | Smaller allocation and initialization copy; possibly little execution-time benefit. |
    | Remove redundant metadata | Fewer reads/writes if that metadata was actually accessed; savings may be offset by recomputation. |

    For the recorded `S=16, P=4, R=8, layers=48` configuration above, the state is **167,200 bytes (about 163 KiB)**. This is one shared device allocation, not that amount per thread or worker.

    A concrete distinction between capacity and traffic is `batch.routing[128 * 16]`: it reserves **8 KiB per descriptor**, but a batch currently selects one or two experts. Dispatch initializes only those expert rows, and the kernel consumes only the selected rows; each routing row contains 16 integers, or 64 bytes. The other rows can remain untouched for that dispatch. See [ExperimentalMoeBatch's routing contract:124](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/experimental_moe_types.h:124). This does not mean actual traffic is exactly 64 or 128 bytes: repeated loads, cache hits, and other descriptor fields also matter. It means eliminating the reserved holes need not produce a proportional reduction in traffic.

    **Tradeoffs and limits:**

    - Bit packing adds masks, shifts, and sometimes read-modify-write operations; independently written fields sharing a word may need additional synchronization.
    - Removing cached or derived information can add recomputation or lookups.
    - Changing alignment or representation can complicate vectorized loads and the existing physical-kernel interface.
    - Reducing the descriptor **pool count**, rather than each descriptor's size, limits the number of outstanding physical batches.
    - Sharing mutable routing storage among outstanding descriptors can overwrite metadata a worker still needs. Preserve each published descriptor's metadata until retirement.
    - Removing validation fields reduces bug detection. A narrower integer that still represents all valid values can be a much simpler change than removing a guard.
    - Shrinking global storage does **not directly increase occupancy**: this state is not allocated in per-CTA shared memory or entirely in registers. A one-byte stored value can still use a full register after loading.
    - A smaller structure does not automatically reduce scan iterations, atomics, or scheduler instruction count.

    Suggested focus: identify the hot metadata and repeated accesses, especially pending-table scans and descriptor headers, before optimizing total `sizeof(ExperimentalMoeState)`. The largest allocation component is not necessarily the largest execution-time cost. This is a source-based explanation and optimization hypothesis, not a measured speedup. No runtime/compiler code was changed.

13. **Make hidden-vector output capture optional; retain token outputs for text generation**

    User proposals:

    - `context_slot_outputs` may be unnecessary for our scheduler; use `context_predicted_tokens` instead.
    - `context_saved_outputs` may be unnecessary for text output; use `context_generated_tokens` instead.

    **Agreed for text generation.** These buffers currently hold different information, so this means omitting hidden-state capture, not aliasing pointers or reinterpreting their element types.

    | Buffer | Current Qwen contents | Purpose |
    |---|---|---|
    | `context_slot_outputs` | BF16 hidden vectors, `[S,2048]`. | Gives the scheduler access to the graph's current hidden output for capture. |
    | `context_predicted_tokens` | Latest int64 prediction per slot, `[S,1]`. | Argmax output used for autoregressive feedback, EOS checking, and token history. |
    | `context_saved_outputs` | Archived BF16 hidden vectors, `[N*64,2048]`. | Post-run numerical validation. |
    | `context_generated_tokens` | Archived int64 predictions, `[N*64,1]`. | Generated text, with valid lengths recorded in `decode_generated_lengths[N]`. |

    Here, `S` is the number of reusable slots and `N` is the number of offline requests. Sources: [field declarations](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/resident_runtime_header.h:370) and [current harness binding](/u/elu1/mpk_prefill_flat_scan/experiments/native_prefill/run.py:176).

    `context_output_width` is **2048** in this harness: Python derives it from the saved tensor's second dimension. It is not the number of predictions per pass. Earlier checksum demos used width 1. The scheduler already saves the token ID separately from the hidden vector:

    ```cpp
    int64_t predicted = config.context_predicted_tokens[slot];
    config.context_generated_tokens[output_row] = predicted;
    save_output(config.context_slot_outputs + size_t(slot) * config.context_output_width,
                config.context_saved_outputs + size_t(output_row) * config.context_output_width,
                config.context_output_width);
    ```

    See [resident_slot_restart.cuh:143](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/resident_slot_restart.cuh:143). The hidden-vector copy transfers **4096 bytes of BF16 payload** per accepted prediction, versus an **eight-byte token-ID store**. Intermediate prompt chunks skip this save branch; the final prompt chunk and each subsequent decode pass enter it. The copy is a serial loop on the advancing scheduler thread, so it occupies the controller while workers may continue executing; see [resident_context_io.cuh:15](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/resident_context_io.cuh:15). Its actual contribution to model latency remains unmeasured here.

    **The live hidden tensor remains necessary.** Removing `context_slot_outputs` from the scheduler interface removes a pointer to that tensor, not the tensor's role in the numerical graph. The current last-row gather writes `last_hidden`, which final normalization and the LM head consume before argmax. Keep that graph path; see [demo.py:303](/u/elu1/mpk_prefill_flat_scan/experiments/qwen_decoder_layer/demo.py:303). For chunk capacity one, the harness uses `layer47_final` directly.

    **The archived hidden tensor is optional for text generation.** Subsequent model passes do not consume `context_saved_outputs`. Generated token history and per-request lengths suffice for CPU tokenizer decoding. Disabling hidden capture can therefore remove that allocation and its scheduler copy while preserving generation. Retain an optional capture mode for numerical investigations and older BF16-vector/checksum experiments.

    A future patch would need to:

    1. Allow omitted hidden-capture buffers in the Python bindings and host validators. They currently require BF16 output buffers and positive output width; see [Python validation](/u/elu1/mpk_prefill_flat_scan/checkout/python/mirage/mpk/persistent_kernel.py:3850) and [host validation](/u/elu1/mpk_prefill_flat_scan/checkout/include/mirage/persistent_kernel/resident_slot_restart.cuh:698). Preserve prediction ownership, terminal-writer, token-history, and feedback checks.
    2. Guard the vector copy in the applicable preparation paths. Keep saving the predicted ID, updating generated lengths, checking EOS, and preparing the next embedding input.
    3. Update the harness's allocations, saved artifacts, and hidden-vector assertions when capture is disabled; see [validation.py:46](/u/elu1/mpk_prefill_flat_scan/experiments/native_prefill/validation.py:46).
    4. Make the separate diagnostic hidden-vector copy optional too. The harness includes `last_hidden` in its [capture list](/u/elu1/mpk_prefill_flat_scan/experiments/native_prefill/run.py:144), and [compare_complete()](/u/elu1/mpk_prefill_flat_scan/experiments/qwen_decoder_layer/bench_io.cuh:7) copies it on the scheduler. Removing only `save_output()` leaves that other copy running.

    This is a proposed capture/interface simplification. It does not require changing the model's numerical operators, batching policy, or slot-progression semantics. No implementation change or measured speedup is claimed here.
