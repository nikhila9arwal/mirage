#pragma once

// Isolated overnight experiment. The opt-in runtime has one metadata owner and
// a fixed, small shape; these are storage records, not another scheduler API.
#include <cstdint>

// Opt in in this isolated experiment. Setting this to zero selects the preserved
// selector, so the same graph/kernels can be used for a matched comparison.
#ifndef NAS_SELECTOR_HISTOGRAM
#define NAS_SELECTOR_HISTOGRAM 0
#endif
#ifndef NAS_SELECTOR_FIFO
#define NAS_SELECTOR_FIFO 0
#endif
#if NAS_SELECTOR_FIFO && NAS_SELECTOR_HISTOGRAM
#error "Select either persistent FIFO bins or rebuilt histograms, not both"
#endif
// Independent opt-ins for the winner-loop experiment. Both zero reproduces
// the previous FIFO path, including its storage layout and per-bin checks.
#ifndef NAS_FIFO_CACHE_OLDEST
#define NAS_FIFO_CACHE_OLDEST 0
#endif
#ifndef NAS_FIFO_SKIP_STATIC_CHECKS
#define NAS_FIFO_SKIP_STATIC_CHECKS 0
#endif
#if (NAS_FIFO_CACHE_OLDEST || NAS_FIFO_SKIP_STATIC_CHECKS) && !NAS_SELECTOR_FIFO
#error "FIFO winner optimizations require NAS_SELECTOR_FIFO=1"
#endif

namespace mirage { namespace runtime {
#ifndef NAS_MOE_STATE_SLOTS
#define NAS_MOE_STATE_SLOTS 16
#endif
// A state slot owns one request's activations. A physical projection can select
// at most sixteen token/branch rows; adding state does not enlarge its GEMM tile.
constexpr uint32_t EXPERIMENTAL_MOE_STATE_SLOTS = NAS_MOE_STATE_SLOTS;
constexpr uint32_t EXPERIMENTAL_MOE_BATCH_CAPACITY = 16;
#ifndef NAS_MOE_TOP_K
#define NAS_MOE_TOP_K 1
#endif
constexpr uint32_t EXPERIMENTAL_MOE_TOP_K = NAS_MOE_TOP_K;
#ifndef NAS_PREFILL_CHUNK_ROWS
#define NAS_PREFILL_CHUNK_ROWS 1
#endif
// A slot still owns one request. Its chunk has capacity P: a full prefill pass
// uses P rows and a decode pass uses one. Physical descriptors select individual
// token/branch members without changing ownership or the capacity-based strides.
constexpr uint32_t EXPERIMENTAL_MOE_CHUNK_ROWS = NAS_PREFILL_CHUNK_ROWS;
constexpr uint32_t EXPERIMENTAL_MOE_PENDING_ROWS =
    EXPERIMENTAL_MOE_STATE_SLOTS * EXPERIMENTAL_MOE_CHUNK_ROWS * EXPERIMENTAL_MOE_TOP_K;
static_assert(EXPERIMENTAL_MOE_CHUNK_ROWS >= 1 && EXPERIMENTAL_MOE_CHUNK_ROWS <= 4,
              "Chunk capacity supports one through four token rows");
static_assert(EXPERIMENTAL_MOE_TOP_K >= 1 && EXPERIMENTAL_MOE_TOP_K <= 8,
              "Joined MoE batching supports one through eight branches");
// These older diagnostics describe one physical row per active slot. Reject
// unsupported combinations instead of silently counting branches as requests.
#if (NAS_MOE_TOP_K > 1 || NAS_PREFILL_CHUNK_ROWS > 1) && \
    (defined(NAS_INVESTIGATE_ALL_ROWS) || defined(NAS_INVESTIGATE_LOCKSTEP_RESTART) || \
     defined(NAS_MOE_OCCUPANCY) || defined(NAS_COMPARE_TASK_TIMING))
#error "Legacy cohort, occupancy, and task-timing modes require one token and one branch per slot"
#endif
constexpr uint32_t EXPERIMENTAL_MOE_EXPERTS = 128;
constexpr uint32_t EXPERIMENTAL_MOE_POOL = 16;
enum ExperimentalBatchKind : uint32_t {
  EXPERIMENTAL_BATCH_W13 = 0,
  EXPERIMENTAL_BATCH_W2 = 1,
  EXPERIMENTAL_BATCH_QKV = 2,
  EXPERIMENTAL_BATCH_WO = 3,
  EXPERIMENTAL_BATCH_KINDS = 4,
};
#ifndef NAS_MOE_LAYERS
#define NAS_MOE_LAYERS 1
#endif
constexpr uint32_t EXPERIMENTAL_MOE_LAYERS = NAS_MOE_LAYERS;
// The full Qwen checkpoint has 48 layers. Keep this prototype's supported
// range explicit; each layer still owns separate weights and operation IDs.
static_assert(EXPERIMENTAL_MOE_LAYERS >= 1 && EXPERIMENTAL_MOE_LAYERS <= 48,
              "This experiment supports one through 48 sequential layers");
// Kinds select CUDA code; concrete operations bind each layer's own tensors.
constexpr uint32_t EXPERIMENTAL_BATCH_OPERATORS =
    EXPERIMENTAL_BATCH_KINDS * EXPERIMENTAL_MOE_LAYERS;
// Branch trace records reserve one byte for op_id. 48 * 4 = 192 fits without
// changing the trace format or the descriptor's ordinary uint32_t operation ID.
static_assert(EXPERIMENTAL_BATCH_OPERATORS <= 256,
              "Branch trace operation IDs must fit in eight bits");
// The comparison uses complete 16-row native windows. Slot indexing and
// pending-table scans also support intermediate multiples of sixteen.
static_assert(EXPERIMENTAL_MOE_STATE_SLOTS >= 16 &&
              EXPERIMENTAL_MOE_STATE_SLOTS <= 4096 &&
              EXPERIMENTAL_MOE_STATE_SLOTS % 16 == 0,
              "This isolated sweep requires 16 through 4096 slots in multiples of 16");
#ifndef NAS_MOE_H
#define NAS_MOE_H 512
#endif
constexpr uint32_t EXPERIMENTAL_MOE_H = NAS_MOE_H;
#ifndef NAS_MOE_I
#define NAS_MOE_I 256
#endif
constexpr uint32_t EXPERIMENTAL_MOE_I = NAS_MOE_I;
constexpr uint32_t EXPERIMENTAL_MOE_ITEM_TAG = 0x80000000u;

// A transient selector row, reconstructed from its slot; never stored in state.
struct ExperimentalMoePending {
  uint32_t pending, data_id, epoch, expert, op_id;
  unsigned long long ready_ns;
};

// One unfinished logical operation owns a slot. Its members share identity and
// readiness, while expert IDs stay per token/branch. A set bit means undispatched
// work; immutable descriptors own the dispatched members. Keep this header and
// expert table until remaining_branches reaches zero, even when the mask is empty.
struct ExperimentalMoePendingSlot {
  uint32_t pending_mask, data_id, epoch, op_id;
  unsigned long long ready_ns;
  uint32_t experts[EXPERIMENTAL_MOE_CHUNK_ROWS][EXPERIMENTAL_MOE_TOP_K];
};
static_assert(EXPERIMENTAL_MOE_CHUNK_ROWS * EXPERIMENTAL_MOE_TOP_K <= 32,
              "One pending bit per token/branch member must fit in uint32_t");

// One native operator family binds one layer's shared tensors. Its kind selects
// CUDA code, while its op_id selects this particular record and weight allocation.
// The host builds these records once; the running kernel never changes them.
struct ExperimentalBatchOperator {
  uint32_t kind, layer_id, resident_task_id, present, enabled;
  uint32_t input_width, output_width, num_experts, num_branches;
  void *input, *output, *weight_map;
  int32_t const *routing; // Only MoE: token-major [S*P,128], ranks 1..R; zero is absent.
  uint32_t threshold;
  unsigned long long timeout_ns;
};

struct ExperimentalMoeMember {
  uint32_t slot, token, branch, data_id, epoch;
};

struct alignas(16) ExperimentalMoeBatch {
  // Sequence identifies an invocation, while the array index identifies reusable
  // storage. A late completion may never refer to a new occupant of that storage.
  uint32_t sequence, live, op_id, expert, count; // expert is the first mask entry.
  uint32_t expert_count; // At most two for early or deadline coalescing.
  ExperimentalMoeMember members[EXPERIMENTAL_MOE_BATCH_CAPACITY];
  // Compact physical rows gather/scatter independently. Let t=slot*P+token:
  // W13 reads t and writes t*R+branch; W2 reads/writes t*R+branch.
  // Legacy dense batching is P=1 and uses the slot row for both.
  alignas(16) uint32_t input_rows[EXPERIMENTAL_MOE_BATCH_CAPACITY];
  alignas(16) uint32_t output_rows[EXPERIMENTAL_MOE_BATCH_CAPACITY];
  // Expert-major membership. Only mask[0..expert_count) rows are read by the
  // kernel; the controller initializes those complete 16-entry rows on reuse.
  // Each physical row has binary membership in exactly one expert. Logical
  // top-k is expanded by the scheduler; the physical kernel still uses top-1.
  alignas(16) int32_t routing[EXPERIMENTAL_MOE_EXPERTS *
                              EXPERIMENTAL_MOE_BATCH_CAPACITY];
  alignas(16) int32_t mask[EXPERIMENTAL_MOE_EXPERTS + 4];
};

// Temporary bin statistics owned by the ONE controller. Separate arrays keep
// the 16-bit counts compact (a count/timestamp struct would have padding).
// A token routes to distinct experts, so one bin has at most S*P members, not
// S*P*R. The routing checks in experimental_moe_ready enforce that invariant.
constexpr uint32_t EXPERIMENTAL_MOE_SELECTION_BINS =
    EXPERIMENTAL_BATCH_OPERATORS * EXPERIMENTAL_MOE_EXPERTS;
static_assert(uint64_t(EXPERIMENTAL_MOE_STATE_SLOTS) *
                  EXPERIMENTAL_MOE_CHUNK_ROWS <= UINT16_MAX,
              "A selector bin must fit in its uint16_t count");
struct ExperimentalMoeSelectionScratch {
  uint16_t count[EXPERIMENTAL_MOE_SELECTION_BINS];
  uint64_t oldest[EXPERIMENTAL_MOE_SELECTION_BINS];
  // There cannot be more touched bins than pending token/branch positions.
  uint32_t touched[EXPERIMENTAL_MOE_PENDING_ROWS];
};

// Persistent bins owned by the ONE controller. count is the FIFO length, not
// another independently maintained histogram. Only undispatched members live
// here; descriptors own members after dispatch. Each flat (slot,token,branch)
// position supplies one link, so we do not reserve a large FIFO for every bin.
struct ExperimentalMoeFifo {
  uint16_t count[EXPERIMENTAL_MOE_SELECTION_BINS];
  uint32_t head[EXPERIMENTAL_MOE_SELECTION_BINS];
  uint32_t tail[EXPERIMENTAL_MOE_SELECTION_BINS];
  uint32_t next[EXPERIMENTAL_MOE_PENDING_ROWS];
  uint32_t active[EXPERIMENTAL_MOE_PENDING_ROWS];
  uint32_t active_count;
  // The head member's slot retains ready_ns until all its members complete.
  // Optional duplication saves the dependent head -> slot lookup on every
  // winner scan. Only first append and a partial dispatch change a bin's age.
#if NAS_FIFO_CACHE_OLDEST
  uint64_t oldest[EXPERIMENTAL_MOE_SELECTION_BINS];
#endif
};

struct ExperimentalMoeState {
  ExperimentalBatchOperator operators[EXPERIMENTAL_BATCH_OPERATORS];
  // -1 denotes ordinary work. A disabled marked dense operation keeps its ID
  // here, but its ready task follows the ordinary worker path.
  int32_t const *data_to_op_id;
  uint32_t in_flight, next_sequence;
  uint32_t early_coalesce; // Host opt-ins; preserved across invocation resets.
  uint32_t deadline_coalesce;
  // Sequential layers permit one unfinished batched logical operation per slot.
  // The single scheduler owns both arrays. remaining_branches counts the live
  // n*R token/branch members (n=P for prefill, n=1 for decode). Only zero
  // completes the logical DataId.
  uint32_t remaining_branches[EXPERIMENTAL_MOE_STATE_SLOTS];
  // active_data[slot] is an OPTIONAL correctness guard, not needed for valid
  // scheduling or the join. Activation checks remaining==0 and active==INVALID
  // to enforce the one-unfinished-operation assumption. Clear active BEFORE
  // complete_logical, because that call may immediately ready another operation.
  uint32_t active_data[EXPERIMENTAL_MOE_STATE_SLOTS];
  // op_id identifies both layer and kind. Legacy dense uses token/branch zero.
  ExperimentalMoePendingSlot pending[EXPERIMENTAL_MOE_STATE_SLOTS];
  // Single-controller storage stays off the thread-local stack. FIFO bins are
  // persistent readiness state. Histogram/legacy instead rebuild scratch on
  // each selection and are retained for comparisons in this isolated copy.
#if NAS_SELECTOR_FIFO
  ExperimentalMoeFifo fifo;
#elif NAS_SELECTOR_HISTOGRAM
  ExperimentalMoeSelectionScratch selection;
#else
  bool selection_seen[EXPERIMENTAL_MOE_PENDING_ROWS];
#endif
  ExperimentalMoeBatch batches[EXPERIMENTAL_MOE_POOL];
};
}} // namespace mirage::runtime
