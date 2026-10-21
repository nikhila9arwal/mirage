#pragma once

// Isolated overnight experiment. The opt-in runtime has one metadata owner and
// a fixed, small shape; these are storage records, not another scheduler API.
#include <cstdint>

namespace mirage { namespace runtime {
#ifndef NAS_MOE_STATE_SLOTS
#define NAS_MOE_STATE_SLOTS 16
#endif
// A state slot owns one request's activations. A physical projection can select
// at most sixteen of those slots; adding state does not enlarge its GEMM tile.
constexpr uint32_t EXPERIMENTAL_MOE_STATE_SLOTS = NAS_MOE_STATE_SLOTS;
constexpr uint32_t EXPERIMENTAL_MOE_BATCH_CAPACITY = 16;
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
static_assert(EXPERIMENTAL_MOE_LAYERS == 1 || EXPERIMENTAL_MOE_LAYERS == 2,
              "This isolated experiment supports one or two sequential layers");
// Kinds select CUDA code; concrete operations bind each layer's own tensors.
constexpr uint32_t EXPERIMENTAL_BATCH_OPERATORS =
    EXPERIMENTAL_BATCH_KINDS * EXPERIMENTAL_MOE_LAYERS;
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

struct ExperimentalMoePending {
  uint32_t pending, data_id, epoch, expert, op_id;
  unsigned long long ready_ns;
};

// One native operator family binds one layer's shared tensors. Its kind selects
// CUDA code, while its op_id selects this particular record and weight allocation.
// The host builds these records once; the running kernel never changes them.
struct ExperimentalBatchOperator {
  uint32_t kind, layer_id, resident_task_id, present, enabled;
  uint32_t input_width, output_width, num_experts;
  void *input, *output, *weight_map;
  int32_t const *routing; // Only MoE: slot-major [S,128] top-1 ranks.
  uint32_t threshold;
  unsigned long long timeout_ns;
};

struct ExperimentalMoeMember {
  uint32_t slot, data_id, epoch;
};

struct alignas(16) ExperimentalMoeBatch {
  // Sequence identifies an invocation, while the array index identifies reusable
  // storage. A late completion may never refer to a new occupant of that storage.
  uint32_t sequence, live, op_id, expert, count; // expert is the first mask entry.
  uint32_t expert_count; // At most two for early or deadline coalescing.
  ExperimentalMoeMember members[EXPERIMENTAL_MOE_BATCH_CAPACITY];
  // Derived from members[j].slot at formation, then immutable until retirement.
  // This contiguous array is the kernel's compact-row -> state-row mapping.
  alignas(16) uint32_t state_rows[EXPERIMENTAL_MOE_BATCH_CAPACITY];
  // Expert-major membership. Only mask[0..expert_count) rows are read by the
  // kernel; the controller initializes those complete 16-entry rows on reuse.
  // Each selected local row has rank 1 for exactly one expert (top-1).
  alignas(16) int32_t routing[EXPERIMENTAL_MOE_EXPERTS *
                              EXPERIMENTAL_MOE_BATCH_CAPACITY];
  alignas(16) int32_t mask[EXPERIMENTAL_MOE_EXPERTS + 4];
};

struct ExperimentalMoeState {
  ExperimentalBatchOperator operators[EXPERIMENTAL_BATCH_OPERATORS];
  // -1 denotes ordinary work. A disabled marked dense operation keeps its ID
  // here, but its ready task follows the ordinary worker path.
  int32_t const *data_to_op_id;
  uint32_t in_flight, next_sequence;
  uint32_t early_coalesce; // Host opt-ins; preserved across invocation resets.
  uint32_t deadline_coalesce;
  // Sequential layers permit at most one undispatched node per kind and slot.
  // The entry's op_id distinguishes layers without adding another table axis.
  ExperimentalMoePending pending[EXPERIMENTAL_BATCH_KINDS]
                                [EXPERIMENTAL_MOE_STATE_SLOTS];
  ExperimentalMoeBatch batches[EXPERIMENTAL_MOE_POOL];
};
}} // namespace mirage::runtime
