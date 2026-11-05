#pragma once
#include "experimental_moe_selection.h"
#ifdef __CUDACC__
#define NAS_MOE_FIFO_INLINE __host__ __device__ __forceinline__
#else
#define NAS_MOE_FIFO_INLINE inline
#endif

namespace mirage { namespace runtime {

// These helpers run on the single controller, and also in the CPU/GPU tests.
// The caller establishes pending metadata before append and reserves a physical
// descriptor before selection. Selection itself never removes any work.
NAS_MOE_FIFO_INLINE bool experimental_moe_fifo_append(
    ExperimentalMoeFifo &fifo, uint32_t key, uint32_t member) {
  if (key >= EXPERIMENTAL_MOE_SELECTION_BINS ||
      member >= EXPERIMENTAL_MOE_PENDING_ROWS ||
      fifo.count[key] == UINT16_MAX) return false;
  if (fifo.count[key] == 0u) {
    if (fifo.active_count >= EXPERIMENTAL_MOE_PENDING_ROWS) return false;
    fifo.active[fifo.active_count++] = key;
    fifo.head[key] = member;
  } else {
    fifo.next[fifo.tail[key]] = member;
  }
  fifo.next[member] = UINT32_MAX;
  fifo.tail[key] = member;
  ++fifo.count[key];
  return true;
}

// Counts and oldest ages are already available. Scan only nonempty bins,
// then follow at most C links for the winner: O(active bins + C).
NAS_MOE_FIFO_INLINE ExperimentalMoeSelection experimental_moe_select_fifo(
    ExperimentalMoePendingSlot const (&pending)[EXPERIMENTAL_MOE_STATE_SLOTS],
    ExperimentalBatchOperator const (&operators)[EXPERIMENTAL_BATCH_OPERATORS],
    ExperimentalMoeFifo const &fifo, unsigned long long now,
    uint32_t eligible_ops = ~0u,
    ExperimentalMoeSelectionTiming *timing = nullptr) {
#ifndef NAS_INVESTIGATE_ALL_ROWS
  (void)eligible_ops;
#endif
  experimental_moe_selection_stamp(timing, 0);
  ExperimentalMoeSelection selected;
  uint64_t best_time = UINT64_MAX;
  uint32_t best_key = UINT32_MAX;
  experimental_moe_selection_stamp(timing, 1);

  for (uint32_t i = 0; i < fifo.active_count; ++i) {
    uint32_t key = fifo.active[i];
    uint32_t op_id = key / EXPERIMENTAL_MOE_EXPERTS;
    auto const &op = operators[op_id];
    if (!op.present || !op.enabled) continue;
#ifdef NAS_INVESTIGATE_ALL_ROWS
    if ((eligible_ops & (1u << op_id)) == 0u) continue;
#endif
    uint32_t count = fifo.count[key];
    constexpr uint32_t members_per_slot =
        EXPERIMENTAL_MOE_CHUNK_ROWS * EXPERIMENTAL_MOE_TOP_K;
    uint64_t oldest = pending[fifo.head[key] / members_per_slot].ready_ns;
    bool threshold = count >= op.threshold;
    bool timeout = now - oldest >= op.timeout_ns;
    // Equal ages use (op_id, expert), independent of active-list swaps. This
    // intentionally replaces the old flat-member-index tie rule. Members of
    // the selected bin leave in FIFO arrival order, not flat slot order.
    if ((threshold || timeout) &&
        (oldest < best_time || (oldest == best_time && key < best_key))) {
      best_time = oldest;
      best_key = key;
      selected.op_id = op_id;
      selected.expert = key % EXPERIMENTAL_MOE_EXPERTS;
      selected.group_count = count;
      selected.reason = threshold ? 1u : 2u;
      selected.active_bin_index = i;
    }
  }
  experimental_moe_selection_stamp(timing, 2);

  if (selected.op_id >= 0) {
    uint32_t target = selected.group_count < EXPERIMENTAL_MOE_BATCH_CAPACITY
                          ? selected.group_count : EXPERIMENTAL_MOE_BATCH_CAPACITY;
    uint32_t member = fifo.head[best_key];
    for (uint32_t j = 0; j < target; ++j) {
      uint32_t token_row = member / EXPERIMENTAL_MOE_TOP_K;
      selected.slots[j] = token_row / EXPERIMENTAL_MOE_CHUNK_ROWS;
      selected.tokens[j] = token_row % EXPERIMENTAL_MOE_CHUNK_ROWS;
      selected.branches[j] = member % EXPERIMENTAL_MOE_TOP_K;
      member = fifo.next[member];
    }
    selected.count = target;
  }
  experimental_moe_selection_stamp(timing, 3);
  if (timing != nullptr) {
    timing->groups = fifo.active_count;
    timing->boundary[4] = timing->boundary[3]; // No per-selection reset.
  }
  return selected;
}

// Commit only after the descriptor has captured all selected members. No
// other controller action can run between select and commit. Removing work
// does NOT complete its logical DataId: the existing branch join still waits
// for GPU completion. Pending bits are cleared by the existing dispatch loop.
NAS_MOE_FIFO_INLINE bool experimental_moe_fifo_commit(
    ExperimentalMoeFifo &fifo, ExperimentalMoeSelection const &selected) {
  uint32_t i = selected.active_bin_index;
  if (selected.op_id < 0 || i >= fifo.active_count || selected.count == 0u ||
      selected.count > EXPERIMENTAL_MOE_BATCH_CAPACITY) return false;
  uint32_t key = uint32_t(selected.op_id) * EXPERIMENTAL_MOE_EXPERTS + selected.expert;
  if (fifo.active[i] != key || fifo.count[key] != selected.group_count ||
      selected.count > fifo.count[key]) return false;
  // Selection already traversed these links. One lookup after the last
  // selected member advances the head past the entire captured prefix.
  uint32_t j = selected.count - 1u;
  uint32_t last = (selected.slots[j] * EXPERIMENTAL_MOE_CHUNK_ROWS +
                   selected.tokens[j]) * EXPERIMENTAL_MOE_TOP_K + selected.branches[j];
  fifo.head[key] = fifo.next[last];
  fifo.count[key] -= selected.count;
  if (fifo.count[key] == 0u) {
    fifo.tail[key] = UINT32_MAX;
    fifo.active[i] = fifo.active[--fifo.active_count];
  }
  return true;
}
}} // namespace mirage::runtime
#undef NAS_MOE_FIFO_INLINE
