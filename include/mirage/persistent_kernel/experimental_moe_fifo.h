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
    ExperimentalMoeFifo &fifo, uint32_t key, uint32_t member,
    unsigned long long ready_ns, uint32_t threshold = 0u,
    unsigned long long timeout_ns = 0u) {
#if NAS_FIFO_HOT_RECORDS
  if (key >= EXPERIMENTAL_MOE_SELECTION_BINS ||
      member >= EXPERIMENTAL_MOE_PENDING_ROWS) return false;
  uint32_t position = fifo.active_position[key];
  if (position == 0u) {
    if (fifo.active_count >= EXPERIMENTAL_MOE_PENDING_ROWS) return false;
    uint32_t index = fifo.active_count++;
    // Threshold and timeout are immutable operation parameters. Capture once
    // when this bin becomes active; the winner scan needs no operator lookup.
    fifo.active[index] = {ready_ns, timeout_ns, key, threshold, 0u, 0u, 0u};
    position = index + 1u;
    fifo.active_position[key] = position;
    fifo.head[key] = member;
  } else {
    if (fifo.active[position - 1u].count == UINT16_MAX) return false;
    fifo.next[fifo.tail[key]] = member;
  }
  fifo.next[member] = UINT32_MAX;
  fifo.tail[key] = member;
  ++fifo.active[position - 1u].count;
#else
  (void)threshold;
  (void)timeout_ns;
#if !NAS_FIFO_CACHE_OLDEST
  (void)ready_ns;
#endif
  if (key >= EXPERIMENTAL_MOE_SELECTION_BINS ||
      member >= EXPERIMENTAL_MOE_PENDING_ROWS ||
      fifo.count[key] == UINT16_MAX) return false;
  if (fifo.count[key] == 0u) {
    if (fifo.active_count >= EXPERIMENTAL_MOE_PENDING_ROWS) return false;
    fifo.active[fifo.active_count++] = key;
    fifo.head[key] = member;
#if NAS_FIFO_CACHE_OLDEST
    fifo.oldest[key] = ready_ns;
#endif
  } else {
    // Appends arrive in timestamp order. An existing head keeps its age.
    fifo.next[fifo.tail[key]] = member;
  }
  fifo.next[member] = UINT32_MAX;
  fifo.tail[key] = member;
  ++fifo.count[key];
#endif
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
#if NAS_FIFO_CACHE_OLDEST
  (void)pending;
#endif
#if NAS_FIFO_HOT_RECORDS
  (void)operators;
#endif
  experimental_moe_selection_stamp(timing, 0);
  ExperimentalMoeSelection selected;
  uint64_t best_time = UINT64_MAX;
  uint32_t best_key = UINT32_MAX;
  experimental_moe_selection_stamp(timing, 1);

  for (uint32_t i = 0; i < fifo.active_count; ++i) {
#if NAS_FIFO_HOT_RECORDS
    auto const &bin = fifo.active[i];
    uint32_t key = bin.key;
    uint32_t op_id = key / EXPERIMENTAL_MOE_EXPERTS;
#ifdef NAS_INVESTIGATE_ALL_ROWS
    if ((eligible_ops & (1u << op_id)) == 0u) continue;
#endif
    uint32_t count = bin.count;
    uint64_t oldest = bin.oldest;
    bool threshold = count >= bin.threshold;
    bool timeout = now - oldest >= bin.timeout_ns;
#else
    uint32_t key = fifo.active[i];
    uint32_t op_id = key / EXPERIMENTAL_MOE_EXPERTS;
    auto const &op = operators[op_id];
#if !NAS_FIFO_SKIP_STATIC_CHECKS
    if (!op.present || !op.enabled) continue;
#endif
    // With the opt-in above, reuse ready()'s validation: disabled operations
    // take its ordinary-worker path and present is checked before FIFO append.
    // These operator fields are immutable during a run. Dynamic cohort gating
    // below still applies; it is not a static validity check.
#ifdef NAS_INVESTIGATE_ALL_ROWS
    if ((eligible_ops & (1u << op_id)) == 0u) continue;
#endif
    uint32_t count = fifo.count[key];
#if NAS_FIFO_CACHE_OLDEST
    uint64_t oldest = fifo.oldest[key];
#else
    constexpr uint32_t members_per_slot =
        EXPERIMENTAL_MOE_CHUNK_ROWS * EXPERIMENTAL_MOE_TOP_K;
    uint64_t oldest = pending[fifo.head[key] / members_per_slot].ready_ns;
#endif
    bool threshold = count >= op.threshold;
    bool timeout = now - oldest >= op.timeout_ns;
#endif
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
    ExperimentalMoeFifo &fifo,
    ExperimentalMoePendingSlot const (&pending)[EXPERIMENTAL_MOE_STATE_SLOTS],
    ExperimentalMoeSelection const &selected) {
#if !NAS_FIFO_CACHE_OLDEST
  (void)pending;
#endif
  uint32_t i = selected.active_bin_index;
  if (selected.op_id < 0 || i >= fifo.active_count || selected.count == 0u ||
      selected.count > EXPERIMENTAL_MOE_BATCH_CAPACITY) return false;
  uint32_t key = uint32_t(selected.op_id) * EXPERIMENTAL_MOE_EXPERTS + selected.expert;
#if NAS_FIFO_HOT_RECORDS
  auto &bin = fifo.active[i];
  if (bin.key != key || fifo.active_position[key] != i + 1u ||
      bin.count != selected.group_count || selected.count > bin.count) return false;
#else
  if (fifo.active[i] != key || fifo.count[key] != selected.group_count ||
      selected.count > fifo.count[key]) return false;
#endif
  // Selection already traversed these links. One lookup after the last
  // selected member advances the head past the entire captured prefix.
  uint32_t j = selected.count - 1u;
  uint32_t last = (selected.slots[j] * EXPERIMENTAL_MOE_CHUNK_ROWS +
                   selected.tokens[j]) * EXPERIMENTAL_MOE_TOP_K + selected.branches[j];
  fifo.head[key] = fifo.next[last];
#if NAS_FIFO_HOT_RECORDS
  bin.count -= selected.count;
  if (bin.count == 0u) {
    fifo.tail[key] = UINT32_MAX;
    uint32_t last_index = --fifo.active_count;
    if (i != last_index) {
      fifo.active[i] = fifo.active[last_index];
      fifo.active_position[fifo.active[i].key] = i + 1u;
    }
    fifo.active_position[key] = 0u;
  } else {
    constexpr uint32_t members_per_slot =
        EXPERIMENTAL_MOE_CHUNK_ROWS * EXPERIMENTAL_MOE_TOP_K;
    bin.oldest = pending[fifo.head[key] / members_per_slot].ready_ns;
  }
#else
  fifo.count[key] -= selected.count;
  if (fifo.count[key] == 0u) {
    fifo.tail[key] = UINT32_MAX;
    fifo.active[i] = fifo.active[--fifo.active_count];
#if NAS_FIFO_CACHE_OLDEST
  } else {
    // Only an oversized bin leaves a new head. Read its timestamp ONCE here,
    // rather than chasing the head on every subsequent selection attempt.
    constexpr uint32_t members_per_slot =
        EXPERIMENTAL_MOE_CHUNK_ROWS * EXPERIMENTAL_MOE_TOP_K;
    fifo.oldest[key] = pending[fifo.head[key] / members_per_slot].ready_ns;
#endif
  }
#endif
  return true;
}
}} // namespace mirage::runtime
#undef NAS_MOE_FIFO_INLINE
