#pragma once
#include "experimental_moe_types.h"

#if defined(NAS_INVESTIGATE_ALL_ROWS_COUNTER) && !defined(NAS_INVESTIGATE_ALL_ROWS)
#error "Readiness counters require the all-row cohort policy"
#endif

// Keep selection independent of queues and device state. The GPU scheduler and
// the small deterministic CPU/GPU tests exercise this same policy.
#ifdef __CUDACC__
#define NAS_MOE_SELECTION_INLINE __host__ __device__ __forceinline__
#else
#define NAS_MOE_SELECTION_INLINE inline
#endif

namespace mirage { namespace runtime {

struct ExperimentalMoeSelection {
  int32_t op_id = -1;
  uint32_t expert = 0, reason = 0, group_count = 0, count = 0;
  uint32_t second_expert = UINT32_MAX; // Optional same-operation companion.
  uint32_t slots[EXPERIMENTAL_MOE_BATCH_CAPACITY] = {};
};

#ifdef NAS_INVESTIGATE_ALL_ROWS
static_assert(EXPERIMENTAL_MOE_STATE_SLOTS == 64 &&
              EXPERIMENTAL_BATCH_OPERATORS == 8 && EXPERIMENTAL_MOE_LAYERS == 2,
              "All-row gating is restricted to S64 and two layers");
constexpr uint32_t NAS_ALL_ROWS_OPERATOR_MASK = 0xffu;

// Only the scheduler owns open_ops. Clearing selected rows must not close it.
NAS_MOE_SELECTION_INLINE bool experimental_moe_latch_all_rows(
    ExperimentalMoePending const (&pending)[EXPERIMENTAL_BATCH_KINDS]
                                          [EXPERIMENTAL_MOE_STATE_SLOTS],
    ExperimentalBatchOperator const (&operators)[EXPERIMENTAL_BATCH_OPERATORS],
    uint32_t epoch, uint32_t &open_ops) {
  if (epoch == 0u || (open_ops & ~NAS_ALL_ROWS_OPERATOR_MASK) != 0u) return false;
  for (uint32_t op_id = 0; op_id < EXPERIMENTAL_BATCH_OPERATORS; ++op_id) {
    auto const &op = operators[op_id];
    if (!op.present || !op.enabled || op.kind != op_id % EXPERIMENTAL_BATCH_KINDS ||
        op.layer_id != op_id / EXPERIMENTAL_BATCH_KINDS) return false;
    uint32_t bit = 1u << op_id;
    if ((open_ops & bit) != 0u) continue;
    bool all_ready = true;
    for (uint32_t slot = 0; slot < EXPERIMENTAL_MOE_STATE_SLOTS; ++slot) {
      auto const &row = pending[op.kind][slot];
      if (!row.pending || row.op_id != op_id) { all_ready = false; continue; }
      if (row.epoch != epoch) return false;
    }
    if (all_ready) open_ops |= bit;
  }
  return true;
}

#ifdef NAS_INVESTIGATE_ALL_ROWS_COUNTER
// A closed operation cannot dispatch, so its cumulative arrivals equal its
// pending rows until it opens. Keep the latch after that operation starts draining.
NAS_MOE_SELECTION_INLINE bool experimental_moe_latch_all_rows(
    uint32_t const (&ready_count)[EXPERIMENTAL_BATCH_OPERATORS],
    ExperimentalBatchOperator const (&operators)[EXPERIMENTAL_BATCH_OPERATORS],
    uint32_t epoch, uint32_t &open_ops) {
  if (epoch == 0u || (open_ops & ~NAS_ALL_ROWS_OPERATOR_MASK) != 0u) return false;
  for (uint32_t op_id = 0; op_id < EXPERIMENTAL_BATCH_OPERATORS; ++op_id) {
    auto const &op = operators[op_id];
    if (!op.present || !op.enabled || op.kind != op_id % EXPERIMENTAL_BATCH_KINDS ||
        op.layer_id != op_id / EXPERIMENTAL_BATCH_KINDS ||
        ready_count[op_id] > EXPERIMENTAL_MOE_STATE_SLOTS) return false;
    if (ready_count[op_id] == EXPERIMENTAL_MOE_STATE_SLOTS) open_ops |= 1u << op_id;
  }
  return true;
}
#endif
#endif

NAS_MOE_SELECTION_INLINE ExperimentalMoeSelection experimental_moe_select(
    ExperimentalMoePending const (&pending)[EXPERIMENTAL_BATCH_KINDS]
                                          [EXPERIMENTAL_MOE_STATE_SLOTS],
    ExperimentalBatchOperator const (&operators)[EXPERIMENTAL_BATCH_OPERATORS],
    unsigned long long now, uint32_t eligible_ops = ~0u,
    bool early_coalesce = false, uint32_t in_flight = 0u,
    bool deadline_coalesce = false) {
  ExperimentalMoeSelection selected;
  unsigned long long best_time = ~0ull;
  uint32_t best_slot = EXPERIMENTAL_MOE_STATE_SLOTS;
  // Keep the usual eligible selection authoritative. This candidate is used
  // only when that selection is empty AND fewer than two descriptors are live.
  bool allow_early = early_coalesce && in_flight < 2u;
  ExperimentalMoeSelection early;
  unsigned long long early_time = ~0ull;

  // Visit the fixed kind/slot tables directly. Each pending row names its
  // concrete operation, so layers with no pending work need no separate scan.
  for (uint32_t kind = 0; kind < EXPERIMENTAL_BATCH_KINDS; kind++) {
    // Scratch for this selection call and kind, not persistent readiness state.
    // Counting a bin marks all its rows, even if it is not eligible yet.
    bool seen[EXPERIMENTAL_MOE_STATE_SLOTS] = {};
    for (uint32_t slot = 0; slot < EXPERIMENTAL_MOE_STATE_SLOTS; slot++) {
      auto const &candidate = pending[kind][slot];
      if (!candidate.pending || seen[slot]) continue;
      uint32_t op_id = candidate.op_id;
      auto const &op = operators[op_id];
      if (!op.present || !op.enabled || (eligible_ops & (1u << op_id)) == 0u) continue;
      uint32_t count = 0u;
      unsigned long long oldest = ~0ull;
      for (uint32_t s = 0; s < EXPERIMENTAL_MOE_STATE_SLOTS; s++) {
        auto const &row = pending[kind][s];
        if (row.pending && row.op_id == op_id && row.expert == candidate.expert) {
          seen[s] = true;
          count++;
          if (row.ready_ns < oldest) oldest = row.ready_ns;
        }
      }
      if (allow_early && op.kind < EXPERIMENTAL_BATCH_QKV &&
          count <= EXPERIMENTAL_MOE_BATCH_CAPACITY) {
        // Largest whole bin, then oldest row, then stable operation/expert IDs.
        bool better = early.op_id < 0 || count > early.group_count ||
            (count == early.group_count && (oldest < early_time ||
             (oldest == early_time && (op_id < static_cast<uint32_t>(early.op_id) ||
              (op_id == static_cast<uint32_t>(early.op_id) && candidate.expert < early.expert)))));
        if (better) {
          early.op_id = op_id;
          early.expert = candidate.expert;
          early.group_count = count;
          early.reason = 3u; // Early progress, neither threshold nor timeout.
          early_time = oldest;
        }
      }
      bool threshold_reached = count >= op.threshold;
      bool timeout_reached = now - oldest >= op.timeout_ns;
      // Traversal is now kind/slot order. Explicitly retain the old tie order:
      // lower op_id, then the bin's first slot (not its oldest member's slot).
      bool earlier_tie = oldest == best_time && selected.op_id >= 0 &&
          (op_id < static_cast<uint32_t>(selected.op_id) ||
           (op_id == static_cast<uint32_t>(selected.op_id) && slot < best_slot));
      if ((threshold_reached || timeout_reached) &&
          (oldest < best_time || earlier_tie)) {
        best_time = oldest;
        best_slot = slot;
        selected.op_id = op_id;
        selected.expert = candidate.expert;
        selected.reason = threshold_reached ? 1u : 2u;
        selected.group_count = count;
      }
      // Keep scanning: this kind may contain other layers' dense operations.
      // Their op_ids differ, so only this bin's members were marked seen.
    }
  }
  if (selected.op_id < 0 && early.op_id >= 0) {
    selected = early;
    // Find one other whole bin for the SAME concrete operation/layer. Never
    // take a small fragment merely to fill the remaining descriptor capacity.
    uint32_t remaining = EXPERIMENTAL_MOE_BATCH_CAPACITY - selected.group_count;
    uint32_t second_count = 0u;
    unsigned long long second_time = ~0ull;
    bool seen[EXPERIMENTAL_MOE_STATE_SLOTS] = {};
    auto const &rows = pending[operators[selected.op_id].kind];
    for (uint32_t slot = 0; slot < EXPERIMENTAL_MOE_STATE_SLOTS; ++slot) {
      auto const &candidate = rows[slot];
      if (!candidate.pending || seen[slot] ||
          candidate.op_id != static_cast<uint32_t>(selected.op_id) ||
          candidate.expert == selected.expert) continue;
      uint32_t count = 0u;
      unsigned long long oldest = ~0ull;
      for (uint32_t s = 0; s < EXPERIMENTAL_MOE_STATE_SLOTS; ++s) {
        auto const &row = rows[s];
        if (row.pending && row.op_id == candidate.op_id && row.expert == candidate.expert) {
          seen[s] = true;
          ++count;
          if (row.ready_ns < oldest) oldest = row.ready_ns;
        }
      }
      if (count <= remaining && (count > second_count ||
          (count == second_count && (oldest < second_time ||
           (oldest == second_time && candidate.expert < selected.second_expert))))) {
        selected.second_expert = candidate.expert;
        second_count = count;
        second_time = oldest;
      }
    }
    selected.group_count += second_count;
  }
  if (selected.op_id < 0) return selected;

  // Choose the normal primary first. Only a timeout-selected MoE bin with
  // unused capacity searches for a companion; its dispatch is never delayed.
  // This path is independent of the early fallback's in_flight < 2 condition.
  auto const &selected_op = operators[selected.op_id];
  if (deadline_coalesce && selected.reason == 2u &&
      selected_op.kind < EXPERIMENTAL_BATCH_QKV &&
      selected.group_count < EXPERIMENTAL_MOE_BATCH_CAPACITY) {
    constexpr unsigned long long small_bin_window_ns = 10000000ull; // 10 ms.
    constexpr unsigned long long large_bin_window_ns =  5000000ull; //  5 ms.
    uint32_t remaining = EXPERIMENTAL_MOE_BATCH_CAPACITY - selected.group_count;
    uint32_t second_count = 0u;
    unsigned long long second_time = ~0ull;
    bool seen[EXPERIMENTAL_MOE_STATE_SLOTS] = {};
    auto const &rows = pending[selected_op.kind];
    for (uint32_t slot = 0; slot < EXPERIMENTAL_MOE_STATE_SLOTS; ++slot) {
      auto const &candidate = rows[slot];
      if (!candidate.pending || seen[slot] ||
          candidate.op_id != static_cast<uint32_t>(selected.op_id) ||
          candidate.expert == selected.expert) continue;

      // A young row cannot establish that its bin is near deadline. Skip this
      // row without marking it seen: an older member later in the table may
      // qualify the bin. Subtraction avoids overflow and handles short timeouts.
      unsigned long long age = now - candidate.ready_ns;
      if (age < selected_op.timeout_ns &&
          selected_op.timeout_ns - age > small_bin_window_ns) continue;

      uint32_t count = 0u;
      unsigned long long oldest = ~0ull;
      for (uint32_t s = 0; s < EXPERIMENTAL_MOE_STATE_SLOTS; ++s) {
        auto const &row = rows[s];
        if (row.pending && row.op_id == candidate.op_id &&
            row.expert == candidate.expert) {
          seen[s] = true;
          ++count;
          if (row.ready_ns < oldest) oldest = row.ready_ns;
        }
      }
      // Take a whole bin. Tiny bins may leave up to 10 ms early; bins with
      // four or more rows may leave up to 5 ms early. Expired bins also qualify.
      unsigned long long window = count <= 3u ? small_bin_window_ns
                                             : large_bin_window_ns;
      age = now - oldest;
      if (count > remaining ||
          (age < selected_op.timeout_ns &&
           selected_op.timeout_ns - age > window)) continue;

      // Same operation means the same timeout: oldest readiness is earliest
      // deadline. Break ties by more rows, then lower expert ID.
      if (oldest < second_time ||
          (oldest == second_time &&
           (count > second_count ||
            (count == second_count && candidate.expert < selected.second_expert)))) {
        selected.second_expert = candidate.expert;
        second_count = count;
        second_time = oldest;
      }
    }
    selected.group_count += second_count;
  }

  selected.count = selected.group_count < EXPERIMENTAL_MOE_BATCH_CAPACITY
                       ? selected.group_count : EXPERIMENTAL_MOE_BATCH_CAPACITY;
  unsigned long long previous_time = 0u;
  uint32_t previous_slot = 0u;
  // Pick the next oldest (ready_ns,slot) each time. At most C*S comparisons;
  // unselected pending records and their original ages remain untouched.
  for (uint32_t j = 0; j < selected.count; j++) {
    unsigned long long next_time = ~0ull;
    uint32_t next_slot = EXPERIMENTAL_MOE_STATE_SLOTS;
    for (uint32_t slot = 0; slot < EXPERIMENTAL_MOE_STATE_SLOTS; slot++) {
      auto const &row = pending[operators[selected.op_id].kind][slot];
      if (!row.pending || row.op_id != static_cast<uint32_t>(selected.op_id) ||
          (row.expert != selected.expert && row.expert != selected.second_expert)) continue;
      if (j != 0 && (row.ready_ns < previous_time ||
          (row.ready_ns == previous_time && slot <= previous_slot))) continue;
      if (row.ready_ns < next_time ||
          (row.ready_ns == next_time && slot < next_slot)) {
        next_time = row.ready_ns;
        next_slot = slot;
      }
    }
    selected.slots[j] = next_slot;
    previous_time = next_time;
    previous_slot = next_slot;
  }
  return selected;
}

}} // namespace mirage::runtime
#undef NAS_MOE_SELECTION_INLINE
