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

// Diagnostic boundaries only; no timer or record writes in an uninstrumented
// invocation. The caller aggregates them AFTER dispatch/queue publication.
struct ExperimentalMoeSelectionTiming {
  unsigned long long boundary[5] = {};
  uint32_t groups = 0u;
};
NAS_MOE_SELECTION_INLINE void experimental_moe_selection_stamp(
    ExperimentalMoeSelectionTiming *timing, uint32_t boundary) {
#if defined(NAS_SELECTOR_BREAKDOWN) && defined(__CUDA_ARCH__)
  if (timing != nullptr) {
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now) :: "memory");
    timing->boundary[boundary] = now;
  }
#else
  (void)timing;
  (void)boundary;
#endif
}

struct ExperimentalMoeSelection {
  int32_t op_id = -1;
  uint32_t expert = 0, reason = 0, group_count = 0, count = 0;
  uint32_t second_expert = UINT32_MAX; // Optional same-operation companion.
  // FIFO-only: the winner's position in the active-bin list allows O(1)
  // removal when its last members leave. No per-bin reverse index is needed.
  uint32_t active_bin_index = UINT32_MAX;
  uint32_t slots[EXPERIMENTAL_MOE_BATCH_CAPACITY] = {};
  uint32_t tokens[EXPERIMENTAL_MOE_BATCH_CAPACITY] = {};
  uint32_t branches[EXPERIMENTAL_MOE_BATCH_CAPACITY] = {};
};

#ifdef NAS_INVESTIGATE_ALL_ROWS
static_assert(EXPERIMENTAL_MOE_STATE_SLOTS == 64 &&
              EXPERIMENTAL_BATCH_OPERATORS == 8 && EXPERIMENTAL_MOE_LAYERS == 2,
              "All-row gating is restricted to S64 and two layers");
constexpr uint32_t NAS_ALL_ROWS_OPERATOR_MASK = 0xffu;

// Only the scheduler owns open_ops. Clearing selected rows must not close it.
NAS_MOE_SELECTION_INLINE bool experimental_moe_latch_all_rows(
    ExperimentalMoePendingSlot const (&pending)[EXPERIMENTAL_MOE_STATE_SLOTS],
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
      auto const &row = pending[slot];
      if ((row.pending_mask & 1u) == 0u || row.op_id != op_id) {
        all_ready = false;
        continue;
      }
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

// A slot owns only one active operation, so each flat token/branch index has
// one possible row. Returning a plain value keeps selection independent of
// the compact storage layout.
NAS_MOE_SELECTION_INLINE ExperimentalMoePending experimental_moe_row(
    ExperimentalMoePendingSlot const (&pending)[EXPERIMENTAL_MOE_STATE_SLOTS],
    uint32_t index) {
  constexpr uint32_t members_per_slot =
      EXPERIMENTAL_MOE_CHUNK_ROWS * EXPERIMENTAL_MOE_TOP_K;
  auto const &slot = pending[index / members_per_slot];
  uint32_t member = index % members_per_slot;
  if ((slot.pending_mask & (1u << member)) == 0u) return {};
  uint32_t expert = slot.experts[member / EXPERIMENTAL_MOE_TOP_K]
                                [member % EXPERIMENTAL_MOE_TOP_K];
  return {1u, slot.data_id, slot.epoch, expert, slot.op_id, slot.ready_ns};
}

// Companion and member scans retain their explicit kind filter. Native
// operator IDs are layer * KINDS + kind (established by host initialization).
NAS_MOE_SELECTION_INLINE ExperimentalMoePending experimental_moe_row(
    ExperimentalMoePendingSlot const (&pending)[EXPERIMENTAL_MOE_STATE_SLOTS],
    uint32_t kind, uint32_t index) {
  auto row = experimental_moe_row(pending, index);
  if (row.op_id % EXPERIMENTAL_BATCH_KINDS != kind) return {};
  return row;
}

// One caller-owned buffer is reused for the primary and optional companion
// scans. Its contents never carry information into the next scan.
NAS_MOE_SELECTION_INLINE void experimental_moe_clear_seen(
    bool (&seen)[EXPERIMENTAL_MOE_PENDING_ROWS]) {
  for (uint32_t row = 0; row < EXPERIMENTAL_MOE_PENDING_ROWS; ++row)
    seen[row] = false;
}

NAS_MOE_SELECTION_INLINE ExperimentalMoeSelection experimental_moe_select(
    ExperimentalMoePendingSlot const (&pending)[EXPERIMENTAL_MOE_STATE_SLOTS],
    ExperimentalBatchOperator const (&operators)[EXPERIMENTAL_BATCH_OPERATORS],
    bool (&seen)[EXPERIMENTAL_MOE_PENDING_ROWS],
    unsigned long long now, uint32_t eligible_ops = ~0u,
    bool early_coalesce = false, uint32_t in_flight = 0u,
    bool deadline_coalesce = false,
    ExperimentalMoeSelectionTiming *timing = nullptr) {
  experimental_moe_selection_stamp(timing, 0);
#ifndef NAS_INVESTIGATE_ALL_ROWS
  // Only the legacy two-layer cohort experiment uses an operation bit mask.
  // Normal async eligibility comes from each bin's threshold or timeout below.
  (void)eligible_ops;
#endif
  ExperimentalMoeSelection selected;
  unsigned long long best_time = ~0ull;
  uint32_t best_row = EXPERIMENTAL_MOE_PENDING_ROWS;
  // Keep the usual eligible selection authoritative. This candidate is used
  // only when that selection is empty AND fewer than two descriptors are live.
  bool allow_early = early_coalesce && in_flight < 2u;
  ExperimentalMoeSelection early;
  unsigned long long early_time = ~0ull;

  // One operation owns each slot: an index cannot also belong to another
  // kind. Count each exact (op_id, expert) bin once, even before it is eligible.
  // The explicit tie rules below preserve the old kind-by-kind result.
  experimental_moe_clear_seen(seen);
  experimental_moe_selection_stamp(timing, 1);
  for (uint32_t index = 0; index < EXPERIMENTAL_MOE_PENDING_ROWS; index++) {
    auto const candidate = experimental_moe_row(pending, index);
    if (!candidate.pending || seen[index]) continue;
    uint32_t op_id = candidate.op_id;
    auto const &op = operators[op_id];
    if (!op.present || !op.enabled) continue;
#ifdef NAS_INVESTIGATE_ALL_ROWS
    // This experiment is explicitly limited to eight operations above.
    // Do not shift a 32-bit mask by arbitrary full-model operation IDs.
    if ((eligible_ops & (1u << op_id)) == 0u) continue;
#endif
    if (timing != nullptr) ++timing->groups;
    uint32_t count = 0u;
    unsigned long long oldest = ~0ull;
    for (uint32_t s = 0; s < EXPERIMENTAL_MOE_PENDING_ROWS; s++) {
      auto const row = experimental_moe_row(pending, s);
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
    // Retain the original tie order explicitly: lower op_id, then the
    // bin's first flat row. It does not depend on which kind is visited first.
    bool earlier_tie = oldest == best_time && selected.op_id >= 0 &&
        (op_id < static_cast<uint32_t>(selected.op_id) ||
         (op_id == static_cast<uint32_t>(selected.op_id) && index < best_row));
    if ((threshold_reached || timeout_reached) &&
        (oldest < best_time || earlier_tie)) {
      best_time = oldest;
      best_row = index;
      selected.op_id = op_id;
      selected.expert = candidate.expert;
      selected.reason = threshold_reached ? 1u : 2u;
      selected.group_count = count;
    }
  }
  if (selected.op_id < 0 && early.op_id >= 0) {
    selected = early;
    // Find one other whole bin for the SAME concrete operation/layer. Never
    // take a small fragment merely to fill the remaining descriptor capacity.
    uint32_t remaining = EXPERIMENTAL_MOE_BATCH_CAPACITY - selected.group_count;
    uint32_t second_count = 0u;
    unsigned long long second_time = ~0ull;
    experimental_moe_clear_seen(seen);
    uint32_t kind = operators[selected.op_id].kind;
    for (uint32_t index = 0; index < EXPERIMENTAL_MOE_PENDING_ROWS; ++index) {
      auto const candidate = experimental_moe_row(pending, kind, index);
      if (!candidate.pending || seen[index] ||
          candidate.op_id != static_cast<uint32_t>(selected.op_id) ||
          candidate.expert == selected.expert) continue;
      uint32_t count = 0u;
      unsigned long long oldest = ~0ull;
      for (uint32_t s = 0; s < EXPERIMENTAL_MOE_PENDING_ROWS; ++s) {
        auto const row = experimental_moe_row(pending, kind, s);
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
  if (selected.op_id < 0) {
    experimental_moe_selection_stamp(timing, 2);
    if (timing != nullptr)
      timing->boundary[3] = timing->boundary[4] = timing->boundary[2];
    return selected;
  }

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
    experimental_moe_clear_seen(seen);
    uint32_t kind = selected_op.kind;
    for (uint32_t index = 0; index < EXPERIMENTAL_MOE_PENDING_ROWS; ++index) {
      auto const candidate = experimental_moe_row(pending, kind, index);
      if (!candidate.pending || seen[index] ||
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
      for (uint32_t s = 0; s < EXPERIMENTAL_MOE_PENDING_ROWS; ++s) {
        auto const row = experimental_moe_row(pending, kind, s);
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

  experimental_moe_selection_stamp(timing, 2);
  selected.count = selected.group_count < EXPERIMENTAL_MOE_BATCH_CAPACITY
                       ? selected.group_count : EXPERIMENTAL_MOE_BATCH_CAPACITY;
  unsigned long long previous_time = 0u;
  uint32_t previous_row = 0u;
  // Pick the next oldest (ready_ns,slot,token,branch). At most C*S*P*R comparisons;
  // unselected pending records and their original ages remain untouched.
  for (uint32_t j = 0; j < selected.count; j++) {
    unsigned long long next_time = ~0ull;
    uint32_t next_row = EXPERIMENTAL_MOE_PENDING_ROWS;
    for (uint32_t index = 0; index < EXPERIMENTAL_MOE_PENDING_ROWS; index++) {
      auto const row = experimental_moe_row(pending, selected_op.kind, index);
      if (!row.pending || row.op_id != static_cast<uint32_t>(selected.op_id) ||
          (row.expert != selected.expert && row.expert != selected.second_expert)) continue;
      if (j != 0 && (row.ready_ns < previous_time ||
          (row.ready_ns == previous_time && index <= previous_row))) continue;
      if (row.ready_ns < next_time ||
          (row.ready_ns == next_time && index < next_row)) {
        next_time = row.ready_ns;
        next_row = index;
      }
    }
    uint32_t token_row = next_row / EXPERIMENTAL_MOE_TOP_K;
    selected.slots[j] = token_row / EXPERIMENTAL_MOE_CHUNK_ROWS;
    selected.tokens[j] = token_row % EXPERIMENTAL_MOE_CHUNK_ROWS;
    selected.branches[j] = next_row % EXPERIMENTAL_MOE_TOP_K;
    previous_time = next_time;
    previous_row = next_row;
  }
  experimental_moe_selection_stamp(timing, 3);
  if (timing != nullptr) timing->boundary[4] = timing->boundary[3];
  return selected;
}

// Single-expert policy: build every bin's statistics in one flat pass, choose
// the oldest eligible bin, then collect its first C members in one more pass.
// It intentionally does not sort the winning bin's members by readiness age.
// Unselected members keep their original readiness timestamps and pending bits.
NAS_MOE_SELECTION_INLINE ExperimentalMoeSelection experimental_moe_select_histogram(
    ExperimentalMoePendingSlot const (&pending)[EXPERIMENTAL_MOE_STATE_SLOTS],
    ExperimentalBatchOperator const (&operators)[EXPERIMENTAL_BATCH_OPERATORS],
    ExperimentalMoeSelectionScratch &scratch, unsigned long long now,
    uint32_t eligible_ops = ~0u,
    ExperimentalMoeSelectionTiming *timing = nullptr) {
#ifndef NAS_INVESTIGATE_ALL_ROWS
  (void)eligible_ops;
#endif
  experimental_moe_selection_stamp(timing, 0);
  ExperimentalMoeSelection selected;
  uint32_t touched_count = 0u;

  // Phase 1: count every (concrete operation/layer, expert) bin once per row.
  // count[] starts at zero. A first touch records the bin, in flat row order,
  // and initializes oldest[]; stale timestamps need no full-array clearing.
  for (uint32_t index = 0; index < EXPERIMENTAL_MOE_PENDING_ROWS; ++index) {
    auto row = experimental_moe_row(pending, index);
    if (!row.pending) continue;
    auto const &op = operators[row.op_id];
    if (!op.present || !op.enabled) continue;
#ifdef NAS_INVESTIGATE_ALL_ROWS
    if ((eligible_ops & (1u << row.op_id)) == 0u) continue;
#endif
    uint32_t key = row.op_id * EXPERIMENTAL_MOE_EXPERTS + row.expert;
    if (scratch.count[key] == 0u) {
      scratch.touched[touched_count++] = key;
      scratch.oldest[key] = row.ready_ns;
    } else if (row.ready_ns < scratch.oldest[key]) {
      scratch.oldest[key] = row.ready_ns;
    }
    ++scratch.count[key];
  }
  experimental_moe_selection_stamp(timing, 1);

  // Phase 2: inspect only touched bins. Ties retain lower op_id, then the bin
  // whose FIRST flat member appeared earlier, exactly as in the old selector.
  // touched[] already supplies that final tie order; no extra index is needed.
  uint64_t best_time = UINT64_MAX;
  for (uint32_t i = 0; i < touched_count; ++i) {
    uint32_t key = scratch.touched[i];
    uint32_t op_id = key / EXPERIMENTAL_MOE_EXPERTS;
    auto const &op = operators[op_id];
    uint32_t count = scratch.count[key];
    uint64_t oldest = scratch.oldest[key];
    bool threshold_reached = count >= op.threshold;
    bool timeout_reached = now - oldest >= op.timeout_ns;
    bool earlier_tie = oldest == best_time && selected.op_id >= 0 &&
        op_id < static_cast<uint32_t>(selected.op_id);
    if ((threshold_reached || timeout_reached) &&
        (oldest < best_time || earlier_tie)) {
      best_time = oldest;
      selected.op_id = op_id;
      selected.expert = key % EXPERIMENTAL_MOE_EXPERTS;
      selected.group_count = count; // Uncapped: permits the >C diagnostic.
      selected.reason = threshold_reached ? 1u : 2u;
    }
  }
  experimental_moe_selection_stamp(timing, 2);

  // Phase 3: select the first C matching positions, without rescanning once
  // per member. Only dispatch later clears these bits; selection is read-only.
  if (selected.op_id >= 0) {
    uint32_t target = selected.group_count < EXPERIMENTAL_MOE_BATCH_CAPACITY
                          ? selected.group_count : EXPERIMENTAL_MOE_BATCH_CAPACITY;
    for (uint32_t index = 0; index < EXPERIMENTAL_MOE_PENDING_ROWS &&
         selected.count < target; ++index) {
      auto row = experimental_moe_row(pending, index);
      if (!row.pending || row.op_id != static_cast<uint32_t>(selected.op_id) ||
          row.expert != selected.expert) continue;
      uint32_t j = selected.count++;
      uint32_t token_row = index / EXPERIMENTAL_MOE_TOP_K;
      selected.slots[j] = token_row / EXPERIMENTAL_MOE_CHUNK_ROWS;
      selected.tokens[j] = token_row % EXPERIMENTAL_MOE_CHUNK_ROWS;
      selected.branches[j] = index % EXPERIMENTAL_MOE_TOP_K;
    }
  }
  experimental_moe_selection_stamp(timing, 3);

  // Phase 4: reset ONLY touched counts, including when no bin is eligible.
  // The next call therefore starts empty without clearing L*K*E entries.
  for (uint32_t i = 0; i < touched_count; ++i)
    scratch.count[scratch.touched[i]] = 0u;
  experimental_moe_selection_stamp(timing, 4);
  if (timing != nullptr) timing->groups = touched_count;
  return selected;
}

}} // namespace mirage::runtime
#undef NAS_MOE_SELECTION_INLINE
