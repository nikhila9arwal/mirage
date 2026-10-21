#pragma once

// Optional population measurements for the finite offline experiment. Only the
// scheduler's thread 0 touches this object. Workers and scheduling decisions do
// not read it. The host reads the buffer after the complete invocation finishes.
//
// A row moves through four measured populations:
//   pending W13 -> dispatched/unretired W13 -> ordinary work ->
//   pending W2  -> dispatched/unretired W2  -> ordinary work / context save.
// The unretired population includes queued, executing, and completed batches
// whose completion the scheduler has not finished handling.
// This recorder describes the demo's fixed FFN chain: SiLU separates W13/W2,
// and slot restart follows W2 retirement. It is not a generic overlapping-DAG
// accounting scheme. "Other active" is the residual ownership population; an
// ordinary consumer may already execute while its producer remains unretired.
#ifdef NAS_MOE_OCCUPANCY

#ifndef NAS_MOE_OCCUPANCY_INTERVAL_NS
#define NAS_MOE_OCCUPANCY_INTERVAL_NS 1000000ULL
#endif

__device__ __forceinline__ unsigned long long nas_moe_occupancy_clock() {
  unsigned long long now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now) :: "memory");
  return now;
}

struct NasMoeOccupancy {
  using Counter = unsigned long long;
  static constexpr Counter WORDS = 32;
  static constexpr Counter SAVED_STRIDE = 128;

  enum SnapshotKind : Counter {
    INITIAL = 0, BOOTSTRAP = 1, PROGRESS = 2, TIME = 3, FINAL = 4,
    ADMISSION_CLOSED = 5,
  };

  Counter *buffer;
  Counter slots, contexts, start_ns, last_ns, last_snapshot_ns;
  Counter records = 0, saved = 0, admitted = 0, active = 0;
  Counter pending[2] = {}, in_flight[2] = {};
  Counter pending_area[2] = {}, in_flight_area[2] = {};
  Counter active_area = 0, absolute_difference_area = 0;
  Counter squared_difference_area = 0;
  Counter batches[2] = {}, rows[2] = {}, full_batches[2] = {};
  Counter timeout_batches[2] = {}, threshold_batches[2] = {};
  Counter ready_rows[2] = {}, retired_rows[2] = {}, retired_batches[2] = {};

  __device__ __forceinline__ NasMoeOccupancy(int num_slots, int num_contexts)
      : slots(num_slots), contexts(num_contexts) {
    static_assert(NAS_MOE_OCCUPANCY_CAPACITY > 0,
                  "Occupancy output needs at least one record");
    require_request_slot_invariant(num_slots > 0 && num_contexts >= 0);
    buffer = reinterpret_cast<Counter *>(NAS_MOE_OCCUPANCY_ADDRESS);
    buffer[6] = 0u; // Clear publication before this invocation writes records.
    start_ns = last_ns = last_snapshot_ns = nas_moe_occupancy_clock();
    snapshot(INITIAL);
  }

  // Integrate the OLD populations up to the transition's timestamp. Integrals
  // are exact for the explicitly placed observation hooks, including long
  // intervals with no scheduler progress. No periodic polling is necessary.
  __device__ __forceinline__ void integrate(Counter now) {
    require_request_slot_invariant(now >= last_ns);
    Counter elapsed = now - last_ns;
    for (int stage = 0; stage < 2; stage++) {
      pending_area[stage] += pending[stage] * elapsed;
      in_flight_area[stage] += in_flight[stage] * elapsed;
    }
    active_area += active * elapsed;
    Counter difference = pending[0] >= pending[1]
                             ? pending[0] - pending[1]
                             : pending[1] - pending[0];
    absolute_difference_area += difference * elapsed;
    squared_difference_area += difference * difference * elapsed;
    last_ns = now;
  }

  __device__ __forceinline__ void validate() const {
    require_request_slot_invariant(saved <= admitted && admitted <= contexts &&
        active <= slots && active == admitted - saved);
    Counter tracked = 0;
    for (int stage = 0; stage < 2; stage++) {
      require_request_slot_invariant(ready_rows[stage] >= rows[stage] &&
          rows[stage] >= retired_rows[stage] &&
          pending[stage] == ready_rows[stage] - rows[stage] &&
          in_flight[stage] == rows[stage] - retired_rows[stage]);
      tracked += pending[stage] + in_flight[stage];
    }
    // All other active slots are in ordinary tasks or waiting for their slot
    // completion/control handling. No context can occupy two measured stages.
    require_request_slot_invariant(tracked <= active);
  }

  __device__ __forceinline__ void snapshot(SnapshotKind kind) {
    Counter index = records++;
    last_snapshot_ns = last_ns;
    if (index >= NAS_MOE_OCCUPANCY_CAPACITY) return;
    Counter *record = buffer + (index + 1) * WORDS;
    record[0] = last_ns;
    record[1] = kind;
    record[2] = saved;
    record[3] = admitted;
    record[4] = active;
    record[5] = pending[0];
    record[6] = pending[1];
    record[7] = in_flight[0];
    record[8] = in_flight[1];
    record[9] = pending_area[0];
    record[10] = pending_area[1];
    record[11] = in_flight_area[0];
    record[12] = in_flight_area[1];
    record[13] = active_area;
    record[14] = absolute_difference_area;
    record[15] = squared_difference_area;
    record[16] = batches[0];
    record[17] = batches[1];
    record[18] = rows[0];
    record[19] = rows[1];
    record[20] = full_batches[0];
    record[21] = full_batches[1];
    record[22] = timeout_batches[0];
    record[23] = timeout_batches[1];
    record[24] = threshold_batches[0];
    record[25] = threshold_batches[1];
    record[26] = ready_rows[0];
    record[27] = ready_rows[1];
    record[28] = retired_rows[0];
    record[29] = retired_rows[1];
    record[30] = retired_batches[0];
    record[31] = retired_batches[1];
  }

  __device__ __forceinline__ void maybe_time_snapshot() {
    // Opportunistic time snapshots use an existing transition's timestamp.
    // Their spacing may exceed the interval during idle periods; the integrals
    // still cover the entire interval. Set the interval to zero to disable.
    if (NAS_MOE_OCCUPANCY_INTERVAL_NS != 0u &&
        last_ns - last_snapshot_ns >= NAS_MOE_OCCUPANCY_INTERVAL_NS) {
      snapshot(TIME);
    }
  }

  __device__ __forceinline__ void ready(uint32_t stage, Counter ready_ns) {
    require_request_slot_invariant(stage < 2u);
    integrate(ready_ns); // Reuse the timestamp already stored in pending[].
    pending[stage]++;
    ready_rows[stage]++;
    validate();
    maybe_time_snapshot();
  }

  __device__ __forceinline__ void dispatch(
      uint32_t stage, uint32_t count, uint32_t reason) {
    require_request_slot_invariant(stage < 2u && count > 0u &&
        count <= EXPERIMENTAL_MOE_BATCH_CAPACITY && pending[stage] >= count &&
        (reason == 1u || reason == 2u));
    integrate(nas_moe_occupancy_clock());
    // Observation boundary: all selected pending flags have just been cleared.
    // Treat this short clear loop as one batch transition, before publication.
    pending[stage] -= count;
    in_flight[stage] += count;
    batches[stage]++;
    rows[stage] += count;
    full_batches[stage] += count == EXPERIMENTAL_MOE_BATCH_CAPACITY;
    threshold_batches[stage] += reason == 1u;
    timeout_batches[stage] += reason == 2u;
    validate();
    maybe_time_snapshot();
  }

  __device__ __forceinline__ void retire(uint32_t stage, uint32_t count) {
    require_request_slot_invariant(stage < 2u && count > 0u &&
                                   in_flight[stage] >= count);
    integrate(nas_moe_occupancy_clock());
    // All logical members have been released and the descriptor is now free.
    in_flight[stage] -= count;
    retired_rows[stage] += count;
    retired_batches[stage]++;
    validate();
    maybe_time_snapshot();
  }

  __device__ __forceinline__ void control(
      uint32_t new_active, uint32_t new_admitted, bool bootstrap) {
    integrate(nas_moe_occupancy_clock());
    bool admission_just_closed = admitted < contexts && new_admitted == contexts;
    // Called only for bootstrap or a slot completion. Stores that save the old
    // result and admit its replacement have finished when this hook executes.
    if (!bootstrap) saved++;
    active = new_active;
    admitted = new_admitted;
    validate();
    if (bootstrap) snapshot(BOOTSTRAP);
    else if (admission_just_closed) snapshot(ADMISSION_CLOSED);
    else if (saved % SAVED_STRIDE == 0u) snapshot(PROGRESS);
    else maybe_time_snapshot();
  }

  __device__ __forceinline__ void finish() {
    integrate(nas_moe_occupancy_clock());
    validate();
    require_request_slot_invariant(saved == contexts && admitted == contexts &&
        active == 0u && pending[0] == 0u && pending[1] == 0u &&
        in_flight[0] == 0u && in_flight[1] == 0u);
    for (int stage = 0; stage < 2; stage++) {
      require_request_slot_invariant(ready_rows[stage] == contexts &&
          rows[stage] == contexts && retired_rows[stage] == contexts &&
          batches[stage] == retired_batches[stage] &&
          batches[stage] == threshold_batches[stage] + timeout_batches[stage]);
    }
    snapshot(FINAL);
    buffer[0] = records;
    buffer[1] = records > NAS_MOE_OCCUPANCY_CAPACITY;
    buffer[2] = start_ns;
    buffer[3] = last_ns;
    buffer[4] = NAS_MOE_OCCUPANCY_CAPACITY;
    buffer[5] = 1u; // Schema version: one header row and 32-word snapshots.
    buffer[7] = WORDS;
    buffer[8] = SAVED_STRIDE;
    buffer[9] = NAS_MOE_OCCUPANCY_INTERVAL_NS;
    buffer[10] = slots;
    buffer[11] = contexts;
    for (Counter i = 12; i < WORDS; i++) buffer[i] = 0u;
    __threadfence();
    buffer[6] = 1u; // Publish once; host synchronization is still required.
  }
};

#else

// Keep call sites readable. With instrumentation disabled the empty recorder
// and all of these methods compile away, including their timestamp arguments.
struct NasMoeOccupancy {
  __device__ __forceinline__ NasMoeOccupancy(int, int) {}
  __device__ __forceinline__ void ready(uint32_t, unsigned long long) {}
  __device__ __forceinline__ void dispatch(uint32_t, uint32_t, uint32_t) {}
  __device__ __forceinline__ void retire(uint32_t, uint32_t) {}
  __device__ __forceinline__ void control(uint32_t, uint32_t, bool) {}
  __device__ __forceinline__ void finish() {}
};

#endif
