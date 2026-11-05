/* Copyright 2026 CMU
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstdlib>
#include <string>
#include <algorithm>
#include "mirage/type.h"
#include "resident_context_io.cuh"
#ifdef NAS_COMPARE_IO_ADDRESS
#include "bench_io.cuh"
#endif

// Internal to resident_persistent_kernel.cuh. Include after the lifecycle,
// queue publication and reset helpers, which these functions reuse.
//
// First preparation policy: finite passes over fixed synthetic tensors.
// MIRAGE_RESIDENT_SLOT_ITERATIONS=4,2 enables four passes in slot 0 and two
// in slot 1. Alternatively, binding context_io enables a finite input queue.
// ContextIO can also bind fixed sequences: one sequence stays in each slot.
// With neither option, retain shared END. Global token/KV metadata is unused.

constexpr uint32_t SLOT_CONTROL_TAG = 0x80000000u;
constexpr uint32_t SLOT_DISPATCH_TAG = 0x40000000u;
constexpr uint32_t SLOT_CONTROL_ID_MASK = 0x3fffffffu;

__host__ __device__ __forceinline__ bool
slot_restart_enabled(ResidentRuntimeConfig const &config) {
  return config.slot_iteration_limits != nullptr;
}

// Queue storage remains 64 bits: full epoch in high 32 bits, then a marker,
// dispatch/completion bit and a 30-bit slot ID. This is a scheduler control
// message, never an index into all_events. Host validation excludes remote
// events and reserves these low-bit tags from ordinary event indices.
__host__ __device__ __forceinline__ EventId
make_slot_control_item(uint32_t slot, uint32_t epoch, bool dispatch) {
  return (static_cast<EventId>(epoch) << 32) | SLOT_CONTROL_TAG |
         (dispatch ? SLOT_DISPATCH_TAG : 0u) | slot;
}

__host__ __device__ __forceinline__ bool is_slot_control_item(EventId item) {
  return (static_cast<uint32_t>(item) & SLOT_CONTROL_TAG) != 0u;
}

__device__ __forceinline__ void
reset_slot_iteration_state(ResidentRuntimeConfig const &config, uint32_t slot) {
  // All terminals of the old pass have reported. For a validated finite DAG,
  // this implies all its predecessor releases are done; no other slot owns
  // these counters. Queue cursors and other slots' readiness stay intact.
  for (uint32_t i = config.slot_data_offsets[slot];
       i < config.slot_data_offsets[slot + 1]; i++) {
    DataId data = config.slot_data_ids[i];
    config.data_pending_predecessor_counts[data] =
        config.data_initial_predecessor_counts[data];
  }
  if (config.slot_completed_terminal_data_count != nullptr) {
    config.slot_completed_terminal_data_count[slot] = 0u;
  }
}

// Capacity is compiled into tensor/TMA geometry. Only native prefill changes
// the number of valid rows: a full prompt chunk has P, a decode pass has one.
// The controller publishes these values before dispatch and leaves them stable
// until every task of this slot's epoch has completed.
__device__ __forceinline__ int
request_slot_active_rows(ResidentRuntimeConfig const &config, uint32_t slot) {
  if (config.prefill_chunk_capacity == 0) return NAS_PREFILL_CHUNK_ROWS;
  require_request_slot_invariant(slot < static_cast<uint32_t>(config.num_request_slots) &&
      config.slot_chunk_rows != nullptr && config.slot_token_position != nullptr &&
      config.prefill_chunk_capacity == NAS_PREFILL_CHUNK_ROWS);
  int rows = config.slot_chunk_rows[slot];
  int position = config.slot_token_position[slot];
  require_request_slot_invariant((rows == 1 || rows == config.prefill_chunk_capacity) &&
      position >= 0 && position + rows <= 64);
  return rows;
}

__device__ __forceinline__ int
request_slot_active_rows(ResidentRuntimeConfig const &config,
                         TaskDesc const *task, int capacity) {
  if (config.prefill_chunk_capacity == 0) return capacity;
  int slot = task->task_metadata.request_id;
  require_request_slot_invariant(slot >= 0 && capacity == config.prefill_chunk_capacity);
  return request_slot_active_rows(config, static_cast<uint32_t>(slot));
}

// Scheduler 0 calls this only for a free/completed slot. Admission changes
// request metadata and a scalar seed; the attention workers install the prefix
// lazily, before their first attention call. Nothing here resets slot epochs.
__device__ __forceinline__ bool
try_admit_next_request(ResidentRuntimeConfig const &config, uint32_t slot) {
  int request = *config.decode_next_request;
  if (request == config.num_decode_requests) {
    config.decode_slot_request_ids[slot] = -1;
    config.request_ids[slot] = -1;
    return false; // Closed offline queue: no future arrivals are expected.
  }
  require_request_slot_invariant(request >= 0 && request < config.num_decode_requests);
  *config.decode_next_request = request + 1;
  require_request_slot_invariant(config.decode_generated_lengths[request] == 0);
  config.decode_slot_request_ids[slot] = request;
#ifdef NAS_LATENCY_PROBE
  nas_probe::admit(request, slot);
#endif
  config.request_ids[slot] = request * 64;
  if (config.prefill_chunk_capacity > 0) {
    config.slot_token_position[slot] = 0;
    config.slot_chunk_rows[slot] = 0;
  } else {
    static_cast<int64_t *>(config.context_slot_inputs)[slot] =
        static_cast<int64_t const *>(config.context_inputs)[request * 64];
  }
  return true;
}

__device__ __forceinline__ bool
prepare_prefill_request(ResidentRuntimeConfig const &config,
                        uint32_t slot, uint32_t completed_epoch) {
  bool needs_request = completed_epoch == 0u;
  if (!needs_request) {
    int request = config.decode_slot_request_ids[slot];
    require_request_slot_invariant(request >= 0 && request < config.num_decode_requests);
    int position = config.slot_token_position[slot];
    int rows = request_slot_active_rows(config, slot);
    int pass = request * 64 + position;
    require_request_slot_invariant(config.request_ids[slot] == pass);
#ifdef NAS_LATENCY_PROBE
    nas_probe::end_pass(pass, config.context_predicted_tokens[slot]);
#endif
    position += rows;
    config.slot_token_position[slot] = position;

    // An intermediate prompt chunk has no accepted prediction, even if its
    // readout happens to be EOS. Only the final prompt row starts generation.
    if (position >= config.decode_prompt_lengths[request]) {
      int generated = config.decode_generated_lengths[request];
      require_request_slot_invariant(generated >= 0 && generated < 64);
      int output_row = request * 64 + generated;
      int64_t predicted = config.context_predicted_tokens[slot];
      require_request_slot_invariant(predicted >= 0);
      config.context_generated_tokens[output_row] = predicted;
      save_output(config.context_slot_outputs + size_t(slot) * config.context_output_width,
                  config.context_saved_outputs + size_t(output_row) * config.context_output_width,
                  config.context_output_width);
      config.decode_generated_lengths[request] = generated + 1;
      ETRACE(39, completed_epoch, slot, DATA_INVALID_ID, -1, output_row);
#ifdef NAS_LATENCY_PROBE
      nas_probe::token(output_row, slot, completed_epoch, predicted);
#endif
      needs_request = predicted == config.eos_token_id || position == 64;
    }
#ifdef NAS_COMPARE_IO_ADDRESS
    // Every pass has a distinct capture key, including ignored prompt chunks.
    // Generated-token output rows use a different index after prefill.
    compare_complete(slot, pass);
#endif
  }
  if (needs_request && !try_admit_next_request(config, slot)) return false;

  int request = config.decode_slot_request_ids[slot];
  int position = config.slot_token_position[slot];
  int prompt_length = config.decode_prompt_lengths[request];
  bool prefill = position < prompt_length;
  int rows = prefill ? config.prefill_chunk_capacity : 1;
  require_request_slot_invariant(position >= 0 && position + rows <= 64 &&
      (!prefill || position + rows <= prompt_length));
  auto *input = static_cast<int64_t *>(config.context_slot_inputs) +
      size_t(slot) * config.prefill_chunk_capacity;
  for (int token = 0; token < config.prefill_chunk_capacity; ++token) {
    // Negative IDs are padding: the existing embedding kernel writes zeros.
    if (token >= rows) input[token] = -1;
    else if (prefill)
      input[token] = static_cast<int64_t const *>(config.context_inputs)[
          request * 64 + position + token];
    else input[token] = config.context_predicted_tokens[slot];
  }
  config.slot_chunk_rows[slot] = rows;
#ifdef NAS_LATENCY_PROBE
  nas_probe::begin_pass(request * 64 + position, slot, completed_epoch + 1u, rows, input);
#endif
  config.request_ids[slot] = request * 64 + position;
  require_request_slot_invariant(*config.next_request_id < config.num_contexts);
  ++*config.next_request_id;
  ETRACE(38, completed_epoch + 1u, slot, DATA_INVALID_ID, -1, config.request_ids[slot]);
#ifdef NAS_COMPARE_IO_ADDRESS
  compare_admit(config.request_ids[slot]);
#endif
  return true;
}

__device__ __forceinline__ bool
prepare_decode_request(ResidentRuntimeConfig const &config,
                       uint32_t slot, uint32_t completed_epoch) {
  bool needs_request = completed_epoch == 0u;
  if (!needs_request) {
    int request = config.decode_slot_request_ids[slot];
    require_request_slot_invariant(request >= 0 && request < config.num_decode_requests);
    int step = config.decode_generated_lengths[request];
    int prefix = config.decode_prefix_lengths[request];
    int previous = request * 64 + step;
    require_request_slot_invariant(step >= 0 && prefix + step < 64 &&
                                    config.request_ids[slot] == previous);
    // The terminal argmax has completed. Save EOS as well as ordinary tokens,
    // and finish all output stores before a replacement can reuse this slot.
    int64_t predicted = config.context_predicted_tokens[slot];
    require_request_slot_invariant(predicted >= 0);
    config.context_generated_tokens[previous] = predicted;
    save_output(config.context_slot_outputs + size_t(slot) * config.context_output_width,
                config.context_saved_outputs + size_t(previous) * config.context_output_width,
                config.context_output_width);
    config.decode_generated_lengths[request] = step + 1;
    ETRACE(39, completed_epoch, slot, DATA_INVALID_ID, -1, previous);
#ifdef NAS_COMPARE_IO_ADDRESS
    compare_complete(slot, previous);
#endif
    needs_request = predicted == config.eos_token_id || prefix + step + 1 >= 64;
    if (!needs_request) {
      config.request_ids[slot] = previous + 1;
      static_cast<int64_t *>(config.context_slot_inputs)[slot] = predicted;
    }
  }
  if (needs_request && !try_admit_next_request(config, slot)) return false;
  // Keep the inherited counter's meaning: admitted graph passes, not requests.
  // It is separate from decode_next_request, which consumes each request once.
  require_request_slot_invariant(*config.next_request_id < config.num_contexts);
  ++*config.next_request_id;
  int context = config.request_ids[slot];
  ETRACE(38, completed_epoch + 1u, slot, DATA_INVALID_ID, -1, context);
#ifdef NAS_COMPARE_IO_ADDRESS
  compare_admit(context);
#endif
  return true;
}

// Called collectively by the attention worker CTA. The working cache remains
// private to the slot/head; an immutable prefilled request cache is only read.
// The 48-layer demo scans its short binding list only on the first decode pass.
__device__ __forceinline__ int
prepare_request_attention(ResidentRuntimeConfig const &config,
                          TaskDesc const *task_desc) {
  int slot = task_desc->task_metadata.request_id;
  require_request_slot_invariant(slot >= 0 && slot < config.num_request_slots);
  if (config.prefill_chunk_capacity > 0) {
    // Attention itself appends current K/V. Previously computed prefix entries
    // stay in the private page; no imported prefix or cache copy is needed.
    return config.slot_token_position[slot] + request_slot_active_rows(config, slot);
  }
  int request = config.decode_slot_request_ids[slot];
  require_request_slot_invariant(request >= 0 && request < config.num_decode_requests);
  int prefix = config.decode_prefix_lengths[request];
  int step = config.decode_generated_lengths[request];
  require_request_slot_invariant(prefix >= 0 && step >= 0 && prefix + step < 64);
  if (step == 0) {
    auto *k = static_cast<__nv_bfloat16 *>(task_desc->input_ptrs[1]);
    auto *v = static_cast<__nv_bfloat16 *>(task_desc->input_ptrs[2]);
    bool matched = false;
    for (int i = 0; i < config.num_request_kv_bindings; i++) {
      auto const &binding = config.request_kv_bindings[i];
      size_t stride = size_t(64) * binding.heads * 128;
      uintptr_t begin = reinterpret_cast<uintptr_t>(binding.slot_k);
      uintptr_t address = reinterpret_cast<uintptr_t>(k);
      if (address < begin || address >= begin + size_t(config.num_request_slots) * stride * 2)
        continue;
      size_t offset = (address - begin) / sizeof(__nv_bfloat16);
      size_t head_offset = offset % stride;
      require_request_slot_invariant(offset / stride == static_cast<size_t>(slot) &&
          head_offset < static_cast<size_t>(binding.heads) * 128 && head_offset % 128 == 0 &&
          v == binding.slot_v + offset);
      // Sources have the same per-request stride as the fixed slot allocation.
      auto const *source_k = binding.prefix_k + size_t(request) * stride + head_offset;
      auto const *source_v = binding.prefix_v + size_t(request) * stride + head_offset;
      for (int element = threadIdx.x; element < prefix * 128; element += blockDim.x) {
        size_t index = size_t(element / 128) * binding.heads * 128 + element % 128;
        k[index] = source_k[index];
        v[index] = source_v[index];
      }
      matched = true;
      break;
    }
    require_request_slot_invariant(matched);
    // Existing attention reads cached tokens immediately. All head-local copy
    // stores must therefore finish before any thread enters that implementation.
    __syncthreads();
  }
  return prefix + step + 1; // Current consumed token is included in attention.
}

__device__ __forceinline__ bool
prepare_slot(ResidentRuntimeConfig const &config,
             uint32_t slot, uint32_t completed_epoch) {
  if (config.prefill_chunk_capacity > 0)
    return prepare_prefill_request(config, slot, completed_epoch);
  if (config.num_decode_requests >= 0)
    return prepare_decode_request(config, slot, completed_epoch);
  if (config.num_contexts >= 0) {
    bool const autoregressive = config.context_predicted_tokens != nullptr;
    int64_t predicted_token = 0; // Bootstrap has no completed prediction.
    // Only scheduler 0 advances slots, after their terminal releases. Save by
    // the OLD context ID before replacing it; completion order can differ
    // from input order. Bootstrap has no output to save.
    if (completed_epoch > 0u) {
      int previous = config.request_ids[slot];
      require_request_slot_invariant(previous >= 0 && previous < config.num_contexts);
      if (config.context_sequence_length > 0) {
        require_request_slot_invariant(
#ifdef NAS_COMPARE_SEQUENCE_REPEATS
            completed_epoch <= NAS_COMPARE_STEPS * NAS_COMPARE_SEQUENCE_REPEATS &&
            previous == compare_context_row(slot, completed_epoch - 1u));
#else
            completed_epoch <= static_cast<uint32_t>(config.context_sequence_length) &&
            previous == static_cast<int>(slot) * config.context_sequence_length +
                            static_cast<int>(completed_epoch) - 1);
#endif
      }
      if (autoregressive) {
        // Terminal argmax is finished and completion has been acquired. Save
        // the ID before either retiring the slot or enabling its next pass.
        predicted_token = config.context_predicted_tokens[slot];
        require_request_slot_invariant(predicted_token >= 0);
        config.context_generated_tokens[previous] = predicted_token;
      }
      save_output(config.context_slot_outputs + size_t(slot) * config.context_output_width,
                  config.context_saved_outputs + size_t(previous) * config.context_output_width,
                  config.context_output_width);
      ETRACE(39, completed_epoch, slot, DATA_INVALID_ID, -1, previous);
#ifdef NAS_COMPARE_IO_ADDRESS
      compare_complete(slot, previous);
#endif
    }
    // Fixed sequences address their own row directly. Reuse get_new_input's
    // copy with a local cursor; the global cursor remains an admitted-pass
    // counter for occupancy accounting, independent of the selected row ID.
    int sequence_row = 0;
    int *input_cursor = config.next_request_id;
    if (config.context_sequence_length > 0) {
#ifdef NAS_COMPARE_SEQUENCE_REPEATS
      // Replace this slot's sequence locally after each T passes. Its epoch
      // remains monotonic; only the attention position wraps to the prefix.
      if (completed_epoch == NAS_COMPARE_STEPS * NAS_COMPARE_SEQUENCE_REPEATS) {
#else
      if (completed_epoch == static_cast<uint32_t>(config.context_sequence_length)) {
#endif
        config.request_ids[slot] = -1; // The final result was saved above.
        return false;
      }
#ifdef NAS_COMPARE_SEQUENCE_REPEATS
      sequence_row = compare_context_row(slot, completed_epoch);
#else
      sequence_row = static_cast<int>(slot) * config.context_sequence_length +
                     static_cast<int>(completed_epoch);
#endif
      input_cursor = &sequence_row;
    }
    // Admission copies IDs only: one for decode, or P for the opt-in fixed
    // chunk. Workers gather the large embedding rows after slot activation.
    int context;
    if (autoregressive && completed_epoch > 0u) {
      // context identifies the next saved pass, not a vocabulary token.
      // Only the token value changes source: use our prediction instead of
      // reading the next preloaded row. Bootstrap still loads the seed below.
      context = sequence_row;
      static_cast<int64_t *>(config.context_slot_inputs)[slot] = predicted_token;
    } else if (config.context_input_is_int64) {
      context = get_new_input(
          static_cast<int64_t const *>(config.context_inputs), config.num_contexts,
          config.context_input_width, input_cursor,
          static_cast<int64_t *>(config.context_slot_inputs) +
              size_t(slot) * config.context_input_width);
    } else {
      context = get_new_input(
          static_cast<__nv_bfloat16 const *>(config.context_inputs),
          config.num_contexts, config.context_input_width, input_cursor,
          static_cast<__nv_bfloat16 *>(config.context_slot_inputs) +
              size_t(slot) * config.context_input_width);
    }
    if (config.context_sequence_length > 0) {
      require_request_slot_invariant(context >= 0 && *config.next_request_id < config.num_contexts);
      ++*config.next_request_id; // Only scheduler 0 prepares slots.
    }
    config.request_ids[slot] = context;
    if (context >= 0) {
      ETRACE(38, completed_epoch + 1u, slot, DATA_INVALID_ID, -1, context);
#ifdef NAS_COMPARE_IO_ADDRESS
      compare_admit(context);
#endif
    }
    // Copies are synchronous stores on this thread. advance_slot() publishes
    // them with its existing fence before dispatching the next pass. Exhaustion
    // still saves the final result, then follows the usual inactive-slot path.
    return context >= 0;
  }
  // Bounded synthetic policy: input/weight addresses and values stay fixed.
  // This is the preparation hook for a later serving policy with stable
  // per-slot token/KV storage and admission. Do not put prepare_next_batch()
  // here: it compacts metadata belonging to slots that may still be running.
  return completed_epoch < config.slot_iteration_limits[slot];
}

__device__ __forceinline__ bool
advance_slot(ResidentRuntimeConfig const &config,
             uint32_t slot, uint32_t completed_epoch) {
  // Only scheduler 0 calls this. The completion payload is authoritative;
  // accepting a stale notification could reset a newer pass's counters.
  require_request_slot_invariant(
      slot < static_cast<uint32_t>(config.num_request_slots));
  require_request_slot_invariant(config.slot_iteration[slot] == completed_epoch);
  uint32_t expected = completed_epoch == 0u ? REQUEST_SLOT_INACTIVE
                                           : REQUEST_SLOT_COMPLETE;
  require_request_slot_invariant(
      atomicAdd(&config.slot_state[slot], 0u) == expected);
  prepare_request_slot_iteration(config, slot);
  ETRACE(31, completed_epoch, slot, DATA_INVALID_ID, -1, 0);
  if (!prepare_slot(config, slot, completed_epoch)) {
    transition_request_slot_state(config, slot, REQUEST_SLOT_PREPARING,
                                   REQUEST_SLOT_INACTIVE);
    ETRACE(33, completed_epoch, slot, DATA_INVALID_ID, -1, 0);
    return false;
  }

  reset_slot_iteration_state(config, slot);
  // Activation fences reset/preparation writes before RUNNING. Each following
  // scheduler publication and worker-queue publication retains its fence.
  activate_request_slot_iteration(config, slot);
  uint32_t epoch = config.slot_iteration[slot];
  ETRACE(32, epoch, slot, DATA_INVALID_ID, -1, 0);
  // One broadcaster gives every scheduler the same activation order. Together
  // with each slot's topological task order, this avoids prelaunch FIFO cycles
  // caused by different schedulers enqueueing independent slots in reverse order.
  for (int scheduler = 0; scheduler < config.num_local_schedulers; scheduler++) {
    publish_scheduler_event(
        config, scheduler, make_slot_control_item(slot, epoch, true));
  }
  return true;
}

__device__ __forceinline__ void
dispatch_slot(ResidentRuntimeConfig const &config,
               uint32_t slot, uint32_t epoch,
               unsigned long long my_first_worker,
               unsigned long long my_last_worker,
               size_t *worker_queue_next_free_task_pos) {
  for (uint32_t i = config.slot_data_offsets[slot];
       i < config.slot_data_offsets[slot + 1]; i++) {
    DataId data = config.slot_data_ids[i];
    bool root = config.data_initial_predecessor_counts[data] == 0u;
    if (!resident_uses_hybrid_prelaunch(config) && !root) {
      continue; // scheduler_dispatch enables successors through completions.
    }
    // Keep one producer per worker queue. Each scheduler handles only workers
    // in its existing ownership range. Different slots can share workers.
    unsigned long long worker = data % config.num_workers;
    if (worker < my_first_worker || worker >= my_last_worker) {
      continue;
    }
    // A scheduler with no tasks for this slot may read an older broadcast
    // after that slot advanced again. It must perform no mutable-slot access.
    // A scheduler with assigned work cannot be overtaken: that work must run
    // before all terminals of this epoch can complete.
    validate_request_slot_iteration(config, data, epoch);
    TaskId task = config.data_to_task_id[data];
    require_request_slot_invariant(task != TASK_INVALID_ID);
    ETRACE(34, epoch, task, data, worker, slot);
    if (root) {
      ETRACE(3, epoch, task, data, worker, DATA_INVALID_ID);
    }
    enqueue_worker_item(config, worker,
                        &worker_queue_next_free_task_pos[worker - my_first_worker],
                        compute_task_id(epoch, task));
  }
}

// Called by scheduler lane 0 before interpreting a queue word as an EventDesc.
// In hybrid mode the caller must still reach its existing warp synchronization.
__device__ __forceinline__ bool
handle_slot_restart_event(ResidentRuntimeConfig const &config,
                          EventId item, int scheduler,
                          unsigned long long my_first_worker,
                          unsigned long long my_last_worker,
                          size_t *worker_queue_next_free_task_pos) {
  if (!slot_restart_enabled(config) || item == 0u) {
    return false; // Retain the existing termination-event handling.
  }
  if (item == static_cast<EventId>(config.end_event_index)) {
    // The existing synthetic END bootstraps the run exactly once. It does not
    // represent completion of any slot. No global BEGIN task is needed here.
    require_request_slot_invariant(
        scheduler == 0 && *config.current_iteration == 0u);
    *config.current_iteration = 1u; // Bootstrap marker, not a slot epoch clock.
    ETRACE(35, 0, 0, DATA_INVALID_ID, -1, 0);
    uint32_t active = 0u;
    for (uint32_t slot = 0;
         slot < static_cast<uint32_t>(config.num_request_slots); slot++) {
      if (advance_slot(config, slot, 0u)) {
        active++;
      }
    }
    // Only scheduler 0 reads/writes this count; it cannot handle completion
    // notifications until this bootstrap loop has finished.
    *config.active_request_slot_count = active;
    if (active == 0u) {
      terminate_schedulers(config);
    }
    return true;
  }

  require_request_slot_invariant(is_slot_control_item(item));
  uint32_t slot = static_cast<uint32_t>(item) & SLOT_CONTROL_ID_MASK;
  uint32_t epoch = static_cast<uint32_t>(item >> 32);
  require_request_slot_invariant(
      slot < static_cast<uint32_t>(config.num_request_slots) && epoch != 0u);
  if ((static_cast<uint32_t>(item) & SLOT_DISPATCH_TAG) != 0u) {
    dispatch_slot(config, slot, epoch, my_first_worker, my_last_worker,
                   worker_queue_next_free_task_pos);
  } else {
    require_request_slot_invariant(scheduler == 0);
    ETRACE(36, epoch, slot, DATA_INVALID_ID, -1, 0);
    if (!advance_slot(config, slot, epoch)) {
      uint32_t active = *config.active_request_slot_count;
      require_request_slot_invariant(active > 0u);
      *config.active_request_slot_count = active - 1u;
      // This policy has a closed, finite workload. A serving policy with open
      // admission must also account for incoming requests before shutting down.
      if (active == 1u) {
        terminate_schedulers(config);
      }
    }
  }
  return true;
}

// Runtime-only plan built from the native graph's existing slot ownership.
// No dependency is inserted or removed. The topological order is used only
// for prelaunch dispatch, and for compact slot-local counter reset.
struct SlotRestartPlan {
  std::vector<uint32_t> limits;
  std::vector<uint32_t> data_offsets;
  std::vector<DataId> data_ids;
};

// Slot attention uses one private page. V0 overwrites position 63; fixed
// sequences append at prefix+epoch-1. K/V are both inputs and outputs. Validate
// resolved addresses as well as shapes: distinct tensor names can still alias
// the same CUDA allocation. This is host initialization, never scheduler work.
static void validate_slot_attention_bindings(
    ResidentRuntimeConfig const &c, std::vector<FullDataDesc> const &data,
    std::vector<ResidentTaskDesc> const &tasks, std::vector<uint32_t> const &slots) {
  bool found = false;
  for (auto const &d : data) {
    if (d.resident_task_id >= tasks.size())
      throw std::runtime_error("Slot attention validation found an invalid resident task ID");
    found |= tasks[d.resident_task_id].task_type == TASK_ATTENTION_SLOT_HOPPER;
  }
  if (!found) return;
  if (c.num_request_slots <= 0 || slots.size() != data.size() ||
      c.num_gpus != 1 || resident_uses_streaming(c) ||
      (c.num_contexts < 0 && std::getenv("MIRAGE_RESIDENT_SLOT_ITERATIONS") == nullptr))
    throw std::runtime_error("Slot attention V0 requires single-GPU resident slot restart");

  struct Range { uintptr_t begin, end; };
  std::vector<Range> cache_rows;
  auto same_view = [](TensorDesc const &a, TensorDesc const &b) {
    if (a.base_ptr != b.base_ptr || a.num_dims != b.num_dims ||
        a.data_type != b.data_type) return false;
    for (int i = 0; i < a.num_dims; i++)
      if (a.dim[i] != b.dim[i] || a.stride[i] != b.stride[i]) return false;
    return true;
  };
  for (size_t d = 0; d < data.size(); d++) {
    auto const &task = tasks[data[d].resident_task_id];
    if (task.task_type != TASK_ATTENTION_SLOT_HOPPER) continue;
    if ((task.num_inputs != 3 && task.num_inputs != 5 && task.num_inputs != 7) ||
        task.num_outputs != 3 ||
        slots[d] >= static_cast<uint32_t>(c.num_request_slots) ||
        data[d].task_metadata.request_id != static_cast<int>(slots[d]))
      throw std::runtime_error("Slot attention V0 descriptor or slot metadata mismatch");
    auto const &q = data[d].inputs[0];
    auto const &o = data[d].outputs[0];
    auto const &k = data[d].inputs[1];
    // Use the same derived widths as registration. The 64-token / D=128
    // limits remain explicit until other kernel configurations are validated.
    if (k.num_dims != 4 || k.dim[1] != 64 || k.dim[3] != 128 ||
        o.num_dims != 2 || o.dim[1] % k.dim[3] != 0 ||
        k.stride[1] % k.dim[3] != 0)
      throw std::runtime_error("Slot attention requires consistent heads, D=128 and P=64");
    int const page_size = k.dim[1], head_dim = k.dim[3];
    int const heads = k.stride[1] / head_dim, ratio = o.dim[1] / head_dim;
    auto row = [&](TensorDesc const &t, int width) {
      return t.base_ptr != nullptr && t.data_type == mirage::type::DT_BFLOAT16 &&
             t.num_dims == 2 && t.dim[0] == NAS_PREFILL_CHUNK_ROWS &&
             t.dim[1] == width &&
             int64_t(t.stride[0]) == int64_t(heads) * width && t.stride[1] == 1;
    };
    if (heads < 1 || (ratio != 4 && ratio != 8) ||
        !row(q, (ratio + 2) * head_dim) || !row(o, ratio * head_dim))
      throw std::runtime_error("Slot attention requires a GQA4/GQA8 query chunk matching NAS_PREFILL_CHUNK_ROWS");
    // Optional pairs are norm weights first, then RoPE tables. A five-input
    // task can have either pair; ranks distinguish them without new GPU state.
    bool saw_rope = false;
    for (int i = 3; i < task.num_inputs; i += 2) {
      bool const norm = data[d].inputs[i].num_dims == 1;
      if ((norm && (i != 3 || saw_rope)) || (!norm && saw_rope))
        throw std::runtime_error("Invalid slot attention normalization/RoPE input order");
      for (int j = i; j < i + 2; j++) {
        auto const &t = data[d].inputs[j];
        if (t.base_ptr == nullptr || t.data_type != mirage::type::DT_BFLOAT16 ||
            (norm ? (t.num_dims != 1 || t.dim[0] != head_dim || t.stride[0] != 1)
                  : (t.num_dims != 2 || t.dim[0] < page_size || t.dim[1] != head_dim ||
                     t.stride[0] != static_cast<size_t>(head_dim) || t.stride[1] != 1)))
          throw std::runtime_error("Slot attention requires norm[D] and full-head RoPE[positions,D]");
      }
      saw_rope |= !norm;
    }
    for (int i = 1; i <= 2; i++) {
      auto const &view = data[d].inputs[i];
      if (view.base_ptr == nullptr || view.data_type != mirage::type::DT_BFLOAT16 ||
          view.num_dims != 4 || view.dim[0] != 1 || view.dim[1] != page_size ||
          view.dim[2] != 1 || view.dim[3] != head_dim ||
          int64_t(view.stride[0]) != int64_t(page_size) * heads * head_dim ||
          int64_t(view.stride[1]) != int64_t(heads) * head_dim ||
          view.stride[2] != static_cast<size_t>(head_dim) || view.stride[3] != 1 ||
          !same_view(view, data[d].outputs[i]))
        throw std::runtime_error("Slot attention V0 requires exact in-place private K/V page views");
      // Different heads interleave inside a page. A single bounding interval
      // would falsely overlap them; check each token/head row using its actual BF16 width.
      uintptr_t base = reinterpret_cast<uintptr_t>(view.base_ptr);
      for (int token = 0; token < page_size; token++) {
        uintptr_t begin = base + size_t(token) * view.stride[1] * 2;
        cache_rows.push_back({begin, begin + size_t(head_dim) * 2});
      }
    }
  }
  std::sort(cache_rows.begin(), cache_rows.end(),
            [](Range const &a, Range const &b) { return a.begin < b.begin; });
  for (size_t i = 1; i < cache_rows.size(); i++) {
    if (cache_rows[i].begin < cache_rows[i - 1].end)
      throw std::runtime_error("Slot attention V0 cache views overlap or have multiple writers");
  }

  // Only the attention task may access its cache. Reject overlap with any
  // other graph view, including aliases attached under different names.
  for (auto const &d : data) {
    auto const &task = tasks[d.resident_task_id];
    auto check = [&](TensorDesc const &t, int index) {
      // Only the K/V input/output aliases were checked above. Additional
      // read-only inputs must still be checked against writable cache memory.
      if (task.task_type == TASK_ATTENTION_SLOT_HOPPER && (index == 1 || index == 2)) return;
      if (t.num_dims <= 0 || t.num_dims > mirage::config::MAX_TENSOR_DIMS ||
          t.base_ptr == nullptr)
        throw std::runtime_error("Slot attention V0 graph contains an invalid tensor view");
      size_t bytes = t.data_type == mirage::type::DT_BFLOAT16 ? 2 :
          t.data_type == mirage::type::DT_INT64 ? 8 :
          (t.data_type == mirage::type::DT_INT32 ||
           t.data_type == mirage::type::DT_FLOAT32) ? 4 : 0;
      if (bytes == 0) throw std::runtime_error("Slot attention V0 graph tensor dtype is unsupported");
      size_t span = 1;
      for (int axis = 0; axis < t.num_dims; axis++) {
        if (t.dim[axis] <= 0 || t.stride[axis] <= 0)
          throw std::runtime_error("Slot attention V0 graph tensor strides are invalid");
        span += size_t(t.dim[axis] - 1) * t.stride[axis];
      }
      uintptr_t begin = reinterpret_cast<uintptr_t>(t.base_ptr);
      uintptr_t end = begin + span * bytes;
      auto next = std::lower_bound(cache_rows.begin(), cache_rows.end(), begin,
          [](Range const &range, uintptr_t address) { return range.end <= address; });
      if (next != cache_rows.end() && next->begin < end)
        throw std::runtime_error("Slot attention V0 cache overlaps another graph tensor");
    };
    for (int i = 0; i < task.num_inputs; i++) check(d.inputs[i], i);
    for (int i = 0; i < task.num_outputs; i++) check(d.outputs[i], i);
  }
}

// Copying into an otherwise read-only graph input adds a write outside the
// compiled DAG. Check resolved addresses, including aliases under other names,
// so that every access to a working row belongs to that row's slot. The first
// interface accepts full contiguous rows; no descriptor rebinding is needed.
static void validate_context_io_bindings(
    ResidentRuntimeConfig const &c, std::vector<FullDataDesc> const &data,
    std::vector<ResidentTaskDesc> const &tasks, std::vector<uint32_t> const &slots) {
  if (c.num_contexts < 0) return;
  // P is compiled into task geometry. Native prefill uses scalar external
  // prompt IDs but P working IDs per slot; the fixed-chunk API uses P for both.
  constexpr int chunk_rows = NAS_PREFILL_CHUNK_ROWS;
  bool const native_prefill = c.prefill_chunk_capacity > 0;
  int const slot_input_width = native_prefill ? chunk_rows : c.context_input_width;
  if ((c.context_input_is_int64 != 0 && c.context_input_is_int64 != 1) ||
      (c.context_input_is_int64 && c.context_input_width != (native_prefill ? 1 : chunk_rows))) {
    throw std::runtime_error("Integer ContextIO width disagrees with its prompt/chunk policy");
  }
  if (native_prefill && (c.prefill_chunk_capacity != chunk_rows ||
      c.num_decode_requests < 0 || !c.context_input_is_int64))
    throw std::runtime_error("Native prefill requires a request queue and matching compiled chunk capacity");
  if (chunk_rows > 1 && !native_prefill &&
      (!c.context_input_is_int64 || c.num_contexts != c.num_request_slots ||
       c.context_sequence_length != 1 || c.context_prefix_length != 0 ||
       c.num_decode_requests >= 0 || c.context_predicted_tokens != nullptr ||
       c.context_generated_tokens != nullptr)) {
    throw std::runtime_error("Fixed chunks require int64 ContextIO, one full chunk per slot, sequence=(1,0), and no decode feedback");
  }
  if (c.context_input_width <= 0 || c.context_output_width <= 0 ||
      c.context_slot_inputs == nullptr || c.context_slot_outputs == nullptr ||
      (c.num_contexts > 0 &&
       (c.context_inputs == nullptr || c.context_saved_outputs == nullptr))) {
    throw std::runtime_error("Invalid context IO pointers or widths");
  }
  bool const autoregressive = c.context_predicted_tokens != nullptr;
  uintptr_t base[] = {reinterpret_cast<uintptr_t>(c.context_inputs),
                      reinterpret_cast<uintptr_t>(c.context_saved_outputs),
                      reinterpret_cast<uintptr_t>(c.context_slot_inputs),
                      reinterpret_cast<uintptr_t>(c.context_slot_outputs),
                      reinterpret_cast<uintptr_t>(c.context_predicted_tokens),
                      reinterpret_cast<uintptr_t>(c.context_generated_tokens)};
  size_t input_element_bytes = c.context_input_is_int64 ? sizeof(int64_t) : sizeof(__nv_bfloat16);
  size_t row[] = {size_t(slot_input_width) * input_element_bytes,
                 size_t(c.context_output_width) * sizeof(__nv_bfloat16)};
  size_t bytes[] = {size_t(c.num_contexts) * c.context_input_width * input_element_bytes,
                    size_t(c.num_contexts) * row[1],
                    size_t(c.num_request_slots) * row[0], size_t(c.num_request_slots) * row[1],
                    size_t(c.num_request_slots) * sizeof(int64_t),
                    size_t(c.num_contexts) * sizeof(int64_t)};
  auto overlaps = [](uintptr_t a, size_t n, uintptr_t b, size_t m) {
    return n != 0 && m != 0 && a < b + m && b < a + n;
  };
  int const num_buffers = autoregressive ? 6 : 4;
  for (int i = 0; i < num_buffers; i++) {
    for (int j = i + 1; j < num_buffers; j++) {
      if (overlaps(base[i], bytes[i], base[j], bytes[j]))
        throw std::runtime_error("Context IO buffers must not overlap");
    }
  }
  std::vector<bool> reads_input(c.num_request_slots, false);
  std::vector<bool> writes_output(c.num_request_slots, false);
  std::vector<int> prediction_writers(c.num_request_slots, 0);
  std::vector<int> prediction_vocab(c.num_request_slots, 0);
  std::vector<int> embedding_vocab(c.num_request_slots, 0);
  for (size_t d = 0; d < data.size(); d++) {
    auto const &task = tasks[data[d].resident_task_id];
    bool const gather = task.task_type == TASK_IDENTITY;
    bool const reads_live_count = gather || task.task_type == TASK_ATTENTION_SLOT_HOPPER ||
        task.task_type == TASK_MOE_TOPK_ROW || task.task_type == TASK_SILU_MUL ||
        task.task_type == TASK_MOE_WEIGHTED_SUM;
    if (native_prefill && reads_live_count &&
        (slots[d] > INT16_MAX || data[d].task_metadata.request_id != static_cast<int>(slots[d])))
      throw std::runtime_error("Chunk-aware task metadata must identify its owning slot");
    if (gather) {
      if (!native_prefill || task.num_inputs != 1 || task.num_outputs != 1)
        throw std::runtime_error("Slot identity is reserved for the native-prefill last-row gather");
      auto const &input = data[d].inputs[0];
      auto const &output = data[d].outputs[0];
      if (input.data_type != mirage::type::DT_BFLOAT16 ||
          output.data_type != mirage::type::DT_BFLOAT16 ||
          input.num_dims != 2 || output.num_dims != 2 ||
          input.dim[0] != chunk_rows || output.dim[0] != 1 ||
          input.dim[1] <= 0 || input.dim[1] != output.dim[1] ||
          input.stride[0] != input.dim[1] || input.stride[1] != 1 ||
          output.stride[0] != output.dim[1] || output.stride[1] != 1)
        throw std::runtime_error("Last-row gather requires contiguous BF16 [P,H] -> [1,H]");
    }
    if (task.task_type == TASK_ELEMENTWISE_ADD_SM100) {
      // A residual task adds two private [P,H] chunks (P=1 for decode).
      // Its kernel uses the output stride for all pointers, so require
      // matching contiguous views. Slot ownership remains unchanged.
      if (task.num_inputs != 2 || task.num_outputs != 1)
        throw std::runtime_error("Slot residual add requires two inputs and one output");
      auto const &output = data[d].outputs[0];
      auto check_row = [&](TensorDesc const &view) {
        if (view.base_ptr == nullptr || view.data_type != mirage::type::DT_BFLOAT16 ||
            view.num_dims != 2 || view.dim[0] != chunk_rows || view.dim[1] <= 0 ||
            view.dim[1] != output.dim[1] || view.stride[0] != view.dim[1] ||
            view.stride[1] != 1)
          throw std::runtime_error("Slot residual add requires matching contiguous BF16 [P,H] views");
      };
      check_row(output);
      check_row(data[d].inputs[0]);
      check_row(data[d].inputs[1]);
    }
    // Ordinary chunk kernels compile their local geometry into CUDA/TMA.
    // Check those rows as well as IO byte ranges, so an incorrectly marked
    // one-row task cannot silently compute only the first token of a chunk.
    if (chunk_rows > 1 &&
        (task.task_type == TASK_EMBEDDING_INPUT ||
         task.task_type == TASK_RMS_NORM_HOPPER ||
         task.task_type == TASK_LINEAR_SWAPAB_HOPPER ||
         task.task_type == TASK_LINEAR_CUTLASS_HOPPER ||
         task.task_type == TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER)) {
      auto const &input = data[d].inputs[0];
      auto const &output = data[d].outputs[0];
      bool embedding = task.task_type == TASK_EMBEDDING_INPUT;
      // The one-row readout after gather intentionally has a smaller local
      // batch. Earlier chunk operators keep P-row geometry in the native DAG.
      int const rows = native_prefill && !embedding && output.num_dims == 2 &&
          output.dim[0] == 1 ? 1 : chunk_rows;
      bool input_shape = embedding
          ? input.num_dims == 2 && input.dim[0] == 1 && input.dim[1] == chunk_rows
          : input.num_dims == 2 && input.dim[0] == rows;
      if (!input_shape || output.num_dims != 2 || output.dim[0] != rows)
        throw std::runtime_error("Ordinary task local rows disagree with the fixed chunk capacity");
    }
    if (task.task_type == TASK_ARGMAX_ROW) {
      if (chunk_rows > 1 && !native_prefill)
        throw std::runtime_error("Fixed chunk prefill stops before readout; argmax is not supported");
      // The row kernel uses only descriptor pointers and reads exactly one
      // vocabulary row. Its scalar token output is captured before slot reuse.
      if (task.num_inputs != 1 || task.num_outputs != 1)
        throw std::runtime_error("Slot argmax requires one input and one output");
      auto const &input = data[d].inputs[0];
      auto const &output = data[d].outputs[0];
      if (input.base_ptr == nullptr || input.data_type != mirage::type::DT_BFLOAT16 ||
          input.num_dims != 2 || input.dim[0] != 1 || input.dim[1] <= 0 ||
          input.stride[0] != input.dim[1] || input.stride[1] != 1)
        throw std::runtime_error("Slot argmax requires a contiguous BF16 [1,V] input");
      if (output.base_ptr == nullptr || output.data_type != mirage::type::DT_INT64 ||
          output.num_dims != 2 || output.dim[0] != 1 || output.dim[1] != 1 ||
          output.stride[0] != 1 || output.stride[1] != 1)
        throw std::runtime_error("Slot argmax requires a contiguous int64 [1,1] output");
    }
    auto check = [&](TensorDesc const &t, bool output, int index) {
      // Routing owns R FP32 weights and an int32 rank row. The weighted sum
      // reads those same weights after every W2 branch has finished. Permit
      // exactly these positions, not arbitrary FP32 ContextIO activations.
      bool topk = task.task_type == TASK_MOE_TOPK_ROW;
      bool logical_moe = task.task_type == TASK_MOE_SLOT_W13 ||
                         task.task_type == TASK_MOE_SLOT_W2;
      bool route_view = (topk && output && index == 1) ||
                        (logical_moe && !output && index == 2);
      bool route_weight = (topk && output && index == 0) ||
          (task.task_type == TASK_MOE_WEIGHTED_SUM && !output && index == 1);
      bool allowed_scratch = (route_view && t.data_type == mirage::type::DT_INT32) ||
                             (route_weight && t.data_type == mirage::type::DT_FLOAT32);
      if ((t.data_type != mirage::type::DT_BFLOAT16 &&
           t.data_type != mirage::type::DT_INT64 && !allowed_scratch) || t.num_dims <= 0)
        throw std::runtime_error("Context IO task tensor dtype is not allowed for this operator position");
      if (route_view || route_weight) {
        int branches = 1;
#ifdef MIRAGE_EXPERIMENTAL_MOE_BATCHING
        branches = EXPERIMENTAL_MOE_TOP_K;
#endif
        int width = route_view ? 128 : branches;
        int dtype = route_view ? mirage::type::DT_INT32 : mirage::type::DT_FLOAT32;
        if (t.data_type != dtype || t.num_dims != 2 || t.dim[0] != chunk_rows ||
            t.dim[1] != width || t.stride[0] != width || t.stride[1] != 1)
          throw std::runtime_error("Experimental routing requires a private [P,width] chunk matching P and top-k");
      }
      // Integer views are token IDs: embedding reads one, and row argmax
      // writes one. Other compute must not reinterpret int64 as BF16 data.
      bool embedding_id = task.task_type == TASK_EMBEDDING_INPUT && !output && index == 0;
      bool predicted_id = task.task_type == TASK_ARGMAX_ROW && output && index == 0;
      if ((t.data_type == mirage::type::DT_INT64) != (embedding_id || predicted_id))
        throw std::runtime_error("Int64 task views must be embedding inputs or row-argmax outputs");
      size_t span = 1, elements = 1;
      for (int axis = 0; axis < t.num_dims; axis++) {
        if (t.dim[axis] <= 0 || t.stride[axis] <= 0)
          throw std::runtime_error("Invalid context task tensor view");
        span += size_t(t.dim[axis] - 1) * t.stride[axis];
        elements *= t.dim[axis];
      }
      uintptr_t ptr = reinterpret_cast<uintptr_t>(t.base_ptr);
      size_t element_bytes = t.data_type == mirage::type::DT_INT64 ? sizeof(int64_t) :
          (t.data_type == mirage::type::DT_BFLOAT16 ? sizeof(__nv_bfloat16) : sizeof(int32_t));
      size_t length = span * element_bytes;
      for (int i = 0; i < 4; i++) {
        if (!overlaps(ptr, length, base[i], bytes[i])) continue;
        if (i < 2)
          throw std::runtime_error("Graph tasks must use slot rows, not the context queue/results");
        int which = i - 2;
        auto expected_type = i == 2 && c.context_input_is_int64
                                 ? mirage::type::DT_INT64 : mirage::type::DT_BFLOAT16;
        if (t.data_type != expected_type)
          throw std::runtime_error("Context task view dtype disagrees with its slot buffer");
        if (ptr != base[i] + size_t(slots[d]) * row[which] ||
            length != row[which] || span != elements || (i == 2 && output))
          throw std::runtime_error("Context task view must be the owning slot's full contiguous row");
        if (i == 2 && !output) {
          reads_input[slots[d]] = true;
          if (autoregressive || chunk_rows > 1) {
            // Feedback and every supplied chunk ID must address the embedding.
            auto const &weight = data[d].inputs[1];
            if (task.num_inputs != 2 || weight.num_dims != 2 || weight.dim[0] <= 0)
              throw std::runtime_error("Autoregressive input requires a valid embedding table");
            int &vocab = embedding_vocab[slots[d]];
            vocab = vocab == 0 ? weight.dim[0] : std::min(vocab, weight.dim[0]);
          }
        }
        if (i == 3 && output) writes_output[slots[d]] = true;
      }
      if (autoregressive) {
        if (overlaps(ptr, length, base[5], bytes[5]))
          throw std::runtime_error("Graph tasks must not access generated-token history");
        if (overlaps(ptr, length, base[4], bytes[4])) {
          if (!predicted_id || ptr != base[4] + size_t(slots[d]) * sizeof(int64_t) ||
              length != sizeof(int64_t) || span != 1)
            throw std::runtime_error("Feedback must bind the owning slot's scalar argmax output");
          ++prediction_writers[slots[d]];
          prediction_vocab[slots[d]] = data[d].inputs[0].dim[1];
        }
      }
    };
    for (int i = 0; i < task.num_inputs; i++) check(data[d].inputs[i], false, i);
    for (int i = 0; i < task.num_outputs; i++) check(data[d].outputs[i], true, i);
  }
  // The embedding kernel does not bounds-check positive IDs. A full chunk
  // consumes every supplied ID, so checking only its first token is unsafe.
  std::vector<int64_t> chunk_ids;
  if (chunk_rows > 1 && !native_prefill) {
    chunk_ids.resize(size_t(c.num_contexts) * chunk_rows);
    CUDA_CHECK(cudaMemcpy(chunk_ids.data(), c.context_inputs,
                          chunk_ids.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));
  }
  for (int slot = 0; slot < c.num_request_slots; slot++) {
    if (chunk_rows > 1 && !native_prefill) {
      for (int token = 0; token < chunk_rows; token++) {
        int64_t id = chunk_ids[size_t(slot) * chunk_rows + token];
        if (id < 0 || id >= embedding_vocab[slot])
          throw std::runtime_error("Every fixed-chunk token must be inside the embedding vocabulary");
      }
    }
    if (!reads_input[slot] || !writes_output[slot])
      throw std::runtime_error("Every slot must read its context input and write its complete output row");
    if (autoregressive && (prediction_writers[slot] != 1 ||
                           prediction_vocab[slot] > embedding_vocab[slot]))
      throw std::runtime_error("Each feedback slot needs one argmax writer within its embedding vocabulary");
  }
}

// Request metadata is outside the numerical graph, but workers read prefixes
// and the controller writes lengths/history. Validate ownership once on the
// host, before any request can run. Call after the existing graph/ContextIO
// validators; they establish valid tensor shapes and fixed slot ownership.
static void validate_decode_request_bindings(
    ResidentRuntimeConfig const &c, std::vector<FullDataDesc> const &data,
    std::vector<ResidentTaskDesc> const &tasks, std::vector<uint32_t> const &slots) {
  if (c.num_decode_requests < 0) return;
  bool const native_prefill = c.prefill_chunk_capacity > 0;
  // Imported-prefix decode and native prefill share admission and output
  // storage, but have different immutable per-request length arrays.
  int32_t const *lengths = native_prefill ? c.decode_prompt_lengths
                                         : c.decode_prefix_lengths;
  if (c.num_decode_requests > INT32_MAX / 64 ||
      c.num_contexts != c.num_decode_requests * 64 ||
      c.context_input_is_int64 != 1 || c.context_input_width != 1 ||
      c.context_sequence_length != 0 || c.context_prefix_length != 0 ||
      c.context_predicted_tokens == nullptr ||
      (c.num_decode_requests > 0 && (c.context_generated_tokens == nullptr ||
       lengths == nullptr || c.decode_generated_lengths == nullptr)))
    throw std::runtime_error("Request queues require int64 ContextIO rows=N*64, feedback, and request length arrays");
  if (native_prefill && (c.prefill_chunk_capacity != NAS_PREFILL_CHUNK_ROWS ||
      c.decode_prefix_lengths != nullptr || !g_request_kv_bindings.empty()))
    throw std::runtime_error("Native prefill uses the compiled chunk capacity and no imported KV prefixes");
  if (c.eos_token_id < -1)
    throw std::runtime_error("Request queue EOS must be -1 (disabled) or a vocabulary token");

  std::vector<int32_t> prefix(c.num_decode_requests);
  std::vector<int64_t> inputs(c.num_contexts);
  if (c.num_decode_requests > 0) {
    CUDA_CHECK(cudaMemcpy(prefix.data(), lengths,
                          prefix.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(inputs.data(), c.context_inputs,
                          inputs.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));
  }
  int vocab = INT32_MAX;
  for (auto const &d : data) {
    if (tasks[d.resident_task_id].task_type == TASK_EMBEDDING_INPUT)
      vocab = std::min(vocab, d.inputs[1].dim[0]);
  }
  if (vocab == INT32_MAX || c.eos_token_id >= vocab)
    throw std::runtime_error("Request queue needs an embedding vocabulary containing EOS");
  for (int request = 0; request < c.num_decode_requests; request++) {
    int length = prefix[request];
    if (native_prefill) {
      if (length <= 0 || length > 64 || length % c.prefill_chunk_capacity != 0)
        throw std::runtime_error("Native prompt lengths must be positive full chunks fitting the 64-token cache");
    } else if (length < 0 || length >= 64) {
      throw std::runtime_error("Imported request prefix lengths must be in [0,63]");
    }
    // Native prefill consumes every prompt token, not just the decode seed.
    int tokens = native_prefill ? length : 1;
    for (int token = 0; token < tokens; token++)
      if (inputs[request * 64 + token] < 0 || inputs[request * 64 + token] >= vocab)
        throw std::runtime_error("Every consumed prompt/seed ID must be inside the embedding vocabulary");
  }

  struct Range { uintptr_t begin, end; };
  auto range = [](void const *ptr, size_t bytes) {
    uintptr_t begin = reinterpret_cast<uintptr_t>(ptr);
    if (bytes != 0 && (ptr == nullptr || bytes > UINTPTR_MAX - begin))
      throw std::runtime_error("Invalid request buffer range");
    return Range{begin, begin + bytes};
  };
  auto overlaps = [](Range a, Range b) {
    return a.begin < a.end && b.begin < b.end && a.begin < b.end && b.begin < a.end;
  };
  size_t n = c.num_decode_requests, rows = c.num_contexts, ns = c.num_request_slots;
  // These buffers must never be addressed by ordinary graph tasks. Cache
  // destinations are checked separately against the exact attention views.
  std::vector<Range> external = {
      range(lengths, n * sizeof(int32_t)),
      range(c.decode_generated_lengths, n * sizeof(int32_t))};
  std::vector<Range> context = {
      range(c.context_inputs, rows * sizeof(int64_t)),
      range(c.context_saved_outputs, rows * c.context_output_width * sizeof(__nv_bfloat16)),
      range(c.context_slot_inputs, ns * (native_prefill ? NAS_PREFILL_CHUNK_ROWS : 1) * sizeof(int64_t)),
      range(c.context_slot_outputs, ns * c.context_output_width * sizeof(__nv_bfloat16)),
      range(c.context_predicted_tokens, ns * sizeof(int64_t)),
      range(c.context_generated_tokens, rows * sizeof(int64_t))};
  std::vector<std::vector<bool>> seen;
  for (auto const &binding : g_request_kv_bindings) {
    if (binding.heads <= 0 || binding.heads > 65535 ||
        binding.slot_k == nullptr || binding.slot_v == nullptr)
      throw std::runtime_error("Invalid request KV binding");
    size_t stride = size_t(64) * binding.heads * 128;
    external.push_back(range(binding.prefix_k, n * stride * sizeof(__nv_bfloat16)));
    external.push_back(range(binding.prefix_v, n * stride * sizeof(__nv_bfloat16)));
    seen.emplace_back(ns * binding.heads, false);
  }
  for (size_t i = 0; i < external.size(); i++) {
    for (size_t j = 0; j < i; j++)
      if (overlaps(external[i], external[j]))
        throw std::runtime_error("Request metadata and prefix buffers must not overlap");
    for (auto const &other : context)
      if (overlaps(external[i], other))
        throw std::runtime_error("Request metadata/prefix buffers overlap ContextIO buffers");
  }
  for (size_t d = 0; d < data.size(); d++) {
    auto const &task = tasks[data[d].resident_task_id];
    // Reject aliases with weights, activations, and mutable slot caches.
    auto check_graph_view = [&](TensorDesc const &view) {
      size_t elements = 1;
      for (int axis = 0; axis < view.num_dims; axis++)
        elements += size_t(view.dim[axis] - 1) * view.stride[axis];
      size_t width = view.data_type == mirage::type::DT_BFLOAT16 ? 2 :
          view.data_type == mirage::type::DT_INT64 ? 8 : 4;
      Range memory = range(view.base_ptr, elements * width);
      for (auto const &buffer : external)
        if (overlaps(memory, buffer))
          throw std::runtime_error("Graph tensors must not alias request metadata or immutable prefixes");
    };
    for (int i = 0; i < task.num_inputs; i++) check_graph_view(data[d].inputs[i]);
    for (int i = 0; i < task.num_outputs; i++) check_graph_view(data[d].outputs[i]);
    // Native attention populates its private cache from position zero. It has
    // no prefix-copy source; private cache ownership was validated separately.
    if (native_prefill || task.task_type != TASK_ATTENTION_SLOT_HOPPER) continue;
    auto const &k = data[d].inputs[1];
    auto const &v = data[d].inputs[2];
    size_t matches = 0;
    for (size_t b = 0; b < g_request_kv_bindings.size(); b++) {
      auto const &binding = g_request_kv_bindings[b];
      size_t stride = size_t(64) * binding.heads * 128;
      uintptr_t base = reinterpret_cast<uintptr_t>(binding.slot_k);
      uintptr_t address = reinterpret_cast<uintptr_t>(k.base_ptr);
      if (address < base || address >= base + ns * stride * sizeof(__nv_bfloat16)) continue;
      size_t offset = (address - base) / sizeof(__nv_bfloat16);
      size_t head_offset = offset % stride;
      if ((address - base) % sizeof(__nv_bfloat16) != 0 ||
          offset / stride != slots[d] || head_offset % 128 != 0 ||
          head_offset >= size_t(binding.heads) * 128 ||
          k.stride[0] != stride || k.stride[1] != size_t(binding.heads) * 128 ||
          reinterpret_cast<uintptr_t>(v.base_ptr) !=
              reinterpret_cast<uintptr_t>(binding.slot_v) + offset * sizeof(__nv_bfloat16))
        throw std::runtime_error("Request KV binding does not match the native slot/head views");
      size_t member = slots[d] * binding.heads + head_offset / 128;
      if (seen[b][member])
        throw std::runtime_error("Request KV binding has multiple attention writers");
      seen[b][member] = true;
      ++matches;
    }
    if (matches != 1)
      throw std::runtime_error("Each attention task needs exactly one request KV binding");
  }
  for (auto const &members : seen)
    if (std::find(members.begin(), members.end(), false) != members.end())
      throw std::runtime_error("Request KV binding is not fully owned by native attention tasks");
}

static SlotRestartPlan make_slot_restart_plan(
    ResidentRuntimeConfig const &config,
    std::vector<TaskDesc> const &tasks,
    std::vector<DataDesc> const &data,
    std::vector<uint32_t> const &slots,
    std::vector<uint32_t> const &edge_offsets,
    std::vector<DataId> const &edge_targets,
    std::vector<TaskId> const &data_to_task,
    std::vector<DataId> const &first_data,
    size_t num_events) {
  SlotRestartPlan plan;
  char const *setting = std::getenv("MIRAGE_RESIDENT_SLOT_ITERATIONS");
  bool context_io = config.num_contexts >= 0;
  bool const autoregressive = config.context_predicted_tokens != nullptr;
  bool const request_queue = config.num_decode_requests >= 0;
  bool const native_prefill = config.prefill_chunk_capacity > 0;
  if (config.prefill_chunk_capacity < 0)
    throw std::runtime_error("Prefill chunk capacity must be zero (disabled) or positive");
  if (native_prefill && (!request_queue ||
      config.prefill_chunk_capacity != NAS_PREFILL_CHUNK_ROWS ||
      config.context_input_width != 1 || !config.context_input_is_int64))
    throw std::runtime_error("Native prefill requires scalar prompt IDs and matching chunk capacity");
  if (NAS_PREFILL_CHUNK_ROWS > 1 && !native_prefill &&
      (!context_io || config.context_sequence_length != 1 ||
       config.context_prefix_length != 0 || config.num_contexts != config.num_request_slots ||
       config.context_input_width != NAS_PREFILL_CHUNK_ROWS ||
       !config.context_input_is_int64 || autoregressive || request_queue))
    throw std::runtime_error("Fixed chunk prefill requires exactly one full ContextIO pass per slot");
  // An empty request queue may have a null zero-element history allocation.
  bool const has_history = config.context_generated_tokens != nullptr ||
      (request_queue && config.num_decode_requests == 0);
  if (autoregressive != has_history ||
      (autoregressive && (!context_io || config.context_input_is_int64 != 1 ||
                          (!request_queue && config.context_sequence_length <= 0))))
    throw std::runtime_error("Autoregressive feedback requires token buffers and fixed sequences or an offline request queue");
  if (request_queue && (!autoregressive || config.context_sequence_length != 0 ||
      config.context_prefix_length != 0 || config.num_decode_requests > INT32_MAX / 64 ||
      config.num_contexts != config.num_decode_requests * 64))
    throw std::runtime_error("Request queues require ContextIO rows=N*64 and their own per-request positions");
#if defined(NAS_COMPARE_SEQUENCE_REPEATS) && NAS_COMPARE_SEQUENCE_REPEATS != 1
  if (autoregressive)
    throw std::runtime_error("Autoregressive feedback supports one sequence per slot, without replacement repeats");
#endif
  if (config.context_sequence_length != 0 || config.context_prefix_length != 0) {
    // Check the native binding too, before device code can select a cache row.
    int steps = config.context_sequence_length, prefix = config.context_prefix_length;
    if (!context_io || steps < 1 || steps > 64 || prefix < 0 || prefix > 64 - steps ||
        config.num_request_slots <= 0 ||
        int64_t(config.num_request_slots) * steps
#ifdef NAS_COMPARE_SEQUENCE_REPEATS
            * NAS_COMPARE_SEQUENCE_REPEATS
#endif
            != config.num_contexts) {
      throw std::runtime_error("Fixed sequences require ContextIO rows=S*T, T>=1, and 0<=prefix<=64-T");
    }
  }
  if (!context_io && setting == nullptr) {
    return plan;
  }
  if (context_io && setting != nullptr) {
    throw std::runtime_error("Choose context_io or MIRAGE_RESIDENT_SLOT_ITERATIONS, not both");
  }
  if (config.num_request_slots <= 0 || config.num_gpus != 1 ||
      resident_uses_streaming(config) || config.num_remote_schedulers != 0 ||
      config.num_local_schedulers < 1 || config.num_workers < config.num_local_schedulers ||
      config.num_workers > MAX_NUM_WORKERS || num_events >= SLOT_DISPATCH_TAG ||
      static_cast<uint32_t>(config.num_request_slots) > SLOT_CONTROL_ID_MASK) {
    throw std::runtime_error("Slot restart requires an annotated single-GPU resident graph and local worker owners");
  }
  // Fixed sequences have T passes per slot; ordinary queue admission uses N
  // as an upper bound. Both retain the conservative whole-invocation bounds.
  if (context_io) {
#ifndef MODE_OFFLINE
    throw std::runtime_error("Context IO currently requires offline mode");
#endif
    if (config.num_request_slots > MPK_MAX_NUM_BATCHED_REQUESTS) {
      throw std::runtime_error("Context slots exceed request_ids allocation");
    }
    plan.limits.assign(config.num_request_slots, config.context_sequence_length > 0
        ? config.context_sequence_length
#ifdef NAS_COMPARE_SEQUENCE_REPEATS
            * NAS_COMPARE_SEQUENCE_REPEATS
#endif
        : config.num_contexts);
  } else {
    // Accept exactly one unsigned decimal limit per slot; zero disables a slot.
    std::string text(setting);
    size_t begin = 0;
    while (true) {
      size_t end = text.find(',', begin);
      if (end == std::string::npos) end = text.size();
      if (end == begin) throw std::runtime_error("Empty slot iteration limit");
      uint32_t value = 0u;
      for (size_t i = begin; i < end; i++) {
        if (text[i] < '0' || text[i] > '9' ||
            value > (UINT32_MAX - static_cast<uint32_t>(text[i] - '0')) / 10u) {
          throw std::runtime_error("Invalid uint32 slot iteration limit");
        }
        value = value * 10u + static_cast<uint32_t>(text[i] - '0');
      }
      plan.limits.push_back(value);
      if (end == text.size()) break;
      begin = end + 1;
    }
    if (plan.limits.size() != static_cast<size_t>(config.num_request_slots)) {
      throw std::runtime_error("Expected one iteration limit per request slot");
    }
  }

  std::vector<uint32_t> incoming(data.size(), 0u);
  std::vector<bool> is_root(data.size(), false), mapped_task(tasks.size(), false);
  for (DataId root : first_data) {
    if (root >= data.size() || is_root[root]) {
      throw std::runtime_error("Invalid or duplicate first DataId for slot restart");
    }
    is_root[root] = true;
  }
  uint64_t total_compute = 0u, total_passes = 0u;
  std::vector<uint32_t> tasks_per_slot(config.num_request_slots, 0u);
  std::vector<uint32_t> feedback_terminals(config.num_request_slots, 0u);
  for (uint32_t limit : plan.limits) total_passes += limit;
  for (DataId d = 0; d < data.size(); d++) {
    TaskId task = data_to_task[d];
    if (task < static_cast<TaskId>(config.num_control_tasks) ||
        task >= tasks.size() || mapped_task[task]) {
      throw std::runtime_error("Invalid DataId/task mapping for slot restart");
    }
    mapped_task[task] = true;
    // These generated operator variants take only fixed tensor descriptors.
    // Several other linear variants read global qo_indptr metadata; allowing
    // those without a serving preparation policy could silently skip work.
    TaskType type = tasks[task].task_type;
    if (autoregressive && edge_offsets[d] == edge_offsets[d + 1]) {
      // Completion may restart this slot only after its bound prediction exists.
      // This first feedback policy has one terminal argmax, owned by this slot.
      if (type != TASK_ARGMAX_ROW || data[d].output_ptrs[0] !=
              static_cast<void const *>(config.context_predicted_tokens + slots[d]))
        throw std::runtime_error("Autoregressive slot terminal must be its bound argmax output");
      ++feedback_terminals[slots[d]];
    }
    bool experimental_moe = false;
    bool joined_moe_task = false;
    uint32_t physical_work = 1u;
#ifdef MIRAGE_EXPERIMENTAL_MOE_BATCHING
    experimental_moe = config.experimental_moe != nullptr &&
        (type == TASK_MOE_SLOT_W13 || type == TASK_MOE_SLOT_W2 ||
         type == TASK_LINEAR_QKV_HOPPER || type == TASK_LINEAR_WO_HOPPER);
    // Host initialization checked these fixed, whole-slot tensor views. SiLU
    // processes R independent rows, and the sum combines them only after W2.
    joined_moe_task = config.experimental_moe != nullptr &&
        (type == TASK_SILU_MUL || type == TASK_MOE_WEIGHTED_SUM);
    if (experimental_moe &&
        (type == TASK_MOE_SLOT_W13 || type == TASK_MOE_SLOT_W2)) {
      // Native prefill's 64*N bound already counts all consumed token rows.
      // Charge R members per row, even if each gets its own physical batch.
      // Legacy fixed chunks are bounded by graph passes, so still charge P*R.
      physical_work = (native_prefill ? 1u : EXPERIMENTAL_MOE_CHUNK_ROWS) *
          EXPERIMENTAL_MOE_TOP_K;
    }
#endif
    // Residual and argmax views are checked by validate_context_io_bindings.
    // Keep these tasks scoped to ContextIO; the fixed-tensor policy does not
    // call those binding checks. Slot advancement itself needs no new behavior.
    bool context_residual = context_io && type == TASK_ELEMENTWISE_ADD_SM100;
    bool context_argmax = context_io && type == TASK_ARGMAX_ROW;
    bool context_gather = native_prefill && type == TASK_IDENTITY;
    if (type != TASK_RMS_NORM_HOPPER &&
        type != TASK_CHECKSUM_HOPPER &&
        type != TASK_EMBEDDING_INPUT &&
        type != TASK_SILU_MUL_ROW &&
        type != TASK_MOE_TOP1_ROW &&
        type != TASK_ATTENTION_SLOT_HOPPER &&
        type != TASK_LINEAR_SWAPAB_HOPPER &&
        type != TASK_LINEAR_CUTLASS_HOPPER &&
        type != TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER &&
        !experimental_moe && !joined_moe_task && !context_residual && !context_argmax &&
        !context_gather) {
      throw std::runtime_error("Slot restart supports descriptor-input embedding, row SiLU/multiply, fixed-prefix slot attention, Hopper checksum/RMSNorm/swapAB/CUTLASS linear tasks, and ContextIO residual add/row argmax");
    }
    if (data[d].dependent_event != EVENT_INVALID_ID &&
        is_nvshmem_event(data[d].dependent_event)) {
      throw std::runtime_error("Remote readiness is unsupported by slot restart");
    }
    total_compute += uint64_t(plan.limits[slots[d]]) * physical_work;
    tasks_per_slot[slots[d]] += physical_work;
    for (uint32_t e = edge_offsets[d]; e < edge_offsets[d + 1]; e++) {
      DataId successor = edge_targets[e];
      if (successor >= data.size() || slots[successor] != slots[d]) {
        throw std::runtime_error("Invalid or cross-slot restart dependency");
      }
      incoming[successor]++;
    }
  }
  if (autoregressive) {
    for (uint32_t count : feedback_terminals) {
      if (count != 1u)
        throw std::runtime_error("Autoregressive slots require exactly one terminal argmax");
    }
  }
  for (size_t task = config.num_control_tasks; task < tasks.size(); task++) {
    if (!mapped_task[task]) throw std::runtime_error("Unmapped compute task in slot restart graph");
  }
  if (context_io) {
    // A fast slot may consume almost the whole queue. Bound every queue by N
    // passes of the largest slot, without assuming balanced admission.
    uint32_t largest = 0u;
    for (uint32_t count : tasks_per_slot) largest = std::max(largest, count);
    total_passes = static_cast<uint64_t>(config.num_contexts);
    // Native prefill consumes at most 64 input rows per request. Ordinary
    // operators run no more often than that, and MoE charges R members per
    // consumed row above. This avoids multiplying the same row bound by P.
    total_compute = total_passes * largest;
  }
  // Existing rings have no consumer backpressure. Conservatively bound the
  // entire invocation below each ring's capacity, even if one worker/owner
  // receives every item. This policy cannot overwrite a slow consumer's slot.
  // Scheduler 0 receives bootstrap, one completion and one dispatch per pass,
  // and termination. Other schedulers receive only dispatches and termination.
  if (total_compute + 1u >= config.per_worker_queue_len ||
      total_compute >= config.per_completion_queue_len ||
      2u * total_passes + 2u >= config.per_sched_queue_len) {
    throw std::runtime_error("Bounded slot run exceeds existing queue capacity; reduce iteration limits or context count");
  }
  std::vector<DataId> order;
  for (DataId d = 0; d < data.size(); d++) {
    if (data[d].initial_predecessor_count < 0 ||
        static_cast<uint32_t>(data[d].initial_predecessor_count) != incoming[d] ||
        is_root[d] != (incoming[d] == 0u)) {
      throw std::runtime_error("Slot restart roots/predecessor counts disagree with data edges");
    }
    if (incoming[d] == 0u) order.push_back(d);
  }
  for (size_t i = 0; i < order.size(); i++) {
    DataId d = order[i];
    for (uint32_t e = edge_offsets[d]; e < edge_offsets[d + 1]; e++) {
      DataId successor = edge_targets[e];
      if (--incoming[successor] == 0u) order.push_back(successor);
    }
  }
  if (order.size() != data.size()) {
    throw std::runtime_error("Slot restart requires an acyclic data graph");
  }
  plan.data_offsets.assign(config.num_request_slots + 1, 0u);
  for (uint32_t slot : slots) plan.data_offsets[slot + 1]++;
  for (size_t i = 1; i < plan.data_offsets.size(); i++) {
    plan.data_offsets[i] += plan.data_offsets[i - 1];
  }
  plan.data_ids.resize(data.size());
  std::vector<uint32_t> cursor = plan.data_offsets;
  for (DataId d : order) plan.data_ids[cursor[slots[d]]++] = d;
  return plan;
}
