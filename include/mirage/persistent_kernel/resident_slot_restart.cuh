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

__device__ __forceinline__ bool
prepare_slot(ResidentRuntimeConfig const &config,
             uint32_t slot, uint32_t completed_epoch) {
  if (config.num_contexts >= 0) {
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
    // Admission copies a single int64 for embedding input. The large table row
    // is gathered later by a regular worker task, after this slot is enabled.
    int context;
    if (config.context_input_is_int64) {
      context = get_new_input(
          static_cast<int64_t const *>(config.context_inputs), config.num_contexts, 1,
          input_cursor, static_cast<int64_t *>(config.context_slot_inputs) + slot);
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
    if (task.num_inputs != 3 || task.num_outputs != 3 ||
        slots[d] >= static_cast<uint32_t>(c.num_request_slots) ||
        data[d].task_metadata.request_id != static_cast<int>(slots[d]))
      throw std::runtime_error("Slot attention V0 descriptor or slot metadata mismatch");
    auto const &q = data[d].inputs[0];
    auto const &o = data[d].outputs[0];
    auto const &k = data[d].inputs[1];
    int heads = k.stride[1] / 128;
    auto row = [&](TensorDesc const &t, int width) {
      return t.base_ptr != nullptr && t.data_type == mirage::type::DT_BFLOAT16 &&
             t.num_dims == 2 && t.dim[0] == 1 && t.dim[1] == width &&
             int64_t(t.stride[0]) == int64_t(heads) * width && t.stride[1] == 1;
    };
    if (heads < 1 || !row(q, 768) || !row(o, 512))
      throw std::runtime_error("Slot attention V0 requires one GQA4/D128 query slice");
    for (int i = 1; i <= 2; i++) {
      auto const &view = data[d].inputs[i];
      if (view.base_ptr == nullptr || view.data_type != mirage::type::DT_BFLOAT16 ||
          view.num_dims != 4 || view.dim[0] != 1 || view.dim[1] != 64 ||
          view.dim[2] != 1 || view.dim[3] != 128 ||
          int64_t(view.stride[0]) != 64ll * heads * 128 ||
          int64_t(view.stride[1]) != int64_t(heads) * 128 ||
          view.stride[2] != 128 || view.stride[3] != 1 ||
          !same_view(view, data[d].outputs[i]))
        throw std::runtime_error("Slot attention V0 requires exact in-place private K/V page views");
      // Different heads interleave inside a page. A single bounding interval
      // would falsely overlap them; each token/head row is exactly 256 bytes.
      uintptr_t base = reinterpret_cast<uintptr_t>(view.base_ptr);
      for (int token = 0; token < 64; token++) {
        uintptr_t begin = base + size_t(token) * view.stride[1] * 2;
        cache_rows.push_back({begin, begin + 256});
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
      if (task.task_type == TASK_ATTENTION_SLOT_HOPPER && index > 0) return;
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
  if ((c.context_input_is_int64 != 0 && c.context_input_is_int64 != 1) ||
      (c.context_input_is_int64 && c.context_input_width != 1)) {
    throw std::runtime_error("Integer ContextIO input must be one int64 ID per slot");
  }
  if (c.context_input_width <= 0 || c.context_output_width <= 0 ||
      c.context_slot_inputs == nullptr || c.context_slot_outputs == nullptr ||
      (c.num_contexts > 0 &&
       (c.context_inputs == nullptr || c.context_saved_outputs == nullptr))) {
    throw std::runtime_error("Invalid context IO pointers or widths");
  }
  uintptr_t base[] = {reinterpret_cast<uintptr_t>(c.context_inputs),
                      reinterpret_cast<uintptr_t>(c.context_saved_outputs),
                      reinterpret_cast<uintptr_t>(c.context_slot_inputs),
                      reinterpret_cast<uintptr_t>(c.context_slot_outputs)};
  size_t input_element_bytes = c.context_input_is_int64 ? sizeof(int64_t) : sizeof(__nv_bfloat16);
  size_t row[] = {size_t(c.context_input_width) * input_element_bytes,
                 size_t(c.context_output_width) * sizeof(__nv_bfloat16)};
  size_t bytes[] = {size_t(c.num_contexts) * row[0], size_t(c.num_contexts) * row[1],
                    size_t(c.num_request_slots) * row[0], size_t(c.num_request_slots) * row[1]};
  auto overlaps = [](uintptr_t a, size_t n, uintptr_t b, size_t m) {
    return n != 0 && m != 0 && a < b + m && b < a + n;
  };
  for (int i = 0; i < 4; i++) {
    for (int j = i + 1; j < 4; j++) {
      if (overlaps(base[i], bytes[i], base[j], bytes[j]))
        throw std::runtime_error("Context IO buffers must not overlap");
    }
  }
  std::vector<bool> reads_input(c.num_request_slots, false);
  std::vector<bool> writes_output(c.num_request_slots, false);
  for (size_t d = 0; d < data.size(); d++) {
    auto const &task = tasks[data[d].resident_task_id];
    auto check = [&](TensorDesc const &t, bool output, int index) {
      // Top-1 owns a scalar FP32 weight and an int32 routing row. Permit exactly
      // these operator positions rather than making all ContextIO views typeless.
      bool top1 = task.task_type == TASK_MOE_TOP1_ROW;
      bool logical_moe = task.task_type == TASK_MOE_SLOT_W13 ||
                         task.task_type == TASK_MOE_SLOT_W2;
      bool route_view = (top1 && output && index == 1) ||
                        (logical_moe && !output && index == 2);
      bool route_weight = top1 && output && index == 0;
      bool allowed_scratch = (route_view && t.data_type == mirage::type::DT_INT32) ||
                             (route_weight && t.data_type == mirage::type::DT_FLOAT32);
      if ((t.data_type != mirage::type::DT_BFLOAT16 &&
           t.data_type != mirage::type::DT_INT64 && !allowed_scratch) || t.num_dims <= 0)
        throw std::runtime_error("Context IO task tensor dtype is not allowed for this operator position");
      if (route_view || route_weight) {
        int width = route_view ? 128 : 1;
        int dtype = route_view ? mirage::type::DT_INT32 : mirage::type::DT_FLOAT32;
        if (t.data_type != dtype || t.num_dims != 2 || t.dim[0] != 1 ||
            t.dim[1] != width || t.stride[0] != width || t.stride[1] != 1)
          throw std::runtime_error("Experimental routing requires one private contiguous top-1 row");
      }
      // Integer views have one supported use: the embedding task's ID input.
      // Keep BF16-only compute from reinterpreting an int64 activation/weight.
      bool embedding_id = task.task_type == TASK_EMBEDDING_INPUT && !output && index == 0;
      if ((t.data_type == mirage::type::DT_INT64) != embedding_id)
        throw std::runtime_error("Only descriptor-input embedding may read int64 ContextIO task views");
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
        if (i == 2 && !output) reads_input[slots[d]] = true;
        if (i == 3 && output) writes_output[slots[d]] = true;
      }
    };
    for (int i = 0; i < task.num_inputs; i++) check(data[d].inputs[i], false, i);
    for (int i = 0; i < task.num_outputs; i++) check(data[d].outputs[i], true, i);
  }
  for (int slot = 0; slot < c.num_request_slots; slot++) {
    if (!reads_input[slot] || !writes_output[slot])
      throw std::runtime_error("Every slot must read its context input and write its complete output row");
  }
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
    bool experimental_moe = false;
#ifdef MIRAGE_EXPERIMENTAL_MOE_BATCHING
    experimental_moe = config.experimental_moe != nullptr &&
        (type == TASK_MOE_SLOT_W13 || type == TASK_MOE_SLOT_W2 ||
         type == TASK_LINEAR_QKV_HOPPER || type == TASK_LINEAR_WO_HOPPER);
#endif
    if (type != TASK_RMS_NORM_HOPPER &&
        type != TASK_CHECKSUM_HOPPER &&
        type != TASK_EMBEDDING_INPUT &&
        type != TASK_SILU_MUL_ROW &&
        type != TASK_MOE_TOP1_ROW &&
        type != TASK_ATTENTION_SLOT_HOPPER &&
        type != TASK_LINEAR_SWAPAB_HOPPER &&
        type != TASK_LINEAR_CUTLASS_HOPPER &&
        type != TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER && !experimental_moe) {
      throw std::runtime_error("Slot restart supports descriptor-input embedding, row SiLU/multiply, fixed-prefix slot attention, Hopper checksum/RMSNorm/swapAB/CUTLASS linear tasks");
    }
    if (data[d].dependent_event != EVENT_INVALID_ID &&
        is_nvshmem_event(data[d].dependent_event)) {
      throw std::runtime_error("Remote readiness is unsupported by slot restart");
    }
    total_compute += plan.limits[slots[d]];
    tasks_per_slot[slots[d]]++;
    for (uint32_t e = edge_offsets[d]; e < edge_offsets[d + 1]; e++) {
      DataId successor = edge_targets[e];
      if (successor >= data.size() || slots[successor] != slots[d]) {
        throw std::runtime_error("Invalid or cross-slot restart dependency");
      }
      incoming[successor]++;
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
