#pragma once

// Explicitly experimental fixed-shape adapter, compiled only on request. The
// ordinary resident paths below this include remain the default implementation.
#include "fixed_moe_adapter.cuh"
#ifdef NAS_COMPARE_TASK_TIMING
#include "task_timing.cuh"
#endif
#include "experimental_moe_scheduler_trace.cuh"
#if defined(NAS_INVESTIGATE_ALL_ROWS) && !defined(NAS_INVESTIGATE_LOCKSTEP_RESTART)
#error "All-row gating requires slot-end lockstep"
#endif
#include "experimental_moe_selection.h"
#include "experimental_moe_occupancy.cuh"
#include <cstdlib>
#include <limits>

__device__ __forceinline__ unsigned long long experimental_moe_now() {
  unsigned long long now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  return now;
}

// A published queue word makes mutable descriptors and outputs visible. Keep
// both the GPU acquire and a compiler memory barrier in this experimental path;
// the existing baseline queue helpers remain unchanged.
__device__ __forceinline__ unsigned long long experimental_moe_acquire(
    unsigned long long *address) {
  unsigned long long value;
  asm volatile("ld.acquire.gpu.u64 %0, [%1];"
               : "=l"(value) : "l"(address) : "memory");
  return value;
}

__host__ __device__ __forceinline__ int experimental_moe_kind(TaskType type) {
  switch (type) {
    case TASK_MOE_SLOT_W13: return EXPERIMENTAL_BATCH_W13;
    case TASK_MOE_SLOT_W2: return EXPERIMENTAL_BATCH_W2;
    case TASK_LINEAR_QKV_HOPPER: return EXPERIMENTAL_BATCH_QKV;
    case TASK_LINEAR_WO_HOPPER: return EXPERIMENTAL_BATCH_WO;
    default: return -1;
  }
}

__device__ __forceinline__ bool experimental_moe_item(TaskId item) {
  return (static_cast<uint32_t>(item) & EXPERIMENTAL_MOE_ITEM_TAG) != 0u;
}

__device__ __forceinline__ ExperimentalMoeBatch &experimental_moe_batch(
    ResidentRuntimeConfig const &config, TaskId item) {
  uint32_t index = static_cast<uint32_t>(item) & ~EXPERIMENTAL_MOE_ITEM_TAG;
  require_request_slot_invariant(index < EXPERIMENTAL_MOE_POOL);
  ExperimentalMoeBatch &batch = config.experimental_moe->batches[index];
  require_request_slot_invariant(batch.live == 1u &&
      batch.sequence == static_cast<uint32_t>(item >> 32) && batch.count > 0u &&
      batch.count <= EXPERIMENTAL_MOE_BATCH_CAPACITY &&
      batch.op_id < EXPERIMENTAL_BATCH_OPERATORS);
  auto const &op = config.experimental_moe->operators[batch.op_id];
  require_request_slot_invariant(op.present && op.enabled && batch.expert < op.num_experts &&
      batch.expert_count >= 1u && batch.expert_count <= 2u &&
      batch.mask[0] == static_cast<int32_t>(batch.expert) &&
      batch.mask[op.num_experts] == static_cast<int32_t>(batch.expert_count));
  if (batch.expert_count == 2u)
    require_request_slot_invariant(op.kind < EXPERIMENTAL_BATCH_QKV &&
        batch.mask[1] >= 0 && static_cast<uint32_t>(batch.mask[1]) < op.num_experts &&
        batch.mask[0] != batch.mask[1]);
  return batch;
}

// All metadata writes in this helper belong to the single scheduler. A pending
// bit expresses undispatched work; an immutable descriptor tracks dispatched work.
__device__ __forceinline__ void experimental_moe_ready(
    ResidentRuntimeConfig const &config, DataId data, uint32_t epoch,
    int &next_worker, size_t *worker_positions, NasMoeOccupancy &occupancy
#ifdef NAS_INVESTIGATE_ALL_ROWS_COUNTER
    , uint32_t (&ready_count)[EXPERIMENTAL_BATCH_OPERATORS], uint32_t cohort_epoch
#endif
    ) {
  validate_request_slot_iteration(config, data, epoch);
  TaskId task = config.data_to_task_id[data];
  ExperimentalMoeState &state = *config.experimental_moe;
  int32_t op_id = state.data_to_op_id[data];
  require_request_slot_invariant(op_id == -1 ||
      (op_id >= 0 && static_cast<uint32_t>(op_id) < EXPERIMENTAL_BATCH_OPERATORS));
  if (op_id < 0 || !state.operators[op_id].enabled) {
    // Marked QKV/W_O nodes retain ordinary dense CUDA bodies. Disabling a bin
    // changes physical dispatch only; the native dependencies remain identical.
    enqueue_worker_item(config, next_worker, &worker_positions[next_worker],
                        compute_task_id(epoch, task));
    next_worker = (next_worker + 1) % config.num_workers;
    return;
  }
  uint32_t slot = config.data_slot_ids[data];
  require_request_slot_invariant(slot < EXPERIMENTAL_MOE_STATE_SLOTS);
  auto const &op = state.operators[op_id];
  require_request_slot_invariant(op.present && op.kind < EXPERIMENTAL_BATCH_KINDS);
  auto &row = state.pending[op.kind][slot];
  require_request_slot_invariant(row.pending == 0u);
#ifdef NAS_INVESTIGATE_ALL_ROWS_COUNTER
  require_request_slot_invariant(epoch == cohort_epoch && cohort_epoch != 0u &&
      ready_count[op_id] < EXPERIMENTAL_MOE_STATE_SLOTS);
#endif
  // Dense projections have expert zero and require no routing tensor or scan.
  int expert = op.num_experts == 1u ? 0 : -1;
  if (op.num_experts != 1u) {
    for (uint32_t e = 0; e < op.num_experts; e++) {
      int rank = op.routing[slot * op.num_experts + e];
      require_request_slot_invariant(rank == 0 || rank == 1);
      if (rank == 1) {
        require_request_slot_invariant(expert == -1);
        expert = e;
      }
    }
  }
  require_request_slot_invariant(expert >= 0);
  row.data_id = data;
  row.epoch = epoch;
  row.expert = expert;
  row.op_id = op_id;
  row.ready_ns = experimental_moe_now();
  row.pending = 1u;
#ifdef NAS_COMPARE_TASK_TIMING
  compare_async_row_time(op_id, slot, epoch, 0u, row.ready_ns);
#endif
#ifdef NAS_INVESTIGATE_ALL_ROWS_COUNTER
  // This counts arrivals in the cohort, not rows still waiting in the table.
  // One controller inserts each op/slot once; dispatch must not decrement it.
  ++ready_count[op_id];
#endif
  occupancy.ready(op.kind, row.ready_ns);
  // Record the stored readiness time itself, so a diagnostic trace can replay
  // timeout and oldest-member decisions without a later clock sample's skew.
  ETRACE_AT(row.ready_ns, 70, epoch, task, data, -1, (op_id << 16) | expert);
}

// Scalar version of the scheduler's logical completion path. Batch members and
// ordinary tasks use exactly this function; it contains no block-wide barrier.
__device__ __forceinline__ void experimental_moe_complete_logical(
    ResidentRuntimeConfig const &config, DataId data, uint32_t epoch,
    int &next_worker, size_t *worker_positions, NasMoeOccupancy &occupancy
#ifdef NAS_INVESTIGATE_ALL_ROWS_COUNTER
    , uint32_t (&ready_count)[EXPERIMENTAL_BATCH_OPERATORS], uint32_t cohort_epoch
#endif
    ) {
  validate_data_completion(config, data, epoch);
  atomicAdd(config.completed_data_this_iteration_count, 1u);
  DataId begin = config.data_edge_offsets[data];
  DataId end = config.data_edge_offsets[data + 1];
  if (begin == end) {
    report_terminal_data_completion(config, data, epoch, -1);
    return;
  }
  for (DataId edge = begin; edge < end; edge++) {
    DataId successor = config.data_edge_targets[edge];
    unsigned int remaining = atomicSub(
        &config.data_pending_predecessor_counts[successor], 1u);
    require_request_slot_invariant(remaining > 0u);
    if (remaining == 1u) {
      ETRACE(3, epoch, config.data_to_task_id[successor], successor, -1, data);
      experimental_moe_ready(config, successor, epoch, next_worker,
                             worker_positions, occupancy
#ifdef NAS_INVESTIGATE_ALL_ROWS_COUNTER
                             , ready_count, cohort_epoch
#endif
                             );
    }
  }
}

__device__ __forceinline__ bool experimental_moe_dispatch(
    ResidentRuntimeConfig const &config, int &next_worker,
    size_t *worker_positions, NasMoeOccupancy &occupancy
#ifdef NAS_INVESTIGATE_ALL_ROWS
    , uint32_t &open_ops, uint32_t cohort_epoch
#endif
#ifdef NAS_INVESTIGATE_ALL_ROWS_COUNTER
    , uint32_t const (&ready_count)[EXPERIMENTAL_BATCH_OPERATORS]
#endif
    ) {
  ExperimentalMoeState &state = *config.experimental_moe;
  int free_batch = -1;
  for (uint32_t b = 0; b < EXPERIMENTAL_MOE_POOL; b++) {
    if (state.batches[b].live == 0u) { free_batch = b; break; }
  }
  if (free_batch < 0) return false;

#ifdef NAS_INVESTIGATE_ALL_ROWS_COUNTER
  require_request_slot_invariant(experimental_moe_latch_all_rows(
      ready_count, state.operators, cohort_epoch, open_ops));
#elif defined(NAS_INVESTIGATE_ALL_ROWS)
  require_request_slot_invariant(experimental_moe_latch_all_rows(
      state.pending, state.operators, cohort_epoch, open_ops));
#endif
  unsigned long long now = experimental_moe_now();
  uint32_t eligible_ops = ~0u;
#ifdef NAS_INVESTIGATE_ALL_ROWS
  eligible_ops = open_ops;
#endif
  auto selected = experimental_moe_select(state.pending, state.operators, now,
      eligible_ops, state.early_coalesce != 0u, state.in_flight,
      state.deadline_coalesce != 0u);
  if (selected.op_id < 0) return false;
  auto const &op = state.operators[selected.op_id];

  auto &batch = state.batches[free_batch];
  require_request_slot_invariant(state.next_sequence != UINT32_MAX);
  batch.sequence = ++state.next_sequence;
  batch.op_id = selected.op_id;
  batch.expert = selected.expert;
  batch.count = selected.count;
  require_request_slot_invariant(batch.count > 0u &&
      batch.count <= EXPERIMENTAL_MOE_BATCH_CAPACITY);
  batch.expert_count = selected.second_expert == UINT32_MAX ? 1u : 2u;
  batch.mask[0] = batch.expert;
  if (batch.expert_count == 2u) batch.mask[1] = selected.second_expert;
  batch.mask[op.num_experts] = batch.expert_count;
  // Initialize only the expert rows the kernel will read. Other rows may retain
  // old bytes: the immutable active-expert list makes them unreachable.
  for (uint32_t e = 0; e < batch.expert_count; ++e)
    for (uint32_t j = 0; j < EXPERIMENTAL_MOE_BATCH_CAPACITY; ++j)
      batch.routing[batch.mask[e] * EXPERIMENTAL_MOE_BATCH_CAPACITY + j] = 0;
  for (uint32_t j = 0; j < EXPERIMENTAL_MOE_BATCH_CAPACITY; ++j)
    batch.state_rows[j] = UINT32_MAX;
  if (selected.reason == 3u)
    require_request_slot_invariant(state.early_coalesce && state.in_flight < 2u);
  if (batch.expert_count == 2u)
    require_request_slot_invariant(op.kind < EXPERIMENTAL_BATCH_QKV &&
        selected.group_count == batch.count && // Both whole bins fit.
        ((selected.reason == 3u && state.early_coalesce) ||
         (selected.reason == 2u && state.deadline_coalesce)));

  // A compact local row maps to an independently owned state row. Keep every
  // member captured until its physical batch retires; only selected pending
  // entries are removed below. Leftovers retain their original ready_ns.
  for (uint32_t j = 0; j < batch.count; j++) {
    uint32_t slot = selected.slots[j];
    require_request_slot_invariant(slot < EXPERIMENTAL_MOE_STATE_SLOTS);
    auto const &row = state.pending[op.kind][slot];
    require_request_slot_invariant(row.pending && row.op_id == batch.op_id &&
        (row.expert == batch.expert || row.expert == selected.second_expert) &&
        config.data_slot_ids[row.data_id] == slot);
    validate_request_slot_iteration(config, row.data_id, row.epoch);
    for (uint32_t earlier = 0; earlier < j; earlier++)
      require_request_slot_invariant(batch.state_rows[earlier] != slot);
#ifdef NAS_COMPARE_TASK_TIMING
    compare_async_row_time(batch.op_id, slot, row.epoch, 1u, now, selected.reason);
#endif
    batch.members[j] = {slot, row.data_id, row.epoch};
    batch.state_rows[j] = slot;
    batch.routing[row.expert * EXPERIMENTAL_MOE_BATCH_CAPACITY + j] = 1;
  }
  batch.live = 1u;
  state.in_flight++;
  int worker = next_worker;
  TaskId item = compute_task_id(batch.sequence,
      EXPERIMENTAL_MOE_ITEM_TAG | static_cast<uint32_t>(free_batch));
  // Kind 77 is a decision snapshot, not a DataId event: data=group_count,
  // task=op_id, epoch=sequence. Its timestamp is the exact eligibility clock.
  ETRACE_AT(now, 77, batch.sequence, batch.op_id, selected.group_count,
            worker, batch.expert);
  ETRACE(71, batch.sequence, batch.op_id, DATA_INVALID_ID, worker,
         (selected.reason << 16) | batch.count);
  for (uint32_t m = 0; m < batch.count; m++) {
    auto const &member = batch.members[m];
    ETRACE(72, member.epoch, config.data_to_task_id[member.data_id], member.data_id,
           worker, batch.sequence);
    state.pending[op.kind][member.slot].pending = 0u;
  }
  occupancy.dispatch(op.kind, batch.count, selected.reason);

  // Runtime validation bounds the entire finite invocation below ring capacity;
  // no producer can overwrite an unread word. Only this controller enqueues.
  size_t position = worker_positions[worker]++;
  st_relaxed_gpu_u64(&config.worker_queues[worker]
      [position % config.per_worker_queue_len], item);
  __threadfence();
  // Publish intent precedes the release itself. It cannot follow worker start,
  // but it is not a claim that the ready tail has already been incremented.
  ETRACE(76, batch.sequence, batch.op_id, DATA_INVALID_ID, worker, batch.expert);
  atom_add_release_gpu_u64(&config.worker_queue_last_ready_task_id[worker], 1);
  next_worker = (next_worker + 1) % config.num_workers;
  return true;
}

__device__ __forceinline__ void execute_worker_experimental_moe(
    ResidentRuntimeConfig config) {
  // Loading one item at a time keeps tagged batch descriptors out of the old
  // TaskDesc prefetch path. This branch is only for the opt-in experiment.
  __shared__ TaskId item;
  __shared__ TaskDesc task;
  unsigned long long cursor = 0u;
  int worker = blockIdx.x;
  while (true) {
    if (threadIdx.x == 0) {
      while (experimental_moe_acquire(&config.worker_queue_last_ready_task_id[worker]) <= cursor)
        __nanosleep(10);
      item = ld_relaxed_gpu_u64(&config.worker_queues[worker]
          [cursor % config.per_worker_queue_len]);
      if (!experimental_moe_item(item)) {
        uint32_t index = get_task_position_index(item);
        task = config.all_tasks[index];
      }
    }
    __syncthreads();
    if (experimental_moe_item(item)) {
      auto &batch = experimental_moe_batch(config, item);
      if (threadIdx.x == 0) {
        ETRACE(73, batch.sequence, batch.op_id, DATA_INVALID_ID, worker, batch.expert);
        for (uint32_t m = 0; m < batch.count; m++) {
          auto const &member = batch.members[m];
          require_request_slot_invariant(member.slot < EXPERIMENTAL_MOE_STATE_SLOTS &&
              batch.state_rows[m] == member.slot);
          validate_request_slot_iteration(config, member.data_id, member.epoch);
          ETRACE(4, member.epoch, config.data_to_task_id[member.data_id],
                 member.data_id, worker, batch.sequence);
        }
      }
      __syncthreads();
      auto const &op = config.experimental_moe->operators[batch.op_id];
#ifdef NAS_COMPARE_TASK_TIMING
      // The descriptor has been acquired, members validated, and the existing
      // CTA barrier passed. Time only this physical projection and its final
      // barrier; leave queue waiting and controller fanout outside the interval.
      unsigned long long task_timing_start = 0u;
      if (threadIdx.x == 0 && compare_task_enabled())
        task_timing_start = compare_task_now();
#endif
      // Native bindings were checked against these compiled dimensions before
      // launch. The operation record selects tensors; kind selects CUDA code.
      switch (op.kind) {
        case EXPERIMENTAL_BATCH_W13:
          nas_moe_probe::moe_linear<true, EXPERIMENTAL_MOE_H, EXPERIMENTAL_MOE_I, true>(
              op.weight_map, op.input, batch.routing, batch.mask,
              op.output, batch.state_rows, batch.count);
          break;
        case EXPERIMENTAL_BATCH_W2:
          nas_moe_probe::moe_linear<false, EXPERIMENTAL_MOE_H, EXPERIMENTAL_MOE_I, true>(
              op.weight_map, op.input, batch.routing, batch.mask,
              op.output, batch.state_rows, batch.count);
          break;
        case EXPERIMENTAL_BATCH_QKV:
          nas_moe_probe::dense_linear<EXPERIMENTAL_MOE_H, 3 * EXPERIMENTAL_MOE_H / 2>(
              op.weight_map, op.input, batch.routing, batch.mask,
              op.output, batch.state_rows, batch.count);
          break;
        case EXPERIMENTAL_BATCH_WO:
          nas_moe_probe::dense_linear<EXPERIMENTAL_MOE_H, EXPERIMENTAL_MOE_H>(
              op.weight_map, op.input, batch.routing, batch.mask,
              op.output, batch.state_rows, batch.count);
          break;
        default:
          require_request_slot_invariant(false);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
#ifdef NAS_COMPARE_TASK_TIMING
        if (task_timing_start != 0u) {
          unsigned long long done = compare_task_now();
          compare_async_task_finish(worker, batch, task_timing_start, done);
        }
#endif
        for (uint32_t m = 0; m < batch.count; m++) {
          auto const &member = batch.members[m];
          ETRACE(5, member.epoch, config.data_to_task_id[member.data_id],
                 member.data_id, worker, batch.sequence);
        }
        ETRACE(74, batch.sequence, batch.op_id, DATA_INVALID_ID, worker, batch.expert);
        // This is the worker's final descriptor access. Publication includes the
        // captured sequence; the controller fans out captured per-member epochs.
        publish_completion_item(config, 0, static_cast<uint32_t>(item),
                                static_cast<uint32_t>(item >> 32));
      }
    } else {
      if (task.task_type == TASK_TERMINATE) return;
      DataId data = config.task_to_data_id[get_task_position_index(item)];
      uint32_t epoch = get_task_iteration_num(item);
      if (threadIdx.x == 0) {
        validate_request_slot_iteration(config, data, epoch);
        ETRACE(4, epoch, get_task_position_index(item), data, worker, 0);
      }
      __syncthreads();
      _execute_task(&task, config);
      if (threadIdx.x == 0) ETRACE_DELAY(epoch, get_task_position_index(item), data, worker);
      __syncthreads();
      if (threadIdx.x == 0) {
        ETRACE(5, epoch, get_task_position_index(item), data, worker, 0);
        publish_completion_item(config, 0, data, epoch);
      }
    }
    __syncthreads();
    cursor++;
  }
}

#ifdef NAS_INVESTIGATE_LOCKSTEP_RESTART
#include "lockstep_restart.cuh"
#endif

__device__ __forceinline__ void execute_scheduler_experimental_moe(
    ResidentRuntimeConfig config) {
  if (threadIdx.x != 0) return;
  require_request_slot_invariant(blockIdx.x == 0 && config.num_local_schedulers == 1);
  NAS_SCHED_TRACE_INIT();
  NasMoeOccupancy occupancy(config.num_request_slots, config.num_contexts);
  size_t worker_positions[MAX_WORKER_PER_SCHEDULER] = {};
  unsigned long long completion_cursor = 0u, event_cursor = 0u;
  int next_worker = 0;
#ifdef NAS_INVESTIGATE_LOCKSTEP_RESTART
  // A local held[] array raises the scheduler launch's local-memory demand.
  // Persistent workers are already running when that launch is submitted.
  // Keep this single-owner state in the scheduler CTA's shared memory instead.
  __shared__ NasInvestigationLockstepRestart lockstep;
  lockstep.initialize();
#endif
  while (true) {
    NAS_SCHED_TRACE_LOOP();
    bool progress = false;
    // Bounded polling is deliberate: completions cannot starve restart/control
    // events, and an empty event queue cannot stop timeout-driven dispatch.
    // Investigation only: service a bounded completion burst before the next
    // full pending-table scan. Control and dispatch still run every outer loop.
    // Default 1 preserves the existing completion/dispatch order.
#ifndef NAS_INVESTIGATE_COMPLETION_BUDGET
#define NAS_INVESTIGATE_COMPLETION_BUDGET 1
#endif
    static_assert(NAS_INVESTIGATE_COMPLETION_BUDGET >= 1 &&
                  NAS_INVESTIGATE_COMPLETION_BUDGET <= 64,
                  "Investigation completion budget must be bounded");
    for (uint32_t served = 0; served < NAS_INVESTIGATE_COMPLETION_BUDGET; ++served) {
    NAS_SCHED_TRACE_BEGIN(poll_completion);
    bool completion_ready = completion_cursor <
        experimental_moe_acquire(&config.completion_queue_last_ready_data_id[0]);
    NAS_SCHED_TRACE_END(poll_completion, NAS_SCHED_POLL_COMPLETION,
                       completion_ready, DATA_INVALID_ID, 0, 0);
    if (!completion_ready) break;
    if (completion_ready) {
      NAS_SCHED_TRACE_BEGIN(handle_completion);
      CompletionItem item = ld_relaxed_gpu_u64(&config.completion_queues[0]
          [completion_cursor++ % config.per_completion_queue_len]);
      bool batch_completion = experimental_moe_item(item);
      if (batch_completion) {
        auto &batch = experimental_moe_batch(config, item);
        for (uint32_t m = 0; m < batch.count; m++) {
          auto const &member = batch.members[m];
          NAS_SCHED_TRACE_BEGIN(complete_logical);
          experimental_moe_complete_logical(config, member.data_id, member.epoch,
                                            next_worker, worker_positions, occupancy
#ifdef NAS_INVESTIGATE_ALL_ROWS_COUNTER
                                            , lockstep.ready_count, lockstep.epoch
#endif
                                            );
          NAS_SCHED_TRACE_END(complete_logical, NAS_SCHED_COMPLETE_LOGICAL, 0,
                             member.data_id, member.epoch, batch.sequence);
        }
        ETRACE(75, batch.sequence, batch.op_id, DATA_INVALID_ID, -1, batch.count);
        // Consumers can execute now, but this single scheduler cannot handle
        // their completions/restart until every member has been released here.
        batch.live = 0u;
        require_request_slot_invariant(config.experimental_moe->in_flight > 0u);
        config.experimental_moe->in_flight--;
        occupancy.retire(config.experimental_moe->operators[batch.op_id].kind, batch.count);
      } else {
        NAS_SCHED_TRACE_BEGIN(complete_logical);
        experimental_moe_complete_logical(config, get_completion_data_id(item),
            get_completion_iteration_num(item), next_worker, worker_positions, occupancy
#ifdef NAS_INVESTIGATE_ALL_ROWS_COUNTER
            , lockstep.ready_count, lockstep.epoch
#endif
            );
        NAS_SCHED_TRACE_END(complete_logical, NAS_SCHED_COMPLETE_LOGICAL, 0,
                           get_completion_data_id(item),
                           get_completion_iteration_num(item), 0);
      }
      NAS_SCHED_TRACE_END(handle_completion, NAS_SCHED_HANDLE_COMPLETION,
                         batch_completion,
                         batch_completion ? DATA_INVALID_ID : get_completion_data_id(item),
                         batch_completion ? 0 : get_completion_iteration_num(item),
                         batch_completion ? get_completion_iteration_num(item) : 0);
      progress = true;
    }
    } // Bounded completion burst.
    NAS_SCHED_TRACE_BEGIN(poll_control);
    bool control_ready = event_cursor <
        experimental_moe_acquire(&config.sched_queue_last_ready_event_id[0]);
    NAS_SCHED_TRACE_END(poll_control, NAS_SCHED_POLL_CONTROL,
                       control_ready, DATA_INVALID_ID, 0, 0);
    if (control_ready) {
      NAS_SCHED_TRACE_BEGIN(handle_control);
      EventId event = ld_relaxed_gpu_u64(&config.sched_queues[0]
          [event_cursor++ % config.per_sched_queue_len]);
      if (event == 0u) {
#ifdef NAS_INVESTIGATE_LOCKSTEP_RESTART
        lockstep.finish(config);
#endif
        require_request_slot_invariant(*config.active_request_slot_count == 0u &&
                                       config.experimental_moe->in_flight == 0u);
        for (uint32_t kind = 0; kind < EXPERIMENTAL_BATCH_KINDS; kind++)
          for (uint32_t slot = 0; slot < EXPERIMENTAL_MOE_STATE_SLOTS; slot++)
            require_request_slot_invariant(config.experimental_moe->pending[kind][slot].pending == 0u);
        for (int worker = 0; worker < config.num_workers; worker++)
          enqueue_worker_item(config, worker, &worker_positions[worker], 0u);
        NAS_SCHED_TRACE_END(handle_control, NAS_SCHED_HANDLE_CONTROL, 1, event, 0, 0);
        occupancy.finish();
        NAS_SCHED_TRACE_FINISH();
        return;
      }
#ifdef NAS_INVESTIGATE_LOCKSTEP_RESTART
      require_request_slot_invariant(lockstep.handle(config, event, worker_positions, occupancy));
#else
      require_request_slot_invariant(handle_slot_restart_event(
          config, event, 0, 0, config.num_workers, worker_positions));
#endif
#ifndef NAS_INVESTIGATE_LOCKSTEP_RESTART
      bool bootstrap = event == static_cast<EventId>(config.end_event_index);
      bool completed_slot = is_slot_control_item(event) &&
          (static_cast<uint32_t>(event) & SLOT_DISPATCH_TAG) == 0u;
      if (bootstrap || completed_slot) {
        occupancy.control(*config.active_request_slot_count,
                          *config.next_request_id, bootstrap);
      }
#endif
      NAS_SCHED_TRACE_END(handle_control, NAS_SCHED_HANDLE_CONTROL, 0,
                         event, event >> 32, 0);
      progress = true;
    }
    // Include unsuccessful scans: they consume scheduler time too.
    NAS_SCHED_TRACE_BEGIN(dispatch);
    bool dispatched = experimental_moe_dispatch(config, next_worker,
                                                worker_positions, occupancy
#ifdef NAS_INVESTIGATE_ALL_ROWS
                                                , lockstep.open_ops, lockstep.epoch
#endif
#ifdef NAS_INVESTIGATE_ALL_ROWS_COUNTER
                                                , lockstep.ready_count
#endif
                                                );
    NAS_SCHED_TRACE_END(dispatch, NAS_SCHED_DISPATCH, dispatched, DATA_INVALID_ID, 0,
                       dispatched ? config.experimental_moe->next_sequence : 0);
    progress = dispatched || progress;
    if (!progress) {
      NAS_SCHED_TRACE_BEGIN(backoff);
      __nanosleep(10);
      NAS_SCHED_TRACE_END(backoff, NAS_SCHED_BACKOFF, 0, DATA_INVALID_ID, 0, 0);
    }
  }
}
