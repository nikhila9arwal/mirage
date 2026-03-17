/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "runtime_header.h"
#include <cstdint>

namespace mirage {
namespace runtime {

typedef uint32_t ResidentTaskId;
typedef uint32_t DataId;
constexpr ResidentTaskId RESIDENT_TASK_INVALID_ID = 0xFFFFFFFFu;
constexpr DataId DATA_INVALID_ID = 0xFFFFFFFFu;

enum ResidentExecutionMode : uint32_t {
  RESIDENT_EXECUTION_SCHEDULER_DISPATCH = 0,
  RESIDENT_EXECUTION_HYBRID_PRELAUNCH = 1,
  RESIDENT_EXECUTION_STREAMING = 2,
};

enum ResidentTaskExecutionKind : uint32_t {
  RESIDENT_TASK_EXECUTION_PRELAUNCHED = 0,
  RESIDENT_TASK_EXECUTION_STREAMING = 1,
};

enum StreamingBaseExecutionMode : uint32_t {
  STREAMING_BASE_EXECUTION_LEGACY_EVENT = 0,
  STREAMING_BASE_EXECUTION_HYBRID_PRELAUNCH = 1,
};

using TaskMetadata = FullTaskDesc::TaskMetadata;

struct ResidentTaskDesc {
  ResidentTaskDesc()
      : task_type(TASK_TERMINATE), variant_id(0),
        profiler_group_id(INVALID_PROFILER_GROUP_ID), num_inputs(0),
        num_outputs(0), execution_kind(RESIDENT_TASK_EXECUTION_PRELAUNCHED),
        max_parallelism(1), total_data_count(0) {}
  ResidentTaskDesc(TaskType t, unsigned variant)
      : task_type(t), variant_id(variant),
        profiler_group_id(INVALID_PROFILER_GROUP_ID), num_inputs(0),
        num_outputs(0), execution_kind(RESIDENT_TASK_EXECUTION_PRELAUNCHED),
        max_parallelism(1), total_data_count(0) {}
  TaskType task_type;
  unsigned variant_id;
  uint32_t profiler_group_id;
  int num_inputs, num_outputs;
  uint32_t execution_kind;
  int max_parallelism;
  int total_data_count;
};

struct FullDataDesc {
  FullDataDesc()
      : resident_task_id(RESIDENT_TASK_INVALID_ID),
        profiler_group_id(INVALID_PROFILER_GROUP_ID),
        initial_predecessor_count(0), trigger_event(EVENT_INVALID_ID),
        dependent_event(EVENT_INVALID_ID) {
    task_metadata.raw_payload = ~0ull;
    for (int i = 0; i < MAX_INPUTS_PER_TASK; i++) {
      inputs[i].num_dims = 0;
      inputs[i].base_ptr = nullptr;
      inputs[i].data_type = 0;
      for (int d = 0; d < mirage::config::MAX_TENSOR_DIMS; d++) {
        inputs[i].dim[d] = 0;
        inputs[i].stride[d] = 0;
      }
#ifdef MPK_ENABLE_TMA
      for (int k = 0; k < mirage::config::MAX_TMA_DESC_PER_TENSOR; k++) {
        inputs[i].tma_desc_ptrs[k] = nullptr;
      }
#endif
    }
    for (int i = 0; i < MAX_OUTPUTS_PER_TASK; i++) {
      outputs[i].num_dims = 0;
      outputs[i].base_ptr = nullptr;
      outputs[i].data_type = 0;
      for (int d = 0; d < mirage::config::MAX_TENSOR_DIMS; d++) {
        outputs[i].dim[d] = 0;
        outputs[i].stride[d] = 0;
      }
#ifdef MPK_ENABLE_TMA
      for (int k = 0; k < mirage::config::MAX_TMA_DESC_PER_TENSOR; k++) {
        outputs[i].tma_desc_ptrs[k] = nullptr;
      }
#endif
    }
  }
  ResidentTaskId resident_task_id;
  uint32_t profiler_group_id;
  int initial_predecessor_count;
  EventId trigger_event;
  EventId dependent_event;
  TensorDesc inputs[MAX_INPUTS_PER_TASK];
  TensorDesc outputs[MAX_OUTPUTS_PER_TASK];
  TaskMetadata task_metadata;
};

struct alignas(16) DataDesc {
  DataDesc()
      : resident_task_id(RESIDENT_TASK_INVALID_ID),
        profiler_group_id(INVALID_PROFILER_GROUP_ID),
        initial_predecessor_count(0), trigger_event(EVENT_INVALID_ID),
        dependent_event(EVENT_INVALID_ID) {
    task_metadata.raw_payload = ~0ull;
    for (int i = 0; i < MAX_INPUTS_PER_TASK; i++) {
      input_ptrs[i] = nullptr;
#ifdef MPK_ENABLE_TMA
      for (int k = 0; k < mirage::config::MAX_TMA_DESC_PER_TENSOR; k++) {
        input_tma_desc_ptrs[i][k] = nullptr;
      }
#endif
    }
    for (int i = 0; i < MAX_OUTPUTS_PER_TASK; i++) {
      output_ptrs[i] = nullptr;
#ifdef MPK_ENABLE_TMA
      for (int k = 0; k < mirage::config::MAX_TMA_DESC_PER_TENSOR; k++) {
        output_tma_desc_ptrs[i][k] = nullptr;
      }
#endif
    }
  }
  DataDesc(FullDataDesc const &d)
      : resident_task_id(d.resident_task_id),
        profiler_group_id(d.profiler_group_id),
        initial_predecessor_count(d.initial_predecessor_count),
        trigger_event(d.trigger_event), dependent_event(d.dependent_event),
        task_metadata(d.task_metadata) {
    for (int i = 0; i < MAX_INPUTS_PER_TASK; i++) {
      input_ptrs[i] = d.inputs[i].base_ptr;
    }
    for (int i = 0; i < MAX_OUTPUTS_PER_TASK; i++) {
      output_ptrs[i] = d.outputs[i].base_ptr;
    }
#ifdef MPK_ENABLE_TMA
    for (int i = 0; i < MAX_INPUTS_PER_TASK; i++) {
      for (int k = 0; k < mirage::config::MAX_TMA_DESC_PER_TENSOR; k++) {
        input_tma_desc_ptrs[i][k] = d.inputs[i].tma_desc_ptrs[k];
      }
    }
    for (int i = 0; i < MAX_OUTPUTS_PER_TASK; i++) {
      for (int k = 0; k < mirage::config::MAX_TMA_DESC_PER_TENSOR; k++) {
        output_tma_desc_ptrs[i][k] = d.outputs[i].tma_desc_ptrs[k];
      }
    }
#endif
  }
  ResidentTaskId resident_task_id;
  uint32_t profiler_group_id;
  int initial_predecessor_count;
  EventId trigger_event;
  EventId dependent_event;
  void *input_ptrs[MAX_INPUTS_PER_TASK];
  void *output_ptrs[MAX_OUTPUTS_PER_TASK];
#ifdef MPK_ENABLE_TMA
  void *input_tma_desc_ptrs[MAX_INPUTS_PER_TASK]
                           [mirage::config::MAX_TMA_DESC_PER_TENSOR];
  void *output_tma_desc_ptrs[MAX_OUTPUTS_PER_TASK]
                            [mirage::config::MAX_TMA_DESC_PER_TENSOR];
#endif
  TaskMetadata task_metadata;
};

struct DataEdgeDesc {
  DataEdgeDesc() : src_data_id(DATA_INVALID_ID), dst_data_id(DATA_INVALID_ID) {}
  DataEdgeDesc(DataId src, DataId dst) : src_data_id(src), dst_data_id(dst) {}
  DataId src_data_id;
  DataId dst_data_id;
};

struct alignas(16) ResolvedTaskDesc {
  __host__ __device__ ResolvedTaskDesc()
      : task_type(TASK_TERMINATE), variant_id(0),
        profiler_group_id(INVALID_PROFILER_GROUP_ID),
        trigger_event(EVENT_INVALID_ID),
        dependent_event(EVENT_INVALID_ID) {
    task_metadata.raw_payload = ~0ull;
    for (int i = 0; i < MAX_INPUTS_PER_TASK; i++) {
      input_ptrs[i] = nullptr;
#ifdef MPK_ENABLE_TMA
      for (int k = 0; k < mirage::config::MAX_TMA_DESC_PER_TENSOR; k++) {
        input_tma_desc_ptrs[i][k] = nullptr;
      }
#endif
    }
    for (int i = 0; i < MAX_OUTPUTS_PER_TASK; i++) {
      output_ptrs[i] = nullptr;
#ifdef MPK_ENABLE_TMA
      for (int k = 0; k < mirage::config::MAX_TMA_DESC_PER_TENSOR; k++) {
        output_tma_desc_ptrs[i][k] = nullptr;
      }
#endif
    }
  }
  TaskType task_type;
  unsigned variant_id;
  uint32_t profiler_group_id;
  EventId trigger_event;
  EventId dependent_event;
  void *input_ptrs[MAX_INPUTS_PER_TASK];
  void *output_ptrs[MAX_OUTPUTS_PER_TASK];
#ifdef MPK_ENABLE_TMA
  void *input_tma_desc_ptrs[MAX_INPUTS_PER_TASK]
                           [mirage::config::MAX_TMA_DESC_PER_TENSOR];
  void *output_tma_desc_ptrs[MAX_OUTPUTS_PER_TASK]
                            [mirage::config::MAX_TMA_DESC_PER_TENSOR];
#endif
  TaskMetadata task_metadata;
};

__device__ __forceinline__ ResolvedTaskDesc
resolve_task_desc(ResidentTaskDesc const *resident_task_desc,
                  DataDesc const *data_desc) {
  ResolvedTaskDesc resolved;
  resolved.task_type = resident_task_desc->task_type;
  resolved.variant_id = resident_task_desc->variant_id;
  resolved.profiler_group_id = data_desc->profiler_group_id;
  resolved.trigger_event = data_desc->trigger_event;
  resolved.dependent_event = data_desc->dependent_event;
  for (int i = 0; i < MAX_INPUTS_PER_TASK; i++) {
    resolved.input_ptrs[i] = data_desc->input_ptrs[i];
  }
  for (int i = 0; i < MAX_OUTPUTS_PER_TASK; i++) {
    resolved.output_ptrs[i] = data_desc->output_ptrs[i];
  }
#ifdef MPK_ENABLE_TMA
  for (int i = 0; i < MAX_INPUTS_PER_TASK; i++) {
    for (int k = 0; k < mirage::config::MAX_TMA_DESC_PER_TENSOR; k++) {
      resolved.input_tma_desc_ptrs[i][k] = data_desc->input_tma_desc_ptrs[i][k];
    }
  }
  for (int i = 0; i < MAX_OUTPUTS_PER_TASK; i++) {
    for (int k = 0; k < mirage::config::MAX_TMA_DESC_PER_TENSOR; k++) {
      resolved.output_tma_desc_ptrs[i][k] =
          data_desc->output_tma_desc_ptrs[i][k];
    }
  }
#endif
  resolved.task_metadata = data_desc->task_metadata;
  return resolved;
}

static_assert(sizeof(ResolvedTaskDesc) == sizeof(TaskDesc),
              "ResolvedTaskDesc must stay layout-compatible with TaskDesc.");
static_assert(alignof(ResolvedTaskDesc) == alignof(TaskDesc),
              "ResolvedTaskDesc alignment must match TaskDesc.");

struct ResidentRuntimeConfig : public RuntimeConfig {
  uint32_t resident_execution_mode;
  uint32_t streaming_base_execution_mode;
  int begin_event_index;
  int end_event_index;
  int num_resident_tasks;
  int num_data;
  int num_data_edges;
  int num_first_data_ids;
  int num_control_tasks;
  int num_terminal_data;
  unsigned long long int per_completion_queue_len;
  unsigned long long int *worker_queue_next_free_task_id;
  unsigned long long int *completion_queue_last_ready_data_id;
  unsigned long long int *completion_queue_next_free_data_id;
  ResidentTaskDesc *resident_tasks;
  DataDesc *all_data;
  uint32_t *data_edge_offsets;
  DataId *data_edge_targets;
  uint32_t *data_initial_predecessor_counts;
  uint32_t *data_pending_predecessor_counts;
  uint32_t *data_last_enqueued_iteration;
  uint32_t *data_last_executed_iteration;
  DataId *task_to_data_id;
  TaskId *data_to_task_id;
  int *worker_owner_scheduler;
  int *resident_owner_worker;
  uint32_t *worker_streaming_resident_offsets;
  ResidentTaskId *worker_streaming_resident_ids;
  DataId *resident_ready_data_head;
  DataId *resident_ready_data_tail;
  DataId *data_ready_next;
  uint32_t *resident_ready_queue_offsets;
  uint32_t *resident_ready_head_positions;
  uint32_t *resident_ready_next_free_positions;
  uint32_t *resident_ready_tail_positions;
  DataId *resident_ready_queue_storage;
  uint32_t *resident_active_workers;
  uint32_t *resident_completed_data;
  uint32_t *completed_terminal_data_count;
  uint32_t *completed_streaming_data_count;
  uint32_t *completed_data_this_iteration_count;
  uint32_t *current_iteration;
  uint32_t *streaming_terminate_flag;
  DataId *first_data_ids;
  TaskId **completion_queues;
};

} // namespace runtime
} // namespace mirage
