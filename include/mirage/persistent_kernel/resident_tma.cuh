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

#include "resident_runtime_header.h"
#include "tma.cuh"

namespace mirage {
namespace runtime {

__host__ inline void create_tma_desc_by_data(ResidentTaskDesc const &resident_task,
                                             FullDataDesc &data_desc) {
  FullTaskDesc task_desc(resident_task.task_type, resident_task.variant_id);
  task_desc.profiler_group_id = data_desc.profiler_group_id;
  task_desc.num_inputs = resident_task.num_inputs;
  task_desc.num_outputs = resident_task.num_outputs;
  task_desc.trigger_event = data_desc.trigger_event;
  task_desc.dependent_event = data_desc.dependent_event;
  task_desc.task_metadata = data_desc.task_metadata;
  for (int i = 0; i < resident_task.num_inputs; i++) {
    task_desc.inputs[i] = data_desc.inputs[i];
  }
  for (int i = 0; i < resident_task.num_outputs; i++) {
    task_desc.outputs[i] = data_desc.outputs[i];
  }

  bool const uses_tma =
      (task_desc.task_type > TASK_HOPPER_TASK_BEGIN &&
       task_desc.task_type < TASK_HOPPER_TASK_END) ||
      (task_desc.task_type > TASK_SM100_TMA_START_TASK &&
       task_desc.task_type < TASK_SM100_TMA_END_TASK);
  if (!uses_tma) {
    return;
  }

  create_tma_desc_by_task(task_desc);
  for (int i = 0; i < resident_task.num_inputs; i++) {
    data_desc.inputs[i] = task_desc.inputs[i];
  }
  for (int i = 0; i < resident_task.num_outputs; i++) {
    data_desc.outputs[i] = task_desc.outputs[i];
  }
}

} // namespace runtime
} // namespace mirage
