/* Copyright 2023-2024 CMU
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

#include "mirage/kernel/graph.h"
#include "mirage/kernel/task_graph_builder.h"
#include "mirage/persistent_kernel/resident_runtime_header.h"
#include "mirage/kernel/task_register.h"
#include "mirage/transpiler/utils.h"
#include "mirage/utils/json_utils.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <set>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <tuple>

namespace mirage {
namespace kernel {

using namespace mirage::runtime;
namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;

namespace {

size_t get_event_id(int my_gpu_id, size_t event_pos, bool nvshmem_event) {
  size_t event_id = ((static_cast<size_t>(my_gpu_id) << 32) | event_pos);
  if (nvshmem_event) {
    event_id = event_id | EVENT_NVSHMEM_TAG;
  }
  return event_id;
}

bool is_nvshmem_event(size_t event_id) {
  return (event_id & EVENT_NVSHMEM_TAG) > 0;
}

/**
 * Get number of subtasks for a task type.
 *
 * For each (bid.x, bid.y, bid.z), there could be multiple subtasks associated
 * with it. These subtasks can be executed on different workers, just like
 * normal tasks. The biggest difference is that these subtasks share the same
 * input/output tensors.
 *
 * For example, for NVSHMEM_ALLGATHER_STRIDED_PUT task, each (bid.x, bid.y,
 * bid.z) will have (num_gpus - 1) subtasks, each subtask is responsible for
 * putting data to one of the other GPUs.
 */
int get_num_subtasks(int num_gpus, TaskType task_type) {
  // TODO(Zepeng) Re-consider this design. Try if task coalescing can result in
  // better performance.
  if (task_type == TASK_NVSHMEM_ALLGATHER_STRIDED_PUT) {
    return num_gpus - 1;
  } else {
    return 1;
  }
}

namespace {

constexpr uint32_t MAX_PROFILER_GROUP_COUNT = 1u << 13;

using ProfilerGroupKey = std::tuple<uint32_t, TaskType, uint32_t>;

void assign_profiler_group_ids(std::vector<FullTaskDesc> &all_tasks) {
  std::map<ProfilerGroupKey, uint32_t> group_ids;

  for (FullTaskDesc &task : all_tasks) {
    if (!uses_dag_profiler_group(task.task_type)) {
      task.profiler_group_id = INVALID_PROFILER_GROUP_ID;
      continue;
    }

    ProfilerGroupKey key(
        static_cast<uint32_t>(task.dependent_event & 0xFFFFFFFFull),
        task.task_type,
        static_cast<uint32_t>(task.trigger_event & 0xFFFFFFFFull));
    auto inserted = group_ids.emplace(key, static_cast<uint32_t>(group_ids.size()));
    uint32_t profiler_group_id = inserted.first->second;
    assert(profiler_group_id < MAX_PROFILER_GROUP_COUNT);
    task.profiler_group_id = profiler_group_id;
  }
}

struct TensorInterval {
  uintptr_t begin;
  uintptr_t end;
};

struct ResidentTaskGraph {
  std::vector<ResidentTaskDesc> resident_tasks;
  std::vector<FullDataDesc> all_data;
  std::vector<DataEdgeDesc> data_edges;
  std::vector<DataId> first_data_ids;
  std::vector<DataId> task_to_data_id;
  std::vector<TaskId> data_to_task_id;
};

enum class DataAwareTaskGraphFlavor {
  RESIDENT,
  STREAMING,
};

bool is_streaming_whitelist_task(TaskType task_type) {
  return task_type == TASK_MOE_W13_LINEAR_SM90 ||
         task_type == TASK_MOE_W2_LINEAR_SM90 ||
         task_type == TASK_MOE_W13_LINEAR_SM100 ||
         task_type == TASK_MOE_W2_LINEAR_SM100;
}

ResidentTaskExecutionKind get_streaming_execution_kind(TaskType task_type) {
  return is_streaming_whitelist_task(task_type)
             ? RESIDENT_TASK_EXECUTION_STREAMING
             : RESIDENT_TASK_EXECUTION_PRELAUNCHED;
}

void append_tensor_signature(std::ostringstream &oss,
                             TensorDesc const &tensor_desc,
                             bool include_base_ptr) {
  if (include_base_ptr) {
    oss << '@' << reinterpret_cast<uintptr_t>(tensor_desc.base_ptr);
  }
  oss << ':' << tensor_desc.data_type << ':' << tensor_desc.num_dims;
  for (int dim = 0; dim < tensor_desc.num_dims; dim++) {
    oss << ':' << tensor_desc.dim[dim] << ':' << tensor_desc.stride[dim];
  }
}

std::string build_streaming_resident_key(kernel::KNOperator const *op,
                                         TaskId task_id,
                                         FullTaskDesc const &task) {
  std::ostringstream oss;
  oss << reinterpret_cast<uintptr_t>(op) << ':'
      << static_cast<int>(task.task_type) << ':' << task.variant_id << ':'
      << task.num_inputs << ':' << task.num_outputs;
  if (!is_streaming_whitelist_task(task.task_type)) {
    // Unsupported operators stay at original MPK task granularity in the
    // streaming pass. Only whitelisted operators collapse multiple data
    // slices into one resident task.
    oss << ":task=" << task_id;
    return oss.str();
  }

  // First-cut streaming groups keep the MoE weight shard and expert slot
  // fixed, and stream only the dynamic token/request slices through them.
  oss << ":expert=" << task.task_metadata.expert_offset;
  if (task.num_inputs > 1) {
    append_tensor_signature(oss, task.inputs[1], true /*include_base_ptr*/);
  }
  for (int output_idx = 0; output_idx < task.num_outputs; output_idx++) {
    append_tensor_signature(oss, task.outputs[output_idx], false);
  }
  return oss.str();
}

size_t runtime_dtype_size(int data_type) {
  switch (static_cast<type::DataType>(data_type)) {
    case type::DT_FLOAT8:
    case type::DT_INT8:
    case type::DT_UINT8:
      return 1;
    case type::DT_FLOAT16:
    case type::DT_BFLOAT16:
    case type::DT_INT16:
    case type::DT_UINT16:
      return 2;
    case type::DT_FLOAT32:
    case type::DT_INT32:
    case type::DT_UINT32:
      return 4;
    case type::DT_DOUBLE:
    case type::DT_INT64:
    case type::DT_UINT64:
      return 8;
    default:
      return 0;
  }
}

bool tensor_interval(TensorDesc const &tensor, TensorInterval &interval) {
  if (tensor.base_ptr == nullptr || tensor.num_dims <= 0) {
    return false;
  }
  size_t elem_size = runtime_dtype_size(tensor.data_type);
  if (elem_size == 0) {
    return false;
  }
  size_t max_elem_offset = 0;
  for (int i = 0; i < tensor.num_dims; i++) {
    if (tensor.dim[i] <= 0) {
      return false;
    }
    max_elem_offset +=
        static_cast<size_t>(tensor.dim[i] - 1) *
        static_cast<size_t>(std::abs(tensor.stride[i]));
  }
  interval.begin = reinterpret_cast<uintptr_t>(tensor.base_ptr);
  interval.end = interval.begin + (max_elem_offset + 1) * elem_size;
  return interval.end > interval.begin;
}

bool tensors_overlap(TensorDesc const &lhs, TensorDesc const &rhs) {
  TensorInterval lhs_interval, rhs_interval;
  if (!tensor_interval(lhs, lhs_interval) || !tensor_interval(rhs, rhs_interval)) {
    return false;
  }
  return lhs_interval.begin < rhs_interval.end &&
         rhs_interval.begin < lhs_interval.end;
}

bool is_control_event(EventDesc const &event_desc) {
  return event_desc.event_type == EVENT_TERMINATION ||
         event_desc.event_type == EVENT_LAUNCH_DEPENDENT_TASKS ||
         event_desc.event_type == EVENT_END_OF_TASK_GRAPH;
}

ResidentExecutionMode get_resident_execution_mode() {
  char const *mode_env = std::getenv("MIRAGE_RESIDENT_EXECUTION_MODE");
  if (mode_env == nullptr || std::strcmp(mode_env, "") == 0 ||
      std::strcmp(mode_env, "scheduler_dispatch") == 0) {
    return RESIDENT_EXECUTION_SCHEDULER_DISPATCH;
  }
  if (std::strcmp(mode_env, "hybrid_prelaunch") == 0) {
    return RESIDENT_EXECUTION_HYBRID_PRELAUNCH;
  }
  throw std::runtime_error(
      "Invalid MIRAGE_RESIDENT_EXECUTION_MODE. Expected "
      "\"scheduler_dispatch\" or \"hybrid_prelaunch\".");
}

StreamingBaseExecutionMode get_streaming_base_execution_mode() {
  char const *mode_env = std::getenv("MIRAGE_STREAMING_BASE_MODE");
  if (mode_env == nullptr || std::strcmp(mode_env, "") == 0) {
    mode_env = std::getenv("MIRAGE_RESIDENT_EXECUTION_MODE");
  }
  if (mode_env == nullptr || std::strcmp(mode_env, "") == 0 ||
      std::strcmp(mode_env, "legacy_event") == 0 ||
      std::strcmp(mode_env, "scheduler_dispatch") == 0) {
    return STREAMING_BASE_EXECUTION_LEGACY_EVENT;
  }
  if (std::strcmp(mode_env, "hybrid_prelaunch") == 0) {
    return STREAMING_BASE_EXECUTION_HYBRID_PRELAUNCH;
  }
  throw std::runtime_error(
      "Invalid streaming base mode. Expected \"legacy_event\" or "
      "\"hybrid_prelaunch\".");
}

ResidentTaskGraph build_data_aware_task_graph(
    mirage::kernel::Graph const &graph,
    std::vector<FullTaskDesc> const &all_tasks,
    std::vector<EventDesc> const &all_events,
    OperatorTaskMaps const &all_task_maps,
    DataAwareTaskGraphFlavor flavor,
    ResidentExecutionMode execution_mode,
    StreamingBaseExecutionMode streaming_base_mode) {
  ResidentTaskGraph result;
  result.task_to_data_id.assign(all_tasks.size(), DATA_INVALID_ID);
  std::unordered_map<std::string, ResidentTaskId> streaming_resident_ids;

  // Keep families and DataIds in registration/grid order. Slot annotations
  // and projection-layer numbering use this order; regular TaskIds may be
  // interleaved by the shared DAG generator and are looked up explicitly.
  for (auto const &op : graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      continue;
    }
    auto const task_map_it = all_task_maps.find(op);
    if (task_map_it == all_task_maps.end()) {
      continue;
    }
    auto const &task_map = task_map_it->second;
    if (task_map.empty()) {
      continue;
    }
    if (flavor == DataAwareTaskGraphFlavor::RESIDENT) {
      TaskId sample_task_id = task_map.begin()->second.front();
      FullTaskDesc const &sample_task = all_tasks[sample_task_id];
      ResidentTaskId resident_task_id =
          static_cast<ResidentTaskId>(result.resident_tasks.size());
      ResidentTaskDesc resident_task(sample_task.task_type, sample_task.variant_id);
      resident_task.profiler_group_id = resident_task_id;
      resident_task.num_inputs = sample_task.num_inputs;
      resident_task.num_outputs = sample_task.num_outputs;
      resident_task.execution_kind = RESIDENT_TASK_EXECUTION_PRELAUNCHED;
      resident_task.max_parallelism = 0;
      int op_data_count = 0;

      for (auto const &entry : task_map) {
        resident_task.max_parallelism += static_cast<int>(entry.second.size());
        for (TaskId task_id : entry.second) {
          FullTaskDesc const &task = all_tasks[task_id];
          if (task.task_type != sample_task.task_type ||
              task.variant_id != sample_task.variant_id ||
              task.num_inputs != sample_task.num_inputs ||
              task.num_outputs != sample_task.num_outputs) {
            throw std::runtime_error(
                "Resident task builder grouped heterogeneous task signatures "
                "into one resident task. Split by subtask family instead.");
          }
          FullDataDesc data_desc;
          data_desc.resident_task_id = resident_task_id;
          data_desc.profiler_group_id = resident_task_id;
          data_desc.trigger_event = task.trigger_event;
          data_desc.dependent_event = task.dependent_event;
          data_desc.task_metadata = task.task_metadata;
          for (int i = 0; i < task.num_inputs; i++) {
            data_desc.inputs[i] = task.inputs[i];
          }
          for (int i = 0; i < task.num_outputs; i++) {
            data_desc.outputs[i] = task.outputs[i];
          }
          DataId data_id = static_cast<DataId>(result.all_data.size());
          result.task_to_data_id[task_id] = data_id;
          result.all_data.push_back(data_desc);
          op_data_count++;
        }
      }
      resident_task.total_data_count = op_data_count;
      result.resident_tasks.push_back(resident_task);
      continue;
    }

    for (auto const &entry : task_map) {
      for (TaskId task_id : entry.second) {
        FullTaskDesc const &task = all_tasks[task_id];
        std::string resident_key =
            build_streaming_resident_key(op, task_id, task);
        auto const [resident_it, inserted] = streaming_resident_ids.emplace(
            resident_key, static_cast<ResidentTaskId>(result.resident_tasks.size()));
        ResidentTaskId resident_task_id = resident_it->second;
        if (inserted) {
          ResidentTaskDesc resident_task(task.task_type, task.variant_id);
          resident_task.profiler_group_id = resident_task_id;
          resident_task.num_inputs = task.num_inputs;
          resident_task.num_outputs = task.num_outputs;
          resident_task.execution_kind =
              get_streaming_execution_kind(task.task_type);
          resident_task.max_parallelism = 1;
          resident_task.total_data_count = 0;
          result.resident_tasks.push_back(resident_task);
        } else {
          ResidentTaskDesc const &existing = result.resident_tasks.at(resident_task_id);
          if (existing.task_type != task.task_type ||
              existing.variant_id != task.variant_id ||
              existing.num_inputs != task.num_inputs ||
              existing.num_outputs != task.num_outputs) {
            throw std::runtime_error(
                "Streaming task builder grouped heterogeneous task signatures "
                "into one resident task.");
          }
        }

        FullDataDesc data_desc;
        data_desc.resident_task_id = resident_task_id;
        data_desc.profiler_group_id = resident_task_id;
        data_desc.trigger_event = task.trigger_event;
        data_desc.dependent_event = task.dependent_event;
        data_desc.task_metadata = task.task_metadata;
        for (int i = 0; i < task.num_inputs; i++) {
          data_desc.inputs[i] = task.inputs[i];
        }
        for (int i = 0; i < task.num_outputs; i++) {
          data_desc.outputs[i] = task.outputs[i];
        }
        DataId data_id = static_cast<DataId>(result.all_data.size());
        result.task_to_data_id[task_id] = data_id;
        result.all_data.push_back(data_desc);
        result.resident_tasks.at(resident_task_id).total_data_count += 1;
      }
    }
  }

  std::unordered_map<uint32_t, std::vector<TaskId>> event_producers;
  std::unordered_map<uint32_t, std::vector<TaskId>> event_consumers;
  for (TaskId task_id = 2; task_id < all_tasks.size(); task_id++) {
    FullTaskDesc const &task = all_tasks[task_id];
    if (task.trigger_event != EVENT_INVALID_ID) {
      event_producers[static_cast<uint32_t>(task.trigger_event & 0xFFFFFFFFull)]
          .push_back(task_id);
    }
    if (task.dependent_event != EVENT_INVALID_ID) {
      event_consumers[static_cast<uint32_t>(task.dependent_event & 0xFFFFFFFFull)]
          .push_back(task_id);
    }
  }

  for (size_t event_idx = 0; event_idx < all_events.size(); event_idx++) {
    bool const local_data_dependencies =
        execution_mode == RESIDENT_EXECUTION_HYBRID_PRELAUNCH ||
        execution_mode == RESIDENT_EXECUTION_STREAMING ||
        (flavor == DataAwareTaskGraphFlavor::STREAMING &&
         streaming_base_mode == STREAMING_BASE_EXECUTION_HYBRID_PRELAUNCH);
    if (local_data_dependencies && is_control_event(all_events[event_idx])) {
      continue;
    }
    auto producer_it = event_producers.find(static_cast<uint32_t>(event_idx));
    auto consumer_it = event_consumers.find(static_cast<uint32_t>(event_idx));
    if (producer_it == event_producers.end() || consumer_it == event_consumers.end()) {
      continue;
    }
    auto const &producer_tasks = producer_it->second;
    auto const &consumer_tasks = consumer_it->second;
    if (local_data_dependencies) {
      bool nvshmem_event = false;
      for (TaskId producer_task_id : producer_tasks) {
        if (is_nvshmem_event(all_tasks[producer_task_id].trigger_event)) {
          nvshmem_event = true;
          break;
        }
      }
      if (!nvshmem_event) {
        for (TaskId consumer_task_id : consumer_tasks) {
          if (is_nvshmem_event(all_tasks[consumer_task_id].dependent_event)) {
            nvshmem_event = true;
            break;
          }
        }
      }
      if (nvshmem_event) {
        continue;
      }
    }
    for (TaskId consumer_task_id : consumer_tasks) {
      std::unordered_set<TaskId> matched_producers;
      FullTaskDesc const &consumer_task = all_tasks[consumer_task_id];
      for (TaskId producer_task_id : producer_tasks) {
        FullTaskDesc const &producer_task = all_tasks[producer_task_id];
        bool overlaps = false;
        for (int out_idx = 0; out_idx < producer_task.num_outputs && !overlaps;
             out_idx++) {
          for (int in_idx = 0; in_idx < consumer_task.num_inputs; in_idx++) {
            if (tensors_overlap(producer_task.outputs[out_idx],
                                consumer_task.inputs[in_idx])) {
              overlaps = true;
              break;
            }
          }
        }
        if (overlaps) {
          matched_producers.insert(producer_task_id);
        }
      }
      if (matched_producers.empty()) {
        matched_producers.insert(producer_tasks.begin(), producer_tasks.end());
      }
      DataId consumer_data_id = result.task_to_data_id[consumer_task_id];
      result.all_data[consumer_data_id].initial_predecessor_count =
          static_cast<int>(matched_producers.size());
      for (TaskId producer_task_id : matched_producers) {
        result.data_edges.emplace_back(result.task_to_data_id[producer_task_id],
                                       consumer_data_id);
      }
    }
  }

  for (DataId data_id = 0; data_id < result.all_data.size(); data_id++) {
    if (result.all_data[data_id].initial_predecessor_count == 0) {
      result.first_data_ids.push_back(data_id);
    }
  }

  result.data_to_task_id.assign(result.all_data.size(), TASK_INVALID_ID);
  for (TaskId task_id = 0; task_id < result.task_to_data_id.size(); task_id++) {
    DataId data_id = result.task_to_data_id[task_id];
    if (data_id == DATA_INVALID_ID) {
      continue;
    }
    if (data_id >= result.data_to_task_id.size()) {
      throw std::runtime_error(
          "Resident task graph contains an out-of-range task-to-data mapping.");
    }
    if (result.data_to_task_id[data_id] != TASK_INVALID_ID) {
      throw std::runtime_error(
          "Resident task graph maps multiple compatibility tasks to one data "
          "instance.");
    }
    result.data_to_task_id[data_id] = task_id;
  }

  return result;
}

void validate_resident_task_graph(ResidentTaskGraph const &graph) {
  size_t const num_data = graph.all_data.size();
  if (graph.data_to_task_id.size() != num_data) {
    throw std::runtime_error(
        "Resident task graph has inconsistent data-to-task mapping size.");
  }
  std::vector<uint32_t> indegree(num_data, 0);
  std::vector<std::vector<DataId>> outgoing(num_data);
  outgoing.reserve(num_data);

  for (DataEdgeDesc const &edge : graph.data_edges) {
    if (edge.src_data_id >= num_data || edge.dst_data_id >= num_data) {
      throw std::runtime_error(
          "Resident task graph contains an out-of-range data edge.");
    }
    if (edge.src_data_id == edge.dst_data_id) {
      throw std::runtime_error(
          "Resident task graph contains a self-cycle.");
    }
    outgoing[edge.src_data_id].push_back(edge.dst_data_id);
    indegree[edge.dst_data_id] += 1;
  }

  std::vector<DataId> zero_indegree;
  zero_indegree.reserve(num_data);
  for (DataId data_id = 0; data_id < num_data; data_id++) {
    if (graph.all_data[data_id].initial_predecessor_count !=
        static_cast<int>(indegree[data_id])) {
      throw std::runtime_error(
          "Resident task graph has inconsistent predecessor counts.");
    }
    if (indegree[data_id] == 0) {
      zero_indegree.push_back(data_id);
    }
  }

  std::vector<DataId> sorted_first = graph.first_data_ids;
  std::sort(sorted_first.begin(), sorted_first.end());
  sorted_first.erase(std::unique(sorted_first.begin(), sorted_first.end()),
                     sorted_first.end());
  if (sorted_first != zero_indegree) {
    throw std::runtime_error(
        "Resident task graph first_data_ids do not match zero-indegree data.");
  }

  for (DataId data_id = 0; data_id < num_data; data_id++) {
    if (graph.data_to_task_id[data_id] == TASK_INVALID_ID) {
      throw std::runtime_error(
          "Resident task graph contains a data node with no compatibility task "
          "mapping.");
    }
  }

  std::queue<DataId> ready;
  for (DataId data_id : zero_indegree) {
    ready.push(data_id);
  }
  size_t visited = 0;
  std::vector<uint32_t> remaining_indegree = indegree;
  while (!ready.empty()) {
    DataId data_id = ready.front();
    ready.pop();
    visited++;
    for (DataId succ_data_id : outgoing[data_id]) {
      assert(remaining_indegree[succ_data_id] > 0);
      remaining_indegree[succ_data_id] -= 1;
      if (remaining_indegree[succ_data_id] == 0) {
        ready.push(succ_data_id);
      }
    }
  }

  if (visited != num_data) {
    throw std::runtime_error(
        "Resident task graph contains a cycle or unreachable strongly "
        "connected component.");
  }
}

} // namespace

// The resident compiler currently supports attached storage tensors. The new
// native compiler's virtual view/alias analysis is deliberately not duplicated
// here: reject such graphs before any owner-op cast or pointer arithmetic.
void validate_resident_bindings(mirage::kernel::Graph const &graph) {
  for (auto const *op : graph.operators) {
    if (op->op_type == type::KN_INPUT_OP) {
      continue;
    }
    auto registered = graph.task_config.find(op);
    if (op->op_type != type::KN_CUSTOMIZED_OP ||
        registered == graph.task_config.end()) {
      throw std::runtime_error("resident_data requires registered customized operators");
    }
    int inputs = std::get<0>(registered->second);
    int outputs = std::get<1>(registered->second);
    if (inputs < 0 || inputs > MAX_INPUTS_PER_TASK ||
        outputs < 0 || outputs > MAX_OUTPUTS_PER_TASK) {
      throw std::runtime_error("Resident task exceeds the shared descriptor input/output bounds");
    }
    auto const *custom = static_cast<kn::KNCustomizedOp const *>(op);
    for (auto const *inner : custom->bgraph.operators) {
      if (inner->op_type != type::TB_INPUT_OP) {
        throw std::runtime_error("Resident operator descriptors require TBInputOp tensors");
      }
      DTensor const &tensor = static_cast<tb::TBInputOp const *>(inner)->dtensor;
      if (tensor == DTensor::EMPTY_TENSOR) {
        continue;
      }
      if (tensor.is_virtual() || tensor.view_offset != 0) {
        throw std::runtime_error(
            "resident_data does not yet support virtual view/narrow/split tensors; "
            "attach the desired storage tensor explicitly");
      }
      auto binding = graph.io_config.find(tensor.guid);
      if (binding == graph.io_config.end() || tensor.owner_op == nullptr ||
          tensor.owner_op->op_type != type::KN_INPUT_OP) {
        throw std::runtime_error("Resident tensor must have an explicit IO binding");
      }
      IODesc const &io = binding->second;
      int64_t contiguous_stride = 1;
      for (int d = tensor.num_dims - 1; d >= 0; --d) {
        if (tensor.dim[d] <= 0 || tensor.stride[d] < 0 ||
            tensor.stride[d] > std::numeric_limits<int>::max()) {
          throw std::runtime_error("Resident tensor dimensions/strides exceed descriptor bounds");
        }
        if (tensor.dim[d] != io.tensor.dim[d] ||
            tensor.stride[d] != io.tensor.stride[d]) {
          throw std::runtime_error("Resident IO binding disagrees with DTensor shape or strides");
        }
        if ((io.type == IODesc::CUDAMallocTensor ||
             io.type == IODesc::NVSHMEMMallocTensor ||
             io.type == IODesc::ShuffledTorchTensor) &&
            tensor.stride[d] != contiguous_stride) {
          throw std::runtime_error("Resident internal allocations require contiguous tensors");
        }
        if (contiguous_stride > std::numeric_limits<int64_t>::max() / tensor.dim[d]) {
          throw std::runtime_error("Resident tensor size overflows int64");
        }
        contiguous_stride *= tensor.dim[d];
      }
    }
  }
}

bool sanity_check(mirage::kernel::Graph const &graph,
                  std::vector<FullTaskDesc> const &all_tasks,
                  std::vector<EventDesc> const &all_events,
                  std::vector<TaskId> const &first_tasks) {
  std::unordered_set<EventId> triggered_events;
  std::unordered_set<TaskId> executed_tasks;
  std::vector<int> event_counts(all_events.size(), 0);
  for (size_t i = 0; i < all_events.size(); i++) {
    event_counts[i] = all_events[i].num_triggers;
  }
  std::queue<TaskId> task_queue;
  std::queue<EventId> event_queue;
  printf("First tasks: %d\n", (int)first_tasks.size());
  for (size_t i = 0; i < first_tasks.size(); i++) {
    task_queue.push(first_tasks[i]);
  }
  while (!(task_queue.empty() && event_queue.empty())) {
    // Execute tasks
    while (!task_queue.empty()) {
      TaskId task = task_queue.front();
      task_queue.pop();
      assert(executed_tasks.count(task) == 0);
      executed_tasks.insert(task);
      FullTaskDesc desc = all_tasks[task];
      if (desc.trigger_event != EVENT_INVALID_ID) {
        EventId event_id = desc.trigger_event;
        size_t event_pos = event_id & 0xffffffff;
        // event_pos 0 is the end of task graph event
        if (event_pos == 0) {
          continue;
        }
        assert(event_counts[event_pos] > 0);
        event_counts[event_pos]--;
        if (event_counts[event_pos] == 0) {
          event_queue.push(event_id);
        }
      }
    }
    while (!event_queue.empty()) {
      EventId event_id = event_queue.front();
      event_queue.pop();
      assert(triggered_events.count(event_id) == 0);
      triggered_events.insert(event_id);
      size_t event_pos = event_id & 0xffffffff;
      EventDesc desc = all_events[event_pos];
      for (TaskId tid = desc.first_task_id; tid < desc.last_task_id; tid++) {
        task_queue.push(tid);
      }
    }
  }
  printf("Number of all events: %zu\n", all_events.size());
  printf("Number of all tasks: %zu\n", all_tasks.size());
  printf("Number of triggered events: %zu\n", triggered_events.size());
  printf("Number of executed tasks: %zu\n", executed_tasks.size());
  return true;
}

// Slot ownership is optional compiler metadata. It neither changes data edges
// nor changes the runtime's current graph-wide iteration protocol.
void add_request_slot_metadata(
    json &task_graph,
    std::vector<int> const &data_slot_ids,
    std::map<mirage::type::GuidType, IODesc> const &io_configs) {
  if (data_slot_ids.empty()) {
    return;
  }
  auto const &all_data = task_graph.at("all_data");
  size_t const num_data = all_data.size();
  if (data_slot_ids.size() != num_data) {
    throw std::runtime_error(
        "Request slot assignment must contain one slot ID per DataId: got " +
        std::to_string(data_slot_ids.size()) + ", expected " +
        std::to_string(num_data) + ".");
  }
  int max_slot = -1;
  for (int slot : data_slot_ids) {
    if (slot < 0 || static_cast<size_t>(slot) >= num_data) {
      throw std::runtime_error(
          "Request slot IDs must be nonnegative and smaller than the number "
          "of DataIds.");
    }
    max_slot = std::max(max_slot, slot);
  }
  size_t const num_slots = static_cast<size_t>(max_slot + 1);
  std::vector<std::vector<DataId>> members(num_slots), roots(num_slots),
      terminals(num_slots);
  std::vector<uint32_t> outgoing(num_data, 0);
  for (auto const &edge : task_graph.at("data_edges")) {
    DataId src = edge.at("src_data_id").get<DataId>();
    DataId dst = edge.at("dst_data_id").get<DataId>();
    if (data_slot_ids[src] != data_slot_ids[dst]) {
      throw std::runtime_error(
          "Cross-slot dependency D" + std::to_string(src) + " (slot " +
          std::to_string(data_slot_ids[src]) + ") -> D" + std::to_string(dst) +
          " (slot " + std::to_string(data_slot_ids[dst]) +
          ") prevents independent request-slot advancement.");
    }
    outgoing[src]++;
  }
  for (DataId d = 0; d < num_data; d++) {
    int slot = data_slot_ids[d];
    members[slot].push_back(d);
    if (all_data[d].at("initial_predecessor_count").get<int>() == 0) {
      roots[slot].push_back(d);
    }
    if (outgoing[d] == 0) {
      terminals[slot].push_back(d);
    }
  }
  for (size_t slot = 0; slot < num_slots; slot++) {
    if (members[slot].empty()) {
      throw std::runtime_error("Request slot IDs must be dense from zero; slot " +
                               std::to_string(slot) + " has no DataIds.");
    }
    if (roots[slot].empty() || terminals[slot].empty()) {
      throw std::runtime_error("Each request slot must have roots and terminals.");
    }
  }

  // TensorDesc.base_ptr is not populated when the resident data edges are
  // constructed. Validate the final native tensor views after serialization,
  // where allocation names and byte offsets have actually been resolved.
  std::unordered_map<std::string, IODesc const *> named_io;
  std::function<void(IODesc const &)> collect_io = [&](IODesc const &io) {
    auto [it, inserted] = named_io.emplace(io.name, &io);
    if (!inserted && (it->second->type != io.type ||
                      it->second->torch_data_ptr != io.torch_data_ptr)) {
      throw std::runtime_error("Ambiguous tensor name in request slots: " + io.name);
    }
    for (auto const &sub : io.sub_descs) {
      collect_io(sub);
    }
  };
  for (auto const &entry : io_configs) {
    collect_io(entry.second);
  }
  struct Access {
    DataId data;
    int slot;
    bool writes;
    uint64_t begin, end;
    std::string tensor;
  };
  std::unordered_map<std::string, std::vector<Access>> accesses;
  auto checked_add = [](uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
      throw std::runtime_error("Tensor interval overflow in request slot validation.");
    }
    return a + b;
  };
  for (DataId d = 0; d < num_data; d++) {
    for (bool writes : {false, true}) {
      for (auto const &tensor : all_data[d].at(writes ? "outputs" : "inputs")) {
        std::string name = tensor.at("base_ptr").get<std::string>();
        auto dims = tensor.at("dims").get<std::vector<int64_t>>();
        auto strides = tensor.at("strides").get<std::vector<int64_t>>();
        if (name == "nullptr" && dims.empty()) {
          continue;
        }
        size_t elem_size = runtime_dtype_size(tensor.at("data_type").get<int>());
        if (dims.empty() || dims.size() != strides.size() || elem_size == 0) {
          throw std::runtime_error("Invalid tensor view in request slots: " + name);
        }
        uint64_t elements = 1;
        for (size_t dim = 0; dim < dims.size(); dim++) {
          if (dims[dim] <= 0 || strides[dim] < 0) {
            throw std::runtime_error(
                "Request slots require positive dimensions and nonnegative strides: " + name);
          }
          uint64_t extent = static_cast<uint64_t>(dims[dim] - 1);
          uint64_t stride = static_cast<uint64_t>(strides[dim]);
          if (stride != 0 && extent > std::numeric_limits<uint64_t>::max() / stride) {
            throw std::runtime_error("Tensor interval overflow in request slots: " + name);
          }
          elements = checked_add(elements, extent * stride);
        }
        if (elements > std::numeric_limits<uint64_t>::max() / elem_size) {
          throw std::runtime_error("Tensor interval overflow in request slots: " + name);
        }
        int64_t offset = tensor.at("offset").get<int64_t>();
        auto io = named_io.find(name);
        if (offset < 0 || io == named_io.end()) {
          throw std::runtime_error("Unresolved tensor view in request slots: " + name);
        }
        std::string allocation = "allocation:" + name;
        uint64_t base = 0;
        if (io->second->type == IODesc::TorchTensor) {
          if (io->second->torch_data_ptr == nullptr) {
            throw std::runtime_error("Unbound torch tensor in request slots: " + name);
          }
          // Different attachment names may alias the same torch storage.
          allocation = "torch-address-space";
          base = reinterpret_cast<uintptr_t>(io->second->torch_data_ptr);
        } else if (io->second->type == IODesc::FusedTorchTensor ||
                   io->second->type == IODesc::NVSHMEMMallocTensor) {
          throw std::runtime_error("Unresolved or remote allocation in request slots: " + name);
        }
        uint64_t begin = checked_add(base, static_cast<uint64_t>(offset));
        accesses[allocation].push_back(
            {d, data_slot_ids[d], writes, begin,
             checked_add(begin, elements * elem_size), name});
      }
    }
  }
  // Conservative byte intervals also cover strided views. Read/read sharing
  // is allowed; any cross-slot overlap involving a write is rejected.
  for (auto const &entry : accesses) {
    for (auto const &writer : entry.second) {
      if (!writer.writes) {
        continue;
      }
      for (auto const &other : entry.second) {
        if (writer.slot != other.slot && writer.begin < other.end &&
            other.begin < writer.end) {
          throw std::runtime_error(
              "Cross-slot writable overlap: D" + std::to_string(writer.data) +
              " (slot " + std::to_string(writer.slot) + ", " + writer.tensor +
              ") overlaps D" + std::to_string(other.data) + " (slot " +
              std::to_string(other.slot) + ", " + other.tensor + ").");
        }
      }
    }
  }

  task_graph["request_slot_metadata_version"] = 1;
  task_graph["num_request_slots"] = num_slots;
  task_graph["data_slot_ids"] = data_slot_ids;
  task_graph["request_slots"] = json::array();
  for (size_t slot = 0; slot < num_slots; slot++) {
    task_graph["request_slots"].push_back(
        {{"slot_id", slot},
         {"data_ids", members[slot]},
         {"first_data_ids", roots[slot]},
         {"terminal_data_ids", terminals[slot]},
         {"num_terminal_data", terminals[slot].size()}});
  }
}

// Compilation validates ownership within each allocation. Runtime model
// rebinding may introduce aliases between previously separate allocations, so
// check their storage spans once at initialization as well. Conservatively
// reject distinct writable attachment names that overlap, even within one slot.
void emit_slot_binding_checks(
    mirage::transpiler::CodeKeeper &code,
    json const &task_graph,
    std::map<mirage::type::GuidType, IODesc> const &io_configs) {
  std::map<std::string, IODesc const *> named_io;
  std::function<void(IODesc const &)> collect = [&](IODesc const &io) {
    named_io.emplace(io.name, &io);
    for (IODesc const &sub : io.sub_descs) collect(sub);
  };
  for (auto const &entry : io_configs) collect(entry.second);
  std::map<std::string, bool> used;
  for (auto const &data : task_graph.at("all_data")) {
    for (bool writes : {false, true}) {
      for (auto const &tensor : data.at(writes ? "outputs" : "inputs")) {
        std::string name = tensor.at("base_ptr").get<std::string>();
        if (name != "nullptr") used[name] |= writes;
      }
    }
  }
  code.e("// Runtime rebinding cannot introduce overlapping writable storages.");
  code.e("struct SlotBoundStorage { uintptr_t begin, bytes; bool writes; };");
  code.e("std::vector<SlotBoundStorage> slot_storages = {");
  for (auto const &entry : used) {
    IODesc const &io = *named_io.at(entry.first);
    uint64_t elements = 1;
    for (int d = 0; d < io.tensor.num_dims; d++) {
      if (io.tensor.dim[d] <= 0 || io.tensor.stride[d] < 0) {
        throw std::runtime_error("Invalid storage span in resident slot binding");
      }
      uint64_t extent = static_cast<uint64_t>(io.tensor.dim[d] - 1) *
                        static_cast<uint64_t>(io.tensor.stride[d]);
      if (extent > std::numeric_limits<uint64_t>::max() - elements) {
        throw std::runtime_error("Resident slot binding span overflow");
      }
      elements += extent;
    }
    size_t element_bytes = runtime_dtype_size(io.tensor.data_type);
    if (element_bytes == 0 ||
        elements > std::numeric_limits<uint64_t>::max() / element_bytes) {
      throw std::runtime_error("Resident slot binding size overflow");
    }
    code.e("{reinterpret_cast<uintptr_t>(all_tensors.at($)), $ull, $},",
           json(entry.first).dump(), elements * element_bytes,
           entry.second ? "true" : "false");
  }
  code.e("};");
  code.e("for (auto const &s : slot_storages) {");
  code.e("if (s.begin == 0 || s.bytes > UINTPTR_MAX - s.begin)");
  code.e("throw std::runtime_error(\"Invalid resident slot storage binding\");");
  code.e("}");
  code.e("for (size_t i = 0; i < slot_storages.size(); ++i) {");
  code.e("for (size_t j = 0; j < i; ++j) {");
  code.e("auto const &a = slot_storages[i]; auto const &b = slot_storages[j];");
  code.e("if ((a.writes || b.writes) && a.begin < b.begin + b.bytes &&");
  code.e("    b.begin < a.begin + a.bytes)");
  code.e("throw std::runtime_error(\"Resident slot bindings alias distinct writable attachment names\");");
  code.e("}");
  code.e("}");
}

TaskGraphResult print_task_graph(
    mirage::kernel::Graph const &graph,
    int num_gpus,
    int my_gpu_id,
    std::vector<FullTaskDesc> const &all_tasks,
    std::vector<EventDesc> const &all_events,
    std::vector<TaskId> const &first_tasks,
    OperatorTaskMaps const &all_task_maps,
    std::unordered_map<kn::KNOperator const *,
                       std::tuple<int, int, TaskType, int>> const &task_configs,
    std::map<mirage::type::GuidType, IODesc> const &io_configs,
    DataAwareTaskGraphFlavor flavor,
    bool use_json_format,
    std::vector<int> const &data_slot_ids = {}) {
  using mirage::runtime::IODesc;
  if (!use_json_format) {
    throw std::runtime_error(
        "Resident task graph generation requires JSON output to preserve "
        "TaskIds.");
  }
  ResidentExecutionMode resident_execution_mode =
      flavor == DataAwareTaskGraphFlavor::STREAMING
          ? RESIDENT_EXECUTION_STREAMING
          : get_resident_execution_mode();
  StreamingBaseExecutionMode streaming_base_mode =
      flavor == DataAwareTaskGraphFlavor::STREAMING
          ? get_streaming_base_execution_mode()
          : STREAMING_BASE_EXECUTION_LEGACY_EVENT;
  ResidentTaskGraph resident_graph =
      build_data_aware_task_graph(graph,
                                  all_tasks,
                                  all_events,
                                  all_task_maps,
                                  flavor,
                                  resident_execution_mode,
                                  streaming_base_mode);
  validate_resident_task_graph(resident_graph);
  mirage::transpiler::CodeKeeper code;
  mirage::transpiler::CodeKeeper tgbody;
  tgbody.inc_indent();
  code.e("#define MIRAGE_RESIDENT_EXECUTION_MODE_VALUE $",
         static_cast<uint32_t>(resident_execution_mode));
  code.e("#define MIRAGE_STREAMING_BASE_EXECUTION_MODE_VALUE $",
         static_cast<uint32_t>(streaming_base_mode));
  code.e("#include \"$\"",
         flavor == DataAwareTaskGraphFlavor::STREAMING
             ? "streaming_persistent_kernel.cuh"
             : "resident_persistent_kernel.cuh");
  if (use_json_format) {
    code.e("#include <nlohmann/json.hpp>");
    code.e("#include <fstream>");
    code.e("#include <filesystem>");
    code.e("using json = nlohmann::json;");
  }
  code.e("using namespace mirage::runtime;");
  code.e("std::string g_task_graph_json_path;");
  code.e("size_t get_event_id(int my_gpu_id, size_t event_pos, bool "
         "nvshmem_event) {");
  code.e("size_t event_id = ((static_cast<size_t>(my_gpu_id) << 32) | "
         "event_pos);");
  code.e("if (nvshmem_event) {");
  code.e("event_id = event_id | EVENT_NVSHMEM_TAG;");
  code.e("}");
  code.e("return event_id;");
  code.e("}");
  code.e("");

  // function that loads json file and generates task graph
  if (use_json_format) {
    code.e("void construct_task_graph(int num_gpus,");
    code.e("                          int my_gpu_id,");
    code.e("                          std::vector<FullTaskDesc> &all_tasks,");
    code.e("                          std::vector<EventDesc> &all_events,");
    code.e("                          std::vector<TaskId> &first_tasks,");
    code.e("                          std::vector<ResidentTaskDesc> "
           "&resident_tasks,");
    code.e("                          std::vector<FullDataDesc> &all_fulldata,");
    code.e("                          std::vector<DataEdgeDesc> &data_edges,");
    code.e("                          std::vector<DataId> &first_data_ids,");
    code.e("                          std::vector<DataId> &task_to_data_id,");
    code.e("                          RequestSlotCompletionMetadata "
           "&request_slot_completion,");
    code.e("                          std::map<std::string, void*> const "
           "&all_tensors) {");
    code.e("std::string json_path = g_task_graph_json_path;");
    code.e("if (json_path.empty()) {");
    code.e("std::filesystem::path file_path(__FILE__);");
    code.e("json_path = file_path.parent_path().string()+\"/task_graph.json\";");
    code.e("}");
    code.e("std::ifstream json_file(json_path);");
    code.e("if (!json_file) throw std::runtime_error(\"Cannot open resident task graph: \" + json_path);");
    code.e("nlohmann::json json_task_graph;");
    code.e("json_file >> json_task_graph;");
    // load tasks
    code.e("for (json const &task : json_task_graph[\"all_tasks\"]) {");
    code.e("FullTaskDesc "
           "task_desc(static_cast<TaskType>(task.at(\"task_type\")),");
    code.e("            task.at(\"variant_id\"));");
    code.e("if (task.contains(\"profiler_group_id\")) {");
    code.e("task_desc.profiler_group_id = "
           "task.at(\"profiler_group_id\").get<uint32_t>();");
    code.e("}");
    code.e("task_desc.task_metadata.request_id = "
           "task.at(\"request_id\").get<int>();");
    code.e("task_desc.task_metadata.expert_offset = "
           "task.at(\"expert_offset\").get<int>();");
    code.e("task_desc.task_metadata.kv_idx = task.at(\"kv_idx\").get<int>();");
    code.e("task_desc.task_metadata.merge_task_offset = "
           "task.at(\"merge_task_offset\").get<int>();");
    code.e("task_desc.task_metadata.task_offset = "
           "task.at(\"task_offset\").get<int>();");
    code.e("if (task.at(\"trigger_event\").is_number_integer()) {");
    code.e("task_desc.trigger_event = task.at(\"trigger_event\").get<unsigned "
           "long long int>();");
    code.e("}");
    code.e("else {");
    code.e("assert(false);");
    code.e("}");
    code.e("if (task.at(\"dependent_event\").is_number_integer()) {");
    code.e("task_desc.dependent_event = "
           "task.at(\"dependent_event\").get<unsigned long long int>();");
    code.e("}");
    code.e("else {");
    code.e("assert(false);");
    code.e("}");

    // load inputs
    code.e("task_desc.num_inputs = 0;");
    code.e("for (json const &tensor : task[\"inputs\"]) {");
    code.e("TensorDesc input;");
    code.e("std::string name = tensor.at(\"base_ptr\").get<std::string>();");
    code.e("assert(all_tensors.find(name) != all_tensors.end());");
    code.e("off_t offset = tensor.at(\"offset\").get<off_t>();");
    code.e("input.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;");
    code.e(
        "assert(tensor.at(\"dims\").size() == tensor.at(\"strides\").size());");
    code.e("input.num_dims = tensor.at(\"dims\").size();");
    code.e("input.data_type = tensor.at(\"data_type\").get<int>();");
    code.e("for (int i = 0; i < input.num_dims; i++) {");
    code.e("input.dim[i] = tensor[\"dims\"][i].get<int>();");
    code.e("input.stride[i] = tensor[\"strides\"][i].get<int>();");
    code.e("}");

    code.e("task_desc.inputs[task_desc.num_inputs++] = input;");
    code.e("}");
    // load outputs
    code.e("task_desc.num_outputs = 0;");
    code.e("for (json const &tensor : task[\"outputs\"]) {");
    code.e("TensorDesc output;");
    code.e("std::string name = tensor.at(\"base_ptr\").get<std::string>();");
    code.e("assert(all_tensors.find(name) != all_tensors.end());");
    code.e("off_t offset = tensor.at(\"offset\").get<off_t>();");
    code.e(
        "output.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;");
    code.e(
        "assert(tensor.at(\"dims\").size() == tensor.at(\"strides\").size());");
    code.e("output.num_dims = tensor.at(\"dims\").size();");
    code.e("output.data_type = tensor.at(\"data_type\").get<int>();");
    code.e("for (int i = 0; i < output.num_dims; i++) {");
    code.e("output.dim[i] = tensor[\"dims\"][i];");
    code.e("output.stride[i] = tensor[\"strides\"][i];");
    code.e("}");

    code.e("task_desc.outputs[task_desc.num_outputs++] = output;");
    code.e("}");

    // create TMA desc for each task
    code.e("#ifdef MPK_ENABLE_TMA");
    // Hopper Tasks
    code.e("if (task.at(\"task_type\") > TASK_HOPPER_TASK_BEGIN && "
           "task.at(\"task_type\") < TASK_HOPPER_TASK_END) {");
    code.e("create_tma_desc_by_task(task_desc);");
    code.e("}");
    // SM100 Tasks
    code.e("if (task.at(\"task_type\") > TASK_SM100_TMA_START_TASK && "
           "task.at(\"task_type\") < TASK_SM100_TMA_END_TASK) {");
    code.e("create_tma_desc_by_task(task_desc);");
    code.e("}");
    code.e("#endif");
    code.e("all_tasks.push_back(task_desc);");
    code.e("if (task.contains(\"data_id\")) {");
    code.e("task_to_data_id.push_back(task.at(\"data_id\").get<uint32_t>());");
    code.e("} else {");
    code.e("task_to_data_id.push_back(DATA_INVALID_ID);");
    code.e("}");
    code.e("}");
    // load events
    // In streaming legacy-base mode, data items' trigger_event and
    // dependent_event fields are indices into the full all_events array.
    // Loading only control_events (3 entries) leaves the runtime's
    // all_event_counters array too small, causing out-of-bounds accesses
    // and a deadlock because the EOG event never fires.
    // In streaming hybrid-base or resident_data modes, only the 3 control
    // events are needed (data-edge predecessor counts handle readiness).
    bool const needs_full_events =
        flavor == DataAwareTaskGraphFlavor::STREAMING &&
        streaming_base_mode == STREAMING_BASE_EXECUTION_LEGACY_EVENT;
    if (needs_full_events) {
      code.e("json event_jsons = json_task_graph[\"all_events\"];");
    } else {
      code.e("json event_jsons = json_task_graph.contains(\"control_events\") "
             "? json_task_graph[\"control_events\"] "
             ": json_task_graph[\"all_events\"];");
    }
    code.e("for (json const &e : event_jsons) {");
    code.e("EventType event_type = "
           "static_cast<EventType>(e.at(\"event_type\").get<int>());");
    code.e("int num_triggers = e.at(\"num_triggers\").get<int>();");
    code.e("int first_task_id = e.at(\"first_task_id\").get<int>();");
    code.e("int last_task_id = e.at(\"last_task_id\").get<int>();");
    code.e("all_events.push_back(EventDesc(event_type, num_triggers, "
           "first_task_id, last_task_id));");
    code.e("}");
    // load first tasks
    code.e("for (json const &t : json_task_graph[\"first_tasks\"]) {");
    code.e("first_tasks.push_back(t.get<int>());");
    code.e("}");
    code.e("if (json_task_graph.contains(\"resident_tasks\")) {");
    code.e("for (json const &task : json_task_graph[\"resident_tasks\"]) {");
    code.e("ResidentTaskDesc resident_task("
           "static_cast<TaskType>(task.at(\"task_type\").get<int>()),");
    code.e("                             task.at(\"variant_id\").get<int>());");
    code.e("resident_task.profiler_group_id = "
           "task.at(\"profiler_group_id\").get<uint32_t>();");
    code.e("resident_task.num_inputs = task.at(\"num_inputs\").get<int>();");
    code.e("resident_task.num_outputs = task.at(\"num_outputs\").get<int>();");
    code.e("resident_task.execution_kind = task.contains(\"execution_kind\") ? "
           "task.at(\"execution_kind\").get<uint32_t>() : "
           "static_cast<uint32_t>(RESIDENT_TASK_EXECUTION_PRELAUNCHED);");
    code.e("resident_task.max_parallelism = "
           "task.at(\"max_parallelism\").get<int>();");
    code.e("resident_task.total_data_count = "
           "task.at(\"total_data_count\").get<int>();");
    code.e("resident_tasks.push_back(resident_task);");
    code.e("}");
    code.e("}");
    code.e("if (json_task_graph.contains(\"all_data\")) {");
    code.e("for (json const &task : json_task_graph[\"all_data\"]) {");
    code.e("FullDataDesc data_desc;");
    code.e("data_desc.resident_task_id = "
           "task.at(\"resident_task_id\").get<uint32_t>();");
    code.e("data_desc.profiler_group_id = "
           "task.at(\"profiler_group_id\").get<uint32_t>();");
    code.e("data_desc.initial_predecessor_count = "
           "task.at(\"initial_predecessor_count\").get<int>();");
    code.e("if (task.contains(\"trigger_event\") && "
           "task.at(\"trigger_event\").is_number_integer()) {");
    code.e("data_desc.trigger_event = "
           "task.at(\"trigger_event\").get<unsigned long long int>();");
    code.e("}");
    code.e("if (task.contains(\"dependent_event\") && "
           "task.at(\"dependent_event\").is_number_integer()) {");
    code.e("data_desc.dependent_event = "
           "task.at(\"dependent_event\").get<unsigned long long int>();");
    code.e("}");
    code.e("data_desc.task_metadata.request_id = "
           "task.at(\"request_id\").get<int>();");
    code.e("data_desc.task_metadata.expert_offset = "
           "task.at(\"expert_offset\").get<int>();");
    code.e("data_desc.task_metadata.kv_idx = task.at(\"kv_idx\").get<int>();");
    code.e("data_desc.task_metadata.merge_task_offset = "
           "task.at(\"merge_task_offset\").get<int>();");
    code.e("data_desc.task_metadata.task_offset = "
           "task.at(\"task_offset\").get<int>();");
    code.e("int input_idx = 0;");
    code.e("for (json const &tensor : task[\"inputs\"]) {");
    code.e("TensorDesc input;");
    code.e("std::string name = tensor.at(\"base_ptr\").get<std::string>();");
    code.e("assert(all_tensors.find(name) != all_tensors.end());");
    code.e("off_t offset = tensor.at(\"offset\").get<off_t>();");
    code.e("input.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;");
    code.e("input.num_dims = tensor.at(\"dims\").size();");
    code.e("input.data_type = tensor.at(\"data_type\").get<int>();");
    code.e("for (int i = 0; i < input.num_dims; i++) {");
    code.e("input.dim[i] = tensor[\"dims\"][i].get<int>();");
    code.e("input.stride[i] = tensor[\"strides\"][i].get<int>();");
    code.e("}");
    code.e("data_desc.inputs[input_idx++] = input;");
    code.e("}");
    code.e("int output_idx = 0;");
    code.e("for (json const &tensor : task[\"outputs\"]) {");
    code.e("TensorDesc output;");
    code.e("std::string name = tensor.at(\"base_ptr\").get<std::string>();");
    code.e("assert(all_tensors.find(name) != all_tensors.end());");
    code.e("off_t offset = tensor.at(\"offset\").get<off_t>();");
    code.e("output.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;");
    code.e("output.num_dims = tensor.at(\"dims\").size();");
    code.e("output.data_type = tensor.at(\"data_type\").get<int>();");
    code.e("for (int i = 0; i < output.num_dims; i++) {");
    code.e("output.dim[i] = tensor[\"dims\"][i].get<int>();");
    code.e("output.stride[i] = tensor[\"strides\"][i].get<int>();");
    code.e("}");
    code.e("data_desc.outputs[output_idx++] = output;");
    code.e("}");
    code.e("#ifdef MPK_ENABLE_TMA");
    code.e("if (data_desc.resident_task_id < resident_tasks.size()) {");
    code.e("create_tma_desc_by_data("
           "resident_tasks.at(data_desc.resident_task_id), data_desc);");
    code.e("}");
    code.e("#endif");
    code.e("all_fulldata.push_back(data_desc);");
    code.e("}");
    code.e("}");
    code.e("if (json_task_graph.contains(\"data_edges\")) {");
    code.e("for (json const &edge : json_task_graph[\"data_edges\"]) {");
    code.e("data_edges.push_back(DataEdgeDesc("
           "edge.at(\"src_data_id\").get<uint32_t>(), "
           "edge.at(\"dst_data_id\").get<uint32_t>()));");
    code.e("}");
    code.e("}");
    code.e("if (json_task_graph.contains(\"first_data_ids\")) {");
    code.e("for (json const &data_id : json_task_graph[\"first_data_ids\"]) {");
    code.e("first_data_ids.push_back(data_id.get<uint32_t>());");
    code.e("}");
    code.e("}");
    code.e("if (json_task_graph.contains(\"request_slot_metadata_version\")) {");
    code.e("if (json_task_graph.at(\"request_slot_metadata_version\").get<int>() != 1) {");
    code.e("throw std::runtime_error(\"Unsupported request-slot metadata version\");");
    code.e("}");
    code.e("uint32_t num_slots = json_task_graph.at(\"num_request_slots\").get<uint32_t>();");
    code.e("if (num_slots == 0 || num_slots > all_fulldata.size()) {");
    code.e("throw std::runtime_error(\"Invalid number of request slots\");");
    code.e("}");
    code.e("request_slot_completion.data_slot_ids =");
    code.e("    json_task_graph.at(\"data_slot_ids\").get<std::vector<uint32_t>>();");
    code.e("request_slot_completion.num_terminal_data.assign(num_slots, 0u);");
    code.e("for (auto const &slot : json_task_graph.at(\"request_slots\")) {");
    code.e("uint32_t slot_id = slot.at(\"slot_id\").get<uint32_t>();");
    code.e("uint32_t count = slot.at(\"num_terminal_data\").get<uint32_t>();");
    code.e("if (slot_id >= num_slots || count == 0 ||");
    code.e("    request_slot_completion.num_terminal_data[slot_id] != 0) {");
    code.e("throw std::runtime_error(\"Invalid or duplicate request-slot terminal count\");");
    code.e("}");
    code.e("request_slot_completion.num_terminal_data[slot_id] = count;");
    code.e("}");
    code.e("}");
    code.e("}");
    code.e("");
  }

  code.e("static void _init_persistent_kernel(std::vector<FullTaskDesc> "
         "&all_tasks,");
  code.e("                                    std::vector<EventDesc> "
         "&all_events,");
  code.e("                                  std::vector<TaskId> &first_tasks,");
  code.e("                                  std::vector<ResidentTaskDesc> "
         "&resident_tasks,");
  code.e("                                  std::vector<FullDataDesc> "
         "&all_fulldata,");
  code.e("                                  std::vector<DataEdgeDesc> "
         "&data_edges,");
  code.e("                                  std::vector<DataId> &first_data_ids,");
  code.e("                                  std::vector<DataId> &task_to_data_id,");
  code.e("                                  RequestSlotCompletionMetadata "
         "&request_slot_completion,");
  code.e("                                  int num_gpus,");
  code.e("                                  int my_gpu_id,");
  code.e("                                  std::map<std::string, void*> const "
         "&model_tensors) {");
  code.e("assert(num_gpus == $);", num_gpus);

  if (use_json_format) {
    code.e("std::map<std::string, void*> all_tensors;");
  }
  for (auto const &iter : io_configs) {
    IODesc desc = iter.second;
    switch (desc.type) {
      case IODesc::TorchTensor: {
        code.e("char *$ = static_cast<char*>(model_tensors.at(\"$\"));",
               desc.name, desc.name);
        if (use_json_format) {
          code.e("all_tensors[\"$\"] = $;", desc.name, desc.name);
        }
        break;
      }
      case IODesc::FusedTorchTensor: {
        for (auto const &sdesc : desc.sub_descs) {
          code.e("char *$ = static_cast<char*>(model_tensors.at(\"$\"));",
                 sdesc.name, sdesc.name);
          if (use_json_format) {
            code.e("all_tensors[\"$\"] = $;", sdesc.name, sdesc.name);
          }
        }
        break;
      }
      case IODesc::CUDAMallocTensor: {
        code.e("void *$;", desc.name);
        size_t size = mirage::type::get_datatype_size(
            static_cast<type::DataType>(desc.tensor.data_type));
        for (int i = 0; i < desc.tensor.num_dims; i++) {
          size *= desc.tensor.dim[i];
        }
        code.e("CUDA_CHECK(cudaMalloc(&$, $));", desc.name, size);
        if (use_json_format) {
          code.e("all_tensors[\"$\"] = $;", desc.name, desc.name);
        }
        break;
      }
      case IODesc::NVSHMEMMallocTensor: {
        size_t size = mirage::type::get_datatype_size(
            static_cast<type::DataType>(desc.tensor.data_type));
        for (int i = 0; i < desc.tensor.num_dims; i++) {
          size *= desc.tensor.dim[i];
        }
        code.e("void *$ = nvshmem_malloc($);", desc.name, size);
        code.e("assert($ != nullptr);", desc.name);
        if (use_json_format) {
          code.e("all_tensors[\"$\"] = $;", desc.name, desc.name);
        }
        break;
      }
      case IODesc::ShuffledTorchTensor: {
        code.e("char *$;", desc.name);
        size_t size = mirage::type::get_datatype_size(
            static_cast<type::DataType>(desc.tensor.data_type));
        for (int i = 0; i < desc.tensor.num_dims; i++) {
          size *= desc.tensor.dim[i];
        }
        code.e("CUDA_CHECK(cudaMalloc(&$, $));", desc.name, size);

        size_t bytes_per_row = size / desc.tensor.dim[0];
        size_t bytes_per_group = 0;
        std::vector<size_t> bytes_per_tensor;
        for (int i = 0; i < desc.sub_descs.size(); i++) {
          bytes_per_group +=
              bytes_per_row * desc.sub_descs[i].tensor.dim[0] / desc.num_groups;
          bytes_per_tensor.push_back(bytes_per_row *
                                     desc.sub_descs[i].tensor.dim[0] /
                                     desc.num_groups);
        }
        size_t start_addr_offset = 0;
        for (int i = 0; i < desc.sub_descs.size(); i++) {
          code.e("CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>($ + "
                 "$), $, "
                 "model_tensors.at(\"$\"), $, $, $, "
                 "cudaMemcpyDeviceToDevice));",
                 desc.name,         /*dst address*/
                 start_addr_offset, /*dst bytes offset between each copy*/
                 bytes_per_group,   /*dst bytes offset between each copy*/
                 desc.sub_descs[i].name, /*runtime source tensor*/
                 bytes_per_tensor[i], /*src bytes offset between each copy*/
                 bytes_per_tensor[i], /*width*/
                 desc.num_groups      /*height*/
          );
          start_addr_offset += bytes_per_tensor[i];
        }
        if (use_json_format) {
          code.e("all_tensors[\"$\"] = $;", desc.name, desc.name);
        }
        break;
      }
      default:
        assert(false);
    }
  }
  json json_task_graph = {{"schema_version", 2},
                          {"resident_execution_mode",
                           static_cast<uint32_t>(resident_execution_mode)},
                          {"streaming_base_execution_mode",
                           static_cast<uint32_t>(streaming_base_mode)},
                          {"all_tasks", {}},
                          {"all_events", {}},
                          {"first_tasks", {}},
                          {"resident_tasks", {}},
                          {"all_data", {}},
                          {"data_edges", {}},
                          {"first_data_ids", {}},
                          {"control_events", {}}};
  // generate task[0]
  {
    tgbody.e("all_tasks.push_back(FullTaskDesc(TASK_TERMINATE));");
    tgbody.e("task_to_data_id.push_back(DATA_INVALID_ID);");
    json_task_graph["all_tasks"].push_back(
        json{{"task_type", TASK_TERMINATE},
             {"variant_id", 0},
             {"profiler_group_id", all_tasks[0].profiler_group_id},
             {"resident_task_id", RESIDENT_TASK_INVALID_ID},
             {"data_id", DATA_INVALID_ID},
             {"inputs", {}},
             {"outputs", {}},
             {"trigger_event", EVENT_INVALID_ID},
             {"dependent_event", EVENT_INVALID_ID},
             {"request_id", -1},
             {"expert_offset", -1},
             {"kv_idx", -1},
             {"merge_task_offset", -1},
             {"task_offset", -1}});
  }
  // generate task[1]
  {
    tgbody.e("all_tasks.push_back(FullTaskDesc(TASK_BEGIN_TASK_GRAPH));");
    tgbody.e("all_tasks.back().profiler_group_id = $;",
             all_tasks[1].profiler_group_id);
    tgbody.e("task_to_data_id.push_back(DATA_INVALID_ID);");
    json_task_graph["all_tasks"].push_back(
        json{{"task_type", TASK_BEGIN_TASK_GRAPH},
             {"variant_id", 0},
             {"profiler_group_id", all_tasks[1].profiler_group_id},
             {"resident_task_id", RESIDENT_TASK_INVALID_ID},
             {"data_id", DATA_INVALID_ID},
             {"inputs", {}},
             {"outputs", {}},
             {"trigger_event",
              get_event_id(my_gpu_id, 1 /*event_pos*/, false /*is_nvshmem*/)},
             {"dependent_event", EVENT_INVALID_ID},
             {"request_id", -1},
             {"expert_offset", -1},
             {"kv_idx", -1},
             {"merge_task_offset", -1},
             {"task_offset", -1}});
  }
  // Branch tasks can interleave across operators, so buffer by global TaskId.
  std::vector<json> json_tasks(all_tasks.size());
  std::vector<bool> task_filled(all_tasks.size(), false);
  std::vector<json> json_data(resident_graph.all_data.size());
  for (auto const &op : graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      continue;
    }
    assert(op->op_type == type::KNOperatorType::KN_CUSTOMIZED_OP);
    std::tuple<int, int, TaskType, int> task_config =
        task_configs.find(op)->second;

    assert(all_task_maps.find(op) != all_task_maps.end());
    TaskMap const &task_map = all_task_maps.find(op)->second;
    // Customized op
    kn::KNCustomizedOp const *cur_op =
        dynamic_cast<kn::KNCustomizedOp const *>(op);
    tb::Graph const &bgraph = cur_op->bgraph;
    dim3 bid;
    std::vector<tb::TBInputOp *> input_ops;
    std::vector<tb::TBInputOp *> output_ops;
    int num_inputs = std::get<0>(task_config);
    // int num_outputs = std::get<1>(task_config);
    TaskType task_type = std::get<2>(task_config);
    for (auto const &op : bgraph.operators) {
      assert(op->op_type == mirage::type::TB_INPUT_OP);
      if (input_ops.size() < (size_t)num_inputs) {
        input_ops.push_back(static_cast<tb::TBInputOp *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBInputOp *>(op));
      }
    }

    unsigned cur_op_num_subtasks = get_num_subtasks(num_gpus, task_type);

    for (bid.x = 0; bid.x < bgraph.grid_dim.x; bid.x++) {
      for (bid.y = 0; bid.y < bgraph.grid_dim.y; bid.y++) {
        for (bid.z = 0; bid.z < bgraph.grid_dim.z; bid.z++) {
          for (int subtask_id = 0; subtask_id < cur_op_num_subtasks;
               subtask_id++) {
            TaskId task_id = task_map.at(bid).at(subtask_id);
            if (task_id < 2 || task_id >= all_tasks.size()) {
              throw std::runtime_error(
                  "Resident serialization has an invalid compute TaskId.");
            }
            if (task_filled[task_id]) {
              throw std::runtime_error(
                  "Resident serialization has a duplicate TaskId.");
            }
            FullTaskDesc task_desc = all_tasks[task_id];
            tgbody.e("// task[$]", task_id);
            tgbody.e("{");
            tgbody.e("FullTaskDesc task_desc(static_cast<TaskType>($));",
                     task_desc.task_type);
            size_t event_gpu_id = ((task_desc.trigger_event >> 32) & 0xffff);
            size_t event_pos = (task_desc.trigger_event & 0xffffffff);
            bool is_nvshmem_event =
                ((task_desc.trigger_event & EVENT_NVSHMEM_TAG) > 0);
            // assert(event_gpu_id == my_gpu_id);
            // assert(!is_nvshmem_event);
            DataId data_id = DATA_INVALID_ID;
            ResidentTaskId resident_task_id = RESIDENT_TASK_INVALID_ID;
            if (task_id < resident_graph.task_to_data_id.size()) {
              data_id = resident_graph.task_to_data_id[task_id];
            }
            if (data_id != DATA_INVALID_ID &&
                data_id < resident_graph.all_data.size()) {
              resident_task_id =
                  resident_graph.all_data[data_id].resident_task_id;
            }
            json json_task;
            json_task = {
                {"task_type", task_desc.task_type},
                {"variant_id", task_desc.variant_id},
                {"profiler_group_id", task_desc.profiler_group_id},
                {"resident_task_id", resident_task_id},
                {"data_id", data_id},
                {"inputs", {}},
                {"outputs", {}},
                {"trigger_event", task_desc.trigger_event},
                {"dependent_event", task_desc.dependent_event},
                {"request_id", task_desc.task_metadata.request_id},
                {"expert_offset", task_desc.task_metadata.expert_offset},
                {"kv_idx", task_desc.task_metadata.kv_idx},
                {"merge_task_offset",
                 task_desc.task_metadata.merge_task_offset},
                {"task_offset", task_desc.task_metadata.task_offset}};
            json json_data_desc;
            bool has_data_desc =
                (data_id != DATA_INVALID_ID &&
                 data_id < resident_graph.all_data.size());
            if (has_data_desc) {
              FullDataDesc const &data_desc = resident_graph.all_data[data_id];
              json_data_desc = {
                  {"resident_task_id", data_desc.resident_task_id},
                  {"profiler_group_id", data_desc.profiler_group_id},
                  {"initial_predecessor_count",
                   data_desc.initial_predecessor_count},
                  {"trigger_event", data_desc.trigger_event},
                  {"dependent_event", data_desc.dependent_event},
                  {"request_id", data_desc.task_metadata.request_id},
                  {"expert_offset", data_desc.task_metadata.expert_offset},
                  {"kv_idx", data_desc.task_metadata.kv_idx},
                  {"merge_task_offset",
                   data_desc.task_metadata.merge_task_offset},
                  {"task_offset", data_desc.task_metadata.task_offset},
                  {"inputs", {}},
                  {"outputs", {}}};
            }

            for (int i = 0; i < task_desc.num_inputs; i++) {
              if (input_ops[i]->dtensor == kernel::DTensor::EMPTY_TENSOR) {
                json json_dims = json::array();
                json json_strides = json::array();
                json_task["inputs"].push_back(
                    json{{"base_ptr", "nullptr"},
                         {"offset", 0},
                         {"data_type", type::DT_UNKNOWN},
                         {"dims", json_dims},
                         {"strides", json_strides}});
                if (has_data_desc) {
                  json_data_desc["inputs"].push_back(
                      json{{"base_ptr", "nullptr"},
                           {"offset", 0},
                           {"data_type", type::DT_UNKNOWN},
                           {"dims", json_dims},
                           {"strides", json_strides}});
                }
                continue;
              }
              off_t offset = 0;
              int num_dims = input_ops[i]->dtensor.num_dims;
              int3 input_map = input_ops[i]->input_map;
              IODesc io_desc =
                  io_configs.find(input_ops[i]->dtensor.guid)->second;
              assert(input_ops[i]->dtensor.owner_op->op_type ==
                     type::KN_INPUT_OP);
              if (io_desc.type == IODesc::FusedTorchTensor) {
                // Currently assert that we fuse the 0-th dim (i.e., 0)
                int fused_group_size = 0;
                std::vector<int> group_sizes;
                for (auto const &sub_desc : io_desc.sub_descs) {
                  assert(sub_desc.tensor.num_dims == num_dims);
                  assert(sub_desc.tensor.dim[0] % io_desc.num_groups == 0);
                  int my_group_size =
                      sub_desc.tensor.dim[0] / io_desc.num_groups;
                  fused_group_size += my_group_size;
                  group_sizes.push_back(my_group_size);
                }
                assert(io_desc.tensor.dim[0] ==
                       fused_group_size * io_desc.num_groups);
                assert(io_desc.tensor.num_dims == num_dims);
                int fused_dim_off = 0;
                if (input_map.x == 0) {
                  fused_dim_off =
                      io_desc.tensor.dim[0] / bgraph.grid_dim.x * bid.x;
                }
                if (input_map.y == 0) {
                  fused_dim_off =
                      io_desc.tensor.dim[0] / bgraph.grid_dim.y * bid.y;
                }
                if (input_map.z == 0) {
                  fused_dim_off =
                      io_desc.tensor.dim[0] / bgraph.grid_dim.z * bid.z;
                }
                int fused_dim_off_in_group = fused_dim_off % fused_group_size;
                size_t index = 0;
                while (index < group_sizes.size()) {
                  if (fused_dim_off_in_group >= group_sizes[index]) {
                    fused_dim_off_in_group -= group_sizes[index];
                    index++;
                  } else {
                    break;
                  }
                }
                IODesc sub_desc = io_desc.sub_descs[index];
                int fused_dim_off_subtensor =
                    fused_dim_off / fused_group_size * group_sizes[index] +
                    fused_dim_off_in_group;
                // Assert that it is within range
                assert(fused_dim_off_subtensor < sub_desc.tensor.dim[0]);
                if (input_map.x > 0) {
                  size_t block_size =
                      sub_desc.tensor.dim[input_map.x] / bgraph.grid_dim.x;
                  offset +=
                      block_size * bid.x * sub_desc.tensor.stride[input_map.x];
                } else if (input_map.x == 0) {
                  offset += fused_dim_off_subtensor *
                            sub_desc.tensor.stride[input_map.x];
                }
                if (input_map.y > 0) {
                  size_t block_size =
                      sub_desc.tensor.dim[input_map.y] / bgraph.grid_dim.y;
                  offset +=
                      block_size * bid.y * sub_desc.tensor.stride[input_map.y];
                } else if (input_map.y == 0) {
                  offset += fused_dim_off_subtensor *
                            sub_desc.tensor.stride[input_map.y];
                }
                if (input_map.z > 0) {
                  size_t block_size =
                      sub_desc.tensor.dim[input_map.z] / bgraph.grid_dim.z;
                  offset +=
                      block_size * bid.z * sub_desc.tensor.stride[input_map.z];
                } else if (input_map.z == 0) {
                  offset += fused_dim_off_subtensor *
                            sub_desc.tensor.stride[input_map.z];
                }
                tgbody.e("TensorDesc input$;", i);
                tgbody.e("input$.base_ptr = static_cast<char*>($) + $;",
                         i,
                         sub_desc.name,
                         offset * type::get_datatype_size(
                                      static_cast<type::DataType>(
                                          sub_desc.tensor.data_type)));
                tgbody.e(
                    "input$.num_dims = $;", i, task_desc.inputs[i].num_dims);
                tgbody.e(
                    "input$.data_type = $;", i, task_desc.inputs[i].data_type);
                json json_dims = json::array();
                json json_strides = json::array();
                for (int d = 0; d < task_desc.inputs[i].num_dims; d++) {
                  tgbody.e(
                      "input$.dim[$] = $;", i, d, task_desc.inputs[i].dim[d]);
                  tgbody.e(
                      "input$.stride[$] = $;", i, d, sub_desc.tensor.stride[d]);
                  json_dims.push_back(task_desc.inputs[i].dim[d]);
                  json_strides.push_back(sub_desc.tensor.stride[d]);
                }
                tgbody.e("task_desc.inputs[$] = input$;", i, i);
                json_task["inputs"].push_back(json{
                    {"base_ptr", sub_desc.name},
                    {"offset",
                     offset *
                         type::get_datatype_size(static_cast<type::DataType>(
                             sub_desc.tensor.data_type))},
                    {"data_type", task_desc.inputs[i].data_type},
                    {"dims", json_dims},
                    {"strides", json_strides}});
                if (has_data_desc) {
                  json_data_desc["inputs"].push_back(json{
                      {"base_ptr", sub_desc.name},
                      {"offset",
                       offset *
                           type::get_datatype_size(static_cast<type::DataType>(
                               sub_desc.tensor.data_type))},
                      {"data_type", task_desc.inputs[i].data_type},
                      {"dims", json_dims},
                      {"strides", json_strides}});
                }
              } else {
                // Non-fused case, use io_desc
                if (input_map.x >= 0) {
                  size_t block_size =
                      io_desc.tensor.dim[input_map.x] / bgraph.grid_dim.x;
                  offset +=
                      block_size * bid.x * io_desc.tensor.stride[input_map.x];
                }
                if (input_map.y >= 0) {
                  size_t block_size =
                      io_desc.tensor.dim[input_map.y] / bgraph.grid_dim.y;
                  offset +=
                      block_size * bid.y * io_desc.tensor.stride[input_map.y];
                }
                if (input_map.z >= 0) {
                  size_t block_size =
                      io_desc.tensor.dim[input_map.z] / bgraph.grid_dim.z;
                  offset +=
                      block_size * bid.z * io_desc.tensor.stride[input_map.z];
                }
                tgbody.e("TensorDesc input$;", i);
                tgbody.e("input$.base_ptr = static_cast<char*>($) + $;",
                         i,
                         io_desc.name,
                         offset * type::get_datatype_size(
                                      static_cast<type::DataType>(
                                          io_desc.tensor.data_type)));
                tgbody.e(
                    "input$.num_dims = $;", i, task_desc.inputs[i].num_dims);
                tgbody.e(
                    "input$.data_type = $;", i, task_desc.inputs[i].data_type);
                json json_dims = json::array();
                json json_strides = json::array();
                for (int d = 0; d < task_desc.inputs[i].num_dims; d++) {
                  tgbody.e(
                      "input$.dim[$] = $;", i, d, task_desc.inputs[i].dim[d]);
                  tgbody.e("input$.stride[$] = $;",
                           i,
                           d,
                           task_desc.inputs[i].stride[d]);
                  json_dims.push_back(task_desc.inputs[i].dim[d]);
                  json_strides.push_back(task_desc.inputs[i].stride[d]);
                }
                tgbody.e("task_desc.inputs[$] = input$;", i, i);
                json_task["inputs"].push_back(json{
                    {"base_ptr", io_desc.name},
                    {"offset",
                     offset *
                         type::get_datatype_size(static_cast<type::DataType>(
                             io_desc.tensor.data_type))},
                    {"data_type", task_desc.inputs[i].data_type},
                    {"dims", json_dims},
                    {"strides", json_strides}});
                if (has_data_desc) {
                  json_data_desc["inputs"].push_back(json{
                      {"base_ptr", io_desc.name},
                      {"offset",
                       offset *
                           type::get_datatype_size(static_cast<type::DataType>(
                               io_desc.tensor.data_type))},
                      {"data_type", task_desc.inputs[i].data_type},
                      {"dims", json_dims},
                      {"strides", json_strides}});
                }
              }
            }

            for (int i = 0; i < task_desc.num_outputs; i++) {
              off_t offset = 0;
              if (task_type == runtime::TASK_NVSHMEM_ALLGATHER_STRIDED_PUT) {
                // A special case for buffer-style tensors.
                offset = my_gpu_id * input_ops[0]->dtensor.num_elements();
              }
              int3 output_map = output_ops[i]->input_map;
              IODesc io_desc =
                  io_configs.find(output_ops[i]->dtensor.guid)->second;
              assert(io_desc.type != IODesc::FusedTorchTensor);
              if (output_map.x >= 0) {
                size_t block_size =
                    io_desc.tensor.dim[output_map.x] / bgraph.grid_dim.x;
                offset +=
                    block_size * bid.x * io_desc.tensor.stride[output_map.x];
              }
              if (output_map.y >= 0) {
                size_t block_size =
                    io_desc.tensor.dim[output_map.y] / bgraph.grid_dim.y;
                offset +=
                    block_size * bid.y * io_desc.tensor.stride[output_map.y];
              }
              if (output_map.z >= 0) {
                size_t block_size =
                    io_desc.tensor.dim[output_map.z] / bgraph.grid_dim.z;
                offset +=
                    block_size * bid.z * io_desc.tensor.stride[output_map.z];
              }

              tgbody.e("TensorDesc output$;", i);
              tgbody.e("output$.base_ptr = static_cast<char*>($) + $;",
                       i,
                       io_desc.name,
                       offset *
                           type::get_datatype_size(static_cast<type::DataType>(
                               io_desc.tensor.data_type)));
              tgbody.e(
                  "output$.num_dims = $;", i, task_desc.outputs[i].num_dims);
              tgbody.e(
                  "output$.data_type = $;", i, task_desc.outputs[i].data_type);
              json json_dims = json::array();
              json json_strides = json::array();
              for (int d = 0; d < task_desc.outputs[i].num_dims; d++) {
                tgbody.e(
                    "output$.dim[$] = $;", i, d, task_desc.outputs[i].dim[d]);
                tgbody.e("output$.stride[$] = $;",
                         i,
                         d,
                         task_desc.outputs[i].stride[d]);
                json_dims.push_back(task_desc.outputs[i].dim[d]);
                json_strides.push_back(task_desc.outputs[i].stride[d]);
              }
              tgbody.e("task_desc.outputs[$] = output$;", i, i);
              json_task["outputs"].push_back(json{
                  {"base_ptr", io_desc.name},
                  {"offset",
                   offset * type::get_datatype_size(static_cast<type::DataType>(
                                io_desc.tensor.data_type))},
                  {"data_type", task_desc.outputs[i].data_type},
                  {"dims", json_dims},
                  {"strides", json_strides}});
              if (has_data_desc) {
                json_data_desc["outputs"].push_back(json{
                    {"base_ptr", io_desc.name},
                    {"offset",
                     offset * type::get_datatype_size(
                                  static_cast<type::DataType>(
                                      io_desc.tensor.data_type))},
                    {"data_type", task_desc.outputs[i].data_type},
                    {"dims", json_dims},
                    {"strides", json_strides}});
              }
            }

            tgbody.e("all_tasks.push_back(task_desc);");
            tgbody.e("task_to_data_id.push_back($);", data_id);
            tgbody.e("}");
            json_tasks[task_id] = json_task;
            task_filled[task_id] = true;
            if (has_data_desc) {
              json_data[data_id] = json_data_desc;
            }
          } // subtask_id
        }   // bid.z
      }     // bid.y
    }       // bid.x
  }
  for (TaskId task_id = 2; task_id < all_tasks.size(); task_id++) {
    if (!task_filled[task_id]) {
      throw std::runtime_error(
          "Resident serialization is missing a compute TaskId.");
    }
    json_task_graph["all_tasks"].push_back(json_tasks[task_id]);
  }
  // Add all events
  for (auto const &event : all_events) {
    tgbody.e(
        "all_events.push_back(EventDesc(static_cast<EventType>($), $, $, $));",
        event.event_type,
        event.num_triggers,
        event.first_task_id,
        event.last_task_id);
    json_task_graph["all_events"].push_back(
        json{{"event_type", event.event_type},
             {"num_triggers", event.num_triggers},
             {"first_task_id", event.first_task_id},
             {"last_task_id", event.last_task_id}});
  }
  // Add first task
  for (auto const &task : first_tasks) {
    tgbody.e("first_tasks.push_back($);", task);
    json_task_graph["first_tasks"].push_back(task);
  }
  for (ResidentTaskId resident_task_id = 0;
       resident_task_id < resident_graph.resident_tasks.size();
       resident_task_id++) {
    ResidentTaskDesc const &resident_task =
        resident_graph.resident_tasks[resident_task_id];
    json_task_graph["resident_tasks"].push_back(
        json{{"task_type", resident_task.task_type},
             {"variant_id", resident_task.variant_id},
             {"profiler_group_id", resident_task.profiler_group_id},
             {"num_inputs", resident_task.num_inputs},
             {"num_outputs", resident_task.num_outputs},
             {"execution_kind", resident_task.execution_kind},
             {"max_parallelism", resident_task.max_parallelism},
             {"total_data_count", resident_task.total_data_count}});
  }
  for (size_t data_id = 0; data_id < json_data.size(); data_id++) {
    assert(!json_data[data_id].is_null());
    json_task_graph["all_data"].push_back(json_data[data_id]);
  }
  for (DataEdgeDesc const &edge : resident_graph.data_edges) {
    json_task_graph["data_edges"].push_back(
        json{{"src_data_id", edge.src_data_id},
             {"dst_data_id", edge.dst_data_id}});
  }
  for (DataId data_id : resident_graph.first_data_ids) {
    json_task_graph["first_data_ids"].push_back(data_id);
  }
  for (EventDesc const &event : all_events) {
    if (event.event_type == EVENT_EMPTY) {
      continue;
    }
    json_task_graph["control_events"].push_back(
        json{{"event_type", event.event_type},
             {"num_triggers", event.num_triggers},
             {"first_task_id", event.first_task_id},
             {"last_task_id", event.last_task_id}});
  }
  if (use_json_format) {
    // Add nullptr for tensors set as None
    code.e("all_tensors[\"nullptr\"] = nullptr;");
    if (!data_slot_ids.empty()) {
      emit_slot_binding_checks(code, json_task_graph, io_configs);
    }
    code.e("construct_task_graph(num_gpus, my_gpu_id, all_tasks, all_events, "
           "first_tasks, resident_tasks, all_fulldata, data_edges, "
           "first_data_ids, task_to_data_id, request_slot_completion, "
           "all_tensors);");
  } else {
    code.e(tgbody.to_string());
  }
  // ensure cudaMemcpyAsync is completed
  code.e("cudaDeviceSynchronize();");
  code.e("}");
  code.e("");

  // Generate task implementation
  std::map<TaskType, std::string> task_type_to_name;
  task_type_to_name[TASK_EMBEDDING] = "TASK_EMBEDDING";
  task_type_to_name[TASK_EMBEDDING_INPUT] = "TASK_EMBEDDING_INPUT";
  task_type_to_name[TASK_CHECKSUM_HOPPER] = "TASK_CHECKSUM_HOPPER";
  task_type_to_name[TASK_ATTENTION_SLOT_HOPPER] = "TASK_ATTENTION_SLOT_HOPPER";
  task_type_to_name[TASK_RMS_NORM] = "TASK_RMS_NORM";
  task_type_to_name[TASK_RMS_NORM_LINEAR] = "TASK_RMS_NORM_LINEAR";
  task_type_to_name[TASK_ATTENTION_1] = "TASK_ATTENTION_1";
  task_type_to_name[TASK_SILU_MUL] = "TASK_SILU_MUL";
  task_type_to_name[TASK_SILU_MUL_ROW] = "TASK_SILU_MUL_ROW";
  task_type_to_name[TASK_MOE_TOP1_ROW] = "TASK_MOE_TOP1_ROW";
  task_type_to_name[TASK_MOE_SLOT_W13] = "TASK_MOE_SLOT_W13";
  task_type_to_name[TASK_MOE_SLOT_W2] = "TASK_MOE_SLOT_W2";
  task_type_to_name[TASK_IDENTITY] = "TASK_IDENTITY";
  task_type_to_name[TASK_SILU_MUL_LINEAR_WITH_RESIDUAL] =
      "TASK_SILU_MUL_LINEAR_WITH_RESIDUAL";
  task_type_to_name[TASK_LINEAR] = "TASK_LINEAR";
  task_type_to_name[TASK_LINEAR_WITH_RESIDUAL] = "TASK_LINEAR_WITH_RESIDUAL";
  task_type_to_name[TASK_ARGMAX_PARTIAL] = "TASK_ARGMAX_PARTIAL";
  task_type_to_name[TASK_ARGMAX_REDUCE] = "TASK_ARGMAX_REDUCE";
  task_type_to_name[TASK_REDUCE] = "TASK_REDUCE";
  task_type_to_name[TASK_FIND_NGRAM_PARTIAL] = "TASK_FIND_NGRAM_PARTIAL";
  task_type_to_name[TASK_FIND_NGRAM_GLOBAL] = "TASK_FIND_NGRAM_GLOBAL";
  task_type_to_name[TASK_TARGET_VERIFY_GREEDY] = "TASK_TARGET_VERIFY_GREEDY";
  task_type_to_name[TASK_SINGLE_BATCH_EXTEND_ATTENTION] =
      "TASK_SINGLE_BATCH_EXTEND_ATTENTION";
  task_type_to_name[TASK_PAGED_ATTENTION_1] = "TASK_PAGED_ATTENTION_1";
  task_type_to_name[TASK_LINEAR_HOPPER] = "TASK_LINEAR_HOPPER";
  task_type_to_name[TASK_LINEAR_WITH_RESIDUAL_HOPPER] =
      "TASK_LINEAR_WITH_RESIDUAL_HOPPER";
  task_type_to_name[TASK_PAGED_ATTENTION_HOPPER] =
      "TASK_PAGED_ATTENTION_HOPPER";
  task_type_to_name[TASK_RMS_NORM_HOPPER] = "TASK_RMS_NORM_HOPPER";
  task_type_to_name[TASK_LINEAR_SWAPAB_HOPPER] = "TASK_LINEAR_SWAPAB_HOPPER";
  task_type_to_name[TASK_LINEAR_QKV_HOPPER] = "TASK_LINEAR_QKV_HOPPER";
  task_type_to_name[TASK_LINEAR_WO_HOPPER] = "TASK_LINEAR_WO_HOPPER";
  task_type_to_name[TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER] =
      "TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER";
  task_type_to_name[TASK_LINEAR_CUTLASS_HOPPER] = "TASK_LINEAR_CUTLASS_HOPPER";
  task_type_to_name[TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER] =
      "TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER";
  task_type_to_name[TASK_SILU_MUL_HOPPER] = "TASK_SILU_MUL_HOPPER";
  task_type_to_name[TASK_EMBEDDING_HOPPER] = "TASK_EMBEDDING_HOPPER";
  task_type_to_name[TASK_LINEAR_SM100] = "TASK_LINEAR_SM100";
  task_type_to_name[TASK_LINEAR_WITH_RESIDUAL_SM100] =
      "TASK_LINEAR_WITH_RESIDUAL_SM100";
  task_type_to_name[TASK_SPLITK_LINEAR_SM100] = "TASK_SPLITK_LINEAR_SM100";
  task_type_to_name[TASK_ATTN_SM100] = "TASK_ATTN_SM100";
  task_type_to_name[TASK_ARGMAX_PARTIAL_SM100] = "TASK_ARGMAX_PARTIAL_SM100";
  task_type_to_name[TASK_ARGMAX_REDUCE_SM100] = "TASK_ARGMAX_REDUCE_SM100";
  task_type_to_name[TASK_SAMPLING_SM100] = "TASK_SAMPLING_SM100";
  task_type_to_name[TASK_TENSOR_INIT] = "TASK_TENSOR_INIT";
  task_type_to_name[TASK_MOE_TOPK_SOFTMAX_SM100] =
      "TASK_MOE_TOPK_SOFTMAX_SM100";
  task_type_to_name[TASK_MOE_W13_LINEAR_SM100] = "TASK_MOE_W13_LINEAR_SM100";
  task_type_to_name[TASK_MOE_W2_LINEAR_SM100] = "TASK_MOE_W2_LINEAR_SM100";
  task_type_to_name[TASK_MOE_MUL_SUM_ADD_SM100] = "TASK_MOE_MUL_SUM_ADD_SM100";
  task_type_to_name[TASK_MOE_W13_LINEAR_SM90] = "TASK_MOE_W13_LINEAR_SM90";
  task_type_to_name[TASK_MOE_W2_LINEAR_SM90] = "TASK_MOE_W2_LINEAR_SM90";
  task_type_to_name[TASK_SPLITK_LINEAR_SWAPAB_HOPPER] =
      "TASK_SPLITK_LINEAR_SWAPAB_HOPPER";
  task_type_to_name[TASK_PAGED_ATTENTION_SPLIT_KV_SM100] =
      "TASK_PAGED_ATTENTION_SPLIT_KV_SM100";
  task_type_to_name[TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100] =
      "TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100";
  task_type_to_name[TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER] =
      "TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER";
  // Multi-gpu tasks
  task_type_to_name[TASK_NVSHMEM_ALLGATHER_STRIDED_PUT] =
      "TASK_NVSHMEM_ALLGATHER_STRIDED_PUT";
  task_type_to_name[TASK_NVSHMEM_TILE_ALLREDUCE] =
      "TASK_NVSHMEM_TILE_ALLREDUCE";

  code.e("__device__ __forceinline__");
  code.e("void _execute_task(TaskDesc const* task_desc,");
  code.e("                   ResidentRuntimeConfig const &runtime_config) {");
  TaskRegister *task_register = TaskRegister::get_instance();
  std::set<std::pair<TaskType, int>> used_variants;
  for (FullTaskDesc const &task : all_tasks) {
    if (task.task_type != TASK_TERMINATE && task.task_type != TASK_BEGIN_TASK_GRAPH) {
      used_variants.emplace(task.task_type, task.variant_id);
    }
  }
  bool first_task = true;
  for (auto const &task : task_register->all_task_variants) {
    for (size_t variant_id = 0; variant_id < task.second.size(); variant_id++) {
      if (used_variants.count({task.first, static_cast<int>(variant_id)}) == 0) {
        continue;
      }
      std::string cond = first_task ? "if" : "else if";
      assert(task_type_to_name.find(task.first) != task_type_to_name.end());
      code.e("$ (task_desc->task_type == $ && task_desc->variant_id == $) {",
             cond,
             task_type_to_name[task.first],
             variant_id);
      code.e("$", task.second[variant_id]);
      code.e("}");
      first_task = false;
    }
  }
  code.e("}");
  code.e("__device__ __forceinline__");
  code.e("void _execute_task(ResidentTaskDesc const* resident_task_desc,");
  code.e("                   DataDesc const* data_desc,");
  code.e("                   ResidentRuntimeConfig const &runtime_config) {");
  code.e("__shared__ ResolvedTaskDesc resolved_task_desc;");
  code.e("if (threadIdx.x == 0) {");
  code.e("resolved_task_desc = resolve_task_desc(resident_task_desc, "
         "data_desc);");
  code.e("}");
  code.e("__syncthreads();");
  code.e("_execute_task(reinterpret_cast<TaskDesc const*>(&resolved_task_desc), "
         "runtime_config);");
  code.e("}");

  add_request_slot_metadata(json_task_graph, data_slot_ids, io_configs);

  // Write json to output file
  // std::ofstream out("task_graph.json");
  // out << json_task_graph.dump(2);
  // out.close();
  TaskGraphResult result;
  result.cuda_code = code.to_string();
  result.json_file = json_task_graph.dump(2);
  return result;
}

} // namespace

TaskGraphResult Graph::generate_resident_task_graph(
    int _num_gpus, int _my_gpu_id, std::vector<int> const &data_slot_ids) {
  if (_num_gpus <= 0 || _my_gpu_id < 0 || _my_gpu_id >= _num_gpus) {
    throw std::runtime_error("Invalid resident GPU count or rank");
  }
  validate_resident_bindings(*this);
  if (!data_slot_ids.empty() && _num_gpus != 1) {
    throw std::runtime_error("Request slot metadata currently supports single-GPU resident_data only.");
  }
  std::vector<FullTaskDesc> all_tasks;
  std::vector<EventDesc> all_events;
  std::vector<TaskId> first_tasks;
  int num_gpus, my_gpu_id;
  OperatorTaskMaps all_task_maps;
  num_gpus = _num_gpus;
  my_gpu_id = _my_gpu_id;
  // add the termination event to the event lists
  EventDesc e(EVENT_TERMINATION, 1, 0, 0);
  all_events.push_back(e);
  FullTaskDesc t(TASK_TERMINATE, 0 /*variant_id*/);
  all_tasks.push_back(t);
  kn::register_mugraph(*this,
                      num_gpus,
                      my_gpu_id,
                      all_tasks,
                      all_events,
                      first_tasks,
                      all_task_maps,
                      task_config);
  assign_profiler_group_ids(all_tasks);
  assert(sanity_check(*this, all_tasks, all_events, first_tasks));
  return print_task_graph(*this,
                          num_gpus,
                          my_gpu_id,
                          all_tasks,
                          all_events,
                          first_tasks,
                          all_task_maps,
                          task_config,
                          io_config,
                          DataAwareTaskGraphFlavor::RESIDENT,
                          true /*use_json_format*/,
                          data_slot_ids);
}


} // namespace kernel
} // namespace mirage
