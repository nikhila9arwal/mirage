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

#include "mirage/persistent_kernel/runtime_header.h"
#include <map>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace mirage {
namespace kernel {

class Graph;
class KNOperator;

struct Dim3Comparator {
  bool operator()(dim3 const &a, dim3 const &b) const {
    if (a.x != b.x) {
      return a.x < b.x;
    }
    if (a.y != b.y) {
      return a.y < b.y;
    }
    return a.z < b.z;
  }
};

using TaskMap = std::map<dim3, std::vector<runtime::TaskId>, Dim3Comparator>;
using OperatorTaskMaps = std::map<KNOperator *, TaskMap>;

// Shared native/resident lowering; TaskIds are global across operators.
void register_mugraph(
    Graph const &graph,
    int num_gpus,
    int my_gpu_id,
    std::vector<runtime::FullTaskDesc> &all_tasks,
    std::vector<runtime::EventDesc> &all_events,
    std::vector<runtime::TaskId> &first_tasks,
    OperatorTaskMaps &all_task_maps,
    std::unordered_map<KNOperator const *,
                       std::tuple<int, int, runtime::TaskType, int>> const
        &task_configs);

} // namespace kernel
} // namespace mirage
