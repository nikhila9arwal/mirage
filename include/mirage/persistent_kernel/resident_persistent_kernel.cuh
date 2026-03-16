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

#include "profiler.h"
#include "tasks/common/copy_sm80.cuh"
#ifdef MPK_ENABLE_TMA
#include "resident_tma.cuh"
#endif
#include "mpk_atoms.cuh"
#include "resident_runtime_header.h"
#ifdef USE_NVSHMEM
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#endif
#include <thread>
#include <unistd.h>
#include <vector>

#if defined(MIRAGE_GRACE_HOPPER)
#include "tasks/hopper/task_header.cuh"
#elif defined(MIRAGE_GRACE_BLACKWELL)
#include "tasks/blackwell/task_header.cuh"
#else
#include "tasks/ampere/task_header.cuh"
#endif

using bfloat16 = type::bfloat16_t;
using namespace mirage::runtime;
using namespace kernel;
// Configurations for the MPK runtime
// #define MPK_MAX_NUM_BATCHED_REQUESTS 16
// #define MPK_MAX_NUM_BATCHED_TOKENS 64
// #define MPK_MAX_NUM_PAGES 1024
// #define MPK_PAGE_SIZE 64

#if defined(MIRAGE_GRACE_HOPPER)
#define WORKER_NUM_THREADS 256
#define SINGLE_KERNEL_NUM_THREADS 256
#elif defined(MIRAGE_GRACE_BLACKWELL)
#define WORKER_NUM_THREADS 256
#define SINGLE_KERNEL_NUM_THREADS 256
#else
#define WORKER_NUM_THREADS 128
#define SINGLE_KERNEL_NUM_THREADS 128
#endif
#define INIT_NUM_THREADS 128

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr,                                                          \
              "CUDA error at %s:%d: %s\n",                                     \
              __FILE__,                                                        \
              __LINE__,                                                        \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#endif

#ifdef USE_NVSHMEM
#ifndef NVSHMEM_CHECK
#define NVSHMEM_CHECK(stmt)                                                    \
  do {                                                                         \
    int result = (stmt);                                                       \
    if (NVSHMEMX_SUCCESS != result) {                                          \
      fprintf(stderr,                                                          \
              "[%s:%d] NVSHMEM failed with error %d\n",                        \
              __FILE__,                                                        \
              __LINE__,                                                        \
              result);                                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif
#endif

// #define MPK_ENABLE_VERBOSE
#ifndef MIRAGE_RESIDENT_EXECUTION_MODE_VALUE
#define MIRAGE_RESIDENT_EXECUTION_MODE_VALUE RESIDENT_EXECUTION_SCHEDULER_DISPATCH
#endif
#ifndef MIRAGE_STREAMING_BASE_EXECUTION_MODE_VALUE
#define MIRAGE_STREAMING_BASE_EXECUTION_MODE_VALUE \
  STREAMING_BASE_EXECUTION_LEGACY_EVENT
#endif

#ifdef MIRAGE_STREAMING_BUILD
#define MIRAGE_COMPILE_RESIDENT_SCHEDULER_DISPATCH 0
#define MIRAGE_COMPILE_RESIDENT_HYBRID_PRELAUNCH 0
#define MIRAGE_COMPILE_STREAMING_RUNTIME 1
#else
#define MIRAGE_COMPILE_RESIDENT_SCHEDULER_DISPATCH 1
#define MIRAGE_COMPILE_RESIDENT_HYBRID_PRELAUNCH 1
#define MIRAGE_COMPILE_STREAMING_RUNTIME 0
#endif

#define MIRAGE_STREAMING_BURST_QUOTA 8
#define MIRAGE_STREAMING_MAX_READY_SUCCESSORS 1024

__device__ __forceinline__ void
    _execute_task(TaskDesc const *task_desc,
                  ResidentRuntimeConfig const &runtime_config);
__device__ __forceinline__ void
    _execute_task(ResidentTaskDesc const *resident_task_desc,
                  DataDesc const *data_desc,
                  ResidentRuntimeConfig const &runtime_config);

__device__ __forceinline__ bool is_termination_event(size_t event_loc,
                                                     EventDesc e) {
  return (event_loc == 0);
}

__host__ __device__ __forceinline__ bool is_nvshmem_event(EventId event_id) {
  return (event_id & EVENT_NVSHMEM_TAG) > 0;
}

__device__ __forceinline__ size_t get_event_gpu_id(EventId event_id) {
  return ((event_id >> 32) & 0xffff);
}

__device__ __forceinline__ size_t get_event_position_index(EventId event_id) {
  return (event_id & 0xffffffff);
}

__device__ __forceinline__ size_t get_task_iteration_num(TaskId task_id) {
  return (task_id >> 32);
}

__device__ __forceinline__ size_t get_task_position_index(TaskId task_id) {
  return (task_id & 0xffffffff);
}

__device__ __forceinline__ TaskId compute_task_id(size_t iteration_num,
                                                  size_t position_index) {
  return ((iteration_num << 32) | position_index);
}

constexpr TaskId BEGIN_TASK_GRAPH_TASK_ID = 1;

__host__ __device__ __forceinline__ ResidentExecutionMode
get_resident_execution_mode(ResidentRuntimeConfig const &config) {
  return static_cast<ResidentExecutionMode>(config.resident_execution_mode);
}

__host__ __device__ __forceinline__ bool
resident_uses_hybrid_prelaunch(ResidentRuntimeConfig const &config) {
  return get_resident_execution_mode(config) ==
         RESIDENT_EXECUTION_HYBRID_PRELAUNCH;
}

__host__ __device__ __forceinline__ bool
resident_uses_streaming(ResidentRuntimeConfig const &config) {
  return get_resident_execution_mode(config) == RESIDENT_EXECUTION_STREAMING;
}

__host__ __device__ __forceinline__ StreamingBaseExecutionMode
get_streaming_base_execution_mode(ResidentRuntimeConfig const &config) {
  return static_cast<StreamingBaseExecutionMode>(
      config.streaming_base_execution_mode);
}

__host__ __device__ __forceinline__ bool
streaming_uses_hybrid_base(ResidentRuntimeConfig const &config) {
  return resident_uses_streaming(config) &&
         get_streaming_base_execution_mode(config) ==
             STREAMING_BASE_EXECUTION_HYBRID_PRELAUNCH;
}

__host__ __device__ __forceinline__ bool
streaming_uses_legacy_base(ResidentRuntimeConfig const &config) {
  return resident_uses_streaming(config) &&
         get_streaming_base_execution_mode(config) ==
             STREAMING_BASE_EXECUTION_LEGACY_EVENT;
}

__host__ __device__ __forceinline__ bool
resident_task_is_streaming(ResidentTaskDesc const &resident_task_desc) {
  return resident_task_desc.execution_kind ==
         RESIDENT_TASK_EXECUTION_STREAMING;
}

__host__ __device__ __forceinline__ bool
resident_task_is_prelaunched(ResidentTaskDesc const &resident_task_desc) {
  return resident_task_desc.execution_kind ==
         RESIDENT_TASK_EXECUTION_PRELAUNCHED;
}

constexpr uint32_t STREAMING_RESIDENT_WORK_ITEM_TAG = 0x80000000u;

__device__ __forceinline__ bool
is_streaming_resident_work_item(TaskId task_id) {
  return (get_task_position_index(task_id) & STREAMING_RESIDENT_WORK_ITEM_TAG) !=
         0;
}

__device__ __forceinline__ ResidentTaskId
get_streaming_resident_task_id(TaskId task_id) {
  return static_cast<ResidentTaskId>(
      get_task_position_index(task_id) & ~STREAMING_RESIDENT_WORK_ITEM_TAG);
}

__device__ __forceinline__ TaskId
compute_streaming_resident_work_item(size_t iteration_num,
                                     ResidentTaskId resident_task_id) {
  return compute_task_id(
      iteration_num,
      STREAMING_RESIDENT_WORK_ITEM_TAG |
          static_cast<uint32_t>(resident_task_id));
}

__global__ void init_kernel(ResidentRuntimeConfig config) {
  assert(gridDim.x == 1);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Only a single thread that initializes everything
  if (threadIdx.x == 0) {
    // initialize metadata
#if defined(MODE_OFFLINE) || defined(MODE_ONLINE)
    for (int i = 0; i < config.total_num_requests; i++) {
      config.step[i] = 0;
    }
    *config.next_request_id = 0;
    for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
      config.request_ids[i] = -1;
    }
    for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS + 1; i++) {
      config.qo_indptr_buffer[i] = 0;
      config.paged_kv_indptr_buffer[i] = 0;
    }
    // Page manager
    *config.page_queue_head = 0;
    *config.page_queue_tail = MPK_MAX_NUM_PAGES;
    for (int i = 0; i < MPK_MAX_NUM_PAGES; i++) {
      config.page_queue[i] = i;
    }
#endif
  }
}

__global__ void prepare_kernel(ResidentRuntimeConfig config,
                               int end_of_task_graph_event_pos) {
  // Initialize worker queue last task id
  // Each worker now maintains a local and a remote worker queue
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < 2 * config.num_workers;
       i += blockDim.x * gridDim.x) {
    config.worker_queue_next_free_task_id[i] = 0;
    config.worker_queue_last_ready_task_id[i] = 0;
  }
  // Initialize scheduler queue last event id
  // We maintain one extra scheduler queue for the global scheduler
  int num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_schedulers + 1;
       i += blockDim.x * gridDim.x) {
    config.sched_queue_last_ready_event_id[i] = 0;
    config.sched_queue_next_free_event_id[i] = 0;
  }
  if (!resident_uses_streaming(config)) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < max(1, config.num_local_schedulers);
         i += blockDim.x * gridDim.x) {
      config.completion_queue_last_ready_data_id[i] = 0;
      config.completion_queue_next_free_data_id[i] = 0;
    }
  }
  // Initialize all event counters
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < config.num_events;
       i += blockDim.x * gridDim.x) {
    config.all_event_counters[i] = 0;
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < config.num_data;
       i += blockDim.x * gridDim.x) {
    config.data_pending_predecessor_counts[i] =
        config.data_initial_predecessor_counts[i];
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *config.completed_terminal_data_count = 0;
    *config.completed_data_this_iteration_count = 0;
    *config.current_iteration = 0;
    if (resident_uses_streaming(config) &&
        config.completed_streaming_data_count != nullptr) {
      *config.completed_streaming_data_count = 0;
    }
  }
  // Send event to scheduler[0]
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    assert(config.all_events[end_of_task_graph_event_pos].event_type ==
           EVENT_END_OF_TASK_GRAPH);
    config.sched_queue_next_free_event_id[0] = 1;
    config.sched_queues[0][0] = end_of_task_graph_event_pos;
    config.sched_queue_last_ready_event_id[0] = 1;
  }
}

#ifdef MODE_OFFLINE
// TODO: parallelize this processing
__device__ __forceinline__ bool
    prepare_next_batch(ResidentRuntimeConfig const &config) {
  __shared__ int smem_kv_indices[MPK_MAX_NUM_PAGES];
  int page_queue_head = *config.page_queue_head;
  int page_queue_tail = *config.page_queue_tail;
  // Step 1: finalize previous batch
  for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    int16_t request_id = config.request_ids[i];
    if (request_id != -1) {
      // Step 1.1: move output_tokens to tokens
      int step = config.step[request_id];
      int qo_indptr = config.qo_indptr_buffer[i];
      int num_tokens = config.qo_indptr_buffer[i + 1] - qo_indptr;
      int prompt_len = config.prompt_length[request_id];
      for (int j = 0; j < num_tokens; j++) {
        if (step + j + 1 >= prompt_len &&
            step + j + 1 < config.max_seq_length) {
          config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step + j + 1] =
              config.output_tokens[qo_indptr + j];
        }
      }
      config.step[request_id] = step + num_tokens;
      if ((step + num_tokens + 1 >= config.max_seq_length) ||
          ((config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step +
                          num_tokens] == config.eos_token_id) &&
           (step + num_tokens >= prompt_len)))
      {
        // Request is done
        config.request_ids[i] = -1;
        // Free pages
        int kv_indptr = config.paged_kv_indptr_buffer[i];
        int num_pages = config.paged_kv_indptr_buffer[i + 1] - kv_indptr;
        for (int j = 0; j < num_pages; j++) {
          config.page_queue[page_queue_tail % MPK_MAX_NUM_PAGES] =
              config.paged_kv_indices_buffer[kv_indptr + j];
          page_queue_tail++;
        }
      }
    }
  }

  // Step 2: copy kv_indices to shared mem
  int num_pages = config.paged_kv_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];
  for (int i = 0; i < num_pages; i++) {
    smem_kv_indices[i] = config.paged_kv_indices_buffer[i];
  }

  // Step 3: prepare next batch
  int num_reqs = 0, num_tokens = 0;
  num_pages = 0;
  for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    int16_t request_id = config.request_ids[i];
    if (request_id != -1) {
      int kv_indptr = config.paged_kv_indptr_buffer[i];
      int num_old_pages = config.paged_kv_indptr_buffer[i + 1] - kv_indptr;
      config.request_ids[num_reqs] = request_id;
      config.qo_indptr_buffer[num_reqs] = num_tokens;
      config.paged_kv_indptr_buffer[num_reqs] = num_pages;
      int step = config.step[request_id];
      int num_new_tokens = config.prompt_length[request_id] - step;
      if (num_new_tokens > 0) {
        // Prefill requests
        num_new_tokens =
            min(num_new_tokens, MPK_MAX_NUM_BATCHED_TOKENS - num_tokens);
      } else {
        // Decode requests
        num_new_tokens = min(1, MPK_MAX_NUM_BATCHED_TOKENS - num_tokens);
      }
      // Move tokens to input_tokens
      for (int j = 0; j < num_new_tokens; j++) {
        config.input_tokens[num_tokens + j] =
            config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step + j];
      }
      // Prepare page indptrs
      int num_new_pages =
          (step + num_new_tokens + MPK_PAGE_SIZE - 1) / MPK_PAGE_SIZE;
      config.paged_kv_last_page_len_buffer[num_reqs] =
          (step + num_new_tokens) % MPK_PAGE_SIZE;
      for (int j = 0; j < num_old_pages; j++) {
        config.paged_kv_indices_buffer[num_pages + j] =
            smem_kv_indices[kv_indptr + j];
      }
      for (int j = num_old_pages; j < num_new_pages; j++) {
        config.paged_kv_indices_buffer[num_pages + j] =
            config.page_queue[page_queue_head % MPK_MAX_NUM_PAGES];
        page_queue_head++;
      }
      num_pages += num_new_pages;
      num_tokens += num_new_tokens;
      num_reqs++;
    }
  }

  // Add new prefill requests until we reach capacity
  while (num_reqs < MPK_MAX_NUM_BATCHED_REQUESTS &&
         num_tokens < MPK_MAX_NUM_BATCHED_TOKENS) {
    int next_request_id = *config.next_request_id;
    if (next_request_id >= config.total_num_requests) {
      break;
    }
    config.request_ids[num_reqs] = next_request_id;
    config.qo_indptr_buffer[num_reqs] = num_tokens;
    config.paged_kv_indptr_buffer[num_reqs] = num_pages;
    // Prefill request
    int num_new_tokens = min(config.prompt_length[next_request_id],
                             MPK_MAX_NUM_BATCHED_TOKENS - num_tokens);
    // Move tokens to input tokens
    for (int j = 0; j < num_new_tokens; j++) {
      config.input_tokens[num_tokens + j] =
          config.tokens[next_request_id * MPK_MAX_SEQ_LENGTH + j];
    }
    int num_new_pages = (num_new_tokens + MPK_PAGE_SIZE - 1) / MPK_PAGE_SIZE;
    config.paged_kv_last_page_len_buffer[num_reqs] =
        num_new_tokens % MPK_PAGE_SIZE;
    for (int j = 0; j < num_new_pages; j++) {
      config.paged_kv_indices_buffer[num_pages + j] =
          config.page_queue[page_queue_head % MPK_MAX_NUM_PAGES];
      page_queue_head++;
    }
    num_tokens += num_new_tokens;
    num_pages += num_new_pages;
    num_reqs++;
    *config.next_request_id = next_request_id + 1;
  }

  // Step 4: Update all unused requests slots
  for (int i = num_reqs; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    config.request_ids[i] = -1;
  }
  for (int i = num_reqs; i <= MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    config.qo_indptr_buffer[i] = num_tokens;
    config.paged_kv_indptr_buffer[i] = num_pages;
  }

  // Step 5: update page head tail
  *config.page_queue_head = page_queue_head;
  *config.page_queue_tail = page_queue_tail;

  // printf("Next batch: steps[%d %d %d %d] num_active_tokens(%d)\n",
  //        config.step[0],
  //        config.step[1],
  //        config.step[2],
  //        config.step[3],
  //        config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);

  if (num_tokens == 0) {
    return false;
  } else {
    return true;
  }
}
#endif

#ifdef MODE_ONLINE
__device__ __forceinline__ bool
    prepare_next_batch(ResidentRuntimeConfig const &config) {
  int step = config.step[0];
#ifdef MPK_ENABLE_VERBOSE
  printf("step: %d, new_token_num(%p): %d, new_token_ids:\n",
         step,
         config.new_token_nums,
         config.new_token_nums[0]);
  for (int i = 0; i < config.new_token_nums[0]; i++) {
    printf("%lld ", config.tokens[step + 1 + i]);
  }
  printf("\n");
#endif
  config.step[0] = step + config.new_token_nums[0];

#ifdef MPK_ENABLE_PROFILING
  return false;
#else
  if ((step + 2 >= config.max_seq_length) ||
      (config.tokens[step + 1] == config.eos_token_id)) {
    return false;
  } else {
    return true;
  }
#endif
}
#endif

#ifdef MODE_ONLINE_NOTOKEN
__device__ __forceinline__ bool prepare_next_batch(ResidentRuntimeConfig const &config,
                                                   size_t iteration_num = 0) {
  // TODO: iteration_num is a current workaround
  // We may consider split EVENT_END_OF_TASK_GRAPH into
  // EVENT_END_OF_TASK_GRAPH and EVENT_START_OF_TASK_GRAPH
  if (iteration_num > 0) {
    return false;
  } else { // iteration_num == 0
    return true;
  }
}
#endif

__device__ __forceinline__ int get_rand_sched_id(size_t event_index,
                                                 int worker_id,
                                                 int num_workers,
                                                 int num_schedulers) {
  // const size_t seed = 0xac4c1b51;
  // size_t x = event_index * seed;
  // x ^= x >> 17;
  // x *= worker_id;
  //  x *= 0xed5ad4bb;
  // x ^= x >> 11;
  size_t x = worker_id;
  return x / ((num_workers + num_schedulers - 1) / num_schedulers);
}

__device__ __forceinline__ void
    get_first_last_ids(unsigned long long int num_elements,
                       unsigned long long int num_workers,
                       unsigned long long int my_id,
                       unsigned long long int *my_first_element,
                       unsigned long long int *my_last_element) {
  unsigned long long int num_elements_per_worker = num_elements / num_workers;
  unsigned long long int reminder = num_elements % num_workers;
  if (my_id < reminder) {
    *my_first_element = (num_elements_per_worker + 1) * my_id;
    *my_last_element = *my_first_element + num_elements_per_worker + 1;
  } else {
    *my_first_element = num_elements_per_worker * my_id + reminder;
    *my_last_element = *my_first_element + num_elements_per_worker;
  }
}

__device__ __forceinline__ void terminate_schedulers(ResidentRuntimeConfig config) {
  // Event ID 0 is the termination event
  int num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  for (int i = 0; i < num_schedulers; i++) {
    // size_t last_event_id =
    //     atomicAdd(&config.sched_queue_next_free_event_id[i], 1);
    size_t last_event_id =
        atom_add_release_gpu_u64(&config.sched_queue_next_free_event_id[i], 1);
    st_relaxed_gpu_u64(
        &config.sched_queues[i][last_event_id % config.per_sched_queue_len], 0);
    // Use st.relaxed to make sure sched_queue updates are visible to scheduler
    // CTAs before incrementing its last_ready_event_id
    size_t old;
    do {
      // old = atomicCAS(&config.sched_queue_last_ready_event_id[i],
      //                 last_event_id,
      //                 last_event_id + 1);
      old = atom_cas_release_gpu_u64(&config.sched_queue_last_ready_event_id[i],
                                     last_event_id,
                                     last_event_id + 1);
    } while (old != last_event_id);
  }
}

__device__ __forceinline__ bool
event_ready_nonblocking(ResidentRuntimeConfig const &config,
                        EventId event_id,
                        TaskId current_task_id) {
  if (event_id == EVENT_INVALID_ID) {
    return true;
  }
  size_t event_index = get_event_position_index(event_id);
  EventCounter needed_counts =
      static_cast<EventCounter>(config.all_event_num_triggers[event_index]) *
      get_task_iteration_num(current_task_id);
  EventCounter actual_counts =
      ld_acquire_sys_u64(&config.all_event_counters[event_index]);
  return actual_counts >= needed_counts;
}

__device__ __forceinline__ bool
prelaunched_task_ready_nonblocking(ResidentRuntimeConfig const &config,
                                   TaskDesc const *task_desc,
                                   TaskId current_task_id,
                                   DataId current_data_id) {
  if (current_data_id == DATA_INVALID_ID) {
    return event_ready_nonblocking(
        config, task_desc->dependent_event, current_task_id);
  }
  DataDesc const &data_desc = config.all_data[current_data_id];
  ResidentTaskDesc const &resident_task_desc =
      config.resident_tasks[data_desc.resident_task_id];
  assert(resident_task_is_prelaunched(resident_task_desc));
  if (streaming_uses_legacy_base(config)) {
    return event_ready_nonblocking(
        config, data_desc.dependent_event, current_task_id);
  }
  if (data_desc.dependent_event != EVENT_INVALID_ID &&
      is_nvshmem_event(data_desc.dependent_event)) {
    return event_ready_nonblocking(
        config, data_desc.dependent_event, current_task_id);
  }
  return atomicAdd(&config.data_pending_predecessor_counts[current_data_id], 0u) ==
         0u;
}

__device__ __forceinline__ bool
compat_task_is_prelaunched(ResidentRuntimeConfig const &config,
                           TaskId task_position) {
  if (task_position < static_cast<TaskId>(config.num_control_tasks)) {
    return true;
  }
  DataId data_id = config.task_to_data_id[task_position];
  if (data_id == DATA_INVALID_ID) {
    return true;
  }
  ResidentTaskId resident_task_id = config.all_data[data_id].resident_task_id;
  return resident_task_is_prelaunched(config.resident_tasks[resident_task_id]);
}

__device__ __forceinline__ void
enqueue_prelaunched_task_range(ResidentRuntimeConfig const &config,
                               int my_first_worker,
                               int my_last_worker,
                               int *next_worker,
                               size_t *worker_queue_next_free_task_pos,
                               size_t iteration_num,
                               TaskId first_task_id,
                               TaskId last_task_id) {
  for (TaskId task_pos = first_task_id; task_pos < last_task_id; task_pos++) {
    if (!compat_task_is_prelaunched(config, task_pos)) {
      continue;
    }
    enqueue_worker_item(
        config,
        *next_worker,
        &worker_queue_next_free_task_pos[*next_worker - my_first_worker],
        compute_task_id(iteration_num, task_pos));
    *next_worker = (*next_worker == my_last_worker - 1) ? my_first_worker
                                                        : *next_worker + 1;
  }
}

__device__ __forceinline__ void
enqueue_prelaunched_dependent_tasks_partitioned(
    ResidentRuntimeConfig const &config,
    int my_first_worker,
    int my_last_worker,
    int *next_worker,
    size_t *worker_queue_next_free_task_pos,
    size_t iteration_num,
    EventDesc const &event_desc) {
  for (size_t chunk_idx = 0;
       chunk_idx <
       (event_desc.last_task_id - event_desc.first_task_id +
        config.num_workers - 1) /
           config.num_workers;
       chunk_idx++) {
    for (size_t worker = my_first_worker; worker < my_last_worker; worker++) {
      size_t position_index =
          event_desc.first_task_id + chunk_idx * config.num_workers + worker;
      if (position_index >= event_desc.last_task_id ||
          !compat_task_is_prelaunched(config, position_index)) {
        continue;
      }
      enqueue_worker_item(
          config,
          *next_worker,
          &worker_queue_next_free_task_pos[*next_worker - my_first_worker],
          compute_task_id(iteration_num, position_index));
      *next_worker = (*next_worker == my_last_worker - 1) ? my_first_worker
                                                          : *next_worker + 1;
    }
  }
}

__device__ __forceinline__ void worker_checker(ResidentRuntimeConfig config) {
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  // int num_schedulers =
  //    config.num_local_schedulers + config.num_remote_schedulers;

  assert(gridDim.x == config.num_workers);
  assert(config.num_workers <= MAX_NUM_WORKERS);
  // We will reinterpret TaskDesc as an array of integers to
  // collectively load it from device to shared memory
  static_assert(sizeof(TaskDesc) % sizeof(int) == 0);
}

__device__ __forceinline__ void scheduler_checker(ResidentRuntimeConfig config) {
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  // int num_schedulers =
  //    config.num_local_schedulers + config.num_remote_schedulers;

  assert(config.num_workers <= MAX_NUM_WORKERS);
}

__device__ __forceinline__ void persistent_checker(ResidentRuntimeConfig config) {
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  int const num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  int const num_schedulers_per_sm = std::min((int)blockDim.x / 32, 4);
  assert(num_schedulers % num_schedulers_per_sm == 0);
  assert(gridDim.x ==
         config.num_workers + num_schedulers / num_schedulers_per_sm);
  assert(config.num_workers <= MAX_NUM_WORKERS);
  // We will reinterpret TaskDesc as an array of integers to
  // collectively load it from device to shared memory
  static_assert(sizeof(TaskDesc) % sizeof(int) == 0);
  // assert(blockDim.x >= 128);
}

__device__ __forceinline__ void
publish_worker_item(ResidentRuntimeConfig const &config,
                    int worker_queue_id,
                    TaskId work_item) {
  unsigned long long int next_pos =
      atomicAdd(&config.worker_queue_next_free_task_id[worker_queue_id], 1ull);
  st_relaxed_gpu_u64(
      &config.worker_queues[worker_queue_id][next_pos %
                                             config.per_worker_queue_len],
      work_item);
  __threadfence();
  while (atomicCAS(&config.worker_queue_last_ready_task_id[worker_queue_id],
                   next_pos,
                   next_pos + 1) != next_pos) {
  }
}

__device__ __forceinline__ void
publish_scheduler_event(ResidentRuntimeConfig const &config,
                        int sched_queue_id,
                        EventId event_id) {
  unsigned long long int next_pos =
      atomicAdd(&config.sched_queue_next_free_event_id[sched_queue_id], 1ull);
  st_relaxed_gpu_u64(
      &config.sched_queues[sched_queue_id][next_pos %
                                           config.per_sched_queue_len],
      event_id);
  __threadfence();
  while (atomicCAS(&config.sched_queue_last_ready_event_id[sched_queue_id],
                   next_pos,
                   next_pos + 1) != next_pos) {
  }
}

__device__ __forceinline__ void
publish_completion_item(ResidentRuntimeConfig const &config,
                        int completion_queue_id,
                        DataId data_id) {
  unsigned long long int next_pos =
      atomicAdd(&config.completion_queue_next_free_data_id[completion_queue_id],
                1ull);
  st_relaxed_gpu_u64(
      &config.completion_queues[completion_queue_id]
                               [next_pos % config.per_completion_queue_len],
      static_cast<TaskId>(data_id));
  __threadfence();
  while (atomicCAS(&config.completion_queue_last_ready_data_id
                                [completion_queue_id],
                   next_pos,
                   next_pos + 1) != next_pos) {
  }
}

__device__ __forceinline__ void
enqueue_worker_item(ResidentRuntimeConfig const &config,
                    int worker_queue_id,
                    size_t *next_free_task_pos,
                    TaskId work_item) {
  size_t last_task_id = (*next_free_task_pos)++;
  st_relaxed_gpu_u64(
      &config.worker_queues[worker_queue_id][last_task_id %
                                             config.per_worker_queue_len],
      work_item);
  atom_add_release_gpu_u64(&config.worker_queue_last_ready_task_id[worker_queue_id],
                           1);
}

__device__ __forceinline__ uint32_t
streaming_resident_queue_capacity(ResidentRuntimeConfig const &config,
                                  ResidentTaskId resident_task_id) {
  return config.resident_ready_queue_offsets[resident_task_id + 1] -
         config.resident_ready_queue_offsets[resident_task_id];
}

__device__ __forceinline__ uint32_t
streaming_worker_resident_begin(ResidentRuntimeConfig const &config,
                                int worker_id) {
  return config.worker_streaming_resident_offsets[worker_id];
}

__device__ __forceinline__ uint32_t
streaming_worker_resident_end(ResidentRuntimeConfig const &config,
                              int worker_id) {
  return config.worker_streaming_resident_offsets[worker_id + 1];
}

__device__ __forceinline__ void
publish_streaming_resident_ready_data(ResidentRuntimeConfig const &config,
                                      ResidentTaskId resident_task_id,
                                      DataId data_id) {
  uint32_t queue_capacity =
      streaming_resident_queue_capacity(config, resident_task_id);
  assert(queue_capacity > 0);
  uint32_t queue_offset = config.resident_ready_queue_offsets[resident_task_id];
  uint32_t next_pos =
      atomicAdd(&config.resident_ready_next_free_positions[resident_task_id], 1u);
  uint32_t head_pos =
      atomicAdd(&config.resident_ready_head_positions[resident_task_id], 0u);
  assert(next_pos < head_pos + queue_capacity);
  config.resident_ready_queue_storage[queue_offset + (next_pos % queue_capacity)] =
      data_id;
  __threadfence();
  while (atomicCAS(&config.resident_ready_tail_positions[resident_task_id],
                   next_pos,
                   next_pos + 1) != next_pos) {
  }
}

__device__ __forceinline__ bool
pop_streaming_resident_ready_data(ResidentRuntimeConfig const &config,
                                  ResidentTaskId resident_task_id,
                                  DataId *data_id) {
  uint32_t head_pos = config.resident_ready_head_positions[resident_task_id];
  uint32_t last_ready =
      atomicAdd(&config.resident_ready_tail_positions[resident_task_id], 0u);
  if (head_pos >= last_ready) {
    return false;
  }
  uint32_t queue_capacity =
      streaming_resident_queue_capacity(config, resident_task_id);
  uint32_t queue_offset = config.resident_ready_queue_offsets[resident_task_id];
  *data_id = config.resident_ready_queue_storage[queue_offset +
                                                 (head_pos % queue_capacity)];
  config.resident_ready_head_positions[resident_task_id] = head_pos + 1;
  return true;
}

__device__ __forceinline__ void
trigger_task_event(ResidentRuntimeConfig const &config,
                   TaskDesc const *task_desc,
                   TaskId current_task_id,
                   int worker_id) {
  EventId const event_id = task_desc->trigger_event;
  TaskType const task_type = task_desc->task_type;
  if (event_id == EVENT_INVALID_ID) {
    return;
  }
  size_t event_index = get_event_position_index(event_id);
  if (!is_nvshmem_event(event_id)) {
    size_t gpu_id = get_event_gpu_id(event_id);
    assert(gpu_id == config.my_gpu_id);
    EventCounter count =
        atom_add_release_gpu_u64(&config.all_event_counters[event_index], 1);
    int num_triggers = config.all_event_num_triggers[event_index];
    if ((count + 1) ==
        static_cast<EventCounter>(num_triggers) *
            get_task_iteration_num(current_task_id)) {
      EventDesc event_desc = config.all_events[event_index];
      if (event_desc.event_type != EVENT_EMPTY) {
        bool use_bcast_queue = false;
        if (event_desc.event_type == EVENT_LAUNCH_MASSIVE_TASKS ||
            event_desc.event_type == EVENT_LAUNCH_DEPENDENT_TASKS) {
          use_bcast_queue = true;
        }
        int sched_id =
            use_bcast_queue
                ? config.num_local_schedulers + config.num_remote_schedulers
                : get_rand_sched_id(event_index,
                                    worker_id,
                                    config.num_workers,
                                    config.num_local_schedulers);
        publish_scheduler_event(config, sched_id, event_index);
      }
    }
  } else {
    assert(task_type == TASK_NVSHMEM_ALLGATHER_STRIDED_PUT);
  }
}

__device__ __forceinline__ void
trigger_data_event(ResidentRuntimeConfig const &config,
                   DataDesc const *data_desc,
                   TaskType task_type,
                   TaskId current_task_id,
                   int worker_id) {
  EventId const event_id = data_desc->trigger_event;
  if (event_id == EVENT_INVALID_ID) {
    return;
  }
  size_t event_index = get_event_position_index(event_id);
  if (!is_nvshmem_event(event_id)) {
    size_t gpu_id = get_event_gpu_id(event_id);
    assert(gpu_id == config.my_gpu_id);
    EventCounter count =
        atom_add_release_gpu_u64(&config.all_event_counters[event_index], 1);
    int num_triggers = config.all_event_num_triggers[event_index];
    if ((count + 1) ==
        static_cast<EventCounter>(num_triggers) *
            get_task_iteration_num(current_task_id)) {
      EventDesc event_desc = config.all_events[event_index];
      if (event_desc.event_type != EVENT_EMPTY) {
        bool use_bcast_queue = false;
        if (event_desc.event_type == EVENT_LAUNCH_MASSIVE_TASKS ||
            event_desc.event_type == EVENT_LAUNCH_DEPENDENT_TASKS) {
          use_bcast_queue = true;
        }
        int sched_id =
            use_bcast_queue
                ? config.num_local_schedulers + config.num_remote_schedulers
                : get_rand_sched_id(event_index,
                                    worker_id,
                                    config.num_workers,
                                    config.num_local_schedulers);
        publish_scheduler_event(config, sched_id, event_index);
      }
    }
  } else {
    assert(task_type == TASK_NVSHMEM_ALLGATHER_STRIDED_PUT);
  }
}

__device__ __forceinline__ void
reset_iteration_state_scalar(ResidentRuntimeConfig const &config) {
  for (DataId data_id = 0; data_id < config.num_data; data_id++) {
    config.data_pending_predecessor_counts[data_id] =
        config.data_initial_predecessor_counts[data_id];
  }
  *config.completed_terminal_data_count = 0;
  *config.completed_data_this_iteration_count = 0;
  __threadfence();
}

__device__ __forceinline__ void
reset_iteration_state_parallel(ResidentRuntimeConfig const &config,
                               int lane_id,
                               int lane_count) {
  for (DataId data_id = lane_id; data_id < config.num_data;
       data_id += lane_count) {
    config.data_pending_predecessor_counts[data_id] =
        config.data_initial_predecessor_counts[data_id];
  }
  if (lane_id == 0) {
    *config.completed_terminal_data_count = 0;
    *config.completed_data_this_iteration_count = 0;
  }
  __syncwarp();
  __threadfence();
  __syncwarp();
}

__device__ __forceinline__ void
reset_streaming_iteration_state_parallel(ResidentRuntimeConfig const &config,
                                         int lane_id,
                                         int lane_count) {
  reset_iteration_state_parallel(config, lane_id, lane_count);
  for (ResidentTaskId resident_task_id = lane_id;
       resident_task_id < config.num_resident_tasks;
       resident_task_id += lane_count) {
    config.resident_ready_head_positions[resident_task_id] = 0u;
    config.resident_ready_next_free_positions[resident_task_id] = 0u;
    config.resident_ready_tail_positions[resident_task_id] = 0u;
  }
  if (lane_id == 0) {
    *config.completed_streaming_data_count = 0u;
  }
  __syncwarp();
  __threadfence();
  __syncwarp();
}

__device__ __forceinline__ void
release_completed_data(ResidentRuntimeConfig const &config,
                       DataId data_id,
                       EventId trigger_event) {
  __threadfence();
  __syncthreads();

  DataId edge_begin = config.data_edge_offsets[data_id];
  DataId edge_end = config.data_edge_offsets[data_id + 1];
  for (DataId edge_idx = edge_begin + threadIdx.x; edge_idx < edge_end;
       edge_idx += blockDim.x) {
    DataId succ_data_id = config.data_edge_targets[edge_idx];
    unsigned int remaining =
        atomicSub(&config.data_pending_predecessor_counts[succ_data_id], 1u);
    assert(remaining > 0);
  }

  __syncthreads();
  bool const has_remote_trigger =
      trigger_event != EVENT_INVALID_ID && is_nvshmem_event(trigger_event);
  if (threadIdx.x == 0 && edge_begin == edge_end && !has_remote_trigger) {
    unsigned int completed_terminal =
        atomicAdd(config.completed_terminal_data_count, 1u) + 1;
    if (completed_terminal ==
        static_cast<unsigned int>(config.num_terminal_data)) {
      publish_scheduler_event(config, 0, config.end_event_index);
    }
  }
}

__device__ __forceinline__ void
release_completed_data_streaming(ResidentRuntimeConfig const &config,
                                 DataId data_id,
                                 bool count_streaming_completion,
                                 bool count_terminal_completion,
                                 uint32_t *ready_streaming_edge_bits) {
  __threadfence();
  __syncthreads();

  DataId edge_begin = config.data_edge_offsets[data_id];
  DataId edge_end = config.data_edge_offsets[data_id + 1];
  DataId edge_count = edge_end - edge_begin;
  if (threadIdx.x == 0) {
    assert(edge_count <= MIRAGE_STREAMING_MAX_READY_SUCCESSORS);
  }
  for (int word_idx = threadIdx.x;
       word_idx < MIRAGE_STREAMING_MAX_READY_SUCCESSORS / 32;
       word_idx += blockDim.x) {
    ready_streaming_edge_bits[word_idx] = 0u;
  }
  __syncthreads();

  for (DataId edge_idx = edge_begin + threadIdx.x; edge_idx < edge_end;
       edge_idx += blockDim.x) {
    DataId succ_data_id = config.data_edge_targets[edge_idx];
    unsigned int remaining =
        atomicSub(&config.data_pending_predecessor_counts[succ_data_id], 1u);
    assert(remaining > 0);
    if (remaining == 1u) {
      ResidentTaskId succ_resident_task_id =
          config.all_data[succ_data_id].resident_task_id;
      if (resident_task_is_streaming(
              config.resident_tasks[succ_resident_task_id])) {
        DataId local_edge_idx = edge_idx - edge_begin;
        atomicOr(&ready_streaming_edge_bits[local_edge_idx / 32],
                 1u << (local_edge_idx % 32));
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    if (count_streaming_completion) {
      atomicAdd(config.completed_streaming_data_count, 1u);
    }
    for (DataId local_edge_idx = 0; local_edge_idx < edge_count;
         local_edge_idx++) {
      if ((ready_streaming_edge_bits[local_edge_idx / 32] &
           (1u << (local_edge_idx % 32))) == 0u) {
        continue;
      }
      DataId succ_data_id = config.data_edge_targets[edge_begin + local_edge_idx];
      ResidentTaskId succ_resident_task_id =
          config.all_data[succ_data_id].resident_task_id;
      publish_streaming_resident_ready_data(
          config, succ_resident_task_id, succ_data_id);
    }
    bool const has_remote_trigger =
        config.all_data[data_id].trigger_event != EVENT_INVALID_ID &&
        is_nvshmem_event(config.all_data[data_id].trigger_event);
    if (count_terminal_completion && edge_begin == edge_end &&
        !has_remote_trigger) {
      unsigned int completed_terminal =
          atomicAdd(config.completed_terminal_data_count, 1u) + 1u;
      if (completed_terminal ==
          static_cast<unsigned int>(config.num_terminal_data)) {
        publish_scheduler_event(config, 0, config.end_event_index);
      }
    }
  }
}

__device__ __forceinline__ void
execute_worker_scheduler_dispatch(ResidentRuntimeConfig config) {
  constexpr int TASK_DESCS_BUFFER_LENGTH = std::min(
      (mirage::runtime::WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE - 60) /
          (int)(sizeof(TaskDesc) + sizeof(TaskId)),
      16);
  __shared__ TaskDesc task_descs[TASK_DESCS_BUFFER_LENGTH];
  __shared__ TaskId task_ids[TASK_DESCS_BUFFER_LENGTH];
  __shared__ DataId current_data_id;
  __shared__ int current_queue_idx;
  __shared__ TaskId *worker_queues[2];
  __shared__ int worker_queue_ids[2];
  __shared__ size_t next_task_pos[2];
  __shared__ size_t last_task_pos[2];

#ifdef MPK_ENABLE_PROFILING
  PROFILER_CLOSURE_PARAMS_DECL;
  PROFILER_INIT(static_cast<uint64_t *>(config.profiler_buffer),
                0,
                1,
                (threadIdx.x % WORKER_NUM_THREADS == 0));

#endif
  int const worker_id = blockIdx.x;
  worker_queues[0] = config.worker_queues[worker_id];
  worker_queue_ids[0] = worker_id;
  int num_worker_queues = 1;
  if (config.num_gpus > 1) {
    worker_queues[num_worker_queues] =
        config.worker_queues[worker_id + config.num_workers];
    worker_queue_ids[num_worker_queues] = worker_id + config.num_workers;
    num_worker_queues++;
  }

  if (threadIdx.x == 0) {
    for (int i = 0; i < 2; i++) {
      next_task_pos[i] = 0;
      last_task_pos[i] = 0;
    }
  }

  int queue_pos = 0, queue_len = 0;
#ifdef MPK_ENABLE_PROFILING
  size_t task_counter = 0;
#endif
  while (true) {
    if (queue_pos == queue_len) {
      if (threadIdx.x == 0) {
        current_queue_idx = 0;
        while (next_task_pos[current_queue_idx] ==
               last_task_pos[current_queue_idx]) {
          last_task_pos[current_queue_idx] =
              ld_acquire_gpu_u64(&config.worker_queue_last_ready_task_id
                                      [worker_queue_ids[current_queue_idx]]);
          if (next_task_pos[current_queue_idx] <
              last_task_pos[current_queue_idx]) {
            break;
          }
          current_queue_idx =
              (current_queue_idx == num_worker_queues - 1) ? 0
                                                           : current_queue_idx + 1;
          __nanosleep(10);
        }
        assert(next_task_pos[current_queue_idx] + config.per_worker_queue_len >
               last_task_pos[current_queue_idx]);
      }
      __syncthreads();
      int num_loaded_tasks =
          min((int)(last_task_pos[current_queue_idx] -
                    next_task_pos[current_queue_idx]),
              TASK_DESCS_BUFFER_LENGTH);
      if (threadIdx.x < num_loaded_tasks) {
        task_ids[threadIdx.x] = ld_relaxed_gpu_u64(
            &worker_queues[current_queue_idx]
                          [(next_task_pos[current_queue_idx] + threadIdx.x) %
                           config.per_worker_queue_len]);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        next_task_pos[current_queue_idx] += num_loaded_tasks;
      }
      static_assert(sizeof(TaskDesc) % 16 == 0);
      constexpr int TASK_SIZE = sizeof(TaskDesc) / 16;
      for (int i = threadIdx.x; i < num_loaded_tasks * TASK_SIZE;
           i += blockDim.x) {
        int task_idx = i / TASK_SIZE;
        int offset = i % TASK_SIZE;
        load_smem(reinterpret_cast<char *>(task_descs) + i * 16,
                  reinterpret_cast<char *>(
                      config.all_tasks +
                      get_task_position_index(task_ids[task_idx])) +
                      offset * 16);
      }
      kernel::cp_async_fence();
      kernel::cp_async_wait<0>();
      __syncthreads();
      queue_pos = 0;
      queue_len = num_loaded_tasks;
    }
    TaskDesc *task_desc = task_descs + queue_pos;
    TaskId current_task_id = task_ids[queue_pos];
    size_t position_index = get_task_position_index(current_task_id);
    if (threadIdx.x == 0) {
      current_data_id = DATA_INVALID_ID;
      if (position_index >= static_cast<size_t>(config.num_control_tasks)) {
        current_data_id = config.task_to_data_id[position_index];
      }
      if (current_data_id == DATA_INVALID_ID &&
          task_desc->dependent_event != EVENT_INVALID_ID) {
        EventId event_id = task_desc->dependent_event;
        assert(get_event_gpu_id(event_id) == config.my_gpu_id);
        size_t event_index = get_event_position_index(event_id);
        EventCounter needed_counts =
            static_cast<EventCounter>(
                config.all_event_num_triggers[event_index]) *
            get_task_iteration_num(current_task_id);
        EventCounter actual_counts = 0;
        if (is_nvshmem_event(event_id)) {
#ifdef USE_NVSHMEM
          nvshmem_signal_wait_until(
              reinterpret_cast<uint64_t *>(
                  &config.all_event_counters[event_index]),
              NVSHMEM_CMP_EQ,
              needed_counts);
#endif
        } else {
          while (actual_counts < needed_counts) {
            actual_counts =
                ld_acquire_sys_u64(&config.all_event_counters[event_index]);
            __nanosleep(10);
          }
        }
      }
    }
    __syncthreads();

#ifdef MPK_ENABLE_PROFILING
    uint32_t profiler_event_no = task_counter;
    if (current_data_id != DATA_INVALID_ID) {
      profiler_event_no = config.all_data[current_data_id].profiler_group_id;
    } else if (uses_dag_profiler_group(task_desc->task_type)) {
      profiler_event_no = task_desc->profiler_group_id;
    }
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_START(task_desc->task_type, profiler_event_no);
      if (current_data_id != DATA_INVALID_ID) {
        PROFILER_EVENT_METADATA(
            task_desc->task_type, profiler_event_no, current_data_id);
      }
    }
#endif

    if (task_desc->task_type == TASK_TERMINATE) {
      return;
    } else if (task_desc->task_type != TASK_BEGIN_TASK_GRAPH) {
      _execute_task(task_desc, config);
    }
    __syncthreads();

#ifdef MPK_ENABLE_PROFILING
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_END(task_desc->task_type, profiler_event_no);
    }
    task_counter++;
#endif

    if (current_data_id != DATA_INVALID_ID) {
      if (threadIdx.x == 0) {
        int completion_queue_id = config.worker_owner_scheduler[worker_id];
        publish_completion_item(config, completion_queue_id, current_data_id);
      }
    } else if (threadIdx.x == 0) {
      trigger_task_event(config, task_desc, current_task_id, worker_id);
    }
    __syncthreads();
    queue_pos += 1;
  }
}

__device__ __forceinline__ void
execute_worker_hybrid_prelaunch(ResidentRuntimeConfig config) {
  constexpr int TASK_DESCS_BUFFER_LENGTH = std::min(
      (mirage::runtime::WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE - 60) /
          (int)(sizeof(TaskDesc) + sizeof(TaskId)),
      16);
  __shared__ TaskDesc task_descs[TASK_DESCS_BUFFER_LENGTH];
  __shared__ TaskId task_ids[TASK_DESCS_BUFFER_LENGTH];
  __shared__ DataId current_data_id;
  __shared__ int current_queue_idx;
  __shared__ TaskId *worker_queues[2];
  __shared__ int worker_queue_ids[2];
  __shared__ size_t next_task_pos[2];
  __shared__ size_t last_task_pos[2];

#ifdef MPK_ENABLE_PROFILING
  PROFILER_CLOSURE_PARAMS_DECL;
  PROFILER_INIT(static_cast<uint64_t *>(config.profiler_buffer),
                0,
                1,
                (threadIdx.x % WORKER_NUM_THREADS == 0));

#endif
  int const worker_id = blockIdx.x;
  worker_queues[0] = config.worker_queues[worker_id];
  worker_queue_ids[0] = worker_id;
  int num_worker_queues = 1;
  if (config.num_gpus > 1) {
    worker_queues[num_worker_queues] =
        config.worker_queues[worker_id + config.num_workers];
    worker_queue_ids[num_worker_queues] = worker_id + config.num_workers;
    num_worker_queues++;
  }

  if (threadIdx.x == 0) {
    for (int i = 0; i < 2; i++) {
      next_task_pos[i] = 0;
      last_task_pos[i] = 0;
    }
  }

  int queue_pos = 0, queue_len = 0;
#ifdef MPK_ENABLE_PROFILING
  size_t task_counter = 0;
#endif
  while (true) {
    if (queue_pos == queue_len) {
      if (threadIdx.x == 0) {
        current_queue_idx = 0;
        while (next_task_pos[current_queue_idx] ==
               last_task_pos[current_queue_idx]) {
          last_task_pos[current_queue_idx] =
              ld_acquire_gpu_u64(&config.worker_queue_last_ready_task_id
                                      [worker_queue_ids[current_queue_idx]]);
          if (next_task_pos[current_queue_idx] <
              last_task_pos[current_queue_idx]) {
            break;
          }
          current_queue_idx =
              (current_queue_idx == num_worker_queues - 1) ? 0
                                                           : current_queue_idx + 1;
          __nanosleep(10);
        }
        assert(next_task_pos[current_queue_idx] + config.per_worker_queue_len >
               last_task_pos[current_queue_idx]);
      }
      __syncthreads();
      int num_loaded_tasks =
          min((int)(last_task_pos[current_queue_idx] -
                    next_task_pos[current_queue_idx]),
              TASK_DESCS_BUFFER_LENGTH);
      if (threadIdx.x < num_loaded_tasks) {
        task_ids[threadIdx.x] = ld_relaxed_gpu_u64(
            &worker_queues[current_queue_idx]
                          [(next_task_pos[current_queue_idx] + threadIdx.x) %
                           config.per_worker_queue_len]);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        next_task_pos[current_queue_idx] += num_loaded_tasks;
      }
      static_assert(sizeof(TaskDesc) % 16 == 0);
      constexpr int TASK_SIZE = sizeof(TaskDesc) / 16;
      for (int i = threadIdx.x; i < num_loaded_tasks * TASK_SIZE;
           i += blockDim.x) {
        int task_idx = i / TASK_SIZE;
        int offset = i % TASK_SIZE;
        load_smem(reinterpret_cast<char *>(task_descs) + i * 16,
                  reinterpret_cast<char *>(
                      config.all_tasks +
                      get_task_position_index(task_ids[task_idx])) +
                      offset * 16);
      }
      kernel::cp_async_fence();
      kernel::cp_async_wait<0>();
      __syncthreads();
      queue_pos = 0;
      queue_len = num_loaded_tasks;
    }
    TaskDesc *task_desc = task_descs + queue_pos;
    TaskId current_task_id = task_ids[queue_pos];
    size_t position_index = get_task_position_index(current_task_id);
    if (threadIdx.x == 0) {
      current_data_id = DATA_INVALID_ID;
      if (position_index >= static_cast<size_t>(config.num_control_tasks)) {
        current_data_id = config.task_to_data_id[position_index];
      }
      bool wait_on_event = false;
      bool wait_on_data = false;
      if (current_data_id != DATA_INVALID_ID) {
        wait_on_event = task_desc->dependent_event != EVENT_INVALID_ID &&
                        is_nvshmem_event(task_desc->dependent_event);
        wait_on_data = !wait_on_event;
      } else {
        wait_on_event = task_desc->dependent_event != EVENT_INVALID_ID;
      }
      if (wait_on_data) {
        while (atomicAdd(&config.data_pending_predecessor_counts[current_data_id],
                         0u) != 0u) {
          __nanosleep(10);
        }
      } else if (wait_on_event) {
        EventId event_id = task_desc->dependent_event;
        assert(get_event_gpu_id(event_id) == config.my_gpu_id);
        size_t event_index = get_event_position_index(event_id);
        EventCounter needed_counts =
            static_cast<EventCounter>(
                config.all_event_num_triggers[event_index]) *
            get_task_iteration_num(current_task_id);
        EventCounter actual_counts = 0;
        if (is_nvshmem_event(event_id)) {
#ifdef USE_NVSHMEM
          nvshmem_signal_wait_until(
              reinterpret_cast<uint64_t *>(
                  &config.all_event_counters[event_index]),
              NVSHMEM_CMP_EQ,
              needed_counts);
#endif
        } else {
          while (actual_counts < needed_counts) {
            actual_counts =
                ld_acquire_sys_u64(&config.all_event_counters[event_index]);
            __nanosleep(10);
          }
        }
      }
    }
    __syncthreads();

#ifdef MPK_ENABLE_PROFILING
    uint32_t profiler_event_no = task_counter;
    if (current_data_id != DATA_INVALID_ID) {
      profiler_event_no = config.all_data[current_data_id].profiler_group_id;
    } else if (uses_dag_profiler_group(task_desc->task_type)) {
      profiler_event_no = task_desc->profiler_group_id;
    }
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_START(task_desc->task_type, profiler_event_no);
      if (current_data_id != DATA_INVALID_ID) {
        PROFILER_EVENT_METADATA(
            task_desc->task_type, profiler_event_no, current_data_id);
      }
    }
#endif

    if (task_desc->task_type == TASK_TERMINATE) {
      return;
    } else if (task_desc->task_type != TASK_BEGIN_TASK_GRAPH) {
      _execute_task(task_desc, config);
    }
    __syncthreads();

#ifdef MPK_ENABLE_PROFILING
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_END(task_desc->task_type, profiler_event_no);
    }
    task_counter++;
#endif

    if (current_data_id != DATA_INVALID_ID) {
      release_completed_data(config, current_data_id, task_desc->trigger_event);
    } else if (threadIdx.x == 0) {
      trigger_task_event(config, task_desc, current_task_id, worker_id);
    }
    __syncthreads();
    queue_pos += 1;
  }
}

__device__ __forceinline__ void
execute_worker_streaming(ResidentRuntimeConfig config) {
  constexpr int TASK_DESCS_BUFFER_LENGTH = std::min(
      (mirage::runtime::WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE - 256) /
          (int)(sizeof(TaskDesc) + sizeof(TaskId)),
      16);
  __shared__ TaskDesc task_descs[TASK_DESCS_BUFFER_LENGTH];
  __shared__ TaskId task_ids[TASK_DESCS_BUFFER_LENGTH];
  __shared__ TaskId current_task_id;
  __shared__ DataId current_data_id;
  __shared__ ResidentTaskId current_streaming_resident_id;
  __shared__ int current_queue_idx;
  __shared__ int queue_pos;
  __shared__ int queue_len;
  __shared__ bool queue_task_ready;
  __shared__ bool executed_streaming_work;
  __shared__ uint32_t next_streaming_resident_slot;
  __shared__ uint32_t ready_streaming_edge_bits
      [MIRAGE_STREAMING_MAX_READY_SUCCESSORS / 32];
  __shared__ TaskId *worker_queues[2];
  __shared__ int worker_queue_ids[2];
  __shared__ size_t next_task_pos[2];
  __shared__ size_t last_task_pos[2];

#ifdef MPK_ENABLE_PROFILING
  PROFILER_CLOSURE_PARAMS_DECL;
  PROFILER_INIT(static_cast<uint64_t *>(config.profiler_buffer),
                0,
                1,
                (threadIdx.x % WORKER_NUM_THREADS == 0));
#endif

  int const worker_id = blockIdx.x;
  worker_queues[0] = config.worker_queues[worker_id];
  worker_queue_ids[0] = worker_id;
  int num_worker_queues = 1;
  if (config.num_gpus > 1) {
    worker_queues[num_worker_queues] =
        config.worker_queues[worker_id + config.num_workers];
    worker_queue_ids[num_worker_queues] = worker_id + config.num_workers;
    num_worker_queues++;
  }

  if (threadIdx.x == 0) {
    for (int i = 0; i < 2; i++) {
      next_task_pos[i] = 0;
      last_task_pos[i] = 0;
    }
    queue_pos = 0;
    queue_len = 0;
    next_streaming_resident_slot = 0;
  }

#ifdef MPK_ENABLE_PROFILING
  size_t task_counter = 0;
#endif

  while (true) {
    if (queue_pos == queue_len) {
      if (threadIdx.x == 0) {
        current_queue_idx = -1;
        for (int attempt = 0; attempt < num_worker_queues; attempt++) {
          int queue_idx = attempt;
          last_task_pos[queue_idx] =
              ld_acquire_gpu_u64(&config.worker_queue_last_ready_task_id
                                      [worker_queue_ids[queue_idx]]);
          if (next_task_pos[queue_idx] < last_task_pos[queue_idx]) {
            current_queue_idx = queue_idx;
            break;
          }
        }
      }
      __syncthreads();
      if (current_queue_idx >= 0) {
        int num_loaded_tasks =
            min((int)(last_task_pos[current_queue_idx] -
                      next_task_pos[current_queue_idx]),
                TASK_DESCS_BUFFER_LENGTH);
        if (threadIdx.x < num_loaded_tasks) {
          task_ids[threadIdx.x] = ld_relaxed_gpu_u64(
              &worker_queues[current_queue_idx]
                            [(next_task_pos[current_queue_idx] + threadIdx.x) %
                             config.per_worker_queue_len]);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          next_task_pos[current_queue_idx] += num_loaded_tasks;
        }
        static_assert(sizeof(TaskDesc) % 16 == 0);
        constexpr int TASK_SIZE = sizeof(TaskDesc) / 16;
        for (int i = threadIdx.x; i < num_loaded_tasks * TASK_SIZE;
             i += blockDim.x) {
          int task_idx = i / TASK_SIZE;
          int offset = i % TASK_SIZE;
          load_smem(reinterpret_cast<char *>(task_descs) + i * 16,
                    reinterpret_cast<char *>(
                        config.all_tasks +
                        get_task_position_index(task_ids[task_idx])) +
                        offset * 16);
        }
        kernel::cp_async_fence();
        kernel::cp_async_wait<0>();
        __syncthreads();
        if (threadIdx.x == 0) {
          queue_pos = 0;
          queue_len = num_loaded_tasks;
        }
      } else if (threadIdx.x == 0) {
        queue_pos = 0;
        queue_len = 0;
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      current_task_id = TASK_INVALID_ID;
      current_data_id = DATA_INVALID_ID;
      queue_task_ready = false;
      if (queue_pos < queue_len) {
        TaskDesc const *task_desc = task_descs + queue_pos;
        current_task_id = task_ids[queue_pos];
        size_t position_index = get_task_position_index(current_task_id);
        if (position_index >= static_cast<size_t>(config.num_control_tasks)) {
          current_data_id = config.task_to_data_id[position_index];
          if (current_data_id != DATA_INVALID_ID) {
            ResidentTaskId resident_task_id =
                config.all_data[current_data_id].resident_task_id;
            assert(resident_task_is_prelaunched(
                config.resident_tasks[resident_task_id]));
          }
        }
        queue_task_ready = prelaunched_task_ready_nonblocking(
            config, task_desc, current_task_id, current_data_id);
      }
    }
    __syncthreads();

    if (queue_task_ready) {
      TaskDesc *task_desc = task_descs + queue_pos;
      if (task_desc->task_type == TASK_TERMINATE) {
        return;
      }

#ifdef MPK_ENABLE_PROFILING
      uint32_t profiler_event_no = task_counter;
      if (current_data_id != DATA_INVALID_ID) {
        profiler_event_no = config.all_data[current_data_id].profiler_group_id;
      } else if (uses_dag_profiler_group(task_desc->task_type)) {
        profiler_event_no = task_desc->profiler_group_id;
      }
      PROFILER_EVENT_START(task_desc->task_type, profiler_event_no);
      if (current_data_id != DATA_INVALID_ID) {
        PROFILER_EVENT_METADATA(
            task_desc->task_type, profiler_event_no, current_data_id);
      }
#endif

      if (task_desc->task_type != TASK_BEGIN_TASK_GRAPH) {
        _execute_task(task_desc, config);
      }
      __syncthreads();

#ifdef MPK_ENABLE_PROFILING
      PROFILER_EVENT_END(task_desc->task_type, profiler_event_no);
      task_counter++;
#endif

      if (current_data_id != DATA_INVALID_ID) {
        release_completed_data_streaming(
            config, current_data_id, false /*count_streaming_completion*/,
            streaming_uses_hybrid_base(config) /*count_terminal_completion*/,
            ready_streaming_edge_bits);
        if (threadIdx.x == 0 && streaming_uses_legacy_base(config)) {
          trigger_data_event(
              config, config.all_data + current_data_id, task_desc->task_type,
              current_task_id, worker_id);
        }
      } else if (threadIdx.x == 0) {
        trigger_task_event(config, task_desc, current_task_id, worker_id);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        queue_pos += 1;
      }
      __syncthreads();
      continue;
    }

    if (threadIdx.x == 0) {
      current_streaming_resident_id = RESIDENT_TASK_INVALID_ID;
      current_data_id = DATA_INVALID_ID;
      executed_streaming_work = false;
      uint32_t resident_begin = streaming_worker_resident_begin(config, worker_id);
      uint32_t resident_end = streaming_worker_resident_end(config, worker_id);
      uint32_t resident_count = resident_end - resident_begin;
      for (uint32_t attempt = 0; attempt < resident_count; attempt++) {
        uint32_t resident_slot =
            resident_begin +
            ((next_streaming_resident_slot + attempt) % resident_count);
        ResidentTaskId resident_task_id =
            config.worker_streaming_resident_ids[resident_slot];
        if (!pop_streaming_resident_ready_data(
                config, resident_task_id, &current_data_id)) {
          continue;
        }
        current_streaming_resident_id = resident_task_id;
        next_streaming_resident_slot =
            (next_streaming_resident_slot + attempt + 1) % resident_count;
        executed_streaming_work = true;
        break;
      }
    }
    __syncthreads();

    if (!executed_streaming_work) {
      __nanosleep(10);
      continue;
    }

    for (int burst = 0; burst < MIRAGE_STREAMING_BURST_QUOTA; burst++) {
      ResidentTaskDesc const *resident_task_desc =
          config.resident_tasks + current_streaming_resident_id;
      DataDesc const *data_desc = config.all_data + current_data_id;
      assert(resident_task_is_streaming(*resident_task_desc));
      if (threadIdx.x == 0 &&
          data_desc->dependent_event != EVENT_INVALID_ID) {
        assert(!is_nvshmem_event(data_desc->dependent_event) &&
               "streaming_data supports only single-GPU execution");
      }
      __syncthreads();

#ifdef MPK_ENABLE_PROFILING
      uint32_t profiler_event_no = data_desc->profiler_group_id;
      PROFILER_EVENT_START(resident_task_desc->task_type, profiler_event_no);
      PROFILER_EVENT_METADATA(
          resident_task_desc->task_type, profiler_event_no, current_data_id);
#endif

      _execute_task(resident_task_desc, data_desc, config);
      __syncthreads();

#ifdef MPK_ENABLE_PROFILING
      PROFILER_EVENT_END(resident_task_desc->task_type, profiler_event_no);
      task_counter++;
#endif

      release_completed_data_streaming(
          config, current_data_id, true /*count_streaming_completion*/,
          streaming_uses_hybrid_base(config) /*count_terminal_completion*/,
          ready_streaming_edge_bits);
      if (threadIdx.x == 0 && streaming_uses_legacy_base(config)) {
        TaskId compat_task_id =
            compute_task_id(atomicAdd(config.current_iteration, 0u),
                            config.data_to_task_id[current_data_id]);
        trigger_data_event(config,
                           data_desc,
                           resident_task_desc->task_type,
                           compat_task_id,
                           worker_id);
      }
      __syncthreads();

      if (burst == MIRAGE_STREAMING_BURST_QUOTA - 1) {
        break;
      }
      if (threadIdx.x == 0) {
        if (!pop_streaming_resident_ready_data(config,
                                               current_streaming_resident_id,
                                               &current_data_id)) {
          current_streaming_resident_id = RESIDENT_TASK_INVALID_ID;
        }
      }
      __syncthreads();
      if (current_streaming_resident_id == RESIDENT_TASK_INVALID_ID) {
        break;
      }
    }
  }
}

__device__ __forceinline__ void
seed_streaming_first_data(ResidentRuntimeConfig const &config) {
  if (threadIdx.x != 0) {
    return;
  }
  for (int i = 0; i < config.num_first_data_ids; i++) {
    DataId data_id = config.first_data_ids[i];
    ResidentTaskId resident_task_id = config.all_data[data_id].resident_task_id;
    if (!resident_task_is_streaming(config.resident_tasks[resident_task_id])) {
      continue;
    }
    publish_streaming_resident_ready_data(config, resident_task_id, data_id);
  }
}

__device__ __forceinline__ void
execute_scheduler_streaming(ResidentRuntimeConfig config) {
  int const sched_id = blockIdx.x;
  if (sched_id >= config.num_local_schedulers) {
    return;
  }

  __shared__ size_t cur_event_pos[2];
  __shared__ size_t last_event_pos[2];
  __shared__ int queue_idx;
  __shared__ EventId current_event_id;
  __shared__ EventDesc current_event_desc;
  __shared__ bool continue_iteration;

  if (threadIdx.x == 0) {
    cur_event_pos[0] = 0;
    cur_event_pos[1] = 0;
    last_event_pos[0] = 0;
    last_event_pos[1] = 0;
    queue_idx = 0;
  }
  __syncthreads();

  EventId *sched_queues[2];
  int sched_queue_ids[2];
  int num_sched_queues = 1;
  sched_queues[0] = config.sched_queues[sched_id];
  sched_queue_ids[0] = sched_id;
  int num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  sched_queues[num_sched_queues] = config.sched_queues[num_schedulers];
  sched_queue_ids[num_sched_queues] = num_schedulers;
  num_sched_queues++;

  unsigned long long int my_first_worker, my_last_worker;
  get_first_last_ids(config.num_workers,
                     config.num_local_schedulers,
                     sched_id,
                     &my_first_worker,
                     &my_last_worker);
  size_t worker_queue_next_free_task_pos[MAX_WORKER_PER_SCHEDULER];
  for (int i = 0; i < MAX_WORKER_PER_SCHEDULER; i++) {
    worker_queue_next_free_task_pos[i] = 0;
  }

  size_t iteration_num = 0;
  while (true) {
    if (threadIdx.x == 0) {
      while (cur_event_pos[queue_idx] == last_event_pos[queue_idx]) {
        last_event_pos[queue_idx] = ld_acquire_gpu_u64(
            &config.sched_queue_last_ready_event_id[sched_queue_ids[queue_idx]]);
        if (cur_event_pos[queue_idx] < last_event_pos[queue_idx]) {
          break;
        }
        queue_idx = (queue_idx == num_sched_queues - 1) ? 0 : queue_idx + 1;
        __nanosleep(10);
      }
      assert(cur_event_pos[queue_idx] + config.per_sched_queue_len >
             last_event_pos[queue_idx]);
      current_event_id = ld_relaxed_gpu_u64(
          &sched_queues[queue_idx]
                       [cur_event_pos[queue_idx] % config.per_sched_queue_len]);
      current_event_desc = config.all_events[current_event_id];
      cur_event_pos[queue_idx] += 1;
    }
    __syncthreads();

    if (is_termination_event(current_event_id, current_event_desc)) {
      if (threadIdx.x == 0) {
        for (int worker = my_first_worker; worker < my_last_worker; worker++) {
          publish_worker_item(config, worker, 0);
        }
      }
      return;
    }

    if (current_event_desc.event_type == EVENT_END_OF_TASK_GRAPH) {
      if (threadIdx.x == 0) {
#ifdef MODE_ONLINE_NOTOKEN
        continue_iteration = prepare_next_batch(config, iteration_num);
#else
        continue_iteration = prepare_next_batch(config);
#endif
      }
      __syncthreads();
      if (!continue_iteration) {
        if (threadIdx.x == 0) {
          terminate_schedulers(config);
        }
        return;
      }

      reset_streaming_iteration_state_parallel(config, threadIdx.x, blockDim.x);
      if (threadIdx.x == 0) {
        int next_worker = static_cast<int>(my_first_worker);
        enqueue_worker_item(
            config,
            next_worker,
            &worker_queue_next_free_task_pos[next_worker - my_first_worker],
            compute_task_id(iteration_num + 1, BEGIN_TASK_GRAPH_TASK_ID));
      }
      __syncthreads();
      continue;
    }

    if (current_event_desc.event_type == EVENT_LAUNCH_DEPENDENT_TASKS) {
      if (threadIdx.x == 0) {
        iteration_num += 1;
        if (current_event_id == static_cast<EventId>(config.begin_event_index)) {
          *config.current_iteration = static_cast<uint32_t>(iteration_num);
        }
        int next_worker = static_cast<int>(my_first_worker);
        enqueue_prelaunched_dependent_tasks_partitioned(
            config,
            my_first_worker,
            my_last_worker,
            &next_worker,
            worker_queue_next_free_task_pos,
            iteration_num,
            current_event_desc);
      }
      __syncthreads();
      if (sched_id == 0 &&
          current_event_id == static_cast<EventId>(config.begin_event_index)) {
        seed_streaming_first_data(config);
      }
      __syncthreads();
      continue;
    }

    if (current_event_desc.event_type == EVENT_LAUNCH_TASKS ||
        current_event_desc.event_type == EVENT_LAUNCH_MASSIVE_TASKS) {
      TaskId my_first_task = current_event_desc.first_task_id;
      TaskId my_last_task = current_event_desc.last_task_id;
      if (current_event_desc.event_type == EVENT_LAUNCH_MASSIVE_TASKS) {
        get_first_last_ids(current_event_desc.last_task_id -
                               current_event_desc.first_task_id,
                           config.num_local_schedulers,
                           sched_id,
                           &my_first_task,
                           &my_last_task);
        my_first_task += current_event_desc.first_task_id;
        my_last_task += current_event_desc.first_task_id;
      }
      if (threadIdx.x == 0) {
        int next_worker = static_cast<int>(my_first_worker);
        enqueue_prelaunched_task_range(config,
                                       my_first_worker,
                                       my_last_worker,
                                       &next_worker,
                                       worker_queue_next_free_task_pos,
                                       iteration_num,
                                       my_first_task,
                                       my_last_task);
      }
      __syncthreads();
      continue;
    }
  }
}

__device__ __forceinline__ void
execute_scheduler_scheduler_dispatch(ResidentRuntimeConfig config, int offset) {
  int const num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  int const num_schedulers_per_sm = std::min((int)blockDim.x / 32, 4);
  int const warp_id = threadIdx.x / 32;
  if (threadIdx.x % 32 != 0 || warp_id >= num_schedulers_per_sm) {
    return;
  }

  int const sched_id = blockIdx.x * num_schedulers_per_sm + warp_id + offset;
  int num_sched_queues = 1;
  size_t iteration_num = 0;
  EventId *sched_queues[2];
  int sched_queue_ids[2];
  sched_queues[0] = config.sched_queues[sched_id];
  sched_queue_ids[0] = sched_id;
  unsigned long long int my_first_worker, my_last_worker;

  if (sched_id < config.num_local_schedulers) {
    sched_queues[num_sched_queues] = config.sched_queues[num_schedulers];
    sched_queue_ids[num_sched_queues] = num_schedulers;
    num_sched_queues++;
    get_first_last_ids(config.num_workers,
                       config.num_local_schedulers,
                       sched_id,
                       &my_first_worker,
                       &my_last_worker);
  } else {
    get_first_last_ids(config.num_workers,
                       config.num_remote_schedulers,
                       sched_id - config.num_local_schedulers,
                       &my_first_worker,
                       &my_last_worker);
    my_first_worker += config.num_workers;
    my_last_worker += config.num_workers;
  }

  size_t cur_event_pos[2], last_event_pos[2];
  for (int i = 0; i < 2; i++) {
    cur_event_pos[i] = 0;
    last_event_pos[i] = 0;
  }
  size_t completion_cur_pos = 0, completion_last_pos = 0;
  size_t worker_queue_next_free_task_pos[MAX_WORKER_PER_SCHEDULER];
  for (int i = 0; i < MAX_WORKER_PER_SCHEDULER; i++) {
    worker_queue_next_free_task_pos[i] = 0;
  }

  int next_worker = my_first_worker;
  int queue_idx = 0;
  while (true) {
    bool processed_completion = false;
    if (sched_id < config.num_local_schedulers) {
      completion_last_pos = ld_acquire_gpu_u64(
          &config.completion_queue_last_ready_data_id[sched_id]);
      if (completion_cur_pos < completion_last_pos) {
        assert(completion_cur_pos + config.per_completion_queue_len >
               completion_last_pos);
        DataId completed_data_id = static_cast<DataId>(ld_relaxed_gpu_u64(
            &config.completion_queues[sched_id]
                                     [completion_cur_pos %
                                      config.per_completion_queue_len]));
        completion_cur_pos += 1;
        atomicAdd(config.completed_data_this_iteration_count, 1u);
        DataId edge_begin = config.data_edge_offsets[completed_data_id];
        DataId edge_end = config.data_edge_offsets[completed_data_id + 1];
        if (edge_begin == edge_end) {
          unsigned int completed_terminal =
              atomicAdd(config.completed_terminal_data_count, 1u) + 1;
          if (completed_terminal ==
              static_cast<unsigned int>(config.num_terminal_data)) {
            publish_scheduler_event(config, 0, config.end_event_index);
          }
        } else {
          for (DataId edge_idx = edge_begin; edge_idx < edge_end; edge_idx++) {
            DataId succ_data_id = config.data_edge_targets[edge_idx];
            unsigned int remaining =
                atomicSub(&config.data_pending_predecessor_counts[succ_data_id],
                          1u);
            if (remaining == 1) {
              TaskId succ_task_id = config.data_to_task_id[succ_data_id];
              assert(succ_task_id != TASK_INVALID_ID);
              enqueue_worker_item(
                  config,
                  next_worker,
                  &worker_queue_next_free_task_pos[next_worker - my_first_worker],
                  compute_task_id(iteration_num, succ_task_id));
              next_worker = (next_worker == my_last_worker - 1)
                                ? my_first_worker
                                : next_worker + 1;
            }
          }
        }
        processed_completion = true;
      }
    }
    if (processed_completion) {
      continue;
    }

    while (cur_event_pos[queue_idx] == last_event_pos[queue_idx]) {
      if (sched_id < config.num_local_schedulers) {
        completion_last_pos = ld_acquire_gpu_u64(
            &config.completion_queue_last_ready_data_id[sched_id]);
        if (completion_cur_pos < completion_last_pos) {
          processed_completion = true;
          break;
        }
      }
      last_event_pos[queue_idx] = ld_acquire_gpu_u64(
          &config.sched_queue_last_ready_event_id[sched_queue_ids[queue_idx]]);
      if (cur_event_pos[queue_idx] < last_event_pos[queue_idx]) {
        break;
      } else {
        queue_idx = (queue_idx == num_sched_queues - 1) ? 0 : queue_idx + 1;
      }
      __nanosleep(10);
    }
    if (processed_completion) {
      continue;
    }

    assert(cur_event_pos[queue_idx] + config.per_sched_queue_len >
           last_event_pos[queue_idx]);
    EventId event_id = ld_relaxed_gpu_u64(
        &sched_queues[queue_idx]
                     [cur_event_pos[queue_idx] % config.per_sched_queue_len]);
    EventDesc e = config.all_events[event_id];
    cur_event_pos[queue_idx] += 1;

    if (is_termination_event(event_id, e)) {
      if (sched_id < config.num_local_schedulers) {
        for (int worker = my_first_worker; worker < my_last_worker; worker++) {
          enqueue_worker_item(
              config,
              worker,
              &worker_queue_next_free_task_pos[worker - my_first_worker],
              0);
        }
      }
      return;
    }

    if (e.event_type == EVENT_END_OF_TASK_GRAPH) {
      if (sched_id != 0) {
        continue;
      }
      bool has_next_batch;
#ifdef MODE_ONLINE_NOTOKEN
      has_next_batch = prepare_next_batch(config, iteration_num);
#else
      has_next_batch = prepare_next_batch(config);
#endif
      if (!has_next_batch) {
        terminate_schedulers(config);
      } else {
        reset_iteration_state_scalar(config);
        enqueue_worker_item(
            config,
            next_worker,
            &worker_queue_next_free_task_pos[next_worker - my_first_worker],
            compute_task_id(iteration_num + 1, BEGIN_TASK_GRAPH_TASK_ID));
        next_worker = (next_worker == my_last_worker - 1) ? my_first_worker
                                                          : next_worker + 1;
      }
      continue;
    }

    if (e.event_type == EVENT_LAUNCH_DEPENDENT_TASKS &&
        event_id == static_cast<EventId>(config.begin_event_index) &&
        config.num_first_data_ids > 0) {
      if (sched_id != 0) {
        continue;
      }
      iteration_num += 1;
      *config.current_iteration = static_cast<uint32_t>(iteration_num);
      for (int i = 0; i < config.num_first_data_ids; i++) {
        DataId data_id = config.first_data_ids[i];
        TaskId task_id = config.data_to_task_id[data_id];
        assert(task_id != TASK_INVALID_ID);
        enqueue_worker_item(
            config,
            next_worker,
            &worker_queue_next_free_task_pos[next_worker - my_first_worker],
            compute_task_id(iteration_num, task_id));
        next_worker = (next_worker == my_last_worker - 1) ? my_first_worker
                                                          : next_worker + 1;
      }
      continue;
    }

    if (e.event_type == EVENT_LAUNCH_DEPENDENT_TASKS) {
      iteration_num = iteration_num + 1;
      assert(sched_id < config.num_local_schedulers);
      for (size_t i = 0;
           i < (e.last_task_id - e.first_task_id + config.num_workers - 1) /
                   config.num_workers;
           i++) {
        for (size_t worker = my_first_worker; worker < my_last_worker;
             worker++) {
          size_t position_index =
              e.first_task_id + i * config.num_workers + worker;
          if (position_index < e.last_task_id) {
            enqueue_worker_item(
                config,
                next_worker,
                &worker_queue_next_free_task_pos[next_worker - my_first_worker],
                compute_task_id(iteration_num, position_index));
            next_worker = (next_worker == my_last_worker - 1)
                              ? my_first_worker
                              : next_worker + 1;
          }
        }
      }
      continue;
    }

    TaskId my_first_task = e.first_task_id, my_last_task = e.last_task_id;
    if (e.event_type == EVENT_LAUNCH_MASSIVE_TASKS) {
      assert(sched_id < config.num_local_schedulers);
      get_first_last_ids(e.last_task_id - e.first_task_id,
                         config.num_local_schedulers,
                         sched_id,
                         &my_first_task,
                         &my_last_task);
      my_first_task += e.first_task_id;
      my_last_task += e.first_task_id;
    }
    for (size_t task_pos = my_first_task; task_pos < my_last_task; task_pos++) {
      enqueue_worker_item(
          config,
          next_worker,
          &worker_queue_next_free_task_pos[next_worker - my_first_worker],
          compute_task_id(iteration_num, task_pos));
      next_worker = (next_worker == my_last_worker - 1) ? my_first_worker
                                                        : next_worker + 1;
    }
  }
}

// need to alter as there is only one warp per block
__device__ __forceinline__ void
execute_scheduler_hybrid_prelaunch(ResidentRuntimeConfig config, int offset) {
  int const num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  int const num_schedulers_per_sm = std::min((int)blockDim.x / 32, 4);
  int const warp_id = threadIdx.x / 32;
  if (warp_id >= num_schedulers_per_sm) {
    return;
  }
  int const lane_id = threadIdx.x % 32;
  __shared__ int warp_action[4];
  constexpr int SCHED_ACTION_NONE = 0;
  constexpr int SCHED_ACTION_RESET = 1;
  constexpr int SCHED_ACTION_RETURN = 2;

  int const sched_id = blockIdx.x * num_schedulers_per_sm + warp_id + offset;
  int num_sched_queues = 1;
  size_t iteration_num = 0;
  EventId *sched_queues[2];
  int sched_queue_ids[2];
  sched_queues[0] = config.sched_queues[sched_id];
  sched_queue_ids[0] = sched_id;
  unsigned long long int my_first_worker, my_last_worker;

  if (sched_id < config.num_local_schedulers) {
    sched_queues[num_sched_queues] = config.sched_queues[num_schedulers];
    sched_queue_ids[num_sched_queues] = num_schedulers;
    num_sched_queues++;
    get_first_last_ids(config.num_workers,
                       config.num_local_schedulers,
                       sched_id,
                       &my_first_worker,
                       &my_last_worker);
  } else {
    get_first_last_ids(config.num_workers,
                       config.num_remote_schedulers,
                       sched_id - config.num_local_schedulers,
                       &my_first_worker,
                       &my_last_worker);
    my_first_worker += config.num_workers;
    my_last_worker += config.num_workers;
  }

  size_t cur_event_pos[2], last_event_pos[2];
  for (int i = 0; i < 2; i++) {
    cur_event_pos[i] = 0;
    last_event_pos[i] = 0;
  }
  size_t worker_queue_next_free_task_pos[MAX_WORKER_PER_SCHEDULER];
  for (int i = 0; i < MAX_WORKER_PER_SCHEDULER; i++) {
    worker_queue_next_free_task_pos[i] = 0;
  }

  int next_worker = my_first_worker;
  int queue_idx = 0;
  while (true) {
    if (lane_id == 0) {
      warp_action[warp_id] = SCHED_ACTION_NONE;

      while (cur_event_pos[queue_idx] == last_event_pos[queue_idx]) {
        last_event_pos[queue_idx] = ld_acquire_gpu_u64(
            &config.sched_queue_last_ready_event_id[sched_queue_ids[queue_idx]]);
        if (cur_event_pos[queue_idx] < last_event_pos[queue_idx]) {
          break;
        } else {
          queue_idx = (queue_idx == num_sched_queues - 1) ? 0 : queue_idx + 1;
        }
        __nanosleep(10);
      }

      assert(cur_event_pos[queue_idx] + config.per_sched_queue_len >
             last_event_pos[queue_idx]);
      EventId event_id = ld_relaxed_gpu_u64(
          &sched_queues[queue_idx]
                       [cur_event_pos[queue_idx] % config.per_sched_queue_len]);
      EventDesc e = config.all_events[event_id];
      cur_event_pos[queue_idx] += 1;

      if (is_termination_event(event_id, e)) {
        if (sched_id < config.num_local_schedulers) {
          for (int worker = my_first_worker; worker < my_last_worker; worker++) {
            enqueue_worker_item(
                config,
                worker,
                &worker_queue_next_free_task_pos[worker - my_first_worker],
                0);
          }
        }
        warp_action[warp_id] = SCHED_ACTION_RETURN;
      } else if (e.event_type == EVENT_END_OF_TASK_GRAPH) {
        if (sched_id == 0) {
          bool has_next_batch;
#ifdef MODE_ONLINE_NOTOKEN
          has_next_batch = prepare_next_batch(config, iteration_num);
#else
          has_next_batch = prepare_next_batch(config);
#endif
          if (!has_next_batch) {
            terminate_schedulers(config);
          } else {
            warp_action[warp_id] = SCHED_ACTION_RESET;
          }
        }
      } else if (e.event_type == EVENT_LAUNCH_DEPENDENT_TASKS) {
        iteration_num = iteration_num + 1;
        assert(sched_id < config.num_local_schedulers);
        if (event_id == static_cast<EventId>(config.begin_event_index) &&
            sched_id == 0) {
          *config.current_iteration = static_cast<uint32_t>(iteration_num);
        }
        for (size_t i = 0;
             i < (e.last_task_id - e.first_task_id + config.num_workers - 1) /
                     config.num_workers;
             i++) {
          for (size_t worker = my_first_worker; worker < my_last_worker;
               worker++) {
            size_t position_index =
                e.first_task_id + i * config.num_workers + worker;
            if (position_index < e.last_task_id) {
              enqueue_worker_item(
                  config,
                  next_worker,
                  &worker_queue_next_free_task_pos[next_worker - my_first_worker],
                  compute_task_id(iteration_num, position_index));
              next_worker = (next_worker == my_last_worker - 1)
                                ? my_first_worker
                                : next_worker + 1;
            }
          }
        }
      } else {
        TaskId my_first_task = e.first_task_id, my_last_task = e.last_task_id;
        if (e.event_type == EVENT_LAUNCH_MASSIVE_TASKS) {
          assert(sched_id < config.num_local_schedulers);
          get_first_last_ids(e.last_task_id - e.first_task_id,
                             config.num_local_schedulers,
                             sched_id,
                             &my_first_task,
                             &my_last_task);
          my_first_task += e.first_task_id;
          my_last_task += e.first_task_id;
        }
        for (size_t task_pos = my_first_task; task_pos < my_last_task;
             task_pos++) {
          enqueue_worker_item(
              config,
              next_worker,
              &worker_queue_next_free_task_pos[next_worker - my_first_worker],
              compute_task_id(iteration_num, task_pos));
          next_worker = (next_worker == my_last_worker - 1) ? my_first_worker
                                                            : next_worker + 1;
        }
      }
    }

    __syncwarp();
    if (warp_action[warp_id] == SCHED_ACTION_RETURN) {
      return;
    }
    if (warp_action[warp_id] == SCHED_ACTION_RESET) {
      reset_iteration_state_parallel(config, lane_id, 32);
      __syncwarp();
      if (lane_id == 0) {
        enqueue_worker_item(
            config,
            next_worker,
            &worker_queue_next_free_task_pos[next_worker - my_first_worker],
            compute_task_id(iteration_num + 1, BEGIN_TASK_GRAPH_TASK_ID));
        next_worker = (next_worker == my_last_worker - 1) ? my_first_worker
                                                          : next_worker + 1;
      }
      __syncwarp();
    }
  }
}

#if MIRAGE_COMPILE_RESIDENT_SCHEDULER_DISPATCH
__global__ __launch_bounds__(WORKER_NUM_THREADS,
                             1) void
persistent_kernel_scheduler_dispatch(ResidentRuntimeConfig config) {
  persistent_checker(config);
  if (blockIdx.x < config.num_workers) {
    execute_worker_scheduler_dispatch(config);
  } else {
    execute_scheduler_scheduler_dispatch(config, -(4 * config.num_workers));
  }
}

__global__ __launch_bounds__(WORKER_NUM_THREADS,
                             1) void
worker_kernel_scheduler_dispatch(ResidentRuntimeConfig config) {
  worker_checker(config);
  execute_worker_scheduler_dispatch(config);
}

__global__ void scheduler_kernel_scheduler_dispatch(
    ResidentRuntimeConfig config) {
  scheduler_checker(config);
  execute_scheduler_scheduler_dispatch(config, 0);
}
#endif

#if MIRAGE_COMPILE_RESIDENT_HYBRID_PRELAUNCH
__global__ __launch_bounds__(WORKER_NUM_THREADS,
                             1) void
persistent_kernel_hybrid_prelaunch(ResidentRuntimeConfig config) {
  persistent_checker(config);
  if (blockIdx.x < config.num_workers) {
    execute_worker_hybrid_prelaunch(config);
  } else {
    execute_scheduler_hybrid_prelaunch(config, -(4 * config.num_workers));
  }
}

__global__ __launch_bounds__(WORKER_NUM_THREADS,
                             1) void
worker_kernel_hybrid_prelaunch(ResidentRuntimeConfig config) {
  worker_checker(config);
  execute_worker_hybrid_prelaunch(config);
}

__global__ void scheduler_kernel_hybrid_prelaunch(
    ResidentRuntimeConfig config) {
  scheduler_checker(config);
  execute_scheduler_hybrid_prelaunch(config, 0);
}
#endif

#if MIRAGE_COMPILE_STREAMING_RUNTIME
__global__ __launch_bounds__(WORKER_NUM_THREADS,
                             1) void
worker_kernel_streaming(ResidentRuntimeConfig config) {
  worker_checker(config);
  execute_worker_streaming(config);
}

__global__ void scheduler_kernel_streaming(ResidentRuntimeConfig config) {
  scheduler_checker(config);
  execute_scheduler_streaming(config);
}
#endif

template <typename DT>
DT *gpu_malloc(size_t size) {
  void *dst_ptr;
#ifdef USE_NVSHMEM
  dst_ptr = nvshmem_malloc(size);
#else
  cudaMalloc(&dst_ptr, size);
#endif
  return static_cast<DT *>(dst_ptr);
}

void gpu_free(void *ptr) {
#ifdef USE_NVSHMEM
  nvshmem_free(ptr);
#else
  cudaFree(ptr);
#endif
}

// The following function will be generated by the transpiler
static void _init_persistent_kernel(std::vector<FullTaskDesc> &all_tasks,
                                    std::vector<EventDesc> &all_events,
                                    std::vector<TaskId> &first_tasks,
                                    std::vector<ResidentTaskDesc> &resident_tasks,
                                    std::vector<FullDataDesc> &all_fulldata,
                                    std::vector<DataEdgeDesc> &data_edges,
                                    std::vector<DataId> &first_data_ids,
                                    std::vector<DataId> &task_to_data_id,
                                    int num_gpus,
                                    int my_gpu_id);

static ResidentRuntimeConfig global_runtime_config;

// meta_tensors[0]: seq_length
// meta_tensors[1]: tokens
// meta_tensors[2]: input_tokens
// meta_tensors[3]: output_tokens
// meta_tensors[4]: new_tokens_nums
// meta_tensors[5]: prompt_length
// meta_tensors[6]: qo_indptr_buffer
// meta_tensors[7]: paged_kv_indptr_buffer
// meta_tensors[8]: paged_kv_indices_buffer
// meta_tensors[9]: paged_kv_last_page_len_buffer

extern "C" void init_request_resources() {
  init_kernel<<<dim3(1, 1, 1), dim3(INIT_NUM_THREADS, 1, 1)>>>(
      global_runtime_config);
  cudaStreamSynchronize(NULL);
}

extern "C" void init_persistent_kernel(std::vector<void *> meta_tensors,
                                       void *profiler_buffer,
                                       int my_rank,
                                       int num_workers,
                                       int num_local_schedulers,
                                       int num_remote_schedulers,
                                       int max_seq_length,
                                       int total_num_requests,
                                       long long eos_token_id,
                                       int allocate_nvshmem_teams) {
  assert(meta_tensors.size() == 10);
  global_runtime_config.step = static_cast<int *>(meta_tensors[0]);
  global_runtime_config.tokens = static_cast<long long *>(meta_tensors[1]);
  global_runtime_config.input_tokens =
      static_cast<long long *>(meta_tensors[2]);
  global_runtime_config.output_tokens =
      static_cast<long long *>(meta_tensors[3]);
  global_runtime_config.new_token_nums = static_cast<int *>(meta_tensors[4]);
  global_runtime_config.prompt_length = static_cast<int *>(meta_tensors[5]);
  global_runtime_config.qo_indptr_buffer = static_cast<int *>(meta_tensors[6]);
  global_runtime_config.paged_kv_indptr_buffer =
      static_cast<int *>(meta_tensors[7]);
  global_runtime_config.paged_kv_indices_buffer =
      static_cast<int *>(meta_tensors[8]);
  global_runtime_config.paged_kv_last_page_len_buffer =
      static_cast<int *>(meta_tensors[9]);
  global_runtime_config.num_workers = num_workers;
  global_runtime_config.num_local_schedulers = num_local_schedulers;
  global_runtime_config.num_remote_schedulers = num_remote_schedulers;
  global_runtime_config.max_seq_length = max_seq_length;
  global_runtime_config.eos_token_id = eos_token_id;
  global_runtime_config.profiler_buffer = profiler_buffer;
  int num_schedulers = num_local_schedulers + num_remote_schedulers;

  // Initialize nvshmem
  cudaSetDevice(my_rank);

#ifdef USE_NVSHMEM
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  nvshmem_barrier_all();
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  printf("MPK: Rank%d is Ready. Worldsize=%d\n", mype, npes);

  // Create nvshmem teams
  // For now, we assume we always need these teams. In the future, we should
  // determine the numebr of teams by scanning kernels.
  if (allocate_nvshmem_teams > 0) {
    int num_teams = allocate_nvshmem_teams;
    printf("MPK: Rank%d is allocating %d nvshmem teams. The more gpus "
           "involved, the longer this takes.\n",
           mype,
           num_teams);
    std::vector<nvshmem_team_t> teams_host(num_teams);
    for (int i = 0; i < num_teams; i++) {
      NVSHMEM_CHECK(nvshmem_team_split_strided(
          NVSHMEM_TEAM_WORLD, 0, 1, npes, nullptr, 0, &teams_host[i]));
      if (mype == 0) {
        printf("MPK: Creating nvshmem team %d/%d, idx %d\n",
               i + 1,
               num_teams,
               teams_host[i]);
      }
    }
    global_runtime_config.nvshmem_teams =
        gpu_malloc<nvshmem_team_t>(num_teams * sizeof(nvshmem_team_t));
    cudaMemcpy(global_runtime_config.nvshmem_teams,
               teams_host.data(),
               num_teams * sizeof(nvshmem_team_t),
               cudaMemcpyHostToDevice);
    printf("MPK: Rank%d finished allocating nvshmem teams\n", mype);
  }
#else
  int mype = 0;
  int npes = 1;
#endif

#if defined(MODE_OFFLINE) || defined(MODE_ONLINE)
  global_runtime_config.request_ids =
      gpu_malloc<int>(sizeof(int) * (MPK_MAX_NUM_BATCHED_REQUESTS + 1));
  global_runtime_config.next_request_id = gpu_malloc<int>(sizeof(int));
  global_runtime_config.page_queue =
      gpu_malloc<int>(MPK_MAX_NUM_PAGES * sizeof(int));
  global_runtime_config.page_queue_head = gpu_malloc<int>(sizeof(int));
  global_runtime_config.page_queue_tail = gpu_malloc<int>(sizeof(int));
  global_runtime_config.total_num_requests = total_num_requests;
#endif
  global_runtime_config.per_worker_queue_len = 1024;
  global_runtime_config.per_sched_queue_len = 1024;
  global_runtime_config.per_completion_queue_len = 4096;
  global_runtime_config.num_gpus = npes;
  global_runtime_config.my_gpu_id = mype;
  global_runtime_config.num_graphs = 1;
  global_runtime_config.resident_execution_mode =
      MIRAGE_RESIDENT_EXECUTION_MODE_VALUE;
  global_runtime_config.streaming_base_execution_mode =
      MIRAGE_STREAMING_BASE_EXECUTION_MODE_VALUE;
  if (resident_uses_streaming(global_runtime_config)) {
    assert(npes == 1 && "streaming_data currently supports only single-GPU execution");
  }
  global_runtime_config.split_worker_scheduler =
      resident_uses_streaming(global_runtime_config)
          ? true
          : (resident_uses_hybrid_prelaunch(global_runtime_config) ? (npes == 1)
                                                                   : true);
  global_runtime_config.num_control_tasks = 2;
  global_runtime_config.completion_queue_last_ready_data_id = nullptr;
  global_runtime_config.completion_queue_next_free_data_id = nullptr;
  global_runtime_config.data_last_enqueued_iteration = nullptr;
  global_runtime_config.data_last_executed_iteration = nullptr;
  global_runtime_config.data_to_task_id = nullptr;
  global_runtime_config.worker_owner_scheduler = nullptr;
  global_runtime_config.resident_owner_worker = nullptr;
  global_runtime_config.worker_streaming_resident_offsets = nullptr;
  global_runtime_config.worker_streaming_resident_ids = nullptr;
  global_runtime_config.resident_ready_data_head = nullptr;
  global_runtime_config.resident_ready_data_tail = nullptr;
  global_runtime_config.data_ready_next = nullptr;
  global_runtime_config.resident_ready_queue_offsets = nullptr;
  global_runtime_config.resident_ready_head_positions = nullptr;
  global_runtime_config.resident_ready_next_free_positions = nullptr;
  global_runtime_config.resident_ready_tail_positions = nullptr;
  global_runtime_config.resident_ready_queue_storage = nullptr;
  global_runtime_config.resident_active_workers = nullptr;
  global_runtime_config.resident_completed_data = nullptr;
  global_runtime_config.completed_data_this_iteration_count = nullptr;
  global_runtime_config.completed_streaming_data_count = nullptr;
  global_runtime_config.first_data_ids = nullptr;
  global_runtime_config.completion_queues = nullptr;

  std::vector<FullTaskDesc> all_fulltasks;
  std::vector<EventDesc> all_events;
  std::vector<TaskId> first_tasks;
  std::vector<ResidentTaskDesc> resident_tasks;
  std::vector<FullDataDesc> all_fulldata;
  std::vector<DataEdgeDesc> data_edges;
  std::vector<DataId> first_data_ids;
  std::vector<DataId> task_to_data_id;
  _init_persistent_kernel(all_fulltasks,
                          all_events,
                          first_tasks,
                          resident_tasks,
                          all_fulldata,
                          data_edges,
                          first_data_ids,
                          task_to_data_id,
                          npes,
                          mype);
  std::vector<TaskDesc> all_tasks;
  for (auto const &ft : all_fulltasks) {
    TaskDesc task_desc(ft);
    // if (ft.task_type == TASK_PAGED_ATTENTION_SPLIT_KV_SM100 || ft.task_type
    // == TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100) {
    //   printf("ft.kv_idx %d\n", ft.kv_idx);
    //   printf("ft.merge_task_offset %d\n", ft.merge_task_offset);
    // }
    all_tasks.push_back(task_desc);
  }
  std::vector<DataDesc> all_data;
  all_data.reserve(all_fulldata.size());
  for (auto const &fd : all_fulldata) {
    all_data.emplace_back(fd);
  }
  std::vector<TaskId> data_to_task_id(all_data.size(), TASK_INVALID_ID);
  for (TaskId task_id = 0; task_id < task_to_data_id.size(); task_id++) {
    DataId data_id = task_to_data_id[task_id];
    if (data_id == DATA_INVALID_ID) {
      continue;
    }
    assert(data_id < data_to_task_id.size());
    assert(data_to_task_id[data_id] == TASK_INVALID_ID);
    data_to_task_id[data_id] = task_id;
  }
  std::vector<uint32_t> host_data_edge_offsets(all_data.size() + 1, 0);
  for (auto const &edge : data_edges) {
    if (edge.src_data_id == DATA_INVALID_ID ||
        edge.src_data_id >= all_data.size() ||
        edge.dst_data_id == DATA_INVALID_ID ||
        edge.dst_data_id >= all_data.size()) {
      continue;
    }
    host_data_edge_offsets[edge.src_data_id + 1] += 1;
  }
  for (size_t i = 1; i < host_data_edge_offsets.size(); i++) {
    host_data_edge_offsets[i] += host_data_edge_offsets[i - 1];
  }
  std::vector<DataId> host_data_edge_targets(data_edges.size(), DATA_INVALID_ID);
  std::vector<uint32_t> host_edge_cursor = host_data_edge_offsets;
  for (auto const &edge : data_edges) {
    if (edge.src_data_id == DATA_INVALID_ID ||
        edge.src_data_id >= all_data.size() ||
        edge.dst_data_id == DATA_INVALID_ID ||
        edge.dst_data_id >= all_data.size()) {
      continue;
    }
    uint32_t cursor = host_edge_cursor[edge.src_data_id]++;
    host_data_edge_targets[cursor] = edge.dst_data_id;
  }
  int num_terminal_data = 0;
  for (size_t data_id = 0; data_id < all_data.size(); data_id++) {
    bool const no_local_successors =
        host_data_edge_offsets[data_id] == host_data_edge_offsets[data_id + 1];
    bool const remote_trigger_only =
        (resident_uses_hybrid_prelaunch(global_runtime_config) ||
         resident_uses_streaming(global_runtime_config)) &&
        all_data[data_id].trigger_event != EVENT_INVALID_ID &&
        is_nvshmem_event(all_data[data_id].trigger_event);
    if (streaming_uses_legacy_base(global_runtime_config)) {
      continue;
    }
    if (no_local_successors && !remote_trigger_only) {
      num_terminal_data++;
    }
  }
  global_runtime_config.num_resident_tasks = (int)resident_tasks.size();
  global_runtime_config.num_data = (int)all_data.size();
  global_runtime_config.num_data_edges = (int)data_edges.size();
  global_runtime_config.num_first_data_ids = (int)first_data_ids.size();
  global_runtime_config.num_terminal_data = num_terminal_data;

  // Initialize worker queue last task id
  // Each worker now maintains a local and a remote worker queue
  global_runtime_config.worker_queue_next_free_task_id =
      gpu_malloc<unsigned long long int>((num_workers * 2) *
                                         sizeof(unsigned long long int));
  global_runtime_config.worker_queue_last_ready_task_id =
      gpu_malloc<unsigned long long int>((num_workers * 2) *
                                         sizeof(unsigned long long int));
  // std::vector<unsigned long long int> host_worker_queue_last_task_id;
  // for (int i = 0; i < 2 * num_workers; i++) {
  //   host_worker_queue_last_task_id.push_back(0);
  // }
  // cudaMemcpy(global_runtime_config.worker_queue_last_ready_task_id,
  //            host_worker_queue_last_task_id.data(),
  //            (num_workers * 2) * sizeof(unsigned long long int),
  //            cudaMemcpyHostToDevice);
  //  Initialize scheduler queue last event id
  //  We maintain one extra scheduler queue for the global scheduler
  global_runtime_config.sched_queue_last_ready_event_id =
      gpu_malloc<unsigned long long int>((num_schedulers + 1) *
                                         sizeof(unsigned long long int));
  global_runtime_config.sched_queue_next_free_event_id =
      gpu_malloc<unsigned long long int>((num_schedulers + 1) *
                                         sizeof(unsigned long long int));

  // std::vector<unsigned long long int> host_sched_queue_last_event_id;
  // for (int i = 0; i < (num_schedulers + 1); i++) {
  //   host_sched_queue_last_event_id.push_back(0);
  // }
  // cudaMemcpy(global_runtime_config.sched_queue_last_ready_event_id,
  //            host_sched_queue_last_event_id.data(),
  //            (num_schedulers + 1) * sizeof(unsigned long long int),
  //            cudaMemcpyHostToDevice);
  // cudaMemcpy(global_runtime_config.sched_queue_next_free_event_id,
  //            host_sched_queue_last_event_id.data(),
  //            (num_schedulers + 1) * sizeof(unsigned long long int),
  //            cudaMemcpyHostToDevice);
  //  Initialize all event counters
  global_runtime_config.all_event_counters =
      gpu_malloc<EventCounter>(all_events.size() * sizeof(EventCounter));
  global_runtime_config.all_event_num_triggers =
      gpu_malloc<int>(all_events.size() * sizeof(int));
  std::vector<int> host_all_event_counters;
  for (size_t i = 0; i < all_events.size(); i++) {
    host_all_event_counters.push_back(all_events.at(i).num_triggers);
  }
  cudaMemcpy(global_runtime_config.all_event_num_triggers,
             host_all_event_counters.data(),
             all_events.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  // cudaMemset(global_runtime_config.all_event_counters,
  //            0,
  //            all_events.size() * sizeof(EventCounter));
  //  Initialize all tasks
  global_runtime_config.all_tasks =
      gpu_malloc<TaskDesc>(all_tasks.size() * sizeof(TaskDesc));
  cudaMemcpy(global_runtime_config.all_tasks,
             all_tasks.data(),
             all_tasks.size() * sizeof(TaskDesc),
             cudaMemcpyHostToDevice);
  // Initialize all events
  global_runtime_config.num_events = (int)all_events.size();
  global_runtime_config.begin_event_index = 1;
  global_runtime_config.end_event_index = (int)all_events.size() - 1;
  for (size_t event_idx = 0; event_idx < all_events.size(); event_idx++) {
    if (all_events[event_idx].event_type == EVENT_LAUNCH_DEPENDENT_TASKS) {
      global_runtime_config.begin_event_index = (int)event_idx;
    } else if (all_events[event_idx].event_type == EVENT_END_OF_TASK_GRAPH) {
      global_runtime_config.end_event_index = (int)event_idx;
    }
  }
  global_runtime_config.all_events =
      gpu_malloc<EventDesc>(all_events.size() * sizeof(EventDesc));
  cudaMemcpy(global_runtime_config.all_events,
             all_events.data(),
             all_events.size() * sizeof(EventDesc),
             cudaMemcpyHostToDevice);
  global_runtime_config.resident_tasks = gpu_malloc<ResidentTaskDesc>(
      resident_tasks.size() * sizeof(ResidentTaskDesc));
  cudaMemcpy(global_runtime_config.resident_tasks,
             resident_tasks.data(),
             resident_tasks.size() * sizeof(ResidentTaskDesc),
             cudaMemcpyHostToDevice);
  global_runtime_config.all_data =
      gpu_malloc<DataDesc>(all_data.size() * sizeof(DataDesc));
  cudaMemcpy(global_runtime_config.all_data,
             all_data.data(),
             all_data.size() * sizeof(DataDesc),
             cudaMemcpyHostToDevice);
  global_runtime_config.data_edge_offsets = gpu_malloc<uint32_t>(
      host_data_edge_offsets.size() * sizeof(uint32_t));
  cudaMemcpy(global_runtime_config.data_edge_offsets,
             host_data_edge_offsets.data(),
             host_data_edge_offsets.size() * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  global_runtime_config.data_edge_targets =
      gpu_malloc<DataId>(host_data_edge_targets.size() * sizeof(DataId));
  cudaMemcpy(global_runtime_config.data_edge_targets,
             host_data_edge_targets.data(),
             host_data_edge_targets.size() * sizeof(DataId),
             cudaMemcpyHostToDevice);
  std::vector<uint32_t> host_initial_predecessor_counts;
  host_initial_predecessor_counts.reserve(all_data.size());
  for (auto const &fd : all_fulldata) {
    host_initial_predecessor_counts.push_back(fd.initial_predecessor_count);
  }
  global_runtime_config.data_initial_predecessor_counts = gpu_malloc<uint32_t>(
      host_initial_predecessor_counts.size() * sizeof(uint32_t));
  global_runtime_config.data_pending_predecessor_counts = gpu_malloc<uint32_t>(
      host_initial_predecessor_counts.size() * sizeof(uint32_t));
  global_runtime_config.task_to_data_id =
      gpu_malloc<DataId>(task_to_data_id.size() * sizeof(DataId));
  cudaMemcpy(global_runtime_config.task_to_data_id,
             task_to_data_id.data(),
             task_to_data_id.size() * sizeof(DataId),
             cudaMemcpyHostToDevice);
  global_runtime_config.data_to_task_id =
      gpu_malloc<TaskId>(data_to_task_id.size() * sizeof(TaskId));
  cudaMemcpy(global_runtime_config.data_to_task_id,
             data_to_task_id.data(),
             data_to_task_id.size() * sizeof(TaskId),
             cudaMemcpyHostToDevice);
  cudaMemcpy(global_runtime_config.data_initial_predecessor_counts,
             host_initial_predecessor_counts.data(),
             host_initial_predecessor_counts.size() * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  global_runtime_config.completed_terminal_data_count =
      gpu_malloc<uint32_t>(sizeof(uint32_t));
  global_runtime_config.completed_streaming_data_count =
      gpu_malloc<uint32_t>(sizeof(uint32_t));
  global_runtime_config.completed_data_this_iteration_count =
      gpu_malloc<uint32_t>(sizeof(uint32_t));
  global_runtime_config.current_iteration = gpu_malloc<uint32_t>(sizeof(uint32_t));
  global_runtime_config.first_data_ids =
      gpu_malloc<DataId>(first_data_ids.size() * sizeof(DataId));
  cudaMemcpy(global_runtime_config.first_data_ids,
             first_data_ids.data(),
             first_data_ids.size() * sizeof(DataId),
             cudaMemcpyHostToDevice);
  if (resident_uses_streaming(global_runtime_config)) {
    std::vector<TaskId> host_resident_canonical_task_id(
        resident_tasks.size(), TASK_INVALID_ID);
    for (TaskId task_id = 0; task_id < task_to_data_id.size(); task_id++) {
      DataId data_id = task_to_data_id[task_id];
      if (data_id == DATA_INVALID_ID) {
        continue;
      }
      ResidentTaskId resident_task_id = all_data[data_id].resident_task_id;
      if (resident_task_id >= host_resident_canonical_task_id.size()) {
        continue;
      }
      if (host_resident_canonical_task_id[resident_task_id] ==
          TASK_INVALID_ID) {
        host_resident_canonical_task_id[resident_task_id] = task_id;
      }
    }
    std::vector<int> host_resident_owner_worker(resident_tasks.size(), 0);
    std::vector<uint32_t> host_worker_streaming_resident_counts(
        num_workers + 1, 0u);
    for (size_t resident_task_id = 0; resident_task_id < resident_tasks.size();
         resident_task_id++) {
      if (!resident_task_is_streaming(resident_tasks[resident_task_id])) {
        continue;
      }
      TaskId canonical_task_id =
          host_resident_canonical_task_id[resident_task_id];
      if (canonical_task_id == TASK_INVALID_ID) {
        canonical_task_id = static_cast<TaskId>(resident_task_id);
      }
      host_resident_owner_worker[resident_task_id] =
          static_cast<int>(canonical_task_id % std::max(1, num_workers));
      host_worker_streaming_resident_counts[host_resident_owner_worker
                                                [resident_task_id] +
                                            1] += 1u;
    }
    global_runtime_config.resident_owner_worker =
        gpu_malloc<int>(host_resident_owner_worker.size() * sizeof(int));
    cudaMemcpy(global_runtime_config.resident_owner_worker,
               host_resident_owner_worker.data(),
               host_resident_owner_worker.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    for (int worker_id = 1; worker_id <= num_workers; worker_id++) {
      host_worker_streaming_resident_counts[worker_id] +=
          host_worker_streaming_resident_counts[worker_id - 1];
    }
    std::vector<ResidentTaskId> host_worker_streaming_resident_ids(
        host_worker_streaming_resident_counts.back(),
        RESIDENT_TASK_INVALID_ID);
    std::vector<uint32_t> host_worker_streaming_cursor =
        host_worker_streaming_resident_counts;
    for (size_t resident_task_id = 0; resident_task_id < resident_tasks.size();
         resident_task_id++) {
      if (!resident_task_is_streaming(resident_tasks[resident_task_id])) {
        continue;
      }
      int owner_worker = host_resident_owner_worker[resident_task_id];
      uint32_t cursor = host_worker_streaming_cursor[owner_worker]++;
      host_worker_streaming_resident_ids[cursor] =
          static_cast<ResidentTaskId>(resident_task_id);
    }
    global_runtime_config.worker_streaming_resident_offsets =
        gpu_malloc<uint32_t>(host_worker_streaming_resident_counts.size() *
                             sizeof(uint32_t));
    cudaMemcpy(global_runtime_config.worker_streaming_resident_offsets,
               host_worker_streaming_resident_counts.data(),
               host_worker_streaming_resident_counts.size() * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    global_runtime_config.worker_streaming_resident_ids =
        gpu_malloc<ResidentTaskId>(std::max<size_t>(
            1, host_worker_streaming_resident_ids.size()) *
                                   sizeof(ResidentTaskId));
    if (!host_worker_streaming_resident_ids.empty()) {
      cudaMemcpy(global_runtime_config.worker_streaming_resident_ids,
                 host_worker_streaming_resident_ids.data(),
                 host_worker_streaming_resident_ids.size() *
                     sizeof(ResidentTaskId),
                 cudaMemcpyHostToDevice);
    }

    std::vector<uint32_t> host_streaming_queue_offsets(
        resident_tasks.size() + 1, 0);
    for (size_t resident_task_id = 0; resident_task_id < resident_tasks.size();
         resident_task_id++) {
      host_streaming_queue_offsets[resident_task_id + 1] =
          host_streaming_queue_offsets[resident_task_id] +
          (resident_task_is_streaming(resident_tasks[resident_task_id])
               ? static_cast<uint32_t>(
                     resident_tasks[resident_task_id].total_data_count)
               : 0u);
    }
    global_runtime_config.resident_ready_queue_offsets = gpu_malloc<uint32_t>(
        host_streaming_queue_offsets.size() * sizeof(uint32_t));
    cudaMemcpy(global_runtime_config.resident_ready_queue_offsets,
               host_streaming_queue_offsets.data(),
               host_streaming_queue_offsets.size() * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    global_runtime_config.resident_ready_head_positions = gpu_malloc<uint32_t>(
        resident_tasks.size() * sizeof(uint32_t));
    global_runtime_config.resident_ready_next_free_positions =
        gpu_malloc<uint32_t>(resident_tasks.size() * sizeof(uint32_t));
    global_runtime_config.resident_ready_tail_positions = gpu_malloc<uint32_t>(
        resident_tasks.size() * sizeof(uint32_t));
    global_runtime_config.resident_ready_queue_storage =
        gpu_malloc<DataId>(std::max<size_t>(
            1, host_streaming_queue_offsets.back()) * sizeof(DataId));
  }
  {
    std::vector<int> host_worker_owner_scheduler(num_workers, 0);
    int owner_scheduler_count = std::max(1, num_local_schedulers);
    for (int sched_id = 0; sched_id < owner_scheduler_count; sched_id++) {
      int workers_per_scheduler = num_workers / owner_scheduler_count;
      int remainder = num_workers % owner_scheduler_count;
      int my_first_worker = 0;
      int my_last_worker = 0;
      if (sched_id < remainder) {
        my_first_worker = (workers_per_scheduler + 1) * sched_id;
        my_last_worker = my_first_worker + workers_per_scheduler + 1;
      } else {
        my_first_worker = workers_per_scheduler * sched_id + remainder;
        my_last_worker = my_first_worker + workers_per_scheduler;
      }
      for (int worker_id = my_first_worker; worker_id < my_last_worker;
           worker_id++) {
        host_worker_owner_scheduler[worker_id] = sched_id;
      }
    }
    global_runtime_config.worker_owner_scheduler =
        gpu_malloc<int>(host_worker_owner_scheduler.size() * sizeof(int));
    cudaMemcpy(global_runtime_config.worker_owner_scheduler,
               host_worker_owner_scheduler.data(),
               host_worker_owner_scheduler.size() * sizeof(int),
               cudaMemcpyHostToDevice);
  }
  // Initialize worker queues
  {
    std::vector<TaskId *> host_worker_queues;
    for (int i = 0; i < (num_workers * 2); i++) {
      TaskId *worker_queue = gpu_malloc<TaskId>(
          global_runtime_config.per_worker_queue_len * sizeof(TaskId));
      host_worker_queues.push_back(worker_queue);
    }
    global_runtime_config.worker_queues =
        gpu_malloc<TaskId *>((num_workers * 2) * sizeof(TaskId *));
    cudaMemcpy(global_runtime_config.worker_queues,
               host_worker_queues.data(),
               (num_workers * 2) * sizeof(TaskId *),
               cudaMemcpyHostToDevice);
  }
  // Initialize scheduler queues
  {
    std::vector<EventId *> host_sched_queues;
    for (int i = 0; i < (num_schedulers + 1); i++) {
      EventId *sched_queue = gpu_malloc<EventId>(
          global_runtime_config.per_sched_queue_len * sizeof(EventId));
      host_sched_queues.push_back(sched_queue);
    }
    global_runtime_config.sched_queues =
        gpu_malloc<EventId *>((num_schedulers + 1) * sizeof(EventId *));
    cudaMemcpy(global_runtime_config.sched_queues,
               host_sched_queues.data(),
               (num_schedulers + 1) * sizeof(EventId *),
               cudaMemcpyHostToDevice);
  }
  {
    if (!resident_uses_streaming(global_runtime_config)) {
      int num_completion_queues = std::max(1, num_local_schedulers);
      global_runtime_config.completion_queue_last_ready_data_id =
          gpu_malloc<unsigned long long int>(num_completion_queues *
                                             sizeof(unsigned long long int));
      global_runtime_config.completion_queue_next_free_data_id =
          gpu_malloc<unsigned long long int>(num_completion_queues *
                                             sizeof(unsigned long long int));
      std::vector<TaskId *> host_completion_queues;
      for (int i = 0; i < num_completion_queues; i++) {
        TaskId *completion_queue = gpu_malloc<TaskId>(
            global_runtime_config.per_completion_queue_len * sizeof(TaskId));
        host_completion_queues.push_back(completion_queue);
      }
      global_runtime_config.completion_queues =
          gpu_malloc<TaskId *>(num_completion_queues * sizeof(TaskId *));
      cudaMemcpy(global_runtime_config.completion_queues,
                 host_completion_queues.data(),
                 num_completion_queues * sizeof(TaskId *),
                 cudaMemcpyHostToDevice);
    }
  }
  // Initialize first tasks
  {
    global_runtime_config.first_tasks =
        gpu_malloc<TaskId>(first_tasks.size() * sizeof(TaskId));
    cudaMemcpy(global_runtime_config.first_tasks,
               first_tasks.data(),
               first_tasks.size() * sizeof(TaskId),
               cudaMemcpyHostToDevice);
  }

  // Set configuration for kernels
#if MIRAGE_COMPILE_RESIDENT_SCHEDULER_DISPATCH
  cudaFuncSetAttribute(worker_kernel_scheduler_dispatch,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       MAX_DYNAMIC_SHARED_MEMORY_SIZE);
#endif
#if MIRAGE_COMPILE_RESIDENT_HYBRID_PRELAUNCH
  cudaFuncSetAttribute(worker_kernel_hybrid_prelaunch,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       MAX_DYNAMIC_SHARED_MEMORY_SIZE);
#endif
#if MIRAGE_COMPILE_STREAMING_RUNTIME
  cudaFuncSetAttribute(worker_kernel_streaming,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       MAX_DYNAMIC_SHARED_MEMORY_SIZE);
#endif
#if MIRAGE_COMPILE_RESIDENT_SCHEDULER_DISPATCH
  cudaFuncSetAttribute(scheduler_kernel_scheduler_dispatch,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       1024);
#endif
#if MIRAGE_COMPILE_RESIDENT_HYBRID_PRELAUNCH
  cudaFuncSetAttribute(scheduler_kernel_hybrid_prelaunch,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       1024);
#endif
#if MIRAGE_COMPILE_STREAMING_RUNTIME
  cudaFuncSetAttribute(scheduler_kernel_streaming,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       1024);
#endif
#if MIRAGE_COMPILE_RESIDENT_SCHEDULER_DISPATCH
  cudaFuncSetAttribute(persistent_kernel_scheduler_dispatch,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       MAX_DYNAMIC_SHARED_MEMORY_SIZE);
#endif
#if MIRAGE_COMPILE_RESIDENT_HYBRID_PRELAUNCH
  cudaFuncSetAttribute(persistent_kernel_hybrid_prelaunch,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       MAX_DYNAMIC_SHARED_MEMORY_SIZE);
#endif
  // Create worker and scheduler streams
  cudaStreamCreateWithFlags(&global_runtime_config.worker_stream,
                            cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&global_runtime_config.scheduler_stream,
                            cudaStreamNonBlocking);
  // Create events
  cudaEventCreateWithFlags(&global_runtime_config.prepare_done_event,
                           cudaEventDisableTiming);
  cudaEventCreateWithFlags(&global_runtime_config.worker_done_event,
                           cudaEventDisableTiming);
  cudaEventCreateWithFlags(&global_runtime_config.scheduler_done_event,
                           cudaEventDisableTiming);

  init_request_resources();
#ifdef USE_NVSHMEM
  // Add a global barrier for all init_kernel to complete
  nvshmem_barrier_all();
#endif
}

// Entry point for C/C++
// TODO: change launch config
extern "C" void launch_persistent_kernel(cudaStream_t default_stream) {
  // int device;
  // cudaGetDevice(&device);
  // int sm_count;
  // cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  //  Prepare next persistent kernel by resetting queue pointers
  {
    prepare_kernel<<<dim3(global_runtime_config.num_workers, 1, 1),
                     dim3(128, 1, 1),
                     0 /*smem*/,
                     default_stream>>>(global_runtime_config,
                                       global_runtime_config.end_event_index);
    // cudaStreamSynchronize(NULL);
    cudaEventRecord(global_runtime_config.prepare_done_event, default_stream);
    // cudaDeviceSynchronize();
#ifdef USE_NVSHMEM
    nvshmem_barrier_all();
#endif
  }
  int num_schedulers = global_runtime_config.num_local_schedulers +
                       global_runtime_config.num_remote_schedulers;
  if (global_runtime_config.split_worker_scheduler) {
    printf("worker kernel & scheduler kernel\n");
    printf("smem size: %d\n", MAX_DYNAMIC_SHARED_MEMORY_SIZE);

    cudaStreamWaitEvent(global_runtime_config.worker_stream,
                        global_runtime_config.prepare_done_event,
                        0);
    cudaStreamWaitEvent(global_runtime_config.scheduler_stream,
                        global_runtime_config.prepare_done_event,
                        0);

    // The split kernel does not support NVSHMEM because
    // nvshmemx_collective_launch launches kernels sequentially, which blocks
    // the interaction between the worker kernel and the scheduler kernel
    if (resident_uses_streaming(global_runtime_config)) {
#if MIRAGE_COMPILE_STREAMING_RUNTIME
      worker_kernel_streaming<<<dim3(global_runtime_config.num_workers, 1, 1),
                                dim3(WORKER_NUM_THREADS, 1, 1),
                                MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/,
                                global_runtime_config.worker_stream>>>(
          global_runtime_config);

      scheduler_kernel_streaming<<<dim3(global_runtime_config.num_local_schedulers,
                                         1,
                                         1),
                                   dim3(32, 1, 1),
                                   0 /*smem*/,
                                   global_runtime_config.scheduler_stream>>>(
          global_runtime_config);
#else
      assert(false && "streaming kernels are not compiled into this TU");
#endif
    } else if (resident_uses_hybrid_prelaunch(global_runtime_config)) {
#if MIRAGE_COMPILE_RESIDENT_HYBRID_PRELAUNCH
      worker_kernel_hybrid_prelaunch<<<dim3(global_runtime_config.num_workers, 1, 1),
                                       dim3(WORKER_NUM_THREADS, 1, 1),
                                       MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/,
                                       global_runtime_config.worker_stream>>>(
          global_runtime_config);

      scheduler_kernel_hybrid_prelaunch<<<dim3(global_runtime_config.num_local_schedulers,
                                                1,
                                                1),
                                          dim3(32, 1, 1),
                                          0 /*smem*/,
                                          global_runtime_config.scheduler_stream>>>(
          global_runtime_config);
#else
      assert(false && "hybrid resident kernels are not compiled into this TU");
#endif
    } else {
#if MIRAGE_COMPILE_RESIDENT_SCHEDULER_DISPATCH
      worker_kernel_scheduler_dispatch<<<dim3(global_runtime_config.num_workers, 1, 1),
                                         dim3(WORKER_NUM_THREADS, 1, 1),
                                         MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/,
                                         global_runtime_config.worker_stream>>>(
          global_runtime_config);

      scheduler_kernel_scheduler_dispatch<<<dim3(global_runtime_config.num_local_schedulers,
                                                  1,
                                                  1),
                                            dim3(32, 1, 1),
                                            0 /*smem*/,
                                            global_runtime_config.scheduler_stream>>>(
          global_runtime_config);
#else
      assert(false &&
             "scheduler-dispatch resident kernels are not compiled into this TU");
#endif
    }

    cudaEventRecord(global_runtime_config.worker_done_event,
                    global_runtime_config.worker_stream);
    cudaEventRecord(global_runtime_config.scheduler_done_event,
                    global_runtime_config.scheduler_stream);

    cudaStreamWaitEvent(
        default_stream, global_runtime_config.worker_done_event, 0);
    cudaStreamWaitEvent(
        default_stream, global_runtime_config.scheduler_done_event, 0);
    printf("Finished Launching Persistent Kernel (Async)\n");

    char const *debug_progress_env =
        std::getenv("MIRAGE_STREAMING_DEBUG_PROGRESS");
    if (resident_uses_streaming(global_runtime_config) &&
        debug_progress_env != nullptr && std::strcmp(debug_progress_env, "0") != 0) {
      std::vector<uint32_t> host_ready_head(
          std::max(1, global_runtime_config.num_resident_tasks), 0u);
      std::vector<uint32_t> host_ready_tail(
          std::max(1, global_runtime_config.num_resident_tasks), 0u);
      std::vector<uint32_t> host_worker_resident_offsets(
          std::max(1, global_runtime_config.num_workers + 1), 0u);
      CUDA_CHECK(cudaMemcpy(host_worker_resident_offsets.data(),
                            global_runtime_config.worker_streaming_resident_offsets,
                            (global_runtime_config.num_workers + 1) *
                                sizeof(uint32_t),
                            cudaMemcpyDeviceToHost));
      std::vector<ResidentTaskId> host_worker_resident_ids(
          std::max<uint32_t>(1, host_worker_resident_offsets.back()),
          RESIDENT_TASK_INVALID_ID);
      if (host_worker_resident_offsets.back() > 0) {
        CUDA_CHECK(cudaMemcpy(host_worker_resident_ids.data(),
                              global_runtime_config.worker_streaming_resident_ids,
                              host_worker_resident_offsets.back() *
                                  sizeof(ResidentTaskId),
                              cudaMemcpyDeviceToHost));
      }
      int debug_step = 0;
      while (true) {
        cudaError_t worker_status =
            cudaEventQuery(global_runtime_config.worker_done_event);
        cudaError_t scheduler_status =
            cudaEventQuery(global_runtime_config.scheduler_done_event);
        if (worker_status == cudaSuccess && scheduler_status == cudaSuccess) {
          break;
        }
        if (worker_status != cudaSuccess && worker_status != cudaErrorNotReady) {
          CUDA_CHECK(worker_status);
        }
        if (scheduler_status != cudaSuccess &&
            scheduler_status != cudaErrorNotReady) {
          CUDA_CHECK(scheduler_status);
        }
        uint32_t host_completed_terminal = 0;
        uint32_t host_completed_streaming = 0;
        uint32_t host_current_iteration = 0;
        CUDA_CHECK(cudaMemcpy(&host_completed_terminal,
                              global_runtime_config.completed_terminal_data_count,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&host_completed_streaming,
                              global_runtime_config.completed_streaming_data_count,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&host_current_iteration,
                              global_runtime_config.current_iteration,
                              sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_ready_head.data(),
                              global_runtime_config.resident_ready_head_positions,
                              global_runtime_config.num_resident_tasks *
                                  sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_ready_tail.data(),
                              global_runtime_config.resident_ready_tail_positions,
                              global_runtime_config.num_resident_tasks *
                                  sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        uint32_t active_residents = 0;
        uint32_t max_worker_active = 0;
        uint32_t workers_with_active = 0;
        for (int resident_task_id = 0;
             resident_task_id < global_runtime_config.num_resident_tasks;
             resident_task_id++) {
          active_residents +=
              (host_ready_head[resident_task_id] < host_ready_tail[resident_task_id]);
        }
        for (int worker_id = 0; worker_id < global_runtime_config.num_workers;
             worker_id++) {
          uint32_t worker_active = 0;
          for (uint32_t cursor = host_worker_resident_offsets[worker_id];
               cursor < host_worker_resident_offsets[worker_id + 1];
               cursor++) {
            ResidentTaskId resident_task_id = host_worker_resident_ids[cursor];
            if (resident_task_id == RESIDENT_TASK_INVALID_ID) {
              continue;
            }
            worker_active +=
                (host_ready_head[resident_task_id] <
                 host_ready_tail[resident_task_id]);
          }
          max_worker_active = max(max_worker_active, worker_active);
          workers_with_active += (worker_active > 0);
        }
        printf("[streaming-debug] tick=%d iteration=%u terminal=%u/%d streaming_completed=%u active_residents=%u workers_with_active=%u max_worker_active=%u\n",
               debug_step++,
               host_current_iteration,
               host_completed_terminal,
               global_runtime_config.num_terminal_data,
               host_completed_streaming,
               active_residents,
               workers_with_active,
               max_worker_active);
        usleep(500000);
      }
    }
  } else {
    printf("a single persistent kernel\n");
    int num_sms_to_use = global_runtime_config.num_workers + num_schedulers / 4;
#ifdef USE_NVSHMEM
    void *args[] = {&global_runtime_config};
    void const *kernel_ptr =
        resident_uses_hybrid_prelaunch(global_runtime_config)
#if MIRAGE_COMPILE_RESIDENT_HYBRID_PRELAUNCH
            ? (void const *)persistent_kernel_hybrid_prelaunch
#else
            ? nullptr
#endif
#if MIRAGE_COMPILE_RESIDENT_SCHEDULER_DISPATCH
            : (void const *)persistent_kernel_scheduler_dispatch;
#else
            : nullptr;
#endif
    assert(kernel_ptr != nullptr);
    nvshmemx_collective_launch(kernel_ptr,
                               dim3(num_sms_to_use, 1, 1),
                               dim3(SINGLE_KERNEL_NUM_THREADS, 1, 1),
                               args,
                               MAX_DYNAMIC_SHARED_MEMORY_SIZE /*sharedmem*/,
                               0 /*stream*/);
#else
    if (resident_uses_streaming(global_runtime_config)) {
#if MIRAGE_COMPILE_STREAMING_RUNTIME
      worker_kernel_streaming<<<dim3(global_runtime_config.num_workers, 1, 1),
                                dim3(WORKER_NUM_THREADS, 1, 1),
                                MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/,
                                global_runtime_config.worker_stream>>>(
          global_runtime_config);
      scheduler_kernel_streaming<<<dim3(global_runtime_config.num_local_schedulers,
                                         1,
                                         1),
                                   dim3(32, 1, 1),
                                   0 /*smem*/,
                                   global_runtime_config.scheduler_stream>>>(
          global_runtime_config);
      cudaEventRecord(global_runtime_config.worker_done_event,
                      global_runtime_config.worker_stream);
      cudaEventRecord(global_runtime_config.scheduler_done_event,
                      global_runtime_config.scheduler_stream);
      cudaStreamWaitEvent(
          default_stream, global_runtime_config.worker_done_event, 0);
      cudaStreamWaitEvent(
          default_stream, global_runtime_config.scheduler_done_event, 0);
      return;
#else
      assert(false && "streaming kernels are not compiled into this TU");
#endif
    } else if (resident_uses_hybrid_prelaunch(global_runtime_config)) {
#if MIRAGE_COMPILE_RESIDENT_HYBRID_PRELAUNCH
      persistent_kernel_hybrid_prelaunch<<<dim3(num_sms_to_use, 1, 1),
                                           dim3(SINGLE_KERNEL_NUM_THREADS, 1, 1),
                                           MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/>>>(
          global_runtime_config);
#else
      assert(false && "hybrid resident kernel is not compiled into this TU");
#endif
    } else {
#if MIRAGE_COMPILE_RESIDENT_SCHEDULER_DISPATCH
      persistent_kernel_scheduler_dispatch<<<dim3(num_sms_to_use, 1, 1),
                                             dim3(SINGLE_KERNEL_NUM_THREADS, 1, 1),
                                             MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/>>>(
          global_runtime_config);
#else
      assert(false &&
             "scheduler-dispatch resident kernel is not compiled into this TU");
#endif
    }
#endif
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    printf("Finished Launch Persistent Kernel\n");
  }
}

extern "C" void finalize_persistent_kernel() {
  gpu_free(global_runtime_config.worker_queue_next_free_task_id);
  gpu_free(global_runtime_config.worker_queue_last_ready_task_id);
  gpu_free(global_runtime_config.sched_queue_last_ready_event_id);
  gpu_free(global_runtime_config.sched_queue_next_free_event_id);
  gpu_free(global_runtime_config.completion_queue_last_ready_data_id);
  gpu_free(global_runtime_config.completion_queue_next_free_data_id);
  gpu_free(global_runtime_config.all_event_counters);
  gpu_free(global_runtime_config.all_event_num_triggers);
  gpu_free(global_runtime_config.resident_tasks);
  gpu_free(global_runtime_config.all_data);
  gpu_free(global_runtime_config.data_edge_offsets);
  gpu_free(global_runtime_config.data_edge_targets);
  gpu_free(global_runtime_config.data_initial_predecessor_counts);
  gpu_free(global_runtime_config.data_pending_predecessor_counts);
  gpu_free(global_runtime_config.task_to_data_id);
  gpu_free(global_runtime_config.data_to_task_id);
  gpu_free(global_runtime_config.worker_owner_scheduler);
  gpu_free(global_runtime_config.resident_owner_worker);
  gpu_free(global_runtime_config.worker_streaming_resident_offsets);
  gpu_free(global_runtime_config.worker_streaming_resident_ids);
  gpu_free(global_runtime_config.resident_ready_queue_offsets);
  gpu_free(global_runtime_config.resident_ready_head_positions);
  gpu_free(global_runtime_config.resident_ready_next_free_positions);
  gpu_free(global_runtime_config.resident_ready_tail_positions);
  gpu_free(global_runtime_config.resident_ready_queue_storage);
  gpu_free(global_runtime_config.completed_terminal_data_count);
  gpu_free(global_runtime_config.completed_streaming_data_count);
  gpu_free(global_runtime_config.completed_data_this_iteration_count);
  gpu_free(global_runtime_config.current_iteration);
  gpu_free(global_runtime_config.first_data_ids);
  gpu_free(global_runtime_config.resident_active_workers);
  gpu_free(global_runtime_config.resident_completed_data);
  gpu_free(global_runtime_config.all_tasks);
  gpu_free(global_runtime_config.all_events);
#if defined(MODE_OFFLINE) || defined(MODE_ONLINE)
  gpu_free(global_runtime_config.next_request_id);
  gpu_free(global_runtime_config.page_queue);
  gpu_free(global_runtime_config.page_queue_head);
  gpu_free(global_runtime_config.page_queue_tail);
#endif
  int num_workers = global_runtime_config.num_workers;
  std::vector<TaskId *> host_worker_queues(num_workers * 2);
  cudaMemcpy(host_worker_queues.data(),
             global_runtime_config.worker_queues,
             (num_workers * 2) * sizeof(TaskId *),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < 2 * num_workers; i++) {
    gpu_free(host_worker_queues[i]);
  }
  gpu_free(global_runtime_config.worker_queues);
  int num_schedulers = global_runtime_config.num_local_schedulers +
                       global_runtime_config.num_remote_schedulers;
  std::vector<EventId *> host_sched_queues(num_schedulers + 1);
  cudaMemcpy(host_sched_queues.data(),
             global_runtime_config.sched_queues,
             (num_schedulers + 1) * sizeof(EventId *),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < num_schedulers + 1; i++) {
    gpu_free(host_sched_queues[i]);
  }
  gpu_free(global_runtime_config.sched_queues);
  if (global_runtime_config.completion_queues != nullptr) {
    int num_completion_queues =
        std::max(1, global_runtime_config.num_local_schedulers);
    std::vector<TaskId *> host_completion_queues(num_completion_queues);
    cudaMemcpy(host_completion_queues.data(),
               global_runtime_config.completion_queues,
               num_completion_queues * sizeof(TaskId *),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_completion_queues; i++) {
      gpu_free(host_completion_queues[i]);
    }
  }
  gpu_free(global_runtime_config.completion_queues);
  gpu_free(global_runtime_config.first_tasks);
#ifdef USE_NVSHMEM
  nvshmem_barrier_all();
  nvshmem_finalize();
#endif
  // Free worker and scheduler streams
  cudaEventDestroy(global_runtime_config.prepare_done_event);
  cudaEventDestroy(global_runtime_config.worker_done_event);
  cudaEventDestroy(global_runtime_config.scheduler_done_event);
  cudaStreamDestroy(global_runtime_config.worker_stream);
  cudaStreamDestroy(global_runtime_config.scheduler_stream);
}
