#pragma once
// Included only when the demo adds -DMIRAGE_END_TRACE. The header row stores
// (record count, overflow); following rows have 12 uint64 words each.
// Same event layout as the existing resident-runtime trace hooks.
#ifndef END_TRACE_CAPACITY
#define END_TRACE_CAPACITY 8192
#endif
__device__ __forceinline__ unsigned long long end_trace_clock() {
  unsigned long long time;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time) :: "memory");
  return time;
}

// A scheduler decision can record the exact clock value used by its policy.
// Publication is unchanged; only the source of the timestamp differs.
__device__ __forceinline__ void end_trace_record_at(
    unsigned long long timestamp, int kind, unsigned long long epoch, unsigned long long task,
    unsigned long long data, int worker, unsigned long long aux) {
  unsigned int sm;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  auto *buffer = reinterpret_cast<unsigned long long *>(END_TRACE_BUFFER_ADDRESS);
  auto index = atomicAdd(buffer, 1ULL);
  if (index >= END_TRACE_CAPACITY) {
    atomicExch(buffer + 1, 1ULL);
    return;
  }
  auto *record = buffer + (index + 1) * 12;
  record[0] = timestamp;
  record[1] = kind;
  record[2] = epoch;
  record[3] = task;
  record[4] = data;
  record[5] = static_cast<unsigned long long>(worker);
  record[6] = aux;
  record[7] = sm;
  record[8] = blockIdx.x;
  record[9] = threadIdx.x;
  record[10] = 0;
  __threadfence();
  atomicExch(record + 11, 1ULL); // Payload is complete before publication.
}

__device__ __forceinline__ void end_trace_record(
    int kind, unsigned long long epoch, unsigned long long task,
    unsigned long long data, int worker, unsigned long long aux) {
  end_trace_record_at(end_trace_clock(), kind, epoch, task, data, worker, aux);
}

#define ETRACE(kind, epoch, task, data, worker, aux) \
  end_trace_record(kind, epoch, task, data, worker, aux)
#define ETRACE_AT(timestamp, kind, epoch, task, data, worker, aux) \
  end_trace_record_at(timestamp, kind, epoch, task, data, worker, aux)

// Older experiment launchers define a single task. New launchers may select
// several native terminal task IDs, without changing the runtime task graph.
#ifndef END_TRACE_DELAY_TASKS
#define END_TRACE_DELAY_TASKS END_TRACE_DELAY_TASK
#endif

__device__ __forceinline__ void end_trace_delay(
    unsigned long long epoch, unsigned long long task,
    unsigned long long data, int worker) {
  if (END_TRACE_DELAY_NS == 0) return;
  constexpr unsigned long long delayed_tasks[] = {END_TRACE_DELAY_TASKS};
  bool selected = false;
  #pragma unroll
  for (int i = 0; i < sizeof(delayed_tasks) / sizeof(delayed_tasks[0]); i++) {
    selected |= task == delayed_tasks[i];
  }
  if (!selected) return;
  ETRACE(6, epoch, task, data, worker, END_TRACE_DELAY_NS);
  auto start = end_trace_clock();
  while (end_trace_clock() - start < END_TRACE_DELAY_NS) __nanosleep(100);
  ETRACE(7, epoch, task, data, worker, END_TRACE_DELAY_NS);
}

#define ETRACE_DELAY(epoch, task, data, worker) end_trace_delay(epoch, task, data, worker)
