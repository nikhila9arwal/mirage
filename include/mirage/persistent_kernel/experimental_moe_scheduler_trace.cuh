#pragma once
// Optional, single-writer scheduler timeline. Only the controller's thread 0
// writes this buffer; the host reads it after the entire invocation completes.
// No per-record atomic, fence, or cross-worker synchronization is needed.
#ifdef NAS_MOE_SCHEDULER_TRACE
enum NasSchedulerTraceKind {
  NAS_SCHED_POLL_COMPLETION = 1,
  NAS_SCHED_HANDLE_COMPLETION = 2,
  NAS_SCHED_COMPLETE_LOGICAL = 3,
  NAS_SCHED_POLL_CONTROL = 4,
  NAS_SCHED_HANDLE_CONTROL = 5,
  NAS_SCHED_DISPATCH = 6,
  NAS_SCHED_BACKOFF = 7,
};

__device__ __forceinline__ unsigned long long nas_scheduler_trace_clock() {
  unsigned long long now;
  // The compiler memory barrier keeps memory work inside its timed scope.
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now) :: "memory");
  return now;
}

struct NasSchedulerTrace {
  unsigned long long *buffer;
  unsigned long long count = 0, loops = 0, start_ns;
  unsigned int sm;

  __device__ __forceinline__ NasSchedulerTrace() {
    buffer = reinterpret_cast<unsigned long long *>(NAS_SCHEDULER_TRACE_ADDRESS);
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    start_ns = nas_scheduler_trace_clock();
  }

  __device__ __forceinline__ void append(
      unsigned long long start, unsigned long long end, unsigned int kind,
      unsigned long long result, unsigned long long subject,
      unsigned long long epoch, unsigned long long batch_sequence) {
    auto index = count++;
    // Keep counting on overflow, but never overwrite an earlier record.
    if (index >= NAS_SCHEDULER_TRACE_CAPACITY) return;
    auto *r = buffer + (index + 2) * 8;
    r[0] = start;
    r[1] = end;
    r[2] = kind;
    r[3] = result;
    r[4] = subject; // DataId for completion; raw EventId for control.
    r[5] = epoch;
    r[6] = batch_sequence; // Distinct from a member's slot epoch.
    r[7] = loops;
  }

  __device__ __forceinline__ void finish() {
    auto end_ns = nas_scheduler_trace_clock();
    buffer[0] = count;
    buffer[1] = count > NAS_SCHEDULER_TRACE_CAPACITY;
    buffer[2] = start_ns;
    buffer[3] = end_ns;
    buffer[4] = loops;
    buffer[5] = NAS_SCHEDULER_TRACE_CAPACITY;
    buffer[6] = 1; // Schema version; two header rows, then eight-word records.
    buffer[8] = sm;
    buffer[9] = blockIdx.x;
    buffer[10] = threadIdx.x;
    // Publish once on controller shutdown. Host synchronization is required
    // before reading; this is not a live producer/consumer ring.
    __threadfence();
    buffer[7] = 1;
  }
};

#define NAS_SCHED_TRACE_INIT() NasSchedulerTrace scheduler_trace
#define NAS_SCHED_TRACE_LOOP() (++scheduler_trace.loops)
#define NAS_SCHED_TRACE_BEGIN(name) auto name = nas_scheduler_trace_clock()
#define NAS_SCHED_TRACE_END(name, kind, result, subject, epoch, sequence) \
  do { \
    auto name##_end = nas_scheduler_trace_clock(); \
    scheduler_trace.append(name, name##_end, kind, result, subject, epoch, sequence); \
  } while (0)
#define NAS_SCHED_TRACE_FINISH() scheduler_trace.finish()
#else
// Compile away every timestamp, metadata expression, and trace write.
#define NAS_SCHED_TRACE_INIT() ((void)0)
#define NAS_SCHED_TRACE_LOOP() ((void)0)
#define NAS_SCHED_TRACE_BEGIN(name) ((void)0)
#define NAS_SCHED_TRACE_END(...) ((void)0)
#define NAS_SCHED_TRACE_FINISH() ((void)0)
#endif
