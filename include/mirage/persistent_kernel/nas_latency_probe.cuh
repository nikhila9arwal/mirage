#pragma once
// Measurement only. Nothing is compiled unless the isolated launcher opts in.
// All timestamps use the GPU-global nanosecond timer, not an SM cycle counter.
// The host reads these buffers only after torch.cuda.synchronize(). Existing
// queue release/acquire operations publish worker records to the controller.
#if defined(NAS_DISPATCH_BREAKDOWN) && !defined(NAS_LATENCY_DIAGNOSTIC)
#error "Dispatch breakdown requires the diagnostic probe buffers"
#endif
#if defined(NAS_SELECTOR_BREAKDOWN) && !defined(NAS_DISPATCH_BREAKDOWN)
#error "Selector breakdown requires the dispatch diagnostic buffers"
#endif
#ifdef NAS_LATENCY_PROBE
namespace nas_probe {
using Word = unsigned long long;
enum Buffer { HEADER, ADMISSIONS, PASSES, TOKENS, WORKER_COUNTS, WORKER_RECORDS,
              DATA_RECORDS, DATA_ENQUEUE, BATCH_RECORDS, BATCH_ENQUEUE, SCHEDULER, DISPATCH,
              SELECTOR_PHASES, SELECTOR_STATS, MEMBERSHIP };
__device__ __forceinline__ Word clock() {
  Word value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
  return value;
}
__device__ __forceinline__ Word *buffer(Buffer index) {
  auto *addresses = reinterpret_cast<Word const *>(NAS_LATENCY_ADDRESS);
  return reinterpret_cast<Word *>(addresses[index]);
}
__device__ __forceinline__ void start() { buffer(HEADER)[0] = clock(); }
__device__ __forceinline__ void finish() { buffer(HEADER)[1] = clock(); }
__device__ __forceinline__ void admit(int request, int slot) {
  Word *record = buffer(ADMISSIONS) + request * 4;
  record[0] = clock(); record[1] = slot; record[2]++;
}
__device__ __forceinline__ void begin_pass(
    int pass, int slot, Word epoch, int rows, int64_t const *input) {
  Word *record = buffer(PASSES) + pass * 16;
  record[0] = clock(); record[2] = slot; record[3] = epoch; record[4] = rows;
  // Four scalar IDs suffice for this unchanged P=4 workload. These small
  // captures verify feedback without copying intermediate activation vectors.
  for (int i = 0; i < NAS_LATENCY_CHUNK_ROWS; ++i) record[6+i] = input[i];
  record[10]++;
}
__device__ __forceinline__ void end_pass(int pass, int64_t predicted) {
  Word *record = buffer(PASSES) + pass * 16;
  record[1] = clock(); record[5] = predicted; record[11]++;
}
__device__ __forceinline__ void token(int output, int slot, Word epoch, int64_t value) {
  Word *record = buffer(TOKENS) + output * 4;
  record[0] = clock(); record[1] = value; record[2] = slot; record[3] = epoch;
}

#ifdef NAS_LATENCY_DIAGNOSTIC
// Each worker owns its append cursor. The controller only fills fields 3/4 of
// a completed record after acquiring the normal completion queue publication.
// A per-DataId/pool lookup is safe because the next use cannot begin before
// the previous completion retires. No new synchronization is introduced.
__device__ __forceinline__ void task(
    int worker, Word begin, Word end, bool batch, Word id, Word epoch,
    Word rows, Word prefill_rows, Word experts, Word slot, Word request,
    Word position, Word expert, Word lookup) {
  Word *header = buffer(WORKER_COUNTS) + worker * 2;
  Word index = header[0]++;
  Word *records = buffer(batch ? BATCH_RECORDS : DATA_RECORDS);
  records[lookup] = 0;
  if (index >= NAS_LATENCY_TASK_CAPACITY) { header[1] = 1; return; }
  Word *r = buffer(WORKER_RECORDS) + (worker * NAS_LATENCY_TASK_CAPACITY + index) * 16;
  unsigned int sm;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  r[0]=begin; r[1]=end;
  r[2]=buffer(batch ? BATCH_ENQUEUE : DATA_ENQUEUE)[lookup];
  r[3]=0; r[4]=0; r[5]=batch; r[6]=id; r[7]=epoch;
  r[8]=rows; r[9]=prefill_rows; r[10]=experts; r[11]=slot;
  r[12]=request; r[13]=position; r[14]=sm; r[15]=expert;
  records[lookup] = reinterpret_cast<Word>(r);
}
__device__ __forceinline__ Word *completion(bool batch, Word id) {
  auto *record = reinterpret_cast<Word *>(buffer(batch ? BATCH_RECORDS : DATA_RECORDS)[id]);
  if (record) record[3] = clock();
  return record;
}
__device__ __forceinline__ void retire(Word *record) {
  if (record) record[4] = clock();
}
__device__ __forceinline__ void scope(int kind, bool result, Word begin, Word end) {
  // One controller owns all statistics. This avoids an unbounded poll trace
  // and a large scheduler-local array. Nested logical completion is reported
  // separately and must not be added to its enclosing handler a second time.
  Word *stat = buffer(SCHEDULER) + (kind * 2 + result) * 3;
  Word duration = end - begin;
  stat[0]++; stat[1] += duration; if (duration > stat[2]) stat[2] = duration;
}
#ifdef NAS_DISPATCH_BREAKDOWN
enum DispatchOutcome { NO_DESCRIPTOR, NO_ELIGIBLE, PUBLISHED };
enum DispatchPhase { LOOKUP, SELECTION, DESCRIPTOR, PUBLICATION };
__device__ __forceinline__ void dispatch_stage(
    DispatchOutcome outcome, DispatchPhase phase, Word begin, Word end) {
  // One controller; fixed-size aggregates, no atomics or per-attempt records.
  Word *stat = buffer(DISPATCH) + (outcome * 4 + phase) * 3;
  Word duration = end - begin;
  stat[0]++;
  stat[1] += duration;
  if (duration > stat[2]) stat[2] = duration;
}
#endif
#ifdef NAS_SELECTOR_BREAKDOWN
__device__ __forceinline__ void membership(
    uint32_t phase, Word begin, Word end, uint32_t members) {
  // Include work moved OUT of selection. Phase 0 installs ready members;
  // phase 1 removes a captured batch. Both are nested in existing scopes.
  Word *stat = buffer(MEMBERSHIP) + phase * 4;
  Word duration = end - begin;
  ++stat[0]; stat[1] += duration;
  if (duration > stat[2]) stat[2] = duration;
  stat[3] += members;
}
__device__ __forceinline__ void selector(
    Word const (&boundary)[5], uint32_t groups, uint32_t winning_count,
    uint32_t members, uint32_t reason, bool published) {
  // One controller owns these bounded aggregates. No per-attempt trace buffer.
  for (int phase = 0; phase < 4; ++phase) {
    Word *stat = buffer(SELECTOR_PHASES) + (published * 4 + phase) * 3;
    Word duration = boundary[phase + 1] - boundary[phase];
    ++stat[0]; stat[1] += duration;
    if (duration > stat[2]) stat[2] = duration;
  }
  // calls, groups sum/max, winning count sum/max, selected rows,
  // winning bins above capacity, threshold-selected wins.
  Word *stat = buffer(SELECTOR_STATS) + published * 8;
  ++stat[0]; stat[1] += groups;
  if (groups > stat[2]) stat[2] = groups;
  stat[3] += winning_count;
  if (winning_count > stat[4]) stat[4] = winning_count;
  stat[5] += members;
  if (published && winning_count > 16u) ++stat[6];
  if (published && reason == 1u) ++stat[7];
}
#endif
#endif
} // namespace nas_probe
#endif
