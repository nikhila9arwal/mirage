// CPU policy checks; compile the same cases with nvcc to check device execution.
#include "experimental_moe_selection.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <tuple>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

using namespace mirage::runtime;
constexpr uint32_t N = EXPERIMENTAL_MOE_PENDING_ROWS;
constexpr uint32_t P = EXPERIMENTAL_MOE_CHUNK_ROWS;
constexpr uint32_t R = EXPERIMENTAL_MOE_TOP_K;
constexpr uint32_t S = EXPERIMENTAL_MOE_STATE_SLOTS;
constexpr uint32_t OPS = EXPERIMENTAL_BATCH_OPERATORS;
constexpr unsigned long long NOW = 10000;

struct Fixture {
  ExperimentalMoePendingSlot pending[S] = {};
  ExperimentalBatchOperator operators[OPS] = {};
  bool seen[N] = {};
  ExperimentalMoeSelectionScratch scratch = {};
};

void init(Fixture &f) {
  for (uint32_t op = 0; op < OPS; ++op) {
    f.operators[op].kind = op % 4;
    f.operators[op].layer_id = op / 4;
    f.operators[op].present = f.operators[op].enabled = 1;
    f.operators[op].num_experts = op % 4 < 2 ? 128 : 1;
    f.operators[op].threshold = 8;
    f.operators[op].timeout_ns = 1000;
  }
}

struct Group {
  unsigned long long oldest = ~0ull;
  uint32_t first = N;
  std::vector<uint32_t> members;
};

std::vector<uint32_t> flat_members(ExperimentalMoeSelection const &s) {
  std::vector<uint32_t> rows;
  for (uint32_t i = 0; i < s.count; ++i)
    rows.push_back((s.slots[i] * P + s.tokens[i]) * R + s.branches[i]);
  return rows;
}

void validate(Fixture const &f, ExperimentalMoeSelection const &result, bool greedy) {
  // Independent reference: construct groups from the actual bit masks and
  // sort their complete member lists, rather than reproducing histogram loops.
  std::map<std::pair<uint32_t,uint32_t>, Group> groups;
  for (uint32_t slot = 0; slot < S; ++slot) {
    auto const &row = f.pending[slot];
    if (!f.operators[row.op_id].present || !f.operators[row.op_id].enabled) continue;
    for (uint32_t token = 0; token < P; ++token)
      for (uint32_t branch = 0; branch < R; ++branch) {
        if (!(row.pending_mask & (1u << (token * R + branch)))) continue;
        uint32_t index = (slot * P + token) * R + branch;
        auto &group = groups[{row.op_id, row.experts[token][branch]}];
        group.oldest = std::min(group.oldest, row.ready_ns);
        group.first = std::min(group.first, index);
        group.members.push_back(index);
      }
  }
  std::tuple<unsigned long long,uint32_t,uint32_t> best{~0ull,UINT32_MAX,UINT32_MAX};
  std::pair<uint32_t,uint32_t> key;
  Group const *winner = nullptr;
  for (auto const &entry : groups) {
    auto const &op = f.operators[entry.first.first];
    auto const &group = entry.second;
    if (group.members.size() < op.threshold && NOW - group.oldest < op.timeout_ns) continue;
    auto order = std::make_tuple(group.oldest,entry.first.first,group.first);
    if (order < best) { best=order; key=entry.first; winner=&group; }
  }
  if (!winner) { assert(result.op_id == -1 && result.count == 0); return; }
  assert(result.op_id == int(key.first) && result.expert == key.second);
  assert(result.second_expert == UINT32_MAX);
  assert(result.group_count == winner->members.size());
  assert(result.reason == (winner->members.size() >= f.operators[key.first].threshold ? 1u : 2u));
  auto expected = winner->members;
  if (!greedy)
    std::sort(expected.begin(),expected.end(),[&](uint32_t a,uint32_t b) {
      return std::make_pair(f.pending[a/(P*R)].ready_ns,a) <
             std::make_pair(f.pending[b/(P*R)].ready_ns,b);
    });
  expected.resize(std::min<size_t>(expected.size(),16));
  assert(flat_members(result) == expected);
}

#ifdef __CUDACC__
void check_cuda(cudaError_t e) {
  if (e != cudaSuccess) { std::cerr << cudaGetErrorString(e) << '\n'; std::abort(); }
}
__global__ void device_check(Fixture *f, ExperimentalMoeSelection *out) {
  if (threadIdx.x || blockIdx.x) return;
  out[0] = experimental_moe_select(f->pending,f->operators,f->seen,NOW);
  out[1] = experimental_moe_select_histogram(f->pending,f->operators,f->scratch,NOW);
  // Reuse catches forgotten scratch reset, including the no-eligible path.
  out[2] = experimental_moe_select_histogram(f->pending,f->operators,f->scratch,NOW);
}
Fixture *gpu_fixture;
ExperimentalMoeSelection *gpu_output;
#endif

void check(Fixture &f, bool gpu) {
  std::vector<unsigned char> before(sizeof(f.pending));
  std::memcpy(before.data(), f.pending, before.size());
  auto old = experimental_moe_select(f.pending,f.operators,f.seen,NOW);
  auto next = experimental_moe_select_histogram(f.pending,f.operators,f.scratch,NOW);
  validate(f,old,false); validate(f,next,true);
  assert(std::memcmp(before.data(),f.pending,before.size()) == 0);
  for (auto count : f.scratch.count) assert(count == 0);
  auto again = experimental_moe_select_histogram(f.pending,f.operators,f.scratch,NOW);
  validate(f,again,true);
  for (auto count : f.scratch.count) assert(count == 0);
#ifdef __CUDACC__
  if (gpu) {
    check_cuda(cudaMemcpy(gpu_fixture,&f,sizeof(f),cudaMemcpyHostToDevice));
    device_check<<<1,32>>>(gpu_fixture,gpu_output);
    check_cuda(cudaGetLastError());
    ExperimentalMoeSelection out[3];
    check_cuda(cudaMemcpy(out,gpu_output,sizeof(out),cudaMemcpyDeviceToHost));
    validate(f,out[0],false); validate(f,out[1],true); validate(f,out[2],true);
    auto downloaded = std::make_unique<Fixture>();
    check_cuda(cudaMemcpy(downloaded.get(),gpu_fixture,sizeof(f),cudaMemcpyDeviceToHost));
    for (auto count : downloaded->scratch.count) assert(count == 0);
    assert(std::memcmp(f.pending,downloaded->pending,sizeof(f.pending)) == 0);
  }
#else
  (void)gpu;
#endif
}

int main() {
#ifdef __CUDACC__
  check_cuda(cudaMalloc(&gpu_fixture,sizeof(Fixture)));
  check_cuda(cudaMalloc(&gpu_output,3*sizeof(ExperimentalMoeSelection)));
#endif
  auto owned = std::make_unique<Fixture>();
  auto &f = *owned; init(f);
  check(f,true); // Empty.
  // All tokens route to the same expert; valid top-k members stay distinct.
  // Newer low slots deliberately expose the change from oldest to first C.
  for (uint32_t slot = 0; slot < S; ++slot) {
    auto &row = f.pending[slot];
    row.pending_mask = 0;
    row.op_id = 4 * (EXPERIMENTAL_MOE_LAYERS - 1);
    row.ready_ns = NOW - slot - 1;
    for (uint32_t token = 0; token < P; ++token) {
      row.pending_mask |= 1u << (token*R);
      row.experts[token][0] = 127;
    }
  }
  check(f,true); // Threshold-selected; maximum count S*P for one bin.
  auto result = experimental_moe_select_histogram(f.pending,f.operators,f.scratch,NOW);
  assert(result.group_count == S*P);
  // Simulate dispatch, not completion. Leftover member ages must be unchanged.
  for (uint32_t j = 0; j < result.count; ++j)
    f.pending[result.slots[j]].pending_mask &= ~(1u << (result.tokens[j]*R+result.branches[j]));
  check(f,true);
  for (auto &row : f.pending) row.pending_mask = 0;
  f.pending[0].pending_mask=1; f.pending[0].ready_ns=NOW;
  check(f,true); // One young row is not eligible; scratch must still reset.
  f.pending[0].ready_ns=NOW-1000;
  check(f,true); // Exactly on deadline.
  f.pending[1]=f.pending[0]; f.pending[1].op_id=0;
  check(f,true); // Equal timestamps choose lower operation, even later slot.
  f.pending[0].op_id=0; f.pending[0].experts[0][0]=99;
  f.pending[1].experts[0][0]=1;
  check(f,true); // Equal op/time chooses first flat row, not lower expert ID.

  std::mt19937 rng(20261103);
  for (int trial = 0; trial < 2000; ++trial) {
    for (uint32_t op = 0; op < OPS; ++op) {
      f.operators[op].present = rng()%5 != 0;
      f.operators[op].enabled = rng()%5 != 0;
      f.operators[op].threshold = 1 + rng()%16;
      f.operators[op].timeout_ns = rng()%10001;
    }
    for (uint32_t slot = 0; slot < S; ++slot) {
      auto &row = f.pending[slot];
      row.pending_mask=0; row.op_id=rng()%OPS; row.ready_ns=NOW-rng()%10001;
      for (uint32_t token=0; token<P; ++token) {
        uint32_t first=rng()%128;
        for (uint32_t branch=0; branch<R; ++branch) {
          bool dense = row.op_id%4 >= 2;
          row.experts[token][branch] = dense ? 0 : (first+branch)%128;
          if ((!dense || (token==0 && branch==0)) && rng()%3)
            row.pending_mask |= 1u << (token*R+branch);
        }
      }
    }
    check(f,trial<40); // The rest exercise policy extensively on the CPU.
  }
#ifdef __CUDACC__
  check_cuda(cudaFree(gpu_fixture)); check_cuda(cudaFree(gpu_output));
  std::cout << "CPU + GPU ";
#else
  std::cout << "CPU ";
#endif
  std::cout << "selector tests passed: P=" << P << " R=" << R << " S=" << S
            << " layers=" << EXPERIMENTAL_MOE_LAYERS << ", 2000 random states + edge cases\n";
}
