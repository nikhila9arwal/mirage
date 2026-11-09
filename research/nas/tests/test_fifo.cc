// Replay the same append/select/commit/reset stream on CPU and GPU. The
// independent oracle uses std::map + std::deque, not intrusive member links.
#include "experimental_moe_fifo.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
using namespace mirage::runtime;
constexpr uint32_t S = EXPERIMENTAL_MOE_STATE_SLOTS;
constexpr uint32_t P = EXPERIMENTAL_MOE_CHUNK_ROWS;
constexpr uint32_t R = EXPERIMENTAL_MOE_TOP_K;
constexpr uint32_t N = EXPERIMENTAL_MOE_PENDING_ROWS;
constexpr uint32_t OPS = EXPERIMENTAL_BATCH_OPERATORS;
struct Fixture {
  ExperimentalMoePendingSlot pending[S] = {};
  ExperimentalBatchOperator operators[OPS] = {};
  ExperimentalMoeFifo fifo = {};
  ExperimentalMoeSelection selected;
};

uint32_t active_key(ExperimentalMoeFifo const &fifo, uint32_t i) {
#if NAS_FIFO_HOT_RECORDS
  return fifo.active[i].key;
#else
  return fifo.active[i];
#endif
}
uint32_t bin_count(ExperimentalMoeFifo const &fifo, uint32_t key) {
#if NAS_FIFO_HOT_RECORDS
  return fifo.active_position[key] ? fifo.active[fifo.active_position[key]-1].count : 0;
#else
  return fifo.count[key];
#endif
}
uint64_t bin_age(ExperimentalMoeFifo const &fifo, uint32_t key) {
#if NAS_FIFO_HOT_RECORDS
  return fifo.active[fifo.active_position[key]-1].oldest;
#elif NAS_FIFO_CACHE_OLDEST
  return fifo.oldest[key];
#else
  (void)fifo; (void)key; return 0;
#endif
}

enum Kind { APPEND, SELECT, COMMIT, RESET };
struct Action { Kind kind; uint32_t member, key; unsigned long long time; };

#ifdef __CUDACC__
__host__ __device__
#endif
void act(Fixture &f, Action a) {
  if (a.kind == APPEND) {
    // Match the production ready() precondition. Operator fields never change
    // during replay, so skipping the duplicate winner-loop check is valid.
    assert(f.operators[a.key/128].present && f.operators[a.key/128].enabled);
    auto &slot = f.pending[a.member / (P*R)];
    uint32_t offset = a.member % (P*R);
    assert(!(slot.pending_mask & (1u << offset)));
    if (slot.pending_mask == 0u) {
      slot.op_id = a.key / 128; slot.ready_ns = a.time; ++slot.epoch;
    }
    assert(slot.op_id == a.key / 128 && slot.ready_ns == a.time);
    slot.experts[offset/R][offset%R] = a.key % 128;
    slot.pending_mask |= 1u << offset;
    assert(experimental_moe_fifo_append(f.fifo,a.key,a.member,a.time,
        f.operators[a.key/128].threshold,f.operators[a.key/128].timeout_ns));
  } else if (a.kind == SELECT) {
    f.selected = experimental_moe_select_fifo(f.pending,f.operators,f.fifo,a.time);
  } else if (a.kind == COMMIT) {
    assert(experimental_moe_fifo_commit(f.fifo,f.pending,f.selected));
    for (uint32_t j=0; j<f.selected.count; ++j) {
      auto &row = f.pending[f.selected.slots[j]];
      uint32_t bit = 1u << (f.selected.tokens[j]*R+f.selected.branches[j]);
      assert(row.pending_mask & bit);
      row.pending_mask &= ~bit;
    }
  } else {
    // Match prepare_kernel: links/heads/tails intentionally retain old bytes.
#if NAS_FIFO_HOT_RECORDS
    for (auto &n : f.fifo.active_position) n=0;
#else
    for (auto &n : f.fifo.count) n=0;
#endif
    f.fifo.active_count=0;
    for (auto &row : f.pending) row.pending_mask=0;
  }
}

std::vector<uint32_t> members(ExperimentalMoeSelection const &s) {
  std::vector<uint32_t> result;
  for (uint32_t j=0; j<s.count; ++j)
    result.push_back((s.slots[j]*P+s.tokens[j])*R+s.branches[j]);
  return result;
}
void same(ExperimentalMoeSelection const &a, ExperimentalMoeSelection const &b) {
  assert(a.op_id==b.op_id && a.expert==b.expert && a.reason==b.reason &&
         a.group_count==b.group_count && a.count==b.count &&
         a.active_bin_index==b.active_bin_index && members(a)==members(b));
}

#ifdef __CUDACC__
void cuda_check(cudaError_t e) {
  if (e!=cudaSuccess) { std::cerr<<cudaGetErrorString(e)<<'\n'; std::abort(); }
}
__global__ void replay(Fixture *f, Action const *actions, size_t n,
                       ExperimentalMoeSelection *results) {
  if (threadIdx.x || blockIdx.x) return;
  size_t out=0;
  for (size_t i=0; i<n; ++i) {
    act(*f,actions[i]);
    if (actions[i].kind==SELECT) results[out++]=f->selected;
  }
}
#endif

int main() {
  auto initial=std::make_unique<Fixture>();
  for (uint32_t i=0; i<OPS; ++i) {
    auto &op=initial->operators[i];
    op.present=op.enabled=1;
    // Operation zero retains the explicit deadline-boundary fixture below.
    // Other operations exercise copying distinct immutable policy parameters.
    op.threshold=i==0 ? 8u : 1u+i%16u;
    op.timeout_ns=i==0 ? 1000u : 1u+(i*137u)%1000u;
  }
  auto owned=std::make_unique<Fixture>(*initial);
  auto &f=*owned;
  std::map<uint32_t,std::deque<uint32_t>> queues;
  std::vector<Action> actions;
  std::vector<ExperimentalMoeSelection> selections;
  unsigned long long now=1000;
  uint64_t oversized=0, empty=0, selected_rows=0;

  auto check = [&]() {
    assert(f.fifo.active_count==queues.size());
    std::vector<bool> seen(N,false);
    std::vector<uint32_t> keys;
    for (uint32_t i=0; i<f.fifo.active_count; ++i) keys.push_back(active_key(f.fifo,i));
    std::sort(keys.begin(),keys.end());
    size_t k=0;
    for (auto const &[key,q] : queues) {
      assert(keys[k++]==key && bin_count(f.fifo,key)==q.size());
#if NAS_FIFO_HOT_RECORDS
      uint32_t pos=f.fifo.active_position[key];
      assert(pos>0 && pos<=f.fifo.active_count && f.fifo.active[pos-1].key==key);
      assert(f.fifo.active[pos-1].threshold==f.operators[key/128].threshold);
      assert(f.fifo.active[pos-1].timeout_ns==f.operators[key/128].timeout_ns);
#endif
      uint32_t index=f.fifo.head[key];
      for (uint32_t member : q) {
        assert(index==member && member<N && !seen[member]); seen[member]=true;
        auto row=experimental_moe_row(f.pending,member);
        assert(row.pending && row.op_id*128+row.expert==key);
        index=f.fifo.next[index];
      }
      assert(index==UINT32_MAX && f.fifo.tail[key]==q.back());
#if NAS_FIFO_CACHE_OLDEST
      // The oracle follows the head even when the selector uses cached age.
      // This catches a stale age after partial dispatch, empty-bin reuse, or
      // prepare_kernel-style reset (which deliberately leaves age bytes alone).
      assert(bin_age(f.fifo,key)==f.pending[q.front()/(P*R)].ready_ns);
#endif
    }
    for (uint32_t key=0; key<EXPERIMENTAL_MOE_SELECTION_BINS; ++key)
      if (!queues.count(key)) assert(bin_count(f.fifo,key)==0);
    for (uint32_t member=0; member<N; ++member)
      assert(seen[member]==bool(experimental_moe_row(f.pending,member).pending));
  };
  auto step = [&](Action a) {
    if (a.kind==APPEND) queues[a.key].push_back(a.member);
    if (a.kind==RESET) queues.clear();
    if (a.kind==COMMIT) {
      uint32_t key=uint32_t(f.selected.op_id)*128+f.selected.expert;
      auto &q=queues.at(key);
      for (uint32_t member : members(f.selected)) { assert(q.front()==member); q.pop_front(); }
      if (q.empty()) queues.erase(key);
    }
    act(f,a); actions.push_back(a);
    if (a.kind==SELECT) {
      uint64_t best_time=UINT64_MAX; uint32_t best_key=UINT32_MAX;
      for (auto const &[key,q] : queues) {
        auto const &op=f.operators[key/128];
        uint64_t age=f.pending[q.front()/(P*R)].ready_ns;
        if (!op.present || !op.enabled ||
            (q.size()<op.threshold && a.time-age<op.timeout_ns)) continue;
        if (std::make_pair(age,key)<std::make_pair(best_time,best_key)) {
          best_time=age; best_key=key;
        }
      }
      if (best_key==UINT32_MAX) { assert(f.selected.op_id==-1); ++empty; }
      else {
        auto const &q=queues.at(best_key);
        assert(uint32_t(f.selected.op_id)*128+f.selected.expert==best_key);
        assert(f.selected.group_count==q.size());
        assert(f.selected.reason==(q.size()>=f.operators[best_key/128].threshold ? 1u : 2u));
        std::vector<uint32_t> expected(q.begin(),q.begin()+std::min<size_t>(16,q.size()));
        assert(members(f.selected)==expected);
        oversized += q.size()>16; selected_rows += f.selected.count;
      }
      selections.push_back(f.selected);
    }
  };
  auto append_slot = [&](uint32_t slot,uint32_t op,uint32_t rows,uint32_t expert_base) {
    assert(f.pending[slot].pending_mask==0);
    for (uint32_t t=0; t<rows; ++t)
      for (uint32_t b=0; b<R; ++b)
        step({APPEND,(slot*P+t)*R+b,op*128+(expert_base+b)%128,now});
  };
  step({SELECT,0,0,now}); // Empty and before/exactly at the deadline.
  step({APPEND,0,7,now});
  step({SELECT,0,0,now+999}); assert(f.selected.op_id==-1);
  step({SELECT,0,0,now+1000}); assert(f.selected.reason==2);
  step({COMMIT,0,0,0}); check();

  // Equal-time bins, reverse slot order, and >16 members. Check FIFO order,
  // partial drains and head age changes; active-list swap order cannot win ties.
  now+=2000;
  for (uint32_t s=S; s-- > 0;) { ++now; append_slot(s,OPS-4,P,7); }
  while (!queues.empty()) {
    step({SELECT,0,0,now+1000});
    auto before=f.selected;
    step({SELECT,0,0,now+1000}); same(before,f.selected); // Read-only peek.
    step({COMMIT,0,0,0}); check();
  }
  // Repeated epochs mix operations/layers, dense expert-zero identities, full
  // prefill and one-row decode, and reset while links still contain old state.
  std::mt19937 rng(5271);
  for (int i=0; i<3000; ++i) {
    now+=1+rng()%30;
    uint32_t slot=rng()%S;
    if (f.pending[slot].pending_mask==0) {
      uint32_t op=rng()%OPS;
      if (op%4<2) append_slot(slot,op,(rng()%2)?P:1,rng()%128);
      else step({APPEND,slot*P*R,op*128,now});
    }
    step({SELECT,0,0,now});
    if (f.selected.op_id>=0) step({COMMIT,0,0,0});
    if (i%113==0) step({RESET,0,0,0});
    check();
  }
  now+=1000;
  while (!queues.empty()) {
    step({SELECT,0,0,now}); step({COMMIT,0,0,0}); check();
  }
  assert(empty && selected_rows && (P==1 || oversized));
  assert(!experimental_moe_fifo_append(f.fifo,EXPERIMENTAL_MOE_SELECTION_BINS,0,now));
  assert(!experimental_moe_fifo_append(f.fifo,0,N,now));
  assert(!experimental_moe_fifo_commit(f.fifo,f.pending,ExperimentalMoeSelection{}));
#ifdef __CUDACC__
  Fixture *device; Action *device_actions; ExperimentalMoeSelection *device_results;
  cuda_check(cudaMalloc(&device,sizeof(Fixture)));
  cuda_check(cudaMalloc(&device_actions,actions.size()*sizeof(Action)));
  cuda_check(cudaMalloc(&device_results,selections.size()*sizeof(ExperimentalMoeSelection)));
  cuda_check(cudaMemcpy(device,initial.get(),sizeof(Fixture),cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(device_actions,actions.data(),actions.size()*sizeof(Action),cudaMemcpyHostToDevice));
  replay<<<1,32>>>(device,device_actions,actions.size(),device_results);
  cuda_check(cudaGetLastError());
  std::vector<ExperimentalMoeSelection> results(selections.size());
  cuda_check(cudaMemcpy(results.data(),device_results,results.size()*sizeof(results[0]),cudaMemcpyDeviceToHost));
  for (size_t i=0; i<results.size(); ++i) same(results[i],selections[i]);
  auto actual=std::make_unique<Fixture>();
  cuda_check(cudaMemcpy(actual.get(),device,sizeof(Fixture),cudaMemcpyDeviceToHost));
  assert(std::memcmp(&actual->fifo,&f.fifo,sizeof(f.fifo))==0);
  assert(std::memcmp(actual->pending,f.pending,sizeof(f.pending))==0);
  cuda_check(cudaFree(device)); cuda_check(cudaFree(device_actions)); cuda_check(cudaFree(device_results));
  std::cout<<"GPU replay passed; ";
#endif
  // Compare this digest across build switches: timing-free selection and FIFO
  // membership must remain identical, including the equal-age tie breaker.
  uint64_t digest=14695981039346656037ull;
  auto mix=[&](uint32_t value) { digest=(digest^value)*1099511628211ull; };
  for (auto const &selected : selections) {
    mix(uint32_t(selected.op_id)); mix(selected.expert); mix(selected.reason);
    mix(selected.group_count); mix(selected.count); mix(selected.active_bin_index);
    for (uint32_t member : members(selected)) mix(member);
  }
  std::cout<<"selection_digest="<<digest<<"; CPU passed: "<<actions.size()<<" actions, "<<selections.size()
           <<" selections, "<<oversized<<" oversized winners, FIFO bytes="
           <<sizeof(ExperimentalMoeFifo)<<"\n";
}
