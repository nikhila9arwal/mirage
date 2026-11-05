#pragma once

// Experimental adapter for the existing MPK Hopper MoE implementation.
// This file changes addressing/metadata plumbing only: the matrix arithmetic
// remains kernel::moe_linear_sm90_task_impl from the isolated checkout.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include "tasks/hopper/moe_linear_swapAB_hopper.cuh"
#include "tasks/hopper/tma_2d.cuh"

namespace nas_moe_probe {

constexpr int physical_rows = 16;
constexpr int num_experts = 128;
constexpr int num_routes = 1;

// Every thread of a 256-thread worker must call this helper. The immutable
// mask lists active expert IDs; mask[E] is their count. Expert-major membership
// routing[expert * 16 + local_row] is 1 only for that expert's rows. state_rows maps
// each active local row to its input row; output_rows may select a different
// destination. Flattening [slot, branch, width] into branch rows lets top-k use
// the same one-expert-per-physical-row arithmetic. row_count is at most 16.
// INDEXED_ROWS=false retains the original dense 16-state-row adapter. All metadata
// arrays, participating activation rows, and the descriptor must stay alive
// until this function and the caller's completion publication have finished.
// Explicit dimensions let dense projections use the same arithmetic with
// E=1, without pretending their allocation contains 128 copies of the weights.
template <bool W13, int K, int O, int E, bool INDEXED_ROWS>
__device__ __forceinline__ void projection(
    void const *weight_tma_desc,
    void const *input,
    int32_t const *routing,
    int32_t const *expert_mask,
    void *output,
    uint32_t const *state_rows = nullptr,
    uint32_t row_count = physical_rows,
    uint32_t const *output_rows = nullptr) {
  using T = cute::bfloat16_t;
  static_assert(K % 64 == 0 && O % 64 == 0,
                "The probe deliberately excludes reduction/output tails");

  using WeightTMA = kernel::tma::tma_2d<
      T, 3, 3, 3, E * O, K, 64, 64, K, 1, 1, 1,
      64 * 64, true>;
  WeightTMA weight_tma(
      static_cast<CUtensorMap *>(const_cast<void *>(weight_tma_desc)));

  auto routing_layout = cute::make_layout(
      cute::make_shape(E, physical_rows),
      // Unlike the single-expert broadcast view, mixed batches give each expert
      // its own immutable membership row. Activations still stay in their slots.
      cute::make_stride(cute::Int<physical_rows>{}, cute::Int<1>{}));
  auto indices = cute::make_tensor(cute::make_gmem_ptr(routing), routing_layout);
  // Include the final count element in the layout, as well as in its allocation.
  auto mask = cute::make_tensor(
      cute::make_gmem_ptr(expert_mask),
      cute::make_layout(cute::make_shape(E + 1),
                        cute::make_stride(cute::Int<1>{})));
  auto output_layout = cute::make_layout(
      cute::make_shape(physical_rows, O, num_routes),
      cute::make_stride(num_routes * O, cute::Int<1>{}, O));
  auto out = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<T *>(output)), output_layout);
  auto bias_layout = cute::make_layout(
      cute::make_shape(physical_rows, O, E),
      cute::make_stride(O, cute::Int<1>{}, physical_rows * O));
  auto bias = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<T *>(nullptr)), bias_layout);

  // This descriptor belongs to one worker, so expert stride 1 visits every
  // selected expert inside this same task. Native partitioned calls are unchanged.
  // Keep the two layout ranks used by native registration. With R=1 their
  // memory strides agree, but W2 still expects the explicit route dimension.
  if constexpr (W13) {
    auto input_layout = cute::make_layout(
        cute::make_shape(physical_rows, K),
        cute::make_stride(K, cute::Int<1>{}));
    auto in = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<T const *>(input)), input_layout);
    kernel::moe_linear_sm90_task_impl<
        T, WeightTMA, decltype(in), decltype(bias), decltype(indices),
        decltype(mask), decltype(out), 64, 16, physical_rows, O, O, K,
        E, num_routes, 1, true, true, 8, INDEXED_ROWS>(
        weight_tma, in, bias, indices, mask, out, 0, state_rows, row_count, output_rows);
  } else {
    auto input_layout = cute::make_layout(
        cute::make_shape(physical_rows, K, num_routes),
        cute::make_stride(num_routes * K, cute::Int<1>{}, K));
    auto in = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<T const *>(input)), input_layout);
    kernel::moe_linear_sm90_task_impl<
        T, WeightTMA, decltype(in), decltype(bias), decltype(indices),
        decltype(mask), decltype(out), 64, 16, physical_rows, O, O, K,
        E, num_routes, 1, false, true, 8, INDEXED_ROWS>(
        weight_tma, in, bias, indices, mask, out, 0, state_rows, row_count, output_rows);
  }
  // Join only this worker's producer/consumer threads before its caller can
  // publish the logical results. This is not a barrier between slots/batches.
  __syncthreads();
}

// Preserve the original W13/W2 layouts and dimensions. W2 keeps its explicit
// route dimension even though the current top-1 path has one route per row.
template <bool W13, int H = 512, int I = 256, bool INDEXED_ROWS = false>
__device__ __forceinline__ void moe_linear(
    void const *weight_tma_desc,
    void const *input,
    int32_t const *routing,
    int32_t const *expert_mask,
    void *output,
    uint32_t const *state_rows = nullptr,
    uint32_t row_count = physical_rows,
    uint32_t const *output_rows = nullptr) {
  projection<W13, W13 ? H : I, W13 ? 2 * I : H, num_experts, INDEXED_ROWS>(
      weight_tma_desc, input, routing, expert_mask, output, state_rows, row_count,
      output_rows);
}

// Dense batches use indexed slot rows and one expert. The caller supplies a
// real [N,K] weight map and immutable mask[0]=0, mask[1]=1 metadata.
template <int K, int N>
__device__ __forceinline__ void dense_linear(
    void const *weight_tma_desc,
    void const *input,
    int32_t const *routing,
    int32_t const *expert_mask,
    void *output,
    uint32_t const *state_rows,
    uint32_t row_count,
    uint32_t const *output_rows = nullptr) {
  projection<true, K, N, 1, true>(
      weight_tma_desc, input, routing, expert_mask, output, state_rows, row_count,
      output_rows);
}

} // namespace nas_moe_probe
