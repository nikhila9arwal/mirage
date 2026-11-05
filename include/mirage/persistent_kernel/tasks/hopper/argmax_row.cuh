/* Copyright 2026 CMU. Licensed under the Apache License, Version 2.0. */
#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>

namespace kernel {

// Compare (value, vocabulary index), so ties do not depend on warp/lane order.
// The first NaN wins, matching torch.argmax; model validation still requires
// finite logits. Infinite values and signed zero follow the same index rule.
__device__ __forceinline__ bool argmax_row_better(
    float value, int index, float best_value, int best_index) {
  bool value_nan = isnan(value);
  bool best_nan = isnan(best_value);
  if (value_nan != best_nan) return value_nan;
  if (value_nan) return index < best_index;
  return value > best_value || (value == best_value && index < best_index);
}

__device__ __forceinline__ void argmax_row_warp_reduce(float &value, int &index) {
  int lane = threadIdx.x % 32;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    float other_value = __shfl_down_sync(0xffffffffu, value, offset);
    int other_index = __shfl_down_sync(0xffffffffu, index, offset);
    if (lane + offset < 32 &&
        argmax_row_better(other_value, other_index, value, index)) {
      value = other_value;
      index = other_index;
    }
  }
}

// One worker CTA consumes one complete row. Its descriptor already points at
// the owning slot, so this kernel needs neither a slot ID nor global metadata.
template <typename T, int VOCAB_SIZE, int THREADS = 256>
__device__ __forceinline__ void argmax_row_impl(
    void const *input_ptr, void *output_ptr) {
  static_assert(VOCAB_SIZE > 0 && THREADS == 256);
  T const *input = static_cast<T const *>(input_ptr);
  long long *output = static_cast<long long *>(output_ptr);
  constexpr int WARPS = THREADS / 32;
  constexpr int NO_INDEX = 0x7fffffff;
  extern __shared__ char smem[];
  float *warp_values = reinterpret_cast<float *>(smem);
  int *warp_indices = reinterpret_cast<int *>(warp_values + WARPS);

  float value = -CUDART_INF_F;
  int index = NO_INDEX;
  // The bound handles arbitrary vocabulary widths, including the final tail.
  for (int column = threadIdx.x; column < VOCAB_SIZE; column += THREADS) {
    float candidate = float(input[column]);
    if (argmax_row_better(candidate, column, value, index)) {
      value = candidate;
      index = column;
    }
  }
  argmax_row_warp_reduce(value, index);
  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;
  if (lane == 0) {
    warp_values[warp] = value;
    warp_indices[warp] = index;
  }
  __syncthreads();

  if (warp == 0) {
    value = lane < WARPS ? warp_values[lane] : -CUDART_INF_F;
    index = lane < WARPS ? warp_indices[lane] : NO_INDEX;
    argmax_row_warp_reduce(value, index);
    if (lane == 0) output[0] = static_cast<long long>(index);
  }
  // The worker's existing completion barrier and publication follow this call.
}

} // namespace kernel
