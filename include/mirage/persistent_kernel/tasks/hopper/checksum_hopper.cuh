/* Copyright 2026 CMU. Licensed under the Apache License, Version 2.0. */
#pragma once

namespace kernel {

// One block reduces one slot's row. Accumulate in FP32 and round only the final
// scalar to BF16, so the existing ContextIO output interface can save it.
// Coefficients are shared read-only; no reduction ever crosses slot boundaries.
template <typename T, int HIDDEN_DIM, int NUM_THREADS = 256>
__device__ __forceinline__ void checksum_hopper_impl(
    void const *input_ptr, void const *coefficient_ptr, void *output_ptr) {
  static_assert(HIDDEN_DIM > 0 && NUM_THREADS == 256);
  T const *input = static_cast<T const *>(input_ptr);
  T const *coefficient = static_cast<T const *>(coefficient_ptr);
  T *output = static_cast<T *>(output_ptr);
  extern __shared__ char smem[];
  float *warp_sums = reinterpret_cast<float *>(smem);

  float sum = 0.0f;
  for (int column = threadIdx.x; column < HIDDEN_DIM; column += NUM_THREADS) {
    sum += float(input[column]) * float(coefficient[column]);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();

  if (warp == 0) {
    sum = lane < NUM_THREADS / 32 ? warp_sums[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset /= 2) {
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) output[0] = T(sum);
  }
  // The worker's existing completion barrier follows this function.
}

} // namespace kernel
