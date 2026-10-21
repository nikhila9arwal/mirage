/* Copyright 2026 CMU. Licensed under the Apache License, Version 2.0. */
#pragma once

#include <cassert>
#include <cstddef>
#include <cuda_bf16.h>

namespace mirage {
namespace runtime {

// Call on scheduler 0's advancing thread, after all work for this slot has
// completed and while the slot is PREPARING. Skip this call at bootstrap.
// The caller selects the saved row using the OLD context ID. Both arguments
// point to contiguous rows of width BF16 elements; their storage cannot overlap.
__device__ __forceinline__ void save_output(
    __nv_bfloat16 const *slot_output, __nv_bfloat16 *saved_output, int width) {
  assert(width > 0);
  for (int column = 0; column < width; column++) {
    saved_output[column] = slot_output[column];
  }
}

// Call on the same advancing thread, at bootstrap or after save_output().
// inputs is a contiguous [num_contexts, width] queue; slot_input points
// to one writable row, disjoint from that queue. Return the admitted context
// ID, or -1 if exhausted. Exhaustion changes neither the row nor the cursor.
// Only this thread consumes the queue, so the cursor needs no atomic operation.
// T is BF16 for context vectors or int64_t for a scalar embedding ID.
template <typename T>
__device__ __forceinline__ int get_new_input(
    T const *inputs, int num_contexts, int width,
    int *next_context_id, T *slot_input) {
  int context = *next_context_id;
  assert(context >= 0 && context <= num_contexts);
  if (context == num_contexts) {
    return -1;
  }
  assert(width > 0);
  size_t source = static_cast<size_t>(context) * width;
  for (int column = 0; column < width; column++) {
    slot_input[column] = inputs[source + column];
  }
  *next_context_id = context + 1;
  // The caller records this ID and retains advance_slot()'s fence/publication
  // sequence before workers can read the newly filled row.
  return context;
}

} // namespace runtime
} // namespace mirage
