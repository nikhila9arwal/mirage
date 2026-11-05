#pragma once

// The existing ContextIO completion hook calls this after saving a context and
// before reusing its slot. Copies are diagnostic work, included in run time.
__device__ inline void compare_admit(int) {}

__device__ inline void compare_complete(int slot, int context) {
  auto *io = reinterpret_cast<uint64_t const *>(NAS_COMPARE_IO_ADDRESS);
  auto *copies = reinterpret_cast<uint64_t const *>(io[0]);
  reinterpret_cast<int32_t *>(io[2])[context] += 1;
  reinterpret_cast<int32_t *>(io[3])[context] = slot;
  for (uint64_t i = 0; i < io[1]; ++i) {
    auto *src = reinterpret_cast<uint16_t const *>(copies[3 * i]);
    auto *dst = reinterpret_cast<uint16_t *>(copies[3 * i + 1]);
    uint64_t words = copies[3 * i + 2];
    // Copy bits, including FP32 routing weights and int32 routing ranks.
    for (uint64_t j = 0; j < words; ++j)
      dst[size_t(context) * words + j] = src[size_t(slot) * words + j];
  }
}
