# Async NAS F1

Research implementation of asynchronous request-slot progress and MoE batching
on Mirage's resident persistent runtime. This repository preserves Mirage's Git
history and Apache-2.0 license. The original README is retained in
[docs/MIRAGE_UPSTREAM_README.md](docs/MIRAGE_UPSTREAM_README.md).

The current Hopper path runs the real 48-layer Qwen3-30B-A3B checkpoint with
top-8 routing, chunked prefill, autoregressive decode, EOS stopping, and offline
request replacement. Each request has a private, single-page KV cache with a
64-position prompt-plus-decode limit. Full prefill chunks use four token rows;
decode uses one active row. This is an offline research prototype, not an online
serving interface.

Slots advance independently. Ready MoE token/branch members enter bins grouped
by operation/layer and expert. A single controller selects a physical batch of
at most 16 rows; logical completion joins all branches before releasing the
next operation. The current prefill harness leaves dense QKV/W_O batching off.

## Start here

- [NAS experiments, tests, and reproduction](research/nas/README.md)
- [Source provenance and rollback](research/nas/PROVENANCE.md)
- [Runtime](include/mirage/persistent_kernel/experimental_moe_runtime.cuh)
- [FIFO selector](include/mirage/persistent_kernel/experimental_moe_fifo.h)
- [Slot preparation/restart](include/mirage/persistent_kernel/resident_slot_restart.cuh)
- [Compiler](src/kernel/runtime_resident.cc)

Build Mirage from this checkout using [INSTALL.md](INSTALL.md), including the
pinned submodules (`git submodule update --init --recursive`). The tested GPU
environment uses Hopper/GH200, CUDA 12.9, and BF16. Model checkpoints, installed
packages, compiled extensions, traces, and raw run outputs are not committed.

The source snapshot has passed GPU checks in its original experiment directory.
Packaging it here does not constitute a new full-model GPU validation. Known
NAS/HF arithmetic differences remain: the preserved 64-prompt test produced
663 NAS tokens and matched 60/64 HF sequences exactly. See the NAS experiment
documentation before interpreting performance or numerical results.
