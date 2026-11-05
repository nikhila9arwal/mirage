# NAS experiment harness

The runtime/compiler live at the repository root. This directory collects the
small programs that previously lived beside isolated `checkout/` copies:

- `experiments/native_prefill`: real prompts -> NAS prefill -> autoregressive
  decode -> offline replacement; an independent HF reference runs separately.
- `experiments/qwen_decoder_layer`: checkpoint loading, model construction,
  intermediate validation, and the earlier HF-prefix demos.
- `measurement`: matched whole-model timing and optional device diagnostics.
- `tests`: standalone selector/FIFO replay and its independent reference.
- `archive`: original DeltaAI launch commands retained for provenance.

No weights, cache tensors, compiled test binaries, or raw GPU traces are included.
Legacy HF-prefix demos and archived scripts contain site-specific default data
paths; pass their explicit path arguments on another machine.

## CPU tests

From the repository root, with a C++17 compiler on PATH:

```bash
python3 research/nas/tests/run_checks.py
```

`CXX` selects the compiler. Binaries go to ignored `research/nas/.build/`.
All four cache/check combinations run at P=1/R=1 and P=4/R=8, and their
selection digests must match. The tests include partial drains, reuse, and reset.
The runner also executes the measurement accounting unit tests. To compile the
small GPU replay with NVCC, without launching a GPU workload:

```bash
python3 research/nas/tests/run_checks.py --compile-gpu
```

`NVCC` can select a different compiler path. Run the resulting `fifo_gpu_*`
binaries on a Hopper GPU. Full-model CUDA generation remains part of the GPU
measurement harness; these standalone tests do not replace it.

## Model workload and measurements

Build/import this checkout's Mirage extension first. The checkpoint manifest
must identify Qwen/Qwen3-30B-A3B at revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` and point to its local snapshot.
The preparation helpers check model dimensions and provenance. Use explicit
data paths; historical data files are not copied into this repository.

```bash
mkdir -p research/nas/data research/nas/results
python3 research/nas/experiments/native_prefill/workload.py \
  --checkpoint /path/to/checkpoint.json --chunk-size 4 \
  --output research/nas/data/prompts.json
```

Run HF reference generation and NAS execution as separate GPU processes so
they do not hold both full models in GPU memory:

```bash
python3 research/nas/experiments/native_prefill/hf_reference.py \
  --workload research/nas/data/prompts.json \
  --output research/nas/results/hf_reference
```

`measurement/measure.py` requires `--workload`, `--hf-reference`, and
`--baseline-outputs`. The last is the preserved NAS `outputs.pt` containing
token IDs and lengths, not checkpoint weights; requiring it retains the
existing exact-output check. Inspect `--help` for variant choices. Results must
be below `research/nas/results`, which may be a symlink to scratch storage.
Each invocation creates a new output directory.

For the existing DeltaAI allocation, submit **from the repository root**:

```bash
export NAS_PYTHON=/path/to/environment/bin/python
export NAS_WORKLOAD=/path/to/prompts_64.json
export NAS_HF_REFERENCE=/path/to/hf_reference.json
export NAS_BASELINE_OUTPUTS=/path/to/preserved/control/outputs.pt
sbatch research/nas/measurement/run.sbatch
```

The launcher requests one GPU for 35 minutes, with 16 workers and one scheduler.
It drains at most 16 available completions per loop. The workload uses 16 slots,
64 prompts, 48 layers, top-8, chunk capacity 4, B=8, T=100 us, batch capacity 16,
and descriptor pool 16. Dense batching and companion policies remain off.
One checkpoint load is shared across the selected variants. Scheduler probes
are separate from uninstrumented throughput controls and can change batching.

## Preserved FIFO baseline

`NAS_SELECTOR_FIFO=1` enables incremental counts and FIFO membership;
`NAS_SELECTOR_HISTOGRAM=0` selects against the older rebuilt-histogram path.
Selection scans nonempty bins, then follows at most 16 member links. Eligibility
is the existing threshold-or-timeout rule. Equal ages break ties by operation
ID and expert; members leave in arrival order.

Original matched job 3529728 measured 114.685 s for histogram versus 68.693 s
for FIFO on the same GH200, with identical 663-token NAS output. These are
single invocations, not variance estimates. The FIFO diagnostic took 77.331 s;
choosing the winning bin remained the largest selection cost. Original raw
results remain on the source machine, outside Git.

## Optional winner-loop optimization

The latest commit adds two independent, compile-time opt-ins, both default off:

- `NAS_FIFO_CACHE_OLDEST=1`: keep each bin's head timestamp in `oldest[key]`.
  First append initializes it; partial dispatch updates it from the new head.
  The winner loop avoids repeatedly chasing head -> slot -> timestamp.
- `NAS_FIFO_SKIP_STATIC_CHECKS=1`: omit duplicate present/enabled checks in the
  winner loop. The existing readiness and host validation remain enabled.

No threshold, timeout, tie-breaking, publication, completion, or iteration
semantics change. The cache adds 192 KiB plus four alignment bytes for the
48-layer configuration. These switches require `NAS_SELECTOR_FIFO=1`.

| Selector name | Cache oldest | Skip duplicate checks |
|---|---:|---:|
| `fifo` | 0 | 0 |
| `fifo_cached` | 1 | 0 |
| `fifo_cached_unchecked` | 1 | 1 |

The launcher defaults to two uninstrumented controls (baseline and both
changes), followed by baseline/cached/both diagnostic invocations. To run
the baseline implementation only:

```bash
sbatch research/nas/measurement/run.sbatch baseline
```

The original comparison was submitted as job 3530459 from the isolated source
copy. Its GPU results are not claimed by this publishing snapshot. CPU replay
and CUDA compilation passed there. The preserved `nas-fifo-baseline` Git tag
and both switches set to zero provide independent rollback paths.
