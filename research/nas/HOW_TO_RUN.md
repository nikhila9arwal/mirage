# How to run asynchronous binned MPK

Use the programs under `research/nas` for our runtime. Running an upstream
`demo/qwen3` script does not automatically enable independent slot restart,
ContextIO, or the binned controller.

This guide covers the `nas-async` checkout reviewed on 2026-11-09, including its
existing local packed-loader/FIFO changes. It separately identifies the dense
branch and online prototype. For the implementation, read the
[reviewer guide](REVIEWER_GUIDE.md).

## Choose the run you need

| Goal | Entry point | GPU / assets |
|---|---|---|
| Check FIFO selection and accounting | [tests/run_checks.py](tests/run_checks.py) | CPU; C++17 compiler, PyTorch, safetensors |
| Validate full native prefill/decode for the first time | [native_prefill/run.py](experiments/native_prefill/run.py) | Hopper GPU, checkpoint manifest and tokenized workload; optional HF reference |
| Measure FIFO/hot-record performance | [measurement/measure.py](measurement/measure.py) | Also requires matching HF reference and preserved NAS outputs; optional packed weights |
| Compare QKV/W_O bins | Same measurement script on `feature/dense-prefill-bins-20261108` | Additional dense-mode options on that branch |
| Submit live requests through native MPK's engine | Separate `online_serving/serve.py` prototype | See [online status](#online-serving-status); adapter not in this checkout |

## 1. Select the source and Python environment

From your existing clone, or after cloning the private repository:

```bash
cd /u/elu1/Async-NAS-F1                 # Replace on another machine.
export NAS_REPO="$PWD"
export NAS_PYTHON=/u/elu1/venvs/reoalece/bin/python
git branch --show-current
git rev-parse HEAD
git status --short
```

The tested full-model path is BF16 on Hopper/GH200, compute capability 9.0.
The research environment inspected for this guide has Python 3.12,
PyTorch `2.10.0+cu129`, Transformers `4.57.1`, safetensors `0.8.0`, and
huggingface-hub `0.36.2`; prior runs used CUDA 12.9. These are provenance, not
instructions to replace a working environment. Use enough GPU memory for the
roughly 60 GB BF16 checkpoint plus caches, activations, and queues. The full
model experiments used GH200 GPUs with roughly 95 GiB visible memory.

Check the import **before** a long allocation:

```bash
export PYTHONPATH="$NAS_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
"$NAS_PYTHON" -B - <<'PY'
import os
from pathlib import Path
import mirage, mirage.core
root = Path(os.environ['NAS_REPO']).resolve()
for module in (mirage, mirage.core):
    path = Path(module.__file__).resolve()
    print(path)
    assert path.is_relative_to(root), f'Wrong checkout: {path}'
PY
```

Both the package and compiled `core` extension must belong to this checkout.
At documentation time, the existing virtualenv without this path selection
imported `/u/elu1/reoalece/mirage`; this publishing checkout had no built `core`
extension. Setting `PYTHONPATH` selects source but does not build that extension.
The harness also checks import provenance and rejects a different checkout.

If a build is needed, use the **source-build** instructions in
[INSTALL.md](../../INSTALL.md), with this fork and its pinned submodules:

```bash
git submodule update --init --recursive
"$NAS_PYTHON" -m pip install -e . -v
```

Use the existing CUDA/C++/Rust/Z3 build environment. The upstream PyPI wheel is
not a substitute for this fork. Recheck imports after building. C++ compiler
changes under `src/` require rebuilding the extension; generated runtime CUDA
is compiled by `pk.compile()` during each fresh harness run. Binary extensions
must also match the node's CPU architecture and Python ABI.

## 2. Quick checks without a model run

```bash
"$NAS_PYTHON" -B research/nas/tests/run_checks.py
"$NAS_PYTHON" -B research/nas/measurement/measure.py --help
```

The replay checks partial FIFO drains, reuse/reset, and selection equivalence
across metadata variants. `CXX` selects the compiler. To compile the additional
GPU replays required by the supplied Slurm launcher:

```bash
"$NAS_PYTHON" -B research/nas/tests/run_checks.py --compile-gpu
```

This invokes NVCC but does not launch GPU work. `NVCC` can select its path.
Outputs are ignored under `research/nas/.build/`. These tests do not load Qwen
or verify full-model CUDA execution.

## 3. Run the prepared 64-request measurement

This is the shortest route when the research fixtures already exist. All four
inputs must describe the same checkpoint/workload:

| Variable | Required content |
|---|---|
| `NAS_WORKLOAD` | Tokenized 64-prompt JSON, chunk size 4, including checkpoint manifest path and workload hash |
| `NAS_HF_REFERENCE` | Independent HF `reference.json` for that exact workload hash |
| `NAS_BASELINE_OUTPUTS` | Preserved NAS `outputs.pt` with `tokens` and `lengths`; not checkpoint weights or HF outputs |
| `NAS_PACKED_WEIGHTS` | Optional complete `nas_qwen_packed_v1` cache directory, including `manifest.json` |

For example, these are existing DeltaAI assets, outside Git:

```bash
export NAS_ASSETS=/work/hdd/bgpd/elu1/nas_packed_hot_integration_20261106T165229Z
export NAS_WORKLOAD="$NAS_ASSETS/verification/prompts_64.json"
export NAS_HF_REFERENCE="$NAS_ASSETS/verification/hf_reference.json"
export NAS_BASELINE_OUTPUTS="$NAS_ASSETS/results/job_3540345/measurement/control_fifo_hot/outputs.pt"
export NAS_PACKED_WEIGHTS=/work/hdd/bgpd/elu1/nas_hot_bins_weights_20261105T043850Z/packed_weights
```

Use your own paths elsewhere. These files are deliberately not committed.
The packed loader reuses already transformed NAS weights and verifies their
identity and bytes. Its cache-creation tools remain outside this repository.
Omit `--packed-weights` to load the checkpoint directly; omit `--compare-loader`
for normal timing. Loading and hash verification can be slow and variable.

Submit **from the repository root**, after compiling the GPU replays:

```bash
cd "$NAS_REPO"
sbatch research/nas/measurement/run.sbatch hot
```

The checked-in launcher requests one GPU, eight CPU cores, 96 GB host RAM,
and 35 minutes on `ghx4,ghx4-interactive`, using account `bgpd-dtai-gh`.
Override site settings with `sbatch` options on another cluster. The argument
`hot` runs:

1. Small fast/hot GPU FIFO replays.
2. `control_fifo_cached_unchecked` and `control_fifo_hot` without scheduler/task probes.
3. The matching two diagnostic invocations.

`baseline` runs the original FIFO control and its diagnostic. The default
`compare` compares the cached/check variants. These arguments are experiments,
not different models. A short allocation can expire during loading or compilation;
35 minutes is the launcher's requested bound, not a model-latency estimate.

To run just one control and one diagnostic interactively instead:

```bash
cd "$NAS_REPO"
mkdir -p research/nas/results
export NAS_RUN="$NAS_REPO/research/nas/results/manual_$(date +%Y%m%d_%H%M%S)"
srun --account=bgpd-dtai-gh --partition=ghx4-interactive --gres=gpu:1 \
  --ntasks=1 --cpus-per-task=8 --mem=96G --time=00:35:00 \
  timeout --signal=TERM --kill-after=30s 2040 \
  "$NAS_PYTHON" -B -u research/nas/measurement/measure.py \
  --workload "$NAS_WORKLOAD" --hf-reference "$NAS_HF_REFERENCE" \
  --baseline-outputs "$NAS_BASELINE_OUTPUTS" \
  --packed-weights "$NAS_PACKED_WEIGHTS" \
  --variants control_fifo_hot diagnostic_fifo_hot \
  --completion-budget 16 --budget-seconds 2000 --output "$NAS_RUN"
```

Remove the packed-weights argument if no complete cache is available. Do not
run `srun` inside an already allocated shell merely to obtain a second GPU;
there, execute the Python command directly with the same timeout.

The output directory must be new and below the resolved
`research/nas/results` directory. That directory may be a symlink to scratch
for `measure.py`; retain existing results rather than replacing them.

## 4. What this measurement actually runs

| Setting | Current measurement value |
|---|---|
| Model | Qwen3-30B-A3B, 48 layers; H=2048, I=768, 128 experts, top-8 |
| Requests / state slots / workers / schedulers | 64 / 16 / 16 / 1 |
| Prefill / decode | P=4 full prompt chunks, then one active token row per slot pass |
| Cache | Private per slot and layer; 64 consumed token positions |
| Graph/runtime | `resident_data` / `scheduler_dispatch`, experimental batching enabled |
| MoE eligibility | B=8 useful members or T=100 **microseconds**, not 100 ms |
| Physical capacity / descriptor pool | 16 selected members / 16 live descriptor slots |
| Dense batching on `nas-async` | Off for this chunked-prefill graph |
| Companions / cohort gating | Off |
| Completion drain | CLI default 1; commands above and Slurm launcher explicitly use 16 |

The harness sets these environment/compile options itself. Exporting a different
`MIRAGE_MOE_TIMEOUT_US` will not override the value assigned inside `measure.py`.
There is no `--slots` or `--layers` option on that measurement entry point.
Changing only a Python slot count is insufficient: graph geometry, compiled
state capacity, queue bounds, probes, and host validation must agree.

`control_fifo_hot` preserves oldest-eligible selection and FIFO member order.
The word `unchecked` in its comparison control means redundant static
present/enabled checks were omitted in the winner loop; it does not disable all
slot invariants or ownership validation.

## 5. Read results and check progress

```bash
squeue -u "$USER"
scontrol show job JOB_ID
tail -n 40 nas-fifo-winner-JOB_ID.log
```

The Slurm output is `research/nas/results/job_JOB_ID/measurement/`. For a direct
run, use `NAS_RUN` above. Inspect:

| Artifact | Meaning |
|---|---|
| `status.json`, `failure.json` | Last completed phase or concrete failure; distinguish loading, generation, compilation, execution, validation |
| `environment.json`, `workload.json` | Actual source imports, GPU, geometry, policy, and inputs |
| `VARIANT/compile_command.json` | Actual CUDA flags; confirm FIFO, hot records, completion budget, P/R/layers |
| `VARIANT/native_graph.json`, `mapping.json`, `graph_validation.json` | Logical graph and task/data/slot mapping |
| `REPORT.md`, `summary.json`, `VARIANT/report.json` | Invocation timing, validation, and diagnostic summaries |
| `VARIANT/outputs.pt` | Saved token IDs, lengths, and hidden outputs |
| Diagnostic `measurements.npz`, `requests.csv`, `tokens.csv`, `passes.csv` | Raw device records and request/pass timing |
| Diagnostic `selector.json`, `selector_timings.csv`, `dispatch_timings.csv` | Selection cost and dispatch outcomes, including unsuccessful attempts |
| `loading.json` with packed loading | Separate load/hash measurements |

Generated CUDA and task-graph files are also copied to each variant directory.
Check `report.json`'s `passed` and exact-output fields, not just process exit or
plausible text. NAS/HF agreement is reported separately from runtime correctness.

Use uninstrumented controls for invocation throughput. Diagnostics record TTFT,
TPOT, scheduler work, batch fill, and worker execution; the probes can change
batch formation. This entry point writes raw measurements and reports, **not
an automatic Perfetto export**, and has no `--trace` flag.

Definitions used by the analyzer:

- **TTFT from admission:** first saved generated token minus slot admission.
- **TTFT from invocation:** includes the offline request's wait for a free slot.
  All requests in this fixture exist at launch.
- **Per-request mean TPOT:** `(last token time - first token time)/(length - 1)`;
  undefined for one-token outputs, and EOS is included.
- **Worker-seconds:** sum of measured task execution durations over workers.
  This is not invocation latency or whole-GPU utilization.
- **Scheduler scopes:** `complete_logical` is nested in completion handling;
  selector phases are nested in dispatch. Do not sum nested scopes twice.

Loading, compiling, hashing, and post-run analysis are outside the GPU invocation
timer. ContextIO hidden-output copies remain inside the runtime. The entire
Slurm allocation can therefore be much longer than the reported inference time.

## 6. Prepare inputs and a first NAS reference

Skip this section if you already have the matching assets in section 3.

The pinned checkpoint is `Qwen/Qwen3-30B-A3B` at revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`. A checkpoint manifest includes:

```json
{
  "model": "Qwen/Qwen3-30B-A3B",
  "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
  "files": {
    "config.json": "/path/to/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json"
  }
}
```

The directory must contain the tokenizer/config files, safetensors index, and
checkpoint shards. The helper expects the snapshot directory's name to be the
pinned revision. [prepare_checkpoint.py](experiments/qwen_decoder_layer/prepare_checkpoint.py)
can fetch missing weight shards **after** that manifest/index exist:

```bash
export NAS_CHECKPOINT=/path/to/checkpoint.json
"$NAS_PYTHON" -B research/nas/experiments/qwen_decoder_layer/prepare_checkpoint.py \
  --checkpoint "$NAS_CHECKPOINT" --layers 48 --include-readout
```

Prepare the default 64 short prompts on the CPU:

```bash
mkdir -p research/nas/data research/nas/results
export NAS_WORKLOAD="$NAS_REPO/research/nas/data/prompts_64.json"
"$NAS_PYTHON" -B research/nas/experiments/native_prefill/workload.py \
  --checkpoint "$NAS_CHECKPOINT" --chunk-size 4 --output "$NAS_WORKLOAD"
```

The generator applies the Qwen chat template with thinking disabled. Default
prompts get short explicit suffixes to fit complete chunks. With `--prompts
questions.json`, supply a JSON list of strings; supplied prompts are not silently
modified, truncated, or padded. Templated lengths must fit 64 positions and be
divisible by 4. This older offline fixture requires full chunks.

On an allocated Hopper GPU, run HF reference generation, then NAS as **separate
processes** so two full models are not resident simultaneously:

```bash
export NAS_STAMP=$(date +%Y%m%d_%H%M%S)
export NAS_HF_DIR="$NAS_REPO/research/nas/results/hf_reference_$NAS_STAMP"
"$NAS_PYTHON" -B -u research/nas/experiments/native_prefill/hf_reference.py \
  --workload "$NAS_WORKLOAD" --batch-size 8 --output "$NAS_HF_DIR"
export NAS_HF_REFERENCE="$NAS_HF_DIR/reference.json"

export NAS_CORRECTNESS="$NAS_REPO/research/nas/results/first_nas_$NAS_STAMP"
"$NAS_PYTHON" -B -u research/nas/experiments/native_prefill/run.py \
  --workload "$NAS_WORKLOAD" --hf-reference "$NAS_HF_REFERENCE" \
  --slots 16 --trials 1 --batch-rows 8 --timeout-us 100 \
  --output "$NAS_CORRECTNESS"
```

These are GPU commands to use inside an allocation with an external timeout,
as in section 3. The first correctness run uses the original loader, completion
captures, and its existing selector configuration; it is not the hot-FIFO timing
control. It builds K/V in NAS from the full prompts, not an HF-prefilled cache.
It does not take `--packed-weights`. Its older output-path check requires a
resolved directory within `research/nas`; unlike `measure.py`, it does not allow
that directory to resolve outside the checkout through a results symlink.

Only after the correctness report passes, adapt its output schema for the
measurement harness (values are copied unchanged):

```bash
export NAS_BASELINE_OUTPUTS="$NAS_REPO/research/nas/data/nas_baseline_$NAS_STAMP.pt"
"$NAS_PYTHON" -B - <<'PY'
import json, os
from pathlib import Path
import torch
source = Path(os.environ['NAS_CORRECTNESS'])
assert json.loads((source / 'report.json').read_text())['passed']
raw = torch.load(source / 'trial_0/outputs.pt', map_location='cpu', weights_only=True)
with Path(os.environ['NAS_BASELINE_OUTPUTS']).open('xb') as f:
    torch.save({'tokens': raw['history'], 'lengths': raw['generated_lengths']}, f)
PY
```

Then use section 3's measurement command. Never point `--baseline-outputs` at
the current run's output to bypass its regression check. The HF reference is an
independent numerical comparison, not the source of NAS generated tokens.

## 7. Dense bins: use the dense branch

The main-branch chunked model rejects enabled dense bins. Its older dense
adapter assumes synthetic shapes; the dense branch supplies actual Qwen QKV
2048→5120 and W_O 4096→2048 geometry and per-token joins.

Preserve an existing dirty checkout. For a separate review/run checkout:

```bash
git worktree add --detach ../Async-NAS-F1-dense-review 3ac9c8d
cd ../Async-NAS-F1-dense-review
export NAS_REPO="$PWD"
```

Build/check imports there as in section 1. On that branch, the measurement
command accepts these additional arguments:

```bash
--dense-modes none qkv wo both --dense-batch-rows 8 --dense-timeout-us 100
```

For example, inside a GPU allocation, with the same three reference assets:

```bash
timeout --signal=TERM --kill-after=30s 2640 \
  "$NAS_PYTHON" -B -u research/nas/measurement/measure.py \
  --workload "$NAS_WORKLOAD" --hf-reference "$NAS_HF_REFERENCE" \
  --baseline-outputs "$NAS_BASELINE_OUTPUTS" --packed-weights "$NAS_PACKED_WEIGHTS" \
  --variants control_fifo_hot diagnostic_fifo_hot \
  --dense-modes none qkv wo both --dense-batch-rows 8 --dense-timeout-us 100 \
  --completion-budget 16 --budget-seconds 2600 \
  --output "research/nas/results/dense_$(date +%Y%m%d_%H%M%S)"
```

This is eight invocations, not one. A dense-mode list marks QKV/W_O consistently
even for the `none` control. Dense B/T settings remain separate from MoE B/T.
There is still one expert-zero bin per dense operation/layer, not one bin shared
across all layers. The first measured dense comparison was slower; enabling
the feature is not an established performance win.

## Online serving status

As of this review, bounded live serving is in this isolated checkout:

```text
/work/hdd/bgpd/elu1/nas_online_serving_20261108T184514Z/checkout
```

Its instructions are in
`research/nas/experiments/online_serving/README.md` **inside that checkout**.
An example launch, using its existing local checkpoint/cache defaults:

```bash
cd /work/hdd/bgpd/elu1/nas_online_serving_20261108T184514Z/checkout
srun --account=bgpd-dtai-gh --partition=ghx4-interactive --gres=gpu:1 \
  --ntasks=1 --cpus-per-task=4 --time=00:20:00 \
  timeout --signal=INT --kill-after=30s 1100 \
  /u/elu1/venvs/reoalece/bin/python -B -u \
  research/nas/experiments/online_serving/serve.py \
  --output "/work/hdd/bgpd/elu1/nas_online_serving_20261108T184514Z/results/server_$(date +%Y%m%d_%H%M%S)" \
  --max-requests 64 --ring-capacity 8 --dense-batching both \
  --host 127.0.0.1 --port 8000
```

The native engine here provides host submission/completion transport and HTTP
handlers; the GPU executes the experimental NAS controller and graph. Sessions
have a finite accepted-request budget. This is not unrestricted long-running
serving. The prototype also handles prompt tails using single-row passes and
requires host output acknowledgement before slot reuse. Use its `smoke.py`
and README for the live-arrival/idle/replacement checks. Do not assume merely
selecting `mode='online_pinned'` in the offline main-branch harness supplies
this integration.

## Common problems

| Symptom | Check |
|---|---|
| Wrong imported checkout / missing `mirage.core` | Build this source and verify both paths; a virtualenv alone does not select a checkout |
| Missing replay binary in the Slurm log | Run `tests/run_checks.py --compile-gpu` first |
| Unknown `--dense-modes` | You are on `nas-async`, not the dense branch |
| Chunk capacity requires dense batching disabled | Same branch mismatch; environment variables alone cannot port the feature |
| Workload/reference mismatch | Keep the tokenized JSON and matching HF hash together; the workload embeds a checkpoint path |
| Missing baseline `tokens`/`lengths` | Use measurement output format or the explicit conversion in section 6 |
| Existing output directory / output-path assertion | Choose a new path under the harness's allowed results root |
| `NVCC_*` or slot-iteration setting rejected | Unset `NVCC_PREPEND_FLAGS`, `NVCC_APPEND_FLAGS`, and `MIRAGE_RESIDENT_SLOT_ITERATIONS` for these controlled runs |
| `Insufficient allocation time` | Earlier loading/compilation consumed the budget; completed variants remain saved but the comparison is incomplete |
| Task trace/probe capacity exceeded | Treat that diagnostic as incomplete; increase its capacity for a separate run rather than interpreting truncated records |

For this documentation change, command-line help, source bindings, and links
were checked. No new GPU benchmark, compiler rebuild, or model download was
performed; archived successful runs remain separate evidence.
