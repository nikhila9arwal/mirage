# ContextIO demos on the ported NAS checkout

These offline demos let each reusable slot consume queued inputs independently.
They run all admissions and graph passes inside the persistent runtime. The
Python trial loop tests reset/reuse of the whole invocation.

## Row FFN

From this checkout:

```bash
export PATH="/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9/bin:$PATH"

srun --account=bgpd-dtai-gh --partition=ghx4-interactive --gres=gpu:1 \
  --time=00:05:00 --ntasks=1 timeout 240 \
  /u/elu1/venvs/reoalece/bin/python -B -u demo/context_io/ffn.py \
  --slots 8 --contexts 24 --width 512 --intermediate-width 256 \
  --mode hybrid_prelaunch --full-output --trace \
  --output /u/elu1/mpk_port_patch_d_20261022T020336Z/results/ffn_manual
```

Use a fresh output directory for each run. `--mode scheduler_dispatch` selects
the other resident mode. `--delay-us 3000 --delay-slots 1` adds a bounded device
delay before the selected slot's terminal completion; it requires `--trace`.

The graph is embedding → RMSNorm → gate/up projection → SiLU/multiply → down
projection, with one row per slot. Weights are shared read-only. By default a
checksum is saved; `--full-output` saves full vectors for numerical checking.

## Other demos

- `demo.py`: queued BF16 vectors through three RMSNorm stages.
- `embedding_checksum.py`: queued integer IDs, embedding, RMSNorm, optional
  row linear, and checksum output.
- [Attention/MoE demo](../../../experiments/integration/demo_attention.py):
  private attention caches and one or two MoE layers, with optional QKV/W_O
  batching. See the port's top-level README for its tested commands.

Trace records are collected on the device and exported after execution. Raw
records, the generated graph, numerical results, and nested Perfetto slices
are written to the output directory. Instrumented timings are diagnostic;
use tracing disabled for performance comparisons.

Supported here: single-GPU offline Hopper, BF16, finite inputs. The MoE demo
uses one scheduler and top-1 routing. These are toy model blocks; tokenizer,
prefill, live serving, and arbitrary model loading are outside this port.
