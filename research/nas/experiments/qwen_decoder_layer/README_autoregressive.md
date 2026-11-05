# Bounded greedy autoregressive demo

This isolated copy also supports real prompt preparation. See the [prompt-to-text workflow](../../README.md) for its commands. The fixture-based workflow below is retained for compatibility.

`autoregressive.py` runs the existing 48-layer Qwen3-30B-A3B graph, final norm, LM head, and argmax. Each slot keeps one sequence. Its argmax output becomes that slot's next embedding input inside the persistent runtime.

The initial condition is the existing cached HF prefix in `layers_48_02/reference.pt`: 16 slots, a 60-token prefix, and a seed token at position 60. Four passes consume positions 60–63 and save four predictions for positions 61–64. The last prediction is saved but not consumed again. The current one-page attention implementation therefore stays within its 64 cache positions.

This first demo has fixed step counts and does not stop on EOS or admit replacement sequences. It retains original kernel arithmetic. The host loop repeats whole invocations only to test reset/reuse; it does not launch once per generated token.

## Run NAS

From the isolated experiment root:

```bash
cd /work/hdd/bgpd/elu1/mpk_qwen_autoregressive_20261027T002756Z
srun --account=bgpd-dtai-gh --partition=ghx4-interactive --gres=gpu:1 \
  --ntasks=1 --cpus-per-task=8 --mem=64G --time=00:45:00 \
  timeout --signal=TERM --kill-after=15s 2400 \
  /u/elu1/venvs/reoalece/bin/python -B -u \
  experiments/qwen_decoder_layer/autoregressive.py \
  --steps 4 --trials 2 --trace --output verification/autoregressive_qwen
```

Use a fresh output directory for each run. Checkpoint loading and packing can take many minutes; `status.json` records the current phase and elapsed time. `--trace` produces `trial_*/trace.perfetto.json` and raw device records. Without tracing, the demo captures only consumed/predicted token IDs and the existing saved final hidden rows.

The ContextIO input allocation remains `[slots * steps, 1]` for compatibility. Only row `slot * steps` contains a seed. All later rows contain `-1`; reading one would be a feedback bug. The runtime instead saves each prediction into `generated_tokens` and copies that scalar into the slot input for its next pass.

## Independent HF comparison

Run `autoregressive_reference.py` in a separate GPU process. Keeping a complete HF model and packed NAS model resident together would exceed the intended VRAM budget.

```bash
srun --account=bgpd-dtai-gh --partition=ghx4-interactive --gres=gpu:1 \
  --ntasks=1 --cpus-per-task=8 --mem=64G --time=00:45:00 \
  timeout --signal=TERM --kill-after=15s 2400 \
  /u/elu1/venvs/reoalece/bin/python -B -u \
  experiments/qwen_decoder_layer/autoregressive_reference.py \
  --slots 16 --steps 4 --output verification/hf_greedy
```

Then add `--hf-reference verification/hf_greedy/reference.pt` to a fresh NAS run. Both start from identical seeds and prefix contents, then follow their own predicted tokens. No new checkpoint download is requested.

## What is checked

- The first consumed token equals the seed; every later consumed token exactly equals the previous generated token.
- Runtime token history equals an independent capture of the argmax output, including the last step.
- Every pass is saved once and belongs to the expected slot.
- Predicted IDs are in the vocabulary, unused input rows remain poisoned, and repeated invocations return identical tokens.
- The final KV cache preserves its prefix and unused future positions; appended positions are finite.
- Read-only weight and input hashes remain unchanged.

`report.json` uses `passed` for these runtime/feedback checks. Optional `hf_agreement` is a separate result. Once NAS and HF predict different tokens, their later input tokens and cache histories also differ; that is not by itself a failure of the feedback copy. Existing arithmetic mismatches remain relevant and are not removed by this patch. The diagnostic invocation time includes capture work and is not a production throughput measurement.
