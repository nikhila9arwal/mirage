# Offline replacement requests with variable prefixes

This harness runs the existing 48-layer Qwen decode graph with reusable slots.
HF prepares every prompt except its final token. That final token seeds NAS;
each subsequent input comes from the preceding NAS prediction. A request stops
when it predicts the checkpoint EOS token or its cached prefix plus generated
length reaches 64. There is no separate maximum-output-length array.

The default workload is `verification/request_prompts.json`: 20 independent
prompts and 16 slots. The pinned tokenizer produces prefix lengths 18–30.
Request admission occurs inside the persistent invocation; the Python harness
does not launch one kernel per token or replacement request.

From the experiment root:

```bash
/u/elu1/venvs/reoalece/bin/python \
  experiments/qwen_decoder_layer/prepare_requests.py \
  --prompts verification/request_prompts.json --tokenize-only \
  --output verification/my_tokenize_check

sbatch --partition=ghx4-interactive verification/request_demo.sbatch
```

Inside an existing GPU allocation, run:

```bash
bash verification/run_request_demo.sh verification/my_request_run
```

The run script launches three separate processes: HF prefix preparation, an
independent HF continuation, then NAS. This keeps only one model allocation
on the GPU at a time. The NAS stage executes twice to verify reset/reuse.
The script uses timeouts for every stage and stops if one fails.

To use other prompts, create a JSON list of strings and pass its path as the
second argument to `run_request_demo.sh`, or submit with
`--export=ALL,TASK_PROMPT_FILE=/absolute/path/prompts.json`. Prompts are neither
truncated nor padded; a templated prompt must leave at least one position
within the 64-token cache. Initial prompt lengths can differ between requests.

## Storage and runtime API

The prepared `prefix/requests.pt` artifact contains:

- `prefix_lengths`: int32 `[requests]`, excluding the seed token.
- `seed_tokens`: int64 `[requests]`, the final token of each templated prompt.
- `initial_k/v`: BF16 `[48, requests, 64, 4, 128]`, immutable HF prefixes.
- Per-request prompt metadata and pinned checkpoint provenance.

Output buffers reserve 64 rows per request. Row `request * 64 + step` belongs
to that request even if another slot completes work first. Only the first
`generated_lengths[request]` rows are valid; unused rows remain poisoned in
the diagnostic run. The length includes EOS. Displayed text omits EOS.

The binding is:

```python
pk.eos_token_id = data['eos_token_id']
pk.compile(
    context_io=(inputs, saved_hidden, slot_ids, final_hidden),
    context_autoregressive=(slot_predictions, generated_history),
    context_requests=(prefix_lengths, generated_lengths, cache_bindings),
    output_dir=output,
)
```

`cache_bindings` has one tuple per layer:
`(slot_k, slot_v, prepared_k, prepared_v)`. The slot tensors retain their fixed
addresses. Each attention worker initializes its own head's prefix when that
request first reaches the layer. Slot epochs remain runtime generations;
request-generated lengths determine token positions.

## Evidence and limits

Each NAS `trial_N/report.json` records:

- Exact seed/feedback behavior and EOS/capacity stopping.
- Valid rows saved exactly once; unused rows untouched.
- One stable owning slot per request and observed replacement when requests
  exceed slots.
- Final prefix contents and finite decoded positions for each slot's final
  request. Values beyond its visible cache length may remain from older work.
- Repeated token and length agreement, plus request-by-request HF comparison.

`trial_N/outputs.pt` stores the actual consumed IDs, predictions, lengths,
visits, owner slots, and saved hidden outputs. The run hashes all input seeds,
prefixes, prefix lengths, and weights before and after execution.

Exact HF token agreement is reported separately from lifecycle correctness.
The previously identified model arithmetic discrepancy remains relevant; HF
and NAS can diverge and then follow different subsequent token/KV histories.
The sequential HF reference is a correctness reference, not a throughput
comparison. NAS timings include diagnostic capture work, with task tracing
disabled. This harness does not currently export a replacement-aware Perfetto
trace; the preserved fixed-pass exporter assumes consecutive context IDs.

CPU checks:

```bash
/u/elu1/venvs/reoalece/bin/python verification/test_request_queue.py
```

They cover actual request-output indexing and stopping contracts, and mock
small model/cache behavior to exercise prefix preparation and HF decoding.
They do not substitute for the GPU runtime verification.
