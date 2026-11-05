#!/usr/bin/env python3
"""Prepare a finite offline queue of independently prefilled text requests.

Each request keeps its own prompt length and immutable HF prefix. NAS slots
are reusable execution storage, so the number of requests can exceed slots.
The seed is the final prompt token; decoding stops at EOS or 64 cached tokens.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

from prepare_prompt import _checkpoint, copy_prefill, tokenize_prompt, MODEL, REVISION
from reference import DEFAULT_CHECKPOINT, D, KV_HEADS, PAGE, digest

FORMAT = 'nas_request_queue_prefix_v1'
LAYERS = 48


def tokenize_requests(texts, checkpoint=DEFAULT_CHECKPOINT):
    """Check all prompt lengths before loading model weights or touching CUDA."""
    if not isinstance(texts, list) or not texts or any(not isinstance(t, str) for t in texts):
        raise ValueError('Expected a nonempty JSON list of prompt strings')
    # One available decode position is sufficient. Every request then receives
    # the entire remaining cache capacity; there is no per-request token limit.
    prompts = [tokenize_prompt(text, steps=1, checkpoint=checkpoint) for text in texts]
    eos = prompts[0]['eos_token_ids']
    if len(eos) != 1 or any(prompt['eos_token_ids'] != eos for prompt in prompts):
        raise ValueError('This first request queue uses one checkpoint EOS token ID')
    return prompts, eos[0]


def validate_requests(data):
    """Validate the CPU artifact layout without scanning the large KV tensors."""
    import torch
    if data.get('format') != FORMAT or data.get('layers') != LAYERS or data.get('capacity') != PAGE:
        raise ValueError('Expected a 48-layer request queue with cache capacity 64')
    prompts = data.get('prompts')
    if not isinstance(prompts, list) or not prompts:
        raise ValueError('Request queue must contain prompt metadata')
    count = len(prompts)
    prefix, seeds = data.get('prefix_lengths'), data.get('seed_tokens')
    for value, dtype, name in ((prefix, torch.int32, 'prefix_lengths'), (seeds, torch.int64, 'seed_tokens')):
        if (not isinstance(value, torch.Tensor) or value.dtype != dtype or
                value.device.type != 'cpu' or tuple(value.shape) != (count,)):
            raise ValueError(f'{name} must be a CPU {dtype}[requests] tensor')
    eos = data.get('eos_token_id')
    if type(eos) is not int or eos < 0:
        raise ValueError('Request queue needs one nonnegative EOS token ID')
    for request, prompt in enumerate(prompts):
        ids, vocab = prompt.get('token_ids'), prompt.get('vocab_size')
        if (not isinstance(ids, list) or not ids or type(vocab) is not int or
                any(type(token) is not int or not 0 <= token < vocab for token in ids) or
                prompt.get('eos_token_ids') != [eos] or eos >= vocab):
            raise ValueError(f'Invalid token metadata for request {request}')
        if not isinstance(prompt.get('text'), str) or not isinstance(prompt.get('rendered'), str):
            raise ValueError(f'Invalid prompt text for request {request}')
        if int(prefix[request]) != len(ids)-1 or not 0 <= int(prefix[request]) < PAGE:
            raise ValueError(f'Invalid cached-prefix length for request {request}')
        if int(seeds[request]) != ids[-1]:
            raise ValueError(f'Request {request} seed must be the final prompt token')
    for name in ('initial_k', 'initial_v'):
        value = data.get(name)
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16 or
                value.device.type != 'cpu' or not value.is_contiguous() or
                tuple(value.shape) != (LAYERS, count, PAGE, KV_HEADS, D)):
            raise ValueError(f'{name} must be contiguous CPU BF16[48,requests,64,4,128]')
    provenance = data.get('provenance', {})
    if (provenance.get('model'), provenance.get('revision')) != (MODEL, REVISION):
        raise ValueError('Request queue provenance must identify the pinned checkpoint')
    return count


def request_fingerprints(data):
    """Match reference results by request and actual prefix contents."""
    return [dict(k=digest(data['initial_k'][:, r, :int(prefix)]),
                 v=digest(data['initial_v'][:, r, :int(prefix)]))
            for r, prefix in enumerate(data['prefix_lengths'])]


def prefill_requests(model, prompts, *, device='cuda', progress=lambda *a, **k: None):
    """Run requests separately, reusing the existing HF-to-NAS cache copier.

    Separate forwards avoid introducing padding or variable-length HF batch
    metadata into this first offline implementation. They are preparation work,
    outside the measured NAS invocation.
    """
    import torch
    count = len(prompts)
    initial_k = torch.full((LAYERS, count, PAGE, KV_HEADS, D), .125, dtype=torch.bfloat16)
    initial_v = torch.full_like(initial_k, -.25)
    with torch.inference_mode():
        for request, prompt in enumerate(prompts):
            prefix = len(prompt['token_ids'])-1
            progress('prefill_request', request=request, prefix=prefix)
            if prefix:
                ids = torch.tensor([prompt['token_ids'][:-1]], device=device, dtype=torch.int64)
                result = model(input_ids=ids, use_cache=True, logits_to_keep=1)
                # Select one actual destination request, so no prefix is shared
                # or broadcast into another request's saved cache allocation.
                copy_prefill(initial_k[:, request:request+1], initial_v[:, request:request+1],
                             result.past_key_values, prefix)
                del result
    return initial_k, initial_v


def prepare_requests(texts, output, *, checkpoint=DEFAULT_CHECKPOINT, tokenize_only=False, prompt_file=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = dict(prepared=False, tokenize_only=tokenize_only, job=os.getenv('SLURM_JOB_ID'))
    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    def progress(phase, **details):
        status.update(phase=phase, elapsed_seconds=time.monotonic()-started, **details)
        save('status.json', status)
        print(f'[{status["elapsed_seconds"]:.1f}s] {phase}: {details}', flush=True)
    try:
        progress('tokenize_requests')
        manifest, snapshot, _ = _checkpoint(checkpoint)
        prompts, eos = tokenize_requests(texts, checkpoint)
        prefixes = [len(prompt['token_ids'])-1 for prompt in prompts]
        provenance = dict(model=manifest['model'], revision=manifest['revision'],
            source_checkpoint=str(Path(checkpoint).resolve()), source_snapshot=str(snapshot.resolve()),
            source_prompt_file=str(Path(prompt_file).resolve()) if prompt_file else None,
            dtype='bfloat16', attention='eager', local_files_only=True,
            prefill='Each HF prompt independently, excluding its final seed token',
            versions={name: importlib.metadata.version(name) for name in ('torch', 'transformers')})
        save('prompts.json', prompts)
        save('provenance.json', provenance)
        progress('tokenized', requests=len(prompts), prefix_lengths=prefixes,
                 generation_capacities=[PAGE-p for p in prefixes], eos_token_id=eos)
        if tokenize_only:
            progress('finished', tokenized=True)
            return dict(prompts=prompts, prefix_lengths=prefixes, eos_token_id=eos)
        import torch
        from transformers import Qwen3MoeForCausalLM
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        progress('load_hf_model')
        model = Qwen3MoeForCausalLM.from_pretrained(snapshot, local_files_only=True,
            dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='eager').eval()
        initial_k, initial_v = prefill_requests(model, prompts, progress=progress)
        data = dict(format=FORMAT, layers=LAYERS, capacity=PAGE,
            prefix_lengths=torch.tensor(prefixes, dtype=torch.int32),
            seed_tokens=torch.tensor([p['token_ids'][-1] for p in prompts], dtype=torch.int64),
            initial_k=initial_k, initial_v=initial_v, prompts=prompts,
            eos_token_id=eos, provenance=provenance)
        validate_requests(data)
        torch.save(data, output/'requests.pt')
        progress('finished', prepared=True, peak_cuda_bytes=torch.cuda.max_memory_allocated())
        return data
    except Exception:
        save('failure.json', dict(error=traceback.format_exc(), status=status))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts', type=Path, required=True, help='JSON list of prompt strings')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--tokenize-only', action='store_true')
    args = parser.parse_args()
    prepare_requests(json.loads(args.prompts.read_text()), args.output, checkpoint=args.checkpoint,
                     tokenize_only=args.tokenize_only, prompt_file=args.prompts)


if __name__ == '__main__':
    main()
