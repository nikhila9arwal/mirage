#!/usr/bin/env python3
"""Independent HF continuation and shared request-level verification helpers.

Both HF and NAS consume the same prepared prefix and seed, then follow their
own predictions until EOS or the common 64-position cache limit. Exact model
agreement is reported separately from feedback/lifecycle correctness.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

from autoregressive_reference import check_feedback
from prepare_requests import validate_requests, request_fingerprints, LAYERS
from reference import DEFAULT_CHECKPOINT, PAGE

FORMAT = 'nas_request_queue_reference_v1'


def check_requests(consumed, generated, seeds, lengths, prefixes, eos, vocab):
    """Validate actual per-request outputs; padding is never treated as work."""
    import torch
    count = seeds.numel()
    if (consumed.dtype != torch.int64 or generated.dtype != torch.int64 or
            consumed.shape != generated.shape or tuple(generated.shape) != (count, PAGE) or
            tuple(lengths.shape) != (count,) or tuple(prefixes.shape) != (count,)):
        raise ValueError('Expected consumed/generated[N,64] with seeds/lengths/prefixes[N]')
    consumed, generated, seeds, lengths, prefixes = (
        value.detach().cpu() for value in (consumed, generated, seeds, lengths, prefixes))
    rows = []
    for request in range(count):
        length, prefix = int(lengths[request]), int(prefixes[request])
        bounded = 1 <= length <= PAGE-prefix
        row = dict(request=request, prefix_length=prefix, generated_length=length,
                   length_in_bounds=bounded, feedback=False, ids_in_vocabulary=False,
                   no_tokens_after_eos=False, finished_by_eos_or_capacity=False)
        if bounded:
            tokens = generated[request, :length]
            row.update(feedback=check_feedback(consumed[request:request+1, :length],
                generated[request:request+1, :length], seeds[request:request+1])['passed'],
                ids_in_vocabulary=bool(((tokens >= 0) & (tokens < vocab)).all()),
                no_tokens_after_eos=bool((tokens[:-1] != eos).all()),
                finished_by_eos_or_capacity=int(tokens[-1]) == eos or prefix+length == PAGE,
                stop_reason='eos' if int(tokens[-1]) == eos else 'capacity')
        row['passed'] = all(row[name] for name in ('length_in_bounds', 'feedback',
            'ids_in_vocabulary', 'no_tokens_after_eos', 'finished_by_eos_or_capacity'))
        rows.append(row)
    return dict(passed=all(row['passed'] for row in rows), requests=rows)


def render_requests(tokenizer, data, generated, lengths):
    result = []
    for request, prompt in enumerate(data['prompts']):
        tokens = generated[request, :int(lengths[request])].tolist()
        visible = tokens[:-1] if tokens and tokens[-1] == data['eos_token_id'] else tokens
        result.append(dict(request=request, prompt=prompt['text'],
            prefix_length=int(data['prefix_lengths'][request]), generated_length=len(tokens),
            token_ids=tokens, raw_text=tokenizer.decode(tokens, skip_special_tokens=False),
            continuation=tokenizer.decode(visible, skip_special_tokens=True),
            full_text=tokenizer.decode(prompt['token_ids']+visible, skip_special_tokens=True)))
    return result


def validate_reference(artifact, data, fingerprints):
    import torch
    expected = dict(format=FORMAT, model=data['provenance']['model'],
        revision=data['provenance']['revision'], capacity=PAGE,
        eos_token_id=data['eos_token_id'], prefix_sha256=fingerprints)
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f'Request reference {key} does not match this input queue')
    for key in ('seed_tokens', 'prefix_lengths'):
        if not torch.equal(artifact[key], data[key]):
            raise ValueError(f'Request reference {key} differs')
    checks = check_requests(artifact['consumed_tokens'], artifact['generated_tokens'],
        artifact['seed_tokens'], artifact['generated_lengths'], artifact['prefix_lengths'],
        data['eos_token_id'], data['prompts'][0]['vocab_size'])
    if not checks['passed']:
        raise ValueError('HF reference has invalid feedback or stopping behavior')


def compare_requests(generated, lengths, artifact):
    """Compare token positions, counting a missing token as a disagreement."""
    rows = []
    for request in range(len(lengths)):
        actual = generated[request, :int(lengths[request])].tolist()
        expected = artifact['generated_tokens'][request, :int(artifact['generated_lengths'][request])].tolist()
        matching = sum(a == b for a, b in zip(actual, expected))
        differences = [i for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
        first = differences[0] if differences else (min(len(actual), len(expected)) if len(actual) != len(expected) else None)
        rows.append(dict(request=request, matching_tokens=matching,
            compared_positions=max(len(actual), len(expected)), exact_sequence=actual == expected,
            nas_length=len(actual), hf_length=len(expected), first_difference_step=first))
    return dict(passed=all(row['exact_sequence'] for row in rows),
        matching_tokens=sum(row['matching_tokens'] for row in rows),
        compared_positions=sum(row['compared_positions'] for row in rows),
        matching_sequences=sum(row['exact_sequence'] for row in rows), requests=rows,
        interpretation='HF and NAS follow their own predictions; after the first divergence '
                       'their later inputs and KV histories differ.')


def decode_request(model, data, request, *, device='cuda'):
    """Decode one reference request; its cache cannot alias another request."""
    import torch
    from transformers.cache_utils import DynamicCache
    prefix = int(data['prefix_lengths'][request])
    cache = DynamicCache(config=model.config)
    if prefix:
        for layer in range(LAYERS):
            k = data['initial_k'][layer, request:request+1, :prefix].transpose(1, 2).contiguous().to(device)
            v = data['initial_v'][layer, request:request+1, :prefix].transpose(1, 2).contiguous().to(device)
            cache.update(k, v, layer)
    consumed, generated = [], []
    token = data['seed_tokens'][request:request+1].to(device)
    for step in range(PAGE-prefix):
        position = prefix+step
        if any(cache.get_seq_length(layer) != position for layer in range(LAYERS)):
            raise RuntimeError('HF cache length does not match the next position')
        consumed.append(int(token.item()))
        result = model(input_ids=token[:, None], past_key_values=cache,
            cache_position=torch.tensor([position], device=device), use_cache=True, logits_to_keep=1)
        logits = result.logits[:, -1]
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError('HF reference produced nonfinite logits')
        token = logits.argmax(-1)
        generated.append(int(token.item()))
        if generated[-1] == data['eos_token_id']:
            break
    return consumed, generated


def run_reference(output, *, requests, checkpoint=DEFAULT_CHECKPOINT):
    import torch
    from transformers import AutoTokenizer, Qwen3MoeForCausalLM
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def save(name, value):
        (output/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    def progress(phase, **details):
        status = dict(phase=phase, elapsed_seconds=time.monotonic()-started, **details)
        save('status.json', status)
        print(f'[{status["elapsed_seconds"]:.1f}s] {phase}: {details}', flush=True)
    try:
        progress('load_request_queue')
        data = torch.load(requests, map_location='cpu', weights_only=True, mmap=True)
        count = validate_requests(data)
        manifest = json.loads(Path(checkpoint).read_text())
        if any(data['provenance'][key] != manifest[key] for key in ('model', 'revision')):
            raise ValueError('Request prefixes and checkpoint differ')
        snapshot = Path(manifest['files']['config.json']).parent
        fingerprints = request_fingerprints(data)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        progress('load_hf_model')
        model = Qwen3MoeForCausalLM.from_pretrained(snapshot, local_files_only=True,
            dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='eager').eval()
        consumed = torch.full((count, PAGE), -1, dtype=torch.int64)
        generated = torch.full_like(consumed, -1)
        lengths = torch.zeros(count, dtype=torch.int32)
        with torch.inference_mode():
            for request in range(count):
                inputs, outputs = decode_request(model, data, request)
                length = len(outputs)
                consumed[request, :length] = torch.tensor(inputs)
                generated[request, :length] = torch.tensor(outputs)
                lengths[request] = length
                progress('request_complete', request=request, prefix=int(data['prefix_lengths'][request]),
                    generated_length=length, stopped_by_eos=outputs[-1] == data['eos_token_id'])
        checks = check_requests(consumed, generated, data['seed_tokens'], lengths,
            data['prefix_lengths'], data['eos_token_id'], data['prompts'][0]['vocab_size'])
        if not checks['passed']:
            raise RuntimeError('HF feedback/stopping checks failed')
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        texts = render_requests(tokenizer, data, generated, lengths)
        artifact = dict(format=FORMAT, model=manifest['model'], revision=manifest['revision'],
            capacity=PAGE, eos_token_id=data['eos_token_id'], prefix_sha256=fingerprints,
            seed_tokens=data['seed_tokens'], prefix_lengths=data['prefix_lengths'],
            consumed_tokens=consumed, generated_tokens=generated, generated_lengths=lengths,
            texts=texts, source_requests=str(Path(requests).resolve()))
        torch.save(artifact, output/'reference.pt')
        save('report.json', dict(passed=checks['passed'], checks=checks, texts=texts,
            generated_lengths=lengths.tolist(), generated_tokens=generated.tolist(),
            elapsed_seconds=time.monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            job=os.getenv('SLURM_JOB_ID'), gpu=str(torch.cuda.get_device_properties(0)),
            scope='Independent HF requests, stopping at EOS or prefix+generated=64'))
        for row in texts:
            print(f'Request {row["request"]}: {row["continuation"]}', flush=True)
        progress('finished', generated_tokens=int(lengths.sum()))
        return artifact
    except Exception:
        save('failure.json', dict(error=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run_reference(args.output, requests=args.requests, checkpoint=args.checkpoint)


if __name__ == '__main__':
    main()
