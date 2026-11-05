"""Independent HF greedy decoding, outside the NAS persistent invocation.

Run this as a separate process: the HF model and packed NAS model each fit on
our GPU, but retaining both copies would not. Only small token artifacts are
saved. The existing cached HF prefix is shared as an initial condition; each
implementation then grows its own KV cache using its own predicted tokens.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

from reference import DEFAULT_CHECKPOINT, digest
from prepare_prompt import render_prompt_text, validate_prompt

DEFAULT_PREFIX = Path(
    '/work/hdd/bgpd/elu1/mpk_qwen_full_stack_20261026T045001Z/'
    'verification/layers_48_02/reference.pt')
FORMAT = 'nas_greedy_reference_v1'


def check_feedback(consumed, generated, seeds):
    """Check exact runtime feedback, independently of agreement with HF.

    consumed/generated are [slot, step]; seeds is [slot]. The last generated
    token is saved too, although the bounded run does not consume it again.
    """
    import torch
    if (consumed.dtype != torch.int64 or generated.dtype != torch.int64 or
            seeds.dtype != torch.int64 or consumed.ndim != 2 or
            generated.shape != consumed.shape or seeds.shape != consumed.shape[:1] or
            consumed.shape[0] < 1 or consumed.shape[1] < 1):
        raise ValueError('Expected int64 consumed/generated[S,T] and seeds[S], S,T>=1')
    consumed, generated, seeds = (x.detach().cpu() for x in (consumed, generated, seeds))
    expected = torch.cat((seeds[:, None], generated[:, :-1]), dim=1)
    mismatches = torch.nonzero(consumed != expected).tolist()
    nonnegative = bool((consumed >= 0).all() and (generated >= 0).all() and (seeds >= 0).all())
    return dict(passed=nonnegative and not mismatches, nonnegative_ids=nonnegative,
                slots=consumed.shape[0], steps=consumed.shape[1],
                mismatch_slot_steps=mismatches,
                rule='consumed[:,0]=seed; consumed[:,step+1]=generated[:,step]')


def prefix_fingerprint(initial_k, initial_v, prefix):
    """Hash only the preloaded prefix, excluding unused future cache positions."""
    if initial_k.ndim != 5 or initial_v.shape != initial_k.shape or not 0 <= prefix <= initial_k.shape[2]:
        raise ValueError('Expected matching K/V[layer,slot,position,head,dim] and valid prefix')
    return dict(initial_k=digest(initial_k[:, :, :prefix]),
                initial_v=digest(initial_v[:, :, :prefix]))


def validate_reference(artifact, seeds, *, steps, prefix, model, revision, prefix_sha256):
    """Reject a token reference from a different model, sequence, or KV prefix."""
    import torch
    expected = dict(format=FORMAT, slots=seeds.numel(), steps=steps, prefix=prefix,
                    layers=48, model=model, revision=revision, prefix_sha256=prefix_sha256)
    for name, value in expected.items():
        if artifact.get(name) != value:
            raise ValueError(f'Autoregressive reference {name} does not match this run')
    if seeds.dtype != torch.int64 or seeds.ndim != 1 or not torch.equal(artifact['seed_tokens'], seeds.cpu()):
        raise ValueError('Autoregressive reference seed tokens do not match this run')
    if not torch.equal(artifact['positions'], torch.arange(prefix, prefix+steps)):
        raise ValueError('Autoregressive reference positions do not match this run')
    feedback = check_feedback(artifact['consumed_tokens'], artifact['generated_tokens'], artifact['seed_tokens'])
    if (feedback['slots'], feedback['steps']) != (seeds.numel(), steps) or not feedback['passed']:
        raise ValueError('Autoregressive reference has invalid internal token feedback')


def compare_reference(generated, artifact):
    """Report greedy agreement separately; divergence is not a feedback failure."""
    import torch
    expected = artifact['generated_tokens']
    if generated.dtype != torch.int64 or generated.shape != expected.shape:
        raise ValueError('NAS and HF generated tokens must be int64 with matching [S,T] shape')
    same = generated.detach().cpu() == expected.cpu()
    first_difference = []
    for row in same:
        indices = torch.where(~row)[0]
        first_difference.append(int(indices[0]) if len(indices) else None)
    return dict(passed=bool(same.all()), matching_tokens=int(same.sum()),
                total_tokens=same.numel(), matching_sequences=int(same.all(dim=1).sum()),
                total_sequences=len(same), first_difference_step=first_difference,
                interpretation='HF and NAS follow their own predictions. After a token diverges, '
                               'later steps also have different inputs and KV history.')


def run_reference(output, *, cached_prefix=DEFAULT_PREFIX, checkpoint=DEFAULT_CHECKPOINT,
                  slots=16, steps=4):
    """Run all 48 HF layers at each step, then feed that step's argmax back.

    Prepare a text prompt separately with prepare_prompt.py. The cached fixture
    supplies its HF K/V prefix and the first token to consume at position prefix.
    No teacher-forced token after that first token is used.
    """
    import torch
    from transformers import AutoTokenizer, Qwen3MoeForCausalLM
    from transformers.cache_utils import DynamicCache

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def progress(phase, **details):
        record = dict(phase=phase, elapsed_seconds=time.monotonic()-started, **details)
        (output/'status.json').write_text(json.dumps(record, indent=2)+'\n')
        print(f'[{record["elapsed_seconds"]:.1f}s] {phase}: {details}', flush=True)

    progress('load_cached_prefix')
    data = torch.load(cached_prefix, map_location='cpu', weights_only=True, mmap=True)
    prompt = validate_prompt(data)
    manifest = json.loads(Path(checkpoint).read_text())
    for key in ('model', 'revision'):
        if data['provenance'][key] != manifest[key]:
            raise ValueError(f'Cached prefix and checkpoint {key} differ')
    prefix = data['prefix']
    if (data['layers'] != 48 or not 1 <= slots <= data['slots'] or
            not 1 <= steps <= 64-prefix):
        raise ValueError('Need a 48-layer prefix, available slots, and 1 <= steps <= 64-prefix')
    snapshot = Path(manifest['files']['config.json']).parent
    seeds = data['token_ids'][:slots, 0].clone().to(torch.int64)
    initial_k, initial_v = (data[name][:, :slots] for name in ('initial_k', 'initial_v'))
    hashes = prefix_fingerprint(initial_k, initial_v, prefix)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()

    progress('load_hf_model')
    # The local loader streams checkpoint shards into the model's CUDA storage.
    # It avoids constructing a second NAS-weight copy or downloading anything.
    model = Qwen3MoeForCausalLM.from_pretrained(
        snapshot, local_files_only=True, dtype=torch.bfloat16,
        device_map={'': 'cuda'}, attn_implementation='eager').eval()
    if model.config.num_hidden_layers != 48 or model.config.sliding_window is not None:
        raise ValueError('This reference expects the current 48-layer full-attention checkpoint')

    with torch.inference_mode():
        progress('initialize_private_reference_cache')
        cache = DynamicCache(config=model.config)
        for layer in range(48):
            # Saved NAS-style prefix layout is [slot,position,kv_head,dim]. HF
            # expects [slot,kv_head,position,dim]. Both are already post-RoPE K.
            k = initial_k[layer, :, :prefix].transpose(1, 2).contiguous().to('cuda')
            v = initial_v[layer, :, :prefix].transpose(1, 2).contiguous().to('cuda')
            cache.update(k, v, layer)
        consumed = torch.empty((slots, steps), dtype=torch.int64, device='cuda')
        generated = torch.empty_like(consumed)
        token = seeds.to('cuda')
        for step in range(steps):
            position = prefix + step
            if any(cache.get_seq_length(layer) != position for layer in range(48)):
                raise RuntimeError('HF cache length does not match the next token position')
            consumed[:, step] = token
            result = model(input_ids=token[:, None], past_key_values=cache,
                           cache_position=torch.tensor([position], device='cuda'),
                           use_cache=True, logits_to_keep=1)
            logits = result.logits[:, -1]
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError('HF reference produced nonfinite logits')
            token = logits.argmax(-1)
            generated[:, step] = token
            progress('decode_step_complete', step=step+1, position=position)
        consumed, generated = consumed.cpu(), generated.cpu()

    feedback = check_feedback(consumed, generated, seeds)
    if not feedback['passed']:
        raise RuntimeError('Independent HF reference failed its own feedback check')
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    text = tokenizer.batch_decode(generated, skip_special_tokens=False)
    # EOS trimming is presentation only; every fixed-step token remains above.
    prompt_display = render_prompt_text(tokenizer, data, generated)
    artifact = dict(format=FORMAT, slots=slots, steps=steps, prefix=prefix, layers=48,
        model=manifest['model'], revision=manifest['revision'], prefix_sha256=hashes,
        seed_tokens=seeds, consumed_tokens=consumed, generated_tokens=generated,
        generated_text=text, prompt=prompt, provenance=data['provenance'], prompt_display=prompt_display,
        positions=torch.arange(prefix, prefix+steps),
        position_convention='Consumed-token position; its generated token predicts position+1',
        source_prefix=str(Path(cached_prefix).resolve()), source_checkpoint=str(Path(checkpoint).resolve()))
    torch.save(artifact, output/'reference.pt')
    report = dict(feedback=feedback, generated_tokens=generated.tolist(),
        consumed_tokens=consumed.tolist(), generated_text=text,
        prompt=prompt, provenance=data['provenance'], prompt_display=prompt_display,
        job=os.getenv('SLURM_JOB_ID'), gpu=str(torch.cuda.get_device_properties(0)),
        transformers=importlib.metadata.version('transformers'), torch=torch.__version__,
        elapsed_seconds=time.monotonic()-started, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        model=manifest['model'], revision=manifest['revision'], prefix_sha256=hashes,
        scope='Independent HF greedy decode from the saved prefix; fixed steps; no EOS stopping')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    if prompt_display is not None:
        print(f'Prompt: {prompt_display["prompt_text"]}', flush=True)
        for slot, continuation in enumerate(prompt_display['continuation_text']):
            print(f'Slot {slot} continuation: {continuation}', flush=True)
    progress('finished', generated_tokens=generated.numel())
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cached-prefix', type=Path, default=DEFAULT_PREFIX)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--slots', type=int, default=16)
    parser.add_argument('--steps', type=int, default=4)
    args = parser.parse_args()
    run_reference(args.output, cached_prefix=args.cached_prefix, checkpoint=args.checkpoint,
                  slots=args.slots, steps=args.steps)


if __name__ == '__main__':
    main()
