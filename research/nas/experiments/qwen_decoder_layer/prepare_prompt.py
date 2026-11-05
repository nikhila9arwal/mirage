#!/usr/bin/env python3
"""Prepare a real chat prompt for the existing 16-slot autoregressive demo.

HF prefills every prompt token except the last. That last token is the seed
consumed at position P by both decoders; their first predictions follow it.
Run --tokenize-only on the CPU before reserving a GPU. Checkpoint files are
read locally, and every invocation writes to a new output directory.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

from reference import DEFAULT_CHECKPOINT, D, KV_HEADS, PAGE

FORMAT = 'nas_prompt_prefix_v1'
MODEL = 'Qwen/Qwen3-30B-A3B'
REVISION = 'ad44e777bcd18fa416d9da3bd8f70d33ebb85d39'
SLOTS, LAYERS = 16, 48
TEMPLATE_KWARGS = dict(add_generation_prompt=True, enable_thinking=False)


def _checkpoint(checkpoint):
    manifest = json.loads(Path(checkpoint).read_text())
    if (manifest['model'], manifest['revision']) != (MODEL, REVISION):
        raise ValueError('Prompt preparation requires the pinned Qwen3-30B-A3B checkpoint')
    snapshot = Path(manifest['files']['config.json']).parent
    if snapshot.name != REVISION:
        raise ValueError('Checkpoint snapshot directory must match the pinned revision')
    config = json.loads((snapshot / 'config.json').read_text())
    if (config['num_hidden_layers'], config['num_key_value_heads'], config['head_dim']) != (LAYERS, KV_HEADS, D):
        raise ValueError('Checkpoint cache shape must be 48 layers, 4 KV heads, and head dimension 128')
    if config.get('sliding_window') is not None:
        raise ValueError('Prompt preparation requires full attention')
    return manifest, snapshot, config


def tokenize_prompt(text, *, steps=16, checkpoint=DEFAULT_CHECKPOINT):
    """Return JSON-safe prompt metadata, without loading weights or using CUDA."""
    from transformers import AutoTokenizer

    if not isinstance(text, str) or type(steps) is not int or not 1 <= steps <= PAGE:
        raise ValueError('Expected prompt text and an integer steps in 1..64')
    _, snapshot, config = _checkpoint(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        [dict(role='user', content=text)], tokenize=False, **TEMPLATE_KWARGS)
    # The template already contains its special tokens. Add none a second time,
    # and never silently truncate or pad a prompt to fit the fixed cache page.
    ids = tokenizer(rendered, add_special_tokens=False, truncation=False,
                    padding=False)['input_ids']
    prefix = len(ids) - 1
    if not ids or prefix + steps > PAGE:
        raise ValueError(f'Prompt has {len(ids)} tokens: prefix={prefix} + steps={steps} '
                         f'must be <= {PAGE}; shorten the prompt or reduce steps (no truncation)')
    eos = config.get('eos_token_id', tokenizer.eos_token_id)
    eos = eos if isinstance(eos, list) else ([] if eos is None else [eos])
    generation_config = snapshot / 'generation_config.json'
    if generation_config.is_file():
        extra = json.loads(generation_config.read_text()).get('eos_token_id', [])
        eos += extra if isinstance(extra, list) else ([] if extra is None else [extra])
    return dict(text=text, rendered=rendered, token_ids=ids,
                template_kwargs=dict(TEMPLATE_KWARGS), eos_token_ids=sorted(set(eos)),
                vocab_size=config['vocab_size'])


def validate_prompt(data):
    """Check the prompt fixture on the CPU; legacy fixtures return None.

    This checks metadata, seed IDs, and cache shapes without scanning the large
    cache tensors or importing a model. The preparation checks cache finiteness
    while copying the much smaller single-sequence HF prefix.
    """
    if data.get('format') != FORMAT:
        if 'prompt' in data or str(data.get('format', '')).startswith('nas_prompt_prefix'):
            raise ValueError('Unknown prompt prefix format')
        return None
    import torch

    prompt = data.get('prompt')
    if not isinstance(prompt, dict):
        raise ValueError('Prompt fixture requires prompt metadata')
    if not isinstance(prompt.get('text'), str) or not isinstance(prompt.get('rendered'), str):
        raise ValueError('Prompt text and rendered template must be strings')
    if prompt.get('template_kwargs') != TEMPLATE_KWARGS:
        raise ValueError('Prompt must use add_generation_prompt=True and enable_thinking=False')
    vocab = prompt.get('vocab_size')
    ids, eos = prompt.get('token_ids'), prompt.get('eos_token_ids')
    if (type(vocab) is not int or vocab < 1 or not isinstance(ids, list) or not ids or
            any(type(token) is not int or not 0 <= token < vocab for token in ids)):
        raise ValueError('Prompt token_ids must be a nonempty list of in-vocabulary integer IDs')
    if not isinstance(eos, list) or any(type(token) is not int or not 0 <= token < vocab for token in eos):
        raise ValueError('Prompt eos_token_ids must be in-vocabulary integer IDs')
    if (data.get('slots'), data.get('layers')) != (SLOTS, LAYERS):
        raise ValueError('Prompt fixture requires 16 slots and 48 layers')
    prefix, steps = data.get('prefix'), data.get('steps')
    if (type(prefix) is not int or prefix != len(ids)-1 or type(steps) is not int or
            not 1 <= steps <= PAGE-prefix):
        raise ValueError('Prompt prefix must equal token count minus one, with prefix+steps<=64')
    seed = data.get('token_ids')
    if (not isinstance(seed, torch.Tensor) or seed.dtype != torch.int64 or
            tuple(seed.shape) != (SLOTS, 1) or seed.device.type != 'cpu' or
            not bool((seed == ids[-1]).all())):
        raise ValueError('Prompt token_ids must be int64[16,1], repeating only the final prompt token')
    for name in ('initial_k', 'initial_v'):
        value = data.get(name)
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16 or
                tuple(value.shape) != (LAYERS, SLOTS, PAGE, KV_HEADS, D) or value.device.type != 'cpu'):
            raise ValueError(f'Prompt {name} must be CPU BF16[48,16,64,4,128]')
    provenance = data.get('provenance', {})
    if (provenance.get('model'), provenance.get('revision')) != (MODEL, REVISION):
        raise ValueError('Prompt provenance must identify the pinned checkpoint')
    return prompt


def render_prompt_text(tokenizer, data, generated):
    """Return readable text per slot; raw fixed-length outputs stay with callers.

    Only this display stops at the first EOS. Decoding the combined IDs keeps
    tokenizer spacing intact in full_text, including the rendered chat prompt.
    """
    prompt = validate_prompt(data)
    if prompt is None:
        return None
    rows = generated.tolist() if hasattr(generated, 'tolist') else generated
    if not isinstance(rows, list) or not rows or not all(isinstance(row, list) for row in rows):
        raise ValueError('Generated token IDs must have [slot,step] shape')
    continuation, full = [], []
    for row in rows:
        if any(type(token) is not int or not 0 <= token < prompt['vocab_size'] for token in row):
            raise ValueError('Generated token IDs must be in-vocabulary integers')
        end = next((i for i, token in enumerate(row) if token in prompt['eos_token_ids']), len(row))
        visible = row[:end]
        continuation.append(tokenizer.decode(visible, skip_special_tokens=True))
        full.append(tokenizer.decode(prompt['token_ids'] + visible, skip_special_tokens=True))
    return dict(prompt_text=prompt['text'], rendered_prompt=prompt['rendered'],
                continuation_text=continuation, full_text=full)


def copy_prefill(initial_k, initial_v, cache, prefix):
    """Copy HF [1,head,P,dim] into private host [layer,slot,P,head,dim] rows."""
    import torch

    for layer in range(initial_k.shape[0]):
        for destination, source in ((initial_k, cache.layers[layer].keys),
                                    (initial_v, cache.layers[layer].values)):
            row = source.detach().cpu().transpose(1, 2)
            expected = (1, prefix, destination.shape[3], destination.shape[4])
            if tuple(row.shape) != expected or row.dtype != torch.bfloat16 or not bool(torch.isfinite(row).all()):
                raise ValueError(f'HF layer {layer} has an invalid or nonfinite prefix cache')
            # copy_ broadcasts the single prompt into actual independent rows;
            # an expanded tensor view must never become the saved NAS cache.
            destination[layer, :, :prefix].copy_(row)


def prepare_prompt(text, output, *, steps=16, checkpoint=DEFAULT_CHECKPOINT,
                   tokenize_only=False, prompt_file=None):
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
        progress('tokenize_prompt')
        manifest, snapshot, _ = _checkpoint(checkpoint)
        provenance = dict(model=manifest['model'], revision=manifest['revision'],
            source_checkpoint=str(Path(checkpoint).resolve()), source_snapshot=str(snapshot.resolve()),
            source_prompt_file=str(Path(prompt_file).resolve()) if prompt_file else None,
            source=str(Path(__file__).resolve()), local_files_only=True,
            source_policy='Checkpoint, tokenizer, prompt file, and baseline artifacts are read-only',
            dtype='bfloat16', attention='eager', prefill_sequences=1, slots=SLOTS, layers=LAYERS,
            prefill='HF prefills all but the final prompt token; copy into private host cache rows',
            versions={name: importlib.metadata.version(name) for name in ('torch', 'transformers')})
        save('provenance.json', provenance)
        prompt = tokenize_prompt(text, steps=steps, checkpoint=checkpoint)
        prefix = len(prompt['token_ids'])-1
        save('prompt.json', prompt)
        progress('tokenized', prompt_tokens=len(prompt['token_ids']), prefix=prefix, steps=steps,
                 seed_token_id=prompt['token_ids'][-1])
        if tokenize_only:
            progress('finished', tokenized=True)
            return prompt

        # Import and load the model only after the CPU capacity check succeeds.
        import torch
        from transformers import Qwen3MoeForCausalLM
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        progress('load_hf_model')
        model = Qwen3MoeForCausalLM.from_pretrained(
            snapshot, local_files_only=True, dtype=torch.bfloat16,
            device_map={'': 'cuda'}, attn_implementation='eager').eval()
        initial_k = torch.full((LAYERS, SLOTS, PAGE, KV_HEADS, D), .125, dtype=torch.bfloat16)
        initial_v = torch.full_like(initial_k, -.25)
        with torch.inference_mode():
            if prefix:
                progress('prefill_prompt', prefix=prefix)
                ids = torch.tensor([prompt['token_ids'][:-1]], device='cuda', dtype=torch.int64)
                result = model(input_ids=ids, use_cache=True, logits_to_keep=1)
                copy_prefill(initial_k, initial_v, result.past_key_values, prefix)
        data = dict(format=FORMAT, slots=SLOTS, layers=LAYERS, prefix=prefix, steps=steps,
            token_ids=torch.full((SLOTS, 1), prompt['token_ids'][-1], dtype=torch.int64),
            initial_k=initial_k, initial_v=initial_v, prompt=prompt, provenance=provenance)
        validate_prompt(data)
        torch.save(data, output / 'prefix.pt')
        progress('finished', prepared=True, peak_cuda_bytes=torch.cuda.max_memory_allocated())
        return data
    except Exception as error:
        save('failure.json', dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        progress('failed', error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt')
    source.add_argument('--prompt-file', type=Path)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True, help='New directory for status and prefix.pt')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--tokenize-only', action='store_true', help='CPU capacity check; save metadata without weights or prefix.pt')
    args = parser.parse_args()
    text = args.prompt_file.read_text() if args.prompt_file is not None else args.prompt
    prepare_prompt(text, args.output, steps=args.steps, checkpoint=args.checkpoint,
                   tokenize_only=args.tokenize_only, prompt_file=args.prompt_file)


if __name__ == '__main__':
    main()
