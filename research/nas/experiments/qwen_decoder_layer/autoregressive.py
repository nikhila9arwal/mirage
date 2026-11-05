#!/usr/bin/env python3
"""Bounded greedy Qwen decoding inside one NAS persistent invocation.

The cached fixture supplies each slot's KV prefix and its first input token.
Every later input comes from that slot's own argmax output. Host Python does
not launch one graph per token. Captures and optional traces are diagnostics.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import demo
from autoregressive_reference import (
    DEFAULT_PREFIX, check_feedback, compare_reference, prefix_fingerprint,
    validate_reference,
)
from reference import DEFAULT_CHECKPOINT, digest
from prepare_prompt import render_prompt_text, validate_prompt

SESSION = Path(__file__).resolve().parents[2]
ROOT = SESSION.parents[1]  # Repository root; research/nas holds the harness.


def load_weights(torch, config, snapshot, owners, progress):
    """Pack one HF layer at a time, so only one model stays on the GPU."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding
    from reference import _load_tensors, _load_layer, _pack_layer

    with torch.inference_mode():
        rotary = Qwen3MoeRotaryEmbedding(config).to('cuda')
        shape_only = torch.empty((1, 64, config.hidden_size), device='cuda', dtype=torch.bfloat16)
        cos, sin = rotary(shape_only, torch.arange(64, device='cuda')[None, :])
        cos, sin = cos[0].contiguous(), sin[0].contiguous()
        weights = dict(embedding=_load_tensors(snapshot, owners, ['model.embed_tokens.weight'])[
            'model.embed_tokens.weight'].to('cuda'), layers=[])
        for layer in range(48):
            progress('load_and_pack_layer', layer=layer + 1, completed_layers=layer)
            block = _load_layer(config, layer, snapshot, owners)
            weights['layers'].append(_pack_layer(block, cos, sin))
            del block
            torch.cuda.empty_cache()
            progress('layer_ready', completed_layers=layer + 1)
        raw = _load_tensors(snapshot, owners, ['model.norm.weight', 'lm_head.weight'])
        weights['readout'] = dict(norm=raw['model.norm.weight'].to('cuda'),
                                  lm_head=raw['lm_head.weight'].to('cuda'))
    return weights


def check_final_cache(torch, buffers, data, prefix, steps):
    """Check the final cache boundaries without capturing every large KV row."""
    checks = []
    for layer in range(48):
        for kind in ('k', 'v'):
            actual = buffers[f'layer{layer}_{kind}_cache'].cpu()
            initial = data['initial_' + kind][layer]
            checks.append(dict(layer=layer, kind=kind,
                prefix_unchanged=torch.equal(actual[:, :prefix], initial[:, :prefix]),
                future_unchanged=torch.equal(actual[:, prefix + steps:], initial[:, prefix + steps:]),
                appended_positions_finite=bool(torch.isfinite(actual[:, prefix:prefix + steps]).all())))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cached-prefix', type=Path, default=DEFAULT_PREFIX)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--batch-rows', type=int, default=4)
    parser.add_argument('--timeout-us', type=int, default=100)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--hf-reference', type=Path,
                        help='Optional artifact produced by autoregressive_reference.py in a separate process')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not (1 <= args.steps <= 64 and 1 <= args.trials <= 3 and
            1 <= args.batch_rows <= 16 and 0 <= args.timeout_us <= 1000000):
        parser.error('Use 1..64 steps, 1..3 trials, B=1..16, T=0..1000000 us')
    if os.environ.get('MIRAGE_COMPARE_SEQUENCE_REPEATS', '1') != '1':
        parser.error('This demo keeps one sequence in each slot; sequence replacement is disabled')
    out = args.output.resolve()
    if SESSION not in out.parents:
        parser.error('Keep output in this isolated experiment directory')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = dict(generated=False, compiled=False, executed=False, validated=False,
                  job=os.getenv('SLURM_JOB_ID'))

    def progress(phase, **details):
        status.update(phase=phase, elapsed_seconds=time.monotonic() - started, **details)
        demo.save_json(out / 'status.json', status)
        print(f'[{status["elapsed_seconds"]:.1f}s] {phase}: {details}', flush=True)

    progress('imports')
    sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'demo/context_io')]
    os.environ.update(MIRAGE_HOME=str(ROOT), MIRAGE_ROOT=str(ROOT),
        MIRAGE_TASK_GRAPH_MODE='resident_data', MIRAGE_RESIDENT_EXECUTION_MODE='scheduler_dispatch',
        MIRAGE_MOE_BATCH_ROWS=str(args.batch_rows), MIRAGE_MOE_TIMEOUT_US=str(args.timeout_us),
        MIRAGE_MOE_EARLY_COALESCE='0', MIRAGE_MOE_DEADLINE_COALESCE='0', MIRAGE_DENSE_BATCHING='none')
    if 'MIRAGE_RESIDENT_SLOT_ITERATIONS' in os.environ:
        parser.error('Unset MIRAGE_RESIDENT_SLOT_ITERATIONS; fixed sequences bound this run')
    pk = None
    try:
        import torch
        import mirage as mi
        import mirage.core
        from transformers import AutoTokenizer, Qwen3MoeConfig
        from reference import selected_experts
        torch.cuda.set_device(0)
        if torch.cuda.get_device_capability(0) != (9, 0):
            raise RuntimeError('This demo requires the tested Hopper GPU path')
        if str(ROOT) not in mi.__file__ or str(ROOT) not in mirage.core.__file__:
            raise RuntimeError('The imported Mirage package must belong to this isolated checkout')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()

        progress('load_cached_prefix')
        data = torch.load(args.cached_prefix, map_location='cpu', weights_only=True, mmap=True)
        prompt = validate_prompt(data)
        # The existing MoE implementation requires a multiple of 16 state
        # slots. This first demo uses the already verified 16-slot prefix.
        slots, layers, prefix = data['slots'], data['layers'], data['prefix']
        if (slots, layers) != (16, 48) or not 0 <= prefix <= 64 - args.steps:
            raise ValueError('Need the 16-slot, 48-layer fixture and prefix+steps<=64')
        manifest = json.loads(args.checkpoint.read_text())
        for name in ('model', 'revision'):
            if data['provenance'][name] != manifest[name]:
                raise ValueError(f'Cached prefix and checkpoint {name} differ')
        snapshot = Path(manifest['files']['config.json']).parent
        owners = json.loads((snapshot / 'model.safetensors.index.json').read_text())['weight_map']
        config = Qwen3MoeConfig.from_json_file(str(snapshot / 'config.json'))
        config._attn_implementation = 'eager'
        if (config.hidden_size, config.moe_intermediate_size, config.num_experts,
            config.num_experts_per_tok, config.num_hidden_layers, config.head_dim,
            config.num_key_value_heads, config.num_attention_heads) != (2048, 768, 128, 8, 48, 128, 4, 32):
            raise ValueError('Checkpoint shape does not match the existing Qwen graph builder')
        seeds = data['token_ids'][:, 0].clone().to(torch.int64)
        prefix_hashes = prefix_fingerprint(data['initial_k'], data['initial_v'], prefix)
        hf = None
        if args.hf_reference is not None:
            hf = torch.load(args.hf_reference, map_location='cpu', weights_only=True)
            validate_reference(hf, seeds, steps=args.steps, prefix=prefix,
                model=manifest['model'], revision=manifest['revision'], prefix_sha256=prefix_hashes)
        demo.save_json(out / 'provenance.json', dict(model=manifest['model'], revision=manifest['revision'],
            cached_prefix=str(args.cached_prefix.resolve()), prefix_sha256=prefix_hashes,
            seed_tokens=seeds.tolist(), slots=slots, steps=args.steps, prefix=prefix, layers=layers,
            prompt=prompt, provenance=data['provenance'],
            input_positions=list(range(prefix, prefix + args.steps)),
            generated_token_positions=list(range(prefix + 1, prefix + args.steps + 1))))
        demo.save_json(out / 'environment.json', dict(root=str(ROOT), mirage=mi.__file__,
            core=mirage.core.__file__, python=sys.executable, torch=torch.__version__,
            gpu=str(torch.cuda.get_device_properties(0)), node=os.uname().nodename,
            job=os.getenv('SLURM_JOB_ID'), workers=16, schedulers=1,
            mode='resident_data/scheduler_dispatch', slots=slots, layers=layers))
        weights = load_weights(torch, config, snapshot, owners, progress)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        contexts = slots * args.steps
        # Preserve the current ContextIO shape and pass IDs. Only each slot's
        # first row is a seed; every unused teacher-forcing row is invalid.
        inputs = torch.full((slots, args.steps), -1, device='cuda', dtype=torch.int64)
        inputs[:, 0].copy_(seeds)
        inputs = inputs.reshape(contexts, 1)
        generated = torch.empty((contexts, 1), device='cuda', dtype=torch.int64)
        saved_hidden = torch.empty((contexts, demo.H), device='cuda', dtype=torch.bfloat16)

        spec = importlib.util.spec_from_file_location('autoregressive_context_demo', ROOT / 'demo/context_io/demo.py')
        context_demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(context_demo)
        pk = context_demo.make_kernel(mi, torch, slots=slots, workers=16, metadata_pages=128)
        pk.num_local_schedulers = 1
        progress('generate_native_graph')
        buffers = demo.build_graph(pk, torch, weights, slots, layers, readout=True)
        graph = json.loads(pk.generate_task_graph()['json_file'])
        pk.set_request_slots(demo.assign_slots(graph, buffers, slots))
        graph = json.loads(pk.generate_task_graph()['json_file'])
        mapping, graph_report = demo.validate_graph(graph, slots, layers, readout=True)
        for name, value in (('native_graph.json', graph), ('mapping.json', mapping),
                            ('graph_validation.json', graph_report)):
            demo.save_json(out / name, value)
        progress('allocate_diagnostics', generated=True)
        # Capture the ID actually consumed BEFORE prepare_slot overwrites it.
        # Predicted IDs are captured independently of the runtime history store.
        names = ['slot_ids', 'predicted_tokens']
        if args.trace:
            names += [f'layer{layer}_routing_ranks' for layer in range(layers)]
        captures = {name: torch.empty((contexts, *buffers[name].shape[1:]),
            device='cuda', dtype=buffers[name].dtype) for name in names}
        descriptors = torch.tensor([[buffers[name].data_ptr(), target.data_ptr(),
            buffers[name][0].numel() * buffers[name].element_size() // 2]
            for name, target in captures.items()], device='cuda', dtype=torch.int64)
        visits = torch.zeros(contexts, device='cuda', dtype=torch.int32)
        slot_owners = torch.empty_like(visits)
        capture_io = torch.tensor([descriptors.data_ptr(), len(captures), visits.data_ptr(), slot_owners.data_ptr()],
                                 device='cuda', dtype=torch.int64)
        args.slots, args.layers = slots, layers
        args.work_queue_capacity = demo.next_power_of_two(contexts * (1 + layers * (14 + 2 * demo.TOP_K) + 3) + 2)
        args.control_queue_capacity = demo.next_power_of_two(2 * contexts + 3)
        trace_capacity = demo.next_power_of_two(1024 + contexts * (32 + 512 * layers + 24))
        trace = torch.zeros((trace_capacity + 1, 12), device='cuda', dtype=torch.int64) if args.trace else None
        readonly = dict(inputs=inputs, embedding=weights['embedding'],
                        final_norm=weights['readout']['norm'], lm_head=weights['readout']['lm_head'])
        for layer, packed in enumerate(weights['layers']):
            readonly.update({f'layer{layer}_{name}': value for name, value in packed.items()})
        progress('hash_readonly_inputs_and_weights')
        before = {name: digest(value) for name, value in readonly.items()}
        demo.save_json(out / 'readonly_before.json', before)
        progress('compile_native_graph')
        with demo.compile_options(args, out, capture_io, trace):
            pk.compile(context_io=(inputs, saved_hidden, buffers['slot_ids'], buffers['layer47_final']),
                context_sequence=(args.steps, prefix),
                context_autoregressive=(buffers['predicted_tokens'], generated), output_dir=str(out))
        progress('ready', compiled=True)
        reports, previous = [], None
        for trial in range(args.trials):
            directory = out / f'trial_{trial}'
            directory.mkdir()
            # Each invocation starts with the SAME prefix and seeds. Epochs and
            # scheduler state reset through the existing launch initializer.
            for value in list(buffers.values()) + list(captures.values()):
                value.fill_(-1 if value.dtype in (torch.int32, torch.int64) else float('nan'))
            for layer in range(layers):
                buffers[f'layer{layer}_k_cache'].copy_(data['initial_k'][layer])
                buffers[f'layer{layer}_v_cache'].copy_(data['initial_v'][layer])
            generated.fill_(-1)
            saved_hidden.fill_(float('nan'))
            visits.zero_()
            slot_owners.fill_(-1)
            if trace is not None:
                trace.zero_()
            torch.cuda.synchronize()
            progress('execute', trial=trial)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            pk()
            end.record()
            torch.cuda.synchronize()
            progress('validate', executed=True, trial=trial)
            consumed = captures['slot_ids'].reshape(slots, args.steps).cpu()
            predictions = captures['predicted_tokens'].reshape(slots, args.steps).cpu()
            history = generated.reshape(slots, args.steps).cpu()
            feedback = check_feedback(consumed, history, seeds)
            cache_checks = check_final_cache(torch, buffers, data, prefix, args.steps)
            checks = dict(prediction_capture_equals_history=torch.equal(predictions, history),
                contexts_saved_once=bool((visits == 1).all()),
                owners_correct=torch.equal(slot_owners, torch.arange(contexts, device='cuda', dtype=torch.int32) // args.steps),
                ids_in_vocabulary=bool(((history >= 0) & (history < config.vocab_size)).all()),
                saved_hidden_finite=bool(torch.isfinite(saved_hidden).all()),
                unused_inputs_still_poisoned=bool((inputs.reshape(slots, args.steps)[:, 1:] == -1).all()),
                final_cache_boundaries=all(all(row[key] for key in
                    ('prefix_unchanged', 'future_unchanged', 'appended_positions_finite')) for row in cache_checks),
                repeated_tokens_exact=previous is None or torch.equal(history, previous))
            previous = history.clone()
            agreement = compare_reference(history, hf) if hf is not None else None
            text = tokenizer.batch_decode(history, skip_special_tokens=False)
            # EOS trimming is presentation only; every fixed-step token remains above.
            prompt_display = render_prompt_text(tokenizer, data, history)
            report = dict(passed=feedback['passed'] and all(checks.values()), feedback=feedback,
                checks=checks, cache_checks=cache_checks, hf_agreement=agreement,
                consumed_tokens=consumed.tolist(), generated_tokens=history.tolist(), generated_text=text,
                prompt=prompt, provenance=data['provenance'], prompt_display=prompt_display,
                diagnostic_invocation_ms=begin.elapsed_time(end))
            torch.save(dict(consumed_tokens=consumed, generated_tokens=history, predicted_tokens=predictions,
                saved_hidden=saved_hidden.cpu(), visits=visits.cpu(), owners=slot_owners.cpu()), directory / 'outputs.pt')
            if trace is not None:
                from trace import export_trace
                experts = {layer: selected_experts(captures[f'layer{layer}_routing_ranks']).cpu().tolist()
                           for layer in range(layers)}
                torch.save(trace.cpu(), directory / 'raw_trace.pt')
                report['trace'] = export_trace(trace, graph, mapping, contexts, directory, expected_experts=experts)
            demo.save_json(directory / 'report.json', report)
            reports.append(report)
            if prompt_display is not None:
                print(f'Prompt: {prompt_display["prompt_text"]}', flush=True)
            for slot in range(slots):
                print(f'Slot {slot}: {history[slot].tolist()} -> {text[slot]!r}', flush=True)
                if prompt_display is not None:
                    print(f'Slot {slot} continuation: {prompt_display["continuation_text"][slot]}', flush=True)
        progress('verify_readonly_inputs_and_weights')
        after = {name: digest(value) for name, value in readonly.items()}
        demo.save_json(out / 'readonly_after.json', after)
        result = dict(passed=before == after and all(row['passed'] for row in reports),
            readonly_unchanged=before == after, trials=reports, graph=graph_report,
            prompt=prompt, provenance=data['provenance'],
            hf_reference=str(args.hf_reference.resolve()) if args.hf_reference else None,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            scope='48 NAS layers, greedy feedback inside the persistent runtime, fixed steps, no EOS stopping',
            interpretation='passed checks feedback and lifecycle; HF agreement is reported separately. '
                           'A prior teacher-forced model numerical mismatch is not erased by this test.')
        demo.save_json(out / 'report.json', result)
        progress('finished', validated=result['passed'])
        if not result['passed']:
            raise RuntimeError('Autoregressive feedback/lifecycle check failed; artifacts saved')
    except Exception:
        demo.save_json(out / 'failure.json', dict(error=traceback.format_exc(), status=status))
        raise
    finally:
        if pk is not None:
            pk.finalize()


if __name__ == '__main__':
    main()
