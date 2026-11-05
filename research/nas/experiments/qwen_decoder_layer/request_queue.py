#!/usr/bin/env python3
"""Run a finite queue of real Qwen requests through reusable NAS slots.

HF prefixes are prepared beforehand. One persistent invocation performs all
admission, EOS/capacity stopping, token feedback, and slot replacement. Output
rows are request*64+step, so completion order never changes request ownership.
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
from autoregressive import load_weights
from prepare_requests import validate_requests, request_fingerprints
from request_reference import check_requests, compare_requests, render_requests, validate_reference
from reference import DEFAULT_CHECKPOINT, PAGE, digest

SESSION = Path(__file__).resolve().parents[2]
ROOT = SESSION.parents[1]  # Repository root; research/nas holds the harness.


def check_output_rows(torch, data, history, lengths, consumed, predictions, visits, owners,
                      saved_hidden, slots):
    """Only each request's actual output rows may have been saved or consumed."""
    count = len(lengths)
    valid = torch.arange(PAGE)[None, :] < lengths[:, None]
    owner_rows = owners.reshape(count, PAGE).cpu()
    visits = visits.reshape(count, PAGE).cpu()
    hidden = saved_hidden.reshape(count, PAGE, -1).cpu()
    lifecycle = check_requests(consumed, history, data['seed_tokens'], lengths,
        data['prefix_lengths'], data['eos_token_id'], data['prompts'][0]['vocab_size'])
    request_owners = []
    for request in range(count):
        row = owner_rows[request, :int(lengths[request])]
        request_owners.append(int(row[0]) if len(row) else -1)
    ownership = all(0 <= owner < slots and bool(
        (owner_rows[r, :int(lengths[r])] == owner).all()) for r, owner in enumerate(request_owners))
    checks = dict(prediction_capture_equals_history=torch.equal(predictions[valid], history[valid]),
        valid_rows_saved_once=bool((visits[valid] == 1).all()),
        unused_rows_not_saved=bool((visits[~valid] == 0).all()),
        unused_history_untouched=bool((history[~valid] == -1).all()),
        unused_captures_untouched=bool((consumed[~valid] == -1).all() and (predictions[~valid] == -1).all()),
        unused_owners_untouched=bool((owner_rows[~valid] == -1).all()),
        output_ownership=ownership, valid_hidden_finite=bool(torch.isfinite(hidden[valid]).all()),
        unused_hidden_untouched=bool(torch.isnan(hidden[~valid]).all()),
        replacement_observed=count <= slots or len(set(request_owners)) < count)
    return lifecycle, checks, request_owners


def check_final_caches(torch, buffers, data, lengths, owners, slots):
    """A reused slot contains its LAST request's cache, not every old request.

    Old values beyond the current sequence length are allowed. Attention must
    exclude those positions; the deterministic runtime test checks that mask.
    """
    last = {slot: max((r for r, owner in enumerate(owners) if owner == slot), default=-1)
            for slot in range(slots)}
    checks = []
    for layer in range(48):
        for kind in ('k', 'v'):
            actual = buffers[f'layer{layer}_{kind}_cache'].cpu()
            for slot, request in last.items():
                if request < 0:
                    continue
                prefix, length = int(data['prefix_lengths'][request]), int(lengths[request])
                checks.append(dict(layer=layer, kind=kind, slot=slot, request=request,
                    prefix_matches=torch.equal(actual[slot, :prefix], data['initial_'+kind][layer, request, :prefix]),
                    decoded_positions_finite=bool(torch.isfinite(actual[slot, prefix:prefix+length]).all())))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--hf-reference', type=Path)
    parser.add_argument('--slots', type=int, choices=(16, 32), default=16)
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--batch-rows', type=int, default=4)
    parser.add_argument('--timeout-us', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not (1 <= args.trials <= 3 and 1 <= args.batch_rows <= 16 and 0 <= args.timeout_us <= 1000000):
        parser.error('Use 1..3 trials, batch rows 1..16, timeout 0..1000000 us')
    if (os.environ.get('MIRAGE_COMPARE_SEQUENCE_REPEATS', '1') != '1' or
            'MIRAGE_RESIDENT_SLOT_ITERATIONS' in os.environ):
        parser.error('Unset comparison-repeat and fixed-iteration overrides for the request queue')
    out = args.output.resolve()
    if SESSION not in out.parents:
        parser.error('Keep output in this isolated experiment directory')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = dict(generated=False, compiled=False, executed=False, validated=False,
                  job=os.getenv('SLURM_JOB_ID'))
    def progress(phase, **details):
        status.update(phase=phase, elapsed_seconds=time.monotonic()-started, **details)
        demo.save_json(out/'status.json', status)
        print(f'[{status["elapsed_seconds"]:.1f}s] {phase}: {details}', flush=True)
    progress('imports')
    sys.path[:0] = [str(ROOT/'python'), str(ROOT/'demo/context_io')]
    os.environ.update(MIRAGE_HOME=str(ROOT), MIRAGE_ROOT=str(ROOT), MIRAGE_TASK_GRAPH_MODE='resident_data',
        MIRAGE_RESIDENT_EXECUTION_MODE='scheduler_dispatch', MIRAGE_MOE_BATCH_ROWS=str(args.batch_rows),
        MIRAGE_MOE_TIMEOUT_US=str(args.timeout_us), MIRAGE_MOE_EARLY_COALESCE='0',
        MIRAGE_MOE_DEADLINE_COALESCE='0', MIRAGE_DENSE_BATCHING='none')
    pk = None
    try:
        import torch
        import mirage as mi
        import mirage.core
        from transformers import AutoTokenizer, Qwen3MoeConfig
        torch.cuda.set_device(0)
        if torch.cuda.get_device_capability(0) != (9, 0):
            raise RuntimeError('This demo uses the tested Hopper path')
        if str(ROOT) not in mi.__file__ or str(ROOT) not in mirage.core.__file__:
            raise RuntimeError('Import Mirage from the isolated checkout')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        progress('load_request_queue')
        data = torch.load(args.requests, map_location='cpu', weights_only=True, mmap=True)
        count = validate_requests(data)
        contexts, slots, layers = count*PAGE, args.slots, 48
        manifest = json.loads(args.checkpoint.read_text())
        if any(data['provenance'][key] != manifest[key] for key in ('model', 'revision')):
            raise ValueError('Request prefixes and checkpoint differ')
        snapshot = Path(manifest['files']['config.json']).parent
        weight_owners = json.loads((snapshot/'model.safetensors.index.json').read_text())['weight_map']
        config = Qwen3MoeConfig.from_json_file(str(snapshot/'config.json'))
        config._attn_implementation = 'eager'
        if (config.hidden_size, config.moe_intermediate_size, config.num_experts,
            config.num_experts_per_tok, config.num_hidden_layers, config.head_dim,
            config.num_key_value_heads, config.num_attention_heads) != (2048, 768, 128, 8, 48, 128, 4, 32):
            raise ValueError('Checkpoint shape differs from the existing graph builder')
        fingerprints = request_fingerprints(data)
        hf = None
        if args.hf_reference:
            hf = torch.load(args.hf_reference, map_location='cpu', weights_only=True)
            validate_reference(hf, data, fingerprints)
        demo.save_json(out/'environment.json', dict(root=str(ROOT), mirage=mi.__file__, core=mirage.core.__file__,
            python=sys.executable, torch=torch.__version__, gpu=str(torch.cuda.get_device_properties(0)),
            node=os.uname().nodename, job=os.getenv('SLURM_JOB_ID'), workers=16, schedulers=1,
            mode='resident_data/scheduler_dispatch', slots=slots, requests=count, layers=layers))
        demo.save_json(out/'provenance.json', dict(source_requests=str(args.requests.resolve()),
            prefix_sha256=fingerprints, prefix_lengths=data['prefix_lengths'].tolist(),
            seed_tokens=data['seed_tokens'].tolist(), prompts=data['prompts'], provenance=data['provenance']))
        weights = load_weights(torch, config, snapshot, weight_owners, progress)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        inputs = torch.full((count, PAGE), -1, dtype=torch.int64, device='cuda')
        inputs[:, 0].copy_(data['seed_tokens'])
        inputs = inputs.reshape(contexts, 1)
        history = torch.empty((contexts, 1), dtype=torch.int64, device='cuda')
        lengths = torch.empty(count, dtype=torch.int32, device='cuda')
        prefixes = data['prefix_lengths'].to('cuda')
        saved_hidden = torch.empty((contexts, demo.H), dtype=torch.bfloat16, device='cuda')
        prefix_k, prefix_v = data['initial_k'].to('cuda'), data['initial_v'].to('cuda')

        spec = importlib.util.spec_from_file_location('request_context_demo', ROOT/'demo/context_io/demo.py')
        context_demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(context_demo)
        pk = context_demo.make_kernel(mi, torch, slots=slots, workers=16, metadata_pages=128)
        pk.num_local_schedulers = 1
        pk.eos_token_id = data['eos_token_id']
        progress('generate_native_graph')
        buffers = demo.build_graph(pk, torch, weights, slots, layers, readout=True)
        graph = json.loads(pk.generate_task_graph()['json_file'])
        pk.set_request_slots(demo.assign_slots(graph, buffers, slots))
        graph = json.loads(pk.generate_task_graph()['json_file'])
        mapping, graph_report = demo.validate_graph(graph, slots, layers, readout=True)
        for name, value in (('native_graph.json', graph), ('mapping.json', mapping), ('graph_validation.json', graph_report)):
            demo.save_json(out/name, value)
        bindings = [(buffers[f'layer{layer}_k_cache'], buffers[f'layer{layer}_v_cache'],
                     prefix_k[layer], prefix_v[layer]) for layer in range(layers)]
        # Capture actual consumed/predicted IDs before the scheduler reuses a
        # slot. These diagnostic copies are included in the reported run time.
        captures = {name: torch.empty((contexts, 1), dtype=torch.int64, device='cuda')
                    for name in ('slot_ids', 'predicted_tokens')}
        descriptors = torch.tensor([[buffers[name].data_ptr(), target.data_ptr(), 4]
            for name, target in captures.items()], dtype=torch.int64, device='cuda')
        visits = torch.zeros(contexts, dtype=torch.int32, device='cuda')
        owners = torch.empty_like(visits)
        capture_io = torch.tensor([descriptors.data_ptr(), len(captures), visits.data_ptr(), owners.data_ptr()],
                                 dtype=torch.int64, device='cuda')
        args.layers = layers
        args.work_queue_capacity = demo.next_power_of_two(contexts*(1+layers*(14+2*demo.TOP_K)+3)+2)
        args.control_queue_capacity = demo.next_power_of_two(2*contexts+3)
        readonly = dict(inputs=inputs, prefix_lengths=prefixes, initial_k=prefix_k, initial_v=prefix_v,
            embedding=weights['embedding'], final_norm=weights['readout']['norm'], lm_head=weights['readout']['lm_head'])
        for layer, packed in enumerate(weights['layers']):
            readonly.update({f'layer{layer}_{name}': value for name, value in packed.items()})
        progress('hash_readonly_inputs_and_weights', generated=True)
        before = {name: digest(value) for name, value in readonly.items()}
        demo.save_json(out/'readonly_before.json', before)
        progress('compile_native_graph')
        with demo.compile_options(args, out, capture_io, None):
            pk.compile(context_io=(inputs, saved_hidden, buffers['slot_ids'], buffers['layer47_final']),
                context_autoregressive=(buffers['predicted_tokens'], history),
                context_requests=(prefixes, lengths, bindings), output_dir=str(out))
        progress('ready', compiled=True)
        reports, previous_history, previous_lengths = [], None, None
        for trial in range(args.trials):
            directory = out/f'trial_{trial}'
            directory.mkdir()
            for value in list(buffers.values()) + list(captures.values()):
                value.fill_(-1 if value.dtype in (torch.int32, torch.int64) else float('nan'))
            history.fill_(-1)
            lengths.fill_(-99)  # Runtime initialization must reset these too.
            saved_hidden.fill_(float('nan'))
            visits.zero_()
            owners.fill_(-1)
            torch.cuda.synchronize()
            progress('execute', trial=trial)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            pk()
            end.record()
            torch.cuda.synchronize()
            progress('validate', executed=True, trial=trial)
            consumed = captures['slot_ids'].reshape(count, PAGE).cpu()
            predictions = captures['predicted_tokens'].reshape(count, PAGE).cpu()
            generated = history.reshape(count, PAGE).cpu()
            actual_lengths = lengths.cpu()
            lifecycle, checks, request_owners = check_output_rows(torch, data, generated, actual_lengths,
                consumed, predictions, visits, owners, saved_hidden, slots)
            cache_checks = check_final_caches(torch, buffers, data, actual_lengths, request_owners, slots)
            checks.update(final_prefixes_match=all(row['prefix_matches'] for row in cache_checks),
                cached_decode_positions_finite=all(row['decoded_positions_finite'] for row in cache_checks),
                unused_inputs_still_poisoned=bool((inputs.reshape(count, PAGE)[:, 1:] == -1).all()),
                repeated_tokens_exact=previous_history is None or torch.equal(generated, previous_history),
                repeated_lengths_exact=previous_lengths is None or torch.equal(actual_lengths, previous_lengths))
            previous_history, previous_lengths = generated.clone(), actual_lengths.clone()
            agreement = compare_requests(generated, actual_lengths, hf) if hf is not None else None
            texts = render_requests(tokenizer, data, generated, actual_lengths)
            report = dict(passed=lifecycle['passed'] and all(checks.values()), lifecycle=lifecycle,
                checks=checks, cache_checks=cache_checks, hf_agreement=agreement,
                generated_lengths=actual_lengths.tolist(), request_owners=request_owners, texts=texts,
                diagnostic_invocation_ms=begin.elapsed_time(end), completed_passes=int(actual_lengths.sum()))
            torch.save(dict(consumed_tokens=consumed, generated_tokens=generated, predicted_tokens=predictions,
                generated_lengths=actual_lengths, visits=visits.cpu(), owners=owners.cpu(),
                saved_hidden=saved_hidden.cpu()), directory/'outputs.pt')
            demo.save_json(directory/'report.json', report)
            reports.append(report)
            for row in texts:
                print(f'Request {row["request"]}, slot {request_owners[row["request"]]}, '
                      f'prefix {row["prefix_length"]}, generated {row["generated_length"]}: '
                      f'{row["continuation"]}', flush=True)
        progress('verify_readonly_inputs_and_weights')
        after = {name: digest(value) for name, value in readonly.items()}
        demo.save_json(out/'readonly_after.json', after)
        result = dict(passed=before == after and all(row['passed'] for row in reports),
            readonly_unchanged=before == after, trials=reports, graph=graph_report,
            requests=count, slots=slots, layers=layers, capacity=PAGE, eos_token_id=data['eos_token_id'],
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            scope='48-layer NAS autoregression with EOS, offline replacement, and variable HF prefixes',
            interpretation='passed checks request lifecycle; exact HF agreement is reported separately '
                           'and does not erase the known model arithmetic discrepancy.')
        demo.save_json(out/'report.json', result)
        progress('finished', validated=result['passed'])
        if not result['passed']:
            raise RuntimeError('Request lifecycle check failed; artifacts saved')
    except Exception:
        demo.save_json(out/'failure.json', dict(error=traceback.format_exc(), status=status))
        raise
    finally:
        if pk is not None:
            pk.finalize()


if __name__ == '__main__':
    main()
