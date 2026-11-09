#!/usr/bin/env python3
"""Compare FIFO winner-loop variants without changing selection policy."""
import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

SESSION = Path(__file__).resolve().parents[1]
ROOT = SESSION.parents[1]  # Repository root; research/nas holds the harness.
sys.path.insert(0, str(SESSION / 'experiments/native_prefill'))
from workload import validate, _checkpoint
from validation import compare_hf
from autoregressive import load_weights
from run import make_kernel, cache_checks
import demo


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


@contextmanager
def compile_options(args, out, pointers, variant):
    """Select the policy and probes; both policies use the same completion budget."""
    module = importlib.import_module('mirage.mpk.persistent_kernel')
    original = module.get_compile_command
    def command(*positional, **keywords):
        cmd = [x for x in original(*positional, **keywords) if x != '-arch=sm_90a']
        policy = variant.split('_', 1)[1]
        flags = [f'-DNAS_INVESTIGATE_COMPLETION_BUDGET={args.completion_budget}',
            f'-DNAS_SELECTOR_HISTOGRAM={int(policy == "histogram")}',
            f'-DNAS_SELECTOR_FIFO={int(policy.startswith("fifo"))}',
            f'-DNAS_FIFO_CACHE_OLDEST={int(policy in ("fifo_cached", "fifo_cached_unchecked", "fifo_hot"))}',
            f'-DNAS_FIFO_SKIP_STATIC_CHECKS={int(policy in ("fifo_cached_unchecked", "fifo_hot"))}',
            f'-DNAS_FIFO_HOT_RECORDS={int(policy == "fifo_hot")}',
            '-DMIRAGE_EXPERIMENTAL_MOE_BATCHING=1',
            '-DNAS_MOE_TOP_K=8', '-DNAS_MOE_STATE_SLOTS=16', '-DNAS_MOE_LAYERS=48',
            '-DNAS_MOE_H=2048', '-DNAS_MOE_I=768',
            f'-DNAS_MOE_WORKER_QUEUE_LEN={args.work_queue_capacity}',
            f'-DNAS_MOE_COMPLETION_QUEUE_LEN={args.work_queue_capacity}',
            f'-DNAS_MOE_SCHED_QUEUE_LEN={args.control_queue_capacity}']
        if variant.startswith('diagnostic'):
            flags += ['-DNAS_LATENCY_PROBE=1', '-DNAS_LATENCY_CHUNK_ROWS=4',
                      f'-DNAS_LATENCY_ADDRESS={pointers.data_ptr()}ULL']
        if variant.startswith('diagnostic'):
            flags += ['-DNAS_LATENCY_DIAGNOSTIC=1', '-DNAS_DISPATCH_BREAKDOWN=1', '-DNAS_SELECTOR_BREAKDOWN=1',
                      f'-DNAS_LATENCY_TASK_CAPACITY={args.task_capacity}ULL']
        cmd[1:1] = flags
        save(out / 'compile_command.json', cmd)
        return cmd
    module.get_compile_command = command
    try:
        yield
    finally:
        module.get_compile_command = original


def allocate_probe(torch, count, num_data, capacity, diagnostic):
    """Buffer order is shared with nas_latency_probe.cuh; no runtime ABI changes."""
    shapes = [(8,), (count, 4), (count*64, 16), (count*64, 4)]
    shapes += ([(16, 2), (16, capacity, 16), (num_data,), (num_data,),
                (16,), (16,), (8, 2, 3)] if diagnostic else [(1,)] * 7)
    shapes += [(3, 4, 3) if diagnostic else (1,)]  # outcome, phase, count/sum/max
    shapes += [(2, 4, 3), (2, 8)] if diagnostic else [(1,), (1,)]
    shapes += [(2, 4) if diagnostic else (1,)] # ready/remove membership maintenance
    buffers = [torch.zeros(shape, dtype=torch.int64, device='cuda') for shape in shapes]
    addresses = torch.tensor([x.data_ptr() for x in buffers], dtype=torch.int64, device='cuda')
    return buffers, addresses


def reset(torch, buffers, history, lengths, saved, probes):
    for value in buffers.values():
        value.fill_(float('nan') if value.is_floating_point() else -1)
    for name, value in buffers.items():
        if name.endswith('_cache'): value.fill_(-7)
    history.fill_(-1)
    lengths.fill_(-99)
    saved.fill_(float('nan'))
    for value in probes: value.zero_()
    torch.cuda.synchronize()


def run(args):
    import numpy as np
    import torch
    out = args.output.resolve()
    # Source lives in home; results may be a symlink to the large /work/hdd
    # experiment directory. Keep writes confined to that chosen results root.
    assert (SESSION / 'results').resolve() in out.parents
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {}
    def progress(phase, **details):
        status.update(phase=phase, elapsed_seconds=time.monotonic()-started, **details)
        save(out / 'status.json', status)
        print(json.dumps(status), flush=True)
    pk = None
    try:
        data = validate(json.loads(args.workload.read_text()))
        assert data['chunk_size'] == 4 and len(data['prompts']) == 64
        hf = json.loads(args.hf_reference.read_text())
        assert hf['passed'] and hf['workload_sha256'] == data['sha256']
        for key in ('NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS', 'MIRAGE_RESIDENT_SLOT_ITERATIONS'):
            assert not os.environ.get(key), key
        os.environ.update(MIRAGE_HOME=str(ROOT), MIRAGE_ROOT=str(ROOT),
            MIRAGE_TASK_GRAPH_MODE='resident_data', MIRAGE_RESIDENT_EXECUTION_MODE='scheduler_dispatch',
            MIRAGE_DENSE_BATCHING='none', MIRAGE_MOE_BATCH_ROWS='8', MIRAGE_MOE_TIMEOUT_US='100',
            MIRAGE_MOE_EARLY_COALESCE='0', MIRAGE_MOE_DEADLINE_COALESCE='0',
            MIRAGE_COMPARE_SEQUENCE_REPEATS='1', MPK_MAX_TOKENS_PER_REQUEST='64')
        sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'demo/context_io')]
        import mirage as mi
        import mirage.core
        from transformers import AutoTokenizer, Qwen3MoeConfig
        torch.cuda.set_device(0)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        assert torch.cuda.get_device_capability(0) == (9, 0)
        assert all(Path(x).resolve().is_relative_to(ROOT) for x in (mi.__file__, mirage.core.__file__))
        manifest, snapshot, _ = _checkpoint(data['checkpoint'])
        config = Qwen3MoeConfig.from_json_file(str(snapshot / 'config.json'))
        config._attn_implementation = 'eager'
        assert (config.hidden_size, config.moe_intermediate_size, config.num_experts,
                config.num_experts_per_tok, config.num_hidden_layers) == (2048, 768, 128, 8, 48)
        save(out / 'environment.json', dict(python=sys.executable, torch=torch.__version__,
            mirage=mi.__file__, extension=mirage.core.__file__, gpu=str(torch.cuda.get_device_properties(0)),
            node=os.uname().nodename, job=os.getenv('SLURM_JOB_ID'), slots=16, workers=16, schedulers=1,
            layers=48, chunk_rows=4, requests=64, threshold=8, timeout_us=100, physical_capacity=16,
            descriptor_pool=16, dense_batching=False, completion_budget=args.completion_budget, selectors=sorted({v.split("_",1)[1] for v in args.variants}), injected_delay=False,
            mode='resident_data/scheduler_dispatch', model=manifest['model'], revision=manifest['revision'],
            task_trace=False, optional_activation_capture=False, context_output_copy=True))
        save(out / 'workload.json', data)
        owners = json.loads((snapshot / 'model.safetensors.index.json').read_text())['weight_map']
        progress('load_checkpoint')
        if args.packed_weights:
            from packed_weights import load_for_experiment, source_identity, cache_manifest
            _, expected_source = source_identity(data['checkpoint'])
            assert cache_manifest(args.packed_weights, snapshot)['source'] == expected_source
            weights = load_for_experiment(torch, config, snapshot, owners, progress,
                args.packed_weights, out/'loading.json', compare_legacy=args.compare_loader)
        else:
            weights = load_weights(torch, config, snapshot, owners, progress)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        progress('weights_ready')
        summary = []
        previous = None
        baseline = torch.load(args.baseline_outputs, map_location='cpu', weights_only=True)
        for variant in args.variants:
            # Preserve completed results if the allocation is too short for a
            # further full invocation. Never truncate a request for a benchmark.
            remaining = args.budget_seconds - (time.monotonic()-started)
            policy = variant.split('_', 1)[1]
            # Estimate each policy separately: a slow legacy invocation must not
            # force the same duration estimate for the optimized selector later.
            # FIFO diagnostics last took 77 seconds. Budget 120 until a new
            # sample exists, plus the compile/setup margin below. This never
            # truncates an invocation; the external job timeout remains final.
            initial_estimate = 120 if policy.startswith('fifo') else (180 if policy == 'histogram' else 1353)
            estimate = max([x['invocation_seconds'] for x in summary if x['selector']==policy] or [initial_estimate])
            if remaining < estimate * 1.15 + 180:
                progress('insufficient_time_for_next_variant', skipped=variant, remaining=remaining)
                raise RuntimeError('Insufficient allocation time: required comparison is incomplete')
            directory = out / variant
            directory.mkdir()
            progress('build_graph', variant=variant)
            count, slots, p = 64, 16, 4
            contexts = count*64
            inputs = torch.full((contexts, 1), -1, dtype=torch.int64, device='cuda')
            for row in data['prompts']:
                inputs[row['request']*64:row['request']*64+row['length'], 0] = torch.tensor(
                    row['token_ids'], dtype=torch.int64, device='cuda')
            prompt_lengths = torch.tensor([row['length'] for row in data['prompts']], dtype=torch.int32, device='cuda')
            lengths = torch.empty(count, dtype=torch.int32, device='cuda')
            history = torch.empty((contexts, 1), dtype=torch.int64, device='cuda')
            saved = torch.empty((contexts, demo.H), dtype=torch.bfloat16, device='cuda')
            pk = make_kernel(mi, torch, slots, p, data['eos_token_id'])
            buffers = demo.build_graph(pk, torch, weights, slots, 48, readout=True, chunk_rows=p)
            last_hidden = buffers['last_hidden']
            graph = json.loads(pk.generate_task_graph()['json_file'])
            pk.set_request_slots(demo.assign_slots(graph, buffers, slots))
            graph = json.loads(pk.generate_task_graph()['json_file'])
            mapping, validation = demo.validate_graph(graph, slots, 48, readout=True, chunk_rows=p)
            save(directory / 'native_graph.json', graph)
            save(directory / 'mapping.json', mapping)
            save(directory / 'graph_validation.json', validation)
            ordinary, moe = [0]*slots, [0]*slots
            for node in mapping:
                (moe if node['operator'] in ('gate_up','down') else ordinary)[node['slot']] += 1
            bound = contexts * max(a + 8*b for a,b in zip(ordinary,moe))
            args.work_queue_capacity = demo.next_power_of_two(bound+2)
            args.control_queue_capacity = demo.next_power_of_two(2*contexts+3)
            save(directory / 'queue_capacity_proof.json', dict(consumed_row_bound=contexts,
                compute_bound=bound, worker=args.work_queue_capacity,
                completion=args.work_queue_capacity, scheduler=args.control_queue_capacity))
            probes, pointers = allocate_probe(torch, count, len(mapping), args.task_capacity, variant.startswith('diagnostic'))
            progress('compile', variant=variant)
            with compile_options(args, directory, pointers, variant):
                pk.compile(context_io=(inputs,saved,buffers['slot_ids'],last_hidden),
                    context_chunk_size=p, context_prefill=(prompt_lengths,lengths),
                    context_autoregressive=(buffers['predicted_tokens'],history), output_dir=str(directory))
            reset(torch, buffers, history, lengths, saved, probes)
            # CUDA events bracket the complete invocation. In the instrumented
            # variants the device-global timer separately brackets preparation
            # through scheduler shutdown; host delivery/network is not included.
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            progress('execute', variant=variant)
            host_start = time.monotonic()
            start.record()
            pk()
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)/1000
            host_elapsed = time.monotonic()-host_start
            progress('executed', variant=variant, invocation_seconds=elapsed)
            tokens = history.reshape(count,64).cpu()
            lens = lengths.cpu()
            saved_cpu = saved.reshape(count,64,demo.H).cpu()
            torch.save(dict(tokens=tokens, lengths=lens, saved_hidden=saved_cpu), directory/'outputs.pt')
            raw = {name:probes[i].cpu().numpy() for i,name in enumerate(
                ('header','admissions','passes','tokens','worker_counts'))}
            if variant.startswith('diagnostic'):
                raw['scheduler'] = probes[10].cpu().numpy()
                raw['dispatch'] = probes[11].cpu().numpy()
                raw['selector_phases'] = probes[12].cpu().numpy()
                raw['selector_stats'] = probes[13].cpu().numpy()
                raw['membership'] = probes[14].cpu().numpy()
                for worker in range(16):
                    n = min(int(raw['worker_counts'][worker,0]), args.task_capacity)
                    raw[f'worker_{worker}'] = probes[5][worker,:n].cpu().numpy()
            np.savez_compressed(directory/'measurements.npz', **raw)
            del raw
            from analyze import analyze_lifecycle, analyze_diagnostic
            checks = []
            for request, prompt in enumerate(data['prompts']):
                length = int(lens[request]); row = tokens[request,:length]
                checks.append(1 <= length <= 65-prompt['length'] and
                    bool(((row >= 0) & (row < data['vocab_size'])).all()) and
                    not bool((row[:-1] == data['eos_token_id']).any()) and
                    (int(row[-1]) == data['eos_token_id'] or prompt['length']+length-1 == 64) and
                    bool((tokens[request,length:] == -1).all()) and
                    bool(torch.isfinite(saved_cpu[request,:length]).all()) and
                    bool(torch.isnan(saved_cpu[request,length:]).all()))
            report = dict(variant=variant, selector=policy, completion_budget=args.completion_budget,
                fifo_cache_oldest=policy in ('fifo_cached', 'fifo_cached_unchecked', 'fifo_hot'),
                fifo_hot_records=policy == 'fifo_hot',
                fifo_skip_static_checks=policy in ('fifo_cached_unchecked', 'fifo_hot'),
                instrumentation="diagnostic" if variant.startswith("diagnostic") else "off",
                invocation_seconds=elapsed, host_launch_to_sync_seconds=host_elapsed,
                same_tokens_as_preserved_control=bool(torch.equal(tokens,baseline["tokens"]) and
                    torch.equal(lens,baseline["lengths"])),
                generated_tokens=int(lens.sum()), basic_output_checks=all(checks),
                hf_agreement=compare_hf(tokens,lens,hf),
                same_tokens_as_first=None if previous is None else bool(torch.equal(tokens,previous)),
                texts=[tokenizer.decode(tokens[i,:int(lens[i])].tolist(),skip_special_tokens=True) for i in range(count)])
            report['amortized_seconds_per_token'] = elapsed/report['generated_tokens']
            if variant.startswith('diagnostic'):
                lifecycle = analyze_lifecycle(directory, data, tokens.numpy(), lens.numpy())
                report['lifecycle'] = lifecycle
                report['clock_agreement_passed'] = abs(lifecycle['device_span_s']-elapsed)<max(1.,.01*elapsed)
                cache = cache_checks(torch,buffers,lifecycle,slots,-7)
                save(directory/'cache_checks.json',cache)
                report['cache_checks_passed'] = all(r['valid_finite'] and r['all_valid_positions_written'] and
                    r['never_used_suffix_poisoned'] for r in cache)
            if variant.startswith('diagnostic'):
                report['diagnostic'] = analyze_diagnostic(directory, mapping, elapsed, selector_policy=policy)
            report['passed'] = (report['basic_output_checks'] and
                report['same_tokens_as_preserved_control'] and report['same_tokens_as_first'] is not False and
                report.get('lifecycle',{}).get('passed',True) and report.get('cache_checks_passed',True) and
                report.get('diagnostic',{}).get('passed',True) and report.get('clock_agreement_passed',True))
            save(directory/'report.json',report)
            summary.append({k:v for k,v in report.items() if k not in ('texts','hf_agreement','diagnostic','lifecycle')})
            save(out/'summary.json',summary)
            from report import write_report
            write_report(out)
            progress('validated',variant=variant,passed=report['passed'])
            assert report['passed'], directory/'report.json'
            previous = tokens.clone()
            pk.finalize()
            pk = None
            del buffers, probes, pointers, inputs, prompt_lengths, history, saved, lengths, last_hidden
            torch.cuda.empty_cache()
        progress('complete', completed_variants=[row['variant'] for row in summary])
    except BaseException:
        save(out/'failure.json',dict(error=traceback.format_exc(),status=status))
        raise
    finally:
        if pk is not None: pk.finalize()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--hf-reference', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', choices=('diagnostic_legacy','diagnostic_histogram','control_legacy','control_histogram',
                        'diagnostic_fifo','control_fifo','diagnostic_fifo_cached','control_fifo_cached',
                        'diagnostic_fifo_cached_unchecked','control_fifo_cached_unchecked',
                        'diagnostic_fifo_hot','control_fifo_hot'),
                        default=['control_fifo','control_fifo_cached_unchecked',
                                 'diagnostic_fifo','diagnostic_fifo_cached','diagnostic_fifo_cached_unchecked'])
    parser.add_argument('--packed-weights',type=Path)
    parser.add_argument('--compare-loader',action='store_true',help='Time and byte-check original loading before packed loading')
    parser.add_argument('--completion-budget',type=int,choices=(1,16),default=1)
    parser.add_argument('--baseline-outputs',type=Path,required=True)
    parser.add_argument('--task-capacity',type=int,default=250000)
    parser.add_argument('--budget-seconds',type=int,default=6900)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.compare_loader and not args.packed_weights:
        parser.error('--compare-loader requires --packed-weights')
    run(args)
