#!/usr/bin/env python3
"""Two reusable NAS slots consume a preloaded queue of context vectors.

One context = one pass through three RMSNorm stages. This is a scheduling/IO
demo, not an FFN or decoder. See README.md for GPU commands and limitations.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SLOTS, WIDTH, STAGES = 64, 16384, 3


def make_kernel(mi, torch, *, slots=None, workers=64, metadata_pages=None):
    # Optional dimensions let the FFN demo run a small correctness case without
    # changing this demo's defaults or the runtime's worker assignment policy.
    if slots is None:
        slots = SLOTS
    if metadata_pages is None:
        metadata_pages = slots * 2
    # FFN-only experiments can keep this unused legacy page pool small as the
    # independent activation-state count grows. It is not used by ContextIO.
    assert metadata_pages > 0
    # The existing constructor still requires token/KV-shaped metadata. Allocate
    # valid CUDA buffers, although the ContextIO preparation path never uses
    # them for serving. These dimensions describe slots, not the context queue.
    def zeros(shape, dtype=torch.int32):
        return torch.zeros(shape, dtype=dtype, device='cuda')

    meta = dict(
        step=zeros(slots), tokens=zeros((slots, 3), torch.int64),
        input_tokens=zeros((slots, 1), torch.int64),
        output_tokens=zeros((slots, 1), torch.int64),
        num_new_tokens=torch.ones(slots, dtype=torch.int32, device='cuda'),
        prompt_lengths=torch.ones(slots, dtype=torch.int32, device='cuda'),
        qo_indptr_buffer=zeros(slots + 1), paged_kv_indptr_buffer_0=zeros(slots + 1),
        paged_kv_indices_buffer_0=zeros(metadata_pages), paged_kv_last_page_len_buffer_0=zeros(slots),
        paged_kv_indices_snapshot_0=zeros(metadata_pages),
    )
    return mi.PersistentKernel(
        mode='offline', world_size=1, mpi_rank=0, num_workers=workers,
        num_local_schedulers=2, num_remote_schedulers=0, max_seq_length=3,
        max_num_batched_requests=slots, max_num_batched_tokens=slots,
        max_num_pages=metadata_pages, page_size=64, meta_tensors=meta,
        profiler_tensor=None, trace_name='', spec_decode_config=None,
        use_cutlass_kernel=True, task_graph_mode='resident_data', eos_token_id=-1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['hybrid_prelaunch', 'scheduler_dispatch'], default='hybrid_prelaunch')
    parser.add_argument('--contexts', type=int, default=7)
    parser.add_argument('--trace', action='store_true', help='write raw CSV, a timeline, and nested Perfetto slices')
    parser.add_argument('--delay-us', type=int, default=0, help='finite delay before slot B terminal completion; requires --trace')
    parser.add_argument('--output', type=Path, required=True, help='new directory for generated graph and results')
    args = parser.parse_args()
    if args.contexts < 0 or not 0 <= args.delay_us <= 100_000:
        parser.error('contexts must be nonnegative; delay-us must be between 0 and 100000')
    if args.delay_us and not args.trace:
        parser.error('--delay-us requires --trace')
    if 'MIRAGE_RESIDENT_SLOT_ITERATIONS' in os.environ:
        parser.error('unset MIRAGE_RESIDENT_SLOT_ITERATIONS: queue exhaustion controls this demo')

    # Import this checkout, including the applied Python binding and JIT headers.
    sys.path.insert(0, str(ROOT / 'python'))
    os.environ.update(MIRAGE_HOME=str(ROOT), MIRAGE_ROOT=str(ROOT),
                      MIRAGE_TASK_GRAPH_MODE='resident_data', MIRAGE_RESIDENT_EXECUTION_MODE=args.mode)
    import torch
    import mirage as mi
    from trace_support import compile_options, export_trace, slot_label

    if torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError('This demo uses the tested Hopper RMSNorm variant (compute capability 9.0)')
    torch.cuda.set_device(0)
    torch.manual_seed(1729)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)  # Preserve previous evidence.
    save = lambda name, value: (out / name).write_text(json.dumps(value, indent=2) + '\n')
    status = dict(generated=False, compiled=False, executed=False, validated=False)
    save('status.json', status)
    save('environment.json', dict(
        branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        mirage=mi.__file__, python=sys.executable, gpu=str(torch.cuda.get_device_properties(0)),
        node=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'), mode=args.mode,
        contexts=args.contexts, slots=SLOTS, width=WIDTH, trace=args.trace, delay_us=args.delay_us))

    # N queued vectors can exceed S working slots. Results keep input order even
    # if their GPU computations complete in a different order.
    inputs = torch.randn(args.contexts, WIDTH, device='cuda', dtype=torch.bfloat16)
    original_inputs = inputs.clone()
    saved_outputs = torch.full_like(inputs, float('nan'))
    slot_inputs = torch.empty(SLOTS, WIDTH, device='cuda', dtype=torch.bfloat16)
    working = [torch.empty_like(slot_inputs) for _ in range(STAGES)]
    weights = [(1 + .1 * torch.randn(WIDTH, device='cuda')).bfloat16() for _ in range(STAGES)]
    pk = make_kernel(mi, torch)
    x = pk.attach_input(slot_inputs, name='slot_inputs')
    for stage, (buffer, weight) in enumerate(zip(working, weights)):
        y = pk.attach_input(buffer, name=f'stage_{stage + 1}')
        w = pk.attach_input(weight, name=f'norm_w_{stage + 1}')
        # grid.x partitions the rows: one task per slot at every stage.
        pk.rmsnorm_layer(x, w, y, grid_dim=(SLOTS, 1, 1), block_dim=(256, 1, 1))
        x = y

    # Derive slot ownership from native tensor slices, never request_id fields.
    plain = json.loads(pk.generate_task_graph()['json_file'])
    slots = []
    for data in plain['all_data']:
        view, = data['outputs']
        assert view['dims'] == [1, WIDTH] and view['offset'] % (WIDTH * 2) == 0
        slots.append(view['offset'] // (WIDTH * 2))
    pk.set_request_slots(slots)
    graph = json.loads(pk.generate_task_graph()['json_file'])
    assert all(slots[e['src_data_id']] == slots[e['dst_data_id']] for e in graph['data_edges'])
    mapping = []
    for task_id, task in enumerate(graph['all_tasks']):
        d = task['data_id']
        if 0 <= d < len(slots):
            data = graph['all_data'][d]
            mapping.append(dict(task=task_id, data=d, resident=data['resident_task_id'],
                                slot=slots[d], stage=data['outputs'][0]['base_ptr']))
    save('mapping.json', mapping)
    save('native_graph.json', graph)
    status['generated'] = True
    save('status.json', status)
    print(f'Contexts={args.contexts}, slots={SLOTS}, width={WIDTH}, mode={args.mode}')
    print(f"Native graph: {len(graph['resident_tasks'])} resident families, {len(slots)} compute DataIds")
    print(' Task  Data  Resident  Slot  Stage')
    for row in mapping:
        print(f" {row['task']:4d}  {row['data']:4d}  {row['resident']:8d}  {slot_label(row['slot'], SLOTS):>4}  {row['stage']}")

    # Allocate tracing only when requested; it does not change graph edges.
    trace = torch.zeros((8193, 12), dtype=torch.int64, device='cuda') if args.trace else None
    terminal, = graph['request_slots'][1]['terminal_data_ids']
    delay_task, = [row['task'] for row in mapping if row['data'] == terminal]
    with compile_options(trace, delay_task, args.delay_us, out):
        pk.compile(context_io=(inputs, saved_outputs, slot_inputs, working[-1]), output_dir=str(out))
    status['compiled'] = True
    save('status.json', status)

    reference = inputs
    for weight in weights:
        reference = (reference.float() * torch.rsqrt(reference.float().square().mean(-1, keepdim=True) + 1e-6)
                     * weight.float()).bfloat16()
    torch.cuda.synchronize()
    pk()  # ONE launch. Every admission/replacement happens inside the runtime.
    torch.cuda.synchronize()
    status['executed'] = True
    save('status.json', status)
    torch.testing.assert_close(saved_outputs, reference, atol=.03125, rtol=.02)
    assert torch.equal(inputs, original_inputs)
    error = (saved_outputs.float() - reference.float()).abs().max().item() if args.contexts else 0.0
    report = dict(contexts=args.contexts, max_abs_error=error, atol=.03125, rtol=.02,
                  persistent_invocations=1, workload='three RMSNorm stages', input_queue_unchanged=True)
    if trace is not None:
        report.update(export_trace(trace, graph, args.contexts, args.delay_us, out))
    torch.save(dict(inputs=inputs.cpu(), outputs=saved_outputs.cpu(), reference=reference.cpu()), out / 'outputs.pt')
    save('report.json', report)
    status['validated'] = True
    save('status.json', status)
    pk.finalize()
    print(f'PASS: saved all {args.contexts} outputs; maximum absolute error={error:g}')
    print(f'Artifacts: {out}')


if __name__ == '__main__':
    main()
