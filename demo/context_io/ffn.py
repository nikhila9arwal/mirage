#!/usr/bin/env python3
"""Independent slots: embedding -> RMSNorm -> gate/up -> SiLU*up -> down.

Each pass consumes a new context ID from a finite GPU queue. The two projection
matrices are shared; every compute task reads/writes only its slot's activation
row. By default save a checksum; --full-output saves every element for checking.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['hybrid_prelaunch', 'scheduler_dispatch'], default='hybrid_prelaunch')
    parser.add_argument('--slots', type=int, default=8)
    parser.add_argument('--contexts', type=int, default=40)
    parser.add_argument('--width', type=int, default=4096, help='H: context and final output width')
    parser.add_argument('--intermediate-width', type=int, default=8192, help='I: width after SiLU/multiply')
    parser.add_argument('--table-rows', type=int, default=64)
    parser.add_argument('--full-output', action='store_true', help='save/check full vectors instead of checksums; adds scheduler copy cost')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--delay-us', type=int, default=0, help='finite delay before terminal completion; requires --trace')
    parser.add_argument('--delay-slots', type=int, nargs='+', default=[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (not 2 <= args.slots <= 128 or args.contexts < 1 or args.width <= 0 or args.width % 512
            or args.intermediate_width <= 0 or args.intermediate_width % 128
            or args.table_rows < 1 or args.trials < 1 or not 0 <= args.delay_us <= 100000):
        parser.error('Use 2..128 slots, positive contexts, H divisible by 512, I divisible by 128, positive table rows/trials, delay 0..100000')
    args.delay_slots = sorted(set(args.delay_slots))
    if any(slot < 0 or slot >= args.slots for slot in args.delay_slots):
        parser.error('--delay-slots must identify existing slots')
    if args.delay_us and not args.trace:
        parser.error('--delay-us requires --trace')
    if 'MIRAGE_RESIDENT_SLOT_ITERATIONS' in os.environ:
        parser.error('Unset MIRAGE_RESIDENT_SLOT_ITERATIONS; the context queue bounds this run')

    sys.path.insert(0, str(ROOT / 'python'))
    os.environ.update(MIRAGE_HOME=str(ROOT), MIRAGE_ROOT=str(ROOT),
                      MIRAGE_TASK_GRAPH_MODE='resident_data', MIRAGE_RESIDENT_EXECUTION_MODE=args.mode)
    import torch
    import mirage as mi
    import mirage.core
    from demo import make_kernel
    from trace_support import compile_options, export_trace

    torch.cuda.set_device(0)
    gpu = torch.cuda.get_device_properties(0)
    if (gpu.major, gpu.minor) != (9, 0) or args.slots + 2 > gpu.multi_processor_count:
        parser.error('Use Hopper with room for one worker per slot and two schedulers')
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save = lambda name, value: (out / name).write_text(json.dumps(value, indent=2) + '\n')
    status = dict(generated=False, compiled=False, executed=False, validated=False)
    save('status.json', status)
    save('environment.json', dict(root=str(ROOT), mirage=mi.__file__, core=mirage.core.__file__,
        python=sys.executable, gpu=str(gpu), node=os.uname().nodename, job=os.getenv('SLURM_JOB_ID'),
        mode=args.mode, task_graph_mode='resident_data', slots=args.slots, workers=args.slots,
        local_schedulers=2, contexts=args.contexts, width=args.width,
        intermediate_width=args.intermediate_width, full_output=args.full_output,
        table_rows=args.table_rows, trials=args.trials, seed=1729,
        trace=args.trace, delay_us=args.delay_us, delay_slots=args.delay_slots))

    S, H, I = args.slots, args.width, args.intermediate_width
    def empty(width):
        return torch.empty(S, width, device='cuda', dtype=torch.bfloat16)

    ids = torch.randint(args.table_rows, (args.contexts, 1), device='cuda', dtype=torch.int64)
    table = torch.randn(args.table_rows, H, device='cuda', dtype=torch.bfloat16)
    norm_w = (1 + .1 * torch.randn(H, device='cuda')).bfloat16()
    # First I rows project gate; second I rows project up. Each is allocated once.
    gate_up_w = (torch.randn(2 * I, H, device='cuda') / H ** .5).bfloat16()
    down_w = (torch.randn(H, I, device='cuda') / I ** .5).bfloat16()
    coefficient = (torch.randn(H, device='cuda') / H ** .5).bfloat16()
    readonly = [ids, table, norm_w, gate_up_w, down_w, coefficient]
    originals = [tensor.clone() for tensor in readonly]
    slot_ids = torch.empty(S, 1, device='cuda', dtype=torch.int64)
    embedded, normalized, gate_up, activated, down = [empty(w) for w in (H, H, 2 * I, I, H)]
    slot_result = down if args.full_output else empty(1)
    saved = torch.full((args.contexts, H if args.full_output else 1), float('nan'),
                       device='cuda', dtype=torch.bfloat16)
    pk = make_kernel(mi, torch, slots=S, workers=S)
    attach = pk.attach_input
    grid, block = (S, 1, 1), (256, 1, 1)
    ids_t = attach(slot_ids.view(1, S), name='slot_ids')
    x = attach(embedded, name='embedded')
    pk.embed_layer(ids_t, attach(table, name='embedding_table'), x,
                   grid_dim=(1, S, 1), block_dim=block, input_source=1)
    n = attach(normalized, name='normalized')
    pk.rmsnorm_layer(x, attach(norm_w, name='norm_weight'), n, grid_dim=grid, block_dim=block)
    p = attach(gate_up, name='gate_up')
    pk.linear_layer(n, attach(gate_up_w, name='gate_up_weight'), p,
                    grid_dim=grid, block_dim=block, one_row_per_task=True)
    a = attach(activated, name='activated')
    pk.silu_mul_layer(p, a, grid_dim=grid, block_dim=block, one_row_per_task=True)
    y = attach(down, name='down')
    pk.linear_layer(a, attach(down_w, name='down_weight'), y,
                    grid_dim=grid, block_dim=block, one_row_per_task=True)
    if not args.full_output:
        pk.checksum_layer(y, attach(coefficient, name='checksum_coefficient'),
                          attach(slot_result, name='checksum'), grid_dim=grid, block_dim=block)

    # Read ownership from the actual native output slices, not task indices or
    # request_id metadata. Then reject any unexpected join between slot rows.
    graph = json.loads(pk.generate_task_graph()['json_file'])
    slots = []
    for data in graph['all_data']:
        view, = data['outputs']
        row_bytes = 2 * view['dims'][1]
        assert view['dims'][0] == 1 and view['offset'] % row_bytes == 0
        slot = view['offset'] // row_bytes
        assert 0 <= slot < S
        slots.append(slot)
    pk.set_request_slots(slots)
    graph = json.loads(pk.generate_task_graph()['json_file'])
    stages = 5 if args.full_output else 6
    assert len(graph['resident_tasks']) == stages and len(slots) == S * stages
    assert len(graph['data_edges']) == S * (stages - 1)
    assert all(slots[e['src_data_id']] == slots[e['dst_data_id']] for e in graph['data_edges'])
    for slot in graph['request_slots']:
        assert len(slot['data_ids']) == stages
        assert len(slot['first_data_ids']) == len(slot['terminal_data_ids']) == 1
    # Both projection matrices must describe one shared allocation, while every
    # linear and activation input slice covers exactly one complete row.
    for name in ('gate_up', 'activated', 'down'):
        data = [d for d in graph['all_data'] if d['outputs'][0]['base_ptr'] == name]
        assert len(data) == S and all(d['inputs'][0]['dims'][0] == 1 for d in data)
        if name != 'activated':
            assert all(d['inputs'][1] == data[0]['inputs'][1] for d in data)
    mapping = []
    for task_id, task in enumerate(graph['all_tasks']):
        d = task['data_id']
        if 0 <= d < len(slots):
            data = graph['all_data'][d]
            mapping.append(dict(task=task_id, data=d, resident=data['resident_task_id'],
                                slot=slots[d], output=data['outputs'][0]['base_ptr']))
    save('native_graph.json', graph)
    save('mapping.json', mapping)
    status['generated'] = True
    save('status.json', status)
    print(f'FFN: {S} slots/workers, H={H}, I={I}, {stages} resident families, {len(slots)} compute DataIds', flush=True)

    trace = torch.zeros((8193, 12), device='cuda', dtype=torch.int64) if args.trace else None
    terminals = {graph['request_slots'][slot]['terminal_data_ids'][0] for slot in args.delay_slots}
    delay_tasks = [row['task'] for row in mapping if row['data'] in terminals]
    save('delay_config.json', dict(slots=args.delay_slots, tasks=delay_tasks, delay_us=args.delay_us))
    with compile_options(trace, delay_tasks, args.delay_us, out):
        pk.compile(context_io=(ids, saved, slot_ids, slot_result), output_dir=str(out))
    status['compiled'] = True
    save('status.json', status)

    # Match rounding between operators, including the existing SiLU kernel's
    # BF16 rounding BEFORE multiplying by up. Matmul accumulates in FP32.
    reference = table[ids[:, 0]].float()
    reference = (reference * torch.rsqrt(reference.square().mean(-1, keepdim=True) + 1e-6)
                 * norm_w.float()).bfloat16()
    packed = (reference.float() @ gate_up_w.float().T).bfloat16()
    gate, up = packed[:, :I], packed[:, I:]
    activated_ref = (torch.nn.functional.silu(gate.float()).bfloat16() * up).bfloat16()
    reference = (activated_ref.float() @ down_w.float().T).bfloat16()
    if not args.full_output:
        reference = (reference.float() * coefficient.float()).sum(-1, keepdim=True).bfloat16()
    elapsed, errors = [], []
    for trial in range(-1, args.trials):
        # Warmup/trials each launch one complete invocation. Context admission
        # and every slot restart happen on the GPU, without per-pass host waits.
        saved.fill_(float('nan'))
        if trace is not None:
            trace.zero_()
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        pk()
        end.record()
        torch.cuda.synchronize()
        status['executed'] = True
        save('status.json', status)
        torch.testing.assert_close(saved, reference, atol=.015625, rtol=.02)
        assert all(torch.equal(a, b) for a, b in zip(readonly, originals))
        if trial >= 0:
            elapsed.append(begin.elapsed_time(end) * 1000)
            errors.append((saved.float() - reference.float()).abs().max().item())
    report = dict(contexts=args.contexts, slots=S, width=H, intermediate_width=I,
                  resident_families=stages, compute_data_ids=len(slots), mode=args.mode,
                  output='full vectors' if args.full_output else 'weighted BF16 checksums',
                  trials=args.trials, warmups=1, input_bytes_per_context=8,
                  saved_bytes_per_context=2 * saved.shape[1], atol=.015625, rtol=.02,
                  max_abs_error=max(errors), invocation_us=elapsed, readonly_inputs_unchanged=True)
    if trace is not None:
        report.update(export_trace(trace, graph, args.contexts, args.delay_us, out))
        report['trace_trial'] = args.trials - 1
    save('report.json', report)
    torch.save(dict(ids=ids.cpu(), saved=saved.cpu(), reference=reference.cpu()), out / 'outputs.pt')
    status['validated'] = True
    save('status.json', status)
    pk.finalize()
    print('PASS', json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
