#!/usr/bin/env python3
"""Independent slots: int64 ID -> embedding -> 3 RMSNorms -> BF16 checksum.

Run on a Hopper GPU, for example:
  python demo/context_io/embedding_checksum.py --contexts 20 --width 16384 \
    --mode hybrid_prelaunch --trials 5 --output experiments/checksum_run

The preloaded embedding table supplies synthetic context vectors. The scheduler
copies 8 input bytes and 2 output bytes per context. Each worker block still
processes a full H-element vector. No attention, KV, or host request submission.
Add --linear-width N for one complete H-to-N linear projection per slot before
the checksum. All slots share one weight allocation.
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
    parser.add_argument('--contexts', type=int, default=20)
    parser.add_argument('--width', type=int, default=16384)
    parser.add_argument('--table-rows', type=int, default=64)
    parser.add_argument('--linear-width', type=int, default=0,
                        help='optional one-row linear output width; positive multiple of 64, or 0 to disable')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--delay-us', type=int, default=0, help='finite terminal delay per selected slot pass; requires --trace')
    parser.add_argument('--delay-slots', type=int, nargs='+', default=[1],
                        help='slot IDs whose terminal tasks receive the delay (default: 1)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.contexts < 0 or args.width <= 0 or args.width % 512 or args.table_rows < 1
            or args.linear_width < 0 or args.linear_width % 64
            or args.trials < 1 or not 0 <= args.delay_us <= 100000):
        parser.error('Use nonnegative contexts, positive width divisible by 512, linear width 0 or a positive multiple of 64, positive table rows/trials, delay 0..100000')
    if args.delay_us and not args.trace:
        parser.error('--delay-us requires --trace')
    if 'MIRAGE_RESIDENT_SLOT_ITERATIONS' in os.environ:
        parser.error('Unset MIRAGE_RESIDENT_SLOT_ITERATIONS; the input queue bounds this run')
    sys.path.insert(0, str(ROOT / 'python'))
    os.environ.update(MIRAGE_HOME=str(ROOT), MIRAGE_ROOT=str(ROOT),
                      MIRAGE_TASK_GRAPH_MODE='resident_data', MIRAGE_RESIDENT_EXECUTION_MODE=args.mode)
    import torch
    import mirage as mi
    import mirage.core
    from demo import make_kernel, SLOTS, STAGES
    from trace_support import compile_options, export_trace

    args.delay_slots = sorted(set(args.delay_slots))
    if any(slot < 0 or slot >= SLOTS for slot in args.delay_slots):
        parser.error(f'--delay-slots must be in 0..{SLOTS - 1}')

    assert torch.cuda.get_device_capability(0) == (9, 0)
    torch.cuda.set_device(0)
    torch.manual_seed(1729)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save = lambda name, value: (out / name).write_text(json.dumps(value, indent=2) + '\n')
    status = dict(generated=False, compiled=False, executed=False, validated=False)
    save('status.json', status)
    save('environment.json', dict(
        root=str(ROOT), mirage=mi.__file__, core=mirage.core.__file__, python=sys.executable,
        gpu=str(torch.cuda.get_device_properties(0)), node=os.uname().nodename,
        job=os.environ.get('SLURM_JOB_ID'), mode=args.mode, contexts=args.contexts,
        width=args.width, linear_width=args.linear_width, table_rows=args.table_rows, slots=SLOTS, stages=STAGES,
        seed=1729, delay_us=args.delay_us, delay_slots=args.delay_slots, trace=args.trace))

    # IDs index a shared, read-only table; context ID still identifies the queue
    # position/results row, even when different contexts choose the same ID.
    ids = torch.randint(args.table_rows, (args.contexts, 1), device='cuda', dtype=torch.int64)
    original_ids = ids.clone()
    table = torch.randn(args.table_rows, args.width, device='cuda', dtype=torch.bfloat16)
    original_table = table.clone()
    slot_ids = torch.empty(SLOTS, 1, device='cuda', dtype=torch.int64)
    embedded = torch.empty(SLOTS, args.width, device='cuda', dtype=torch.bfloat16)
    working = [torch.empty_like(embedded) for _ in range(STAGES)]
    norm_weights = [(1 + .1 * torch.randn(args.width, device='cuda')).bfloat16() for _ in range(STAGES)]
    output_width = args.linear_width or args.width
    if args.linear_width:
        # Allocate W once. Both row tasks refer to this same read-only tensor.
        linear_weight = (torch.randn(args.linear_width, args.width, device='cuda') / args.width ** .5).bfloat16()
        original_linear_weight = linear_weight.clone()
        projected = torch.empty(SLOTS, args.linear_width, device='cuda', dtype=torch.bfloat16)
    # Scale coefficients to keep this diagnostic dot product near unit scale.
    coefficient = (torch.randn(output_width, device='cuda') / output_width ** .5).bfloat16()
    slot_checksum = torch.empty(SLOTS, 1, device='cuda', dtype=torch.bfloat16)
    saved = torch.full((args.contexts, 1), float('nan'), device='cuda', dtype=torch.bfloat16)
    pk = make_kernel(mi, torch)
    # embed_layer partitions ID dimension 1 with grid.y. This view has exactly
    # the same storage as slot_ids[S,1]; each task still reads one owning ID.
    id_tensor = pk.attach_input(slot_ids.view(1, SLOTS), name='slot_ids')
    table_tensor = pk.attach_input(table, name='embedding_table')
    x = pk.attach_input(embedded, name='embedded')
    pk.embed_layer(id_tensor, table_tensor, x, grid_dim=(1, SLOTS, 1),
                   block_dim=(256, 1, 1), input_source=1)
    for stage, (buffer, weight) in enumerate(zip(working, norm_weights)):
        y = pk.attach_input(buffer, name=f'stage_{stage + 1}')
        w = pk.attach_input(weight, name=f'norm_w_{stage + 1}')
        pk.rmsnorm_layer(x, w, y, grid_dim=(SLOTS, 1, 1), block_dim=(256, 1, 1))
        x = y
    if args.linear_width:
        w = pk.attach_input(linear_weight, name='linear_weight')
        y = pk.attach_input(projected, name='linear_output')
        pk.linear_layer(x, w, y, grid_dim=(SLOTS, 1, 1), block_dim=(256, 1, 1),
                        one_row_per_task=True)
        x = y
    c = pk.attach_input(coefficient, name='checksum_coefficient')
    y = pk.attach_input(slot_checksum, name='checksum')
    pk.checksum_layer(x, c, y, grid_dim=(SLOTS, 1, 1), block_dim=(256, 1, 1))

    graph = json.loads(pk.generate_task_graph()['json_file'])
    data_slots = []
    for data in graph['all_data']:
        view, = data['outputs']
        assert view['dims'][0] == 1
        row_bytes = view['dims'][1] * 2  # Every compute output is BF16.
        assert view['offset'] % row_bytes == 0
        slot = view['offset'] // row_bytes
        assert 0 <= slot < SLOTS
        data_slots.append(slot)
    pk.set_request_slots(data_slots)
    graph = json.loads(pk.generate_task_graph()['json_file'])
    assert len(graph['resident_tasks']) == STAGES + 2 + bool(args.linear_width)
    assert all(data_slots[e['src_data_id']] == data_slots[e['dst_data_id']] for e in graph['data_edges'])
    for slot in graph['request_slots']:
        assert len(slot['first_data_ids']) == len(slot['terminal_data_ids']) == 1
        assert graph['all_data'][slot['terminal_data_ids'][0]]['outputs'][0]['base_ptr'] == 'checksum'
    if args.linear_width:
        linear_data = [d for d in graph['all_data'] if d['outputs'][0]['base_ptr'] == 'linear_output']
        assert len(linear_data) == SLOTS
        assert all(d['inputs'][0]['dims'] == [1, args.width] for d in linear_data)
        assert all(d['outputs'][0]['dims'] == [1, args.linear_width] for d in linear_data)
        assert all(d['inputs'][1] == linear_data[0]['inputs'][1] for d in linear_data)
    mapping = []
    for task_id, task in enumerate(graph['all_tasks']):
        d = task['data_id']
        if 0 <= d < len(data_slots):
            data = graph['all_data'][d]
            mapping.append(dict(task=task_id, data=d, resident=data['resident_task_id'],
                                slot=data_slots[d], output=data['outputs'][0]['base_ptr']))
    save('mapping.json', mapping)
    save('native_graph.json', graph)
    status['generated'] = True
    save('status.json', status)
    print(json.dumps(mapping, indent=2), flush=True)
    trace = torch.zeros((8193, 12), device='cuda', dtype=torch.int64) if args.trace else None
    # Select actual native terminal tasks; task indices need not equal slot IDs.
    delayed_data = {graph['request_slots'][slot]['terminal_data_ids'][0]
                    for slot in args.delay_slots}
    delay_tasks = [m['task'] for m in mapping if m['data'] in delayed_data]
    save('delay_config.json', dict(slots=args.delay_slots, tasks=delay_tasks,
                                  delay_us=args.delay_us, applies_to='every selected slot pass'))
    with compile_options(trace, delay_tasks, args.delay_us, out):
        pk.compile(context_io=(ids, saved, slot_ids, slot_checksum), output_dir=str(out))
    status['compiled'] = True
    save('status.json', status)

    reference = table[ids[:, 0]]
    for weight in norm_weights:
        reference = (reference.float() * torch.rsqrt(reference.float().square().mean(-1, keepdim=True) + 1e-6)
                     * weight.float()).bfloat16()
    if args.linear_width:
        reference = (reference.float() @ linear_weight.float().T).bfloat16()
    reference = (reference.float() * coefficient.float()).sum(-1, keepdim=True).bfloat16()
    elapsed, errors = [], []
    # One warmup, then independent whole-invocation trials. Admission and slot
    # restart happen inside each invocation, without any per-context host wait.
    for trial in range(-1, args.trials):
        saved.fill_(float('nan'))
        if trace is not None: trace.zero_()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        pk()
        end.record()
        torch.cuda.synchronize()
        status['executed'] = True
        save('status.json', status)
        torch.testing.assert_close(saved, reference, atol=.015625, rtol=.02)
        assert torch.equal(ids, original_ids) and torch.equal(table, original_table)
        if args.linear_width:
            assert torch.equal(linear_weight, original_linear_weight)
        error = (saved.float() - reference.float()).abs().max().item() if args.contexts else 0.0
        if trial >= 0:
            elapsed.append(start.elapsed_time(end) * 1000)
            errors.append(error)
    report = dict(contexts=args.contexts, width=args.width, linear_width=args.linear_width,
                  trials=args.trials, warmups=1,
                  input_bytes_per_context=8, saved_bytes_per_context=2,
                  checksum='BF16(sum_fp32(hidden[j] * coefficient[j]))',
                  atol=.015625, rtol=.02, max_abs_error=max(errors), invocation_us=elapsed,
                  input_queue_and_table_unchanged=True)
    if trace is not None:
        report.update(export_trace(trace, graph, args.contexts, args.delay_us, out))
        report['trace_trial'] = args.trials - 1
    save('report.json', report)
    torch.save(dict(ids=ids.cpu(), checksum=saved.cpu(), reference=reference.cpu()), out / 'outputs.pt')
    status['validated'] = True
    save('status.json', status)
    pk.finalize()
    print('PASS', json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
