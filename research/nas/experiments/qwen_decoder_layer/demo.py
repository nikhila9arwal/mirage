#!/usr/bin/env python3
"""Validate a configurable stack of real Qwen decoder layers in NAS.

Each fixed slot consumes successive, predetermined token IDs. The reference
prefills its private KV prefix; NAS then executes the complete layer stack for every
new token. Captures are correctness diagnostics, not performance measurements.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

SESSION = Path(__file__).resolve().parents[2]
ROOT = SESSION.parents[1]  # Repository root; research/nas holds the harness.
HERE = Path(__file__).resolve().parent
H, I, E, TOP_K = 2048, 768, 128, 8
KV_HEADS, Q_PER_KV, HEAD_DIM = 4, 8, 128
READOUT_STAGES = ('readout_norm', 'readout_logits', 'predicted_tokens')
STAGES = (
    'input_norm', 'qkv', 'attention', 'projected',
    'attention_residual', 'post_norm', 'router_logits', 'routing_weights',
    'gate_up', 'activated', 'down', 'combined', 'final',
)


def next_power_of_two(value):
    return 1 << (value - 1).bit_length()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def assign_slots(graph, buffers, slots):
    """Derive ownership from native output views, including each KV head."""
    result = []
    for node in graph['all_data']:
        owners = []
        for view in node['outputs']:
            tensor = buffers[view['base_ptr']]
            # Chunk activations flatten [slot, token, feature] to [S*P,F].
            # Caches/readout still use [S,...]; total bytes per slot covers both.
            row_bytes = tensor.numel() * tensor.element_size() // slots
            slot = view['offset'] // row_bytes
            extent = tensor.element_size() * (1 + sum(
                (size - 1) * stride for size, stride in zip(view['dims'], view['strides'])))
            assert 0 <= slot < slots
            assert view['offset'] + extent <= (slot + 1) * row_bytes
            owners.append(slot)
        assert owners and len(set(owners)) == 1
        result.append(owners[0])
    return result


def validate_graph(graph, slots, layers, readout=False, chunk_rows=1):
    """Check native layer edges and the optional slot-local readout terminal."""
    data, families = graph['all_data'], graph['resident_tasks']
    stages = ['embedded'] + [f'layer{layer}_{stage}' for layer in range(layers) for stage in STAGES]
    if readout and chunk_rows > 1:
        stages += ['last_hidden']
    if readout:
        stages += list(READOUT_STAGES)
    extra = (len(READOUT_STAGES) + (chunk_rows > 1)) if readout else 0
    assert len(families) == 1 + layers * len(STAGES) + extra
    assert len(data) == slots * (1 + layers * (len(STAGES) + KV_HEADS - 1) + extra)
    assert graph['num_request_slots'] == slots
    tasks = {task['data_id']: index for index, task in enumerate(graph['all_tasks'])
             if 0 <= task['data_id'] < len(data)}
    assert len(tasks) == len(data)
    edges = {(edge['src_data_id'], edge['dst_data_id']) for edge in graph['data_edges']}
    assert len(edges) == len(graph['data_edges'])
    assert all(graph['data_slot_ids'][a] == graph['data_slot_ids'][b] for a, b in edges)
    incoming = Counter(b for _, b in edges)
    assert all(node['initial_predecessor_count'] == incoming[d] for d, node in enumerate(data))
    mapping, required, allowed, roots, terminals = [], set(), set(), [], []
    for slot, desc in enumerate(graph['request_slots']):
        groups = defaultdict(list)
        for d in desc['data_ids']:
            node = data[d]
            name = node['outputs'][0]['base_ptr']
            groups[name].append(d)
            assert graph['data_slot_ids'][d] == slot
            if name in ('embedded', 'last_hidden') or name in READOUT_STAGES:
                layer, operator = None, name
            else:
                tag, operator = name.split('_', 1)
                layer = int(tag.removeprefix('layer'))
                assert layer in range(layers)
            label = name
            if operator == 'attention':
                assert len(node['inputs']) == 7 and len(node['outputs']) == 3
                assert node['inputs'][1:3] == node['outputs'][1:3]
                assert node['inputs'][0]['dims'] == [chunk_rows, (Q_PER_KV + 2) * HEAD_DIM]
                assert node['outputs'][0]['dims'] == [chunk_rows, Q_PER_KV * HEAD_DIM]
                assert node['inputs'][1]['dims'] == [1, 64, 1, HEAD_DIM]
                assert node['inputs'][1]['base_ptr'] == f'layer{layer}_k_cache'
                assert node['inputs'][2]['base_ptr'] == f'layer{layer}_v_cache'
                row_bytes = chunk_rows * KV_HEADS * Q_PER_KV * HEAD_DIM * 2
                head = (node['outputs'][0]['offset'] % row_bytes) // (Q_PER_KV * HEAD_DIM * 2)
                label += f' / KV head {head}'
                assert all(view['offset'] == 0 for view in node['inputs'][3:])
            if operator in ('attention_residual', 'final'):
                assert len(node['inputs']) == 2 and len(node['outputs']) == 1
                assert all(view['dims'] == [chunk_rows, H] for view in node['inputs'] + node['outputs'])
            if operator == 'last_hidden':
                assert node['inputs'][0]['dims'] == [chunk_rows, H]
                assert node['outputs'][0]['dims'] == [1, H]
            if operator == 'predicted_tokens':
                assert len(node['inputs']) == len(node['outputs']) == 1
                assert node['inputs'][0]['dims'][0] == 1
                assert node['outputs'][0]['dims'] == [1, 1]
            mapping.append(dict(data=d, task=tasks[d], resident=node['resident_task_id'],
                slot=slot, layer=layer, operator=operator, name=label,
                task_type=families[node['resident_task_id']]['task_type']))
        assert set(groups) == set(stages)
        for name in stages:
            assert len(groups[name]) == (KV_HEADS if name.endswith('_attention') else 1)
        assert desc['first_data_ids'] == groups['embedded']
        terminal = 'predicted_tokens' if readout else f'layer{layers-1}_final'
        assert desc['terminal_data_ids'] == groups[terminal]
        assert desc['num_terminal_data'] == 1
        roots += groups['embedded']
        terminals += groups[terminal]
        for before, after in zip(stages, stages[1:]):
            required.update((a, b) for a in groups[before] for b in groups[after])
        for layer in range(layers):
            tag = f'layer{layer}_'
            layer_input = 'embedded' if layer == 0 else f'layer{layer-1}_final'
            # Redundant skip edges may be omitted when the chain already
            # enforces the same dependency. They may not cross slots/layers.
            for before, after in (
                (layer_input, tag + 'attention_residual'),
                (tag + 'attention_residual', tag + 'final'),
                (tag + 'post_norm', tag + 'gate_up'),
                (tag + 'routing_weights', tag + 'down'),
                (tag + 'routing_weights', tag + 'combined'),
            ):
                allowed.update((a, b) for a in groups[before] for b in groups[after])
    assert required <= edges <= required | allowed, 'Unexpected native dependency'
    assert set(graph['first_data_ids']) == set(roots)
    for layer in range(layers):
        for stage in ('qkv', 'projected', 'router_logits', 'gate_up', 'down'):
            name = f'layer{layer}_{stage}'
            views = [node['inputs'][1] for node in data if node['outputs'][0]['base_ptr'] == name]
            assert views and all(view == views[0] for view in views), 'Weights must be shared within a layer'
    if readout:
        for name in ('readout_norm', 'readout_logits'):
            views = [node['inputs'][1] for node in data if node['outputs'][0]['base_ptr'] == name]
            assert views and all(view == views[0] for view in views), 'Readout weights must be shared'
    return mapping, dict(layers=layers, readout=readout, resident_families=len(families), compute_data_ids=len(data),
        edges=len(edges), independent_slots=True, attention_tasks=layers * slots * KV_HEADS,
        roots=roots, terminals=terminals, residual_adds=2 * layers * slots,
        logical_moe_nodes=2 * layers * slots, branches_per_logical_moe=TOP_K,
        prefill_members_per_logical_moe=chunk_rows * TOP_K, chunk_rows=chunk_rows,
        layer_handoff_edges_verified=True, terminal_only_after_last_layer=not readout,
        terminal_after_readout=readout)


@contextmanager
def compile_options(args, out, capture_io, trace):
    """Use the existing bounded batching/capture switches for this demo only."""
    module = importlib.import_module('mirage.mpk.persistent_kernel')
    original = module.get_compile_command

    def command(*positional, **keywords):
        cmd = [arg for arg in original(*positional, **keywords) if arg != '-arch=sm_90a']
        flags = [f'-I{HERE}', '-DMIRAGE_EXPERIMENTAL_MOE_BATCHING=1',
            f'-DNAS_MOE_TOP_K={TOP_K}', f'-DNAS_MOE_STATE_SLOTS={args.slots}',
            f'-DNAS_MOE_LAYERS={args.layers}', f'-DNAS_MOE_H={H}', f'-DNAS_MOE_I={I}',
            f'-DNAS_COMPARE_IO_ADDRESS={capture_io.data_ptr()}ULL',
            f'-DNAS_MOE_WORKER_QUEUE_LEN={args.work_queue_capacity}',
            f'-DNAS_MOE_COMPLETION_QUEUE_LEN={args.work_queue_capacity}',
            f'-DNAS_MOE_SCHED_QUEUE_LEN={args.control_queue_capacity}']
        if trace is not None:
            flags += [f'-I{ROOT / "demo/context_io"}', '-DMIRAGE_END_TRACE',
                f'-DEND_TRACE_BUFFER_ADDRESS={trace.data_ptr()}ULL',
                f'-DEND_TRACE_CAPACITY={trace.shape[0] - 1}ULL',
                '-DEND_TRACE_DELAY_NS=0ULL', '-DEND_TRACE_DELAY_TASKS=0',
                '-DNAS_MOE_BRANCH_TRACE=1']
        cmd[1:1] = flags
        save_json(out / 'compile_command.json', cmd)
        return cmd

    module.get_compile_command = command
    try:
        yield
    finally:
        module.get_compile_command = original


def build_layer(pk, torch, x, weights, slots, layer_id, buffers, chunk_rows=1):
    """Append one complete layer; x is the previous layer's actual output."""
    tag = f'layer{layer_id}_'
    grid, block = (slots, 1, 1), (256, 1, 1)

    def work(name, shape, dtype=torch.bfloat16):
        # A cache belongs to the sequence; activations belong to its P-token
        # chunk. We reuse the same weights and operator order as decode.
        rows = slots if name.endswith('_cache') else slots * chunk_rows
        tensor = torch.empty((rows, *shape), device='cuda', dtype=dtype)
        buffers[tag + name] = tensor
        return pk.attach_input(tensor, name=tag + name)

    def weight(name):
        return pk.attach_input(weights[name], name=tag + name + '_weight')

    def linear(x, w, y):
        pk.linear_layer(x, w, y, grid_dim=grid, block_dim=block,
                        chunk_rows_per_task=chunk_rows)

    def add(x, residual, y):
        pk.elementwise_add_layer(x, residual, y, grid_dim=grid, block_dim=block)

    # Self attention, followed by the first residual. QKV packs Q,K,V per KV head.
    normalized = work('input_norm', (H,))
    pk.rmsnorm_layer(x, weight('input_norm'), normalized, grid_dim=grid, block_dim=block)
    qkv = work('qkv', (KV_HEADS * (Q_PER_KV + 2) * HEAD_DIM,))
    linear(normalized, weight('qkv'), qkv)
    k_cache = work('k_cache', (64, KV_HEADS, HEAD_DIM))
    v_cache = work('v_cache', (64, KV_HEADS, HEAD_DIM))
    attention = work('attention', (KV_HEADS * Q_PER_KV * HEAD_DIM,))
    pk.attention_slot_layer(qkv, k_cache, v_cache, attention,
        q_norm=weight('q_norm'), k_norm=weight('k_norm'),
        cos_pos_embed=weight('cos'), sin_pos_embed=weight('sin'), qk_norm_eps=1e-6,
        chunk_rows_per_task=chunk_rows)
    projected = work('projected', (H,))
    linear(attention, weight('wo'), projected)
    h = work('attention_residual', (H,))
    add(projected, x, h)

    # One logical MoE node owns all eight branches. Its existing completion
    # counter joins W13 before SiLU and W2 before the weighted reduction.
    post = work('post_norm', (H,))
    pk.rmsnorm_layer(h, weight('post_norm'), post, grid_dim=grid, block_dim=block)
    logits = work('router_logits', (E,))
    linear(post, weight('router_weight'), logits)
    probabilities = work('routing_weights', (TOP_K,), torch.float32)
    ranks = work('routing_ranks', (E,), torch.int32)
    pk.moe_topk_row_layer(logits, probabilities, ranks, grid_dim=grid, block_dim=block,
                         chunk_rows_per_task=chunk_rows)
    gate_up = work('gate_up', (TOP_K, 2 * I))
    pk.moe_slot_w13_layer(post, weight('w13'), ranks, gate_up, grid_dim=grid, block_dim=block,
                         chunk_rows_per_task=chunk_rows)
    activated = work('activated', (TOP_K, I))
    pk.moe_silu_mul_layer(gate_up, activated, grid_dim=grid, block_dim=block,
                         chunk_rows_per_task=chunk_rows)
    down = work('down', (TOP_K, H))
    pk.moe_slot_w2_layer(activated, weight('w2'), ranks, down, grid_dim=grid, block_dim=block,
                        chunk_rows_per_task=chunk_rows)
    combined = work('combined', (H,))
    pk.moe_weighted_sum_layer(down, probabilities, combined, grid_dim=grid, block_dim=block,
                            chunk_rows_per_task=chunk_rows)
    final = work('final', (H,))
    add(combined, h, final)
    return final


def build_readout(pk, torch, x, weights, slots, buffers):
    """Append three ordinary tasks per slot; this helper also accepts saved states.

    The LM head is untied: it reads its own checkpoint allocation. The predicted
    int64 is a diagnostic output in this patch, not the next embedding input.
    """
    grid, block = (slots, 1, 1), (256, 1, 1)
    width, vocab = weights['norm'].numel(), weights['lm_head'].shape[0]
    assert tuple(weights['lm_head'].shape) == (vocab, width)

    def work(name, shape, dtype=torch.bfloat16):
        tensor = torch.empty((slots, *shape), device='cuda', dtype=dtype)
        buffers[name] = tensor
        return pk.attach_input(tensor, name=name)

    normalized = work('readout_norm', (width,))
    logits = work('readout_logits', (vocab,))
    prediction = work('predicted_tokens', (1,), torch.int64)
    pk.rmsnorm_layer(x, pk.attach_input(weights['norm'], name='final_norm_weight'),
                     normalized, grid_dim=grid, block_dim=block)
    pk.linear_layer(normalized, pk.attach_input(weights['lm_head'], name='lm_head_weight'),
                    logits, grid_dim=grid, block_dim=block, one_row_per_task=True)
    pk.argmax_row_layer(logits, prediction, grid_dim=grid, block_dim=block)
    return prediction


def build_graph(pk, torch, weights, slots, layers, readout=False, chunk_rows=1):
    """Embed once, then connect all layers inside the same slot epoch."""
    buffers = {}
    slot_ids = torch.empty((slots, chunk_rows), device='cuda', dtype=torch.int64)
    embedded = torch.empty((slots * chunk_rows, H), device='cuda', dtype=torch.bfloat16)
    buffers.update(slot_ids=slot_ids, embedded=embedded)
    x = pk.attach_input(embedded, name='embedded')
    pk.embed_layer(pk.attach_input(slot_ids.view(1, slots * chunk_rows), name='slot_ids'),
        pk.attach_input(weights['embedding'], name='embedding_weight'), x,
        grid_dim=(1, slots, 1), block_dim=(256, 1, 1), input_source=1)
    for layer_id in range(layers):
        x = build_layer(pk, torch, x, weights['layers'][layer_id], slots, layer_id, buffers,
                        chunk_rows=chunk_rows)
    if readout and chunk_rows > 1:
        # One prediction per pass: choose the final valid row, not every prompt
        # token. The existing runtime ignores intermediate-chunk predictions.
        buffers['last_hidden'] = torch.empty((slots, H), device='cuda', dtype=torch.bfloat16)
        last = pk.attach_input(buffers['last_hidden'], name='last_hidden')
        pk.gather_last_slot_row_layer(x, last, grid_dim=(slots, 1, 1), block_dim=(256, 1, 1),
                                     chunk_rows_per_task=chunk_rows)
        x = last
    if readout:
        build_readout(pk, torch, x, weights['readout'], slots, buffers)
    return buffers


def cpu_tree(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item, torch) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_tree(item, torch) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers', type=int, default=48, help='Number of checkpoint layers, 1..48')
    parser.add_argument('--slots', type=int, choices=(16, 32), default=16)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--prefix', type=int, default=60)
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--batch-rows', type=int, default=4)
    parser.add_argument('--timeout-us', type=int, default=100)
    parser.add_argument('--readout', action='store_true',
                        help='Append checkpoint final norm, LM head, and per-slot argmax (48 layers only)')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not (1 <= args.layers <= 48 and 1 <= args.steps <= 4 and 0 <= args.prefix <= 64 - args.steps
            and 1 <= args.trials <= 3 and 1 <= args.batch_rows <= 16
            and 1 <= args.timeout_us <= 1000000):
        parser.error('Use 1..48 layers, 1..4 steps, prefix+steps<=64, 1..3 trials, B=1..16, T=1..1000000 us')
    if args.readout and args.layers != 48:
        parser.error('--readout requires all 48 checkpoint layers')
    layers = args.layers
    # A layer has 14 ordinary tasks and two logical MoE tasks, each hiding
    # eight branch computations. Assume no batching savings. The existing host
    # validator uses this same conservative whole-invocation bound; larger
    # graphs need larger rings, not a change to queue/scheduling semantics.
    contexts = args.slots * args.steps
    readout_tasks = len(READOUT_STAGES) if args.readout else 0
    physical_work_bound = contexts * (1 + layers * (14 + 2 * TOP_K) + readout_tasks)
    args.work_queue_capacity = next_power_of_two(physical_work_bound + 2)
    args.control_queue_capacity = next_power_of_two(2 * contexts + 3)
    # Bound diagnostic events separately; none of these sizes changes readiness.
    args.trace_capacity = next_power_of_two(1024 + contexts * (32 + 512 * layers + 8 * readout_tasks))
    out = args.output.resolve()
    if SESSION not in out.parents:
        parser.error('Keep output in the isolated experiment directory')
    out.mkdir(parents=True, exist_ok=False)
    status = dict(generated=False, compiled=False, executed=False, validated=False)
    save_json(out / 'status.json', status)
    save_json(out / 'arguments.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'demo/context_io')]
    os.environ.update(MIRAGE_HOME=str(ROOT), MIRAGE_ROOT=str(ROOT),
        MIRAGE_TASK_GRAPH_MODE='resident_data', MIRAGE_RESIDENT_EXECUTION_MODE='scheduler_dispatch',
        MIRAGE_MOE_BATCH_ROWS=str(args.batch_rows), MIRAGE_MOE_TIMEOUT_US=str(args.timeout_us),
        MIRAGE_MOE_EARLY_COALESCE='0', MIRAGE_MOE_DEADLINE_COALESCE='0', MIRAGE_DENSE_BATCHING='none')
    os.environ.pop('MIRAGE_RESIDENT_SLOT_ITERATIONS', None)
    pk = None
    try:
        import torch
        import mirage as mi
        import mirage.core
        from reference import qwen, check_all, digest, selected_experts
        torch.cuda.set_device(0)
        assert torch.cuda.get_device_capability(0) == (9, 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        assert str(ROOT) in mi.__file__ and str(ROOT) in mirage.core.__file__
        spec = importlib.util.spec_from_file_location('context_demo', ROOT / 'demo/context_io/demo.py')
        context_demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(context_demo)
        save_json(out / 'environment.json', dict(root=str(ROOT), mirage=mi.__file__, core=mirage.core.__file__,
            python=sys.executable, torch=torch.__version__, gpu=str(torch.cuda.get_device_properties(0)),
            node=os.uname().nodename, job=os.getenv('SLURM_JOB_ID'), workers=16, schedulers=1,
            mode='resident_data/scheduler_dispatch', layers=layers, readout=args.readout, slots=args.slots, steps=args.steps, H=H, I=I, E=E, top_k=TOP_K,
            head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
        print(f'Preparing {layers} checkpoint layers with a layerwise reference...', flush=True)
        with torch.inference_mode():
            data = qwen(slots=args.slots, steps=args.steps, prefix=args.prefix, layers=layers, readout=args.readout)
        save_json(out / 'provenance.json', data['provenance'])
        # Reuse the cached checkpoint; save reference outputs without duplicating
        # its large read-only weights in every diagnostic run directory.
        torch.save(cpu_tree({k: v for k, v in data.items() if k != 'weights'}, torch), out / 'reference.pt')
        contexts = args.slots * args.steps
        weights = data['weights']
        ids = data['token_ids'].reshape(contexts, 1).contiguous().to('cuda')
        pk = context_demo.make_kernel(mi, torch, slots=args.slots, workers=16, metadata_pages=128)
        pk.num_local_schedulers = 1
        print(f'Generating the native {layers}-layer graph...', flush=True)
        buffers = build_graph(pk, torch, weights, args.slots, layers, readout=args.readout)
        graph = json.loads(pk.generate_task_graph()['json_file'])
        pk.set_request_slots(assign_slots(graph, buffers, args.slots))
        graph = json.loads(pk.generate_task_graph()['json_file'])
        save_json(out / 'native_graph.json', graph)
        status['generated'] = True
        save_json(out / 'status.json', status)
        mapping, graph_report = validate_graph(graph, args.slots, layers, readout=args.readout)
        save_json(out / 'mapping.json', mapping)
        save_json(out / 'graph_validation.json', graph_report)

        # Top-k destroys its logit scratch. Capture all stable outputs, including
        # every branch and the cache snapshot before this slot is reused.
        captures = {name: torch.empty((contexts, *value.shape[1:]), device='cuda', dtype=value.dtype)
            for name, value in buffers.items() if name != 'slot_ids' and not name.endswith('_router_logits')}
        descriptors = torch.tensor([[buffers[name].data_ptr(), target.data_ptr(),
            buffers[name][0].numel() * buffers[name].element_size() // 2]
            for name, target in captures.items()], device='cuda', dtype=torch.int64)
        visits = torch.zeros(contexts, device='cuda', dtype=torch.int32)
        owners = torch.empty_like(visits)
        capture_io = torch.tensor([descriptors.data_ptr(), len(captures), visits.data_ptr(), owners.data_ptr()],
                                 device='cuda', dtype=torch.int64)
        saved = torch.empty((contexts, H), device='cuda', dtype=torch.bfloat16)
        trace = torch.zeros((args.trace_capacity + 1, 12), device='cuda', dtype=torch.int64) if args.trace else None
        readonly = dict(ids=ids, embedding=weights['embedding'])
        for layer_id, layer_weights in enumerate(weights['layers']):
            readonly.update({f'layer{layer_id}_{name}': value for name, value in layer_weights.items()})
        if args.readout:
            readonly.update({f'readout_{name}': value for name, value in weights['readout'].items()})
        print('Hashing read-only weights before execution...', flush=True)
        before = {name: digest(value) for name, value in readonly.items()}
        save_json(out / 'readonly_before.json', before)
        print('Compiling the native-generated persistent kernel...', flush=True)
        # ContextIO still saves the decoder hidden vector. Its raw-bit capture
        # hook also saves the int64 prediction before the slot is reused.
        with compile_options(args, out, capture_io, trace):
            pk.compile(context_io=(ids, saved, buffers['slot_ids'], buffers[f'layer{layers-1}_final']),
                       context_sequence=(args.steps, args.prefix), output_dir=str(out))
        status['compiled'] = True
        save_json(out / 'status.json', status)
        reports, previous = [], None
        for trial in range(args.trials):
            print(f'Running trial {trial + 1}/{args.trials}...', flush=True)
            trial_out = out / f'trial_{trial}'
            trial_out.mkdir()
            for tensor in list(buffers.values()) + list(captures.values()):
                tensor.fill_(-1 if tensor.dtype in (torch.int32, torch.int64) else float('nan'))
            for layer_id in range(layers):
                buffers[f'layer{layer_id}_k_cache'].copy_(data['initial_k'][layer_id])
                buffers[f'layer{layer_id}_v_cache'].copy_(data['initial_v'][layer_id])
            visits.zero_()
            owners.fill_(-1)
            saved.fill_(float('nan'))
            if trace is not None:
                trace.zero_()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            pk()
            end.record()
            torch.cuda.synchronize()
            status['executed'] = True
            save_json(out / 'status.json', status)
            host = {name: value.cpu() for name, value in captures.items()}
            torch.save(dict(saved=saved.cpu(), captures=host, visits=visits.cpu(), owners=owners.cpu()),
                       trial_out / 'outputs.pt')
            if trace is not None:
                torch.save(trace.cpu(), trial_out / 'raw_trace.pt')
            print(f'Checking all {layers} layers after trial {trial + 1}...', flush=True)
            with torch.inference_mode():
                numerical = check_all(captures, saved, data)
            repeated = previous is None or all(torch.equal(value, previous[name]) for name, value in host.items())
            previous = host
            checks = dict(contexts_saved_once=bool((visits == 1).all()),
                sequence_owners_correct=bool(torch.equal(owners, torch.arange(contexts, device='cuda', dtype=torch.int32) // args.steps)),
                saved_equals_terminal=torch.equal(saved, captures[f'layer{layers-1}_final']),
                embedding_exact=torch.equal(host['embedded'], data['inputs'].reshape(contexts, H).cpu()),
                repeated_outputs_exact=repeated)
            # Verify prefix and future positions at EACH observed pass, not just
            # at the final cache state (which could hide an earlier overwrite).
            for layer_id in range(layers):
                for name in ('k', 'v'):
                    snapshots = host[f'layer{layer_id}_{name}_cache'].reshape(
                        args.slots, args.steps, 64, KV_HEADS, HEAD_DIM)
                    initial = data['initial_' + name][layer_id].cpu()
                    checks[f'layer{layer_id}_{name}_prefix_unchanged'] = all(torch.equal(
                        snapshots[:, t, :args.prefix], initial[:, :args.prefix]) for t in range(args.steps))
                    checks[f'layer{layer_id}_{name}_future_unchanged'] = all(torch.equal(
                        snapshots[:, t, args.prefix + t + 1:], initial[:, args.prefix + t + 1:]) for t in range(args.steps))
            report = dict(numerical=numerical, checks=checks, diagnostic_invocation_ms=start.elapsed_time(end))
            if trace is not None:
                from trace import export_trace
                experts = {layer: selected_experts(captures[f'layer{layer}_routing_ranks']).cpu().tolist()
                           for layer in range(layers)}
                report['trace'] = export_trace(trace, graph, mapping, contexts, trial_out, expected_experts=experts)
            report['passed'] = numerical['passed'] and all(checks.values())
            save_json(trial_out / 'report.json', report)
            reports.append(report)
        print('Checking read-only weight hashes after execution...', flush=True)
        after = {name: digest(value) for name, value in readonly.items()}
        save_json(out / 'readonly_after.json', after)
        passed = before == after and all(report['passed'] for report in reports)
        save_json(out / 'report.json', dict(passed=passed, graph=graph_report, trials=reports,
            readonly_unchanged=before == after, timing='Correctness captures/tracing included; not a performance benchmark'))
        status['validated'] = passed
        save_json(out / 'status.json', status)
        print(('PASS: ' if passed else 'FAIL: ') + str(out), flush=True)
        if not passed:
            raise RuntimeError('Full-stack validation failed; raw outputs and reports were preserved')
    except BaseException as error:
        status['error'] = repr(error)
        save_json(out / 'status.json', status)
        (out / 'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        if pk is not None and status['compiled']:
            pk.finalize()


if __name__ == '__main__':
    main()
