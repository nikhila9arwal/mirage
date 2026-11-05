"""Complete Qwen decoder-stack reference, evaluated outside the NAS invocation.

The primary result is the installed Transformers decoder layer with checkpoint
weights. Local checks use actual NAS intermediates to explain rounding or locate
a failed stage; they never replace NAS routes or rescue a failed model check.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path

H, I, EXPERTS, TOP_K = 2048, 768, 128, 8
D, KV_HEADS, Q_PER_KV, PAGE = 128, 4, 8, 64
DEFAULT_CHECKPOINT = Path(
    '/work/hdd/bgpd/elu1/mpk_qwen_block_20261025T185208Z/'
    'verification/reference_01/checkpoint.json')


def digest(tensor):
    import torch
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(raw).hexdigest()


def error(actual, expected, *, atol=.03, rtol=.03, relative_limit=.02):
    """Declared before the GPU tests; report every failed element as well."""
    import torch
    a, b = actual.float(), expected.to(actual.device).float()
    delta = (a - b).abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    close = delta <= atol + rtol * b.abs()
    relative = (torch.linalg.vector_norm(delta) /
                torch.linalg.vector_norm(b).clamp_min(1e-12)).item()
    return dict(passed=finite and bool(close.all()) and relative <= relative_limit,
                finite=finite, max_abs=delta.max().item(), relative_l2=relative,
                atol=atol, rtol=rtol, relative_l2_limit=relative_limit,
                failing_elements=int((~close).sum()), elements=delta.numel())


def selected_experts(ranks):
    """Invert rank[expert]=branch+1; zero means this expert was not selected."""
    import torch
    assert bool(((ranks != 0).sum(-1) == TOP_K).all())
    choices = []
    for branch in range(TOP_K):
        matches = ranks == branch + 1
        assert bool((matches.sum(-1) == 1).all()), 'Missing or repeated top-k rank'
        choices.append(matches.to(torch.int64).argmax(-1))
    return torch.stack(choices, dim=-1)


def _routing(logits):
    import torch
    full = torch.softmax(logits, dim=-1, dtype=torch.float32)
    gates, experts = torch.topk(full, TOP_K, dim=-1)
    return experts, gates / gates.sum(-1, keepdim=True)


def _inspect_moe(x, logits, block):
    """Expose branches in the same expert order/arithmetic as HF's MoE block."""
    import torch
    experts, gates = _routing(logits)
    gates_bf16 = gates.to(x.dtype)
    gate_up = torch.empty((len(x), TOP_K, 2 * I), device=x.device, dtype=x.dtype)
    activated = torch.empty((len(x), TOP_K, I), device=x.device, dtype=x.dtype)
    down = torch.empty((len(x), TOP_K, H), device=x.device, dtype=x.dtype)
    combined = torch.zeros_like(x)
    for expert_id in experts.unique(sorted=True).tolist():
        rows, branches = torch.where(experts == expert_id)
        expert = block.experts[expert_id]
        gate, up = expert.gate_proj(x[rows]), expert.up_proj(x[rows])
        middle = expert.act_fn(gate) * up
        result = expert.down_proj(middle)
        gate_up[rows, branches] = torch.cat((gate, up), dim=-1)
        activated[rows, branches] = middle
        down[rows, branches] = result
        combined.index_add_(0, rows, result * gates_bf16[rows, branches, None])
    return dict(experts=experts, routing_weights=gates_bf16,
                routing_weights_fp32=gates, gate_up=gate_up,
                activated=activated, down=down, manual_combined=combined)


def _load_tensors(snapshot, owners, names):
    """Read only this layer's tensors; safetensors keeps the shard memory-mapped."""
    import torch
    from safetensors import safe_open

    state = {}
    for shard in sorted({owners[name] for name in names}):
        with safe_open(str(snapshot / shard), framework='pt', device='cpu') as handle:
            for name in names:
                if owners[name] == shard:
                    state[name] = handle.get_tensor(name).to(torch.bfloat16)
    return state


def _load_layer(config, layer_id, snapshot, owners):
    """Instantiate one real HF block, without allocating placeholder weights."""
    import torch
    from transformers.modeling_utils import no_init_weights
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer

    tag = f'model.layers.{layer_id}.'
    names = [name for name in owners if name.startswith(tag)]
    state = _load_tensors(snapshot, owners, names)
    # assign=True replaces meta parameters with checkpoint tensors. Only this
    # one block is copied to CUDA, so the full HF model never occupies VRAM.
    with torch.device('meta'), no_init_weights():
        block = Qwen3MoeDecoderLayer(config, layer_id)
    block.load_state_dict({name.removeprefix(tag): value for name, value in state.items()},
                          strict=True, assign=True)
    return block.to(device='cuda', dtype=torch.bfloat16).eval()


def _pack_layer(block, cos, sin):
    """Keep one NAS weight allocation per expert family, shared by all slots."""
    import torch
    attn = block.self_attn
    qkv = torch.cat((attn.q_proj.weight.reshape(KV_HEADS, Q_PER_KV, D, H),
                     attn.k_proj.weight.reshape(KV_HEADS, 1, D, H),
                     attn.v_proj.weight.reshape(KV_HEADS, 1, D, H)), dim=1).reshape(-1, H)
    # Copy each expert directly into its final slice. A list of concatenations
    # would retain another large temporary W13 tensor during torch.stack().
    w13 = torch.empty((EXPERTS, 2 * I, H), device='cuda', dtype=torch.bfloat16)
    w2 = torch.empty((EXPERTS, H, I), device='cuda', dtype=torch.bfloat16)
    for expert_id, expert in enumerate(block.mlp.experts):
        w13[expert_id, :I].copy_(expert.gate_proj.weight)
        w13[expert_id, I:].copy_(expert.up_proj.weight)
        w2[expert_id].copy_(expert.down_proj.weight)
    weights = dict(input_norm=block.input_layernorm.weight, qkv=qkv,
        q_norm=attn.q_norm.weight, k_norm=attn.k_norm.weight, cos=cos, sin=sin,
        wo=attn.o_proj.weight, post_norm=block.post_attention_layernorm.weight,
        router_weight=block.mlp.gate.weight, w13=w13, w2=w2)
    return {name: value.detach().contiguous() for name, value in weights.items()}


def _reference_layer(block, hidden, cos, sin, prefix, steps):
    """Evaluate one layer's prefix and decode sequence with its own HF cache.

    Causality permits this layer-major order: all of layer L's token outputs
    become layer L+1's inputs. Within a layer, decode still runs in token order,
    so each query sees exactly its prefix and preceding decoded tokens.
    """
    import torch
    from transformers.models.qwen3_moe.modeling_qwen3_moe import DynamicCache

    slots = hidden.shape[0]
    layer_id = block.self_attn.layer_idx
    cache = DynamicCache()
    output = torch.empty_like(hidden)
    initial_k = torch.full((slots, PAGE, KV_HEADS, D), .125,
                           device='cuda', dtype=torch.bfloat16)
    initial_v = torch.full_like(initial_k, -.25)
    if prefix:
        mask = torch.full((prefix, prefix), float('-inf'), device='cuda',
                          dtype=torch.bfloat16).triu(1)[None, None]
        output[:, :prefix] = block(hidden[:, :prefix],
            (cos[None, :prefix], sin[None, :prefix]), attention_mask=mask,
            past_key_values=cache)
        initial_k[:, :prefix] = cache.layers[layer_id].keys.transpose(1, 2)
        initial_v[:, :prefix] = cache.layers[layer_id].values.transpose(1, 2)

    captured = {}
    def output_hook(name):
        def hook(module, inputs, result):
            captured[name] = result.detach().clone()
        return hook
    def input_hook(name):
        def hook(module, inputs):
            captured[name] = inputs[0].detach().clone()
        return hook
    def moe_hook(module, inputs, result):
        captured['combined'] = result[0].detach().clone()
        captured['logits'] = result[1].detach().clone()

    attn = block.self_attn
    hooks = [
        block.input_layernorm.register_forward_hook(output_hook('input_norm')),
        attn.q_proj.register_forward_hook(output_hook('q')),
        attn.k_proj.register_forward_hook(output_hook('k')),
        attn.v_proj.register_forward_hook(output_hook('v')),
        attn.o_proj.register_forward_pre_hook(input_hook('attention')),
        attn.o_proj.register_forward_hook(output_hook('projected')),
        block.post_attention_layernorm.register_forward_pre_hook(input_hook('attention_residual')),
        block.post_attention_layernorm.register_forward_hook(output_hook('post_norm')),
        block.mlp.register_forward_hook(moe_hook),
    ]
    records = []
    try:
        for step in range(steps):
            pos = prefix + step
            x = block(hidden[:, pos:pos+1],
                (cos[None, pos:pos+1], sin[None, pos:pos+1]),
                attention_mask=None, past_key_values=cache)
            output[:, pos:pos+1] = x
            record = {name: captured[name].reshape(slots, -1) for name in
                      ('input_norm', 'attention', 'projected', 'attention_residual',
                       'post_norm', 'combined', 'logits')}
            record['qkv'] = torch.cat((
                captured['q'].reshape(slots, KV_HEADS, Q_PER_KV, D),
                captured['k'].reshape(slots, KV_HEADS, 1, D),
                captured['v'].reshape(slots, KV_HEADS, 1, D)), dim=2).reshape(slots, -1)
            record.update(_inspect_moe(record['post_norm'], record['logits'], block.mlp))
            torch.testing.assert_close(record.pop('manual_combined'), record['combined'],
                                       atol=.002, rtol=.002)
            record['input'] = hidden[:, pos]
            record['final'] = x[:, 0]
            record['k_cache'] = initial_k.clone()
            record['v_cache'] = initial_v.clone()
            record['k_cache'][:, :pos+1] = cache.layers[layer_id].keys.transpose(1, 2)
            record['v_cache'][:, :pos+1] = cache.layers[layer_id].values.transpose(1, 2)
            # Full-vector diagnostics live on the CPU. Only the current layer's
            # live intermediates are needed while preparing the next layer.
            records.append({name: value.detach().cpu() for name, value in record.items()})
    finally:
        for hook in hooks:
            hook.remove()
    reference = {name: torch.stack([record[name] for record in records], dim=1)
                 for name in records[0]}
    return output, initial_k.cpu(), initial_v.cpu(), reference


def qwen(slots=16, steps=4, prefix=60, checkpoint=DEFAULT_CHECKPOINT, layers=48, readout=False):
    """Build an exact checkpoint reference and packed NAS weights layer by layer.

    Previous NAS weights remain resident, but only one HF block exists at a
    time. Peak memory is the NAS model plus a temporary layer, not two models.
    Reference outputs and prefix snapshots are returned on CPU; weights, input
    token IDs, and input embeddings remain on CUDA for the NAS invocation.
    """
    import torch
    from transformers import AutoTokenizer, Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeDecoderLayer, Qwen3MoeRotaryEmbedding)

    assert slots >= 1 and steps >= 1 and 0 <= prefix <= PAGE - steps
    manifest = json.loads(Path(checkpoint).read_text())
    snapshot = Path(manifest['files']['config.json']).parent
    index_path = snapshot / 'model.safetensors.index.json'
    owners = json.loads(index_path.read_text())['weight_map']
    config = Qwen3MoeConfig.from_json_file(str(snapshot / 'config.json'))
    config._attn_implementation = 'eager'
    assert 1 <= layers <= config.num_hidden_layers == 48
    if readout:
        assert layers == config.num_hidden_layers and not config.tie_word_embeddings
    assert (config.hidden_size, config.moe_intermediate_size, config.num_experts,
            config.num_experts_per_tok) == (H, I, EXPERTS, TOP_K)
    assert (config.head_dim, config.num_key_value_heads,
            config.num_attention_heads) == (D, KV_HEADS, KV_HEADS * Q_PER_KV)
    assert config.norm_topk_prob and config.rms_norm_eps == 1e-6
    selected = ['model.embed_tokens.weight']
    if readout:
        selected += ['model.norm.weight', 'lm_head.weight']
    for layer_id in range(layers):
        selected += [name for name in owners if name.startswith(f'model.layers.{layer_id}.')]
    shards = sorted({owners[name] for name in selected})
    missing = [shard for shard in shards if not (snapshot / shard).is_file()]
    if missing:
        raise FileNotFoundError('Missing checkpoint shards: ' + ', '.join(missing) +
                                '. Run prepare_checkpoint.py before reserving a GPU.')

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    prompts = [
        'Explain how rain forms and why clouds sometimes produce snow instead. ',
        'Write a Python function that sums the squares of a list of integers. ',
        'Compare a bicycle and a train as ways to travel across a city. ',
        'A library has twelve shelves with twenty books each. Count the books. ',
    ]
    tokens = torch.tensor([
        tokenizer.encode(prompts[slot % len(prompts)] * 8)[:prefix + steps]
        for slot in range(slots)], device='cuda', dtype=torch.int64)
    assert tokens.shape == (slots, prefix + steps)
    embedding = _load_tensors(snapshot, owners, ['model.embed_tokens.weight'])[
        'model.embed_tokens.weight'].to('cuda').contiguous()
    hidden = torch.nn.functional.embedding(tokens, embedding)
    inputs = hidden[:, prefix:].contiguous()
    rotary = Qwen3MoeRotaryEmbedding(config).to('cuda')
    cos, sin = rotary(hidden, torch.arange(PAGE, device='cuda')[None, :])
    cos, sin = cos[0].contiguous(), sin[0].contiguous()

    layer_weights, initial_keys, initial_values, reference, memory = [], [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    for layer_id in range(layers):
        print(f'Reference layer {layer_id + 1}/{layers}: load, prefill, decode, pack', flush=True)
        block = _load_layer(config, layer_id, snapshot, owners)
        hidden, initial_k, initial_v, record = _reference_layer(
            block, hidden, cos, sin, prefix, steps)
        layer_weights.append(_pack_layer(block, cos, sin))
        initial_keys.append(initial_k)
        initial_values.append(initial_v)
        reference.append(record)
        # Packed expert tensors are independent copies. Small norm/dense weights
        # intentionally retain their allocations through layer_weights.
        del block
        torch.cuda.empty_cache()
        memory.append(dict(layer=layer_id, allocated_bytes=torch.cuda.memory_allocated(),
                           reserved_bytes=torch.cuda.memory_reserved(),
                           peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        print(f'  retained CUDA weights/state: {memory[-1]["allocated_bytes"] / 1e9:.3f} GB', flush=True)

    weights = dict(embedding=embedding, layers=layer_weights)
    readout_reference = None
    if readout:
        loaded = _load_tensors(snapshot, owners, ['model.norm.weight', 'lm_head.weight'])
        weights['readout'] = dict(norm=loaded['model.norm.weight'].to('cuda').contiguous(),
                                  lm_head=loaded['lm_head.weight'].to('cuda').contiguous())
        assert tuple(weights['readout']['norm'].shape) == (config.hidden_size,)
        assert tuple(weights['readout']['lm_head'].shape) == (config.vocab_size, config.hidden_size)
        # Only observed decode positions need vocabulary outputs. Prefix-cache
        # preparation and the 48-layer reference remain exactly as before.
        readout_reference = {name: value.cpu() for name, value in
            hf_readout(hidden[:, prefix:], weights['readout'], config.rms_norm_eps).items()}

    module_path = Path(__import__(Qwen3MoeDecoderLayer.__module__, fromlist=['']).__file__)
    return dict(token_ids=tokens[:, prefix:].contiguous(), inputs=inputs,
        weights=weights, readout_enabled=readout, readout_reference=readout_reference,
        initial_k=torch.stack(initial_keys), initial_v=torch.stack(initial_values),
        reference=reference, prefix=prefix, steps=steps, slots=slots, layers=layers,
        ratio=Q_PER_KV, kv_heads=KV_HEADS, eps=config.rms_norm_eps,
        provenance=dict(model=manifest['model'], revision=manifest['revision'],
            original_layers=config.num_hidden_layers, instantiated_layers=layers,
            layer_ids=list(range(layers)), checkpoint_shards=shards,
            checkpoint_index=str(index_path),
            source='Transformers Qwen3MoeDecoderLayer.forward; eager attention; DynamicCache',
            boundary=f'checkpoint embedding -> complete layers 0 through {layers - 1}, including every residual' +
                     (' -> final norm -> LM head -> greedy token' if readout else ''),
            readout_enabled=readout, vocab_size=config.vocab_size,
            prefill='Each complete layer feeds the next; separate KV entries per layer',
            reference_order='Layer-major; causal prefix then chronological decode steps within each layer',
            memory_policy='Packed NAS weights plus one temporary HF block; reference snapshots on CPU',
            preparation_memory=memory,
            token_ids=tokens.tolist(), prefix=prefix, steps=steps, slots=slots,
            H=H, I=I, experts=EXPERTS, top_k=TOP_K, head_dim=D,
            q_heads=KV_HEADS*Q_PER_KV, kv_heads=KV_HEADS,
            rms_norm_eps=config.rms_norm_eps, rope_theta=config.rope_theta,
            reference_source=str(module_path),
            reference_source_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
            versions={name: importlib.metadata.version(name) for name in
                      ('torch', 'transformers', 'safetensors')},
            tolerance=dict(atol=.03, rtol=.03, relative_l2=.02,
                           routing_weights_atol=.003, routing_weights_rtol=.03)))

def hf_readout(hidden, weights, eps):
    """The installed HF final norm and untied LM head, on the supplied states."""
    import torch
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm
    norm = Qwen3MoeRMSNorm(hidden.shape[-1], eps=eps)
    norm.weight = torch.nn.Parameter(weights['norm'], requires_grad=False)
    normalized = norm(hidden)
    logits = torch.nn.functional.linear(normalized, weights['lm_head'])
    return dict(readout_norm=normalized, readout_logits=logits,
                predicted_tokens=logits.argmax(-1, keepdim=True))


def check_readout(captures, hidden, weights, eps, hf_reference=None):
    """Separate local operator correctness from accumulated HF disagreement.

    Linear is checked on the captured NAS norm output; argmax is checked exactly
    on the captured NAS logits. The HF-style whole readout is also reported on
    the same incoming hidden vectors, without substituting any reference input.
    """
    import torch
    device = hidden.device
    c = {name: captures[name].to(device) for name in
         ('readout_norm', 'readout_logits', 'predicted_tokens')}
    w = {name: value.to(device) for name, value in weights.items()}
    contexts = hidden.shape[0]
    assert c['predicted_tokens'].dtype == torch.int64
    assert tuple(c['predicted_tokens'].shape) == (contexts, 1)
    assert tuple(c['readout_norm'].shape) == tuple(hidden.shape)
    assert tuple(c['readout_logits'].shape) == (contexts, w['lm_head'].shape[0])

    def predictions(expected):
        expected = expected.reshape(contexts, 1)
        same = c['predicted_tokens'] == expected
        return dict(passed=bool(same.all()), matching=int(same.sum()), contexts=contexts,
                    mismatch_contexts=torch.where(~same[:, 0])[0].cpu().tolist(),
                    nas_ids=c['predicted_tokens'][:, 0].cpu().tolist(),
                    expected_ids=expected[:, 0].cpu().tolist())

    finite = bool(torch.isfinite(hidden).all() and
                  torch.isfinite(c['readout_norm']).all() and
                  torch.isfinite(c['readout_logits']).all())
    in_range = bool(((c['predicted_tokens'] >= 0) &
                     (c['predicted_tokens'] < w['lm_head'].shape[0])).all())
    local = dict(
        norm=error(c['readout_norm'], _norm(hidden, w['norm'], eps)),
        # Retain the decoder's strict local-linear tolerance. A failed rounding
        # check remains failed even when the predicted token is unchanged.
        linear=error(c['readout_logits'],
                     torch.nn.functional.linear(c['readout_norm'], w['lm_head']),
                     atol=.002, rtol=.002),
        argmax=predictions(c['readout_logits'].argmax(-1, keepdim=True)))
    expected = hf_readout(hidden, w, eps)
    same_input_hf = dict(
        norm=error(c['readout_norm'], expected['readout_norm']),
        logits=error(c['readout_logits'], expected['readout_logits']),
        argmax=predictions(expected['predicted_tokens']))
    independent_hf = None
    if hf_reference is not None:
        # The live harness keeps [slot, step, ...]; cached standalone callers
        # may already flatten those leading axes to [context, ...].
        expected = {name: value.reshape(c[name].shape).to(device)
                    for name, value in hf_reference.items()}
        independent_hf = dict(
            norm=error(c['readout_norm'], expected['readout_norm']),
            logits=error(c['readout_logits'], expected['readout_logits']),
            argmax=predictions(expected['predicted_tokens']))
    local_passed = finite and in_range and all(item['passed'] for item in local.values())
    return dict(passed=local_passed, local_passed=local_passed, finite=finite,
                predicted_ids_in_range=in_range, local=local,
                same_input_hf=same_input_hf, independent_hf=independent_hf,
                policy='Local checks use actual NAS inputs and the existing NAS norm arithmetic. '
                       'HF comparisons are retained separately; readout success does not replace '
                       'the independent decoder numerical verdict.')


def _norm(x, weight, eps):
    """Current NAS norm rounds after multiplying weight; HF rounds before it."""
    import torch
    x = x.float()
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
            * weight.float()).bfloat16()


def _attention(data, actual_qkv):
    """Kernel-arithmetic diagnostic, including all preceding decoded KV writes."""
    import torch
    slots, steps = data['slots'], data['steps']
    groups = actual_qkv.reshape(slots, steps, KV_HEADS, Q_PER_KV+2, D)
    w = data['weights']
    k = data['initial_k'].to(actual_qkv.device).clone()
    v = data['initial_v'].to(actual_qkv.device).clone()
    outputs, keys, values = [], [], []
    def rotate(x, position):
        half = torch.cat((-x[..., D//2:], x[..., :D//2]), dim=-1)
        return (x.float()*w['cos'][position].float() +
                half.float()*w['sin'][position].float()).bfloat16()
    for step in range(steps):
        pos = data['prefix'] + step
        q = rotate(_norm(groups[:, step, :, :Q_PER_KV], w['q_norm'], data['eps']), pos)
        k[:, pos] = rotate(_norm(groups[:, step, :, Q_PER_KV], w['k_norm'], data['eps']), pos)
        v[:, pos] = groups[:, step, :, Q_PER_KV+1]
        scores = torch.einsum('shrd,sthd->shrt', q.float(), k[:, :pos+1].float()) * D**-.5
        probabilities = scores.softmax(-1).bfloat16()
        outputs.append(torch.einsum('shrt,sthd->shrd', probabilities.float(),
                                    v[:, :pos+1].float()).reshape(slots, -1).bfloat16())
        keys.append(k.clone()); values.append(v.clone())
    return dict(attention=torch.stack(outputs, 1), k_cache=torch.stack(keys, 1),
                v_cache=torch.stack(values, 1))


def _route_report(actual, expected, logits):
    import torch
    same = (actual.sort(-1).values == expected.sort(-1).values).all(-1)
    sorted_logits = logits.float().sort(descending=True).values
    margin = sorted_logits[:, TOP_K-1] - sorted_logits[:, TOP_K]
    return dict(passed=bool(same.all()), contexts=len(actual), same_sets=int(same.sum()),
                same_rank_order=int((actual == expected).all(-1).sum()),
                mismatch_contexts=torch.where(~same)[0].tolist(),
                boundary_tie_contexts=torch.where(margin == 0)[0].tolist(),
                eighth_ninth_logit_margin=margin.tolist(),
                actual_experts=actual.tolist(), reference_experts=expected.tolist())


def check_layer(captures, saved, data):
    """Full model comparisons plus stage-local diagnostics after the GPU run."""
    import torch
    C = data['slots'] * data['steps']
    device = saved.device
    c = {name: value.to(device) for name, value in captures.items()}
    w = {name: value.to(device) for name, value in data['weights'].items()}
    expected = {name: value.reshape(C, *value.shape[2:]).to(device)
                for name, value in data['reference'].items()}
    model = {name: error(c[name], expected[name]) for name in
             ('input', 'input_norm', 'qkv', 'attention', 'projected',
              'attention_residual', 'post_norm', 'combined', 'final', 'k_cache', 'v_cache')}
    experts = selected_experts(c['routing_ranks'])
    routing = _route_report(experts, expected['experts'], expected['logits'])
    matches = experts[:, :, None] == expected['experts'][:, None, :]
    present = matches.any(-1)
    reorder = matches.to(torch.int64).argmax(-1)
    rows = torch.arange(C, device=device)[:, None]
    # A different selected expert has no matching model branch. Report those
    # branches explicitly instead of comparing it to an arbitrary HF branch.
    branch_checks = {}
    for name in ('gate_up', 'activated', 'down', 'routing_weights'):
        if bool(present.any()):
            aligned = expected[name][rows, reorder]
            opts = dict(atol=.003, rtol=.03) if name == 'routing_weights' else {}
            branch_checks[name] = error(c[name][present], aligned[present], **opts)
            branch_checks[name]['compared_branches'] = int(present.sum())
            branch_checks[name]['unmatched_branches'] = int((~present).sum())
        else:
            branch_checks[name] = dict(passed=False, compared_branches=0,
                                       unmatched_branches=C*TOP_K)

    # These checks condition on NAS inputs. They localize errors without
    # mistaking expected upstream BF16 differences for a broken downstream op.
    linear = torch.nn.functional.linear
    local = dict(
        input_norm=error(c['input_norm'], _norm(c['input'], w['input_norm'], data['eps'])),
        qkv=error(c['qkv'], linear(c['input_norm'], w['qkv']), atol=.002, rtol=.002),
        projected=error(c['projected'], linear(c['attention'], w['wo']), atol=.002, rtol=.002),
        attention_residual=error(c['attention_residual'], c['input'] + c['projected'], atol=0, rtol=0),
        post_norm=error(c['post_norm'], _norm(c['attention_residual'], w['post_norm'], data['eps'])),
        activated=error(c['activated'], torch.nn.functional.silu(c['gate_up'][..., :I]) *
                        c['gate_up'][..., I:], atol=.002, rtol=.002),
        final=error(c['final'], c['attention_residual'] + c['combined'], atol=0, rtol=0))
    attention = _attention(data, c['qkv'])
    for name, value in attention.items():
        local[name] = error(c[name], value.reshape_as(c[name]))
    logits = linear(c['post_norm'], w['router_weight'])
    local_experts, local_weights = _routing(logits)
    local_routing = _route_report(experts, local_experts, logits)
    local_match = experts[:, :, None] == local_experts[:, None, :]
    local_present = local_match.any(-1)
    local_reorder = local_match.to(torch.int64).argmax(-1)
    if bool(local_present.any()):
        aligned = local_weights[rows, local_reorder]
        local['routing_weights'] = error(c['routing_weights'][local_present],
                                        aligned[local_present], atol=.001, rtol=.01)
    else:
        local['routing_weights'] = dict(passed=False)
    up, down = torch.empty_like(c['gate_up']), torch.empty_like(c['down'])
    for expert_id in experts.unique(sorted=True).tolist():
        context, branch = torch.where(experts == expert_id)
        up[context, branch] = linear(c['post_norm'][context], w['w13'][expert_id])
        down[context, branch] = linear(c['activated'][context, branch], w['w2'][expert_id])
    local['gate_up'] = error(c['gate_up'], up, atol=.002, rtol=.002)
    local['down'] = error(c['down'], down, atol=.002, rtol=.002)
    combined = torch.zeros_like(saved, dtype=torch.float32)
    for branch in range(TOP_K):
        combined += c['down'][:, branch].float() * c['routing_weights'][:, branch, None]
    local['combined'] = error(c['combined'], combined.bfloat16(), atol=.002, rtol=.002)
    return dict(passed=routing['passed'] and local_routing['passed'] and
                all(item['passed'] for group in (model, branch_checks, local) for item in group.values()),
                model=model, model_branches=branch_checks, routing=routing,
                local=local, local_routing=local_routing,
                router_logits='The top-k kernel overwrites its logits scratch; local logits are recomputed from captured post_norm.',
                policy='Primary HF full-layer result and routes must pass; local checks cannot replace them.')


def check_all(captures, saved, data):
    """Validate each layer against HF, while keeping NAS's own layer handoff."""
    import torch
    C = data['slots'] * data['steps']
    reports = []
    actual_input = captures['embedded']
    for layer_id in range(data['layers']):
        tag = f'layer{layer_id}_'
        local_captures = {name.removeprefix(tag): value for name, value in captures.items()
                          if name.startswith(tag)}
        local_captures['input'] = actual_input
        layer_data = dict(data, weights=data['weights']['layers'][layer_id],
            reference=data['reference'][layer_id],
            initial_k=data['initial_k'][layer_id], initial_v=data['initial_v'][layer_id])
        reports.append(check_layer(local_captures, local_captures['final'], layer_data))
        actual_input = local_captures['final']
    exact = dict(input_gather=torch.equal(captures['embedded'], data['inputs'].reshape(C, H)),
                 saved_equals_last_layer=torch.equal(saved, actual_input))
    result = dict(passed=all(item['passed'] for item in reports) and all(exact.values()),
                  layers=reports, exact=exact,
                  policy='Every independent HF layer reference must pass; each NAS layer feeds the next.')
    if data.get('readout_enabled', False):
        result['readout'] = check_readout(captures, actual_input, data['weights']['readout'],
                                          data['eps'], data['readout_reference'])
        result['passed'] = result['passed'] and result['readout']['local_passed']
    return result
