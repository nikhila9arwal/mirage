"""Load the existing packed NAS weights and check their exact bytes.

Reload helpers are reused unchanged from the verified cache experiment.
Preparation and resumable cache-writing tools stay in that experiment; this
module only loads an existing complete cache. Hashing is outside load timing.
"""
import hashlib
import json
from pathlib import Path
import time

FORMAT = "nas_qwen_packed_v1"


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def fingerprint(tensor):
    # Called outside GPU loading timings. Hash exact BF16 bytes, not float casts.
    return hashlib.sha256(memoryview(tensor.detach().contiguous().view(
        __import__('torch').uint8).cpu().numpy())).hexdigest()


def source_identity(checkpoint):
    manifest = json.loads(Path(checkpoint).read_text())
    snapshot = Path(manifest['files']['config.json']).parent
    return snapshot, dict(model=manifest['model'], revision=manifest['revision'],
                          config_sha256=hashlib.sha256((snapshot/'config.json').read_bytes()).hexdigest())


def cache_manifest(directory, snapshot):
    directory = Path(directory)
    info = json.loads((directory/'manifest.json').read_text())
    if info['format'] != FORMAT or info['layers'] != 48:
        raise ValueError('Unsupported packed-cache format or layer count')
    if info['source']['config_sha256'] != hashlib.sha256((snapshot/'config.json').read_bytes()).hexdigest():
        raise ValueError('Packed cache config differs from the workload')
    expected = {f'layer_{i:02d}.safetensors' for i in range(48)} | {'common.safetensors'}
    if {f['file'] for f in info['files']} != expected or len(info['files']) != len(expected):
        raise ValueError('Incomplete/duplicate packed-cache files')
    for f in info['files']:
        if (directory/f['file']).stat().st_size != f['bytes']:
            raise ValueError('Packed-cache size mismatch: ' + f['file'])
    return info


def verify_weights(weights, manifest):
    # Original and packed loaders are checked against exactly the same
    # verified packed-cache hashes. This avoids holding two full models in GPU memory.
    checked = 0
    for item in manifest['files']:
        name = item['file']
        if name == 'common.safetensors':
            actual = dict(embedding=weights['embedding'], **weights['readout'])
        else:
            actual = weights['layers'][int(name[6:8])]
        for key, expected in item['tensors'].items():
            tensor = actual[key]
            if list(tensor.shape) != expected['shape'] or str(tensor.dtype) != expected['dtype']:
                raise ValueError(f'Shape/dtype mismatch in {name}:{key}')
            if fingerprint(tensor) != expected['sha256']:
                raise ValueError(f'Weight-byte mismatch in {name}:{key}')
            checked += 1
    for key, expected in manifest.get('rope', {}).items():
        if fingerprint(weights['layers'][0][key]) != expected:
            raise ValueError('RoPE bytes differ from cache preparation: ' + key)
    return checked


def load_packed(torch, config, snapshot, directory, progress):
    from safetensors import safe_open
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding
    info = cache_manifest(directory, snapshot)
    directory = Path(directory)
    with torch.inference_mode():
        rotary = Qwen3MoeRotaryEmbedding(config).to('cuda')
        shape = torch.empty((1,64,config.hidden_size), device='cuda', dtype=torch.bfloat16)
        cos, sin = rotary(shape, torch.arange(64,device='cuda')[None,:])
        cos, sin = cos[0].contiguous(), sin[0].contiguous()
        weights = dict(layers=[])
        for layer in range(config.num_hidden_layers):
            with safe_open(str(directory/f'layer_{layer:02d}.safetensors'), framework='pt', device='cpu') as f:
                packed = {key:f.get_tensor(key).to('cuda') for key in f.keys()}
            packed.update(cos=cos, sin=sin)
            weights['layers'].append(packed)
            progress('packed_layer_ready', completed_layers=layer+1)
        with safe_open(str(directory/'common.safetensors'), framework='pt', device='cpu') as f:
            common = {key:f.get_tensor(key).to('cuda') for key in f.keys()}
        weights.update(embedding=common.pop('embedding'), readout=common)
    return weights, info


def load_for_experiment(torch, config, snapshot, owners, progress, directory, report_path,
                        compare_legacy=True):
    import gc
    from autoregressive import load_weights
    manifest = cache_manifest(directory, snapshot)
    report = dict(cache=str(directory), source=manifest['source'],
                  verification='Exact tensor bytes vs verified packed-cache manifest; hashes excluded from load timings',
                  order=['original','packed'] if compare_legacy else ['packed'],
                  cache_state='Not controlled; no filesystem cache flushing')
    save_json(report_path, report)
    if compare_legacy:
        torch.cuda.synchronize()
        begin = time.monotonic()
        progress('original_load')
        weights = load_weights(torch, config, snapshot, owners, progress)
        torch.cuda.synchronize()
        report['original_load_seconds'] = time.monotonic()-begin
        save_json(report_path, report)
        begin = time.monotonic()
        progress('verify_original_weights')
        report['original_tensors_verified'] = verify_weights(weights, manifest)
        report['original_hash_seconds'] = time.monotonic()-begin
        # RoPE is regenerated by the same GPU implementation in both loaders.
        rope = {k:weights['layers'][0][k].clone() for k in ('cos','sin')}
        del weights
        gc.collect()
        torch.cuda.empty_cache()
        save_json(report_path, report)
    begin = time.monotonic()
    progress('packed_load')
    weights, manifest = load_packed(torch, config, snapshot, directory, progress)
    torch.cuda.synchronize()
    report['packed_load_seconds'] = time.monotonic()-begin
    save_json(report_path, report)
    begin = time.monotonic()
    progress('verify_packed_weights')
    report['packed_tensors_verified'] = verify_weights(weights, manifest)
    report['packed_hash_seconds'] = time.monotonic()-begin
    if compare_legacy:
        report['rope_equal'] = all(torch.equal(rope[k],weights['layers'][0][k]) for k in rope)
        if not report['rope_equal']: raise ValueError('RoPE values changed')
    report['passed'] = True
    save_json(report_path, report)
    return weights
