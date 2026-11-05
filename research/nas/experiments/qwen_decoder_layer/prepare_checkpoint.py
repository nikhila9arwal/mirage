#!/usr/bin/env python3
"""Download the pinned checkpoint shards needed by the requested layer count.

This is a CPU-only preparation step. The existing HDD cache is reused; neither
GPU jobs nor the reference loader download files implicitly.
"""
import argparse
import json
from pathlib import Path

from reference import DEFAULT_CHECKPOINT


def prepare(layers, checkpoint=DEFAULT_CHECKPOINT, include_readout=False):
    from huggingface_hub import snapshot_download

    manifest = json.loads(Path(checkpoint).read_text())
    snapshot = Path(manifest['files']['config.json']).parent
    owners = json.loads((snapshot / 'model.safetensors.index.json').read_text())['weight_map']
    names = ['model.embed_tokens.weight']
    if include_readout:
        names += ['model.norm.weight', 'lm_head.weight']
    for layer in range(layers):
        names.extend(name for name in owners if name.startswith(f'model.layers.{layer}.'))
    shards = sorted({owners[name] for name in names})
    missing = [name for name in shards if not (snapshot / name).is_file()]
    print(f'Pinned model: {manifest["model"]}@{manifest["revision"]}', flush=True)
    print(f'{layers} layers need {len(shards)} shards; {len(missing)} are missing.', flush=True)
    if missing:
        # Keep all large downloads in the already-established HDD cache. Two
        # concurrent transfers keep this preparation straightforward and bounded.
        downloaded = Path(snapshot_download(
            repo_id=manifest['model'], revision=manifest['revision'],
            cache_dir=str(snapshot.parents[2]), allow_patterns=missing, max_workers=2))
        assert downloaded.resolve() == snapshot.resolve()
    for shard in shards:
        if not (snapshot / shard).is_file():
            raise FileNotFoundError(snapshot / shard)
    print('Checkpoint preparation complete.', flush=True)
    return dict(model=manifest['model'], revision=manifest['revision'],
                layers=layers, snapshot=str(snapshot), shards=shards,
                downloaded=missing, include_readout=include_readout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers', type=int, default=48)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--include-readout', action='store_true',
                        help='Also prepare the final norm and untied LM-head weights')
    args = parser.parse_args()
    if not 1 <= args.layers <= 48:
        parser.error('layers must be between 1 and 48')
    report = prepare(args.layers, args.checkpoint, args.include_readout)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
