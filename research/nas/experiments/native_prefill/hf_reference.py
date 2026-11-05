"""Independent HF prefill and greedy continuation; nothing is imported into NAS."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
import traceback

from workload import validate, _checkpoint


def run(args):
    import torch
    from transformers import AutoTokenizer, Qwen3MoeForCausalLM
    from transformers.cache_utils import DynamicCache
    data = validate(json.loads(args.workload.read_text()))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def save(name, value):
        (out/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    def progress(phase, **details):
        row = dict(phase=phase, elapsed_seconds=time.monotonic()-started, **details)
        save('status.json', row)
        print(json.dumps(row), flush=True)
    try:
        manifest, snapshot, _ = _checkpoint(data['checkpoint'])
        if any(data[k] != manifest[k] for k in ('model','revision')):
            raise ValueError('Checkpoint provenance mismatch')
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        progress('load_checkpoint')
        model = Qwen3MoeForCausalLM.from_pretrained(snapshot, local_files_only=True,
            dtype=torch.bfloat16, device_map={'':'cuda'}, attn_implementation='eager').eval()
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        groups = defaultdict(list)
        for row in data['prompts']:
            groups[row['length']].append(row['request'])
        outputs = [[] for _ in data['prompts']]
        first_margins = [None for _ in data['prompts']]
        with torch.inference_mode():
            for length, requests in sorted(groups.items()):
                for begin in range(0, len(requests), args.batch_size):
                    members = requests[begin:begin+args.batch_size]
                    # Equal prompt lengths avoid padding and position ambiguity.
                    # Completed members may remain in HF's physical batch; only
                    # active members' predictions are accepted into their history.
                    tokens = torch.tensor([data['prompts'][r]['token_ids'] for r in members],
                                          dtype=torch.int64, device='cuda')
                    cache = DynamicCache(config=model.config)
                    progress('prefill', requests=members, prompt_length=length)
                    result = model(input_ids=tokens, past_key_values=cache, use_cache=True,
                        cache_position=torch.arange(length,device='cuda'), logits_to_keep=1)
                    active = [True]*len(members)
                    position = length
                    while any(active):
                        if cache.get_seq_length() != position:
                            raise RuntimeError('HF cache length is inconsistent')
                        logits = result.logits[:, -1]
                        if not bool(torch.isfinite(logits).all()):
                            raise RuntimeError('HF produced nonfinite logits')
                        prediction = logits.argmax(-1)
                        if position == length:
                            best = logits.float().topk(2, dim=-1).values
                            for j, r in enumerate(members):
                                first_margins[r] = float(best[j,0]-best[j,1])
                        for j, token in enumerate(prediction.tolist()):
                            if active[j]:
                                outputs[members[j]].append(token)
                                active[j] = token != data['eos_token_id'] and position < 64
                        if any(active):
                            result = model(input_ids=prediction[:,None], past_key_values=cache,
                                cache_position=torch.tensor([position],device='cuda'),
                                use_cache=True, logits_to_keep=1)
                            position += 1
                    progress('batch_complete', requests=members,
                             generated_lengths=[len(outputs[r]) for r in members])
                    del result, cache
        rows = []
        for row, generated in zip(data['prompts'],outputs):
            if (not generated or len(generated)>65-row['length'] or
                data['eos_token_id'] in generated[:-1] or
                (generated[-1] != data['eos_token_id'] and row['length']+len(generated)-1 != 64)):
                raise RuntimeError('HF stopping contract failed')
            rows.append(dict(request=row['request'], prompt=row['text'],
                prompt_length=row['length'], generated_length=len(generated),
                continuation=tokenizer.decode(generated,skip_special_tokens=True),
                tokens=generated, first_argmax_margin=first_margins[row['request']]))
        result = dict(passed=True, workload_sha256=data['sha256'], model=data['model'],
            revision=data['revision'], generated_tokens=outputs, texts=rows,
            total_generated_tokens=sum(map(len,outputs)), elapsed_seconds=time.monotonic()-started,
            gpu=str(torch.cuda.get_device_properties(0)), node=os.uname().nodename,
            job=os.getenv('SLURM_JOB_ID'), batch_size=args.batch_size,
            scope='HF independently prefills complete prompts and follows its own greedy tokens')
        save('reference.json',result)
        progress('finished',generated_tokens=result['total_generated_tokens'])
    except Exception:
        save('failure.json',dict(error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.batch_size<=16:
        parser.error('Use 1..16 HF reference rows per batch')
    run(args)
