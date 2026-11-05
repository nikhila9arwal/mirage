#!/usr/bin/env python3
"""Real 48-layer Qwen: NAS full-chunk prefill, autoregression, and replacement.

Only token IDs enter from the host. The existing persistent runtime owns every
prompt/decode transition. Small completion captures audit its input/output path.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

from workload import validate, _checkpoint, signature
from validation import check_run, compare_hf
from autoregressive import load_weights
from reference import digest
import demo

SESSION = Path(__file__).resolve().parents[2]
ROOT = SESSION.parents[1]  # Repository root; research/nas holds the harness.


def make_kernel(mi, torch, slots, p, eos):
    """Reuse chunk harness metadata. No shared batch preparation runs between passes."""
    def zeros(shape, dtype=torch.int32):
        return torch.zeros(shape,dtype=dtype,device='cuda')
    meta=dict(step=zeros(slots),tokens=zeros((slots,64),torch.int64),
        input_tokens=zeros((slots,1),torch.int64),output_tokens=zeros((slots,1),torch.int64),
        num_new_tokens=torch.ones(slots,dtype=torch.int32,device='cuda'),
        prompt_lengths=torch.ones(slots,dtype=torch.int32,device='cuda'),
        qo_indptr_buffer=zeros(slots+1),paged_kv_indptr_buffer_0=zeros(slots+1),
        paged_kv_indices_buffer_0=zeros(slots),paged_kv_last_page_len_buffer_0=zeros(slots),
        paged_kv_indices_snapshot_0=zeros(slots))
    return mi.PersistentKernel(mode='offline',world_size=1,mpi_rank=0,num_workers=16,
        num_local_schedulers=1,num_remote_schedulers=0,max_seq_length=64,
        max_num_batched_requests=slots,max_num_batched_tokens=slots*p,max_num_pages=slots,
        page_size=64,meta_tensors=meta,profiler_tensor=None,trace_name='',spec_decode_config=None,
        use_cutlass_kernel=True,task_graph_mode='resident_data',eos_token_id=eos)


def cache_checks(torch, buffers, lifecycle, slots, poison):
    """A reused slot retains the last request's cache; older suffixes are allowed."""
    checks=[]
    for slot in range(slots):
        members=[row for row in lifecycle['requests'] if row.get('slot')==slot]
        if not members:
            continue
        last=max(members,key=lambda row:row['request'])
        high_water=max(row['consumed_rows'] for row in members)
        for layer in range(48):
            for kind in ('k','v'):
                value=buffers[f'layer{layer}_{kind}_cache'][slot]
                checks.append(dict(slot=slot,request=last['request'],layer=layer,kind=kind,
                    valid_finite=bool(torch.isfinite(value[:last['consumed_rows']]).all()),
                    all_valid_positions_written=bool((value[:last['consumed_rows']] != poison).any(dim=-1).all()),
                    never_used_suffix_poisoned=bool((value[high_water:] == poison).all())))
    return checks


def run(args):
    import torch
    data=validate(json.loads(args.workload.read_text()))
    p, count=data['chunk_size'],len(data['prompts'])
    out=args.output.resolve()
    if SESSION not in out.parents:
        raise ValueError('Keep outputs in this isolated experiment directory')
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    status=dict(generated=False,compiled=False,executed=False,validated=False)
    def save(name,value):
        (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    def progress(phase,**details):
        status.update(phase=phase,elapsed_seconds=time.monotonic()-started,**details)
        save('status.json',status)
        print(json.dumps(status),flush=True)
    pk=None
    try:
        for name in ('MIRAGE_RESIDENT_SLOT_ITERATIONS','NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS'):
            if os.environ.get(name):
                raise ValueError(f'Unset {name} for this controlled experiment')
        os.environ.update(MIRAGE_HOME=str(ROOT),MIRAGE_ROOT=str(ROOT),MIRAGE_TASK_GRAPH_MODE='resident_data',
            MIRAGE_RESIDENT_EXECUTION_MODE='scheduler_dispatch',MIRAGE_DENSE_BATCHING='none',
            MIRAGE_MOE_BATCH_ROWS=str(args.batch_rows),MIRAGE_MOE_TIMEOUT_US=str(args.timeout_us),
            MIRAGE_MOE_EARLY_COALESCE='0',MIRAGE_MOE_DEADLINE_COALESCE='0',
            MIRAGE_COMPARE_SEQUENCE_REPEATS='1',MPK_MAX_TOKENS_PER_REQUEST=str(args.slots*p))
        sys.path[:0]=[str(ROOT/'python'),str(ROOT/'demo/context_io')]
        import mirage as mi
        import mirage.core
        from transformers import AutoTokenizer,Qwen3MoeConfig
        torch.cuda.set_device(0)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        if torch.cuda.get_device_capability(0)!=(9,0):
            raise RuntimeError('Use the tested Hopper path')
        if not all(Path(path).resolve().is_relative_to(ROOT) for path in (mi.__file__,mirage.core.__file__)):
            raise RuntimeError('Imported Mirage is outside the isolated checkout')
        manifest,snapshot,_=_checkpoint(data['checkpoint'])
        if any(data[k]!=manifest[k] for k in ('model','revision')):
            raise ValueError('Checkpoint provenance mismatch')
        config=Qwen3MoeConfig.from_json_file(str(snapshot/'config.json'))
        config._attn_implementation='eager'
        if (config.hidden_size,config.moe_intermediate_size,config.num_experts,
            config.num_experts_per_tok,config.num_hidden_layers,config.head_dim,
            config.num_key_value_heads,config.num_attention_heads)!=(2048,768,128,8,48,128,4,32):
            raise ValueError('Unexpected model dimensions')
        hf=json.loads(args.hf_reference.read_text()) if args.hf_reference else None
        if hf and (not hf['passed'] or hf['workload_sha256']!=data['sha256']):
            raise ValueError('HF reference is not for these exact prompts')
        save('environment.json',dict(python=sys.executable,torch=torch.__version__,mirage=mi.__file__,
            extension=mirage.core.__file__,gpu=str(torch.cuda.get_device_properties(0)),
            node=os.uname().nodename,job=os.getenv('SLURM_JOB_ID'),slots=args.slots,workers=16,
            schedulers=1,layers=48,chunk_size=p,requests=count,mode='resident_data/scheduler_dispatch',
            batch_rows=args.batch_rows,timeout_us=args.timeout_us,dense_batching=False,
            model=manifest['model'],revision=manifest['revision']))
        save('workload.json',data)
        progress('load_checkpoint')
        owners=json.loads((snapshot/'model.safetensors.index.json').read_text())['weight_map']
        weights=load_weights(torch,config,snapshot,owners,progress)
        tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
        contexts=count*64
        inputs=torch.full((contexts,1),-1,dtype=torch.int64,device='cuda')
        for row in data['prompts']:
            inputs[row['request']*64:row['request']*64+row['length'],0]=torch.tensor(
                row['token_ids'],dtype=torch.int64,device='cuda')
        prompt_lengths=torch.tensor([r['length'] for r in data['prompts']],dtype=torch.int32,device='cuda')
        lengths=torch.empty(count,dtype=torch.int32,device='cuda')
        history=torch.empty((contexts,1),dtype=torch.int64,device='cuda')
        saved_hidden=torch.empty((contexts,demo.H),dtype=torch.bfloat16,device='cuda')
        pk=make_kernel(mi,torch,args.slots,p,data['eos_token_id'])
        progress('generate_native_graph')
        buffers=demo.build_graph(pk,torch,weights,args.slots,48,readout=True,chunk_rows=p)
        last_hidden=buffers['last_hidden'] if p>1 else buffers['layer47_final']
        graph=json.loads(pk.generate_task_graph()['json_file'])
        pk.set_request_slots(demo.assign_slots(graph,buffers,args.slots))
        graph=json.loads(pk.generate_task_graph()['json_file'])
        mapping,graph_report=demo.validate_graph(graph,args.slots,48,readout=True,chunk_rows=p)
        for name,value in [('native_graph.json',graph),('mapping.json',mapping),('graph_validation.json',graph_report)]:
            save(name,value)
        # All captures are small and use the existing completion hook. Position
        # request*64+start identifies a pass; accepted outputs have separate indices.
        source=dict(slot_ids=buffers['slot_ids'],predicted_tokens=buffers['predicted_tokens'],last_hidden=last_hidden)
        captures={name:torch.empty((contexts,*value.shape[1:]),dtype=value.dtype,device='cuda')
                  for name,value in source.items()}
        copies=torch.tensor([[value.data_ptr(),captures[name].data_ptr(),
                              value.numel()*value.element_size()//(2*args.slots)]
                             for name,value in source.items()],dtype=torch.int64,device='cuda')
        visits=torch.zeros(contexts,dtype=torch.int32,device='cuda')
        slot_owners=torch.empty_like(visits)
        capture_io=torch.tensor([copies.data_ptr(),len(copies),visits.data_ptr(),slot_owners.data_ptr()],
                                dtype=torch.int64,device='cuda')
        ordinary=[0]*args.slots
        moe=[0]*args.slots
        for node in mapping:
            (moe if node['operator'] in ('gate_up','down') else ordinary)[node['slot']]+=1
        # 64N bounds consumed TOKEN rows. Count R members per row, not P*R
        # per row: counting P twice was fixed in the preserved sweep baseline.
        work_bound=contexts*max(a+demo.TOP_K*b for a,b in zip(ordinary,moe))
        args.layers=48
        args.work_queue_capacity=demo.next_power_of_two(work_bound+2)
        args.control_queue_capacity=demo.next_power_of_two(2*contexts+3)
        save('queue_capacity_proof.json',dict(consumed_row_bound=contexts,ordinary_by_slot=ordinary,
            moe_by_slot=moe,compute_bound=work_bound,worker_capacity=args.work_queue_capacity,
            completion_capacity=args.work_queue_capacity,scheduler_capacity=args.control_queue_capacity))
        readonly=dict(inputs=inputs,prompt_lengths=prompt_lengths,embedding=weights['embedding'],
                      **weights['readout'])
        for layer,packed in enumerate(weights['layers']):
            readonly.update({f'layer{layer}_{name}':value for name,value in packed.items()})
        progress('hash_weights_and_inputs',generated=True)
        before={name:digest(value) for name,value in readonly.items()}
        save('readonly_before.json',before)
        progress('compile_native_graph')
        with demo.compile_options(args,out,capture_io,None):
            pk.compile(context_io=(inputs,saved_hidden,buffers['slot_ids'],last_hidden),
                context_chunk_size=p,context_prefill=(prompt_lengths,lengths),
                context_autoregressive=(buffers['predicted_tokens'],history),output_dir=str(out))
        progress('ready',compiled=True)
        reports=[]
        previous=None
        for trial in range(args.trials):
            directory=out/f'trial_{trial}'
            directory.mkdir()
            for value in list(buffers.values())+list(captures.values()):
                value.fill_(float('nan') if value.dtype.is_floating_point else -1)
            # Match the existing prefill tests: poison unused K/V with finite
            # values. Prefix length and causal masking must exclude this suffix.
            cache_poison=-7.0-trial
            for name,value in buffers.items():
                if name.endswith('_cache'):
                    value.fill_(cache_poison)
            history.fill_(-1)
            saved_hidden.fill_(float('nan'))
            lengths.fill_(-99)
            visits.zero_()
            slot_owners.fill_(-1)
            torch.cuda.synchronize()
            progress('execute',trial=trial)
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            pk()
            end.record()
            torch.cuda.synchronize()
            progress('validate',executed=True,trial=trial)
            got={name:value.cpu() for name,value in captures.items()}
            generated=history.reshape(count,64).cpu()
            actual_lengths=lengths.cpu()
            saved=saved_hidden.cpu()
            lifecycle=check_run(data,generated,actual_lengths,got['slot_ids'],got['predicted_tokens'],
                visits.cpu(),slot_owners.cpu(),got['last_hidden'],saved,args.slots)
            checks=cache_checks(torch,buffers,lifecycle,args.slots,cache_poison) if lifecycle['passed'] else []
            repeated=previous is None or (torch.equal(generated,previous[0]) and torch.equal(actual_lengths,previous[1]))
            previous=(generated.clone(),actual_lengths.clone())
            texts=[]
            for prompt,length in zip(data['prompts'],actual_lengths.tolist()):
                r=prompt['request']
                tokens=generated[r,:max(0,min(length,64))].tolist()
                texts.append(dict(request=r,prompt=prompt['text'],prompt_length=prompt['length'],
                    generated_length=length,tokens=tokens,
                    continuation=tokenizer.decode(tokens,skip_special_tokens=True)))
            report=dict(passed=lifecycle['passed'] and lifecycle['replacement_observed'] and repeated and
                bool(checks) and all(row['valid_finite'] and row['all_valid_positions_written'] and
                                    row['never_used_suffix_poisoned'] for row in checks),
                lifecycle=lifecycle,cache_checks=checks,repeated_tokens_and_lengths_exact=repeated,
                hf_agreement=compare_hf(generated,actual_lengths,hf) if hf and lifecycle['passed'] else None,
                texts=texts,diagnostic_invocation_ms=begin.elapsed_time(end))
            torch.save(dict(captures=got,history=generated,generated_lengths=actual_lengths,
                visits=visits.cpu(),owners=slot_owners.cpu(),saved_hidden=saved),directory/'outputs.pt')
            (directory/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
            reports.append(report)
            for row in texts:
                print(f'Request {row["request"]}: {row["prompt"]}\n  NAS: {row["continuation"]}',flush=True)
            if not report['passed']:
                raise RuntimeError(f'Trial {trial} failed runtime validation; preserved report contains details')
        progress('verify_readonly_weights_and_inputs')
        after={name:digest(value) for name,value in readonly.items()}
        save('readonly_after.json',after)
        result=dict(passed=before==after and all(row['passed'] for row in reports),
            readonly_unchanged=before==after,requests=count,slots=args.slots,layers=48,chunk_size=p,
            trials=reports,graph=graph_report,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            scope='Real checkpoint; NAS consumes all prompt tokens, builds KV, decodes, and replaces slots',
            interpretation='passed covers execution/lifecycle; HF token agreement is separately reported')
        save('report.json',result)
        progress('finished',validated=result['passed'])
        if not result['passed']:
            raise RuntimeError('Readonly validation failed')
        # The standard batch script keeps both stages together. Export its
        # human-readable report automatically, after all GPU timing/checks.
        if args.hf_reference and args.hf_reference.resolve()==out.parent/'hf/reference.json':
            sys.path.insert(0,str(SESSION/'verification'))
            from summarize_real_prefill import summarize
            summary=summarize(out.parent)
            (out.parent/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            (SESSION/'VERIFICATION.md').write_text(
                '# Real Qwen native-prefill verification\n\n'
                f'GPU job **{os.getenv("SLURM_JOB_ID")}** completed the HF and NAS stages. '
                f'NAS runtime/lifecycle checks passed for {count} requests, 48 layers, '
                f'P={p}, and {args.trials} invocations.\n\n'
                f'Exact HF sequences: {summary["hf_exact_sequences"]}/{count}; '
                'this agreement is separate from lifecycle correctness.\n\n'
                f'[Full results and all continuations]({out.parent.relative_to(SESSION)}/RESULTS.md).\n')
            print(json.dumps(summary),flush=True)
    except Exception:
        save('failure.json',dict(error=traceback.format_exc(),status=status))
        raise
    finally:
        if pk is not None:
            pk.finalize()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload',type=Path,required=True)
    parser.add_argument('--hf-reference',type=Path)
    parser.add_argument('--slots',type=int,choices=(16,32),default=16)
    parser.add_argument('--trials',type=int,choices=(1,2,3),default=2)
    parser.add_argument('--batch-rows',type=int,default=8)
    parser.add_argument('--timeout-us',type=int,default=100)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.batch_rows<=16 or not 0<=args.timeout_us<=1000000:
        parser.error('Batch threshold 1..16, timeout 0..1000000 us')
    run(args)
