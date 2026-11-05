"""Post-run analysis. No host observation or output processing runs in the kernel."""
import csv
import json
from pathlib import Path
import numpy as np


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not a.size: return dict(count=0)
    return dict(count=int(a.size), mean=float(a.mean()), min=float(a.min()),
        p50=float(np.percentile(a,50)), p90=float(np.percentile(a,90)),
        p95=float(np.percentile(a,95)), p99=float(np.percentile(a,99)), max=float(a.max()))


def write_csv(path, rows):
    if not rows: return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def analyze_lifecycle(directory, workload, history, lengths):
    raw=np.load(directory/'measurements.npz')
    header,admissions=raw['header'],raw['admissions']
    passes=raw['passes'].reshape(64,64,16)
    saved=raw['tokens'].reshape(64,64,4)
    start,end=map(int,header[:2])
    assert start > 0 and end > start
    requests,token_rows,pass_rows,intervals,visible_intervals=[],[],[],[],[]
    prefill_durations,decode_durations=[],[]
    eos=workload['eos_token_id']
    for request,prompt in enumerate(workload['prompts']):
        length,n=prompt['length'],int(lengths[request])
        tokens=history[request,:n]
        starts=list(range(0,length,4))+list(range(length,length+n-1))
        accepted=[length-4]+list(range(length,length+n-1))
        slot=int(admissions[request,1]); admit=int(admissions[request,0])
        output=saved[request,:n]
        times=output[:,0]
        used=np.zeros(64,dtype=bool); used[starts]=True
        checks=dict(admitted_once=int(admissions[request,2])==1,
            bounded_clock=start<=admit<=int(times[0])<=int(times[-1])<=end,
            ordered_tokens=bool((np.diff(times)>0).all()),
            token_values=np.array_equal(output[:,1],tokens),
            one_slot=0<=slot<16 and bool((output[:,2]==slot).all()),
            exact_pass_counts=bool((passes[request,used,10:12]==1).all()),
            no_extra_passes=bool((passes[request,~used]==0).all()),
            no_extra_tokens=bool((saved[request,n:]==0).all()))
        expected_input=True
        previous_done=admit
        for j,position in enumerate(starts):
            record=passes[request,position]
            rows=4 if position<length else 1
            expected=prompt['token_ids'][position:position+4] if rows==4 else [int(tokens[position-length]),-1,-1,-1]
            expected_input &= record[4]==rows and np.array_equal(record[6:10],expected)
            expected_input &= record[2]==slot and previous_done<=record[0]<record[1]<=end
            previous_done=int(record[1])
            duration=float(record[1]-record[0])/1e9
            (prefill_durations if rows==4 else decode_durations).append(duration)
            pass_rows.append(dict(request=request,slot=slot,epoch=int(record[3]),position=position,
                phase='prefill' if rows==4 else 'decode',active_rows=rows,
                start_s=float(record[0]-start)/1e9,done_s=float(record[1]-start)/1e9,duration_s=duration))
        checks['input_preparation_and_feedback']=bool(expected_input)
        checks['accepted_prediction']=all(int(passes[request,pos,5])==int(tokens[j]) and
            int(passes[request,pos,1])<=int(times[j]) and
            int(passes[request,pos,3])==int(output[j,3]) for j,pos in enumerate(accepted))
        itl=np.diff(times).astype(float)/1e9
        intervals.extend(itl.tolist())
        visible_itl=[float(itl[j-1]) for j in range(1,n) if int(tokens[j])!=eos]
        visible_intervals.extend(visible_itl)
        service=float(times[0]-admit)/1e9
        queued=float(admit-start)/1e9
        requests.append(dict(request=request,slot=slot,prompt_length=length,generated_length=n,
            consumed_rows=length+n-1,prefill_passes=length//4,decode_passes=n-1,
            graph_passes=len(starts),stopped_by_eos=int(tokens[-1])==eos,
            queue_wait_s=queued,ttft_from_admission_s=service,ttft_from_invocation_s=queued+service,
            mean_tpot_s=None if n<2 else float(itl.mean()),
            p50_itl_s=None if n<2 else float(np.median(itl)),
            final_output_from_invocation_s=float(times[-1]-start)/1e9,
            passed=all(checks.values()),checks=checks))
        for j in range(n):
            token_rows.append(dict(request=request,slot=slot,generated_index=j,token=int(tokens[j]),
                eos=int(tokens[j])==eos,time_s=float(times[j]-start)/1e9,
                from_admission_s=float(times[j]-admit)/1e9,
                interval_s=None if j==0 else float(itl[j-1]),epoch=int(output[j,3])))
    stats=dict(queue_wait_s=distribution([r['queue_wait_s'] for r in requests]),
        ttft_from_admission_s=distribution([r['ttft_from_admission_s'] for r in requests]),
        ttft_from_invocation_s=distribution([r['ttft_from_invocation_s'] for r in requests]),
        per_request_mean_tpot_s=distribution([r['mean_tpot_s'] for r in requests if r['mean_tpot_s'] is not None]),
        all_intertoken_intervals_s=distribution(intervals),
        intertoken_intervals_excluding_eos_s=distribution(visible_intervals),
        prefill_pass_s=distribution(prefill_durations),decode_pass_s=distribution(decode_durations))
    # All 64 offline requests exist at launch. Admission time excludes their
    # wait for a free slot. Neither measure includes loading, compilation,
    # tokenization, network transport, or a host streaming interface.
    report=dict(passed=all(r['passed'] for r in requests),requests=requests,statistics=stats,
        device_span_s=(end-start)/1e9,generated_tokens=int(lengths.sum()),
        completed_graph_passes=len(pass_rows),consumed_rows=sum(r['consumed_rows'] for r in requests),
        timestamp_source='GPU %globaltimer, nanoseconds; token becomes saved/visible in device output storage',
        tpot_definition='(last saved token time - first saved token time)/(generated_length - 1); EOS included')
    save(directory/'latency.json',report)
    write_csv(directory/'requests.csv',[{k:v for k,v in r.items() if k!='checks'} for r in requests])
    write_csv(directory/'tokens.csv',token_rows)
    write_csv(directory/'passes.csv',pass_rows)
    return report


def concurrency(records,start,end):
    # Integrate worker compute intervals. Persistent polling, scheduler work,
    # and queue handling are not counted as useful task execution.
    times=np.concatenate(([start,end],records[:,0],records[:,1]))
    delta=np.concatenate(([0,0],np.ones(len(records),dtype=np.int64),-np.ones(len(records),dtype=np.int64)))
    order=np.argsort(times,kind='stable'); times=times[order]; delta=delta[order]
    unique,indices=np.unique(times,return_index=True)
    changes=np.add.reduceat(delta,indices)
    level=np.cumsum(changes)[:-1]; lengths=np.diff(unique)
    assert int(level.min())>=0 and int(level.max())<=16
    seconds=np.bincount(level,weights=lengths/1e9,minlength=17)
    return dict(seconds_by_busy_workers=seconds.tolist(),
        mean_busy_workers=float(np.dot(seconds,np.arange(17))/((end-start)/1e9)),
        no_recorded_compute_s=float(seconds[0]),at_least_one_task_s=float(seconds[1:].sum()))


def analyze_diagnostic(directory,mapping,invocation, selector_policy=None):
    raw=np.load(directory/'measurements.npz')
    counts=raw['worker_counts']
    overflow=bool((counts[:,1]!=0).any())
    workers=[raw[f'worker_{w}'] for w in range(16)]
    records=np.concatenate(workers,axis=0)
    records=records[np.argsort(records[:,0],kind='stable')]
    checks=dict(no_overflow=not overflow,
        count_matches=len(records)==int(counts[:,0].sum()),
        completed=bool((records[:,1]>=records[:,0]).all()),
        enqueue_before_start=bool(((records[:,2]>0)&(records[:,2]<=records[:,0])).all()),
        handler_after_compute=bool((records[:,3]>=records[:,1]).all()),
        retirement_after_handler=bool((records[:,4]>=records[:,3]).all()))
    lifecycle=json.loads((directory/'latency.json').read_text())
    slot_passes=np.zeros(16,dtype=np.int64)
    for request in lifecycle['requests']:
        slot_passes[request['slot']]+=request['graph_passes']
    # Physical batching changes queue-item counts, but it cannot skip a native
    # ordinary node or a (layer, token, expert-branch) projection.
    ordinary=records[records[:,5]==0]
    observed=np.bincount(ordinary[:,6],minlength=len(mapping))
    expected=np.zeros(len(mapping),dtype=np.int64)
    for node in mapping:
        if node['operator'] not in ('gate_up','down'):
            expected[node['data']]=slot_passes[node['slot']]
    checks['ordinary_execution_counts_exact']=np.array_equal(observed,expected)
    branches_per_family=lifecycle['consumed_rows']*48*8
    for kind,name in ((0,'W13'),(1,'W2')):
        selected=records[(records[:,5]==1)&(records[:,6]%4==kind)]
        checks[name+'_branch_rows_exact']=int(selected[:,8].sum())==branches_per_family
    names={row['data']:row['operator'] for row in mapping}
    grouped={}
    for r in records:
        operator=('gate_up' if int(r[6])%4==0 else 'down') if r[5] else names[int(r[6])]
        grouped.setdefault(operator,[]).append(r)
    rows=[]
    for name,values in grouped.items():
        a=np.asarray(values)
        duration=(a[:,1]-a[:,0]).astype(float)/1e6
        row=dict(operator=name,tasks=len(a),worker_seconds=float(duration.sum()/1000),
            mean_ms=float(duration.mean()),p50_ms=float(np.percentile(duration,50)),
            p95_ms=float(np.percentile(duration,95)),max_ms=float(duration.max()),
            mean_enqueue_to_start_ms=float(((a[:,0]-a[:,2])/1e6).mean()),
            mean_done_to_handler_ms=float(((a[:,3]-a[:,1])/1e6).mean()),
            mean_handler_ms=float(((a[:,4]-a[:,3])/1e6).mean()),
            mean_useful_output_rows=float((np.ones(len(a)) if name in
                ('last_hidden','readout_norm','readout_logits','predicted_tokens') else a[:,8]).mean()),
            mean_slot_active_rows=None if bool(a[0,5]) else float(a[:,8].mean()),
            prefill_only_tasks=int(((a[:,9]>0)&(a[:,8]==a[:,9])).sum()),
            decode_only_tasks=int((a[:,9]==0).sum()),
            mixed_phase_tasks=int(((a[:,9]>0)&(a[:,8]!=a[:,9])).sum()))
        for phase,mask in [('prefill',(a[:,9]>0)&(a[:,8]==a[:,9])),('decode',a[:,9]==0)]:
            row[phase+'_worker_seconds']=float(duration[mask].sum()/1000)
            row[phase+'_mean_ms']=None if not mask.any() else float(duration[mask].mean())
        rows.append(row)
    rows.sort(key=lambda x:x['worker_seconds'],reverse=True)
    write_csv(directory/'operator_timings.csv',rows)
    worker_rows=[dict(worker=w,tasks=len(a),compute_s=float((a[:,1]-a[:,0]).sum()/1e9),
        compute_fraction_of_invocation=float((a[:,1]-a[:,0]).sum()/1e9/invocation)) for w,a in enumerate(workers)]
    write_csv(directory/'worker_utilization.csv',worker_rows)
    labels={1:'poll_completion',2:'handle_completion',3:'complete_logical_nested',
            4:'poll_control',5:'handle_control',6:'dispatch',7:'backoff'}
    sched=[]
    for kind,name in labels.items():
        for result in (0,1):
            n,total,maximum=map(int,raw['scheduler'][kind,result])
            sched.append(dict(scope=name,result=result,calls=n,seconds=total/1e9,
                mean_us=None if not n else total/n/1e3,max_us=maximum/1e3))
    write_csv(directory/'scheduler_timings.csv',sched)
    header=raw['header']
    sched_span=float(header[3]-header[2])/1e9
    exclusive_scopes=sum(r['seconds'] for r in sched if r['scope']!='complete_logical_nested')
    work_seconds=sum(r['seconds'] for r in sched if r['scope'] in ('handle_completion','handle_control','dispatch'))
    probe_span=float(header[1]-header[0])/1e9
    checks['timer_agrees_with_cuda_events']=abs(probe_span-invocation)<max(1.0,.01*invocation)
    from dispatch_analysis import analyze_dispatch
    split = analyze_dispatch(raw['dispatch'], sched)
    write_csv(directory/'dispatch_timings.csv', split['rows'])
    save(directory/'dispatch.json', split)
    checks['dispatch_accounting'] = split['passed']
    selector = None
    if selector_policy is not None:
        from selector_analysis import analyze_selector
        selector = analyze_selector(raw['selector_phases'], raw['selector_stats'], split, selector_policy)
        write_csv(directory/'selector_timings.csv', selector['rows'])
        save(directory/'selector.json', selector)
        checks['selector_accounting'] = selector['passed']
    membership = None
    if 'membership' in raw:
        from selector_analysis import analyze_membership
        membership = analyze_membership(raw['membership'],
            int(raw['selector_stats'][1, 5]), split['successes'])
        write_csv(directory/'membership_timings.csv', membership['rows'])
        save(directory/'membership.json', membership)
        checks['membership_accounting'] = membership['passed']
    report=dict(dispatch=split,selector=selector,membership=membership,passed=all(checks.values()),checks=checks,physical_tasks=len(records),
        summed_worker_seconds=sum(r['worker_seconds'] for r in rows),operators=rows,workers=worker_rows,
        concurrency=concurrency(records,int(header[0]),int(header[1])),scheduler=sched,
        scheduler_span_s=sched_span,scheduler_handling_and_dispatch_s=work_seconds,
        scheduler_unattributed_s=sched_span-exclusive_scopes,
        compute_done_to_handler_ms=distribution((records[:,3]-records[:,1])/1e6),
        enqueue_to_compute_start_ms=distribution((records[:,0]-records[:,2])/1e6),
        instrumentation_note='Worker intervals end after the existing CTA join; record writes are outside them. '
          'Scheduler scope totals exclude their own aggregate writes. Nested complete_logical is part of handle_completion. '
          'This separate diagnostic invocation may change batching and is not the primary TTFT/TPOT baseline.')
    save(directory/'diagnostic.json',report)
    return report
