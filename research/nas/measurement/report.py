"""Write a self-contained report for completed selector variants only."""
import json
from pathlib import Path


def write_report(output):
    reports = [json.loads(p.read_text()) for p in sorted(output.glob('*/report.json'))]
    text = ['# FIFO winner-loop comparison', '',
        'Real 48-layer Qwen3-30B-A3B, identical 64 offline prompts; 16 slots, 16 worker CTAs, '
        'one controller; P=4, top-k=8, H=2048, I=768, 128 experts. MoE threshold 8, timeout '
        '100 microseconds, batch capacity 16, descriptor pool 16. Dense batching and both '
        'companion policies are off. ContextIO hidden-output saving remains on.', '',
        'All FIFO variants use the same counts, member queues, oldest-eligible rule, '
        'and (op_id, expert) equal-age tie rule. fifo is the preserved baseline. '
        'fifo_cached stores the head timestamp per bin. fifo_cached_unchecked also '
        'omits duplicate per-bin present/enabled checks, reusing the validation in ready(). '
        'No heap, extra scheduler, kernel, or compiler change is included.', '',
        'Each variant is one full invocation. GPU timing excludes checkpoint loading, compilation, '
        'and host analysis. Compare within the same instrumentation and completion budget. '
        'Diagnostics can alter batch formation; uninstrumented controls are reported separately.', '',
        '| Variant | Completion budget | Invocation s | Generated tokens | Tokens/s | Speedup vs matching FIFO baseline | Checks |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in reports:
        old = next((x for x in reports if x['selector']=='fifo' and
            x['instrumentation']==r['instrumentation'] and x['completion_budget']==r['completion_budget']),None)
        speed = f"{old['invocation_seconds']/r['invocation_seconds']:.3f}x" if old else 'not run'
        text.append(f"| {r['variant']} | {r['completion_budget']} | {r['invocation_seconds']:.6f} | "
            f"{r['generated_tokens']} | {r['generated_tokens']/r['invocation_seconds']:.3f} | {speed} | {r['passed']} |")
    for r in reports:
        if 'diagnostic' not in r: continue
        name=r['variant']; d=r['diagnostic']; split=d['dispatch']; selector=d['selector']
        text += ['', f'## {name}', '',
            f"Dispatch attempts: **{split['attempts']:,}**; published: **{split['successes']:,}**; "
            f"unsuccessful: **{split['failures']:,}**. No descriptor: "
            f"{split['outcomes']['no_descriptor']:,}; no eligible bin: {split['outcomes']['no_eligible']:,}.", '',
            '| Outcome | Selector phase | Calls | Summed s | Mean microseconds | Max microseconds |',
            '|---|---|---:|---:|---:|---:|']
        for phase in selector['rows']:
            if phase['calls']:
                text.append(f"| {phase['outcome']} | {phase['phase']} | {phase['calls']:,} | "
                    f"{phase['seconds']:.6f} | {phase['mean_us']:.3f} | {phase['max_us']:.3f} |")
        wins=selector['outcomes'][1]
        text += ['', f"Winning bins above 16 rows: **{wins['oversized_winners']:,}/{wins['calls']:,}** "
            f"({100*(wins['oversized_fraction'] or 0):.4f}%). "
            f"Maximum winning count: {wins['max_winning_rows']}; mean selected rows: {wins['mean_selected_rows']:.3f}.", '',
            f"Selector phases total **{selector['measured_seconds']:.6f} s**; enclosing selection "
            f"**{selector['outer_seconds']:.6f} s**; outer dispatch **{split['outer_seconds']:.6f} s**. "
            'These are nested scopes, not times to add together. FIFO reset is unused/zero.', '',
            '| Outcome | Mean distinct bins inspected | Maximum | Winner ns/inspected bin |', '|---|---:|---:|---:|']
        for outcome in selector['outcomes']:
            if outcome['calls']:
                winner = next((phase for phase in selector['rows']
                    if phase['outcome']==outcome['outcome'] and phase['phase']=='choose_winner'), None)
                normalized = (f"{winner['seconds']*1e9/outcome['groups_inspected']:.3f}"
                    if winner and outcome['groups_inspected'] else 'n/a')
                text.append(f"| {outcome['outcome']} | {outcome['mean_groups']:.3f} | {outcome['max_groups']} | {normalized} |")
        if d.get('membership'):
            text += ['', '| Membership maintenance | Calls | Members | Summed s | Mean us |',
                     '|---|---:|---:|---:|---:|']
            for phase in d['membership']['rows']:
                text.append(f"| {phase['phase']} | {phase['calls']:,} | {phase['members']:,} | "
                    f"{phase['seconds']:.6f} | {(phase['mean_us'] or 0):.3f} |")
            total = selector['outer_seconds'] + sum(p['seconds'] for p in d['membership']['rows'])
            text += ['', f'Selection plus these disjoint maintenance scopes: **{total:.6f} s**. '
                'This includes work moved out of selection; it excludes their probe aggregate writes. '
                'Maintenance is already included in the enclosing scheduler scopes below.']
        text += ['', '| Scheduler scope | Calls | Summed s | Mean us |', '|---|---:|---:|---:|']
        for scope in dict.fromkeys(row['scope'] for row in d['scheduler']):
            entries=[row for row in d['scheduler'] if row['scope']==scope]
            calls=sum(row['calls'] for row in entries); seconds=sum(row['seconds'] for row in entries)
            text.append(f'| {scope} | {calls:,} | {seconds:.6f} | {seconds*1e6/calls if calls else 0:.3f} |')
        text += ['', 'Nested logical completion is inside handle_completion; do not add it twice.', '',
            '| Operation | Physical tasks | Useful rows/task | Worker-seconds |', '|---|---:|---:|---:|']
        for op in d['operators']:
            if op['operator'] in ('gate_up','down','qkv','projected'):
                text.append(f"| {op['operator']} | {op['tasks']:,} | {op['mean_useful_output_rows']:.3f} | {op['worker_seconds']:.3f} |")
        text += ['', f"Mean busy workers: {d['concurrency']['mean_busy_workers']:.3f}/16. "
            f"Mean GPU completion to handler entry: {d['compute_done_to_handler_ms']['mean']:.3f} ms.", '',
            '| Request latency | Mean s | Median s | P95 s |','|---|---:|---:|---:|']
        for key,label in [('ttft_from_admission_s','TTFT after slot admission'),
                          ('ttft_from_invocation_s','TTFT including waiting for a slot'),
                          ('per_request_mean_tpot_s','Per-request mean TPOT')]:
            stat=r['lifecycle']['statistics'][key]
            text.append(f"| {label} | {stat['mean']:.3f} | {stat['p50']:.3f} | {stat['p95']:.3f} |")
        text += ['', f"Selector accounting passed: **{selector['passed']}**. "
            f"[Selector CSV]({name}/selector_timings.csv), [counts]({name}/selector.json), "
            f"[dispatch CSV]({name}/dispatch_timings.csv), [raw measurements]({name}/measurements.npz), "
            f"[validation]({name}/report.json)."]
    text += ['', 'Only completed variants appear above. See status.json, failure.json (if present), '
        'and the Slurm log for incomplete runs. Faster selection can itself change arrival timing '
        'and batching. One run does not measure variance or establish each optimization’s isolated '
        'contribution. Compare diagnostic_fifo to diagnostic_fifo_cached for age caching, '
        'then diagnostic_fifo_cached to diagnostic_fifo_cached_unchecked for the additional '
        'effect of removing duplicate validity checks. Their changed execution timing can '
        'alter bin populations; use both per-call and per-inspected-bin costs.']
    (output/'REPORT.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    import sys
    write_report(Path(sys.argv[1]))
