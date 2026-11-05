"""Analyze bounded selector counters, separate from enclosing dispatch time."""
import numpy as np

PHASES = {
    'legacy': ('clear_seen', 'count_and_choose', 'collect_members', 'reset_unused'),
    'histogram': ('accumulate_bins', 'choose_winner', 'collect_members', 'reset_counts'),
    'fifo': ('prepare_selection', 'choose_winner', 'collect_fifo_prefix', 'reset_unused'),
}


def analyze_membership(raw, selected_rows, published):
    raw = np.asarray(raw)
    assert raw.shape == (2, 4)
    rows = []
    for i, name in enumerate(('ready_membership', 'dispatch_membership')):
        calls, total, maximum, members = map(int, raw[i])
        rows.append(dict(phase=name, calls=calls, seconds=total/1e9,
            mean_us=total/calls/1e3 if calls else None, max_us=maximum/1e3,
            members=members))
    checks = dict(nonnegative=bool((raw >= 0).all()),
        admitted_equals_selected=int(raw[0, 3]) == selected_rows,
        removed_equals_selected=int(raw[1, 3]) == selected_rows,
        removal_calls_equal_publications=int(raw[1, 0]) == published,
        max_consistent=all(r['max_us'] <= r['seconds']*1e6+1e-6 for r in rows))
    return dict(passed=all(checks.values()), checks=checks, rows=rows,
        note='ready_membership includes pending metadata writes and FIFO appends; '
             'dispatch_membership includes FIFO commit and pending-bit clears. '
             'These are nested in completion/control and dispatch respectively.')


def analyze_selector(raw, stats, dispatch, policy):
    raw, stats = np.asarray(raw), np.asarray(stats)
    assert raw.shape == (2, 4, 3), raw.shape
    assert stats.shape == (2, 8), stats.shape
    assert policy in PHASES
    rows, summaries = [], []
    for outcome, label in enumerate(('no_eligible', 'published')):
        calls, groups, max_groups, winning_rows, max_winning, members, overflow, threshold = map(int, stats[outcome])
        summaries.append(dict(outcome=label, calls=calls, groups_inspected=groups,
            mean_groups=groups/calls if calls else None, max_groups=max_groups,
            winning_rows=winning_rows, mean_winning_rows=winning_rows/calls if calls else None,
            max_winning_rows=max_winning, selected_rows=members,
            mean_selected_rows=members/calls if calls else None,
            oversized_winners=overflow, oversized_fraction=overflow/calls if calls else None,
            threshold_winners=threshold, timeout_winners=(calls-threshold) if outcome else 0))
        for phase, name in enumerate(PHASES[policy]):
            count, total, maximum = map(int, raw[outcome, phase])
            rows.append(dict(outcome=label, phase=name, calls=count, seconds=total/1e9,
                mean_us=total/count/1e3 if count else None, max_us=maximum/1e3))
    outer = [r for r in dispatch['rows'] if r['phase']=='selection']
    outer_seconds = sum(r['seconds'] for r in outer)
    measured = sum(r['seconds'] for r in rows)
    checks = dict(
        nonnegative=bool((raw>=0).all() and (stats>=0).all()),
        all_calls_have_four_phases=all(bool((raw[i,:,0]==stats[i,0]).all()) for i in range(2)),
        classified_calls_match_dispatch=all(int(stats[i,0])==dispatch['outcomes'][label]
            for i,label in enumerate(('no_eligible','published'))),
        ineligible_has_no_members=bool((stats[0,3:]==0).all()),
        greedy_capacity_respected=int(stats[1,0])<=int(stats[1,5])<=16*int(stats[1,0]),
        all_selected_members_belong_to_winning_bin=int(stats[1,5])<=int(stats[1,3]),
        overflow_bounded=0<=int(stats[1,6])<=int(stats[1,0]),
        threshold_bounded=0<=int(stats[1,7])<=int(stats[1,0]),
        max_consistent=all((r['calls']>0 or (r['seconds']==0 and r['max_us']==0))
            and r['max_us']<=r['seconds']*1e6+1e-6 for r in rows),
        phases_within_selection=measured<=outer_seconds+1e-6,
    )
    return dict(policy=policy, passed=all(checks.values()), checks=checks,
        rows=rows, outcomes=summaries, measured_seconds=measured,
        outer_seconds=outer_seconds, outer_minus_phases_seconds=outer_seconds-measured,
        note='GPU-global timer; counts/sum/max per phase. Only calls with an available descriptor '
             'enter selection. FIFO counts persist; histogram calls reset scratch. Aggregate writes occur '
             'after selection and publication. Overflow means winning group_count > 16, '
             'not pending rows elsewhere in the graph.')
