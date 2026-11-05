"""Check fixed-size controller aggregates after GPU synchronization."""
import numpy as np

OUTCOMES = ('no_descriptor', 'no_eligible', 'published')
PHASES = ('lookup', 'selection', 'descriptor', 'publication')


def analyze_dispatch(raw, scheduler):
    raw = np.asarray(raw)
    assert raw.shape == (3, 4, 3), raw.shape
    rows = []
    for outcome, name in enumerate(OUTCOMES):
        for phase, label in enumerate(PHASES):
            count, total, maximum = map(int, raw[outcome, phase])
            rows.append(dict(outcome=name, phase=label, calls=count,
                seconds=total / 1e9, mean_us=total / count / 1e3 if count else None,
                max_us=maximum / 1e3))
    outer = [r for r in scheduler if r['scope'] == 'dispatch']
    attempts = sum(r['calls'] for r in outer)
    successes = sum(r['calls'] for r in outer if r['result'] == 1)
    outer_seconds = sum(r['seconds'] for r in outer)
    phase_seconds = sum(r['seconds'] for r in rows)
    lookup_counts = [int(raw[o, 0, 0]) for o in range(3)]
    checks = dict(
        nonnegative=bool((raw >= 0).all()),
        every_attempt_classified=sum(lookup_counts) == attempts,
        publications_match_outer_success=lookup_counts[2] == successes,
        pool_full_skips_later_phases=bool((raw[0, 1:] == 0).all()),
        ineligible_stops_after_selection=int(raw[1, 1, 0]) == lookup_counts[1]
            and bool((raw[1, 2:] == 0).all()),
        all_successful_phases_recorded=bool((raw[2, :, 0] == successes).all()),
        durations_consistent=all((r['calls'] > 0 or (r['seconds'] == 0 and r['max_us'] == 0))
            and r['max_us'] <= r['seconds'] * 1e6 + 1e-6 for r in rows),
        phases_inside_outer_scope=phase_seconds <= outer_seconds + 1e-6,
    )
    return dict(passed=all(checks.values()), checks=checks, rows=rows,
        attempts=attempts, successes=successes, failures=attempts-successes,
        success_fraction=successes/attempts if attempts else None,
        outcomes=dict(zip(OUTCOMES, lookup_counts)), outer_seconds=outer_seconds,
        phase_seconds=phase_seconds, outer_minus_phases_seconds=outer_seconds-phase_seconds,
        note='Four contiguous phases exclude their final aggregate writes. The outer dispatch scope '
             'includes those writes, timer overhead, and function entry/exit. No phase waits for GEMM completion.')
