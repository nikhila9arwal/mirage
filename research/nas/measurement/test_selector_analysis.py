import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from selector_analysis import analyze_selector, analyze_membership
from report import write_report


class SelectorAccounting(unittest.TestCase):
    def fixture(self):
        phases=np.zeros((2,4,3),dtype=np.int64)
        phases[0]=[[3,3000,1000],[3,300,100],[3,0,0],[3,300,100]]
        phases[1]=[[2,2000,1000],[2,200,100],[2,800,400],[2,200,100]]
        stats=np.array([[3,12,5,0,0,0,0,0], [2,20,11,22,18,20,1,1]],dtype=np.int64)
        dispatch=dict(outcomes=dict(no_descriptor=4,no_eligible=3,published=2),
            rows=[dict(phase='selection',seconds=4000/1e9),dict(phase='selection',seconds=4000/1e9)])
        return phases,stats,dispatch

    def test_units_outcomes_and_overflow(self):
        p,s,d=self.fixture();r=analyze_selector(p,s,d,'histogram')
        self.assertTrue(r['passed'],r['checks'])
        self.assertEqual(r['rows'][0]['mean_us'],1)
        self.assertEqual(r['outcomes'][1]['oversized_fraction'],.5)
        self.assertEqual(r['outcomes'][1]['mean_selected_rows'],10)
        self.assertEqual(r['outcomes'][1]['threshold_winners'],1)
        self.assertEqual(r['outcomes'][1]['timeout_winners'],1)

    def test_missing_call_fails(self):
        p,s,d=self.fixture();p[1,3,0]-=1
        self.assertFalse(analyze_selector(p,s,d,'histogram')['passed'])

    def test_bad_overflow_and_nested_time_fail(self):
        p,s,d=self.fixture();s[1,6]=3
        self.assertFalse(analyze_selector(p,s,d,'histogram')['passed'])
        p,s,d=self.fixture();p[1,0,1]=10**9
        self.assertFalse(analyze_selector(p,s,d,'histogram')['passed'])

    def test_empty_work_is_valid(self):
        d=dict(outcomes=dict(no_descriptor=12,no_eligible=0,published=0),rows=[])
        self.assertTrue(analyze_selector(np.zeros((2,4,3),dtype=np.int64),
            np.zeros((2,8),dtype=np.int64),d,'histogram')['passed'])

    def test_fifo_phases_and_maintenance(self):
        p,s,d=self.fixture()
        result=analyze_selector(p,s,d,'fifo')
        self.assertTrue(result['passed'])
        self.assertEqual(result['rows'][2]['phase'],'collect_fifo_prefix')
        raw=np.array([[4,8000,3000,20],[2,1000,600,20]],dtype=np.int64)
        self.assertTrue(analyze_membership(raw,20,2)['passed'])
        raw[0,3]-=1
        self.assertFalse(analyze_membership(raw,20,2)['passed'])
        raw[0,3]+=1; raw[1,0]+=1
        self.assertFalse(analyze_membership(raw,20,2)['passed'])

    def test_fifo_winner_variants_use_the_same_phase_boundaries(self):
        p,s,d=self.fixture()
        baseline=analyze_selector(p,s,d,'fifo')
        for name in ('fifo_cached','fifo_cached_unchecked'):
            result=analyze_selector(p,s,d,name)
            self.assertTrue(result['passed'])
            self.assertEqual(result['rows'],baseline['rows'])

    def test_report_uses_existing_latency_field_names(self):
        p,s,d=self.fixture();selector=analyze_selector(p,s,d,'histogram')
        report=dict(variant='diagnostic_histogram',selector='histogram',instrumentation='diagnostic',
            completion_budget=1,invocation_seconds=1,generated_tokens=663,passed=True,
            lifecycle=dict(statistics={name:dict(mean=1,p50=1,p95=1) for name in
                ('ttft_from_admission_s','ttft_from_invocation_s','per_request_mean_tpot_s')}),
            diagnostic=dict(selector=selector,dispatch=dict(**d,attempts=9,successes=2,
                failures=7,outer_seconds=.1),operators=[],concurrency=dict(mean_busy_workers=1),
                compute_done_to_handler_ms=dict(mean=.1),scheduler=[]))
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);(output/'diagnostic_histogram').mkdir()
            (output/'diagnostic_histogram/report.json').write_text(json.dumps(report))
            write_report(output)
            text=(output/'REPORT.md').read_text()
            self.assertIn('1/2',text);self.assertIn('accumulate_bins',text)


if __name__=='__main__': unittest.main()
