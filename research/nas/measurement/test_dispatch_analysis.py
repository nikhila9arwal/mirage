import unittest
import numpy as np
from dispatch_analysis import analyze_dispatch


class DispatchAccounting(unittest.TestCase):
    def fixture(self):
        stats = np.zeros((3, 4, 3), dtype=np.int64)
        stats[0, 0] = [2, 200, 150]  # Two attempts with an exhausted pool.
        stats[1, 0] = [3, 300, 100]
        stats[1, 1] = [3, 3000, 1500]  # Three attempts without an eligible bin.
        stats[2] = [[4, 400, 100], [4, 40000, 10000],
                    [4, 8000, 2000], [4, 2000, 500]]
        sched = [dict(scope='dispatch', result=0, calls=5, seconds=4e-6),
                 dict(scope='dispatch', result=1, calls=4, seconds=60e-6)]
        return stats, sched

    def test_complete_accounting_and_units(self):
        stats, sched = self.fixture()
        result = analyze_dispatch(stats, sched)
        self.assertTrue(result['passed'])
        self.assertEqual(result['attempts'], 9)
        self.assertEqual(result['successes'], 4)
        self.assertEqual(result['failures'], 5)
        self.assertAlmostEqual(result['success_fraction'], 4/9)
        self.assertEqual(result['outcomes'], dict(no_descriptor=2,no_eligible=3,published=4))
        selection = next(r for r in result['rows'] if r['outcome']=='published' and r['phase']=='selection')
        self.assertEqual(selection['mean_us'], 10)
        self.assertAlmostEqual(result['phase_seconds'], 53.9e-6)

    def test_missing_successful_phase_fails(self):
        stats, sched = self.fixture()
        stats[2, 3] = 0
        self.assertFalse(analyze_dispatch(stats, sched)['passed'])

    def test_bad_early_return_accounting_fails(self):
        stats, sched = self.fixture()
        stats[0, 2] = [1, 100, 100]
        self.assertFalse(analyze_dispatch(stats, sched)['passed'])

    def test_phases_cannot_exceed_enclosing_scope(self):
        stats, sched = self.fixture()
        stats[2, 1, 1] = 1000000
        self.assertFalse(analyze_dispatch(stats, sched)['passed'])

    def test_zero_work_is_valid(self):
        result = analyze_dispatch(np.zeros((3,4,3),dtype=np.int64),[])
        self.assertTrue(result['passed'])
        self.assertEqual(result['attempts'], 0)
        self.assertEqual(result['successes'], 0)
        self.assertEqual(result['failures'], 0)
        self.assertIsNone(result['success_fraction'])


if __name__ == '__main__':
    unittest.main()
