"""Test selection/reporting without using the GPU."""
import json
from pathlib import Path
import tempfile
import unittest

from kernel_sprint import report
from paths import ROOT, save, storage


class SprintReportTests(unittest.TestCase):
    def test_confirmation_uses_only_fresh_final_pairs(self):
        storage()
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp', prefix='sprint-report-') as folder:
            job = Path(folder)
            (job / 'results').mkdir()
            save(job / 'config.json', {'deadline_utc': 'test'})
            template = dict(status='measured', distribution='ternary',
                            plugins=[{'module': 'kernels.sprint_example', 'name': 'candidate'}],
                            p95_ratio=.95)
            # A large search win cannot conceal a regression in independent confirmation.
            save(job / 'results/0.json', {**template, 'phase': 'search', 'energy_ratio': .7})
            for i, energy in enumerate((1.01, 1.02, 1.03), 1):
                save(job / f'results/{i}.json', {**template, 'phase': 'confirm', 'energy_ratio': energy})
            report(job)
            row = json.loads((job / 'summary.json').read_text())['candidates'][0]
            self.assertFalse(row['confirmed'])
            self.assertAlmostEqual(row['energy_ratio'], 1.02)
            for i, energy in enumerate((.95, .96, .97), 1):
                save(job / f'results/{i}.json', {**template, 'phase': 'confirm', 'energy_ratio': energy})
            report(job)
            row = json.loads((job / 'summary.json').read_text())['candidates'][0]
            self.assertTrue(row['confirmed'])
            self.assertEqual(row['confirmation_pairs'], 3)


if __name__ == '__main__':
    unittest.main()
