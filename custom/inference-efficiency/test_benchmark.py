"""Decision gates must not mistake incomplete or runtime-only gains for evidence."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import benchmark
from paths import ROOT, save, storage


class DecisionTests(unittest.TestCase):
    def run_report(self, *, packed=.4, dense=1.0, seconds=60, missing=False):
        storage()
        test_root = ROOT / 'test-artifacts'
        test_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            path = Path(folder)
            save(path / 'config.json', dict(variants=list(benchmark.VARIANTS), repeats=5, seconds=60))
            for repeat in range(5):
                for variant in benchmark.VARIANTS:
                    if missing and variant == 'ct2-int8':
                        continue
                    energy = packed if variant == 'ternary-packed' else dense if variant == 'ternary-dense' else 1.0
                    save(path / f'{repeat:02d}-{variant}.json', dict(
                        variant=variant, repeat=repeat, gpu_j_per_audio_second=energy,
                        p95_seconds=.5, avg_watts=100, clip_latency_seconds=[.5]*16,
                        completed_clips=16, measurement=dict(elapsed_seconds=seconds)))
            with contextlib.redirect_stdout(io.StringIO()):
                benchmark.report(path)
            return json.loads((path / 'summary.json').read_text())

    def test_large_external_and_same_runtime_savings_pass(self):
        result = self.run_report()
        self.assertTrue(result['qualified'])
        self.assertEqual(result['decision'], 'go')

    def test_runtime_only_gain_does_not_justify_ternary(self):
        result = self.run_report(dense=.3)
        self.assertEqual(result['decision'], 'runtime_gain_only')

    def test_missing_int8_cannot_qualify(self):
        result = self.run_report(missing=True)
        self.assertFalse(result['qualified'])
        self.assertEqual(result['decision'], 'incomplete')

    def test_short_actual_windows_cannot_qualify(self):
        result = self.run_report(seconds=1)
        self.assertFalse(result['qualified'])
        self.assertEqual(result['decision'], 'incomplete')


if __name__ == '__main__':
    unittest.main()
