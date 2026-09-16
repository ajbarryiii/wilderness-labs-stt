"""Failure-oriented benchmark provenance, output and qualification checks."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import benchmark
from storage import storage


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary_root = storage() / 'tmp'
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root, prefix='benchmark-tests-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.gpu_root = self.root / 'gpu'
        (self.gpu_root / 'data').mkdir(parents=True)
        (self.gpu_root / 'models').mkdir()
        self.gpu_patch = mock.patch.object(benchmark, 'GPU_ROOT', self.gpu_root)
        self.gpu_patch.start()
        self.addCleanup(self.gpu_patch.stop)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def test_frozen_source_manifest_and_model_lock_mutations_fail(self):
        source = self.root / 'runtime.py'
        source.write_text('original')
        manifest = self.write(self.gpu_root / 'data/manifest.json', {'test': 'manifest'})
        lock = self.write(self.gpu_root / 'models/model-lock.json', {'test': 'lock'})
        config = {'kind': 'run', 'source_hashes': {str(source): benchmark.digest(source)},
                  'manifest_sha256': benchmark.digest(manifest),
                  'model_lock_sha256': benchmark.digest(lock)}
        benchmark.verify_config(config)
        for path, message in ((source, 'Source changed'), (manifest, 'manifest changed'),
                              (lock, 'Model lock changed')):
            with self.subTest(path=path):
                original = path.read_text()
                path.write_text(original + '\n')
                with self.assertRaisesRegex(RuntimeError, message):
                    benchmark.verify_config(config)
                path.write_text(original)

    def test_frozen_pcm_and_mel_checks_cannot_be_bypassed(self):
        tokens = list(range(128))
        pcm, mel = self.root / 'gpu/data/pcm.npy', self.root / 'gpu/data/mel.npy'
        pcm.write_bytes(b'pcm fixture')
        mel.write_bytes(b'mel fixture')
        clip = {'seconds': 30, 'replay_tokens': tokens, 'pcm': str(pcm), 'mel': str(mel),
                'pcm_sha256': benchmark.digest(pcm), 'mel_sha256': benchmark.digest(mel)}
        manifest = {'clips': [{**clip, 'id': f'clip-{i}'} for i in range(16)],
                    'decoder_steps': 128, 'replay_tokens': tokens}
        path = self.write(self.gpu_root / 'data/manifest.json', manifest)
        self.assertEqual(len(benchmark.workload()['clips']), 16)
        mel.write_bytes(b'altered mel fixture')
        with self.assertRaisesRegex(ValueError, 'frozen workload'):
            benchmark.workload()
        mel.write_bytes(b'mel fixture')
        manifest['clips'][0]['replay_tokens'] = tokens[:-1]
        self.write(path, manifest)
        with self.assertRaisesRegex(ValueError, 'duration or tokens differ'):
            benchmark.workload()

    def test_frontend_is_verified_even_without_loading_ct2_weights(self):
        frontend = self.write(self.gpu_root / 'models/whisper-medium.en/preprocessor_config.json', {'n_mels': 80})
        lock = {'openai/whisper-medium.en': {
            'path': str(frontend.parent), 'files': {
                frontend.name: {'bytes': frontend.stat().st_size, 'sha256': benchmark.digest(frontend)}}}}
        self.write(self.gpu_root / 'models/model-lock.json', lock)
        benchmark.verify_models(include_ct2=False)
        with self.assertRaisesRegex(ValueError, 'required artifact'):
            benchmark.verify_models()
        frontend.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            benchmark.verify_models(include_ct2=False)

    def test_output_digest_rejects_unverified_or_degenerate_outputs(self):
        for output in (torch.ones(1, 2), torch.tensor([[1., float('nan')]]), torch.empty(0)):
            with self.subTest(output=output), self.assertRaisesRegex(RuntimeError, 'degenerate'):
                benchmark.output_digest(SimpleNamespace(), output)
        with self.assertRaisesRegex(RuntimeError, 'not verified'):
            benchmark.output_digest(SimpleNamespace(replay_verified=False), [])
        self.assertEqual(benchmark.output_digest(SimpleNamespace(replay_verified=True), []),
                         {'forced_prefix_verified': True})
        model = SimpleNamespace(last_predictions=torch.tensor([1, 2]))
        first = benchmark.output_digest(model, torch.tensor([[0., 1.]]))
        second = benchmark.output_digest(model, torch.tensor([[0., 2.]]))
        self.assertNotEqual(first['logits'], second['logits'])
        self.assertEqual(first['predictions'], second['predictions'])

    def test_placements_are_bounded_by_physical_core_presets(self):
        topology = {'affinity_presets': {'large_l3': [0, 2, 4, 6], 'small_l3': [1, 3]}}
        actual = benchmark.placements(topology, ['large_l3:3', 'small_l3:1'])
        self.assertEqual(actual, [{'placement': 'large_l3', 'threads': 3, 'cpus': [0, 2, 4]},
                                  {'placement': 'small_l3', 'threads': 1, 'cpus': [1]}])
        for request in ('large_l3:0', 'large_l3:5', 'large_l3:-1', 'large_l3:abc', 'absent:1'):
            with self.subTest(request=request), self.assertRaises((ValueError, KeyError)):
                benchmark.placements(topology, [request])

    def make_report(self, variants=('w1a1-dense',), windows=3):
        job = self.root / 'job'
        job.mkdir()
        schedule = []
        for repeat in range(windows):
            for variant in variants:
                schedule.append({'id': f'window-{len(schedule):03d}', 'variant': variant,
                                 'placement': 'large_l3', 'threads': 4, 'cpus': [0, 1, 2, 3],
                                 'repeat': repeat})
        config = {'kind': 'run', 'seconds': 60, 'clips': 16, 'repeats': windows,
                  'topology': {'model_name': 'test CPU'}, 'schedule': schedule}
        self.write(job / 'config.json', config)
        self.write(job / 'workload.json', {'clips': [{'id': f'clip-{i}'} for i in range(16)]})
        return job, schedule

    def row(self, spec, **overrides):
        hashes = ({'forced_prefix_verified': True} if spec['variant'].startswith('ct2-')
                  else {'logits': 'a' * 64, 'predictions': 'b' * 64})
        return {**spec, 'kind': 'run', 'clip_latency_seconds': [4.] * 16,
                'completed_clips': 16, 'joules_per_clip': 100., 'average_watts': 25.,
                'measurement': {'elapsed_seconds': 64., 'iterations': 1, 'energy_joules': 1600.},
                'outputs': {f'clip-{i}': hashes.copy() for i in range(16)},
                'peak_rss_bytes': 1024, 'estimated_other_busy_cores': 0.,
                'qualified_energy': True, **overrides}

    def summary(self, job):
        benchmark.report(job)
        return json.loads((job / 'summary.json').read_text())

    def test_partial_group_cannot_qualify_then_complete_group_can(self):
        job, schedule = self.make_report()
        self.write(job / f"{schedule[0]['id']}.json", self.row(schedule[0]))
        value = self.summary(job)['w1a1-dense@large_l3:4']
        self.assertFalse(value['qualified_energy'])
        self.assertFalse(value['complete_group'])
        self.assertEqual(value['planned_windows'], 3)
        for spec in schedule[1:]:
            self.write(job / f"{spec['id']}.json", self.row(spec))
        self.assertTrue(self.summary(job)['w1a1-dense@large_l3:4']['qualified_energy'])

    def test_missing_dense_reference_cannot_qualify_native_group(self):
        job, schedule = self.make_report(('w1a1-avx512',))
        for spec in schedule:
            self.write(job / f"{spec['id']}.json", self.row(spec))
        value = self.summary(job)['w1a1-avx512@large_l3:4']
        self.assertIsNone(value['exact_dense_match'])
        self.assertFalse(value['qualified_energy'])

    def test_optimized_backend_requires_matching_original_avx512_reference(self):
        job, schedule = self.make_report(('w1a1-avx512', 'w1a1-avx512_opt'))
        config = json.loads((job / 'config.json').read_text())
        config['reference_backend'] = 'avx512'
        self.write(job / 'config.json', config)
        for spec in schedule:
            if spec['variant'].endswith('_opt'):
                self.write(job / f"{spec['id']}.json", self.row(spec))
        value = self.summary(job)['w1a1-avx512_opt@large_l3:4']
        self.assertFalse(value['qualified_energy'])
        self.assertIsNone(value['exact_reference_match'])
        for spec in schedule:
            self.write(job / f"{spec['id']}.json", self.row(spec))
        value = self.summary(job)['w1a1-avx512_opt@large_l3:4']
        self.assertTrue(value['qualified_energy'])
        self.assertTrue(value['exact_reference_match'])
        self.assertIsNone(value['exact_dense_match'])
        changed = self.row(schedule[-1])
        changed['outputs']['clip-0']['logits'] = 'c' * 64
        self.write(job / f"{schedule[-1]['id']}.json", changed)
        with self.assertRaisesRegex(RuntimeError, 'Full-model output mismatch'):
            self.summary(job)

    def test_missing_energy_or_short_actual_interval_cannot_qualify(self):
        job, schedule = self.make_report(('ct2-int8',))
        for spec in schedule:
            self.write(job / f"{spec['id']}.json", self.row(spec))
        self.assertTrue(self.summary(job)['ct2-int8@large_l3:4']['qualified_energy'])
        first = schedule[0]
        for overrides in ({'measurement': {'elapsed_seconds': 59., 'iterations': 1, 'energy_joules': 100.}},
                          {'joules_per_clip': None, 'average_watts': None,
                           'measurement': {'elapsed_seconds': 64., 'iterations': 1, 'energy_joules': None}}):
            with self.subTest(overrides=overrides):
                self.write(job / f"{first['id']}.json", self.row(first, **overrides))
                self.assertFalse(self.summary(job)['ct2-int8@large_l3:4']['qualified_energy'])

    def test_schedule_mismatch_and_duplicate_window_evidence_fail(self):
        job, schedule = self.make_report()
        first = schedule[0]
        for overrides in ({'cpus': [16, 17, 18, 19]}, {'threads': 8}, {'variant': 'w2a2-dense'},
                          {'id': 'window-999'}):
            with self.subTest(overrides=overrides):
                self.write(job / f"{first['id']}.json", self.row(first, **overrides))
                with self.assertRaisesRegex(RuntimeError, 'schedule|unscheduled'):
                    self.summary(job)
        self.write(job / f"{first['id']}.json", self.row(first))
        self.write(job / 'window-001.json', self.row(first))
        with self.assertRaisesRegex(RuntimeError, 'Duplicate|unscheduled'):
            self.summary(job)

    def test_missing_outputs_incomplete_cycles_and_false_replay_fail(self):
        job, schedule = self.make_report(('ct2-int8',))
        first = schedule[0]
        valid = self.row(first)
        malformed = [dict(valid, outputs={}), dict(valid, completed_clips=15),
                     dict(valid, outputs={key: {'forced_prefix_verified': False}
                                          for key in valid['outputs']})]
        for row in malformed:
            with self.subTest(row=row):
                self.write(job / f"{first['id']}.json", row)
                with self.assertRaisesRegex(RuntimeError, 'output evidence|cycle|forced replay'):
                    self.summary(job)

    def test_same_model_output_mismatch_is_never_averaged_away(self):
        job, schedule = self.make_report(('w1a1-dense', 'w1a1-avx512'))
        for spec in schedule:
            row = self.row(spec)
            if spec['variant'].endswith('avx512'):
                row['outputs']['clip-0']['logits'] = 'c' * 64
            self.write(job / f"{spec['id']}.json", row)
        with self.assertRaisesRegex(RuntimeError, 'Full-model output mismatch'):
            self.summary(job)

    def test_selected_scalar_reference_qualifies_only_matching_complete_groups(self):
        job, schedule = self.make_report(('w1a1-scalar', 'w1a1-avx512'))
        config = json.loads((job / 'config.json').read_text())
        config['reference_backend'] = 'scalar'
        self.write(job / 'config.json', config)
        for spec in schedule:
            if spec['variant'].endswith('avx512'):
                self.write(job / f"{spec['id']}.json", self.row(spec))
        value = self.summary(job)['w1a1-avx512@large_l3:4']
        self.assertEqual(value['reference_backend'], 'scalar')
        self.assertIsNone(value['exact_reference_match'])
        self.assertIsNone(value['exact_dense_match'])
        self.assertFalse(value['qualified_energy'])
        for spec in schedule:
            if spec['variant'].endswith('scalar'):
                self.write(job / f"{spec['id']}.json", self.row(spec))
        for value in self.summary(job).values():
            self.assertTrue(value['exact_reference_match'])
            self.assertIsNone(value['exact_dense_match'])
            self.assertTrue(value['qualified_energy'])
        report = (job / 'REPORT.md').read_text()
        self.assertIn('Exact scalar match', report)
        self.assertNotIn('Exact dense match', report)
        changed = next(spec for spec in schedule if spec['variant'].endswith('avx512'))
        row = self.row(changed)
        row['outputs']['clip-0']['logits'] = 'c' * 64
        self.write(job / f"{changed['id']}.json", row)
        with self.assertRaisesRegex(RuntimeError, 'Full-model output mismatch'):
            self.summary(job)
        self.write(job / f"{changed['id']}.json", self.row(changed))
        # Old configs default to dense and must not borrow a scalar reference.
        del config['reference_backend']
        self.write(job / 'config.json', config)
        for value in self.summary(job).values():
            self.assertEqual(value['reference_backend'], 'dense')
            self.assertIsNone(value['exact_reference_match'])
            self.assertFalse(value['qualified_energy'])

    def test_model_worker_retains_metadata_before_close_and_checks_outside_measurement(self):
        class Meter:
            closed = False
            measuring = False

            def metadata(self):
                if self.closed:
                    raise RuntimeError('CpuEnergyMeter is closed')
                return {'available': False, 'test_meter': True}

            def measure(self, call, **kwargs):
                self.measuring = True
                call()
                self.measuring = False
                return {'elapsed_seconds': 1., 'energy_joules': None,
                        'average_watts': None, 'iterations': 1}

            def close(self):
                self.closed = True

        job, schedule = self.make_report(('ct2-int8',))
        config = {'kind': 'screen', 'clips': 1, 'warmup': 1, 'seconds': 0,
                  'idle_seconds': 0, 'cycles': 1}
        model = SimpleNamespace(replay_verified=True, metadata=lambda: {'engine': 'fake-ct2'})
        meter = Meter()
        checked_while_measuring = []
        run_while_measuring = []
        real_digest = benchmark.output_digest

        def checked(model, output):
            checked_while_measuring.append(meter.measuring)
            return real_digest(model, output)

        def call(index):
            run_while_measuring.append(meter.measuring)
            return []

        with mock.patch.object(benchmark, 'workload', return_value={'clips': [{'id': 'clip-0'}]}), \
                mock.patch.object(benchmark, 'verify_models') as verify, \
                mock.patch.object(benchmark, 'make_runner', return_value=(model, call)), \
                mock.patch('hardware.CpuEnergyMeter.detect', return_value=meter), \
                mock.patch.object(benchmark, 'output_digest', side_effect=checked), \
                mock.patch.object(benchmark, 'telemetry', return_value={}), \
                mock.patch.object(benchmark, 'proc_cpu', return_value=0), \
                mock.patch('builtins.print'):
            benchmark.model_worker(job, config, schedule[0])
        self.assertTrue(meter.closed)
        self.assertTrue(checked_while_measuring)
        self.assertFalse(any(checked_while_measuring))
        self.assertIn(True, run_while_measuring)
        verify.assert_called_once_with(include_ct2=True)
        row = json.loads((job / f"{schedule[0]['id']}.json").read_text())
        self.assertEqual(row['energy_meter'], {'available': False, 'test_meter': True})
        self.assertEqual(row['completed_clips'], 1)
        self.assertIsNone(row['joules_per_clip'])


if __name__ == '__main__':
    unittest.main()
