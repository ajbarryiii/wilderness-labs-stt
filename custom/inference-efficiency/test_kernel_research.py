"""Controller safety and selection rules; no GPU work."""
import unittest

from kernel_research import confirmation, improves, key, proposal
from kernels.research import AXES, BASE, METHODS, validate_policy


def result(energy=1., latency=.16):
    return {'gpu_j_per_audio_second': energy, 'p95_seconds': latency, 'avg_watts': 250.}


class ResearchTests(unittest.TestCase):
    def test_energy_is_objective_with_latency_guard(self):
        self.assertTrue(improves(result(.98), result()))
        self.assertFalse(improves(result(1.01, .1), result()))
        self.assertFalse(improves(result(.9, .17), result()))
        self.assertFalse(improves(result(float('nan')), result()))

    def test_confirmation_requires_independent_repeats_and_separation(self):
        rows = [{'role': role, 'result': result(e)} for role, e in
                [('baseline', 1.), ('winner', .96)] * 3]
        self.assertTrue(confirmation(rows)['confirmed'])
        self.assertFalse(confirmation(rows[:-1])['confirmed'])
        rows[-1]['result'] = result(1.001)
        self.assertFalse(confirmation(rows)['confirmed'])

    def test_confirmation_rejects_latency_regression(self):
        rows = [{'role': role, 'result': result(e, t)} for role, e, t in
                [('baseline', 1., .16), ('winner', .9, .17)] * 3]
        self.assertFalse(confirmation(rows)['confirmed'])

    def test_proposal_is_adaptive_and_never_repeats(self):
        profile = {a: {m: {'median_us': 1. if m == 'base' else 2.} for m in METHODS[a]}
                   for a in AXES}
        profile['down']['split4']['median_us'] = .5
        seen = {key(BASE)}
        policy, _, _ = proposal(BASE, seen, profile)
        self.assertEqual(policy, {**BASE, 'down': 'split4'})
        seen.add(key(policy))
        profile['point']['acc2']['median_us'] = .8
        next_policy, _, _ = proposal(policy, seen, profile)
        self.assertEqual(next_policy['down'], 'split4')
        self.assertEqual(next_policy['point'], 'acc2')
        self.assertNotIn(key(next_policy), seen)
        validate_policy(next_policy)

    def test_policy_rejects_model_or_evaluator_changes(self):
        with self.assertRaises(ValueError):
            validate_policy({**BASE, 'decoder_steps': 64})
        with self.assertRaises(ValueError):
            validate_policy({**BASE, 'down': 'fp8'})


if __name__ == '__main__':
    unittest.main()
