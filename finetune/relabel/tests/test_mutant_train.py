"""mutant_train contracts (mutant plan "Shared decoder", "Training protocol", "Code"): the schedule (pilot warmup then
constant, main warmup over 2% then cosine to 10% at S, a constant 10% extension) and its accounting (pilots are the
first 3,000 positions of epoch 0 and do not count toward S = 2 epochs; the main run serves epochs 0 and 1 once each;
the extension epoch 2); learning-rate selection (lowest held-out WER, ties to the smaller rate); the initial state
(B0's prediction network bitwise, one fresh joint identical in every arm, only the decoder trainable, dropout 0); one
plan hash and one seeding in every arm's configuration; resume equals uninterrupted; the held-out metric is
deterministic and is evaluate.score's corpus WER; the non-finite guard; budget stops; smoke-path isolation.

CPU: the real B0 for the initial state; a tiny Parakeet-TDT (tests/decoder_fixtures.py) with a stand-in loss (the TDT
loss is CUDA-only) that draws from Python's random module as the fused loss's omega draw does. The CUDA versions of
the resume and determinism checks are in test_mutant_gpu.py.

Run from finetune/relabel:  ./python -m unittest discover -s tests -v
"""
from __future__ import annotations

import math
import os
import random
import shutil
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import evaluate  # noqa: E402
import mutant_data as md  # noqa: E402
import mutant_train as mt  # noqa: E402
import relabel_paths  # noqa: E402
from decoder_fixtures import tiny_model  # noqa: E402
from mutant_fixtures import FakeBatches, heldout_items, stand_in_loss  # noqa: E402

TMP = relabel_paths.ARTIFACTS / "tmp" / "tests" / "mutant-train"


def tiny(seed=0):
    model = tiny_model(seed)
    mt.init_decoder(model, joint_seed=mt.JOINT_SEED)
    return model



class TestSchedule(unittest.TestCase):
    def test_pilot_warmup_then_constant(self):
        s = mt.schedule("pilot", 3e-4, n_batches=10_000)
        self.assertEqual((s.start_step, s.end_step), (0, 3000))
        self.assertAlmostEqual(s.lr_at(0), 3e-4 / 300)
        self.assertAlmostEqual(s.lr_at(299), 3e-4)
        self.assertTrue(all(s.lr_at(k) == 3e-4 for k in (300, 1500, 2999)))
        self.assertEqual(s.eval_steps, [3000])
        self.assertEqual(s.keep_steps, [])

    def test_main_warmup_cosine_to_ten_percent(self):
        n = 9_000
        s = mt.schedule("main", 1e-3, n_batches=n)
        S = 2 * n
        w = round(0.02 * S)
        self.assertEqual((s.start_step, s.end_step, s.total), (0, S, S))
        self.assertAlmostEqual(s.lr_at(0), 1e-3 / w)
        self.assertAlmostEqual(s.lr_at(w - 1), 1e-3)
        self.assertAlmostEqual(s.lr_at(w), 1e-3)
        self.assertAlmostEqual(s.factor(S), 0.10)
        mid = (w + S) // 2
        self.assertAlmostEqual(s.factor(mid), 0.10 + 0.90 * 0.5 * (1 + math.cos(math.pi * (mid - w) / (S - w))))
        lrs = [s.lr_at(k) for k in range(w, S)]
        self.assertTrue(all(a >= b for a, b in zip(lrs, lrs[1:])))
        self.assertGreater(min(lrs), 1e-4)
        self.assertEqual(s.eval_steps, [round(f * S) for f in (0.25, 0.5, 0.75, 1.0)])
        self.assertEqual(s.keep_steps, [round(0.75 * S), S])

    def test_extension_constant_ten_percent_for_one_epoch(self):
        n = 9_000
        s = mt.schedule("extension", 1e-3, n_batches=n)
        self.assertEqual((s.start_step, s.end_step), (2 * n, 3 * n))
        self.assertTrue(all(abs(s.lr_at(k) - 1e-4) < 1e-18 for k in (2 * n, 2 * n + 7, 3 * n - 1)))
        self.assertEqual(s.eval_steps, [3 * n])

    def test_pilot_accounting(self):
        train = [{"id": f"c{i}", "shard": i % 5, "row": i, "samples": 16000 * (1 + i % 20)} for i in range(400)]
        plan = md.build_plan(train, 5, 2, 40 * 16000)
        pilot = mt.schedule("pilot", 1e-4, plan.n_batches, pilot_steps=plan.n_batches // 2)
        served = [plan.position(t)[2] for t in range(pilot.start_step, pilot.end_step)]
        self.assertEqual(served, plan.epoch_order(0)[: plan.n_batches // 2].tolist())
        main = mt.schedule("main", 1e-4, plan.n_batches)
        self.assertEqual(main.total, 2 * plan.n_batches)
        served = [plan.position(t) for t in range(main.start_step, main.end_step)]
        for e in (0, 1):
            self.assertEqual(sorted(b for ep, _, b in served if ep == e), list(range(plan.n_batches)))
        ext = mt.schedule("extension", 1e-4, plan.n_batches)
        self.assertEqual({plan.position(t)[0] for t in range(ext.start_step, ext.end_step)}, {2})


class TestSelectLr(unittest.TestCase):
    def test_lowest_wins_and_ties_go_to_the_smaller_rate(self):
        res = {1e-4: {"errors": 50, "words": 1000}, 3e-4: {"errors": 40, "words": 1000},
               1e-3: {"errors": 45, "words": 1000}}
        self.assertEqual(mt.select_lr(res), 3e-4)
        res[1e-3]["errors"] = 40
        self.assertEqual(mt.select_lr(res), 3e-4)
        res[1e-4]["errors"] = 40
        self.assertEqual(mt.select_lr(res), 1e-4)

    def test_every_pilot_is_needed(self):
        with self.assertRaises(ValueError):
            mt.select_lr({1e-4: {"errors": 1, "words": 10}, 3e-4: {"errors": 1, "words": 10}})


class TestInitialStateB0(unittest.TestCase):
    """The real B0 (CPU), one model in memory at a time: each arm's initial state is built from its own load."""

    @classmethod
    def setUpClass(cls):
        import gc
        b0 = evaluate.load_pretrained("cpu")
        cls.ref_decoder = {k: v.clone() for k, v in b0.decoder.state_dict().items()}
        cls.ref_joint = {k: v.clone() for k, v in b0.joint.state_dict().items()}
        cls.fresh = mt.fresh_joint_state(b0, mt.JOINT_SEED)
        cls.fresh_other = mt.fresh_joint_state(b0, mt.JOINT_SEED + 1)
        del b0
        gc.collect()
        cls.states = {}
        for arm in ("D-B", "D-T", "D-Q"):
            m = evaluate.load_pretrained("cpu")
            info = mt.init_decoder(m)
            cls.states[arm] = ({k: v.clone() for k, v in m.decoder.state_dict().items()},
                               {k: v.clone() for k, v in m.joint.state_dict().items()}, info)
            cls.model = m
            gc.collect()

    def test_prediction_network_is_b0s_bitwise(self):
        for arm, (dec, _, info) in self.states.items():
            self.assertEqual(set(dec), set(self.ref_decoder))
            self.assertTrue(all(torch.equal(dec[k], self.ref_decoder[k]) for k in dec), arm)
            self.assertEqual(info["pred_sha256"], mt.state_sha256(self.ref_decoder))

    def test_joint_is_fresh_and_identical_in_every_arm(self):
        for arm, (_, joint, info) in self.states.items():
            self.assertEqual(set(joint), set(self.fresh))
            self.assertTrue(all(torch.equal(joint[k], self.fresh[k]) for k in joint), arm)
        first = self.states["D-B"][1]
        self.assertFalse(any(torch.equal(first[k], self.ref_joint[k]) for k in self.ref_joint
                             if self.ref_joint[k].dim() == 2))
        self.assertEqual(len({info["joint_init_sha256"] for _, _, info in self.states.values()}), 1)
        self.assertNotEqual(mt.state_sha256(self.fresh_other), self.states["D-B"][2]["joint_init_sha256"])
        self.assertEqual(self.model.joint.enc.in_features, 1024)

    def test_only_the_decoder_trains_without_dropout(self):
        m = self.model
        names = {n for n, p in m.named_parameters() if p.requires_grad}
        self.assertTrue(names)
        self.assertTrue(all(n.startswith(("decoder.", "joint.")) for n in names))
        self.assertEqual(names, {n for n, _ in m.named_parameters() if n.startswith(("decoder.", "joint."))})
        self.assertEqual([p for p in mt.trainable_parameters(m)], [p for n, p in m.named_parameters()
                                                                    if n.startswith(("decoder.", "joint."))])
        drop = [x.p for part in (m.decoder, m.joint) for x in part.modules() if isinstance(x, torch.nn.Dropout)]
        drop += [x.dropout for x in m.decoder.modules() if isinstance(x, torch.nn.RNNBase)]
        self.assertTrue(all(p == 0 for p in drop))


class TestRunConfig(unittest.TestCase):
    def test_one_plan_hash_and_one_seeding_in_every_arm(self):
        plan = mock.Mock(sha256="p" * 64, n_batches=100)
        data = mock.Mock(plan=plan, summary={"clips_sha256": "c" * 64})
        configs = [mt.run_config(arm, "main", 3e-4, data, {"features": arm}, {"pred_sha256": "x",
                                 "joint_init_sha256": "y"}, smoke=False) for arm in ("D-B", "D-T", "D-Q")]
        self.assertEqual({c["plan_sha256"] for c in configs}, {"p" * 64})
        self.assertEqual({json_key(c["seeds"]) for c in configs}, {json_key({"joint": mt.JOINT_SEED, "train":
                                                                            mt.TRAIN_SEED, "plan": md.PLAN_SEED})})
        self.assertEqual(len({mt.config_sha256(c) for c in configs}), 3)

    def test_initial_rng_state_is_the_same_for_every_arm(self):
        states = []
        for _ in range(3):
            mt.seed_training()
            states.append((random.random(), np.random.random(), float(torch.rand(1))))
        self.assertEqual(len(set(states)), 1)


def json_key(d):
    import json
    return json.dumps(d, sort_keys=True)


class TestTrainer(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(TMP, ignore_errors=True)

    def make(self, run, budget=None, stop_after=None, loss=stand_in_loss, items=None):
        model = tiny()
        sched = mt.schedule("main", 1e-3, n_batches=6)       # 12 steps, evals at 3, 6, 9, 12
        config = {"test": True, "schedule": sched.to_dict()}
        return mt.Trainer(model, FakeBatches(), sched, TMP / run, config, torch.device("cpu"), loss_fn=loss,
                          heldout=items if items is not None else heldout_items(), budget_minutes=budget,
                          stop_after=stop_after)

    def test_resume_equals_uninterrupted(self):
        full = self.make("full")
        self.assertEqual(full.run(), "complete")
        part = self.make("part", stop_after=5)
        self.assertEqual(part.run(), "stopped")
        self.assertEqual(part.step, 5)
        again = self.make("part")
        self.assertEqual(again.step, 5)
        self.assertEqual(again.run(), "complete")
        a, b = full.model.state_dict(), again.model.state_dict()
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a if k.startswith(("decoder.", "joint."))))
        self.assertEqual(full.evals, again.evals)
        self.assertEqual([e["step"] for e in full.evals], [3, 6, 9, 12])
        self.assertTrue((TMP / "full" / "ckpt" / "step-9.pt").exists())
        self.assertTrue((TMP / "full" / "summary.json").exists())

    def test_learning_rate_of_every_step_is_the_schedule(self):
        t = self.make("lr")
        seen = []
        orig = t.optimizer.step

        def spy(*a, **k):
            seen.append(t.optimizer.param_groups[0]["lr"])
            return orig(*a, **k)
        t.optimizer.step = spy
        t.run()
        self.assertEqual(seen, [t.schedule.lr_at(k) for k in range(12)])

    def test_resume_of_another_configuration_is_refused(self):
        self.make("x", stop_after=2).run()
        model = tiny()
        sched = mt.schedule("main", 3e-4, n_batches=6)
        with self.assertRaises(mt.ResumeError):
            mt.Trainer(model, FakeBatches(), sched, TMP / "x", {"test": True, "schedule": sched.to_dict()},
                       torch.device("cpu"), loss_fn=stand_in_loss, heldout=heldout_items())

    def test_budget_stop_checkpoints_and_resumes(self):
        t = self.make("budget", budget=0.0)
        self.assertEqual(t.run(), "budget")
        self.assertGreaterEqual(t.step, 1)
        again = self.make("budget")
        self.assertEqual(again.step, t.step)
        self.assertEqual(again.run(), "complete")

    def test_nonfinite_steps_are_skipped_then_stop(self):
        calls = {"n": 0}

        def bad(model, batch):
            calls["n"] += 1
            loss = stand_in_loss(model, batch)
            return loss * float("nan") if calls["n"] in (2, 4, 5) else loss
        t = self.make("nan", loss=bad)
        before = None

        def snap():
            return {k: v.clone() for k, v in t.model.joint.state_dict().items()}
        t.run_steps(1)
        before = snap()
        t.run_steps(1)                       # step 2: non-finite, skipped
        after = snap()
        self.assertTrue(all(torch.equal(before[k], after[k]) for k in before))
        self.assertEqual(t.nonfinite, 1)
        self.assertEqual(t.step, 2)
        t.run_steps(3)                       # steps 3, 4 (skipped), 5 (skipped)
        self.assertEqual(t.nonfinite, 3)
        calls["n"] = 1                       # the next call is the 2nd: non-finite, the 4th of the run
        with self.assertRaises(mt.NonfiniteSteps):
            t.run_steps(1)


class TestHeldoutMetric(unittest.TestCase):
    def test_deterministic_and_scored_by_evaluate(self):
        model = tiny()
        items = heldout_items()
        a = mt.heldout_metric(model, items, torch.device("cpu"))
        b = mt.heldout_metric(model, items, torch.device("cpu"))
        self.assertEqual(a, b)
        rows = [evaluate.score(it["text"], h) for it, h in zip(items, a["hyps"])]
        scored = [r for r in rows if r["scored"]]
        self.assertEqual(a["errors"], sum(r["S"] + r["D"] + r["I"] for r in scored))
        self.assertEqual(a["words"], sum(len(r["ref_norm"].split()) for r in scored))
        self.assertAlmostEqual(a["wer"], 100 * a["errors"] / a["words"])
        self.assertEqual(set(a["sources"]), set(md.SOURCES))

    def test_metric_leaves_rng_and_modes_unchanged(self):
        model = tiny()
        model.decoder.train()
        model.joint.train()
        random.seed(3)
        torch.manual_seed(3)
        mt.heldout_metric(model, heldout_items(), torch.device("cpu"))
        self.assertTrue(model.decoder.training and model.joint.training)
        x = (random.random(), float(torch.rand(1)))
        random.seed(3)
        torch.manual_seed(3)
        self.assertEqual(x, (random.random(), float(torch.rand(1))))


class TestSmokeIsolation(unittest.TestCase):
    def test_run_dirs(self):
        prod, smoke = md.layout(), md.layout(40)
        for stage, lr in (("pilot", 1e-4), ("main", None), ("extension", None)):
            p = mt.run_dir(prod, "D-T", stage, lr)
            s = mt.run_dir(smoke, "D-T", stage, lr)
            self.assertTrue(p.is_relative_to(md.production_root()))
            self.assertTrue(s.is_relative_to(relabel_paths.SMOKE))
            self.assertFalse(s.is_relative_to(md.production_root()))
        self.assertTrue(mt.records_dir(smoke, "D-Q", "main").is_relative_to(relabel_paths.SMOKE))
        self.assertEqual(mt.records_dir(prod, "D-Q", "main"), relabel_paths.ARTIFACTS / "encoder-probe" / "decoder"
                         / "D-Q")


if __name__ == "__main__":
    unittest.main()
