"""Contract tests (CUDA: the TDT loss is CUDA-only): stage 3 through the pinned train.main with the stage 3 seam, on
the fixture plan (b0prime_fixtures: real C1 shards, C2 labels, materialized pool audio) and the tiny model.

- arm A1 trains every parameter (encoder, prediction and joint networks change; the model is in training mode at
  every step, SpecAugment included), the learning rate of every step is its planned batch's, and a scored run writes
  model.nemo with its manifest and is scored on the model restored from it;
- a trunk interrupted and resumed twice, then its branches (one interrupted and resumed), equal the uninterrupted
  trunk and branches bit for bit, with the same planned batches and learning rates.
"""
from __future__ import annotations

import contextlib
import json
import shutil
import unittest
from unittest import mock

import torch

import b0prime_fixtures as fx
import train  # finetune/parakeet-ternary
import train_b0prime as tb
import train_decoder as td
from decoder_fixtures import tiny_model
from training import seam as tseam
from training import stage3

HAS_CUDA = torch.cuda.is_available()


@contextlib.contextmanager
def nolock(label):
    yield


@unittest.skipUnless(HAS_CUDA, "the TDT loss is CUDA-only")
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = fx.World()
        cls.h = fx.Harness(cls.world, "cuda")
        train.deterministic_mode()

    @classmethod
    def tearDownClass(cls):
        cls.world.close()

    def setUp(self):
        shutil.rmtree(tb.root(), ignore_errors=True)
        self.scored = []

    def fake_score(self, net, out, source, device, label):
        self.scored.append({"label": label, "sha256": source["sha256"], "training": net.training,
                            "nemo": out / "model.nemo"})
        return {"dev_wer": {n: 0.05 for n in tb.DEV_SETS}, "dev_utterances": {n: 1 for n in tb.DEV_SETS},
                "domain_sets": {}}

    def main(self, name: str, stop_after: int | None = None) -> tuple[int, list[dict]]:
        """train.main(--run name) inside the stage 3 seam (fixture data, the tiny model); with stop_after, a SIGTERM
        is requested once that many steps are done. Returns (exit code, per-step records)."""
        seen, label, guarded = [], tseam.label_batch, tseam.guarded_train_step

        def remember(c, batch):
            out = label(c, batch)
            if out is not None:
                seen.append({"index": int(batch["index"]), "lrs": {g["lr"] for g in c.optimizer.param_groups},
                             "training": all(m.training for m in c.student.modules()),
                             "spec": c.student.spec_augmentation.training})
            return out

        def step(c, labeled, trainable):
            values = guarded(c, labeled, trainable)
            if stop_after is not None and c.state.step + 1 >= stop_after:
                c.stop_reason = "SIGTERM"
            return values

        patches = [mock.patch.dict(stage3.RUNS, fx.RUNS, clear=True),
                   mock.patch.object(tb, "offline_data", lambda run, labels_run, peak, cv: self.h.data(run, peak)),
                   mock.patch.object(tb, "peak_of", lambda run: (fx.PEAK, None)),
                   mock.patch.object(tb, "grad_checkpointing", lambda: False),
                   mock.patch.object(tb, "_sha256_base", lambda: "b0"),
                   mock.patch.object(tb, "build_model", fx.model_factory()),
                   mock.patch.object(train, "DevEvaluator", fx.FakeEvaluator),
                   mock.patch.object(tb, "score_model", self.fake_score),
                   mock.patch.object(tb, "label_manifest", lambda run: {"fixture": True}),
                   mock.patch.object(td, "_PATHS_GPU_LOCK", nolock)]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            stack.enter_context(tb.seam(None))
            stack.enter_context(mock.patch.object(train, "label_batch", remember))
            stack.enter_context(mock.patch.object(train, "train_step", step))
            try:
                train.main(["--run", name, "--labels-run", "fixture", "--cleaning-version", "0"])
            except SystemExit as stop:
                return stop.code, seen
        return 0, seen

    def latest(self, name):
        return torch.load(train.ckpt_paths(tb.run_dir(name))[0], map_location="cpu", weights_only=False)["student"]

    def test_a1_trains_every_parameter_and_scores_the_restored_nemo(self):
        code, seen = self.main("pw")
        self.assertEqual(code, 0)
        summary = json.loads((tb.run_dir("pw") / "summary.json").read_text())
        data = self.h.data(fx.RUNS["pw"])
        self.assertEqual([s["index"] for s in seen], [k for k in range(data.stop) if k not in data.plan["empty_batches"]])
        for s in seen:
            self.assertEqual(s["lrs"], {data.schedule.lr(s["index"])})
            self.assertTrue(s["training"] and s["spec"])
        self.assertEqual(summary["steps"], data.plan["steps"])
        init, final = tiny_model(7).state_dict(), self.latest("pw")
        for part in ("encoder.", "decoder.", "joint."):
            params = [k for k, _ in tiny_model(7).named_parameters() if k.startswith(part)]
            self.assertTrue(params)
            unchanged = [k for k in params if torch.equal(init[k], final[k])]
            self.assertEqual(unchanged, [], part)
        manifest = json.loads((tb.run_dir("pw") / "model.json").read_text())
        nemo = tb.run_dir("pw") / "model.nemo"
        self.assertEqual(manifest["sha256"], tb._sha256_file(nemo))
        self.assertEqual(manifest["labels"]["labels_sha256"], self.world.labels.identity["sha256"])
        self.assertIn("relabel/train_b0prime.py", manifest["training_source_hashes"])
        self.assertEqual(self.scored, [{"label": "pw", "sha256": manifest["sha256"], "training": False, "nemo": nemo}])
        restored = train.load_nemo(nemo, "cpu").state_dict()
        for k, v in final.items():
            self.assertTrue(torch.equal(restored[k].float(), v.float()), k)

    def test_resumed_trunk_and_branches_equal_uninterrupted_ones(self):
        reference = {}
        for name in ("trunk", "ba", "bb"):
            code, seen = self.main(name)
            self.assertEqual(code, 0, name)
            reference[name] = ([(s["index"], s["lrs"]) for s in seen], self.latest(name))
        self.assertTrue(json.loads((tb.run_dir("trunk") / "summary.json").read_text())["b0prime"]["final_branch_points"])
        shutil.rmtree(tb.root())
        seen = []
        for stop in (2, 5, None):
            code, part = self.main("trunk", stop)
            self.assertEqual(code, train.EXIT_INTERRUPTED if stop else 0)
            seen += part
        self.assertEqual([(s["index"], s["lrs"]) for s in seen], reference["trunk"][0])
        for k, v in self.latest("trunk").items():
            self.assertTrue(torch.equal(v, reference["trunk"][1][k]), k)
        start = tb.read_branch_point(tb.run_dir("trunk"), 1)["step"]
        seen = []
        for stop in (start + 1, None):
            code, part = self.main("ba", stop)
            self.assertEqual(code, train.EXIT_INTERRUPTED if stop else 0)
            seen += part
        self.assertEqual([(s["index"], s["lrs"]) for s in seen], reference["ba"][0])
        self.assertEqual(seen[0]["lrs"], {fx.PEAK})
        for name in ("ba", "bb"):
            if name == "bb":
                self.assertEqual(self.main("bb")[0], 0)
            for k, v in self.latest(name).items():
                self.assertTrue(torch.equal(v, reference[name][1][k]), f"{name} {k}")


if __name__ == "__main__":
    unittest.main()
