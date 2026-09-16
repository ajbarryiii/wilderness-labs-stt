"""Bounded CPU integration: real model, exact resume, and destructive-step guards."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from binary_stt.health import TrainingCollapse
from binary_stt.features import CausalLogMel
from binary_stt.model import BinaryCTCModel
from binary_stt.storage import atomic_torch_save, storage, write_json
from binary_stt.train import Batcher, UniqueLedger, check_finite_state, evaluate, quantization_at, train
from binary_stt.tokenizer import CharacterTokenizer


def example(index, text="ab", samples=8000):
    audio = .1 * torch.sin(torch.arange(samples) * (.019 + .001 * index))
    return {"id": str(index), "content_id": hashlib.sha256(audio.numpy().tobytes()).hexdigest(),
            "source": "fixture", "speaker": str(index), "audio": audio,
            "sample_rate": 16000, "seconds": samples / 16000, "text": text}


class RepeatDataset:
    def __init__(self, examples):
        self.examples, self.position = examples, 0

    def __iter__(self):
        return self

    def __next__(self):
        value = deepcopy(self.examples[self.position % len(self.examples)])
        self.position += 1
        return value

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]

    @property
    def stats(self):
        totals = {"accepted": self.position, "rejected": 0, "seconds": 0., "reasons": {}, "sources": {}}
        for index in range(self.position):
            value = self.examples[index % len(self.examples)]
            source = totals["sources"].setdefault(value["source"], {"accepted": 0, "rejected": 0, "seconds": 0.})
            source["accepted"] += 1
            source["seconds"] += value["seconds"]
            totals["seconds"] += value["seconds"]
        return totals


class SmallModel(nn.Module):
    """Has stochastic training so exact-resume tests also exercise RNG restore."""
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(123)
            self.projection = nn.Linear(80, 4)
        self.dropout = nn.Dropout(.2)

    def set_quantization(self, weight, activation):
        self.quantization = (weight, activation)

    def forward(self, features, lengths):
        value = features.transpose(1, 2)[:, ::8]
        return self.projection(self.dropout(value)), (lengths + 7) // 8


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        root = storage() / "tests"
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_run(self, name="run", updates=None):
        path = self.root / name
        path.mkdir()
        config = {
            "seed": 83,
            "model": {"d_model": 8, "ff_dim": 16, "num_layers": 1,
                      "num_heads": 2, "stem_channels": 2, "vocab_size": 4,
                      "chunk_size": 2, "left_context": 4, "depthwise_kernel": 3,
                      "dropout": 0., "activation_checkpointing": False},
            "tokenizer": {"kind": "character", "alphabet": " ab"},
            "training": {"device": "cpu", "max_steps": 4, "warmup_steps": 0,
                         "lr": .001, "min_lr": .00001, "grad_accumulation": 2,
                         "batch_audio_seconds": .7, "max_batch_size": 2,
                         "eval_every": 2, "checkpoint_every": 1, "log_every": 1,
                         "minimum_free_disk_gib": 0},
            "quantization": {"weight_start_step": 0, "weight_ramp_steps": 2,
                             "activation_start_step": 1, "activation_ramp_steps": 2},
            "notifications": {"desktop": False, "email_enabled": False, "email_required": False},
            "health": {"min_train_steps": 100},
            "augmentation": {"enabled": False},
            "data": {"train_sources": []},
        }
        if updates:
            for key, value in updates.items():
                config[key].update(value)
        write_json(path / "config.json", config)
        atomic_torch_save(path / "validation.pt", [example(100), example(101)])
        return path, config

    def dataset(self):
        return RepeatDataset([example(index) for index in range(5)])

    def checkpoint(self, path, filename="latest.pt"):
        return torch.load(path / filename, map_location="cpu", weights_only=False)

    def test_real_binary_model_cpu_end_to_end(self):
        path, config = self.make_run(updates={"training": {"max_steps": 2, "eval_every": 1}})
        result = train(path, dataset=self.dataset(), model=BinaryCTCModel(config["model"]))
        self.assertEqual(result["state"], "completed")
        self.assertEqual(result["step"], 2)
        self.assertEqual(result["audio_seconds"], 2.)
        self.assertEqual(result["unique_recordings"], 4)
        self.assertTrue((path / "last_known_good.pt").exists())
        records = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        self.assertTrue(any(record["kind"] == "validation" for record in records))
        self.assertTrue(any(record["kind"] == "train" for record in records))

    def test_resume_matches_uninterrupted_with_pending_audio_dropout_and_augmentation(self):
        updates = {"augmentation": {"enabled": True, "freq_masks": 1, "freq_width": 4,
                                    "time_masks": 1, "max_time_width": 2, "max_time_fraction": .1}}
        full, _ = self.make_run("full", updates)
        split, _ = self.make_run("split", updates)
        train(full, dataset=self.dataset(), model=SmallModel())
        first = train(split, stop_after=2, dataset=self.dataset(), model=SmallModel())
        self.assertEqual(first["state"], "stopped")
        middle = self.checkpoint(split)
        self.assertIsNotNone(middle["batcher"]["pending"])
        train(split, resume=True, dataset=self.dataset(), model=SmallModel())
        expected, actual = self.checkpoint(full), self.checkpoint(split)
        for name in expected["model"]:
            torch.testing.assert_close(actual["model"][name], expected["model"][name], rtol=0, atol=0)
        self.assertEqual(expected["stream"], actual["stream"])
        self.assertEqual(expected["source_health"], actual["source_health"])
        self.assertEqual(expected["unique"], actual["unique"])
        self.assertEqual(actual["unique"]["unique_recordings"], 5)
        self.assertEqual(actual["progress"]["audio_seconds"], 4.)
        self.assertEqual(actual["progress"]["optimized_sources"], expected["progress"]["optimized_sources"])
        self.assertEqual(actual["progress"]["optimized_sources"]["fixture"], {"examples": 8, "audio_seconds": 4.})
        self.assertEqual(actual["progress"]["reader_stats"]["accepted"], 9)
        self.assertEqual(actual["progress"]["optimized_examples"], 8)

    def test_corrupt_optimizer_step_preserves_last_good_and_recovery(self):
        path, _ = self.make_run()
        train(path, stop_after=2, dataset=self.dataset(), model=SmallModel())
        last_good = (path / "last_known_good.pt").read_bytes()
        recovery = (path / "latest.pt").read_bytes()
        original = torch.optim.AdamW.step
        def poison(optimizer, *args, **kwargs):
            result = original(optimizer, *args, **kwargs)
            next(iter(optimizer.state.values()))["exp_avg"].fill_(float("nan"))
            return result
        with patch.object(torch.optim.AdamW, "step", poison):
            with self.assertRaisesRegex(TrainingCollapse, "optimizer state"):
                train(path, resume=True, dataset=self.dataset(), model=SmallModel())
        self.assertEqual(last_good, (path / "last_known_good.pt").read_bytes())
        self.assertEqual(recovery, (path / "latest.pt").read_bytes())
        failure = json.loads((path / "failure.json").read_text())
        self.assertEqual(failure["state"], "collapse")
        with self.assertRaisesRegex(RuntimeError, "reviewed new run"):
            train(path, resume=True, dataset=self.dataset(), model=SmallModel())

    def test_latent_nonfinite_is_detected_even_when_sign_hides_it(self):
        model = SmallModel()
        with torch.no_grad():
            model.projection.weight[0, 0] = float("nan")
        signs = torch.where(model.projection.weight >= 0, 1., -1.)
        self.assertTrue(torch.isfinite(signs).all())
        with self.assertRaisesRegex(TrainingCollapse, "latent model"):
            check_finite_state(model)

    def test_alignment_counts_adjacent_repeats_and_bounded_rejection(self):
        # 0.08 sec yields one encoder frame; "aa" needs three frames.
        dataset = RepeatDataset([example(0, text="aa", samples=1280)])
        batcher = Batcher(dataset, CharacterTokenizer(" ab"),
                          {"max_consecutive_rejections": 2}, set())
        with self.assertRaisesRegex(TrainingCollapse, "CTC-unalignable"):
            batcher.next()
        self.assertEqual(batcher.rejected, 2)

    def test_stop_file_and_configuration_mismatch(self):
        path, config = self.make_run()
        (path / "STOP").touch()
        result = train(path, dataset=self.dataset(), model=SmallModel())
        self.assertEqual(result["step"], 0)
        self.assertEqual(result["reason"], "STOP file")
        config["training"]["lr"] *= 2
        write_json(path / "config.json", config)
        with self.assertRaisesRegex(ValueError, "Configuration/tokenizer changed"):
            train(path, resume=True, dataset=self.dataset(), model=SmallModel())

    def test_unique_ledger_rolls_back_uncheckpointed_exposure_on_resume(self):
        path, _ = self.make_run()
        train(path, stop_after=2, dataset=self.dataset(), model=SmallModel())
        ledger = UniqueLedger(path / "unique.sqlite3", 2)
        ledger.add([example(500)], 3)
        self.assertEqual(ledger.summary()["unique_recordings"], 5)
        ledger.close()
        result = train(path, resume=True, stop_after=2, dataset=self.dataset(), model=SmallModel())
        self.assertEqual(result["unique_recordings"], 4)

    def test_low_lifetime_rejection_rate_does_not_eventually_kill_a_run(self):
        batcher = Batcher(self.dataset(), CharacterTokenizer(" ab"), {}, set(),
                          {"attempted": 1000000, "rejected": 1000})
        batcher._reject("one isolated invalid example")
        self.assertEqual(batcher.rejected, 1001)

    def test_quantization_schedule_has_stable_phase_names(self):
        config = {"weight_start_step": 0, "weight_ramp_steps": 4,
                  "activation_start_step": 4, "activation_ramp_steps": 4}
        self.assertEqual(quantization_at(1, config)["phase"], quantization_at(3, config)["phase"])
        self.assertEqual(quantization_at(4, config)["weight"], 1)
        self.assertEqual(quantization_at(8, config)["activation"], 1)

    def test_validation_metrics_are_partitioned_by_source(self):
        path, _ = self.make_run()
        examples = [example(100), example(101)]
        tokenizer = CharacterTokenizer(" ab")
        for value, name in zip(examples, ["clean", "meeting"]):
            value.update(target=tokenizer.encode(value["text"]), source=name)
        class Predictions(SmallModel):
            def forward(self, features, lengths):
                steps = (features.shape[-1] + 7) // 8
                logits = torch.full((len(features), steps, 4), -4.)
                logits[:, :, 0] = 4.
                logits[0, 0, 0], logits[0, 0, 2] = -4., 4.
                logits[0, 1, 0], logits[0, 1, 3] = -4., 4.
                return logits, (lengths + 7) // 8
        result = evaluate(Predictions(), examples, tokenizer, CausalLogMel(), torch.device("cpu"), path, 0)
        self.assertEqual(result["wer"], .5)
        self.assertEqual(result["cer"], .5)
        self.assertEqual(result["sources"]["clean"]["wer"], 0.)
        self.assertEqual(result["sources"]["meeting"]["wer"], 1.)
        self.assertEqual(result["sources"]["meeting"]["empty_fraction"], 1.)
        self.assertEqual(result["sources"]["meeting"]["blank_fraction"], 1.)
        self.assertEqual(result["sources"]["meeting"]["reference_words"], 1)

    def test_minority_source_collapse_stops_even_when_aggregate_is_healthy(self):
        path, _ = self.make_run(updates={"training": {"eval_every": 1},
                                        "health": {"val_patience": 1, "phase_grace_steps": 0},
                                        "quantization": {"weight_ramp_steps": 0, "activation_ramp_steps": 0,
                                                         "activation_start_step": 0}})
        calls = 0
        def metrics(*args, **kwargs):
            nonlocal calls
            calls += 1
            clean = {"loss": 1., "wer": .1, "cer": .1, "blank_fraction": .8, "empty_fraction": 0.}
            meeting = clean if calls == 1 else {**clean, "wer": 1., "empty_fraction": 1., "blank_fraction": 1.}
            return {**clean, "sources": {"clean": clean, "meeting": meeting}}
        with patch("binary_stt.train.evaluate", side_effect=metrics):
            with self.assertRaisesRegex(TrainingCollapse, "source meeting"):
                train(path, dataset=self.dataset(), model=SmallModel())
        self.assertEqual(self.checkpoint(path, "last_known_good.pt")["progress"]["step"], 0)
        failure = json.loads((path / "failure.json").read_text())
        self.assertEqual(failure["metrics"]["source"], "meeting")

    def test_incompatible_feature_geometry_is_rejected(self):
        for index, change in enumerate(({"model": {"n_mels": 40}}, {"features": {"hop_length": 80}})):
            path, config = self.make_run(f"geometry-{index}")
            for name, value in change.items():
                config.setdefault(name, {}).update(value)
            write_json(path / "config.json", config)
            with self.assertRaisesRegex(ValueError, "80-bin features"):
                train(path, dataset=self.dataset(), model=SmallModel())


if __name__ == "__main__":
    unittest.main()
