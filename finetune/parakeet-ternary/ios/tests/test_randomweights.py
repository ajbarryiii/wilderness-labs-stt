"""randomweights.py: determinism, cross-machine manifests, statistical fidelity, and a full-depth forward pass.

No NeMo. Runs on NixOS (through ../heavy, full model about 3.5 GB) and on the Mac (through
macguard); both write <artifacts>/random/manifests/seed{0,1,2}.json, whose per-tensor SHA-256s must
agree across machines (compare the files or their "digest"). Steps, in order:
1. manifests of seeds 0, 1, 2 (seed 0 twice: identical), digests pairwise distinct;
2. seed 0 written to <artifacts>/random/seed0/model.safetensors: the write manifest equals the
   in-memory one and every tensor read back hashes the same;
3. fidelity of seed 0 against weight_stats.json: per module the code fractions within 6 binomial
   standard deviations of the histogram; for every scale and float tensor the empirical CDF at the
   stored quantiles within the KS bound 2.7 / sqrt(n) (about 1e-6 false-alarm rate per tensor);
   codes in {-1, 0, 1}, scales and BatchNorm variances positive, zero rows zero;
4. the reference at full depth on the written file: 2 s of seeded synthetic audio gives finite
   features, encoder output and joint logits, and replaying its own greedy trace reproduces the
   logits and decisions exactly.

  NixOS: ../heavy ios-wp1-random --mem-max 8G --runtime 30min --wait -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
             <repo>/finetune/parakeet-ternary/python <repo>/finetune/parakeet-ternary/ios/tests/test_randomweights.py -v
  Mac:   ios/macguard --rss-cap 6G --timeout 1800 -- ios/pyenv/.venv/bin/python ios/tests/test_randomweights.py -v
"""
from __future__ import annotations

import json
import platform
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import randomweights as rw  # noqa: E402
import reference  # noqa: E402

SEEDS = (0, 1, 2)
KS_C = 2.7
SIGMAS = 6.0


def cdf_ok(values: np.ndarray, quantiles: np.ndarray) -> tuple[bool, float]:
    """Empirical CDF of values at the quantile points vs their probabilities, allowing for ties:
    P(x < q_k) - tol <= p_k <= P(x <= q_k) + tol. Returns (ok, worst excess over 0 before tol)."""
    x = np.sort(values.astype(np.float64).ravel())
    q = quantiles.astype(np.float64)
    p = rw.quantile_grid(len(q))
    below = np.searchsorted(x, q, side="left") / x.size
    upto = np.searchsorted(x, q, side="right") / x.size
    excess = np.maximum(below - p, p - upto).max()
    return bool(excess <= KS_C / np.sqrt(x.size)), float(excess)


class RandomWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stats = rw.load_stats()
        cls.out = common.artifacts_dir() / "random"
        (cls.out / "manifests").mkdir(parents=True, exist_ok=True)

    def test_1_manifests(self) -> None:
        digests = {}
        for seed in SEEDS:
            manifest = rw.manifest_only(self.stats, seed)
            manifest["machine"] = {"platform": platform.platform(), "python": platform.python_version(),
                                   "numpy": np.__version__, "torch": torch.__version__}
            (self.out / "manifests" / f"seed{seed}.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
            digests[seed] = manifest["digest"]
        again = rw.manifest_only(self.stats, 0)
        common.report("random_manifests", digests=digests, seed0_repeat_digest=again["digest"],
                      tensors=len(again["tensors"]), weight_stats_sha256=self.stats["_sha256"])
        self.assertEqual(again["digest"], digests[0])
        self.assertEqual(len(set(digests.values())), len(SEEDS))

    def test_2_written_file(self) -> None:
        from safetensors import safe_open

        written = rw.write(self.stats, 0, self.out / "seed0")
        in_memory = json.loads((self.out / "manifests" / "seed0.json").read_text())
        self.assertEqual(written["tensors"], in_memory["tensors"])
        mismatched = 0
        with safe_open(str(self.out / "seed0" / rw.MODEL_FILE), framework="numpy") as f:
            self.assertEqual(f.metadata()["format"], rw.MODEL_FORMAT)
            for name in f.keys():
                mismatched += rw.sha256_bytes(f.get_tensor(name).tobytes()) != written["tensors"][name]["sha256"]
        common.report("random_written", file_sha256=written["file_sha256"], digest=written["digest"],
                      bytes=(self.out / "seed0" / rw.MODEL_FILE).stat().st_size, read_back_mismatches=mismatched)
        self.assertEqual(mismatched, 0)

    def test_3_fidelity(self) -> None:
        worst = {"codes_sigma": 0.0, "scale_ks": 0.0, "float_ks": 0.0}
        failures = []
        for name, value in rw.generate(self.stats, 0):
            module = name.rsplit(".", 1)[0]
            if name.endswith(".codes") and module in self.stats["ternary"]:
                counts = self.stats["ternary"][module]["codes"]
                total = sum(counts.values())
                if not np.isin(value, (-1, 0, 1)).all():
                    failures.append(f"{name}: codes outside -1/0/1")
                for code, key in ((-1, "minus_one"), (0, "zero"), (1, "plus_one")):
                    p = counts[key] / total
                    sigma = abs(float((value == code).mean()) - p) / np.sqrt(p * (1 - p) / value.size)
                    worst["codes_sigma"] = max(worst["codes_sigma"], sigma)
                    if sigma > SIGMAS:
                        failures.append(f"{name}: code {code} off by {sigma:.1f} sigma")
            elif name.endswith(".scale") and module in self.stats["ternary"]:
                ok, excess = cdf_ok(value, rw.decode_f32(self.stats["ternary"][module]["scale"]["quantiles"]))
                worst["scale_ks"] = max(worst["scale_ks"], excess * np.sqrt(value.size))
                if not ok or not (value > 0).all():
                    failures.append(f"{name}: scale distribution (excess {excess:.3g})")
            elif name in self.stats["float"]:
                entry = self.stats["float"][name]
                rows = entry.get("zero_rows", [])
                kept = np.delete(value, rows, axis=0) if rows else value
                ok, excess = cdf_ok(kept, rw.decode_f32(entry["quantiles"]))  # (stats include the zero rows: 0.1%)
                worst["float_ks"] = max(worst["float_ks"], excess * np.sqrt(kept.size))
                if not ok:
                    failures.append(f"{name}: value distribution (excess {excess:.3g})")
                if rows and value[rows].any():
                    failures.append(f"{name}: zero rows are not zero")
                if name.endswith("running_var") and not (value > 0).all():
                    failures.append(f"{name}: non-positive variance")
        common.report("random_fidelity", worst_code_sigma=worst["codes_sigma"],
                      worst_scale_ks_sqrt_n=worst["scale_ks"], worst_float_ks_sqrt_n=worst["float_ks"],
                      ks_bound_sqrt_n=KS_C, failures=failures)
        self.assertEqual(failures, [])

    def test_4_forward(self) -> None:
        torch.set_grad_enabled(False)
        model = reference.build(reference.Config.from_model_config(self.stats["model_config"]))
        load = reference.load_weights(model, self.out / "seed0" / rw.MODEL_FILE)
        rng = np.random.Generator(np.random.PCG64(0))
        t = np.arange(32000) / 16000.0
        audio = (0.1 * np.sin(2 * np.pi * (200 + 300 * t) * t) + 0.01 * rng.standard_normal(32000)).astype(np.float32)
        signal, length = torch.from_numpy(audio)[None], torch.tensor([32000])
        features, feat_len = model.preprocessor(signal, length)
        encoded, enc_len = model.encoder(features, feat_len)
        trace, steps = reference.greedy_decode(model, encoded, enc_len, record=True)[0]
        replayed = reference.replay(model, encoded, int(enc_len[0]), trace)
        common.report("random_forward", load=load, encoder_frames=int(enc_len[0]), steps=len(trace),
                      tokens=len(trace.tokens), forced_advances=sum(trace.forced_advance),
                      encoder_rms=float(encoded.pow(2).mean().sqrt()), finite=common.finite(features, encoded, steps.logits),
                      replay_identical=torch.equal(replayed.logits, steps.logits), peak_rss_mb=common.peak_rss_mb())
        self.assertEqual(load["ternary_modules"], 264)
        self.assertTrue(common.finite(features, encoded, steps.logits, steps.h, steps.c))
        self.assertTrue(torch.equal(replayed.logits, steps.logits))
        self.assertEqual(replayed.argmax_token.tolist(), trace.token)
        self.assertEqual(replayed.argmax_duration.tolist(), trace.duration)


def tearDownModule() -> None:
    common.report("test_randomweights_peak", peak_rss_mb=common.peak_rss_mb())


if __name__ == "__main__":
    unittest.main()
