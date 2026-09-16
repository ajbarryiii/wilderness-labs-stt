"""Deterministic, target-preserving acoustic augmentation, keyed by presentation."""

import collections
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from common import digest, encode
from prepare import feature
from recovery_core import read
from recovery_train import Features


def active_rms(audio):
    """20 ms energy gate excludes silence when specifying speaker level ratios."""
    count = len(audio) // 320
    if not count:
        return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    energies = np.mean(audio[:count * 320].reshape(count, 320).astype(np.float64) ** 2, axis=1)
    active = energies >= max(float(energies.max()) * 0.01, 1e-12)
    return float(np.sqrt(energies[active].mean())) if active.any() else 0.0


class Mixer:
    def __init__(self, rows, noise):
        self.rows, self.noise = rows, noise
        self.waveforms = collections.OrderedDict()
        self.noise_arrays = {}
        self.speech = {split: [r for r in rows if r["domain"] == "general" and r["split"] == split]
                       for split in ("train", "calibration")}
        assert all(self.speech.values())

    def audio(self, row, trim_digit=True):
        key = row["id"]
        if key not in self.waveforms:
            assert digest(row["audio"]) == row["audio_sha256"]
            x, sr = sf.read(row["audio"], dtype="float32")
            assert sr == 16000 and x.ndim == 1 and np.isfinite(x).all()
            self.waveforms[key] = x
            if len(self.waveforms) > 4000:
                self.waveforms.popitem(last=False)
        x = self.waveforms[key]
        if trim_digit and row["domain"] == "digits":
            assert not np.any(x[:3200])
            return x[3200:]
        return x

    def background(self, item):
        if item["id"] not in self.noise_arrays:
            assert digest(item["audio"]) == item["sha256"]
            x, sr = sf.read(item["audio"], dtype="float32")
            assert sr == 16000 and x.ndim == 1 and np.isfinite(x).all()
            self.noise_arrays[item["id"]] = x
        return self.noise_arrays[item["id"]]

    def mix(self, target, row, rng, kind, db, validation=False):
        if kind == "noise":
            pool = [n for n in self.noise if n["split"] == ("validation" if validation else "train")]
            source = pool[int(rng.integers(len(pool)))]
            wave = self.background(source)
            assert len(wave) >= len(target)
            start = int(rng.integers(len(wave) - len(target) + 1))
            background = wave[start:start + len(target)].copy()
            brms = float(np.sqrt(np.mean(background.astype(np.float64) ** 2)))
            placement = dict(source_offset_samples=start)
        else:
            assert kind == "speech"
            pool = self.speech["calibration" if validation else "train"]
            # Exclude the target recording AND known target speaker, including during validation.
            candidates = [r for r in pool if r["id"] != row["id"] and r.get("speaker") != row.get("speaker")]
            source = candidates[int(rng.integers(len(candidates)))]
            wave = self.audio(source)
            # One continuous interferer, random placement; no artificial tiled word repetitions.
            if len(wave) > len(target):
                offset = int(rng.integers(len(wave) - len(target) + 1))
                background = wave[offset:offset + len(target)].copy()
                placement = dict(source_offset_samples=offset, target_offset_samples=0)
            else:
                offset = int(rng.integers(len(target) - len(wave) + 1))
                background = np.zeros_like(target)
                background[offset:offset + len(wave)] = wave
                placement = dict(source_offset_samples=0, target_offset_samples=offset)
            brms = active_rms(background)
        trms = active_rms(target)
        if trms < 1e-8 or brms < 1e-8:
            raise ValueError(f"Silent mixing source: {row['id']}, {source['id']}")
        scale = trms / brms * 10 ** (-db / 20)
        mixture = target + background * scale
        peak = float(np.max(np.abs(mixture)))
        limiter = min(1.0, 0.98 / max(peak, 1e-8))
        mixture = (mixture * limiter).astype(np.float32)
        assert np.isfinite(mixture).all() and np.max(np.abs(mixture)) <= 0.980001
        return mixture, dict(kind=kind, source_id=source["id"], source_split=source["split"],
            db=float(db), measured_db=float(20 * np.log10(trms / (brms * scale))),
            target_active_rms=trms, background_rms=brms, background_scale=scale,
            common_peak_scale=limiter, **placement)


class AugmentedFeatures(Features):
    def __init__(self, rows, run, spec):
        super().__init__(rows)
        self.spec = spec
        self.policy = spec["acoustic_augmentation"]
        self.kind = self.policy["kind"]
        sources = read(Path(run) / "augmentation-sources.json")
        assert digest(sources["noise_manifest"]) == sources["noise_manifest_sha256"]
        assert digest(sources["validation_manifest"]) == sources["validation_manifest_sha256"]
        self.mixer = Mixer(read(Path(run) / "manifest.json")["rows"], read(sources["noise_manifest"])["rows"])
        # Pin decoded source waveforms in process memory before allocating the GPU model.
        # HDD seeks otherwise stall the GPU on each new random source, especially while
        # checkpoints are flushing. The pilot's complete training audio is only ~1.5 GiB.
        if self.kind in {"noise", "speech"}:
            for row in self.mixer.rows:
                if row["split"] == "train":
                    self.mixer.audio(row, trim_digit=False)
            for item in self.mixer.noise:
                if item["split"] == "train":
                    self.mixer.background(item)
        self.last_details = {}
        self.validation = {}
        fixture = read(sources["validation_manifest"])
        for name in fixture["selection_conditions"]:
            items = fixture["conditions"][name]
            feats = {}
            for row in items:
                assert digest(row["features"]) == row["feature_sha256"]
                feats[row["id"]] = torch.from_numpy(np.load(row["features"], allow_pickle=False))
            self.validation[name] = (items, feats)

    def augmented(self, row, prefix, gain, presentation=None):
        assert row["split"] == "train"
        assert presentation is not None
        rng = np.random.default_rng(np.random.SeedSequence([self.spec["seed"], 617, presentation]))
        probability = self.policy.get("probability", 0.5) * min(1.0, presentation / self.policy.get("ramp_examples", 12000))
        enabled = self.kind != "baseline" and rng.random() < probability
        detail = dict(kind=self.kind, enabled=bool(enabled), presentation=presentation)
        if not enabled or self.kind == "specaugment":
            x = super().augmented(row, prefix, gain)
            if enabled:
                x = x.clone()
                width = int(rng.integers(1, 7))
                start = int(rng.integers(81 - width))
                fill = x[:, prefix:].mean()
                x[start:start + width, prefix:] = fill
                duration = x.shape[-1] - prefix
                cap = min(3 if row["domain"] == "digits" else 10, max(1, int(duration * 0.02)))
                tw = int(rng.integers(1, cap + 1))
                ts = int(rng.integers(prefix, x.shape[-1] - tw + 1))
                x[:, ts:ts + tw] = fill
                detail.update(frequency_width=width, time_width=tw, time_start=ts)
        else:
            target = np.pad(self.mixer.audio(row), (prefix * 160, 0))
            db = float(rng.uniform(*self.policy["db_range"]))
            mixed, info = self.mixer.mix(target, row, rng, self.kind, db)
            x = torch.from_numpy(feature(mixed * 10 ** (gain / 20)))
            detail.update(info)
        y = encode(row["text"])
        assert (x.shape[-1] + 1) // 2 >= len(y) + sum(a == b for a, b in zip(y, y[1:]))
        assert torch.isfinite(x).all()
        self.last_details[row["id"]] = detail
        return x
