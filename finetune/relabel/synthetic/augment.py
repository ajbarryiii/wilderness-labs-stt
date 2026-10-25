"""The on-the-fly augmentation of synthetic clips (synthetic plan "Augmentation"; the TS arms only).

Augmenter(pool, seed)(item, index) -> a new item with augmented audio and the SAME label:
- the random generator is seeded from (seed, consumed-batch index, clip id), so a run is reproducible and resumes
  identically, and a clip repeated in another batch gets a fresh draw;
- with probability P_APPLY the clip is augmented, otherwise returned unchanged (a copy);
- an augmented clip gets 1, 2 or 3 operations (uniform), distinct, drawn uniformly from OPS and applied in the
  order of OPS; parameters are drawn from the same generator:
    speed     speed perturbation, factor U(0.9, 1.1) (resampling: duration and pitch change together)
    pitch     pitch shift U(-2, +2) semitones, duration kept (librosa pitch_shift)
    reverb    convolution with a room impulse response drawn uniformly from the TRAIN RIR pool (onset-aligned,
              tail cut, output scaled to the input's RMS)
    noise     an additive noise segment drawn uniformly from the TRAIN noise pool (random offset, looped if
              short) at an SNR of U(5, 30) dB over the whole clip
    channel   with probability 1/2 telephone band-limiting (300-3400 Hz band-pass, 8 kHz round trip), otherwise a
              low-bitrate codec round trip (Opus in Ogg, compression level U(0.9, 1.0), about 12-40 kbit/s)
    gain      gain U(-10, +15) dB followed by soft clipping (tanh)
- the output is float32 in [-1, 1]; "duration" is updated from the output length; every other key is copied, so
  "label" (and "id", "source") never change. Applied operations and parameters are returned in "augment".

The TSn arms pass no transform (mixing.SyntheticSource(augment=None)): their clips are the pool's audio as stored.
"""
from __future__ import annotations

import hashlib
import io
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from synthetic import layout

P_APPLY = 0.8
OPS = ("speed", "pitch", "reverb", "noise", "channel", "gain")
SPEED = (0.9, 1.1)
PITCH = (-2.0, 2.0)
SNR_DB = (5.0, 30.0)
GAIN_DB = (-10.0, 15.0)
CODEC_LEVEL = (0.9, 1.0)
TELEPHONE_HZ = (300.0, 3400.0)
SR = layout.SAMPLE_RATE


def rng_for(seed: int, index: int, clip_id: str) -> np.random.Generator:
    h = hashlib.sha256(f"{seed}|{index}|{clip_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def _resample(x: np.ndarray, up: int, down: int) -> np.ndarray:
    from scipy.signal import resample_poly
    return resample_poly(x.astype(np.float64), up, down).astype(np.float32)


def speed(x: np.ndarray, factor: float) -> np.ndarray:
    f = Fraction(1.0 / factor).limit_denominator(200)
    return _resample(x, f.numerator, f.denominator)


def pitch(x: np.ndarray, semitones: float) -> np.ndarray:
    import warnings
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)     # librosa 1.0's deprecated, unused STFT arguments
        y = librosa.effects.pitch_shift(x.astype(np.float32), sr=SR, n_steps=float(semitones), res_type="soxr_hq")
    return y[:len(x)].astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def reverb(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve
    rir = np.asarray(rir, dtype=np.float64)
    peak = np.max(np.abs(rir)) if len(rir) else 0.0
    if peak <= 0:
        return x.copy()
    onset = int(np.argmax(np.abs(rir) > 0.1 * peak))
    y = fftconvolve(x.astype(np.float64), rir[onset:])[:len(x)]
    r_in, r_out = rms(x), rms(y)
    return (y * (r_in / r_out)).astype(np.float32) if r_out > 0 else x.copy()


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float, offset: int) -> np.ndarray:
    n = np.asarray(noise, dtype=np.float64)
    if not len(n) or not len(x):
        return x.copy()
    reps = int(np.ceil((offset % len(n) + len(x)) / len(n))) + 1
    seg = np.tile(n, reps)[offset % len(n): offset % len(n) + len(x)]
    ps, pn = rms(x), rms(seg)
    if pn <= 0 or ps <= 0:
        return x.copy()
    seg = seg * (ps / pn) / (10 ** (snr_db / 20))
    return (x + seg).astype(np.float32)


def telephone(x: np.ndarray) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, TELEPHONE_HZ, btype="bandpass", fs=SR, output="sos")
    y = sosfiltfilt(sos, x.astype(np.float64))
    y = _resample(_resample(y, 1, 2), 2, 1)
    return y[:len(x)].astype(np.float32)


def codec(x: np.ndarray, level: float) -> np.ndarray:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1, 1).astype(np.float32), SR, format="OGG", subtype="OPUS", compression_level=float(level))
    buf.seek(0)
    y, rate = sf.read(buf, dtype="float32")
    if rate != SR:
        raise RuntimeError("codec round trip changed the sample rate")
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y[:len(x)].astype(np.float32)


def gain(x: np.ndarray, db: float) -> np.ndarray:
    return np.tanh(x.astype(np.float64) * 10 ** (db / 20)).astype(np.float32)


class Pool:
    """The TRAIN noise / RIR pool (noise.py writes pool.json): file lists, audio read on demand."""

    def __init__(self, noise_paths: list[str], rir_paths: list[str], identity: str | None = None):
        if not noise_paths or not rir_paths:
            raise RuntimeError("the augmentation pool needs noise clips and RIRs")
        self.noise_paths, self.rir_paths, self.identity = list(noise_paths), list(rir_paths), identity

    @classmethod
    def load(cls, path: Path | None = None) -> "Pool":
        path = layout.noise_dir() / "pool.json" if path is None else Path(path)
        data = json.loads(path.read_text())
        if not data.get("train_check", {}).get("pass"):
            raise RuntimeError(f"{path}: the TRAIN-pool disjointness check did not pass")
        return cls([c["path"] for c in data["noise"]], [r["path"] for r in data["rirs"]],
                   identity=hashlib.sha256(path.read_bytes()).hexdigest())

    @staticmethod
    def read(path: str) -> np.ndarray:
        import soundfile as sf
        a, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate != SR:
            raise ValueError(f"{path}: expected {SR} Hz")
        return a[:, 0]


class Augmenter:
    """transform(item, index) for mixing.SyntheticSource (see the module docstring)."""

    def __init__(self, pool: Pool, seed: int = layout.SEED, p_apply: float = P_APPLY):
        self.pool, self.seed, self.p_apply = pool, int(seed), float(p_apply)

    @property
    def identity(self) -> dict:
        return {"seed": self.seed, "p_apply": self.p_apply, "ops": OPS, "pool": self.pool.identity}

    def draw(self, rng: np.random.Generator) -> list[tuple[str, dict]]:
        if rng.random() >= self.p_apply:
            return []
        k = int(rng.integers(1, 4))
        chosen = sorted(rng.choice(len(OPS), size=k, replace=False).tolist())
        plan = []
        for i in chosen:
            op = OPS[i]
            if op == "speed":
                plan.append((op, {"factor": float(rng.uniform(*SPEED))}))
            elif op == "pitch":
                plan.append((op, {"semitones": float(rng.uniform(*PITCH))}))
            elif op == "reverb":
                plan.append((op, {"rir": int(rng.integers(len(self.pool.rir_paths)))}))
            elif op == "noise":
                plan.append((op, {"noise": int(rng.integers(len(self.pool.noise_paths))),
                                  "snr_db": float(rng.uniform(*SNR_DB)), "offset": int(rng.integers(0, 2 ** 31))}))
            elif op == "channel":
                if rng.random() < 0.5:
                    plan.append((op, {"kind": "telephone"}))
                else:
                    plan.append((op, {"kind": "codec", "level": float(rng.uniform(*CODEC_LEVEL))}))
            else:
                plan.append((op, {"db": float(rng.uniform(*GAIN_DB))}))
        return plan

    def apply(self, x: np.ndarray, plan: list[tuple[str, dict]]) -> np.ndarray:
        y = np.asarray(x, dtype=np.float32)
        for op, p in plan:
            if op == "speed":
                y = speed(y, p["factor"])
            elif op == "pitch":
                y = pitch(y, p["semitones"])
            elif op == "reverb":
                y = reverb(y, self.pool.read(self.pool.rir_paths[p["rir"]]))
            elif op == "noise":
                y = add_noise(y, self.pool.read(self.pool.noise_paths[p["noise"]]), p["snr_db"], p["offset"])
            elif op == "channel":
                y = telephone(y) if p["kind"] == "telephone" else codec(y, p["level"])
            else:
                y = gain(y, p["db"])
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(y, -1.0, 1.0).astype(np.float32)

    def __call__(self, item: dict, index: int) -> dict:
        plan = self.draw(rng_for(self.seed, index, item["id"]))
        out = dict(item)
        if plan:
            out["audio"] = self.apply(item["audio"], plan)
        else:
            out["audio"] = np.array(item["audio"], dtype=np.float32, copy=True)
        out["duration"] = len(out["audio"]) / SR
        out["augment"] = [[op, p] for op, p in plan]
        return out
