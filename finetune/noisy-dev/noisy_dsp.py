"""Signal processing of the noisy development-set copy. See plans/build.md "Conditions".

Every function is deterministic for given inputs (float64 numpy/scipy, single-threaded; the
Codec2 binary is deterministic per process) and draws no random numbers: the build passes in
everything it drew from the per-utterance generator.

- Levels: utterance RMS over the whole length ("added-noise ratio", not a measured acoustic SNR);
  the 20 ms, -30 dB-relative activity gate is a diagnostic and places impulsive events.
- Continuous noise: a source shorter than the utterance is looped with a 50 ms raised-cosine
  crossfade (loop).
- Impulsive events: detect_events cuts a clip at frames within 30 dB of its peak frame (gaps under
  100 ms merged, events of 30 ms to 3 s kept); event_count, place_event and fit_event implement
  the count, placement and truncation rules.
- Reverberation: align_rir drops everything before the first sample above 10 % of the maximum;
  reverberate keeps the full tail, truncates to the input length plus REVERB_TAIL and rescales so
  that the input region [0, len(x)) has the input's RMS.
- Radio: radio() is the Codec2 3200 channel surrogate of "Radio channel", with the codec delay
  measured once (measure_codec_delay) and compensated.
"""
from __future__ import annotations

import math
import subprocess

import numpy as np
from scipy import signal

import noisy_paths

SR = noisy_paths.SAMPLE_RATE
FRAME = SR // 50                  # 20 ms frames (320 samples)
ACTIVITY_GATE_DB = 30.0           # active frame: energy within 30 dB of the loudest frame
CROSSFADE = SR // 20              # 50 ms looping crossfade
EVENT_GATE_DB = 30.0
EVENT_MERGE_GAP_S = 0.1
EVENT_MIN_S, EVENT_MAX_S = 0.03, 3.0
EVENT_PER_SECONDS, EVENT_MAX_COUNT = 3.0, 6
EVENT_FADE = SR // 100            # 10 ms fade on a truncated event
RIR_ONSET_FRACTION = 0.1
REVERB_TAIL = SR // 2             # 0.5 s of reverberant tail after the input end
HEADROOM_PEAK = 0.99
RADIO_INPUT_DBFS = -20.0
RADIO_BAND = (300.0, 3400.0)
RADIO_BUTTER_N = 4                # scipy N=4 band-pass: an 8th-order Butterworth, run zero-phase
RADIO_CLIP_DRIVE = 2.0            # tanh(2x) / tanh(2)
CODEC2_MODE = "3200"
CODEC2_RATE = 8000
CODEC2_FRAME = 160                # 20 ms at 8 kHz in 3200 bit/s mode
CODEC2_FLUSH_FRAMES = 4
DELAY_SIGNAL = {"seconds": 12.0, "seed": 0, "sample_rate": CODEC2_RATE, "envelope_window": CODEC2_FRAME}

PARAMETERS = {  # recorded in the build lock
    "sample_rate": SR, "frame": FRAME, "activity_gate_db": ACTIVITY_GATE_DB, "crossfade": CROSSFADE,
    "crossfade_shape": "raised cosine, equal gain", "event_gate_db": EVENT_GATE_DB,
    "event_merge_gap_s": EVENT_MERGE_GAP_S, "event_min_s": EVENT_MIN_S, "event_max_s": EVENT_MAX_S,
    "event_per_seconds": EVENT_PER_SECONDS, "event_max_count": EVENT_MAX_COUNT, "event_fade": EVENT_FADE,
    "rir_onset_fraction": RIR_ONSET_FRACTION, "reverb_tail": REVERB_TAIL,
    "reverb_normalization": "RMS over the input region [0, len(x)) equals the input RMS",
    "headroom_peak": HEADROOM_PEAK, "radio_input_dbfs": RADIO_INPUT_DBFS, "radio_band_hz": list(RADIO_BAND),
    "radio_filter": f"scipy.signal.butter(N={RADIO_BUTTER_N}, band-pass, sos), sosfiltfilt (zero phase)",
    "radio_clip": f"tanh({RADIO_CLIP_DRIVE}x)/tanh({RADIO_CLIP_DRIVE}) at 8 kHz",
    "resampler": "scipy.signal.resample_poly (default Kaiser window)", "codec2_mode": CODEC2_MODE,
    "codec2_frame": CODEC2_FRAME, "codec2_flush_frames": CODEC2_FLUSH_FRAMES, "delay_signal": DELAY_SIGNAL,
    "pcm": "16-bit, round(x * 32768) clipped to int16",
}


# --------------------------------------------------------------------------------------------- levels

def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(x, dtype=np.float64) * 32768.0), -32768, 32767).astype(np.int16)


def from_int16(pcm: np.ndarray) -> np.ndarray:
    return np.asarray(pcm, dtype=np.float64) / 32768.0


def power(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x * x)) if x.size else 0.0


def rms(x: np.ndarray) -> float:
    return math.sqrt(power(x))


def level_db(x: np.ndarray) -> float:
    """RMS level in dBFS (-inf for silence)."""
    p = power(x)
    return 10 * math.log10(p) if p > 0 else -math.inf


def frame_energies(x: np.ndarray, frame: int = FRAME) -> np.ndarray:
    """Mean-square energy of consecutive frames; the last partial frame is zero-padded."""
    x = np.asarray(x, dtype=np.float64)
    n = max(1, math.ceil(len(x) / frame))
    pad = np.zeros(n * frame)
    pad[: len(x)] = x
    return np.mean(pad.reshape(n, frame) ** 2, axis=1)


def active_frames(x: np.ndarray, gate_db: float = ACTIVITY_GATE_DB, frame: int = FRAME) -> np.ndarray:
    """Frames whose energy is within gate_db of the loudest frame (all False for silence)."""
    energy = frame_energies(x, frame)
    top = energy.max()
    if top <= 0:
        return np.zeros(len(energy), dtype=bool)
    return energy >= top * 10 ** (-gate_db / 10)


def active_fraction(x: np.ndarray, gate_db: float = ACTIVITY_GATE_DB) -> float:
    return float(np.mean(active_frames(x, gate_db)))


def noise_gain(reference: np.ndarray, noise: np.ndarray, ratio_db: float) -> float:
    """Gain g with 10 log10(P(reference) / P(g * noise)) = ratio_db, powers over the whole length."""
    p_ref, p_noise = power(reference), power(noise)
    if p_ref <= 0 or p_noise <= 0:
        raise ValueError("reference and noise must have nonzero power")
    return math.sqrt(p_ref / (p_noise * 10 ** (ratio_db / 10)))


def added_noise_ratio_db(reference: np.ndarray, noise: np.ndarray) -> float:
    return 10 * math.log10(power(reference) / power(noise))


def headroom(x: np.ndarray, peak: float = HEADROOM_PEAK) -> tuple[np.ndarray, float]:
    """(x scaled as a whole so that its peak is at most `peak`, the scale)."""
    top = float(np.max(np.abs(x))) if len(x) else 0.0
    scale = peak / top if top > peak else 1.0
    return (x * scale if scale != 1.0 else np.array(x, dtype=np.float64)), scale


# --------------------------------------------------------------------------------------------- continuous noise

def _fade_in(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)


def loop(clip: np.ndarray, length: int, offset: int, crossfade: int = CROSSFADE) -> np.ndarray:
    """`length` samples of the clip looped with a raised-cosine crossfade, starting at `offset`
    (0 <= offset < len(clip)) into the looped stream."""
    clip = np.asarray(clip, dtype=np.float64)
    if len(clip) <= 2 * crossfade:
        raise ValueError(f"clip of {len(clip)} samples is too short to loop with a {crossfade}-sample crossfade")
    if not 0 <= offset < len(clip):
        raise ValueError("offset must lie within the clip")
    period = len(clip) - crossfade
    copies = max(1, math.ceil((offset + length - len(clip)) / period) + 1)
    stream = np.zeros((copies - 1) * period + len(clip))
    fade_in = _fade_in(crossfade)
    for k in range(copies):
        piece = clip.copy()
        if k > 0:
            piece[:crossfade] *= fade_in
        if k < copies - 1:
            piece[-crossfade:] *= 1.0 - fade_in
        stream[k * period: k * period + len(clip)] += piece
    return stream[offset: offset + length]


# --------------------------------------------------------------------------------------------- impulsive events

def detect_events(x: np.ndarray, gate_db: float = EVENT_GATE_DB, merge_gap_s: float = EVENT_MERGE_GAP_S,
                  min_s: float = EVENT_MIN_S, max_s: float = EVENT_MAX_S) -> list[tuple[int, int]]:
    """(start, length) in samples of the clip's events: runs of 20 ms frames within gate_db of the
    clip's peak frame, gaps under merge_gap_s merged; runs shorter than min_s or longer than max_s
    are dropped."""
    mask = active_frames(x, gate_db)
    runs, start = [], None
    for i, on in enumerate(np.append(mask, False)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append([start, i])
            start = None
    merged: list[list[int]] = []
    gap = merge_gap_s * SR / FRAME
    for run in runs:
        if merged and run[0] - merged[-1][1] < gap:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    events = []
    for a, b in merged:
        s, e = a * FRAME, min(b * FRAME, len(x))
        if min_s * SR <= e - s <= max_s * SR:
            events.append((int(s), int(e - s)))
    return events


def event_count(n: int) -> int:
    """One event per started EVENT_PER_SECONDS of utterance, at most EVENT_MAX_COUNT."""
    return min(EVENT_MAX_COUNT, max(1, math.ceil(n / (EVENT_PER_SECONDS * SR))))


def place_event(length: int, n: int, onset: int) -> tuple[int, int, bool, bool]:
    """(start, kept length, shifted, truncated) of an event of `length` samples drawn at `onset`
    in an utterance of n samples: an event past the end is shifted earlier; one longer than the
    utterance starts at 0 and is truncated (fit_event fades it)."""
    if length > n:
        return 0, n, True, True
    if onset + length > n:
        return n - length, length, True, False
    return onset, length, False, False


def fit_event(event: np.ndarray, n: int) -> np.ndarray:
    """The event truncated to n samples with a 10 ms fade-out (unchanged if it fits)."""
    event = np.asarray(event, dtype=np.float64)
    if len(event) <= n:
        return event.copy()
    out = event[:n].copy()
    k = min(EVENT_FADE, n)
    out[-k:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(k) / max(1, k - 1))
    return out


# --------------------------------------------------------------------------------------------- reverberation

def align_rir(h: np.ndarray, fraction: float = RIR_ONSET_FRACTION) -> np.ndarray:
    """The RIR from its first sample whose magnitude exceeds `fraction` of the maximum."""
    h = np.asarray(h, dtype=np.float64)
    if h.ndim != 1:
        raise ValueError("align_rir takes one channel")
    top = np.max(np.abs(h))
    if top <= 0:
        raise ValueError("all-zero RIR")
    onset = int(np.argmax(np.abs(h) > fraction * top))
    return h[onset:].copy()


def reverberate(x: np.ndarray, h: np.ndarray, tail: int = REVERB_TAIL) -> np.ndarray:
    """x convolved with the (aligned) RIR, full tail kept, truncated to len(x) + tail and scaled so
    that its RMS over [0, len(x)) equals the input's RMS."""
    x = np.asarray(x, dtype=np.float64)
    full = signal.fftconvolve(x, np.asarray(h, dtype=np.float64))
    y = np.zeros(len(x) + tail)
    k = min(len(full), len(y))
    y[:k] = full[:k]
    region = rms(y[: len(x)])
    if region <= 0:
        raise ValueError("reverberated input region is silent")
    return y * (rms(x) / region)


# --------------------------------------------------------------------------------------------- radio

def bandpass_sos(fs: int) -> np.ndarray:
    return signal.butter(RADIO_BUTTER_N, RADIO_BAND, btype="bandpass", fs=fs, output="sos")


class Codec2:
    """c2enc | c2dec of the pinned nixpkgs Codec2, 3200 bit/s, headerless, through pipes."""

    def __init__(self, enc=noisy_paths.CODEC2_ENC, dec=noisy_paths.CODEC2_DEC, mode: str = CODEC2_MODE) -> None:
        self.enc, self.dec, self.mode = str(enc), str(dec), mode

    def encode_decode(self, pcm: np.ndarray) -> np.ndarray:
        """Decoded int16 PCM at 8 kHz; the input must be a whole number of frames."""
        pcm = np.asarray(pcm)
        if pcm.dtype != np.int16 or len(pcm) % CODEC2_FRAME:
            raise ValueError("Codec2 input must be int16 and a whole number of 160-sample frames")
        env = {"PATH": "/nonexistent", "LC_ALL": "C"}
        bits = subprocess.run([self.enc, self.mode, "-", "-"], input=pcm.astype("<i2").tobytes(),
                              capture_output=True, check=True, env=env).stdout
        out = subprocess.run([self.dec, self.mode, "-", "-"], input=bits, capture_output=True, check=True,
                             env=env).stdout
        decoded = np.frombuffer(out, dtype="<i2").astype(np.int16)
        if len(decoded) != len(pcm):
            raise RuntimeError(f"Codec2 returned {len(decoded)} samples for {len(pcm)}")
        return decoded


def delay_test_signal(seconds: float = DELAY_SIGNAL["seconds"], seed: int = DELAY_SIGNAL["seed"],
                      sr: int = CODEC2_RATE) -> np.ndarray:
    """Aperiodic harmonic bursts (60-300 ms, 80-300 ms apart, pitch 100-200 Hz) at -20 dBFS RMS: a
    fixed signal whose envelope has a unique cross-correlation peak."""
    rng = np.random.Generator(np.random.PCG64(seed))
    n = int(round(seconds * sr))
    t = np.arange(n) / sr
    x = np.zeros(n)
    pos = 0.2
    while pos < seconds - 0.5:
        dur, f0 = rng.uniform(0.06, 0.3), rng.uniform(100, 200)
        seg = (t >= pos) & (t < pos + dur)
        x[seg] = sum(np.sin(2 * np.pi * f0 * h * t[seg]) / h for h in range(1, int(3400 / f0)))
        pos += dur + rng.uniform(0.08, 0.3)
    return x * (10 ** (RADIO_INPUT_DBFS / 20) / rms(x))


def _envelope(x: np.ndarray, window: int) -> np.ndarray:
    w = np.hanning(window + 2)[1:-1]
    e = np.sqrt(np.maximum(signal.fftconvolve(np.asarray(x, dtype=np.float64) ** 2, w / w.sum(), mode="same"), 0))
    return e - e.mean()


def envelope_lag(ref: np.ndarray, out: np.ndarray, max_lag: int, window: int) -> int:
    """Lag in samples (positive: `out` is late) maximizing the cross-correlation of the two
    signals' smoothed RMS envelopes, searched over [-max_lag, max_lag]."""
    a, b = _envelope(ref, window), _envelope(out, window)
    cc = signal.correlate(b, a, mode="full", method="fft")
    lags = signal.correlation_lags(len(b), len(a), mode="full")
    m = np.abs(lags) <= max_lag
    return int(lags[m][int(np.argmax(cc[m]))])


def _pad_frames(x8: np.ndarray) -> np.ndarray:
    frames = math.ceil(len(x8) / CODEC2_FRAME) + CODEC2_FLUSH_FRAMES
    buf = np.zeros(frames * CODEC2_FRAME)
    buf[: len(x8)] = x8
    return buf


def measure_codec_delay(codec) -> int:
    """Codec delay in 8 kHz samples, by envelope cross-correlation on the fixed delay_test_signal
    (seed 0, 12 s), searched over [0, CODEC2_FLUSH_FRAMES frames]."""
    x = delay_test_signal(DELAY_SIGNAL["seconds"], DELAY_SIGNAL["seed"], CODEC2_RATE)
    buf = _pad_frames(x)
    out = from_int16(codec.encode_decode(to_int16(buf)))
    a, b = _envelope(buf, DELAY_SIGNAL["envelope_window"]), _envelope(out, DELAY_SIGNAL["envelope_window"])
    cc = signal.correlate(b, a, mode="full", method="fft")
    lags = signal.correlation_lags(len(b), len(a), mode="full")
    m = (lags >= 0) & (lags <= CODEC2_FLUSH_FRAMES * CODEC2_FRAME)
    return int(lags[m][int(np.argmax(cc[m]))])


def radio(x: np.ndarray, codec, delay: int) -> np.ndarray:
    """The radio channel: -20 dBFS RMS, band-pass, 8 kHz, Codec2 3200 (frame padding plus four
    flush frames, `delay` compensated), tanh(2x)/tanh(2), band-pass, 16 kHz, input RMS restored.
    Output has the input's length."""
    x = np.asarray(x, dtype=np.float64)
    level = rms(x)
    if level <= 0:
        raise ValueError("radio input is silent")
    if not 0 <= delay <= CODEC2_FLUSH_FRAMES * CODEC2_FRAME:
        raise ValueError(f"codec delay {delay} is not covered by the flush frames")
    a = signal.sosfiltfilt(bandpass_sos(SR), x * (10 ** (RADIO_INPUT_DBFS / 20) / level))
    a8 = signal.resample_poly(a, CODEC2_RATE, SR)
    decoded = from_int16(codec.encode_decode(to_int16(_pad_frames(a8))))
    o = decoded[delay: delay + len(a8)]
    c = np.tanh(RADIO_CLIP_DRIVE * o) / math.tanh(RADIO_CLIP_DRIVE)
    c = signal.sosfiltfilt(bandpass_sos(CODEC2_RATE), c)
    y16 = signal.resample_poly(c, SR, CODEC2_RATE)
    y = np.zeros(len(x))
    k = min(len(y16), len(y))
    y[:k] = y16[:k]
    out_level = rms(y)
    if out_level <= 0:
        raise RuntimeError("radio output is silent")
    return y * (level / out_level)


def bandwidth_ratio_db(y: np.ndarray, fs: int = SR) -> float:
    """10 log10(energy above 3.6 kHz / energy in 300-3400 Hz), Welch PSD (4096-point segments)."""
    f, p = signal.welch(np.asarray(y, dtype=np.float64), fs, nperseg=min(4096, len(y)))
    return 10 * math.log10(p[f > 3600].sum() / p[(f >= 300) & (f <= 3400)].sum())
