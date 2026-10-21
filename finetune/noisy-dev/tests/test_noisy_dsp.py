"""Signal-processing contracts (plans/build.md "Code and tests" 2, 6, 7): the added-noise ratio and
event levels, impulsive-event count and placement rules, reverberation identity / delayed identity /
tail preservation, and the radio channel (final-output bandwidth, delay alignment within 1 ms, no
truncation of the last 100 ms). The radio tests run the real pinned Codec2 binary."""
from __future__ import annotations

import math
import unittest

import noisy_fixtures as fx
import numpy as np
from scipy import signal

import noisy_dsp as dsp

SR = fx.SR


def to_pcm16(x: np.ndarray) -> np.ndarray:
    """What a written 16-bit FLAC holds."""
    return np.clip(np.round(np.asarray(x) * 32768.0), -32768, 32767).astype(np.int16)


def level_db(x: np.ndarray) -> float:
    return 10 * math.log10(float(np.mean(np.square(x))))


class RatioTest(unittest.TestCase):
    def test_noise_gain_sets_utterance_rms_ratio(self):
        speech = fx.speech_like(3.0, 1)
        for ratio in (10.0, 5.0, 0.0, -3.0):
            noise = fx.noise_clip(3.0, 2, "brown")
            g = dsp.noise_gain(speech, noise, ratio)
            measured = level_db(speech) - level_db(g * noise)  # whole-length RMS, no activity gate
            self.assertAlmostEqual(measured, ratio, delta=1e-9)
            self.assertAlmostEqual(dsp.added_noise_ratio_db(speech, g * noise), ratio, delta=1e-9)

    def test_looped_segment_has_requested_length_and_crossfade(self):
        clip = fx.noise_clip(0.7, 3)
        seg = dsp.loop(clip, 3 * SR, offset=1234)
        self.assertEqual(len(seg), 3 * SR)
        # The stream starts at the offset and continues with the clip itself (no gap of silence).
        np.testing.assert_allclose(seg[:2000], clip[1234:3234])
        frames = seg[: len(seg) // 320 * 320].reshape(-1, 320)
        self.assertGreater(np.sqrt(np.mean(frames ** 2, axis=1)).min(), 0.03)  # no dropouts at the seams

    def test_active_frames_gate(self):
        x = np.zeros(SR)
        x[3200:6400] = 0.1 * np.sin(np.arange(3200) * 0.3)    # frames 10-19 at full level
        x[9600:12800] = 0.002 * np.sin(np.arange(3200) * 0.3)  # 34 dB down: inactive at a 30 dB gate
        mask = dsp.active_frames(x, gate_db=30.0)
        self.assertEqual(len(mask), SR // dsp.FRAME)
        self.assertEqual(np.nonzero(mask)[0].tolist(), list(range(10, 20)))


class EventRuleTest(unittest.TestCase):
    def test_count_is_one_per_started_three_seconds_at_most_six(self):
        for seconds, count in ((0.5, 1), (3.0, 1), (3.01, 2), (8.9, 3), (17.9, 6), (40.0, 6)):
            self.assertEqual(dsp.event_count(int(round(seconds * SR))), count, seconds)

    def test_placement_shifts_and_truncates(self):
        n = 2 * SR
        start, length, shifted, truncated = dsp.place_event(4000, n, onset=1000)
        self.assertEqual((start, length, shifted, truncated), (1000, 4000, False, False))
        start, length, shifted, truncated = dsp.place_event(4000, n, onset=n - 100)
        self.assertEqual((start, length, shifted, truncated), (n - 4000, 4000, True, False))
        start, length, shifted, truncated = dsp.place_event(3 * SR, n, onset=500)
        self.assertEqual((start, length, shifted, truncated), (0, n, True, True))

    def test_truncated_event_fades_over_10_ms(self):
        event = np.ones(3 * SR)
        out = dsp.fit_event(event, 2 * SR)
        self.assertEqual(len(out), 2 * SR)
        self.assertEqual(out[-1], 0.0)
        np.testing.assert_array_equal(out[: 2 * SR - 160], 1.0)
        self.assertTrue(np.all(np.diff(out[-160:]) <= 0))

    def test_event_detection_windows(self):
        x = fx.bursts(4.0, [0.5, 0.56, 2.0], seed=5)  # the first two merge (gap under 100 ms)
        events = dsp.detect_events(x)
        self.assertEqual(len(events), 2)
        (s0, l0), (s1, l1) = events
        self.assertLessEqual(abs(s0 - int(0.5 * SR)), dsp.FRAME)
        self.assertLessEqual(abs(s1 - int(2.0 * SR)), dsp.FRAME)
        for _start, length in events:
            self.assertGreaterEqual(length, int(0.03 * SR))
            self.assertLessEqual(length, 3 * SR)


class ReverbTest(unittest.TestCase):
    def test_delta_and_delayed_delta_reproduce_the_input(self):
        x = fx.speech_like(2.0, 7)
        for delay in (0, 37, 480):
            h = np.zeros(2000)
            h[delay] = 0.6  # any gain: the output is rescaled to the input's RMS
            y = dsp.reverberate(x, dsp.align_rir(h))
            self.assertEqual(len(y), len(x) + dsp.REVERB_TAIL)
            np.testing.assert_array_equal(to_pcm16(y[: len(x)]), to_pcm16(x))
            np.testing.assert_array_equal(to_pcm16(y[len(x):]), 0)

    def test_onset_is_first_sample_above_ten_percent_of_max(self):
        h = np.zeros(1000)
        h[100] = 0.05   # below 10 % of the maximum: dropped with everything before it
        h[150] = 0.2    # the onset
        h[300] = 1.0
        aligned = dsp.align_rir(h)
        self.assertEqual(aligned[0], 0.2)
        self.assertEqual(len(aligned), 850)

    def test_tail_kept_and_last_200_ms_preserved(self):
        # Broadband input (so the RIR's spectral colouring affects every region alike), with a
        # louder burst in the last 200 ms: energy right at the end must survive.
        rng = np.random.default_rng(9)
        x = 0.02 * rng.standard_normal(3 * SR)
        x[-int(0.2 * SR):] *= 4.0
        h = np.zeros(800)
        h[0] = 1.0
        h[1:] = 0.05 * rng.standard_normal(799) * np.exp(-np.arange(799) / 120)  # 50 ms, short RIR
        y = dsp.reverberate(x, dsp.align_rir(h))
        last = slice(len(x) - int(0.2 * SR), len(x))
        self.assertLess(abs(level_db(y[last]) - level_db(x[last])), 1.0)
        self.assertGreater(np.abs(y[len(x):len(x) + 700]).max(), 0)  # the tail after the input end is kept
        self.assertAlmostEqual(level_db(y[: len(x)]), level_db(x), delta=1e-6)


class RadioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.codec = dsp.Codec2()
        cls.delay = dsp.measure_codec_delay(cls.codec)

    def test_delay_is_measured_in_plausible_range(self):
        self.assertGreaterEqual(self.delay, 0)
        self.assertLessEqual(self.delay, 4 * dsp.CODEC2_FRAME)  # covered by the four flush frames

    def test_final_output_bandwidth(self):
        noise = np.random.default_rng(10).standard_normal(10 * SR) * 0.1
        y = dsp.radio(noise, self.codec, self.delay)
        self.assertEqual(len(y), len(noise))
        f, p = signal.welch(y, SR, nperseg=4096)
        passband = p[(f >= 300) & (f <= 3400)].sum()
        above = p[f > 3600].sum()
        self.assertLessEqual(10 * math.log10(above / passband), -40.0)
        self.assertLessEqual(dsp.bandwidth_ratio_db(y), -40.0)

    def test_delay_compensation_aligns_within_1_ms(self):
        x = dsp.delay_test_signal(seconds=12.0, seed=3, sr=SR)  # not the measurement signal (seed 0)
        y = dsp.radio(x, self.codec, self.delay)
        lag = dsp.envelope_lag(x, y, max_lag=int(0.05 * SR), window=2 * dsp.CODEC2_FRAME)
        self.assertLessEqual(abs(lag), SR // 1000, f"residual lag {lag} samples")

    def test_compensation_of_a_known_delay(self):
        class DelayCodec:  # a stand-in codec that only delays by a known amount
            def encode_decode(self, pcm):
                return np.concatenate([np.zeros(93, np.int16), pcm[:-93]])
        stub = DelayCodec()
        delay = dsp.measure_codec_delay(stub)
        self.assertLessEqual(abs(delay - 93), 2)

    def test_last_100_ms_not_truncated(self):
        x = fx.speech_like(2.0, 11)
        tail = int(0.1 * SR)
        x[-tail:] = 0.08 * signal.sawtooth(2 * np.pi * 180 * np.arange(tail) / SR)
        y = dsp.radio(x, self.codec, self.delay)
        self.assertEqual(len(y), len(x))
        self.assertGreater(level_db(y[-tail:]), level_db(x[-tail:]) - 6.0)

    def test_output_rescaled_to_input_rms(self):
        x = fx.speech_like(3.0, 12, level_db=-31.0)
        y = dsp.radio(x, self.codec, self.delay)
        self.assertAlmostEqual(level_db(y), level_db(x), delta=1e-6)

    def test_codec_is_deterministic(self):
        x = fx.speech_like(1.0, 13)
        a = dsp.radio(x, self.codec, self.delay)
        b = dsp.radio(x, self.codec, self.delay)
        np.testing.assert_array_equal(a, b)


class SerializedRadioTest(unittest.TestCase):
    """Review round 2, finding 8: the bandwidth contract holds on the decoded, serialized output
    (the build's FLAC writer) across admitted input levels, and the chain handles lengths that are
    not whole frames."""

    @classmethod
    def setUpClass(cls):
        cls.codec = dsp.Codec2()
        cls.delay = dsp.measure_codec_delay(cls.codec)
        cls.root = fx.fresh_dir("radio-serialized")

    def serialized(self, y: np.ndarray, name: str) -> np.ndarray:
        import noisy_build
        y, _ = dsp.headroom(y)
        path = self.root / f"{name}.flac"
        noisy_build.write_flac(path, y)
        return fx.read_pcm(path)

    def test_bandwidth_after_serialization_across_levels(self):
        # From just above the silent threshold (-70 dBFS) to loud input; white noise and speech-like.
        for level in (-69.0, -60.0, -45.0, -30.0, -20.0, -10.0, -3.0):
            for kind in ("noise", "speech"):
                x = (np.random.default_rng(int(-level)).standard_normal(5 * SR) if kind == "noise"
                     else fx.speech_like(5.0, int(-level)))
                x *= 10 ** (level / 20) / np.sqrt(np.mean(x ** 2))
                y = self.serialized(dsp.radio(x, self.codec, self.delay), f"{kind}{-level:.0f}")
                self.assertLessEqual(dsp.bandwidth_ratio_db(y), -40.0, (kind, level))

    def test_lengths_that_are_not_whole_frames(self):
        for n in (int(0.3 * SR) + 1, 2 * SR + 1, 2 * SR + 159, 2 * SR + 161, 2 * SR + 319, 3 * SR + 7):
            x = fx.speech_like(n / SR + 0.01, n)[:n]
            tail = int(0.1 * SR)
            x[-tail:] = 0.08 * signal.sawtooth(2 * np.pi * 180 * np.arange(tail) / SR)
            y = dsp.radio(x, self.codec, self.delay)
            self.assertEqual(len(y), n)
            self.assertGreater(level_db(y[-tail:]), level_db(x[-tail:]) - 6.0, n)
            self.assertAlmostEqual(level_db(y), level_db(x), delta=1e-6)
            r = dsp.reverberate(x, dsp.align_rir(np.array([0.0, 1.0, 0.3])))
            self.assertEqual(len(r), n + dsp.REVERB_TAIL)

    def test_24_bit_writer_is_exact_for_16_bit_input(self):
        import noisy_build
        pcm = np.array([-32768, -1, 0, 1, 12345, 32767], dtype=np.int16)
        x = dsp.from_int16(pcm)
        path = self.root / "exact.flac"
        noisy_build.write_flac(path, x)
        np.testing.assert_array_equal(fx.read_pcm(path), x)


class HeadroomTest(unittest.TestCase):
    def test_peak_limited_as_a_whole(self):
        x = np.array([0.2, -1.5, 0.5])
        y, scale = dsp.headroom(x)
        self.assertAlmostEqual(np.abs(y).max(), 0.99)
        self.assertAlmostEqual(scale, 0.99 / 1.5)
        y, scale = dsp.headroom(np.array([0.1, -0.5]))
        self.assertEqual(scale, 1.0)


if __name__ == "__main__":
    unittest.main()
