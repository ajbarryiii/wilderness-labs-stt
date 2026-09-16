"""Full small-graph parity and fixed-token CPU control contract checks."""
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import runtime
from runtime import CPUCTranslate2Control, CPUReplayWhisper, WhisperConfig, quantize_activation


class CPUReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_quantization_boundaries(self):
        below = torch.nextafter(torch.tensor(-0.5), torch.tensor(0.0))
        above = torch.nextafter(torch.tensor(0.5), torch.tensor(0.0))
        x = torch.tensor([-1.0, -0.5, below, -0.0, 0.0, above, 0.5, 1.0])
        torch.testing.assert_close(quantize_activation(x, 1),
                                   torch.tensor([-1., -1., -1., 1., 1., 1., 1., 1.]))
        torch.testing.assert_close(quantize_activation(x, 2),
                                   torch.tensor([-1., -1., 0., 0., 0., 0., 1., 1.]))

    def test_factory_restored_on_success_and_failure(self):
        original = runtime._shared._Weight
        CPUReplayWhisper(WhisperConfig.tiny_smoke(), implementation="dense")
        self.assertIs(runtime._shared._Weight, original)
        mismatched = WhisperConfig(n_audio_state=64, n_text_state=32)
        with self.assertRaises(ValueError):
            CPUReplayWhisper(mismatched, implementation="dense")
        self.assertIs(runtime._shared._Weight, original)

    def test_dense_and_native_complete_graphs_match_exactly(self):
        from native import available_backends
        backends = available_backends()
        self.assertIn("scalar", backends)
        config = WhisperConfig.tiny_smoke()
        generator = torch.Generator().manual_seed(71)
        mel = torch.randn(1, config.n_mels, 2 * config.n_audio_ctx, generator=generator)
        changed_mel = torch.randn(mel.shape, generator=generator)
        tokens = [3, 4, 5, 6]
        for distribution, activation_bits in (("binary", 1), ("ternary", 1), ("ternary", 2)):
            with self.subTest(distribution=distribution, activation_bits=activation_bits):
                dense = CPUReplayWhisper(config, distribution=distribution,
                                         activation_bits=activation_bits, implementation="dense")
                encoded = dense.encode(mel)
                expected = dense.decode(encoded, tokens).clone()
                predictions = dense.last_predictions.clone()
                changed = dense.run(changed_mel, tokens).clone()
                self.assertFalse(torch.equal(changed, expected), "Mel input must affect logits")
                changed_tokens = dense.run(mel, [8, 9]).clone()
                repeated = dense.run(mel, tokens)
                torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
                self.assertEqual(dense.factory.parameter_count, config.parameter_count())
                for implementation in backends:
                    with self.subTest(implementation=implementation):
                        model = CPUReplayWhisper(
                            config, distribution=distribution,
                            activation_bits=activation_bits, implementation=implementation)
                        # Integer-valued dense FP32 dots are exact at these
                        # widths. Matching epilogues permit bitwise checks,
                        # avoiding tolerances that hide quantization tie flips.
                        actual_encoded = model.encode(mel)
                        torch.testing.assert_close(actual_encoded, encoded, rtol=0, atol=0)
                        actual = model.decode(actual_encoded, tokens)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(model.last_predictions, predictions, rtol=0, atol=0)
                        torch.testing.assert_close(model.run(changed_mel, tokens), changed, rtol=0, atol=0)
                        torch.testing.assert_close(model.run(mel, [8, 9]), changed_tokens, rtol=0, atol=0)
                        torch.testing.assert_close(model.run(mel, tokens), expected, rtol=0, atol=0)
                        self.assertEqual(model.metadata()["activation_bits"], activation_bits)
                        self.assertLess(model.metadata()["weight_storage_bytes"],
                                        dense.metadata()["weight_storage_bytes"])

    def test_invalid_quantized_runtime_options(self):
        for kwargs in ({"activation_bits": 3}, {"implementation": "cuda"},
                       {"threads": 0}, {"distribution": "binary", "activation_bits": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CPUReplayWhisper(WhisperConfig.tiny_smoke(), **kwargs)

    def test_optimized_ragged_graph_and_decoder_cache_reuse(self):
        from native import available_backends
        if "avx512_opt" not in available_backends():
            self.skipTest("Optimized AVX512 backend is unavailable on this CPU")
        # Encoder lengths 14 then 7 exercise tiled row tails. Width 40 and
        # vocabulary 67 also exercise packed word and output-channel tails.
        config = WhisperConfig(
            name="ragged-smoke-only", n_mels=5, n_audio_ctx=7,
            n_audio_state=40, n_audio_head=5, n_audio_layer=1,
            n_vocab=67, n_text_ctx=12, n_text_state=40,
            n_text_head=5, n_text_layer=1, sot_id=65, no_timestamps_id=66)
        generator = torch.Generator().manual_seed(219)
        inputs = [torch.randn(1,config.n_mels,2*config.n_audio_ctx,generator=generator)
                  for _ in range(2)]
        # Grow and shrink token caches, then return to the original input and
        # replay length. Every decoder step's greedy result is compared too.
        cases = ((0,[3]), (1,[3,4,5,6,7]), (0,[8,9]), (0,[3,4,5,6,7]), (0,[3]))
        for distribution,activation_bits in (("binary",1),("ternary",1),("ternary",2)):
            models = {
                implementation: CPUReplayWhisper(
                    config,distribution=distribution,activation_bits=activation_bits,
                    implementation=implementation,threads=3)
                for implementation in ("dense","avx512","avx512_opt")
            }
            for input_index,tokens in cases:
                mel = inputs[input_index]
                dense = models["dense"]
                expected_encoded = dense.encode(mel)
                expected = dense.decode(expected_encoded,tokens).clone()
                expected_predictions = dense.last_predictions.clone()
                for implementation in ("avx512","avx512_opt"):
                    with self.subTest(distribution=distribution,a=activation_bits,
                                      implementation=implementation,input=input_index,tokens=tokens):
                        model = models[implementation]
                        encoded = model.encode(mel)
                        torch.testing.assert_close(encoded,expected_encoded,rtol=0,atol=0)
                        torch.testing.assert_close(model.decode(encoded,tokens),expected,rtol=0,atol=0)
                        torch.testing.assert_close(model.last_predictions,expected_predictions,
                                                   rtol=0,atol=0)


class CPUControlTests(unittest.TestCase):
    def make_control(self, returned=None, actual_type=None, compute_type="int8_float32"):
        model = mock.Mock(compute_type=actual_type or compute_type)
        model.generate.side_effect = lambda encoded, prompts, **kw: [SimpleNamespace(
            sequences_ids=[prompts[0][2:] if returned is None else returned])]
        constructor = mock.Mock(return_value=model)
        fake_ct2 = SimpleNamespace(
            __version__="test", models=SimpleNamespace(Whisper=constructor),
            get_supported_compute_types=lambda device: {"float32", "int8_float32"})
        with mock.patch.dict(sys.modules, {"ctranslate2": fake_ct2}):
            control = CPUCTranslate2Control("/mnt/hd/wilderness-labs-stt/test-model", threads=3,
                                           compute_type=compute_type)
        return control, constructor

    def test_cpu_precision_and_checked_forced_replay(self):
        for compute_type in ("float32", "int8_float32"):
            with self.subTest(compute_type=compute_type):
                control, constructor = self.make_control(compute_type=compute_type)
                self.assertEqual(constructor.call_args.kwargs["device"], "cpu")
                self.assertEqual(constructor.call_args.kwargs["intra_threads"], 3)
                self.assertEqual(constructor.call_args.kwargs["compute_type"], compute_type)
                self.assertIs(CPUCTranslate2Control.decode, runtime._GPUControl.decode)
                control.decode(object(), [3, 4, 5, 6])
                self.assertTrue(control.replay_verified)
                self.assertEqual(control.metadata()["device"], "cpu")
                args, kwargs = control.model.generate.call_args
                self.assertEqual(args[1], [[50257, 50362, 3, 4, 5, 6]])
                self.assertEqual(kwargs["max_length"], 8)
                with self.assertRaises(ValueError):
                    control.decode(object(), [50256])

    def test_cpu_replay_mismatch_and_precision_fallback_fail(self):
        control, _ = self.make_control(returned=[3, 4, 9])
        with self.assertRaisesRegex(RuntimeError, "forced replay contract failed"):
            control.decode(object(), [3, 4, 5])
        self.assertFalse(control.replay_verified)
        with self.assertRaisesRegex(RuntimeError, "precision fallback"):
            self.make_control(actual_type="float32")


if __name__ == "__main__":
    unittest.main()
