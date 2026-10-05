"""Optional integration test against the local trained export.

PARAKEET_KERNEL_EXPORT=/mnt/hd/.../export ../python -m unittest inference.test_model_graphs -v
"""
import os
import unittest

import torch


@unittest.skipUnless(os.environ.get("PARAKEET_KERNEL_EXPORT"), "set PARAKEET_KERNEL_EXPORT to run model integration")
class ModelGraphTests(unittest.TestCase):
    @torch.inference_mode()
    def test_optimized_pipeline_multiple_shapes_and_restore(self):
        import evaluate as ev
        import paths
        from .runtime import load_packed
        from .optimized import enable_optimizations, disable_optimizations
        from nemo.utils import logging
        logging.setLevel(logging.ERROR)
        with paths.gpu_lock("optimized model integration"), ev.strict_fp32("cuda"):
            model = load_packed(os.environ["PARAKEET_KERNEL_EXPORT"])
            with ev.inference_settings(model):
                original_keys = set(model.state_dict())
                dw = model.encoder.layers[0].conv.depthwise_conv
                original_weight = dw.weight.clone()
                audio = torch.randn(2, 48000, device="cuda")*.05
                length = torch.tensor([48000, 31000], device="cuda")
                ref = model(input_signal=audio, input_signal_length=length)
                enable_optimizations(model, max_graphs=2)
                cache = model.forward
                out = model(input_signal=audio, input_signal_length=length)
                torch.testing.assert_close(out, ref, atol=3e-6, rtol=3e-5)
                saved = tuple(t.clone() for t in out)
                for samples in [64000, 72000]:
                    x = torch.randn(2, samples, device="cuda")*.04
                    lens = torch.tensor([samples, samples-17000], device="cuda")
                    actual = model(input_signal=x, input_signal_length=lens)
                    eager = cache.model_forward(input_signal=x, input_signal_length=lens)
                    torch.testing.assert_close(actual, eager, atol=3e-6, rtol=3e-5)
                # Revisit first graph after another shape changes positional buffers.
                audio.mul_(.7).add_(.001); length.sub_(2000)
                actual = model(input_signal=audio, input_signal_length=length)
                eager = cache.model_forward(input_signal=audio, input_signal_length=length)
                torch.testing.assert_close(actual, eager, atol=3e-6, rtol=3e-5)
                torch.testing.assert_close(out, saved, atol=0, rtol=0)
                self.assertEqual(len(cache.entries), 2)
                with self.assertRaises(RuntimeError):
                    model.encoder(audio_signal=torch.randn(2,128,128,device="cuda"), length=torch.tensor([128,90],device="cuda"), bypass_pre_encode=True)
                with torch.inference_mode(False):
                    disable_optimizations(model)
                self.assertEqual(set(model.state_dict()), original_keys)
                torch.testing.assert_close(dw.weight, original_weight, atol=0, rtol=0)
                self.assertIsNone(dw.bias)
                restored = model(input_signal=audio, input_signal_length=length)
                torch.testing.assert_close(restored, actual, atol=3e-6, rtol=3e-5)

    @torch.inference_mode()
    def test_new_features_lengths_and_owned_outputs(self):
        import evaluate as ev
        import paths
        from .runtime import load_packed
        from .graphs import enable_encoder_graphs, disable_encoder_graphs
        from nemo.utils import logging

        logging.setLevel(logging.ERROR)
        with paths.gpu_lock("packed model graph integration"), ev.strict_fp32("cuda"):
            model = load_packed(os.environ["PARAKEET_KERNEL_EXPORT"])
            cache = enable_encoder_graphs(model, max_graphs=1)
            torch.manual_seed(418)
            features = torch.randn(2, 128, 384, device="cuda")
            lengths = torch.tensor([384, 270], device="cuda")
            first = model.encoder(audio_signal=features, length=lengths)
            saved = tuple(t.clone() for t in first)
            features.mul_(.7).add_(.3)
            lengths.sub_(37)
            second = model.encoder(audio_signal=features, length=lengths)
            eager = cache.original_forward(audio_signal=features, length=lengths)
            torch.testing.assert_close(second, eager, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(first, saved, atol=0, rtol=0)
            self.assertEqual(len(cache.entries), 1)
            disable_encoder_graphs(model)


if __name__ == "__main__":
    unittest.main()
