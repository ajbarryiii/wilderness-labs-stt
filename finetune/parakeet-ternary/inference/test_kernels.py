"""GPU numerical, layout, stream, and graph regression checks.

Run with ../python -m unittest inference.test_kernels from parakeet-ternary.
"""
import unittest

import torch
import torch.nn.functional as F

from export import pack_codes
from .kernels import matmul
from .runtime import PackedLinear, PackedPointwiseConv1d


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from paths import gpu_lock
        cls.lock = gpu_lock("packed kernel tests")
        cls.lock.__enter__()
        torch.manual_seed(417)
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        cls.lock.__exit__(None, None, None)

    def make(self, n, k, conv=False, bias=True):
        codes = torch.randint(-1, 2, (n, k), dtype=torch.int8)
        codes[0] = 0
        scale = torch.rand(n) * .04
        b = torch.randn(n) * .01 if bias else None
        cls = PackedPointwiseConv1d if conv else PackedLinear
        layer = cls(pack_codes(codes), scale, k, b).cuda()
        return layer, (codes.float() * scale[:, None]).cuda()

    @torch.inference_mode()
    def test_shapes_precision_and_split_k(self):
        for n, k, m in [(17, 37, 5), (1, 1, 1), (1024, 1024, 1), (2048, 1024, 65),
                         (4096, 1024, 129), (1024, 4096, 33)]:
            layer, w = self.make(n, k)
            x = torch.randn(m, k, device="cuda")
            ref = F.linear(x, w, layer.bias)
            for mode in ("tf32x3", "bf16x3"):
                for split in (1, 4):
                    with self.subTest(n=n, k=k, m=m, mode=mode, split=split):
                        y = matmul(x, layer.packed_t, layer.scale, layer.bias, k, mode=mode,
                                   config=(32, 64, 64, split, 4, 3))
                        torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)
                        torch.testing.assert_close(y[:, 0], layer.bias[0].expand(m), atol=0, rtol=0)

    @torch.inference_mode()
    def test_convolution_and_strided_linear(self):
        layer, w = self.make(97, 65, conv=True)
        for x in (torch.randn(2, 65, 39, device="cuda"),
                  torch.randn(2, 78, 65, device="cuda")[:, ::2].transpose(1, 2)):
            ref = F.conv1d(x, w[:, :, None], layer.bias)
            for mode in ("tf32x3", "bf16x3"):
                layer.mode = mode
                torch.testing.assert_close(layer(x), ref, atol=1e-5, rtol=1e-5)
                y = matmul(x.transpose(1, 2), layer.packed_t, layer.scale, layer.bias, 65, mode=mode)
                torch.testing.assert_close(y.transpose(1, 2), ref, atol=1e-5, rtol=1e-5)

    @torch.inference_mode()
    def test_graph_replay_changed_input_and_nondefault_stream(self):
        layer, w = self.make(1024, 1024, bias=False)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            x = torch.randn(31, 1024, device="cuda")
            for mode in ("tf32x3", "bf16x3"):
                def run():
                    return matmul(x, layer.packed_t, layer.scale, None, 1024, mode=mode,
                                  config=(32, 64, 64, 4, 4, 3))
                run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    y = run()
                for factor in (.25, -2):
                    x.mul_(factor)
                    graph.replay()
                    torch.testing.assert_close(y, F.linear(x, w), atol=1e-5, rtol=1e-5)
        torch.cuda.current_stream().wait_stream(stream)

    @torch.inference_mode()
    def test_output_reuse_empty_vector_and_fp16(self):
        layer, w = self.make(33, 67, bias=False)
        for shape in [(67,), (2, 3, 4, 67), (0, 67), (2, 0, 67)]:
            x = torch.randn(shape, device="cuda")
            out = torch.empty((*shape[:-1], 33), device="cuda")
            y = matmul(x, layer.packed_t, layer.scale, None, 67, out=out)
            self.assertIs(y, out)
            torch.testing.assert_close(y, F.linear(x, w), atol=1e-5, rtol=1e-5)
        x = torch.randn(17, 67, device="cuda", dtype=torch.float16)
        y = matmul(x, layer.packed_t, layer.scale, None, 67, mode="fp16")
        torch.testing.assert_close(y, F.linear(x.float(), w).half(), atol=3e-4, rtol=1e-3)

    @torch.inference_mode()
    def test_dynamic_range_and_cancellation(self):
        layer, w = self.make(64, 1024, bias=False)
        for magnitude in (1e-20, 1e20):
            x = torch.randn(7, 1024, device="cuda") * magnitude
            for mode in ("tf32x3", "bf16x3"):
                y = matmul(x, layer.packed_t, layer.scale, None, 1024, mode=mode)
                ref = F.linear(x.double(), w.double()).float()
                torch.testing.assert_close(y / magnitude, ref / magnitude, atol=1e-5, rtol=1e-5)
        # Adjacent products cancel exactly; low components must not leak into zeros.
        codes = torch.ones(64, 1024, dtype=torch.int8)
        codes[:, 1::2] = -1
        layer = PackedLinear(pack_codes(codes), torch.ones(64), 1024).cuda()
        x = torch.randn(7, 512, device="cuda").repeat_interleave(2, dim=-1)
        for mode in ("tf32x3", "bf16x3"):
            y = matmul(x, layer.packed_t, layer.scale, None, 1024, mode=mode)
            torch.testing.assert_close(y, torch.zeros_like(y), atol=1e-5, rtol=0)

    def test_validation(self):
        with self.assertRaises(ValueError):
            PackedLinear(torch.tensor([[3]], dtype=torch.uint8), torch.ones(1), 4)
        with self.assertRaises(ValueError):
            PackedLinear(torch.tensor([[4]], dtype=torch.uint8), torch.ones(1), 1)
        layer, _ = self.make(8, 16)
        with self.assertRaises(RuntimeError):
            layer(torch.randn(2, 16, device="cuda", requires_grad=True))
        with self.assertRaises(ValueError):
            layer(torch.randn(2, 15, device="cuda"))
        square, _ = self.make(16, 16)
        x = torch.randn(2, 16, device="cuda")
        with self.assertRaises(ValueError):
            matmul(x, square.packed_t, square.scale, square.bias, 16, out=x)
        with self.assertRaises(ValueError):
            layer.half()(torch.randn(2, 16, device="cuda"))

    @torch.inference_mode()
    def test_gemv_dispatch(self):
        for n, k in [(17, 37), (1024, 4096), (1024, 2057)]:
            layer, w = self.make(n, k)
            for m in (1, 2, 4):
                x = torch.randn(m, k, device="cuda")
                torch.testing.assert_close(layer(x), F.linear(x, w, layer.bias), atol=1e-5, rtol=1e-5)

    @torch.inference_mode()
    def test_encoder_graph_cache(self):
        from .graphs import enable_encoder_graphs, disable_encoder_graphs

        class Encoder(torch.nn.Module):
            def forward(self, audio_signal, length=None):
                return audio_signal.sin() + length[:, None, None].float(), length + 1

        model = torch.nn.Module()
        model.encoder = Encoder().eval()
        original = model.encoder.forward
        cache = enable_encoder_graphs(model, max_graphs=1)
        x = torch.randn(2, 3, 17, device="cuda")
        lengths = torch.tensor([17, 12], device="cuda")
        a = model.encoder(x, lengths)
        old = a[0].clone()
        x.mul_(-2)
        lengths.sub_(3)
        b = model.encoder(x, lengths)
        torch.testing.assert_close(b, original(x, lengths))
        torch.testing.assert_close(a[0], old)
        self.assertEqual(len(cache.entries), 1)
        # A new shape runs eagerly once the cache is full.
        c = x[:, :, :9]
        torch.testing.assert_close(model.encoder(c, lengths), original(c, lengths))
        self.assertEqual(len(cache.entries), 1)
        disable_encoder_graphs(model)
        self.assertEqual(model.encoder.forward, original)
        self.assertFalse(cache.entries)


if __name__ == "__main__":
    unittest.main()
