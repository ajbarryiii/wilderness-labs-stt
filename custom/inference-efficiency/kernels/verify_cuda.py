"""Explicit CUDA GEMV regression checks; never silently validate a fallback.

Run with ./inference-efficiency/python -m kernels.verify_cuda. References and
packing run on CPU; GPU work is limited to small kernel correctness checks.
"""

import math
import unittest

import torch

from . import cuda_gemv
from .packed import PackedWeight


class CudaGemvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernels = cuda_gemv.load()
        torch.manual_seed(974)

    def inputs(self, bits, n, k, m=1, *, codes=None, x=None, scale=None, bias=False):
        if codes is None:
            codes = (torch.randint(0, 2, (n, k), dtype=torch.int8) * 2 - 1
                     if bits == 1 else torch.randint(-1, 2, (n, k), dtype=torch.int8))
        if x is None:
            # Dyadic values keep the FP32 sum exact, isolating kernel errors
            # from differing reduction orders in reference matrix libraries.
            x = (torch.randint(-64, 65, (m, k)).float() / 64).half()
        if scale is None:
            scale = torch.linspace(-1.3, 1.3, n) / math.sqrt(k)
        host = PackedWeight.from_codes(codes, bits, scale)
        b = torch.linspace(-0.125, 0.125, n).half() if bias else None
        expected = x.float() @ host.dense().float().T
        if b is not None:
            expected += b.float()
        return (host.to("cuda"), x.cuda(), None if b is None else b.cuda(),
                expected.half())

    def launch(self, packed, x, bias, *, variant=None, out=None):
        m = x.shape[0]
        if out is None:
            out = torch.empty((m, packed.n), device="cuda", dtype=torch.float16)
        kw = packed.words.shape[1]
        if variant is None:
            cuda_gemv.gemv(self.kernels, x, packed.words, packed.scales, bias,
                           out, packed.n, packed.k, kw, packed.bits, m)
        else:
            rows = 128 if variant == "rpl" else 4
            shared = 4 * (packed.k + 128 * (kw | 1)) if variant == "rpl" else 0
            self.kernels[(packed.bits, variant)](
                grid=((packed.n + rows - 1) // rows, m, 1), block=(128, 1, 1),
                args=[x, packed.words, packed.scales, bias if bias is not None else out,
                      out, packed.n, packed.k, kw, int(bias is not None)],
                shared_mem=shared)
        return out

    def test_scale_rounding_on_every_kernel(self):
        for bits, variants in ((1, (0, 1, 2, 4, "rpl")),
                               (2, (0, 1, 2, 4, 8, "rpl"))):
            for variant in variants:
                k = 73 if variant == 0 else 1024 if variant == "rpl" else variant * 1024 // bits
                for bias in (False, True):
                    with self.subTest(bits=bits, variant=variant, bias=bias):
                        x = torch.zeros((1, k), dtype=torch.float16)
                        x[0, 0] = 3
                        args = self.inputs(bits, 5, k, x=x, bias=bias,
                                           codes=torch.ones((5, k), dtype=torch.int8),
                                           scale=torch.tensor([1.0004, -1.0004, 2e-8, 0., .10003]))
                        actual = self.launch(*args[:3], variant=variant)
                        torch.testing.assert_close(actual.cpu(), args[3], atol=0, rtol=0)

    def test_row_per_lane_cancellation(self):
        for bits in (1, 2):
            with self.subTest(bits=bits):
                n, k = 5, 1024
                x = torch.full((1, k), 1 / 1024, dtype=torch.float16)
                if bits == 1:
                    x[0, :2] = 32752
                    codes = torch.ones((n, k), dtype=torch.int8)
                    codes[:, 0] = -1
                else:
                    x[0, 0] = 65504
                    codes = torch.zeros((n, k), dtype=torch.int8)
                args = self.inputs(bits, n, k, x=x, codes=codes, scale=1.)
                expected = (torch.full((1, n), (k - 2) / 1024, dtype=torch.float16)
                            if bits == 1 else torch.zeros((1, n), dtype=torch.float16))
                actual = self.launch(*args[:3], variant="rpl")
                torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)

    def test_variants_tails_and_batches(self):
        for bits in (1, 2):
            # Includes ragged words, non-power-of-two WPL, and large K fallback.
            shapes = [(5, k) for k in (1, 15, 16, 17, 31, 32, 33, 73, 512,
                                       1024, 1536, 2048, 3072, 4096, 4097)]
            shapes += [(129, 1024), (8193, 32), (8193, 2048), (8193, 33)]
            for n, k in shapes:
                for m in (1, 4):
                    with self.subTest(bits=bits, n=n, k=k, m=m):
                        args = self.inputs(bits, n, k, m, bias=m == 4)
                        guarded = torch.full((m * n + 16,), 321., device="cuda", dtype=torch.float16)
                        actual = self.launch(*args[:3], out=guarded[8:-8].reshape(m, n))
                        torch.testing.assert_close(actual.cpu(), args[3], atol=.003, rtol=.003)
                        self.assertTrue(bool((guarded[:8] == 321).all()))
                        self.assertTrue(bool((guarded[-8:] == 321).all()))

    def test_nondefault_stream_and_graph_replay(self):
        for bits in (1, 2):
            for n, k in ((5, 73), (5, 1024), (8193, 32)):
                with self.subTest(bits=bits, n=n, k=k):
                    packed, x, bias, expected = self.inputs(bits, n, k, 3, bias=True)
                    original = x.clone()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        out = self.launch(packed, x, bias)
                    stream.synchronize()
                    torch.testing.assert_close(out.cpu(), expected, atol=.003, rtol=.003)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        self.launch(packed, x, bias, out=out)
                    x.zero_()
                    graph.replay()
                    torch.testing.assert_close(out.cpu(), bias.cpu().expand_as(expected), atol=0, rtol=0)
                    x.copy_(original)
                    graph.replay()
                    torch.testing.assert_close(out.cpu(), expected, atol=.003, rtol=.003)


if __name__ == "__main__":
    with torch.inference_mode():
        unittest.main()
