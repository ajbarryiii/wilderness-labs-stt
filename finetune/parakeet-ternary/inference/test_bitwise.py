"""Checks for the isolated FP32 LUT/SIMT experiments."""
import unittest

import torch

from export import pack_codes
from .runtime import PackedLinear
from .bitwise import linear


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class BitwiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from paths import gpu_lock
        cls.lock = gpu_lock("bitwise kernel tests")
        cls.lock.__enter__()
        torch.manual_seed(420)
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        cls.lock.__exit__(None, None, None)

    @torch.inference_mode()
    def test_ragged_bias_zero_rows_and_model_shapes(self):
        for m, n, k in [(3, 17, 37), (64, 1024, 1024), (1, 1024, 4096)]:
            c = torch.randint(-1, 2, (n, k), dtype=torch.int8)
            c[0] = 0
            scale, bias = torch.rand(n) * .1, torch.randn(n)
            layer = PackedLinear(pack_codes(c), scale, k, bias).cuda()
            x = torch.randn(m, k, device="cuda")
            ref = torch.nn.functional.linear(x, (c.float()*scale[:, None]).cuda(), bias.cuda())
            for variant in ("lookup", "simt"):
                y = linear(x, layer, variant=variant)
                torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(y[:, 0], layer.bias[0].expand(m), atol=0, rtol=0)

    @torch.inference_mode()
    def test_graph_replay_uses_changed_inputs(self):
        from .benchmark import capture
        c = torch.randint(-1, 2, (97, 129), dtype=torch.int8)
        layer = PackedLinear(pack_codes(c), torch.ones(97), 129).cuda()
        x = torch.randn(7, 129, device="cuda")
        for variant in ("lookup", "simt"):
            graph, (y,) = capture(lambda: (linear(x, layer, variant=variant),))
            x.mul_(-.25)
            graph.replay()
            torch.testing.assert_close(y, torch.nn.functional.linear(x, c.float().cuda()), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
