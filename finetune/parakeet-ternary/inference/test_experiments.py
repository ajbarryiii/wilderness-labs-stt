"""Numerical checks for the new arithmetic and fusion experiments."""
import unittest

import torch
import torch.nn.functional as F

from export import pack_codes
from .kernels import matmul
from .runtime import PackedLinear


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import paths
        cls.lock = paths.gpu_lock("experimental kernel tests")
        cls.lock.__enter__()
        torch.manual_seed(570)
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        cls.lock.__exit__(None, None, None)

    @torch.inference_mode()
    def test_arithmetic_and_silu(self):
        for n, k, m in [(33, 79, 3), (127, 1024, 31), (64, 4096, 129)]:
            codes = torch.randint(-1, 2, (n, k), dtype=torch.int8)
            codes[0].zero_()
            scales = torch.rand(n) * .03
            layer = PackedLinear(pack_codes(codes), scales, k, torch.randn(n)*.01).cuda()
            w = (codes.float()*scales[:, None]).cuda()
            for mode in ["bf16x3", "bf16x2", "fp16x2"]:
                for magnitude in ([1., 1e-20, 1e20] if mode != "fp16x2" else [1e-4, 1., 100.]):
                    x = torch.randn(m, k, device="cuda")*magnitude
                    # Compare normalized output to avoid absolute tolerances
                    # hiding failures at very small magnitudes.
                    ref = F.linear(x, w)
                    y = matmul(x, layer.packed_t, layer.scale, None, k, mode=mode)
                    torch.testing.assert_close(y/magnitude, ref/magnitude, atol=2e-5, rtol=2e-5)
                x = torch.randn(m, k, device="cuda")
                for split in [1, 4]:
                    y = matmul(x, layer.packed_t, layer.scale, layer.bias, k, mode=mode,
                               activation="silu", config=(32, 64, 64, split, 4, 2))
                    torch.testing.assert_close(y, F.silu(F.linear(x, w, layer.bias)), atol=2e-5, rtol=2e-5)

    @torch.inference_mode()
    def test_layer_norm(self):
        from .fused_kernels import layer_norm
        norm = torch.nn.LayerNorm(1024).cuda()
        norm.weight.uniform_(-2, 2); norm.bias.uniform_(-1, 1)
        for x in [torch.randn(2, 37, 1024, device="cuda"), torch.randn(2, 1024, 17, device="cuda").transpose(1, 2)]:
            torch.testing.assert_close(layer_norm(x, norm), norm(x), atol=2e-6, rtol=2e-5)

    @torch.inference_mode()
    def test_relative_attention(self):
        from nemo.collections.asr.parts.submodules.multi_head_attention import RelPositionMultiHeadAttention
        from .fused_kernels import relative_attention
        att = RelPositionMultiHeadAttention(2, 128, 0., None, None).cuda().eval()
        att.pos_bias_u.normal_(); att.pos_bias_v.normal_()
        for t in [17, 65, 129]:
            x = torch.randn(2, t, 128, device="cuda")
            pos = torch.randn(1, 2*t-1, 128, device="cuda")
            for masked in [False, True]:
                mask = None
                if masked:
                    mask = torch.zeros(2, t, t, dtype=torch.bool, device="cuda")
                    mask[1, :, t//2:] = True
                    mask[1, t//2:, :] = True  # includes all-masked rows
                ref = att(x, x, x, mask, pos)
                for position_dot in [False, True]:
                    y = relative_attention(att, x, x, x, mask, pos, position_dot=position_dot)
                    torch.testing.assert_close(y, ref, atol=3e-6, rtol=2e-5)

    @torch.inference_mode()
    def test_int8_matches_quantized_reference(self):
        from .fused_kernels import int8_matmul
        codes = torch.randint(-1, 2, (97, 129), dtype=torch.int8)
        layer = PackedLinear(pack_codes(codes), torch.rand(97)*.1, 129, torch.randn(97)*.01).cuda()
        x = torch.randn(2, 37, 129, device="cuda"); x[0, 0].zero_()
        scale = x.abs().amax(-1, keepdim=True).div(127).clamp_min(1e-30)
        qx = (x/scale).round()*scale
        ref = F.linear(qx, codes.cuda().float()*layer.scale[:, None], layer.bias)
        torch.testing.assert_close(int8_matmul(x, layer), ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(int8_matmul(x.transpose(1, 2), layer, conv=True).transpose(1, 2), ref,
                                   atol=1e-5, rtol=1e-5)
        for components, tol in [(2, 1e-4), (3, 2e-6)]:
            y = int8_matmul(x, layer, components=components)
            dense = F.linear(x, codes.cuda().float()*layer.scale[:, None], layer.bias)
            torch.testing.assert_close(y, dense, atol=tol, rtol=tol)
        norm = torch.nn.LayerNorm(129).cuda()
        norm.weight.uniform_(-2, 2); norm.bias.uniform_(-1, 1)
        ref = F.silu(F.linear(norm(x), codes.cuda().float()*layer.scale[:, None], layer.bias))
        y = int8_matmul(x, layer, components=3, norm=norm, activation="silu")
        torch.testing.assert_close(y, ref, atol=3e-6, rtol=2e-5)
        y = int8_matmul(x, layer, components=3, norm=norm, activation="silu", config=(32,64,128,1,4,2))
        torch.testing.assert_close(y, ref, atol=3e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
