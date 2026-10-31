"""Contract tests (CUDA, inside a job with a lease): the stage 4 activation quantizer against the i8x1 kernel's actual
codes and scales (keab-prime plan "Code and review": `_quantize` of finetune/parakeet-ternary/inference/
fused_kernels.py with COMPONENTS = 1, called as the x1 runtime calls it: gpu1bit measured-x2-parts/layer_parts.py
m1_quantize_parts).

- the training quantizer's codes and scales equal the kernel's, bit for bit, on random, heavy-tailed, zero, tiny,
  large and rounding-tie inputs, at the model's channel counts (1024, 4096) and an odd one, for linear [B, T, K] and
  pointwise-convolution [B, C, T] layouts (a strided view, as the runtime quantizes x.transpose(1, 2)), from FP32 and
  from BF16 inputs (quantized in FP32);
- x̂ is q s computed in FP32, returned in the input's dtype;
- the straight-through gradient on CUDA;
- a binary module's forward under BF16 autocast quantizes its input with the kernel's grid.
The count of codes where Triton's division and an IEEE formula disagree is printed (information: why the training
quantizer runs the kernel's arithmetic rather than a PyTorch formula).
"""
from __future__ import annotations

import unittest

import torch

import train  # noqa: F401  finetune/parakeet-ternary first
from training import actquant

HAS_CUDA = torch.cuda.is_available()


def kernel_codes(xx: torch.Tensor):
    """`_quantize` at COMPONENTS = 1 on a [B, T, K] FP32 view (any strides): (q int8 [B T, K], s FP32 [B T])."""
    actquant._triton_libcuda()
    import triton
    from inference.fused_kernels import _quantize
    b, t, k = xx.shape
    m = b * t
    q = torch.empty(1, m, k, device=xx.device, dtype=torch.int8)
    qs = torch.empty(1, m, device=xx.device, dtype=torch.float32)
    _quantize[(m,)](xx, q, qs, None, None, m, k, t, *xx.stride(), triton.next_power_of_2(k), 1, False, 0.,
                    INPUT_SILU=False, enable_fp_fusion=False)
    return q[0], qs[0]


def inputs(k: int, device: str = "cuda") -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(k)
    x = {"normal": torch.randn(3, 41, k, generator=g),
         "heavy": torch.randn(2, 37, k, generator=g) ** 3 * torch.logspace(-4, 4, k)[torch.randperm(k, generator=g)],
         "large": torch.randn(2, 5, k, generator=g) * 1e30,
         "tiny": torch.randn(2, 5, k, generator=g) * 1e-33}
    zeros = torch.randn(2, 6, k, generator=g)
    zeros[0, 2] = 0
    zeros[1, :3] = 0
    x["zero-rows"] = zeros
    # ties: s = 2^e exactly (max |x| = 127 2^e), every other value an odd multiple of 2^(e-1): x / s = n + 0.5
    ties = torch.randint(-126, 126, (2, 9, k), generator=g).float() + 0.5
    ties[..., 0] = 127.0
    x["ties"] = ties * 2.0 ** torch.randint(-20, 20, (2, 9, 1), generator=g).float()
    return {n: v.to(device) for n, v in x.items()}


@unittest.skipUnless(HAS_CUDA, "the kernel comparison needs CUDA")
class AgainstTheKernel(unittest.TestCase):
    def check(self, x: torch.Tensor, dim: int, label: str) -> int:
        y, q, s = actquant.quantize(x, dim)
        xx = x.float().transpose(1, 2) if dim == actquant.CONV_DIM else x.float()
        kq, ks = kernel_codes(xx if xx.dim() == 3 else xx.unsqueeze(0))         # a 2-D input is one batch row
        self.assertTrue(torch.equal(q, kq), f"{label}: codes differ in {int((q != kq).sum())} places")
        self.assertTrue(torch.equal(s, ks), f"{label}: scales differ in {int((s != ks).sum())} frames")
        want = (kq.float() * ks[:, None]).to(x.dtype)
        shape = xx.shape
        want = want.reshape(shape).transpose(1, 2) if dim == actquant.CONV_DIM else want.reshape(x.shape)
        self.assertEqual(y.dtype, x.dtype)
        self.assertTrue(torch.equal(y, want), f"{label}: x_hat is not q s in FP32 cast to {x.dtype}")
        _, rq, rs = actquant.reference(x.cpu(), dim)
        return int((rq.to(q.device) != q).sum()) + int((rs.to(s.device) != s).sum())

    def test_linear_layout_fp32_and_bf16(self):
        ieee = 0
        for k in (1024, 4096, 176):
            for name, x in inputs(k).items():
                ieee += self.check(x, actquant.LINEAR_DIM, f"linear K={k} {name} fp32")
                ieee += self.check(x.to(torch.bfloat16), actquant.LINEAR_DIM, f"linear K={k} {name} bf16")
        print(f"\ncodes or scales where an IEEE formula differs from the kernel (linear): {ieee}")

    def test_conv_layout_is_per_frame_across_channels(self):
        ieee = 0
        for k in (1024, 176):
            for name, x in inputs(k).items():
                conv = x.transpose(1, 2).contiguous()                         # [B, C, T], the module's input
                ieee += self.check(conv, actquant.CONV_DIM, f"conv C={k} {name} fp32")
                ieee += self.check(conv.to(torch.bfloat16), actquant.CONV_DIM, f"conv C={k} {name} bf16")
        print(f"\ncodes or scales where an IEEE formula differs from the kernel (conv): {ieee}")

    def test_two_dimensional_and_strided_inputs(self):
        x = inputs(1024)["normal"]
        self.check(x[0], actquant.LINEAR_DIM, "2-D")
        self.check(x[:, ::2], actquant.LINEAR_DIM, "strided frames")
        self.check(x.transpose(0, 1), actquant.LINEAR_DIM, "transposed batch")

    def test_the_straight_through_gradient_on_cuda(self):
        for dtype in (torch.float32, torch.bfloat16):
            x = (torch.randn(2, 9, 1024, device="cuda") * 5).to(dtype).requires_grad_(True)
            w = torch.randn_like(x)
            (actquant.fake_quant(x) * w).sum().backward()
            self.assertEqual(x.grad.dtype, dtype)
            self.assertTrue(torch.equal(x.grad, w))

    def test_a_binary_module_under_autocast(self):
        import binary_quant
        lin = torch.nn.Linear(1024, 512).cuda()
        mod = binary_quant.BinaryLinear.from_linear(lin, "ternary")
        seen = {}

        class Tap(actquant.Hook):
            def __call__(self, module, inputs):
                out = super().__call__(module, inputs)
                seen.update(x=inputs[0].detach(), y=out[0].detach())
                return out

        mod.register_forward_pre_hook(Tap("tap", actquant.LINEAR_DIM))
        x = torch.randn(2, 30, 1024, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mod(x.to(torch.bfloat16))
        self.assertEqual(seen["y"].dtype, torch.bfloat16)
        kq, ks = kernel_codes(seen["x"].float())
        self.assertTrue(torch.equal(seen["y"], (kq.float() * ks[:, None]).to(torch.bfloat16).reshape(seen["x"].shape)))


if __name__ == "__main__":
    unittest.main()
