"""The activation quantizer of stages 4-5 (keab-prime plan "Training", "Activation quantization"; program.md, "Activation
precision target"): fake quantization of the input of every binary module onto the i8x1 kernels' per-frame INT8 grid.

The grid is the x1 runtime's quantizer, `_quantize` of finetune/parakeet-ternary/inference/fused_kernels.py at
COMPONENTS = 1 (gpu1bit's int8_mma:x1 calls it so: measured-x2-parts/layer_parts.m1_quantize_parts). Per frame, across
the input channels: s = max(max|x| / 127, 1e-30), q = round-half-even(x / s) as int8, x̂ = q · s.

  - On CUDA the codes and scales are computed by _x1_kernel below, which is `_quantize`'s arithmetic statement for
    statement (load, |x|, max, "/ 127.0", the 1e-30 floor, "x / scale", libdevice nearbyint, the int8 cast; launched
    with enable_fp_fusion=False as the runtime launches it), with the frame count and strides as runtime arguments
    instead of compile-time constants (`_quantize` specializes on M and T, so it would recompile for every batch
    shape). Triton's FP32 division is not IEEE's, so a PyTorch formula would not reproduce the kernel's codes at
    every near-tie; the CUDA contract test (tests/test_keab_actquant_gpu.py) checks _x1_kernel's codes and scales
    against `_quantize` itself.
  - On the CPU (the tests' tiny models) reference(): the same formula in PyTorch with IEEE division.
  - The input is read as the module receives it, in whatever dtype (BF16 under autocast, FP32 where NeMo's attention
    or LayerNorm give FP32), and quantized in FP32 (x.float()); x̂ = q · s is computed in FP32 and returned in the
    input's dtype.
  - Gradient: straight-through, the identity to the input (s is not differentiated).

Axis (CHANNEL_DIM): a linear module's input [..., K] is quantized per frame across its last axis; a pointwise
convolution's input [B, C, T] per (b, t) across C (the runtime quantizes x.transpose(1, 2)).

Boundary (install): one forward pre-hook (Hook) on every binary module (binary_quant.BINARY_TYPES; 264 in Parakeet)
replaces its input by x̂. That is the i8x1 kernels' quantization boundary, because the module's input is what the
kernel quantizes: the LayerNorm output for the modules whose norm the kernel fuses (feed_forward1/2.linear1,
conv.pointwise_conv1, self_attn.linear_q), the post-SiLU input of feed_forward linear2 and conv.pointwise_conv2, the
shared normalized input of linear_q, linear_k and linear_v (each quantized identically), the positional embeddings for
linear_pos, the attention context for linear_out. Nothing else is quantized: not encoder.pre_encode.out (its W8A8
option is measured in step 0, not trained), not the prediction or joint networks. Hooks are not in state_dict; a
checkpoint loads into a model with or without them. ramp: none, on from the first step (no fraction).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn

FLOOR = 1e-30
QMAX = 127
LINEAR_DIM, CONV_DIM = -1, 1
IDENTITY = {"kind": "x1 per-frame INT8 fake quantization (keab-prime plan)",
            "grid": "inference.fused_kernels._quantize, COMPONENTS = 1: s = max(max|x| / 127, 1e-30), "
                    "q = nearbyint(x / s), x_hat = q s, per frame across input channels",
            "compute": "FP32 from the module's input, output in the input's dtype", "gradient": "straight-through",
            "where": "the input of every binary module (forward pre-hook)", "ramp": "none (on from step 0)"}
_ATTR = "_keab_actquant"


# --- the quantizer ---------------------------------------------------------------------------------

_KERNEL = {}


def _triton_libcuda() -> None:
    """Triton looks libcuda up with /sbin/ldconfig, which some systems lack: unless TRITON_LIBCUDA_PATH is set, point
    it at the first LD_LIBRARY_PATH directory holding libcuda.so.1 (as train_b0prime.py does)."""
    import os
    if os.environ.get("TRITON_LIBCUDA_PATH"):
        return
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if d and os.path.exists(os.path.join(d, "libcuda.so.1")):
            os.environ["TRITON_LIBCUDA_PATH"] = d
            return


def _kernel():
    if "k" not in _KERNEL:
        _triton_libcuda()
        import triton
        import triton.language as tl

        @triton.jit
        def _x1_kernel(X, Q, S, Y, T, XB, XT, XK, K: tl.constexpr, BLOCK: tl.constexpr, STORE_Q: tl.constexpr):
            m = tl.program_id(0)
            k = tl.arange(0, BLOCK)
            x = tl.load(X + (m // T) * XB + (m % T) * XT + k * XK, k < K, other=0)
            scale = tl.maximum(tl.max(tl.abs(x), 0) / 127.0, 1.e-30)
            q = tl.extra.cuda.libdevice.nearbyint(x / scale).to(tl.int8)
            if STORE_Q:
                tl.store(Q + m * K + k, q, k < K)
            tl.store(S + m, scale)
            y = q.to(tl.float32) * scale
            tl.store(Y + m * K + k, y.to(Y.dtype.element_ty), k < K)

        _KERNEL["k"] = (_x1_kernel, triton.next_power_of_2)
    return _KERNEL["k"]


def _frames(x32: Tensor, dim: int) -> tuple[Tensor, int, int, tuple[int, int, int], tuple]:
    """(the [B, T, K] view, M = B T, K, (XB, XT, XK) strides, the output's [B, T, K] shape) of x32 quantized across
    `dim` (the last axis, or axis 1 of a [B, C, T] input)."""
    if dim in (-1, x32.dim() - 1):
        k = x32.shape[-1]
        xx = x32.reshape(-1, k) if x32.dim() != 2 else x32
        m = xx.shape[0]
        return xx, m, k, (0, xx.stride(0), xx.stride(1)), (1, m, k)
    if dim == CONV_DIM and x32.dim() == 3:
        b, c, t = x32.shape
        xx = x32.transpose(1, 2)                                 # [B, T, C], strides (C T, 1, T) for a contiguous x
        return xx, b * t, c, (xx.stride(0), xx.stride(1), xx.stride(2)), (b, t, c)
    raise ValueError(f"quantization across axis {dim} of a {x32.dim()}-D input is not a binary module's layout")


def quantize(x: Tensor, dim: int = LINEAR_DIM, codes: bool = True) -> tuple[Tensor, Tensor | None, Tensor]:
    """(x̂ in x's dtype and shape, q int8 [M, K] or None, s FP32 [M]) of x quantized per frame across `dim`
    (module docstring); M frames in [B, T] order (the runtime's m = b T + t), K channels. Not differentiable."""
    with torch.no_grad():
        x32 = x.detach().to(torch.float32)
        if x32.numel() == 0:
            raise ValueError("cannot quantize an empty input")
        if not (dim in (-1, x32.dim() - 1) or (dim == CONV_DIM and x32.dim() == 3)):
            raise ValueError(f"quantization across axis {dim} of a {x32.dim()}-D input is not a binary module's layout")
        if x32.is_cuda:
            return _quantize_cuda(x32, dim, x.dtype, codes)
        return reference(x, dim)


def _quantize_cuda(x32: Tensor, dim: int, dtype: torch.dtype, codes: bool):
    kernel, pow2 = _kernel()
    xx, m, k, (xb, xt, xk), shape = _frames(x32, dim)
    t = shape[1]
    y = torch.empty((m, k), dtype=dtype, device=x32.device)
    s = torch.empty(m, dtype=torch.float32, device=x32.device)
    q = torch.empty((m, k), dtype=torch.int8, device=x32.device) if codes else s
    kernel[(m,)](xx, q, s, y, t, xb, xt, xk, k, pow2(k), bool(codes), enable_fp_fusion=False)
    out = y.view(shape)
    out = out.transpose(1, 2) if dim == CONV_DIM and x32.dim() == 3 else out.view(x32.shape)
    return out, (q if codes else None), s


def reference(x: Tensor, dim: int = LINEAR_DIM) -> tuple[Tensor, Tensor, Tensor]:
    """quantize()'s formula in PyTorch with IEEE division (the CPU path; on CUDA a reference only)."""
    with torch.no_grad():
        x32 = x.detach().to(torch.float32)
        moved = x32.movedim(dim, -1)
        amax = moved.abs().amax(dim=-1)
        s = (amax / torch.full_like(amax, float(QMAX))).clamp_min(FLOOR)
        q = torch.round(moved / s.unsqueeze(-1))                  # round half to even
        y = (q * s.unsqueeze(-1)).movedim(-1, dim).to(x.dtype)
        return y, q.reshape(-1, q.shape[-1]).to(torch.int8), s.reshape(-1)


class _FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, dim: int) -> Tensor:
        return quantize(x, dim, codes=False)[0]

    @staticmethod
    def backward(ctx, grad: Tensor):
        return grad, None


def fake_quant(x: Tensor, dim: int = LINEAR_DIM) -> Tensor:
    """x̂ (module docstring) with the straight-through gradient."""
    return _FakeQuant.apply(x, dim)


# --- the boundary ----------------------------------------------------------------------------------

class Hook:
    """The forward pre-hook of one binary module: its first positional input replaced by fake_quant(x, dim)."""

    def __init__(self, name: str, dim: int):
        self.name, self.dim, self.calls = name, dim, 0

    def __call__(self, module: nn.Module, inputs: tuple):
        if not inputs or not isinstance(inputs[0], Tensor):
            raise RuntimeError(f"{self.name}: a binary module called without a positional input tensor")
        self.calls += 1
        return (fake_quant(inputs[0], self.dim),) + tuple(inputs[1:])


def channel_dim(module: nn.Module) -> int:
    import binary_quant
    if isinstance(module, binary_quant.BinaryPointwiseConv1d):
        return CONV_DIM
    if isinstance(module, binary_quant.BinaryLinear):
        return LINEAR_DIM
    raise ValueError(f"{type(module).__name__} is not a binary module")


def install(model: nn.Module, expected: int | None = None) -> list[str]:
    """One Hook on the input of every binary module (module docstring); returns their names. Refuses a model that
    already has one, no binary module, or (with `expected`) another count."""
    import binary_quant
    if hooks(model):
        raise ValueError("the model already has activation quantizers")
    names = binary_quant.quantized_module_names(model)
    if not names or (expected is not None and len(names) != expected):
        raise ValueError(f"{len(names)} binary modules, expected {expected or 'some'}")
    handles = []
    for name in names:
        module = model.get_submodule(name)
        handles.append(module.register_forward_pre_hook(Hook(name, channel_dim(module))))
    object.__setattr__(model, _ATTR, {"names": names, "handles": handles})
    return names


def hooks(model: nn.Module) -> dict[str, list[Hook]]:
    """{module name: its activation-quantizer hooks} over every module of the model (empty lists left out)."""
    out = {}
    for name, module in model.named_modules():
        found = [h for h in module._forward_pre_hooks.values() if isinstance(h, Hook)]
        if found:
            out[name] = found
    return out


def verify(model: nn.Module, expected: int | None = None) -> dict:
    """The boundary check: exactly one Hook on every binary module and none elsewhere (ValueError otherwise)."""
    import binary_quant
    names = binary_quant.quantized_module_names(model)
    found = hooks(model)
    wrong = sorted(n for n in names if len(found.get(n, [])) != 1)
    stray = sorted(n for n in found if n not in names)
    if wrong or stray or not names or (expected is not None and len(names) != expected):
        raise ValueError(f"activation quantizers: {len(names)} binary modules, missing or repeated on {wrong[:3]}, "
                         f"stray on {stray[:3]}")
    return {"modules": len(names), "names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
            "conv_modules": sum(found[n][0].dim == CONV_DIM for n in names), **identity()}


def identity() -> dict:
    """IDENTITY plus this file's SHA-256 (the kernel's source is part of it)."""
    return {**IDENTITY, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
