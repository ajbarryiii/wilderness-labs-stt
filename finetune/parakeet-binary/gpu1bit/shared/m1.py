"""M1's runtimes, imported unchanged, on MB1's derived 2-bit export (plans/gpu-kernel.md A1, A11).

Baselines (each a fresh model; never stacked):
- R-compact:  load_packed(derived) + enable_optimizations(model)                      (2-bit, INT8x3)
- R-expanded: load_packed(derived) + enable_optimizations(model, encoder_storage="expanded")
- R-hybrid:   load_packed(derived) + experimental_pipeline(model, "ptx_row_best") applied once (A11)
- R-bf16x3:   kernel level only, inference.kernels.matmul(mode="bf16x3") on a PackedLinear
Per-layer kernels (for numerics and P-kernel): `layer_ops(codes_mb1, scale, bias, K)` returns the
derived PackedLinear plus callables for R-bf16x3 and R-expanded's INT8x3 projection op.
"""
from __future__ import annotations

import contextlib

import torch

from gpu1bit.shared import env


def derived_packed_linear(codes, scale, bias, in_features: int, device="cuda", conv: bool = False):
    """M1's PackedLinear / PackedPointwiseConv1d holding the derived 2-bit codes of an MB1 layer."""
    from inference.runtime import PackedLinear, PackedPointwiseConv1d
    from gpu1bit.shared.derive import derive_codes
    cls = PackedPointwiseConv1d if conv else PackedLinear
    return cls(derive_codes(codes.cpu(), in_features), scale.cpu().float(), in_features,
               None if bias is None else bias.cpu().float()).to(device)


class LayerOps:
    """R-bf16x3 and R-expanded INT8x3 ops for one layer, on the derived 2-bit weights.

    bf16x3(x, conv): inference.kernels.matmul(..., mode="bf16x3") (compact BF16x3 Tensor Core path;
        1-4 rows use its FP32 SIMT path, as M1 does).
    int8x3(x, conv): ExpandedProjection("triton_col_tuned")(x, mod, conv, components=3): INT8x3
        residual activation quantization + expanded INT8 column-major dot + epilogue, exactly the
        op R-expanded runs per projection (without fused norm/activation).
    """

    def __init__(self, codes, scale, bias, in_features: int, device="cuda"):
        from inference.encoder4 import ExpandedProjection
        self.module = derived_packed_linear(codes, scale, bias, in_features, device)
        self.conv_module = derived_packed_linear(codes, scale, bias, in_features, device, conv=True)
        # Share the same packed buffer so both views use one INT8 expansion cache entry.
        self.conv_module.packed_t = self.module.packed_t
        self.projection = ExpandedProjection("triton_col_tuned")
        self.in_features = in_features

    def bf16x3(self, x, conv: bool = False):
        from inference.kernels import matmul
        m = self.module
        return matmul(x, m.packed_t, m.scale, m.bias, m.in_features, conv=conv, mode="bf16x3")

    def int8x3(self, x, conv: bool = False):
        return self.projection(x, self.conv_module if conv else self.module, conv, 3)

    def ops(self) -> dict:
        return {"r_bf16x3": self.bf16x3, "r_expanded_int8x3": self.int8x3}


def load_derived(device="cuda"):
    """M1's load_packed on the derived export (SHA-256 verified by load_packed)."""
    from inference.runtime import load_packed
    return load_packed(env.DERIVED_EXPORT, device)


@contextlib.contextmanager
def baseline(name: str, model, *, max_graphs: int = 4):
    """Apply one baseline's transform stack to a FRESH load_derived() model for the block."""
    from inference.optimized import enable_optimizations, disable_optimizations
    if name == "R-compact":
        enable_optimizations(model, max_graphs=max_graphs)
        try:
            yield None
        finally:
            disable_optimizations(model)
    elif name == "R-expanded":
        enable_optimizations(model, max_graphs=max_graphs, encoder_storage="expanded")
        try:
            yield model._packed_optimization
        finally:
            disable_optimizations(model)
    elif name == "R-hybrid":
        # experimental_pipeline always enables a 4-entry pipeline graph cache (cuda_pipeline.py);
        # encoder-scope measurements capture model.encoder directly and do not use it.
        from inference.cuda_pipeline import experimental_pipeline
        with experimental_pipeline(model, "ptx_row_best") as provider:
            yield provider
    else:
        raise ValueError(f"unknown baseline {name!r}")
