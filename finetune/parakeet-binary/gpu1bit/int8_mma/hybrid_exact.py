"""B-hybrid with an exact deferred split-K (review c1 finding 1); session_patch.py serves it as "B-hybrid"
and "B-hybrid-b01" in the re-run selection, deferred.py exposes its provider to H's deferred A3.

hybrid.py's B-hybrid (frozen) hands R-hybrid's row epilogue FP32 split partials for the deferred
N = 1024 calls (XCHG 4), which can lose a small residual before a cancellation across slices.
`HybridProjectionExact` is HybridProjection with every deferred call made exact:
  * split deferred calls (choose_exact_deferred(M, N, K) is a config): b1_dot XCHG 5 writes packed
    INT32x3 slices as int32 word planes [C, 2, M, N] (kernels_exact.py) and returns an ExactDeferredProjection, which
    row_exact._row_epilogue_exact finishes in integers before M1's SPLIT == 1 epilogue;
  * otherwise (None): b1_dot's exact in-kernel finish and a SPLIT = 1 DeferredProjection, as in B-hybrid.
Either way the value entering the residual / LayerNorm / quantization is bitwise b1_dot's SPLIT == 1
result, so the encoder output equals that of HybridProjection(split_deferred=False) bit for bit (tested).
This provider never returns FP32 split slices. `exact_hybrid_pipeline` is hybrid.hybrid_pipeline with
row_exact's finish_residual routing installed. The b01 form serves Int8MmaB01Layer tables with the
b01 XCHG 5 kernel.

`deferred_project` is the deferred provider as one call (quantize, dot, row epilogue with a zero
residual, factor 1, no norms) for the A3 qualification of the deferred path (H).
"""
from __future__ import annotations

import contextlib
import math
import types

import torch

from gpu1bit.int8_mma import configs, cuda, kernels_exact
from gpu1bit.int8_mma.hybrid import HybridProjection, hybrid_pipeline
from gpu1bit.int8_mma.kernels import DotConfig
from gpu1bit.int8_mma.layer import Int8MmaLayer, m1_quantize

# Deferred-split configs for XCHG 5 per (N, K, M); None keeps the exact in-kernel finish (SPLIT = 1).
# tune_exact.py (B/tune/20261020T1319*-exact-*): dot + _row_epilogue_exact (LayerNorm + INT8x3 of the next
# input), every candidate bitwise against the exact in-kernel path first; a config is used only where
# it beats "exact in-kernel b1_dot + M1's SPLIT = 1 _row_epilogue" by at least 1%. us per call, hot.
TUNED_EXACT_DEFERRED: dict[tuple[int, int, int], DotConfig | None] = {
    (1024, 1024, 38): DotConfig(1, 8, 1, 1, 4, 3, 5, 1, 0),   # w1x8_r1x1_s4_c3_e 4.83 vs exact 4.90 (frozen FP32 4.02)
    (1024, 1024, 126): None,                                  # best 6.23 vs exact 5.83 (frozen FP32 5.54)
    (1024, 1024, 256): None,                                  # best 8.81 vs exact 7.40 (frozen FP32 7.32)
    (1024, 1024, 368): None,                                  # best 11.83 vs exact 9.13 (frozen: exact too)
    (1024, 4096, 38): DotConfig(1, 8, 1, 1, 2, 6, 5, 1, 0),   # w1x8_r1x1_s2_c6_e 7.02 vs exact 7.17 (frozen FP32 6.13)
    (1024, 4096, 126): DotConfig(1, 4, 1, 4, 2, 4, 5, 1, 0),  # w1x4_r1x4_s2_c4_e 10.76 vs exact 11.10 (frozen FP32 9.83)
    (1024, 4096, 256): None,                                  # best 17.82 vs exact 17.37 (frozen FP32 16.11)
    (1024, 4096, 368): None,                                  # best 24.83 vs exact 23.63 (frozen FP32 22.07)
}


def choose_exact_deferred(m: int, n: int, k: int) -> kernels_exact.ExactDeferredConfig | None:
    """The XCHG 5 config for (M, N, K) (nearest tuned M in log scale), or None (in-kernel finish)."""
    if 127 * k >= kernels_exact.PACK_LIMIT:
        return None
    tuned = [mm for (nn, kk, mm) in TUNED_EXACT_DEFERRED if (nn, kk) == (n, k)]
    if not tuned or m < 1:
        return None
    nearest = min(tuned, key=lambda mm: (abs(math.log(m / mm)), mm))
    cfg = TUNED_EXACT_DEFERRED[n, k, nearest]
    return None if cfg is None else kernels_exact.ExactDeferredConfig.of(cfg)


def is_b01(layer) -> bool:
    from gpu1bit.int8_mma.layer_b01 import Int8MmaB01Layer
    return isinstance(layer, Int8MmaB01Layer) or bool(getattr(layer, "B01", False))


def dot_slices(layer: Int8MmaLayer, q: torch.Tensor, qs: torch.Tensor, ldq: int, m: int, slices: torch.Tensor,
               config: DotConfig, order: int = 0) -> None:
    """slices int32 [C, 2, m, N] <- packed INT32x3 K-slice sums of b1_dot XCHG 5 (b01 kernel for b01 layers)."""
    n, k = layer.out_features, layer.in_features
    cfg = kernels_exact.ExactDeferredConfig.of(config)
    if not layer.fast:
        raise ValueError("exact deferred slices need the fragment layout (layer.fast)")
    if 127 * k >= kernels_exact.PACK_LIMIT:
        raise ValueError("packed int64 lanes would overflow for this K")
    if ldq % 16 or q.stride(1) != ldq or q.stride(2) != 1 or q.data_ptr() % 16 or q.shape[0] != 3:
        raise ValueError("fast path needs q [3, M, ldq] with 16-byte aligned rows")
    if slices.dtype != torch.int32 or tuple(slices.shape) != (cfg.cluster, 2, m, n) or not slices.is_contiguous():
        raise ValueError(f"slices must be contiguous int32 [{cfg.cluster}, 2, {m}, {n}]")
    b01 = is_b01(layer)
    source, name = kernels_exact.dot_source(cfg, order, b01)
    cuda.load(source, name, kernels_exact.shared_bytes(cfg, b01)).launch(
        cfg.grid(m, n), (cfg.threads, 1, 1),
        [cuda.ptr(q), cuda.i64(q.stride(0)), cuda.i32(ldq), cuda.i32(-(-k // 16) * 16), cuda.ptr(qs),
         cuda.i32(qs.stride(0)), cuda.ptr(layer.words), cuda.i32(layer.words.shape[0]),
         cuda.i32(layer.words.shape[1]), cuda.ptr(layer.scale), cuda.ptr(layer.bias), cuda.ptr(slices),
         cuda.i32(n), cuda.i32(m), cuda.i32(n), cuda.ptr(None)])


class HybridProjectionExact(HybridProjection):
    """HybridProjection whose deferred calls are exact (module docstring). split_deferred=False keeps
    every deferred call on the in-kernel finish (then identical to HybridProjection's False mode)."""

    def __call__(self, x, mod, conv=False, components=3, activation="", norm=None, config=None, input_activation=""):
        from inference.row_fusion import DeferredProjection
        from gpu1bit.int8_mma.row_exact import ExactDeferredProjection
        if components != 3 or config is not None:
            raise ValueError("HybridProjectionExact serves M1's INT8x3 calls without tile configs")
        entry, offset, count = self._resolver._lookup(mod)
        layer = entry.layer
        if count != entry.out_features:  # a fused member called alone (never inside the pipeline)
            y = layer.project(x, conv=conv, components=3, activation=activation, norm=norm,
                              input_activation=input_activation)
            return y[:, offset:offset + count] if conv else y[..., offset:offset + count]
        xx = x.transpose(1, 2) if conv else x
        if xx.dim() == 2:
            xx = xx.unsqueeze(0)
        b, t, k = xx.shape
        m, n = b * t, layer.out_features
        cached = self.prequantized.pop((xx.data_ptr(), m, k, id(norm)), None) if norm is not None else None
        if cached is not None:
            q, qs = cached
            self.calls["prequantized"] += 1
        else:
            q, qs = m1_quantize(xx, norm, input_activation)
            self.calls["quantized"] += 1
        deferred = (self.defer_epilogue and n == 1024 and mod.packed_t.data_ptr() not in self.no_defer_weights
                    and activation == "")
        if deferred:
            self.calls["deferred"] += 1
            cfg = choose_exact_deferred(m, n, k) if self.split_deferred and layer.fast and m > 0 else None
            if cfg is not None:
                slices = torch.empty(cfg.cluster, 2, m, n, device=xx.device, dtype=torch.int32)
                dot_slices(layer, q, qs, k, m, slices, cfg)
                self.calls["exact_slices"] = self.calls.get("exact_slices", 0) + 1
                return ExactDeferredProjection(slices, layer.scale, layer.bias, cfg.cluster, (b, t, n), qs, 0)
        y = torch.empty(b, t, n, device=xx.device, dtype=torch.float32)
        layer.dot(q, qs, k, m, y.view(m, n), silu=activation == "silu", config=configs.choose(m, n, k))
        if deferred:  # exact in-kernel finish; _row_epilogue with SPLIT = 1 adds residual, norm, quantization
            return DeferredProjection(y, layer.scale, layer.bias, 1, (b, t, n))
        return y.transpose(1, 2) if conv else y.reshape(*x.shape[:-1], n)


@contextlib.contextmanager
def exact_hybrid_pipeline(model, provider):
    """hybrid.hybrid_pipeline (R-hybrid's contexts) with row_exact's finish_residual routing installed."""
    from gpu1bit.int8_mma import row_exact
    row_exact.install()
    with hybrid_pipeline(model, provider):
        yield provider


@contextlib.contextmanager
def exact_model(layer_cls=Int8MmaLayer, max_graphs: int = 4, split_deferred: bool = True):
    """(model, info) for P-paired: MB1 through load_binary_model(layer_cls) + exact_hybrid_pipeline."""
    import evaluate as ev
    from gpu1bit.shared import env, provider as P
    if max_graphs != 4:
        raise ValueError("R-hybrid's pipeline fixes max_graphs=4 (as gpu1bit.shared.m1.baseline)")
    model, table = P.load_binary_model(env.MB1_EXPORT, layer_cls)
    with ev.inference_settings(model):
        model.decoding.decoding.decoding_computer.force_cuda_graphs_mode("full_graph")
        provider = HybridProjectionExact(table, P.module_names(model))
        provider.split_deferred = split_deferred
        with exact_hybrid_pipeline(model, provider):
            yield model, {"kind": "candidate",
                          "stack": "experimental_pipeline(ptx_row_best) contexts + HybridProjectionExact (exact integer deferred slices)",
                          "projection": provider, "weight_bytes": table.weight_bytes()}


def deferred_project(layer: Int8MmaLayer, x: torch.Tensor, *, conv: bool = False, exact: bool = True,
                     config: DotConfig | None = None, norm=None, input_activation: str = "") -> torch.Tensor:
    """The deferred provider path of one projection as a plain op (for A3 on the deferred path).

    x: [B, T, K] (conv=False) or [B, K, T] (conv=True), as api.call passes it; returns y in the same
    layout as layer.linear / layer.conv. M1's _quantize (optionally fused LayerNorm / input SiLU),
    then:
      exact=True:  XCHG 5 integer slices + row_exact._row_epilogue_exact (config: an XCHG 5 / any split
                   config, default choose_exact_deferred; None there -> exact in-kernel finish +
                   M1's _row_epilogue with SPLIT = 1, exactly as HybridProjectionExact does);
      exact=False: frozen B-hybrid: XCHG 4 FP32 slices (config default configs.choose_deferred) + M1's
                   _row_epilogue summing them in FP32 (or the in-kernel finish where that is None).
    The row epilogue runs with a zero residual, factor 1 and no norms, so y is the projection value."""
    from inference.row_fusion import DeferredProjection, finish_residual
    from gpu1bit.int8_mma.row_exact import ExactDeferredProjection, finish_residual_exact
    xx = x.transpose(1, 2) if conv else x
    if xx.dim() == 2:
        xx = xx.unsqueeze(0)
    b, t, k = xx.shape
    m, n = b * t, layer.out_features
    provider = types.SimpleNamespace(row_warps="adaptive", prequantized={},
                                     null=torch.empty(0, device=layer.device, dtype=torch.uint8))
    residual = torch.zeros(b, t, n, device=layer.device, dtype=torch.float32)
    if m == 0:
        return residual.transpose(1, 2) if conv else residual.view(*x.shape[:-1], n)
    q, qs = m1_quantize(xx, norm, input_activation)
    if exact:
        cfg = choose_exact_deferred(m, n, k) if config is None else kernels_exact.ExactDeferredConfig.of(config)
        if cfg is not None:
            slices = torch.empty(cfg.cluster, 2, m, n, device=layer.device, dtype=torch.int32)
            dot_slices(layer, q, qs, k, m, slices, cfg)
            y = finish_residual_exact(provider, ExactDeferredProjection(slices, layer.scale, layer.bias, cfg.cluster,
                                                                        (b, t, n), qs, 0), residual, 1.0)
            return y.transpose(1, 2) if conv else y.view(*x.shape[:-1], n)
    else:
        cfg = configs.choose_deferred(m, n, k) if config is None else config
        if cfg is not None and cfg.cluster > 1:
            partial = torch.empty(cfg.cluster, m, n, device=layer.device, dtype=torch.float32)
            layer.dot(q, qs, k, m, partial[0], config=cfg)
            y = finish_residual(provider, DeferredProjection(partial, layer.scale, layer.bias, cfg.cluster, (b, t, n)),
                                residual, 1.0)
            return y.transpose(1, 2) if conv else y.view(*x.shape[:-1], n)
    value = torch.empty(b, t, n, device=layer.device, dtype=torch.float32)
    layer.dot(q, qs, k, m, value.view(m, n), config=configs.choose(m, n, k))
    y = finish_residual(provider, DeferredProjection(value, layer.scale, layer.bias, 1, (b, t, n)), residual, 1.0)
    return y.transpose(1, 2) if conv else y.view(*x.shape[:-1], n)
