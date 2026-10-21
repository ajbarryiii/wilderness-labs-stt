"""B-hybrid-x2: B's exact deferred hybrid (hybrid_exact) at two activation parts (plans/x2-default.md).

Stack (Z6): `PartCount(HybridProjectionX2(table, names), 2)` (H's gpu1bit.shared.fulldev.PartCount, which
forwards attribute writes to the provider, X1) inside `exact_hybrid_pipeline_x2(model, provider)`, i.e.
R-hybrid's context stack (hybrid.hybrid_pipeline, frozen) with row_exact_x2's router installed. The
table's layers are B's `int8_mma:x2` variant (layer_parts.Int8MmaPartsLayer, PARTS 2), the same layer
objects as the provider-x2 arm.

HybridProjectionX2 is HybridProjectionExact (hybrid_exact.py) with these 3-to-2 changes, and nothing else:
  * it serves components == 2 only (PartCount sets it on every call, layer_fusions4.py:72 included)
    and raises on anything else, so a missing adapter fails loudly;
  * activations are M1's _quantize at COMPONENTS = 2 (layer_parts.m1_quantize_parts, as the provider
    arm's project), or the cached row-epilogue quantization's parts 0 and 1 (q[:2], qs[:2]; F2);
  * the in-kernel route is the layer's P-part b1_dot (kernels_parts) with the frozen tiles
    (configs.choose), finished by M1's _row_epilogue with SPLIT = 1 when deferred;
  * the XCHG 5 route (hybrid_exact.choose_exact_deferred, the x3 table, F4) is kernels_exact_x2's
    2-part kernel, returned as an ExactDeferredProjectionX2 that row_exact_x2 finishes exactly;
  * fused members called alone use the layer's project(components=2).
Deferral decisions depend on (M, N, K), the weights and the defer flag only, so the route of every call
equals the x3 stack's (X1).

F5: the part-count files are B's hash-pinned B/wip/parts files. `parts()` imports them through
gpu1bit.actprec.parts.load_parts() (verified bytes, loud on any mismatch) and is the first step of
every x2 entry point (HybridProjectionX2.__init__, exact_model_x2, dot_slices_x2, deferred_project_x2).
Importing this module imports no gpu1bit.actprec module.
"""
from __future__ import annotations

import contextlib
import types

import torch

from gpu1bit.int8_mma import configs, cuda, kernels_exact, row_exact_x2
from gpu1bit.int8_mma.hybrid import hybrid_pipeline
from gpu1bit.int8_mma.hybrid_exact import HybridProjectionExact, choose_exact_deferred
from gpu1bit.int8_mma.kernels import DotConfig

PARTS = 2
KERNEL = "int8_mma:x2"
_PARTS: dict = {}


def parts() -> dict:
    """{"kernels_parts", "layer_parts", "parts_run"}: B's pinned part-count modules (F5), loaded once."""
    if not _PARTS:
        import importlib
        from gpu1bit.actprec.parts import load_parts
        load_parts()  # raises ValueError naming the file and both hashes on any mismatch
        for name in ("kernels_parts", "layer_parts", "parts_run"):
            _PARTS[name] = importlib.import_module(f"gpu1bit.int8_mma.{name}")
    return _PARTS


def x2_layer_class():
    return parts()["layer_parts"].IMPLEMENTATIONS["x2"]


def is_x2_layer(layer) -> bool:
    lp = parts()["layer_parts"]
    return isinstance(layer, lp.Int8MmaPartsLayer) and layer.PARTS == PARTS and not layer.B01


def quantize_x2(xx, norm=None, input_activation: str = ""):
    """M1's _quantize at COMPONENTS = 2 (exactly the provider arm's project quantization)."""
    return parts()["layer_parts"].m1_quantize_parts(xx, PARTS, norm, input_activation)


def dot_slices_x2(layer, q: torch.Tensor, qs: torch.Tensor, ldq: int, m: int, slices: torch.Tensor,
                  config: DotConfig, order: int = 0) -> None:
    """slices int32 [C, 2, m, N] <- packed P-part K-slice sums (P = q.shape[0] in (2, 3)) of b1_dot XCHG 5."""
    from gpu1bit.int8_mma import kernels_exact_x2
    parts()
    p = q.shape[0]
    n, k = layer.out_features, layer.in_features
    cfg = kernels_exact.ExactDeferredConfig.of(config)
    if p not in kernels_exact_x2.PARTS or qs.shape[0] != p:
        raise ValueError(f"q / qs must hold 2 or 3 parts, got {tuple(q.shape)} / {tuple(qs.shape)}")
    if not layer.fast:
        raise ValueError("exact deferred slices need the fragment layout (layer.fast)")
    if 127 * k >= kernels_exact.PACK_LIMIT:
        raise ValueError("packed int64 lanes would overflow for this K")
    if ldq % 16 or q.stride(1) != ldq or q.stride(2) != 1 or q.data_ptr() % 16:
        raise ValueError("fast path needs q [P, M, ldq] with 16-byte aligned rows")
    if qs.stride(1) != 1:
        raise ValueError("qs rows must be contiguous")
    if slices.dtype != torch.int32 or tuple(slices.shape) != (cfg.cluster, 2, m, n) or not slices.is_contiguous():
        raise ValueError(f"slices must be contiguous int32 [{cfg.cluster}, 2, {m}, {n}]")
    source, name = kernels_exact_x2.dot_source(cfg, order, p)
    cuda.load(source, name, kernels_exact_x2.shared_bytes(cfg, p)).launch(
        cfg.grid(m, n), (cfg.threads, 1, 1),
        [cuda.ptr(q), cuda.i64(q.stride(0)), cuda.i32(ldq), cuda.i32(-(-k // 16) * 16), cuda.ptr(qs),
         cuda.i32(qs.stride(0)), cuda.ptr(layer.words), cuda.i32(layer.words.shape[0]),
         cuda.i32(layer.words.shape[1]), cuda.ptr(layer.scale), cuda.ptr(layer.bias), cuda.ptr(slices),
         cuda.i32(n), cuda.i32(m), cuda.i32(n), cuda.ptr(None)])


class HybridProjectionX2(HybridProjectionExact):
    """HybridProjectionExact at two activation parts (module docstring). Use inside PartCount(·, 2)."""

    PARTS = PARTS

    def __init__(self, table, names=None):
        parts()  # F5: first step, before any table or CUDA work
        bad = [e.name for e in table.entries if not is_x2_layer(e.layer)]
        if bad:
            raise ValueError(f"HybridProjectionX2 needs {KERNEL} layers (layer_parts, PARTS 2); not: {bad[:3]}")
        super().__init__(table, names)
        row_exact_x2.install()

    def __call__(self, x, mod, conv=False, components=3, activation="", norm=None, config=None, input_activation=""):
        from inference.row_fusion import DeferredProjection
        if components != PARTS:
            raise ValueError(f"HybridProjectionX2 serves INT8x2 calls only (components={components!r}); "
                             "wrap it in gpu1bit.shared.fulldev.PartCount(provider, 2)")
        if config is not None:
            raise ValueError("HybridProjectionX2 serves M1's calls without tile configs")
        entry, offset, count = self._resolver._lookup(mod)
        layer = entry.layer
        if count != entry.out_features:  # a fused member called alone (never inside the pipeline)
            y = layer.project(x, conv=conv, components=PARTS, activation=activation, norm=norm,
                              input_activation=input_activation)
            return y[:, offset:offset + count] if conv else y[..., offset:offset + count]
        xx = x.transpose(1, 2) if conv else x
        if xx.dim() == 2:
            xx = xx.unsqueeze(0)
        b, t, k = xx.shape
        m, n = b * t, layer.out_features
        cached = self.prequantized.pop((xx.data_ptr(), m, k, id(norm)), None) if norm is not None else None
        if cached is not None:  # M1's row epilogue quantized 3 parts; parts 0, 1 are its 2-part output (F2)
            q, qs = cached
            if q.shape[0] < PARTS or qs.shape[0] < PARTS:
                raise ValueError(f"cached activation has {q.shape[0]} parts")
            q, qs = q[:PARTS], qs[:PARTS]
            self.calls["prequantized"] += 1
        else:
            q, qs = quantize_x2(xx, norm, input_activation)
            self.calls["quantized"] += 1
        deferred = (self.defer_epilogue and n == 1024 and mod.packed_t.data_ptr() not in self.no_defer_weights
                    and activation == "")
        if deferred:
            self.calls["deferred"] += 1
            cfg = choose_exact_deferred(m, n, k) if self.split_deferred and layer.fast and m > 0 else None
            if cfg is not None:
                slices = torch.empty(cfg.cluster, 2, m, n, device=xx.device, dtype=torch.int32)
                dot_slices_x2(layer, q, qs, k, m, slices, cfg)
                self.calls["exact_slices"] = self.calls.get("exact_slices", 0) + 1
                return row_exact_x2.ExactDeferredProjectionX2(slices, layer.scale, layer.bias, cfg.cluster, (b, t, n),
                                                              qs, 0, PARTS)
        y = torch.empty(b, t, n, device=xx.device, dtype=torch.float32)
        layer.dot(q, qs, k, m, y.view(m, n), silu=activation == "silu", config=configs.choose(m, n, k), parts=PARTS)
        if deferred:  # exact in-kernel finish; M1's _row_epilogue with SPLIT = 1 adds residual, norm, quantization
            return DeferredProjection(y, layer.scale, layer.bias, 1, (b, t, n))
        return y.transpose(1, 2) if conv else y.reshape(*x.shape[:-1], n)


def check_write_forwarding(provider, inner) -> None:
    """X1: a write to the wrapper must reach the inner provider (R-hybrid's contexts set defer_epilogue
    and no_defer_weights on the object they are given). Raises if it does not."""
    if provider is inner:
        return
    before = inner.defer_epilogue
    provider.defer_epilogue = not before
    reached = inner.defer_epilogue is (not before)
    provider.defer_epilogue = before
    if not reached or inner.defer_epilogue is not before:
        raise RuntimeError(f"{type(provider).__name__} does not forward attribute writes to the provider "
                           "(plans/x2-default.md X1); the hybrid stack would never defer")


def unwrap(provider):
    """The innermost provider of a chain of .provider / .inner wrappers."""
    seen = set()
    while id(provider) not in seen:
        seen.add(id(provider))
        nxt = provider.__dict__.get("provider") or provider.__dict__.get("inner")
        if nxt is None:
            break
        provider = nxt
    return provider


@contextlib.contextmanager
def exact_hybrid_pipeline_x2(model, provider):
    """hybrid.hybrid_pipeline (R-hybrid's contexts, frozen) with row_exact_x2's router; `provider` is
    PartCount(HybridProjectionX2(...), 2) (possibly with transparent probes in between)."""
    inner = unwrap(provider)
    if not isinstance(inner, HybridProjectionX2):
        raise TypeError(f"expected a HybridProjectionX2 inside the adapter, got {type(inner).__name__}")
    check_write_forwarding(provider, inner)
    row_exact_x2.install()
    with hybrid_pipeline(model, provider):
        yield provider


@contextlib.contextmanager
def exact_model_x2(max_graphs: int = 4, split_deferred: bool = True, probe=None, partcount=None):
    """(model, info): MB1 on int8_mma:x2 layers, PartCount(probe(HybridProjectionX2), 2), the x2 pipeline.
    `probe(inner)` may wrap the provider transparently (tests); `partcount` overrides the adapter class
    (default gpu1bit.shared.fulldev.PartCount)."""
    parts()  # F5: first step
    import evaluate as ev
    from gpu1bit.shared import env, provider as P, registry
    if max_graphs != 4:
        raise ValueError("R-hybrid's pipeline fixes max_graphs=4")
    if partcount is None:
        from gpu1bit.shared.fulldev import PartCount as partcount
    impl = registry.get(KERNEL)
    model, table = P.load_binary_model(env.MB1_EXPORT, impl)
    with ev.inference_settings(model):
        model.decoding.decoding.decoding_computer.force_cuda_graphs_mode("full_graph")
        inner = HybridProjectionX2(table, P.module_names(model))
        inner.split_deferred = split_deferred
        wrapped = probe(inner) if probe is not None else inner
        pc = partcount(wrapped, PARTS)
        with exact_hybrid_pipeline_x2(model, pc):
            yield model, {"kind": "candidate", "parts": PARTS, "kernel": KERNEL,
                          "stack": "experimental_pipeline(ptx_row_best) contexts + PartCount(HybridProjectionX2, 2) "
                                   "(exact integer deferred slices at 2 parts)",
                          "projection": pc, "inner": inner, "partcount": pc, "weight_bytes": table.weight_bytes()}


def deferred_project_x2(layer, x: torch.Tensor, *, conv: bool = False, exact: bool = True,
                        config: DotConfig | None = None, norm=None, input_activation: str = "") -> torch.Tensor:
    """The x2 deferred path of one projection as a plain op (unit tests, sanitizer): 2-part quantize, then
    XCHG 5 + _row_epilogue_exact_parts (exact=True and a split config) or the in-kernel x2 b1_dot + M1's
    _row_epilogue SPLIT = 1, with a zero residual, factor 1 and no norms (y = the projection value)."""
    parts()
    from inference.row_fusion import DeferredProjection
    from gpu1bit.int8_mma.row_exact import _ORIGINAL
    row_exact_x2.install()
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
    q, qs = quantize_x2(xx, norm, input_activation)
    cfg = None
    if exact:
        cfg = choose_exact_deferred(m, n, k) if config is None else kernels_exact.ExactDeferredConfig.of(config)
    if cfg is not None:
        slices = torch.empty(cfg.cluster, 2, m, n, device=layer.device, dtype=torch.int32)
        dot_slices_x2(layer, q, qs, k, m, slices, cfg)
        y = row_exact_x2.finish_residual_exact_x2(
            provider, row_exact_x2.ExactDeferredProjectionX2(slices, layer.scale, layer.bias, cfg.cluster, (b, t, n), qs, 0,
                                                             PARTS), residual, 1.0)
    else:
        value = torch.empty(b, t, n, device=layer.device, dtype=torch.float32)
        layer.dot(q, qs, k, m, value.view(m, n), config=configs.choose(m, n, k), parts=PARTS)
        original = _ORIGINAL
        if original is None:
            raise RuntimeError("row_exact's router is not installed")
        y = original(provider, DeferredProjection(value, layer.scale, layer.bias, 1, (b, t, n)), residual, 1.0)
    return y.transpose(1, 2) if conv else y.view(*x.shape[:-1], n)
