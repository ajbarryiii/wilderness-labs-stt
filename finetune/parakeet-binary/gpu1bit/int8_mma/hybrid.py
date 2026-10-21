"""R-hybrid's pipeline served from B's 1-bit layers ("B-hybrid"; plans/gpu-kernel.md A2, runtime vs runtime).

M1's fastest runtime, R-hybrid = experimental_pipeline(model, "ptx_row_best") (inference/cuda_pipeline.py),
stacks encoder_transform(int8x3, qkv, silu, position, fold_bn, norm, attention, norm_quant,
position_dot) + layer_fusions(residual=False) + row_fusions + the fused decoder and the pipeline
graph cache, with a PtxCudaProjection provider that (a) returns DeferredProjection objects for the
N = 1024, no-activation projections so row_fusion's _row_epilogue folds the split-K finish, the
residual add, the next LayerNorm and the next INT8x3 quantization into one kernel, and (b) consumes
those prequantized activations (provider.prequantized) instead of re-quantizing.

`HybridProjection` implements that same provider protocol (defer_epilogue, prequantized,
no_defer_weights, row_warps, null) over the BinaryLayer table of gpu1bit.shared.provider
.load_binary_model, and `hybrid_pipeline` applies exactly R-hybrid's context stack with it. Deferred
calls run b1_dot's in-kernel split-K finish and hand row_fusion a SPLIT = 1 DeferredProjection (the
finished projection; _row_epilogue then adds the residual, normalizes and quantizes). Activation
quantization of the remaining calls is M1's own _quantize, so q/qs equal R-hybrid's.
"""
from __future__ import annotations

import contextlib

import torch

from gpu1bit.int8_mma import configs
from gpu1bit.int8_mma.layer import m1_quantize


class HybridProjection:
    def __init__(self, table, names=None):
        from gpu1bit.shared.provider import BinaryProjection
        self._resolver = BinaryProjection(table, names)  # handle -> (entry, offset, count) lookup only
        self.table = table
        self.defer_epilogue = False
        self.prequantized: dict = {}
        self.no_defer_weights: set = set()
        self.row_warps = "adaptive"
        self.null = torch.empty(0, device="cuda", dtype=torch.uint8)
        if self.null.data_ptr() != 0:
            raise RuntimeError("expected a null empty-tensor pointer")
        self.calls = {"deferred": 0, "prequantized": 0, "quantized": 0}
        # Deferred N = 1024 calls: True hands _row_epilogue FP32 split slices (configs.TUNED_DEFERRED,
        # R-hybrid's own split-K arithmetic); False keeps b1_dot's exact in-kernel finish (SPLIT = 1).
        self.split_deferred = True

    @property
    def bytes(self) -> int:
        return self.table.weight_bytes()

    def __call__(self, x, mod, conv=False, components=3, activation="", norm=None, config=None, input_activation=""):
        from inference.row_fusion import DeferredProjection
        if components != 3 or config is not None:
            raise ValueError("HybridProjection serves M1's INT8x3 calls without tile configs")
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
            cfg = configs.choose_deferred(m, n, k) if self.split_deferred else None
            if cfg is not None and cfg.cluster > 1:  # S FP32 slices; _row_epilogue sums them (R-hybrid's split-K)
                partial = torch.empty(cfg.cluster, m, n, device=xx.device, dtype=torch.float32)
                layer.dot(q, qs, k, m, partial[0], config=cfg)
                return DeferredProjection(partial, layer.scale, layer.bias, cfg.cluster, (b, t, n))
        y = torch.empty(b, t, n, device=xx.device, dtype=torch.float32)
        layer.dot(q, qs, k, m, y.view(m, n), silu=activation == "silu", config=configs.choose(m, n, k))
        if deferred:  # exact in-kernel finish; _row_epilogue with SPLIT = 1 adds residual, norm, quantization
            return DeferredProjection(y, layer.scale, layer.bias, 1, (b, t, n))
        return y.transpose(1, 2) if conv else y.reshape(*x.shape[:-1], n)


@contextlib.contextmanager
def hybrid_pipeline(model, provider):
    """inference.cuda_pipeline.experimental_pipeline(model, "ptx_row_best") with `provider`: the same
    contexts in the same order (encoder_provider's encoder_transform / layer_fusions(residual=False)
    / row_fusions inside decoder_graph_pool, then decoder and control transforms, then graphs)."""
    from inference.decoder_kernels import control_transform, decoder_transform
    from inference.graph_memory import decoder_graph_pool
    from inference.graphs import disable_pipeline_graphs, enable_pipeline_graphs
    from inference.layer_fusions4 import layer_fusions
    from inference.row_fusion import row_fusions
    from inference.transforms import encoder_transform
    with decoder_graph_pool(model), \
            encoder_transform(model, mode="int8x3", qkv=True, silu=True, position=True, fold_bn=True, norm=True,
                              attention=True, norm_quant=True, position_dot=True, projection=provider), \
            layer_fusions(model, provider, residual=False), row_fusions(model, provider), \
            decoder_transform(model, joint=True, lstm=True, precompute=True, block=1), \
            control_transform(model, storage=True):
        enable_pipeline_graphs(model, max_graphs=4)
        try:
            yield provider
        finally:
            disable_pipeline_graphs(model)


@contextlib.contextmanager
def b_hybrid_model(max_graphs: int = 4):
    """(model, info) for P-paired: MB1 through load_binary_model(Int8MmaLayer) + hybrid_pipeline."""
    import evaluate as ev
    from gpu1bit.int8_mma.layer import Int8MmaLayer
    from gpu1bit.shared import env, provider as P
    if max_graphs != 4:
        raise ValueError("R-hybrid's pipeline fixes max_graphs=4 (as gpu1bit.shared.m1.baseline)")
    model, table = P.load_binary_model(env.MB1_EXPORT, Int8MmaLayer)
    with ev.inference_settings(model):
        model.decoding.decoding.decoding_computer.force_cuda_graphs_mode("full_graph")
        provider = HybridProjection(table, P.module_names(model))
        with hybrid_pipeline(model, provider):
            yield model, {"kind": "candidate", "stack": "experimental_pipeline(ptx_row_best) contexts + HybridProjection",
                          "projection": provider, "weight_bytes": table.weight_bytes()}
