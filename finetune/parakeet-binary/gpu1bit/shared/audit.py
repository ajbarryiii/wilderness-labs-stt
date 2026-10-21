"""A5 storage audit: unique GPU storage attributable to the 264 quantized layers' weights across a
whole transformed model (plans/gpu-kernel.md A5). Authoritative budget check:

    weight_code_bytes <= floor(1.10 * 75,497,472) = 83,047,219 bytes (MB1's exact packed code bytes)

Method. Every CUDA tensor reachable from the roots is found by walking the Python object graph
(module __dict__s incl. patched forward functions, closures, default arguments, bound methods,
functools.partial, containers, dataclasses, ExitStack callbacks and suspended context-manager
generator frames, so fused-QKV copies, retained layouts, provider caches and transform keepalives
are all seen). Storages are deduplicated by untyped_storage().data_ptr() and assigned to the first
category whose roots reach them, in this order:
  dense       parameters/buffers of every non-quantized module (the FP32 rest of NeMo)
  weights     parameters/buffers of the 264 quantized modules themselves (claimed before the graph
              cache, whose keepalive holds encoder.buffers())
  dense       decoder-side transform contexts (decoder_transform, control_transform,
              decoder_graph_pool: e.g. the precomputed token table), fold_bn originals and the
              fold_bn temporaries left in encoder_transform's frame (alpha, weight, bias)
  activation  graph-cache entries (static inputs/outputs), positional-projection caches, provider
              prequantized activations: activation-derived, excluded, reported
  weights     everything else reachable from the quantized modules, the projection provider, the
              patched functions of encoder modules, the encoder-side transform contexts, and the
              module globals of candidate packages (gpu1bit.<dir>.*), so a module-level cache counts
Within `weights`, a storage whose entire contents equal a layer's FP32 scale vector (or a
concatenation of the Q/K/V scales), or its bias, is reported as `scale_bytes` / `bias_bytes`
(excluded from the budget); every other weights storage counts as `weight_code_bytes` (codes in any
layout, padding, auxiliaries such as row popcounts, pre-multiplied scales, duplicates).
The report lists every weights storage with its size, dtype, shape and first path.
"""
from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import inspect
import types

import torch
from torch import nn

from gpu1bit.shared import api

MB1_CODE_BYTES = 75_497_472
CODE_BUDGET_BYTES = (MB1_CODE_BYTES * 11) // 10   # 83,047,219


def _children(obj):
    """(label, child) pairs of a Python object, excluding module globals and code objects."""
    if isinstance(obj, torch.Tensor):
        return []
    if isinstance(obj, (str, bytes, int, float, bool, type(None), type, types.ModuleType, types.CodeType,
                        torch.device, torch.dtype)):
        return []
    if isinstance(obj, dict):
        return [(f"[{k!r}]", v) for k, v in list(obj.items())]
    if isinstance(obj, (list, tuple, set, frozenset, collections.deque)):
        return [(f"[{i}]", v) for i, v in enumerate(list(obj))]
    if isinstance(obj, types.FunctionType):
        out = []
        if obj.__closure__:
            out += [(f".<closure {n}>", c.cell_contents) for n, c in zip(obj.__code__.co_freevars, obj.__closure__)
                    if _cell_ok(c)]
        if obj.__defaults__:
            out += [(f".<default {i}>", v) for i, v in enumerate(obj.__defaults__)]
        if obj.__kwdefaults__:
            out += [(f".<kwdefault {k}>", v) for k, v in obj.__kwdefaults__.items()]
        out += [(f".{k}", v) for k, v in obj.__dict__.items()]
        return out
    if isinstance(obj, types.MethodType):
        return [(".__self__", obj.__self__), (".__func__", obj.__func__)]
    if isinstance(obj, functools.partial):
        return [(".func", obj.func), (".args", obj.args), (".keywords", obj.keywords)]
    if isinstance(obj, types.GeneratorType):
        frame = obj.gi_frame
        return [] if frame is None else [(f".<local {k}>", v) for k, v in frame.f_locals.items()]
    if isinstance(obj, contextlib._GeneratorContextManager):
        return [(".gen", obj.gen)]
    if isinstance(obj, contextlib.ExitStack):
        return [(".<callback>", cb) for cb in list(obj._exit_callbacks)]
    if isinstance(obj, torch.cuda.CUDAGraph):
        return []
    out = []
    if hasattr(obj, "__dict__"):
        out += [(f".{k}", v) for k, v in vars(obj).items()]
    for slot in getattr(type(obj), "__slots__", ()) or ():
        if isinstance(slot, str) and hasattr(obj, slot):
            out.append((f".{slot}", getattr(obj, slot)))
    return out


def _cell_ok(cell) -> bool:
    try:
        cell.cell_contents
        return True
    except ValueError:
        return False


class _Walker:
    def __init__(self, stop_modules: set[int]):
        self.stop_modules = stop_modules   # nn.Modules not to descend into from this walk
        self.storages: dict[int, dict] = {}

    def walk(self, roots: list[tuple[str, object]], category: str, owned: dict[int, dict]):
        seen = set()
        stack = list(roots)
        while stack:
            path, obj = stack.pop()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if isinstance(obj, torch.Tensor):
                if obj.is_cuda and obj.layout == torch.strided:
                    storage = obj.untyped_storage()
                    ptr = storage.data_ptr()
                    if ptr and ptr not in owned:
                        owned[ptr] = {"category": category, "bytes": storage.nbytes(), "path": path,
                                      "dtype": str(obj.dtype), "shape": list(obj.shape), "tensor": obj}
                continue
            if isinstance(obj, nn.Module) and id(obj) in self.stop_modules:
                continue
            for label, child in _children(obj):
                stack.append((path + label, child))


def audit(model: nn.Module, quantized_names: list[str], extra_roots: dict[str, object] | None = None,
          scales: dict[str, torch.Tensor] | None = None, biases: dict[str, torch.Tensor] | None = None) -> dict:
    """The A5 report for a transformed model. `scales`/`biases`: {layer name: CPU FP32 [N]} used to
    recognise scale/bias storages by content (Q/K/V concatenations are recognised too)."""
    torch.cuda.synchronize()
    modules = dict(model.named_modules())
    quantized = {modules[n] for n in quantized_names}
    quantized_ids = {id(m) for m in quantized}
    owned: dict[int, dict] = {}
    # 1. dense: parameters and buffers of non-quantized modules
    dense_roots = []
    for name, m in modules.items():
        if id(m) in quantized_ids:
            continue
        for k, t in list(m._parameters.items()) + list(m._buffers.items()):
            if t is not None:
                dense_roots.append((f"{name}.{k}", t))
    walker = _Walker(stop_modules=set())
    walker.walk(dense_roots, "dense", owned)
    # fold_bn and similar keepalives: (Conv1d, weight clone, bias) tuples retained for restoration
    # are found below via the optimization context; classify tuples whose first item is a dense
    # Conv1d as dense copies before the weights walk.
    all_module_ids = {id(m) for m in modules.values()}
    opt = getattr(model, "_packed_optimization", None)
    ctx_roots = [("model._packed_optimization", opt)] if opt is not None else []
    dense_copies = []
    activation_roots = []
    for path, obj in _iter_objects(ctx_roots + list((extra_roots or {}).items()) +
                                   [(f"{n}.__dict__", vars(m)) for n, m in modules.items()], all_module_ids):
        if isinstance(obj, tuple) and len(obj) == 3 and isinstance(obj[0], nn.Conv1d) and id(obj[0]) not in quantized_ids:
            dense_copies.append((path, obj[1:]))
        if isinstance(obj, list) and path.rsplit(".", 1)[-1].startswith(("<default 1>", "<default 2>")) \
                and "linear_pos" in path:
            activation_roots.append((path, obj))   # positional cache / captured lists
        if type(obj).__name__ in ("PipelineGraphCache", "EncoderGraphCache"):
            activation_roots.append((path + ".entries", obj.entries))
        if hasattr(obj, "prequantized") and isinstance(getattr(obj, "prequantized"), dict):
            activation_roots.append((path + ".prequantized", obj.prequantized))
    for name, m in modules.items():
        for key in ("_packed_positional_capture",):
            if key in vars(m):
                activation_roots.append((f"{name}.{key}", vars(m)[key]))
    # 2. the quantized modules' own parameters and buffers are weights
    direct = []
    for name in quantized_names:
        m = modules[name]
        for k, t in list(m._parameters.items()) + list(m._buffers.items()):
            if t is not None:
                direct.append((f"{name}.{k}", t))
    walker.walk(direct, "weights", owned)
    # 3. decoder-side transform contexts and fold_bn originals are dense model state
    decoder_side = []
    for path, obj in _iter_objects(ctx_roots, all_module_ids):
        if isinstance(obj, types.GeneratorType) and any(
                key in obj.gi_code.co_name for key in ("decoder", "control", "graph_pool")):
            decoder_side.append((path, obj))
        # fold_bn temporaries left in encoder_transform's frame (alpha, weight/bias clones, bn, dw)
        if isinstance(obj, types.GeneratorType) and obj.gi_code.co_name == "encoder_transform" and obj.gi_frame:
            decoder_side += [(f"{path}.<local {k}>", v) for k, v in obj.gi_frame.f_locals.items()
                             if k in ("alpha", "weight", "bias", "bn", "dw")]
    _Walker(stop_modules=all_module_ids).walk(decoder_side, "dense", owned)
    walker.walk(dense_copies, "dense", owned)
    # 4. activations
    walker.walk(activation_roots, "activation", owned)
    # 3. weights: quantized modules, patched encoder functions, optimization context, extra roots
    stop = {id(m) for m in modules.values() if id(m) not in quantized_ids}
    weight_walker = _Walker(stop_modules=stop)
    roots = [(n, modules[n]) for n in quantized_names]
    for name, m in modules.items():
        for k, v in vars(m).items():
            if callable(v) and not isinstance(v, nn.Module):
                roots.append((f"{name}.{k}", v))
    roots += ctx_roots + list((extra_roots or {}).items())
    import sys
    for mod_name, module in list(sys.modules.items()):
        if mod_name.startswith("gpu1bit.") and not mod_name.startswith(("gpu1bit.shared", "gpu1bit.tests")) \
                and module is not None:
            roots.append((f"<module {mod_name}>", dict(vars(module))))
    weight_walker.walk(roots, "weights", owned)
    # classify scale / bias storages by content
    vectors = []
    for kind, table in (("scale", scales or {}), ("bias", biases or {})):
        for name, v in table.items():
            if v is not None:
                vectors.append((kind, name, v.detach().float().cpu().contiguous()))
        if kind == "scale":
            blocks = collections.defaultdict(dict)
            for name, v in table.items():
                for q in ("linear_q", "linear_k", "linear_v"):
                    if name.endswith(".self_attn." + q):
                        blocks[name[: -len(q) - 1]][q] = v
            for block, parts in blocks.items():
                if len(parts) == 3:
                    vectors.append(("scale", block + ".qkv",
                                    torch.cat([parts[q] for q in ("linear_q", "linear_k", "linear_v")]).float()))
    by_size = collections.defaultdict(list)
    for kind, name, v in vectors:
        by_size[v.numel() * 4].append((kind, name, v))
    report = {"dense_bytes": 0, "activation_bytes": 0, "weight_code_bytes": 0, "scale_bytes": 0, "bias_bytes": 0,
              "weights": []}
    for ptr, rec in owned.items():
        cat = rec["category"]
        if cat != "weights":
            report[f"{cat}_bytes"] += rec["bytes"]
            continue
        label = "code"
        t = rec["tensor"]
        if rec["bytes"] in by_size:
            flat = torch.empty(0, dtype=torch.uint8, device=t.device).set_(t.untyped_storage()).view(torch.float32).cpu()
            for kind, name, v in by_size[rec["bytes"]]:
                if torch.equal(flat, v):
                    label = kind
                    rec["matches"] = name
                    break
        report["weight_code_bytes" if label == "code" else f"{label}_bytes"] += rec["bytes"]
        report["weights"].append({k: v for k, v in rec.items() if k not in ("tensor", "category")} | {"class": label})
    report["weights"].sort(key=lambda r: -r["bytes"])
    report["code_budget_bytes"] = CODE_BUDGET_BYTES
    report["within_budget"] = report["weight_code_bytes"] <= CODE_BUDGET_BYTES
    report["storages"] = {c: sum(1 for r in owned.values() if r["category"] == c) for c in ("dense", "activation", "weights")}
    return report


def _iter_objects(roots, stop_ids):
    seen, stack = set(), list(roots)
    while stack:
        path, obj = stack.pop()
        if id(obj) in seen or isinstance(obj, torch.Tensor):
            continue
        seen.add(id(obj))
        yield path, obj
        if isinstance(obj, nn.Module) and id(obj) in stop_ids:
            continue
        for label, child in _children(obj):
            stack.append((path + label, child))


def layer_storage(layer, scale: torch.Tensor, bias: torch.Tensor | None) -> dict:
    """Per-layer version for the contract tests: storages reachable from one BinaryLayer object."""
    owned: dict[int, dict] = {}
    _Walker(stop_modules=set()).walk([("layer", layer)], "weights", owned)
    total = sum(r["bytes"] for r in owned.values())
    s = scale.detach().float().cpu()
    b = None if bias is None else bias.detach().float().cpu()
    scale_bytes = bias_bytes = 0
    for rec in owned.values():
        t = rec["tensor"]
        if rec["bytes"] == 4 * s.numel():
            flat = torch.empty(0, dtype=torch.uint8, device=t.device).set_(t.untyped_storage()).view(torch.float32).cpu()
            if torch.equal(flat, s) and not scale_bytes:
                scale_bytes = rec["bytes"]
            elif b is not None and torch.equal(flat, b) and not bias_bytes:
                bias_bytes = rec["bytes"]
    return {"total_bytes": total, "scale_bytes": scale_bytes, "bias_bytes": bias_bytes,
            "code_bytes": total - scale_bytes - bias_bytes,
            "storages": [{k: v for k, v in r.items() if k != "tensor"} for r in owned.values()]}
