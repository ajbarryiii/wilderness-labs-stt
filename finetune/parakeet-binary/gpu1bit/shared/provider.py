"""BinaryProjection: candidates' 1-bit layers at M1's projection-provider seam (plans/gpu-kernel.md A2).

M1's optimized runtimes route every quantized-layer call through one `projection` object, called by
inference.transforms.encoder_transform and inference.layer_fusions4.layer_fusions as

    provider(x, mod, conv=False, components=3, activation="", norm=None, config=None, input_activation="")

with these semantics (inference/encoder4.py ExpandedProjection.__call__, the R-expanded reference):
    xx = x.transpose(1, 2) if conv else x           # conv: [B, K, T] -> [B, T, K] view
    z  = LayerNorm(norm)(xx) if norm else xx        # fused input normalization (norm_quant=True)
    z  = z * sigmoid(z) if input_activation == "silu" else z
    y  = alpha * (W z) + bias                       # the quantized layer, [B, T, N]
    y  = y * sigmoid(y) if activation == "silu" else y
    return y.transpose(1, 2) if conv else y          # conv returns a [B, N, T] view
and the call sites are (per Conformer block, eval, offline): fused QKV (norm=norm_self_att, mod is a
SimpleNamespace whose packed_t is torch.cat of the three modules' packed_t along dim 1),
linear_out, linear_pos (eager at warm-up, cached in graphs), ff{1,2}.linear1 (norm, activation="silu"),
ff{1,2}.linear2, pointwise_conv1 (conv, norm=norm_conv), pointwise_conv2 (conv, input_activation="silu").
`components` is always 3 (INT8x3 residual activations in M1); `config` is always None.

THE CANDIDATE-FACING OPERATION (optional; A/B/C implement it to fuse):

    layer.project(x, *, conv, components, activation, norm, input_activation) -> Tensor

  on a BinaryLayer, with exactly the semantics above (x and the result as above; conv results may
  have any strides). `components` says what M1 would do (3 INT8 residual parts); an INT8 candidate
  honours it, other arithmetic classes ignore it and say so. If a layer has no `project`, the
  provider composes it from F.layer_norm, F.silu and layer.linear/layer.conv (correct, unfused:
  fine for tests, not what screening should time).

MODEL CONSTRUCTION (A5: one copy of the weights): `load_binary_model(MB1 export, Implementation)`
builds NeMo from MB1's manifest and replaces the 264 quantized modules by BinaryLinear /
BinaryPointwiseConv1d (subclasses of M1's PackedLinear / PackedPointwiseConv1d, so M1's transform
type checks pass). Q, K and V of each block are prepared as ONE fused [3072, 1024] layer; the three
modules are row slices of it, so fusion replaces the separate weights instead of duplicating them.
A module's `packed_t` is a CPU int64 HANDLE [1, N] (global row ids, never on the GPU); torch.cat of
the Q/K/V handles in encoder_transform yields the fused handle, which the provider resolves to the
fused layer. `scale` is a host copy (CPU) that M1's QKV concatenation needs and nothing computes
with. The 1-bit layer object holds the only GPU copy of codes and scales.

RUNTIME: `enable_binary_optimizations(model, provider)` runs M1's own
inference.optimized.enable_optimizations(model, encoder_storage="expanded", ...) with
inference.encoder4.ExpandedProjection temporarily replaced by a factory returning `provider`: the
identical transform stack, fusions, decoder kernels and graph cache as R-expanded; only the
projection differs (A2 "controlled substitution"). M1's disable_optimizations undoes it.
R-hybrid's deferred row-epilogue variant (row_fusion.py: defer_epilogue / prequantized) is not
supported by this provider; a winner that supports it gets its own adapter.
"""
from __future__ import annotations

import contextlib
import json
import types
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gpu1bit.shared import api, env  # noqa: F401  (env fixes sys.path for M1's packages)

QKV = ("linear_q", "linear_k", "linear_v")


@dataclass
class Entry:
    """One prepared 1-bit layer and the global handle range of its rows."""
    name: str
    layer: object
    kind: str            # "linear" or "conv"
    out_features: int
    in_features: int
    start: int           # first global row id
    members: tuple = ()  # for a fused QKV entry: the three module names


@dataclass
class Table:
    entries: list[Entry] = field(default_factory=list)
    next_row: int = 0

    def add(self, name, layer, kind, n, k, members=()) -> Entry:
        e = Entry(name, layer, kind, n, k, self.next_row, tuple(members))
        self.entries.append(e)
        self.next_row += n
        return e

    def resolve(self, start: int, count: int) -> tuple[Entry, int]:
        for e in self.entries:
            if e.start <= start < e.start + e.out_features:
                if start + count > e.start + e.out_features:
                    raise ValueError("handle range crosses layers")
                return e, start - e.start
        raise KeyError(f"no layer owns row handle {start}")

    def weight_bytes(self) -> int:
        return sum(e.layer.weight_bytes() for e in self.entries)


def _handle(start: int, n: int) -> Tensor:
    return torch.arange(start, start + n, dtype=torch.int64).view(1, n)


class _BinaryMixin:
    """Attributes M1's transforms read: in_features, out_features, mode, packed_t, scale, bias."""

    def _setup(self, entry: Entry, offset: int, n: int, k: int, scale_cpu: Tensor, bias_cpu: Tensor | None):
        nn.Module.__init__(self)
        self.in_features, self.out_features, self.mode = k, n, "binary"
        self._entry, self._offset = entry, offset
        # Plain attributes (not buffers): never moved to the GPU, absent from state_dict.
        object.__setattr__(self, "packed_t", _handle(entry.start + offset, n))
        object.__setattr__(self, "scale", scale_cpu.detach().clone().contiguous())
        object.__setattr__(self, "bias", None if bias_cpu is None else bias_cpu.detach().clone().contiguous())
        self.unfused_calls = 0
        self.eval()

    def _run(self, x: Tensor, conv: bool) -> Tensor:
        e = self._entry
        if self.out_features == e.out_features:
            return e.layer.conv(x) if conv else e.layer.linear(x)
        self.unfused_calls += 1  # a member of a fused layer called alone: compute all rows, slice
        y = e.layer.conv(x) if conv else e.layer.linear(x)
        return y[:, self._offset:self._offset + self.out_features] if conv else \
            y[..., self._offset:self._offset + self.out_features]

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, layer={self._entry.name!r}"


def _packed_classes():
    from inference.runtime import PackedLinear, PackedPointwiseConv1d

    class BinaryLinear(_BinaryMixin, PackedLinear):
        def __init__(self, *a):
            self._setup(*a)

        def forward(self, x):
            return self._run(x, False)

    class BinaryPointwiseConv1d(_BinaryMixin, PackedPointwiseConv1d):
        def __init__(self, *a):
            self._setup(*a)

        def forward(self, x):
            return self._run(x, True)

    return BinaryLinear, BinaryPointwiseConv1d


class BinaryProjection:
    """The provider object handed to M1's transforms. See the module docstring."""

    def __init__(self, table: Table, names: dict[int, str] | None = None):
        self.table = table
        self._resolved: dict[tuple, tuple[Entry, int, int]] = {}
        self.names = names or {}
        self.record: list[dict] | None = None   # set to [] to record every call (instrumentation)
        self.composed_calls = 0

    @property
    def bytes(self) -> int:  # ExpandedProjection exposes .bytes; same meaning here
        return self.table.weight_bytes()

    def _lookup(self, mod) -> tuple[Entry, int, int]:
        handle = mod.packed_t
        key = (handle.data_ptr(), tuple(handle.shape))
        hit = self._resolved.get(key)
        if hit is None:
            if handle.is_cuda or handle.dtype != torch.int64 or handle.dim() != 2 or handle.shape[0] != 1:
                raise TypeError("BinaryProjection serves BinaryLinear / BinaryPointwiseConv1d handles only")
            start, count = int(handle[0, 0]), handle.shape[1]
            if not torch.equal(handle[0], torch.arange(start, start + count)):
                raise ValueError("non-contiguous row handle")
            entry, offset = self.table.resolve(start, count)
            hit = (entry, offset, count)
            self._resolved[key] = hit
        return hit

    def __call__(self, x, mod, conv=False, components=3, activation="", norm=None, config=None, input_activation=""):
        if config is not None:
            raise ValueError("BinaryProjection takes no M1 tile config")
        if activation not in ("", "silu") or input_activation not in ("", "silu"):
            raise ValueError(f"unsupported activation {activation!r} / {input_activation!r}")
        entry, offset, count = self._lookup(mod)
        if self.record is not None:
            self.record.append({"layer": entry.name, "rows": [offset, offset + count], "N": count,
                                "K": entry.in_features, "conv": bool(conv), "components": components,
                                "activation": activation, "norm": self.names.get(id(norm)) if norm is not None else None,
                                "input_activation": input_activation, "x_shape": list(x.shape),
                                "x_stride": list(x.stride()), "fused_op": hasattr(entry.layer, "project")})
        layer = entry.layer
        if hasattr(layer, "project"):
            y = layer.project(x, conv=conv, components=components, activation=activation, norm=norm,
                              input_activation=input_activation)
        else:
            self.composed_calls += 1
            if conv and norm is None and not input_activation:
                y = layer.conv(x)
                if activation == "silu":
                    y = F.silu(y)
            else:
                z = x.transpose(1, 2) if conv else x
                if norm is not None:
                    z = F.layer_norm(z, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
                if input_activation == "silu":
                    z = F.silu(z)
                y = layer.linear(z)
                if activation == "silu":
                    y = F.silu(y)
                if conv:
                    y = y.transpose(1, 2)
        if count != entry.out_features:
            y = y[:, offset:offset + count] if conv else y[..., offset:offset + count]
        return y


def _verify_export(directory: Path) -> dict:
    import binary_export
    manifest = json.loads((directory / binary_export.MANIFEST).read_text())
    if manifest["format"] != binary_export.FORMAT:
        raise ValueError(f"unknown export format {manifest['format']!r}")
    for rel, digest in manifest["files"].items():
        if env.sha256_file(directory / rel) != digest:
            raise ValueError(f"{directory / rel} does not match the manifest SHA-256")
    return manifest


def load_binary_model(directory, implementation, device="cuda", *, fuse_qkv: bool = True):
    """(NeMo model on `device`, Table) from a parakeet-binary-v1 export; see module docstring.

    Non-quantized tensors are loaded as binary_export.load_export does (FP16 -> FP32, strict key
    check). The 264 quantized modules hold no GPU tensors; `implementation.prepare` makes the only
    GPU copy of codes and scales (Q/K/V fused when fuse_qkv).
    """
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from omegaconf import OmegaConf, open_dict
    from safetensors.torch import load_file
    import binary_export
    from quant import EXPECTED_MODULES, QUANTIZED_PATTERN

    directory = Path(directory).resolve()
    device = torch.device(device)
    manifest = _verify_export(directory)
    layers = manifest["quantized_layers"]
    if len(layers) != EXPECTED_MODULES or any(not QUANTIZED_PATTERN.fullmatch(n) for n in layers):
        raise ValueError("export must contain the 264 Parakeet quantized layers")
    cfg = OmegaConf.create(manifest["config"])
    with open_dict(cfg):
        cfg.tokenizer.dir = str(directory / binary_export.TOKENIZER_DIR)
        for key, filename in binary_export.TOKENIZER_FILES.items():
            cfg.tokenizer[key] = str(directory / binary_export.TOKENIZER_DIR / filename)
        for ds in ("train_ds", "validation_ds", "test_ds"):
            if ds in cfg:
                cfg[ds] = None
    model = EncDecRNNTBPEModel(cfg=cfg)
    tensors = load_file(directory / manifest["file"])
    BinaryLinear, BinaryPointwiseConv1d = _packed_classes()
    table = Table()
    pending = dict(layers)
    replacements = {}
    if fuse_qkv:
        blocks = sorted({n.rsplit(".", 1)[0] for n in layers if n.endswith(".self_attn.linear_q")})
        for block in blocks:
            names = [f"{block}.{q}" for q in QKV]
            shapes = [tuple(layers[n]["shape"]) for n in names]
            if len(set(shapes)) != 1 or any(layers[n]["kind"] != "linear" or layers[n]["bias"] for n in names):
                raise ValueError(f"{block}: Q/K/V must be equal-shape bias-free linear layers")
            n, k = shapes[0]
            codes = torch.cat([tensors.pop(f"{x}.codes") for x in names])
            scales = torch.cat([tensors.pop(f"{x}.scale") for x in names])
            layer = implementation.prepare(codes, scales, None, k, device)
            entry = table.add(f"{block}.qkv", layer, "linear", 3 * n, k, names)
            for i, x in enumerate(names):
                replacements[x] = BinaryLinear(entry, i * n, n, k, scales[i * n:(i + 1) * n], None)
                del pending[x]
    for name, info in pending.items():
        n, k = info["shape"]
        codes, scale = tensors.pop(f"{name}.codes"), tensors.pop(f"{name}.scale")
        bias = tensors.pop(f"{name}.bias") if info["bias"] else None
        kind = "conv" if info["kind"] == "pointwise_conv1d" else "linear"
        layer = implementation.prepare(codes, scale, bias, k, device)
        entry = table.add(name, layer, kind, n, k)
        cls = BinaryPointwiseConv1d if kind == "conv" else BinaryLinear
        replacements[name] = cls(entry, 0, n, k, scale, bias)
    for name, module in replacements.items():
        original = model.get_submodule(name)
        if tuple(original.weight.shape[:2]) != (module.out_features, module.in_features):
            raise ValueError(f"invalid shape for {name}")
        model.set_submodule(name, module)
    state = {key: value.float() if value.is_floating_point() else value for key, value in tensors.items()}
    model.load_state_dict(state, strict=True)  # the binary modules hold no parameters or buffers
    model = model.to(device=device).eval()
    return model, table


def module_names(model) -> dict[int, str]:
    return {id(m): n for n, m in model.named_modules()}


@contextlib.contextmanager
def _replace_expanded_projection(provider):
    """inference.optimized.enable_optimizations imports ExpandedProjection from inference.encoder4
    at call time; swap that module attribute for a factory returning `provider` during the call."""
    import inference.encoder4 as encoder4
    original = encoder4.ExpandedProjection

    def factory(backend="triton", config=None):
        if backend != "triton_col_tuned" or config is not None:
            raise AssertionError(f"unexpected ExpandedProjection({backend!r}, {config!r})")
        return provider

    encoder4.ExpandedProjection = factory
    try:
        yield
    finally:
        encoder4.ExpandedProjection = original


def enable_binary_optimizations(model, provider: BinaryProjection, *, max_graphs: int = 4, decoder: bool = True,
                                position_dot: bool = True):
    """M1's enable_optimizations(model, encoder_storage="expanded") with `provider` as the projection."""
    from inference.optimized import enable_optimizations
    with _replace_expanded_projection(provider):
        return enable_optimizations(model, max_graphs=max_graphs, decoder=decoder, position_dot=position_dot,
                                    encoder_storage="expanded")


def r_expanded_recorder(model, projection):
    """Instrument an R-expanded model's ExpandedProjection: returns the list its calls append to,
    in BinaryProjection.record's format (QKV namespaces resolved to '<block>.qkv' by their scales)."""
    names = module_names(model)
    record: list[dict] = []
    qkv = {}
    for block, m in model.named_modules():
        if block.endswith(".self_attn") and hasattr(m, "linear_q"):
            qkv[block] = torch.cat([getattr(m, q).scale for q in QKV])
    resolved: dict[int, str] = {}
    base = type(projection)

    class Recording(base):
        def __call__(self, x, mod, conv=False, components=3, activation='', norm=None, config=None,
                     input_activation=''):
            name = names.get(id(mod))
            if name is None:
                key = id(mod.packed_t)
                if key not in resolved:
                    hits = [b for b, s in qkv.items() if s.shape == mod.scale.shape and torch.equal(s, mod.scale)]
                    if len(hits) != 1:
                        raise AssertionError("cannot identify a fused projection")
                    resolved[key] = hits[0] + ".qkv"
                name = resolved[key]
            n = mod.out_features
            k = x.shape[1 if conv else -1]
            record.append({"layer": name, "rows": [0, n], "N": n, "K": k, "conv": bool(conv),
                           "components": components, "activation": activation,
                           "norm": names.get(id(norm)) if norm is not None else None,
                           "input_activation": input_activation, "x_shape": list(x.shape),
                           "x_stride": list(x.stride())})
            return base.__call__(self, x, mod, conv, components, activation, norm, config, input_activation)

    projection.__class__ = Recording
    return record
