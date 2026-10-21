"""The BinaryLayer interface for native 1-bit GPU kernels, and its FP64 reference.

Owner: agent H (plans/gpu-kernel.md "Interface"). A, B and C implement BinaryLayer; the contract
tests in gpu1bit/tests/ check every registered implementation against the rules below. Changes to
this contract are announced in gpu1bit/NOTES.md first, never made silently.

Weights (export layout "parakeet-binary-v1", binary_export.BIT_LAYOUT): codes uint8 [N, ceil(K/8)];
code j of a row is bit j % 8 of byte j // 8 (least significant bit first); bit 1 = +1, bit 0 = -1;
unused tail bits are 0. Per-row FP32 scale alpha [N]; optional FP32 bias [N] (MB1 has none).

    y[..., j] = alpha_j * sum_i w_ji x[..., i] + bias_j,   w_ji = 2 b_ji - 1 in {-1, +1}
              = alpha_j * (2 * sum_i b_ji x_i - sum_i x_i) + bias_j          (the {0, 1} identity)

CONTRACT (every implementation; checked by gpu1bit/tests/test_contract.py)

prepare(codes, scale, bias, in_features, device) -> layer
  * codes uint8 [N, ceil(K/8)], scale FP32 [N], bias None or FP32 [N]; they may live on the CPU (the
    loader's case, from safetensors) or on `device`. Invalid input raises ValueError: wrong dtype or
    shape, nonzero tail bits, non-finite scale or bias, K < 1 or N < 1. `validate_weights` does this.
  * Every persistent GPU allocation is made here (any re-layout: tiling, interleaving, bit
    transposition, row popcounts). The layer keeps no reference to the caller's tensors and never
    changes them. Nothing persistent may be created lazily in linear()/conv() (it would break CUDA
    graph capture and hide memory); per-call scratch from torch's caching allocator is fine.

weight_bytes() -> int
  * Bytes of every persistent GPU allocation this layer owns (count each storage's full nbytes;
    `resident_bytes` does this), INCLUDING its FP32 scales and bias. The contract test checks the
    report is honest: it equals the storages reachable from the layer object (audit.layer_storage)
    and torch.cuda.memory_allocated() growth across prepare() and warm calls is at most
    weight_bytes() + UNACCOUNTED_SLACK_BYTES (no hidden or lazily created state).
  * STORAGE BUDGET (plans/gpu-kernel.md A5; authoritative check = gpu1bit/shared/audit.py `audit`):
    unique GPU storage (deduplicated by untyped_storage().data_ptr()) attributable to the 264
    layers' weights across the WHOLE transformed model, measured after preparation and after all
    warm-ups and graph captures (fused-QKV copies, retained layouts, padding, closures and provider
    caches included) must be <= floor(1.10 x 75,497,472) = 83,047,219 bytes of weight-code
    storage. Scales, biases and activation workspaces are EXCLUDED from that budget and reported
    separately; the audit recognises them by content (a storage equal to a layer's FP32 scale or
    bias vector, or the Q/K/V concatenation); anything else (codes in any layout, row popcounts,
    pre-multiplied or duplicated scales, lookup tables) counts as code storage. Expanding weights to
    INT8, FP16 or 2-bit cannot fit (that is the baseline).
  * Per-layer test (necessary, not sufficient): code storage of one prepared layer (its reachable
    storages minus exact scale/bias copies) <= code_budget_bytes(N, K) = floor(1.10 N ceil(K/8)).
    weight_budget_bytes(N, K, bias) = code_budget_bytes + 4N (+4N) is the matching bound on
    weight_bytes().

linear(x) -> y
  * x: FP32 CUDA tensor on the layer's device, shape [..., K] (1-D or more), ANY strides (NeMo
    passes transposed views of [B, K, T] storage). Leading dims may be empty (returns an empty
    [..., N]). Returns a NEW contiguous FP32 tensor [..., N] sharing storage with nothing else
    (not x, not weights, not a previous output).
conv(x) -> y
  * Kernel-1 pointwise Conv1d: x FP32 CUDA [B, K, T], ANY strides (NeMo passes x.transpose(1, 2) of
    contiguous [B, T, K]). Returns a NEW FP32 tensor [B, N, T]; its strides are free (M1 returns a
    transposed view of [B, T, N] storage), it must not share storage with x, weights or earlier
    outputs. Same arithmetic as linear on x.transpose(1, 2).
Both
  * Wrong dtype, device, rank or K raises ValueError (cheap metadata checks only: no host sync).
  * Inference only: RuntimeError if torch.is_grad_enabled() and x.requires_grad. Works under
    torch.no_grad() and torch.inference_mode(); the output never requires grad.
  * Batch 1 is the benchmark regime; B > 1 (and any number of leading linear dims) must be correct.
  * Work is enqueued on torch.cuda.current_stream(); correct on non-default streams; no host
    synchronization, no inference-time autotuning.
  * Deterministic: the same input gives bitwise the same output, eagerly and under CUDA-graph replay
    (so no atomics in reductions). CUDA-graph capturable after one eager warm-up call of the same
    shape and strides (the warm-up may compile kernels and allocate scratch): capture, write new data
    into the captured input, replay, and the output equals an eager call on that data bitwise.
  * Numerics (plans/gpu-kernel.md A3). Semantics are checked exactly on integer-valued data
    (|y - y_ref| < 0.2 elementwise, where every true result lies on a grid of spacing >= 0.5).
    Precision, per layer (each of the 264 on captured dev inputs, and each synthetic case), against
    the FP64 reference (`errors`):
      - every output finite;
      - relative Frobenius error ||y - y_ref||_F / max(||y_ref||_F, TINY);
      - elementwise guard max_ij |y - y_ref|_ij / (|alpha_j| * ||x_i||_1 + TINY), x_i the input row;
    both at most NUMERICS_FACTOR (1.5) x the worse of M1's R-bf16x3 kernel
    (inference.kernels.matmul mode="bf16x3") and R-expanded's INT8x3 projection
    (inference.encoder4.ExpandedProjection("triton_col_tuned")) on the same weights (derived 2-bit
    layout) and inputs. No floor: where both M1 kernels are exact (e.g. silence, x = 0, gives
    y = bias exactly) the candidate must be exact too.

OPTIONAL (A2, not part of the Protocol): `project(x, *, conv, components, activation, norm,
input_activation)`, the fused op the pipeline calls through gpu1bit/shared/provider.py
(BinaryProjection); semantics in provider.py's docstring. Layers without it are composed there.

Helpers here are pure PyTorch and import nothing outside torch at module import time.
"""
from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

FORMAT = "parakeet-binary-v1"
BUDGET_FACTOR = 1.10
NUMERICS_FACTOR = 1.5
TINY = 2.2250738585072014e-308  # torch.finfo(torch.float64).tiny
M1_KERNELS = ("r_bf16x3", "r_expanded_int8x3")
UNACCOUNTED_SLACK_BYTES = 64 * 1024  # allocator rounding and tiny process-wide tables
_SHIFTS = torch.arange(8, dtype=torch.uint8)


@runtime_checkable
class BinaryLayer(Protocol):
    """One quantized MB1 layer (linear or kernel-1 pointwise conv) running from 1-bit weights."""

    @classmethod
    def prepare(cls, codes: Tensor, scale: Tensor, bias: Tensor | None, in_features: int,
                device) -> "BinaryLayer": ...      # codes: MB1 uint8 [N, ceil(K/8)], LSB first

    def weight_bytes(self) -> int: ...             # everything resident on the GPU for this layer

    def linear(self, x: Tensor) -> Tensor: ...     # [..., K] FP32 (any strides) -> [..., N] FP32

    def conv(self, x: Tensor) -> Tensor: ...       # [B, K, T] FP32 (any strides) -> [B, N, T] FP32


# --- weights -------------------------------------------------------------------------------------

def packed_row_bytes(in_features: int) -> int:
    return math.ceil(in_features / 8)


def validate_weights(codes: Tensor, scale: Tensor, bias: Tensor | None, in_features: int) -> tuple[int, int]:
    """(N, K) after the checks prepare() must make; ValueError otherwise. Syncs if tensors are on GPU."""
    if not isinstance(in_features, int) or isinstance(in_features, bool) or in_features < 1:
        raise ValueError(f"in_features must be a positive int, got {in_features!r}")
    if codes.dtype != torch.uint8 or codes.dim() != 2 or codes.shape[0] < 1 \
            or codes.shape[1] != packed_row_bytes(in_features):
        raise ValueError(f"codes must be uint8 [N, {packed_row_bytes(in_features)}], got {codes.dtype} {tuple(codes.shape)}")
    n = codes.shape[0]
    tail = 8 * codes.shape[1] - in_features
    if tail and (codes[:, -1] >> (8 - tail)).any():
        raise ValueError("invalid packed codes: nonzero tail padding")
    if scale.dtype != torch.float32 or scale.shape != (n,) or not torch.isfinite(scale).all():
        raise ValueError("scale must be finite FP32 [N]")
    if bias is not None and (bias.dtype != torch.float32 or bias.shape != (n,) or not torch.isfinite(bias).all()):
        raise ValueError("bias must be None or finite FP32 [N]")
    if not (codes.device == scale.device and (bias is None or bias.device == codes.device)):
        raise ValueError("codes, scale and bias must be on one device")
    return n, in_features


def unpack_bits(codes: Tensor, in_features: int) -> Tensor:
    """uint8 [N, ceil(K/8)] -> uint8 {0, 1} [N, K] (bit 1 = +1). Rejects nonzero tail bits."""
    if codes.dtype != torch.uint8 or codes.dim() != 2 or codes.shape[1] != packed_row_bytes(in_features):
        raise ValueError(f"expected uint8 [N, {packed_row_bytes(in_features)}], got {codes.dtype} {tuple(codes.shape)}")
    bits = ((codes[..., None] >> _SHIFTS.to(codes.device)) & 1).flatten(1)
    if bits[:, in_features:].any():
        raise ValueError("invalid packed codes: nonzero tail padding")
    return bits[:, :in_features].contiguous()


def unpack_signs(codes: Tensor, in_features: int) -> Tensor:
    """uint8 [N, ceil(K/8)] -> int8 {-1, +1} [N, K]; same result as binary_export.unpack_codes."""
    return (2 * unpack_bits(codes, in_features).to(torch.int8) - 1).to(torch.int8)


def pack_signs(signs: Tensor) -> Tensor:
    """int8 {-1, +1} [N, K] -> MB1 codes uint8 [N, ceil(K/8)]; same result as binary_export.pack_codes."""
    if signs.dtype != torch.int8 or signs.dim() != 2 or ((signs != 1) & (signs != -1)).any():
        raise ValueError("expected 2-D int8 signs in {-1, +1}")
    rows, k = signs.shape
    bits = torch.zeros(rows, 8 * packed_row_bytes(k), dtype=torch.uint8, device=signs.device)
    bits[:, :k] = (signs == 1).to(torch.uint8)
    bits = bits.view(rows, -1, 8) << _SHIFTS.to(signs.device)
    return bits.sum(dim=2, dtype=torch.uint8)  # disjoint bits: sum == OR


def dense_weight(codes: Tensor, scale: Tensor, in_features: int, dtype: torch.dtype = torch.float64) -> Tensor:
    """Dequantized [N, K] = signs * scale[:, None] (exact in FP32 and FP64), on codes.device."""
    return unpack_signs(codes, in_features).to(dtype) * scale.to(device=codes.device, dtype=dtype)[:, None]


# --- references ----------------------------------------------------------------------------------

def reference_linear(x: Tensor, codes: Tensor, scale: Tensor, bias: Tensor | None, in_features: int,
                     dtype: torch.dtype = torch.float64) -> Tensor:
    """Dense reference [..., K] -> [..., N] in `dtype`, any input strides, on x.device.

    FP64 (the numerics reference): x converted exactly, w in {-1, +1} exact, s_j = sum_i w_ji x_i
    by FP64 GEMM, then y = alpha_j * s_j + bias_j in FP64. FP32: F.linear with the dequantized FP32
    weight, as binary_export.load_export's NeMo model computes it (TF32 must be off: use
    evaluate.strict_fp32 or torch.backends.cuda.matmul.allow_tf32 = False).
    """
    if x.shape[-1] != in_features:
        raise ValueError(f"expected [..., {in_features}] input, got {tuple(x.shape)}")
    codes = codes.to(x.device)
    scale = scale.to(x.device)
    if dtype == torch.float64:
        signs = unpack_signs(codes, in_features).to(torch.float64)
        y = torch.matmul(x.to(torch.float64), signs.t()) * scale.to(torch.float64)
        if bias is not None:
            y = y + bias.to(device=x.device, dtype=torch.float64)
        return y
    if dtype == torch.float32:
        weight = dense_weight(codes, scale, in_features, torch.float32)
        return torch.nn.functional.linear(x.to(torch.float32), weight,
                                          None if bias is None else bias.to(device=x.device, dtype=torch.float32))
    raise ValueError("reference dtype must be float64 or float32")


def reference_conv(x: Tensor, codes: Tensor, scale: Tensor, bias: Tensor | None, in_features: int,
                   dtype: torch.dtype = torch.float64) -> Tensor:
    """Kernel-1 pointwise Conv1d reference: [B, K, T] (any strides) -> [B, N, T] in `dtype`."""
    if x.dim() != 3 or x.shape[1] != in_features:
        raise ValueError(f"expected [B, {in_features}, T] input, got {tuple(x.shape)}")
    return reference_linear(x.transpose(1, 2), codes, scale, bias, in_features, dtype).transpose(1, 2)


def reference(kind: str, x: Tensor, codes: Tensor, scale: Tensor, bias: Tensor | None, in_features: int,
              dtype: torch.dtype = torch.float64) -> Tensor:
    if kind == "linear":
        return reference_linear(x, codes, scale, bias, in_features, dtype)
    if kind == "conv":
        return reference_conv(x, codes, scale, bias, in_features, dtype)
    raise ValueError(f"kind must be linear or conv, got {kind!r}")


def call(layer: BinaryLayer, kind: str, x: Tensor) -> Tensor:
    if kind == "linear":
        return layer.linear(x)
    if kind == "conv":
        return layer.conv(x)
    raise ValueError(f"kind must be linear or conv, got {kind!r}")


def _rows(kind: str, t: Tensor, width: int) -> Tensor:
    """[..., width] (linear) or [B, width, T] (conv) -> [M, width] rows, as FP64."""
    if kind == "conv":
        t = t.transpose(1, 2)
    return t.reshape(-1, width).to(torch.float64)


def relative_error(y: Tensor, y_ref: Tensor) -> float:
    """||y - y_ref||_F / max(||y_ref||_F, TINY), in FP64 (y_ref normally FP64)."""
    if y.shape != y_ref.shape:
        raise ValueError(f"shape mismatch {tuple(y.shape)} vs {tuple(y_ref.shape)}")
    ref = y_ref.to(torch.float64)
    diff = y.to(device=ref.device, dtype=torch.float64) - ref
    return torch.linalg.vector_norm(diff).item() / max(torch.linalg.vector_norm(ref).item(), TINY)


def elementwise_guard(kind: str, x: Tensor, y: Tensor, y_ref: Tensor, scale: Tensor) -> float:
    """max_ij |y - y_ref|_ij / (|alpha_j| * ||x_i||_1 + TINY): error relative to the worst-case
    accumulation magnitude of output (i, j); defined even where y_ref = 0."""
    if y.shape != y_ref.shape:
        raise ValueError(f"shape mismatch {tuple(y.shape)} vs {tuple(y_ref.shape)}")
    n = scale.numel()
    k = x.shape[1] if kind == "conv" else x.shape[-1]
    ref = _rows(kind, y_ref, n)
    if ref.numel() == 0:
        return 0.0
    diff = (_rows(kind, y.to(ref.device), n) - ref).abs()
    l1 = _rows(kind, x.to(ref.device), k).abs().sum(dim=1)
    bound = scale.to(device=ref.device, dtype=torch.float64).abs()[None, :] * l1[:, None] + TINY
    return (diff / bound).max().item()


def errors(kind: str, x: Tensor, y: Tensor, y_ref: Tensor, scale: Tensor) -> dict:
    """The A3 metrics of one output: finite, relative_frobenius, elementwise_guard."""
    return {"finite": bool(torch.isfinite(y).all().item()),
            "relative_frobenius": relative_error(y, y_ref),
            "elementwise_guard": elementwise_guard(kind, x, y, y_ref, scale)}


def numerics_thresholds(m1: dict[str, dict]) -> dict[str, float]:
    """{metric: NUMERICS_FACTOR x max over M1_KERNELS} from {kernel: errors(...)}."""
    return {metric: NUMERICS_FACTOR * max(m1[name][metric] for name in M1_KERNELS)
            for metric in ("relative_frobenius", "elementwise_guard")}


def within(candidate: dict, thresholds: dict[str, float]) -> bool:
    return candidate["finite"] and all(candidate[m] <= t for m, t in thresholds.items())


# --- memory --------------------------------------------------------------------------------------

def code_budget_bytes(out_features: int, in_features: int) -> int:
    """floor(1.10 * N * ceil(K/8)): per-layer weight-code storage bound (scales/bias excluded)."""
    return (out_features * packed_row_bytes(in_features) * 11) // 10


def weight_budget_bytes(out_features: int, in_features: int, has_bias: bool) -> int:
    """code_budget_bytes + 4N scale bytes + 4N bias bytes if a bias exists: bound on weight_bytes()."""
    return code_budget_bytes(out_features, in_features) + 4 * out_features + (4 * out_features if has_bias else 0)


def resident_bytes(*tensors: Tensor | None) -> int:
    """Sum of distinct CUDA storages' full nbytes (a view counts its whole storage once)."""
    seen: dict[int, int] = {}
    for t in tensors:
        if t is None or not t.is_cuda:
            continue
        storage = t.untyped_storage()
        seen[storage.data_ptr()] = max(seen.get(storage.data_ptr(), 0), storage.nbytes())
    return sum(seen.values())
