"""488,270,080-parameter chunk-aware binary/ternary CTC training model.

BinaryCTCModel accepts a ModelConfig or a dictionary of its fields. Inputs are
[B, n_mels, T] and actual frame lengths; outputs are [B, ceil(T/8), vocab_size]
and ceil(lengths/8). Parameters stay FP32; use autocast for BF16 QAT matmuls.
At fractions (1, 1), projections use binary signs or scaled ternary codes,
according to config.quantizer. Fraction zero bypasses the corresponding quantizer.
Training executes ordinary differentiable matmuls, not packed popcount kernels.
This file implements masked training, not an exported streaming cache runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 1024
    ff_dim: int = 4096
    num_layers: int = 20
    num_heads: int = 16
    stem_channels: int = 256
    vocab_size: int = 2048  # Includes CTC blank.
    n_mels: int = 80
    chunk_size: int = 4
    left_context: int = 64
    depthwise_kernel: int = 9
    dropout: float = 0.1
    activation_checkpointing: bool = True
    weight_fraction: float = 0.0
    activation_fraction: float = 0.0
    center_weights: bool = False
    quantizer: str = "binary"

    @classmethod
    def from_dict(cls, values: Mapping | "ModelConfig") -> "ModelConfig":
        if isinstance(values, cls):
            return values
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown model configuration fields: {sorted(unknown)}")
        return cls(**values)

    def __post_init__(self) -> None:
        for name in (
            "d_model", "ff_dim", "num_layers", "num_heads", "stem_channels",
            "vocab_size", "n_mels", "chunk_size", "depthwise_kernel",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.num_heads or (self.d_model // self.num_heads) % 2:
            raise ValueError("Head dimension must be an even integer for fixed RoPE")
        if self.left_context < 0 or not 0 <= self.dropout < 1:
            raise ValueError("left_context must be >= 0 and dropout must be in [0,1)")
        _validate_fraction(self.weight_fraction)
        _validate_fraction(self.activation_fraction)
        if self.quantizer not in {"binary", "ternary"}:
            raise ValueError("quantizer must be binary or ternary")


def _validate_fraction(value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Quantization fractions must be finite numbers in [0, 1]")


def binary_sign(value: Tensor) -> Tensor:
    """Deterministic +/-1 signs, including sign(0) == +1."""
    return torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))


class _ActivationSignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor) -> Tensor:
        return binary_sign(value)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> Tensor:
        return gradient


class _DequantizedWeightSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: Tensor, scale: Tensor, center: bool) -> Tensor:
        signs = binary_sign(weight - weight.mean() if center else weight)
        ctx.save_for_backward(signs)
        return signs * scale

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, Tensor, None]:
        (signs,) = ctx.saved_tensors
        return gradient, (gradient * signs).sum(dim=1, keepdim=True), None


class _WeightSignsSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: Tensor, scale: Tensor, center: bool) -> Tensor:
        ctx.save_for_backward(scale)
        return binary_sign(weight - weight.mean() if center else weight)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None, None]:
        (scale,) = ctx.saved_tensors
        # The external output scale cancels this factor. Thus the effective
        # dequantized matrix has the identity latent-weight derivative.
        return gradient / scale, None, None


def activation_quantize(value: Tensor, threshold: Tensor, fraction: float) -> Tensor:
    """FP32 sign statistics with an identity straight-through estimator."""
    if fraction == 0:
        return value
    shifted = value.float() - threshold.float()
    quantized = _ActivationSignSTE.apply(shifted)
    return torch.lerp(value.float(), quantized, fraction).to(value.dtype)


def ternary_codes(value: Tensor) -> Tensor:
    """Row/vector absmean-normalized {-1,0,+1} codes; exact half ties round to zero."""
    scale = value.float().abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    return (value.float() / scale).round().clamp(-1, 1)


class _TernaryWeightSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: Tensor, scale: Tensor, center: bool) -> Tensor:
        codes = ternary_codes(weight - weight.mean() if center else weight)
        ctx.save_for_backward(codes)
        return codes * scale

    @staticmethod
    def backward(ctx, gradient: Tensor):
        (codes,) = ctx.saved_tensors
        return gradient, (gradient * codes).sum(dim=1, keepdim=True), None


def ternary_activation_quantize(value: Tensor, threshold: Tensor, fraction: float) -> Tensor:
    """Three codes per token, with detached absmean scale and identity input STE."""
    if fraction == 0:
        return value
    shifted = value.float() - threshold.float()
    scale = shifted.detach().abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    quantized = scale * (shifted / scale).round().clamp(-1, 1)
    surrogate = shifted + (quantized - shifted).detach()
    return torch.lerp(value.float(), surrogate, fraction).to(value.dtype)


class BinaryLinear(nn.Module):
    """Bias-free latent FP32 matrix, input thresholds, positive output scales.

    Following the STE idiom in Microsoft's BitNet training reference, the
    derivative of the dequantized weight w.r.t. its latent weight is identity.
    The custom STE also trains this model's separate positive output scales.
    The default codebook is {-1,+1}. The opt-in ternary branch uses absmean
    {-1,0,+1} codes with learned output scales and optional ternary activations.
    It is a project adaptation, not an exact BitNet b1.58/INT8 reproduction.
    """

    def __init__(self, in_features: int, out_features: int, center_weights: bool = False) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.center_weights = center_weights
        self.quantizer = "binary"
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        self.threshold = nn.Parameter(torch.zeros(in_features, dtype=torch.float32))
        self.log_scale = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        self.weight_fraction = 0.0
        self.activation_fraction = 0.0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        with torch.no_grad():
            self.log_scale.copy_(self.weight.float().abs().mean(dim=1).clamp_min(1e-8).log())
            self.threshold.zero_()

    def set_quantization(self, weight_fraction: float, activation_fraction: float) -> None:
        _validate_fraction(weight_fraction)
        _validate_fraction(activation_fraction)
        self.weight_fraction = float(weight_fraction)
        self.activation_fraction = float(activation_fraction)

    @property
    def output_scale(self) -> Tensor:
        return self.log_scale.float().exp()

    def effective_weight(self) -> Tensor:
        weight = self.weight.float()
        if self.weight_fraction == 0:
            return weight
        scale = self.output_scale[:, None]
        if self.quantizer == "ternary":
            quantized = _TernaryWeightSTE.apply(weight, scale, self.center_weights)
            return torch.lerp(weight, quantized, self.weight_fraction)
        binary = _DequantizedWeightSTE.apply(weight, scale, self.center_weights)
        return torch.lerp(weight, binary, self.weight_fraction)

    def forward(self, value: Tensor) -> Tensor:
        if self.quantizer == "ternary":
            activation = ternary_activation_quantize(value, self.threshold, self.activation_fraction)
            return F.linear(activation, self.effective_weight())
        activation = activation_quantize(value, self.threshold, self.activation_fraction)
        if self.weight_fraction == 1:
            scale = self.output_scale
            signs = _WeightSignsSTE.apply(self.weight.float(), scale[:, None].detach(), self.center_weights)
            projected = F.linear(activation, signs)
            # Scale after the binary dot, matching the intended popcount
            # operator. Autocast may still round the matmul output to BF16;
            # packed-export numerical parity requires a separate validation.
            return (projected.float() * scale).to(projected.dtype)
        return F.linear(activation, self.effective_weight())


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, value: Tensor) -> Tensor:
        return F.layer_norm(
            value.float(), self.normalized_shape, self.weight.float(),
            self.bias.float(), self.eps,
        ).to(value.dtype)


def _valid_mask(lengths: Tensor, size: int) -> Tensor:
    return torch.arange(size, device=lengths.device)[None, :] < lengths[:, None]


def _mask_frames(value: Tensor, valid: Tensor) -> Tensor:
    return value.masked_fill(~valid[..., None], 0)


def chunk_attention_mask(length: int, chunk_size: int, left_context: int,
                         device: torch.device | str | None = None) -> Tensor:
    """True = visible; cached context is measured from the chunk start."""
    positions = torch.arange(length, device=device)
    start = (positions // chunk_size) * chunk_size
    return ((positions[None, :] >= (start - left_context)[:, None]) &
            (positions[None, :] < (start + chunk_size)[:, None]))


class CausalSubsampler(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        c = cfg.stem_channels
        self.convolutions = nn.ModuleList([
            nn.Conv2d(1, c, 3, stride=2, bias=False),
            nn.Conv2d(c, c, 3, stride=2, groups=c, bias=False),
            nn.Conv2d(c, c, 3, stride=2, groups=c, bias=False),
        ])
        self.pointwise = nn.ModuleList([
            nn.Identity(), nn.Conv2d(c, c, 1, bias=False), nn.Conv2d(c, c, 1, bias=False),
        ])
        self.norms = nn.ModuleList([FP32LayerNorm(c) for _ in range(3)])
        frequency_bins = (cfg.n_mels + 7) // 8
        self.projection = nn.Linear(c * frequency_bins, cfg.d_model, bias=False)

    def forward(self, features: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        # CNN layout [B, C, frequency, time]. Normalize C independently at
        # every time/frequency cell; no temporal mean or variance is computed.
        value = features.masked_fill(~_valid_mask(lengths, features.shape[-1])[:, None, :], 0)
        value = value[:, None]
        for convolution, pointwise, norm in zip(self.convolutions, self.pointwise, self.norms):
            value = convolution(F.pad(value, (2, 0, 1, 1)))
            value = pointwise(value)
            value = F.silu(norm(value.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
            lengths = (lengths + 1) // 2
            value = value.masked_fill(~_valid_mask(lengths, value.shape[-1])[:, None, None, :], 0)
        value = value.permute(0, 3, 1, 2).flatten(2)
        return self.projection(value), lengths


class BinaryFeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.up = BinaryLinear(cfg.d_model, cfg.ff_dim, cfg.center_weights)
        self.down = BinaryLinear(cfg.ff_dim, cfg.d_model, cfg.center_weights)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, value: Tensor) -> Tensor:
        # The down projection owns the learned hidden threshold/nonlinearity.
        # In FP reference mode this is identity; during QAT it blends to sign.
        return self.dropout(self.down(self.up(value)))


class ChunkAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.qkv = BinaryLinear(cfg.d_model, 3 * cfg.d_model, cfg.center_weights)
        self.output = BinaryLinear(cfg.d_model, cfg.d_model, cfg.center_weights)
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.d_model // cfg.num_heads
        self.dropout = cfg.dropout
        self.output_dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "inverse_frequency", 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)),
            persistent=False,
        )

    def _rope(self, value: Tensor) -> Tensor:
        positions = torch.arange(value.shape[-2], device=value.device, dtype=torch.float32)
        angles = positions[:, None] * self.inverse_frequency.float()[None, :]
        cosine, sine = angles.cos()[None, None], angles.sin()[None, None]
        even, odd = value.float()[..., ::2], value.float()[..., 1::2]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2).to(value.dtype)

    def forward(self, value: Tensor, attention_mask: Tensor) -> Tensor:
        batch, steps, width = value.shape
        qkv = self.qkv(value).view(batch, steps, 3, self.num_heads, self.head_dim)
        query, key, projected_value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query, key = self._rope(query), self._rope(key)
        attended = F.scaled_dot_product_attention(
            query, key, projected_value, attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, steps, width)
        return self.output_dropout(self.output(attended))


class CausalConvolution(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.expansion = BinaryLinear(cfg.d_model, 2 * cfg.d_model, cfg.center_weights)
        self.depthwise = nn.Conv1d(
            cfg.d_model, cfg.d_model, cfg.depthwise_kernel,
            groups=cfg.d_model, bias=False,
        )
        self.internal_norm = FP32LayerNorm(cfg.d_model)
        self.output = BinaryLinear(cfg.d_model, cfg.d_model, cfg.center_weights)
        self.kernel = cfg.depthwise_kernel
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, value: Tensor, valid: Tensor) -> Tensor:
        value = _mask_frames(F.glu(self.expansion(value), dim=-1), valid)
        value = self.depthwise(F.pad(value.transpose(1, 2), (self.kernel - 1, 0))).transpose(1, 2)
        value = F.silu(self.internal_norm(value))
        return self.dropout(self.output(value))


class BinaryConformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ffn1 = BinaryFeedForward(cfg)
        self.attention = ChunkAttention(cfg)
        self.convolution = CausalConvolution(cfg)
        self.ffn2 = BinaryFeedForward(cfg)
        # Five external LNs + convolution.internal_norm = six affine LNs.
        self.norms = nn.ModuleList([FP32LayerNorm(cfg.d_model) for _ in range(5)])

    def forward(self, value: Tensor, valid: Tensor, attention_mask: Tensor) -> Tensor:
        value = _mask_frames(value + 0.5 * self.ffn1(self.norms[0](value)), valid)
        value = _mask_frames(value + self.attention(self.norms[1](value), attention_mask), valid)
        value = _mask_frames(value + self.convolution(self.norms[2](value), valid), valid)
        value = _mask_frames(value + 0.5 * self.ffn2(self.norms[3](value)), valid)
        return _mask_frames(self.norms[4](value), valid)


class BinaryCTCModel(nn.Module):
    def __init__(self, cfg: Mapping | ModelConfig | None = None) -> None:
        super().__init__()
        self.config = ModelConfig.from_dict({} if cfg is None else cfg)
        self.stem = CausalSubsampler(self.config)
        self.blocks = nn.ModuleList([BinaryConformerBlock(self.config) for _ in range(self.config.num_layers)])
        self.classifier = nn.Linear(self.config.d_model, self.config.vocab_size, bias=True)
        for module in self.modules():
            if isinstance(module, BinaryLinear):
                module.quantizer = self.config.quantizer
        self.set_quantization(self.config.weight_fraction, self.config.activation_fraction)

    def set_quantization(self, weight_fraction: float, activation_fraction: float) -> None:
        _validate_fraction(weight_fraction)
        _validate_fraction(activation_fraction)
        for module in self.modules():
            if isinstance(module, BinaryLinear):
                module.set_quantization(weight_fraction, activation_fraction)

    @staticmethod
    def encoded_lengths(feature_lengths: Tensor) -> Tensor:
        return (feature_lengths + 7) // 8

    def forward(self, features: Tensor, feature_lengths: Tensor) -> tuple[Tensor, Tensor]:
        if features.ndim != 3 or features.shape[1] != self.config.n_mels:
            raise ValueError(f"Expected features [batch,{self.config.n_mels},time]")
        if feature_lengths.ndim != 1 or feature_lengths.shape[0] != features.shape[0]:
            raise ValueError("feature_lengths must have shape [batch]")
        if feature_lengths.is_floating_point() or feature_lengths.dtype == torch.bool:
            raise ValueError("feature_lengths must contain integers")
        lengths = feature_lengths.to(device=features.device, dtype=torch.long)
        if torch.any(lengths <= 0) or torch.any(lengths > features.shape[-1]):
            raise ValueError("Feature lengths must be positive and within the padded input")
        value, lengths = self.stem(features, lengths)
        valid = _valid_mask(lengths, value.shape[1])
        mask = chunk_attention_mask(
            value.shape[1], self.config.chunk_size, self.config.left_context, value.device,
        )[None, None] & valid[:, None, None, :]
        for block in self.blocks:
            if self.training and self.config.activation_checkpointing and torch.is_grad_enabled():
                value = checkpoint(block, value, valid, mask, use_reentrant=False)
            else:
                value = block(value, valid, mask)
        logits = _mask_frames(self.classifier(value), valid)
        return logits, lengths
