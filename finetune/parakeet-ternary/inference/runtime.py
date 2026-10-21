"""Load the existing export directly into packed, inference-only NeMo modules."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .kernels import matmul


class PackedLinear(nn.Module):
    def __init__(self, packed: torch.Tensor, scale: torch.Tensor, in_features: int,
                 bias: torch.Tensor | None = None, *, mode: str = "bf16x3"):
        super().__init__()
        from export import unpack_codes

        if in_features < 1 or packed.ndim != 2 or packed.shape[0] < 1:
            raise ValueError("weight dimensions must be positive")
        if packed.device != scale.device or (bias is not None and bias.device != packed.device):
            raise ValueError("codes, scales, and bias must be on the same device")
        # Validate on CPU once, including unused 11 codes and tail bits.
        unpack_codes(packed, in_features)
        n = packed.shape[0]
        if scale.shape != (n,) or scale.dtype != torch.float32 or not torch.isfinite(scale).all():
            raise ValueError("scale must be finite FP32 [N]")
        if bias is not None and (bias.shape != (n,) or bias.dtype != torch.float32 or not torch.isfinite(bias).all()):
            raise ValueError("bias must be finite FP32 [N]")
        if mode not in ("tf32x3", "bf16x3", "fp16"):
            raise ValueError(f"unknown mode {mode!r}")
        self.in_features, self.out_features = in_features, n
        self.mode = mode
        # Four little-endian export bytes form one uint32 decode word.
        padded = torch.nn.functional.pad(packed, (0, (-packed.shape[1]) % 4))
        self.register_buffer("packed_t", padded.contiguous().view(torch.int32).t().contiguous())
        self.register_buffer("scale", scale.contiguous())
        self.register_buffer("bias", bias.contiguous() if bias is not None else None)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return matmul(x, self.packed_t, self.scale, self.bias, self.in_features, mode=self.mode)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, mode={self.mode!r}"


class PackedPointwiseConv1d(PackedLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return matmul(x, self.packed_t, self.scale, self.bias, self.in_features, conv=True, mode=self.mode)


def load_packed(directory: str | Path, device: str | torch.device = "cuda", *, mode: str = "bf16x3"):
    """Construct NeMo from the verified export, replacing all 264 ternary layers.

    No requantization and no .nemo download. The initial NeMo constructor uses
    CPU weights, which are discarded as modules are replaced. No dense ternary
    weight is ever moved to the GPU. The decoder and remaining layers are FP32.
    """
    from export import FORMAT, MANIFEST, TOKENIZER_DIR, TOKENIZER_FILES, sha256_file
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from omegaconf import OmegaConf, open_dict
    from safetensors.torch import load_file
    from quant import EXPECTED_MODULES, QUANTIZED_PATTERN

    if mode not in ("tf32x3", "bf16x3"):
        raise ValueError("the model loader requires a high-accuracy FP32-input mode (tf32x3 or bf16x3)")
    device = torch.device(device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
        raise ValueError("this runtime targets NVIDIA SM120 (RTX 5090)")
    directory = Path(directory).resolve()
    manifest = json.loads((directory / MANIFEST).read_text())
    if manifest["format"] != FORMAT:
        raise ValueError("unsupported export format")
    for rel, digest in manifest["files"].items():
        if sha256_file(directory / rel) != digest:
            raise ValueError(f"SHA-256 mismatch: {rel}")
    layers = manifest["quantized_layers"]
    if len(layers) != EXPECTED_MODULES or any(not QUANTIZED_PATTERN.fullmatch(n) for n in layers):
        raise ValueError("export must contain the 264 Parakeet ternary layers")
    cfg = OmegaConf.create(manifest["config"])
    if cfg.get("target") != "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel":
        raise ValueError("unexpected model target")
    with open_dict(cfg):
        cfg.tokenizer.dir = str(directory / TOKENIZER_DIR)
        for key, filename in TOKENIZER_FILES.items():
            cfg.tokenizer[key] = str(directory / TOKENIZER_DIR / filename)
        for ds in ("train_ds", "validation_ds", "test_ds"):
            if ds in cfg:
                cfg[ds] = None
    model = EncDecRNNTBPEModel(cfg=cfg)
    tensors = load_file(directory / manifest["file"])
    packed_state = {}
    for name, layer in layers.items():
        original = model.get_submodule(name)
        n, k = layer["shape"]
        if tuple(original.weight.shape[:2]) != (n, k):
            raise ValueError(f"invalid shape for {name}")
        cls = {"linear": PackedLinear, "pointwise_conv1d": PackedPointwiseConv1d}[layer["kind"]]
        module = cls(tensors.pop(f"{name}.codes"), tensors.pop(f"{name}.scale"), k,
                     tensors.pop(f"{name}.bias") if layer["bias"] else None, mode=mode)
        model.set_submodule(name, module)
        packed_state.update((f"{name}.{key}", value) for key, value in module.state_dict().items())
    state = {key: value.float() if value.is_floating_point() else value for key, value in tensors.items()}
    state.update(packed_state)
    model.load_state_dict(state, strict=True)
    return model.to(device=device).eval()
