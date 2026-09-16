"""CPU Whisper replay with identical quantization in dense and packed arms.

This reuses the GPU experiment's architecture and seeded weight factory, but no
CUDA kernels. Random low-bit weights measure execution, not speech accuracy.
The dense reference deliberately quantizes activations too: it isolates the
effect of packing and CPU popcount without changing the numerical model.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import sys
import threading

import torch
import torch.nn.functional as F


_GPU_EXPERIMENT = Path(__file__).resolve().parents[1] / "inference-efficiency"
if str(_GPU_EXPERIMENT) not in sys.path:
    sys.path.append(str(_GPU_EXPERIMENT))
import runtime_model as _shared
from control import CTranslate2Control as _GPUControl

WhisperConfig = _shared.WhisperConfig
IMPLEMENTATIONS = ("dense", "scalar", "avx512", "avx512_opt")
_CONSTRUCTION_LOCK = threading.RLock()


def quantize_activation(x, activation_bits):
    """Fixed GPU-experiment A1/A2 thresholds; no activation rescaling.

    Zero is +1 for A1. The inclusive A2 boundaries are -0.5 and +0.5.
    Nonfinite activations are rejected by the benchmark health checks rather
    than scanned on every timed linear call.
    """
    if activation_bits == 1:
        return torch.where(x >= 0, 1.0, -1.0)
    if activation_bits == 2:
        return (x >= 0.5).to(torch.float32) - (x <= -0.5).to(torch.float32)
    raise ValueError("activation_bits must be 1 or 2")


class _CPUWeight:
    def __init__(self, factory, n, k, scale, *, implementation,
                 activation_bits, threads):
        self.n, self.k, self.factory = n, k, factory
        self.scale = float(scale)
        self.activation_bits = activation_bits
        self.data = None
        self.packed = None
        codes = factory.codes((n, k))
        if implementation == "dense":
            # Unscaled integer-valued FP32 codes make the GEMM accumulation
            # exact at Whisper widths; apply scale once, as popcount does.
            self.data = codes.float()
        else:
            from native import PackedWeight
            self.packed = PackedWeight(
                codes, self.scale, activation_bits=activation_bits,
                weight_bits=1 if factory.distribution == "binary" else 2,
                backend=implementation, threads=threads)

    def linear(self, x, bias=None):
        if self.packed is not None:
            return self.packed.linear(x, bias=bias)
        if x.device.type != "cpu" or x.dtype != torch.float32:
            raise ValueError("Expected CPU float32 input")
        out = F.linear(quantize_activation(x, self.activation_bits), self.data)
        out.mul_(self.scale)
        if bias is not None:
            out.add_(bias)
        return out

    def embedding(self, ids):
        # IDs are indices, not quantized activations. Keep the tied embedding
        # and vocabulary projection backed by the same weight object.
        if self.packed is not None:
            return self.packed.embedding(ids)
        return F.embedding(ids, self.data).mul_(self.scale)

    def assign(self, data):
        raise ValueError("This quantized execution experiment uses seeded random weights")

    @property
    def storage_bytes(self):
        if self.packed is not None:
            return self.packed.storage_bytes
        return self.data.numel() * self.data.element_size() + 4


@contextmanager
def _cpu_weight_construction(implementation, activation_bits, threads):
    """Limit the shared factory substitution to construction and always restore.

    Existing models keep their own weight objects, so switching implementations
    does not modify an already constructed model's dispatch.
    """
    with _CONSTRUCTION_LOCK:
        original = _shared._Weight

        def weight(factory, n, k, scale):
            return _CPUWeight(factory, n, k, scale,
                              implementation=implementation,
                              activation_bits=activation_bits, threads=threads)

        _shared._Weight = weight
        try:
            yield
        finally:
            _shared._Weight = original


class CPUReplayWhisper(_shared.ReplayWhisper):
    """CPU FP32 residual graph with dense or packed quantized projections.

    ``threads`` controls native OpenMP. The benchmark worker also sets PyTorch
    and BLAS thread counts before construction; this class does not change
    process-global PyTorch threading when creating an individual model.
    """

    def __init__(self, config=None, distribution="ternary", activation_bits=2,
                 implementation="avx512_opt", seed=20260911, threads=1):
        if implementation not in IMPLEMENTATIONS:
            raise ValueError(f"implementation must be one of {IMPLEMENTATIONS}")
        if activation_bits not in (1, 2):
            raise ValueError("activation_bits must be 1 or 2")
        if activation_bits == 2 and distribution != "ternary":
            raise ValueError("A2 is defined for ternary weights in this experiment")
        if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
            raise ValueError("threads must be a positive integer")
        self.cpu_implementation = implementation
        self.activation_bits = activation_bits
        self.threads = threads
        # Pass a valid shared factory tag. Actual linear dispatch lives on the
        # injected weight objects; CPU metadata and convolution are overridden.
        with _cpu_weight_construction(implementation, activation_bits, threads):
            super().__init__(config=config, distribution=distribution,
                             implementation="dense", seed=seed,
                             device="cpu", dtype=torch.float32)
        self.implementation = implementation

    def _convolve(self, x, weight, bias, stride):
        # Every arm quantizes identical padded windows. A normal dense conv1d
        # would leave activations unquantized and would be a different model.
        windows = F.pad(x, (1, 1)).unfold(-1, 3, stride)
        windows = windows.permute(0, 2, 1, 3).contiguous().flatten(2)
        return weight.linear(windows, bias).transpose(1, 2)

    def load_hf_safetensors(self, directory):
        raise ValueError("Use CPUCTranslate2Control for pretrained model measurements")

    def metadata(self):
        c = self.config
        vector_bytes = self.factory.vector_count * 4
        native_metadata = None
        if self.cpu_implementation != "dense":
            native_metadata = self.factory.matrices[0].packed.metadata
        return {
            "engine": "pytorch-sdpa-cpu-quantized",
            "architecture": asdict(c),
            "implementation": self.cpu_implementation,
            "distribution": self.distribution,
            "weight_bits": 1 if self.distribution == "binary" else 2,
            "activation_bits": self.activation_bits,
            "seed": self.seed,
            "parameter_count": self.factory.parameter_count,
            "weight_storage_bytes": sum(w.storage_bytes for w in self.factory.matrices) + vector_bytes,
            "learned_vector_parameters_stored_dense": self.factory.vector_count,
            "additional_fixed_encoder_position_bytes": self.encoder_position.numel() * 4,
            "additional_structural_zero_bias_bytes": (
                c.n_audio_layer * c.n_audio_state + 2 * c.n_text_layer * c.n_text_state) * 4,
            "activation_dtype": "torch.float32",
            "matrix_accumulation": "exact integer dot, float32 weight scale then bias",
            "activation_quantization": ("x >= 0 gives +1, otherwise -1" if self.activation_bits == 1
                                        else "x >= 0.5 gives +1; x <= -0.5 gives -1; otherwise 0"),
            "threads": self.threads,
            "torch_threads": torch.get_num_threads(),
            "native": native_metadata,
            "cuda_graph": False,
            "pretrained_checkpoint": None,
            "quality_claim": "none: seeded random weights, execution experiment only",
            "replay": "one SOT prefill without logits; N sequential decode steps beginning at no_timestamps; N forced text outputs",
            "exceptions": [
                "Attention, normalization, residuals, biases, GELU and positions use FP32",
                "Embedding lookup decodes scaled weight values without activation quantization",
                "All custom arms include padded im2col and activation quantization for convolutions",
                "Dense reference stores unscaled FP32 codes and applies scale after the integer-valued dot",
                "CTranslate2 is a separate pretrained external control with a different execution engine",
            ],
        }


class CPUCTranslate2Control(_GPUControl):
    """Pretrained CPU control sharing the existing checked forced replay contract."""

    def __init__(self, model_path, compute_type="int8_float32", threads=1,
                 sot_id=50257, no_timestamps_id=50362):
        import ctranslate2
        if compute_type not in ("float32", "int8_float32"):
            raise ValueError("CPU control precision must be float32 or int8_float32")
        if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
            raise ValueError("threads must be a positive integer")
        supported = sorted(ctranslate2.get_supported_compute_types("cpu"))
        if compute_type not in supported:
            raise RuntimeError(f"CTranslate2 CPU lacks {compute_type}; supports {supported}")
        self.ct2 = ctranslate2
        self.model_path = str(Path(model_path).resolve())
        self.requested_compute_type = compute_type
        self.flash_attention = False
        self.device_index = 0
        self.sot_id, self.no_timestamps_id = sot_id, no_timestamps_id
        self.threads = threads
        self.supported_compute_types = supported
        self.model = ctranslate2.models.Whisper(
            self.model_path, device="cpu", compute_type=compute_type,
            inter_threads=1, intra_threads=threads)
        if self.model.compute_type != compute_type:
            raise RuntimeError(
                f"CTranslate2 precision fallback: requested {compute_type}, got {self.model.compute_type}")
        self.replay_verified = False

    def metadata(self):
        return {
            "engine": "ctranslate2", "version": self.ct2.__version__,
            "device": "cpu", "model_path": self.model_path,
            "compute_type": self.model.compute_type,
            "requested_compute_type": self.requested_compute_type,
            "supported_compute_types": self.supported_compute_types,
            "threads": self.threads, "inter_threads": 1,
            "replay_verified": self.replay_verified,
            "replay": "one SOT prefill without logits; N sequential decode steps beginning at no_timestamps; N forced text outputs",
            "source_contract": "https://github.com/OpenNMT/CTranslate2/blob/v4.8.2/src/decoding.cc",
            "same_engine_as_candidates": False,
            "quality_claim": "pretrained external control; this fixed-token replay does not measure WER",
        }
