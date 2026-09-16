"""Whisper-shaped, fixed-token inference with shared dense and packed execution.

This is a PyTorch/SDPA runtime, not a CTranslate2 modification. Its dense twins
isolate the kernel effect; the separately measured CTranslate2 control remains
the investment baseline. No transcription quality is claimed for random weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class WhisperConfig:
    name: str = "medium.en"
    n_mels: int = 80
    n_audio_ctx: int = 1500
    n_audio_state: int = 1024
    n_audio_head: int = 16
    n_audio_layer: int = 24
    n_vocab: int = 51864
    n_text_ctx: int = 448
    n_text_state: int = 1024
    n_text_head: int = 16
    n_text_layer: int = 24
    sot_id: int = 50257
    no_timestamps_id: int = 50362

    @classmethod
    def medium_en(cls):
        return cls()

    @classmethod
    def tiny_smoke(cls):
        return cls(name="smoke-only", n_mels=8, n_audio_ctx=16,
                   n_audio_state=64, n_audio_head=4, n_audio_layer=2,
                   n_vocab=256, n_text_ctx=32, n_text_state=64,
                   n_text_head=4, n_text_layer=2, sot_id=254,
                   no_timestamps_id=255)

    def parameter_count(self):
        """Learned parameters, counting tied token/output weights once."""
        a, d = self.n_audio_state, self.n_text_state
        return (3 * self.n_mels * a + a + 3 * a * a + a
                + self.n_audio_layer * (12 * a * a + 12 * a) + 2 * a
                + (self.n_vocab + self.n_text_ctx) * d
                + self.n_text_layer * (16 * d * d + 17 * d) + 2 * d)


class _Factory:
    def __init__(self, distribution, implementation, seed, device, dtype):
        if distribution not in {"binary", "ternary"}:
            raise ValueError("distribution must be binary or ternary")
        if implementation not in {"dense", "packed"}:
            raise ValueError("implementation must be dense or packed")
        self.distribution, self.implementation = distribution, implementation
        self.device, self.dtype = torch.device(device), dtype
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.parameter_count = 0
        self.vector_count = 0
        self.matrices = []

    def codes(self, shape):
        if self.distribution == "binary":
            return torch.randint(0, 2, shape, generator=self.generator,
                                 dtype=torch.int8).mul_(2).sub_(1)
        return torch.randint(-1, 2, shape, generator=self.generator,
                             dtype=torch.int8)

    def variance_scale(self, std):
        return std * (math.sqrt(1.5) if self.distribution == "ternary" else 1)

    def vector(self, size, std=0.01):
        self.parameter_count += size
        self.vector_count += size
        return (self.codes((size,)).float() * self.variance_scale(std)).to(
            device=self.device, dtype=self.dtype)

    def matrix(self, n, k, std=None):
        std = k ** -0.5 if std is None else std
        self.parameter_count += n * k
        weight = _Weight(self, n, k, self.variance_scale(std))
        self.matrices.append(weight)
        return weight


class _Weight:
    def __init__(self, factory, n, k, scale):
        self.n, self.k = n, k
        self.factory = factory
        codes = factory.codes((n, k))
        self.scale = float(scale)
        self.packed = None
        self.data = None
        if factory.implementation == "packed":
            from kernels.packed import PackedWeight
            # Packing and host/device copies occur during model construction.
            self.packed = PackedWeight.from_codes(
                codes, bits=1 if factory.distribution == "binary" else 2,
                scale=torch.tensor(scale, dtype=torch.float32)).to(factory.device)
        else:
            self.data = (codes.float() * scale).to(factory.device, factory.dtype)

    def linear(self, x, bias=None):
        if self.packed is not None:
            return self.packed.linear(x, bias=bias)
        return F.linear(x, self.data, bias)

    def embedding(self, ids):
        if self.packed is not None:
            return self.packed.embedding(ids)
        return F.embedding(ids, self.data)

    def assign(self, data):
        if self.packed is not None:
            raise ValueError("Pretrained checkpoint loading requires the dense runtime")
        if data.shape != (self.n, self.k):
            raise ValueError(f"Expected {(self.n, self.k)}, got {tuple(data.shape)}")
        self.data.copy_(data.to(device=self.data.device, dtype=self.data.dtype))

    @property
    def storage_bytes(self):
        if self.packed is not None:
            return self.packed.storage_bytes
        return self.data.numel() * self.data.element_size()


class _Norm:
    def __init__(self, factory, width):
        self.weight = factory.vector(width, std=1)
        self.bias = factory.vector(width)

    def __call__(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, eps=1e-5)


class _Attention:
    def __init__(self, factory, width, heads, cross=False):
        self.width, self.heads, self.cross = width, heads, cross
        # Fuse Q/K/V and K/V linear calls just as an optimized dense runtime does.
        if cross:
            self.q = factory.matrix(width, width)
            self.kv = factory.matrix(2 * width, width)
            self.q_bias = factory.vector(width)
            self.kv_bias = torch.cat((torch.zeros(width, device=factory.device,
                                                dtype=factory.dtype),
                                      factory.vector(width)))
        else:
            self.qkv = factory.matrix(3 * width, width)
            self.qkv_bias = torch.cat((factory.vector(width),
                                       torch.zeros(width, device=factory.device,
                                                   dtype=factory.dtype),
                                       factory.vector(width)))
        self.out = factory.matrix(width, width)
        self.out_bias = factory.vector(width)

    def split_heads(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads,
                         self.width // self.heads).transpose(1, 2)

    def memory_kv(self, memory):
        k, v = self.kv.linear(memory, self.kv_bias).chunk(2, dim=-1)
        return self.split_heads(k).contiguous(), self.split_heads(v).contiguous()

    def __call__(self, x, *, memory_kv=None, cache=None, step=None):
        if self.cross:
            q = self.split_heads(self.q.linear(x, self.q_bias))
            k, v = memory_kv
        else:
            q, k, v = self.qkv.linear(x, self.qkv_bias).chunk(3, dim=-1)
            q, k, v = map(self.split_heads, (q, k, v))
            if cache is not None:
                if step is None or x.shape[1] != 1:
                    raise ValueError("Cached attention requires one sequential input token")
                cache[0][:, :, step:step + 1, :].copy_(k)
                cache[1][:, :, step:step + 1, :].copy_(v)
                k, v = cache[0][:, :, :step + 1, :], cache[1][:, :, :step + 1, :]
        # A one-token query can attend every populated cache entry. is_causal=True
        # with Q length one would incorrectly mask all but the first key.
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0, is_causal=False)
        y = y.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.width)
        return self.out.linear(y, self.out_bias)


class _Block:
    def __init__(self, factory, width, heads, decoder=False):
        self.attn = _Attention(factory, width, heads)
        self.attn_ln = _Norm(factory, width)
        self.cross_attn = _Attention(factory, width, heads, cross=True) if decoder else None
        self.cross_attn_ln = _Norm(factory, width) if decoder else None
        self.mlp_ln = _Norm(factory, width)
        self.mlp_up = factory.matrix(4 * width, width)
        self.mlp_up_bias = factory.vector(4 * width)
        self.mlp_down = factory.matrix(width, 4 * width)
        self.mlp_down_bias = factory.vector(width)

    def __call__(self, x, *, memory_kv=None, cache=None, step=None):
        x = x + self.attn(self.attn_ln(x), cache=cache, step=step)
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.cross_attn_ln(x), memory_kv=memory_kv)
        y = F.gelu(self.mlp_up.linear(self.mlp_ln(x), self.mlp_up_bias))
        return x + self.mlp_down.linear(y, self.mlp_down_bias)


class ReplayWhisper:
    """Exact Whisper dimensions, tied embedding/logit weights, and growing KV.

    Matrices, convolutions and embeddings use seeded binary/ternary codes.
    Learned bias and normalization vectors also contain scaled codes but remain
    FP16 storage: tiny-vector unpacking is an explicitly recorded exception.
    Fixed encoder sinusoids retain the original architecture's definition.
    """

    def __init__(self, config=None, distribution="ternary", implementation="packed",
                 seed=20260911, device="cuda", dtype=torch.float16):
        self.config = config or WhisperConfig.medium_en()
        c = self.config
        if c.n_audio_state != c.n_text_state:
            raise ValueError("This Whisper implementation expects equal encoder/decoder width")
        self.distribution, self.implementation, self.seed = distribution, implementation, seed
        self.device, self.dtype = torch.device(device), dtype
        self.factory = f = _Factory(distribution, implementation, seed, device, dtype)
        self.conv1 = f.matrix(c.n_audio_state, 3 * c.n_mels)
        self.conv1_bias = f.vector(c.n_audio_state)
        self.conv2 = f.matrix(c.n_audio_state, 3 * c.n_audio_state)
        self.conv2_bias = f.vector(c.n_audio_state)
        timescales = torch.exp(-math.log(10000) * torch.arange(c.n_audio_state // 2)
                              / (c.n_audio_state // 2 - 1))
        angles = torch.arange(c.n_audio_ctx)[:, None] * timescales[None, :]
        self.encoder_position = torch.cat((angles.sin(), angles.cos()), dim=1).to(device, dtype)
        self.encoder_blocks = [_Block(f, c.n_audio_state, c.n_audio_head)
                               for _ in range(c.n_audio_layer)]
        self.encoder_ln = _Norm(f, c.n_audio_state)
        self.token_embedding = f.matrix(c.n_vocab, c.n_text_state, std=0.02)
        self.decoder_position = f.matrix(c.n_text_ctx, c.n_text_state, std=0.02)
        self.decoder_blocks = [_Block(f, c.n_text_state, c.n_text_head, decoder=True)
                               for _ in range(c.n_text_layer)]
        self.decoder_ln = _Norm(f, c.n_text_state)
        if f.parameter_count != c.parameter_count():
            raise RuntimeError("Constructed parameter count does not match the architecture")
        self._cache = None
        self._cache_steps = 0
        self._token_cache = {}
        self._graph = None
        self.pretrained_checkpoint = None

    def _convolve(self, x, weight, bias, stride):
        if self.implementation == "dense":
            return F.conv1d(x, weight.data.reshape(weight.n, x.shape[1], 3),
                            bias, stride=stride, padding=1)
        # The unpack stays inside the matrix kernel. im2col activation materialization
        # is included in candidate energy; this is a known convolution limitation.
        windows = F.pad(x, (1, 1)).unfold(-1, 3, stride)
        windows = windows.permute(0, 2, 1, 3).contiguous().flatten(2)
        return weight.linear(windows, bias).transpose(1, 2)

    @torch.inference_mode()
    def encode(self, mel):
        c = self.config
        if tuple(mel.shape) != (1, c.n_mels, 2 * c.n_audio_ctx):
            raise ValueError(f"Expected mel {(1, c.n_mels, 2 * c.n_audio_ctx)}, got {tuple(mel.shape)}")
        x = mel.to(device=self.device, dtype=self.dtype)
        x = F.gelu(self._convolve(x, self.conv1, self.conv1_bias, 1))
        x = F.gelu(self._convolve(x, self.conv2, self.conv2_bias, 2))
        x = x.transpose(1, 2) + self.encoder_position
        for block in self.encoder_blocks:
            x = block(x)
        return self.encoder_ln(x)

    def _prepare_decode(self, tokens: Sequence[int], sot_id, no_timestamps_id):
        tokens = tuple(int(t) for t in tokens)
        if not tokens or len(tokens) + 1 > self.config.n_text_ctx:
            raise ValueError("Replay requires 1..n_text_ctx-1 forced output tokens")
        if not all(0 <= t < self.config.n_vocab for t in tokens):
            raise ValueError("Replay token outside vocabulary")
        key = (tokens, sot_id, no_timestamps_id)
        if key not in self._token_cache:
            inputs = (sot_id, no_timestamps_id) + tokens[:-1]
            self._token_cache[key] = (
                torch.tensor(inputs, device=self.device, dtype=torch.long).view(-1, 1, 1),
                torch.arange(len(inputs), device=self.device, dtype=torch.long).view(-1, 1, 1))
        if self._cache_steps != len(tokens) + 1:
            shape = (1, self.config.n_text_head, len(tokens) + 1,
                     self.config.n_text_state // self.config.n_text_head)
            self._cache = [(torch.empty(shape, device=self.device, dtype=self.dtype),
                            torch.empty(shape, device=self.device, dtype=self.dtype))
                           for _ in self.decoder_blocks]
            self._cache_steps = len(tokens) + 1
        return self._token_cache[key]

    @torch.inference_mode()
    def decode(self, encoded, replay_tokens, sot_id=None, no_timestamps_id=None):
        sot_id = self.config.sot_id if sot_id is None else sot_id
        no_timestamps_id = self.config.no_timestamps_id if no_timestamps_id is None else no_timestamps_id
        inputs, positions = self._prepare_decode(replay_tokens, sot_id, no_timestamps_id)
        memory = [b.cross_attn.memory_kv(encoded) for b in self.decoder_blocks]
        predictions = []
        for step in range(inputs.shape[0]):
            x = self.token_embedding.embedding(inputs[step]) + self.decoder_position.embedding(positions[step])
            for i, block in enumerate(self.decoder_blocks):
                x = block(x, memory_kv=memory[i], cache=self._cache[i], step=step)
            x = self.decoder_ln(x)
            if step:
                logits = self.token_embedding.linear(x)
                # Keep vocabulary projection and greedy reduction at every step;
                # forced inputs control scheduling regardless of the predicted IDs.
                predictions.append(logits.argmax(dim=-1))
        self.last_predictions = torch.stack(predictions)
        return logits

    @torch.inference_mode()
    def run(self, mel, replay_tokens):
        return self.decode(self.encode(mel), replay_tokens)

    @torch.inference_mode()
    def capture(self, mel, replay_tokens):
        """Capture the shared runtime's complete fixed-shape CUDA graph, untimed.

        Use only while the GPU is available. Both dense and packed variants must
        use this same setting. Input host/device transfers remain the caller's
        measured responsibility; replay_graph() copies mel into the static input.
        """
        if self.device.type != "cuda":
            raise ValueError("CUDA graph capture requires CUDA")
        self._static_mel = mel.to(self.device, self.dtype).clone()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.run(self._static_mel, replay_tokens)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_output = self.run(self._static_mel, replay_tokens)
        return self

    @torch.inference_mode()
    def replay_graph(self, mel):
        if self._graph is None:
            raise RuntimeError("Call capture() before replay_graph()")
        self._static_mel.copy_(mel)
        self._graph.replay()
        return self._static_output

    @torch.inference_mode()
    def validate_outputs(self, mel, replay_tokens):
        encoded = self.encode(mel)
        logits = self.decode(encoded, replay_tokens)
        result = {}
        for name, tensor in (("encoded", encoded), ("logits", logits)):
            value = tensor.float()
            result[name] = {"finite": bool(value.isfinite().all()),
                            "std": float(value.std()), "abs_max": float(value.abs().max()),
                            "shape": list(value.shape)}
            if not result[name]["finite"] or result[name]["std"] == 0:
                raise RuntimeError(f"Nonfinite or degenerate {name}: {result[name]}")
        return result

    def metadata(self):
        decode_backend = "torch-dense"
        if self.implementation == "packed":
            from kernels import cuda_gemv
            if cuda_gemv.try_load() is None:
                decode_backend = "triton"
            else:
                decode_backend = ("cuda-gemv-with-triton-large-n-ternary"
                                  if self.distribution == "ternary" else "cuda-gemv")
        vectors_bytes = self.factory.vector_count * torch.empty((), dtype=self.dtype).element_size()
        element_bytes = torch.empty((), dtype=self.dtype).element_size()
        structural_padding_bytes = ((self.config.n_audio_layer * self.config.n_audio_state
                                     + 2 * self.config.n_text_layer * self.config.n_text_state)
                                    * element_bytes)
        return {"engine": "pytorch-sdpa-packed" if self.implementation == "packed" else "pytorch-sdpa-dense",
                "architecture": asdict(self.config), "distribution": self.distribution,
                "seed": self.seed, "implementation": self.implementation,
                "decode_backend": decode_backend,
                "parameter_count": self.factory.parameter_count,
                "learned_vector_parameters_stored_dense": self.factory.vector_count,
                "weight_storage_bytes": sum(w.storage_bytes for w in self.factory.matrices) + vectors_bytes,
                "additional_fixed_encoder_position_bytes": self.encoder_position.numel() * element_bytes,
                "additional_structural_zero_bias_bytes": structural_padding_bytes,
                "activation_dtype": str(self.dtype), "matrix_accumulation": "float32",
                "scale_definition": "fan-in standard deviation; ternary multiplied by sqrt(3/2); embeddings std0.02; vectors std0.01; layernorm gamma std1",
                "ternary_zero_probability": 1 / 3 if self.distribution == "ternary" else 0,
                "cuda_graph": self._graph is not None,
                "pretrained_checkpoint": self.pretrained_checkpoint,
                "exceptions": ["Bias and layernorm vectors contain random codes but use FP16 storage",
                               "Structural missing key biases are zero and not counted as parameters",
                               "Encoder positions are fixed sinusoids",
                               "Packed convolutions materialize activation windows before packed matmul",
                               "Shared candidate runtime is PyTorch SDPA, not CTranslate2; compare dense twins and external optimized controls",
                               "CTranslate2 synchronizes host sampling per token; candidates retain predicted IDs on GPU"]}

    @torch.inference_mode()
    def load_hf_safetensors(self, directory):
        """Load original Hugging Face Whisper weights into the dense replay graph.

        This optional calibration quantifies the shared runtime's overhead against
        stock CTranslate2 using identical pretrained weights and forced tokens.
        """
        if self.implementation != "dense":
            raise ValueError("Pretrained calibration requires implementation='dense'")
        from safetensors import safe_open
        directory = Path(directory)
        model_file = directory / "model.safetensors"
        if not model_file.is_file():
            raise FileNotFoundError(model_file)
        with safe_open(model_file, framework="pt", device="cpu") as checkpoint:
            def tensor(key):
                return checkpoint.get_tensor("model." + key)

            def vector(dest, key):
                source = tensor(key)
                if source.shape != dest.shape:
                    raise ValueError(f"Shape mismatch for {key}")
                dest.copy_(source.to(dest.device, dest.dtype))

            def norm(dest, key):
                vector(dest.weight, key + ".weight")
                vector(dest.bias, key + ".bias")

            for i in (1, 2):
                weight = getattr(self, f"conv{i}")
                weight.assign(tensor(f"encoder.conv{i}.weight").flatten(1))
                vector(getattr(self, f"conv{i}_bias"), f"encoder.conv{i}.bias")
            self.encoder_position.copy_(tensor("encoder.embed_positions.weight").to(self.device, self.dtype))
            norm(self.encoder_ln, "encoder.layer_norm")
            self.token_embedding.assign(tensor("decoder.embed_tokens.weight"))
            self.decoder_position.assign(tensor("decoder.embed_positions.weight"))
            norm(self.decoder_ln, "decoder.layer_norm")
            for side, blocks in (("encoder", self.encoder_blocks), ("decoder", self.decoder_blocks)):
                for i, block in enumerate(blocks):
                    prefix = f"{side}.layers.{i}"
                    a = prefix + ".self_attn"
                    block.attn.qkv.assign(torch.cat([tensor(a + f".{p}_proj.weight") for p in ("q", "k", "v")]))
                    block.attn.qkv_bias.copy_(torch.cat((tensor(a + ".q_proj.bias"),
                                                       torch.zeros(self.config.n_text_state),
                                                       tensor(a + ".v_proj.bias"))).to(self.device, self.dtype))
                    block.attn.out.assign(tensor(a + ".out_proj.weight"))
                    vector(block.attn.out_bias, a + ".out_proj.bias")
                    norm(block.attn_ln, prefix + ".self_attn_layer_norm")
                    if side == "decoder":
                        a = prefix + ".encoder_attn"
                        block.cross_attn.q.assign(tensor(a + ".q_proj.weight"))
                        vector(block.cross_attn.q_bias, a + ".q_proj.bias")
                        block.cross_attn.kv.assign(torch.cat([tensor(a + f".{p}_proj.weight") for p in ("k", "v")]))
                        block.cross_attn.kv_bias.copy_(torch.cat((torch.zeros(self.config.n_text_state),
                                                               tensor(a + ".v_proj.bias"))).to(self.device, self.dtype))
                        block.cross_attn.out.assign(tensor(a + ".out_proj.weight"))
                        vector(block.cross_attn.out_bias, a + ".out_proj.bias")
                        norm(block.cross_attn_ln, prefix + ".encoder_attn_layer_norm")
                    norm(block.mlp_ln, prefix + ".final_layer_norm")
                    block.mlp_up.assign(tensor(prefix + ".fc1.weight"))
                    vector(block.mlp_up_bias, prefix + ".fc1.bias")
                    block.mlp_down.assign(tensor(prefix + ".fc2.weight"))
                    vector(block.mlp_down_bias, prefix + ".fc2.bias")
        self.pretrained_checkpoint = str(directory.resolve())
        self.distribution = "pretrained"
        return self
