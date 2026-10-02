"""Pure-PyTorch FP32 reference of nvidia/parakeet-tdt-0.6b-v2 (no NeMo import). See DESIGN.md "Implementation decisions".

Mirrors NeMo 3.0 at inference (eval mode, dither off, pad_to as configured, 0 for this model) op for
op, so that on the same weights and input it agrees with NeMo to float rounding. Runs on torch 2.7
(Mac uv env) and the NeMo env's torch (NixOS). Module and parameter names equal NeMo's
EncDecRNNTBPEModel state_dict keys under preprocessor./encoder./decoder./joint.

- Front end (AudioToMelSpectrogramPreprocessor / FilterbankFeatures): preemphasis 0.97 within the
  valid samples, torch.stft (n_fft 512, hop 160, centred zero padding, 400-sample symmetric Hann
  window), |X|^2 via sqrt then square, Slaney mel filterbank (128 x 257), log(x + 2^-24),
  per-feature normalization over the valid frames (unbiased std + 1e-5), padded frames set to 0.
  Feature length = samples // 160. hann_window() and mel_filterbank() compute the two constants in
  numpy; NVIDIA's checkpoint stores its own copies, which differ from any portable computation by
  at most one float32 ulp in a few entries (an older librosa and float32 torch arithmetic).
  load_weights() uses the stored copy when the source has one (after a closeness check), so parity
  with NeMo is on identical constants.
- Encoder (ConformerEncoder): dw_striding 8x subsampling (Conv2d 3x3/2 + ReLU, then twice depthwise
  3x3/2 + pointwise 1x1 + ReLU, each layer input masked to the valid frames as NeMo's
  MaskedConvSequential does), flatten 256 x 16 -> Linear 4096 -> 1024; relative positional
  encoding (no xscale); n_layers x ConformerLayer: x + 0.5 FF1, x + RelPosMHSA (untied
  pos_bias_u/v, Transformer-XL rel_shift, scores / sqrt(128), masked fill -1e4), x + conv module
  (pointwise 1024->2048, GLU, padded frames zeroed, depthwise k=9, BatchNorm, SiLU, pointwise),
  x + 0.5 FF2, LayerNorm. No biases in the layers (use_bias false).
- Prediction network (RNNTDecoder): Embedding 1025 x 640 (blank 1024 is the padding row and the
  start symbol), 2-layer LSTM 640.
- Joint (RNNTJoint): enc 1024->640 and pred 640->640 projections, sum, ReLU, Linear 640 -> 1030
  (1024 tokens, blank, 5 duration logits). With log_softmax None NeMo log-softmaxes the whole
  1030-vector on CPU; decisions here use that same vector (the shift never changes an argmax except
  in exact float ties).
- Greedy TDT (the config's strategy greedy_batch: GreedyBatchedTDTInfer ->
  GreedyBatchedTDTLabelLoopingComputer.torch_impl), reset for every utterance and started as
  GreedyTDTInfer starts: zero LSTM state and the start symbol as a zero input vector (NeMo's
  predict(None); label looping embeds the blank instead, the same thing because the blank row is
  the zero padding row, which load_weights checks), run per utterance as one linear sequence of
  joint evaluations ("steps"): the token is the argmax over tokens+blank, the duration the
  durations[argmax] of the duration head; a blank with duration 0 advances 1 frame; after an
  emitted token the prediction network is run on it, and if the number of tokens emitted at the
  same frame reaches max_symbols (10) while the frame did not advance, the frame advances by 1;
  decoding stops when the frame reaches the encoder length. This is NeMo's label-looping
  semantics. (The frame-looping GreedyTDTInfer used by strategy "greedy" differs in corner cases,
  e.g. a blank of duration 0 there repeats the joint call until max_symbols.)

greedy_decode() returns a Trace per utterance: every step's frame, prediction-net input token,
chosen token and duration, emitted flag, symbols-at-frame counter, forced advance and whether the
prediction state was updated. replay() consumes a Trace, feeds its decisions (not its own argmax)
to the prediction network and returns per step the raw joint logits, the LSTM state that produced
the prediction output, and its own argmax, so a model under test is compared on logits before the
decisions are overridden.

Ternary modules may be supplied as int8 codes [out, in] in {-1, 0, 1} plus FP32 per-row scales
[out]; they are dequantized in place into the model's single dense FP32 weight (one module at a
time, no second dense copy).
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

PREFIXES = ("preprocessor.", "encoder.", "decoder.", "joint.")
TERNARY_SUFFIXES = ("feed_forward1.linear1", "feed_forward1.linear2", "feed_forward2.linear1",
                    "feed_forward2.linear2", "self_attn.linear_q", "self_attn.linear_k", "self_attn.linear_v",
                    "self_attn.linear_out", "self_attn.linear_pos", "conv.pointwise_conv1",
                    "conv.pointwise_conv2")
NORM_CONSTANT = 1e-5          # features.CONSTANT: std guard of per_feature normalization
INF_VAL = 10000.0             # multi_head_attention.INF_VAL
WINDOW_TOLERANCE = 1e-6       # max |computed - stored| accepted for the Hann window
FILTERBANK_TOLERANCE = 1e-8   # max |computed - stored| accepted for the mel filterbank


# --- configuration ---------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Config:
    """The architecture constants this reference implements; from_model_config() validates a NeMo config."""
    sample_rate: int = 16000
    win_length: int = 400
    hop_length: int = 160
    n_fft: int = 512
    features: int = 128
    preemph: float = 0.97
    log_guard: float = 2.0 ** -24
    pad_to: int = 0
    pad_value: float = 0.0
    n_layers: int = 24
    d_model: int = 1024
    n_heads: int = 8
    ff_expansion: int = 4
    subsampling_factor: int = 8
    subsampling_channels: int = 256
    conv_kernel_size: int = 9
    vocab_size: int = 1024
    pred_hidden: int = 640
    pred_rnn_layers: int = 2
    joint_hidden: int = 640
    durations: tuple[int, ...] = (0, 1, 2, 3, 4)
    max_symbols: int = 10

    @property
    def blank(self) -> int:
        return self.vocab_size

    @property
    def num_outputs(self) -> int:
        return self.vocab_size + 1 + len(self.durations)

    @classmethod
    def from_model_config(cls, cfg: Mapping, num_layers: int | None = None) -> Config:
        """Config from a NeMo model config (model_config.yaml as a dict); ValueError on anything not implemented."""
        pre, enc, dec, joint, decoding = (cfg[k] for k in ("preprocessor", "encoder", "decoder", "joint", "decoding"))
        expect = {
            "preprocessor": (pre, {"normalize": "per_feature", "window": "hann", "log": True, "frame_splicing": 1,
                                   "exact_pad": False, "mel_norm": "slaney", "lowfreq": 0, "highfreq": None,
                                   "log_zero_guard_type": "add", "log_zero_guard_value": 2 ** -24, "mag_power": 2.0,
                                   "preemph": 0.97, "nb_augmentation_prob": 0.0}),
            "encoder": (enc, {"feat_out": -1, "use_bias": False, "subsampling": "dw_striding",
                              "causal_downsampling": False, "reduction": None, "self_attention_model": "rel_pos",
                              "att_context_size": [-1, -1], "att_context_style": "regular", "xscaling": False,
                              "untie_biases": True, "conv_norm_type": "batch_norm", "conv_context_size": None,
                              "subsampling_conv_chunking_factor": 1}),
            "decoder": (dec, {"normalization_mode": None, "random_state_sampling": False, "blank_as_pad": True}),
            "joint": (joint, {"log_softmax": None}),
            "decoding": (decoding, {"strategy": "greedy_batch", "model_type": "tdt"}),
        }
        defaults = {"exact_pad": False, "mel_norm": "slaney", "lowfreq": 0, "highfreq": None,
                    "log_zero_guard_type": "add", "log_zero_guard_value": 2 ** -24, "mag_power": 2.0,
                    "preemph": 0.97, "nb_augmentation_prob": 0.0, "subsampling_conv_chunking_factor": 1}
        for section, (values, wanted) in expect.items():
            for key, want in wanted.items():
                got = values.get(key, defaults.get(key))
                if got != want:
                    raise ValueError(f"{section}.{key} = {got!r} is not implemented (expected {want!r})")
        if joint["jointnet"]["activation"] != "relu" or joint["jointnet"]["encoder_hidden"] != enc["d_model"]:
            raise ValueError("joint network must be relu with encoder_hidden == d_model")
        if joint["num_classes"] != dec["vocab_size"] or joint["jointnet"]["pred_hidden"] != dec["prednet"]["pred_hidden"]:
            raise ValueError("joint and decoder sizes disagree")
        durations = tuple(int(d) for d in decoding["durations"])
        if joint["num_extra_outputs"] != len(durations):
            raise ValueError("num_extra_outputs must equal the number of TDT durations")
        if pre.get("n_window_size") or pre.get("n_window_stride"):
            raise ValueError("n_window_size/n_window_stride are not implemented (window_size/window_stride are)")
        sr = int(pre["sample_rate"])
        pad_to = pre.get("pad_to", 16)
        if not isinstance(pad_to, int) or pad_to < 0:
            raise ValueError(f"pad_to {pad_to!r} is not implemented")
        layers = int(enc["n_layers"]) if num_layers is None else int(num_layers)
        if not 1 <= layers <= int(enc["n_layers"]):
            raise ValueError(f"num_layers must be in 1..{enc['n_layers']}")
        greedy = decoding.get("greedy") or {}
        max_symbols = greedy.get("max_symbols") or greedy.get("max_symbols_per_step")
        if max_symbols is None:
            raise ValueError("decoding.greedy.max_symbols must be set")
        return cls(sample_rate=sr, win_length=int(pre["window_size"] * sr), hop_length=int(pre["window_stride"] * sr),
                   n_fft=int(pre["n_fft"]), features=int(pre["features"]), pad_to=pad_to,
                   pad_value=float(pre.get("pad_value", 0.0)), n_layers=layers, d_model=int(enc["d_model"]),
                   n_heads=int(enc["n_heads"]), ff_expansion=int(enc["ff_expansion_factor"]),
                   subsampling_factor=int(enc["subsampling_factor"]),
                   subsampling_channels=int(enc["subsampling_conv_channels"]),
                   conv_kernel_size=int(enc["conv_kernel_size"]), vocab_size=int(dec["vocab_size"]),
                   pred_hidden=int(dec["prednet"]["pred_hidden"]), pred_rnn_layers=int(dec["prednet"]["pred_rnn_layers"]),
                   joint_hidden=int(joint["jointnet"]["joint_hidden"]), durations=durations, max_symbols=int(max_symbols))


def load_model_config(path: str | Path) -> dict:
    """The NeMo model config from a model_config.yaml or a weight_stats.json ("model_config" entry)."""
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())["model_config"]
    import yaml
    return yaml.safe_load(path.read_text())


# --- front-end constants (numpy) ---------------------------------------------------------------------

def hann_window(length: int) -> np.ndarray:
    """Symmetric Hann window (torch.hann_window(length, periodic=False)), computed in float64, float32 result."""
    n = np.arange(length, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / (length - 1))).astype(np.float32)


def _hz_to_mel(freq: float) -> float:
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    if freq >= min_log_hz:
        return min_log_hz / f_sp + math.log(freq / min_log_hz) / (math.log(6.4) / 27.0)
    return freq / f_sp


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    freqs = f_sp * mels
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Slaney-scale, Slaney-normalized mel filterbank [n_mels, n_fft // 2 + 1] float32, as librosa.filters.mel
    (fmin 0, fmax sr/2, htk False) computes it: float64 ramps, float32 weights, float32 normalization product."""
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sample_rate)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sample_rate / 2.0), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    weights = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        weights[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights


# --- modules -------------------------------------------------------------------------------------------

class Featurizer(nn.Module):
    """FilterbankFeatures at inference: audio [B, N] float32 + lengths [B] -> log-mel [B, features, frames], lengths."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("window", torch.from_numpy(hann_window(cfg.win_length)))
        self.register_buffer("fb", torch.from_numpy(mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.features))[None])

    def reset_constants(self) -> None:
        self.window = torch.from_numpy(hann_window(self.cfg.win_length))
        self.fb = torch.from_numpy(mel_filterbank(self.cfg.sample_rate, self.cfg.n_fft, self.cfg.features))[None]

    def get_seq_len(self, lengths: Tensor) -> Tensor:
        return torch.floor_divide(lengths + self.cfg.n_fft // 2 * 2 - self.cfg.n_fft, self.cfg.hop_length).to(torch.long)

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.cfg
        seq_len = self.get_seq_len(lengths)
        seq_len = torch.where(lengths == 0, torch.zeros_like(seq_len), seq_len)
        timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - cfg.preemph * x[:, :-1]), dim=1)
        x = x.masked_fill(~timemask, 0.0)
        x = torch.stft(x, n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length, center=True,
                       window=self.window.to(dtype=torch.float, device=x.device), return_complex=True,
                       pad_mode="constant")
        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1))
        x = x.pow(2.0)
        x = torch.matmul(self.fb.to(x.dtype), x)
        x = torch.log(x + cfg.log_guard)
        x = normalize_per_feature(x, seq_len)
        mask = torch.arange(x.size(-1), device=x.device).repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1), cfg.pad_value)
        if cfg.pad_to > 0 and x.size(-1) % cfg.pad_to:
            x = F.pad(x, (0, cfg.pad_to - x.size(-1) % cfg.pad_to), value=cfg.pad_value)
        return x, seq_len


def normalize_per_feature(x: Tensor, seq_len: Tensor) -> Tensor:
    """features.normalize_batch(..., "per_feature"): mean and unbiased std over valid frames, std + 1e-5."""
    batch, _, max_time = x.shape
    valid = torch.arange(max_time, device=x.device).unsqueeze(0).expand(batch, max_time) < seq_len.unsqueeze(1)
    mean = torch.where(valid.unsqueeze(1), x, 0.0).sum(axis=2) / valid.sum(axis=1).unsqueeze(1)
    std = torch.sqrt(torch.sum(torch.where(valid.unsqueeze(1), x - mean.unsqueeze(2), 0.0) ** 2, axis=2)
                     / (valid.sum(axis=1).unsqueeze(1) - 1.0))
    std = std.masked_fill(std.isnan(), 0.0) + NORM_CONSTANT
    out = (x - mean.unsqueeze(2)) / std.unsqueeze(2)
    return out.masked_fill(~valid.unsqueeze(1), 0.0)


class Preprocessor(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.featurizer = Featurizer(cfg)

    def forward(self, audio: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        return self.featurizer(audio.to(torch.float32), lengths)


class Subsampling(nn.Module):
    """ConvSubsampling "dw_striding" with MaskedConvSequential masking; [B, T, F] -> [B, T / 8, d_model]."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        c, stages = cfg.subsampling_channels, int(math.log2(cfg.subsampling_factor))
        layers: list[nn.Module] = [nn.Conv2d(1, c, 3, stride=2, padding=1), nn.ReLU()]
        for _ in range(stages - 1):
            layers += [nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c), nn.Conv2d(c, c, 1), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        freq = cfg.features
        for _ in range(stages):
            freq = (freq + 2 - 3) // 2 + 1
        self.out = nn.Linear(c * freq, cfg.d_model)

    def forward(self, x: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        x = x.unsqueeze(1)
        current = lengths.clone().float()
        mask = _time_mask(x, current.long())
        for layer in self.conv:
            x = x * mask.unsqueeze(1).expand_as(x)
            x = layer(x)
            if isinstance(layer, nn.Conv2d) and layer.stride != (1, 1):
                current = (current + layer.padding[0] + layer.padding[1] - layer.kernel_size[0]) // layer.stride[0] + 1
                mask = _time_mask(x, current.long())
        x = x * mask.unsqueeze(1).expand_as(x)
        b, _, t, _ = x.size()
        return self.out(x.transpose(1, 2).reshape(b, t, -1)), current.long()


def _time_mask(x: Tensor, lengths: Tensor) -> Tensor:
    batch, _, time, freq = x.shape
    valid = torch.arange(time, device=x.device).expand(batch, time) < lengths.unsqueeze(1)
    return valid.unsqueeze(-1).expand(batch, time, freq).to(x.dtype)


def rel_positional_embedding(length: int, d_model: int, device: torch.device | str = "cpu") -> Tensor:
    """RelPositionalEncoding's pos_emb for an input of `length` frames: positions length-1 .. -(length-1), [1, 2L-1, d]."""
    positions = torch.arange(length - 1, -length, -1, dtype=torch.float32, device=device).unsqueeze(1)
    pe = torch.zeros(positions.size(0), d_model, device=device)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
                         * -(math.log(INF_VAL) / d_model))
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe.unsqueeze(0)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(F.silu(self.linear1(x)))


class RelPositionAttention(nn.Module):
    """RelPositionMultiHeadAttention (manual attention path, use_pytorch_sdpa False)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.h, self.d_k = n_heads, d_model // n_heads
        self.s_d_k = math.sqrt(self.d_k)
        for name in ("linear_q", "linear_k", "linear_v", "linear_out", "linear_pos"):
            setattr(self, name, nn.Linear(d_model, d_model, bias=False))
        self.pos_bias_u = nn.Parameter(torch.empty(n_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.empty(n_heads, self.d_k))

    @staticmethod
    def rel_shift(x: Tensor) -> Tensor:
        b, h, qlen, pos_len = x.size()
        x = F.pad(x, pad=(1, 0)).view(b, h, -1, qlen)
        return x[:, :, 1:].view(b, h, qlen, pos_len)

    def forward(self, x: Tensor, mask: Tensor, pos_emb: Tensor) -> Tensor:
        n_batch = x.size(0)
        q = self.linear_q(x).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        q = q.transpose(1, 2)
        p = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, self.h, self.d_k).transpose(1, 2)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_bd = self.rel_shift(torch.matmul(q_with_bias_v, p.transpose(-2, -1)))
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = matrix_bd[:, :, :, :matrix_ac.size(-1)]
        scores = (matrix_ac + matrix_bd) / self.s_d_k
        mask = mask.unsqueeze(1)
        attn = torch.softmax(scores.masked_fill(mask, -INF_VAL), dim=-1).masked_fill(mask, 0.0)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)
        return self.linear_out(out)


class ConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) // 2
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, 1, bias=False)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, bias=False)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1, bias=False)

    def forward(self, x: Tensor, pad_mask: Tensor) -> Tensor:
        x = F.glu(self.pointwise_conv1(x.transpose(1, 2)), dim=1)
        x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = self.depthwise_conv(F.pad(x, pad=(self.pad, self.pad)))
        x = self.pointwise_conv2(F.silu(self.batch_norm(x)))
        return x.transpose(1, 2)


class ConformerLayer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        d = cfg.d_model
        self.norm_feed_forward1 = nn.LayerNorm(d)
        self.feed_forward1 = FeedForward(d, d * cfg.ff_expansion)
        self.norm_conv = nn.LayerNorm(d)
        self.conv = ConvModule(d, cfg.conv_kernel_size)
        self.norm_self_att = nn.LayerNorm(d)
        self.self_attn = RelPositionAttention(d, cfg.n_heads)
        self.norm_feed_forward2 = nn.LayerNorm(d)
        self.feed_forward2 = FeedForward(d, d * cfg.ff_expansion)
        self.norm_out = nn.LayerNorm(d)

    def forward(self, x: Tensor, att_mask: Tensor, pos_emb: Tensor, pad_mask: Tensor) -> Tensor:
        residual = x + self.feed_forward1(self.norm_feed_forward1(x)) * 0.5
        residual = residual + self.self_attn(self.norm_self_att(residual), att_mask, pos_emb)
        residual = residual + self.conv(self.norm_conv(residual), pad_mask)
        residual = residual + self.feed_forward2(self.norm_feed_forward2(residual)) * 0.5
        return self.norm_out(residual)


class Encoder(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.pre_encode = Subsampling(cfg)
        self.layers = nn.ModuleList(ConformerLayer(cfg) for _ in range(cfg.n_layers))

    def forward(self, features: Tensor, lengths: Tensor, return_hidden: bool = False):
        """features [B, F, T_mel] -> encoded [B, d_model, T], lengths [B] (+ {"pre_encode", "layers"} if return_hidden)."""
        x, lengths = self.pre_encode(features.transpose(1, 2), lengths)
        hidden = {"pre_encode": x, "layers": []}
        time = x.size(1)
        pos_emb = rel_positional_embedding(time, self.cfg.d_model, x.device)
        valid = torch.arange(time, device=x.device).expand(lengths.size(0), -1) < lengths.unsqueeze(-1)
        pair = valid.unsqueeze(1).repeat([1, time, 1])
        att_mask = ~torch.logical_and(pair, pair.transpose(1, 2))
        pad_mask = ~valid
        for layer in self.layers:
            x = layer(x, att_mask, pos_emb, pad_mask)
            if return_hidden:
                hidden["layers"].append(x)
        out = (x.transpose(1, 2), lengths.to(torch.int64))
        return (*out, hidden) if return_hidden else out


class DecRNN(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.lstm = nn.LSTM(cfg.pred_hidden, cfg.pred_hidden, cfg.pred_rnn_layers)


class Decoder(nn.Module):
    """RNNTDecoder prediction network; predict() is NeMo's predict(y, state, add_sos=False)."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.prediction = nn.ModuleDict({"embed": nn.Embedding(cfg.vocab_size + 1, cfg.pred_hidden,
                                                               padding_idx=cfg.blank),
                                         "dec_rnn": DecRNN(cfg)})

    def initial_state(self, batch: int = 1) -> tuple[Tensor, Tensor]:
        shape = (self.cfg.pred_rnn_layers, batch, self.cfg.pred_hidden)
        return torch.zeros(shape), torch.zeros(shape)

    def predict(self, labels: Tensor | None, state: tuple[Tensor, Tensor] | None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """labels [B, U] -> output [B, U, H], (h, c) each [layers, B, H]. labels None is the start symbol as
        GreedyTDTInfer feeds it (predict(None): a zero input vector, batch 1 or the state's)."""
        if labels is None:
            batch = 1 if state is None else state[0].size(1)
            y = torch.zeros(1, batch, self.cfg.pred_hidden)
        else:
            y = self.prediction["embed"](labels).transpose(0, 1)
        g, hid = self.prediction["dec_rnn"].lstm(y, state)
        return g.transpose(0, 1), hid

    def forward(self, targets: Tensor) -> Tensor:
        """Teacher-forced outputs with the start symbol prepended (NeMo forward, add_sos): [B, U] -> [B, U + 1, H]."""
        y = self.prediction["embed"](targets)
        y = torch.cat([torch.zeros_like(y[:, :1]), y], dim=1).transpose(0, 1)
        g, _ = self.prediction["dec_rnn"].lstm(y, None)
        return g.transpose(0, 1)


class Joint(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.pred = nn.Linear(cfg.pred_hidden, cfg.joint_hidden)
        self.enc = nn.Linear(cfg.d_model, cfg.joint_hidden)
        self.joint_net = nn.Sequential(nn.ReLU(), nn.Identity(), nn.Linear(cfg.joint_hidden, cfg.num_outputs))

    def joint_after_projection(self, f: Tensor, g: Tensor, log_softmax: bool = False) -> Tensor:
        """f [B, T, H] projected encoder, g [B, U, H] projected prediction -> [B, T, U, outputs] (raw logits by default)."""
        res = self.joint_net(f.unsqueeze(2) + g.unsqueeze(1))
        return res.log_softmax(dim=-1) if log_softmax else res

    def forward(self, encoder_output: Tensor, decoder_output: Tensor, log_softmax: bool = True) -> Tensor:
        """encoder [B, T, D], decoder [B, U, H] -> [B, T, U, outputs]; log_softmax=True is NeMo's joint() on CPU."""
        return self.joint_after_projection(self.enc(encoder_output), self.pred(decoder_output), log_softmax)


class ParakeetReference(nn.Module):
    """preprocessor + encoder + decoder + joint with NeMo state_dict names. Build with build(); fill with load_weights()."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.preprocessor = Preprocessor(cfg)
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.joint = Joint(cfg)

    def forward(self, audio: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """audio [B, N] float32 at 16 kHz, lengths [B] -> encoder output [B, d_model, T], encoded lengths [B]."""
        features, feat_len = self.preprocessor(audio, lengths)
        return self.encoder(features, feat_len)


def build(cfg: Config) -> ParakeetReference:
    """Allocate an eval-mode FP32 model with uninitialized weights (no random init); load_weights() fills it."""
    with torch.device("meta"):
        model = ParakeetReference(cfg)
    model = model.to_empty(device="cpu")
    model.preprocessor.featurizer.reset_constants()
    return model.eval()


# --- weights ----------------------------------------------------------------------------------------------

def is_ternary(key: str) -> bool:
    """True for a quantized module's weight key encoder.layers.N.<one of TERNARY_SUFFIXES>.weight."""
    parts = key.split(".")
    return (len(parts) >= 5 and parts[0] == "encoder" and parts[1] == "layers" and parts[-1] == "weight"
            and ".".join(parts[3:-1]) in TERNARY_SUFFIXES)


def _layer_index(key: str) -> int | None:
    parts = key.split(".")
    return int(parts[2]) if len(parts) > 2 and parts[:2] == ["encoder", "layers"] else None


class _Source:
    """Uniform read access to a dict of tensors/arrays or a safetensors file (one tensor in memory at a time)."""

    def __init__(self, source: Mapping | str | Path) -> None:
        if isinstance(source, (str, Path)):
            from safetensors import safe_open
            self._file = safe_open(str(source), framework="pt")
            self.keys = set(self._file.keys())
            self.get = lambda k: self._file.get_tensor(k)
        else:
            self.keys = set(source.keys())
            self.get = lambda k: torch.as_tensor(source[k])


@torch.no_grad()
def load_weights(model: ParakeetReference, source: Mapping | str | Path) -> dict:
    """Fill every parameter and buffer of model from source (NeMo state_dict names); return a report.

    A ternary weight may be given as "<module>.codes" (int8 [out, in], values in {-1, 0, 1}) plus
    "<module>.scale" (FP32 [out]) instead of "<module>.weight"; it is dequantized in place. Keys of
    encoder layers beyond model.cfg.n_layers and keys outside preprocessor/encoder/decoder/joint are
    ignored (counted in the report); any other unused or missing key raises. The window and
    filterbank, if present, must agree with the computed ones within WINDOW_TOLERANCE /
    FILTERBANK_TOLERANCE and are then used as given.
    """
    src = _Source(source)
    used: set[str] = set()
    report = {"ternary_modules": 0, "dense_tensors": 0, "ignored_deeper_layers": 0, "ignored_other": 0}
    persistent = set(model.state_dict().keys())
    targets = dict(model.named_parameters())
    targets.update((k, v) for k, v in model.named_buffers() if k in persistent)
    for key, target in targets.items():
        prefix = key[: -len(".weight")]
        if key in src.keys:
            value = src.get(key)
            used.add(key)
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(f"{key}: shape {tuple(value.shape)} != {tuple(target.shape)}")
            if key.startswith("preprocessor.featurizer."):
                tol = WINDOW_TOLERANCE if key.endswith("window") else FILTERBANK_TOLERANCE
                diff = float((value.float() - target).abs().max())
                report[f"{key.rsplit('.', 1)[1]}_max_abs_diff_vs_computed"] = diff
                report[f"{key.rsplit('.', 1)[1]}_entries_differing_from_computed"] = int((value.float() != target).sum())
                if diff > tol:
                    raise ValueError(f"{key} differs from the computed constant by {diff} > {tol}")
            target.copy_(value)
            report["dense_tensors"] += 1
        elif is_ternary(key) and f"{prefix}.codes" in src.keys:
            codes, scale = src.get(f"{prefix}.codes"), src.get(f"{prefix}.scale")
            used.update((f"{prefix}.codes", f"{prefix}.scale"))
            out_features = target.shape[0]
            if codes.dtype != torch.int8 or tuple(codes.shape) != (out_features, math.prod(target.shape[1:])):
                raise ValueError(f"{prefix}.codes must be int8 [{out_features}, {math.prod(target.shape[1:])}]")
            if scale.dtype != torch.float32 or tuple(scale.shape) != (out_features,):
                raise ValueError(f"{prefix}.scale must be float32 [{out_features}]")
            if bool(((codes < -1) | (codes > 1)).any()):
                raise ValueError(f"{prefix}.codes has values outside {{-1, 0, 1}}")
            matrix = target.view(out_features, -1)
            matrix.copy_(codes)
            matrix.mul_(scale[:, None])
            report["ternary_modules"] += 1
        else:
            raise KeyError(f"source has no value for {key}")
    blank_row = model.decoder.prediction["embed"].weight[model.cfg.blank] if hasattr(model, "decoder") else None
    if blank_row is not None and bool(blank_row.any()):
        raise ValueError("the blank embedding row (padding_idx, start symbol) must be zero")
    for key in src.keys - used:
        layer = _layer_index(key)
        if layer is not None and layer >= model.cfg.n_layers:
            report["ignored_deeper_layers"] += 1
        elif not key.startswith(PREFIXES):
            report["ignored_other"] += 1
        else:
            raise KeyError(f"unused source key {key}")
    return report


# --- greedy TDT decoding, trace and replay ----------------------------------------------------------------

TRACE_FIELDS = ("frame", "pred_input", "token", "duration", "emitted", "pred_updated", "symbols_at_frame",
                "forced_advance", "advance")


@dataclasses.dataclass
class Trace:
    """Complete greedy TDT execution of one utterance; one entry per joint evaluation (step).

    frame: encoder frame of the joint evaluation; pred_input: token whose prediction-net output the
    step used (the blank/start symbol before the first emission); token: chosen token (blank =
    vocab_size); duration: chosen duration value (durations[argmax], before the blank 0 -> 1 rule);
    emitted: token is not blank (appended to the hypothesis); pred_updated: the prediction network
    was run on token after this step; symbols_at_frame: tokens emitted so far at the frame of the
    latest emission (NeMo's last_nb_timestamp_lasts); forced_advance: the max-symbols rule added a
    frame; advance: frames advanced after this step in total.
    """
    num_frames: int
    frame: list[int] = dataclasses.field(default_factory=list)
    pred_input: list[int] = dataclasses.field(default_factory=list)
    token: list[int] = dataclasses.field(default_factory=list)
    duration: list[int] = dataclasses.field(default_factory=list)
    emitted: list[bool] = dataclasses.field(default_factory=list)
    pred_updated: list[bool] = dataclasses.field(default_factory=list)
    symbols_at_frame: list[int] = dataclasses.field(default_factory=list)
    forced_advance: list[bool] = dataclasses.field(default_factory=list)
    advance: list[int] = dataclasses.field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def tokens(self) -> list[int]:
        return [t for t, e in zip(self.token, self.emitted) if e]

    @property
    def token_durations(self) -> list[int]:
        return [d for d, e in zip(self.duration, self.emitted) if e]

    @property
    def timestamps(self) -> list[int]:
        return [f for f, e in zip(self.frame, self.emitted) if e]

    def to_arrays(self) -> dict[str, np.ndarray]:
        out = {k: np.asarray(getattr(self, k), dtype=np.bool_ if k in ("emitted", "pred_updated", "forced_advance")
                             else np.int64) for k in TRACE_FIELDS}
        out["num_frames"] = np.asarray(self.num_frames, dtype=np.int64)
        return out

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray], prefix: str = "") -> Trace:
        return cls(num_frames=int(arrays[prefix + "num_frames"]),
                   **{k: [x.item() for x in np.asarray(arrays[prefix + k])] for k in TRACE_FIELDS})


@dataclasses.dataclass
class StepOutputs:
    """Per-step tensors of a decode or replay: raw joint logits [S, outputs], the LSTM state (h, c) [S, layers, H]
    that produced the step's prediction output, and the model's own argmax token and duration value [S]."""
    logits: Tensor
    h: Tensor
    c: Tensor
    argmax_token: Tensor
    argmax_duration: Tensor


Decider = Callable[[int, int, int, int], tuple[int, int]]
Predict = Callable[[int, tuple[Tensor, Tensor]], tuple[Tensor, tuple[Tensor, Tensor]]]
JointStep = Callable[[Tensor, Tensor], Tensor]


@torch.no_grad()
def run_steps(cfg: Config, enc_proj: Tensor, length: int, state: tuple[Tensor, Tensor], predict: Predict,
              joint_step: JointStep, decide: Decider, record: bool) -> tuple[Trace, StepOutputs | None]:
    """The label-looping step sequence of one utterance, independent of the network implementation.

    enc_proj [1, T, H] is the projected encoder output, state the initial LSTM state, predict(token,
    state) -> (projected prediction output [1, 1, H], state), joint_step(enc_proj[:, t] [1, 1, H], g)
    -> raw logits [1, outputs]. decide(step, frame, argmax token, argmax duration) -> (token,
    duration) chooses the decision (greedy: the argmax; replay: the trace's). The argmax is taken
    over the log-softmaxed logits, as NeMo's CPU joint output; the frame rules are in the module
    docstring.
    """
    blank, durations = cfg.blank, cfg.durations
    num_d = len(durations)
    pred_input = blank
    g, state = predict(blank, state)
    trace = Trace(num_frames=length)
    rec: dict[str, list[Tensor]] = {"logits": [], "h": [], "c": [], "t": [], "d": []}
    t, last_nb, lasts = 0, -1, 0
    while t < length:
        raw = joint_step(enc_proj[:, t].unsqueeze(1), g)  # [1, outputs]
        scores = raw.log_softmax(dim=-1)
        _, arg_token = scores[:, :-num_d].max(dim=-1)
        arg_duration = durations[int(scores[:, -num_d:].argmax(dim=-1))]
        arg_token = int(arg_token)
        if record:
            rec["logits"].append(raw[0]); rec["h"].append(state[0][:, 0]); rec["c"].append(state[1][:, 0])
            rec["t"].append(torch.tensor(arg_token)); rec["d"].append(torch.tensor(arg_duration))
        token, duration = decide(len(trace), t, arg_token, arg_duration)
        emitted = token != blank
        advance = 1 if (not emitted and duration == 0) else duration
        forced = False
        if emitted:
            lasts = lasts + 1 if last_nb == t else 1
            last_nb = t
            g, state = predict(token, state)
            if t + advance < length and lasts >= cfg.max_symbols and last_nb == t + advance:
                advance += 1
                forced = True
        for key, value in (("frame", t), ("pred_input", pred_input), ("token", token), ("duration", duration),
                           ("emitted", emitted), ("pred_updated", emitted), ("symbols_at_frame", lasts),
                           ("forced_advance", forced), ("advance", advance)):
            getattr(trace, key).append(value)
        if emitted:
            pred_input = token
        t += advance
    outputs = None
    if record:
        outputs = StepOutputs(*(torch.stack(rec[k]) if rec[k] else torch.empty(0)
                                for k in ("logits", "h", "c", "t", "d")))
    return trace, outputs


def _run(model: ParakeetReference, encoder_output: Tensor, length: int, decide: Decider,
         record: bool) -> tuple[Trace, StepOutputs | None]:
    """run_steps with this model's prediction and joint networks (encoder_output [1, d_model, T])."""
    joint, decoder = model.joint, model.decoder

    def predict(token: int, state: tuple[Tensor, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        labels = None if token == model.cfg.blank else torch.full((1, 1), token, dtype=torch.long)
        g, state = decoder.predict(labels, state)
        return joint.pred(g), state

    def joint_step(f: Tensor, g: Tensor) -> Tensor:
        return joint.joint_after_projection(f, g).squeeze(1).squeeze(1)

    with torch.no_grad():
        enc_proj = joint.enc(encoder_output.transpose(1, 2))  # NeMo projects all frames at once
    return run_steps(model.cfg, enc_proj, length, decoder.initial_state(1), predict, joint_step, decide, record)


def _utterances(encoder_output: Tensor, lengths: Tensor) -> Iterable[tuple[Tensor, int]]:
    for b in range(encoder_output.size(0)):
        yield encoder_output[b:b + 1], int(lengths[b])


def greedy_decode(model: ParakeetReference, encoder_output: Tensor, lengths: Tensor,
                  record: bool = False) -> list[Trace] | list[tuple[Trace, StepOutputs]]:
    """Greedy TDT per utterance of encoder_output [B, d_model, T]; Traces (with StepOutputs if record)."""
    out = []
    for enc, length in _utterances(encoder_output, lengths):
        trace, steps = _run(model, enc, length, lambda i, t, tok, dur: (tok, dur), record)
        out.append((trace, steps) if record else trace)
    return out


def replay(model: ParakeetReference, encoder_output: Tensor, length: int, trace: Trace) -> StepOutputs:
    """Forced replay of trace on one utterance (encoder_output [1, d_model, T]): the trace's tokens and durations
    drive the prediction network and frame advance; returns this model's per-step logits, states and argmax.

    ValueError if the trace does not fit this utterance (frame sequence, length or step count)."""
    if trace.num_frames != length:
        raise ValueError(f"trace has {trace.num_frames} frames, utterance {length}")

    def decide(i: int, t: int, tok: int, dur: int) -> tuple[int, int]:
        if i >= len(trace):
            raise ValueError(f"trace ended after {len(trace)} steps before the last frame")
        if trace.frame[i] != t:
            raise ValueError(f"step {i}: trace frame {trace.frame[i]} != replayed frame {t}")
        return trace.token[i], trace.duration[i]

    replayed, steps = _run(model, encoder_output, length, decide, record=True)
    if len(replayed) != len(trace):
        raise ValueError(f"trace has {len(trace)} steps, replay ended after {len(replayed)}")
    for key in TRACE_FIELDS:
        if getattr(replayed, key) != getattr(trace, key):
            raise ValueError(f"trace field {key} is inconsistent with its own decisions")
    return steps
