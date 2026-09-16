"""Causal fixed-statistic log-mel extraction without torchaudio/torchcodec.

Frame t uses samples [t * hop - window + 1, t * hop], with zero initial
history. Appending audio never changes an existing feature frame. There is
no per-utterance normalization, which would leak future audio into training.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CausalLogMel(nn.Module):
    """16 kHz audio -> [..., 80, ceil(samples / 160)] FP32 log-mel frames.

    ``mean`` and ``std`` are fixed scalar training-set normalization constants;
    the default leaves natural-log mel power unchanged. Waveforms may have
    shape [samples] or [batch, samples]. Resampling is the data reader's job.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        window_length: int = 400,
        hop_length: int = 160,
        n_fft: int = 512,
        mean: float = 0.0,
        std: float = 1.0,
        log_floor: float = 1e-10,
    ) -> None:
        super().__init__()
        if sample_rate != 16000:
            raise ValueError("Resample audio to 16000 Hz before feature extraction")
        if not 0 < hop_length <= window_length <= n_fft:
            raise ValueError("Require 0 < hop_length <= window_length <= n_fft")
        if n_mels <= 0 or not math.isfinite(std) or std <= 0:
            raise ValueError("n_mels and finite std must be positive")
        if not math.isfinite(mean) or not math.isfinite(log_floor) or log_floor <= 0:
            raise ValueError("mean must be finite and log_floor must be positive")
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.mean, self.std, self.log_floor = mean, std, log_floor

        # STFT otherwise centers a short window inside n_fft, accidentally
        # shifting the causal window. Explicit left padding keeps it aligned.
        window = F.pad(torch.hann_window(window_length), (n_fft - window_length, 0))
        self.register_buffer("window", window, persistent=False)
        frequencies = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        high_mel = 2595 * math.log10(1 + (sample_rate / 2) / 700)
        mel_points = torch.linspace(0, high_mel, n_mels + 2)
        hz_points = 700 * (torch.pow(10.0, mel_points / 2595) - 1)
        rising = (frequencies[None] - hz_points[:-2, None]) / (
            hz_points[1:-1] - hz_points[:-2]
        )[:, None]
        falling = (hz_points[2:, None] - frequencies[None]) / (
            hz_points[2:] - hz_points[1:-1]
        )[:, None]
        filters = torch.minimum(rising, falling).clamp_min(0)
        filters *= (2 / (hz_points[2:] - hz_points[:-2]))[:, None]
        self.register_buffer("mel_filters", filters, persistent=False)

    def feature_lengths(self, sample_lengths: Tensor | int) -> Tensor | int:
        return (sample_lengths + self.hop_length - 1) // self.hop_length

    def forward(self, waveform: Tensor, sample_rate: int | None = None) -> Tensor:
        if sample_rate is not None and sample_rate != self.sample_rate:
            raise ValueError("Waveform sample rate must be 16000 Hz")
        if waveform.ndim not in (1, 2) or waveform.shape[-1] == 0:
            raise ValueError("Expected nonempty waveform [samples] or [batch, samples]")
        # Feature extraction and its reductions remain FP32 under autocast.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            padded = F.pad(waveform.float(), (self.n_fft - 1, 0))
            spectrum = torch.stft(
                padded,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window.to(device=waveform.device, dtype=torch.float32),
                center=False,
                return_complex=True,
            )
            power = spectrum.abs().square()
            mel = torch.matmul(self.mel_filters.to(power.device), power)
            return (mel.clamp_min(self.log_floor).log() - self.mean) / self.std


def log_mel(waveform: Tensor, sample_rate: int = 16000) -> Tensor:
    """Convenience helper; reuse CausalLogMel for dataset-scale extraction."""
    return CausalLogMel(sample_rate=sample_rate).to(waveform.device)(waveform)
