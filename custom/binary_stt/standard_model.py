"""Conventional offline Conformer CTC baseline, independent of binary layers.

TorchAudio's SiLU/GLU Conformer blocks, 2x temporal subsampling, sinusoidal
positions, utterance CMVN, and character outputs. This is a diagnostic baseline,
not a causal or quantized deployment model.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.models import Conformer


class StandardCTC(nn.Module):
    def __init__(self, vocab_size=29, width=256, layers=6, dropout=0.1):
        super().__init__()
        self.subsample = nn.Conv1d(80, width, 5, stride=2, padding=2)
        self.norm = nn.LayerNorm(width)
        self.encoder = Conformer(width, 4, 4 * width, layers, 31,
                                 dropout=dropout, use_group_norm=True)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, features, lengths):
        valid = torch.arange(features.shape[-1], device=features.device)[None] < lengths[:, None]
        # Normalize each mel bin using actual utterance frames, before padding.
        x = features.float().masked_fill(~valid[:, None], 0)
        count = lengths[:, None, None]
        mean = x.sum(-1, keepdim=True) / count
        centered = (x - mean).masked_fill(~valid[:, None], 0)
        var = centered.square().sum(-1, keepdim=True) / count
        x = centered / var.clamp_min(1e-5).sqrt()
        x = F.silu(self.norm(self.subsample(x).transpose(1, 2)))
        lengths = (lengths + 1) // 2
        t, d = x.shape[1:]
        angles = torch.arange(t, device=x.device).float()[:, None] * torch.exp(
            torch.arange(0, d, 2, device=x.device).float() * (-math.log(10000) / d))
        position = torch.stack((angles.sin(), angles.cos()), -1).flatten(-2)
        x = x + position.to(x.dtype)
        valid = torch.arange(t, device=x.device)[None] < lengths[:, None]
        x = x.masked_fill(~valid[..., None], 0)
        x, lengths = self.encoder(x, lengths)
        return self.head(x), lengths
