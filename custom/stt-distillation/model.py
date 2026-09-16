"""Causal Conformer-CTC pilot with identical FP32/ternary learned topology."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from common import ALPHABET


class Weight(nn.Module):
    def effective(self):
        if self.precision == "fp":
            return self.weight
        w = self.weight
        scale = w.detach().abs().mean().clamp_min(1e-8)
        q = (w / scale).round().clamp(-1, 1) * scale
        return w + (q - w).detach()


class Linear(Weight):
    def __init__(self, a, b, precision):
        super().__init__()
        self.precision = precision
        self.weight = nn.Parameter(torch.empty(b, a))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return F.linear(x, self.effective())


class Conv(Weight):
    def __init__(self, a, b, k, precision, stride=1, groups=1):
        super().__init__()
        self.precision = precision
        self.stride = stride
        self.groups = groups
        self.k = k
        self.weight = nn.Parameter(torch.empty(b, a // groups, k))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return F.conv1d(
            F.pad(x, (self.k - 1, 0)),
            self.effective(),
            stride=self.stride,
            groups=self.groups,
        )


def norm(x):
    return F.layer_norm(x, (x.shape[-1],))


def rope(x):
    n = x.shape[-1]
    t = x.shape[-2]
    freqs = torch.outer(
        torch.arange(t, device=x.device, dtype=torch.float32),
        10000 ** (-torch.arange(0, n, 2, device=x.device, dtype=torch.float32) / n),
    )
    co, si = freqs.cos().to(x.dtype), freqs.sin().to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * co - b * si, a * si + b * co), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, heads, precision, context, dropout):
        super().__init__()
        self.heads = heads
        self.context = context
        self.dropout = dropout
        self.ff1 = nn.ModuleList(
            [Linear(d, 4 * d, precision), Linear(4 * d, d, precision)]
        )
        self.ff2 = nn.ModuleList(
            [Linear(d, 4 * d, precision), Linear(4 * d, d, precision)]
        )
        self.qkv = Linear(d, 3 * d, precision)
        self.proj = Linear(d, d, precision)
        self.cin = Linear(d, 2 * d, precision)
        self.depth = Conv(d, d, 15, precision, groups=d)
        self.cout = Linear(d, d, precision)

    def drop(self, x):
        return F.dropout(x, self.dropout, self.training)

    def ff(self, x, mods):
        return self.drop(mods[1](self.drop(F.silu(mods[0](norm(x))))))

    def forward(self, x, mask):
        x = x + 0.5 * self.ff(x, self.ff1)
        b, t, d = x.shape
        q, k, v = (
            self.qkv(norm(x))
            .reshape(b, t, 3, self.heads, d // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        a = F.scaled_dot_product_attention(
            rope(q),
            rope(k),
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0,
        )
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, t, d)))
        c = F.glu(self.cin(norm(x)), dim=-1).transpose(1, 2)
        c = self.depth(c).transpose(1, 2)
        x = x + self.drop(self.cout(F.silu(norm(c))))
        return norm(x + 0.5 * self.ff(x, self.ff2))


class Model(nn.Module):
    def __init__(self, cfg, precision="ternary"):
        super().__init__()
        self.cfg = cfg
        self.precision = precision
        d = cfg["width"]
        self.front = Conv(80, d, 5, precision, stride=2)
        self.blocks = nn.ModuleList(
            [
                Block(d, cfg["heads"], precision, cfg["context"], cfg["dropout"])
                for _ in range(cfg["depth"])
            ]
        )
        self.head = Linear(d, len(ALPHABET), precision)

    def forward(self, x):
        x = F.silu(norm(self.front(x).transpose(1, 2)))
        t = x.shape[1]
        p = torch.arange(t, device=x.device)
        mask = (
            (p[:, None] >= p[None, :]) & (p[:, None] - p[None, :] < self.cfg["context"])
        )[None, None]
        for b in self.blocks:
            x = (
                checkpoint(b, x, mask, use_reentrant=False)
                if self.training and self.cfg.get("checkpoint", True)
                else b(x, mask)
            )
        return self.head(x)


def set_precision(model, precision):
    for m in model.modules():
        if isinstance(m, Weight):
            m.precision = precision
    model.precision = precision
