"""Four-hour learning pilot: ternary transformer matrices, floating CTC head/convs.

This retains the causal Conformer normalization and floating activations to keep
this pilot focused on learnability and optimizer scheduling, not W2A8 fidelity.
"""

import torch
from torch import nn

from common import ALPHABET
from model import Linear, Model


class PilotModel(Model):
    def __init__(self, cfg, precision="fp"):
        super().__init__(cfg, "fp")
        self.precision = precision
        # An unbiased random output layer can commit CTC to arbitrary early
        # symbols. Test an initially uniform character head with a blank prior.
        self.head = nn.Linear(cfg["width"], len(ALPHABET), bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[0] = cfg.get("initial_blank_bias", 2.0)
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, Linear):
                    module.precision = precision

    def forward(self, x):
        logits = super().forward(x)
        # Features have a fixed -2 digital-silence floor. A causal recognizer
        # cannot identify a label while every observed frame is at that floor.
        # Only suppress the leading floor region; later pauses remain available
        # to CTC. A future streaming runtime must carry this one-bit onset state.
        seen_signal = (
            (x.amax(dim=1) > self.cfg.get("startup_floor", -1.95))
            .cummax(dim=1)
            .values[:, ::2]
        )
        blank = torch.full_like(logits, -1e4)
        blank[..., 0] = 0
        return torch.where(seen_signal[..., None], logits, blank)

    def precision_counts(self):
        ternary = sum(
            m.weight.numel()
            for m in self.modules()
            if isinstance(m, Linear) and m.precision == "ternary"
        )
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "ternary": ternary, "floating": total - ternary}
