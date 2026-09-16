"""Causal signal-onset confirmation and first-emission warm-up for CTC repair.

This detects departure from the feature floor, not speech versus background noise.
All audio reaches the encoder. Only leading CTC emissions are held at blank.
"""

import torch
from torch.nn import functional as F

from model import Model
from pilot4_model import PilotModel


def startup_mask(x, floor=-1.95, confirmation_frames=3, hold_frames=10):
    """Return a feature-rate emission mask using only present/past frames.

    Confirm after consecutive above-floor frames, latch once, then wait the
    requested number of additional 10 ms feature hops. No audio is removed.
    Batch rows are independent. Full streaming use must retain state across chunks.
    """
    if confirmation_frames < 1 or hold_frames < 0:
        raise ValueError("Onset confirmation must be positive and hold nonnegative")
    active = x.amax(dim=1) > floor
    positions = torch.arange(x.shape[-1], device=x.device)[None]
    last_inactive = torch.where(active, -1, positions).cummax(dim=1).values
    confirmed = (positions - last_inactive >= confirmation_frames).cummax(dim=1).values
    if hold_frames:
        confirmed = F.pad(confirmed, (hold_frames, 0), value=False)[:, : x.shape[-1]]
    return confirmed


class OnsetModel(PilotModel):
    def forward(self, x):
        # Bypass PilotModel's old single-frame gate, retaining its initialization
        # and precision placement. Encoder ingests the complete original signal.
        logits = Model.forward(self, x)
        allowed = startup_mask(
            x,
            self.cfg.get("startup_floor", -1.95),
            self.cfg.get("startup_confirmation_frames", 3),
            self.cfg.get("startup_hold_frames", 10),
        )[:, ::2]
        blank = torch.full_like(logits, -1e4)
        blank[..., 0] = 0
        return torch.where(allowed[..., None], logits, blank)
