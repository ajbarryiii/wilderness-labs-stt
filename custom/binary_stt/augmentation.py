"""Training-only frequency/time masking of fixed-statistic log-mel features.

This is a mild, configurable variant of the masking operations in SpecAugment
(Park et al., 2019), https://arxiv.org/abs/1904.08779. It does not apply time
warping or utterance normalization. The defaults are engineering starting
points for this model, not a published optimum for binary speech recognition.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor


def _nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def spec_augment(
    features: Tensor,
    lengths: Tensor,
    config: Mapping | None,
    step: int,
) -> Tensor:
    """Return a clone of ``[batch, mel, time]`` features with valid frames masked.

    Call this only for training, after collation. ``lengths`` and CTC targets are
    unchanged. Mask values are zero in the fixed feature coordinate system;
    they are not an utterance-dependent mean. Padding is never modified.

    Configuration: ``enabled=False``, ``start_step=0``, ``freq_masks=2``,
    ``freq_width=8``, ``time_masks=2``, ``max_time_width=20``, and
    ``max_time_fraction=0.05``. Width limits apply independently to each mask;
    mask widths are sampled uniformly from zero through the inclusive limit.
    Time limits round down, so very short examples can receive no time mask.
    A single mask cannot cover every frame or every frequency channel.

    Sampling uses the ordinary CPU torch RNG, whose state is saved by the
    trainer. Disabled and pre-start calls consume no random numbers.
    """
    if features.ndim != 3 or not features.is_floating_point():
        raise ValueError("features must be floating-point [batch, mel, time]")
    batch, frequencies, padded_frames = features.shape
    if frequencies < 1:
        raise ValueError("features must contain at least one mel channel")
    if lengths.ndim != 1 or lengths.numel() != batch:
        raise ValueError("lengths must contain one valid-frame count per example")
    if lengths.is_floating_point() or lengths.is_complex() or lengths.dtype == torch.bool:
        raise ValueError("lengths must contain integer frame counts")
    valid_lengths = lengths.detach().cpu().tolist()
    if any(length < 0 or length > padded_frames for length in valid_lengths):
        raise ValueError("lengths must lie between zero and the padded frame count")
    _nonnegative_integer("step", step)
    settings = dict(config or {})
    allowed = {
        "enabled", "start_step", "freq_masks", "freq_width", "time_masks",
        "max_time_width", "max_time_fraction",
    }
    unknown = settings.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown SpecAugment configuration keys: {sorted(unknown)}")
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    start_step = _nonnegative_integer("start_step", settings.get("start_step", 0))
    freq_masks = _nonnegative_integer("freq_masks", settings.get("freq_masks", 2))
    freq_width = _nonnegative_integer("freq_width", settings.get("freq_width", 8))
    time_masks = _nonnegative_integer("time_masks", settings.get("time_masks", 2))
    time_width = _nonnegative_integer("max_time_width", settings.get("max_time_width", 20))
    fraction = settings.get("max_time_fraction", 0.05)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise ValueError("max_time_fraction must be a finite fraction in [0, 1]")
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("max_time_fraction must be a finite fraction in [0, 1]")

    result = features.clone()
    if not enabled or step < start_step:
        return result

    def sample_span(size: int, maximum: int) -> tuple[int, int]:
        width = int(torch.randint(maximum + 1, ()).item())
        if width == 0:
            return 0, 0
        start = int(torch.randint(size - width + 1, ()).item())
        return start, start + width

    frequency_limit = min(freq_width, frequencies - 1)
    for index, valid_frames in enumerate(valid_lengths):
        if valid_frames == 0:
            continue
        if frequency_limit:
            for _ in range(freq_masks):
                start, end = sample_span(frequencies, frequency_limit)
                result[index, start:end, :valid_frames] = 0
        time_limit = min(time_width, math.floor(valid_frames * fraction), valid_frames - 1)
        if time_limit:
            for _ in range(time_masks):
                start, end = sample_span(valid_frames, time_limit)
                result[index, :, start:end] = 0
    return result
