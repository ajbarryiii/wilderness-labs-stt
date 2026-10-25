"""The warmup-stable-decay (WSD) schedule on the batch index (b0prime plan "Schedule (WSD)").

The learning rate is a function of the batch index k (counted from 0, including batches that take no optimizer
step), so every phase ends exactly at a batch boundary. Since the batch plan (2026-10-24) k is the planned-batch
index, and spans and branch points are pools (pool_trunk, pool_branch: a boundary at pool p is the first planned
batch of pool p):

1. warmup: for k < w, lr = peak * k / w (linear, reaching the peak at batch w); w is 2% of the trunk's planned
   length (WARMUP_FRACTION) and is part of the trunk;
2. stable: from w on, lr = peak (the trunk; a trunk has no end);
3. cooldown (a branch): from the peak at k = s to 0 at the branch's last batch s + c - 1,
   lr = peak * (s + c - 1 - k) / (c - 1). A branch is a new WSD value; the trunk it branches from is unchanged.
   Past the last batch the schedule refuses (the run has stopped: stop_batch = s + c).

Stage 2b (decoder_only): pools 0-155, warmup 2% of their planned batches, stable through pool 132, cooldown over
pools 133-155. Stage 3: pool_trunk(peak, plan, 664) and pool_branch for b25 (166-194), b50 (332-390) and b100
(664-781); the pilot runs are pool_branch(pool_trunk(peak, plan, 78), plan, 66, 78).
"""
from __future__ import annotations

import dataclasses

WARMUP_FRACTION = 0.02
BRANCH_FRACTION = 0.15      # b0prime plan: each cooldown branch is 15% of its own total length


@dataclasses.dataclass(frozen=True)
class WSD:
    peak: float
    warmup: int
    cooldown_start: int | None = None     # s; None for a trunk
    cooldown: int | None = None           # c

    def __post_init__(self):
        if not self.peak > 0 or self.warmup < 0:
            raise ValueError(f"invalid WSD {self}")
        if (self.cooldown_start is None) != (self.cooldown is None):
            raise ValueError("a branch needs both its start and its cooldown length")
        if self.cooldown_start is not None:
            if self.cooldown_start < self.warmup:
                raise ValueError(f"cooldown starts at {self.cooldown_start}, inside the warmup (w = {self.warmup})")
            if self.cooldown < 2:
                raise ValueError("a linear cooldown to 0 needs at least 2 batches")

    @property
    def is_branch(self) -> bool:
        return self.cooldown_start is not None

    @property
    def last_batch(self) -> int | None:
        """The branch's last batch s + c - 1 (lr 0 there); None for a trunk."""
        return None if self.cooldown_start is None else self.cooldown_start + self.cooldown - 1

    @property
    def stop_batch(self) -> int | None:
        """Consumed batches when the branch ends (s + c); None for a trunk."""
        return None if self.cooldown_start is None else self.cooldown_start + self.cooldown

    def factor(self, k: int) -> float:
        if k < 0:
            raise ValueError(f"batch index {k} < 0")
        if k < self.warmup:
            return k / self.warmup
        if self.cooldown_start is None or k < self.cooldown_start:
            return 1.0
        if k > self.last_batch:
            raise ValueError(f"batch {k} is past the branch's last batch {self.last_batch}")
        return (self.last_batch - k) / (self.cooldown - 1)

    def lr(self, k: int) -> float:
        f = self.factor(k)
        return self.peak if f == 1.0 else self.peak * f

    def to_dict(self) -> dict:
        return dataclasses.asdict(self) | {"rule": "b0prime plan WSD on the planned-batch index"}

    @classmethod
    def from_dict(cls, d: dict) -> "WSD":
        return cls(peak=float(d["peak"]), warmup=int(d["warmup"]),
                   cooldown_start=None if d.get("cooldown_start") is None else int(d["cooldown_start"]),
                   cooldown=None if d.get("cooldown") is None else int(d["cooldown"]))


def trunk(peak: float, planned_length: int) -> WSD:
    """A trunk: warmup over 2% of its planned length, then the peak, without an end."""
    if planned_length < 1:
        raise ValueError("planned length must be positive")
    return WSD(peak=float(peak), warmup=round(WARMUP_FRACTION * planned_length))


def branch(base: WSD, start: int, cooldown: int) -> WSD:
    """A cooldown branch of a trunk at consumed batch `start`, over `cooldown` batches."""
    if base.is_branch:
        raise ValueError("branches start from a trunk, not from another branch")
    return WSD(peak=base.peak, warmup=base.warmup, cooldown_start=int(start), cooldown=int(cooldown))


def fraction_branch(base: WSD, total: int, fraction: float = BRANCH_FRACTION) -> WSD:
    """b0prime plan branches: a cooldown of `fraction` of the branch's own total length, ending at `total`."""
    c = round(fraction * total)
    return branch(base, total - c, c)


def pool_trunk(peak: float, plan, n_pools: int) -> WSD:
    """A trunk whose planned length is the plan's first n_pools pools (b0prime plan "Units", 2026-10-24)."""
    if not 0 < n_pools <= plan.n_pools:
        raise ValueError(f"{n_pools} pools outside the plan's {plan.n_pools}")
    return trunk(peak, plan.pool_start(n_pools))


def pool_branch(base: WSD, plan, start_pool: int, end_pool: int) -> WSD:
    """A cooldown branch over pools [start_pool, end_pool): from the first planned batch of start_pool to the last
    planned batch before end_pool."""
    if not 0 <= start_pool < end_pool <= plan.n_pools:
        raise ValueError(f"pools [{start_pool}, {end_pool}) outside the plan's {plan.n_pools}")
    s = plan.pool_start(start_pool)
    return branch(base, s, plan.pool_start(end_pool) - s)


DECODER_ONLY_POOLS = 156             # pools 0-155: consumed batches 0-19,967
DECODER_ONLY_COOLDOWN_POOL = 133     # cooldown over pools 133-155 (23 pools)


def decoder_only(peak: float, plan) -> WSD:
    """decoder-only plan (2026-10-24): warmup over 2% of the planned batches of pools 0-155, stable through
    pool 132, cooldown over pools 133-155 to 0 at the last planned batch."""
    if plan.n_pools != DECODER_ONLY_POOLS:
        raise ValueError(f"the decoder-only plan has {DECODER_ONLY_POOLS} pools, not {plan.n_pools}")
    return pool_branch(pool_trunk(peak, plan, DECODER_ONLY_POOLS), plan, DECODER_ONLY_COOLDOWN_POOL, DECODER_ONLY_POOLS)
