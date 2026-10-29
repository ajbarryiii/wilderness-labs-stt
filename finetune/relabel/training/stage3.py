"""Stage 3 (b0prime plan): the runs, their spans and schedules, and the preregistered rules (CPU, no torch).

Spans are whole pools of the batch plan (b0prime plan "Units"): a run that ends after pool q stops at B(q + 1), the
first planned batch of pool q + 1 (plan.pool_start), exclusive; B(PLAN_POOLS) is the plan's last batch + 1, so no run
goes past the end of pool 781 (consumed batch 99,999).

Runs (RUNS, in ORDER after the B0 re-score):
    pilot-lr3e-5, pilot-lr1e-4, pilot-lr3e-4   pools 0-77, WSD: warmup 2% of B(78), stable through pool 65, cooldown
                                               over pools 66-77 to 0 at B(78) - 1; seed SEED
    sanity-wsd, sanity-linear                  pools 0-77 at the selected peak, seed SANITY_SEED (new): the WSD above,
                                               and the trainer's schedule (Linear: train.lr_factor on the planned-batch
                                               index, warmup round(2% of B(78)), linear decay to 0 at B(78))
    trunk                                      pools 0-663 at the selected peak, seed SEED: warmup 2% of B(664), then the
                                               peak (no end); branch points at the first planned batch of pools 166,
                                               332 and 664 (weights, optimizer, RNG, plan position)
    b25, b50, b100                             from the trunk's branch point at pool 166 / 332 / 664: linear cooldown
                                               from the peak at B(p) to 0 at B(q + 1) - 1 over pools 166-194 / 332-390 /
                                               664-781. B0' is b100, fixed in advance.

Rules (b0prime plan; points of WER; EPS absorbs float error at the thresholds):
    select_lr      the pilot run with the lowest full-dev H4; runs within NEAR_TIE of it go to the lowest learning rate
    wsd_rule       d = H4(WSD) - H4(linear) with its 90% paired cluster-bootstrap interval: stage 3 stops if d > MAX_D
                   or the interval's lower bound is above 0; otherwise the main run uses WSD
    gate           B0 re-score within RESCORE_POINTS of stage 1 on every dev set; H4 delta = WER(B0) - WER(B0') >= 0.20
                   with the interval's lower bound above 0; no H4 set more than 0.25 points worse; a domain set more than
                   0.5 points worse is flagged for the user; b25 or b50 beating b100 by more than 0.10 H4 is flagged
"""
from __future__ import annotations

import dataclasses

from training import schedule as wsd

PLAN_POOLS = 782                  # pools 0-781 of the batch plan: consumed batches 0-99,999
PILOT_POOLS = 78                  # pools 0-77 (consumed batches 0-9,983)
PILOT_COOLDOWN_POOL = 66          # pilot cooldown over pools 66-77 (12 pools)
PILOT_LRS = (3e-5, 1e-4, 3e-4)
TRUNK_POOLS = 664                 # the trunk ends after pool 663 (consumed batch 84,991)
BRANCHES = {"b25": (166, 195), "b50": (332, 391), "b100": (664, 782)}   # cooldown pools [start, end)
BRANCH_POINTS = tuple(start for start, _ in BRANCHES.values())          # 166, 332, 664
B0PRIME = "b100"
SEED = 20260930                   # parakeet-ternary paths.SEED: the pilot learning-rate runs and the trunk
SANITY_SEED = 20260931            # the WSD sanity pair's new seed (b0prime plan Step 1.2)
LR_NAMES = {3e-5: "lr3e-5", 1e-4: "lr1e-4", 3e-4: "lr3e-4"}
NEAR_TIE = 0.05
MAX_D = 0.05
RESCORE_POINTS = 0.05
MIN_DELTA = 0.20
MAX_SET_REGRESSION = 0.25
DOMAIN_FLAG = 0.5
BRANCH_FLAG = 0.10
EPS = 1e-9


@dataclasses.dataclass(frozen=True)
class Run:
    name: str
    kind: str                                   # pilot | sanity | trunk | branch
    stop_pool: int                              # the run stops at B(stop_pool)
    length_pools: int                           # the warmup is 2% of B(length_pools)
    seed: int
    schedule: str = "wsd"                       # wsd | linear
    cooldown: tuple[int, int] | None = None     # WSD cooldown pools [start, end)
    lr: float | None = None                     # fixed (pilot learning-rate runs) or None: the selected peak
    branch_pool: int | None = None              # a branch: the trunk's branch point it starts from
    branch_points: tuple[int, ...] = ()         # the trunk: its branch points

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def scored(self) -> bool:
        """Only cooled-down runs are models (b0prime plan: a trunk checkpoint is never scored)."""
        return self.kind != "trunk"


def _runs() -> dict[str, Run]:
    out = {}
    for lr in PILOT_LRS:
        name = f"pilot-{LR_NAMES[lr]}"
        out[name] = Run(name, "pilot", PILOT_POOLS, PILOT_POOLS, SEED, "wsd", (PILOT_COOLDOWN_POOL, PILOT_POOLS), lr)
    out["sanity-wsd"] = Run("sanity-wsd", "sanity", PILOT_POOLS, PILOT_POOLS, SANITY_SEED, "wsd",
                            (PILOT_COOLDOWN_POOL, PILOT_POOLS))
    out["sanity-linear"] = Run("sanity-linear", "sanity", PILOT_POOLS, PILOT_POOLS, SANITY_SEED, "linear")
    out["trunk"] = Run("trunk", "trunk", TRUNK_POOLS, TRUNK_POOLS, SEED, branch_points=BRANCH_POINTS)
    for name, (start, end) in BRANCHES.items():
        out[name] = Run(name, "branch", end, TRUNK_POOLS, SEED, "wsd", (start, end), branch_pool=start)
    return out


RUNS = _runs()
PILOT = tuple(n for n, r in RUNS.items() if r.kind == "pilot")
SANITY = ("sanity-wsd", "sanity-linear")
MAIN = ("trunk",) + tuple(BRANCHES)


class Linear:
    """The trainer's schedule (train.lr_factor: warmup reaching the peak at batch warmup - 1, then linear decay to 0
    at `total`) on the planned-batch index k, for the WSD sanity check's linear arm."""

    kind = "linear"

    def __init__(self, peak: float, warmup: int, total: int):
        if not peak > 0 or not 0 < warmup < total:
            raise ValueError(f"invalid linear schedule ({peak}, {warmup}, {total})")
        self.peak, self.warmup, self.total = float(peak), int(warmup), int(total)

    @property
    def stop_batch(self) -> int:
        return self.total

    @property
    def last_batch(self) -> int:
        return self.total - 1

    def factor(self, k: int) -> float:
        import train  # finetune/parakeet-ternary: the trainer's own function
        if not 0 <= k < self.total:
            raise ValueError(f"batch {k} outside the run's [0, {self.total})")
        return train.lr_factor(k, self.warmup, self.total)

    def lr(self, k: int) -> float:
        return self.peak * self.factor(k)

    def to_dict(self) -> dict:
        return {"kind": "linear", "peak": self.peak, "warmup": self.warmup, "total": self.total,
                "rule": "train.lr_factor(k, warmup, total) on the planned-batch index (the trainer's schedule)"}

    def __eq__(self, other) -> bool:
        return isinstance(other, Linear) and self.to_dict() == other.to_dict()


def stop_batch(run: Run, plan) -> int:
    """B(run.stop_pool): the run's planned-batch stop (exclusive)."""
    if not 0 < run.stop_pool <= min(plan.n_pools, PLAN_POOLS):
        raise ValueError(f"{run.name}: stop pool {run.stop_pool} outside the plan's {plan.n_pools} pools")
    return plan.pool_start(run.stop_pool)


def schedule_of(run: Run, plan, peak: float):
    """The run's learning rate on the planned-batch index (module docstring)."""
    if run.lr is not None and float(peak) != run.lr:
        raise ValueError(f"{run.name} runs at {run.lr}, not {peak}")
    base = wsd.pool_trunk(peak, plan, run.length_pools)
    if run.schedule == "linear":
        return Linear(peak, base.warmup, stop_batch(run, plan))
    if run.cooldown is None:
        return base
    start, end = run.cooldown
    if end != run.stop_pool:
        raise ValueError(f"{run.name}: the cooldown must end where the run stops")
    return wsd.pool_branch(base, plan, start, end)


def branch_points(run: Run, plan) -> dict[int, int]:
    """{pool: planned batch B(pool)} of the trunk's branch points."""
    return {p: plan.pool_start(p) for p in run.branch_points}


def schedule_dict(s) -> dict:
    return s.to_dict() if isinstance(s, Linear) else {"kind": "wsd", **s.to_dict()}


# ----------------------------------------------------------------------------- rules

def select_lr(h4: dict[float, float]) -> dict:
    """The pilot rule on {learning rate: full-dev H4}: the lowest; within NEAR_TIE of it, the lowest learning rate."""
    if set(h4) != set(PILOT_LRS):
        raise ValueError(f"the selection needs the three pilot runs {PILOT_LRS}, got {sorted(h4)}")
    best = min(h4.values())
    chosen = min(lr for lr, v in h4.items() if v <= best + NEAR_TIE + EPS)
    return {"selected": chosen, "selected_name": LR_NAMES[chosen], "h4": {LR_NAMES[lr]: v for lr, v in h4.items()},
            "rule": f"lowest full-dev H4; within {NEAR_TIE} points of it, the lowest learning rate"}


def wsd_rule(d: float, ci: list[float]) -> dict:
    """The preregistered WSD sanity rule (b0prime plan Step 1): stop if d > MAX_D or the interval's lower bound > 0."""
    too_high, worse = d > MAX_D + EPS, ci[0] > EPS
    return {"d": d, "ci": list(ci), "stop": too_high or worse, "use_wsd": not (too_high or worse),
            "reasons": [r for r, hit in ((f"d > {MAX_D}", too_high), ("the interval's lower bound is above 0", worse))
                        if hit],
            "rule": f"d = H4(WSD) - H4(linear); stop if d > {MAX_D} or the 90% interval's lower bound is above 0"}


def rescore_check(rescored: dict[str, float], stage1: dict[str, float]) -> dict:
    """B0 re-scored against its stage 1 dev WER (points), every set within RESCORE_POINTS."""
    if set(rescored) != set(stage1):
        raise ValueError("the re-score and stage 1 cover different sets")
    sets = {n: {"wer": rescored[n], "stage1_wer": stage1[n], "difference_points": rescored[n] - stage1[n]}
            for n in sorted(rescored)}
    return {"sets": sets, "tolerance_points": RESCORE_POINTS,
            "pass": all(abs(v["difference_points"]) <= RESCORE_POINTS + EPS for v in sets.values())}


def gate(delta: dict, per_set: dict[str, float], rescore: dict, domain: dict[str, float] | None = None) -> dict:
    """The gate to stage 4. delta: {"point", "ci"} of H4 WER(B0) - WER(B0') (points); per_set: {H4 set: WER(B0') -
    WER(B0)}; rescore: rescore_check's result; domain: {domain set: WER(B0') - WER(B0)} (flags only)."""
    improvement = delta["point"] >= MIN_DELTA - EPS and delta["ci"][0] > EPS
    worst = max(per_set.values())
    no_regression = worst <= MAX_SET_REGRESSION + EPS
    flags = sorted(n for n, v in (domain or {}).items() if v > DOMAIN_FLAG + EPS)
    return {"rescore_pass": rescore["pass"], "improvement": improvement, "no_large_regression": no_regression,
            "worst_set_regression": worst, "pass": rescore["pass"] and improvement and no_regression,
            "domain_regressions_over_flag": flags,
            "rules": {"rescore": f"B0 re-scored within {RESCORE_POINTS} points of stage 1 on every dev set",
                      "improvement": f"H4 WER(B0) - WER(B0') >= {MIN_DELTA}, 90% interval excluding 0",
                      "regression": f"no H4 set more than {MAX_SET_REGRESSION} points worse than B0",
                      "domain": f"a domain regression above {DOMAIN_FLAG} points is reported to the user"}}


def branch_flags(h4: dict[str, float]) -> list[str]:
    """Branches (b25, b50) whose H4 beats b100's by more than BRANCH_FLAG (reported to the user before stage 4)."""
    return [n for n in ("b25", "b50") if n in h4 and B0PRIME in h4 and h4[B0PRIME] - h4[n] > BRANCH_FLAG + EPS]
