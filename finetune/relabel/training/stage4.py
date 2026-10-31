"""Stage 4 (keab-prime plan): the arms, the runs, their spans and schedules, and the preregistered rules (CPU, no torch).

Arms (ARMS; the teacher package, keab-prime plan "Decisions" 1-2, "Training"):
    main      T*'s labels (label set "T*") read through cleaning version 3, B0' as the encoder target, B0''s prediction
              and joint networks frozen in the 8-bit form of step 0's scope
    control   B0's labels (label set "B0": b0_raw under T*'s disposition mask) through the same cleaning, B0 as the
              encoder target, B0's prediction and joint networks in the same 8-bit form
Everything else is shared: the batch plan (version 3), the extra sources and their labels, the cleaning version, the
student (MB1's latent checkpoint, binary at f = 1), the activation quantizer, the loss seam, the schedule and the seed
(SEED, one for every run).

Spans are whole pools of the version 3 plan (b0prime plan "Units", stage3.stop_batch): a run that ends after pool q
stops at B(q + 1). Runs (RUNS):
    pilot-<arm>-<lr>      pools 0-77 at peak 2e-5, 6e-5 or 2e-4 (PILOT_LRS; 2e-4 is MB1's own peak): WSD with warmup
                          2% of B(78), stable through pool 65, cooldown over pools 66-77 to 0 at B(78) - 1
    trunk-<arm>           pools 0-663 at the selected peak: warmup 2% of B(664), then the peak; branch points at the
                          first planned batch of pools 332 and 664 (weights, AdamW, RNG, plan position)
    b100-<arm>            from its trunk's branch point at pool 664: linear cooldown over pools 664-781
    b50-main              from the main trunk's branch point at pool 332: linear cooldown over pools 332-390
Kea-B' is b100-main, fixed in advance (KEAB_PRIME). Every run has its arm's twin with the same spec but the arm, except
b50 (main only); the control trunk keeps the same branch points so the two trunks are the same run spec.

Rules (keab-prime plan; points of WER; EPS absorbs float error at the thresholds):
    select_lr   the learning rate with the lowest mean H4 over the two arms' pilots (training-equivalent PyTorch mode);
                rates within NEAR_TIE of it go to the lowest
    gate        on deployed-mode scores of the b100 models: improvement delta = H4(control b100) - H4(Kea-B') >=
                MIN_DELTA with the 90% paired cluster-bootstrap interval's lower bound above 0; no H4 set more than
                MAX_SET_REGRESSION points worse than the control; a domain set more than DOMAIN_FLAG points worse is
                reported to AJ before stage 5 (no threshold)
    mb1_check   MB1 re-scored by the x1 builder within MB1_CHECK points of gpu1bit/RESULTS.md's x1 dev numbers per set
"""
from __future__ import annotations

import dataclasses

from training import stage3

ARMS = ("main", "control")
LABEL_SETS = {"main": "T*", "control": "B0"}
TEACHERS = {"main": "B0p", "control": "B0"}           # the encoder target and the decoder's source
PLAN_POOLS = stage3.PLAN_POOLS                        # 782
PILOT_POOLS = stage3.PILOT_POOLS                      # 78
PILOT_COOLDOWN_POOL = stage3.PILOT_COOLDOWN_POOL      # 66
PILOT_LRS = (2e-5, 6e-5, 2e-4)
LR_NAMES = {2e-5: "lr2e-5", 6e-5: "lr6e-5", 2e-4: "lr2e-4"}
TRUNK_POOLS = 664
BRANCHES = {"b50": (332, 391), "b100": (664, 782)}   # cooldown pools [start, end)
BRANCH_ARMS = {"b50": ("main",), "b100": ARMS}
BRANCH_POINTS = tuple(start for start, _ in BRANCHES.values())          # 332, 664
SEED = 20261031
KEAB_PRIME = "b100-main"
NEAR_TIE = 0.05
MIN_DELTA = 0.10
MAX_SET_REGRESSION = 0.25
DOMAIN_FLAG = 0.5
MB1_CHECK = 0.02
EPS = 1e-9


@dataclasses.dataclass(frozen=True)
class Run:
    name: str
    arm: str                                    # main | control
    kind: str                                   # pilot | trunk | branch
    stop_pool: int
    length_pools: int
    seed: int
    schedule: str = "wsd"
    cooldown: tuple[int, int] | None = None
    lr: float | None = None
    branch_pool: int | None = None
    branch_points: tuple[int, ...] = ()
    trunk: str | None = None                    # a branch's trunk run

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def scored(self) -> bool:
        return self.kind != "trunk"

    @property
    def label_set(self) -> str:
        return LABEL_SETS[self.arm]

    @property
    def teacher(self) -> str:
        return TEACHERS[self.arm]

    def spec(self) -> dict:
        """The run's spec without its name and arm (the arms' twins have equal specs)."""
        d = self.to_dict()
        for key in ("name", "arm", "trunk"):
            d.pop(key)
        return d


def pilot_name(arm: str, lr: float) -> str:
    return f"pilot-{arm}-{LR_NAMES[lr]}"


def _runs() -> dict[str, Run]:
    out = {}
    for lr in PILOT_LRS:
        for arm in ARMS:
            name = pilot_name(arm, lr)
            out[name] = Run(name, arm, "pilot", PILOT_POOLS, PILOT_POOLS, SEED, "wsd",
                            (PILOT_COOLDOWN_POOL, PILOT_POOLS), lr)
    for arm in ARMS:
        out[f"trunk-{arm}"] = Run(f"trunk-{arm}", arm, "trunk", TRUNK_POOLS, TRUNK_POOLS, SEED,
                                  branch_points=BRANCH_POINTS)
    for branch, (start, end) in BRANCHES.items():
        for arm in BRANCH_ARMS[branch]:
            name = f"{branch}-{arm}"
            out[name] = Run(name, arm, "branch", end, TRUNK_POOLS, SEED, "wsd", (start, end), branch_pool=start,
                            trunk=f"trunk-{arm}")
    return out


RUNS = _runs()
PILOT = tuple(n for n, r in RUNS.items() if r.kind == "pilot")
TRUNKS = tuple(f"trunk-{a}" for a in ARMS)
BRANCH_RUNS = ("b50-main", "b100-main", "b100-control")
GATE = {"main": "b100-main", "control": "b100-control"}
MAIN = TRUNKS + BRANCH_RUNS

stop_batch = stage3.stop_batch
schedule_of = stage3.schedule_of
branch_points = stage3.branch_points
schedule_dict = stage3.schedule_dict


def twin(name: str) -> str | None:
    """The other arm's run of the same spec (None for b50-main)."""
    run = RUNS[name]
    other = ARMS[1 - ARMS.index(run.arm)]
    candidate = name.replace(f"-{run.arm}", f"-{other}") if run.kind != "pilot" else pilot_name(other, run.lr)
    return candidate if candidate in RUNS else None


# ----------------------------------------------------------------------------- rules

def select_lr(h4: dict[str, float]) -> dict:
    """The pilot rule on {pilot run name: H4 (points, PyTorch fake-quant mode)} of all six pilots."""
    if set(h4) != set(PILOT):
        raise ValueError(f"the selection needs the six pilot runs {PILOT}, got {sorted(h4)}")
    mean = {lr: sum(h4[pilot_name(a, lr)] for a in ARMS) / len(ARMS) for lr in PILOT_LRS}
    best = min(mean.values())
    chosen = min(lr for lr, v in mean.items() if v <= best + NEAR_TIE + EPS)
    return {"selected": chosen, "selected_name": LR_NAMES[chosen], "mean_h4": {LR_NAMES[lr]: v for lr, v in mean.items()},
            "h4": dict(h4), "rule": f"lowest mean H4 over the two arms (PyTorch fake-quant mode); within {NEAR_TIE} "
                                    "points of it, the lowest learning rate"}


def gate(delta: dict, per_set: dict[str, float], domain: dict[str, float] | None = None) -> dict:
    """delta: {"point", "ci"} of H4(control b100) - H4(Kea-B') (points, deployed mode); per_set: {H4 set: WER(Kea-B')
    - WER(control)}; domain: {domain set: WER(Kea-B') - WER(control)} (flags only)."""
    improvement = delta["point"] >= MIN_DELTA - EPS and delta["ci"][0] > EPS
    worst = max(per_set.values())
    no_regression = worst <= MAX_SET_REGRESSION + EPS
    flags = sorted(n for n, v in (domain or {}).items() if v > DOMAIN_FLAG + EPS)
    return {"improvement": improvement, "no_large_regression": no_regression, "worst_set_regression": worst,
            "pass": improvement and no_regression, "domain_regressions_over_flag": flags,
            "rules": {"improvement": f"H4(control b100) - H4(Kea-B') >= {MIN_DELTA}, 90% interval excluding 0",
                      "regression": f"no H4 set more than {MAX_SET_REGRESSION} points worse than the control",
                      "domain": f"a domain regression above {DOMAIN_FLAG} points is reported to AJ before stage 5"}}


def mb1_check(rescored: dict[str, float], recorded: dict[str, float]) -> dict:
    """MB1 re-scored by the x1 builder (points per dev set) against gpu1bit/RESULTS.md's x1 arm."""
    if set(rescored) != set(recorded):
        raise ValueError("the re-score and the record cover different sets")
    sets = {n: {"wer": rescored[n], "recorded": recorded[n], "difference_points": rescored[n] - recorded[n]}
            for n in sorted(recorded)}
    return {"sets": sets, "tolerance_points": MB1_CHECK,
            "pass": all(abs(v["difference_points"]) <= MB1_CHECK + EPS for v in sets.values())}
