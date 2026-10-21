"""cpu1bit stage analysis on NixOS (plans/cpu-benchmark-mac.md "Design and selection"; history M13.8, M14.7,
M15.3, M15.4, M16.3, M16.6; Codex code review c1 findings 2, 3, 4). Reads a stage directory pulled from
the Mac (shared/pull.py).

    finetune/parakeet-binary/python cpu1bit/shared/analysis.py describe   STAGE_DIR
    ...                                                         acceptance STAGE_DIR
    ...                                                         screen     STAGE_DIR --baseline R-MB1 [--candidates V,T,P]
    ...                                                         tune       STAGE_DIR
    ...                                                         gap        STAGE_DIR --arm R-MB1 --control CTRL-MB1
    ...                                                         confirm    CONFIRM_DIR --screen SCREEN_DIR --baseline R-MB1 --candidate V

Loading (findings 3, 4): stage.json and design.json must agree; for every (replicate, sequence) of the
fixed design the lowest attempt whose end record is valid and whose every block record matches the
design block, the attempt and the fingerprint and is valid is used (runner.valid_attempt), so arms are
paired only within one complete, jointly valid attempt. Any sequence without one -> Incomplete: no
statistic is computed on a reduced design. Energy is the gross padded energy per replay (E_lower, E_est,
E_upper); guard energy per replay and window duration are separate descriptive columns. No idle-corrected
(net) energy is computed (plan review p9).

Statistic (stage 1 and 2): per (replicate, sequence, clip) the paired % change 100 (C - B) / B of energy
per replay, the mean per clip over the (replicate, sequence) cells, then the median over clips.

Confirmation interval (finding 2, p8 major 2): the frozen 95% percentile interval (2.5th and 97.5th
percentiles) of a crossed bootstrap, 10,000 draws, seed 20261020: each draw resamples the replicate IDs and
one shared clip-ID vector with replacement, keeps the pairing of each sampled replicate (its sequences
stay together), takes each sampled clip's mean over the sampled replicates, then the median. The gate
(M10) uses candidate E_upper against baseline E_lower in every window.

Rejections (M15.3): every rejected confirmation sequence attempt (any attempt that is not wholly valid,
whatever its number of failed blocks) is imputed as one additional sequence observation of its replicate
whose paired change is +ENVELOPE_PCT (+50%) in every clip; a replicate's value per clip is the mean over
its valid and imputed sequences. An improvement is declared only if the imputed M10 interval's upper bound
is < 0, the stage has at most MAX_REJECTED (2) rejected sequence attempts, and the simulation check
passes. This is a sensitivity analysis under the stated envelope; no protection is claimed beyond it.

Acceptance audit (M13.8, M14.7): every block attempt and sequence attempt is counted, overall, by arm, by
position and against host-state covariates; every rejected sequence attempt is listed with the gates that
fired. Three conditions are flags that never gate: block acceptance < 95%, a two-sided exact test
(Fisher for two levels, Freeman-Halton for more) p < 0.10 for arm or position, an arm or position
acceptance gap > 5 points.

Simulation check (M15.4, M16.3): the whole stage-2 procedure (retries, +50% imputation, the 2-rejection
cap, the crossed bootstrap) is simulated on null paired changes u_r + v_c + e_rsc with magnitudes from the
screening stage's paired changes for the same two arms (two-way method of moments, conservative:
sigma_rep^2 = MS_seq / n_clips, sigma_clip^2 = MS_clip / n_seq, sigma_e^2 = MS_res). A re-measured
sequence attempt keeps its replicate offset and the clip effects and draws fresh attempt noise. Rejection
rates r in {0, r_U/4, r_U/2, 3 r_U/4, r_U}, r_U the one-sided 95% Clopper-Pearson upper bound of the
confirmation's sequence-attempt rejection rate; each sequence's number of rejected attempts is geometric
in r (independent model), and in the adversarial model the same total number of rejections falls, one at
a time, on the currently accepted attempt least favourable to the candidate (largest mean change).
Pass: maximum false-improvement rate over the grid and both models <= 3.5%, and coverage of 0 at r = 0
>= 93% (support for the directional gate, not two-sided coverage of imputed intervals); otherwise the
result is descriptive only. The error control holds for this generating model only. Stress family
(reported, not gating): sequence outcomes are a mixture of +45% with probability p in {5%, 10%} and
-p 45 / (1 - p) otherwise (mean change 0) on top of the screening noise; every unfavourable attempt is
rejected, and zero-rejection realisations are included.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "mac"))
import runner  # noqa: E402  (pure functions only: record selection)

BOOT_DRAWS = 10_000
BOOT_SEED = 20261020
SIM_SEED = 20261021
SIM_COUNT = 2000
COVERAGE_MIN = 0.93
FALSE_IMPROVEMENT_MAX = 0.035
ENVELOPE_PCT = 50.0            # M15.3: a rejected sequence attempt is imputed with the candidate 50% worse
MAX_REJECTED = 2               # M14.7: more rejected sequence attempts -> descriptive only
ACCEPT_MIN = 0.95              # audit flag: block acceptance below this
FISHER_FLAG_P = 0.10           # audit flag: arm or position dependence
GAP_FLAG_PP = 5.0              # audit flag: arm or position acceptance gap, percentage points
CP_CONF = 0.95                 # one-sided Clopper-Pearson upper bound of the rejection rate
STRESS_P = (0.05, 0.10)
STRESS_UNFAVOURABLE_PCT = 45.0
SIGMA_FLOOR_PCT = 1e-9         # variance components below this are floating-point round-off: 0
MAX_SEQUENCE_ATTEMPTS = runner.MAX_ATTEMPTS * runner.PASSES
MODELS = ("independent", "adversarial")

GENERATING_MODEL = ("null paired changes u_r + v_c + e_rsc resampled from the screening stage's variance "
                    "components (replicate offset, clip effect, attempt noise), with independent or adversarial "
                    "(least favourable to the candidate) rejection of sequence attempts on that distribution")
CONDITIONS = (
    "A1 producer continuity (powermetrics emits every sample contiguously, and each record's energy and "
    "elapsed_ns cover the same interval)",
    "A2 post-spawn origin (the first powermetrics record starts at or after spawn)",
    "A3 constant rate within a block (elapsed_ns follows CLOCK_UPTIME_RAW to a constant |rho| <= 50 ppm)",
    "rejections arise only from the external disturbance and coverage gates (foreign CPU, thermal, power, "
    "user input, memory, QoS, powermetrics coverage and the alignment bound), and every rejected sequence "
    f"attempt is imputed at the +{ENVELOPE_PCT:.0f}% envelope (the candidate {ENVELOPE_PCT:.0f}% worse than "
    "its baseline in every clip); no protection is claimed beyond the envelope",
    f"the simulated error control (maximum false-improvement rate <= {FALSE_IMPROVEMENT_MAX:.1%} over "
    "r in {0, r_U/4, r_U/2, 3r_U/4, r_U} under independent and adversarial rejection, and coverage at r = 0 "
    f">= {COVERAGE_MIN:.0%}, which supports the directional gate only) holds only for the simulation's "
    f"generating model: {GENERATING_MODEL}",
    "limitation (stress family, reported, not gating): rare unfavourable outcomes that are always rejected "
    "can go unobserved, and the error control does not cover them",
)


class Incomplete(Exception):
    pass


# --- loading ------------------------------------------------------------------------------------------

def load_stage(stage_dir: str) -> dict:
    stage = runner.read_json(os.path.join(stage_dir, "stage.json"))
    design = runner.read_json(os.path.join(stage_dir, "design.json"))
    fp = stage["fingerprint_sha256"]
    if design["fingerprint_sha256"] != fp:
        raise Incomplete("design.json does not belong to stage.json")
    obs, attempts, missing = [], {}, []
    for seq in runner.sequences(design["blocks"]):
        key = (seq[0]["replicate"], seq[0]["sequence"])
        n = runner.valid_attempt(stage_dir, seq, fp)
        if n is None:
            missing.append(key)
            continue
        attempts[key] = n
        for b in seq:
            rec = runner.read_json(os.path.join(stage_dir, f"{b['block_id']}.att{n}.json"))
            for w in rec["analysis"]["windows"]:
                obs.append({"replicate": key[0], "sequence": key[1], "position": b["position"], "arm": b["arm"],
                            "attempt": n, "clip": w["clip"], "energy": w["energy_j_per_replay"],
                            "latency": w["latency_s_per_replay"], "guard": w.get("guard_j_per_replay"),
                            "duration": w.get("duration_s"), "replays": w.get("replays")})
    if missing:
        raise Incomplete(f"no complete, jointly valid attempt for (replicate, sequence) {missing}; "
                         f"resume the stage before analysis")
    return {"stage": stage, "design": design, "obs": obs, "attempts": {f"{k[0]}/{k[1]}": v for k, v in attempts.items()}}


def metric(o: dict, which: str) -> float:
    return o["latency"] if which == "latency" else o["energy"][which]


def paired(data: dict, base: str, cand: str, which_c: str = "est", which_b: str = "est") -> dict:
    """{(replicate, sequence, clip): pct} for sequences holding both arms (one attempt each, by load)."""
    cell: dict = {}
    for o in data["obs"]:
        if o["arm"] in (base, cand):
            k = (o["replicate"], o["sequence"], o["clip"])
            cell.setdefault(k, {})[o["arm"]] = o
    out = {}
    for k, v in cell.items():
        if base in v and cand in v:
            if v[base]["attempt"] != v[cand]["attempt"]:
                raise AssertionError("pairing across attempts")
            b, c = metric(v[base], which_b), metric(v[cand], which_c)
            out[k] = 100.0 * (c - b) / b
    return out


def statistic(pcts: dict) -> float:
    by_clip: dict = {}
    for (_, _, clip), v in pcts.items():
        by_clip.setdefault(clip, []).append(v)
    return statistics.median(statistics.mean(v) for v in by_clip.values())


# --- descriptive summaries ----------------------------------------------------------------------------

def _clip_means(data: dict, get) -> dict:
    """{arm: {clip: mean of get(obs)}}; None where any window lacks the field."""
    acc: dict = {}
    for o in data["obs"]:
        acc.setdefault(o["arm"], {}).setdefault(o["clip"], []).append(get(o))
    return {a: {c: (statistics.mean(v) if all(x is not None for x in v) else None) for c, v in cl.items()}
            for a, cl in acc.items()}


def _median_or_none(values) -> float | None:
    v = list(values)
    return None if not v or any(x is None for x in v) else statistics.median(v)


def per_arm(data: dict, which: str) -> dict:
    return _clip_means(data, lambda o: metric(o, which))


def describe(data: dict) -> dict:
    """Per arm and clip, as separate columns: gross padded energy per replay (E_est, E_lower, E_upper), guard
    energy per replay, window duration, latency per replay and N. No idle-corrected energy."""
    cols = {"energy_j_per_replay_est": lambda o: o["energy"]["est"],
            "energy_j_per_replay_lower": lambda o: o["energy"]["lower"],
            "energy_j_per_replay_upper": lambda o: o["energy"]["upper"],
            "guard_j_per_replay": lambda o: o.get("guard"),
            "duration_s": lambda o: o.get("duration"),
            "latency_s_per_replay": lambda o: o["latency"],
            "replays": lambda o: o.get("replays")}
    m = {k: _clip_means(data, g) for k, g in cols.items()}
    out = {}
    for a in sorted(m["energy_j_per_replay_est"]):
        clips = sorted(m["energy_j_per_replay_est"][a])
        out[a] = {"energy_j_median_over_clips": _median_or_none(m["energy_j_per_replay_est"][a][c] for c in clips),
                  "latency_s_median_over_clips": _median_or_none(m["latency_s_per_replay"][a][c] for c in clips),
                  "guard_j_median_over_clips": _median_or_none(m["guard_j_per_replay"][a][c] for c in clips),
                  "duration_s_median_over_clips": _median_or_none(m["duration_s"][a][c] for c in clips),
                  "per_clip": {c: {k: m[k][a][c] for k in cols} for c in clips},
                  "energy_j_per_clip": m["energy_j_per_replay_est"][a],
                  "latency_s_per_clip": m["latency_s_per_replay"][a]}
    return out


def tune_pick(data: dict) -> dict:
    d = describe(data)
    pick = min(d, key=lambda a: (d[a]["energy_j_median_over_clips"], d[a]["latency_s_median_over_clips"], a))
    return {"pick": pick, "summary": d}


def screen(data: dict, baseline: str, candidates: list[str]) -> dict:
    stats = {c: statistic(paired(data, baseline, c)) for c in candidates}
    lat = {c: statistic(paired(data, baseline, c, "latency", "latency")) for c in candidates}
    best = min(candidates, key=lambda c: (stats[c], c))
    return {"statistic_pct_energy": stats, "statistic_pct_latency": lat,
            "confirmation_candidate": best if stats[best] < 0 else None,
            "outcome": f"confirm {best}" if stats[best] < 0 else "baseline wins (no negative screening median)"}


def gap(data: dict, arm: str, control: str) -> dict:
    return {"gap_pct_latency": statistic(paired(data, control, arm, "latency", "latency")),
            "gap_pct_energy": statistic(paired(data, control, arm)),
            "limit_pct": 25.0}


# --- exact tests and the rejection-rate bound ---------------------------------------------------------

def exact_2xk(valid: list[int], totals: list[int]) -> float:
    """Two-sided exact test of independence for a (valid, invalid) x level table with both margins fixed
    (Fisher for two levels, Freeman-Halton for more): the summed probability of every table no more likely
    than the observed one (relative tolerance 1e-9). Exact integer arithmetic."""
    lv = [(int(v), int(n)) for v, n in zip(valid, totals) if n > 0]
    ns = [n for _, n in lv]
    A, N = sum(v for v, _ in lv), sum(ns)
    if len(lv) < 2 or A == 0 or A == N:
        return 1.0
    w0 = math.prod(math.comb(n, v) for v, n in lv)
    tail = [sum(ns[j:]) for j in range(len(ns))] + [0]
    total = 0

    def walk(j: int, rem: int, w: int) -> None:
        nonlocal total
        if j == len(ns) - 1:
            ww = w * math.comb(ns[j], rem)
            if ww * 10**9 <= w0 * (10**9 + 1):
                total += ww
            return
        for a in range(max(0, rem - tail[j + 1]), min(ns[j], rem) + 1):
            walk(j + 1, rem - a, w * math.comb(ns[j], a))
    walk(0, A, 1)
    return min(1.0, total / math.comb(N, A))


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]]."""
    return exact_2xk([a, c], [a + b, c + d])


def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    lc = math.lgamma(n + 1)
    return sum(math.exp(lc - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
               for i in range(k + 1))


def clopper_pearson_upper(k: int, n: int, conf: float = CP_CONF) -> float:
    """One-sided upper confidence bound u for a binomial rate: P(X <= k | p = u) = 1 - conf."""
    if n <= 0 or k >= n:
        return 1.0
    alpha = 1.0 - conf
    if k <= 0:
        return 1.0 - alpha ** (1.0 / n)
    lo, hi = k / n, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return (lo + hi) / 2


# --- acceptance audit ---------------------------------------------------------------------------------

def _block_reasons(rec: dict) -> list[str]:
    a = rec.get("analysis") or {}
    out = list(a.get("block_invalid_reasons") or [])
    for w in a.get("windows") or []:
        if not w.get("valid", True):
            out += list(w.get("invalid_reasons") or [])
    return out


def _host_covariates(rec: dict) -> dict:
    a = rec.get("analysis") or {}
    idle = a.get("idle") or {}
    quiet = rec.get("quiet") or {}
    therm = (quiet.get("checks") or {}).get("thermal_nominal")
    ref = idle.get("reference") or {}           # runner.analyse_block: idle.reference = guard.idle_reference
    return {"idle_power_w": ref.get("p_ref_w"), "p_cluster_idle_share": idle.get("p_cluster_busy_max"),
            "load1": quiet.get("load1"), "thermal_nominal": None if therm is None else float(bool(therm))}


def _levels(blocks: list[dict], key: str) -> dict:
    out: dict = {}
    for b in blocks:
        d = out.setdefault(b[key], {"attempts": 0, "valid": 0})
        d["attempts"] += 1
        d["valid"] += int(b["valid"])
    for d in out.values():
        d["acceptance"] = d["valid"] / d["attempts"]
    return dict(sorted(out.items()))


def _dependence(levels: dict) -> tuple[float, float]:
    """(exact-test p, max - min acceptance in points) over the levels of one factor."""
    vals = list(levels.values())
    p = exact_2xk([d["valid"] for d in vals], [d["attempts"] for d in vals])
    acc = [d["acceptance"] for d in vals]
    return p, (100.0 * (max(acc) - min(acc)) if acc else 0.0)


def acceptance(stage_dir: str) -> dict:
    """Audit of every block attempt and sequence attempt of a stage (complete or not). Flags never gate."""
    stage = runner.read_json(os.path.join(stage_dir, "stage.json"))
    design = runner.read_json(os.path.join(stage_dir, "design.json"))
    fp = stage["fingerprint_sha256"]
    blocks, seq_atts = [], []
    for seq in runner.sequences(design["blocks"]):
        for n in runner.attempt_numbers(stage_dir, seq):
            pre = os.path.join(stage_dir, f"{runner.seq_prefix(seq)}.att{n}")
            end = runner.read_json(pre + ".end.json") if os.path.exists(pre + ".end.json") else None
            ok, reasons, att_blocks = end is not None and bool(end.get("valid")), [], []
            if end is None:
                reasons.append("sequence attempt has no end record (interrupted)")
            for b in seq:
                path = os.path.join(stage_dir, f"{b['block_id']}.att{n}.json")
                if not os.path.exists(path):
                    ok = False
                    reasons.append(f"{b['block_id']}: no block record for this attempt")
                    continue
                rec = runner.read_json(path)
                br = _block_reasons(rec)
                valid = bool((rec.get("analysis") or {}).get("valid"))
                if rec.get("block") != b or rec.get("attempt") != n or rec.get("fingerprint_sha256") != fp:
                    valid = False
                    br.append("record does not match the design block, the attempt or the fingerprint")
                if not valid and not br:
                    br.append("block invalid (no reason recorded)")
                br = list(dict.fromkeys(br))
                ok = ok and valid
                if not valid:
                    reasons += br
                info = {"block_id": b["block_id"], "arm": b["arm"], "position": b["position"], "valid": valid,
                        "reasons": br, "started_unix": rec.get("started_unix"), "ended_unix": rec.get("ended_unix"),
                        **_host_covariates(rec)}
                blocks.append(info)
                att_blocks.append(info)
            if end is not None and not end.get("valid") and not reasons:
                reasons.append("sequence end record invalid")
            seq_atts.append({"replicate": seq[0]["replicate"], "sequence": seq[0]["sequence"], "attempt": n,
                             "accepted": ok, "reasons": list(dict.fromkeys(reasons)),
                             "invalid_blocks": [{"block_id": x["block_id"], "arm": x["arm"], "position": x["position"],
                                                 "reasons": x["reasons"]} for x in att_blocks if not x["valid"]]})
    # time since the previous block attempt (execution order), where the records carry wall times
    timed = sorted((b for b in blocks if b["started_unix"] is not None), key=lambda b: b["started_unix"])
    prev = None
    for b in blocks:
        b["since_previous_block_s"] = None
    for b in timed:
        if prev is not None and prev["ended_unix"] is not None:
            b["since_previous_block_s"] = b["started_unix"] - prev["ended_unix"]
        prev = b
    host = {}
    for cov in ("idle_power_w", "p_cluster_idle_share", "load1", "thermal_nominal", "since_previous_block_s"):
        side = {}
        for label, want in (("valid", True), ("invalid", False)):
            xs = [b[cov] for b in blocks if b["valid"] is want and isinstance(b[cov], (int, float))]
            side[label] = {"n": len(xs), "mean": statistics.mean(xs) if xs else None}
        if side["valid"]["n"] or side["invalid"]["n"]:
            host[cov] = side
    n_blocks = len(blocks)
    overall = sum(b["valid"] for b in blocks) / n_blocks if n_blocks else None
    by_arm, by_pos = _levels(blocks, "arm"), _levels(blocks, "position")
    arm_p, arm_gap = _dependence(by_arm)
    pos_p, pos_gap = _dependence(by_pos)
    flags = []
    if overall is not None and overall < ACCEPT_MIN:
        flags.append(f"block acceptance {overall:.1%} < {ACCEPT_MIN:.0%}")
    for name, p, g in (("arm", arm_p, arm_gap), ("position", pos_p, pos_gap)):
        if p < FISHER_FLAG_P:
            flags.append(f"{name} dependence: exact test p = {p:.3g} < {FISHER_FLAG_P}")
        if g > GAP_FLAG_PP:
            flags.append(f"{name} acceptance gap {g:.1f} points > {GAP_FLAG_PP:.0f} points")
    rejected = [{k: s[k] for k in ("replicate", "sequence", "attempt", "reasons", "invalid_blocks")}
                for s in seq_atts if not s["accepted"]]
    n_seq = len(seq_atts)
    return {"attempts": n_blocks, "valid": sum(b["valid"] for b in blocks), "overall": overall,
            "by_arm": by_arm, "by_position": by_pos, "sequence_attempts": n_seq,
            "sequence_rejected": len(rejected),
            "sequence_acceptance": (n_seq - len(rejected)) / n_seq if n_seq else None,
            "arm_p": arm_p, "arm_gap_points": arm_gap, "position_p": pos_p, "position_gap_points": pos_gap,
            "host_state": host, "flags": flags, "rejected": rejected,
            "flag_rules": {"acceptance_min": ACCEPT_MIN, "exact_test_p": FISHER_FLAG_P, "gap_points": GAP_FLAG_PP,
                           "gating": False}}


# --- crossed bootstrap and the null simulation (numpy) ------------------------------------------------

def _replicate_ids(pcts: dict) -> list:
    return sorted({k[0] for k in pcts})


def to_array(pcts: dict):
    """pcts {(r, s, clip): v} -> (R, S, C) array; every cell must exist (complete design)."""
    import numpy as np
    reps = _replicate_ids(pcts)
    clips = sorted({k[2] for k in pcts})
    by_rep: dict = {}
    for (r, s, _) in pcts:
        by_rep.setdefault(r, set()).add(s)
    s_per = {len(v) for v in by_rep.values()}
    if len(s_per) != 1:
        raise Incomplete("replicates hold different numbers of sequences")
    S = s_per.pop()
    D = np.full((len(reps), S, len(clips)), np.nan)
    for (r, s, c), v in pcts.items():
        D[reps.index(r), sorted(by_rep[r]).index(s), clips.index(c)] = v
    if np.isnan(D).any():
        raise Incomplete("missing (replicate, sequence, clip) cells")
    return D


def stat_array(D) -> float:
    import numpy as np
    return float(np.median(D.mean(axis=(0, 1))))


def stat_means(M) -> float:
    """The statistic on (R, C) replicate means: median over clips of the mean over replicates."""
    import numpy as np
    return float(np.median(M.mean(axis=0)))


def replicate_means(D, n_imputed=None, envelope_pct: float | None = ENVELOPE_PCT):
    """(R, C) value of each replicate per clip: the mean over its S valid sequences and its n_imputed[r]
    imputed sequences at +envelope_pct (M15.3). Replicates without imputation are exactly D.mean(axis=1)."""
    import numpy as np
    M = D.mean(axis=1)
    if envelope_pct is None or n_imputed is None or not np.any(n_imputed):
        return M
    n = np.asarray(n_imputed, dtype=float)[:, None]
    imp = (D.sum(axis=1) + n * envelope_pct) / (D.shape[1] + n)
    return np.where(n > 0, imp, M)


def bootstrap(D, draws: int = BOOT_DRAWS, seed: int = BOOT_SEED, rng=None):
    """Crossed bootstrap over replicates and clips; returns (lo, hi, statistic per draw)."""
    return bootstrap_means(D.mean(axis=1), draws, seed, rng)   # the sequences of a replicate stay together


def bootstrap_means(A, draws: int = BOOT_DRAWS, seed: int = BOOT_SEED, rng=None):
    """The crossed bootstrap on (R, C) replicate means (each replicate's sequences, valid and imputed,
    already averaged); returns (lo, hi, statistic per draw), the 95% percentile interval."""
    import numpy as np
    rng = rng if rng is not None else np.random.default_rng(seed)
    R, C = A.shape
    reps = rng.integers(0, R, size=(draws, R))
    clips = rng.integers(0, C, size=(draws, C))          # one shared clip vector per draw
    W = (reps[:, :, None] == np.arange(R)[None, None, :]).sum(axis=1) / R
    M = np.take_along_axis(W @ A, clips, axis=1)         # each sampled clip's mean over the sampled replicates
    s = np.median(M, axis=1)
    lo, hi = np.percentile(s, [2.5, 97.5])
    return float(lo), float(hi), s


def variance_components(Y) -> dict:
    """Two-way (sequence x clip) method of moments on screening paired changes (conservative). Components
    below SIGMA_FLOOR_PCT are the round-off of identical paired changes and are reported as 0."""
    import numpy as np
    n_r, n_c = Y.shape
    grand = Y.mean()
    row, col = Y.mean(axis=1), Y.mean(axis=0)
    ms_row = n_c * ((row - grand) ** 2).sum() / (n_r - 1)
    ms_col = n_r * ((col - grand) ** 2).sum() / (n_c - 1)
    res = Y - row[:, None] - col[None, :] + grand
    ms_res = (res ** 2).sum() / ((n_r - 1) * (n_c - 1))

    def floor(x: float) -> float:
        x = float(x)
        return x if x > SIGMA_FLOOR_PCT else 0.0
    return {"sigma_replicate": floor(np.sqrt(ms_row / n_c)), "sigma_clip": floor(np.sqrt(ms_col / n_r)),
            "sigma_noise": floor(np.sqrt(ms_res)), "n_sequences": n_r, "n_clips": n_c,
            "sigma_floor_pct": SIGMA_FLOOR_PCT}


def _rejection_counts(rng, r: float, shape):
    """Rejected attempts per sequence before its accepted one: geometric in r, capped at the runner's
    attempt limit (a sequence at the cap leaves the stage incomplete; such stages exceed MAX_REJECTED)."""
    import numpy as np
    if r <= 0.0:
        return np.zeros(shape, dtype=int)
    if r >= 1.0:
        return np.full(shape, MAX_SEQUENCE_ATTEMPTS, dtype=int)
    return np.minimum(rng.geometric(1.0 - r, size=shape) - 1, MAX_SEQUENCE_ATTEMPTS)


def _simulate(sig: dict, R: int, S: int, C: int, sims: int, draws: int, seed: int, reject_rate: float,
              model: str, envelope_pct: float | None, mixture: tuple | None = None) -> dict:
    """The stage-2 procedure on simulated stages. Streams: `rng` (seed) draws the first attempts and the
    bootstrap exactly as the plain simulation; [seed, 1] the rejections and mixture outcomes; [seed, 2] the
    re-measured attempts. So r = 0 reproduces the plain simulation under either model."""
    import numpy as np
    rng = np.random.default_rng(seed)
    rej = np.random.default_rng([seed, 1])
    new = np.random.default_rng([seed, 2])
    s_rep, s_clip, s_e = sig["sigma_replicate"], sig["sigma_clip"], sig["sigma_noise"]
    cover = fi = zero = over_cap = 0
    n_rej = 0
    stat_sum = 0.0
    for _ in range(sims):
        u = rng.normal(0, s_rep, (R, 1, 1))
        v = rng.normal(0, s_clip, (1, 1, C))
        D = u + v + rng.normal(0, s_e, (R, S, C))
        n_imp = np.zeros(R, dtype=int)
        if mixture is not None:                                  # stress family: outcome-dependent rejection
            p, up, fav = mixture
            unfav = rej.random((R, S)) < p
            D = D + np.where(unfav, up, fav)[:, :, None]
            tries = np.zeros((R, S), dtype=int)
            while unfav.any() and (tries < MAX_SEQUENCE_ATTEMPTS).any():
                unfav &= tries < MAX_SEQUENCE_ATTEMPTS
                n_imp += unfav.sum(axis=1)
                tries += unfav
                again = rej.random((R, S)) < p
                fresh = u + v + new.normal(0, s_e, (R, S, C)) + np.where(again, up, fav)[:, :, None]
                D = np.where(unfav[:, :, None], fresh, D)
                unfav = unfav & again
        elif reject_rate > 0.0:
            counts = _rejection_counts(rej, reject_rate, (R, S))
            if model == "independent":
                n_imp = counts.sum(axis=1)
                fresh = u + v + new.normal(0, s_e, (R, S, C))
                D = np.where((counts > 0)[:, :, None], fresh, D)
            else:                                                # adversarial: the same number of rejections
                for _k in range(int(counts.sum())):              # on the least favourable accepted attempt
                    r_, s_ = np.unravel_index(int(np.argmax(D.mean(axis=2))), (R, S))
                    n_imp[r_] += 1
                    D[r_, s_] = u[r_, 0, 0] + v[0, 0] + new.normal(0, s_e, C)
        K = int(n_imp.sum())
        M = replicate_means(D, n_imp, envelope_pct)
        stat_sum += float(stat_means(M))                         # the stage's point statistic (as gated)
        lo, hi, _ = bootstrap_means(M, draws, rng=rng)
        cover += lo <= 0.0 <= hi
        fi += hi < 0.0 and K <= MAX_REJECTED
        zero += K == 0
        over_cap += K > MAX_REJECTED
        n_rej += K
    return {"coverage": cover / sims, "false_improvement_rate": fi / sims, "zero_rejection_rate": zero / sims,
            "over_cap_rate": over_cap / sims, "mean_rejected": n_rej / sims, "mean_statistic": stat_sum / sims,
            "sims": sims, "draws": draws,
            "seed": seed, "reject_rate": reject_rate, "model": model, "envelope_pct": envelope_pct}


def simulate(sig: dict, R: int, S: int, C: int, sims: int = SIM_COUNT, draws: int = BOOT_DRAWS,
             seed: int = SIM_SEED, reject_rate: float = 0.0, model: str = "independent",
             envelope_pct: float | None = ENVELOPE_PCT) -> dict:
    """One grid point of the full stage-2 gate on null data u_r + v_c + e_rsc: sequence attempts rejected
    at rate `reject_rate` (independent, or adversarial: least favourable to the candidate), re-measured,
    imputed at +envelope_pct (None: simply dropped), the crossed bootstrap, and a declaration only if the
    upper bound is < 0 with at most MAX_REJECTED rejections. Coverage is of 0 by the (imputed) interval."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    return _simulate(sig, R, S, C, sims, draws, seed, float(reject_rate), model, envelope_pct)


def simulate_grid(sig: dict, R: int, S: int, C: int, r_upper: float, sims: int = SIM_COUNT,
                  draws: int = BOOT_DRAWS, seed: int = SIM_SEED) -> dict:
    """M15.4: r in {0, r_U/4, r_U/2, 3 r_U/4, r_U} x {independent, adversarial}, common random numbers
    (one seed); pass = max false-improvement rate <= FALSE_IMPROVEMENT_MAX and coverage at r = 0 >=
    COVERAGE_MIN. r = 0 is one simulation (both models coincide there)."""
    rates = [0.0, r_upper / 4, r_upper / 2, 3 * r_upper / 4, r_upper]
    r0 = simulate(sig, R, S, C, sims, draws, seed)
    grid = []
    for model in MODELS:
        for r in rates:
            res = r0 if r == 0.0 else simulate(sig, R, S, C, sims, draws, seed, reject_rate=r, model=model)
            grid.append({"r": r, "model": model, "false_improvement_rate": res["false_improvement_rate"],
                         "coverage": res["coverage"], "zero_rejection_rate": res["zero_rejection_rate"],
                         "over_cap_rate": res["over_cap_rate"], "mean_rejected": res["mean_rejected"],
                         "mean_statistic": res["mean_statistic"]})
    max_fi = max(p["false_improvement_rate"] for p in grid)
    return {"grid": grid, "rates": rates, "r_upper": r_upper, "max_false_improvement": max_fi,
            "coverage_r0": r0["coverage"],
            "pass": max_fi <= FALSE_IMPROVEMENT_MAX and r0["coverage"] >= COVERAGE_MIN,
            "limits": {"false_improvement_max": FALSE_IMPROVEMENT_MAX, "coverage_min": COVERAGE_MIN},
            "envelope_pct": ENVELOPE_PCT, "max_rejected": MAX_REJECTED, "sims": sims, "draws": draws, "seed": seed,
            "generating_model": GENERATING_MODEL}


def stress_family(sig: dict, R: int, S: int, C: int, sims: int = SIM_COUNT, draws: int = BOOT_DRAWS,
                  seed: int = SIM_SEED) -> list[dict]:
    """M16.3 (reported, not gating): a +45% outcome with probability p, else -p 45 / (1 - p) (mean change 0),
    on top of the screening noise; every unfavourable attempt is rejected and re-measured, then the full
    gate (imputation, cap, bootstrap). Zero-rejection realisations are included."""
    out = []
    for p in STRESS_P:
        u = STRESS_UNFAVOURABLE_PCT
        f = -p * u / (1 - p)
        res = _simulate(sig, R, S, C, sims, draws, seed, 0.0, "stress", ENVELOPE_PCT, mixture=(p, u, f))
        out.append({"p": p, "unfavourable_pct": u, "favourable_pct": f,
                    "false_improvement_rate": res["false_improvement_rate"],
                    "zero_rejection_rate": res["zero_rejection_rate"], "coverage": res["coverage"],
                    "mean_rejected": res["mean_rejected"], "sims": sims, "draws": draws, "seed": seed})
    return out


# --- verdict ------------------------------------------------------------------------------------------

def _stress_text(stress: list[dict] | None) -> str:
    if not stress:
        return ""
    cases = "; ".join(f"p = {c['p']:.0%}: false-improvement rate {c['false_improvement_rate']:.1%}, zero-rejection "
                      f"realisations {c['zero_rejection_rate']:.1%}" for c in stress)
    return f" (+{STRESS_UNFAVOURABLE_PCT:.0f}% outcomes, always rejected, mean change 0: {cases})"


def verdict(ci_m10: tuple, coverage: dict, rejected: int = 0, stress: list[dict] | None = None,
            flags: list[str] | None = None) -> dict:
    """The M10 gate. `coverage`: {"coverage" (at r = 0), "false_improvement_rate" (the grid maximum)}.
    Descriptive only if the simulation check fails or more than MAX_REJECTED sequence attempts were
    rejected; otherwise improvement iff the imputed interval's upper bound < 0. Audit flags never gate."""
    conditions = list(CONDITIONS)
    conditions[-1] += _stress_text(stress)
    reasons = []
    if rejected > MAX_REJECTED:
        reasons.append(f"{rejected} rejected sequence attempts > {MAX_REJECTED}: the rejection cap is exceeded")
    if coverage["coverage"] < COVERAGE_MIN:
        reasons.append(f"simulation: coverage at r = 0 {coverage['coverage']:.3f} < {COVERAGE_MIN}")
    if coverage["false_improvement_rate"] > FALSE_IMPROVEMENT_MAX:
        reasons.append(f"simulation: maximum false-improvement rate {coverage['false_improvement_rate']:.3f} "
                       f"> {FALSE_IMPROVEMENT_MAX}")
    if reasons:
        return {"descriptive_only": True, "improvement": False, "reasons": reasons, "conditions": conditions,
                "verdict": "descriptive only: " + "; ".join(reasons) + " (no inferential claim)"}
    imp = ci_m10[1] < 0.0
    reasons.append(f"imputed M10 interval upper bound {ci_m10[1]:.4g} {'<' if imp else '>='} 0; "
                   f"{rejected} rejected sequence attempt(s) <= {MAX_REJECTED}; simulation check passed")
    if flags:
        reasons.append("audit flags (reported, not gating): " + "; ".join(flags))
    if not imp:
        return {"descriptive_only": False, "improvement": False, "reasons": reasons, "conditions": conditions,
                "verdict": "baseline wins"}
    text = ("improvement (imputed M10 interval upper bound < 0), conditional on: "
            + "; ".join(f"({i + 1}) {c}" for i, c in enumerate(conditions)))
    if flags:
        text += f". Audit flags (reported, not gating): {'; '.join(flags)}"
    return {"descriptive_only": False, "improvement": True, "reasons": reasons, "conditions": conditions,
            "verdict": text}


def confirm(confirm_dir: str, screen_dir: str, baseline: str, candidate: str, sims: int = SIM_COUNT) -> dict:
    conf = load_stage(confirm_dir)
    scr = load_stage(screen_dir)
    acc = acceptance(confirm_dir)
    p_est = paired(conf, baseline, candidate)
    p_m10 = paired(conf, baseline, candidate, which_c="upper", which_b="lower")
    D_est, D_m10 = to_array(p_est), to_array(p_m10)
    reps = _replicate_ids(p_m10)
    n_imp = [sum(1 for x in acc["rejected"] if x["replicate"] == r) for r in reps]
    K = acc["sequence_rejected"]
    if sum(n_imp) != K:
        raise Incomplete("rejected sequence attempts outside the analysed replicates")
    M_imp = replicate_means(D_m10, n_imp, ENVELOPE_PCT)
    lo_e, hi_e, _ = bootstrap(D_est)
    lo_0, hi_0, _ = bootstrap(D_m10)
    lo_m, hi_m, _ = bootstrap_means(M_imp)
    Y = to_array(paired(scr, baseline, candidate))
    sig = variance_components(Y.reshape(-1, Y.shape[2]))  # rows = the screening's (replicate, sequence) cells
    R, S, C = D_est.shape
    r_upper = clopper_pearson_upper(K, acc["sequence_attempts"])
    grid = simulate_grid(sig, R, S, C, r_upper, sims=sims)
    stress = stress_family(sig, R, S, C, sims=sims)
    check = {"coverage": grid["coverage_r0"], "false_improvement_rate": grid["max_false_improvement"]}
    metric_m10 = "candidate E_upper vs baseline E_lower per window"
    out = {"baseline": baseline, "candidate": candidate, "replicates": R, "sequences_per_replicate": S, "clips": C,
           "attempts_used": conf["attempts"],
           "point_pct_energy_est": stat_array(D_est), "ci95_energy_est": [lo_e, hi_e],
           "m10": {"metric": metric_m10, "imputation": f"+{ENVELOPE_PCT:.0f}% per rejected sequence attempt",
                   "point_pct": stat_means(M_imp), "ci95": [lo_m, hi_m],
                   "imputed_per_replicate": dict(zip((str(r) for r in reps), n_imp))},
           "m10_without_imputation": {"metric": metric_m10, "point_pct": stat_array(D_m10), "ci95": [lo_0, hi_0]},
           "imputed_sequences": K, "imputed_change_pct": ENVELOPE_PCT, "max_rejected": MAX_REJECTED,
           "r_upper": r_upper,
           "bootstrap": {"draws": BOOT_DRAWS, "seed": BOOT_SEED, "type": "crossed: replicates x shared clip vector"},
           "acceptance": acc,
           "simulation": {**grid, "rejection_rate_basis": {"rejected": K, "sequence_attempts": acc["sequence_attempts"],
                                                           "confidence": CP_CONF},
                          "calibration": {"screen_dir": screen_dir, **sig}, "stress_family": stress},
           "coverage_check": {**check, "pass": grid["pass"], "calibration": {"screen_dir": screen_dir, **sig},
                              "limits": {"coverage_min": COVERAGE_MIN, "false_improvement_max": FALSE_IMPROVEMENT_MAX}}}
    out.update(verdict((lo_m, hi_m), check, rejected=K, stress=stress, flags=acc["flags"]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=["describe", "acceptance", "screen", "tune", "gap", "confirm"])
    p.add_argument("stage_dir")
    p.add_argument("--baseline", default="R-MB1")
    p.add_argument("--candidates", default="V,T,P")
    p.add_argument("--candidate")
    p.add_argument("--screen")
    p.add_argument("--arm", default="R-MB1")
    p.add_argument("--control", default="CTRL-MB1")
    a = p.parse_args()
    try:
        if a.cmd == "confirm":
            out = confirm(a.stage_dir, a.screen, a.baseline, a.candidate)
        elif a.cmd == "acceptance":
            out = acceptance(a.stage_dir)
        else:
            data = load_stage(a.stage_dir)
            fn = {"describe": lambda: describe(data), "tune": lambda: tune_pick(data),
                  "screen": lambda: screen(data, a.baseline, a.candidates.split(",")),
                  "gap": lambda: gap(data, a.arm, a.control)}[a.cmd]
            out = {"attempts_used": data["attempts"], **fn(), "acceptance": acceptance(a.stage_dir)}
    except Incomplete as e:
        print(json.dumps({"refused": str(e)}))
        sys.exit(3)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
