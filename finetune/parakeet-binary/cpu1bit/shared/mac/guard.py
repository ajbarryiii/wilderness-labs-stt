"""Spawn-based mapping, guard-banded window energy bounds and the per-block A1-A3 checks
(plans/cpu-benchmark-mac.md "Measurement": "Energy", "Mapping", "Samples summed", "Window validity",
"Bounds", "Assumptions A1-A3 and validation"; history M13.3, M14.2-M14.5, M15.1, M15.6, M16.1-M16.4).
Contract: tests/test_guard.py. Stdlib only (runs on the Mac under /usr/bin/python3 3.9).

Mapping. powermetrics is spawned by the runner; s = CLOCK_UPTIME_RAW read just before the spawn. The
samples are mapped from the leading run of parsed records: from the capture's first record up to the first
malformed record or the first lost record that pm's segmentation detects (pm._split_lost: a persistent
step in arrival - cumulative elapsed). Within it, sample k (1-based) has elapsed el_k, cumulative elapsed
c_k = sum_{j<=k} el_j (c_0 = 0) and delivery time d_k. Under A1-A3 its interval is
[s + o + c_{k-1}(1 +- rho), s + o + c_k(1 +- rho)] with one offset o in [-delta_A2, U], where
U = min_k (d_k - s - c_k(1 - rho)) over the whole run (a sample is not delivered before it ends; U is a
block quantity, the tightest causal bound) and delta_A2 >= 0 is the A2 pre-spawn residual (default 0).
So per sample:
  earliest start es_k = s - delta_A2 + c_{k-1}(1 - rho)    latest start ls_k = s + U + c_{k-1}(1 + rho)
  earliest end   ee_k = s - delta_A2 + c_k(1 - rho)        latest end   le_k = s + U + c_k(1 + rho)
A sample lies inside [lo, hi] for every admissible offset iff es_k >= lo and le_k <= hi; it can meet
(lo, hi) for some offset iff le_k > lo and es_k < hi. U < 0 contradicts causality (block invalid);
U > U_MAX invalidates the window.

Window [t0, t1], padded interval PI = [p0, p1] = [t0 - G, t1 + G]:
  P_out = samples that can meet PI; P_in = samples inside PI for every offset; X = P_out - P_in.
  E_upper = sum_{P_out} (E + q); E_lower = sum_{P_in} (E - q); E_est = the energy of P_out prorated by
  its overlap with PI at the mid-offset o_mid = (U - delta_A2) / 2 with rate 1 (every P_in sample counts
  whole, so E_lower <= E_est <= E_upper). Joules, for the whole window (the runner divides by N).
  guard_j = (sum E / sum el over the samples inside [t0 - G, t0] or [t1, t1 + G] for every offset) * 2G
  (descriptive; None without such a sample).
Validity (every failing check adds a reason): idle reference valid; U in [0, U_MAX]; coverage (P_out in
the mapped run, the first sample of P_out starts by p0 and the last ends after p1 for every offset, and
the bracketing record, the one after P_out, whose earliest start is >= p1, has arrived); every P_out
sample <= L_MAX; block age at the bracketing record's latest end <= A_MAX; |P_in| >= 2K; and, once
coverage is complete (a truncated P_out has no meaningful edges), edges: every X sample and the K
outermost P_in samples on each side have excess E - p_ref el <= tol; work: some sample inside W for
every offset has excess > tol. Flags (never gating): the delivery-span corroboration
|(d_last - d_first) - (c_last - c_first)| over P_out above J + rho (c_last - c_first), when J is given
(the runner flags windows shorter than 12 s).

Units: times and lengths in ns (ints), energies in mJ unless the key ends in _j, powers in W.
Every public function is total on data: bad input gives an invalid result with a reason ([] or NaN for
samples_within and spin_excess_power_w), never an exception.
"""
from __future__ import annotations

import functools
import math
import statistics
from collections import deque

import pm

MS = 1_000_000
S = 1_000_000_000

G_NS = 1 * S                  # guard on each side of W
S_NS = 1200 * MS              # settle after the untimed replay, before the leading guard
S_POST_NS = 1 * S             # settle after the trailing guard, before the next clip load
K = 2                         # outermost P_in samples per side that must be idle-level
L_MAX_NS = 130 * MS           # longest admissible sample in P_out
A_MAX_NS = 600 * S            # block age: spawn to the bracketing record's latest end
U_MAX_NS = 600 * MS           # largest admissible upper offset bound
RHO = 50e-6                   # |rate error| of elapsed_ns against CLOCK_UPTIME_RAW (A3)
TAU_BAND_NS = 50 * MS         # A1 band slack: width <= 1 s + tau
Q_MJ = 1.0                    # integer-mJ reporting granularity

BAND_LIMIT_NS = 1 * S + TAU_BAND_NS
TOL_FLOOR_MJ = 5.0            # M13.3: tol = max(5 mJ, largest idle deviation + 3 mJ)
TOL_MARGIN_MJ = 3.0
IDLE_MIN_SAMPLES = 8          # sanity floor for an idle reference (a 10 s idle window holds ~90)
JITTER_STRETCH_NS = 10 * S    # A3: J = largest range of d - c within any 10 s stretch
LOOKBACK_NS = 1 * S           # energy-interval check: look-back region before each spin's start
SPIN_SETTLE_NS = G_NS         # ... which must start at least this long after the previous spin's end


# --- input hygiene ------------------------------------------------------------------------------------

def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _as_list(samples):
    if samples is None or isinstance(samples, (str, bytes)):
        return None
    try:
        return list(samples)
    except TypeError:
        return None


def _fields(x):
    """(el_ns, energy_mj, arrival_ns) of a parsed record, or None when the record is malformed (not ok, or
    fields corrupted after parsing)."""
    if getattr(x, "ok", False) is not True:
        return None
    el, e, d = getattr(x, "elapsed_ns", None), getattr(x, "cpu_energy_mj", None), getattr(x, "arrival_ns", None)
    if not _is_int(el) or el <= 0 or not _is_int(d):
        return None
    if isinstance(e, bool) or not isinstance(e, (int, float)) or not math.isfinite(e) or e < 0:
        return None
    return el, float(e), d


def _timestamp_ns(x):
    """The record's 1 s `timestamp` (POSIX) in integer ns, or None."""
    ts = getattr(x, "timestamp_s", None)
    if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
        return None
    whole = math.floor(ts)
    return int(whole) * S + int(round((ts - whole) * S))


def _ref_values(ref):
    """(p_ref_w, tol_mj, "") from an idle_reference result, or (None, None, reason)."""
    if not isinstance(ref, dict) or ref.get("valid") is not True:
        return None, None, "idle reference: missing or invalid"
    p, tol = ref.get("p_ref_w"), ref.get("tol_mj")
    for v in (p, tol):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            return None, None, "idle reference: p_ref_w / tol_mj not finite and non-negative"
    if tol <= 0:
        return None, None, "idle reference: tol_mj must be positive"
    return float(p), float(tol), ""


def _spins(spins):
    """Sorted [(on, off)] with integer ns and off > on, or (None, reason)."""
    xs = _as_list(spins)
    if xs is None:
        return None, "spins: not a sequence"
    out = []
    for i, sp in enumerate(xs):
        try:
            on, off = sp
        except (TypeError, ValueError):
            return None, f"spins: entry {i} is not an (on_ns, off_ns) pair"
        if not (_is_int(on) and _is_int(off)) or off <= on:
            return None, f"spins: entry {i} needs integer ns with off > on"
        out.append((on, off))
    if not out:
        return None, "spins: none given"
    return sorted(out), ""


def _total(fallback):
    """Safety net behind the explicit checks: an unexpected failure on data becomes an invalid result."""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 - total on data: never raise, fail closed
                return fallback(f"{fn.__name__}: unexpected input ({type(e).__name__}: {e})")
        return inner
    return wrap


def _invalid(template):
    def make(reason):
        out = template()
        out["valid"] = False
        out["reasons"] = [reason]
        return out
    return make


# --- the spawn-based mapping ----------------------------------------------------------------------------

class _Map:
    """The mapping of the leading run of parsed records (see the module doc)."""

    def __init__(self, run, fields, spawn_ns, delta_ns, end_reason):
        self.run, self.end_reason, self.s, self.delta = run, end_reason, spawn_ns, delta_ns
        self.el = [f[0] for f in fields]
        self.e = [f[1] for f in fields]
        self.d = [f[2] for f in fields]
        self.c, c = [], 0
        for el in self.el:
            c += el
            self.c.append(c)
        self.cp = [c - el for c, el in zip(self.c, self.el)]
        self.u = min(d - spawn_ns - c * (1 - RHO) for d, c in zip(self.d, self.c))
        lo, hi = spawn_ns - delta_ns, spawn_ns + self.u
        self.es = [lo + c * (1 - RHO) for c in self.cp]
        self.ls = [hi + c * (1 + RHO) for c in self.cp]
        self.ee = [lo + c * (1 - RHO) for c in self.c]
        self.le = [hi + c * (1 + RHO) for c in self.c]
        self.n = len(run)

    def u_ns(self) -> int:
        return int(math.ceil(self.u))

    def excess_mj(self, k: int, p_ref_w: float) -> float:
        return self.e[k] - p_ref_w * self.el[k] * 1e-6

    def inside(self, lo, hi) -> list[int]:
        return [k for k in range(self.n) if self.es[k] >= lo and self.le[k] <= hi]


def _mapping(samples, spawn_ns, delta_a2_ns):
    """(_Map, "") or (None, reason)."""
    if not _is_int(spawn_ns):
        return None, "mapping: spawn time is not an integer ns"
    if not _is_int(delta_a2_ns) or delta_a2_ns < 0:
        return None, "mapping: delta_a2_ns must be a non-negative integer ns"
    xs = _as_list(samples)
    if xs is None:
        return None, "mapping: samples is not a sequence"
    run, fields, end_reason = [], [], ""
    for i, x in enumerate(xs):
        f = _fields(x)
        if f is None:
            end_reason = f"malformed record at position {i}"
            break
        run.append(x)
        fields.append(f)
    if run:
        first = pm._split_lost(run, pm.GAP_TOLERANCE_NS)[0]
        if len(first) < len(run):
            end_reason = (f"lost record before position {len(first)} "
                          "(persistent step in arrival - cumulative elapsed)")
            run, fields = run[:len(first)], fields[:len(first)]
    if not run:
        return None, f"mapping: no parsed record from the capture's first record ({end_reason or 'no records'})"
    return _Map(run, fields, spawn_ns, delta_a2_ns, end_reason), ""


# --- idle reference (M13.3) ------------------------------------------------------------------------------

def _ref_template():
    return {"valid": False, "reasons": [], "p_ref_w": None, "tol_mj": None, "max_dev_mj": None, "n": 0}


@_total(_invalid(_ref_template))
def idle_reference(idle_samples) -> dict:
    """p_ref = median per-sample power (W) and tol = max(5 mJ, max_i |E_i - p_ref el_i| + 3 mJ) over the idle
    window's samples (select them with samples_within). Any malformed sample, or fewer than
    IDLE_MIN_SAMPLES, makes the reference invalid. -> {"valid", "reasons", "p_ref_w", "tol_mj",
    "max_dev_mj", "n"}."""
    out = _ref_template()
    xs = _as_list(idle_samples)
    if xs is None:
        out["reasons"].append("idle reference: samples is not a sequence")
        return out
    fs = [_fields(x) for x in xs]
    if any(f is None for f in fs):
        out["reasons"].append(f"idle reference: {sum(f is None for f in fs)} malformed sample(s)")
        return out
    out["n"] = len(fs)
    if len(fs) < IDLE_MIN_SAMPLES:
        out["reasons"].append(f"idle reference: {len(fs)} sample(s) < {IDLE_MIN_SAMPLES}")
        return out
    p_ref = statistics.median(e * 1e6 / el for el, e, _ in fs)
    dev = max(abs(e - p_ref * el * 1e-6) for el, e, _ in fs)
    out.update(valid=True, p_ref_w=p_ref, max_dev_mj=dev, tol_mj=max(TOL_FLOOR_MJ, dev + TOL_MARGIN_MJ))
    return out


# --- window bounds (M14.2-M14.5, M15.1, M15.6, M16.1) ---------------------------------------------------

def _window_template():
    return {"valid": False, "reasons": [], "flags": [], "u_ns": None, "delta_a2_ns": None, "first_bound_ns": None,
            "p_out_count": 0, "p_in_count": 0, "x_count": 0, "e_lower_j": None, "e_est_j": None, "e_upper_j": None,
            "guard_j": None, "guard_samples": 0, "edge_excess_mj": {"leading": [], "trailing": []},
            "max_edge_excess_mj": None, "work_max_excess_mj": None, "l_max_ns": None, "age_ns": None,
            "bracketed": False, "delivery_mismatch_ns": None, "p_ref_w": None, "tol_mj": None}


@_total(_invalid(_window_template))
def window_bounds(samples, spawn_ns, t0, t1, ref, delivery_jitter_ns=None, delta_a2_ns=0) -> dict:
    """Energy bounds and validity of the timed window [t0, t1] (CLOCK_UPTIME_RAW ns) over the padded
    interval [t0 - G, t1 + G]; see the module doc. samples: the block's records in arrival order, from the
    capture's first record. -> {"valid", "reasons", "flags", "u_ns", "delta_a2_ns", "first_bound_ns"
    (d_1 - s - el_1), "p_out_count", "p_in_count", "x_count", "e_lower_j", "e_est_j", "e_upper_j",
    "guard_j", "guard_samples", "edge_excess_mj" {"leading", "trailing"} (X then the K outermost P_in
    samples, in time order), "max_edge_excess_mj", "work_max_excess_mj", "l_max_ns", "age_ns",
    "bracketed", "delivery_mismatch_ns" (signed, over P_out), "p_ref_w", "tol_mj"}. Energies are for the
    whole window (not per replay)."""
    out = _window_template()
    reasons, flags = out["reasons"], out["flags"]
    out["delta_a2_ns"] = delta_a2_ns
    if not (_is_int(t0) and _is_int(t1)) or t1 <= t0:
        reasons.append("window: t0 and t1 must be integer ns with t1 > t0")
        return out
    if delivery_jitter_ns is not None and (isinstance(delivery_jitter_ns, bool)
                                           or not isinstance(delivery_jitter_ns, (int, float))
                                           or not math.isfinite(delivery_jitter_ns) or delivery_jitter_ns < 0):
        reasons.append("window: delivery_jitter_ns must be None or a finite non-negative number")
        return out
    p_ref, tol, why = _ref_values(ref)
    if why:
        reasons.append(why)
    out.update(p_ref_w=p_ref, tol_mj=tol)
    m, why = _mapping(samples, spawn_ns, delta_a2_ns)
    if m is None:
        reasons.append(why)
        return out
    s = spawn_ns
    out["u_ns"] = m.u_ns()
    out["first_bound_ns"] = m.d[0] - s - m.el[0]
    if m.u < 0:
        reasons.append(f"alignment: causality contradiction, U = {m.u / MS:.1f} ms < 0 (a sample delivered before "
                       "spawn + its cumulative elapsed; invalidates the block)")
        return out
    if m.u > U_MAX_NS:
        reasons.append(f"alignment: U = {m.u / MS:.1f} ms > U_MAX = {U_MAX_NS / MS:.0f} ms")
    p0, p1 = t0 - G_NS, t1 + G_NS
    ks = [k for k in range(m.n) if m.le[k] > p0 and m.es[k] < p1]
    if not ks:
        where = f" (the parsed run ends: {m.end_reason})" if m.end_reason else ""
        reasons.append(f"coverage: no mapped sample can meet [t0 - G, t1 + G]{where}")
        return out
    first, last = ks[0], ks[-1]
    if m.ls[first] > p0:
        reasons.append("coverage: the first sample may start after t0 - G")
    bracket = last + 1 if last + 1 < m.n else None
    out["bracketed"] = bracket is not None
    if bracket is None:
        if m.end_reason:
            reasons.append(f"coverage: the parsed run ends at the padded region ({m.end_reason})")
        else:
            reasons.append("bracketing: the capture ends before the record whose earliest start exceeds t1 + G")
    if m.ee[last] < p1:
        reasons.append("coverage: the last mapped sample may end before t1 + G")
    covered = m.ls[first] <= p0 and bracket is not None and m.ee[last] >= p1
    out["l_max_ns"] = max(m.el[k] for k in ks)
    if out["l_max_ns"] > L_MAX_NS:
        reasons.append(f"sample length: {out['l_max_ns'] / MS:.1f} ms > L_MAX = {L_MAX_NS / MS:.0f} ms in P_out")
    age = (m.le[bracket] if bracket is not None else p1) - s
    out["age_ns"] = int(math.ceil(age))
    if age > A_MAX_NS:
        reasons.append(f"block age: {age / S:.1f} s at the bracketing record > A_MAX = {A_MAX_NS / S:.0f} s")
    k_in = [k for k in ks if m.es[k] >= p0 and m.le[k] <= p1]
    out.update(p_out_count=len(ks), p_in_count=len(k_in), x_count=len(ks) - len(k_in))
    if len(k_in) < 2 * K:
        reasons.append(f"coverage: {len(k_in)} sample(s) inside [t0 - G, t1 + G] for every offset < 2K = {2 * K}")
    # bounds (mJ -> J)
    out["e_upper_j"] = sum(m.e[k] + Q_MJ for k in ks) / 1000.0
    out["e_lower_j"] = sum(m.e[k] - Q_MJ for k in k_in) / 1000.0
    o_mid = (m.u - m.delta) / 2
    est = 0.0
    for k in ks:
        a = s + o_mid + m.cp[k]
        ov = min(a + m.el[k], p1) - max(a, p0)
        if ov > 0:
            est += m.e[k] * ov / m.el[k]
    out["e_est_j"] = est / 1000.0
    # guard energy (descriptive)
    kg = [k for k in ks if (m.es[k] >= p0 and m.le[k] <= t0) or (m.es[k] >= t1 and m.le[k] <= p1)]
    out["guard_samples"] = len(kg)
    if kg:
        p_guard = sum(m.e[k] for k in kg) * 1e6 / sum(m.el[k] for k in kg)
        out["guard_j"] = p_guard * (2 * G_NS) / S
    # delivery-span corroboration (flag only)
    span_c = m.c[last] - m.c[first]
    out["delivery_mismatch_ns"] = (m.d[last] - m.d[first]) - span_c
    if delivery_jitter_ns is not None and abs(out["delivery_mismatch_ns"]) > delivery_jitter_ns + RHO * span_c:
        flags.append(f"delivery span differs from the summed elapsed by {out['delivery_mismatch_ns'] / MS:.1f} ms "
                     f"> J = {delivery_jitter_ns / MS:.1f} ms (+ rho)")
    if p_ref is None or not covered:     # edges and work are judged only on a complete P_out
        return out
    # edges and work present
    if k_in:
        lead = [k for k in ks if k < k_in[0]] + k_in[:K]
        trail = k_in[-K:] + [k for k in ks if k > k_in[-1]]
    else:
        lead, trail = ks, []
    out["edge_excess_mj"] = {"leading": [m.excess_mj(k, p_ref) for k in lead],
                             "trailing": [m.excess_mj(k, p_ref) for k in trail]}
    edge = out["edge_excess_mj"]["leading"] + out["edge_excess_mj"]["trailing"]
    out["max_edge_excess_mj"] = max(edge) if edge else None
    for side in ("leading", "trailing"):
        hot = [x for x in out["edge_excess_mj"][side] if x > tol]
        if hot:
            reasons.append(f"edge: {len(hot)} {side} edge sample(s) above idle level "
                           f"(max excess {max(hot):.1f} mJ > tol {tol:.1f} mJ)")
    kw = [k for k in k_in if m.es[k] >= t0 and m.le[k] <= t1]
    out["work_max_excess_mj"] = max((m.excess_mj(k, p_ref) for k in kw), default=None)
    if out["work_max_excess_mj"] is None or out["work_max_excess_mj"] <= tol:
        reasons.append("work: no sample inside W for every offset exceeds tol")
    out["valid"] = not reasons
    return out


@_total(lambda reason: [])
def samples_within(samples, spawn_ns, lo_ns, hi_ns, delta_a2_ns=0) -> list:
    """The parsed samples whose mapped interval lies inside [lo_ns, hi_ns] for every admissible offset (for
    the idle window's reference and P-cluster share). [] when the mapping fails (no parsed run, U < 0, bad
    arguments) or the parsed run does not reach past hi_ns (a malformed or lost record inside, or the
    capture ending, could hide samples)."""
    if not (_is_int(lo_ns) and _is_int(hi_ns)) or hi_ns <= lo_ns:
        return []
    m, _ = _mapping(samples, spawn_ns, delta_a2_ns)
    if m is None or m.u < 0 or m.es[-1] < hi_ns:
        return []
    return [m.run[k] for k in m.inside(lo_ns, hi_ns)]


# --- energy-interval falsification and spin power (M15.1, M16.1) ----------------------------------------

def _eic_template():
    return {"valid": False, "reasons": [], "max_excess_mj": None, "u_ns": None, "spins": []}


@_total(_invalid(_eic_template))
def energy_interval_check(samples, spawn_ns, spins, ref, delta_a2_ns=0, lookback_ns=LOOKBACK_NS,
                          settle_ns=SPIN_SETTLE_NS) -> dict:
    """M15.1: for each one-thread spin [on, off] of known timing, no sample lying, for every admissible
    offset, inside the look-back region [on - lookback_ns, on] (so ending before `on`) may carry excess above
    tol. Each region must lie in the mapped run, after the first sample's latest start and at least settle_ns
    after the previous spin's end, and must hold at least one sample. -> {"valid", "reasons",
    "max_excess_mj", "u_ns", "spins": [{"on_ns", "off_ns", "checked", "max_excess_mj"}]}."""
    out = _eic_template()
    reasons = out["reasons"]
    p_ref, tol, why = _ref_values(ref)
    if why:
        reasons.append(why)
    sp, why = _spins(spins)
    if why:
        reasons.append(why)
    if not all(_is_int(v) and v >= 0 for v in (lookback_ns, settle_ns)) or lookback_ns < LOOKBACK_NS:
        reasons.append(f"energy interval: lookback_ns must be an integer >= {LOOKBACK_NS} and settle_ns >= 0")
    m, why = _mapping(samples, spawn_ns, delta_a2_ns)
    if why:
        reasons.append(why)
    if reasons:
        return out
    out["u_ns"] = m.u_ns()
    if m.u < 0:
        reasons.append(f"alignment: causality contradiction, U = {m.u / MS:.1f} ms < 0")
        return out
    worst = None
    for i, (on, off) in enumerate(sp):
        lo = on - lookback_ns
        if i and lo < sp[i - 1][1] + settle_ns:
            reasons.append(f"spin {i}: look-back region starts before the previous spin's end + settle")
        if lo < m.ls[0]:
            reasons.append(f"spin {i}: look-back region starts before the capture's first sample")
        if m.es[-1] < on:
            reasons.append(f"spin {i}: the parsed run ends before the spin"
                           + (f" ({m.end_reason})" if m.end_reason else ""))
        ks = m.inside(lo, on)
        xs = [m.excess_mj(k, p_ref) for k in ks]
        mx = max(xs) if xs else None
        out["spins"].append({"on_ns": on, "off_ns": off, "checked": len(ks), "max_excess_mj": mx})
        if not ks:
            reasons.append(f"spin {i}: no sample lies in the look-back region for every offset")
            continue
        worst = mx if worst is None else max(worst, mx)
        if mx > tol:
            reasons.append(f"spin {i}: excess {mx:.1f} mJ > tol {tol:.1f} mJ in a sample that ends before the "
                           "spin under every offset (energy and elapsed_ns cover different intervals)")
    out["max_excess_mj"] = worst
    out["valid"] = not reasons
    return out


@_total(lambda reason: math.nan)
def spin_excess_power_w(samples, spawn_ns, spins, ref, delta_a2_ns=0) -> float:
    """M16.1: the median over spins of (sum E - p_ref sum el) / sum el over the samples wholly inside the
    spin for every admissible offset, in W; spins holding no such sample are skipped. delta_A2 = tol / this
    (mJ / W = ms). NaN when undefined (no spin with a sample, invalid reference, mapping failure)."""
    p_ref, _, why = _ref_values(ref)
    sp, why2 = _spins(spins)
    m, why3 = _mapping(samples, spawn_ns, delta_a2_ns)
    if why or why2 or why3 or m.u < 0:
        return math.nan
    vals = []
    for on, off in sp:
        ks = m.inside(on, off)
        if ks:
            el = sum(m.el[k] for k in ks)
            vals.append((sum(m.e[k] for k in ks) - p_ref * el * 1e-6) * 1e6 / el)
    return statistics.median(vals) if vals else math.nan


# --- per-block checks: A1 band, A2 pre-spawn trial, A3 rate -----------------------------------------------

def _band_template():
    return {"valid": False, "reasons": [], "width_ns": None, "records": 0}


@_total(_invalid(_band_template))
def band_check(samples) -> dict:
    """M15.1 (A1, per block): b_k = timestamp_k - sum_{j<=k} el_j over every record of the block; the band
    width max b - min b must be <= 1 s + tau_band. Every record must parse and carry its timestamp. Limit
    (M16.4): with exactly 100 ms samples a single skip only widens 0.9 s to 1.0 s and passes.
    -> {"valid", "reasons", "width_ns", "records"}."""
    out = _band_template()
    xs = _as_list(samples)
    if not xs:
        out["reasons"].append("band: no records")
        return out
    c, lo, hi = 0, None, None
    for i, x in enumerate(xs):
        f, ts = _fields(x), _timestamp_ns(x)
        if f is None or ts is None:
            out["reasons"].append(f"band: record {i} is malformed or has no timestamp (the band needs every record)")
            return out
        c += f[0]
        b = ts - c
        lo = b if lo is None else min(lo, b)
        hi = b if hi is None else max(hi, b)
    out.update(width_ns=hi - lo, records=len(xs))
    if hi - lo > BAND_LIMIT_NS:
        out["reasons"].append(f"band: width {(hi - lo) / S:.3f} s > 1 s + tau_band ({BAND_LIMIT_NS / S:.3f} s): "
                              "a producer skip or discontinuity (A1)")
    out["valid"] = not out["reasons"]
    return out


def _prespawn_template():
    return {"valid": False, "reasons": [], "first_excess_mj": None, "first_bound_ns": None}


@_total(_invalid(_prespawn_template))
def prespawn_trial(samples, spawn_ns, ref) -> dict:
    """M16.1 (A2) trial: after a >= 500 ms one-thread spin ending at the spawn call, the capture's first
    record must be idle-level (excess <= tol), and causal (el_1 <= d_1 - s). -> {"valid", "reasons",
    "first_excess_mj", "first_bound_ns" (d_1 - s - el_1)}."""
    out = _prespawn_template()
    reasons = out["reasons"]
    p_ref, tol, why = _ref_values(ref)
    if why:
        reasons.append(why)
    if not _is_int(spawn_ns):
        reasons.append("prespawn: spawn time is not an integer ns")
    xs = _as_list(samples)
    f = _fields(xs[0]) if xs else None
    if f is None:
        reasons.append("prespawn: the capture's first record is missing or malformed")
    if reasons:
        return out
    el, e, d = f
    out["first_bound_ns"] = d - spawn_ns - el
    out["first_excess_mj"] = e - p_ref * el * 1e-6
    if out["first_bound_ns"] < 0:
        reasons.append(f"prespawn: causality, the first record arrives {-out['first_bound_ns'] / MS:.1f} ms before "
                       "spawn + its elapsed")
    if out["first_excess_mj"] > tol:
        reasons.append(f"prespawn: the first record carries excess {out['first_excess_mj']:.1f} mJ > tol {tol:.1f} mJ "
                       "(its interval starts before the spawn: A2)")
    out["valid"] = not reasons
    return out


def _rate_template():
    return {"valid": False, "reasons": [], "max_excess_ns": None, "jitter_ns": None, "records": 0}


def _jitter(c: list[int], v: list[int]) -> int:
    """Largest range of v over any stretch {k: c_i <= c_k <= c_i + JITTER_STRETCH_NS}."""
    best, j, hi, lo = 0, 0, deque(), deque()
    for i in range(len(c)):
        while j < len(c) and c[j] - c[i] <= JITTER_STRETCH_NS:
            while hi and v[hi[-1]] <= v[j]:
                hi.pop()
            hi.append(j)
            while lo and v[lo[-1]] >= v[j]:
                lo.pop()
            lo.append(j)
            j += 1
        while hi[0] < i:
            hi.popleft()
        while lo[0] < i:
            lo.popleft()
        best = max(best, v[hi[0]] - v[lo[0]])
    return best


@_total(_invalid(_rate_template))
def rate_check(samples) -> dict:
    """M16.2 (A3, per block and validate's 10-minute capture): with E_k = (d_k - d_1) - (c_k - c_1) for every
    prefix k, require |E_k| - J <= rho (c_k - c_1), J the delivery jitter (largest range of d - c within any
    10 s stretch). Every record must parse. max_excess_ns = max_k (|E_k| - J - rho (c_k - c_1)) (<= 0 when
    valid). jitter_ns is J, which the runner passes to window_bounds as delivery_jitter_ns.
    -> {"valid", "reasons", "max_excess_ns", "jitter_ns", "records"}."""
    out = _rate_template()
    xs = _as_list(samples)
    if not xs:
        out["reasons"].append("rate: no records")
        return out
    fs = [_fields(x) for x in xs]
    bad = [i for i, f in enumerate(fs) if f is None]
    if bad:
        out["reasons"].append(f"rate: record {bad[0]} is malformed (the cumulative elapsed is broken)")
        return out
    out["records"] = len(fs)
    if len(fs) < 2:
        out["reasons"].append("rate: fewer than 2 records")
        return out
    c, acc = [], 0
    for el, _, _ in fs:
        acc += el
        c.append(acc)
    d = [f[2] for f in fs]
    jit = _jitter(c, [dk - ck for dk, ck in zip(d, c)])
    out["jitter_ns"] = jit
    worst = max(abs((dk - d[0]) - (ck - c[0])) - jit - RHO * (ck - c[0]) for dk, ck in zip(d, c))
    out["max_excess_ns"] = worst
    if worst > 0:
        out["reasons"].append(f"rate: accumulated delivery-vs-elapsed error exceeds J + rho c by {worst / MS:.2f} ms "
                              f"(A3, rho = {RHO * 1e6:.0f} ppm)")
    out["valid"] = not out["reasons"]
    return out
