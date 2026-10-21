"""Exact deferred split-K (XCHG 5) at 2 activation parts for B-hybrid-x2 (plans/x2-default.md).

The x3 kernel is kernels_exact.dot_source (frozen b1_dot + XCHG 5: each K slice writes its three INT32
part sums packed into one int64, as int32 low / high word planes [C, 2, M, N]). The x2 kernel is that
source passed through kernels_parts.transform(source, 2), B's hash-pinned part-count transform
(B/wip/parts, loaded by gpu1bit.actprec.parts.load_parts): the A tile, the cp.async loop and the
ldmatrix / IMMA loop cover parts 0 and 1 only. Accumulator slot 2 is never written and stays 0, so
the packed word's lane 2 is exactly 0 and lanes 0 / 1 are the two INT32 part sums. The XCHG 5 branch
reads no scale, so the part-count guards of the epilogues are not involved. Both edit sets are
anchored (asserted) and touch disjoint text: kernels_exact edits the rank line, adds the XCHG 5
branch and the entry's template arguments; kernels_parts edits the A tile, the loops, the QS reads
and the epilogue calls, and leaves COMMON's prefix unchanged except for its own functions.
Tile configurations are the x3 tables (R6 / F4): hybrid_exact.TUNED_EXACT_DEFERRED.

parts = 3 returns kernels_exact.dot_source unchanged (the x3 kernel); only 2 and 3 are accepted.
"""
from __future__ import annotations

from gpu1bit.int8_mma import kernels_exact
from gpu1bit.int8_mma.kernels import DotConfig

PARTS = (2, 3)


def _kernels_parts():
    from gpu1bit.int8_mma import hybrid_x2
    return hybrid_x2.parts()["kernels_parts"]


def _replace(source: str, old: str, new: str, count: int = 1) -> str:
    found = source.count(old)
    if found != count:
        raise AssertionError(f"x2 exact transform expected {count} x {old[:70]!r}, found {found}")
    return source.replace(old, new)


def dot_source(config: DotConfig, order: int = 0, parts: int = 2) -> tuple[str, str]:
    """(source, kernel name) of the XCHG 5 kernel for `config`'s tiles at `parts` activation parts."""
    if parts not in PARTS:
        raise ValueError(f"parts must be one of {PARTS}, got {parts!r}")
    source, name = kernels_exact.dot_source(config, order, False)
    if parts == 3:
        return source, name
    new_name = f"x{parts}_{name}"
    source = _replace(source, f"\n{name}(", f"\n{new_name}(")
    source = _kernels_parts().transform(source, parts)
    return source, new_name


def shared_bytes(config: DotConfig, parts: int = 2) -> int:
    """Dynamic shared memory of the `parts`-part XCHG 5 kernel (the P-part A tile; frozen layout else)."""
    if parts not in PARTS:
        raise ValueError(f"parts must be one of {PARTS}, got {parts!r}")
    if parts == 3:
        return kernels_exact.shared_bytes(config)
    cfg = kernels_exact.base(kernels_exact.ExactDeferredConfig.of(config), xchg=4)
    return _kernels_parts().PartsConfig.of(cfg, parts, False).shared_bytes
