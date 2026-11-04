"""LocalFlow's encoder geometry on top of the upstream MIL builder (as LocalFlow's scripts/prepare-parakeet.py).

apply_buckets(buckets) switches mil.encoder to the given LocalFlow functions (mel frames seconds x 100 + 1).
The default, BUCKETS = (15,), is the one function LocalFlow runs (LocalParakeetCore.encoderSeconds, since
LocalFlow 897f13e), with upstream's 15 s folded relative-position tables (188 encoder frames). ALL_BUCKETS
(4/8/15/30) extends the tables to the 30 s window (376 frames). Every function slices the same centre
positions, so b15 holds the same values either way. Idempotent for one bucket set per process.
Also sets sys.path for the upstream ios/ modules.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IOS = HERE.parent / "parakeet-ternary" / "ios"
for _p in (str(IOS), str(IOS.parent), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BUCKETS = (15,)  # what LocalFlow loads; the other functions' position tables cost ~52 MB in a 1-bit bundle
ALL_BUCKETS = (4, 8, 15, 30)
FUNCTIONS = [f"b{s}" for s in ALL_BUCKETS]  # screen.py's research functions
UPSTREAM_BUCKETS = {2: 201, 4: 401, 8: 801, 15: 1501}


def apply_buckets(buckets: tuple[int, ...] = BUCKETS) -> None:
    from mil import encoder

    if 15 not in buckets or any(s not in ALL_BUCKETS for s in buckets):
        raise ValueError(f"LocalFlow functions must include b15 and come from {ALL_BUCKETS}; got {buckets}")
    want = {s: s * 100 + 1 for s in sorted(buckets)}
    if encoder.BUCKETS == want:
        return
    if encoder.BUCKETS != UPSTREAM_BUCKETS or encoder.MAX_T != 188:
        raise ValueError("Unexpected upstream encoder buckets; review the LocalFlow override")
    encoder.BUCKETS = want
    encoder.MAX_T = encoder.encoder_frames(want[max(buckets)])
    fold_positions = encoder.fold_positions
    encoder.fold_positions = lambda w_pos, length=encoder.MAX_T: fold_positions(w_pos, length)
