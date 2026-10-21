"""LocalFlow's encoder geometry on top of the upstream MIL builder (as LocalFlow's scripts/prepare-parakeet.py).

apply_buckets() switches mil.encoder to LocalFlow's 4/8/15/30-second functions (b4, b8, b15, b30; mel frames
seconds x 100 + 1) and extends the folded relative-position tables to the 30 s window (376 encoder frames).
Every function slices the same centre positions, so shorter functions keep their values. Idempotent.
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

BUCKETS = (4, 8, 15, 30)
FUNCTIONS = [f"b{s}" for s in BUCKETS]
UPSTREAM_BUCKETS = {2: 201, 4: 401, 8: 801, 15: 1501}


def apply_buckets() -> None:
    from mil import encoder

    want = {s: s * 100 + 1 for s in BUCKETS}
    if encoder.BUCKETS == want:
        return
    if encoder.BUCKETS != UPSTREAM_BUCKETS or encoder.MAX_T != 188:
        raise ValueError("Unexpected upstream encoder buckets; review the LocalFlow override")
    encoder.BUCKETS = want
    encoder.MAX_T = encoder.encoder_frames(want[max(BUCKETS)])
    fold_positions = encoder.fold_positions
    encoder.fold_positions = lambda w_pos, length=encoder.MAX_T: fold_positions(w_pos, length)
