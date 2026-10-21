"""Patch module for the re-run selection session (import AFTER gpu1bit.int8_mma.b01_run).

bench.variant_model("B-hybrid") and ("B-hybrid-b01") build the FIXED stacks: hybrid_exact.exact_model
(R-hybrid's context stack, HybridProjectionExact, exact integer deferred split-K finished by
row_exact._row_epilogue_exact) over load_binary_model(MB1, Int8MmaLayer) and (MB1, Int8MmaB01Layer).
"B-hybrid@frozen" and "B-hybrid-b01@frozen" keep the frozen stacks (FP32 deferred slices) reachable
for comparisons. Every other name goes to the previously installed variant_model (b01_run's, then
paired.py's, then H's). Importing this module also registers int8_mma:b01 (b01_run) and installs
row_exact's finish_residual routing.
"""
from __future__ import annotations

import contextlib

from gpu1bit.int8_mma import b01_run  # noqa: F401  (registers int8_mma:b01 and "B-hybrid-b01"@frozen)
from gpu1bit.int8_mma import row_exact  # noqa: F401  (installs the finish_residual routing)
from gpu1bit.shared import bench

_previous = bench.variant_model
FROZEN = {"B-hybrid@frozen": "B-hybrid", "B-hybrid-b01@frozen": "B-hybrid-b01"}


@contextlib.contextmanager
def variant_model(name: str, *, max_graphs: int = 4):
    from gpu1bit.int8_mma.hybrid_exact import exact_model
    if name in ("B-hybrid", "B-hybrid-b01"):
        if name == "B-hybrid":
            from gpu1bit.int8_mma.layer import Int8MmaLayer as layer_cls
        else:
            from gpu1bit.int8_mma.layer_b01 import Int8MmaB01Layer as layer_cls
        with exact_model(layer_cls, max_graphs=4) as pair:
            yield pair
    elif name in FROZEN:
        with _previous(FROZEN[name], max_graphs=max_graphs) as pair:
            yield pair
    else:
        with _previous(name, max_graphs=max_graphs) as pair:
            yield pair


bench.variant_model = variant_model
