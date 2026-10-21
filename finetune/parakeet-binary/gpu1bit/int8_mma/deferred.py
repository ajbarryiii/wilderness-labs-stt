"""B's fixed deferred providers for H's deferred A3 (gpu1bit/shared/deferred_a3.py; interface in NOTES).

DEFERRED_PROVIDERS[name] = {"kernel": registry name, "factory": callable(table, names) -> provider}.
The provider is hybrid_exact.HybridProjectionExact, the one the fixed B-hybrid / B-hybrid-b01 stacks
use (session_patch.py): R-hybrid's provider protocol (defer_epilogue, prequantized, no_defer_weights,
row_warps, null); deferred calls return a SPLIT = 1 DeferredProjection or an ExactDeferredProjection
(integer slice planes). Importing this module installs row_exact's routing of inference.row_fusion
.finish_residual, so M1's finish_residual looked up at call time finishes both exactly.

    timeout 600 gpu1bit/shared/python -m gpu1bit.shared.dispatch \
        --patches gpu1bit.int8_mma.b01_run,gpu1bit.int8_mma.session_patch \
        deferred qualify --provider B-hybrid --module gpu1bit.int8_mma.deferred
"""
from __future__ import annotations

from gpu1bit.int8_mma import b01_run  # noqa: F401  (registers int8_mma:b01)
from gpu1bit.int8_mma import row_exact  # noqa: F401  (installs the finish_residual routing)
from gpu1bit.int8_mma.hybrid_exact import HybridProjectionExact

DEFERRED_PROVIDERS = {
    "B-hybrid": {"kernel": "int8_mma", "factory": HybridProjectionExact},
    "B-hybrid-b01": {"kernel": "int8_mma:b01", "factory": HybridProjectionExact},
}
