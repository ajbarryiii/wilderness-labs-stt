"""B's x2 deferred provider for H's deferred-exactness qualification at x2 (plans/x2-default.md Z5).

DEFERRED_PROVIDERS[name] = {"kernel": registry name, "factory": callable(table, names) -> provider,
"components": the part count the harness must pass}. The provider is hybrid_x2.HybridProjectionX2, the
one B-hybrid-x2 runs inside PartCount(·, 2). The harness calls it directly with components 2 (it sets
defer_epilogue on the provider itself, so no adapter is involved) and finishes a returned
DeferredProjection through inference.row_fusion.finish_residual looked up at call time; constructing
the provider installs row_exact_x2's router there. Importing this module loads no gpu1bit.actprec
module; the registry name int8_mma:x2 needs gpu1bit.actprec.parts.load_parts() first (the factory and
hybrid_x2.parts() call it).
"""
from __future__ import annotations

from gpu1bit.int8_mma.hybrid_x2 import KERNEL, PARTS, HybridProjectionX2

DEFERRED_PROVIDERS = {
    "B-hybrid-x2": {"kernel": KERNEL, "factory": HybridProjectionX2, "components": PARTS},
}
