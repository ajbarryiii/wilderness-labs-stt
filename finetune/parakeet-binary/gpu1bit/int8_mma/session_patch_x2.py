"""Patch module: bench.variant_model("B-hybrid-x2") builds B-hybrid-x2 (plans/x2-default.md).

The stack is hybrid_x2.exact_model_x2(): MB1 on int8_mma:x2 layers, PartCount(HybridProjectionX2, 2)
(H's gpu1bit.shared.fulldev.PartCount, which must forward attribute writes, X1) inside
hybrid_x2.exact_hybrid_pipeline_x2. Every other name goes to the previously installed variant_model.
Importing this module loads no gpu1bit.actprec module; building the variant calls
gpu1bit.actprec.parts.load_parts() first (F5). Import it after gpu1bit.int8_mma.session_patch if both
are used (it does not import that module).
"""
from __future__ import annotations

import contextlib

from gpu1bit.shared import bench

NAME = "B-hybrid-x2"
_previous = bench.variant_model


@contextlib.contextmanager
def variant_model(name: str, *, max_graphs: int = 4):
    if name == NAME:
        from gpu1bit.int8_mma.hybrid_x2 import exact_model_x2
        with exact_model_x2(max_graphs=4) as pair:
            yield pair
    else:
        with _previous(name, max_graphs=max_graphs) as pair:
            yield pair


bench.variant_model = variant_model
