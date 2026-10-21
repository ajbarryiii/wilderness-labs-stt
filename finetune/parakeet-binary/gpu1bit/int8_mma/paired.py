"""H's harness (gpu1bit/shared/bench.py) with one more variant, "B-hybrid" (gpu1bit/int8_mma/hybrid.py).

bench.variant_model is extended by a monkeypatch that lives here (plans/gpu-kernel.md: patches live
in gpu1bit): "B-hybrid" builds MB1 on Int8MmaLayer under R-hybrid's own context stack; "R-hybrid" is
H's unchanged baseline, always with its fixed max_graphs=4 (m1.baseline refuses anything else, so the
encoder scope keeps the pipeline graph cache on for both hybrids; the encoder graph is captured
directly either way). Every other name goes to H's variant_model unchanged. All protocols, checks,
locks and outputs are H's:

    timeout 600 gpu1bit/shared/python -m gpu1bit.int8_mma.paired paired --agent B --run <name> \
        --scope encoder --variants R-hybrid B-hybrid --utterances 0-5
    timeout 600 gpu1bit/shared/python -m gpu1bit.int8_mma.paired gate timing --agent B --impl B-hybrid
"""
from __future__ import annotations

import contextlib

from gpu1bit.shared import bench

_original = bench.variant_model


@contextlib.contextmanager
def variant_model(name: str, *, max_graphs: int = 4):
    if name == "B-hybrid":
        from gpu1bit.int8_mma.hybrid import b_hybrid_model
        with b_hybrid_model(max_graphs=4) as pair:
            yield pair
    elif name == "R-hybrid":
        with _original(name, max_graphs=4) as pair:
            yield pair
    else:
        with _original(name, max_graphs=max_graphs) as pair:
            yield pair


bench.variant_model = variant_model

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "gate":  # H's A12 gate, unchanged, with the same patch
        sys.argv.pop(1)
        from gpu1bit.shared import gate
        gate.main()
    else:
        bench.main()
