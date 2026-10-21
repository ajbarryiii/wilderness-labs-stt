"""Patch module: H's timing harness (shared/bench.py) for the x2 benchmark roster b1 to b6
(plans/x2-default.md X5), imported through gpu1bit.shared.dispatch --patches gpu1bit.x2.bench_patch.

    MB1-B-hybrid-x2 / -x3   gpu1bit.runtime.load_binary_packed(MB1, runtime="hybrid", parts=n) under the
                            selection in use (GPU1BIT_SELECTION = shared/selected-x2.json), via
                            gpu1bit.actprec.parts.build (the deployed loader)
    R-expanded              the existing H builder (MB1, H's derived 2-bit layout)
    M1-x2 / M1-x3           gpu1bit.runtime_m1.build_stack via gpu1bit.actprec.parts.build
    M1-R-hybrid-x3          M1's export on experimental_pipeline(model, "ptx_row_best") (shared/m1.baseline)

Every variant's info records its roster id, model, complete export identity (fulldev.export_identity),
the selection content hash and, for fulldev arms, the arm identity, so P-paired and P-clips/P-memory
artifacts carry the labels the report verifies (bench_roster.verify_labels). Other names are delegated.
"""
from __future__ import annotations

import contextlib

from gpu1bit.shared import bench, env, fulldev
from gpu1bit.x2 import bench_roster as BR

VARIANTS = tuple(b["variant"] for b in BR.BENCH_ROSTER)
_previous = bench.variant_model


def _labels(variant: str) -> dict:
    from gpu1bit.shared import seldoc
    bid = BR.VARIANT_TO_ID[variant]
    return {"roster_id": bid, "model": BR.ROW[bid]["model"], "stack_label": BR.ROW[bid]["stack"],
            "parts": BR.ROW[bid]["parts"], "export": fulldev.export_identity(BR.export_dir(bid)),
            "selection_sha256": seldoc.load()["sha256"]}


def _require_candidate() -> None:
    """Y1: the roster runs under the candidate document's content (GPU1BIT_SELECTION=shared/selected-x2.json,
    or the pointer once it is a byte-identical promoted copy); anything else is refused before model work."""
    from gpu1bit.shared import seldoc
    if seldoc.load()["sha256"] != env.sha256_file(seldoc.CANDIDATE):
        raise ValueError(f"the x2 benchmark roster runs under {seldoc.CANDIDATE.name} (set GPU1BIT_SELECTION); "
                         f"the selection in use is {seldoc.current_path()}")


@contextlib.contextmanager
def variant_model(name: str, *, max_graphs: int = 4):
    if name not in VARIANTS:
        with _previous(name, max_graphs=max_graphs) as pair:
            yield pair
        return
    _require_candidate()
    import evaluate as ev
    from gpu1bit.actprec import parts
    labels = _labels(name)
    if name == "R-expanded":
        with _previous(name, max_graphs=max_graphs) as (model, info):
            yield model, {**info, **labels}
        return
    if name == "M1-R-hybrid-x3":
        from inference.runtime import load_packed
        from gpu1bit.shared import m1
        model = load_packed(env.M1_EXPORT)
        with ev.inference_settings(model):
            model.decoding.decoding.decoding_computer.force_cuda_graphs_mode("full_graph")
            with m1.baseline("R-hybrid", model) as extra:
                yield model, {"kind": "baseline", "stack": "R-hybrid on M1's export", "projection": extra, **labels}
        return
    parts.install()
    spec = fulldev.parse_arm(name)
    identity = fulldev.identity(name, "full")["hash"]
    model, info = parts.build(spec["model"], spec["parts"], kernel=spec["kernel"] or "int8_mma")
    try:
        with ev.inference_settings(model):
            yield model, {"kind": "x2-roster", "arm": name, "identity": identity, **labels,
                          **{k: v for k, v in info.items() if k != "partcount"}, "projection": info["partcount"]}
    finally:
        parts.close(model)


bench.variant_model = variant_model
