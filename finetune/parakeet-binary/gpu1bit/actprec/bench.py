"""Patch module: H's frozen timing harness for the activation-precision arms (plans/activation-precision.md
"Measurements" 2: P-paired at encoder and full scope on dev_24, P-clips and P-memory in fresh processes).

    gpu1bit/shared/python -m gpu1bit.shared.dispatch --patches gpu1bit.actprec.bench bench paired \
        --agent H --run <run> --scope full --variants M1-x3 M1-x2 M1-x1 MB1-int8_mma-x3 MB1-int8_mma-x2 MB1-int8_mma-x1 --utterances 0-3

Variant names are the full-dev arm names. The x3 arms are the deployed stacks exactly; the xN arms
are PartCount over the same stacks. Each is built by gpu1bit.actprec.parts.build.

Each variant is bound to its recognition arm. Its identity is recomputed and must equal the one
frozen for the recognition run (fulldev.check_identity refuses otherwise), and its x3 reference in
that run must pass the R2 gate (parts.require_reference; review c2). The run is explicit: it is read
from GPU1BIT_ACTPREC_RUN, which rerun_timing.py sets from the roster's identity_run, and there is no
default (review c3). The identity, the run and the reference's arm and identity go into the variant's
info, which P-paired and P-clips/P-memory both record. This file is part of those identities
(gpu1bit/actprec/). The hash goes into the
variant's info, so every timing record carries the identity of the recognition rows (R4).

Graph cache: the pipeline graph cache is the deployed 4 entries in both scopes. The encoder scope
calls model.encoder directly, so the unused cache does not take part in it. Any name that is not
an activation-precision arm goes to the previously installed variant_model.
"""
from __future__ import annotations

import contextlib
import os

from gpu1bit.actprec import parts
from gpu1bit.shared import bench, fulldev



def recognition_run() -> str:
    """The recognition run the timed variants are bound to: GPU1BIT_ACTPREC_RUN, which the timing
    orchestrator sets from the roster's identity_run (review c3). No default."""
    run = os.environ.get("GPU1BIT_ACTPREC_RUN")
    if not run:
        raise SystemExit("GPU1BIT_ACTPREC_RUN is not set: the recognition run of an activation-precision timing "
                         "variant must be explicit (rerun_timing.py sets it from the roster's identity_run)")
    return run

_previous = bench.variant_model


def is_arm(name: str) -> bool:
    try:
        spec = fulldev.parse_arm(name)
    except ValueError:
        return False
    return spec["kind"] != "fp32" and (spec["model"] == "M1" or spec["kernel"] in parts.SUPPORTED_KERNELS)


@contextlib.contextmanager
def variant_model(name: str, *, max_graphs: int = 4):
    if not is_arm(name):
        with _previous(name, max_graphs=max_graphs) as pair:
            yield pair
        return
    import evaluate as ev
    run = recognition_run()
    frozen = fulldev.check_identity(run, name, "full")
    reference = parts.require_reference(run, name)   # R2 gate (review c2): x3 itself for x3 arms
    spec = fulldev.parse_arm(name)
    model, info = parts.build(spec["model"], spec["parts"], kernel=spec["kernel"] or "int8_mma")
    try:
        with ev.inference_settings(model):
            yield model, {"kind": "actprec", "arm": name, "identity": frozen["hash"], "identity_run": run,
                          "reference": {"arm": reference["arm"], "identity": reference["identity"]},
                          "parts": spec["parts"], "max_graphs": 4, "requested_max_graphs": max_graphs,
                          **{k: v for k, v in info.items() if k != "partcount"}, "projection": info["partcount"]}
    finally:
        parts.close(model)


parts.install()
bench.variant_model = variant_model
