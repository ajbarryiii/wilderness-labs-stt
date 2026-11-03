"""Deployed-mode scoring of a Kea-T scoring view (parakeet-ternary-v1) on M1's own x1 stack (keat-prime plan "Scoring
and gate"). The ternary counterpart of training/deployed.py, whose load_x1 is binary-only; its type-agnostic parts
are reused unchanged: score_records (the per-utterance procedure and the set's corpus WER) and require_wrapper.

What exists (gpu1bit, unchanged): runtime_m1.build_stack(export_dir, parts, wrap) builds M1's deployed stack, the
research builder of the recorded arm M1-x1 of run ap-20261020r4 (actprec/parts.build("M1", 1)):
  - inference.runtime.load_packed(export_dir), which verifies the export's files against its manifest;
  - NeMo's greedy TDT decoder pinned to full CUDA graphs;
  - enable_optimizations(max_graphs=4, encoder_storage="expanded"), with the stack's one
    ExpandedProjection('triton_col_tuned') wrapped in fulldev.PartCount(provider, parts) when parts != 3. PartCount
    forces the component count on every call, including layer_fusions4's hard-coded 3 for pointwise_conv2.
  The kernels are _quantize and _int8_dot at the Triton constant COMPONENTS = 1; their cache is $RELABEL_ROOT/caches/
  triton (relabel_boot.py). No kernel code is added here.
What is missing, and is added here: the public runtime_m1.load_ternary_packed refuses parts 1 (seldoc.check_parts),
actprec/parts.py hard-codes M1's export, and gpu1bit.runtime.load_binary_packed refuses ternary exports. So:

  load_x1(export_dir)   build_stack(export_dir, 1, wrap=hook) on any export directory; refused unless exactly one
                        provider was built and the stack's provider is a PartCount with n = 1 around it. The
                        PartCount records every call's (requested, effective) part count (part_calls).
  load_x3(export_dir)   the same builder at 3 parts, with no adapter (for fidelity: x1 against x3).
  Both set the featurizer's dither to 0 and pad_to to 0, as runtime_m1.load_ternary_packed does, and close()
  restores them (also if releasing the stack fails). The caller holds the parakeet GPU lock and calls close().

Interpreter: the recorded x1 runs used gpu1bit/shared/python (NVRTC on the library path, which NeMo's full
CUDA-graph decoder needs); finetune/relabel/python-gpu1bit runs relabel code in it. require_wrapper (stage 4's)
refuses any other interpreter before any model work.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

from training import deployed as _deployed

PARTS = 1
X3_PARTS = 3
STACK = ("gpu1bit.runtime_m1.build_stack(export_dir, parts): inference.runtime.load_packed; full CUDA-graph decoder; "
         "enable_optimizations(max_graphs=4, encoder_storage='expanded') with ExpandedProjection('triton_col_tuned') "
         "wrapped in fulldev.PartCount(provider, parts) when parts != 3")
require_wrapper = _deployed.require_wrapper
score_records = _deployed.score_records


class PartCalls:
    """fulldev.PartCount's `record`: counts its (requested, effective) part counts instead of keeping every call."""

    def __init__(self):
        self.counts: collections.Counter = collections.Counter()

    def append(self, pair) -> None:
        requested, effective = pair
        self.counts[(int(requested), int(effective))] += 1

    def summary(self) -> dict[str, int]:
        return {f"{r}->{e}": n for (r, e), n in sorted(self.counts.items())}


def _load(export_dir: Path, parts: int, probe=None):
    import torch
    from gpu1bit import runtime_m1
    from gpu1bit.shared import fulldev
    if parts not in (PARTS, X3_PARTS):
        raise ValueError(f"parts must be {PARTS} or {X3_PARTS}")
    wrapper = require_wrapper()
    export_dir = Path(export_dir).resolve()
    made = []

    def hook(provider):
        made.append(provider)
        return probe(provider) if probe else provider

    with torch.inference_mode():
        model, (inner, prov) = runtime_m1.build_stack(export_dir, parts, wrap=hook)
    adapter = isinstance(prov, fulldev.PartCount)
    if len(made) != 1 or made[0] is not inner:
        raise RuntimeError(f"{len(made)} projection providers were built; expected exactly the stack's one")
    if parts == PARTS and not (adapter and prov.n == PARTS):
        raise RuntimeError(f"x1 build differs from the recorded arm: provider {type(prov).__name__}, "
                           f"n {getattr(prov, 'n', None)}")
    if parts == X3_PARTS and adapter:
        raise RuntimeError("x3 is the deployed stack without the part-count adapter")
    calls = None
    if adapter:
        calls = PartCalls()
        prov.record = calls
    featurizer = model.preprocessor.featurizer
    saved = {k: getattr(featurizer, k) for k in ("dither", "pad_to")}
    featurizer.dither, featurizer.pad_to = 0.0, 0
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    manifest = json.loads((export_dir / "manifest.json").read_text())
    comp = model.decoding.decoding.decoding_computer
    object.__setattr__(model, "_keat_x", {
        "export": str(export_dir), "export_sha256": manifest.get("sha256"), "parts": parts,
        "provider": type(prov).__name__, "inner_provider": type(inner).__name__, "stack": STACK,
        "adapter": "fulldev.PartCount(provider, 1): components = 1 on every projection call" if adapter else None,
        "wrapper": wrapper, "featurizer_saved": dict(saved),
        "decoder_cuda_graphs_mode": str(getattr(comp, "cuda_graphs_mode", None))})
    object.__setattr__(model, "_keat_state", {"provider": prov, "calls": calls, "saved": saved})
    return model


def load_x1(export_dir: Path, probe=None):
    """The export at export_dir on M1's stack at one activation part (module docstring); `probe(provider)` may wrap
    the stack's provider transparently (instrumentation, as actprec.parts.build)."""
    return _load(export_dir, PARTS, probe)


def load_x3(export_dir: Path, probe=None):
    """The export at export_dir on M1's deployed stack (three parts, no adapter)."""
    return _load(export_dir, X3_PARTS, probe)


def _state(model) -> dict:
    state = getattr(model, "_keat_state", None)
    if state is None:
        raise ValueError("not a model loaded by training/keat_deployed.py")
    return state


def provider(model):
    """The stack's projection provider: the PartCount at x1, the (probed) ExpandedProjection at x3."""
    return _state(model)["provider"]


def part_calls(model) -> dict[str, int] | None:
    """{"<requested>-><effective>": calls} recorded by the x1 PartCount (None at x3)."""
    calls = _state(model)["calls"]
    return None if calls is None else calls.summary()


def info(model) -> dict:
    return {**model._keat_x, "part_calls": part_calls(model)}


def close(model) -> None:
    """Release the stack (inference.optimized.disable_optimizations) and restore dither and pad_to."""
    import torch
    from inference.optimized import disable_optimizations
    state = _state(model)
    try:
        with torch.inference_mode():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            disable_optimizations(model)
    finally:
        featurizer = model.preprocessor.featurizer
        for k, v in state["saved"].items():
            setattr(featurizer, k, v)
