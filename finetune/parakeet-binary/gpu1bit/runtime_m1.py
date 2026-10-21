"""M1 (ternary Parakeet) on its R-expanded stack at a selectable activation part count (plans/x2-default.md
X3, Y1). M1's own files under parakeet-ternary/ stay frozen; its `enable_optimizations` keeps x3 as M1's
historical entry point. This loader is the M1 deployment entry point:

    from gpu1bit.runtime_m1 import load_ternary_packed, transcribe, close_ternary_packed
    model = load_ternary_packed()              # M1's export; parts from the selection document in use
    print(transcribe(model, "clip.wav"))
    close_ternary_packed(model)

- parts: 2 or 3; default the selection document's `m1_default_parts` (absent = 3). Before promotion the
  pointer shared/selected.json is the x3 document, so unqualified callers get x3. Any other value is
  refused before any model work.
- selection: the selection document (path; default GPU1BIT_SELECTION or shared/selected.json).
- The stack is exactly the activation-precision M1 arm:
  - inference.runtime.load_packed;
  - the full CUDA-graph decoder mode;
  - enable_optimizations(max_graphs=4, encoder_storage="expanded"), with the stack's own
    ExpandedProjection('triton_col_tuned') wrapped in fulldev.PartCount(provider, parts) when parts != 3.
  gpu1bit/actprec/parts.py builds its M1 arms with this same `build_stack`.
- The record (`model._gpu1bit_m1_runtime`) carries parts, the selection hash and, for M1's canonical export,
  the fulldev identity of arm M1-x<parts> under that selection (X3: equal to the frozen b4/b5 arm identity).
- Dither 0 and pad_to 0 for the deployment frontend (as gpu1bit.runtime), restored by close.

GPU discipline: callers hold paths.gpu_lock (A9). Inference only; one model per thread.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

from gpu1bit.shared import env


@dataclass
class M1Runtime:
    export: str
    parts: int
    provider: object
    featurizer_saved: dict
    identity: str | None
    selection_sha256: str


def resolve_parts(parts, selection=None) -> int:
    from gpu1bit.shared import seldoc
    if parts is not None:
        return seldoc.check_parts(parts)
    return seldoc.m1_default_parts(seldoc.load(selection)["doc"])


def build_stack(export_dir, parts: int, wrap=None):
    """(model, provider) of M1's R-expanded stack at `parts`. wrap(provider) -> provider wraps the
    stack's ExpandedProjection before PartCount (instrumentation; actprec R1 contracts)."""
    from gpu1bit.shared import fulldev
    from inference.runtime import load_packed
    from inference.optimized import enable_optimizations
    import inference.encoder4 as e4
    model = load_packed(Path(export_dir))
    model.decoding.decoding.decoding_computer.force_cuda_graphs_mode("full_graph")
    original = e4.ExpandedProjection
    made = []

    def factory(backend="triton", config=None):
        if backend != "triton_col_tuned" or config is not None:
            raise AssertionError(f"unexpected ExpandedProjection({backend!r}, {config!r})")
        inner = original(backend, config)
        hooked = wrap(inner) if wrap else inner
        prov = fulldev.PartCount(hooked, parts) if parts != 3 else hooked
        made.append((inner, prov))
        return prov
    e4.ExpandedProjection = factory
    try:
        enable_optimizations(model, max_graphs=4, encoder_storage="expanded")
    finally:
        e4.ExpandedProjection = original
    if len(made) != 1:
        raise AssertionError(f"{len(made)} projection providers were built; expected exactly 1")
    return model, made[0]


def load_ternary_packed(export_dir=None, *, parts: int | None = None, selection=None):
    import torch
    from gpu1bit.shared import seldoc
    sel = seldoc.load(selection)
    parts = seldoc.m1_default_parts(sel["doc"]) if parts is None else seldoc.check_parts(parts)
    export = Path(export_dir or env.M1_EXPORT).resolve()
    with torch.inference_mode():
        model, (inner, prov) = build_stack(export, parts)
        model.eval()
        featurizer = model.preprocessor.featurizer
        saved = {k: getattr(featurizer, k) for k in ("dither", "pad_to")}
        featurizer.dither, featurizer.pad_to = 0.0, 0
        torch.cuda.synchronize()
    identity = None
    if export == Path(env.M1_EXPORT).resolve():
        from gpu1bit.actprec import parts as AP
        identity = AP.identity(f"M1-x{parts}", "full", selection=sel["path"])["hash"]
    model._gpu1bit_m1_runtime = M1Runtime(str(export), parts, prov, saved, identity, sel["sha256"])
    return model


def runtime_of(model) -> M1Runtime:
    rt = getattr(model, "_gpu1bit_m1_runtime", None)
    if rt is None:
        raise ValueError("not a gpu1bit.runtime_m1 model (use load_ternary_packed)")
    return rt


def close_ternary_packed(model) -> None:
    import torch
    from inference.optimized import disable_optimizations
    rt = getattr(model, "_gpu1bit_m1_runtime", None)
    if rt is None:
        return
    with torch.inference_mode():
        torch.cuda.synchronize()
        disable_optimizations(model)
        for k, v in rt.featurizer_saved.items():
            setattr(model.preprocessor.featurizer, k, v)
    del model._gpu1bit_m1_runtime


def transcribe(model, paths_or_audio):
    """Greedy TDT transcripts at batch 1 under inference_mode (M1's hf.load_ternary.transcribe)."""
    import torch
    from hf.load_ternary import transcribe as m1_transcribe
    runtime_of(model)
    with torch.inference_mode():
        return m1_transcribe(model, paths_or_audio, batch_size=1)


@contextlib.contextmanager
def loaded(export_dir=None, **kw):
    model = load_ternary_packed(export_dir, **kw)
    try:
        yield model
    finally:
        close_ternary_packed(model)
