"""MB1 end to end from its 1-bit export on the RTX 5090: the selected gpu1bit runtime (phase 3).

    timeout 600 finetune/parakeet-binary/gpu1bit/shared/python -m gpu1bit.runtime \\
        --export /mnt/hd/wilderness-labs-stt/parakeet-binary/runs/main-binB-m1-lr2e-4/export clip.wav

    from gpu1bit.runtime import load_binary_packed, transcribe, close_binary_packed
    model = load_binary_packed(export_dir)                 # runtime, kernel, parts: the selection document
    print(transcribe(model, "clip.wav"))                   # str; a list of items gives a list
    close_binary_packed(model)

What it does (nothing here is copied from M1 or from B; everything is imported):
1. `validate_export`: the directory must be a "parakeet-binary-v1" export (binary_export.FORMAT, also
   in the safetensors metadata); every file's SHA-256 must match the manifest; exactly the 264
   Parakeet quantized layers; every layer's codes / scale / bias pass api.validate_weights (uint8
   [N, ceil(K/8)], no nonzero tail bits, finite FP32 scales). A 2-bit "parakeet-ternary-v1" export
   (M1's, or H's derived MB1 layout) is refused. No .nemo is read: the NeMo model is built from the
   manifest's config and the export's own tokenizer files (gpu1bit.shared.provider.load_binary_model).
2. The 264 layers are prepared by the selected kernel (registry name, gpu1bit/shared/registry.py;
   default shared/selected.json "kernel", now "int8_mma" = B's Int8MmaLayer). Q/K/V of each block
   are one fused [3072, 1024] layer; no other GPU copy of the weights exists (A5).
3. The selected stack (default shared/selected.json "runtime"):
   - "hybrid" (the selection winner, B-hybrid): R-hybrid's context stack, i.e. M1's
     experimental_pipeline(model, "ptx_row_best") contexts in the same order, served by B's
     gpu1bit.int8_mma.hybrid.HybridProjection (deferred row epilogues, prequantized activations),
     entered via gpu1bit.int8_mma.hybrid.hybrid_pipeline; 4-entry pipeline graph cache. Needs a
     kernel class with B's `dot` op. The provider class, the pipeline context and modules to import
     come from shared/selected.json's "hybrid" entry (default: B's frozen hybrid.py), so B's exact
     deferred stack (hybrid_exact) can replace it by a data change, like the kernel.
   - "provider": M1's enable_optimizations(encoder_storage="expanded") stack (R-expanded's) with
     gpu1bit.shared.provider.BinaryProjection: any registered BinaryLayer (its `project` op if any).
   Both use NeMo's fused decoder kernels and greedy TDT decoding unchanged (M1's decoder_transform /
   control_transform, full CUDA-graph decoder mode).
4. `transcribe` is M1's hf.load_ternary.transcribe (strict FP32, dither 0, pad_to 0, short clips
   padded to its minimum) at batch 1, under torch.inference_mode(), the regime every gpu1bit
   measurement used (M1's --experimental-cuda is likewise batch-1 only).

5. Activation parts (plans/x2-default.md X3, Y1): `parts` is 2 or 3 (CLI --parts), default the selection
   document's "default_parts" (absent = 3) for a canonical call (no kernel, no hybrid spec); a call that
   names a kernel gets that kernel's own part count and one naming only a hybrid spec gets 3, so explicit
   x3 callers stay x3 after promotion (resolve_load); any other value is refused. The selection document is
   `selection=` (CLI --selection), else GPU1BIT_SELECTION, else the pointer shared/selected.json (the x3
   document until promotion; shared/seldoc.py). A document with a "parts" map names each part count's
   kernel and hybrid stack (shared/selected-x2.json: B's int8_mma:x2 layers and HybridProjectionX2 at 2);
   at parts 2 the provider is wrapped in fulldev.PartCount(provider, 2). The Runtime record carries
   parts, the selection document's content hash and, for an unmodified hybrid build of MB1's export, the
   fulldev identity of arm MB1-B-hybrid-x<parts> under that document (equal to the gate and benchmark
   arm's identity).

Swapping the kernel (e.g. B's post-selection "int8_mma:b01"): it must first win under the same rules
(dev selection session, A12, A5, A14); then change "kernel" in shared/selected.json. This file is
not edited. `load_binary_packed(..., kernel=NAME, runtime=...)` overrides both for experiments.

GPU discipline: the CLI takes paths.gpu_lock for its whole life and exits inside it (A9). Python
callers on this machine must hold the lock themselves. Inference only; one model per thread.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

from gpu1bit.shared import api, env  # env puts parakeet-binary and parakeet-ternary on sys.path

SELECTED = env.SHARED / "selected.json"   # the deployment pointer (shared/seldoc.py)
RUNTIMES = ("hybrid", "provider")


def selected(selection=None) -> dict:
    """The selection document in use (shared/seldoc.py: `selection`, else GPU1BIT_SELECTION, else the
    pointer shared/selected.json)."""
    from gpu1bit.shared import seldoc
    return seldoc.load(selection)["doc"]


def resolve_load(choice: dict, *, runtime: str | None = None, kernel: str | None = None, hybrid: dict | None = None,
                 parts: int | None = None, research_parts: int | None = None) -> dict:
    """What load_binary_packed builds from a selection document (pure; Codex x2 c1, finding 7):
    {parts, runtime, kernel, canonical, entry, hybrid}. Only a CANONICAL call (no kernel, no hybrid spec)
    takes the document's default_parts; a call naming a kernel gets that kernel's own part count
    (seldoc.part_count: int8_mma 3, int8_mma:x2 2), and a call naming only a hybrid spec gets 3, so explicit
    x3 callers stay x3 after promotion. A kernel at another part count than its own is refused."""
    from gpu1bit.shared import seldoc
    canonical = kernel is None and hybrid is None
    if research_parts is not None:
        if parts is not None or kernel is None or runtime != "provider" or research_parts not in (1, 2, 3):
            raise ValueError("_research_parts (1, 2 or 3) needs runtime='provider', an explicit kernel and no `parts`")
        parts = research_parts
    elif parts is None:
        parts = seldoc.default_parts(choice) if canonical else (seldoc.part_count(kernel) if kernel else 3)
    else:
        parts = seldoc.check_parts(parts)
    if kernel is not None and seldoc.part_count(kernel) != parts:
        raise ValueError(f"kernel {kernel!r} serves {seldoc.part_count(kernel)} activation parts, not the requested "
                         f"part count {parts}")
    runtime = runtime or choice["runtime"]
    if runtime not in RUNTIMES:
        raise ValueError(f"runtime must be one of {RUNTIMES}, got {runtime!r}")
    ent = seldoc.entry(choice, parts) if (runtime == "hybrid" or kernel is None) else {"kernel": kernel, "imports": [], "hybrid": {}}
    return {"parts": parts, "runtime": runtime, "kernel": kernel or ent["kernel"], "canonical": canonical, "entry": ent,
            "hybrid": {**DEFAULT_HYBRID, **(ent.get("hybrid") or {}), **(hybrid or {})}}


def resolve_parts(parts, selection=None) -> int:
    """The activation part count: `parts` if given (2 or 3 only), else the document's default_parts
    (absent = 3) (plans/x2-default.md X3, Y1)."""
    from gpu1bit.shared import seldoc
    if parts is not None:
        return seldoc.check_parts(parts)
    return seldoc.default_parts(seldoc.load(selection)["doc"])


DEFAULT_HYBRID = {"provider": "gpu1bit.int8_mma.hybrid:HybridProjection",
                  "pipeline": "gpu1bit.int8_mma.hybrid:hybrid_pipeline", "imports": []}


def _resolve(spec: str):
    import importlib
    module, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module), attr)


def hybrid_stack(choice: dict):
    """(provider class, pipeline context) of the hybrid runtime from selected.json's "hybrid" entry
    (default: B's frozen hybrid.py). "imports" are modules imported first (e.g. a registry patch for
    a variant kernel, or B's row_exact finish_residual routing for the exact deferred path)."""
    import importlib
    spec = {**DEFAULT_HYBRID, **choice.get("hybrid", {})}
    for module in spec["imports"]:
        importlib.import_module(module)
    return _resolve(spec["provider"]), _resolve(spec["pipeline"])


def import_selected_modules(choice: dict) -> None:
    """Modules selected.json asks for before the registry is consulted (e.g. b01_run for int8_mma:b01)."""
    import importlib
    for module in choice.get("imports", []):
        importlib.import_module(module)


def validate_export(directory) -> dict:
    """Refuse anything but an intact parakeet-binary-v1 export of the 264 layers; return its manifest."""
    import binary_export
    from safetensors import safe_open
    from quant import EXPECTED_MODULES, QUANTIZED_PATTERN
    directory = Path(directory).resolve()
    manifest_path = directory / binary_export.MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"{directory}: no {binary_export.MANIFEST}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != binary_export.FORMAT:
        raise ValueError(f"{directory}: export format {manifest.get('format')!r}, expected {binary_export.FORMAT!r} "
                         "(2-bit parakeet-ternary-v1 exports, including derived ones, are refused)")
    if manifest.get("file") != binary_export.FILE or binary_export.FILE not in manifest.get("files", {}):
        raise ValueError(f"{directory}: manifest does not list {binary_export.FILE}")
    for rel, digest in manifest["files"].items():
        path = directory / rel
        if not path.is_file() or env.sha256_file(path) != digest:
            raise ValueError(f"{path} does not match the manifest SHA-256")
    layers = manifest["quantized_layers"]
    if len(layers) != EXPECTED_MODULES or any(not QUANTIZED_PATTERN.fullmatch(n) for n in layers):
        raise ValueError(f"{directory}: expected the {EXPECTED_MODULES} Parakeet quantized layers, got {len(layers)}")
    with safe_open(directory / manifest["file"], framework="pt") as f:
        if (f.metadata() or {}).get("format") != binary_export.FORMAT:
            raise ValueError(f"{directory}: safetensors metadata format is not {binary_export.FORMAT!r}")
        keys = set(f.keys())
        for name, info in layers.items():
            if info.get("kind") not in ("linear", "pointwise_conv1d"):
                raise ValueError(f"{name}: unknown kind {info.get('kind')!r}")
            n, k = info["shape"]
            if f"{name}.weight" in keys:
                raise ValueError(f"{name}: dense weight present in a 1-bit export")
            codes, scale = f.get_tensor(f"{name}.codes"), f.get_tensor(f"{name}.scale")
            bias = f.get_tensor(f"{name}.bias") if info["bias"] else None
            try:
                got = api.validate_weights(codes, scale, bias, k)
            except ValueError as exc:
                raise ValueError(f"{name}: {exc}") from None
            if got != (n, k):
                raise ValueError(f"{name}: codes give {got}, manifest says {(n, k)}")
    return manifest


@dataclass
class Runtime:
    runtime: str
    kernel: str
    export: str
    export_sha256: str
    provider: object
    table: object
    stack: contextlib.ExitStack
    featurizer_saved: dict
    sources: dict = field(default_factory=dict)
    parts: int = 3
    identity: str | None = None          # the fulldev arm identity of an unmodified selection build (X3)
    selection_sha256: str | None = None


def load_binary_packed(export_dir, *, runtime: str | None = None, kernel: str | None = None,
                       device: str | torch.device = "cuda", max_graphs: int = 4, hybrid: dict | None = None,
                       parts: int | None = None, selection=None, _provider_hook=None, _research_parts: int | None = None):
    """MB1 (or any parakeet-binary-v1 export of this architecture) as an inference-only NeMo model on
    the selected 1-bit runtime. See the module docstring. Call close_binary_packed before moving,
    editing or saving the model.

    parts (plans/x2-default.md X3, Y1): the activation part count, 2 or 3. Default (resolve_load): the
    selection document's default_parts (absent = 3) for a canonical call; with an explicit kernel, that
    kernel's own part count; with only a hybrid spec, 3. Any other value is refused. At parts 2 the provider is
    wrapped in fulldev.PartCount(provider, 2). selection: the selection document (path; default
    GPU1BIT_SELECTION or shared/selected.json). The Runtime record carries parts and, for an unmodified
    hybrid build of the canonical MB1 export, the fulldev identity of arm MB1-B-hybrid-x<parts>.
    _provider_hook(provider) -> provider wraps the provider before PartCount (instrumentation).
    _research_parts: activation-precision research arms only (the x1 arm): a part count outside the
    deployment's {2, 3}, given instead of `parts`, with an explicit kernel on the provider runtime."""
    from gpu1bit.shared import fulldev, provider as P, registry, seldoc
    sel = seldoc.load(selection)
    choice = sel["doc"]
    r = resolve_load(choice, runtime=runtime, kernel=kernel, hybrid=hybrid, parts=parts, research_parts=_research_parts)
    parts, runtime, kernel, canonical, ent, hybrid_spec = (r["parts"], r["runtime"], r["kernel"], r["canonical"],
                                                           r["entry"], r["hybrid"])
    if runtime == "hybrid" and max_graphs != 4:
        raise ValueError("the hybrid stack (R-hybrid's experimental_pipeline) fixes a 4-entry graph cache")
    manifest = validate_export(export_dir)
    import_selected_modules(choice)
    import_selected_modules({"imports": ent.get("imports", [])})
    if ":x" in kernel:   # a part-count layer variant: B's hash-pinned files first (plans/x2-default.md F5)
        from gpu1bit.actprec import parts as AP
        AP.load_parts()
    impl = registry.get(kernel)
    if runtime == "hybrid" and not callable(getattr(impl, "dot", None)):
        raise ValueError(f"kernel {kernel!r} has no dot(q, qs, ...) op; the hybrid stack needs one (use runtime='provider')")
    device = torch.device(device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
        raise ValueError("this runtime targets NVIDIA SM120 (RTX 5090)")
    stack = contextlib.ExitStack()
    with torch.inference_mode():
        model, table = P.load_binary_model(Path(export_dir).resolve(), impl, device)
        featurizer = model.preprocessor.featurizer
        saved = {k: getattr(featurizer, k) for k in ("dither", "pad_to")}
        featurizer.dither, featurizer.pad_to = 0.0, 0
        model.eval()
        model.decoding.decoding.decoding_computer.force_cuda_graphs_mode("full_graph")
        try:
            def wrap(inner):
                hooked = _provider_hook(inner) if _provider_hook else inner
                return fulldev.PartCount(hooked, parts) if parts != 3 else hooked
            if runtime == "hybrid":
                provider_cls, pipeline = hybrid_stack({"hybrid": hybrid_spec})
                prov = wrap(provider_cls(table, P.module_names(model)))
                stack.enter_context(pipeline(model, prov))
            else:
                from inference.optimized import disable_optimizations
                prov = wrap(P.BinaryProjection(table, P.module_names(model)))
                P.enable_binary_optimizations(model, prov, max_graphs=max_graphs)
                stack.callback(disable_optimizations, model)
        except BaseException:
            stack.close()
            raise
    sources = {"runtime.py": env.sha256_file(Path(__file__)), "kernel": env.sha256_file(registry.source_path(kernel)),
               "selection": {"path": sel["path"], "sha256": sel["sha256"]}}
    if runtime == "hybrid":
        spec = hybrid_spec
        sources["hybrid_stack"] = spec
        for part in ("provider", "pipeline"):
            module = spec[part].partition(":")[0]
            path = env.REPO / "finetune" / "parakeet-binary" / (module.replace(".", "/") + ".py")
            sources[f"hybrid_{part}_sha256"] = env.sha256_file(path)
    identity = None
    if runtime == "hybrid" and canonical and _provider_hook is None and Path(export_dir).resolve() == env.MB1_EXPORT.resolve():
        from gpu1bit.actprec import parts as AP
        identity = AP.identity(f"MB1-B-hybrid-x{parts}", "full", selection=sel["path"])["hash"]
    model._gpu1bit_runtime = Runtime(runtime, kernel, str(Path(export_dir).resolve()), manifest["sha256"], prov, table,
                                     stack, saved, sources, parts=parts, identity=identity, selection_sha256=sel["sha256"])
    return model


def runtime_of(model) -> Runtime:
    rt = getattr(model, "_gpu1bit_runtime", None)
    if rt is None:
        raise ValueError("not a gpu1bit.runtime model (use load_binary_packed)")
    return rt


def close_binary_packed(model) -> None:
    """Release graphs and restore M1's transformed modules and the frontend settings."""
    rt = getattr(model, "_gpu1bit_runtime", None)
    if rt is None:
        return
    with torch.inference_mode():
        torch.cuda.synchronize()
        rt.stack.close()
        for k, v in rt.featurizer_saved.items():
            setattr(model.preprocessor.featurizer, k, v)
    del model._gpu1bit_runtime


def transcribe(model, paths_or_audio):
    """Greedy TDT transcripts (punctuated, cased): M1's hf.load_ternary.transcribe at batch 1 under
    inference_mode. One path / 1-D 16 kHz array gives a str; a list gives a list in input order."""
    from hf.load_ternary import transcribe as m1_transcribe
    runtime_of(model)
    with torch.inference_mode():
        return m1_transcribe(model, paths_or_audio, batch_size=1)


def decoded(model) -> dict:
    """Token IDs and frame timestamps of the most recent batch-1 decode (the fused decoder's state)."""
    bh = model.decoding.decoding.decoding_computer.state.batched_hyps
    n = int(bh.current_lengths[0])
    return {"tokens": bh.transcript[0, :n].tolist(), "timestamps": bh.timestamps[0, :n].tolist()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("audio", nargs="+", type=Path)
    p.add_argument("--export", required=True, type=Path, help="parakeet-binary-v1 export directory (MB1)")
    p.add_argument("--runtime", choices=RUNTIMES, default=None, help="default: shared/selected.json")
    p.add_argument("--kernel", default=None, help="registry name; default: shared/selected.json")
    p.add_argument("--parts", type=int, choices=(2, 3), default=None,
                   help="activation parts; default: the selection document's default_parts (absent = 3)")
    p.add_argument("--selection", type=Path, default=None,
                   help="selection document; default: GPU1BIT_SELECTION or shared/selected.json")
    p.add_argument("--label", default="runtime:transcribe", help="GPU lock label")
    p.add_argument("--tokens", action="store_true",
                   help="JSON lines: also token IDs, timestamps, parts, runtime identity and selection hash")
    args = p.parse_args()
    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    code = 0
    with env.gpu_lock(args.label):  # before the first CUDA call; the process exits inside it (A9)
        try:
            model = load_binary_packed(args.export, runtime=args.runtime, kernel=args.kernel, parts=args.parts,
                                       selection=args.selection)
            for path in args.audio:
                text = transcribe(model, path)
                if args.tokens:
                    rt = runtime_of(model)
                    print(json.dumps({"audio": str(path), "text": text, **decoded(model), "parts": rt.parts,
                                      "identity": rt.identity, "selection_sha256": rt.selection_sha256}))
                else:
                    print(f"{path}\t{text}" if len(args.audio) > 1 else text)
            close_binary_packed(model)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 - reported, nonzero exit
            print(f"error: {exc}", file=sys.stderr)
            code = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


if __name__ == "__main__":
    main()
