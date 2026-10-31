"""Deployed-mode scoring of a Kea-B scoring view with the real i8x1 kernels (keab-prime plan "Scoring and gate").

The loader (load_x1): the gpu1bit runtime's public `parts` accepts only 2 or 3, so x1 uses the research path that
produced the recorded x1 arm (gpu1bit/actprec/parts.py `build("MB1", 1)`, run ap-20261020r4, arm MB1-int8_mma-x1):
  1. load the pinned part-count variants (actprec.parts.load_parts: B's SHA-256-pinned kernels_parts, layer_parts,
     parts_run, which register int8_mma:x1);
  2. gpu1bit.runtime.load_binary_packed(export_dir, runtime="provider", kernel="int8_mma:x1", _research_parts=1)
     on the given export directory (gpu1bit's hard-coded MB1 path is neither used nor changed). It wraps the
     provider in fulldev.PartCount(provider, 1) and pins NeMo's fused greedy TDT decoder to full CUDA graphs.
One builder serves MB1, step 0's views and both stage 4 arms.

Per utterance (score_records): gpu1bit.shared.fulldev.process_one, the recorded arm's own procedure: evaluate.py's
audio loading and short-clip padding, a [1, T] FP32 batch through model.forward under evaluate.strict_fp32 and
inference_settings, the fused greedy decoder, evaluate.score; the row keeps the token IDs, timestamps and encoder
lengths. Audio is prefetched one utterance ahead (fulldev.prefetched). Set WER: evaluate.wer_summary (corpus WER over
scored records), as fulldev.summarize_set.

Kernel cache: int8_mma compiles its kernels with NVRTC into gpu1bit's cubin cache. relabel_cubin_cache (installed by
load_x1) reads that cache on a hit and compiles a miss into $RELABEL_ROOT/keab-prime/caches/cubin instead, so nothing
is written outside $RELABEL_ROOT (the cubins are identical either way: the key covers source, kernel name,
architecture, options and NVRTC version).

Interpreter: the recorded x1 runs used gpu1bit/shared/python (finetune/parakeet-ternary/python with
finetune/parakeet-binary on PYTHONPATH and NVRTC 12.9 on the library path, which int8_mma compiles with and NeMo's
full CUDA-graph decoder needs); finetune/relabel/python-gpu1bit runs relabel code in that environment.
require_wrapper refuses any other (without NVRTC NeMo silently falls back to another decoder mode).
"""
from __future__ import annotations

import contextlib
import hashlib
import time
from pathlib import Path

from training import eightbit as _eb  # noqa: F401  puts finetune/parakeet-binary on sys.path

KERNEL = "int8_mma:x1"
PARTS = 1
RUNTIME = "provider"
WRAPPER = "gpu1bit/shared/python"


def require_wrapper() -> dict:
    """gpu1bit's environment facts (env.wrapper()); RuntimeError unless this process runs under WRAPPER with NVRTC."""
    from gpu1bit.shared import env
    facts = env.wrapper()
    if facts["wrapper"] != WRAPPER or not facts["nvrtc_on_library_path"]:
        raise RuntimeError(f"deployed-mode scoring needs {WRAPPER} with NVRTC on the library path "
                           f"(finetune/relabel/python-gpu1bit); got {facts}")
    return facts


def cubin_key(source: str, name: str, arch: str, options, version: str) -> str:
    """gpu1bit.int8_mma.cuda.compile_cubin's cache key."""
    return hashlib.sha256("\0".join([source, name, arch, *options, version]).encode()).hexdigest()


@contextlib.contextmanager
def relabel_cubin_cache(root: Path):
    """int8_mma's compile_cubin with misses compiled into root/caches/cubin (module docstring)."""
    from gpu1bit.int8_mma import cuda
    original = cuda.compile_cubin

    def compile_cubin(source: str, name: str, arch: str = "120a", options=cuda.OPTIONS) -> bytes:
        key = cubin_key(source, name, arch, options, cuda.setup())
        for base in (cuda.ARTIFACTS, Path(root)):
            path = base / "caches" / "cubin" / f"{key}.cubin"
            if path.is_file():
                return path.read_bytes()
        shared = cuda.ARTIFACTS
        cuda.ARTIFACTS = Path(root)
        try:
            return original(source, name, arch, options)
        finally:
            cuda.ARTIFACTS = shared

    cuda.compile_cubin = compile_cubin
    try:
        yield
    finally:
        cuda.compile_cubin = original


def load_x1(export_dir: Path, device: str = "cuda", probe=None):
    """The scoring view at export_dir on the x1 runtime (module docstring); `probe(provider)` may wrap the provider
    transparently (instrumentation, as actprec.parts.build). The caller holds the GPU lock and calls close()."""
    import torch
    from gpu1bit import runtime
    from gpu1bit.actprec import parts as AP
    from gpu1bit.shared import fulldev
    require_wrapper()
    from training import keab_export
    stack = contextlib.ExitStack()
    stack.enter_context(relabel_cubin_cache(keab_export.keab_root()))
    loaded = AP.load_parts()
    made = []

    def hook(provider):
        made.append(provider)
        return probe(provider) if probe else provider

    try:
        with torch.inference_mode():
            model = runtime.load_binary_packed(Path(export_dir), runtime=RUNTIME, kernel=KERNEL, device=device,
                                               _provider_hook=hook, _research_parts=PARTS)
    except BaseException:
        stack.close()
        raise
    model._keab_stack = stack                      # the cubin seam stays until close(): graphs compile lazily
    rt = model._gpu1bit_runtime
    if len(made) != 1 or rt.parts != PARTS or rt.kernel != KERNEL or not isinstance(rt.provider, fulldev.PartCount) \
            or rt.provider.n != PARTS:
        raise RuntimeError(f"x1 build differs from the recorded arm: parts {rt.parts}, kernel {rt.kernel}, "
                           f"provider {type(rt.provider).__name__}, providers built {len(made)}")
    torch.cuda.synchronize()
    comp = model.decoding.decoding.decoding_computer
    model._keab_x1 = {"export": str(Path(export_dir).resolve()), "export_sha256": rt.export_sha256,
                      "kernel": KERNEL, "parts": PARTS, "runtime": RUNTIME, "wip_loaded": loaded,
                      "sources": rt.sources, "selection_sha256": rt.selection_sha256,
                      "decoder_cuda_graphs_mode": str(getattr(comp, "cuda_graphs_mode", None))}
    return model


def info(model) -> dict:
    return dict(model._keab_x1)


def close(model) -> None:
    from gpu1bit import runtime
    try:
        runtime.close_binary_packed(model)
    finally:
        stack = getattr(model, "_keab_stack", None)
        if stack is not None:
            stack.close()


def score_records(model, records: list[tuple[int, dict]], label: str, identity: str) -> dict:
    """Rows (fulldev.process_one) of [(index, manifest record)] in order, and the set's corpus WER."""
    import evaluate
    from gpu1bit.shared import fulldev
    start = time.perf_counter()
    rows = [fulldev.process_one(model, i, rec, identity, audio=audio)
            for i, rec, audio in fulldev.prefetched(records)]
    wall = time.perf_counter() - start
    wer = evaluate.wer_summary(rows)
    structural = {"not_finite": [r["id"] for r in rows if r.get("finite") is False],
                  "decode_errors": [r["id"] for r in rows if r.get("decode_error")]}
    return {"set": label, "utterances": len(rows), "wer": wer, "wer_percent": 100 * wer["wer"],
            "audio_hours": sum(r["duration"] for r in rows) / 3600, "wall_s": wall,
            "empty_hypotheses": sum(r["empty"] for r in rows), "padded_short": sum(r["padded_short"] for r in rows),
            "structural": structural, "structurally_valid": not structural["not_finite"] and not structural["decode_errors"],
            "records": rows}
