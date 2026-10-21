"""Full-dev transcription driver (plans/activation-precision.md "Measurements" 1 and R1-R7; agent H).

Runs one runtime ARM over every utterance of the five dev manifests (paths.DEV_SETS) at batch 1,
through the arm's full runtime as deployed (CPU waveform -> frontend -> encoder -> greedy TDT
decoder -> text, with its 4-entry pipeline CUDA-graph cache; shapes beyond the cache run eagerly,
the deployed overflow path), and scores with frozen evaluate.py. Dev manifests only.

ARMS (`parse_arm`)
  M1-x3              M1's export, R-expanded: inference.runtime.load_packed +
                     inference.optimized.enable_optimizations(max_graphs=4, encoder_storage="expanded").
  MB1-<kernel>-x3    MB1's 1-bit export through gpu1bit.runtime.load_binary_packed(runtime="provider",
                     kernel=<registry name>): R-expanded's transforms with BinaryProjection.
  M1-fp32, MB1-fp32  The FP32 NeMo model rebuilt from the export (export.load_export /
                     binary_export.load_export), batch 1, NeMo's own forward and decoding: R2 step 1 and
                     the fidelity reference (token agreement of runtime-x3 on the 512-utterance sample).
  M1-x2/-x1, MB1-<kernel>-x2/-x1
                     HELD until B's components-1/2 kernels exist and the plan review is CLEAN. They will
                     use `PartCount(provider, n)` (R1) with the x3 tile tables unchanged (R6).
SUBSETS: "full" (every utterance) or "sample" (workloads.json dev_sample, 512 ids); tests use
"first:<N>". The subset is part of the arm identity.

PER UTTERANCE (`process_one`; equivalent to evaluate.transcribe_records at batch_size=1):
evaluate.load_audio (16 kHz mono asserted) + evaluate.pad_short (`load_clip`; the worker prefetches
the next utterance's audio in one background thread, `prefetched`), a [1, T] float32 batch,
model.forward under evaluate.strict_fp32 + evaluate.inference_settings + torch.inference_mode,
model.decoding.rnnt_decoder_predictions_tensor. Recorded per utterance: text, token IDs (the
hypothesis' y_sequence), frame timestamps when available, empty flag, encoder output length and
tensor frames, finiteness of the encoder output, decoding error (R5), evaluate.score fields, and
the arm identity hash. Manifest order.

STRUCTURAL GATES (R5, mandatory): an arm is INVALID (not a trade-off) if on any utterance the encoder
output is not finite, decoding raised, or (xN arms) the encoder length differs from the same model's
x3 arm. Invalidity is reported by `finish` / `compare`; rows are still written so counts are exact.

SCORING (`summarize_set`, `dev_mean`): evaluate.score, evaluate.wer_summary (corpus WER over scored
records), the unweighted mean over paths.DEV_SETS (train.mean_wer). A set summary requires every
utterance of the subset exactly once. Bitwise parity with the recorded evaluations is a test.

CROSS-CHECK (R2): a full-subset x3 arm passes only if its dev mean is within 0.02 points AND every set
within 0.10 points of the model's pinned FP32-NeMo scores (runs/<main run>/summary.json). On failure,
frozen procedure: (1) the model's -fp32 arm at batch 1 over full dev, (2) if that also misses, a
historical-batch reproduction, (3) report; no xN arm runs until the cause is known.

IDENTITY (R4): `identity(arm, subset)` lists the complete export (every file of the export directory,
including manifest.json with the model configuration and the tokenizer; review c1), the SHA-256 of the
CONTENT of the selection document in use (shared/seldoc.py; plans/x2-default.md Y1), the stack and part count, the kernel
and its configuration tables, the SHA-256 of every source file under the experiment directories
(gpu1bit/ including every new file, M1's inference/, the parakeet-ternary and parakeet-binary
modules, hf/load_ternary.py, whisper-ternary/wer.py and quant.py, the wrappers), the scorer dependencies
(evaluate.py, wer.py, normalizer.json, transformers' english_normalizer.py) and the dev manifests'
SHA-256. `freeze` writes it once per run and arm; every worker recomputes and refuses a mismatch,
also refuses if any repo module it imported is outside the hashed set; every row and chunk carries
the hash; `finish` / `compare` refuse mixed or mismatched identities.

PROCESSES (A9, R3): `run` is one GPU process that takes ONE gpu lock before CUDA initialises, works at
most --work-budget seconds (default 480 = 8 min; it predicts the next utterance from the slowest seen
so far and stops early), flushes every row, fsyncs periodically and at the end, and exits inside the
lock. `drive` (no CUDA) freezes the identity, then spawns `timeout 600 <shared/python> -m
gpu1bit.shared.fulldev run ...` until the subset is complete, then calls `finish`.

    gpu1bit/shared/python -m gpu1bit.shared.fulldev drive --run xcheck --arm M1-x3
    gpu1bit/shared/python -m gpu1bit.shared.fulldev compare --run xcheck --arm MB1-int8_mma-x3 --ref MB1-fp32 --subset sample
Output: /mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/H/fulldev/<run>/<arm>[@<subset>]/
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from gpu1bit.shared import env

DEV_SETS = ("librispeech_dev_clean", "librispeech_dev_other", "ami_dev", "voxpopuli_dev", "yodas_dev")
ROOT = env.ARTIFACTS / "fulldev"
PY = str(env.SHARED / "python")
PINNED = {"M1": Path("/mnt/hd/wilderness-labs-stt/parakeet-ternary/runs/main-M1-P2-lr5e-4/summary.json"),
          "MB1": Path("/mnt/hd/wilderness-labs-stt/parakeet-binary/runs/main-binB-m1-lr2e-4/summary.json")}
MEAN_POINTS, SET_POINTS = 0.02, 0.10
WORK_BUDGET_S, HOLD_CAP_S = 480.0, 540.0
_ARM = re.compile(r"^(M1|MB1)(?:-(.+?))?-(x[123]|fp32)$")
STACKS = {
    "M1-x": {"loader": "inference.runtime.load_packed(M1 export)",
             "transform": "inference.optimized.enable_optimizations(model, max_graphs=4, encoder_storage='expanded')",
             "encoder_transform": "mode int8x3, qkv, silu, position, fold_bn, norm, attention, norm_quant, position_dot, "
                                  "projection=ExpandedProjection('triton_col_tuned'); layer_fusions; decoder_transform; "
                                  "control_transform; pipeline graphs (4)", "decoder_graphs": "full_graph"},
    "MB1-x": {"loader": "gpu1bit.runtime.load_binary_packed(MB1 export, runtime='provider', kernel=K)",
              "transform": "gpu1bit.shared.provider.enable_binary_optimizations = enable_optimizations(expanded) "
                           "with BinaryProjection", "decoder_graphs": "full_graph"},
    "fp32": {"loader": "export.load_export / binary_export.load_export (FP32 NeMo rebuilt from the export)",
             "transform": "none (NeMo forward and greedy TDT decoding, batch 1)", "decoder_graphs": "NeMo default"},
}


# --- manifests, subsets, arms --------------------------------------------------------------------

def manifest(name: str) -> tuple[list[dict], dict]:
    """(records in manifest order, metadata) of one dev set; anything else is refused."""
    import paths
    from gpu1bit.shared import workloads
    if name not in DEV_SETS or name not in paths.DEV_SETS:
        raise ValueError(f"{name!r} is not one of the dev sets {DEV_SETS}; test and other manifests are refused")
    records, meta = workloads._manifest(name)  # asserts dev split, manifest path and dev.done.json hash
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name}: duplicate ids in the manifest")
    return records, meta


def subset_records(name: str, subset: str) -> list[tuple[int, dict]]:
    """[(manifest index, record)] of the subset, in manifest order."""
    records = manifest(name)[0]
    if subset == "full":
        return list(enumerate(records))
    if subset == "sample":
        from gpu1bit.shared import workloads
        want = {r["id"] for r in workloads.load()["dev_sample"] if r["corpus"] == name}
        out = [(i, r) for i, r in enumerate(records) if r["id"] in want]
        if len(out) != len(want):
            raise ValueError(f"{name}: dev_sample ids missing from the manifest")
        return out
    m = re.fullmatch(r"first:(\d+)", subset)
    if m:
        return list(enumerate(records))[:int(m.group(1))]
    raise ValueError(f"unknown subset {subset!r}")


def parse_sets(names: list[str] | None) -> list[str]:
    names = list(names or DEV_SETS)
    for n in names:
        if n not in DEV_SETS:
            raise ValueError(f"{n!r} is not a dev set; test manifests are refused")
    return names


def parse_arm(arm: str) -> dict:
    m = _ARM.fullmatch(arm)
    if not m:
        raise ValueError(f"arm {arm!r}: expected M1-x<N>, MB1-<kernel>-x<N> (N in 1..3), M1-fp32 or MB1-fp32")
    model, kernel, tail = m.group(1), m.group(2), m.group(3)
    if tail == "fp32":
        if kernel is not None:
            raise ValueError(f"arm {arm!r}: fp32 arms take no kernel")
        return {"model": model, "kernel": None, "parts": None, "kind": "fp32"}
    if (model == "M1") != (kernel is None):
        raise ValueError(f"arm {arm!r}: M1 arms take no kernel, MB1 arms need one")
    return {"model": model, "kernel": kernel, "parts": int(tail[1]), "kind": f"{model}-x"}


def check_arm_enabled(spec: dict) -> None:
    if spec["parts"] not in (None, 3):
        raise RuntimeError("part-count arms (x2, x1) are held until B's components-1/2 kernels exist and the "
                           "plan review of plans/activation-precision.md is CLEAN (coordinator 2026-10-20)")


def reference_arm(arm: str) -> str | None:
    """The same model's x3 arm, against which an xN arm's lengths and tokens are compared."""
    spec = parse_arm(arm)
    if spec["parts"] in (None, 3):
        return None
    return "M1-x3" if spec["model"] == "M1" else f"MB1-{spec['kernel']}-x3"


class PartCount:
    """R1: wraps a projection provider (ExpandedProjection for M1, BinaryProjection for MB1) and sets
    `components` to n on EVERY call, including layer_fusions4.py's hard-coded components=3 for the
    second pointwise convolution; every other argument, the stack and its fusions are unchanged.
    Records the (requested, effective) part counts of each call when `record` is a list.
    Attribute WRITES are forwarded to the wrapped provider for every name except the adapter's own
    (provider, n, record), so stack contexts that set provider state (row_fusion's defer_epilogue,
    row_fusions' no_defer_weights) reach the provider that uses it (plans/x2-default.md X1)."""

    OWN = ("provider", "n", "record")

    def __init__(self, provider, n: int):
        if n not in (1, 2, 3):
            raise ValueError("n must be 1, 2 or 3")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "record", None)

    def __setattr__(self, name, value):
        if name in PartCount.OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self.provider, name, value)

    def __call__(self, x, mod, conv=False, components=3, activation="", norm=None, config=None, input_activation=""):
        if self.record is not None:
            self.record.append((components, self.n))
        return self.provider(x, mod, conv, self.n, activation, norm, config, input_activation)

    def __getattr__(self, name):  # .weights, .bytes, .table, ... of the wrapped provider
        return getattr(self.provider, name)


def build_arm(spec: dict):
    """(model, info) for an enabled arm; the caller holds the gpu lock, inference_mode on."""
    import torch
    check_arm_enabled(spec)
    if spec["kind"] == "M1-x":
        from inference.runtime import load_packed
        from inference.optimized import enable_optimizations
        model = load_packed(env.M1_EXPORT)
        model.decoding.decoding.decoding_computer.force_cuda_graphs_mode("full_graph")
        enable_optimizations(model, max_graphs=4, encoder_storage="expanded")
    elif spec["kind"] == "MB1-x":
        from gpu1bit import runtime
        model = runtime.load_binary_packed(env.MB1_EXPORT, runtime="provider", kernel=spec["kernel"], parts=spec["parts"])
    else:
        if spec["model"] == "M1":
            import export as ternary_export
            model = ternary_export.load_export(env.M1_EXPORT, "cuda")
        else:
            import binary_export
            model = binary_export.load_export(env.MB1_EXPORT, "cuda")
        model.eval()
    torch.cuda.synchronize()
    comp = model.decoding.decoding.decoding_computer
    return model, {"stack": STACKS[spec["kind"]], "decoder_cuda_graphs_mode": str(getattr(comp, "cuda_graphs_mode", None)),
                   "pipeline_graph_cache": type(model.forward).__name__}


# --- identity (R4) -------------------------------------------------------------------------------

CANDIDATE_DIRS = ("int8_mma", "triton_bf16", "bitserial")


def _source_files(spec: dict) -> list[Path]:
    """Every source an arm can import. Common: H's shared/ and runtime.py, M1's inference/, the
    parakeet-ternary and parakeet-binary modules, hf/load_ternary.py, whisper-ternary/wer.py, the
    wrappers and data files. MB1 arms add every file of every candidate directory (the registry
    imports all of them), so any new or changed file there (e.g. B's b01 / part-count files) changes
    the identity. check_imports_covered enforces that nothing imported falls outside this set."""
    ternary = env.TERNARY_DIR
    files = [p for p in env.SHARED.glob("*.py")] + [env.GPU1BIT / "runtime.py", env.GPU1BIT / "runtime_m1.py"]
    files += [p for p in (env.GPU1BIT / "x2").rglob("*.py") if "__pycache__" not in p.parts]   # x2-default code
    files += sorted((ternary / "inference").glob("*.py"))
    files += [ternary / f for f in ("evaluate.py", "export.py", "quant.py", "paths.py", "python")]
    files += [ternary / "hf" / "load_ternary.py"]
    files += sorted(env.BINARY_DIR.glob("binary_*.py")) + [env.BINARY_DIR / "python"]
    files += [env.REPO / "finetune" / "whisper-ternary" / f for f in ("wer.py", "quant.py")]  # quant.load_whisper_module
    files += [env.SHARED / "python", env.SHARED / "workloads.json"]   # selected.json: by content, see identity()
    if spec["model"] == "MB1" and spec["kind"] != "fp32":
        for d in CANDIDATE_DIRS:
            files += [p for p in (env.GPU1BIT / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    return sorted(set(files))


def export_identity(directory: Path) -> dict:
    """The complete export as the loaders consume it (review c1 major 2): every file of the export
    directory hashed (weights, manifest.json with its model configuration, tokenizer files, ...), plus the
    manifest's weights digest and a hash of its configuration for readability."""
    manifest = json.loads((directory / "manifest.json").read_text())
    files = {str(p.relative_to(directory)): env.sha256_file(p) for p in sorted(directory.rglob("*")) if p.is_file()}
    return {"path": str(directory), "sha256": manifest["sha256"], "files": files,
            "config_sha256": hashlib.sha256(json.dumps(manifest["config"], sort_keys=True).encode()).hexdigest()}


def identity(arm: str, subset: str, selection=None) -> dict:
    """The arm identity; `selection` is the selection document in use (default: seldoc.current_path())."""
    spec = parse_arm(arm)
    import importlib.metadata as md
    import evaluate as ev
    import paths
    files = _source_files(spec)
    sources = {str(p.relative_to(env.REPO)): env.sha256_file(p) for p in files}
    export = env.M1_EXPORT if spec["model"] == "M1" else env.MB1_EXPORT
    tables = [env.TERNARY_DIR / "inference" / f for f in ("encoder4.py", "fused_kernels.py", "kernels.py")]
    if spec["kernel"]:
        tables.append(env.GPU1BIT / spec["kernel"].split(":")[0] / "configs.py")
    import transformers.models.whisper.english_normalizer as en
    doc = {
        "arm": arm, "spec": spec, "subset": subset, "sets": list(DEV_SETS),
        "export": export_identity(export),
        "stack": STACKS[spec["kind"]], "parts": spec["parts"],
        "kernel": spec["kernel"], "config_tables": {str(p.relative_to(env.REPO)): env.sha256_file(p) for p in tables if p.exists()},
        "sources": sources,
        "scorer": {"evaluate.py": env.sha256_file(Path(ev.__file__)), "wer.py": env.sha256_file(env.REPO / "finetune/whisper-ternary/wer.py"),
                   "normalizer.json": env.sha256_file(ev.NORMALIZER_JSON), "english_normalizer.py": env.sha256_file(Path(en.__file__)),
                   "excluded_references": sorted(ev.EXCLUDED_REFERENCES), "min_decode_samples": ev.MIN_DECODE_SAMPLES},
        "manifests": {n: manifest(n)[1]["sha256"] for n in DEV_SETS},
        "versions": {k: md.version(k) for k in ("torch", "triton", "nemo_toolkit", "transformers")},
        "dev_sets": list(paths.DEV_SETS),
    }
    if spec["kind"] != "fp32":
        from gpu1bit.shared import seldoc
        doc["selection"] = {"sha256": seldoc.load(selection)["sha256"]}   # the CONTENT in use, not a path (x2-default Y1)
    doc["hash"] = hashlib.sha256(json.dumps({k: v for k, v in doc.items() if k != "hash"}, sort_keys=True).encode()).hexdigest()
    return doc


def arm_dir(run: str, arm: str, subset: str) -> Path:
    return ROOT / run / (arm if subset == "full" else f"{arm}@{subset.replace(':', '-')}")


def freeze(run: str, arm: str, subset: str) -> str:
    """Write the arm's identity once; an existing one must be identical (else refuse)."""
    d = arm_dir(run, arm, subset)
    env.require_mount()
    d.mkdir(parents=True, exist_ok=True)
    doc = identity(arm, subset)
    path = d / "identity.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["hash"] != doc["hash"]:
            raise SystemExit(f"identity of {arm}@{subset} changed since it was frozen: {diff_identity(old, doc)[:10]}")
    else:
        path.write_text(json.dumps(doc, indent=1) + "\n")
    return doc["hash"]


def diff_identity(a: dict, b: dict) -> list[str]:
    out = []
    for key in sorted(set(a) | set(b)):
        if key == "hash":
            continue
        va, vb = a.get(key), b.get(key)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += [f"{key}.{k}" for k in sorted(set(va) | set(vb)) if va.get(k) != vb.get(k)]
        elif va != vb:
            out.append(key)
    return out


def check_identity(run: str, arm: str, subset: str) -> dict:
    path = arm_dir(run, arm, subset) / "identity.json"
    if not path.exists():
        raise SystemExit(f"no frozen identity for {arm}@{subset} in run {run}; run `fulldev freeze` (or drive) first")
    frozen = json.loads(path.read_text())
    now = identity(arm, subset)
    if now["hash"] != frozen["hash"]:
        raise SystemExit(f"identity mismatch for {arm}@{subset}: {diff_identity(frozen, now)[:10]}")
    return frozen


def check_imports_covered(doc: dict) -> None:
    """Every repo module this process imported must be among the identity's hashed sources."""
    covered = set(doc["sources"])
    stray = []
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f or not os.path.isabs(f):  # e.g. torch._classes / torch._ops report bare file names
            continue
        p = Path(f).resolve()
        if p.is_relative_to(env.REPO / "finetune") and p.suffix == ".py":
            rel = str(p.relative_to(env.REPO))
            if rel not in covered and "/tests/" not in rel:
                stray.append(rel)
    if stray:
        raise RuntimeError(f"imported repo modules outside the arm identity: {sorted(stray)[:10]}")


# --- rows, resume and scoring --------------------------------------------------------------------

def make_row(i: int, uid: str, hyp: str, *, tokens: list[int], padded_short: bool, reference: str,
             duration: float, timestamps: list[int] | None = None, enc_len: int | None = None,
             enc_frames: int | None = None, finite: bool = True, decode_error: str | None = None,
             identity_hash: str | None = None) -> dict:
    import evaluate as ev
    return {"i": i, "id": uid, "hyp": hyp, "tokens": tokens, "timestamps": timestamps or [],
            "empty": not hyp.strip(), "padded_short": padded_short, "duration": duration,
            "enc_len": enc_len, "enc_frames": enc_frames, "finite": finite, "decode_error": decode_error,
            "identity": identity_hash, "ref": reference, **ev.score(reference, hyp)}


def load_clip(rec: dict):
    """(clip, padded_short) of a manifest record exactly as evaluate.py prepares it: load_audio + pad_short."""
    import evaluate as ev
    return ev.pad_short(ev.load_audio(rec["audio_filepath"]))


def prefetched(items, loader=load_clip):
    """Yield (index, record, loader(record)) for [(index, record)] in order, loading the NEXT record's
    audio in one background thread while the caller processes the current one (one ahead, never more).
    A loading error is raised at its own record, after every earlier record was delivered."""
    from concurrent.futures import ThreadPoolExecutor
    items = list(items)
    if not items:
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="fulldev-prefetch") as pool:
        future = pool.submit(loader, items[0][1])
        for k, (i, rec) in enumerate(items):
            audio = future.result()
            if k + 1 < len(items):
                future = pool.submit(loader, items[k + 1][1])
            yield i, rec, audio


def process_one(model, i: int, rec: dict, identity_hash: str, audio=None) -> dict:
    """Transcribe one manifest record at batch 1 and return its row (structural facts included).
    `audio` is load_clip(rec) when the caller prefetched it; otherwise it is loaded here."""
    import torch
    import evaluate as ev
    clip, padded = audio if audio is not None else load_clip(rec)
    audio = torch.from_numpy(clip).unsqueeze(0)
    length = torch.tensor([len(clip)], dtype=torch.long)
    hyp, tokens, stamps, error = "", [], [], None
    device = _device(model)
    with torch.inference_mode(), ev.strict_fp32(device), ev.inference_settings(model):
        enc, enc_len = model.forward(input_signal=audio.to(device), input_signal_length=length.to(device))
        finite = bool(torch.isfinite(enc).all())
        try:
            hyps = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=enc_len,
                                                                   return_hypotheses=True)
            if isinstance(hyps, tuple):
                hyps = hyps[0]
            h = hyps[0]
            hyp = h.text if hasattr(h, "text") else str(h)
            seq = getattr(h, "y_sequence", None)
            tokens = [int(t) for t in (seq.tolist() if hasattr(seq, "tolist") else (seq or []))]
            ts = getattr(h, "timestamp", None)
            if ts is not None and hasattr(ts, "tolist"):
                stamps = [int(t) for t in ts.tolist()]
            elif isinstance(ts, list) and all(isinstance(t, int) for t in ts):
                stamps = ts
        except Exception as exc:  # noqa: BLE001 - R5: recorded; invalidates the arm
            error = repr(exc)
    return make_row(i, rec["id"], hyp, tokens=tokens, timestamps=stamps, padded_short=padded, reference=rec["text"],
                    duration=rec["duration"], enc_len=int(enc_len[0]), enc_frames=int(enc.shape[-1]), finite=finite,
                    decode_error=error, identity_hash=identity_hash)


def _device(model):
    return next(model.parameters()).device


def work_path(run: str, arm: str, name: str, subset: str = "full") -> Path:
    return arm_dir(run, arm, subset) / f"{name}.jsonl"


def load_rows(path: Path) -> list[dict]:
    """Rows of a working file; a torn final line (a killed writer) is cut off the file."""
    if not path.exists():
        return []
    data = path.read_bytes()
    keep = data.rfind(b"\n") + 1
    if keep != len(data):
        with open(path, "r+b") as f:
            f.truncate(keep)
        data = data[:keep]
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def summarize_set(name: str, rows: list[dict], subset: str = "full", identity_hash: str | None = None) -> dict:
    """The set summary: every subset utterance exactly once and (if given) one identity, else ValueError."""
    import evaluate as ev
    records = [r for _, r in subset_records(name, subset)]
    meta = manifest(name)[1]
    want = [r["id"] for r in records]
    got = [r["id"] for r in rows]
    if len(set(got)) != len(got):
        dup = sorted({g for g in got if got.count(g) > 1})[:5]
        raise ValueError(f"{name}: utterances counted twice: {dup}")
    if set(got) != set(want):
        missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
        raise ValueError(f"{name}: {len(missing)} missing (e.g. {missing[:3]}), {len(extra)} not in the subset (e.g. {extra[:3]})")
    if identity_hash is not None:
        mixed = sorted({r.get("identity") for r in rows} - {identity_hash}, key=str)
        if mixed:
            raise ValueError(f"{name}: rows from other arm identities {mixed[:3]}")
    by_id = {r["id"]: r for r in rows}
    ordered = [by_id[u] for u in want]
    for rec, row in zip(records, ordered):
        if row["ref"] != rec["text"]:
            raise ValueError(f"{name}: {row['id']}: reference text differs from the manifest")
    wer = ev.wer_summary(ordered)
    structural = {"not_finite": [r["id"] for r in ordered if r.get("finite") is False],
                  "decode_errors": [r["id"] for r in ordered if r.get("decode_error")]}
    return {"set": name, "subset": subset, "manifest": meta["path"], "manifest_sha256": meta["sha256"],
            "split": meta["split"], "utterances": len(ordered), "wer": wer, "wer_percent": 100 * wer["wer"],
            "empty_hypotheses": sum(r["empty"] for r in ordered), "padded_short": sum(r["padded_short"] for r in ordered),
            "audio_hours": sum(r["duration"] for r in ordered) / 3600, "structural": structural,
            "structurally_valid": not structural["not_finite"] and not structural["decode_errors"], "records": ordered}


def dev_mean(per_set: dict[str, dict]) -> float:
    """Unweighted mean of the five per-set corpus WERs (a fraction), as train.mean_wer."""
    if set(per_set) != set(DEV_SETS):
        raise ValueError(f"dev mean needs all of {DEV_SETS}")
    return sum(per_set[n]["wer"]["wer"] for n in DEV_SETS) / len(DEV_SETS)


def cross_check(model: str, per_set_percent: dict[str, float], mean_percent: float) -> dict:
    """R2: mean within 0.02 points AND every set within 0.10 points of the pinned FP32-NeMo scores."""
    pinned = json.loads(PINNED[model].read_text())
    pinned_sets = {n: 100 * pinned["dev_wer"][n] for n in DEV_SETS}
    pinned_mean = 100 * pinned["dev_mean_wer"]
    sets = {n: per_set_percent[n] - pinned_sets[n] for n in DEV_SETS}
    ok = abs(mean_percent - pinned_mean) <= MEAN_POINTS and all(abs(v) <= SET_POINTS for v in sets.values())
    return {"pinned": str(PINNED[model]), "pinned_mean_percent": pinned_mean, "pinned_sets_percent": pinned_sets,
            "mean_difference_points": mean_percent - pinned_mean, "set_difference_points": sets,
            "limits_points": {"mean": MEAN_POINTS, "each_set": SET_POINTS}, "pass": ok}


def chunks_path(run: str, arm: str, subset: str = "full") -> Path:
    return arm_dir(run, arm, subset) / "chunks.jsonl"


def chunks(run: str, arm: str, subset: str = "full") -> list[dict]:
    p = chunks_path(run, arm, subset)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def finish(run: str, arm: str, subset: str = "full") -> dict | None:
    """Per-set JSON (summary + records) for complete sets; summary.json once all are complete (CPU).
    Refuses rows or chunks of another identity than the frozen one."""
    d = arm_dir(run, arm, subset)
    frozen = check_identity(run, arm, subset)
    bad_chunks = [c["pid"] for c in chunks(run, arm, subset) if c.get("identity") != frozen["hash"]]
    if bad_chunks:
        raise ValueError(f"chunks from other identities: {bad_chunks[:5]}")
    per_set, status = {}, {}
    for name in DEV_SETS:
        rows = load_rows(work_path(run, arm, name, subset))
        total = len(subset_records(name, subset))
        status[name] = f"{len(rows)}/{total}"
        if len(rows) == total:
            summary = summarize_set(name, rows, subset, frozen["hash"])
            tmp = d / f"{name}.json.tmp{os.getpid()}"
            tmp.write_text(json.dumps(summary, ensure_ascii=False) + "\n")
            os.replace(tmp, d / f"{name}.json")
            per_set[name] = summary
    if len(per_set) != len(DEV_SETS):
        print(json.dumps({"arm": arm, "subset": subset, "complete": False, "sets": status}))
        return None
    spec = parse_arm(arm)
    mean = dev_mean(per_set)
    out = {"run": run, "arm": arm, "subset": subset, "spec": spec, "identity": frozen["hash"], "complete": True,
           "dev_wer_percent": {n: per_set[n]["wer_percent"] for n in DEV_SETS}, "dev_mean_wer_percent": 100 * mean,
           "utterances": {n: per_set[n]["utterances"] for n in DEV_SETS},
           "empty_hypotheses": {n: per_set[n]["empty_hypotheses"] for n in DEV_SETS},
           "padded_short": {n: per_set[n]["padded_short"] for n in DEV_SETS},
           "excluded_empty_reference": {n: per_set[n]["wer"]["excluded_empty_reference"] for n in DEV_SETS},
           "structural": {n: per_set[n]["structural"] for n in DEV_SETS},
           "structurally_valid": all(per_set[n]["structurally_valid"] for n in DEV_SETS),
           "chunks": [{k: c.get(k) for k in ("pid", "rows", "held_s", "work_s", "stopped_by_budget", "graph_entries",
                                             "lock_before_cuda", "lock_waited_s")} for c in chunks(run, arm, subset)]}
    if subset == "full" and spec["parts"] in (3, None):
        out["cross_check"] = cross_check(spec["model"], out["dev_wer_percent"], out["dev_mean_wer_percent"])
    (d / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out.get(k) for k in ("arm", "subset", "dev_mean_wer_percent", "dev_wer_percent", "empty_hypotheses",
                                              "structurally_valid", "cross_check")}, indent=1))
    return out


def length_differences(rows: list[dict], ref_rows: list[dict]) -> list[str]:
    """R5: ids whose encoder output length differs from the reference arm's (matched by id)."""
    ref = {r["id"]: r["enc_len"] for r in ref_rows}
    return [r["id"] for r in rows if r["id"] in ref and r["enc_len"] != ref[r["id"]]]


def compare(run: str, arm: str, ref: str, subset: str = "full") -> dict:
    """Token-ID agreement and structural comparison of `arm` against `ref` (same subset), per set and
    pooled (R2 fidelity, R5 lengths, R7 per-set churn)."""
    ids_a, ids_r = check_identity(run, arm, subset)["hash"], check_identity(run, ref, subset)["hash"]
    out = {"run": run, "arm": arm, "ref": ref, "subset": subset, "identities": {"arm": ids_a, "ref": ids_r}, "sets": {}}
    total = mism = 0
    for name in DEV_SETS:
        a = summarize_set(name, load_rows(work_path(run, arm, name, subset)), subset, ids_a)["records"]
        r = summarize_set(name, load_rows(work_path(run, ref, name, subset)), subset, ids_r)["records"]
        m = [x["id"] for x, y in zip(a, r) if x["tokens"] != y["tokens"]]
        lengths = length_differences(a, r)
        out["sets"][name] = {"utterances": len(a), "token_mismatches": len(m), "mismatch_rate": len(m) / len(a),
                             "mismatch_ids": m[:50], "encoder_length_differences": lengths[:50],
                             "text_mismatches": sum(x["hyp"] != y["hyp"] for x, y in zip(a, r))}
        total += len(a)
        mism += len(m)
    out["pooled"] = {"utterances": total, "token_mismatches": mism, "mismatch_rate": mism / total}
    d = ROOT / run / "compare"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{arm}--vs--{ref}@{subset.replace(':', '-')}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"pooled": out["pooled"], "per_set": {n: (s["token_mismatches"], s["utterances"]) for n, s in out["sets"].items()}}))
    return out


# --- the GPU process -----------------------------------------------------------------------------

def run(args) -> None:
    """One locked GPU process (see module docstring); exits inside the lock."""
    code = 0
    spec = parse_arm(args.arm)
    check_arm_enabled(spec)
    sets = parse_sets(args.sets)
    import torch  # importing torch does not initialise CUDA
    cuda_before = torch.cuda.is_initialized()
    label = f"H:fulldev {args.arm}@{args.subset} {args.run}"
    with env.gpu_lock(label) as lock:
        chunk = {"pid": os.getpid(), "label": label, "lock_before_cuda": not cuda_before and not torch.cuda.is_initialized(),
                 "locks_taken": 1, "lock_waited_s": lock["waited_s"], "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "rows": {}, "stopped_by_budget": False, "wrapper": env.wrapper()}
        sink = None
        try:
            from nemo.utils import logging
            logging.setLevel(logging.ERROR)
            frozen = check_identity(args.run, args.arm, args.subset)
            chunk["identity"] = frozen["hash"]
            todo = {}
            for name in sets:
                done = {r["id"] for r in load_rows(work_path(args.run, args.arm, name, args.subset))}
                todo[name] = [(i, r) for i, r in subset_records(name, args.subset) if r["id"] not in done]
            if any(todo.values()):
                with torch.inference_mode():
                    model, info = build_arm(spec)
                check_imports_covered(frozen)
                chunk["arm_info"] = info
                work_start = time.monotonic()
                slowest = 0.0
                chunk["audio_prefetch"] = "one utterance ahead, one thread (fulldev.prefetched)"
                for name in sets:
                    with open(work_path(args.run, args.arm, name, args.subset), "a") as sink, \
                            contextlib.closing(prefetched(todo[name])) as stream:
                        last = time.monotonic()
                        for n_done, (i, rec, audio) in enumerate(stream):
                            now = time.monotonic()
                            margin = 3 * slowest + 1.0
                            if now - work_start + margin > args.work_budget or now - lock["acquired"] + margin > HOLD_CAP_S:
                                chunk["stopped_by_budget"] = True
                                break
                            row = process_one(model, i, rec, frozen["hash"], audio=audio)
                            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                            sink.flush()
                            if n_done % 64 == 63:
                                os.fsync(sink.fileno())
                            end = time.monotonic()
                            slowest = max(slowest, end - last)  # includes any wait for the prefetched audio
                            last = end
                            chunk["rows"][name] = chunk["rows"].get(name, 0) + 1
                        sink.flush()
                        os.fsync(sink.fileno())
                    if chunk["stopped_by_budget"]:
                        break
                chunk["work_s"] = time.monotonic() - work_start
                chunk["slowest_utterance_s"] = slowest
                chunk["graph_entries"] = len(getattr(model.forward, "entries", {}) or {})
                torch.cuda.synchronize()
        except BaseException as exc:  # noqa: BLE001 - recorded and reported, nonzero exit
            import traceback
            traceback.print_exc()
            chunk["error"] = repr(exc)
            code = 1
        chunk["held_s"] = time.monotonic() - lock["acquired"]
        chunk["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        env.require_mount()
        arm_dir(args.run, args.arm, args.subset).mkdir(parents=True, exist_ok=True)
        with open(chunks_path(args.run, args.arm, args.subset), "a") as f:
            f.write(json.dumps(chunk) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps({k: chunk.get(k) for k in ("pid", "rows", "held_s", "work_s", "stopped_by_budget", "error")}), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


# --- the driver (no CUDA) ------------------------------------------------------------------------

def remaining(run: str, arm: str, sets: list[str], subset: str) -> dict:
    return {n: len(subset_records(n, subset)) - len(load_rows(work_path(run, arm, n, subset))) for n in sets}


def drive(args) -> None:
    spec = parse_arm(args.arm)
    check_arm_enabled(spec)
    sets = parse_sets(args.sets)
    frozen_hash = freeze(args.run, args.arm, args.subset)
    d = arm_dir(args.run, args.arm, args.subset)
    for attempt in range(args.max_processes):
        left = remaining(args.run, args.arm, sets, args.subset)
        if not any(left.values()):
            break
        check_identity(args.run, args.arm, args.subset)
        cmd = ["timeout", "600", PY, "-m", "gpu1bit.shared.fulldev", "run", "--run", args.run, "--arm", args.arm,
               "--subset", args.subset, "--sets", *sets, "--work-budget", str(args.work_budget)]
        env_child = dict(os.environ)
        env_child.pop("CUBLAS_WORKSPACE_CONFIG", None)
        env_child.pop("CUDA_VISIBLE_DEVICES", None)
        with open(d / "driver.log", "a") as log:
            log.write(f"\n### {time.strftime('%Y-%m-%dT%H:%M:%S')} identity={frozen_hash[:16]} left={left} {' '.join(cmd)}\n")
            log.flush()
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(env.BINARY_DIR), env=env_child).returncode
        print(json.dumps({"process": attempt, "rc": rc, "left_before": left}), flush=True)
        if rc != 0:
            raise SystemExit(f"run process failed (rc {rc}); see {d / 'driver.log'}")
    else:
        raise SystemExit("max processes reached before completion")
    if set(sets) == set(DEV_SETS):
        finish(args.run, args.arm, args.subset)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    for name in ("run", "drive", "freeze", "finish"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--arm", required=True)
        s.add_argument("--subset", default="full")
        if name in ("run", "drive"):
            s.add_argument("--sets", nargs="+", default=None)
            s.add_argument("--work-budget", type=float, default=WORK_BUDGET_S, help="seconds of work per process (<= 480)")
        if name == "drive":
            s.add_argument("--max-processes", type=int, default=60)
    s = sub.add_parser("compare")
    s.add_argument("--run", required=True)
    s.add_argument("--arm", required=True)
    s.add_argument("--ref", required=True)
    s.add_argument("--subset", default="full")
    args = p.parse_args()
    if args.action in ("run", "drive") and not 0 < args.work_budget <= WORK_BUDGET_S:
        p.error(f"--work-budget must be in (0, {WORK_BUDGET_S:.0f}]")
    if args.action == "run":
        env.require_shared_wrapper()
        run(args)
    elif args.action == "drive":
        drive(args)
    elif args.action == "freeze":
        print(freeze(args.run, args.arm, args.subset))
    elif args.action == "finish":
        finish(args.run, args.arm, args.subset)
    else:
        compare(args.run, args.arm, args.ref, args.subset)


if __name__ == "__main__":
    main()
