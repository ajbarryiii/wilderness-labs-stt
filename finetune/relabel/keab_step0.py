"""Stage 4 step 0 (keab-prime plan "Step 0: 8-bit check of the floating-point parts"): which of Kea-B's floating-point
parts stages 4-5 train and export in 8 bits.

    ./python keab_step0.py prepare                  MB1's bundles and scoring views from its training checkpoint (GPU:
                                                    the binary row scales are computed on CUDA, the device MB1's
                                                    export was quantized on; the "none" view must reproduce MB1's
                                                    export.safetensors byte for byte, else exit 81)
    ./python-gpu1bit keab_step0.py mb1-check [--limit N]
                                                    MB1's export re-scored with the x1 builder (training/deployed.py)
                                                    on full dev; every set within 0.02 points of RESULTS.md's x1 arm,
                                                    else exit 81
    ./python-gpu1bit keab_step0.py run --model MB1 --variant V [--sets dev domain] [--limit N]
    ./python keab_step0.py run --model B0p --variant V [--sets dev domain] [--limit N]
    ./python keab_step0.py report [--limit N]       the preregistered rule and the step 0 tables (CPU)
    ./python keab_step0.py plan                     the production run's jobs (FULL_SETS), one per line
    ./python keab_step0.py status

Models and how each is scored ("Models"; each against its own unquantized scoring, variant "none"):
  MB1  Kea-B in deployed mode: the variant's scoring view (exports/<variant>/view) on the real i8x1 kernels at x1
       (training/deployed.py), batch 1, under ./python-gpu1bit with CUBLAS_WORKSPACE_CONFIG unset, as the recorded
       x1 arm ran.
  B0p  B0' ($RELABEL_ROOT/b0prime/runs/b100/model.nemo, sha 497ac8ca...) in FP32 with only its decoder parts
       quantized (eightbit.apply_decoder_scope, replaced once at load): stage 3's scorer (evaluate.evaluate_set at
       batch 64 and 1200 s per batch, decoder pinned to the stage's CUDA-graph mode, train.deterministic_mode), with
       the decoder's token IDs captured from the same decode.
Variants (VARIANTS): none; S1-int8, S2-int8 (the rule's scopes); S1-palette, S2-palette (informational); MB1 only,
S2-int8-w8a8 (informational: the S2 view with pre_encode.out's input passed through the x1 per-frame activation
quantizer, eightbit.x1_fake_quant, by a forward pre-hook, so pre_encode.out runs as W8A8).
Sets: "dev", the five stage 1 dev sets (H4 for the rule, YODAS reported); "domain", the dev-medical and dev-technical
parts with critical-term recall (stage 3's lists: medical, technical). The production run (FULL_SETS, `plan`) scores
the rule's variants on both groups and the informational ones (palettization, W8A8) on the dev sets only.

Rule (preregistered, report): an INT8 scope passes on a model if its H4 cost (quantized minus unquantized, points)
is at most 0.03, the upper bound of its 90% paired cluster-bootstrap interval (training/bootstrap.py, clusters as
training/devsplit.py) at most 0.06, and no H4 set is worse by more than 0.10. The largest scope that passes on both
models is chosen; if none does, the floating-point parts stay (the architect reports to AJ). Reported with it: every
variant's dev and domain WER, term recall and token-sequence mismatch against "none", per set and pooled, and the
bundles' bytes per part.

Outputs under $RELABEL_ROOT/keab-prime/step0/: exports/<variant>/{bundle,view}, exports/index.json,
mb1-x1-check/, scores/<model>/<variant>/<set>.json (records with token IDs) and run.json, report.json, report.md.
--limit N (first N utterances per set: manifest order for MB1's dev sets as fulldev, id order otherwise) writes under
$RELABEL_ROOT/keab-prime/smoke/step0-limit<N>/ instead, never the production path; its report is not a decision.
A set already written with the same configuration is skipped. Each GPU process holds the parakeet GPU lock for its
whole run (exit 3 if it is not free within --lock-wait seconds); a --limit smoke, and prepare (seconds of CUDA work
per variant), may skip it with --no-gpu-lock (paths.gpu_lock's convention: the lock is for GPU jobs longer than a quick
smoke test; the job's lease still bounds the VRAM).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import relabel_paths
from training import keab_export as ke

MODELS = ("MB1", "B0p")
VARIANTS = {  # name -> (scope, format, w8a8)
    "none": ("none", None, False),
    "S1-int8": ("S1", "int8", False), "S2-int8": ("S2", "int8", False),
    "S1-palette": ("S1", "palette", False), "S2-palette": ("S2", "palette", False),
    "S2-int8-w8a8": ("S2", "int8", True),
}
MODEL_VARIANTS = {"MB1": tuple(VARIANTS), "B0p": ("none", "S1-int8", "S2-int8", "S1-palette", "S2-palette")}
VIEW_VARIANTS = ("none", "S1-int8", "S2-int8", "S1-palette", "S2-palette")   # S2-int8-w8a8 scores the S2-int8 view
RULE_SCOPES = ("S1", "S2")
RULE_VARIANTS = ("none", "S1-int8", "S2-int8")
ALSO_AGAINST = {"S2-int8-w8a8": "S2-int8"}       # W8A8 is reported against "none" and against S2 (its INT8 weights)
FULL_SETS = {v: ("dev", "domain") if v in RULE_VARIANTS else ("dev",) for v in VARIANTS}
FULL_NOTE = ("Full run: the rule's variants (none, S1-int8, S2-int8) on the dev and domain sets; palettization and W8A8 "
             "on the dev sets only, as information. The rule uses H4 only; the domain sets are reported for the rule's "
             "variants.")
COST_MAX, CI_UPPER_MAX, SET_MAX = 0.03, 0.06, 0.10
H4 = ("librispeech_dev_clean", "librispeech_dev_other", "ami_dev", "voxpopuli_dev")
YODAS = "yodas_dev"
DEV_SETS = H4 + (YODAS,)
DOMAIN_SETS = ("dev-medical", "dev-technical")
TERM_KIND = {"dev-medical": "medical", "dev-technical": "technical"}     # stage 3's (train_b0prime.TERM_KIND)
B0P_SHA256 = "497ac8ca2f08807797f0de89dd93e24c689a03013eaa5af207b34201251175af"
MB1_EXPORT_SHA256 = "9873d2f1a53127c673d0201db98cd3ba859ba432f334fef55e9e24fd057b9651"
MB1_STEP = 250_000
# gpu1bit/RESULTS.md "Activation precision", arm MB1-int8_mma-x1 (dev mean 6.3333%): the per-set WERs of its run
# ap-20261020r4 (identical in ap-20261020, r2 and r3), $PARAKEET_BINARY_ROOT/gpu1bit/H/fulldev/<run>/MB1-int8_mma-x1.
RECORDED_X1 = {"librispeech_dev_clean": 2.1570130296726155, "librispeech_dev_other": 4.690653625213477,
               "ami_dev": 11.997454796118564, "voxpopuli_dev": 6.084065758914341, "yodas_dev": 6.737252288050863}
RECORDED_X1_MEAN = 6.333287899593973
CHECK_POINTS = 0.02
EVAL_BATCH, EVAL_MAX_SECONDS = 64, 1200.0     # stage 3's (train_decoder.EVAL_BATCH, score_domain.BATCH_SIZE)
EXIT_LOCK_BUSY, EXIT_CHECK = 3, 81


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# --- locations ---------------------------------------------------------------------------------------

def step0_root() -> Path:
    return ke.keab_root() / "step0"


def out_root(limit: int | None) -> Path:
    """The production root, or the smoke root of a --limit run (never inside the production one)."""
    if limit is None:
        return step0_root()
    if limit <= 0:
        raise ValueError("--limit must be positive")
    out = ke.keab_root() / "smoke" / f"step0-limit{limit}"
    assert not out.is_relative_to(step0_root())
    return out


def exports_root() -> Path:
    return step0_root() / "exports"


def view_dir(variant: str) -> Path:
    if variant not in VIEW_VARIANTS:
        raise ValueError(f"{variant} has no view of its own")
    return exports_root() / variant / "view"


def scores_dir(model: str, variant: str, limit: int | None) -> Path:
    check_variant(model, variant)
    return out_root(limit) / "scores" / model / variant


def mb1_export() -> Path:
    return relabel_paths.PARAKEET_BINARY / "runs" / "main-binB-m1-lr2e-4" / "export"


def mb1_checkpoint() -> Path:
    return relabel_paths.PARAKEET_BINARY / "runs" / "main-binB-m1-lr2e-4" / "ckpt" / "latest.pt"


def b0p_nemo() -> Path:
    return relabel_paths.ARTIFACTS / "b0prime" / "runs" / "b100" / "model.nemo"


def check_variant(model: str, variant: str) -> tuple:
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    if variant not in MODEL_VARIANTS[model]:
        raise ValueError(f"{model} variants are {MODEL_VARIANTS[model]}, got {variant!r}")
    return VARIANTS[variant]


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
        try:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
    return digest.hexdigest()


def cached_sha256(path: Path) -> str:
    """SHA-256 of a large file, cached under $RELABEL_ROOT/keab-prime/caches/sha256 by the file's identity."""
    path = Path(path).resolve()
    st = path.stat()
    identity = {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "inode": st.st_ino}
    entry = ke.keab_root() / "caches" / "sha256" / (hashlib.sha256(str(path).encode()).hexdigest()[:32] + ".json")
    if entry.exists():
        cached = read_json(entry)
        if {k: cached.get(k) for k in identity} == identity:
            return cached["sha256"]
    digest = sha256_file(path)
    if path.stat().st_mtime_ns == st.st_mtime_ns:
        write_json(entry, {**identity, "sha256": digest})
    return digest


# --- sets and records ----------------------------------------------------------------------------------

def expand_sets(groups) -> list[tuple[str, str]]:
    """[(set or part name, "dev" | domain set)] for the groups "dev" and "domain"."""
    import devsets
    out = []
    for g in groups:
        if g == "dev":
            out += [(n, "dev") for n in DEV_SETS]
        elif g == "domain":
            out += [(p, s) for s in DOMAIN_SETS for p in devsets.SETS[s]]
        else:
            raise ValueError(f"set group must be dev or domain, got {g!r}")
    return out


def manifest_of(name: str, group: str) -> Path:
    if group == "dev":
        import paths
        return paths.MANIFESTS / f"{name}.jsonl"
    import score_domain
    return score_domain.part_manifests([name], None)[name][0]


def deployed_records(name: str, group: str, limit: int | None) -> list[tuple[int, dict]]:
    """[(index, record)]: fulldev's dev-set order (manifest order, its dev-split checks) or id order for parts."""
    import evaluate
    if group == "dev":
        from gpu1bit.shared import fulldev
        return fulldev.subset_records(name, "full" if limit is None else f"first:{limit}")
    return list(enumerate(evaluate.read_manifest(manifest_of(name, group))[:limit]))


def add_term_recall(result: dict, group: str) -> None:
    if group in TERM_KIND:
        import score_domain
        import terms
        result["term_recall"] = score_domain.term_recall(result["records"], terms.matcher(TERM_KIND[group]))
        result["term_kind"] = TERM_KIND[group]


# --- prepare: MB1's bundles and views --------------------------------------------------------------------

class Mb1Models:
    """Fresh copies of MB1's training model: the pinned base with the 264 binary modules (MB1's ramp kind) and its
    checkpoint's latent FP32 weights (step 250,000, weight fraction 1), on the CPU in eval mode. The checkpoint and
    the pinned .nemo are read once; each call builds the model from the base's config and tokenizer (NeMo models do
    not deep-copy) and loads the same state. `facts` records the checkpoint."""

    def __init__(self):
        import tempfile

        import torch
        import train  # noqa: F401  finetune/parakeet-ternary first (CUDA/numba configuration)
        import binary_quant
        import export as ternary_export
        from omegaconf import OmegaConf
        path = mb1_checkpoint()
        digest = cached_sha256(path)
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        args = payload["args"]
        self.facts = {"path": str(path), "sha256": digest, "step": int(payload["step"]),
                      "weight_fraction": payload.get("weight_fraction"), "init": args.get("init"),
                      "quantizer": args.get("quantizer"), "ramp_kind": args.get("ramp_kind"),
                      "run_name": args.get("run_name")}
        if self.facts["step"] != MB1_STEP or self.facts["quantizer"] != "binary" or self.facts["weight_fraction"] != 1.0:
            raise SystemExit(f"{path} is not MB1's final binary checkpoint: {self.facts}")
        train.verify_base()
        base = train.load_pretrained("cpu")
        self.config = OmegaConf.to_container(base.cfg, resolve=True)
        relabel_paths.require_mount()
        self.tokenizer = Path(tempfile.mkdtemp(prefix="keab-tokenizer-", dir=relabel_paths.ARTIFACTS / "tmp"))
        ternary_export._extract_tokenizer(base, self.tokenizer / "tokenizer")
        binary_quant.quantize_parakeet(base, self.facts["ramp_kind"])
        base.load_state_dict(payload["student"], strict=True)
        del payload
        self.state = {k: v.detach().clone() for k, v in base.state_dict().items()}
        del base

    def __call__(self):
        import binary_quant
        model = ke._build_model(self.config, self.tokenizer / "tokenizer")
        binary_quant.quantize_parakeet(model, self.facts["ramp_kind"])
        model.load_state_dict(self.state, strict=True)
        binary_quant.set_weight_fraction(model, 1.0)
        return model.eval()

    def close(self) -> None:
        import shutil
        shutil.rmtree(self.tokenizer, ignore_errors=True)


def prepare(args) -> int:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("prepare computes MB1's binary scales on CUDA, the device its export was quantized on")
    root = exports_root()
    index_path = root / "index.json"
    index = read_json(index_path) if index_path.exists() else {"variants": {}}
    models = Mb1Models()
    try:
        return _prepare(args, models, index, index_path)
    finally:
        models.close()


def _prepare(args, models, index: dict, index_path: Path) -> int:
    import contextlib

    import torch
    import score_teacher
    from training import eightbit as eb
    ckpt = models.facts
    index["checkpoint"] = ckpt
    index["mb1_export"] = {"path": str(mb1_export()), "sha256": MB1_EXPORT_SHA256}
    for variant in args.variants:
        scope, fmt, _ = VARIANTS[variant]
        vdir = view_dir(variant)
        entry = index["variants"].get(variant)
        if entry and entry.get("checkpoint_sha256") == ckpt["sha256"] and (vdir / "manifest.json").exists() \
                and read_json(vdir / "manifest.json")["sha256"] == entry["view"]["sha256"] and not args.force:
            log(f"{variant}: view exists ({entry['view']['sha256'][:12]}), skipping")
            continue
        t = time.perf_counter()
        extra = {"stage": "keab-prime step 0", "variant": variant, "checkpoint": ckpt}
        entry = {"scope": scope, "format": fmt, "checkpoint_sha256": ckpt["sha256"], "bundle": None}
        lock = (contextlib.nullcontext() if args.no_gpu_lock
                else score_teacher.gpu_lock(f"keab-step0 prepare {variant}", args.lock_wait))
        with lock:
            model = models().to("cuda")
            if fmt == "palette":
                entry["form"] = eb.apply_keab_palette(model, scope)
                entry["palette"] = {k: v["info"] for k, v in eb.frozen_palette(model).items()}
                view = ke.write_view(model, vdir, extra={**extra, "informational": "palettized, never bundled",
                                                         "form": entry["form"]})
            else:
                entry["form"] = eb.apply_keab_scope(model, scope)
                bundle = ke.write_bundle(model, vdir.parent / "bundle", extra=extra)
                entry["bundle"] = {"path": str(vdir.parent / "bundle"), "sha256": bundle["sha256"],
                                   "bytes": bundle["bytes"], "tensor_mb": ke.tensor_megabytes(bundle)}
            del model
            torch.cuda.empty_cache()
        if fmt != "palette":
            view = ke.derive_view(vdir.parent / "bundle", vdir, extra={"variant": variant})
        entry["view"] = {"path": str(vdir), "sha256": view["sha256"], "bytes": view["bytes"],
                         "tensor_mb": ke.megabytes(ke.view_tensor_bytes(view))}
        if variant == "none" and view["sha256"] != MB1_EXPORT_SHA256:
            write_json(index_path, index)
            log(f"none: the view's export.safetensors is {view['sha256']}, not MB1's {MB1_EXPORT_SHA256}")
            return EXIT_CHECK
        index["variants"][variant] = entry
        write_json(index_path, index)
        log(f"{variant}: view {view['sha256'][:12]}, bundle "
            + (f"{entry['bundle']['tensor_mb']:.3f} MB" if entry["bundle"] else "none")
            + f" ({time.perf_counter() - t:.0f} s)")
    return 0


# --- scoring ---------------------------------------------------------------------------------------------

def set_config(model: str, variant: str, name: str, group: str, limit: int | None, weights: dict) -> dict:
    return {"model": model, "variant": variant, "set": name, "group": group, "limit": limit, "weights": weights,
            "manifest_sha256": sha256_file(manifest_of(name, group)),
            "scorer": ({"kind": "deployed x1", "batch": 1} if model == "MB1" else
                       {"kind": "stage 3 evaluate_set", "batch": EVAL_BATCH, "max_batch_seconds": EVAL_MAX_SECONDS})}


def pending(out: Path, configs: dict) -> list[str]:
    todo = []
    for name, config in configs.items():
        path = out / f"{name}.json"
        if not (path.exists() and read_json(path).get("config") == config):
            todo.append(name)
    return todo


def gpu_lock(args, label: str):
    """The parakeet GPU lock (score_teacher.gpu_lock), or none for a --limit smoke run with --no-gpu-lock."""
    import contextlib
    import score_teacher
    if getattr(args, "no_gpu_lock", False):
        if args.limit is None:
            raise SystemExit("--no-gpu-lock is for --limit smoke runs only")
        return contextlib.nullcontext()
    return score_teacher.gpu_lock(label, args.lock_wait)


def vram_lease() -> None:
    import torch
    fraction = os.environ.get("AGENT_JOBS_VRAM_FRACTION")
    if fraction:
        torch.cuda.set_per_process_memory_fraction(float(fraction))


def run(args) -> int:
    import score_teacher
    scope, fmt, w8a8 = check_variant(args.model, args.variant)
    relabel_paths.require_mount()
    out = scores_dir(args.model, args.variant, args.limit)
    sets = expand_sets(args.sets)
    if args.model == "MB1":
        from training import deployed
        deployed.require_wrapper()
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
            raise SystemExit("MB1 is scored as the recorded x1 arm ran: unset CUBLAS_WORKSPACE_CONFIG")
        vdir = view_dir("S2-int8" if w8a8 else args.variant)
        vman = read_json(vdir / "manifest.json")
        weights = {"view": str(vdir), "view_sha256": vman["sha256"], "w8a8_pre_encode_out": w8a8}
    else:
        weights = {"nemo": str(b0p_nemo()), "sha256": B0P_SHA256, "scope": scope, "format": fmt,
                   "quantizer_sha256": sha256_file(Path(__file__).resolve().parent / "training" / "eightbit.py")}
    configs = {n: set_config(args.model, args.variant, n, g, args.limit, weights) for n, g in sets}
    todo = pending(out, configs)
    if not todo:
        log(f"{args.model} {args.variant}: every set already scored")
        return 0
    groups = dict(sets)
    label = f"keab-step0 {args.model} {args.variant}"
    try:
        with gpu_lock(args, label):
            vram_lease()
            if args.model == "MB1":
                facts = _run_mb1(out, todo, groups, configs, args.limit, vdir, w8a8)
            else:
                facts = _run_b0p(out, todo, groups, configs, args.limit, scope, fmt)
    except score_teacher.LockBusy as busy:
        print(f"refused: {busy}", file=sys.stderr)
        return EXIT_LOCK_BUSY
    write_json(out / "run.json", {"model": args.model, "variant": args.variant, "limit": args.limit,
                                  "weights": weights, "sets": [n for n, _ in sets], **facts,
                                  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return 0


def _finish_set(out: Path, name: str, group: str, result: dict, config: dict) -> None:
    add_term_recall(result, group)
    result.update(config=config, model=config["model"], variant=config["variant"], group=group)
    write_json(out / f"{name}.json", result)
    tr = result.get("term_recall")
    log(f"{config['model']} {config['variant']} {name}: n={result['utterances']} WER {100 * result['wer']['wer']:.3f}"
        + (f" term recall {tr['recall']:.4f}" if tr and tr["recall"] is not None else "")
        + f" ({result['wall_s']:.0f} s)")


def _run_mb1(out, todo, groups, configs, limit, vdir: Path, w8a8: bool) -> dict:
    import torch
    from training import deployed
    from training import eightbit as eb
    model = deployed.load_x1(vdir)
    calls = {"pre_encode_out_quantized": 0}
    if w8a8:
        def pre_hook(module, inputs):
            calls["pre_encode_out_quantized"] += 1
            return (eb.x1_fake_quant(inputs[0]),) + tuple(inputs[1:])
        model.encoder.pre_encode.out.register_forward_pre_hook(pre_hook)
    try:
        identity = hashlib.sha256(json.dumps(configs[todo[0]]["weights"], sort_keys=True).encode()).hexdigest()
        for name in todo:
            result = deployed.score_records(model, deployed_records(name, groups[name], limit), name, identity)
            _finish_set(out, name, groups[name], result, configs[name])
        facts = {"x1": deployed.info(model), "w8a8_hook_calls": calls["pre_encode_out_quantized"],
                 "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2**30}
        if w8a8 and calls["pre_encode_out_quantized"] == 0:
            raise RuntimeError("the W8A8 pre-hook never ran")
    finally:
        deployed.close(model)
    return facts


def decoder_sha256(net, keys) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode())
        digest.update(net.get_parameter(key).detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def capture_tokens(net):
    """Wrap net.decoding.rnnt_decoder_predictions_tensor (unchanged arguments and result) to keep each batch's
    hypotheses' token IDs (NeMo returns Hypothesis(score, y_sequence, text) either way)."""
    batches = []
    original = net.decoding.rnnt_decoder_predictions_tensor

    def wrapped(*a, **k):
        result = original(*a, **k)
        hyps = result[0] if isinstance(result, tuple) else result
        batches.append([[int(t) for t in (h.y_sequence.tolist() if hasattr(h.y_sequence, "tolist")
                                          else (h.y_sequence or []))] for h in hyps])
        return result
    net.decoding.rnnt_decoder_predictions_tensor = wrapped
    return batches, lambda: setattr(net.decoding, "rnnt_decoder_predictions_tensor", original)


def attach_tokens(records: list[dict], manifest_records: list[dict], captured: list[list[list[int]]]) -> None:
    """Token IDs per record: evaluate.make_batches gives the batch order evaluate.decode ran (records keep manifest
    order); ValueError if the capture does not match it."""
    import evaluate
    batches = evaluate.make_batches(manifest_records, EVAL_BATCH, EVAL_MAX_SECONDS)
    if len(batches) != len(captured) or any(len(b) != len(c) for b, c in zip(batches, captured)):
        raise ValueError("captured decoder batches do not match evaluate.make_batches")
    for idx, toks in zip(batches, captured):
        for i, t in zip(idx, toks):
            records[i]["tokens"] = t
    if any("tokens" not in r for r in records):
        raise ValueError("a record got no tokens")


def _run_b0p(out, todo, groups, configs, limit, scope: str, fmt: str | None) -> dict:
    import torch
    import train  # noqa: I001  finetune/parakeet-ternary first, as stage 3's score_run
    import binary_train
    import evaluate
    from training import eightbit as eb
    train.deterministic_mode()
    nemo = b0p_nemo()
    source = evaluate.describe_source("nemo", nemo, "B0' (stage 3 b100), FP32, decoder parts as the step 0 variant")
    if source["sha256"] != B0P_SHA256:
        raise SystemExit(f"{nemo}: SHA-256 {source['sha256']} is not B0''s {B0P_SHA256}")
    net = train.load_nemo(nemo, "cuda").eval()
    label = "B0p"
    mode = binary_train.pin_decoder_graphs_mode(net, label)
    form = eb.apply_decoder_scope(net, scope, fmt or "int8")
    keys = eb.keys_of(("prediction_network", "joint"))
    facts = {"decoder_form": form, "decoder_sha256": decoder_sha256(net, keys), "decoder_graphs_mode": mode,
             "palette": {k: v["info"] for k, v in eb.frozen_palette(net).items()} or None}
    captured, restore = capture_tokens(net)
    device = torch.device("cuda")
    try:
        for name in todo:
            manifest = manifest_of(name, groups[name])
            captured.clear()
            result = evaluate.evaluate_set(net, name, manifest, source, EVAL_BATCH, limit, device, EVAL_MAX_SECONDS)
            attach_tokens(result["records"], evaluate.read_manifest(manifest)[:limit], captured)
            result["decoder_graphs_mode"] = binary_train.require_decoder_graphs_mode(net, label)
            result["decoder_sha256"] = facts["decoder_sha256"]
            _finish_set(out, name, groups[name], result, configs[name])
    finally:
        restore()
    facts["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
    return facts


# --- the MB1 x1 check ------------------------------------------------------------------------------------

def check_against_recorded(per_set: dict[str, float]) -> dict:
    diffs = {n: per_set[n] - RECORDED_X1[n] for n in DEV_SETS}
    mean = sum(per_set[n] for n in DEV_SETS) / len(DEV_SETS)
    return {"recorded": RECORDED_X1, "recorded_mean": RECORDED_X1_MEAN, "wer_percent": per_set, "mean": mean,
            "difference_points": diffs, "mean_difference_points": mean - RECORDED_X1_MEAN,
            "limit_points_per_set": CHECK_POINTS, "pass": all(abs(d) <= CHECK_POINTS for d in diffs.values())}


def mb1_check(args) -> int:
    import torch
    import score_teacher
    from training import deployed
    deployed.require_wrapper()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        raise SystemExit("the recorded x1 arm ran with CUBLAS_WORKSPACE_CONFIG unset; unset it")
    export = mb1_export()
    if read_json(export / "manifest.json")["sha256"] != MB1_EXPORT_SHA256:
        raise SystemExit(f"{export} is not MB1's export {MB1_EXPORT_SHA256}")
    out = out_root(args.limit) / "mb1-x1-check"
    weights = {"export": str(export), "sha256": MB1_EXPORT_SHA256}
    configs = {n: set_config("MB1", "export", n, "dev", args.limit, weights) for n in DEV_SETS}
    todo = pending(out, configs)
    if todo:
        try:
            with gpu_lock(args, "keab-step0 mb1-check"):
                vram_lease()
                model = deployed.load_x1(export)
                try:
                    for name in todo:
                        result = deployed.score_records(model, deployed_records(name, "dev", args.limit), name,
                                                        MB1_EXPORT_SHA256)
                        result.update(config=configs[name])
                        write_json(out / f"{name}.json", result)
                        log(f"mb1-check {name}: WER {result['wer_percent']:.4f} (recorded {RECORDED_X1[name]:.4f})"
                            f" {result['wall_s']:.0f} s")
                    info = deployed.info(model)
                finally:
                    deployed.close(model)
                write_json(out / "x1.json", {**info, "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2**30})
        except score_teacher.LockBusy as busy:
            print(f"refused: {busy}", file=sys.stderr)
            return EXIT_LOCK_BUSY
    per_set = {n: read_json(out / f"{n}.json")["wer_percent"] for n in DEV_SETS}
    summary = check_against_recorded(per_set)
    summary["limit"] = args.limit
    if args.limit is not None:
        summary["pass"] = None
        summary["note"] = "a --limit smoke: not the check"
    write_json(out / "summary.json", summary)
    log(f"mb1-check: {json.dumps(summary['difference_points'])} pass={summary['pass']}")
    return 0 if summary["pass"] in (True, None) else EXIT_CHECK


# --- the report -------------------------------------------------------------------------------------------

def rule(h4_cost: float, ci_upper: float, set_deltas: dict[str, float]) -> dict:
    worst = max(set_deltas.values())
    checks = {"h4_cost": h4_cost <= COST_MAX, "ci_upper": ci_upper <= CI_UPPER_MAX, "worst_set": worst <= SET_MAX}
    return {"h4_cost": h4_cost, "ci_upper": ci_upper, "worst_set_delta": worst,
            "worst_set": max(set_deltas, key=set_deltas.get), "checks": checks, "pass": all(checks.values()),
            "limits": {"h4_cost": COST_MAX, "ci_upper": CI_UPPER_MAX, "worst_set": SET_MAX}}


def choose(passes: dict[str, dict[str, bool]]) -> str:
    """The largest scope (RULE_SCOPES order) that passes on every model, else "none"."""
    chosen = "none"
    for scope in RULE_SCOPES:
        if all(p.get(scope) is True for p in passes.values()):
            chosen = scope
    return chosen


def mismatch(rows: list[dict], ref: list[dict]) -> dict:
    """Token-sequence mismatch of rows against the reference rows (matched by id)."""
    by_id = {r["id"]: r["tokens"] for r in ref}
    if set(by_id) != {r["id"] for r in rows}:
        raise ValueError("token comparison over different utterances")
    n = sum(r["tokens"] != by_id[r["id"]] for r in rows)
    return {"utterances": len(rows), "mismatches": n, "rate": n / len(rows) if rows else None}


def load_variant(model: str, variant: str, limit: int | None, names: list[str]) -> dict[str, dict] | None:
    d = scores_dir(model, variant, limit)
    out = {}
    for n in names:
        p = d / f"{n}.json"
        if not p.exists():
            return None
        out[n] = read_json(p)
    return out


def summarize_variant(model: str, variant: str, res: dict, base: dict, domain_parts: dict) -> dict:
    from training import bootstrap, devsplit
    wer = {n: res[n]["wer_percent"] if "wer_percent" in res[n] else 100 * res[n]["wer"]["wer"] for n in res}
    ref = {n: base[n]["wer_percent"] if "wer_percent" in base[n] else 100 * base[n]["wer"]["wer"] for n in base}
    out = {"wer": {n: wer[n] for n in DEV_SETS}, "delta": {n: wer[n] - ref[n] for n in DEV_SETS},
           "h4": sum(wer[n] for n in H4) / len(H4)}
    out["h4_delta"] = out["h4"] - sum(ref[n] for n in H4) / len(H4)
    tables = {n: bootstrap.SetTable.from_records({variant: res[n]["records"], "none": base[n]["records"]},
                                                 cluster_of=lambda i, n=n: devsplit.cluster_of(n, i)) for n in H4}
    boot = bootstrap.paired(tables, {"q-none": (variant, "none")}, list(H4))
    out["h4_ci"] = boot["mean"]["q-none"]["ci"]
    out["set_ci"] = {n: boot["sets"][n]["q-none"]["ci"] for n in H4}
    out["rule"] = rule(boot["mean"]["q-none"]["point"], out["h4_ci"][1], {n: out["delta"][n] for n in H4})
    out["token_mismatch"] = {n: mismatch(res[n]["records"], base[n]["records"]) for n in res}
    dev_rows = sum(out["token_mismatch"][n]["utterances"] for n in DEV_SETS)
    dev_mis = sum(out["token_mismatch"][n]["mismatches"] for n in DEV_SETS)
    out["token_mismatch_dev_pooled"] = {"utterances": dev_rows, "mismatches": dev_mis, "rate": dev_mis / dev_rows}
    domain = {}
    for s, parts in domain_parts.items():
        if not all(p in res for p in parts):
            continue
        entry = {"parts": {p: {"wer": wer[p], "delta": wer[p] - ref[p]} for p in parts},
                 "mean_wer": sum(wer[p] for p in parts) / len(parts)}
        entry["mean_delta"] = entry["mean_wer"] - sum(ref[p] for p in parts) / len(parts)
        tr = [res[p].get("term_recall") for p in parts]
        tr0 = [base[p].get("term_recall") for p in parts]
        if all(t and t["recall"] is not None for t in tr + tr0):
            occ, rep = sum(t["occurrences"] for t in tr), sum(t["reproduced"] for t in tr)
            occ0, rep0 = sum(t["occurrences"] for t in tr0), sum(t["reproduced"] for t in tr0)
            entry["pooled_term_recall"] = rep / occ
            entry["pooled_term_recall_delta"] = rep / occ - rep0 / occ0
        domain[s] = entry
    out["domain"] = domain
    return out


def compare_variant(model: str, variant: str, res: dict, baselines: dict[str, dict | None], domain_parts: dict) -> dict:
    """summarize_variant against "none", plus, for the variants in ALSO_AGAINST, the same summary against that
    variant (over the sets both have) under "against"."""
    out = summarize_variant(model, variant, res, baselines["none"], domain_parts)
    other = ALSO_AGAINST.get(variant)
    if other is not None and baselines.get(other) is not None:
        ref = {n: baselines[other][n] for n in res if n in baselines[other]}
        out["against"] = {other: summarize_variant(model, variant, {n: res[n] for n in ref}, ref, domain_parts)}
    return out


def palette_summary(infos: dict[str, dict]) -> dict:
    """Per tensor: entries, Lloyd iterations, convergence and the largest error; the tensors that did not converge
    within eightbit.PALETTE_MAX_ITERATIONS."""
    from training import eightbit as eb
    tensors = {k: {f: v.get(f) for f in ("entries", "iterations", "converged", "max_abs_error")}
               for k, v in sorted(infos.items())}
    return {"tensors": tensors, "not_converged": [k for k, v in tensors.items() if not v["converged"]],
            "max_iterations": eb.PALETTE_MAX_ITERATIONS}


def stage3_records(name: str) -> Path:
    """B0''s stage 3 record of a dev set or domain part (b100's eval-dev/ or domain/)."""
    run = relabel_paths.ARTIFACTS / "b0prime" / "runs" / "b100"
    return run / ("eval-dev" if name in DEV_SETS else "domain") / f"{name}.json"


def crosschecks(limit: int | None) -> dict:
    """The unquantized scorings against earlier records of the same weights: B0p "none" against stage 3's b100
    transcripts (same scorer), MB1 "none" (the scoring view) against the mb1-check tokens of MB1's own export."""
    out = {}
    d = scores_dir("B0p", "none", limit)
    sets = {}
    for path in sorted(d.glob("*.json")) if d.exists() else []:
        if path.stem == "run" or not stage3_records(path.stem).exists():
            continue
        old = {r["id"]: r["hyp"] for r in read_json(stage3_records(path.stem))["records"]}
        recs = read_json(path)["records"]
        sets[path.stem] = {"utterances": len(recs), "identical_text": sum(old.get(r["id"]) == r["hyp"] for r in recs)}
    out["B0p_none_vs_stage3"] = sets
    d, chk = scores_dir("MB1", "none", limit), step0_root() / "mb1-x1-check"
    sets = {}
    for name in DEV_SETS:
        if (d / f"{name}.json").exists() and (chk / f"{name}.json").exists():
            ref = {r["id"]: r["tokens"] for r in read_json(chk / f"{name}.json")["records"]}
            recs = read_json(d / f"{name}.json")["records"]
            sets[name] = {"utterances": len(recs), "identical_tokens": sum(ref.get(r["id"]) == r["tokens"] for r in recs)}
    out["MB1_none_view_vs_export"] = sets
    return out


def report(args) -> int:
    import devsets
    names_dev = list(DEV_SETS)
    domain_parts = {s: list(devsets.SETS[s]) for s in DOMAIN_SETS}
    names_all = names_dev + [p for s in DOMAIN_SETS for p in domain_parts[s]]
    out = {"limit": args.limit, "decision": args.limit is None, "models": {}, "rule_scopes": list(RULE_SCOPES),
           "full_run": {"sets": {v: list(g) for v, g in FULL_SETS.items()}, "note": FULL_NOTE}}
    passes: dict[str, dict[str, bool]] = {}
    for model in MODELS:
        base = load_variant(model, "none", args.limit, names_all) or load_variant(model, "none", args.limit, names_dev)
        if base is None:
            out["models"][model] = {"status": "unquantized scoring (none) incomplete"}
            continue
        entry = {"none": {"wer": {n: base[n]["wer_percent"] if "wer_percent" in base[n] else 100 * base[n]["wer"]["wer"]
                                  for n in names_dev}}, "variants": {}}
        entry["none"]["h4"] = sum(entry["none"]["wer"][n] for n in H4) / len(H4)
        loaded = {"none": base}
        for variant in MODEL_VARIANTS[model]:
            if variant == "none":
                continue
            required = [n for n, _ in expand_sets(FULL_SETS[variant]) if n in base]
            res = load_variant(model, variant, args.limit, required)
            if res is not None:                                   # sets beyond the full run's (a smoke scores all)
                more = load_variant(model, variant, args.limit, [n for n in base if n not in res]) or {}
                res.update(more)
            if res is None:
                entry["variants"][variant] = {"status": "incomplete"}
                continue
            loaded[variant] = res
            entry["variants"][variant] = compare_variant(model, variant, res, loaded, domain_parts)
        passes[model] = {VARIANTS[v][0]: entry["variants"][v]["rule"]["pass"] for v in entry["variants"]
                         if VARIANTS[v][1] == "int8" and not VARIANTS[v][2] and "rule" in entry["variants"][v]}
        out["models"][model] = entry
    complete = all(set(passes.get(m, {})) == set(RULE_SCOPES) for m in MODELS)
    out["passes"] = passes
    out["chosen_scope"] = choose(passes) if complete and args.limit is None else None
    if complete and args.limit is not None:
        out["slice_rule_outcome"] = choose(passes)       # a smoke slice: never a decision
    if not complete:
        out["note"] = "the rule needs S1-int8 and S2-int8 on both models"
    out["palette"] = {}
    idx = exports_root() / "index.json"
    if idx.exists():
        index = read_json(idx)
        out["palette"]["MB1"] = {v: palette_summary(e["palette"]) for v, e in index["variants"].items()
                                 if e.get("palette")}
        out["exports"] = {v: {"bundle_tensor_mb": (e.get("bundle") or {}).get("tensor_mb"),
                              "bundle_parts": ((e.get("bundle") or {}).get("bytes") or {}).get("parts"),
                              "view_sha256": e["view"]["sha256"]} for v, e in index["variants"].items()}
    check = step0_root() / "mb1-x1-check" / "summary.json"
    if check.exists():
        out["mb1_x1_check"] = read_json(check)
    out["crosschecks"] = crosschecks(args.limit)
    b0p = {}
    for v in MODEL_VARIANTS["B0p"]:
        run_json = scores_dir("B0p", v, args.limit) / "run.json"
        if run_json.exists() and read_json(run_json).get("palette"):
            b0p[v] = palette_summary(read_json(run_json)["palette"])
    out["palette"]["B0p"] = b0p
    root = out_root(args.limit)
    write_json(root / "report.json", out)
    (root / "report.md").write_text(markdown(out))
    print(markdown(out))
    return 0


def markdown(r: dict) -> str:
    lines = [f"# Stage 4 step 0: 8-bit floating-point parts{'' if r['decision'] else ' (smoke, not a decision)'}", "",
             r["full_run"]["note"], ""]
    if r.get("chosen_scope") is not None:
        lines += [f"**Chosen scope: {r['chosen_scope']}** (largest INT8 scope passing on both models).", ""]
    elif r.get("slice_rule_outcome") is not None:
        lines += [f"Rule outcome on this slice: {r['slice_rule_outcome']} (a smoke; not a decision).", ""]
    for model, m in r["models"].items():
        lines += [f"## {model}", ""]
        if "variants" not in m:
            lines += [m.get("status", ""), ""]
            continue
        lines += ["| Variant | H4 | H4 Δ | 90% CI | worst H4 set Δ | YODAS Δ | token mismatch (dev) | rule |",
                  "|---|---:|---:|---|---:|---:|---:|---|",
                  f"| none | {m['none']['h4']:.3f} | | | | | | |"]
        for v, e in m["variants"].items():
            if "rule" not in e:
                lines.append(f"| {v} | {e.get('status')} | | | | | | |")
                continue
            ru = e["rule"]
            rule_text = ("PASS" if ru["pass"] else "fail") if VARIANTS[v][1] == "int8" and not VARIANTS[v][2] else "info"
            lines.append(f"| {v} | {e['h4']:.3f} | {e['h4_delta']:+.3f} | [{e['h4_ci'][0]:+.3f}, {e['h4_ci'][1]:+.3f}] | "
                         f"{ru['worst_set_delta']:+.3f} ({ru['worst_set']}) | {e['delta'][YODAS]:+.3f} | "
                         f"{100 * e['token_mismatch_dev_pooled']['rate']:.2f}% | {rule_text} |")
        lines.append("")
        dom = [(v, s, d) for v, e in m["variants"].items() for s, d in (e.get("domain") or {}).items()]
        if dom:
            lines += ["| Variant | Domain set | mean WER | Δ | pooled term recall Δ |", "|---|---|---:|---:|---:|"]
            for v, s, d in dom:
                tr = d.get("pooled_term_recall_delta")
                lines.append(f"| {v} | {s} | {d['mean_wer']:.3f} | {d['mean_delta']:+.3f} | "
                             + ("–" if tr is None else f"{100 * tr:+.2f}") + " |")
            lines.append("")
    if r.get("exports"):
        lines += ["## MB1 bundles", "", "| Variant | bundle MB (tensors) |", "|---|---:|"]
        for v, e in r["exports"].items():
            lines.append(f"| {v} | {'–' if e['bundle_tensor_mb'] is None else f'{e['bundle_tensor_mb']:.3f}'} |")
        lines.append("")
    for model, variants in (r.get("palette") or {}).items():
        for v, summ in variants.items():
            nc = summ["not_converged"]
            lines.append(f"Palette {model} {v}: {len(summ['tensors'])} tensors, "
                         + (f"not converged within {summ['max_iterations']} iterations: {', '.join(nc)}" if nc
                            else "all converged"))
    lines.append("")
    for model, m in r["models"].items():
        for v, e in (m.get("variants") or {}).items():
            for other, a in (e.get("against") or {}).items():
                lines.append(f"{model} {v} against {other}: H4 Δ {a['h4_delta']:+.3f} [{a['h4_ci'][0]:+.3f}, "
                             f"{a['h4_ci'][1]:+.3f}], worst H4 set {a['rule']['worst_set_delta']:+.3f} "
                             f"({a['rule']['worst_set']}), token mismatch (dev) "
                             f"{100 * a['token_mismatch_dev_pooled']['rate']:.2f}%")
    lines.append("")
    for name, sets in (r.get("crosschecks") or {}).items():
        if sets:
            key = "identical_text" if "B0p" in name else "identical_tokens"
            lines += [f"Crosscheck {name}: " + ", ".join(f"{n} {e[key]}/{e['utterances']}" for n, e in sets.items()), ""]
    if r.get("mb1_x1_check"):
        c = r["mb1_x1_check"]
        lines += ["## MB1 x1 re-score", "", "| Set | WER | recorded | Δ |", "|---|---:|---:|---:|"]
        for n in DEV_SETS:
            lines.append(f"| {n} | {c['wer_percent'][n]:.4f} | {c['recorded'][n]:.4f} | {c['difference_points'][n]:+.4f} |")
        lines += ["", f"pass: {c['pass']}", ""]
    return "\n".join(lines) + "\n"


def status(args) -> int:
    idx = exports_root() / "index.json"
    print("exports:", sorted(read_json(idx)["variants"]) if idx.exists() else "none")
    for model in MODELS:
        for variant in MODEL_VARIANTS[model]:
            d = scores_dir(model, variant, None)
            done = sorted(p.stem for p in d.glob("*.json") if p.stem != "run") if d.exists() else []
            print(f"{model} {variant}: {len(done)} sets {done}")
    check = step0_root() / "mb1-x1-check" / "summary.json"
    print("mb1-check:", read_json(check)["pass"] if check.exists() else "not run")
    return 0


# --- CLI --------------------------------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("prepare")
    s.add_argument("--variants", nargs="+", default=list(VIEW_VARIANTS), choices=VIEW_VARIANTS)
    s.add_argument("--force", action="store_true")
    s.add_argument("--lock-wait", type=float, default=1800.0)
    s.add_argument("--no-gpu-lock", action="store_true",
                   help="skip the parakeet GPU lock (prepare's CUDA work is seconds per variant)")
    s = sub.add_parser("mb1-check")
    s.add_argument("--limit", type=int)
    s.add_argument("--lock-wait", type=float, default=1800.0)
    s.add_argument("--no-gpu-lock", action="store_true", help="--limit smoke runs only")
    s = sub.add_parser("run")
    s.add_argument("--model", required=True, choices=MODELS)
    s.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    s.add_argument("--sets", nargs="+", default=["dev", "domain"], choices=("dev", "domain"))
    s.add_argument("--limit", type=int)
    s.add_argument("--lock-wait", type=float, default=1800.0)
    s.add_argument("--no-gpu-lock", action="store_true", help="--limit smoke runs only")
    s = sub.add_parser("report")
    s.add_argument("--limit", type=int)
    sub.add_parser("status")
    sub.add_parser("plan")
    args = p.parse_args(argv)
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        p.error("--limit must be positive")
    if args.action in ("run", "mb1-check") and args.no_gpu_lock and args.limit is None:
        p.error("--no-gpu-lock is for --limit smoke runs only")
    return args


def full_plan() -> list[tuple[str, str, tuple[str, ...]]]:
    """(model, variant, set groups) of the production run, in order: every MB1 variant, then every B0' variant."""
    return [(m, v, FULL_SETS[v]) for m in MODELS for v in MODEL_VARIANTS[m]]


def plan(args) -> int:
    """One line per production job: MODEL VARIANT GROUP... (training/keab_launch.sh full reads it)."""
    for m, v, groups in full_plan():
        print(m, v, *groups)
    return 0


ACTIONS = {"plan": plan, "prepare": prepare, "mb1-check": mb1_check, "run": run, "report": report, "status": status}


def main(argv=None) -> int:
    """Every subcommand runs under training/write_guard.py: no write outside $RELABEL_ROOT except the parakeet GPU
    lock file (the shared lock, written by design)."""
    from training import write_guard
    args = parse_args(argv)
    relabel_paths.require_mount()
    with write_guard.active([relabel_paths.ARTIFACTS], files=[relabel_paths.GPU_LOCK]):
        return ACTIONS[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
