"""Stage 5 step 0 (keat-prime plan "Step 0: 8-bit check of the floating-point parts (M1)"): which of Kea-T's
floating-point parts stage 5 trains and exports in 8 bits, and the M1 check of the x1 loader ("Scoring and gate").

    ./python keat_step0.py prepare                  M1's bundles and scoring views from its latent checkpoint (GPU: the
                                                    ternary scales are computed on CUDA, the device M1's export was
                                                    quantized on; the "none" view must reproduce M1's
                                                    export.safetensors byte for byte, else exit 81)
    ./python-gpu1bit keat_step0.py m1-check [--limit N]
                                                    M1's export re-scored with the ternary x1 loader
                                                    (training/keat_deployed.py) on full dev; every set within 0.02
                                                    points of run ap-20261020r4's M1-x1 arm, else exit 81
    ./python-gpu1bit keat_step0.py run --variant V [--sets dev domain] [--limit N]
    ./python keat_step0.py report [--limit N]       the preregistered rule and the step 0 tables (CPU)
    ./python keat_step0.py plan                     the production run's jobs (full_plan), one per line
    ./python keat_step0.py status

Model and scoring: M1 in deployed mode, each variant on M1's own x1 stack at one activation part (gpu1bit's
runtime_m1.build_stack through training/keat_deployed.load_x1), batch 1, under ./python-gpu1bit with
CUBLAS_WORKSPACE_CONFIG unset, as the recorded x1 arm ran. "none" is M1's own export; S1-int8 and S2-int8 are the
scoring views of their bundles (training/keat_export.py), built from M1's latent checkpoint (training/keat_m1.py's
init check) in the training forms of training/keat_eightbit.py. B0' is not re-scored: its rows are stage 4's
($RELABEL_ROOT/keab-prime/step0/report.json, read only and refused unless its SHA-256 is STAGE4_REPORT_SHA256).
M1's decoder is B0's, so M1's rows also cover the control's decoder. Palettization and W8A8 are not measured.
Sets: "dev", the five stage 1 dev sets (H4 for the rule, YODAS reported); "domain", the dev-medical and dev-technical
parts with critical-term recall. The production run (full_plan) scores the three variants on both groups.

Rule (preregistered, unchanged from stage 4; keab_step0.rule and keab_step0.choose): an INT8 scope passes on a model
if its H4 cost (quantized minus unquantized, points) is at most 0.03, the upper bound of its 90% paired
cluster-bootstrap interval at most 0.06, and no H4 set is worse by more than 0.10. The largest scope that passes on
M1 and on B0' is chosen. Reported with it: every variant's dev and domain WER, term recall and token-sequence
mismatch against "none", the bundles' bytes per part, the M1 check, and "none" against the M1 check's tokens.

Outputs under $RELABEL_ROOT/keat-prime/step0/: exports/<variant>/{bundle,view}, exports/index.json,
m1-x1-check/{<set>.json, x1.json, summary.json}, scores/M1/<variant>/<set>.json (records with token IDs) and run.json,
report.json, report.md. Every scoring output records the code identity (code_identity: the git head and the
SHA-256 of every imported file under finetune/, of the x1 runtime's source trees and of the interpreter wrappers) and
the GPU share it ran under (lease_record: under MPS a compute share caps the SMs the process sees, which library
heuristics choose kernels from; the recorded x1 arm ran on the whole GPU, so the production M1 check runs at
--compute 100).
Order and gates: prepare builds "none" first and refuses every other view (also on a retry) unless the none view
reproduces M1's export (none_matches, the plan's A2). Every scored set's configuration names the x1 scorer
(scorer_identity); production scoring (run without --limit) needs the production M1 check to have passed with that
scorer (require_m1_check), and the report takes the decision only then (check_gate).
--limit N (first N utterances per set) writes under $RELABEL_ROOT/keat-prime/smoke/step0-limit<N>/ instead, never the
production path; its report is not a decision. A set already written with the same configuration is skipped. Each
GPU process holds the parakeet GPU lock for its run (exit 3 if it is not free within --lock-wait seconds); a --limit
smoke, and prepare (seconds of CUDA work per variant), may skip it with --no-gpu-lock. Every subcommand runs under
training/write_guard.py with only $RELABEL_ROOT/keat-prime, $RELABEL_ROOT/caches and $RELABEL_ROOT/tmp writable (and
the GPU lock file): nothing is written under stage 4's keab-prime/.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import keab_step0 as k0   # stage 4's step 0 driver, unchanged: its set, record, rule, summary and lock helpers
import relabel_paths
from training import keat_export as tke
from training import keat_m1 as km1

VARIANTS = ("none", "S1-int8", "S2-int8")
SCOPE_OF = {"none": "none", "S1-int8": "S1", "S2-int8": "S2"}
RULE_SCOPES = k0.RULE_SCOPES                       # ("S1", "S2")
FULL_SETS = ("dev", "domain")
H4, DEV_SETS, YODAS, DOMAIN_SETS = k0.H4, k0.DEV_SETS, k0.YODAS, k0.DOMAIN_SETS
# gpu1bit run ap-20261020r4, arm M1-x1 (actprec/parts.build("M1", 1)): its summary.json dev_wer_percent.
RECORDED_RUN, RECORDED_ARM = "ap-20261020r4", "M1-x1"
RECORDED_X1 = {"librispeech_dev_clean": 1.989123690645644, "librispeech_dev_other": 4.236531594472908,
               "ami_dev": 11.732329391802322, "voxpopuli_dev": 6.008925725215174, "yodas_dev": 6.583433742661573}
RECORDED_X1_H4 = sum(RECORDED_X1[n] for n in H4) / len(H4)
CHECK_POINTS = 0.02
# $RELABEL_ROOT/keab-prime/step0/report.json, stage 4's step 0 decision (S2 chosen, 2026-10-31): B0''s rows.
STAGE4_REPORT_SHA256 = "383d6b5a1e76f09fb4d0ffae3550049ef5380271cc2e2c8133d2ea680c65ba4e"
EXIT_LOCK_BUSY, EXIT_CHECK = 3, 81
FINETUNE = Path(__file__).resolve().parents[1]
X1_TREES = ("parakeet-binary/gpu1bit", "parakeet-ternary/inference")
X1_TREE_SUFFIXES = (".py", ".json")
WRAPPERS = ("relabel/python", "relabel/python-gpu1bit", "parakeet-binary/gpu1bit/shared/python",
            "parakeet-ternary/python", "parakeet-binary/python")
OWN_FILES = ("relabel/keat_step0.py", "relabel/training/keat_deployed.py", "relabel/training/keat_export.py",
             "relabel/training/keat_eightbit.py", "relabel/training/keat_m1.py", "relabel/training/keat_launch.sh",
             "relabel/training/deployed.py", "relabel/training/eightbit.py", "relabel/training/keab_export.py",
             "relabel/keab_step0.py", "relabel/training/write_guard.py")
FULL_NOTE = "Full run: none, S1-int8 and S2-int8 on the dev and domain sets; the rule uses H4 only."
rule = k0.rule
choose = k0.choose
log = k0.log
read_json, write_json = k0.read_json, k0.write_json


# --- locations ---------------------------------------------------------------------------------------

def step0_root() -> Path:
    return tke.keat_root() / "step0"


def out_root(limit: int | None) -> Path:
    """The production root, or the smoke root of a --limit run (never inside the production one)."""
    if limit is None:
        return step0_root()
    if limit <= 0:
        raise ValueError("--limit must be positive")
    out = tke.keat_root() / "smoke" / f"step0-limit{limit}"
    assert not out.is_relative_to(step0_root())
    return out


def exports_root() -> Path:
    return step0_root() / "exports"


def check_variant(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"M1's variants are {VARIANTS}, got {variant!r}")
    return SCOPE_OF[variant]


def bundle_dir(variant: str) -> Path:
    check_variant(variant)
    return exports_root() / variant / "bundle"


def view_dir(variant: str) -> Path:
    check_variant(variant)
    return exports_root() / variant / "view"


def scores_dir(variant: str, limit: int | None) -> Path:
    check_variant(variant)
    return out_root(limit) / "scores" / "M1" / variant


def recorded_summary_path() -> Path:
    return relabel_paths.PARAKEET_BINARY / "gpu1bit" / "H" / "fulldev" / RECORDED_RUN / RECORDED_ARM / "summary.json"


def stage4_report_path() -> Path:
    return relabel_paths.ARTIFACTS / "keab-prime" / "step0" / "report.json"


def allowed_roots() -> list[Path]:
    """The write guard's roots: this stage's root and the caches and temporary directory relabel_boot.py set."""
    return [tke.keat_root(), relabel_paths.ARTIFACTS / "caches", relabel_paths.ARTIFACTS / "tmp"]


def weights_of(variant: str) -> tuple[Path, dict]:
    """(the export directory scored for a variant, its record): M1's own export for "none", else the variant's view."""
    check_variant(variant)
    if variant == "none":
        return km1.m1_export(), {"export": str(km1.m1_export()), "sha256": km1.M1_EXPORT_SHA256}
    vdir = view_dir(variant)
    return vdir, {"view": str(vdir)}


def checked_weights(variant: str) -> tuple[Path, dict]:
    """weights_of with the export's manifest read: M1's pinned SHA-256 for "none", the view's own (and its bundle's)
    otherwise; SystemExit if missing or different."""
    wdir, weights = weights_of(variant)
    manifest_path = wdir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{wdir}: no export (run keat_step0.py prepare first)")
    manifest = read_json(manifest_path)
    if variant == "none":
        if manifest["sha256"] != km1.M1_EXPORT_SHA256:
            raise SystemExit(f"{wdir} is not M1's export {km1.M1_EXPORT_SHA256}")
        return wdir, weights
    keat = (manifest.get("extra") or {}).get("keat") or {}
    if keat.get("scope") != SCOPE_OF[variant]:
        raise SystemExit(f"{wdir}: a view of scope {keat.get('scope')}, not {SCOPE_OF[variant]}")
    return wdir, {**weights, "view_sha256": manifest["sha256"], "bundle_sha256": keat.get("bundle_sha256")}


# --- the code identity ---------------------------------------------------------------------------------

def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(["git", "--no-optional-locks", "-C", str(FINETUNE), *args], capture_output=True,
                              text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def code_identity() -> dict:
    """The git head, whether tracked files under finetune/ differ from it (untracked machine-local files such as
    private/local.env do not count; imported ones are hashed below), and the SHA-256 of this stage's files
    (OWN_FILES), of every file under finetune/ imported by this process, of every .py and .json file of the x1
    runtime's source trees and of the interpreter wrappers."""
    rels = {r for r in WRAPPERS + OWN_FILES if (FINETUNE / r).is_file()}
    for tree in X1_TREES:
        for p in (FINETUNE / tree).rglob("*"):
            rel = p.relative_to(FINETUNE)
            if p.is_file() and p.suffix in X1_TREE_SUFFIXES and "__pycache__" not in rel.parts:
                rels.add(rel.as_posix())
    for module in list(sys.modules.values()):
        f = getattr(module, "__file__", None)
        if f:
            p = Path(f).resolve()
            if p.is_relative_to(FINETUNE) and p.is_file():
                rels.add(p.relative_to(FINETUNE).as_posix())
    files = {r: k0.sha256_file(FINETUNE / r) for r in sorted(rels)}
    status = _git("status", "--porcelain", "--untracked-files=no", "--", ".")
    return {"sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "git_head": _git("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status),
            "files": files}


SCORER_FILES = ("relabel/training/keat_deployed.py", "relabel/training/deployed.py", "relabel/keab_step0.py",
                "relabel/relabel_boot.py", "relabel/relabel_paths.py", "relabel/python", "relabel/python-gpu1bit",
                "parakeet-ternary/evaluate.py", "parakeet-ternary/export.py", "parakeet-ternary/quant.py",
                "parakeet-ternary/paths.py", "parakeet-ternary/python", "parakeet-binary/python",
                "whisper-ternary/wer.py", "whisper-ternary/quant.py", "stt/models/whisper-tiny.en/normalizer.json")
_SCORER: dict = {}


def scorer_identity() -> dict:
    """The x1 scorer the M1 check validates: SCORER_FILES (the loader, stage 4's per-utterance procedure, the dev-set
    record order, the evaluator, the normalizer data and the interpreter wrappers) and every .py and .json file of the
    x1 runtime's source trees, hashed once per process. Every scored set's configuration names it, so a set scored by
    another scorer is scored again, and production scoring and the step 0 decision need the M1 check of this one
    (require_m1_check, report). This driver's own file is not part of it (its report code does not change a score);
    it is in code_identity."""
    if not _SCORER:
        rels = set(SCORER_FILES)
        for tree in X1_TREES:
            for p in (FINETUNE / tree).rglob("*"):
                rel = p.relative_to(FINETUNE)
                if p.is_file() and p.suffix in X1_TREE_SUFFIXES and "__pycache__" not in rel.parts:
                    rels.add(rel.as_posix())
        files = {r: k0.sha256_file(FINETUNE / r) for r in sorted(rels)}
        _SCORER.update(sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), files=files)
    return dict(_SCORER)


def code_record(identity: dict) -> dict:
    return {k: identity[k] for k in ("sha256", "git_head", "git_dirty")}


LEASE_PREFIXES = ("AGENT_JOBS_", "CUDA_MPS_")


def lease_record() -> dict:
    """The GPU share this process scores under: the job's lease environment (its compute share and VRAM fraction;
    under MPS a share caps the SMs the process sees, and library heuristics choose kernels from that count) and the
    device's SM count as this process sees it. The recorded x1 arm ran on the whole GPU."""
    import torch
    env = {k: v for k, v in sorted(os.environ.items()) if k.startswith(LEASE_PREFIXES) or k == "CUDA_VISIBLE_DEVICES"}
    out = {"env": env}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        out.update(device=props.name, sm_count=props.multi_processor_count)
    return out


# --- prepare: M1's bundles and views -------------------------------------------------------------------------

class M1Models:
    """Fresh copies of M1's training model: the pinned base with the 264 ternary modules (quant.quantize_parakeet) and
    M1's latent FP32 weights (training/keat_m1.py's init check: step 250,000, weight fraction 1), on the CPU in eval
    mode. The checkpoint and the pinned .nemo are read once; each call builds the model from the base's config and
    tokenizer (NeMo models do not deep-copy) and loads the same state. `facts` records the checkpoint."""

    def __init__(self):
        import tempfile

        import train  # noqa: F401  finetune/parakeet-ternary first (CUDA/numba configuration)
        import export as ternary_export
        import quant
        from omegaconf import OmegaConf
        state, self.facts = km1.load_m1_state()
        train.verify_base()
        base = train.load_pretrained("cpu")
        self.config = OmegaConf.to_container(base.cfg, resolve=True)
        relabel_paths.require_mount()
        tmp = relabel_paths.ARTIFACTS / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        self.tokenizer = Path(tempfile.mkdtemp(prefix="keat-tokenizer-", dir=tmp))
        ternary_export._extract_tokenizer(base, self.tokenizer / "tokenizer")
        quant.quantize_parakeet(base)
        base.load_state_dict(state, strict=True)
        del state
        self.state = {k: v.detach().clone() for k, v in base.state_dict().items()}
        del base

    def __call__(self):
        import quant
        from training import keab_export
        model = keab_export._build_model(self.config, self.tokenizer / "tokenizer")
        quant.quantize_parakeet(model)
        model.load_state_dict(self.state, strict=True)
        quant.set_weight_fraction(model, 1.0)
        return model.eval()

    def close(self) -> None:
        import shutil
        shutil.rmtree(self.tokenizer, ignore_errors=True)


def none_matches(index: dict, ckpt: dict) -> bool:
    """The A2 check holds: the "none" view of this checkpoint is on disk and its export.safetensors is M1's."""
    entry = index["variants"].get("none") or {}
    manifest = view_dir("none") / "manifest.json"
    return (entry.get("checkpoint_sha256") == ckpt["sha256"] and entry.get("matches_m1_export") is True
            and entry["view"]["sha256"] == km1.M1_EXPORT_SHA256 and manifest.exists()
            and read_json(manifest)["sha256"] == km1.M1_EXPORT_SHA256)


def prepare(args) -> int:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("prepare computes M1's ternary scales on CUDA, the device its export was quantized on")
    index_path = exports_root() / "index.json"
    index = read_json(index_path) if index_path.exists() else {"variants": {}}
    models = M1Models()
    try:
        return _prepare(args, models, index, index_path)
    finally:
        models.close()


def _prepare(args, models, index: dict, index_path: Path) -> int:
    import torch
    from training import keat_eightbit as teb
    ckpt = models.facts
    index.update(checkpoint=ckpt, m1_export={"path": str(km1.m1_export()), "sha256": km1.M1_EXPORT_SHA256})
    for variant in sorted(set(args.variants), key=VARIANTS.index):         # "none" (the A2 check) first
        scope = SCOPE_OF[variant]
        vdir, bdir = view_dir(variant), bundle_dir(variant)
        entry = index["variants"].get(variant)
        if variant != "none" and not none_matches(index, ckpt):
            log(f"{variant}: refused, the none view does not reproduce M1's export (prepare --variants none first)")
            return EXIT_CHECK
        if entry and entry.get("checkpoint_sha256") == ckpt["sha256"] and (vdir / "manifest.json").exists() \
                and read_json(vdir / "manifest.json")["sha256"] == entry["view"]["sha256"] and not args.force \
                and (variant != "none" or none_matches(index, ckpt)):
            log(f"{variant}: view exists ({entry['view']['sha256'][:12]}), skipping")
            continue
        t = time.perf_counter()
        extra = {"stage": "keat-prime step 0", "variant": variant, "checkpoint": ckpt}
        lock = (contextlib.nullcontext() if args.no_gpu_lock
                else k0.waiting_lock(f"keat-step0 prepare {variant}", args.lock_wait))
        with lock:
            model = models().to("cuda")
            form = teb.apply_keat_scope(model, scope)
            bundle = tke.write_bundle(model, bdir, extra=extra)
            del model
            torch.cuda.empty_cache()
        view = tke.derive_view(bdir, vdir, extra={"variant": variant})
        entry = {"scope": scope, "checkpoint_sha256": ckpt["sha256"], "form": form,
                 "bundle": {"path": str(bdir), "sha256": bundle["sha256"], "bytes": bundle["bytes"],
                            "tensor_mb": tke.tensor_megabytes(bundle)},
                 "view": {"path": str(vdir), "sha256": view["sha256"], "bytes": view["bytes"],
                          "tensor_mb": tke.megabytes(tke.view_tensor_bytes(view))},
                 "code": code_record(code_identity())}
        if variant == "none":
            entry["matches_m1_export"] = view["sha256"] == km1.M1_EXPORT_SHA256
        index["variants"][variant] = entry
        write_json(index_path, index)
        if variant == "none" and not entry["matches_m1_export"]:
            log(f"none: the view's export.safetensors is {view['sha256']}, not M1's {km1.M1_EXPORT_SHA256}")
            return EXIT_CHECK
        log(f"{variant}: view {view['sha256'][:12]}, bundle {entry['bundle']['tensor_mb']:.3f} MB "
            f"({time.perf_counter() - t:.0f} s)")
    return 0


# --- scoring ---------------------------------------------------------------------------------------------

def set_config(variant: str, name: str, group: str, limit: int | None, weights: dict) -> dict:
    return {"model": "M1", "variant": variant, "set": name, "group": group, "limit": limit, "weights": weights,
            "manifest_sha256": k0.sha256_file(k0.manifest_of(name, group)),
            "scorer": {"kind": "deployed x1", "loader": "training/keat_deployed.load_x1", "batch": 1,
                       "code_sha256": scorer_identity()["sha256"]}}


def require_x1_process() -> None:
    from training import keat_deployed as tdep
    tdep.require_wrapper()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        raise SystemExit("M1 is scored as the recorded x1 arm ran: unset CUBLAS_WORKSPACE_CONFIG")


def _score(model, out: Path, todo: list[str], groups: dict, configs: dict, limit: int | None, identity: str,
           code: dict, lease: dict) -> None:
    from training import keat_deployed as tdep
    for name in todo:
        result = tdep.score_records(model, k0.deployed_records(name, groups[name], limit), name, identity)
        k0.add_term_recall(result, groups[name])
        result.update(config=configs[name], model="M1", variant=configs[name]["variant"], group=groups[name],
                      code=code, lease=lease)
        write_json(out / f"{name}.json", result)
        tr = result.get("term_recall")
        log(f"M1 {configs[name]['variant']} {name}: n={result['utterances']} WER {result['wer_percent']:.4f}"
            + (f" term recall {tr['recall']:.4f}" if tr and tr["recall"] is not None else "")
            + f" ({result['wall_s']:.0f} s)")


def _scored(out: Path, configs: dict, groups: dict, limit: int | None, wdir: Path, label: str, args) -> dict | None:
    """Score the pending sets of `configs` on the export at wdir (x1) under the GPU lock; the run's facts, or None if
    every set was already scored."""
    import torch
    from training import keat_deployed as tdep
    todo = k0.pending(out, configs)
    if not todo:
        return None
    identity = hashlib.sha256(json.dumps(configs[todo[0]]["weights"], sort_keys=True).encode()).hexdigest()
    with k0.gpu_lock(args, label):
        k0.vram_lease()
        model = tdep.load_x1(wdir)
        try:
            code, lease = code_identity(), lease_record()
            _score(model, out, todo, groups, configs, limit, identity, code_record(code), lease)
            facts = {"x1": tdep.info(model), "lease": lease,
                     "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2**30}
        finally:
            tdep.close(model)
    facts["code"] = code_identity()          # after scoring: every module the run imported
    return facts


def m1_check_summary() -> Path:
    return step0_root() / "m1-x1-check" / "summary.json"


def require_m1_check(scorer_sha256: str) -> dict:
    """The production M1 check's summary; SystemExit unless it passed on full dev with this scorer ("Before anything
    else is scored, M1's export is re-scored with this loader")."""
    path = m1_check_summary()
    if not path.exists():
        raise SystemExit(f"no M1 check ({path}): run keat_step0.py m1-check first")
    summary = read_json(path)
    if summary.get("pass") is not True or summary.get("limit") is not None:
        raise SystemExit(f"{path}: the M1 check did not pass on full dev (pass {summary.get('pass')}, "
                         f"limit {summary.get('limit')})")
    if summary.get("scorer_sha256") != scorer_sha256:
        raise SystemExit(f"{path}: the M1 check validated scorer {summary.get('scorer_sha256')}, not this one "
                         f"({scorer_sha256}); run keat_step0.py m1-check again")
    return summary


def run(args) -> int:
    import score_teacher
    check_variant(args.variant)
    relabel_paths.require_mount()
    require_x1_process()
    if args.limit is None:
        require_m1_check(scorer_identity()["sha256"])
    wdir, weights = checked_weights(args.variant)
    out = scores_dir(args.variant, args.limit)
    sets = k0.expand_sets(args.sets)
    configs = {n: set_config(args.variant, n, g, args.limit, weights) for n, g in sets}
    try:
        facts = _scored(out, configs, dict(sets), args.limit, wdir, f"keat-step0 M1 {args.variant}", args)
    except score_teacher.LockBusy as busy:
        print(f"refused: {busy}", file=sys.stderr)
        return EXIT_LOCK_BUSY
    if facts is None:
        log(f"M1 {args.variant}: every set already scored")
        return 0
    write_json(out / "run.json", {"model": "M1", "variant": args.variant, "limit": args.limit, "weights": weights,
                                  "sets": [n for n, _ in sets], **facts,
                                  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return 0


# --- the M1 x1 check -------------------------------------------------------------------------------------

def check_against_recorded(per_set: dict[str, float]) -> dict:
    diffs = {n: per_set[n] - RECORDED_X1[n] for n in DEV_SETS}
    h4 = sum(per_set[n] for n in H4) / len(H4)
    return {"recorded": {"run": RECORDED_RUN, "arm": RECORDED_ARM, "wer_percent": RECORDED_X1, "h4": RECORDED_X1_H4},
            "wer_percent": per_set, "h4": h4, "difference_points": diffs, "h4_difference_points": h4 - RECORDED_X1_H4,
            "limit_points_per_set": CHECK_POINTS, "pass": all(abs(d) <= CHECK_POINTS for d in diffs.values())}


def m1_check(args) -> int:
    import score_teacher
    require_x1_process()
    export, weights = checked_weights("none")
    out = out_root(args.limit) / "m1-x1-check"
    configs = {n: set_config("none", n, "dev", args.limit, {**weights, "check": "m1-x1"}) for n in DEV_SETS}
    try:
        facts = _scored(out, configs, dict.fromkeys(DEV_SETS, "dev"), args.limit, export, "keat-step0 m1-check",
                        args)
    except score_teacher.LockBusy as busy:
        print(f"refused: {busy}", file=sys.stderr)
        return EXIT_LOCK_BUSY
    if facts is not None:
        write_json(out / "x1.json", {**facts, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    rows = {n: read_json(out / f"{n}.json") for n in DEV_SETS}
    summary = check_against_recorded({n: rows[n]["wer_percent"] for n in DEV_SETS})
    scorers = {(rows[n].get("config") or {}).get("scorer", {}).get("code_sha256") for n in DEV_SETS}
    codes = {json.dumps(rows[n].get("code"), sort_keys=True) for n in DEV_SETS}
    leases = {json.dumps(rows[n].get("lease"), sort_keys=True) for n in DEV_SETS}
    summary.update(limit=args.limit, code=[json.loads(c) for c in sorted(codes)],
                   lease=[json.loads(x) for x in sorted(leases)],
                   structurally_valid=all(rows[n]["structurally_valid"] for n in DEV_SETS),
                   utterances={n: rows[n]["utterances"] for n in DEV_SETS},
                   empty_hypotheses={n: rows[n]["empty_hypotheses"] for n in DEV_SETS})
    summary["scorer_sha256"] = scorers.pop() if len(scorers) == 1 else None
    if not summary["structurally_valid"] or summary["scorer_sha256"] is None:
        summary["pass"] = False
    if args.limit is not None:
        summary["pass"] = None
        summary["note"] = "a --limit smoke: not the check"
    write_json(out / "summary.json", summary)
    log(f"m1-check: {json.dumps(summary['difference_points'])} H4 {summary['h4']:.4f} pass={summary['pass']}")
    return 0 if summary["pass"] in (True, None) else EXIT_CHECK


# --- the report -------------------------------------------------------------------------------------------

def stage4_b0p(path: Path | None = None) -> dict:
    """B0''s step 0 rows from stage 4's report (read only): SystemExit unless the file's SHA-256 is
    STAGE4_REPORT_SHA256 and it is a decision (full sets) with both rule scopes."""
    path = Path(path or stage4_report_path())
    digest = k0.sha256_file(path)
    if digest != STAGE4_REPORT_SHA256:
        raise SystemExit(f"{path}: SHA-256 {digest} is not stage 4's recorded step 0 report {STAGE4_REPORT_SHA256}")
    r = read_json(path)
    passes = (r.get("passes") or {}).get("B0p") or {}
    if r.get("limit") is not None or r.get("decision") is not True or set(passes) != set(RULE_SCOPES):
        raise SystemExit(f"{path}: not stage 4's full step 0 decision with B0''s {RULE_SCOPES} rows")
    b0p = r["models"]["B0p"]
    rows = {v: {"h4_delta": e.get("h4_delta"), "h4_ci": e.get("h4_ci"), "rule": e.get("rule")}
            for v, e in b0p["variants"].items() if v in ("S1-int8", "S2-int8")}
    return {"path": str(path), "sha256": digest, "passes": {s: passes[s] is True for s in RULE_SCOPES},
            "rows": rows, "none_h4": (b0p.get("none") or {}).get("h4")}


def load_variant(variant: str, limit: int | None, names: list[str]) -> dict[str, dict] | None:
    d = scores_dir(variant, limit)
    out = {}
    for n in names:
        p = d / f"{n}.json"
        if not p.exists():
            return None
        out[n] = read_json(p)
    return out


def _wer(res: dict) -> float:
    return res["wer_percent"] if "wer_percent" in res else 100 * res["wer"]["wer"]


def crosscheck_none(limit: int | None) -> dict:
    """"none" (M1's export, scored by `run`) against the M1 check's tokens of the same export, per dev set."""
    d, chk = scores_dir("none", limit), out_root(limit) / "m1-x1-check"
    sets = {}
    for name in DEV_SETS:
        if (d / f"{name}.json").exists() and (chk / f"{name}.json").exists():
            ref = {r["id"]: r["tokens"] for r in read_json(chk / f"{name}.json")["records"]}
            recs = read_json(d / f"{name}.json")["records"]
            sets[name] = {"utterances": len(recs), "identical_tokens": sum(ref.get(r["id"]) == r["tokens"] for r in recs)}
    return sets


def check_gate(base: dict | None, limit: int | None) -> dict:
    """Whether the step 0 decision may be taken: the production M1 check passed on full dev, and every M1 scoring the
    report reads (none, S1-int8, S2-int8, at this limit) names the scorer it validated."""
    path = m1_check_summary()
    if not path.exists():
        return {"pass": False, "reason": "no M1 check"}
    summary = read_json(path)
    if summary.get("pass") is not True or summary.get("limit") is not None:
        return {"pass": False, "reason": f"the M1 check did not pass on full dev (pass {summary.get('pass')})"}
    names = list(base or [])
    scorers = set()
    for variant in VARIANTS:
        for res in (load_variant(variant, limit, names) or {}).values():
            scorers.add(((res.get("config") or {}).get("scorer") or {}).get("code_sha256"))
    if scorers != {summary.get("scorer_sha256")}:
        return {"pass": False, "reason": f"M1 scored by {sorted(map(str, scorers))}, the M1 check validated "
                                         f"{summary.get('scorer_sha256')}"}
    return {"pass": True, "reason": None, "scorer_sha256": summary["scorer_sha256"]}


def report(args) -> int:
    import devsets
    domain_parts = {s: list(devsets.SETS[s]) for s in DOMAIN_SETS}
    names_dev = list(DEV_SETS)
    names_all = names_dev + [p for s in DOMAIN_SETS for p in domain_parts[s]]
    b0p = stage4_b0p()
    out = {"limit": args.limit, "decision": args.limit is None, "rule_scopes": list(RULE_SCOPES),
           "full_run": {"sets": {v: list(FULL_SETS) for v in VARIANTS}, "note": FULL_NOTE}, "models": {}, "b0p": b0p}
    m1_passes: dict[str, bool] = {}
    base = load_variant("none", args.limit, names_all) or load_variant("none", args.limit, names_dev)
    if base is None:
        out["models"]["M1"] = {"status": "unquantized scoring (none) incomplete"}
    else:
        entry = {"none": {"wer": {n: _wer(base[n]) for n in names_dev}}, "variants": {}}
        entry["none"]["h4"] = sum(entry["none"]["wer"][n] for n in H4) / len(H4)
        for variant in VARIANTS[1:]:
            res = load_variant(variant, args.limit, list(base))
            if res is None:
                entry["variants"][variant] = {"status": "incomplete"}
                continue
            entry["variants"][variant] = k0.summarize_variant("M1", variant, res, base, domain_parts)
        m1_passes = {SCOPE_OF[v]: e["rule"]["pass"] for v, e in entry["variants"].items() if "rule" in e}
        out["models"]["M1"] = entry
    passes = {"M1": m1_passes, "B0p": b0p["passes"]}
    complete = set(m1_passes) == set(RULE_SCOPES)
    out["passes"] = passes
    out["m1_check_gate"] = check_gate(base, args.limit)
    out["chosen_scope"] = choose(passes) if complete and args.limit is None and out["m1_check_gate"]["pass"] else None
    if complete and args.limit is not None:
        out["slice_rule_outcome"] = choose(passes)           # a smoke slice: never a decision
    if not complete:
        out["note"] = "the rule needs M1's S1-int8 and S2-int8 (B0''s rows are stage 4's)"
    elif not out["m1_check_gate"]["pass"] and args.limit is None:
        out["note"] = f"no decision: {out['m1_check_gate']['reason']}"
    idx = exports_root() / "index.json"
    if idx.exists():
        index = read_json(idx)
        out["exports"] = {v: {"bundle_tensor_mb": e["bundle"]["tensor_mb"], "bundle_parts": e["bundle"]["bytes"]["parts"],
                              "view_sha256": e["view"]["sha256"], "matches_m1_export": e.get("matches_m1_export")}
                          for v, e in index["variants"].items()}
    check = m1_check_summary()
    if check.exists():
        out["m1_x1_check"] = read_json(check)
    out["crosscheck_none_vs_m1_check"] = crosscheck_none(args.limit)
    root = out_root(args.limit)
    write_json(root / "report.json", out)
    text = markdown(out)
    (root / "report.md").write_text(text)
    print(text)
    return 0


def markdown(r: dict) -> str:
    lines = [f"# Stage 5 step 0: 8-bit floating-point parts of M1{'' if r['decision'] else ' (smoke, not a decision)'}",
             "", r["full_run"]["note"], ""]
    if r.get("chosen_scope") is not None:
        lines += [f"**Chosen scope: {r['chosen_scope']}** (largest INT8 scope passing on M1 and on B0').", ""]
    elif r.get("slice_rule_outcome") is not None:
        lines += [f"Rule outcome on this slice: {r['slice_rule_outcome']} (a smoke; not a decision).", ""]
    elif r.get("note"):
        lines += [r["note"], ""]
    m = r["models"].get("M1", {})
    lines += ["## M1 (deployed x1)", ""]
    if "variants" not in m:
        lines += [m.get("status", ""), ""]
    else:
        lines += ["| Variant | H4 | H4 Δ | 90% CI | worst H4 set Δ | YODAS Δ | token mismatch (dev) | rule |",
                  "|---|---:|---:|---|---:|---:|---:|---|", f"| none | {m['none']['h4']:.3f} | | | | | | |"]
        for v, e in m["variants"].items():
            if "rule" not in e:
                lines.append(f"| {v} | {e.get('status')} | | | | | | |")
                continue
            ru = e["rule"]
            lines.append(f"| {v} | {e['h4']:.3f} | {e['h4_delta']:+.3f} | [{e['h4_ci'][0]:+.3f}, {e['h4_ci'][1]:+.3f}] | "
                         f"{ru['worst_set_delta']:+.3f} ({ru['worst_set']}) | {e['delta'][YODAS]:+.3f} | "
                         f"{100 * e['token_mismatch_dev_pooled']['rate']:.2f}% | {'PASS' if ru['pass'] else 'fail'} |")
        lines.append("")
        dom = [(v, s, d) for v, e in m["variants"].items() for s, d in (e.get("domain") or {}).items()]
        if dom:
            lines += ["| Variant | Domain set | mean WER | Δ | pooled term recall Δ |", "|---|---|---:|---:|---:|"]
            for v, s, d in dom:
                tr = d.get("pooled_term_recall_delta")
                lines.append(f"| {v} | {s} | {d['mean_wer']:.3f} | {d['mean_delta']:+.3f} | "
                             + ("–" if tr is None else f"{100 * tr:+.2f}") + " |")
            lines.append("")
    b = r["b0p"]
    lines += ["## B0' (stage 4's rows, read only)", "", f"Report {b['path']} (sha {b['sha256'][:12]}).", "",
              "| Variant | H4 Δ | 90% CI | worst H4 set Δ | rule |", "|---|---:|---|---:|---|"]
    for v, e in b["rows"].items():
        ru = e.get("rule") or {}
        ci = e.get("h4_ci") or [float("nan"), float("nan")]
        lines.append(f"| {v} | {e['h4_delta']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                     f"{ru.get('worst_set_delta', float('nan')):+.3f} | {'PASS' if ru.get('pass') else 'fail'} |")
    lines.append("")
    if r.get("exports"):
        lines += ["## M1 bundles", "", "| Variant | bundle MB (tensors) | view = M1's export |", "|---|---:|---|"]
        for v, e in r["exports"].items():
            same = e.get("matches_m1_export")
            lines.append(f"| {v} | {e['bundle_tensor_mb']:.3f} | {'' if same is None else same} |")
        lines.append("")
    if r.get("m1_x1_check"):
        c = r["m1_x1_check"]
        lines += [f"## M1 x1 check (against {c['recorded']['run']} {c['recorded']['arm']})", "",
                  "| Set | WER | recorded | Δ |", "|---|---:|---:|---:|"]
        for n in DEV_SETS:
            lines.append(f"| {n} | {c['wer_percent'][n]:.4f} | {c['recorded']['wer_percent'][n]:.4f} | "
                         f"{c['difference_points'][n]:+.4f} |")
        lines += ["", f"H4 {c['h4']:.4f} (recorded {c['recorded']['h4']:.4f}); pass: {c['pass']}", ""]
        for lease in c.get("lease") or []:
            if lease:
                lines += [f"Scored on {lease.get('sm_count')} SMs; lease environment {json.dumps(lease.get('env'))}", ""]
    sets = r.get("crosscheck_none_vs_m1_check") or {}
    if sets:
        lines += ["Crosscheck none vs the M1 check: " + ", ".join(f"{n} {e['identical_tokens']}/{e['utterances']}"
                                                                for n, e in sets.items()), ""]
    return "\n".join(lines) + "\n"


def status(args) -> int:
    idx = exports_root() / "index.json"
    print("exports:", sorted(read_json(idx)["variants"]) if idx.exists() else "none")
    for variant in VARIANTS:
        d = scores_dir(variant, None)
        done = sorted(p.stem for p in d.glob("*.json") if p.stem != "run") if d.exists() else []
        print(f"M1 {variant}: {len(done)} sets {done}")
    check = step0_root() / "m1-x1-check" / "summary.json"
    print("m1-check:", read_json(check)["pass"] if check.exists() else "not run")
    return 0


# --- CLI --------------------------------------------------------------------------------------------------

def full_plan() -> list[tuple[str, tuple[str, ...]]]:
    """(variant, set groups) of the production run, in order."""
    return [(v, FULL_SETS) for v in VARIANTS]


def plan(args) -> int:
    """One line per production job: VARIANT GROUP... (training/keat_launch.sh full reads it)."""
    for v, groups in full_plan():
        print(v, *groups)
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("prepare")
    s.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    s.add_argument("--force", action="store_true")
    s.add_argument("--lock-wait", type=float, default=1800.0)
    s.add_argument("--no-gpu-lock", action="store_true",
                   help="skip the parakeet GPU lock (prepare's CUDA work is seconds per variant)")
    s = sub.add_parser("m1-check")
    s.add_argument("--limit", type=int)
    s.add_argument("--lock-wait", type=float, default=1800.0)
    s.add_argument("--no-gpu-lock", action="store_true", help="--limit smoke runs only")
    s = sub.add_parser("run")
    s.add_argument("--variant", required=True, choices=VARIANTS)
    s.add_argument("--sets", nargs="+", default=list(FULL_SETS), choices=FULL_SETS)
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
    if args.action in ("run", "m1-check") and args.no_gpu_lock and args.limit is None:
        p.error("--no-gpu-lock is for --limit smoke runs only")
    return args


ACTIONS = {"plan": plan, "prepare": prepare, "m1-check": m1_check, "run": run, "report": report, "status": status}


def main(argv=None) -> int:
    """Every subcommand runs under training/write_guard.py with allowed_roots() writable (and the parakeet GPU lock
    file, written by design)."""
    from training import write_guard
    args = parse_args(argv)
    relabel_paths.require_mount()
    with write_guard.active(allowed_roots(), files=[relabel_paths.GPU_LOCK]):
        return ACTIONS[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
