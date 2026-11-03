"""Test-set scoring of Kea-B′ (stage 4's run b100-main, its scoring view export/view), once per mode (keab-prime-test
plan v1.2).

    ./python keab_test.py --precheck                     CPU, read-only; writes keab-prime/test/precheck.json
    ./python keab_test.py --scorer-hashes                print the scorer files' SHA-256 (pinned after code review)
    ./python keab_test.py --dev-check fp [--limit N]     MB1's export, then Kea-B′'s view, through the K-fp path on dev
    env -u CUBLAS_WORKSPACE_CONFIG ./python-gpu1bit keab_test.py --dev-check x1 [--limit N]
    ./python keab_test.py --test fp                      the K-fp test run (resumes an open attempt)
    env -u CUBLAS_WORKSPACE_CONFIG ./python-gpu1bit keab_test.py --test x1
    ./python keab_test.py --summarize fp|x1              rebuild summary.json of a fully decoded attempt; no decoding

Modes (plan "What is scored"):
  x1  deployed: training/deployed.load_x1(view) (the i8x1 kernels), deployed.score_records per utterance (batch 1,
      gpu1bit fulldev.process_one) over evaluate.read_manifest(manifest) in manifest order; exactly the stage 4 gate's
      mode, under ./python-gpu1bit (deployed.require_wrapper) with CUBLAS_WORKSPACE_CONFIG unset
      (train_keab.deployed_environment) and the relabel cubin cache (load_x1).
  fp  MB1's test procedure: binary_export.load_export(view) on CUDA, the greedy TDT decoder pinned to
      binary_train's DECODER_GRAPHS_MODE (no_while_loops) and checked before and after every set,
      evaluate.evaluate_set at batch 64 and 1200 s under evaluate.py's strict_fp32 and inference_settings, no extra
      cuDNN flags; under ./python (no NVRTC on the library path, as MB1-test's interpreter), with the torch, NeMo
      and CUDA versions MB1-test recorded.

Every pinned input is a constant in keab_test_paths. --dev-check and --test re-run the precheck in-process and refuse
unless it passes. Refusals that need no GPU come before any model is loaded; the shared GPU lock (paths.gpu_lock's
file, through keab_step0.waiting_lock) is held while a model exists.

Dev cross-check (gate.json last, in keab-prime/test/<mode>-dev-check/): x1 must reproduce stage 4's recorded x1 dev
WER (scores/b100-main/x1) on the same dev records within 0.02 points on the five-set mean and 0.10 points on every
set (token-sequence identity reported); fp validates the path on MB1 first (its recorded dev check
parakeet-binary/eval/MB1-dev-check, same tolerances), then records Kea-B′ in K-fp mode beside its PyTorch fake-quant
dev numbers. Any structurally invalid row (finite false or a decode_error) fails the gate.

Test runs: outputs fixed in code (keab-prime/test/<mode>/), an append-only attempt record per mode
(keab-prime/test/<mode>.attempts.jsonl): "started" (with the launch gate's SHA-256), "resumed" on every later
launch, "set_done" with the per-set file's SHA-256 (the file written and fsynced first), "complete" after
summary.json. --test continues the open attempt: a set with a set_done record whose file still matches is never
decoded again (a mismatch refuses); a set without one is decoded from the start. --test refuses once the mode is
complete, and once all eight sets are done (then only --summarize). --test and --summarize take the mode's attempt
lock (--test waits for it, --summarize refuses if it is held); lock order: the attempt lock, then the GPU lock.
"""
from __future__ import annotations

import train_keab as tk  # noqa: I001  first: train.py (CUDA/numba settings); records CUBLAS_WORKSPACE_CONFIG at start

import argparse
import contextlib
import csv
import fcntl
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import transformers.models.whisper.english_normalizer  # noqa: F401  the pinned normalizer (precheck item 3)

import ami_sdm
import binary_export  # noqa: E402  finetune/parakeet-binary (on sys.path through training.eightbit)
import binary_train  # noqa: E402
import evaluate
import keab_step0 as k0
import keab_test_paths as tp
import paths
import relabel_paths
import score_teacher
import testsets  # finetune/parakeet-ternary (the ESB parquet files of the Common Voice check)
from corpora import registry
from labeling import filters
from training import deployed
from training import eightbit  # noqa: F401  (pinned scorer module)
from training import keab_export  # noqa: F401  (pinned scorer module; load_x1's cubin cache root)
import gpu1bit.actprec.parts  # noqa: E402, F401  load_x1's modules, imported here so precheck item 3 sees them
import gpu1bit.int8_mma.cuda  # noqa: E402, F401
import gpu1bit.runtime  # noqa: E402, F401
import gpu1bit.shared.fulldev  # noqa: E402, F401

DEVICE = "cuda"
STATES = ("started", "resumed", "set_done", "complete")
ACTIVE_JOB_STATES = ("starting", "queued", "running")
LOCK_WAIT_S = 21600.0
EXIT_LOCK_BUSY = 3
# Environment variables that can change numerics or the decoder backend, recorded in every output (MB1's list and the
# gpu1bit wrapper's marker).
SETTINGS_ENV = ("CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_HOME", "NUMBA_FORCE_CUDA_CC",
                "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                "OMP_NUM_THREADS", "GPU1BIT_WRAPPER")
csv.field_size_limit(1 << 30)


def refuse(message: str):
    raise SystemExit(f"keab_test: refused: {message}")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(path: Path) -> str | None:
    """SHA-256 of a file, None if it is missing or unreadable."""
    try:
        with open(path, "rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError:
        return None


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, data) -> None:
    """Atomic and durable: a temporary file written and fsynced, renamed over the path, the directory fsynced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as handle:
        handle.write(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# ----------------------------------------------------------------------------- recorded settings

def pinned_keys() -> list[str]:
    return sorted(tp.SCORER_HASHES) if tp.SCORER_HASHES else tp.scorer_keys()


def scorer_hashes(keys=None) -> dict[str, str | None]:
    return {key: digest(tp.scorer_path(key)) for key in (pinned_keys() if keys is None else keys)}


def scorer_record() -> dict:
    """The scoring sources' current hashes, keab_test_paths.py's own and the scorer modules' files (every output)."""
    return {"hashes": scorer_hashes(), "keab_test_paths_sha256": digest(Path(tp.__file__)),
            "modules": {name: str(module_file(name)) for name in sorted(tp.SCORER_MODULES)}}


def module_file(name: str) -> Path | None:
    if name == "keab_test":
        return Path(__file__).resolve()
    f = getattr(sys.modules.get(name), "__file__", None)
    return None if f is None else Path(f).resolve()


def numerical_settings(mode: str) -> dict:
    """cuDNN flags, the relevant environment, the interpreter and library versions of this process (x1: also the
    gpu1bit wrapper facts). keab_test sets no cuDNN flags of its own."""
    import nemo
    import transformers
    try:
        cudnn = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001  no cuDNN library: recorded as None
        cudnn = None
    out = {"mode": mode, "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
           "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
           "env": {key: os.environ.get(key) for key in SETTINGS_ENV}, "executable": sys.executable,
           "versions": {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "cudnn": cudnn,
                        "nemo": nemo.__version__, "transformers": transformers.__version__,
                        "python": sys.version.split()[0]}}
    if mode == "x1":
        out["wrapper"] = deployed.require_wrapper()
    return out


def identity() -> dict:
    """The pinned Kea-B′ identity (keab_test_paths.identity)."""
    return tp.identity()


def manifest_hashes() -> dict[str, str | None]:
    """Current SHA-256 of every dev and test manifest."""
    return {name: digest(tp.manifest_path(name)) for name in [*paths.DEV_SETS, *paths.TEST_SETS]}


def structural(records: list[dict]) -> dict:
    """Rows a decoder failure or a non-finite encoder output produced (plan "Structural validity")."""
    return {"not_finite": [r.get("id") for r in records if r.get("finite") is False],
            "decode_errors": [r.get("id") for r in records if r.get("decode_error")]}


def is_valid(s: dict) -> bool:
    return not s["not_finite"] and not s["decode_errors"]


# ----------------------------------------------------------------------------- stage 4 state

def active_stage4_jobs() -> list[str] | None:
    """Active relabel-keab-* jobs (the `job` supervisor), this process's own run excluded; None if the query failed."""
    try:
        out = subprocess.run(["job", "ls", "--json"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    try:
        runs = json.loads(out.stdout)
    except ValueError:
        return None
    own = os.environ.get("AGENT_JOBS_RUN_DIR")
    return sorted(str(r.get("name")) for r in runs
                  if str(r.get("name", "")).startswith(tp.JOB_PREFIX) and r.get("state") in ACTIVE_JOB_STATES
                  and not (own and r.get("run_dir") == own))


def score_chain_pending() -> list[str]:
    """Stage 4's chain items (train_keab.ORDER) not done."""
    return [" ".join(map(str, item)) for item in tk.ORDER if not tk.item_done(item)]


# ----------------------------------------------------------------------------- precheck

def precheck() -> dict:
    """Plan "Preconditions" items 1-7: {"pass", "checks": [{"item", "check", "ok", "detail"}], "facts"}. Reads only;
    uses no GPU. An unreadable input fails its item (it never passes silently)."""
    checks: list[dict] = []
    facts: dict = {}

    def checker(item: int):
        def check(name: str, ok, detail="") -> None:
            checks.append({"item": item, "check": name, "ok": bool(ok), "detail": str(detail)[:400]})
        return check

    for item, fn in ((1, _stage4_checks), (2, _identity_checks), (3, _code_checks)):
        try:
            fn(checker(item))
        except Exception as error:  # noqa: BLE001
            checker(item)("inputs present and readable", False, f"{type(error).__name__}: {error}")
    try:
        _reused_checks(checker(4), checker(5), checker(6))
    except Exception as error:  # noqa: BLE001
        for item in (4, 5, 6):
            checker(item)("inputs present and readable", False, f"{type(error).__name__}: {error}")
    try:
        facts["hygiene"] = _hygiene_checks(checker(7))
    except Exception as error:  # noqa: BLE001
        checker(7)("inputs present and readable", False, f"{type(error).__name__}: {error}")
    return {"pass": bool(checks) and all(c["ok"] for c in checks), "utc": utc(), "checks": checks, "facts": facts}


def _stage4_checks(check) -> None:
    """Item 1: stage 4 finished (the chain's items, its report with the gate PASS) and none of its jobs active."""
    pending = score_chain_pending()
    check("stage 4 chain items (train_keab.ORDER) all done", not pending, ", ".join(pending[:6]))
    check(f"{tp.STAGE4_RESULTS_MD.name} written by the stage 4 report", tp.STAGE4_RESULTS_MD.is_file())
    results = read_json(tp.STAGE4_RESULTS_JSON)
    gate = results.get("gate") or {}
    decision = gate.get("decision") or {}
    check("stage 4 results.json: gate complete with decision PASS",
          gate.get("status") == "complete" and decision.get("pass") is True,
          f"status {gate.get('status')!r}, pass {decision.get('pass')!r}")
    jobs = active_stage4_jobs()
    check(f"no {tp.JOB_PREFIX}* job active (query succeeded)", jobs == [],
          "job query failed" if jobs is None else ", ".join(jobs))


def _identity_checks(check) -> None:
    """Item 2: Kea-B′'s pinned files, its summary fields and stage 4's own records of the view."""
    for rel, want in tp.PINNED_FILES.items():
        got = digest(tp.RUN / rel)
        check(f"pinned SHA-256 of runs/{tp.RUN_NAME}/{rel}", got == want, got)
    entries = {p.relative_to(tp.VIEW).as_posix() for p in tp.VIEW.rglob("*")}
    check("scoring view directory file set exact", entries == tp.VIEW_ENTRIES,
          f"extra {sorted(entries - tp.VIEW_ENTRIES)}, missing {sorted(tp.VIEW_ENTRIES - entries)}")
    view = read_json(tp.VIEW / "manifest.json")
    check(f"view manifest format {tp.VIEW_FORMAT}, file {tp.VIEW_FILE}",
          view.get("format") == tp.VIEW_FORMAT and view.get("file") == tp.VIEW_FILE,
          f"{view.get('format')!r} {view.get('file')!r}")
    files = view.get("files") or {}
    want_files = {tp.VIEW_FILE: tp.VIEW_SHA256,
                  **{rel: tp.PINNED_FILES[f"export/view/{rel}"] for rel in tp.VIEW_TOKENIZER}}
    check("view manifest sha256 and files are the pinned view and tokenizer hashes",
          view.get("sha256") == tp.VIEW_SHA256 and files == want_files, view.get("sha256"))
    extra = view.get("extra") or {}
    keab = extra.get("keab") or {}
    check("view manifest records its bundle (stage 4's derive_view, checked tensor by tensor)",
          keab.get("bundle_sha256") == tp.BUNDLE_SHA256 and keab.get("bundle_format") == tp.BUNDLE_FORMAT,
          keab.get("bundle_sha256"))
    check("view manifest records the training model and run",
          extra.get("model_sha256") == tp.MODEL_SHA256 and extra.get("run") == tp.RUN_NAME,
          f"{extra.get('model_sha256')} {extra.get('run')}")
    bundle = read_json(tp.BUNDLE / "manifest.json")
    check(f"bundle manifest: format {tp.BUNDLE_FORMAT}, sha256 the pinned bundle, model and run recorded",
          bundle.get("format") == tp.BUNDLE_FORMAT and bundle.get("sha256") == tp.BUNDLE_SHA256
          and (bundle.get("extra") or {}).get("model_sha256") == tp.MODEL_SHA256
          and (bundle.get("extra") or {}).get("run") == tp.RUN_NAME, bundle.get("sha256"))
    check(f"bundle tensor bytes == {tp.BUNDLE_TENSOR_BYTES}",
          (bundle.get("bytes") or {}).get("tensor_bytes") == tp.BUNDLE_TENSOR_BYTES,
          (bundle.get("bytes") or {}).get("tensor_bytes"))
    model = read_json(tp.RUN / "model.json")
    export = model.get("export") or {}
    check("model.json: the model, bundle and view it exported",
          model.get("sha256") == tp.MODEL_SHA256 and model.get("run") == tp.RUN_NAME
          and export.get("bundle_sha256") == tp.BUNDLE_SHA256 and export.get("view_sha256") == tp.VIEW_SHA256,
          f"{model.get('sha256')} {export}")
    summary = read_json(tp.RUN / "summary.json")
    for key, want in tp.EXPECT_SUMMARY.items():
        got = summary.get(key, "<missing>")
        check(f"summary.{key} == {want!r}",
              key in summary and got == want and isinstance(got, bool) == isinstance(want, bool), repr(got))
    check("summary.config_sha256 == the pinned config.json hash", summary.get("config_sha256") == tp.CONFIG_SHA256,
          summary.get("config_sha256"))
    check("summary.keab.model.sha256 == the pinned model.pt hash",
          ((summary.get("keab") or {}).get("model") or {}).get("sha256") == tp.MODEL_SHA256)
    runtime_path = tp.SCORES_X1 / "runtime.json"
    check("scores x1 runtime.json is the pinned file", digest(runtime_path) == tp.RUNTIME_X1_SHA256,
          digest(runtime_path))
    runtime = read_json(runtime_path).get("runtime") or {}
    check("scores x1 runtime.json: the x1 runtime loaded this view (SHA-256, path, kernel, parts)",
          runtime.get("export_sha256") == tp.VIEW_SHA256 and runtime.get("export") is not None
          and Path(runtime["export"]).resolve() == tp.VIEW.resolve() and runtime.get("kernel") == tp.X1_KERNEL
          and runtime.get("parts") == tp.X1_PARTS, f"{runtime.get('export_sha256')} {runtime.get('export')}")
    for name, want in tp.DEV_X1_FILES.items():
        path = tp.SCORES_X1 / f"{name}.json"
        got = digest(path)
        check(f"recorded x1 dev score {name}.json is the pinned file", got == want, got)
        if got != want:
            continue
        doc = read_json(path)
        config = doc.get("config") or {}
        check(f"recorded x1 dev score {name}: this view, full set, current manifest, structurally valid",
              (config.get("model") or {}).get("view_sha256") == tp.VIEW_SHA256 and config.get("mode") == "x1"
              and config.get("limit") is None and config.get("manifest_sha256") == digest(tp.manifest_path(name))
              and doc.get("structurally_valid") is True and doc.get("utterances") == len(doc.get("records") or []),
              f"{config.get('mode')} {config.get('limit')} {config.get('manifest_sha256')}")
    results = read_json(tp.STAGE4_RESULTS_JSON)
    exp = (results.get("eightbit") or {}).get("keab_prime_export") or {}
    check("stage 4 report records this bundle and view",
          exp.get("bundle_sha256") == tp.BUNDLE_SHA256 and exp.get("view_sha256") == tp.VIEW_SHA256,
          f"{exp.get('bundle_sha256')} {exp.get('view_sha256')}")


def stray_x1_modules() -> list[str]:
    """Imported modules of the x1 packages (gpu1bit, inference) whose file is not a pinned scorer file."""
    pinned = {tp.scorer_path(k).resolve() for k in pinned_keys()}
    out = []
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] not in tp.X1_PACKAGES:
            continue
        f = getattr(module, "__file__", None)
        if f and Path(f).resolve() not in pinned:
            out.append(f"{name} ({f})")
    return sorted(out)


def plan_status() -> str:
    """The plan's bold Status paragraph ("**Status: ... **")."""
    text = tp.PLAN.read_text()
    start = text.find("**Status:")
    end = text.find("**", start + len("**Status:")) if start >= 0 else -1
    return text[start:end] if end > start else ""


def _code_checks(check) -> None:
    """Item 3: every pinned scorer file matches and is the one imported."""
    check("SCORER_HASHES pinned (keab_test_paths; keab_test.py --scorer-hashes after code review)",
          bool(tp.SCORER_HASHES), "not pinned yet" if not tp.SCORER_HASHES else "")
    if tp.SCORER_HASHES:
        keys = set(tp.scorer_keys())
        check("SCORER_HASHES covers exactly the scorer files", set(tp.SCORER_HASHES) == keys,
              f"unpinned {sorted(keys - set(tp.SCORER_HASHES))[:4]}, gone {sorted(set(tp.SCORER_HASHES) - keys)[:4]}")
        changed = [k for k, want in tp.SCORER_HASHES.items() if digest(tp.scorer_path(k)) != want]
        check("every scorer file matches SCORER_HASHES", not changed, ", ".join(changed[:6]))
    check("parakeet-ternary/evaluate.py is the frozen evaluator",
          digest(tp.FINETUNE / "parakeet-ternary" / "evaluate.py") == tp.EVALUATE_SHA256)
    for name, key in tp.SCORER_MODULES.items():
        got = module_file(name)
        check(f"module {name} imported from {key}", got is not None and got == tp.scorer_path(key).resolve(), got)
    stray = stray_x1_modules()
    check("every imported gpu1bit / inference module is a pinned scorer file", not stray, ", ".join(stray[:4]))
    check("evaluate.NORMALIZER_JSON is the pinned normalizer.json",
          Path(evaluate.NORMALIZER_JSON).resolve() == tp.scorer_path(tp.NORMALIZER_JSON).resolve(),
          evaluate.NORMALIZER_JSON)
    own = digest(Path(tp.__file__))
    check("keab_test_paths.py SHA-256 is in the plan's Status line", own is not None and own in plan_status(), own)


def load_arm(directory: Path) -> tuple[dict | None, dict[str, dict | None]]:
    """(summary.json, {test set: per-set output}); None where unreadable."""
    def load(path: Path):
        try:
            return read_json(path)
        except (OSError, ValueError):
            return None
    return load(directory / "summary.json"), {n: load(directory / f"{n}.json") for n in paths.TEST_SETS}


def _reused_checks(check4, check5, check6) -> None:
    """Items 4 (corpus), 5 (normalizer) and 6 (reused arms)."""
    for arm, files in tp.REUSED_FILES.items():
        changed = [f for f, want in files.items() if digest(tp.REUSED[arm] / f) != want]
        check6(f"{arm}: the pinned output files", not changed, ", ".join(changed))
    arms = {arm: load_arm(d) for arm, d in tp.REUSED.items()}
    cache: dict[str, str] = {}

    def norm(text: str) -> str:
        if text not in cache:
            cache[text] = evaluate.normalize(text)
        return cache[text]

    for name in paths.TEST_SETS:
        manifest = tp.manifest_path(name)
        current = digest(manifest)
        recorded = {arm: (docs.get(name) or {}).get("manifest_sha256") for arm, (_, docs) in arms.items()}
        check4(f"{name}: manifest SHA-256 equals B0's, M1's and MB1's records",
               current is not None and all(r == current for r in recorded.values()), f"current {current}, {recorded}")
        ids = [r["id"] for r in evaluate.read_manifest(manifest)] if current else []
        check4(f"{name}: manifest utterance IDs unique", ids and len(ids) == len(set(ids)),
               f"{len(ids) - len(set(ids))} duplicates")
        for arm, (_, docs) in arms.items():
            got = [r.get("id") for r in (docs.get(name) or {}).get("records") or []]
            check6(f"{arm} {name}: one record per manifest utterance", ids and sorted(got) == sorted(ids),
                   f"{len(got)} records, {len(ids)} in the manifest")
        records = (arms["MB1"][1].get(name) or {}).get("records") or []
        bad = [r.get("id") for r in records
               if norm(r["ref"]) != r.get("ref_norm") or norm(r["hyp"]) != r.get("hyp_norm")]
        check5(f"MB1-test {name}: the current normalizer reproduces every saved ref_norm and hyp_norm",
               records and not bad, "no records" if not records else f"{len(bad)} differ, first {bad[:3]}")
    for arm, (summary, docs) in arms.items():
        listed = set((summary or {}).get("sets") or {})
        partial = [n for n, d in docs.items() if d is None or d.get("set") != n or d.get("limit") is not None
                   or d.get("utterances") != len(d.get("records") or [])
                   or ((summary or {}).get("sets") or {}).get(n, {}).get("limit") is not None]
        check6(f"{arm}: all 8 test sets, no --limit, complete", summary is not None
               and listed == set(paths.TEST_SETS) and not partial, f"sets {sorted(listed)}, partial {partial}")
        check6(f"{arm}: the frozen evaluator's normalizer", all(d and d.get("normalizer") == evaluate.NORMALIZER
                                                                 for d in docs.values()))

    def sources(arm: str) -> list[dict]:
        summary, docs = arms[arm]
        return [(summary or {}).get("source") or {}] + [(d or {}).get("source") or {} for d in docs.values()]

    mb1 = {s.get("sha256") for s in sources("MB1")}
    check6(f"MB1-test: source SHA-256 is MB1's pinned export {tp.MB1_EXPORT_SHA256[:8]}", mb1 == {tp.MB1_EXPORT_SHA256},
           sorted(map(str, mb1)))
    m1 = {s.get("sha256") for s in sources("M1")}
    check6(f"M1-test: source SHA-256 is M1's pinned export {tp.M1_EXPORT_SHA256[:8]}", m1 == {tp.M1_EXPORT_SHA256},
           sorted(map(str, m1)))
    check6("B0: the pinned pretrained model (kind, .nemo SHA-256, revision)",
           all(s.get("kind") == "pretrained" and s.get("sha256") == tp.BASE_MODEL["nemo_sha256"]
               and s.get("base_model") == tp.BASE_MODEL for s in sources("B0")))
    decodings = {json.dumps((d or {}).get("decoding"), sort_keys=True) for d in arms["MB1"][1].values()}
    check6("MB1-test: one decoding record at batch 64 and 1200 s (the K-fp reference)",
           len(decodings) == 1 and all(((d or {}).get("decoding") or {}).get("batch_size") == tp.BATCH_SIZE
                                       and (d["decoding"]).get("max_batch_seconds") == tp.MAX_BATCH_SECONDS
                                       for d in arms["MB1"][1].values()))
    check6("MB1-test attempt record is the pinned file", digest(tp.MB1_ATTEMPTS) == tp.MB1_ATTEMPTS_SHA256)
    entries = read_jsonl(tp.MB1_ATTEMPTS)
    complete = {e.get("attempt") for e in entries if e.get("state") == "complete"}
    done = {e.get("set"): e.get("sha256") for e in entries if e.get("state") == "set_done"
            and e.get("attempt") in complete}
    current = {n: digest(tp.MB1_TEST / f"{n}.json") for n in paths.TEST_SETS}
    check6("MB1-test: one complete attempt whose set_done hashes are the current per-set files",
           len(complete) == 1 and done == current, f"complete {sorted(map(str, complete))}")


def require_precheck() -> dict:
    result = precheck()
    if not result["pass"]:
        failing = [f"[{c['item']}] {c['check']} ({c['detail']})" for c in result["checks"] if not c["ok"]]
        refuse(f"precheck failed ({len(failing)} checks): " + "; ".join(failing[:6]))
    return result


def run_precheck() -> int:
    result = precheck()
    relabel_paths.require_mount()
    write_json(tp.PRECHECK, result)
    for c in result["checks"]:
        print(f"{'PASS' if c['ok'] else 'FAIL'}  [{c['item']}] {c['check']}"
              + (f"  [{c['detail']}]" if c["detail"] and not c["ok"] else ""))
    hygiene = result["facts"].get("hygiene")
    if hygiene:
        print("hygiene:", json.dumps(hygiene, sort_keys=True))
    print("ALL PASS" if result["pass"] else "PRECHECK FAILED", "->", tp.PRECHECK)
    return 0 if result["pass"] else 1


# ----------------------------------------------------------------------------- training-data hygiene (item 7)
# The Common Voice held-out rules of stage 3's production code (finetune/relabel/cv27/heldout.py on the stage 3
# branch, not on this one), reimplemented here: sentences compared after the evaluation normalizer (stripped), in both
# forms (as written and as filters version 3's training text); speakers from ESB's `id` values, the release's test and
# dev client_ids, and the client_id any of the six tsvs records for a held-out clip path (a changed id).

def cv_normalizer():
    cache: dict[str, str] = {}

    def normalize(text: str | None) -> str:
        text = text or ""
        out = cache.get(text)
        if out is None:
            out = cache[text] = evaluate.normalize(text).strip()
        return out
    return normalize


def cv_training_text(sentence: str | None) -> str:
    return filters.map_text(sentence or "", version=tp.CV_FILTERS_VERSION)


def cv_sentence_keys(sentence: str | None, normalize) -> tuple[str, ...]:
    """The distinct normalized forms of a sentence: as written, then its filters version 3 text."""
    a, b = normalize(sentence or ""), normalize(cv_training_text(sentence))
    return (a,) if a == b else (a, b)


def cv_read_tsv(path: Path, columns=("client_id", "path", "sentence")):
    """Rows of one release tsv (tab-separated, no quoting), the columns checked present."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        missing = [c for c in columns if c not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError(f"{path}: missing columns {missing}")
        yield from reader


def cv_esb_rows(files) -> list[tuple[str, str, str]]:
    """(id, text, audio path) of the ESB Common Voice test split's parquet files (no audio bytes read)."""
    import pyarrow.parquet as pq
    out = []
    for path in files:
        t = pq.read_table(path, columns=["id", "text", "audio.path"])
        out += [(str(u), x or "", str(p)) for u, x, p in zip(t.column("id").to_pylist(), t.column("text").to_pylist(),
                                                              t.column("path").to_pylist())]
    return out


def cv_heldout(esb, tsv_rows, normalize) -> dict:
    """{"speakers", "paths", "sentences", "counts"}: the held-out sets of the production rules (block comment above);
    tsv_rows(name) yields the rows of one of tp.CV_TSVS. counts uses the production record's keys."""
    esb_spk, esb_path, esb_sent = set(), set(), set()
    n_esb = 0
    for uid, text, path in esb:
        n_esb += 1
        esb_spk.add(uid)
        esb_path.add(path)
        esb_sent.update(cv_sentence_keys(text, normalize))
    speakers, held_paths, sentences = set(esb_spk), set(esb_path), set(esb_sent)
    counts = {"esb_test": n_esb, "esb_test_paths": len(esb_path)}
    for name in tp.CV_HELD_SPLITS:
        n = 0
        for r in tsv_rows(name):
            n += 1
            speakers.add(r["client_id"])
            held_paths.add(r["path"])
            sentences.update(cv_sentence_keys(r["sentence"], normalize))
        counts[name] = n
    ids_of_path: dict[str, set[str]] = {}
    for name in tp.CV_TSVS:
        for r in tsv_rows(name):
            if r["path"] in held_paths:
                ids_of_path.setdefault(r["path"], set()).add(r["client_id"])
    joined = set().union(*ids_of_path.values()) if ids_of_path else set()
    counts.update({
        "held_paths_found_in_release": len(ids_of_path),
        "esb_paths_found_in_release": sum(1 for p in esb_path if p in ids_of_path),
        "esb_paths_with_changed_id": sum(1 for p in esb_path if ids_of_path.get(p) and not (ids_of_path[p] & esb_spk)),
        "speaker_ids_added_by_path_join": len(joined - speakers),
    })
    speakers |= joined
    counts.update({"held_speakers": len(speakers), "held_paths": len(held_paths), "held_sentences": len(sentences)})
    return {"speakers": speakers, "paths": held_paths, "sentences": sentences, "counts": counts}


def cv_raw_dir() -> Path:
    """The release's language directory (registry alias cv27-en, local.raw_dir under $RELABEL_ROOT)."""
    return relabel_paths.ARTIFACTS / registry.source(tp.CV_ALIAS)["local"]["raw_dir"]


def cv_esb_files() -> list[Path]:
    """The ESB test split of the same corpus (registry local.esb_test_set: a parakeet-ternary test set)."""
    return testsets.parquet_files(registry.source(tp.CV_ALIAS)["local"]["esb_test_set"])


def cv_selection_rows() -> dict[str, dict]:
    """{clip id: row} of the accepted rows (reason null) of the selection table."""
    import pyarrow.parquet as pq
    t = pq.read_table(tp.CV_SELECTION, columns=["id", "path", "client_id", "sentence", "reason"]).to_pydict()
    return {i: {"path": p, "client_id": c, "sentence": s}
            for i, p, c, s, r in zip(t["id"], t["path"], t["client_id"], t["sentence"], t["reason"]) if r is None}


def _hygiene_checks(check) -> dict:
    """Item 7: Common Voice (a) the store bound to the run and to the accepted selection, (b) no store clip with a
    held-out speaker, clip path or normalized sentence; AMI: no ESB AMI test meeting in the SDM pair store."""
    facts: dict = {}
    config = read_json(tp.RUN / "config.json")
    extra = (config.get("args") or {}).get("extra_sources") or {}
    cv_pool = (extra.get(f"{tp.CV_ALIAS}:") or {}).get("clip_pool")
    clips_sha = digest(tp.CV_STORE_CLIPS)
    check(f"Common Voice store clips.jsonl SHA-256 == run {tp.RUN_NAME}'s extra_sources clip_pool == the pinned hash",
          clips_sha == cv_pool == tp.CV_CLIP_POOL, f"store {clips_sha}, config {cv_pool}")
    meta = read_json(tp.CV_STORE_META)
    selection_sha = digest(tp.CV_SELECTION)
    check("selection.json is the pinned file", digest(tp.CV_SELECTION_META) == tp.CV_SELECTION_META_SHA256)
    sel_meta = read_json(tp.CV_SELECTION_META)
    check("clips.meta.json: clips_sha256 the store, selection_sha256 the accepted selection table (pinned)",
          meta.get("clips_sha256") == clips_sha and meta.get("selection_sha256") == selection_sha
          == sel_meta.get("table_sha256") == tp.CV_SELECTION_SHA256 and meta.get("complete") is True
          and meta.get("provisional") is False, f"meta {meta.get('selection_sha256')}, table {selection_sha}")
    selected = cv_selection_rows()
    store = []
    with open(tp.CV_STORE_CLIPS) as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                store.append((r["id"], r.get("sentence")))
    not_accepted = [i for i, _ in store if i not in selected]
    other_text = [i for i, s in store if i in selected and selected[i]["sentence"] != s]
    check("every store clip is an accepted selection row with the same sentence", store and not not_accepted
          and not other_text, f"{len(store)} clips; not accepted {not_accepted[:3]}, other sentence {other_text[:3]}")
    facts["cv_store_clips"] = len(store)
    # (b) the held-out sets recomputed from the same inputs (their hashes as the selection recorded them)
    raw = cv_raw_dir()
    recorded = {r["name"]: r["sha256"] for r in (sel_meta.get("inputs") or {}).get("tsvs") or []}
    changed = [n for n in tp.CV_TSVS if digest(raw / n) != recorded.get(n)]
    check("the release's six tsvs are the selection's recorded inputs", not changed, ", ".join(changed))
    esb_files = cv_esb_files()
    esb_recorded = {r["name"]: r["sha256"] for r in (sel_meta.get("inputs") or {}).get("esb_test") or []}
    esb_changed = [p.name for p in esb_files if digest(p) != esb_recorded.get(p.name)]
    check("the ESB Common Voice test parquet files are the selection's recorded inputs",
          esb_files and not esb_changed and {p.name for p in esb_files} == set(esb_recorded), ", ".join(esb_changed))
    normalize = cv_normalizer()
    held = cv_heldout(cv_esb_rows(esb_files), lambda name: cv_read_tsv(raw / name), normalize)
    want_counts = (sel_meta.get("counts") or {}).get("heldout")
    check("recomputed held-out counts equal the selection's recorded counts", held["counts"] == want_counts,
          f"recomputed {held['counts']}, recorded {want_counts}")
    hits = {"speaker": [], "clip": [], "sentence": []}
    for cid, _ in store:
        row = selected.get(cid)
        if row is None:
            continue
        if row["client_id"] in held["speakers"]:
            hits["speaker"].append(cid)
        if row["path"] in held["paths"]:
            hits["clip"].append(cid)
        if any(k in held["sentences"] for k in cv_sentence_keys(row["sentence"], normalize)):
            hits["sentence"].append(cid)
    for kind, found in hits.items():
        check(f"no Common Voice store clip with a held-out {kind}", not found, f"{len(found)}: {found[:3]}")
    facts["cv_heldout"] = held["counts"]
    # AMI
    ami_pool = (extra.get("ami-sdm:") or {}).get("clip_pool")
    ami_sha = digest(tp.AMI_STORE_CLIPS)
    check(f"AMI SDM store clips.jsonl SHA-256 == run {tp.RUN_NAME}'s extra_sources clip_pool == the pinned hash",
          ami_sha == ami_pool == tp.AMI_CLIP_POOL, f"store {ami_sha}, config {ami_pool}")
    test_meetings = {ami_sdm.meeting_of(r["id"]) for r in evaluate.read_manifest(tp.manifest_path(tp.AMI_TEST_SET))}
    clips = read_jsonl(tp.AMI_STORE_CLIPS)
    inconsistent = [c["id"] for c in clips if not (ami_sdm.meeting_of(c["id"]) == ami_sdm.meeting_of(c["ihm_id"])
                                                   == c.get("meeting_id"))]
    check("AMI store: every clip's id, IHM id and meeting_id name one meeting", clips and not inconsistent,
          f"{len(clips)} clips; {inconsistent[:3]}")
    leaked = sorted({c.get("meeting_id") for c in clips} & test_meetings
                    | {ami_sdm.meeting_of(c["id"]) for c in clips} & test_meetings)
    check("no ESB AMI test meeting in the AMI SDM pair store", test_meetings and not leaked, ", ".join(leaked))
    facts.update(ami_store_clips=len(clips), ami_test_meetings=len(test_meetings))
    return facts


# ----------------------------------------------------------------------------- environment of a decoding process

def require_environment(mode: str) -> dict:
    """The interpreter environment a mode decodes in (module docstring); refuses before anything is loaded."""
    if mode == "x1":
        tk.deployed_environment()          # drops the CUBLAS_WORKSPACE_CONFIG train.py set, if the caller had none
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
            refuse("x1 decodes as the recorded x1 arm ran: start the process with CUBLAS_WORKSPACE_CONFIG unset "
                   "(env -u CUBLAS_WORKSPACE_CONFIG ./python-gpu1bit keab_test.py ...)")
        try:
            return deployed.require_wrapper()
        except RuntimeError as error:
            refuse(str(error))
    if os.environ.get("GPU1BIT_WRAPPER"):
        refuse("K-fp runs under ./python (MB1-test's environment, no NVRTC on the library path), not ./python-gpu1bit")
    nvrtc = [e for e in os.environ.get("LD_LIBRARY_PATH", "").split(":") if e and any(Path(e).glob("libnvrtc*"))]
    if nvrtc:
        refuse(f"K-fp runs without NVRTC on the library path (as MB1-test's interpreter); found it in {nvrtc}")
    recorded = read_json(tp.MB1_TEST / "summary.json").get("settings") or {}
    current = numerical_settings("fp")
    want, got = recorded.get("versions") or {}, current["versions"]
    differ = [k for k in tp.FP_VERSION_KEYS if got.get(k) != want.get(k)]
    if differ:
        refuse(f"K-fp needs MB1-test's interpreter versions; {differ} differ: {got} vs {want}")
    flags = [k for k in ("cudnn_deterministic", "cudnn_benchmark") if current[k] != recorded.get(k)]
    if flags:
        refuse(f"K-fp decodes with MB1-test's cuDNN flags; {flags} differ")
    return {"versions": got}


def require_device() -> torch.device:
    device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        refuse("CUDA is required")
    return device


def gpu_lock(label: str, wait_s: float):
    """The shared parakeet GPU lock (paths.gpu_lock's file) through stage 4's waiting lock."""
    return k0.waiting_lock(label, wait_s)


def free_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def release_on_exit(fn, *args, **kwargs):
    """fn(*args): the model exists only in fn's frames; on an exception the traceback's frames are cleared (they hold
    the model) before it propagates; GPU memory is collected and flushed while the caller still holds the GPU lock."""
    try:
        return fn(*args, **kwargs)
    except BaseException as error:
        seen: set[int] = set()
        while error is not None and id(error) not in seen:
            seen.add(id(error))
            traceback.clear_frames(error.__traceback__)
            error = error.__cause__ or error.__context__
        raise
    finally:
        free_gpu_memory()


# ----------------------------------------------------------------------------- decoding

def manifests_for(names) -> list[tuple[str, Path]]:
    sets = [(n, tp.manifest_path(n)) for n in names]
    missing = [str(p) for _, p in sets if not p.exists()]
    if missing:
        refuse(f"missing manifests: {missing}")
    return sets


def view_source_x1() -> dict:
    vman = read_json(tp.VIEW / "manifest.json")
    if vman.get("sha256") != tp.VIEW_SHA256:
        refuse(f"{tp.VIEW} manifest sha256 {vman.get('sha256')} is not the pinned view {tp.VIEW_SHA256}")
    return {"kind": "deployed-x1", "model": tp.MODEL_LABEL, "view": str(tp.VIEW), "view_sha256": tp.VIEW_SHA256,
            "kernel": tp.X1_KERNEL, "parts": tp.X1_PARTS, "pinned_identity": identity()}


def export_source_fp(export_dir: Path, description: str, want_sha256: str, label: str) -> dict:
    """evaluate.describe_source("export", ...) plus the export manifest; refuses weights other than the pinned ones."""
    source = evaluate.describe_source("export", Path(export_dir), description)
    if source.get("sha256") != want_sha256:
        refuse(f"{export_dir} weights SHA-256 {source.get('sha256')} is not the pinned {want_sha256}")
    manifest = read_json(Path(export_dir) / "manifest.json")
    source["export_manifest"] = {k: v for k, v in manifest.items()
                                 if k in ("format", "file", "sha256", "files", "parameter_accounting", "bytes")}
    source["model"] = label
    source["pinned_identity"] = identity()
    return source


def finish_set(result: dict, *, mode: str, model: str, phase: str, name: str, manifest: Path, source: dict,
               limit: int | None, settings: dict, **extra) -> dict:
    s = structural(result["records"])
    result.update(set=name, mode=mode, model=model, phase=phase, manifest=str(manifest),
                  manifest_sha256=digest(manifest), limit=limit, normalizer=evaluate.NORMALIZER, source=source,
                  identity=identity(), scorer=scorer_record(), settings=settings, structural=s,
                  structurally_valid=is_valid(s),
                  keab_test={"mode": mode, "phase": phase, "model": model, "limit": limit,
                             "batch_size": 1 if mode == "x1" else tp.BATCH_SIZE,
                             "max_batch_seconds": None if mode == "x1" else tp.MAX_BATCH_SECONDS}, **extra)
    return result


def _write_set(out_dir: Path, name: str, result: dict, on_done) -> None:
    path = out_dir / f"{name}.json"
    write_json(path, result)
    flag = "" if result["structurally_valid"] else f"  STRUCTURALLY INVALID: {result['structural']}"
    print(f"{result['model']} {result['mode']} {name}: n={result['utterances']} WER {100 * result['wer']['wer']:.3f}%"
          + flag, flush=True)
    if on_done is not None:
        on_done(name, path)


def decode_x1(sets, out_dir: Path, limit: int | None, settings: dict, phase: str, on_done=None) -> dict[str, dict]:
    """Kea-B′'s view on the x1 runtime (training/deployed.py), per set into out_dir/<set>.json."""
    return release_on_exit(_decode_x1, sets, out_dir, limit, settings, phase, on_done)


def _decode_x1(sets, out_dir, limit, settings, phase, on_done):
    source = view_source_x1()
    tk.vram_lease()
    model = deployed.load_x1(tp.VIEW)
    try:
        info = deployed.info(model)
        if (info.get("export_sha256"), info.get("kernel"), info.get("parts")) != (tp.VIEW_SHA256, tp.X1_KERNEL,
                                                                                tp.X1_PARTS):
            refuse(f"the x1 runtime loaded {info.get('export_sha256')} {info.get('kernel')} parts {info.get('parts')}")
        runtime = {k: v for k, v in info.items() if k != "sources"}
        row_identity = hashlib.sha256(json.dumps({"model": tp.MODEL_LABEL, "view_sha256": tp.VIEW_SHA256,
                                                  "mode": "x1"}, sort_keys=True).encode()).hexdigest()
        results = {}
        for name, manifest in sets:
            stray = stray_x1_modules()
            if stray:
                refuse(f"imported x1 modules outside the pinned scorer files: {stray[:4]}")
            if numerical_settings("x1") != settings:
                refuse(f"numerical settings changed since start-up (before {name})")
            records = list(enumerate(evaluate.read_manifest(manifest)))[:limit]
            result = deployed.score_records(model, records, name, row_identity)
            finish_set(result, mode="x1", model=tp.MODEL_LABEL, phase=phase, name=name, manifest=manifest,
                       source=source, limit=limit, settings=settings, runtime=runtime)
            _write_set(out_dir, name, result, on_done)
            results[name] = result
        return results
    finally:
        deployed.close(model)


def decode_fp(export_dir: Path, model_label: str, source: dict, sets, out_dir: Path, limit: int | None,
              settings: dict, phase: str, on_done=None) -> dict[str, dict]:
    """An export rebuilt by binary_export.load_export, its decoder pinned, evaluate.evaluate_set per set."""
    return release_on_exit(_decode_fp, export_dir, model_label, source, sets, out_dir, limit, settings, phase, on_done)


def _decode_fp(export_dir, model_label, source, sets, out_dir, limit, settings, phase, on_done):
    device = require_device()
    tk.vram_lease()
    model = binary_export.load_export(Path(export_dir), device)
    binary_train.pin_decoder_graphs_mode(model, f"{model_label} export")
    results = {}
    for name, manifest in sets:
        if numerical_settings("fp") != settings:
            refuse(f"numerical settings changed since start-up (before {name})")
        before = binary_train.require_decoder_graphs_mode(model, f"{model_label} before {name}")
        result = evaluate.evaluate_set(model, name, manifest, source, tp.BATCH_SIZE, limit, device,
                                       max_batch_seconds=tp.MAX_BATCH_SECONDS)
        after = binary_train.require_decoder_graphs_mode(model, f"{model_label} after {name}")
        finish_set(result, mode="fp", model=model_label, phase=phase, name=name, manifest=manifest, source=source,
                   limit=limit, settings=settings, decoder_graphs_mode={"before": before, "after": after})
        _write_set(out_dir, name, result, on_done)
        results[name] = result
    return results


# ----------------------------------------------------------------------------- dev cross-check

def within(diff: float, tolerance: float) -> bool:
    """|diff| <= tolerance, float noise below 1e-12 ignored (exactly at the tolerance passes)."""
    return round(abs(diff), 12) <= tolerance


def gate_verdict(measured: dict[str, float], recorded: dict[str, float]) -> dict:
    """WER fractions of the five dev sets: the mean within MEAN_TOLERANCE of the recorded mean and every set within
    SET_TOLERANCE of its recorded value."""
    names = list(paths.DEV_SETS)
    if set(measured) != set(names) or set(recorded) != set(names):
        raise ValueError(f"dev sets {sorted(measured)} / {sorted(recorded)} are not {names}")
    per_set = {}
    for name in names:
        diff = measured[name] - recorded[name]
        per_set[name] = {"wer": measured[name], "recorded": recorded[name], "diff": diff, "diff_points": 100 * diff,
                         "within": within(diff, tp.SET_TOLERANCE)}
    mean = sum(measured[n] for n in names) / len(names)
    recorded_mean = sum(recorded[n] for n in names) / len(names)
    mean_diff = mean - recorded_mean
    mean_within = within(mean_diff, tp.MEAN_TOLERANCE)
    return {"pass": mean_within and all(e["within"] for e in per_set.values()), "mean": mean,
            "recorded_mean": recorded_mean, "mean_diff": mean_diff, "mean_diff_points": 100 * mean_diff,
            "mean_within": mean_within, "per_set": per_set,
            "tolerances": {"mean_points": tp.MEAN_TOLERANCE_POINTS, "set_points": tp.SET_TOLERANCE_POINTS,
                           "mean_fraction": tp.MEAN_TOLERANCE, "set_fraction": tp.SET_TOLERANCE}}


def subset_wer(records: list[dict], ids) -> float:
    ids = set(ids)
    return evaluate.wer_summary([r for r in records if r.get("id") in ids])["wer"]


def token_identity(results: dict[str, dict], recorded: dict[str, dict]) -> dict:
    per_set, total, same = {}, 0, 0
    for name, res in results.items():
        ref = {r["id"]: r.get("tokens") for r in recorded[name]["records"]}
        n = len(res["records"])
        k = sum(1 for r in res["records"] if r["id"] in ref and r.get("tokens") == ref[r["id"]])
        per_set[name] = {"utterances": n, "identical": k, "rate": k / max(1, n)}
        total, same = total + n, same + k
    return {"utterances": total, "identical": same, "rate": same / max(1, total), "per_set": per_set}


def dev_check(mode: str, limit: int | None, lock_wait: float) -> int:
    """Decode the five dev sets into dev_check_dir(mode, limit); gate.json last. Exit 0 if the gate passed (a --limit
    smoke always exits 0; it is never a launch gate)."""
    out = tp.dev_check_dir(mode, limit)

    def refuse_if_gated() -> None:
        if limit is None and (out / tp.GATE).exists():
            refuse(f"{out / tp.GATE} exists; a dev check is not rerun into the same directory "
                   "(move it aside by hand, with the reason)")

    refuse_if_gated()           # early; repeated under the GPU lock
    pre = require_precheck()
    sets = manifests_for(paths.DEV_SETS)
    settings = numerical_settings(mode)
    with gpu_lock(f"keab_test:{mode}-dev-check" + ("" if limit is None else f"-limit{limit}"), lock_wait):
        refuse_if_gated()
        relabel_paths.require_mount()
        out.mkdir(parents=True, exist_ok=True)
        for stale in out.rglob("*.json"):   # an unfinished earlier attempt (no gate.json) or a smoke rerun
            stale.unlink()
        gate = (_dev_check_x1 if mode == "x1" else _dev_check_fp)(out, sets, limit, settings)
        gate.update(mode=mode, limit=limit, test_launch_gate=limit is None, out_dir=str(out), identity=identity(),
                    scorer=scorer_record(), manifests=manifest_hashes(), settings=settings,
                    precheck={"pass": pre["pass"], "utc": pre["utc"]},
                    files_sha256={p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob("*.json"))
                                  if p.name != tp.GATE},
                    utc=utc())
        gate["pass"] = bool(gate["within_tolerance"] and gate["structurally_valid"] and gate.get("checks_ok", True)
                            and limit is None)
        write_json(out / tp.GATE, gate)
    print(f"{mode} dev check: gate {'PASS' if gate['pass'] else 'FAIL'}"
          + ("" if limit is None else " (a --limit smoke, never a launch gate)") + f" -> {out / tp.GATE}")
    return 0 if gate["pass"] or limit is not None else 1


def _dev_check_x1(out: Path, sets, limit, settings) -> dict:
    results = decode_x1(sets, out, limit, settings, "dev-check")
    recorded = {n: read_json(tp.SCORES_X1 / f"{n}.json") for n in paths.DEV_SETS}
    same = {}
    for n in paths.DEV_SETS:
        got = [(r.get("i"), r.get("id")) for r in results[n]["records"]]
        want = [(r.get("i"), r.get("id")) for r in recorded[n]["records"]][:len(got) if limit else None]
        same[n] = got == want
    recorded_wer = {n: (recorded[n]["wer"]["wer"] if limit is None else
                        subset_wer(recorded[n]["records"], [r["id"] for r in results[n]["records"]]))
                    for n in paths.DEV_SETS}
    verdict = gate_verdict({n: results[n]["wer"]["wer"] for n in paths.DEV_SETS}, recorded_wer)
    s = {n: results[n]["structural"] for n in paths.DEV_SETS}
    gate = {**verdict, "within_tolerance": verdict["pass"], "recorded_from": str(tp.SCORES_X1),
            "same_records": same, "checks_ok": all(same.values()),
            "token_identity": token_identity(results, recorded), "structural": s,
            "structurally_valid": all(is_valid(v) for v in s.values()),
            "runtime": results[paths.DEV_SETS[0]]["runtime"]}
    for name, e in gate["per_set"].items():
        print(f"{name:22s} {100 * e['wer']:.3f}% recorded {100 * e['recorded']:.3f}% diff {e['diff_points']:+.3f} "
              f"points {'ok' if e['within'] else 'OUTSIDE 0.10'}; same records {same[name]}")
    print(f"mean {100 * gate['mean']:.4f}% recorded {100 * gate['recorded_mean']:.4f}% diff "
          f"{gate['mean_diff_points']:+.4f} points; token identity {100 * gate['token_identity']['rate']:.2f}%")
    return gate


def mb1_dev_check_records() -> tuple[dict, dict[str, dict]]:
    """(MB1's recorded dev-check gate, {dev set: its per-set output}); refuses files other than the pinned gate's."""
    data = tp.MB1_DEV_CHECK_GATE.read_bytes()
    if hashlib.sha256(data).hexdigest() != tp.MB1_DEV_CHECK_GATE_SHA256:
        refuse(f"{tp.MB1_DEV_CHECK_GATE} is not the pinned MB1 dev-check gate")
    gate = json.loads(data)
    docs = {}
    for n in paths.DEV_SETS:
        path = tp.MB1_DEV_CHECK_GATE.parent / f"{n}.json"
        if digest(path) != (gate.get("per_set_sha256") or {}).get(n):
            refuse(f"{path} is not the file MB1's dev-check gate records")
        docs[n] = read_json(path)
    return gate, docs


def mb1_test_decoding() -> dict:
    """MB1-test's decoding record (one record for all eight sets, precheck item 6)."""
    return read_json(tp.MB1_TEST / f"{next(iter(paths.TEST_SETS))}.json")["decoding"]


def _dev_check_fp(out: Path, sets, limit, settings) -> dict:
    mb1_gate, mb1_docs = mb1_dev_check_records()
    reference = mb1_test_decoding()
    mb1_source = export_source_fp(tp.MB1_EXPORT, tp.MB1_DESCRIPTION, tp.MB1_EXPORT_SHA256, "MB1")
    mb1 = decode_fp(tp.MB1_EXPORT, "MB1", mb1_source, sets, out / "MB1", limit, settings, "dev-check")
    if limit is None:
        recorded = {n: mb1_gate["per_set"][n]["wer"] for n in paths.DEV_SETS}
    else:
        recorded = {n: subset_wer(mb1_docs[n]["records"], [r["id"] for r in mb1[n]["records"]]) for n in paths.DEV_SETS}
    verdict = gate_verdict({n: mb1[n]["wer"]["wer"] for n in paths.DEV_SETS}, recorded)
    if limit is None and abs(verdict["recorded_mean"] - tp.MB1_DEV_MEAN) > 1e-12:
        refuse(f"MB1's recorded dev-check mean {verdict['recorded_mean']} is not the pinned {tp.MB1_DEV_MEAN}")
    s = {f"MB1/{n}": mb1[n]["structural"] for n in paths.DEV_SETS}
    decoding_ok = {f"MB1/{n}": mb1[n]["decoding"] == reference for n in paths.DEV_SETS}
    gate = {"within_tolerance": verdict["pass"], "mb1": {**verdict, "recorded_from": str(tp.MB1_DEV_CHECK_GATE)}}
    for name, e in verdict["per_set"].items():
        print(f"MB1 {name:22s} {100 * e['wer']:.3f}% recorded {100 * e['recorded']:.3f}% diff "
              f"{e['diff_points']:+.3f} points {'ok' if e['within'] else 'OUTSIDE 0.10'}")
    print(f"MB1 mean {100 * verdict['mean']:.4f}% recorded {100 * verdict['recorded_mean']:.4f}% diff "
          f"{verdict['mean_diff_points']:+.4f} points")
    if verdict["pass"] and all(is_valid(v) for v in s.values()) and all(decoding_ok.values()):
        source = export_source_fp(tp.VIEW, tp.VIEW_DESCRIPTION, tp.VIEW_SHA256, tp.MODEL_LABEL)
        keab = decode_fp(tp.VIEW, tp.MODEL_LABEL, source, sets, out / "keab", limit, settings, "dev-check")
        torch_dev = read_json(tp.RUN / "summary.json")["dev_wer"]
        measured = {n: keab[n]["wer"]["wer"] for n in paths.DEV_SETS}
        gate["keab"] = {"wer": measured, "mean": sum(measured.values()) / len(measured),
                        "pytorch_fake_quant": torch_dev,
                        "pytorch_fake_quant_mean": sum(torch_dev.values()) / len(torch_dev),
                        "diff_points": {n: 100 * (measured[n] - torch_dev[n]) for n in paths.DEV_SETS},
                        "threshold": None}
        s.update({f"keab/{n}": keab[n]["structural"] for n in paths.DEV_SETS})
        decoding_ok.update({f"keab/{n}": keab[n]["decoding"] == reference for n in paths.DEV_SETS})
        for n in paths.DEV_SETS:
            print(f"Kea-B' fp {n:22s} {100 * measured[n]:.3f}% (PyTorch fake-quant {100 * torch_dev[n]:.3f}%)")
    else:
        gate["keab"] = None
        print("MB1's K-fp validation did not pass: Kea-B' is not scored in K-fp mode")
    gate.update(structural=s, structurally_valid=all(is_valid(v) for v in s.values()),
                decoding_matches_mb1_test=decoding_ok,
                checks_ok=all(decoding_ok.values()) and gate["keab"] is not None)
    return gate


def require_gate(mode: str) -> tuple[dict, str]:
    """(the mode's production dev-check gate, the SHA-256 of exactly the bytes validated); refuses unless it passed and
    records this process's identity, scorer hashes, manifest hashes and settings."""
    path = tp.dev_check_dir(mode) / tp.GATE
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        refuse(f"no dev-check gate {path}; run --dev-check {mode} first")
    try:
        gate = json.loads(data)
    except ValueError:
        refuse(f"dev-check gate {path} is not JSON")
    if gate.get("limit") is not None:
        refuse(f"dev-check gate {path} is a --limit {gate.get('limit')} run")
    if gate.get("mode") != mode or gate.get("out_dir") != str(tp.dev_check_dir(mode)) \
            or gate.get("test_launch_gate") is not True:
        refuse(f"dev-check gate {path} is not the production {mode} dev check")
    if gate.get("pass") is not True:
        refuse(f"dev-check gate {path} did not pass; no test set is decoded in mode {mode}")
    for key, current in (("identity", identity()), ("scorer", scorer_record()), ("manifests", manifest_hashes()),
                         ("settings", numerical_settings(mode))):
        if gate.get(key) != current:
            refuse(f"dev-check gate {path} records other {key} than this process")
    return gate, hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------------------- attempts

@contextlib.contextmanager
def attempt_lock(mode: str, wait: bool, label: str):
    """Exclusive flock on the mode's attempt lock for one transition. wait=False refuses if it is held."""
    relabel_paths.require_mount()
    path = tp.attempts_lock(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if not wait:
                refuse(f"another --test or --summarize holds {path} (an attempt may be running); retry after it ends")
            print(f"waiting for the attempt lock {path} ...", flush=True)
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()} {label} {utc()}\n")
            handle.flush()
            yield
        finally:
            handle.seek(0)
            handle.truncate()
            handle.flush()
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_attempts(mode: str) -> list[dict]:
    path = tp.attempts_path(mode)
    if not path.exists():
        return []
    entries = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                entries.append(json.loads(line))
            except ValueError:
                refuse(f"{path} line {number} is not JSON")
    return entries


def attempt_state(mode: str, entries: list[dict]) -> dict:
    """{"started": entry or None, "resumed": [entries], "set_done": {set: sha256}, "complete": bool} of the mode's one
    attempt; refuses an unreadable record."""
    path = tp.attempts_path(mode)
    out = {"started": None, "resumed": [], "set_done": {}, "complete": False}
    for e in entries:
        state = e.get("state")
        if e.get("attempt") != 1 or state not in STATES:
            refuse(f"{path}: unreadable entry {e}")
        if state == "started":
            if out["started"] is not None:
                refuse(f"{path}: two started records")
            out["started"] = e
        elif out["started"] is None:
            refuse(f"{path}: a {state} record before the started record")
        elif state == "resumed":
            out["resumed"].append(e)
        elif state == "set_done":
            if e.get("set") not in paths.TEST_SETS or e.get("set") in out["set_done"]:
                refuse(f"{path}: set_done for {e.get('set')!r} twice or for an unknown set")
            out["set_done"][e["set"]] = e.get("sha256")
        else:
            out["complete"] = True
    return out


def all_done(state: dict) -> bool:
    return set(state["set_done"]) == set(paths.TEST_SETS)


def append_attempt(mode: str, entry: dict) -> None:
    """Append one line to the mode's attempt record (append-only, flushed and fsynced)."""
    relabel_paths.require_mount()
    path = tp.attempts_path(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps({"attempt": 1, **entry}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def check_done_files(mode: str, state: dict) -> None:
    """Every set_done set's file still matches its recorded hash (never decoded again, never replaced)."""
    for name, want in state["set_done"].items():
        path = tp.test_dir(mode) / f"{name}.json"
        got = digest(path)
        if got != want:
            refuse(f"{path} SHA-256 {got} is not its set_done hash {want}: investigate by hand (a set recorded as "
                   "done is never decoded again)")


def run_test(mode: str, lock_wait: float) -> int:
    """The mode's test run into test_dir(mode), under its attempt lock (waits for it); resumes an open attempt."""
    with attempt_lock(mode, wait=True, label=f"--test {mode}"):
        return _run_test_locked(mode, lock_wait)


def _run_test_locked(mode: str, lock_wait: float) -> int:
    state = attempt_state(mode, read_attempts(mode))     # read after the attempt lock: never a stale view
    if state["complete"]:
        refuse(f"mode {mode} is complete: Kea-B' has been scored on test in this mode")
    if all_done(state):
        refuse(f"mode {mode} has all eight sets done; only --summarize is allowed")
    out = tp.test_dir(mode)

    def early() -> None:
        if state["started"] is None:
            if out.exists():
                refuse(f"{out} exists without an attempt record; investigate by hand")
        else:
            check_done_files(mode, state)

    early()
    require_precheck()
    require_gate(mode)                      # early refusal; repeated under the GPU lock
    sets = manifests_for(paths.TEST_SETS)
    settings = numerical_settings(mode)
    with gpu_lock(f"keab_test:test-{mode}", lock_wait):
        gate, gate_sha256 = require_gate(mode)       # the bytes validated here are the ones recorded
        early()
        if state["started"] is None:
            append_attempt(mode, {"state": "started", "utc": utc(), "pid": os.getpid(), "gate_sha256": gate_sha256,
                                  "gate_utc": gate.get("utc"), "job_run_dir": os.environ.get("AGENT_JOBS_RUN_DIR")})
            out.mkdir(parents=True)
        else:
            if gate_sha256 != state["started"].get("gate_sha256"):
                refuse(f"the {mode} dev-check gate (SHA-256 {gate_sha256}) is not the gate this attempt started with "
                       f"({state['started'].get('gate_sha256')})")
            replayed = [n for n in paths.TEST_SETS if n not in state["set_done"] and (out / f"{n}.json").exists()]
            for n in replayed:              # written before a crash, never recorded: decoded again from the start
                (out / f"{n}.json").unlink()
            append_attempt(mode, {"state": "resumed", "utc": utc(), "pid": os.getpid(), "gate_sha256": gate_sha256,
                                  "done": sorted(state["set_done"]), "replayed": replayed,
                                  "job_run_dir": os.environ.get("AGENT_JOBS_RUN_DIR")})
        todo = [(n, m) for n, m in sets if n not in state["set_done"]]

        def done(name: str, path: Path) -> None:
            append_attempt(mode, {"set": name, "state": "set_done", "sha256": digest(path), "utc": utc()})

        if mode == "x1":
            decode_x1(todo, out, None, settings, "test", on_done=done)
        else:
            source = export_source_fp(tp.VIEW, tp.VIEW_DESCRIPTION, tp.VIEW_SHA256, tp.MODEL_LABEL)
            decode_fp(tp.VIEW, tp.MODEL_LABEL, source, todo, out, None, settings, "test", on_done=done)
        finalize(mode, "test")
    print(f"Kea-B' {mode} test complete -> {out}")
    return 0


def test_summary(mode: str, docs: dict[str, dict], digests: dict[str, str], by: str) -> dict:
    first = docs[next(iter(paths.TEST_SETS))]
    sets = {n: {"wer": 100 * d["wer"]["wer"], "utterances": d["utterances"],
                "scored_utterances": d["wer"]["utterances"],
                "audio_hours": d.get("audio_hours"), "wall_s": d.get("wall_s"), "limit": d["limit"],
                "structurally_valid": d["structurally_valid"]} for n, d in docs.items()}
    return {"mode": mode, "model": tp.MODEL_LABEL, "sets": sets,
            "mean_wer": sum(sets[n]["wer"] for n in paths.MEAN_SETS) / len(paths.MEAN_SETS),
            "mean_sets": list(paths.MEAN_SETS), "common_voice_wer": sets["common_voice"]["wer"],
            "structurally_valid": all(s["structurally_valid"] for s in sets.values()),
            "per_set_sha256": digests, "source": first["source"], "identity": first["identity"],
            "scorer": first["scorer"], "settings": first["settings"], "attempt": 1, "written_by": by,
            "written_utc": utc()}


def finalize(mode: str, by: str) -> None:
    """summary.json from the eight per-set files (each checked against its set_done hash), then "complete"."""
    state = attempt_state(mode, read_attempts(mode))
    if not all_done(state):
        refuse(f"mode {mode}: set_done records for {sorted(state['set_done'])}, not all eight sets")
    docs, digests = {}, {}
    for name in paths.TEST_SETS:
        path = tp.test_dir(mode) / f"{name}.json"
        data = path.read_bytes()
        digests[name] = hashlib.sha256(data).hexdigest()
        if digests[name] != state["set_done"][name]:
            refuse(f"{path} SHA-256 {digests[name]} is not its set_done hash {state['set_done'][name]}")
        docs[name] = json.loads(data)
    invalid = [n for n, d in docs.items() if not d.get("structurally_valid")]
    write_json(tp.test_dir(mode) / "summary.json", test_summary(mode, docs, digests, by))
    append_attempt(mode, {"state": "complete", "utc": utc(), "by": by})
    if invalid:
        print(f"WARNING: structurally invalid rows in {invalid}: the report refuses a numerical result for mode {mode};"
              " investigate the rows (nothing is re-scored automatically)", flush=True)


def summarize(mode: str) -> int:
    """Rebuild summary.json of the mode's fully decoded attempt without decoding, then append "complete" once."""
    with attempt_lock(mode, wait=False, label=f"--summarize {mode}"):
        state = attempt_state(mode, read_attempts(mode))
        if state["complete"]:
            print(f"mode {mode} is already complete; nothing to do")
            return 0
        if not all_done(state):
            refuse(f"mode {mode}: no attempt has set_done records for all eight test sets")
        if scorer_hashes() != tp.SCORER_HASHES:
            refuse("the current scoring sources differ from SCORER_HASHES")
        for name in paths.TEST_SETS:
            doc = read_json(tp.test_dir(mode) / f"{name}.json")
            if (doc.get("scorer") or {}).get("hashes") != tp.SCORER_HASHES:
                refuse(f"{name}: recorded scorer hashes differ from SCORER_HASHES")
        finalize(mode, "summarize")
    print(f"mode {mode} summarized and complete -> {tp.test_dir(mode) / 'summary.json'}")
    return 0


# ----------------------------------------------------------------------------- CLI

def print_scorer_hashes() -> int:
    hashes = scorer_hashes(tp.scorer_keys())
    missing = [k for k, v in hashes.items() if v is None]
    print("SCORER_HASHES = " + json.dumps(hashes, indent=4))
    print(f"# keab_test_paths.py SHA-256 now: {digest(Path(tp.__file__))}")
    print("# Pin: paste SCORER_HASHES into keab_test_paths.py, then record the new keab_test_paths.py SHA-256 in the "
          "plan's Status line (precheck item 3 checks both).")
    if missing:
        print(f"# MISSING files: {missing}")
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--precheck", action="store_true", help="plan 'Preconditions'; read-only, no GPU")
    action.add_argument("--scorer-hashes", action="store_true", help="print the scorer files' current SHA-256")
    action.add_argument("--dev-check", choices=tp.MODES, metavar="MODE", help="the dev cross-check of x1 or fp")
    action.add_argument("--test", choices=tp.MODES, metavar="MODE", help="the test run of x1 or fp (resumable)")
    action.add_argument("--summarize", choices=tp.MODES, metavar="MODE", help="rebuild summary.json; no decoding")
    parser.add_argument("--limit", type=int, help="--dev-check only: the first N utterances per set (a smoke)")
    parser.add_argument("--lock-wait", type=float, default=LOCK_WAIT_S, help="seconds to wait for the GPU lock")
    args = parser.parse_args(argv)
    if args.limit is not None and args.dev_check is None:
        parser.error("--limit is accepted only with --dev-check: each mode is scored on test once")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.scorer_hashes:
        return print_scorer_hashes()
    relabel_paths.require_mount()
    if args.precheck:
        return run_precheck()
    if args.summarize:
        return summarize(args.summarize)
    mode = args.dev_check or args.test
    require_environment(mode)
    try:
        if args.dev_check:
            return dev_check(mode, args.limit, args.lock_wait)
        return run_test(mode, args.lock_wait)
    except score_teacher.LockBusy as busy:
        print(f"keab_test: refused: {busy}", file=sys.stderr)
        return EXIT_LOCK_BUSY


if __name__ == "__main__":
    sys.exit(main())
