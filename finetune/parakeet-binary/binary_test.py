"""Test-set scoring of the binary main run MB1, exactly once (plans/test-scoring.md).

    python binary_test.py --precheck                        # read-only, no GPU; writes eval/MB1-precheck.json
    python binary_test.py --dev-check [--limit N] [--dev-batch-size 32]
    python binary_test.py --test                            # the one test-set run, eval/MB1-test
    python binary_test.py --summarize                       # rebuild summary.json after a late failure
    python binary_test.py --record-failure "<evidence>"     # after investigating a crashed attempt

Only the loading is new: the MB1 export is rebuilt with binary_export.load_export (the loader MB1's
full-dev score used) and the greedy TDT decoder's CUDA-graph mode is pinned to
binary_paths.DECODER_GRAPHS_MODE with binary_train's helpers and checked before and after every
set. Decoding and scoring are the frozen parakeet-ternary evaluate.py's (evaluate_set, summarize,
describe_source, resolve_manifest), at batch 64 and max_batch_seconds 1200 (M1-test's values, fixed
here), in its strict FP32 and inference settings. The shared GPU lock is held for the whole run.

Every pinned input is a constant in binary_test_paths. --dev-check and --test re-run the precheck
in-process and refuse unless it passes; --test also refuses unless eval/MB1-dev-check/gate.json
passed at batch 64 without --limit and records this process's identity, scorer hashes, manifest
hashes and numerical settings. Refusals that need no GPU come before any model is loaded.

Attempts on test are recorded in eval/MB1-test.attempts.jsonl (append-only): "started" before
decoding, "set_done" with the per-set file's SHA-256 after each set, "complete" after summary.json.
A started attempt without a "failed" record (a crash) blocks every new attempt until it is
investigated and recorded with --record-failure, which renames eval/MB1-test to
MB1-test.failed-<time>. Once an attempt has all eight sets done, only --summarize is allowed.

Attempt transitions are serialized (code review round 1): --test, --record-failure and --summarize run
under the attempt lock eval/MB1-test.attempts.lock and read the attempt record only after acquiring
it. --test waits for it; the recovery commands refuse at once if it is held (an attempt may be
running). Lock order: the attempt lock, then the shared GPU lock (paths.gpu_lock), never the reverse;
nothing else takes the attempt lock, so waiting on the GPU lock (other agents' kernel jobs) cannot
deadlock. --test holds the GPU lock through summary.json, the "complete" record and the model's release.

Under the shared GPU lock (code review round 2): the model lives only inside one call
(decode_and_release), whose frames are cleared on an exception, and GPU memory is collected and flushed
before the lock is released. --test reads the launch gate's bytes once under the lock, validates exactly
those bytes and records their SHA-256; the output directory and the export's weight hash are checked
there too. --dev-check checks and cleans its output directory under the lock. The same checks before
the lock are early refusals only (no waiting for the GPU when the answer is already no).
"""
from __future__ import annotations

import binary_train  # noqa: I001  first: imports train.py, which configures CUDA/numba (as for MB1's dev scoring)

import argparse
import contextlib
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

import binary_export
import binary_paths
import binary_test_paths as tp
import evaluate
import paths

DEVICE = "cuda"
STATES = ("started", "set_done", "failed", "complete")
# Environment variables that can change numerics or the decoder backend (train.py sets the first
# four on import; ./python sets the library path), recorded in every output.
SETTINGS_ENV = ("CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_HOME", "NUMBA_FORCE_CUDA_CC",
                "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                "OMP_NUM_THREADS")


def refuse(message: str):
    raise SystemExit(f"binary_test: refused: {message}")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(path: Path) -> str | None:
    """SHA-256 of a file, None if it is missing or unreadable."""
    try:
        return binary_paths.sha256_file(Path(path))
    except OSError:
        return None


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, data) -> None:
    evaluate.write_json(path, data)


# ----------------------------------------------------------------------------- recorded settings

def scorer_hashes() -> dict[str, str | None]:
    return {key: digest(tp.scorer_path(key)) for key in tp.SCORER_HASHES}


def scorer_record() -> dict:
    """The scoring sources' current hashes (and binary_test_paths.py's), recorded in every output."""
    return {"hashes": scorer_hashes(), "binary_test_paths_sha256": digest(Path(tp.__file__)),
            "modules": {name: getattr(sys.modules.get(name), "__file__", None) for name in tp.SCORER_MODULES}}


def numerical_settings() -> dict:
    """cuDNN flags, the relevant environment and the library versions of this process. binary_test
    sets no cuDNN flags of its own: evaluate.strict_fp32 keeps the process's deterministic and
    benchmark values, so these are what the dev check and the test decode with."""
    import nemo
    import transformers
    try:
        cudnn = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001  no cuDNN library: recorded as None
        cudnn = None
    return {"cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "env": {key: os.environ.get(key) for key in SETTINGS_ENV},
            "versions": {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "cudnn": cudnn,
                         "nemo": nemo.__version__, "transformers": transformers.__version__,
                         "python": sys.version.split()[0]}}


def identity() -> dict:
    """The pinned MB1 identity (binary_test_paths), with the recorded per-set dev WERs."""
    return {"run_name": tp.RUN_NAME, "run": str(tp.RUN), "pinned_files": dict(tp.PINNED_FILES),
            "weights_sha256": tp.WEIGHTS_SHA256, "dev_mean_wer": tp.DEV_MEAN,
            "dev_wer": read_json(tp.RUN / "summary.json")["dev_wer"]}


def manifest_hashes() -> dict[str, str | None]:
    """Current SHA-256 of every dev and test manifest (evaluate.resolve_manifest)."""
    return {name: digest(evaluate.resolve_manifest(name)[1]) for name in [*paths.DEV_SETS, *paths.TEST_SETS]}


# ----------------------------------------------------------------------------- unit state

def journal_messages(unit: str) -> list[str] | None:
    """Every message of the unit in the user journal, None if the query failed."""
    out = subprocess.run(["journalctl", "--user", "-u", unit, "--no-pager", "-o", "cat"],
                         capture_output=True, text=True)
    return None if out.returncode != 0 else [line for line in out.stdout.splitlines() if line.strip()]


def active_units() -> list[str] | None:
    """Active parakeet-* user units, None if the query failed."""
    out = subprocess.run(["systemctl", "--user", "list-units", "--state=active", "--no-legend", "--plain",
                          "parakeet-*"], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return [line.split()[0] for line in out.stdout.splitlines() if line.strip()]


def own_unit() -> str | None:
    """The systemd service this process runs in (./heavy's parakeet-binary-NAME.service), if any."""
    try:
        text = Path("/proc/self/cgroup").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        last = line.rstrip().rsplit("/", 1)[-1]
        if last.endswith(".service"):
            return last
    return None


# ----------------------------------------------------------------------------- precheck

def precheck() -> dict:
    """Plan "Preconditions" items 1-6; {"pass", "checks": [{"item", "check", "ok", "detail"}], ...}.
    Reads only; uses no GPU. An unreadable input fails its item (it never passes silently)."""
    checks: list[dict] = []

    def checker(item: int):
        def check(name: str, ok, detail="") -> None:
            checks.append({"item": item, "check": name, "ok": bool(ok), "detail": str(detail)[:400]})
        return check

    for item, run in ((1, _unit_checks), (2, _identity_checks), (3, _code_checks)):
        try:
            run(checker(item))
        except Exception as error:  # noqa: BLE001
            checker(item)("inputs present and readable", False, f"{type(error).__name__}: {error}")
    try:
        _baseline_checks(checker(4), checker(5), checker(6))
    except Exception as error:  # noqa: BLE001
        for item in (4, 5, 6):
            checker(item)("inputs present and readable", False, f"{type(error).__name__}: {error}")
    return {"pass": bool(checks) and all(c["ok"] for c in checks), "utc": utc(),
            "weights_sha256": tp.WEIGHTS_SHA256, "checks": checks}


def _unit_checks(check) -> None:
    """Item 1: unit success from durable evidence."""
    messages = journal_messages(tp.MAIN_UNIT)
    failed = [m for m in messages or [] if "Failed with result" in m]
    check(f"journal of {tp.MAIN_UNIT} readable, not empty, no 'Failed with result'",
          bool(messages) and not failed,
          "journal query failed or empty" if not messages else " | ".join(failed[:2]))
    lines = [line.rstrip() for line in tp.MAIN_LOG.read_text(errors="replace").splitlines()
             if line.startswith("[binary_sweep")]
    main_phase = any("] phase main;" in line for line in lines)
    check("logs/binary-main.log: the main phase's last '[binary_sweep ...]' line is 'done'",
          main_phase and bool(lines) and lines[-1].endswith("] done"),
          (lines[-1][:120] if lines else "no [binary_sweep lines") + ("" if main_phase else "; no 'phase main' line"))
    units, own = active_units(), own_unit()
    others = None if units is None else [u for u in units if u != own]
    check("no parakeet-* unit active other than the caller's own (query succeeded)", others == [],
          "systemctl query failed" if others is None else ", ".join(others))


def _identity_checks(check) -> None:
    """Item 2: final-MB1 identity."""
    for rel, want in tp.PINNED_FILES.items():
        got = digest(tp.RUN / rel)
        check(f"pinned SHA-256 of {rel}", got == want, got)
    entries = {p.relative_to(tp.EXPORT).as_posix() for p in tp.EXPORT.rglob("*")}
    check("export directory file set exact", entries == tp.EXPORT_ENTRIES,
          f"extra {sorted(entries - tp.EXPORT_ENTRIES)}, missing {sorted(tp.EXPORT_ENTRIES - entries)}")
    manifest = read_json(tp.EXPORT / "manifest.json")
    check(f"manifest format == {tp.FORMAT}", manifest.get("format") == tp.FORMAT, repr(manifest.get("format")))
    check(f"manifest file == {tp.WEIGHTS_FILE}", manifest.get("file") == tp.WEIGHTS_FILE, repr(manifest.get("file")))
    check("manifest sha256 == the pinned weight hash", manifest.get("sha256") == tp.WEIGHTS_SHA256,
          manifest.get("sha256"))
    entry = (manifest.get("files") or {}).get(tp.WEIGHTS_FILE)
    check(f"manifest files[{tp.WEIGHTS_FILE}] == the pinned weight hash", entry == tp.WEIGHTS_SHA256, entry)
    summary = read_json(tp.RUN / "summary.json")
    for key, want in tp.EXPECT_SUMMARY.items():
        got = summary.get(key, "<missing>")
        check(f"summary.{key} == {want!r}", key in summary and got == want and not isinstance(got, bool), repr(got))
    binary = summary.get("binary") or {}
    for key, want in tp.EXPECT_BINARY.items():
        check(f"summary.binary.{key} == {want!r}", binary.get(key) == want, repr(binary.get(key)))
    check("summary.config_sha256 == the pinned config.json hash",
          summary.get("config_sha256") == tp.PINNED_FILES["config.json"], summary.get("config_sha256"))
    check(f"summary.dev_mean_wer == the pinned {tp.DEV_MEAN!r}", summary.get("dev_mean_wer") == tp.DEV_MEAN,
          repr(summary.get("dev_mean_wer")))
    check("summary.dev_wer covers the five dev sets", set(summary.get("dev_wer") or {}) == set(paths.DEV_SETS),
          sorted(summary.get("dev_wer") or {}))
    resumes = tp.RUN / "resumes.jsonl"
    lines = [line for line in resumes.read_text().splitlines() if line.strip()] if resumes.exists() else []
    check("no resumes recorded (resumes.jsonl absent or empty)", not lines, f"{len(lines)} entries")
    recon = read_json(tp.EXPORT / "reconstruction.json")
    exact = {k: recon.get(k) for k in ("codes_exact", "scales_exact", "greedy_hyps_equal")}
    check("reconstruction codes, scales and greedy hypotheses exact", all(v is True for v in exact.values()), exact)
    check("summary.reconstruction equals reconstruction.json", summary.get("reconstruction") == recon)


def module_files() -> list[tuple[str, object, Path]]:
    """(module name, loaded module or None, the file it must come from)."""
    out = [(name, sys.modules.get(name), tp.scorer_path(key)) for name, key in tp.SCORER_MODULES.items()]
    out.append(("binary_test", sys.modules[__name__], tp.scorer_path("parakeet-binary/binary_test.py")))
    out.append(("binary_test_paths", tp, tp.HERE / "binary_test_paths.py"))
    return out


def plan_status() -> str:
    """The plan's bold Status paragraph ("**Status: ... **")."""
    text = tp.PLAN.read_text()
    start = text.find("**Status:")
    end = text.find("**", start + len("**Status:")) if start >= 0 else -1
    return text[start:end] if end > start else ""


def _code_checks(check) -> None:
    """Item 3: MB1's recorded sources unchanged; every pinned scorer file and module path."""
    recorded = read_json(tp.RUN / "summary.json").get("source_hashes") or {}
    changed = [key for key, value in recorded.items() if digest(tp.FINETUNE / key) != value]
    check("sources recorded in MB1's summary.json unchanged", recorded and not changed, ", ".join(changed))
    for key, want in tp.SCORER_HASHES.items():
        got = digest(tp.scorer_path(key))
        check(f"scorer file {key} matches SCORER_HASHES", bool(want) and got == want, got)
    check("parakeet-ternary/evaluate.py is the frozen evaluator",
          digest(tp.FINETUNE / "parakeet-ternary" / "evaluate.py") == tp.EVALUATE_SHA256)
    for name, module, path in module_files():
        file = getattr(module, "__file__", None)
        check(f"module {name} imported from {path}",
              file is not None and Path(file).resolve() == path.resolve(), file)
    check("evaluate.NORMALIZER_JSON is the pinned normalizer.json",
          Path(evaluate.NORMALIZER_JSON).resolve() == tp.scorer_path(tp.NORMALIZER_JSON).resolve(),
          evaluate.NORMALIZER_JSON)
    own = digest(Path(tp.__file__))
    check("binary_test_paths.py SHA-256 is in the plan's Status line", own is not None and own in plan_status(), own)


def load_outputs(directory: Path) -> tuple[dict | None, dict[str, dict | None]]:
    """(summary.json, {test set: per-set output}) of an evaluate.py output; None where unreadable."""
    def load(path: Path):
        try:
            return read_json(path)
        except (OSError, ValueError):
            return None
    return load(directory / "summary.json"), {name: load(directory / f"{name}.json") for name in paths.TEST_SETS}


def _baseline_checks(check4, check5, check6) -> None:
    """Items 4 (corpus), 5 (normalizer compatibility) and 6 (reused arms)."""
    arms = {"M1-test": load_outputs(tp.M1_TEST), "b0-pretrained": load_outputs(tp.B0_TEST)}
    cache: dict[str, str] = {}

    def norm(text: str) -> str:
        if text not in cache:
            cache[text] = evaluate.normalize(text)
        return cache[text]

    for name in paths.TEST_SETS:
        manifest = evaluate.resolve_manifest(name)[1]
        current = digest(manifest)
        recorded = {arm: (docs.get(name) or {}).get("manifest_sha256") for arm, (_, docs) in arms.items()}
        check4(f"{name}: manifest SHA-256 equals M1-test's and b0-pretrained's",
               current is not None and all(r == current for r in recorded.values()), f"current {current}, {recorded}")
        ids = [r["id"] for r in evaluate.read_manifest(manifest)] if current else []
        check4(f"{name}: manifest utterance IDs unique", ids and len(ids) == len(set(ids)),
               f"{len(ids) - len(set(ids))} duplicates")
        for arm, (_, docs) in arms.items():
            doc = docs[name]
            records = (doc or {}).get("records") or []
            bad = [r.get("id") for r in records
                   if norm(r["ref"]) != r.get("ref_norm") or norm(r["hyp"]) != r.get("hyp_norm")]
            check5(f"{arm} {name}: the current normalizer reproduces every saved ref_norm and hyp_norm",
                   records and not bad, "no records" if not records else f"{len(bad)} differ, first {bad[:3]}")
    for arm, (summary, docs) in arms.items():
        listed = set((summary or {}).get("sets") or {})
        partial = [n for n, d in docs.items() if d is None or d.get("set") != n or d.get("limit") is not None
                   or ((summary or {}).get("sets") or {}).get(n, {}).get("limit") is not None]
        check6(f"{arm}: all 8 test sets, no --limit", summary is not None and listed == set(paths.TEST_SETS)
               and not partial, f"summary sets {sorted(listed)}, missing or partial {partial}")
    (m1_summary, m1), (b0_summary, b0) = arms["M1-test"], arms["b0-pretrained"]
    differ = [n for n in paths.TEST_SETS if not (m1[n] and b0[n] and m1[n].get("decoding") == b0[n].get("decoding")
                                                 and m1[n].get("normalizer") == b0[n].get("normalizer")
                                                 == evaluate.NORMALIZER)]
    check6("M1-test and b0-pretrained: same decoding and normalizer records (the frozen evaluator's normalizer)",
           not differ, ", ".join(differ))
    batch = [n for n in paths.TEST_SETS if m1[n] and ((m1[n].get("decoding") or {}).get("batch_size"),
                                                      (m1[n].get("decoding") or {}).get("max_batch_seconds"))
             != (tp.BATCH_SIZE, tp.MAX_BATCH_SECONDS)]
    check6(f"M1-test decoding at batch {tp.BATCH_SIZE}, max_batch_seconds {tp.MAX_BATCH_SECONDS:g}", not batch,
           ", ".join(batch))
    m1_sources = [(m1_summary or {}).get("source") or {}] + [(d or {}).get("source") or {} for d in m1.values()]
    shas = {s.get("sha256") for s in m1_sources}
    check6("M1-test: source SHA-256 is M1's pinned export (summary and every set)",
           shas == {tp.M1_EXPORT_SHA256}, sorted(map(str, shas)))
    check6("M1-test: source base_model is the pinned base model",
           all(s.get("base_model") == tp.BASE_MODEL for s in m1_sources))
    b0_sources = [(b0_summary or {}).get("source") or {}] + [(d or {}).get("source") or {} for d in b0.values()]
    check6("b0-pretrained: the pinned pretrained model (kind, revision, .nemo SHA-256), as M1-test's base_model",
           all(s.get("kind") == "pretrained" and s.get("sha256") == tp.BASE_MODEL["nemo_sha256"]
               and s.get("base_model") == tp.BASE_MODEL for s in b0_sources),
           sorted({f"{s.get('kind')} {s.get('sha256')} {(s.get('base_model') or {}).get('revision')}"
                   for s in b0_sources}))


def require_precheck() -> dict:
    result = precheck()
    if not result["pass"]:
        failing = [f"[{c['item']}] {c['check']} ({c['detail']})" for c in result["checks"] if not c["ok"]]
        refuse(f"precheck failed ({len(failing)} checks): " + "; ".join(failing[:6]))
    return result


def run_precheck() -> int:
    result = precheck()
    binary_paths.require_mount()
    tp.PRECHECK.parent.mkdir(parents=True, exist_ok=True)
    write_json(tp.PRECHECK, result)
    for c in result["checks"]:
        print(f"{'PASS' if c['ok'] else 'FAIL'}  [{c['item']}] {c['check']}"
              + (f"  [{c['detail']}]" if c["detail"] and not c["ok"] else ""))
    print("ALL PASS" if result["pass"] else "PRECHECK FAILED", "->", tp.PRECHECK)
    return 0 if result["pass"] else 1


# ----------------------------------------------------------------------------- loading and scoring

def require_device() -> torch.device:
    device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        refuse("CUDA is required")
    return device


def source_record() -> dict:
    """evaluate.describe_source("export", ...) plus the binary manifest and the pinned identity;
    refuses unless the weights it hashed are the pinned ones."""
    source = evaluate.describe_source("export", tp.EXPORT, tp.SOURCE_DESCRIPTION)
    if source.get("sha256") != tp.WEIGHTS_SHA256:
        refuse(f"{tp.EXPORT} weights SHA-256 {source.get('sha256')} is not the pinned {tp.WEIGHTS_SHA256}")
    manifest = read_json(tp.EXPORT / "manifest.json")
    source["export_manifest"] = {k: v for k, v in manifest.items()
                                 if k in ("format", "file", "sha256", "files", "extra", "parameter_accounting", "bytes")}
    source["pinned_identity"] = identity()
    return source


def load_model(device: torch.device):
    """(the MB1 export rebuilt by binary_export.load_export, its pinned decoder mode)."""
    model = binary_export.load_export(tp.EXPORT, device)
    return model, binary_train.pin_decoder_graphs_mode(model, "MB1 export")


def score_sets(model, sets: list[tuple[str, Path]], source: dict, out_dir: Path, batch_size: int,
               limit: int | None, device: torch.device, label: str, settings: dict, on_done=None) -> dict[str, dict]:
    """evaluate.evaluate_set per set, the decoder mode checked before and after each; writes
    out_dir/<set>.json and calls on_done(set, path) after each. `settings` were recorded before the
    model was loaded; they must still hold before every set."""
    scorer = scorer_record()
    results = {}
    for name, manifest in sets:
        if numerical_settings() != settings:
            refuse(f"numerical settings changed since start-up (before {name}): {numerical_settings()}")
        before = binary_train.require_decoder_graphs_mode(model, f"MB1 export before {name}")
        result = evaluate.evaluate_set(model, name, manifest, source, batch_size, limit, device,
                                       max_batch_seconds=tp.MAX_BATCH_SECONDS)
        after = binary_train.require_decoder_graphs_mode(model, f"MB1 export after {name}")
        result.update(decoder_graphs_mode={"before": before, "after": after}, scorer=scorer, settings=settings,
                      binary_test={"mode": label, "batch_size": batch_size,
                                   "max_batch_seconds": tp.MAX_BATCH_SECONDS, "limit": limit})
        path = out_dir / f"{name}.json"
        write_json(path, result)
        results[name] = result
        w = result["wer"]
        print(f"{name}: n={result['utterances']} WER {100 * w['wer']:.3f}% {result['wall_s']:.0f} s", flush=True)
        if on_done is not None:
            on_done(name, path)
    return results


def summary_of(results: dict[str, dict], out_dir: Path) -> dict:
    """evaluate.summarize plus the per-set file hashes and the recorded scorer and settings."""
    first = next(iter(results.values()))
    if any(r["source"] != first["source"] for r in results.values()):
        refuse(f"{out_dir}: the per-set outputs record different sources")
    summary = evaluate.summarize(results, first["source"])
    summary.update(per_set_sha256={n: digest(out_dir / f"{n}.json") for n in results},
                   scorer=first["scorer"], settings=first["settings"], binary_test=first["binary_test"],
                   decoder_graphs_mode=sorted({r["decoder_graphs_mode"]["after"] for r in results.values()}))
    return summary


def manifests_for(names) -> list[tuple[str, Path]]:
    sets = [evaluate.resolve_manifest(n) for n in names]
    missing = [str(p) for _, p in sets if not p.exists()]
    if missing:
        refuse(f"missing manifests: {missing}")
    return sets


# ----------------------------------------------------------------------------- dev cross-check

def within(diff: float, tolerance: float) -> bool:
    """|diff| <= tolerance, with float noise below 1e-12 ignored (exactly at the tolerance passes)."""
    return round(abs(diff), 12) <= tolerance


def gate_verdict(measured: dict[str, float], recorded: dict[str, float], recorded_mean: float) -> dict:
    """The dev cross-check on WER fractions: mean within MEAN_TOLERANCE of the recorded mean and every
    set within SET_TOLERANCE of its recorded value."""
    names = list(paths.DEV_SETS)
    if set(measured) != set(names) or set(recorded) != set(names):
        raise ValueError(f"dev sets {sorted(measured)} / {sorted(recorded)} are not {names}")
    per_set = {}
    for name in names:
        diff = measured[name] - recorded[name]
        per_set[name] = {"wer": measured[name], "recorded": recorded[name], "diff": diff, "diff_points": 100 * diff,
                         "within": within(diff, tp.SET_TOLERANCE)}
    mean = sum(measured[n] for n in names) / len(names)
    mean_diff = mean - recorded_mean
    mean_within = within(mean_diff, tp.MEAN_TOLERANCE)
    return {"pass": mean_within and all(e["within"] for e in per_set.values()), "mean": mean,
            "recorded_mean": recorded_mean, "mean_diff": mean_diff, "mean_diff_points": 100 * mean_diff,
            "mean_within": mean_within, "per_set": per_set,
            "tolerances": {"mean_points": tp.MEAN_TOLERANCE_POINTS, "set_points": tp.SET_TOLERANCE_POINTS,
                           "mean_fraction": tp.MEAN_TOLERANCE, "set_fraction": tp.SET_TOLERANCE}}


def dev_check(limit: int | None, batch_size: int) -> int:
    """Decode the five dev sets into DEV_CHECK (batch 64), DEV_CHECK_B32 (the batch-32 diagnostic) or a
    smoke directory (--limit); gate.json last. Exit 0 if the gate passed (smoke runs: always 0)."""
    if batch_size not in (tp.BATCH_SIZE, tp.DIAGNOSTIC_DEV_BATCH):
        refuse(f"--dev-batch-size {batch_size}: only {tp.BATCH_SIZE} or the diagnostic {tp.DIAGNOSTIC_DEV_BATCH}")
    if limit is not None and limit < 1:
        refuse("--limit must be positive")
    out = tp.smoke_dir(limit, batch_size) if limit is not None else (
        tp.DEV_CHECK if batch_size == tp.BATCH_SIZE else tp.DEV_CHECK_B32)

    def refuse_if_gated() -> None:
        if limit is None and (out / tp.GATE).exists():
            refuse(f"{out / tp.GATE} exists; a dev check is not rerun into the same directory "
                   "(move it aside by hand, with the reason)")

    refuse_if_gated()  # early; repeated under the GPU lock
    require_precheck()
    sets = manifests_for(paths.DEV_SETS)
    device = require_device()
    label = "dev-check" if limit is None else "dev-smoke"
    settings = numerical_settings()
    with paths.gpu_lock(f"binary_test:{label}"):
        refuse_if_gated()  # a concurrent dev check may have completed while this one waited
        source = source_record()
        binary_paths.require_mount()
        out.mkdir(parents=True, exist_ok=True)
        for stale in out.glob("*.json"):  # an unfinished earlier attempt (no gate.json) or a smoke rerun
            stale.unlink()
        results, mode = decode_and_release(device, sets, source, out, batch_size, limit, label, settings)
        write_json(out / "summary.json", summary_of(results, out))
        recorded = read_json(tp.RUN / "summary.json")["dev_wer"]
        verdict = gate_verdict({n: results[n]["wer"]["wer"] for n in paths.DEV_SETS}, recorded, tp.DEV_MEAN)
        gate = {**verdict, "within_tolerance": verdict["pass"], "pass": verdict["pass"] and limit is None,
                "test_launch_gate": limit is None and batch_size == tp.BATCH_SIZE,
                "limit": limit, "batch_size": batch_size, "max_batch_seconds": tp.MAX_BATCH_SECONDS,
                "out_dir": str(out), "decoder_graphs_mode": mode, "identity": identity(),
                "scorer": scorer_record(), "manifests": manifest_hashes(), "settings": settings,
                "per_set_sha256": {n: digest(out / f"{n}.json") for n in paths.DEV_SETS},
                "summary_sha256": digest(out / "summary.json"), "utc": utc()}
        write_json(out / tp.GATE, gate)
    for name, e in gate["per_set"].items():
        print(f"{name:22s} {100 * e['wer']:.3f}% recorded {100 * e['recorded']:.3f}% "
              f"diff {e['diff_points']:+.3f} points {'ok' if e['within'] else 'OUTSIDE 0.10'}")
    print(f"mean {100 * gate['mean']:.4f}% recorded {100 * gate['recorded_mean']:.4f}% diff "
          f"{gate['mean_diff_points']:+.4f} points; gate {'PASS' if gate['pass'] else 'FAIL'}"
          + ("" if gate["test_launch_gate"] else " (not a test-launch gate)") + f" -> {out / tp.GATE}")
    return 0 if gate["pass"] or limit is not None else 1


def require_gate() -> tuple[dict, str]:
    """(the production dev-check gate, the SHA-256 of exactly the bytes validated): read once, passed and
    bound to this process; refuses otherwise."""
    path = tp.DEV_CHECK / tp.GATE
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        refuse(f"no dev-check gate {path}; run --dev-check first")
    try:
        gate = json.loads(data)
    except ValueError:
        refuse(f"dev-check gate {path} is not JSON")
    if gate.get("limit") is not None:
        refuse(f"dev-check gate {path} is a --limit {gate.get('limit')} run")
    if gate.get("batch_size") != tp.BATCH_SIZE:
        refuse(f"dev-check gate {path} decoded at batch {gate.get('batch_size')}, not {tp.BATCH_SIZE}")
    if gate.get("out_dir") != str(tp.DEV_CHECK) or gate.get("test_launch_gate") is not True:
        refuse(f"dev-check gate {path} is not the production dev check ({gate.get('out_dir')})")
    if gate.get("pass") is not True:
        refuse(f"dev-check gate {path} did not pass; the test run cannot start")
    for key, current in (("identity", identity()), ("scorer", scorer_record()), ("manifests", manifest_hashes()),
                         ("settings", numerical_settings())):
        if gate.get(key) != current:
            refuse(f"dev-check gate {path} records other {key} than this process")
    return gate, hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------------------- attempts

@contextlib.contextmanager
def attempt_lock(wait: bool, label: str):
    """Exclusive flock on ATTEMPTS_LOCK for one attempt transition. wait=False refuses if it is held."""
    binary_paths.require_mount()
    tp.ATTEMPTS_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(tp.ATTEMPTS_LOCK, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if not wait:
                refuse(f"another --test, --record-failure or --summarize holds the attempt lock {tp.ATTEMPTS_LOCK} "
                       "(an attempt may be running); retry after it ends")
            print(f"waiting for the attempt lock {tp.ATTEMPTS_LOCK} ...", flush=True)
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


def free_gpu_memory() -> None:
    """Collect unreachable objects (NeMo models hold reference cycles) and flush the CUDA cache."""
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def decode_and_release(device, sets, source, out_dir, batch_size, limit, label, settings, on_done=None):
    """(results, decoder mode): load the MB1 export and score the sets. The model and its decoder exist
    only in _load_and_score's frames; on an exception the frames of its traceback chain are cleared
    (they hold the model) before it propagates. GPU memory is collected and flushed before returning,
    while the caller still holds the GPU lock."""
    try:
        return _load_and_score(device, sets, source, out_dir, batch_size, limit, label, settings, on_done)
    except BaseException as error:
        seen: set[int] = set()
        while error is not None and id(error) not in seen:
            seen.add(id(error))
            traceback.clear_frames(error.__traceback__)  # skips frames still executing
            error = error.__cause__ or error.__context__
        raise
    finally:
        free_gpu_memory()


def _load_and_score(device, sets, source, out_dir, batch_size, limit, label, settings, on_done):
    model, mode = load_model(device)
    results = score_sets(model, sets, source, out_dir, batch_size, limit, device, label, settings, on_done)
    return results, mode


def read_attempts() -> list[dict]:
    if not tp.ATTEMPTS.exists():
        return []
    entries = []
    for number, line in enumerate(tp.ATTEMPTS.read_text().splitlines(), 1):
        if line.strip():
            try:
                entries.append(json.loads(line))
            except ValueError:
                refuse(f"{tp.ATTEMPTS} line {number} is not JSON")
    return entries


def attempt_states(entries: list[dict]) -> dict[int, dict]:
    """{attempt: {"attempt", "started", "set_done": {set: sha256}, "failed", "complete"}}."""
    out: dict[int, dict] = {}
    for e in entries:
        n, state = e.get("attempt"), e.get("state")
        if not isinstance(n, int) or state not in STATES:
            refuse(f"{tp.ATTEMPTS}: unreadable entry {e}")
        a = out.setdefault(n, {"attempt": n, "started": False, "set_done": {}, "failed": False, "complete": False})
        if state == "set_done":
            if e.get("set") in a["set_done"]:
                refuse(f"{tp.ATTEMPTS}: attempt {n} records {e.get('set')} twice")
            a["set_done"][e.get("set")] = e.get("sha256")
        else:
            a[state] = True
    return out


def all_done(attempt: dict) -> bool:
    return set(attempt["set_done"]) == set(paths.TEST_SETS)


def append_attempt(entry: dict) -> None:
    """Append one line to the attempt record (append-only, flushed and fsynced)."""
    binary_paths.require_mount()
    tp.ATTEMPTS.parent.mkdir(parents=True, exist_ok=True)
    with open(tp.ATTEMPTS, "a") as handle:
        handle.write(json.dumps(entry) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_test_summary(results: dict[str, dict], attempt: int, written_by: str) -> None:
    summary = summary_of(results, tp.TEST_OUT)
    summary.update(attempt=attempt, written_by=written_by, written_utc=utc())
    write_json(tp.TEST_OUT / "summary.json", summary)


def run_test() -> int:
    """The single test-set run of MB1 into TEST_OUT, under the attempt lock (waits for it)."""
    with attempt_lock(wait=True, label="--test"):
        return _run_test_locked()


def _run_test_locked() -> int:
    attempts = attempt_states(read_attempts())  # read after the attempt lock: never a stale view
    for a in attempts.values():
        if a["complete"]:
            refuse(f"attempt {a['attempt']} is complete: MB1 has been scored on test")
    for a in attempts.values():
        if all_done(a):
            refuse(f"attempt {a['attempt']} has all eight sets done; only --summarize is allowed")
    for a in attempts.values():
        if not a["failed"]:
            refuse(f"attempt {a['attempt']} started without a failed record: investigate by hand, then "
                   "--record-failure \"<evidence>\"")
    require_precheck()

    def refuse_if_output_exists() -> None:
        if tp.TEST_OUT.exists():
            refuse(f"{tp.TEST_OUT} exists without an open attempt; investigate by hand")

    require_gate()  # early refusals; both repeated under the GPU lock
    refuse_if_output_exists()
    sets = manifests_for(paths.TEST_SETS)
    device = require_device()
    n = 1 + max(attempts, default=0)
    settings = numerical_settings()  # = the gate's (require_gate)
    with paths.gpu_lock("binary_test:test"):  # held through finalization and the model's release
        gate, gate_sha256 = require_gate()  # the bytes validated here are the ones recorded
        refuse_if_output_exists()
        source = source_record()
        append_attempt({"attempt": n, "state": "started", "utc": utc(), "pid": os.getpid(), "unit": own_unit(),
                        "gate_sha256": gate_sha256, "gate_utc": gate.get("utc")})
        tp.TEST_OUT.mkdir(parents=True)
        results, _ = decode_and_release(device, sets, source, tp.TEST_OUT, tp.BATCH_SIZE, None, "test", settings,
                                        on_done=lambda name, path: append_attempt(
                                            {"attempt": n, "set": name, "state": "set_done", "sha256": digest(path)}))
        write_test_summary(results, n, "test")
        append_attempt({"attempt": n, "state": "complete", "utc": utc()})
    print(f"MB1 test attempt {n} complete -> {tp.TEST_OUT}")
    return 0


def record_failure(evidence: str) -> int:
    """Record the open attempt as failed (after investigation by hand) and move its partial output aside."""
    if not evidence.strip():
        refuse("--record-failure needs the evidence (what failed, where it is logged)")
    with attempt_lock(wait=False, label="--record-failure"):
        return _record_failure_locked(evidence)


def _record_failure_locked(evidence: str) -> int:
    attempts = attempt_states(read_attempts())
    open_ = [a for a in attempts.values() if not a["failed"] and not a["complete"]]
    if not open_:
        refuse("no open attempt to record a failure for")
    a = max(open_, key=lambda x: x["attempt"])
    if all_done(a):
        refuse(f"attempt {a['attempt']} has all eight sets done; use --summarize, not --record-failure")
    renamed = None
    if tp.TEST_OUT.exists():
        renamed = tp.EVAL / f"{tp.TEST_OUT.name}.failed-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        if renamed.exists():
            refuse(f"{renamed} exists")
        os.rename(tp.TEST_OUT, renamed)
    append_attempt({"attempt": a["attempt"], "state": "failed", "evidence": evidence,
                    "renamed_to": None if renamed is None else str(renamed), "utc": utc()})
    print(f"attempt {a['attempt']} recorded as failed" + (f"; partial output moved to {renamed}" if renamed else ""))
    return 0


def summarize() -> int:
    """Rebuild TEST_OUT/summary.json from the complete per-set files of the attempt with all eight sets
    done (hashes checked against its set_done records), then append "complete" once. No decoding."""
    with attempt_lock(wait=False, label="--summarize"):
        return _summarize_locked()


def _summarize_locked() -> int:
    attempts = attempt_states(read_attempts())
    complete = [a for a in attempts.values() if a["complete"]]
    if complete:
        print(f"attempt {complete[0]['attempt']} is already complete; nothing to do")
        return 0
    done = [a for a in attempts.values() if all_done(a) and not a["failed"]]
    if not done:
        refuse("no attempt has set_done records for all eight test sets")
    a = max(done, key=lambda x: x["attempt"])
    results = {}
    for name in paths.TEST_SETS:
        path = tp.TEST_OUT / f"{name}.json"
        got = digest(path)
        if got != a["set_done"][name]:
            refuse(f"{name}: {path} SHA-256 {got} != attempt {a['attempt']}'s set_done {a['set_done'][name]}")
        results[name] = read_json(path)
    if scorer_hashes() != tp.SCORER_HASHES:
        refuse("the current scoring sources differ from SCORER_HASHES")
    for name, doc in results.items():
        if (doc.get("scorer") or {}).get("hashes") != tp.SCORER_HASHES:
            refuse(f"{name}: recorded scorer hashes differ from SCORER_HASHES")
    write_test_summary(results, a["attempt"], "summarize")
    append_attempt({"attempt": a["attempt"], "state": "complete", "utc": utc(), "by": "summarize"})
    print(f"attempt {a['attempt']} summarized and complete -> {tp.TEST_OUT / 'summary.json'}")
    return 0


# ----------------------------------------------------------------------------- CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--precheck", action="store_true", help="plan 'Preconditions'; read-only, no GPU")
    mode.add_argument("--dev-check", action="store_true", help="the dev cross-check (launch gate for --test)")
    mode.add_argument("--test", action="store_true", help="the single test-set run")
    mode.add_argument("--summarize", action="store_true", help="rebuild summary.json of a fully decoded attempt")
    mode.add_argument("--record-failure", metavar="EVIDENCE", help="record the open attempt as failed")
    parser.add_argument("--limit", type=int, help="--dev-check only: first N id-sorted utterances per set (smoke)")
    parser.add_argument("--dev-batch-size", type=int, choices=(tp.BATCH_SIZE, tp.DIAGNOSTIC_DEV_BATCH),
                        help="--dev-check only: 32 is the diagnostic reproduction (eval/MB1-dev-check-b32)")
    args = parser.parse_args(argv)
    if args.limit is not None and not args.dev_check:
        parser.error("--limit is accepted only with --dev-check: MB1 is scored on test exactly once")
    if args.dev_batch_size is not None and not args.dev_check:
        parser.error("--dev-batch-size is accepted only with --dev-check")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    binary_paths.require_mount()
    if args.precheck:
        return run_precheck()
    if args.dev_check:
        return dev_check(args.limit, args.dev_batch_size or tp.BATCH_SIZE)
    if args.test:
        return run_test()
    if args.summarize:
        return summarize()
    return record_failure(args.record_failure)


if __name__ == "__main__":
    sys.exit(main())
