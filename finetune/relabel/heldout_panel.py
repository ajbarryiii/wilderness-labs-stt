"""Held-out-source panel, step 1 (heldout-panel plan): every model we have on the ten domain sets; this scores the gaps.

    ./python-gpu1bit heldout_panel.py score --model b100-main|b100-control [--limit N] [options]
    ./python heldout_panel.py score --model B0p|tstar|tstar-encoder [--limit N] [options]
    ./python heldout_panel.py report        # heldout_report.py: report.json and tables.md

Sets (SETS; panels PANELS): the ten parts of devsets.SETS, by alias, read only from their complete production builds
(score_domain.part_manifests: the manifest must match its build meta). Nothing else is accepted.

Existing records are reused (heldout_report.location); SCORED lists the gaps. Each model is scored through the path
its existing records came from (Scorer subclasses):

  b100-main, b100-control  lectures-a and medical-b at x1 (X1Arm): stage 4's scoring view on the real i8x1 kernels
        (training/deployed.load_x1; per utterance at batch 1, deployed.score_records), with train_keab.set_config's
        configuration, under ./python-gpu1bit with CUBLAS_WORKSPACE_CONFIG unset (train_keab.deployed_environment).
        The view must be the one the arm's recorded x1 records name, and the loaded runtime the recorded one
        (runtime.json: export SHA-256, kernel, parts, provider, selection, decoder graphs mode).
  B0p    lectures-a and medical-b in FP32 (B0Prime): stage 3's score_run (train.deterministic_mode, train.load_nemo,
        the decoder pinned to the stages' CUDA-graph mode, evaluate.evaluate_set at batch 64 and 1200 s, configuration
        train_decoder._config). B0' is stage 3's b100 (the longest cooldown branch at the selected rate, sha
        497ac8ca...): the model.nemo SHA-256 must equal model.json's and every stage 3 domain and eval-stem record's.
  tstar  all ten sets (TStar): score_teacher.score_set with teachers.GraniteSpeech, at the diagnostic's revision and
        configuration (bfloat16, length-sorted batches of at most 32 clips and 600 s, 256 new tokens): the run
        configuration must equal the diagnostic summary's.
  tstar-encoder  all ten sets (TStarEncoder): encoder_probe_ctc.score_set (one clip per encoder pass), BPE head
        record only, after probe 1's implementation checks (encoder_probe_ctc.run_checks on its dev clips; exit 5 if
        one fails); the configuration must equal probe 1's summary and out_llm's SHA-256 its recorded one.

Every path uses evaluate.score's normalizer, edit counts and "scored" rule and evaluate.pad_short. Critical-term recall
is added as score_domain computes it (score_domain.term_recall with devsets.TERM_KIND's list). Each written record
gains "panel": its configuration (model, identity, set, manifest SHA-256, limit, scorer), the code commit and whether
the tracked tree was dirty, and CUBLAS_WORKSPACE_CONFIG as the process saw it.

Cross-check: after loading, each path re-scores one set of an existing record of the same model (crosscheck_input:
stem-chem for the x1 arms and B0p, yodas_dev for T* and its encoder) and compares: same configuration, same
utterances, WER within score_domain.CROSSCHECK_POINTS. A production run that fails it writes crosscheck.json and
nothing else (exit 6); a --limit smoke compares its first N utterances and only records the result.

Outputs: $RELABEL_ROOT/heldout-panel/<model>/<set>.json, crosscheck.json and run.json (this process: timings, peak
GPU and host memory, the job lease). --limit N writes to relabel_paths.SMOKE/heldout-panel/<model>-limit<N>/, never
the production path. A set already written with the same panel configuration is skipped unless --force.

GPU: the parakeet GPU lock (score_teacher.gpu_lock) is taken before anything touches CUDA (exit 3 if not free within
--lock-wait), the job's VRAM lease applied (score_teacher.vram_lease), and no model is loaded below
--min-available-gb of host RAM (exit 4). The job lease itself is the caller's.

Process environment: this module imports nothing that imports finetune/parakeet-ternary/train.py (which sets
CUBLAS_WORKSPACE_CONFIG) at module level; the x1 and B0p scorers import their stage's trainer when constructed, as
their stage's scoring processes did, and the T* scorers never do, as the diagnostic and probe 1 never did.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import relabel_paths

PLAN = "heldout-panel plan"
ROOT_NAME = "heldout-panel"
TEACHER = "granite-speech-5.0-2b"
B0P_RUN = "b100"
B0P_LABEL = "stage 3 b100: B0 fine-tuned on T* labels (A1, FP32)"   # stage 3's score_run description

SETS = ("tech-dev-1", "tech-dev-2", "stem-chem", "stem-cs", "stem-math",
        "med-dev-1", "med-dev-2", "med-dev-3", "lectures-a", "medical-b")
PANELS = {
    "T5": ("tech-dev-1", "tech-dev-2", "stem-chem", "stem-cs", "stem-math"),
    "T5-no-tech-dev-2": ("tech-dev-1", "stem-chem", "stem-cs", "stem-math"),
    "M3": ("med-dev-1", "med-dev-2", "med-dev-3"),
    "X2": ("lectures-a", "medical-b"),
}
PANEL_LABELS = {"T5": "T5", "T5-no-tech-dev-2": "T5 without tech-dev-2", "M3": "M3", "X2": "X2"}
X1_ARMS = ("b100-main", "b100-control")
EVAL_EXT = ("lectures-a", "medical-b")
SCORED = {"b100-main": EVAL_EXT, "b100-control": EVAL_EXT, "B0p": EVAL_EXT, "tstar": SETS, "tstar-encoder": SETS}
CROSSCHECK_STEM = "stem-chem"
CROSSCHECK_DEV = "yodas_dev"

LOCK_BUSY_EXIT = 3
LOW_MEMORY_EXIT = 4
CHECKS_FAILED_EXIT = 5
CROSSCHECK_FAILED_EXIT = 6


# --- paths and inputs -----------------------------------------------------------------------------------------------

def out_root() -> Path:
    return relabel_paths.ARTIFACTS / ROOT_NAME


def output_dir(model: str, limit: int | None) -> Path:
    """out_root()/<model> for a full run, relabel_paths.SMOKE/heldout-panel/<model>-limit<N> for a --limit smoke."""
    if model not in SCORED:
        raise ValueError(f"no panel scoring for {model!r}; one of {list(SCORED)}")
    if limit is None:
        return out_root() / model
    if limit <= 0:
        raise ValueError("--limit must be positive")
    out = relabel_paths.SMOKE / ROOT_NAME / f"{model}-limit{limit}"
    assert not out.is_relative_to(out_root())
    return out


def domain_manifests(names) -> dict[str, tuple[Path, str]]:
    """{set: (manifest, its SHA-256)} for the panel's sets only, from their complete production builds."""
    import score_domain

    names = list(names)
    unknown = [n for n in names if n not in SETS]
    if unknown:
        raise SystemExit(f"not a held-out panel set: {unknown} (the ten domain parts only)")
    got = score_domain.part_manifests(names, None)
    return {n: (got[n][0], got[n][1]["manifest_sha256"]) for n in names}


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def code_commit() -> dict:
    """The repository's HEAD and whether tracked files differ from it."""
    here = Path(__file__).resolve().parent

    def git(*args) -> str | None:
        out = subprocess.run(["git", "-C", str(here), *args], capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- scorers --------------------------------------------------------------------------------------------------------

class Scorer:
    """One model's existing scoring path. prepare() checks identities without CUDA; load() builds the model."""

    model: str
    identity: dict
    scorer: dict

    def prepare(self, limit: int | None, device_name: str) -> None:
        raise NotImplementedError

    def crosscheck_input(self) -> tuple[str, Path, Path]:
        """(set, manifest, the existing record of this model on it)."""
        raise NotImplementedError

    def load(self, device) -> None:
        raise NotImplementedError

    def score(self, name: str, manifest: Path, limit: int | None) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def info(self) -> dict:
        return {}


def _stem_manifest(name: str) -> Path:
    return domain_manifests([name])[name][0]


class X1Arm(Scorer):
    def __init__(self, model: str):
        if model not in X1_ARMS:
            raise ValueError(model)
        import train_keab as tk  # first: it records CUBLAS_WORKSPACE_CONFIG as the process started

        tk.deployed_environment()
        self.model, self.tk, self.net = model, tk, None

    def prepare(self, limit, device_name):
        from training import deployed

        deployed.require_wrapper()
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
            raise SystemExit("x1 is scored as the recorded x1 arm ran: unset CUBLAS_WORKSPACE_CONFIG")
        view = Path(self.tk.view_of(self.model))
        vman = read_json(view / "manifest.json")
        self.spec = {"name": self.model, "view": str(view), "view_sha256": vman["sha256"]}
        recorded_dir = Path(self.tk.scores_dir(self.model, "x1"))
        recorded = [read_json(recorded_dir / f"{n}.json")["config"]["model"]
                    for n in SETS if n not in EVAL_EXT and (recorded_dir / f"{n}.json").exists()]
        if not recorded or any(m != self.spec for m in recorded):
            raise SystemExit(f"{self.model}: scoring view {self.spec} is not the one its x1 records name")
        self.runtime = read_json(recorded_dir / "runtime.json")["runtime"]
        if self.runtime["export_sha256"] != vman["sha256"]:
            raise SystemExit(f"{self.model}: recorded runtime export {self.runtime['export_sha256']} != view")
        self.recorded_dir = recorded_dir
        self.identity = {"kind": "x1 scoring view", "view_sha256": vman["sha256"]}
        self.scorer = {"kind": "deployed", "mode": "x1", "batch": 1, "path": "train_keab x1 (training/deployed.py)"}
        self.record_identity = hashlib.sha256(json.dumps(self.spec, sort_keys=True).encode()).hexdigest()

    def crosscheck_input(self):
        return CROSSCHECK_STEM, _stem_manifest(CROSSCHECK_STEM), self.recorded_dir / f"{CROSSCHECK_STEM}.json"

    def load(self, device):
        from training import deployed

        self.net = deployed.load_x1(Path(self.spec["view"]))
        got = {k: self.net._keab_x1.get(k) for k in self.runtime}
        if got != self.runtime:
            self.close()
            raise SystemExit(f"{self.model}: the x1 runtime differs from the recorded arm: {got} != {self.runtime}")

    def score(self, name, manifest, limit):
        import devsets
        import evaluate
        from training import deployed

        group = devsets.set_of(name)
        records = list(enumerate(evaluate.read_manifest(manifest)[:limit]))
        result = deployed.score_records(self.net, records, name, self.record_identity)
        result.update(config=self.tk.set_config(self.spec, "x1", name, group, limit), group=group)
        return result

    def close(self):
        if self.net is not None:
            from training import deployed

            net, self.net = self.net, None
            deployed.close(net)

    def info(self):
        return {"runtime": None if self.net is None else
                {k: v for k, v in self.net._keab_x1.items() if k != "sources"}, "recorded_runtime": self.runtime}


class B0Prime(Scorer):
    model = "B0p"

    def __init__(self):
        import train_b0prime as tb  # imports train.py first, as stage 3's scorer did

        self.tb, self.net = tb, None

    def prepare(self, limit, device_name):
        import evaluate

        run = Path(self.tb.run_dir(B0P_RUN))
        nemo = run / "model.nemo"
        recorded = read_json(run / "model.json")["sha256"]
        self.source = evaluate.describe_source("nemo", nemo, B0P_LABEL)
        records = [p for sub in ("domain", "eval-stem") for p in sorted((run / sub).glob("*.json"))
                   if p.name != "summary.json"]
        shas = {read_json(p)["config"]["weights_sha256"] for p in records}
        if self.source["sha256"] != recorded or shas != {recorded}:
            raise SystemExit(f"B0p: model.nemo {self.source['sha256']}, model.json {recorded}, stage 3 records {shas}")
        self.run = run
        self.identity = {"kind": "weights", "run": B0P_RUN, "weights_sha256": self.source["sha256"]}
        self.scorer = {"kind": "evaluate_set", "batch": 64, "max_batch_seconds": 1200.0,
                       "path": "train_b0prime score_run (FP32)"}

    def crosscheck_input(self):
        return CROSSCHECK_STEM, _stem_manifest(CROSSCHECK_STEM), self.run / "eval-stem" / f"{CROSSCHECK_STEM}.json"

    def load(self, device):
        import score_domain
        import train

        score_domain.binary_on_path()
        import binary_train

        self.deterministic = train.deterministic_mode()
        self.device = device
        self.net = train.load_nemo(Path(self.source["path"]), str(device)).eval()
        self.mode = binary_train.pin_decoder_graphs_mode(self.net, B0P_RUN)

    def score(self, name, manifest, limit):
        import binary_train
        import devsets
        import evaluate
        import score_domain
        import train_decoder as td

        group = devsets.set_of(name)
        result = evaluate.evaluate_set(self.net, name, manifest, self.source, score_domain.BATCH_SIZE, limit,
                                       self.device, score_domain.MAX_BATCH_SECONDS)
        config = td._config(self.source, group, {"manifest_sha256": evaluate.sha256_file(manifest)})
        result.update(config=config, part=name, set=group,
                      decoder_graphs_mode=binary_train.require_decoder_graphs_mode(self.net, B0P_RUN))
        return result

    def close(self):
        self.net = None

    def info(self):
        return {"deterministic_mode": getattr(self, "deterministic", None), "decoder_graphs_mode":
                getattr(self, "mode", None)}


class TStar(Scorer):
    model = "tstar"

    def prepare(self, limit, device_name):
        import score_teacher
        import teachers

        self.teacher = teachers.get(TEACHER)
        self.args = argparse.Namespace(limit=limit, shortest=0, batch_size=self.teacher.batch_size,
                                       max_batch_seconds=self.teacher.max_batch_seconds, device=device_name)
        self.recorded_dir = relabel_paths.output_dir(TEACHER, None)
        diagnostic = read_json(self.recorded_dir / "summary.json")["config"]
        config = score_teacher.run_config(self.teacher, self.args)
        if config != {**diagnostic, "limit": limit}:
            raise SystemExit(f"T*: configuration {config} is not the diagnostic's {diagnostic}")
        self.identity = {"kind": "hub revision", "hf_id": self.teacher.hf_id, "revision": self.teacher.revision}
        self.scorer = {"kind": "score_teacher.score_set", "batch_size": self.args.batch_size,
                       "max_batch_seconds": self.args.max_batch_seconds, "path": "the diagnostic's teacher path"}

    def crosscheck_input(self):
        import evaluate

        manifest = evaluate.resolve_manifest(CROSSCHECK_DEV)[1]
        return CROSSCHECK_DEV, manifest, self.recorded_dir / f"{CROSSCHECK_DEV}.json"

    def load(self, device):
        self.device = device
        self.teacher.load(str(device))
        self.source = self.teacher.info()
        self.source["model_load_s"] = self.teacher.load_s

    def score(self, name, manifest, limit):
        import score_teacher

        return score_teacher.score_set(self.teacher, name, manifest, self.args, self.device, self.source)

    def close(self):
        self.teacher = None


class TStarEncoder(Scorer):
    model = "tstar-encoder"

    def prepare(self, limit, device_name):
        import encoder_probe_ctc as ep

        self.ep = ep
        self.args = argparse.Namespace(limit=limit, shortest=0, device=device_name)
        self.config = ep.run_config(self.args)
        self.recorded = read_json(ep.OUT / "summary.json")
        if self.config != {**self.recorded["config"], "limit": limit}:
            raise SystemExit(f"T* encoder: configuration {self.config} is not probe 1's {self.recorded['config']}")
        self.identity = {"kind": "hub revision", "hf_id": ep.HF_ID, "revision": ep.REVISION, "head": ep.BPE,
                         "conventions_sha256": self.config["conventions_sha256"],
                         "out_llm_sha256": self.recorded["source"]["out_llm"]["sha256"]}
        self.scorer = {"kind": "encoder_probe_ctc.score_set", "head": ep.BPE, "clips_per_pass": 1,
                       "path": "probe 1's scorer"}

    def crosscheck_input(self):
        import evaluate

        return (CROSSCHECK_DEV, evaluate.resolve_manifest(CROSSCHECK_DEV)[1],
                self.ep.OUT / self.ep.BPE / f"{CROSSCHECK_DEV}.json")

    def load(self, device):
        import evaluate

        self.probe, self.source = self.ep.load_probe(device)
        if self.source["out_llm"]["sha256"] != self.identity["out_llm_sha256"]:
            raise SystemExit(f"T* encoder: out_llm {self.source['out_llm']['sha256']} is not probe 1's")
        manifests = {n: evaluate.read_manifest(evaluate.resolve_manifest(n)[1]) for n in relabel_paths.SETS}
        self.checks = self.ep.run_checks(self.probe, manifests)
        print("checks: " + ", ".join(f"{k} {'pass' if self.checks.get(k, {}).get('passed') else 'FAIL'}"
                                     for k in ("hooks_passive", "pre_feedback", "determinism")), flush=True)
        if not self.checks["passed"]:
            raise ChecksFailed(self.checks)

    def score(self, name, manifest, limit):
        return self.ep.score_set(self.probe, name, manifest, self.args, self.config, self.source)[self.ep.BPE]

    def close(self):
        self.probe = None

    def info(self):
        return {"checks": getattr(self, "checks", None)}


class ChecksFailed(RuntimeError):
    def __init__(self, checks: dict):
        super().__init__("implementation checks failed")
        self.checks = checks


SCORERS = {"b100-main": lambda: X1Arm("b100-main"), "b100-control": lambda: X1Arm("b100-control"),
           "B0p": B0Prime, "tstar": TStar, "tstar-encoder": TStarEncoder}


# --- scoring ----------------------------------------------------------------------------------------------------------

def panel_config(model: str, scorer: Scorer, name: str, manifest_sha256: str, limit: int | None) -> dict:
    return {"plan": PLAN, "model": model, "identity": scorer.identity, "set": name, "manifest_sha256": manifest_sha256,
            "limit": limit, "scorer": scorer.scorer}


def _done(path: Path, config: dict) -> bool:
    return path.exists() and read_json(path).get("panel", {}).get("config") == config


def add_term_recall(result: dict, name: str) -> None:
    """Critical-term recall as score_domain adds it (devsets.TERM_KIND's list for the set's group)."""
    import devsets
    import score_domain
    import terms

    kind = devsets.TERM_KIND.get(devsets.set_of(name))
    if kind is not None and result.get("records"):
        result["term_recall"] = score_domain.term_recall(result["records"], terms.matcher(kind))
        result["term_kind"] = kind


def compare(new: dict, recorded: dict, name: str, recorded_path: Path, limit: int | None) -> dict:
    """The new scoring of an existing record's set against that record, on the utterances scored now."""
    import evaluate
    import score_domain

    old = {r["id"]: r for r in recorded["records"]}
    rows = new["records"]
    missing = [r["id"] for r in rows if r["id"] not in old]
    matched = [old[r["id"]] for r in rows if r["id"] in old]
    wer_new = 100 * evaluate.wer_summary(rows)["wer"]
    wer_old = 100 * evaluate.wer_summary(matched)["wer"] if matched else None
    difference = None if wer_old is None else wer_new - wer_old
    same_population = not missing and (limit is not None or len(rows) == len(old))
    config_equal = new.get("config") == recorded.get("config")
    within = difference is not None and abs(difference) <= score_domain.CROSSCHECK_POINTS
    return {"set": name, "recorded": str(recorded_path), "limit": limit, "recorded_wer": wer_old, "wer": wer_new,
            "difference_points": difference, "tolerance_points": score_domain.CROSSCHECK_POINTS,
            "identical_hypotheses": sum(r["hyp"] == old[r["id"]]["hyp"] for r in rows if r["id"] in old),
            "utterances": len(rows), "recorded_utterances": len(old), "missing_ids": missing[:20],
            "config_equal": config_equal, "pass": same_population and config_equal and within,
            "gating": limit is None}


def _peaks(device) -> dict:
    import torch

    cuda = device.type == "cuda"
    return {"peak_gpu_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30 if cuda else None,
            "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30 if cuda else None,
            "host_max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "scope": "this process, since it started (allocator peaks never reset)"}


def run(model: str, limit: int | None = None, lock_wait: float = 3600.0, force: bool = False,
        min_available_gb: float = 16.0, device: str = "cuda") -> int:
    import score_teacher

    relabel_paths.require_mount()
    out = output_dir(model, limit)
    names = SCORED[model]
    inputs = domain_manifests(names)
    scorer = SCORERS[model]()
    scorer.prepare(limit, device)
    configs = {n: panel_config(model, scorer, n, sha, limit) for n, (_, sha) in inputs.items()}
    todo = [n for n in names if force or not _done(out / f"{n}.json", configs[n])]
    if not todo:
        print(f"{model}: every set is written with this configuration", flush=True)
        return 0
    try:
        with score_teacher.gpu_lock(f"relabel heldout_panel {model}", lock_wait):
            return _run_locked(model, scorer, inputs, configs, todo, out, limit, min_available_gb, device)
    except score_teacher.LockBusy as busy:
        print(f"refusing to run: {busy}", file=sys.stderr)
        return LOCK_BUSY_EXIT


def _run_locked(model, scorer, inputs, configs, todo, out, limit, min_available_gb, device_name) -> int:
    import torch

    import score_teacher

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    score_teacher.vram_lease(device)
    if score_teacher.available_memory_gb() < min_available_gb:
        print(f"refusing to load: {score_teacher.available_memory_gb():.1f} GB host memory available, "
              f"--min-available-gb {min_available_gb}", file=sys.stderr)
        return LOW_MEMORY_EXIT
    out.mkdir(parents=True, exist_ok=True)
    timings, start = {}, time.perf_counter()
    try:
        try:
            scorer.load(device)
        except ChecksFailed as failed:
            write_json(out / "run.json", {"model": model, "status": "checks_failed", "checks": failed.checks,
                                          "code": code_commit(), "utc": utc()})
            print("implementation checks failed: nothing scored", file=sys.stderr)
            return CHECKS_FAILED_EXIT
        timings["load_s"] = time.perf_counter() - start
        print(f"{model}: loaded in {timings['load_s']:.1f} s", flush=True)
        name, manifest, recorded_path = scorer.crosscheck_input()
        started = time.perf_counter()
        new = scorer.score(name, manifest, limit)
        check = compare(new, read_json(recorded_path), name, recorded_path, limit)
        check["wall_s"] = time.perf_counter() - started
        write_json(out / "crosscheck.json", check)
        print(f"cross-check {name}: WER {check['wer']:.3f} vs recorded {check['recorded_wer']:.3f} "
              f"({check['identical_hypotheses']}/{check['utterances']} identical, configuration "
              f"{'equal' if check['config_equal'] else 'DIFFERENT'}): {'pass' if check['pass'] else 'FAIL'}",
              flush=True)
        if not check["pass"] and limit is None:
            print(f"{model}: cross-check failed; nothing scored", file=sys.stderr)
            return CROSSCHECK_FAILED_EXIT
        sets = {}
        for name in todo:
            manifest, _ = inputs[name]
            started = time.perf_counter()
            result = scorer.score(name, manifest, limit)
            add_term_recall(result, name)
            result["panel"] = {"config": configs[name], "code": code_commit(),
                               "env": {"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
                               "utc": utc()}
            write_json(out / f"{name}.json", result)
            wer = result.get("wer")
            sets[name] = {"wall_s": time.perf_counter() - started, "utterances": result["utterances"],
                          "audio_hours": result.get("audio_hours"), "wer": None if not wer else 100 * wer["wer"]}
            tr = result.get("term_recall")
            print(f"{name}: n={result['utterances']} WER " + ("n/a" if not wer else f"{100 * wer['wer']:.2f}%")
                  + ("" if not tr or tr["recall"] is None else f" term recall {tr['recall']:.4f}"), flush=True)
        write_json(out / "run.json", {
            "model": model, "status": "complete", "limit": limit, "identity": scorer.identity, "scorer": scorer.scorer,
            "scorer_info": scorer.info(), "timings": timings, "sets": sets, "memory": _peaks(device),
            "job_lease_env": {k: v for k, v in os.environ.items() if k.startswith("AGENT_JOBS_")},
            "gpu_lock": str(relabel_paths.GPU_LOCK), "code": code_commit(), "utc": utc()})
    finally:
        scorer.close()
    return 0


# --- command line ---------------------------------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--model", required=True, choices=list(SCORED))
    s.add_argument("--limit", type=int, help="smoke: the first N id-sorted utterances per set (smoke path)")
    s.add_argument("--lock-wait", type=float, default=3600.0)
    s.add_argument("--force", action="store_true")
    s.add_argument("--min-available-gb", type=float, default=16.0)
    sub.add_parser("report")
    args = parser.parse_args(argv)
    if args.cmd == "score" and args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.cmd == "report":
        import heldout_report

        return heldout_report.main()
    return run(args.model, args.limit, args.lock_wait, args.force, args.min_available_gb)


if __name__ == "__main__":
    sys.exit(main())
