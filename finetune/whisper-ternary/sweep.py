"""Run the experiment in DESIGN.md end to end, one GPU job at a time.

Two protocols exist: ``v1`` (the preregistered recipe) and ``v2`` (Revision 2:
progressive quantization, distillation, longer schedule, more data). Phases:
zeroshot, ptq, sweep, select, a3, test, report. Each step is skipped when its
output already exists and validates, so the sweep can be resumed. Selection
uses full dev-clean WER only; test splits are scored once per reported arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import paths
import powerlog

PY = paths.HERE / "python"
TRAIN = paths.HERE / "train.py"
EVALUATE = paths.HERE / "evaluate.py"
RESULTS = paths.HERE / "results"
TEST_SPLITS = ["test-clean", "test-other"]
BATCH_SIZE = 32
# Four loader workers instead of eight: the Ryzen 9950X3D at full load is the
# second-largest draw on a circuit that has tripped twice. Only the first epoch
# is loader-bound; later epochs read from the page cache.
WORKERS = 4

PROTOCOLS: dict[str, dict] = {
    "v1": {
        "prefix": "", "max_steps": 4000, "warmup": 200, "eval_every": 500,
        "lrs": {"fp32": ["1e-5", "3e-5", "1e-4"], "ternary": ["5e-5", "1e-4", "3e-4"]},
        "train_splits": ["train-clean-100"], "extra": [],
        "summary_checks": {"distill_weight": 0.0, "quant_ramp_fraction": 0.0},
    },
    "v2": {
        "prefix": "v2-", "max_steps": 12000, "warmup": 600, "eval_every": 1000,
        "lrs": {"fp32": ["1e-5", "3e-5"], "ternary": ["3e-4", "1e-3"]},
        "train_splits": "auto",  # train-clean-100 plus train-clean-360 when verified
        "extra": ["--quant-ramp-fraction", "0.25", "--distill-weight", "0.5",
                  "--distill-temperature", "1.0"],
        "summary_checks": {"distill_weight": 0.5, "quant_ramp_fraction": 0.25,
                           "distill_temperature": 1.0},
    },
}


def log(msg: str) -> None:
    print(f"[sweep {time.strftime('%H:%M:%S')}] {msg}", flush=True)


_RECORDER: dict = {"proc": None, "name": None, "restarts": 0}
MAX_RECORDER_RESTARTS = 3


def ensure_recorder() -> None:
    """Reap a dead power recorder and restart it a bounded number of times.

    Called before every GPU job so a recorder that died (disk full, sensor
    crash) is noticed within one job rather than at sweep shutdown. Failures
    here are logged and never propagate; the experiment does not depend on it.
    """
    proc = _RECORDER["proc"]
    if proc is None or proc.poll() is None:
        return
    log(f"warning: power recorder (pid {proc.pid}) exited with {proc.returncode}; "
        f"see {powerlog.POWER_DIR / (_RECORDER['name'] + '.log')}")
    _RECORDER["proc"] = None
    if _RECORDER["restarts"] >= MAX_RECORDER_RESTARTS:
        log("warning: power recorder restart limit reached; continuing without it")
        return
    _RECORDER["restarts"] += 1
    try:
        _RECORDER["proc"] = powerlog.start_recorder(_RECORDER["name"])
        log(f"power recorder restarted (pid {_RECORDER['proc'].pid}, "
            f"restart {_RECORDER['restarts']}/{MAX_RECORDER_RESTARTS})")
    except Exception as error:  # the power log must never block the experiment
        log(f"warning: power recorder restart failed ({type(error).__name__}: {error})")


def call(cmd: list[str], dry: bool) -> None:
    log(" ".join(cmd))
    if not dry:
        ensure_recorder()
        subprocess.run(cmd, check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Protocol:
    """One protocol's fixed settings plus its training splits, pinned on first use.

    The resolved split list is written to RUNS/<prefix>protocol.json the first
    time a non-dry invocation resolves it, and every later invocation reuses that
    pin, so a split that becomes available between phases cannot change the
    data seen by later arms. An explicit --train-splits must match the pin.
    """

    def __init__(self, name: str, train_splits: str | None, dry: bool) -> None:
        spec = PROTOCOLS[name]
        self.name = name
        self.prefix: str = spec["prefix"]
        self.max_steps: int = spec["max_steps"]
        self.warmup: int = spec["warmup"]
        self.eval_every: int = spec["eval_every"]
        self.lrs: dict[str, list[str]] = spec["lrs"]
        self.extra: list[str] = spec["extra"]
        self.summary_checks: dict = spec["summary_checks"]
        pinned = read_json(self.pin_path).get("train_splits") if self.pin_path.exists() else None
        if train_splits:
            splits = train_splits.split(",")
            if pinned is not None and splits != pinned:
                raise RuntimeError(f"--train-splits {splits} differs from the pinned "
                                   f"{pinned} in {self.pin_path}")
        elif pinned is not None:
            splits = pinned
            log(f"using pinned training splits {splits} from {self.pin_path}")
        elif spec["train_splits"] == "auto":
            splits = ["train-clean-100"]
            if manifest_verified("train-clean-360", dry):
                splits.append("train-clean-360")
        else:
            splits = list(spec["train_splits"])
        for split in splits:
            if split not in paths.EXPECTED_UTTERANCES or not split.startswith("train-"):
                raise ValueError(f"not a training split: {split}")
        if len(set(splits)) != len(splits):
            raise ValueError(f"duplicate training split in {splits}")
        self.train_splits: list[str] = splits
        self.train_utterances = sum(paths.EXPECTED_UTTERANCES[s] for s in self.train_splits)
        if pinned is None and not dry:
            self.pin_path.parent.mkdir(parents=True, exist_ok=True)
            self.pin_path.write_text(json.dumps(
                {"protocol": name, "train_splits": self.train_splits,
                 "train_utterances": self.train_utterances,
                 "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
            log(f"pinned training splits {self.train_splits} to {self.pin_path}")

    @property
    def pin_path(self) -> Path:
        return paths.RUNS / f"{self.prefix}protocol.json"

    def run_name(self, arm: str, lr: str) -> str:
        return f"{self.prefix}{arm}-lr{lr}"

    @property
    def selection_path(self) -> Path:
        return paths.RUNS / f"{self.prefix}selection.json"

    def final_path(self, label: str, split: str) -> Path:
        return paths.RUNS / "final" / f"{self.prefix}{label}-{split}.json"


def manifest_verified(split: str, dry: bool = False) -> bool:
    """True when the split is usable for training: data.build_manifest succeeds
    with the published count, every record is well formed, and every audio file
    exists. This exercises the same walker, transcript parsing and count
    assertion that training uses (and warms its manifest cache), so a partial
    extraction or a stale cache falls back to the default splits instead of
    aborting a later training run. Dry runs only count FLAC files, so they
    never write manifest or provenance files.
    """
    root = paths.SPLITS.get(split)
    expected = paths.EXPECTED_UTTERANCES.get(split)
    if root is None or expected is None or not root.is_dir():
        return False
    if dry:
        return sum(1 for _ in root.rglob("*.flac")) == expected
    try:
        import data
        manifest = data.build_manifest(split)
        if len(manifest) != expected:
            raise ValueError(f"{len(manifest)} records, expected {expected}")
        for record in manifest:
            if not (record.get("id") and record.get("text") and record.get("path")
                    and isinstance(record.get("duration_s"), (int, float))):
                raise ValueError(f"malformed manifest record: {record!r}"[:200])
        missing = sum(1 for record in manifest if not Path(record["path"]).is_file())
        if missing:
            raise FileNotFoundError(f"{missing} audio files listed in the manifest are missing")
    except Exception as error:  # any failure means "not usable", by design
        log(f"{split} present but not usable ({type(error).__name__}: {error}); skipping it")
        return False
    return True


def expected_source(source: str, path: Path | None, ptq: str | None) -> tuple[str, Path]:
    """(kind, path) that evaluate.py records in its JSON for these arguments."""
    if ptq is not None:
        return "export", paths.RUNS / f"{ptq}-ptq" / "export"
    if source == "pretrained":
        return "pretrained", paths.MODEL_DIR
    return source, path


def validate_evaluation(out: Path, source: str, split: str, path: Path | None,
                        ptq: str | None) -> None:
    """Refuse to reuse an evaluation of a different model, split, subset or weight file."""
    doc = read_json(out)
    kind, expected = expected_source(source, path, ptq)
    recorded = doc.get("source", {})
    problems = []
    if doc.get("split") != split:
        problems.append(f"split {doc.get('split')!r} != {split!r}")
    if doc.get("limit") is not None:
        problems.append(f"partial evaluation (limit={doc.get('limit')})")
    if doc.get("utterances") != paths.EXPECTED_UTTERANCES[split]:
        problems.append(f"{doc.get('utterances')} utterances != {paths.EXPECTED_UTTERANCES[split]}")
    generation = doc.get("generation", {})
    if (generation.get("num_beams"), generation.get("do_sample"),
            generation.get("max_new_tokens")) != (1, False, 225):
        problems.append(f"decoding settings {generation} differ from DESIGN.md")
    if recorded.get("kind") != kind or Path(recorded.get("path", "")) != Path(expected):
        problems.append(f"source {recorded.get('kind')!r} {recorded.get('path')!r} != "
                        f"{kind!r} {str(expected)!r}")
    weights = Path(expected) / str(recorded.get("weights_file", ""))
    if not weights.is_file() or sha256(weights) != recorded.get("sha256"):
        problems.append(f"weights at {weights} missing or changed since this evaluation")
    if problems:
        raise RuntimeError(f"stale or mismatched evaluation {out}: {'; '.join(problems)}. "
                           "Remove the file to re-score.")


def evaluate(source: str, split: str, out: Path, dry: bool, path: Path | None = None,
             ptq: str | None = None) -> None:
    if out.exists():
        validate_evaluation(out, source, split, path, ptq)
        log(f"exists and matches, skip: {out}")
        return
    if not dry:
        out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(PY), str(EVALUATE), "--source", source, "--split", split, "--out", str(out)]
    if path is not None:
        cmd += ["--path", str(path)]
    if ptq is not None:
        cmd += ["--ptq", ptq]
    call(cmd, dry)


def validate_run(proto: Protocol, run: Path, arm: str, lr: str) -> None:
    """A reusable run must be this protocol's full run with its artifacts present."""
    summary = read_json(run / "summary.json")
    problems = []
    for key, want in (("arm", arm), ("run_name", run.name)):
        if summary.get(key) != want:
            problems.append(f"{key} {summary.get(key)!r} != {want!r}")
    if float(summary.get("lr", "nan")) != float(lr):
        problems.append(f"lr {summary.get('lr')!r} != {lr}")
    wants = {"max_steps": proto.max_steps, "steps": proto.max_steps, "seed": paths.SEED,
             "batch_size": BATCH_SIZE, "train_utterances": proto.train_utterances}
    for key, want in wants.items():
        if summary.get(key) != want:
            problems.append(f"{key} {summary.get(key)!r} != {want}")
    for key, want in proto.summary_checks.items():
        # v1 summaries predate these keys; a missing key means the v1 default.
        if key in summary and summary.get(key) != want:
            problems.append(f"{key} {summary.get(key)!r} != {want}")
        if key not in summary and want:
            problems.append(f"{key} missing (protocol requires {want})")
    recorded_splits = summary.get("train_splits", ["train-clean-100"])
    if list(recorded_splits) != proto.train_splits:
        problems.append(f"train_splits {recorded_splits!r} != {proto.train_splits!r} (order matters)")
    weights = run / ("best-hf/model.safetensors" if arm == "fp32" else "export/export.safetensors")
    required = [weights, run / "eval-dev-clean.json", run / "config.json"]
    if arm != "fp32":
        required.append(run / "export/manifest.json")
    for item in required:
        if not item.is_file():
            problems.append(f"missing {item}")
    if problems:
        raise RuntimeError(f"run {run} cannot be reused: {'; '.join(problems)}. "
                           "Rename or remove it, or rerun train.py with --overwrite.")


def train(proto: Protocol, arm: str, lr: str, dry: bool) -> Path:
    name = proto.run_name(arm, lr)
    run = paths.RUNS / name
    if (run / "summary.json").exists():
        validate_run(proto, run, arm, lr)
        log(f"complete and matches, skip: {name}")
        return run
    cmd = [str(PY), str(TRAIN), "--arm", arm, "--lr", lr, "--run-name", name,
           "--max-steps", str(proto.max_steps), "--warmup", str(proto.warmup),
           "--eval-every", str(proto.eval_every), "--batch-size", str(BATCH_SIZE),
           "--workers", str(WORKERS),
           "--train-splits", ",".join(proto.train_splits), *proto.extra]
    if proto.name == "v1":
        # The v1 train.py had no --train-splits flag; keep its command reproducible.
        cmd = cmd[:cmd.index("--train-splits")]
    call(cmd, dry)
    return run


def phase_zeroshot(proto: Protocol, dry: bool) -> None:
    evaluate("pretrained", "dev-clean", paths.RUNS / "fp32-zeroshot" / "eval-dev-clean.json", dry)


def phase_ptq(proto: Protocol, dry: bool) -> None:
    for arm in ("ternary", "ternary-embed"):
        evaluate("pretrained", "dev-clean", paths.RUNS / f"{arm}-ptq" / "eval-dev-clean.json",
                 dry, ptq=arm)


def phase_sweep(proto: Protocol, dry: bool) -> None:
    for arm, lrs in proto.lrs.items():
        for lr in lrs:
            train(proto, arm, lr, dry)


def phase_select(proto: Protocol, dry: bool) -> dict:
    selection: dict[str, dict] = {}
    for arm, lrs in proto.lrs.items():
        rows = []
        for lr in lrs:
            summary = paths.RUNS / proto.run_name(arm, lr) / "summary.json"
            if not summary.exists():
                if dry:
                    continue
                raise FileNotFoundError(summary)
            # The sweep's string LR must win over summary.json's float so the
            # selected value can be passed back on the command line unchanged.
            rows.append({**read_json(summary), "lr": lr, "run": proto.run_name(arm, lr)})
        if not rows:
            continue
        best = min(rows, key=lambda r: r["dev_clean_wer"])
        selection[arm] = {"selected_lr": best["lr"], "selected_run": best["run"],
                          "criterion": "lowest full dev-clean WER of the run's best checkpoint",
                          "candidates": [{k: r[k] for k in ("lr", "run", "dev_clean_wer",
                                                            "dev_subset_wer", "best_step")}
                                         for r in rows]}
        log(f"{proto.name} {arm}: selected lr {best['lr']} (dev-clean WER {best['dev_clean_wer']:.4f})")
    if not dry:
        selection["protocol"] = {"name": proto.name, "max_steps": proto.max_steps,
                                 "warmup": proto.warmup, "train_splits": proto.train_splits,
                                 "extra": proto.extra}
        proto.selection_path.write_text(json.dumps(selection, indent=1))
    return selection


def phase_a3(proto: Protocol, dry: bool) -> None:
    selection = read_json(proto.selection_path) if proto.selection_path.exists() else {}
    lr = selection.get("ternary", {}).get("selected_lr", proto.lrs["ternary"][0])
    train(proto, "ternary-embed", lr, dry)


def reported_arms(proto: Protocol) -> list[tuple[str, str, Path | None]]:
    """(label, source, path) for every arm in the DESIGN.md main table."""
    selection = read_json(proto.selection_path) if proto.selection_path.exists() else {}
    fp32_run = selection.get("fp32", {}).get("selected_run", proto.run_name("fp32", "?"))
    ternary_run = selection.get("ternary", {}).get("selected_run", proto.run_name("ternary", "?"))
    ternary_lr = selection.get("ternary", {}).get("selected_lr", "?")
    return [
        ("A0-fp32-zeroshot", "pretrained", None),
        ("A1-fp32-finetune", "hf-dir", paths.RUNS / fp32_run / "best-hf"),
        ("A2-ternary-ptq", "export", paths.RUNS / "ternary-ptq" / "export"),
        ("A2-ternary", "export", paths.RUNS / ternary_run / "export"),
        ("A3-ternary-embed", "export",
         paths.RUNS / proto.run_name("ternary-embed", ternary_lr) / "export"),
    ]


def phase_test(proto: Protocol, dry: bool) -> None:
    for label, source, path in reported_arms(proto):
        for split in TEST_SPLITS:
            evaluate(source, split, proto.final_path(label, split), dry, path=path)


def dev_clean_json(label: str, source: str, path: Path | None) -> Path:
    if source == "pretrained":
        return paths.RUNS / "fp32-zeroshot" / "eval-dev-clean.json"
    if label == "A2-ternary-ptq":
        return paths.RUNS / "ternary-ptq" / "eval-dev-clean.json"
    return path.parent / "eval-dev-clean.json"


def pct(v) -> str:
    return "—" if v is None else f"{100 * v:.2f}%"


def mb(v) -> str:
    return "—" if v is None else f"{v / 1e6:.1f} MB"


def phase_report(proto: Protocol, dry: bool) -> None:
    if dry:
        log("report: skipped in dry run")
        return
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for label, source, path in reported_arms(proto):
        row = {"arm": label, "source": source, "path": str(path) if path else None}
        dev = dev_clean_json(label, source, path)
        row["dev_clean_wer"] = read_json(dev)["wer"]["wer"] if dev.exists() else None
        for split in TEST_SPLITS:
            out = proto.final_path(label, split)
            row[f"{split}_wer"] = read_json(out)["wer"]["wer"] if out.exists() else None
        manifest = (path / "manifest.json") if path and source == "export" else None
        if manifest and manifest.exists():
            m = read_json(manifest)
            row["ternary_parameters"] = m["parameter_accounting"]["ternary_parameters"]
            row["packed_code_bytes"] = m["bytes"]["packed_code_bytes"]
            row["artifact_bytes"] = m["bytes"]["file_bytes"]
            row["code_histogram"] = {k: v for k, v in m["code_histogram"].items()
                                     if k != "per_layer"}
            (RESULTS / f"{proto.prefix}{label}-manifest.json").write_text(json.dumps(
                {k: v for k, v in m.items() if k != "config"}, indent=1))
        rows.append(row)
    selection = read_json(proto.selection_path)
    (RESULTS / f"{proto.prefix}results.json").write_text(
        json.dumps({"protocol": proto.name, "rows": rows, "selection": selection}, indent=1))

    lines = [f"# Results: protocol {proto.name}", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. "
             "WER uses the Whisper English normalizer on both sides; see DESIGN.md. "
             f"Training splits: {', '.join(proto.train_splits)}; {proto.max_steps} steps.", "",
             "| Arm | dev-clean WER | test-clean WER | test-other WER | Ternary params | "
             "Packed code bytes | Artifact bytes |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['arm']} | {pct(r['dev_clean_wer'])} | {pct(r['test-clean_wer'])} | "
                     f"{pct(r['test-other_wer'])} | {r.get('ternary_parameters', '—')} | "
                     f"{mb(r.get('packed_code_bytes'))} | {mb(r.get('artifact_bytes'))} |")
    lines += ["", "## Learning-rate selection (full dev-clean WER)", "",
              "| Arm | LR | dev-clean WER | best step | selected |",
              "| --- | ---: | ---: | ---: | :-: |"]
    for arm in ("fp32", "ternary"):
        sel = selection.get(arm)
        if not sel:
            continue
        for c in sel["candidates"]:
            lines.append(f"| {arm} | {c['lr']} | {pct(c['dev_clean_wer'])} | {c['best_step']} | "
                         f"{'yes' if c['lr'] == sel['selected_lr'] else ''} |")
    out = RESULTS / f"{proto.prefix}RESULTS.md"
    out.write_text("\n".join(lines) + "\n")
    log(f"wrote {out}")


PHASES = {"zeroshot": phase_zeroshot, "ptq": phase_ptq, "sweep": phase_sweep,
          "select": phase_select, "a3": phase_a3, "test": phase_test, "report": phase_report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=list(PROTOCOLS), default="v1")
    parser.add_argument("--phase", choices=[*PHASES, "all"], default="all")
    parser.add_argument("--train-splits", help="comma-separated override of the protocol's splits")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-powerlog", action="store_true",
                        help="do not run the powerlog.py flight recorder alongside the sweep")
    args = parser.parse_args()
    paths.require_mount()
    proto = Protocol(args.protocol, args.train_splits, args.dry_run)
    recorder = None
    if not args.dry_run and not args.no_powerlog:
        _RECORDER["name"] = f"sweep-{proto.name}"
        try:
            recorder = powerlog.start_recorder(_RECORDER["name"])
            _RECORDER["proc"] = recorder
            log(f"power log: recorder pid {recorder.pid}, writing under {powerlog.POWER_DIR}")
        except Exception as error:  # the power log must never block the experiment
            log(f"warning: power recorder not started ({type(error).__name__}: {error}); continuing")
    try:
        log(f"protocol {proto.name}: {proto.max_steps} steps, splits {proto.train_splits} "
            f"({proto.train_utterances} utterances), lrs {proto.lrs}")
        names = list(PHASES) if args.phase == "all" else [args.phase]
        for name in names:
            log(f"phase {name}")
            PHASES[name](proto, args.dry_run)
        log("done")
    finally:
        recorder = _RECORDER["proc"]  # may have been restarted by ensure_recorder
        if recorder is not None:
            try:
                if powerlog.stop_recorder(recorder) is None:
                    log("warning: power recorder did not exit within the bounded wait; "
                        "its --parent-pid guard ends it after this process exits")
            except Exception as error:
                log(f"warning: stopping the power recorder failed ({type(error).__name__}: {error})")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        log(f"command failed with exit {error.returncode}")
        sys.exit(error.returncode)
