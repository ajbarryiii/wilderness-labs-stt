#!/usr/bin/python3
"""kea-b1 encoder benchmark driver (Mac, /usr/bin/python3 3.9+, standard library only). See README.md.

    kb1bench.py run --out RUN_DIR --bundle NAME=PATH [--bundle NAME=PATH ...] [--units ane[,gpu,cpu,all]]
        [--variants NAME@UNITS,...] [--external NAME=COMMAND_TEMPLATE] --baseline V --reference V --ternary V
        [--localflow PATH] [--librispeech PATH] [--asr-per-split 150] [--rounds 2] [--phases ...] [--copies-from RUN]
    kb1bench.py report RUN_DIR
    kb1bench.py verify BUNDLE

A variant is NAME@UNITS (ane = cpuAndNeuralEngine, gpu = cpuAndGPU, cpu = cpuOnly, all = all, ext = an external
encoder command). Every heavy step is its own process under macguard; refusals (lock held, low free memory) are
retried until --guard-wait. Raw JSON lines go to RUN_DIR/raw/<variant>/; a finished step is not rerun, so an
interrupted run resumes. Source bundles are only read; every Core ML variant runs from an APFS clone at a new path.
"""
from __future__ import annotations

import argparse
import array
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kb1lib as lib  # noqa: E402

ART = Path("/Users/ajbarry/wilderness-labs-stt-artifacts")
DEFAULTS = {
    "localflow": Path("/Users/ajbarry/workspace/github.com/LocalFlow-kea-b"),
    "librispeech": ART / "librispeech",
    "spelling": ART / "apple-speech-2026-10-15/whisper-english-spelling.json",
    "guard": Path("/Users/ajbarry/workspace/github.com/wilderness-labs-stt/finetune/parakeet-ternary/ios/macguard"),
}
SWIFT_FLAGS = ["-parse-as-library", "-O", "-target", "arm64-apple-macosx26.0", "-Xcc", "-DACCELERATE_NEW_LAPACK"]
TOOL = "kb1bench"  # executable name: Core ML keys its e5rt cache directory by it (~/Library/Caches/kb1bench)
PHASES = ("build", "inputs", "load", "latency", "accuracy", "report")
E2E_LENGTHS = (2, 4, 7, 14, 18, 28, 33)
E2E_OFFSETS = (0.0, 9.3, 21.7)
# Invented text for `say` (no user content). Long enough for 33 s windows starting at 21.7 s.
SPEECH = (
    "The ferry left the harbour a little after seven, and the gulls followed it out past the breakwater. "
    "Marta counted the crates twice before she signed the delivery note, because the last shipment of lamps "
    "had arrived with two missing and nobody could say where they went. Her brother said the weather would "
    "turn by noon, so they moved the canvas chairs under the awning and tied the umbrellas to the railing. "
    "In the afternoon a teacher brought her class to see the lighthouse, and the children asked how far the "
    "beam could reach on a clear night. The keeper told them about the old brass lens, the clockwork that once "
    "turned it, and the winter when ice covered every window. Later, the bakery on the corner sold out of "
    "rye bread, and the owner wrote a note on the chalkboard promising more on Thursday. By evening the wind "
    "had dropped, the boats were tied in neat rows, and the market stalls were folded away for the night. "
    "Somebody played an accordion near the fountain while the last bus waited at the square with its engine "
    "running and its windows glowing yellow against the blue dusk. On Saturday the town will hold its annual "
    "regatta, and the committee still needs volunteers to hand out maps, sell tickets and judge the sandcastles."
)


# ---------------------------------------------------------------- process helpers

class GuardFailure(RuntimeError):
    pass


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def guarded(args, cmd, timeout, stdout_path=None, rss_cap=None, light=False):
    """Run cmd under macguard, retrying refusals (exit 3: lock held or free memory low) until --guard-wait.
    light=True (compiling, `say`, cutting WAVs, front-end features, harness decoding: no Core ML model, a few
    hundred MB at most) runs directly with a timeout, because macguard admits one guarded job per machine."""
    deadline = time.time() + args.guard_wait
    full = [str(args.guard), "--rss-cap", rss_cap or args.rss_cap, "--timeout", str(timeout), "--threads", str(args.threads),
            "--", *map(str, cmd)]
    if light:
        full = list(map(str, cmd))
    while True:
        out = open(stdout_path, "w") if stdout_path else subprocess.PIPE
        try:
            p = subprocess.run(full, stdout=out, stderr=subprocess.PIPE, text=True, timeout=timeout if light else None)
        finally:
            if stdout_path:
                out.close()
        if p.returncode == 3 and time.time() < deadline:
            reason = "lock held" if "lock held" in p.stderr else "free memory low" if "free memory" in p.stderr else "refused"
            log(f"macguard refused ({reason}); retrying in 30 s")
            time.sleep(30)
            continue
        if p.returncode != 0:
            # macguard's own status lines (abort reason such as "rss ... > cap" or "timeout", end status) plus the
            # tool's last error line; the per-process "top" listing is dropped.
            err = p.stderr.strip().splitlines()
            keep = [l for l in err if "end: top" not in l and ("macguard" in l or "kb1bench" in l or "cap" in l)]
            tail = "\n".join((keep or err)[-8:])
            raise GuardFailure(f"step failed (exit {p.returncode}): {' '.join(map(str, cmd[:3]))} ...\n{tail}")
        m = re.search(r"peak group RSS (\d+)KB, swap growth (-?\d+)KB", p.stderr)
        metrics = {"guard_peak_group_rss_kib": int(m[1]), "guard_swap_growth_kib": int(m[2])} if m else {}
        return (p.stdout if not stdout_path else None), metrics


def read_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.startswith("{")] if path.exists() else []


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def step(args, path: Path, cmd, timeout, extra=None, light=False):
    """One recorded tool run: stdout JSON lines to path, plus a final {"type": "step"} line with guard metrics.
    Skipped when path already holds a finished step (resume)."""
    if any(r.get("type") == "step" for r in read_jsonl(path)):
        return read_jsonl(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    t0 = time.time()
    _, metrics = guarded(args, cmd, timeout, stdout_path=tmp, light=light)
    with open(tmp, "a") as f:
        f.write(json.dumps({"type": "step", "wall_s": time.time() - t0, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            **metrics, **(extra or {})}) + "\n")
    tmp.replace(path)
    return read_jsonl(path)


def cache_roots():
    roots = [Path.home() / "Library/Caches" / TOOL]
    for key in ("DARWIN_USER_CACHE_DIR", "DARWIN_USER_TEMP_DIR"):
        d = subprocess.run(["getconf", key], capture_output=True, text=True).stdout.strip()
        if d:
            roots.append(Path(d) / TOOL)
    return roots


def cache_snapshot() -> dict:
    snap = {}
    for root in cache_roots():
        files = {}
        if root.exists():
            for p in root.rglob("*"):
                try:
                    if p.is_file() and not p.is_symlink():
                        files[str(p.relative_to(root))] = p.stat().st_size
                except OSError:
                    pass
        snap[str(root)] = files
    return snap


# ---------------------------------------------------------------- plan

def parse_plan(args) -> dict:
    bundles = {}
    for spec in args.bundle:
        name, _, path = spec.partition("=")
        if not name or not path:
            raise ValueError(f"--bundle {spec!r} is not NAME=PATH")
        bundles[name] = Path(path).resolve()
    externals = {}
    for spec in args.external or []:
        name, _, cmd = spec.partition("=")
        if not name or not cmd:
            raise ValueError(f"--external {spec!r} is not NAME=COMMAND")
        externals[name] = cmd
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    else:
        variants = [f"{b}@{u}" for u in args.units.split(",") for b in bundles]
    variants += [f"{e}@ext" for e in externals if f"{e}@ext" not in variants]
    for v in variants:
        name, unit, _ = lib.parse_variant(v)
        if unit == "ext" and name not in externals or unit != "ext" and name not in bundles:
            raise ValueError(f"variant {v} has no matching --bundle/--external")
    for role in ("baseline", "reference", "ternary"):
        if getattr(args, role) not in variants:
            raise ValueError(f"--{role} {getattr(args, role)} is not one of the variants {variants}")
    if lib.parse_variant(args.reference)[1] == "ext":
        raise ValueError("--reference must be a Core ML variant")
    return {"bundles": {k: str(v) for k, v in bundles.items()}, "externals": externals, "variants": variants,
            "baseline": args.baseline, "reference": args.reference, "ternary": args.ternary}


def safe(v: str) -> str:
    return v.replace("@", "_at_")


# ---------------------------------------------------------------- phases

def phase_build(args, run: Path, plan: dict) -> None:
    exe = run / "bin" / TOOL
    lf = Path(args.localflow)
    sources = sorted([*(lf / "Sources/Parakeet").glob("*.swift"), *(lf / "Sources/Dictionary").glob("*.swift")])
    if not sources:
        raise ValueError(f"no LocalFlow sources under {lf}")
    if not exe.exists():
        exe.parent.mkdir(parents=True, exist_ok=True)
        sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
        guarded(args, ["xcrun", "swiftc", *SWIFT_FLAGS, "-sdk", sdk, *sources, HERE / "kb1bench.swift", "-o", exe], 900, light=True)
    git = lambda *a: subprocess.run(["git", "-C", str(lf), *a], capture_output=True, text=True).stdout.strip()
    plan["build"] = {"localflow": str(lf), "localflow_commit": git("rev-parse", "HEAD"),
                     "localflow_runtime_dirty": git("status", "--porcelain", "--", "Sources/Parakeet", "Sources/Dictionary"),
                     "runtime_sources_sha256": {str(s.relative_to(lf)): lib.sha256_file(s) for s in sources},
                     "harness_swift_sha256": lib.sha256_file(HERE / "kb1bench.swift"), "executable_sha256": lib.sha256_file(exe),
                     "swift_flags": SWIFT_FLAGS}


def load_librispeech(root: Path, split: str):
    items = []
    for t in sorted((root / "LibriSpeech" / split).glob("*/*/*.trans.txt")):
        for line in t.read_text().splitlines():
            ident, _, ref = line.partition(" ")
            items.append({"id": ident, "path": str(t.parent / f"{ident}.flac"), "reference": ref, "split": split})
    if not items:
        raise ValueError(f"no LibriSpeech {split} under {root}")
    return items


def phase_inputs(args, run: Path, plan: dict) -> None:
    inp = run / "inputs"
    inp.mkdir(exist_ok=True)
    exe = run / "bin" / TOOL
    speech = inp / "speech.aiff"
    if not speech.exists():
        (inp / "speech.txt").write_text(SPEECH + "\n")
        guarded(args, ["say", "-v", args.voice, "-o", speech, "-f", inp / "speech.txt"], 300, light=True)
    fixtures_tsv = inp / "fixtures.tsv"
    if not fixtures_tsv.exists():
        out, _ = guarded(args, [exe, "fixtures", "--speech", speech, "--out", inp / "fixtures",
                                "--lengths", ",".join(map(str, args.e2e_lengths)), "--offsets", ",".join(map(str, E2E_OFFSETS))], 300, light=True)
        rows = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
        fixtures_tsv.write_text("".join(f"{r['id']}\t{r['path']}\n" for r in rows))
    asr = []
    for split in ("test-clean", "test-other"):
        asr += lib.select_subset(load_librispeech(Path(args.librispeech), split), args.asr_per_split, seed=0)
    for item in asr:
        if not Path(item["path"]).is_file():
            raise ValueError(f"missing LibriSpeech audio {item['id']}")
    (inp / "asr.tsv").write_text("".join(f"{i['id']}\t{i['path']}\n" for i in asr))
    write_json(inp / "asr-references.json", {i["id"]: {"reference": i["reference"], "split": i["split"]} for i in asr})
    features = inp / "features"
    if not (features / "manifest.jsonl").exists():
        (inp / "features.tsv").write_text(fixtures_tsv.read_text() + (inp / "asr.tsv").read_text())
        ref_bundle = plan["bundles"][lib.parse_variant(plan["reference"])[0]]
        guarded(args, [exe, "features", "--bundle", ref_bundle, "--list", inp / "features.tsv", "--out", features], 1800, light=True)
    rows = read_jsonl(features / "manifest.jsonl")
    asr_ids = {i["id"] for i in asr}
    enc_keys = [k for b, ks in lib.select_chunks(rows, args.encoder_per_bucket).items() for k in ks]
    diff_keys = [k for b, ks in lib.select_chunks([r for r in rows if r["id"] in asr_ids], args.encdiff_per_bucket).items() for k in ks]
    asr_keys = [r["key"] for r in rows if r["id"] in asr_ids and not r["silent"]]
    for name, keys in (("keys-encoder.txt", enc_keys), ("keys-encdiff.txt", diff_keys), ("keys-asr.txt", asr_keys)):
        (inp / name).write_text("\n".join(keys) + "\n")
    counts = {}
    for r in rows:
        if not r["silent"]:
            counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1
    plan["inputs"] = {"voice": args.voice, "e2e_lengths_s": list(args.e2e_lengths), "e2e_offsets_s": list(E2E_OFFSETS),
                      "asr_per_split": args.asr_per_split, "asr_utterances": len(asr), "feature_chunks_by_bucket": counts,
                      "encoder_keys_by_bucket": {b: len(k) for b, k in lib.select_chunks(rows, args.encoder_per_bucket).items()},
                      "encdiff_keys": len(diff_keys), "frontend_from": plan["reference"]}


def skipped(args, plan, v) -> bool:
    """A variant whose step failed (e.g. a load timeout on one compute unit) is skipped for the rest of the run,
    and on resume unless --retry-failed, so one failing arm never stops the comparison of the others."""
    return v in plan.get("failures", {}) and not args.retry_failed


def record_failure(run: Path, plan: dict, v: str, phase: str, error: Exception) -> None:
    if v == plan["reference"] and phase == "accuracy":
        raise error
    log(f"{v}: {phase} failed; skipping this variant from here on: {str(error).splitlines()[0]}")
    plan.setdefault("failures", {})[v] = {"phase": phase, "error": str(error)[-2000:], "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_json(run / "plan.json", plan)


def coreml_variants(plan):
    return [v for v in plan["variants"] if lib.parse_variant(v)[1] != "ext"]


def copy_for(run: Path, plan: dict, variant: str, args) -> Path:
    name = lib.parse_variant(variant)[0]
    dest = run / "copies" / safe(variant) / "bundle"
    if args.copies_from:
        prior = Path(args.copies_from) / "copies" / safe(variant) / "bundle"
        if not prior.exists():
            raise ValueError(f"--copies-from has no copy for {variant}")
        return prior
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        # APFS clone: a new path and inode (so Core ML specializes again) without duplicating the blocks.
        subprocess.run(["cp", "-c", "-R", plan["bundles"][name], str(dest)], check=True)
    return dest


def phase_load(args, run: Path, plan: dict) -> None:
    for v in coreml_variants(plan):
        if skipped(args, plan, v):
            continue
        try:
            units = lib.parse_variant(v)[2]
            raw = run / "raw" / safe(v)
            bundle = copy_for(run, plan, v, args)
            order = args.load_order
            if not args.copies_from:
                path = raw / "load-new-path.jsonl"
                if not path.exists():
                    log(f"{v}: first load at a new path (Core ML specialization)")
                    before = cache_snapshot()
                    step(args, path, [run / "bin" / TOOL, "load", "--bundle", bundle, "--units", units, "--order", order],
                         args.first_load_timeout)
                    after = cache_snapshot()
                    write_json(raw / "load-new-path-cache.json",
                               {root: lib.cache_delta(before.get(root, {}), files) for root, files in after.items()})
            log(f"{v}: deployed prepare and same-path reload (fresh processes)")
            step(args, raw / "prepare.jsonl", [run / "bin" / TOOL, "prepare", "--bundle", bundle, "--units", units], 900)
            step(args, raw / "load-same-path.jsonl", [run / "bin" / TOOL, "load", "--bundle", bundle, "--units", units,
                                                       "--order", order], 900)
        except GuardFailure as error:
            record_failure(run, plan, v, "load", error)


def external_cmd(plan, name, **fields):
    return [part.format(**{k: str(v) for k, v in fields.items()}) for part in shlex.split(plan["externals"][name])]


def phase_latency(args, run: Path, plan: dict) -> None:
    inp = run / "inputs"
    orders = lib.counterbalanced_orders(plan["variants"], args.rounds)
    plan["latency_orders"] = orders
    for r, order in enumerate(orders):
        for v in order:
            if skipped(args, plan, v):
                continue
            try:
                name, unit, units = lib.parse_variant(v)
                raw = run / "raw" / safe(v)
                log(f"round {r + 1}/{len(orders)}: {v}")
                if unit == "ext":
                    out = raw / f"external-r{r}"
                    path = raw / f"encoder-r{r}.jsonl"
                    if not path.exists():
                        out.mkdir(parents=True, exist_ok=True)
                        guarded(args, external_cmd(plan, name, mode="time", features=inp / "features", keys=inp / "keys-encoder.txt",
                                                   out=out, warmups=args.encoder_warmups, reps=args.encoder_reps), args.step_timeout)
                        rows = [{"type": "encoder_call", **x} for x in read_jsonl(out / "timings.jsonl")]
                        mem = json.loads((out / "memory.json").read_text()) if (out / "memory.json").exists() else None
                        path.write_text("".join(json.dumps(x) + "\n" for x in rows + [{"type": "external_memory", "memory": mem},
                                                                                         {"type": "step"}]))
                    continue
                bundle = copy_for(run, plan, v, args)
                step(args, raw / f"encoder-r{r}.jsonl", [run / "bin" / TOOL, "encoder", "--bundle", bundle, "--units", units,
                                                         "--features", inp / "features", "--keys", inp / "keys-encoder.txt",
                                                         "--warmups", args.encoder_warmups, "--reps", args.encoder_reps], args.step_timeout)
                step(args, raw / f"e2e-r{r}.jsonl", [run / "bin" / TOOL, "service", "--bundle", bundle, "--units", units,
                                                     "--fixtures", inp / "fixtures.tsv", "--warmups", args.e2e_warmups,
                                                     "--reps", args.e2e_reps], args.step_timeout)
            except GuardFailure as error:
                record_failure(run, plan, v, "latency", error)


def encdiff(ref_dir: Path, cand_dir: Path, keys) -> dict:
    errors = {}
    for k in keys:
        ref, cand = array.array("f"), array.array("f")
        ref.frombytes((ref_dir / f"{k}.f32").read_bytes())
        cand.frombytes((cand_dir / f"{k}.f32").read_bytes())
        errors[k] = lib.encoder_error(ref, cand)
    return {"aggregate": lib.aggregate_encoder_errors(list(errors.values())),
            "per_chunk": {k: {m: e[m] for m in ("rel_l2", "max_abs", "cosine")} for k, e in errors.items()}}


def phase_accuracy(args, run: Path, plan: dict) -> None:
    inp = run / "inputs"
    exe = run / "bin" / TOOL
    ref = plan["reference"]
    ref_bundle = copy_for(run, plan, ref, args)
    ref_raw = run / "raw" / safe(ref)
    diff_keys = (inp / "keys-encdiff.txt").read_text().split()
    # Reference: encoder outputs of every ASR chunk, and the decode-path self-check (precomputed outputs decoded
    # by the harness must reproduce the service's transcripts exactly; validates the external-encoder seam).
    step(args, ref_raw / "encode.jsonl", [exe, "encode", "--bundle", ref_bundle, "--units", lib.parse_variant(ref)[2],
                                          "--features", inp / "features", "--keys", inp / "keys-asr.txt",
                                          "--out", ref_raw / "encoded"], args.step_timeout)
    step(args, ref_raw / "decode.jsonl", [exe, "decode", "--bundle", ref_bundle, "--list", inp / "asr.tsv",
                                          "--encoded", ref_raw / "encoded"], args.step_timeout, light=True)
    for v in plan["variants"]:
        if skipped(args, plan, v):
            continue
        try:
            name, unit, units = lib.parse_variant(v)
            raw = run / "raw" / safe(v)
            log(f"accuracy: {v}")
            if unit == "ext":
                out = raw / "external-encoded"
                if not (out / "manifest.jsonl").exists():
                    out.mkdir(parents=True, exist_ok=True)
                    guarded(args, external_cmd(plan, name, mode="encode", features=inp / "features", keys=inp / "keys-asr.txt",
                                               out=out, warmups=0, reps=0), args.step_timeout)
                step(args, raw / "asr.jsonl", [exe, "decode", "--bundle", ref_bundle, "--list", inp / "asr.tsv", "--encoded", out],
                     args.step_timeout, light=True)
                cand_dir = out
            else:
                bundle = copy_for(run, plan, v, args)
                step(args, raw / "asr.jsonl", [exe, "service", "--bundle", bundle, "--units", units, "--asr", inp / "asr.tsv"],
                     args.step_timeout)
                if v == ref:
                    cand_dir = ref_raw / "encoded"
                else:
                    tmp_keys = raw / "keys-encdiff.txt"
                    raw.mkdir(parents=True, exist_ok=True)
                    tmp_keys.write_text("\n".join(diff_keys) + "\n")
                    step(args, raw / "encode.jsonl", [exe, "encode", "--bundle", bundle, "--units", units, "--features",
                                                      inp / "features", "--keys", tmp_keys, "--out", raw / "encoded"], args.step_timeout)
                    cand_dir = raw / "encoded"
            if not (raw / "encdiff.json").exists():
                write_json(raw / "encdiff.json", encdiff(ref_raw / "encoded", cand_dir, diff_keys))
        except GuardFailure as error:
            record_failure(run, plan, v, "accuracy", error)


# ---------------------------------------------------------------- report

def mb(x):
    return None if x is None else round(x / 1e6, 1)


def fmt_ratio(p):
    if not p:
        return None
    lo, hi = p["typical_ratio_ci95"]
    return f"{p['typical_ratio']:.3f} [{lo:.3f}, {hi:.3f}]"


def phase_mem(rows, phase):
    for r in rows:
        if r.get("type") in ("load", "prepare", "memory"):
            for p in r["memory"]:
                if p["phase"] == phase:
                    return p
    return None


def build_report(run: Path) -> dict:
    plan = json.loads((run / "plan.json").read_text())
    spelling = lib.load_spelling_map(Path(plan["config"]["spelling"]))
    refs = json.loads((run / "inputs/asr-references.json").read_text())
    n_boot = plan["config"]["bootstrap"]
    variants = plan["variants"]
    raw = lambda v: run / "raw" / safe(v)
    out = {"schema": 1, "run": run.name, "plan": plan, "variants": {}}

    enc_records, e2e_records, thermal = [], [], {}
    for v in variants:
        states = []
        for p in sorted(raw(v).glob("encoder-r*.jsonl")):
            for r in read_jsonl(p):
                if r.get("type") == "encoder_call" and not r.get("warmup"):
                    enc_records.append({**r, "variant": v})
                if r.get("type") == "thermal_block":
                    states += [r["thermal_start"], r["thermal_end"]]
        for p in sorted(raw(v).glob("e2e-r*.jsonl")):
            for r in read_jsonl(p):
                if r.get("type") == "e2e_call" and not r.get("warmup"):
                    e2e_records.append({**r, "variant": v, "length": int(r["clip"].split("-")[1].rstrip("s"))})
                if r.get("type") == "thermal":
                    states += r["states"]
        thermal[v] = lib.thermal_summary(states) if states else None
    enc = lib.group_samples(enc_records, ("variant", "bucket"))
    e2e_len = lib.group_samples(e2e_records, ("variant", "length"))
    e2e_all = lib.group_samples(e2e_records, ("variant",))
    base = plan["baseline"]

    def latency_block(groups, key_of, v, keys):
        block = {}
        for k in keys:
            arm = groups.get(key_of(v, k))
            if not arm:
                continue
            clip_medians = [lib.median(x) for x in arm.values()]
            pooled = [y for x in arm.values() for y in x]
            entry = {"clips": len(arm), "calls": len(pooled), "typical_ms": lib.median(clip_medians),
                     "p95_ms": lib.quantile(pooled, 0.95)}
            b = groups.get(key_of(base, k))
            if b and v != base:
                try:
                    entry["vs_baseline"] = lib.paired_ratio(arm, b, n_boot=n_boot, seed=0)
                except ValueError:
                    pass
            block[str(k)] = entry
        return block

    ref_hyp = {r["id"]: r["text"] for r in read_jsonl(raw(plan["reference"]) / "asr.jsonl") if r.get("type") == "asr"}
    ref_e2e = {}
    for r in e2e_records:
        if r["variant"] == plan["reference"]:
            ref_e2e.setdefault(r["clip"], r["text"])
    ternary_mem = None
    for v in variants:
        name, unit, units = lib.parse_variant(v)
        entry = {"units": units}
        if unit != "ext":
            entry["bundle"] = plan["bundle_info"][name]
        # load
        load = {}
        for label in ("load-new-path", "load-same-path", "prepare"):
            rows = read_jsonl(raw(v) / f"{label}.jsonl")
            main = next((r for r in rows if r.get("type") in ("load", "prepare")), None)
            if main:
                load[label] = {"all_ready_ms": main["all_ready_ms"], "functions": main.get("functions")}
                idle = phase_mem([main], "loaded_idle" if label != "prepare" else "prepared_idle")
                if idle:
                    load[label]["memory_after_load"] = idle["last"]
                    load[label]["system_wired_delta_bytes"] = idle["system_wired_end"] - (main["memory"][0]["system_wired_start"])
                step_row = next((r for r in rows if r.get("type") == "step"), {})
                load[label]["process_wall_s"] = step_row.get("wall_s")
                load[label]["guard_peak_group_rss_kib"] = step_row.get("guard_peak_group_rss_kib")
                load[label]["daemons_kib"] = main.get("daemons_kib")
                load[label]["thermal"] = main.get("thermal") or main.get("thermal_end")
        cachefile = raw(v) / "load-new-path-cache.json"
        if cachefile.exists():
            load["new_path_cache"] = json.loads(cachefile.read_text())
            load["new_path_cache_added_bytes"] = sum(c["added_bytes"] for c in load["new_path_cache"].values())
        entry["load"] = load
        # memory
        memory = {}
        prep = load.get("prepare", {}).get("memory_after_load")
        if prep:
            memory["all_four_loaded"] = prep
        peaks = []
        for p in sorted(raw(v).glob("e2e-r*.jsonl")):
            t = phase_mem(read_jsonl(p), "transcribe")
            if t:
                peaks.append({"max_phys_footprint": t["max_phys_footprint"], "max_footprint_plus_neural": t["max_footprint_plus_neural"]})
        if peaks:
            memory["transcribe_peak_sampled"] = {k: max(p[k] for p in peaks) for k in peaks[0]}
        if unit == "ext":
            mem_rows = [r for p in sorted(raw(v).glob("encoder-r*.jsonl")) for r in read_jsonl(p) if r.get("type") == "external_memory"]
            memory["external_reported"] = mem_rows[0]["memory"] if mem_rows else None
        entry["memory"] = memory
        if v == plan["ternary"] and prep:
            ternary_mem = (prep["footprint_plus_neural"], prep["phys_footprint"],
                           memory.get("transcribe_peak_sampled", {}).get("max_footprint_plus_neural"),
                           entry.get("bundle", {}).get("encoder_weight_bytes"))
        # latency
        entry["encoder_latency_by_bucket"] = latency_block(enc, lambda x, k: (x, k), v, lib.LOCALFLOW_BUCKETS)
        entry["e2e_latency_by_length"] = latency_block(e2e_len, lambda x, k: (x, k), v, sorted({r["length"] for r in e2e_records}))
        entry["e2e_latency_all"] = latency_block(e2e_all, lambda x, k: (x,), v, ["all"]).get("all")
        entry["thermal"] = thermal.get(v)
        buckets_used = {}
        for r in e2e_records:
            if r["variant"] == v:
                buckets_used.setdefault(r["clip"], r["buckets"])
        entry["e2e_buckets_used"] = buckets_used
        # accuracy
        hyp = {r["id"]: r["text"] for r in read_jsonl(raw(v) / "asr.jsonl") if r.get("type") == "asr"}
        if hyp and set(hyp) == set(refs):
            items = {i: {"reference": refs[i]["reference"], "hypothesis": hyp[i]} for i in hyp}
            acc = {"all": lib.wer_summary(list(items.values()), spelling, seed=1, n_boot=n_boot)}
            for split in ("test-clean", "test-other"):
                sub = [items[i] for i in items if refs[i]["split"] == split]
                if sub:
                    acc[split] = lib.wer_summary(sub, spelling, seed=1, n_boot=n_boot)
            if ref_hyp and set(ref_hyp) == set(hyp):
                ref_items = {i: {"reference": refs[i]["reference"], "hypothesis": ref_hyp[i]} for i in hyp}
                acc["vs_reference"] = lib.agreement(ref_hyp, hyp)
                acc["vs_reference"]["paired_wer_delta"] = lib.paired_wer_delta(ref_items, items, spelling, seed=1, n_boot=n_boot)
            entry["accuracy"] = acc
        mine = {}
        for r in e2e_records:
            if r["variant"] == v:
                mine.setdefault(r["clip"], r["text"])
        if mine and set(mine) == set(ref_e2e):
            entry["e2e_fixture_agreement"] = {k: x for k, x in lib.agreement(ref_e2e, mine).items()}
        if (raw(v) / "encdiff.json").exists():
            entry["encoder_error_vs_reference"] = json.loads((raw(v) / "encdiff.json").read_text())["aggregate"]
        out["variants"][v] = entry
    dec = {r["id"]: r["text"] for r in read_jsonl(raw(plan["reference"]) / "decode.jsonl") if r.get("type") == "asr"}
    if dec and ref_hyp:
        out["decode_path_self_check"] = lib.agreement(ref_hyp, {i: dec.get(i, "") for i in ref_hyp})
    if ternary_mem:
        for v, entry in out["variants"].items():
            m = entry["memory"].get("all_four_loaded")
            verdict = {}
            if m:
                verdict["all_four_loaded_footprint_plus_neural"] = lib.memory_verdict(m["footprint_plus_neural"], ternary_mem[0])
                verdict["all_four_loaded_phys_footprint"] = lib.memory_verdict(m["phys_footprint"], ternary_mem[1])
            peak = entry["memory"].get("transcribe_peak_sampled", {}).get("max_footprint_plus_neural")
            if peak and ternary_mem[2]:
                verdict["transcribe_peak_footprint_plus_neural"] = lib.memory_verdict(peak, ternary_mem[2])
            w = entry.get("bundle", {}).get("encoder_weight_bytes")
            if w and ternary_mem[3]:
                verdict["encoder_weight_bin"] = lib.memory_verdict(w, ternary_mem[3])
            entry["memory_vs_ternary"] = verdict
    return out


def summary_markdown(rep: dict) -> str:
    plan = rep["plan"]
    V = rep["variants"]
    lines = [f"# kea-b1 encoder benchmark {rep['run']}", "",
             f"Baseline (latency ratios): `{plan['baseline']}`. Reference (accuracy): `{plan['reference']}`. "
             f"Ternary (memory): `{plan['ternary']}`. Rounds: {plan['config']['rounds']} (counterbalanced). "
             f"Machine: {plan.get('machine', {}).get('hardware')} macOS {plan.get('machine', {}).get('macos')}. "
             f"LocalFlow runtime {plan.get('build', {}).get('localflow_commit', '?')[:7]}.", ""]
    for v, f in sorted(plan.get("failures", {}).items()):
        lines += [f"**{v} failed in phase {f['phase']}** (skipped afterwards): `{f['error'].splitlines()[0]}`", ""]
    lines += ["## Encoder latency, warm `MLModel.prediction` (typical ms = median of clip medians; ratio vs baseline [95% CI])", ""]
    rows = []
    for v, e in V.items():
        row = [v]
        for b in lib.LOCALFLOW_BUCKETS:
            x = e["encoder_latency_by_bucket"].get(str(b))
            row.append(None if not x else f"{x['typical_ms']:.1f} / p95 {x['p95_ms']:.1f}" +
                       (f" ({fmt_ratio(x.get('vs_baseline'))})" if x.get("vs_baseline") else ""))
        rows.append(row)
    lines.append(lib.markdown_table(["variant"] + [f"b{b}" for b in lib.LOCALFLOW_BUCKETS], rows))
    lengths = sorted({k for e in V.values() for k in e["e2e_latency_by_length"]}, key=int)
    lines += ["## End-to-end `LocalParakeetService.transcribe(fileURL:directory:)` (typical ms per clip length)", ""]
    rows = []
    for v, e in V.items():
        row = [v] + [None if not e["e2e_latency_by_length"].get(k) else f"{e['e2e_latency_by_length'][k]['typical_ms']:.0f}"
                     for k in lengths]
        a = e.get("e2e_latency_all")
        row.append(fmt_ratio(a.get("vs_baseline")) if a and a.get("vs_baseline") else ("baseline" if v == plan["baseline"] else None))
        rows.append(row)
    lines.append(lib.markdown_table(["variant"] + [f"{k} s" for k in lengths] + ["all clips ratio [CI]"], rows))
    lines += ["## Load (ms)", ""]
    rows = []
    for v, e in V.items():
        L = e["load"]
        newp = L.get("load-new-path") or {}
        fn = {f["bucket"]: f for f in newp.get("functions") or []}
        same = {f["bucket"]: f for f in (L.get("load-same-path") or {}).get("functions") or []}
        rows.append([v] + [None if b not in fn else round(fn[b]["ready_ms"]) for b in (15, 4, 8, 30)] +
                    [None if not newp else round(newp["all_ready_ms"]),
                     None if not same else "/".join(str(round(same[b]["ready_ms"])) for b in (15, 4, 8, 30) if b in same),
                     None if not L.get("load-same-path") else round(L["load-same-path"]["all_ready_ms"]),
                     None if not L.get("prepare") else round(L["prepare"]["all_ready_ms"]),
                     mb(L.get("new_path_cache_added_bytes"))])
    lines.append(lib.markdown_table(["variant", "new b15", "new b4", "new b8", "new b30", "new all-4", "fresh 15/4/8/30",
                                     "fresh all-4", "service prepare all-4", "cache MB created"], rows))
    lines += ["## Memory (MB). phys = phys_footprint; +neural adds the Neural Engine ledger (neural_nofootprint)", ""]
    rows = []
    for v, e in V.items():
        m, b = e["memory"], e.get("bundle", {})
        loaded = m.get("all_four_loaded") or {}
        peak = m.get("transcribe_peak_sampled") or {}
        vt = e.get("memory_vs_ternary", {}).get("all_four_loaded_footprint_plus_neural")
        rows.append([v, mb(b.get("encoder_weight_bytes")), mb(b.get("listed_bytes")), mb(loaded.get("phys_footprint")),
                     mb(loaded.get("footprint_plus_neural")), mb(peak.get("max_footprint_plus_neural")),
                     "ternary" if v == plan["ternary"] else
                     None if not vt else f"{vt['ratio']:.3f} ({'less' if vt['less_than_ternary'] else 'NOT less'})"])
    lines.append(lib.markdown_table(["variant", "weight.bin", "bundle (listed)", "loaded phys", "loaded phys+neural",
                                     "transcribe peak phys+neural", "loaded vs ternary"], rows))
    lines += ["## Accuracy (LibriSpeech subset; WER Whisper-normalized)", ""]
    rows = []
    for v, e in V.items():
        a = e.get("accuracy") or {}
        vs = a.get("vs_reference") or {}
        err = e.get("encoder_error_vs_reference") or {}
        d = vs.get("paired_wer_delta")
        rows.append([v, None if "test-clean" not in a else f"{100 * a['test-clean']['wer']:.2f}",
                     None if "test-other" not in a else f"{100 * a['test-other']['wer']:.2f}",
                     None if not d else f"{100 * d['wer_delta']:+.2f} [{100 * d['wer_delta_ci95'][0]:+.2f}, {100 * d['wer_delta_ci95'][1]:+.2f}]",
                     None if not vs else f"{100 * vs['exact_match_rate']:.1f}%",
                     None if not e.get("e2e_fixture_agreement") else f"{100 * e['e2e_fixture_agreement']['exact_match_rate']:.0f}%",
                     None if not err else f"{err['pooled_rel_l2']:.4f} (max {err['max_rel_l2']:.4f})"])
    lines.append(lib.markdown_table(["variant", "WER clean %", "WER other %", "dWER vs ref (pp) [CI]", "exact = ref",
                                     "synthetic exact = ref", "encoder rel L2 vs ref"], rows))
    sc = rep.get("decode_path_self_check")
    if sc:
        lines.append(f"Decode-path self-check (reference encoder outputs decoded by the harness vs the service): "
                     f"{100 * sc['exact_match_rate']:.1f}% exact of {sc['utterances']}.\n")
    th = {v: e.get("thermal") for v, e in V.items() if e.get("thermal")}
    if th:
        lines.append("Thermal (ProcessInfo.thermalState at block boundaries): " +
                     "; ".join(f"{v}: {t['counts']}" for v, t in th.items()) + "\n")
    return "\n".join(lines)


def summary_text(rep: dict) -> str:
    plan = rep["plan"]
    out = [f"kea-b1 bench {rep['run']}: baseline {plan['baseline']}, reference {plan['reference']}, ternary {plan['ternary']}"]
    for v, e in rep["variants"].items():
        if v in plan.get("failures", {}):
            out.append(f"{v}: FAILED in {plan['failures'][v]['phase']}: {plan['failures'][v]['error'].splitlines()[0][:120]}")
            continue
        enc = e["encoder_latency_by_bucket"]
        parts = [f"{v}:"]
        parts.append("enc " + " ".join(f"b{b} {enc[str(b)]['typical_ms']:.1f}" for b in lib.LOCALFLOW_BUCKETS if str(b) in enc) + " ms")
        a = e.get("e2e_latency_all")
        if a and a.get("vs_baseline"):
            parts.append(f"e2e x{fmt_ratio(a['vs_baseline'])}")
        m = e["memory"].get("all_four_loaded")
        if m:
            parts.append(f"mem {mb(m['footprint_plus_neural'])} MB")
        vt = e.get("memory_vs_ternary", {}).get("all_four_loaded_footprint_plus_neural")
        if vt:
            parts.append(f"({vt['ratio']:.3f} of ternary)")
        acc = e.get("accuracy", {})
        if "all" in acc:
            parts.append(f"WER {100 * acc['all']['wer']:.2f}%")
        if acc.get("vs_reference"):
            parts.append(f"exact=ref {100 * acc['vs_reference']['exact_match_rate']:.1f}%")
        out.append(" ".join(parts))
    return "\n".join(out) + "\n"


def phase_report(run: Path) -> None:
    rep = build_report(run)
    write_json(run / "results.json", rep)
    (run / "summary.md").write_text(summary_markdown(rep))
    (run / "summary.txt").write_text(summary_text(rep))
    log(f"report: {run / 'summary.md'}")


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True, help="run directory (new, or an interrupted run to resume)")
    r.add_argument("--bundle", action="append", default=[], help="NAME=PATH of a LocalFlow bundle (repeatable)")
    r.add_argument("--units", default="ane", help="compute units for every bundle: ane,gpu,cpu,all (comma list)")
    r.add_argument("--variants", help="explicit NAME@UNITS list (overrides --units)")
    r.add_argument("--external", action="append", help="NAME=COMMAND template for a non-Core-ML encoder (README)")
    r.add_argument("--baseline", required=True, help="variant latency ratios are relative to")
    r.add_argument("--reference", required=True, help="Core ML variant accuracy and encoder error are relative to")
    r.add_argument("--ternary", required=True, help="variant the memory acceptance compares against")
    r.add_argument("--localflow", type=Path, default=DEFAULTS["localflow"])
    r.add_argument("--librispeech", type=Path, default=DEFAULTS["librispeech"])
    r.add_argument("--spelling", type=Path, default=DEFAULTS["spelling"])
    r.add_argument("--guard", type=Path, default=DEFAULTS["guard"])
    r.add_argument("--guard-wait", type=int, default=7200, help="seconds to keep retrying macguard refusals")
    r.add_argument("--rss-cap", default="4G")
    r.add_argument("--threads", type=int, default=8, help="macguard thread cap (BLAS/OMP)")
    r.add_argument("--phases", default=",".join(PHASES))
    r.add_argument("--copies-from", type=Path, help="reuse warm bundle copies of an earlier run (skips first loads)")
    r.add_argument("--voice", default="Samantha")
    r.add_argument("--e2e-lengths", type=lambda s: [int(x) for x in s.split(",")], default=list(E2E_LENGTHS))
    r.add_argument("--asr-per-split", type=int, default=150, help="LibriSpeech utterances per split (test-clean, test-other)")
    r.add_argument("--encoder-per-bucket", type=int, default=6)
    r.add_argument("--encdiff-per-bucket", type=int, default=8)
    r.add_argument("--rounds", type=int, default=2)
    r.add_argument("--encoder-warmups", type=int, default=3)
    r.add_argument("--encoder-reps", type=int, default=10)
    r.add_argument("--e2e-warmups", type=int, default=2)
    r.add_argument("--e2e-reps", type=int, default=5)
    r.add_argument("--load-order", default="15,4,8,30", help="function load order (the app prepares 15 s first)")
    r.add_argument("--first-load-timeout", type=int, default=2400)
    r.add_argument("--step-timeout", type=int, default=2400)
    r.add_argument("--bootstrap", type=int, default=2000)
    r.add_argument("--retry-failed", action="store_true", help="retry variants that failed in an earlier attempt")
    rp = sub.add_parser("report")
    rp.add_argument("run", type=Path)
    vp = sub.add_parser("verify")
    vp.add_argument("bundle", type=Path)
    args = ap.parse_args()

    if args.cmd == "verify":
        info = lib.verify_bundle(args.bundle)
        print(json.dumps({k: v for k, v in info.items() if k != "manifest"}, indent=2))
        return
    if args.cmd == "report":
        phase_report(args.run.resolve())
        return
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("run requires an Apple Silicon Mac")
    run = args.out.resolve()
    run.mkdir(parents=True, exist_ok=True)
    plan_path = run / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    fresh = parse_plan(args)
    if plan and {k: plan[k] for k in fresh} != fresh:
        raise SystemExit("run directory holds a different plan; use a new --out")
    plan.update(fresh)
    plan["config"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ("cmd",)}
    plan["machine"] = {"hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                       "macos": platform.mac_ver()[0], "ram_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))}
    lib.load_spelling_map(args.spelling)
    for name, path in plan["bundles"].items():
        if "bundle_info" not in plan or name not in plan["bundle_info"]:
            log(f"verifying bundle {name}")
            info = lib.verify_bundle(Path(path))
            plan.setdefault("bundle_info", {})[name] = {k: v for k, v in info.items() if k != "manifest"}
            plan["bundle_info"][name]["manifest_encoder"] = info["manifest"].get("encoder")
            plan["bundle_info"][name]["frontend_f32bin_sha256"] = info["manifest"]["files"]["frontend.f32bin"]
    fronts = {i["frontend_f32bin_sha256"] for i in plan["bundle_info"].values()}
    plan["frontends_identical"] = len(fronts) == 1
    write_json(plan_path, plan)
    phases = args.phases.split(",")
    for name, fn in (("build", phase_build), ("inputs", phase_inputs), ("load", phase_load), ("latency", phase_latency),
                     ("accuracy", phase_accuracy)):
        if name in phases:
            log(f"phase {name}")
            fn(args, run, plan)
            write_json(plan_path, plan)
    if "report" in phases:
        phase_report(run)


if __name__ == "__main__":
    try:
        main()
    except (GuardFailure, ValueError, subprocess.CalledProcessError) as e:
        print(f"kb1bench: {e}", file=sys.stderr)
        sys.exit(1)
