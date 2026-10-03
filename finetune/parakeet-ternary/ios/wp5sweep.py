"""WP5/WP7: informational Mac latency sweeps of eligible deployed pipelines, each paired with C0 in the same guarded job.

DESIGN.md "Measurements" and "Repetition and statistics" (3 warm-up + 10 timed calls per clip, 64 natural clips);
the Mac is shared, so every number here is informational and no claim rests on it. Arm groups:
  (a) WP5 encoder comparison, front end A + F2 (multifunction on cpuAndNeuralEngine, C4 fixed 15 s, G0);
  (b) WP5 pipeline factors on C4 multifunction: front end {c0pre, vdsp} x decode {f0, f1, f2};
  (c) WP7, front end A + F2: the ANE-layout builds C4-ane, C3-ane, C6s8-ane (cpuAndNeuralEngine), C4, C3, C6s8
      multifunction on the GPU backend (cpuAndGPU), and C6s8 multifunction on cpuAndNeuralEngine (within-session anchor).
Every timed combination needs its revision-9 deployed-pipeline record (ios/pipegate.py); the Swift runner verifies
it, component SHA-256s and executable included, and verifies C0 against c0.json.

  ./python ios/wp5sweep.py plan [--groups c]
  ./python ios/wp5sweep.py verify --groups c                               # deployment check, nothing timed
  ./python ios/wp5sweep.py sweep --name wp7 --groups c [--settle-ms 500]   # counterbalanced, manifest-driven
  ./python ios/wp5sweep.py report --sweep SWEEP_ID --tag wp7                # from the sweep manifest only
  ./python ios/wp5sweep.py wp5-manifest                                     # manifest of WP5's migrated runs

Sweep design (review WP7 r1 finding 9): the 64 natural clips are split into halves A and B (alternate clips of each
bucket in clips.json order: 8 + 8 per bucket); half A runs the arms in the listed order, half B in reverse, so every
arm's position in the session is counterbalanced and each arm's 64 clips come from both ends of the session. Per run
one macguard job (ios/sweep_job.sh): disk check, Core ML cache purge, C0 and the arm interleaved clip by clip in one
process ("post-purge loads"; C0 first on even clips), then a fresh process for the arm alone ("subsequent
fresh-process load", arm-only footprint), purge. Before every block parakeet-bench waits --settle-ms and, while
ProcessInfo.thermalState is serious or critical, up to 120 s; block records hold the thermal states.

Manifest (review WP7 r1 finding 6): /mnt/hd/wilderness-labs-stt/parakeet-ios/results/wp5/sweeps/<sweep id>.json lists
the commit and executable SHA-256 the sweep must use, the halves, the order, and every (arm, half) run with its run
id and status, rewritten atomically after each run. Runs land in <arm>/<run id>/ on both machines (never reused),
retrieved completely into a .part directory and renamed. report reads only the runs a complete manifest names
(every expected run status 0, the reviewed executable in every load record, 64-clip coverage per arm) and refuses
anything else; no "latest run" pointer is consulted.
Summaries: ios/results/<tag>/.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

IOS = Path(__file__).resolve().parent
sys.path.insert(0, str(IOS))
sys.path.insert(0, str(IOS.parent))

MAC_REPO = "/Users/ajbarry/workspace/github.com/wilderness-labs-stt"
MAC_IOS = MAC_REPO + "/finetune/parakeet-ternary/ios"
MAC_A = "/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios"
LOCAL = Path("/mnt/hd/wilderness-labs-stt/parakeet-ios/results/wp5")
SUMMARY = IOS / "results" / "wp5"
CANDIDATES = ("C1", "C3", "C4", "C6s2", "C6s4", "C6s8", "C6d4", "C6d8", "C7", "C8")
CACHED_CLIP = "n15-2412-153947-0005"
WARMUPS, TIMED = 3, 10


def arms() -> list[dict]:
    out = []
    for arm in CANDIDATES:
        out.append({"name": f"{arm}-multi-vdsp-f2", "group": "a", "model": "mp2", "arm": arm, "variant": "multi",
                    "frontend": "vdsp", "decode": "f2"})
    out.append({"name": "C4-fixed-vdsp-f2", "group": "a", "model": "mp2", "arm": "C4", "variant": "fixed",
                "frontend": "vdsp", "decode": "f2"})
    out.append({"name": "G0-fixed-vdsp-f2", "group": "a", "model": "c0", "arm": "G0", "variant": "fixed",
                "frontend": "vdsp", "decode": "f2"})
    for fe in ("c0pre", "vdsp"):
        for dec in ("f0", "f1", "f2"):
            if (fe, dec) != ("vdsp", "f2"):  # that one is in (a)
                out.append({"name": f"C4-multi-{fe}-{dec}", "group": "b", "model": "mp2", "arm": "C4", "variant": "multi",
                            "frontend": fe, "decode": dec})
    for arm in ("C4-ane", "C3-ane", "C6s8-ane"):
        out.append({"name": f"{arm}-multi-vdsp-f2", "group": "c", "model": "mp2", "arm": arm, "variant": "multi",
                    "frontend": "vdsp", "decode": "f2"})
    for arm in ("C4", "C3", "C6s8"):
        out.append({"name": f"{arm}-multi-gpu-vdsp-f2", "group": "c", "model": "mp2", "arm": arm, "variant": "multi",
                    "backend": "gpu", "frontend": "vdsp", "decode": "f2"})
    out.append({"name": "C6s8-multi-vdsp-f2-anchor-wp7", "group": "c", "model": "mp2", "arm": "C6s8", "variant": "multi",
                "frontend": "vdsp", "decode": "f2"})
    for a in out:
        a.setdefault("backend", "ane")
    return out


UNITS = {"ane": "cpuAndNeuralEngine", "gpu": "cpuAndGPU", "cpu": "cpuOnly"}


def pipeline_record(a: dict) -> Path:
    return IOS / "results" / "eligibility" / "pipelines" / (
        f"{a['model']}-{a['arm']}-{a['variant']}-{a['backend']}-{a['frontend']}-{a['decode']}.json")


def eligibility(a: dict) -> tuple[bool, str]:
    from mil.eligibility import Ineligible, check

    try:
        rec = check(a["model"], a["arm"], a["variant"], a["backend"])
    except Ineligible as exc:
        return False, str(exc)
    p = pipeline_record(a)
    if not p.exists():
        return False, f"no pipeline record {p.name}"
    prec = json.loads(p.read_text())
    if prec.get("design_revision") != 9 or not prec.get("timing_allowed"):
        return False, f"pipeline record {p.name}: not timing-allowed ({prec.get('reasons')})"
    return True, f"revision {rec.get('design_revision')} encoder and pipeline records, timing allowed"


def mac(cmd: str, timeout: int = 600) -> str:
    from mil.archive import local_config

    out = subprocess.run([*local_config("WP3_MAC_RUN").split(), "--repo", MAC_IOS, "--", "sh", "-c", cmd],
                         capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"Mac command failed ({out.returncode}): {cmd[:120]}: {out.stderr.strip()[-400:]}")
    return out.stdout


def bench_args(a: dict, ids: list[str] | None = None) -> list[str]:
    weights = "b0" if a["model"] == "c0" else "mp2"
    select = ["--ids", ",".join(ids)] if ids else ["--kinds", "natural"]
    args = ["--clips", f"{MAC_IOS}/clips.json", "--pcm", f"{MAC_A}/clips", *select, "--mode", "free",
            "--warmups", str(WARMUPS), "--timed", str(TIMED), "--compute-units", UNITS[a["backend"]],
            "--arm", "custom", "--arm-name", a["name"], "--models", f"{MAC_A}/c0", "--frontend", a["frontend"],
            "--encoder", f"{MAC_A}/arms/{a['model']}/{a['arm']}/{a['variant']}.mlmodelc",
            "--encoder-variant", {"fixed": "fixed15", "multi": "multifunction", "enum": "enumerated"}[a["variant"]],
            "--eligibility", f"{a['model']}:{a['arm']}", "--decode", a["decode"]]
    if a["frontend"] == "vdsp":
        args += ["--frontend-constants", f"{MAC_A}/native/{weights}"]
    if a["decode"] == "f2":
        args += ["--native-weights", f"{MAC_A}/native/{weights}"]
    else:
        args += ["--decoder-models", f"{MAC_A}/arms/mp2/decoder-fp32"]
    return args


RAW = ("arm.jsonl", "c0.jsonl", "cached.jsonl", "cache.log", "job.log")
SWEEPS = LOCAL / "sweeps"


def write_atomic(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, path)


def halves() -> dict[str, list[str]]:
    """Natural clips split alternately within each bucket (clips.json order): A = 0th, 2nd, ...; B = 1st, 3rd, ..."""
    clips = [c for c in json.loads((IOS / "clips.json").read_text())["clips"] if c["kind"] == "natural"]
    out = {"A": [], "B": []}
    for b in sorted({c["bucket"] for c in clips}):
        mine = [c["id"] for c in clips if c["bucket"] == b]
        out["A"] += mine[0::2]
        out["B"] += mine[1::2]
    return out


def run_arm(a: dict, ids: list[str], settle_ms: int) -> dict:
    """One guarded job for one arm on the given clips; returns the run's entry (published only if status 0)."""
    from pipegate import disk_need_gb

    cap = "6G" if a["arm"] == "C1" or a["backend"] != "ane" else "4G"  # C1, GPU: see README
    timeout = 7200 if a["backend"] == "gpu" else 3600  # C6s8's GPU load alone took about 310 s per function (WP6a)
    need = disk_need_gb(a)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out = f"{MAC_A}/results/wp5/{a['name']}/{run_id}"  # unique per run: never reuses or deletes an earlier run
    args = bench_args(a, ids) + ["--pair-c0", f"{MAC_A}/c0", "--c0-out", f"{out}/c0.jsonl",
                                 "--c0-compute-units", "cpuAndNeuralEngine", "--settle-ms", str(settle_ms)]
    quoted = " ".join(shlex.quote(x) for x in args)
    line = (f"mkdir -p {out}; i=0; while :; do ./macguard --rss-cap {cap} --timeout {timeout} -- sh sweep_job.sh {need} {out} "
            f"{CACHED_CLIP} -- {quoted} > {out}/job.log 2>&1; s=$?; [ $s -ne 3 ] && break; i=$((i+1)); "
            f"[ $i -gt 120 ] && break; sleep 60; done; echo $s > {out}/STATUS")
    mac(f"nohup sh -c {shlex.quote(line)} > /dev/null 2>&1 < /dev/null & echo started")
    entry = {"run": run_id, "rss_cap": cap, "timeout_s": timeout, "need_gb": need, "clips": len(ids), "status": None}
    t0, status = time.time(), ""
    while not status and time.time() - t0 < 5 * 3600:
        time.sleep(60)
        try:
            status = mac(f"cat {out}/STATUS 2>/dev/null || true").strip()
        except Exception as exc:  # transient SSH failure: keep polling
            print(f"poll: {exc}", file=sys.stderr, flush=True)
    entry["minutes"] = round((time.time() - t0) / 60, 1)
    if not status.lstrip("-").isdigit():
        entry["error"] = f"no status from {out}"
        return entry
    entry["status"] = int(status)
    if int(status) != 0:
        return entry  # nothing is published for a failed run
    # complete retrieval into a temporary directory, then an atomic rename (review WP4/5 finding 6)
    final = LOCAL / a["name"] / run_id
    tmp = LOCAL / a["name"] / f".{run_id}.part"
    tmp.mkdir(parents=True, exist_ok=False)
    try:
        for f in RAW:
            (tmp / f).write_text(mac(f"cat {out}/{f}"))  # any missing file raises: no partial publish
        for f in ("arm.jsonl", "c0.jsonl", "cached.jsonl"):
            if not load_records(tmp / f):
                raise RuntimeError(f"empty {f}")
    except Exception as exc:
        entry.update({"status": None, "error": f"retrieval failed: {exc}"})
        return entry
    (tmp / "STATUS").write_text(status + "\n")
    os.replace(tmp, final)
    entry["dir"] = str(final)
    return entry


def mac_build() -> dict:
    from pipegate import check_deployment

    commit = check_deployment()
    exe = mac(f"shasum -a 256 {MAC_A}/reviewed/parakeet-bench | cut -d' ' -f1").strip()
    info = json.loads(mac(f"cat {MAC_A}/reviewed/BUILD_INFO.json"))
    if info.get("executable_sha256") != exe:
        raise RuntimeError("the Mac binary is not the one build_reviewed.sh stamped")
    return {"commit": commit, "executable_sha256": exe, "built_from_commit": info.get("commit"),
            "source_tree": info.get("source_tree")}


def cmd_sweep(args) -> int:
    groups = set(args.groups.split(","))
    todo = [a for a in arms() if a["group"] in groups and (not args.only or a["name"] in args.only.split(","))]
    if args.only and len(todo) != len(args.only.split(",")):
        print("unknown arm names in --only", file=sys.stderr)
        return 1
    blocked = [(a["name"], why) for a in todo for ok, why in [eligibility(a)] if not ok]
    if blocked:
        print(f"refusing: not eligible: {blocked}", file=sys.stderr)
        return 1
    build = mac_build()
    stale = [a["name"] for a in todo
             if json.loads(pipeline_record(a).read_text())["components"].get("executable_sha256") != build["executable_sha256"]]
    if stale:
        print(f"refusing: the Mac binary is not the one the pipeline gates ran ({stale})", file=sys.stderr)
        return 1
    h = halves()
    order = [(a["name"], "A") for a in todo] + [(a["name"], "B") for a in reversed(todo)]
    sweep_id = f"{args.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    manifest = {"sweep_id": sweep_id, "build": build, "settle_ms": args.settle_ms, "warmups": WARMUPS, "timed": TIMED,
                "arms": [a["name"] for a in todo], "halves": h, "order": order, "runs": {}, "complete": False,
                "design": "half A in the listed arm order, half B in reverse (position counterbalanced)"}
    path = SWEEPS / f"{sweep_id}.json"
    write_atomic(path, manifest)
    by_name = {a["name"]: a for a in todo}
    restored: set = set()
    for i, (name, half) in enumerate(order):
        a = by_name[name]
        ensure_model(a, restored)
        entry = run_arm(a, h[half], args.settle_ms)
        entry["position"] = i
        manifest["runs"].setdefault(name, {})[half] = entry
        write_atomic(path, manifest)
        print(json.dumps({"arm": name, "half": half, **entry}), flush=True)
        later = {(by_name[n]["model"], by_name[n]["arm"]) for n, _ in order[i + 1:]}
        for key in sorted(restored - later):  # archive arms this run restored once no later run needs them
            subprocess.run([str(IOS.parent / "python"), str(IOS / "mil" / "archive.py"), "out", *key], check=True)
            restored.discard(key)
    manifest["complete"] = all(manifest["runs"].get(n, {}).get(hh, {}).get("status") == 0 for n, hh in order)
    write_atomic(path, manifest)
    print(json.dumps({"sweep_id": sweep_id, "complete": manifest["complete"], "manifest": str(path)}))
    return 0 if manifest["complete"] else 1


def cmd_verify(args) -> int:
    """Deployment check before a sweep (no model loaded, nothing timed): the Mac checkout is this commit and clean,
    its binary is the one the pipeline gates ran, and `parakeet-bench run --verify-only` accepts every arm with its
    exact sweep arguments (pipeline record + component SHA-256s + WP3 record) and verifies C0 against c0.json.
    Archived encoder packages are restored first (mil/archive.py back: guarded, disk-checked, SHA-256-verified) and
    archived again (mil/archive.py out) once no later arm needs them (review WP7 r2 finding 4)."""
    groups = set(args.groups.split(","))
    todo = [a for a in arms() if a["group"] in groups and (not args.only or a["name"] in args.only.split(","))]
    build = mac_build()
    failures = []
    restored: set = set()
    for i, a in enumerate(todo):
        try:
            ensure_model(a, restored)
        except Exception as exc:
            failures.append(f"{a['name']}: restore failed: {exc}")
            continue
        rec = json.loads(pipeline_record(a).read_text()) if pipeline_record(a).exists() else {}
        if rec.get("components", {}).get("executable_sha256") != build["executable_sha256"]:
            failures.append(f"{a['name']}: record executable differs from the Mac binary")
        argv = bench_args(a, halves()["A"]) + ["--pair-c0", f"{MAC_A}/c0", "--c0-out", f"{MAC_A}/wp7/verify-c0.jsonl",
                                                "--c0-compute-units", "cpuAndNeuralEngine", "--out", f"{MAC_A}/wp7/verify.jsonl",
                                                "--verify-only"]
        cmd = f"./macguard --rss-cap 1G --timeout 600 -- {MAC_A}/reviewed/parakeet-bench run " + " ".join(shlex.quote(x) for x in argv)
        line = (f"i=0; while :; do {cmd} > {MAC_A}/wp7/verify.out 2>&1; s=$?; [ $s -ne 3 ] && break; i=$((i+1)); "
                f"[ $i -gt 30 ] && break; sleep 30; done; grep -v '^macguard: ' {MAC_A}/wp7/verify.out; "
                f"rm -f {MAC_A}/wp7/verify.out; echo \"exit $s\"")
        try:
            outp = mac(f"mkdir -p {MAC_A}/wp7; {line}", timeout=1500).strip().splitlines()
            if not outp or outp[-1] != "exit 0":
                raise RuntimeError(" | ".join(outp[-3:]))
            doc = json.loads(outp[-2])
            ok = doc.get("verified") and doc.get("c0_identity", {}).get("verified")
            print(f"{a['name']}: {'verified' if ok else 'FAILED'}", flush=True)
            if not ok:
                failures.append(a["name"])
        except Exception as exc:
            failures.append(f"{a['name']}: {str(exc)[-300:]}")
            print(f"{a['name']}: FAILED {str(exc)[-300:]}", flush=True)
        later = {(b["model"], b["arm"]) for b in todo[i + 1:]}
        for key in sorted(restored - later):
            subprocess.run([str(IOS.parent / "python"), str(IOS / "mil" / "archive.py"), "out", *key], check=True)
            restored.discard(key)
    print(json.dumps({"build": build, "failures": failures}))
    return 1 if failures else 0


def cmd_wp5_manifest(args) -> int:
    """A manifest for WP5's runs (migrated into <arm>/wp5-20261003/; one run of all 64 clips per arm, no halves,
    build not recorded then): makes the WP5 report manifest-driven too. Marked legacy."""
    names = [a["name"] for a in arms() if a["group"] in ("a", "b")]
    runs = {}
    for n in names:
        d = LOCAL / n / "wp5-20261003"
        status = (d / "STATUS").read_text().strip() if (d / "STATUS").exists() else None
        runs[n] = {"all": {"run": "wp5-20261003", "status": int(status) if status else None, "dir": str(d)}}
    natural = [c["id"] for c in json.loads((IOS / "clips.json").read_text())["clips"] if c["kind"] == "natural"]
    manifest = {"sweep_id": "wp5-20261003", "legacy": "WP5 runs, before manifests, session ids, C0 identity and build "
                "identity were recorded", "build": None, "arms": names, "halves": {"all": natural},
                "order": [(n, "all") for n in names], "runs": runs,
                "complete": all(v["all"]["status"] == 0 for v in runs.values())}
    write_atomic(SWEEPS / "wp5-20261003.json", manifest)
    print(json.dumps({"complete": manifest["complete"], "arms": len(names)}))
    return 0


def ensure_model(a: dict, restored: set) -> None:
    path = f"{MAC_A}/arms/{a['model']}/{a['arm']}/{a['variant']}.mlmodelc"
    if mac(f"test -d {path} && echo yes || true").strip() == "yes":
        return
    subprocess.run([str(IOS.parent / "python"), str(IOS / "mil" / "archive.py"), "back", a["model"], a["arm"],
                    f"{a['variant']}.mlmodelc"], check=True)
    restored.add((a["model"], a["arm"]))


def cmd_plan(args) -> None:
    groups = set(args.groups.split(","))
    h = halves()
    print(f"halves: A {len(h['A'])} clips, B {len(h['B'])} clips")
    for a in [a for a in arms() if a["group"] in groups]:
        ok, why = eligibility(a)
        print(f"{a['name']:24s} {'TIME' if ok else 'skip'}  {why[:150]}")


LOAD_NOTE = ("Loads: 'post-purge load' = the first load after this binary's Core ML cache directory was purged; "
             "'subsequent fresh-process load' = a new process loading the arm afterwards. Whether Core ML specialized "
             "or reused a cache in either is not established (no Instruments cache events).")


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def identity_problems(manifest: dict, a: dict, d: Path) -> list[str]:
    """Do the run's paired arm, paired C0 and cached-load records describe exactly the manifest's arm and protocol?
    (review WP7 r2 finding 5: a report must not label one arm's records with another arm's name)"""
    legacy = bool(manifest.get("legacy"))
    units = UNITS[a["backend"]]
    loads = {f: next((r for r in load_records(d / f) if r.get("record") == "load"), {}) for f in ("arm.jsonl", "c0.jsonl", "cached.jsonl")}
    problems = []

    def want(f: str, key: str, value) -> None:
        if loads[f].get(key) != value:
            problems.append(f"{f}: {key} = {loads[f].get(key)!r}, expected {value!r}")

    for f in ("arm.jsonl", "cached.jsonl"):
        want(f, "arm", a["name"])
        want(f, "compute_units", units)
        spec = loads[f].get("arm_spec") or {}
        if spec.get("front_end") != a["frontend"] or spec.get("decode") != a["decode"] or spec.get("encoder_input") is not None:
            problems.append(f"{f}: arm_spec {spec} is not front end {a['frontend']} + {a['decode']}")
        e = loads[f].get("eligibility") or {}
        if legacy:  # WP5: the encoder record's file name
            if not isinstance(e, dict) or e.get("record") != f"{a['model']}-{a['arm']}-{a['variant']}-{a['backend']}.json":
                problems.append(f"{f}: eligibility record {e.get('record') if isinstance(e, dict) else e!r} is not this arm's")
        else:
            enc = ((e.get("components") or {}).get("encoder") or {}) if isinstance(e, dict) else {}
            if not isinstance(e, dict) or e.get("pipeline_record") != pipeline_record(a).name \
                    or (enc.get("model"), enc.get("arm"), enc.get("variant")) != (a["model"], a["arm"], a["variant"]) \
                    or (e.get("components") or {}).get("compute_units") != units:
                problems.append(f"{f}: eligibility is not the pipeline record {pipeline_record(a).name} on {units}")
    want("arm.jsonl", "mode", "free")
    want("arm.jsonl", "warmups", manifest.get("warmups", WARMUPS))
    want("arm.jsonl", "timed", manifest.get("timed", TIMED))
    want("cached.jsonl", "warmups", 1)
    want("cached.jsonl", "timed", 0)
    want("c0.jsonl", "arm", "C0")
    want("c0.jsonl", "compute_units", "cpuAndNeuralEngine")
    want("c0.jsonl", "preprocessor_units", "cpuOnly")
    for key in ("mode", "warmups", "timed"):
        want("c0.jsonl", key, loads["arm.jsonl"].get(key))
    if not legacy:
        want("arm.jsonl", "settle_ms", manifest.get("settle_ms"))
        want("c0.jsonl", "settle_ms", manifest.get("settle_ms"))
        if (loads["c0.jsonl"].get("pairing") or {}).get("arm") != a["name"]:
            problems.append("c0.jsonl: pairing block names another arm")
    return problems


def verified_runs(manifest: dict, name: str) -> list[Path]:
    """The run directories the manifest names for one arm, after every check; SystemExit otherwise."""
    problems, dirs = [], []
    for half in manifest["halves"]:
        e = manifest["runs"].get(name, {}).get(half)
        if not e or e.get("status") != 0 or not e.get("dir"):
            problems.append(f"{name} half {half}: {e and (e.get('status'), e.get('error'))}")
            continue
        d = Path(e["dir"])
        if d.parent != LOCAL / name or d.name != e["run"] or not all((d / f).exists() for f in RAW) \
                or (d / "STATUS").read_text().strip() != "0":
            problems.append(f"{name} half {half}: run directory {d} incomplete or not the manifest's")
            continue
        problems += [f"{name} half {half}: {x}" for x in identity_problems(manifest, spec_of(name), d)]
        build = manifest.get("build")
        if build:
            for f in ("arm.jsonl", "c0.jsonl", "cached.jsonl"):
                load = next((r for r in load_records(d / f) if r.get("record") == "load"), {})
                if load.get("executable_sha256") != build["executable_sha256"]:
                    problems.append(f"{name} half {half}: {f} was not made by the sweep's executable")
            load = next(r for r in load_records(d / "arm.jsonl") if r.get("record") == "load")
            if sorted(load.get("clip_ids") or []) != sorted(manifest["halves"][half]):
                problems.append(f"{name} half {half}: clips differ from the manifest's half")
        dirs.append(d)
    if problems:
        raise SystemExit("incomplete sweep: " + "; ".join(problems))
    return dirs


def spec_of(name: str) -> dict:
    return {a["name"]: a for a in arms()}[name]


def cmd_report(args) -> None:
    manifest = json.loads((SWEEPS / f"{args.sweep}.json").read_text())
    if not manifest.get("complete"):
        raise SystemExit(f"sweep {args.sweep} is not complete: refusing to report")
    natural = sorted(c["id"] for c in json.loads((IOS / "clips.json").read_text())["clips"] if c["kind"] == "natural")
    if sorted(c for v in manifest["halves"].values() for c in v) != natural:
        raise SystemExit("the sweep's halves do not cover the 64 natural clips exactly once")
    summary_dir = IOS / "results" / args.tag
    summary_dir.mkdir(parents=True, exist_ok=True)
    spec = {a["name"]: a for a in arms()}
    rows = []
    for name in manifest["arms"]:
        a = spec[name]
        dirs = verified_runs(manifest, name)
        out = summary_dir / f"{a['name']}.summary.json"
        ref = [] if a["model"] == "c0" else ["--ref", "/mnt/hd/wilderness-labs-stt/parakeet-ios/native/mp2"]
        subprocess.run([str(IOS.parent / "python"), str(IOS / "armreport.py"), *[str(d / "arm.jsonl") for d in dirs],
                        "--baseline", *[str(d / "c0.jsonl") for d in dirs], "--out", str(out), *ref],
                       check=True, stdout=subprocess.DEVNULL)
        c0out = summary_dir / f"{a['name']}.c0block.summary.json"
        subprocess.run([str(IOS.parent / "python"), str(IOS / "armreport.py"), *[str(d / "c0.jsonl") for d in dirs],
                        "--out", str(c0out)], check=True, stdout=subprocess.DEVNULL)
        s, c0s = json.loads(out.read_text()), json.loads(c0out.read_text())
        arm_loads = [next((r for r in load_records(d / "arm.jsonl") if r.get("record") == "load"), {}) for d in dirs]
        cached = [load_records(d / "cached.jsonl") for d in dirs]
        cloads = [next((r for r in c if r.get("record") == "load"), {}) for c in cached]
        cends = [next((r for r in c if r.get("record") == "end"), {}) for c in cached]
        enc_loads = [(l.get("encoder") or {}) for l in arm_loads]
        peaks = [e.get("phys_footprint_peak_mb") for e in cends if e.get("phys_footprint_peak_mb") is not None]
        rows.append({
            "arm": a["name"], "group": a["group"], "label": s["label"],
            "wer": s.get("wer_vs_reference", {}).get("wer"),
            "tokens_equal_mp2_reference": s.get("token_sequence_equal_to_model_reference"),
            "paired_total": {b: v["total"] for b, v in s["paired_vs_baseline"]["per_bucket"].items()},
            "paired_total_all": s["paired_vs_baseline"]["all_clips"]["total"],
            "encoder_ms_typical": {b: v.get("encoder_ms_typical") for b, v in s["per_bucket"].items()},
            "encoder_ms_p95_hd": {b: v.get("encoder_ms_p95_hd") for b, v in s["per_bucket"].items()},
            "total_ms_typical": {b: v.get("total_ms_typical") for b, v in s["per_bucket"].items()},
            "total_ms_p95_hd": {b: v.get("total_ms_p95_hd") for b, v in s["per_bucket"].items()},
            "preprocess_ms_typical": {b: v.get("preprocess_ms_typical") for b, v in s["per_bucket"].items()},
            "decode_ms_typical": {b: v.get("decode_ms_typical") for b, v in s["per_bucket"].items()},
            "paired_p95_ratio": {b: {"ratio": v["total"]["p95_ratio"], "ci95": v["total"]["p95_ratio_ci95"]}
                                 for b, v in s["paired_vs_baseline"]["per_bucket"].items()},
            "runs": [d.name for d in dirs], "positions": [manifest["runs"][name][h].get("position") for h in manifest["halves"]],
            "backend": a["backend"], "thermal": s.get("thermal"),
            "c0_post_purge_load_ms": [next((r for r in load_records(d / "c0.jsonl") if r.get("record") == "load"), {}).get("load_ms")
                                      for d in dirs],
            "wer_b0_same_clips": s.get("b0_wer_vs_reference_same_clips", {}).get("wer"),
            "rss_cap": "6G" if a["arm"] == "C1" or a["backend"] != "ane" else "4G",
            "c0_total_ms_typical": {b: v.get("total_ms_typical") for b, v in c0s["per_bucket"].items()},
            "physical_calls": s["calls"]["physical_totals"],
            "post_purge_load_ms": {"encoder": [e.get("load_ms") for e in enc_loads],
                                   "encoder_total": [e.get("total_ms") for e in enc_loads],
                                   "compile_ms": [e.get("compile_ms") for e in enc_loads],
                                   "decode": [l.get("decode_load_ms") for l in arm_loads]},
            "subsequent_fresh_process_load_ms": {"encoder": [(c.get("encoder") or {}).get("load_ms") for c in cloads],
                                                 "encoder_total": [(c.get("encoder") or {}).get("total_ms") for c in cloads],
                                                 "decode": [c.get("decode_load_ms") for c in cloads]},
            "phys_footprint_peak_mb_arm_only": max(peaks) if peaks else None,
            "phys_footprint_peak_mb_with_c0": s["phys_footprint_mb"]["peak"],
            "cache_log": [(d / "cache.log").read_text().strip().splitlines() for d in dirs],
        })
    (summary_dir / "sweep.json").write_text(json.dumps({"informational": "Mac (M1 Pro) is shared; no claims",
                                                       "sweep_id": manifest["sweep_id"], "build": manifest.get("build"),
                                                       "design": manifest.get("design") or manifest.get("legacy"),
                                                       "load_labels": LOAD_NOTE, "rows": rows}, indent=1) + "\n")
    lines = ["| arm | total ratio vs C0 (2 / 4 / 8 / 15 s) [95% CI] | encoder ms (2 / 4 / 8 / 15 s) | load post-purge / "
             "subsequent fresh process (enc ms) | footprint MB |",
             "|---|---|---|---|---|"]
    for r in rows:
        ratio = " / ".join(f"{r['paired_total'][b]['typical_ratio']:.2f} [{r['paired_total'][b]['typical_ratio_ci95'][0]:.2f}, "
                           f"{r['paired_total'][b]['typical_ratio_ci95'][1]:.2f}]" for b in ("2", "4", "8", "15") if b in r["paired_total"])
        enc = " / ".join(f"{r['encoder_ms_typical'][b]:.1f}" for b in ("2", "4", "8", "15") if b in r["encoder_ms_typical"])
        fl = ", ".join(str(round(x)) if x is not None else "-" for x in r["post_purge_load_ms"]["encoder_total"])
        cl = ", ".join(str(round(x)) if x is not None else "-" for x in r["subsequent_fresh_process_load_ms"]["encoder_total"])
        lines.append(f"| {r['arm']} | {ratio} | {enc} | {fl} / {cl} | "
                     f"{r['phys_footprint_peak_mb_arm_only'] and round(r['phys_footprint_peak_mb_arm_only'])} |")
    lines += ["", "| arm | WER % | tokens = mp2 FP32 reference | preprocess / decode ms (15 s) | physical calls (64 clips, 1 call each) | C0 encoder post-purge load ms |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        eq = r["tokens_equal_mp2_reference"]
        lines.append(f"| {r['arm']} | {100 * r['wer']:.2f} | {eq['clips'] if eq else '-'}/{eq['of'] if eq else '-'} | "
                     f"{r['preprocess_ms_typical'].get('15', 0):.1f} / {r['decode_ms_typical'].get('15', 0):.1f} | "
                     f"{', '.join(f'{k} {v}' for k, v in r['physical_calls'].items())} | "
                     f"{', '.join(str(round((x or {}).get('Encoder', 0))) for x in r['c0_post_purge_load_ms'])} |")
    (summary_dir / "sweep_table.md").write_text(f"Sweep {manifest['sweep_id']}. Informational (shared Mac M1 Pro, macOS 27; no claims). " + LOAD_NOTE
                                                 + "\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--groups", default="a,b,c"); p.set_defaults(func=cmd_plan)
    p = sub.add_parser("sweep"); p.add_argument("--name", required=True); p.add_argument("--groups", default="c")
    p.add_argument("--only", help="comma-separated arm names (default: every arm of the groups)")
    p.add_argument("--settle-ms", type=int, default=500)
    p.set_defaults(func=cmd_sweep)
    p = sub.add_parser("report"); p.add_argument("--sweep", required=True, help="sweep id (manifest under results/wp5/sweeps)")
    p.add_argument("--tag", required=True, help="summary directory under ios/results (wp5 or wp7)")
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("verify"); p.add_argument("--groups", default="c"); p.add_argument("--only")
    p.set_defaults(func=cmd_verify)
    sub.add_parser("wp5-manifest").set_defaults(func=cmd_wp5_manifest)
    args = parser.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
