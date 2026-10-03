"""WP5: informational Mac latency sweep of the eligible arms, each paired with C0 in the same guarded job (NixOS).

DESIGN.md "Measurements" and "Repetition and statistics" (3 warm-up + 10 timed calls per clip, 64 natural clips);
the Mac is shared, so every number here is informational and no claim rests on it. Arms (WP5 package):
  (a) encoder comparison, front end A + F2: every encoder arm with a passing revision-8 eligibility record as a
      multifunction model on cpuAndNeuralEngine, plus C4 fixed 15 s and G0 (graph control, C0's weights; F2 and
      front end A with B0's decoder/joint and constants);
  (b) pipeline factors on C4 multifunction: front end {c0pre, vdsp} x decode {f0, f1, f2}; F0/F1 use the FP32
      decoder models (DESIGN.md revision 8, "Decoder and joint precision");
  (c) WP7, new arms with front end A + F2: the ANE-layout builds C4-ane, C3-ane, C6s8-ane (cpuAndNeuralEngine) and
      C4, C3, C6s8 multifunction on the GPU backend (cpuAndGPU), plus C6s8 multifunction on cpuAndNeuralEngine
      re-timed in the same session as a within-session anchor.
Every timed combination needs its deployed-pipeline record (ios/pipegate.py; the Swift runner verifies it).

  ./python ios/wp5sweep.py plan                 # arms and their eligibility (mil.eligibility.check)
  ./python ios/wp5sweep.py run [--only A,B]     # restore models, time on the Mac, fetch, archive restored arms out
  ./python ios/wp5sweep.py report               # armreport per arm (paired vs its C0 block) + results/wp5 table

Per arm, one macguard job (ios/sweep_job.sh): Core ML cache purged, C0 and the arm interleaved clip by clip
("post-purge loads"), then a fresh process for the arm alone ("subsequent fresh-process load", arm-only footprint),
cache purged. Disk headroom per arm is measured (package size + Core ML cache estimate, ios/pipegate.disk_need_gb).
The Swift CLI itself refuses an arm without a passing pipeline record; C0 is exempt (baseline).
Raw records: /mnt/hd/wilderness-labs-stt/parakeet-ios/results/wp5/<arm>/<run id>/ (Mac: <artifacts>/results/wp5/<arm>/
<run id>/), published atomically with <arm>/LATEST naming the run; report reads LATEST runs with status 0 only.
Summaries: ios/results/wp5/ (groups a, b) and ios/results/wp7/ (group c).
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
    if prec.get("design_revision") != 8 or not prec.get("timing_allowed"):
        return False, f"pipeline record {p.name}: not timing-allowed ({prec.get('reasons')})"
    return True, f"revision {rec.get('design_revision')} encoder and pipeline records, timing allowed"


def mac(cmd: str, timeout: int = 600) -> str:
    from mil.archive import local_config

    out = subprocess.run([*local_config("WP3_MAC_RUN").split(), "--repo", MAC_IOS, "--", "sh", "-c", cmd],
                         capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"Mac command failed ({out.returncode}): {cmd[:120]}: {out.stderr.strip()[-400:]}")
    return out.stdout


def bench_args(a: dict) -> list[str]:
    weights = "b0" if a["model"] == "c0" else "mp2"
    args = ["--clips", f"{MAC_IOS}/clips.json", "--pcm", f"{MAC_A}/clips", "--kinds", "natural", "--mode", "free",
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


def run_arm(a: dict, cap: str = "4G") -> dict:
    from pipegate import disk_need_gb

    need = disk_need_gb(a)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out = f"{MAC_A}/results/wp5/{a['name']}/{run_id}"  # unique per run: never reuses or deletes an earlier run
    args = bench_args(a) + ["--pair-c0", f"{MAC_A}/c0", "--c0-out", f"{out}/c0.jsonl", "--c0-compute-units", "cpuAndNeuralEngine"]
    quoted = " ".join("'" + x.replace("'", "'\\''") + "'" for x in args)
    line = (f"mkdir -p {out}; i=0; while :; do ./macguard --rss-cap {cap} --timeout 3600 -- sh sweep_job.sh {need} {out} "
            f"{CACHED_CLIP} -- {quoted} > {out}/job.log 2>&1; s=$?; [ $s -ne 3 ] && break; i=$((i+1)); "
            f"[ $i -gt 120 ] && break; sleep 60; done; echo $s > {out}/STATUS")
    mac(f"nohup sh -c {shlex.quote(line)} > /dev/null 2>&1 < /dev/null & echo started")
    t0, status = time.time(), ""
    while not status and time.time() - t0 < 5 * 3600:
        time.sleep(60)
        try:
            status = mac(f"cat {out}/STATUS 2>/dev/null || true").strip()
        except Exception as exc:  # transient SSH failure: keep polling
            print(f"poll: {exc}", file=sys.stderr, flush=True)
    if not status.lstrip("-").isdigit():
        raise RuntimeError(f"{a['name']}: no status from {out}; stopping")
    result = {"name": a["name"], "status": int(status), "rss_cap": cap, "need_gb": need, "run": run_id}
    if int(status) != 0:
        return result  # nothing is published for a failed run
    # complete retrieval into a temporary directory, then an atomic rename and LATEST (review WP4/5 finding 6)
    final = LOCAL / a["name"] / run_id
    tmp = LOCAL / a["name"] / f".{run_id}.part"
    tmp.mkdir(parents=True, exist_ok=False)
    for f in RAW:
        (tmp / f).write_text(mac(f"cat {out}/{f}"))  # any missing file raises: no partial publish
    for f in ("arm.jsonl", "c0.jsonl", "cached.jsonl"):
        if not load_records(tmp / f):
            raise RuntimeError(f"{a['name']}: empty {f}; not published")
    (tmp / "STATUS").write_text(status + "\n")
    os.replace(tmp, final)
    latest = LOCAL / a["name"] / ".LATEST.part"
    latest.write_text(run_id + "\n")
    os.replace(latest, LOCAL / a["name"] / "LATEST")
    return result


def latest_run(name: str) -> Path | None:
    """The published run of an arm (LATEST), only if it completed with status 0."""
    p = LOCAL / name / "LATEST"
    if not p.exists():
        return None
    d = LOCAL / name / p.read_text().strip()
    if not (d / "STATUS").exists() or (d / "STATUS").read_text().strip() != "0" or not all((d / f).exists() for f in RAW):
        return None
    return d


def ensure_model(a: dict, restored: set) -> None:
    path = f"{MAC_A}/arms/{a['model']}/{a['arm']}/{a['variant']}.mlmodelc"
    if mac(f"test -d {path} && echo yes || true").strip() == "yes":
        return
    subprocess.run([str(IOS.parent / "python"), str(IOS / "mil" / "archive.py"), "back", a["model"], a["arm"],
                    f"{a['variant']}.mlmodelc"], check=True)
    restored.add((a["model"], a["arm"]))


def cmd_plan(args) -> None:
    for a in arms():
        ok, why = eligibility(a)
        print(f"{a['name']:24s} {'TIME' if ok else 'skip'}  {why[:150]}")


def cmd_run(args) -> None:
    only = set(args.only.split(",")) if args.only else None
    restored: set = set()
    log = []
    todo = [a for a in arms() if not only or a["name"] in only]
    for i, a in enumerate(todo):
        ok, why = eligibility(a)
        if not ok:
            log.append({"name": a["name"], "skipped": why})
            print(f"skip {a['name']}: {why}", flush=True)
            continue
        ensure_model(a, restored)
        t0 = time.time()
        r = run_arm(a, "6G" if a["arm"] == "C1" else args.cap)  # dense FP16 C1: see README
        r["minutes"] = round((time.time() - t0) / 60, 1)
        log.append(r)
        print(json.dumps(r), flush=True)
        later = {(b["model"], b["arm"]) for b in todo[i + 1:]}
        for key in sorted(restored - later):  # archive arms this run restored once no later arm needs them
            subprocess.run([str(IOS.parent / "python"), str(IOS / "mil" / "archive.py"), "out", *key], check=True)
            restored.discard(key)
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / f"run-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(log, indent=1) + "\n")


LOAD_NOTE = ("Loads: 'post-purge load' = the first load after this binary's Core ML cache directory was purged; "
             "'subsequent fresh-process load' = a new process loading the arm afterwards. Whether Core ML specialized "
             "or reused a cache in either is not established (no Instruments cache events).")


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def cmd_report(args) -> None:
    groups = set(args.groups.split(","))
    summary_dir = IOS / "results" / args.tag
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for a in arms():
        if a["group"] not in groups:
            continue
        d = latest_run(a["name"])
        if d is None:
            print(f"{a['name']}: no published status-0 run; not reported", file=sys.stderr)
            continue
        out = summary_dir / f"{a['name']}.summary.json"
        ref = [] if a["model"] == "c0" else ["--ref", "/mnt/hd/wilderness-labs-stt/parakeet-ios/native/mp2"]
        subprocess.run([str(IOS.parent / "python"), str(IOS / "armreport.py"), str(d / "arm.jsonl"), "--baseline",
                        str(d / "c0.jsonl"), "--out", str(out), *ref], check=True, stdout=subprocess.DEVNULL)
        c0out = summary_dir / f"{a['name']}.c0block.summary.json"
        subprocess.run([str(IOS.parent / "python"), str(IOS / "armreport.py"), str(d / "c0.jsonl"), "--out", str(c0out)],
                       check=True, stdout=subprocess.DEVNULL)
        s, c0s = json.loads(out.read_text()), json.loads(c0out.read_text())
        arm_load = next((r for r in load_records(d / "arm.jsonl") if r.get("record") == "load"), {})
        cached = load_records(d / "cached.jsonl")
        cload = next((r for r in cached if r.get("record") == "load"), {})
        cend = next((r for r in cached if r.get("record") == "end"), {})
        enc_load = (arm_load.get("encoder") or {})
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
            "run": d.name, "backend": a["backend"],
            "c0_post_purge_load_ms": next((r for r in load_records(d / "c0.jsonl") if r.get("record") == "load"), {}).get("load_ms"),
            "wer_b0_same_clips": s.get("b0_wer_vs_reference_same_clips", {}).get("wer"),
            "rss_cap": "6G" if a["arm"] == "C1" else "4G",
            "c0_total_ms_typical": {b: v.get("total_ms_typical") for b, v in c0s["per_bucket"].items()},
            "physical_calls": s["calls"]["physical_totals"],
            "post_purge_load_ms": {"encoder": enc_load.get("load_ms"), "encoder_total": enc_load.get("total_ms"),
                              "compile_ms": enc_load.get("compile_ms"), "decode": arm_load.get("decode_load_ms")},
            "subsequent_fresh_process_load_ms": {"encoder": (cload.get("encoder") or {}).get("load_ms"),
                               "encoder_total": (cload.get("encoder") or {}).get("total_ms"),
                               "decode": cload.get("decode_load_ms")},
            "phys_footprint_peak_mb_arm_only": cend.get("phys_footprint_peak_mb"),
            "phys_footprint_peak_mb_with_c0": s["phys_footprint_mb"]["peak"],
            "cache_log": (d / "cache.log").read_text().strip().splitlines() if (d / "cache.log").exists() else None,
        })
    (summary_dir / "sweep.json").write_text(json.dumps({"informational": "Mac (M1 Pro) is shared; no claims",
                                                       "load_labels": LOAD_NOTE, "rows": rows}, indent=1) + "\n")
    lines = ["| arm | total ratio vs C0 (2 / 4 / 8 / 15 s) [95% CI] | encoder ms (2 / 4 / 8 / 15 s) | load post-purge / "
             "subsequent fresh process (enc ms) | footprint MB |",
             "|---|---|---|---|---|"]
    for r in rows:
        ratio = " / ".join(f"{r['paired_total'][b]['typical_ratio']:.2f} [{r['paired_total'][b]['typical_ratio_ci95'][0]:.2f}, "
                           f"{r['paired_total'][b]['typical_ratio_ci95'][1]:.2f}]" for b in ("2", "4", "8", "15") if b in r["paired_total"])
        enc = " / ".join(f"{r['encoder_ms_typical'][b]:.1f}" for b in ("2", "4", "8", "15") if b in r["encoder_ms_typical"])
        fl = r["post_purge_load_ms"]["encoder_total"]
        cl = r["subsequent_fresh_process_load_ms"]["encoder_total"]
        lines.append(f"| {r['arm']} | {ratio} | {enc} | {fl and round(fl)} / {cl and round(cl)} | "
                     f"{r['phys_footprint_peak_mb_arm_only'] and round(r['phys_footprint_peak_mb_arm_only'])} |")
    lines += ["", "| arm | WER % | tokens = mp2 FP32 reference | preprocess / decode ms (15 s) | physical calls (64 clips, 1 call each) | C0 encoder post-purge load ms |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        eq = r["tokens_equal_mp2_reference"]
        lines.append(f"| {r['arm']} | {100 * r['wer']:.2f} | {eq['clips'] if eq else '-'}/{eq['of'] if eq else '-'} | "
                     f"{r['preprocess_ms_typical'].get('15', 0):.1f} / {r['decode_ms_typical'].get('15', 0):.1f} | "
                     f"{', '.join(f'{k} {v}' for k, v in r['physical_calls'].items())} | "
                     f"{round((r['c0_post_purge_load_ms'] or {}).get('Encoder', 0))} |")
    (summary_dir / "sweep_table.md").write_text("Informational (shared Mac M1 Pro, macOS 27; no claims). " + LOAD_NOTE
                                                 + "\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan").set_defaults(func=cmd_plan)
    p = sub.add_parser("run"); p.add_argument("--only")
    p.add_argument("--cap", default="4G", help="macguard RSS cap; 6G for dense FP16 multifunction arms whose mapped "
                   "per-function programs exceed 4 GB of group RSS (justified in README)")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("--groups", default="a,b", help="arm groups to report (a,b = WP5; c = WP7)")
    p.add_argument("--tag", default="wp5", help="summary directory under ios/results (wp5 or wp7)")
    p.set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
