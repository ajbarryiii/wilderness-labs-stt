"""WP3 summary: builds, gate 2, gates 3-5, compute plans and probes in one table (pure JSON; any machine).

  python mil/report.py   # reads ios/results/{builds,gate2,gates,probes,diag}; writes results/wp3_summary.json and
                         # results/wp3_summary_table.txt (a Markdown table; *.md under results/ is git-ignored)
"""
from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
IOS_DIR = RES.parent


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def plan_share(plan: dict | None) -> dict:
    """Per function: ANE share of estimated cost and op counts by device (cpuAndNeuralEngine plan)."""
    if not plan:
        return {}
    out = {}
    for fn, f in plan.get("functions", {}).items():
        if "error" in f:
            out[fn] = {"error": f["error"][:200]}
            continue
        cost = f.get("estimated_cost_share_by_preferred_device", {})
        ops = f.get("ops_with_usage_by_preferred_device", {})
        out[fn] = {"ane_cost": round(cost.get("ane", 0.0), 4), "ops": ops,
                   "cpu_ops": f.get("operators_by_preferred_device", {}).get("cpu", {})}
    return out


def main() -> None:
    rows = []
    for p in sorted((RES / "builds").glob("*.json")):
        m = load(p)
        if "models" in m:  # decoder
            for name, e in m["models"].items():
                rows.append({"model": m["model"], "arm": name, "variant": "-", "mlmodelc_mb": round(e.get("compile", {}).get("mlmodelc_bytes", 0) / 2 ** 20, 1),
                             "compile_s": e.get("compile", {}).get("compile_s"),
                             "plan": plan_share(e.get("compute_plan", {}).get("cpuAndNeuralEngine"))})
            continue
        row = {"model": m["model"], "arm": m["arm"], "variant": m["variant"],
               "mlpackage_mb": round(m["sizes"]["mlpackage_bytes"] / 2 ** 20, 1),
               "mlmodelc_mb": round(m.get("compile", {}).get("mlmodelc_bytes", 0) / 2 ** 20, 1),
               "compile_s": m.get("compile", {}).get("compile_s"), "convert_s": m["timing_s"]["convert"],
               "peak_rss_mb": max(m.get("peak_rss_mb_after_save") or 0, m.get("peak_rss_mb_after_compile") or 0),
               "encoded_mb": round(m["encoding"].get("encoded_bytes_total", 0) / 2 ** 20, 1) if "encoded_bytes_total" in m["encoding"] else None,
               "plan": plan_share(m.get("compute_plan", {}).get("cpuAndNeuralEngine"))}
        g2 = load(RES / "gate2" / f"{m['model']}-{m['arm']}.json")
        if g2:
            row["gate2"] = {"bit_exact": g2["bit_exact"], "modules": g2["modules_checked"]}
        for units in ("cpuAndNeuralEngine", "cpuOnly"):
            g = load(RES / "gates" / f"{m['model']}-{m['arm']}-{m['variant']}-{units}.json")
            if not g:
                continue
            s = g["summary"]
            if "gate4_encoder" not in s:  # G0: compared with C0's encoder, no heads
                row[f"gates_{units}"] = {"g0_vs_c0": s.get("pass"), "g4_rel_max": round(s["rel_max"], 4),
                                         "g4_abs_max": round(s["abs_max"], 3), "g4_rel_median": round(s["rel_median"], 4)}
                continue
            e4, e5, h = s["gate4_encoder"], s["gate5_buckets_vs_15s"], s["gate4_heads"]
            row[f"gates_{units}"] = {
                "g4_encoder": e4["pass"], "g4_rel_max": round(e4["rel_max"], 4), "g4_abs_max": round(e4["abs_max"], 3),
                "g4_rel_median": round(e4["rel_median"], 4),
                # counted from the per-case records: the summary's failure list is capped at 40 examples
                "g4_failing_cases": sum(1 for r in g["encoder_cases"].values() if not r["pass"]),
                "g5": e5.get("pass"), "g5_rel_max": e5.get("gated_rel_max"),
                "heads": h["pass"], "token_agree": h["token"]["agreement_on_decisive"],
                "duration_agree": h["duration"]["agreement_on_decisive"],
                "heads_rel_max": {q: round(h[q]["rel_max"], 4) for q in ("token", "duration", "h", "c")},
                "load_s": g.get("load_s")}
        rows.append(row)
    doc = {"rows": rows, "gate3": {m: load(RES / "gates" / f"{m}-gate3.json")["summary"]
                                   for m in ("mp2", "seed0") if (RES / "gates" / f"{m}-gate3.json").exists()},
           "decoder_gates": {p.stem: load(p)["summary"]["gate4_heads"] for p in (RES / "gates").glob("*-decoder-*.json")},
           "g0": {p.stem: load(p)["summary"] for p in (RES / "gates").glob("c0-G0-*.json")},
           "probes": load(RES / "probes" / "summary.json"), "diag": load(RES / "diag" / "fp16_depth.json")}
    (RES / "wp3_summary.json").write_text(json.dumps(doc, indent=1) + "\n")
    lines = ["| model | arm | variant | mlmodelc MB | compile s | convert s | peak RSS MB | gate 2 (fixed build) | ANE cost share (plan, per function) | "
             "g4 enc ANE (rel max / abs max / failing) | g5 ANE | heads ANE | g4 enc CPU (rel max / failing) | heads CPU |",
             "|" + "---|" * 14]

    def fmt_plan(pl):
        return ", ".join(f"{fn}: {v['ane_cost']:.3f}" if "ane_cost" in v else f"{fn}: error" for fn, v in sorted(pl.items())) or "-"

    for r in rows:
        if r["variant"] == "-":
            continue
        a, c = r.get("gates_cpuAndNeuralEngine"), r.get("gates_cpuOnly")
        if a and "g0_vs_c0" in a:
            lines.append(f"| {r['model']} | {r['arm']} | {r['variant']} | {r['mlmodelc_mb']} | {r['compile_s']} | {r['convert_s']} | "
                         f"{r['peak_rss_mb']} | - | {fmt_plan(r['plan'])} | vs C0: {'pass' if a['g0_vs_c0'] else 'FAIL'} "
                         f"{a['g4_rel_max']} / {a['g4_abs_max']} (median {a['g4_rel_median']}) | - | - | - | - |")
            continue
        lines.append("| " + " | ".join(str(x) for x in (
            r["model"], r["arm"], r["variant"], r["mlmodelc_mb"], r["compile_s"], r["convert_s"], r["peak_rss_mb"],
            ("bit-exact" if r["gate2"]["bit_exact"] else "FAIL") + f" ({r['gate2']['modules']})" if "gate2" in r else "-",
            fmt_plan(r["plan"]),
            f"{'pass' if a['g4_encoder'] else 'FAIL'} {a['g4_rel_max']} / {a['g4_abs_max']} / {a['g4_failing_cases']}" if a else "-",
            ("pass" if a["g5"] else ("-" if a["g5"] is None else "FAIL")) if a else "-",
            ("pass" if a["heads"] else "FAIL") if a else "-",
            f"{'pass' if c['g4_encoder'] else 'FAIL'} {c['g4_rel_max']} / {c['g4_failing_cases']}" if c else "-",
            ("pass" if c["heads"] else "FAIL") if c else "-")) + " |")
    (RES / "wp3_summary_table.txt").write_text("\n".join(lines) + "\n")  # *.md under results/ is git-ignored
    print("\n".join(lines))


def rev7() -> str:
    """Revision 7 tables (4a, 4b per arm x variant x backend, diagnostics) -> results/wp3_rev7_table.txt."""
    v7 = RES / "gates" / "v7"
    elig = {(r["model"], r["arm"], r["variant"], r["backend"], r.get("decoder_precision") or "fp32"): r
            for r in (load(RES / "eligibility" / "summary.json") or {"records": []})["records"]}
    out = ["Gate 4a (FP32 builds, CPU_ONLY, full depth; ceilings rel <= 1e-5, abs <= 1e-4)", "",
           "| build | variant | cases | rel max | abs max | pass |", "|---|---|---|---|---|---|"]
    for p in sorted(v7.glob("*-4a.json")):
        d = load(p)
        s = d["summary"]
        if d["gate"] == "4a":
            out.append(f"| {d['label']} | {d['variant']} | {s['cases']} | {s['rel_max']:.2e} | {s['abs_max']:.2e} | {d['pass']} |")
        else:
            out.append(f"| decoder-fp32 (replay + free decoding, both deployed paths) | - | {s['clips']} clips | "
                       f"{s['logits_rel_max']:.2e} (logits) | {s['logits_abs_max']:.2e} | {d['pass']} |")
    out += ["", "Gate 4b (FP16 encoder builds; decoder/joint FP32 = deployed (revision 8), FP16 = recorded, ineligible; "
            "WER from eligibility.py; jd = Decoder + JointDecision, dj = DecoderJoint)", "",
            "| arm | variant | backend | decoder | enc rel max (<= 0.1) | rev2-5 enc failing (diag) | agree all tok / dur (jd) | "
            "identical seqs jd / dj (of 64; >= 61) | WER jd / ref (%) | gate 5 | stress | eligible |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    runs = [load(p) for p in sorted(v7.glob("mp2-*-dec-*.json"))]
    runs.sort(key=lambda d: (d.get("decoder_precision", "fp16") != "fp32", d["arm"], d["variant"], d["backend"]))
    for d in runs:
        if d["gate"] != "4b":
            continue
        s = d["summary"]
        h = s["heads_4b"]["paths"]["jd"]
        fd = s["free_decoding"]
        dec = d.get("decoder_precision", "fp16")
        e = elig.get((d["model"], d["arm"], d["variant"], d["backend"], dec), {})
        w = e.get("wer", {})
        stress_ok = "stress" not in e.get("failed_checks", [])
        out.append(f"| {d['arm']} | {d['variant']} | {d['backend']} | {dec} | {s['encoder_4b']['rel_max']:.4f} | "
                   f"{s['encoder_rev5_diagnostic']['failing_cases']} | {h['token']['agreement_all_steps']:.4f} / "
                   f"{h['duration']['agreement_all_steps']:.4f} | {fd['jd']['identical']} / {fd['dj']['identical']} | "
                   f"{w.get('jd_wer_pct', '-')} / {w.get('reference_wer_pct', '-')} | "
                   f"{'-' if s['gate5']['pass'] is None else s['gate5']['pass']} | {'pass' if stress_ok else 'FAIL'} | "
                   f"{'yes' if e.get('timing_allowed') else 'no'} |")
    out += ["", "Diagnostics: decoder-only 4b (FP16 decoder paths on the FP32 reference encoder output) and the "
            "revision-7 FP32-decoder diagnostic runs (superseded by the revision-8 4b runs above)", "",
            "| run | identical seqs jd / dj (of 64) | agree all tok / dur (jd) | heads pass |", "|---|---|---|---|"]
    for p in sorted(list(v7.glob("*-4b.json")) + list((v7 / "superseded").glob("*-decoder-fp32.json"))):
        d = load(p)
        s = d["summary"]
        h = s["heads_4b"]["paths"]["jd"]
        out.append(f"| {p.stem} | {s['free_decoding']['jd']['identical']} / {s['free_decoding']['dj']['identical']} | "
                   f"{h['token']['agreement_all_steps']:.4f} / {h['duration']['agreement_all_steps']:.4f} | "
                   f"{s['heads_4b']['pass']} |")
    text = "\n".join(out) + "\n"
    (RES / "wp3_rev7_table.txt").write_text(text)
    return text


def wp6a() -> str:
    """WP6a tables (ANE graph layout; GPU backend) -> results/wp6a_table.txt."""
    v7 = RES / "gates" / "v7"
    elig = {(r["model"], r["arm"], r["variant"], r["backend"], r.get("decoder_precision") or "fp32"): r
            for r in (load(RES / "eligibility" / "summary.json") or {"records": []})["records"]}
    stress = (load(RES / "probes" / "stress.json") or {}).get("verdict", {})

    def plan_row(arm, units, dev):
        m = load(RES / "builds" / f"mp2-{arm}-multi.json") or {}
        fs = m.get("compute_plan", {}).get(units, {}).get("functions", {})
        cells = []
        for f in ("b2", "b4", "b8", "b15"):
            x = fs.get(f, {})
            ops = x.get("ops_with_usage_by_preferred_device", {})
            cells.append(f"{ops.get(dev, 0)}/{sum(ops.values())} ({x.get('estimated_cost_share_by_preferred_device', {}).get(dev, 0):.3f})"
                         if ops else "-")
        cpu15 = fs.get("b15", {}).get("operators_by_preferred_device", {}).get("cpu", {})
        return cells, m, cpu15

    out = ["ANE graph layout (DESIGN.md D) vs plain, multifunction, CPU_AND_NE compute plan: ops on the ANE / ops with "
           "usage (estimated ANE cost share), per function", "",
           "| arm | layout | b2 | b4 | b8 | b15 | CPU ops (b15) | mlmodelc MB | convert s | compile s | build peak MB |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ("C4", "C3", "C6s8"):
        for lab, lay in ((a, "plain"), (f"{a}-ane", "ane")):
            cells, m, cpu15 = plan_row(lab, "cpuAndNeuralEngine", "ane")
            if not m:
                continue
            out.append(f"| {a} | {lay} | " + " | ".join(cells) + f" | {sum(cpu15.values())}: " +
                       ", ".join(f"{k.split('.')[-1]} {v}" for k, v in sorted(cpu15.items())) +
                       f" | {m.get('compile', {}).get('mlmodelc_bytes', 0) / 2 ** 20:.1f} | {m['timing_s']['convert']} | "
                       f"{m.get('compile', {}).get('compile_s')} | {max(m.get('peak_rss_mb_after_save') or 0, m.get('peak_rss_mb_after_compile') or 0)} |")
    out += ["", "Gates and eligibility (FP32 decoder/joint, both deployed paths; load = first / cached, s, per function "
            "b2..b15; peak = gate job RSS MB)", "",
            "| arm | backend | 4a (FP32 build) rel / abs max | 4b enc rel max | identical seqs jd / dj | agree all tok / dur | "
            "gate 5 | stress | eligible | load first / cached (s) | peak MB |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab, units in (("C4-ane", "cpuAndNeuralEngine"), ("C3-ane", "cpuAndNeuralEngine"), ("C6s8-ane", "cpuAndNeuralEngine"),
                       ("C4", "cpuAndGPU"), ("C7", "cpuAndGPU"), ("C3", "cpuAndGPU"), ("C6s8", "cpuAndGPU"), ("C1", "cpuAndGPU")):
        backend = {"cpuAndNeuralEngine": "ane", "cpuAndGPU": "gpu"}[units]
        d = load(v7 / f"mp2-{lab}-multi-{units}-dec-fp32.json")
        rec = load(RES / "eligibility" / f"mp2-{lab}-multi-{backend}.json") or {}
        src = rec.get("checks", {}).get("gate4a", {}).get("source")  # the arm's own or its topology's FP32 build
        g4a = load(IOS_DIR / src) if src else None
        a4 = f"{g4a['summary']['rel_max']:.1e} / {g4a['summary']['abs_max']:.1e} ({'pass' if g4a['pass'] else 'FAIL'})" if g4a else "-"
        sv = stress.get(lab, {}).get(backend, {})
        if d is None:
            out.append(f"| {lab} | {backend} | {a4} | not run | - | - | - | {'pass' if sv.get('pass') else 'FAIL'} | no | - | - |")
            continue
        s = d["summary"]
        h = s["heads_4b"]["paths"]["jd"]
        e = elig.get(("mp2", lab, "multi", backend, "fp32"), {})
        loads = d.get("load_s", {})
        ld = " ".join(f"{loads[f]['first_s']:.0f}/{loads[f]['cached_s']:.2f}" for f in ("b2", "b4", "b8", "b15") if f in loads)
        out.append(f"| {lab} | {backend} | {a4} | {s['encoder_4b']['rel_max']:.4f} | {s['free_decoding']['jd']['identical']} / "
                   f"{s['free_decoding']['dj']['identical']} | {h['token']['agreement_all_steps']:.4f} / "
                   f"{h['duration']['agreement_all_steps']:.4f} | {s['gate5']['pass']} | {'pass' if sv.get('pass') else 'FAIL'} | "
                   f"{'yes' if e.get('timing_allowed') else 'no'} | {ld} | {d.get('peak_rss_mb')} |")
    out += ["", "CPU_AND_GPU compute plans (plain multifunction): ops on the GPU / ops with usage (estimated GPU cost share)", "",
            "| arm | b2 | b4 | b8 | b15 |", "|---|---|---|---|---|"]
    for a in ("C4", "C7", "C3", "C6s8", "C1"):
        cells, m, _ = plan_row(a, "cpuAndGPU", "gpu")
        out.append(f"| {a} | " + " | ".join(cells) + " |")
    text = "\n".join(out) + "\n"
    (RES / "wp6a_table.txt").write_text(text)
    return text


if __name__ == "__main__":
    main()
    print(rev7())
    print(wp6a())
