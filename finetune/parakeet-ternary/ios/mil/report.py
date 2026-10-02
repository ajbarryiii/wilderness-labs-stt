"""WP3 summary: builds, gate 2, gates 3-5, compute plans and probes in one table (pure JSON; any machine).

  python mil/report.py      # reads ios/results/{builds,gate2,gates,probes,diag}, writes results/wp3_summary.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"


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
            e4, e5, h = s["gate4_encoder"], s["gate5_buckets_vs_15s"], s["gate4_heads"]
            row[f"gates_{units}"] = {
                "g4_encoder": e4["pass"], "g4_rel_max": round(e4["rel_max"], 4), "g4_abs_max": round(e4["abs_max"], 3),
                "g4_rel_median": round(e4["rel_median"], 4), "g4_failing_cases": len(e4["failures"]),
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
    lines = ["| model | arm | variant | mlmodelc MB | compile s | convert s | peak RSS MB | gate 2 | ANE cost share (plan, per function) | "
             "g4 enc ANE (rel max / abs max / failing) | g5 ANE | heads ANE | g4 enc CPU (rel max / failing) | heads CPU |",
             "|" + "---|" * 14]

    def fmt_plan(pl):
        return ", ".join(f"{fn}: {v['ane_cost']:.3f}" if "ane_cost" in v else f"{fn}: error" for fn, v in sorted(pl.items())) or "-"

    for r in rows:
        if r["variant"] == "-":
            continue
        a, c = r.get("gates_cpuAndNeuralEngine"), r.get("gates_cpuOnly")
        lines.append("| " + " | ".join(str(x) for x in (
            r["model"], r["arm"], r["variant"], r["mlmodelc_mb"], r["compile_s"], r["convert_s"], r["peak_rss_mb"],
            ("bit-exact" if r["gate2"]["bit_exact"] else "FAIL") + f" ({r['gate2']['modules']})" if "gate2" in r else "-",
            fmt_plan(r["plan"]),
            f"{'pass' if a['g4_encoder'] else 'FAIL'} {a['g4_rel_max']} / {a['g4_abs_max']} / {a['g4_failing_cases']}" if a else "-",
            ("pass" if a["g5"] else ("-" if a["g5"] is None else "FAIL")) if a else "-",
            ("pass" if a["heads"] else "FAIL") if a else "-",
            f"{'pass' if c['g4_encoder'] else 'FAIL'} {c['g4_rel_max']} / {c['g4_failing_cases']}" if c else "-",
            ("pass" if c["heads"] else "FAIL") if c else "-")) + " |")
    (RES / "wp3_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
