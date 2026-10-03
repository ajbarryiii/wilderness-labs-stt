"""Mechanical arm eligibility (DESIGN.md revision 7, "Arm disposition"): records per (model, arm, variant, backend).

  python mil/eligibility.py build            # NixOS (needs the parent experiment's scorer and SentencePiece)
  python mil/eligibility.py check --model mp2 --arm C4 --variant multi --backend ane    # exit 0 / 10

Timing runs must call check() (or the CLI) and refuse an arm/backend without a passing record:

  from mil.eligibility import check, Ineligible
  record = check("mp2", "C4", "multi", "ane")      # raises Ineligible (with the reasons) otherwise

C0 (the product baseline) needs no record. G0 (graph control) gets a control record: timing is allowed if it is
finite and equals C0 exactly wherever masking is moot; it can never be selected.

A record is eligible when every check passes:
- scope: not an arm excluded by DESIGN.md (C5 exploratory);
- gate 2: bit-exact effective matrices (results/gate2/<model>-<arm>.json, the arm's fixed build; the multifunction
  and enumerated builds use the same encode() constants);
- gate 4a: the FP32 build of the arm's graph topology passed at full depth on every clip and bucket
  (results/gates/v7/*-4a.json): the arm's own FP32 build if one exists for that variant, else the topology's
  representative (dense: C4-fp32; post-matmul scale: C7-fp32; stacked planes: C8-fp32); plus 4a-decoder;
- gate 4b on that backend (results/gates/v7/<model>-<arm>-<variant>-<units>.json): finite, encoder rel <= 0.1,
  decision agreement through both deployed decoder paths, free-decoding sequence identity >= 95%, gate 5;
- 4b WER (evaluated here): for both deployed paths, WER over the natural clips <= reference WER + 0.2 points;
- stress rule on that backend (results/probes/stress.json, revision 6/7 relative rule).
Every record lists the SHA-256 of each input file; check() recomputes them, so a changed or re-run input
invalidates the record until it is rebuilt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

IOS = Path(__file__).resolve().parents[1]
RES = IOS / "results"
ELIG = RES / "eligibility"
DESIGN_REVISION = 7
CODE_VERSION = "wp3-eligibility-1"
EXIT_INELIGIBLE = 10
BACKENDS = {"ane": "cpuAndNeuralEngine", "cpu": "cpuOnly"}
SCOPE_EXCLUDED = {"C5": "exploratory W8A8 (DESIGN.md: scope reduction; numerics fail)"}
TOPOLOGY_REP = {"dense": "C4", "post_scale": "C7", "planes": "C8"}
TOPOLOGY = {"C1": "dense", "C3": "dense", "C4": "dense", "C6s2": "dense", "C6s4": "dense", "C6s8": "dense",
            "C6d4": "dense", "C6d8": "dense", "C7": "post_scale", "C8": "planes", "C5": "dense+activation_quantization"}
WER_POINTS = 0.2


class Ineligible(Exception):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(IOS))


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def record_path(model: str, arm: str, variant: str, backend: str) -> Path:
    return ELIG / f"{model}-{arm}-{variant}-{backend}.json"


# --- check (any machine; the timing runner calls this) -----------------------------------------------------

def check(model: str, arm: str, variant: str, backend: str, root: Path | None = None) -> dict:
    """The passing eligibility record of an arm/backend, or Ineligible. C0 (baseline) always passes."""
    if arm == "C0":
        return {"model": model, "arm": "C0", "variant": variant, "backend": backend, "timing_allowed": True,
                "eligible": False, "role": "product baseline (no gate record; always timed)"}
    path = (Path(root) if root else ELIG) / f"{model}-{arm}-{variant}-{backend}.json"
    if not path.exists():
        raise Ineligible(f"no eligibility record {path.name}: run the revision-{DESIGN_REVISION} gates and "
                         "eligibility.py build first")
    rec = json.loads(path.read_text())
    if rec.get("design_revision") != DESIGN_REVISION:
        raise Ineligible(f"{path.name} is for design revision {rec.get('design_revision')}, not {DESIGN_REVISION}")
    stale = [f for f, digest in rec.get("inputs", {}).items() if not (IOS / f).exists() or sha(IOS / f) != digest]
    if stale:
        raise Ineligible(f"{path.name}: inputs changed since the record was built: {stale[:5]}")
    if not rec.get("timing_allowed"):
        raise Ineligible(f"{model}/{arm}/{variant}/{backend} is not eligible: " + "; ".join(rec.get("reasons", [])))
    return rec


# --- build (NixOS) ---------------------------------------------------------------------------------------

def _wer(token_lists: dict, clips: dict, sp) -> dict:
    sys.path.insert(0, str(IOS.parent))
    import evaluate

    records = [evaluate.score(clips[c]["transcript"], sp.decode_ids(toks)) for c, toks in sorted(token_lists.items())]
    return evaluate.wer_summary(records)


def _tokenizer():
    import sentencepiece as spm

    sys.path.insert(0, str(IOS))
    from mil.weights import default_path

    path = default_path("mp2") / "tokenizer" / "tokenizer.model"
    return spm.SentencePieceProcessor(model_file=str(path)), path


def build(args) -> int:
    ELIG.mkdir(parents=True, exist_ok=True)
    clips = {c["id"]: c for c in json.loads((IOS / "clips.json").read_text())["clips"]}
    sp, sp_path = _tokenizer()
    stress_path = RES / "probes" / "stress.json"
    stress = load(stress_path)
    v7 = RES / "gates" / "v7"
    dec4a_cache = {}
    rows = []
    built = set()
    for p in sorted(v7.glob("*.json")):
        d = load(p)
        if d.get("gate") != "4b":
            continue
        built.add((d["model"], d["arm"], d["variant"], d["backend"]))
    expected = set(built)
    for arm in ("C1", "C3", "C4", "C5", "C6s2", "C6s4", "C6s8", "C6d4", "C6d8", "C7", "C8"):
        for variant in ("fixed", "multi"):
            for backend in BACKENDS:
                expected.add(("mp2", arm, variant, backend))
    for model, arm, variant, backend in sorted(expected):
        units = BACKENDS[backend]
        inputs, checks, reasons = {}, {}, []

        def use(path: Path):
            if path.exists():
                inputs[rel(path)] = sha(path)
            return load(path)

        # scope
        checks["scope"] = {"pass": arm not in SCOPE_EXCLUDED, "note": SCOPE_EXCLUDED.get(arm)}
        # gate 2
        g2 = use(RES / "gate2" / f"{model}-{arm}.json")
        checks["gate2"] = {"pass": bool(g2 and g2["bit_exact"]), "modules": g2 and g2["modules_checked"]}
        # gate 4a (encoder topology + decoder)
        topo = TOPOLOGY.get(arm)
        own = v7 / f"{model}-{arm}-fp32-{variant}-cpuOnly-4a.json"
        rep = TOPOLOGY_REP.get(topo)
        repf = v7 / f"{model}-{rep}-fp32-{variant}-cpuOnly-4a.json" if rep else None
        src = own if own.exists() else repf
        g4a = use(src) if src is not None else None
        checks["gate4a"] = {"pass": bool(g4a and g4a["pass"] and g4a.get("design_revision") == DESIGN_REVISION),
                            "topology": topo, "source": rel(src) if src is not None and src.exists() else None,
                            "own_fp32_build": own.exists(),
                            "rel_max": g4a and g4a["summary"]["rel_max"], "abs_max": g4a and g4a["summary"]["abs_max"]}
        d4a = use(v7 / f"{model}-decoder-fp32-cpuOnly-4a.json")
        checks["gate4a_decoder"] = {"pass": bool(d4a and d4a["pass"])}
        # gate 4b
        g4b_path = v7 / f"{model}-{arm}-{variant}-{units}.json"
        g4b = use(g4b_path)
        if g4b is None:
            checks["gate4b"] = {"pass": False, "note": "not run"}
            checks["wer"] = {"pass": False, "note": "not run"}
            checks["gate5"] = {"pass": False, "note": "not run"}
        else:
            s = g4b["summary"]
            checks["gate4b"] = {"pass": bool(g4b["pass"] and g4b.get("design_revision") == DESIGN_REVISION),
                                "encoder_rel_max": s["encoder_4b"]["rel_max"], "encoder_failing": s["encoder_4b"]["failing_cases"],
                                "heads": {p: {h: {k: v[k] for k in ("decisive_fraction", "agreement_on_decisive",
                                                                     "agreement_all_steps", "pass")}
                                              for h, v in s["heads_4b"]["paths"][p].items()} for p in ("jd", "dj")},
                                "sequence_identity": {p: s["free_decoding"][p]["identity_fraction"] for p in ("jd", "dj")},
                                "rev5_diagnostic_encoder_pass": s["encoder_rev5_diagnostic"]["pass"]}
            failed = []
            if not s["encoder_4b"]["pass"]:
                failed.append(f"encoder rel <= 0.1 fails on {s['encoder_4b']['failing_cases']} cases")
            for path_name in ("jd", "dj"):
                for head, v in s["heads_4b"]["paths"][path_name].items():
                    if not v["pass"]:
                        failed.append(f"{path_name} {head} agreement (decisive {v['agreement_on_decisive']:.4f}, "
                                      f"all {v['agreement_all_steps']:.4f}, decisive fraction {v['decisive_fraction']:.3f})")
                fr = s["free_decoding"][path_name]
                if not fr["pass_identity"]:
                    failed.append(f"{path_name} free-decoding sequence identity {fr['identical']}/{s['free_decoding']['clips']}"
                                  f" = {fr['identity_fraction']:.3f} < 0.95")
            if not s["heads_4b"]["all_finite"]:
                failed.append("non-finite decoder outputs")
            if s["gate5"]["pass"] is False:
                failed.append("gate 5")
            checks["gate4b"]["failed_conditions"] = failed
            checks["gate5"] = {"pass": s["gate5"]["pass"] is not False, "applicable": s["gate5"]["pass"] is not None,
                               "gated_rel_max": s["gate5"].get("gated_rel_max")}
            ft = g4b["free_decoding_tokens"]
            ref_w = _wer({c: v["reference"] for c, v in ft.items()}, clips, sp)
            wers = {p: _wer({c: v[p] for c, v in ft.items()}, clips, sp) for p in ("jd", "dj")}
            checks["wer"] = {"pass": all(w["wer"] <= ref_w["wer"] + WER_POINTS / 100 for w in wers.values()),
                             "reference_wer_pct": round(100 * ref_w["wer"], 3),
                             **{f"{p}_wer_pct": round(100 * w["wer"], 3) for p, w in wers.items()},
                             "clips": len(ft), "limit_points": WER_POINTS}
        # stress
        sv = (stress or {}).get("verdict", {}).get(arm, {}).get(backend)
        if stress is not None:
            inputs[rel(stress_path)] = sha(stress_path)
        checks["stress"] = {"pass": bool(sv and sv["pass"] and stress.get("design_revision") == DESIGN_REVISION),
                            "detail": sv}
        for name, c in checks.items():
            if not c["pass"]:
                detail = ("; ".join(c["failed_conditions"]) if c.get("failed_conditions")
                          else json.dumps({k: v for k, v in c.items() if k != "pass"}, default=str)[:300])
                reasons.append(f"{name}: {detail}")
        eligible = not reasons
        rec = {"design_revision": DESIGN_REVISION, "code_version": CODE_VERSION, "model": model, "arm": arm,
               "variant": variant, "backend": backend, "compute_units": units, "eligible": eligible,
               "timing_allowed": eligible, "selection_eligible": eligible, "reasons": reasons, "checks": checks,
               "inputs": inputs, "tokenizer_sha256": sha(sp_path), "built": time.strftime("%Y-%m-%d %H:%M")}
        record_path(model, arm, variant, backend).write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n")
        rows.append(rec)
    rows += build_g0(stress)
    summary = [{k: r[k] for k in ("model", "arm", "variant", "backend", "timing_allowed", "selection_eligible")}
               | {"failed_checks": [n for n, c in r["checks"].items() if not c["pass"]],
                  **({"wer": r["checks"]["wer"]} if "wer" in r["checks"] else {})} for r in rows]
    (ELIG / "summary.json").write_text(json.dumps({"design_revision": DESIGN_REVISION, "code_version": CODE_VERSION,
                                                   "records": summary}, indent=1, allow_nan=False) + "\n")
    print(table(rows))
    return 0


def build_g0(stress) -> list[dict]:
    """Control record for G0 (graph control vs C0): timing allowed on a backend whose G0-vs-C0 result exists,
    is finite and matches C0 exactly on every clip where masking is moot (M mod 8 in {0, 7})."""
    out = []
    clips = {c["id"]: c for c in json.loads((IOS / "clips.json").read_text())["clips"]}
    for backend, units in BACKENDS.items():
        path = RES / "gates" / f"c0-G0-fixed-{units}.json"
        d = load(path)
        checks, reasons, inputs = {}, [], {}
        if d is None:
            checks["g0_vs_c0"] = {"pass": False, "note": "not run"}
        else:
            inputs[rel(path)] = sha(path)
            moot = [c for c in d["per_clip"] if clips[c]["mel_frames"] % 8 in (0, 7)]
            checks["g0_vs_c0"] = {"pass": all(d["per_clip"][c]["rel"] == 0 for c in moot) and
                                  all(v["finite"] for v in d["per_clip"].values()),
                                  "moot_clips": len(moot), "moot_exact": sum(1 for c in moot if d["per_clip"][c]["rel"] == 0),
                                  "other_rel_max": max(v["rel"] for c, v in d["per_clip"].items() if c not in moot)}
        reasons = [f"{n}: {c}" for n, c in checks.items() if not c["pass"]]
        rec = {"design_revision": DESIGN_REVISION, "code_version": CODE_VERSION, "model": "c0", "arm": "G0",
               "variant": "fixed", "backend": backend, "compute_units": units, "role": "graph control (vs C0)",
               "eligible": False, "selection_eligible": False, "timing_allowed": not reasons, "reasons": reasons,
               "checks": checks, "inputs": inputs, "built": time.strftime("%Y-%m-%d %H:%M")}
        record_path("c0", "G0", "fixed", backend).write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n")
        out.append(rec)
    return out


def table(rows: list[dict]) -> str:
    lines = ["| model | arm | variant | backend | timing allowed | failed checks | WER jd / dj / ref (%) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        w = r["checks"].get("wer", {})
        wer = (f"{w.get('jd_wer_pct')} / {w.get('dj_wer_pct')} / {w.get('reference_wer_pct')}"
               if "jd_wer_pct" in w else "-")
        lines.append(f"| {r['model']} | {r['arm']} | {r['variant']} | {r['backend']} | "
                     f"{'yes' if r['timing_allowed'] else 'no'} | "
                     f"{', '.join(n for n, c in r['checks'].items() if not c['pass']) or '-'} | {wer} |")
    text = "\n".join(lines) + "\n"
    (ELIG / "table.txt").write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    p = sub.add_parser("check")
    for k in ("model", "arm", "variant", "backend"):
        p.add_argument(f"--{k}", required=True)
    args = parser.parse_args()
    if args.cmd == "build":
        sys.exit(build(args))
    try:
        rec = check(args.model, args.arm, args.variant, args.backend)
        print(json.dumps({"timing_allowed": True, "record": f"{args.model}-{args.arm}-{args.variant}-{args.backend}"}))
    except Ineligible as e:
        print(f"INELIGIBLE: {e}", file=sys.stderr)
        sys.exit(EXIT_INELIGIBLE)


if __name__ == "__main__":
    main()
