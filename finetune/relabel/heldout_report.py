"""Held-out-source panel, step 1 (heldout-panel plan "Reported"): the consolidated table, descriptive, no intervals.

    ./python heldout_report.py      # or: ./python heldout_panel.py report

Reads one record per model and set (location: the existing record, or heldout_panel's for a gap) and writes
$RELABEL_ROOT/heldout-panel/report.json and tables.md (aliases only).

Models (MODELS): B0 (the pinned FP32 Parakeet-TDT-0.6B-v2), Kea-B (MB1's export, as score_domain scored it), the
stage 4 control and Kea-B' (b100-control, b100-main, x1), B0' (stage 3's b100, FP32), T* (granite-speech-5.0-2b, the
diagnostic's teacher path) and T*'s encoder (probe 1's BPE head). Reference rows: H4 (the unweighted mean of its four
dev sets), YODAS, ami-sdm-dev and cv27-en-dev from existing records, "-" where none exist.

A cell counts ("ok") only if its record is of the model's expected identity (expectations: B0's pinned .nemo, MB1's
export, each arm's scoring view, B0''s model.json, T*'s revision for both T* rows), on the set's current manifest
(SHA-256), unlimited, with every manifest utterance; otherwise it is "not_comparable" and shown as "-", as is a missing
record. Each counted cell names its record, the record file's SHA-256, the model identity, the manifest SHA-256 and,
for a record this panel wrote, the code commit. A cell whose record came through another scoring path than its row's
(MAIN_KIND) is counted and flagged ("path_differs"); only reference cells can be such.

Reported (tabulate):
- main: WER per set, the unweighted means of each panel's set WERs (heldout_panel.PANELS: T5, T5 without tech-dev-2, M3,
  X2), "-" if any of its sets is "-"; the reference sets and H4 likewise.
- differences: A - B per set and per panel mean (DIFFERENCES), "-" if either side is "-".
- error_profile: substitutions, deletions and insertions per 100 reference words, per set (ERROR_PROFILE_MODELS).
- term_recall: critical-term recall on eval-stem, recomputed from each record's stored normalized texts as score_domain
  computes it (score_domain.term_recall, terms.matcher("technical")); the unweighted mean of the three parts and the
  pooled counts. The value stored in a record (computed when it was scored) is kept beside it.
- sizes: hours and utterances per set (its manifest) and reference words (the counted records', which must agree;
  otherwise the set is flagged "size_disagreement").
No intervals: the differences are point estimates with each set's size.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import namedtuple
from pathlib import Path

import heldout_panel as hp
import relabel_paths

MODELS = ("B0", "MB1", "b100-control", "b100-main", "B0p", "tstar", "tstar-encoder")
LABELS = {"B0": "B0", "MB1": "Kea-B", "b100-control": "control", "b100-main": "Kea-B′", "B0p": "B0′", "tstar": "T*",
          "tstar-encoder": "T* encoder"}
MAIN_KIND = {"B0": "FP32 (score_domain)", "MB1": "dequantized export, FP32 (score_domain)", "b100-control": "x1",
             "b100-main": "x1", "B0p": "FP32 (stage 3 scorer)", "tstar": "teacher (diagnostic path)",
             "tstar-encoder": "encoder probe 1, BPE head"}
NEW_MODELS = ("b100-control", "b100-main", "B0p", "tstar", "tstar-encoder")   # new to this panel's tables
DIFFERENCES = (("b100-main", "b100-control"), ("B0p", "B0"), ("tstar", "B0"), ("tstar-encoder", "B0"),
               ("tstar-encoder", "tstar"))
ERROR_PROFILE_MODELS = ("tstar", "tstar-encoder", "B0")
STEM = ("stem-chem", "stem-math", "stem-cs")
H4 = tuple(relabel_paths.H4)
REFERENCE_SETS = H4 + (relabel_paths.YODAS, "ami-sdm-dev", "cv27-en-dev")
ALL_SETS = hp.SETS + REFERENCE_SETS
AMI_SDM_DEV = "corpora/ami-sdm/dev/ami-sdm-dev.jsonl"      # train_keab.AMI_SDM_DEV
CV_DEV = "corpora/cv27-en/dev/cv27-en-dev.jsonl"            # train_b0prime.CV_DEV_SAMPLE

Location = namedtuple("Location", "path kind")


# --- where each record is ------------------------------------------------------------------------------------------

def location(model: str, name: str) -> Location | None:
    """The record of `model` on set `name` (heldout-panel plan "Models and records"), None if none exists or is
    planned."""
    import devsets

    r = relabel_paths.ARTIFACTS
    keab, b100 = r / "keab-prime" / "scores", r / "b0prime" / "runs" / hp.B0P_RUN
    kind = MAIN_KIND[model]
    if name in hp.SETS:
        group = devsets.set_of(name)
        if model in hp.SCORED and name in hp.SCORED[model]:
            return Location(hp.output_dir(model, None) / f"{name}.json", kind)
        if model in ("B0", "MB1"):
            sub = r / "devsets" / "scores"
            base = sub / model if group in devsets.DEFAULT_SETS else sub / group / model
            return Location(base / f"{name}.json", kind)
        if model in hp.X1_ARMS:
            return Location(keab / model / "x1" / f"{name}.json", kind)
        if model == "B0p":
            return Location(b100 / ("eval-stem" if group == "eval-stem" else "domain") / f"{name}.json", kind)
        raise ValueError(f"no record location for {model} on {name}")
    if name not in REFERENCE_SETS:
        raise ValueError(f"not a panel or reference set: {name!r}")
    dev = name in relabel_paths.SETS
    if model in ("B0", "MB1"):
        if dev:
            return Location(relabel_paths.STUDENT_RECORDS[model] / f"{name}.json", kind)
        if model == "B0":
            return Location(keab / "B0" / "fp32" / f"{name}.json" if name == "ami-sdm-dev" else
                            r / "b0prime" / "scores" / "B0" / "cv-dev" / f"{name}.json", kind)
        return Location(keab / "MB1" / "x1" / f"{name}.json", "x1")
    if model in hp.X1_ARMS:
        return Location(keab / model / "x1" / f"{name}.json", kind)
    if model == "B0p":
        if dev:
            return Location(b100 / "eval-dev" / f"{name}.json", kind)
        return Location(keab / "B0p" / "fp32" / f"{name}.json" if name == "ami-sdm-dev" else
                        b100 / "cv-dev" / f"{name}.json", kind)
    if not dev:
        return None
    if model == "tstar":
        return Location(relabel_paths.DIAGNOSTIC / hp.TEACHER / f"{name}.json", kind)
    import encoder_probe_ctc as ep
    return Location(ep.OUT / ep.BPE / f"{name}.json", kind)


def expectations() -> tuple[dict, dict]:
    """({model: expected identity}, {set: {manifest_sha256, utterances, hours}}) from the artifacts as they are now."""
    import evaluate
    import paths
    import score_domain
    import teachers

    r = relabel_paths.ARTIFACTS
    arms = {m: json.loads((r / "keab-prime" / "runs" / m / "export" / "view" / "manifest.json").read_text())["sha256"]
            for m in hp.X1_ARMS}
    identity = {"B0": evaluate.base_lock()["files"][paths.MODEL_FILE.name],
                "MB1": score_domain.expected_sha256("MB1"), **arms,
                "B0p": json.loads((r / "b0prime" / "runs" / hp.B0P_RUN / "model.json").read_text())["sha256"],
                "tstar": teachers.REGISTRY[hp.TEACHER].revision}
    identity["tstar-encoder"] = identity["tstar"]
    manifests = {n: m for n, (m, _) in hp.domain_manifests(hp.SETS).items()}
    manifests.update({n: evaluate.resolve_manifest(n)[1] for n in relabel_paths.SETS})
    manifests.update({"ami-sdm-dev": r / AMI_SDM_DEV, "cv27-en-dev": r / CV_DEV})
    sets = {}
    for name in ALL_SETS:
        rows = evaluate.read_manifest(manifests[name])
        sets[name] = {"manifest_sha256": evaluate.sha256_file(manifests[name]), "utterances": len(rows),
                      "hours": sum(x["duration"] for x in rows) / 3600}
    return identity, sets


# --- one cell -------------------------------------------------------------------------------------------------------

def record_identity(record: dict) -> str | None:
    """The model identity a record states, in each scorer's layout (x1 view, stage 4 fp32, stage 3 and score_domain
    weights, evaluate.py source, score_teacher and encoder_probe_ctc revision)."""
    config = record.get("config") or {}
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    for value in (model.get("view_sha256"), model.get("sha256"), config.get("weights_sha256"),
                  (record.get("source") or {}).get("sha256"), config.get("revision")):
        if value:
            return value
    return None


def display(path: Path) -> str:
    """A path relative to the root it lives under, by the root's variable name (no machine paths in reports)."""
    roots = (("$RELABEL_ROOT", relabel_paths.ARTIFACTS), ("$PARAKEET_TERNARY_ROOT", relabel_paths.PARAKEET_TERNARY),
             ("$PARAKEET_BINARY_ROOT", relabel_paths.PARAKEET_BINARY))
    for key, root in roots:
        if Path(path).is_relative_to(root):
            return f"{key}/{Path(path).relative_to(root)}"
    return str(path)


def read_cell(model: str, name: str, loc: Location, expect: dict) -> dict:
    """The cell of `model` on `name` from its record at `loc` (module docstring: counted only if comparable)."""
    path = Path(loc.path)
    if not path.exists():
        return {"status": "missing", "record": display(path), "path": loc.kind}
    raw = path.read_bytes()
    record = json.loads(raw)
    config = record.get("config") or {}
    identity = record_identity(record)
    manifest = record.get("manifest_sha256") or config.get("manifest_sha256")
    limit = record.get("limit", config.get("limit"))
    problems = []
    if identity != expect["identity"]:
        problems.append(f"identity {identity} is not the model's {expect['identity']}")
    if manifest != expect["manifest_sha256"]:
        problems.append(f"manifest {manifest} is not the set's {expect['manifest_sha256']}")
    if limit is not None:
        problems.append(f"limited to {limit} utterances")
    if record.get("utterances") != expect["utterances"]:
        problems.append(f"{record.get('utterances')} utterances, the manifest has {expect['utterances']}")
    if model == "tstar-encoder" and record.get("head", "bpe") != "bpe":
        problems.append(f"head {record.get('head')}")
    wer = record.get("wer")
    if not isinstance(wer, dict):
        problems.append("no edit counts (WER only)" if wer is not None else "no WER")
    panel = record.get("panel") or {}
    cell = {"status": "not_comparable" if problems else "ok", "problems": problems, "record": display(path),
            "record_sha256": hashlib.sha256(raw).hexdigest(), "identity": identity,
            "manifest_sha256": manifest, "path": loc.kind, "code": panel.get("code"),
            "origin": "new" if path.is_relative_to(hp.out_root()) else "existing",
            "utterances": record.get("utterances")}
    if isinstance(wer, dict):
        cell.update(wer=100 * wer["wer"], substitutions=wer["substitutions"], deletions=wer["deletions"],
                    insertions=wer["insertions"], ref_words=wer["ref_words"], scored_utterances=wer["utterances"])
    if name in STEM and record.get("records"):
        import score_domain
        import terms

        cell["term_recall"] = score_domain.term_recall(record["records"], terms.matcher("technical"))
        cell["term_recall_stored"] = record.get("term_recall")
    return cell


# --- the tables -----------------------------------------------------------------------------------------------------

def _value(cells: dict, model: str, name: str) -> float | None:
    c = cells.get(model, {}).get(name)
    return c["wer"] if c and c.get("status") == "ok" else None


def _mean(values: list) -> float | None:
    return None if not values or any(v is None for v in values) else sum(values) / len(values)


def _diff(a, b):
    return None if a is None or b is None else a - b


def tabulate(cells: dict) -> dict:
    """main, differences, error_profile, term_recall and flags from {model: {set: cell}} (module docstring)."""
    main = {}
    for model in MODELS:
        sets = {n: _value(cells, model, n) for n in hp.SETS}
        ref = {n: _value(cells, model, n) for n in REFERENCE_SETS}
        main[model] = {"sets": sets, "panels": {p: _mean([sets[n] for n in names]) for p, names in hp.PANELS.items()},
                       "reference": {"H4": _mean([ref[n] for n in H4]), **ref}}
    differences = {}
    for a, b in DIFFERENCES:
        differences[f"{a} - {b}"] = {
            "a": a, "b": b, "label": f"{LABELS[a]} − {LABELS[b]}",
            "sets": {n: _diff(main[a]["sets"][n], main[b]["sets"][n]) for n in hp.SETS},
            "panels": {p: _diff(main[a]["panels"][p], main[b]["panels"][p]) for p in hp.PANELS}}
    profile = {}
    for model in ERROR_PROFILE_MODELS:
        profile[model] = {}
        for name in hp.SETS:
            c = cells.get(model, {}).get(name)
            profile[model][name] = None if not c or c.get("status") != "ok" else {
                "substitutions": 100 * c["substitutions"] / c["ref_words"],
                "deletions": 100 * c["deletions"] / c["ref_words"],
                "insertions": 100 * c["insertions"] / c["ref_words"], "wer": c["wer"]}
    recall = {}
    for model in MODELS:
        got = {}
        for name in STEM:
            c = cells.get(model, {}).get(name)
            got[name] = c["term_recall"] if c and c.get("status") == "ok" and c.get("term_recall") else None
        complete = all(v is not None for v in got.values())
        occ = sum(v["occurrences"] for v in got.values()) if complete else None
        rep = sum(v["reproduced"] for v in got.values()) if complete else None
        recall[model] = {"parts": got, "mean": _mean([v["recall"] if v else None for v in got.values()]),
                         "pooled": None if not complete else {"occurrences": occ, "reproduced": rep,
                                                              "recall": rep / occ if occ else None}}
    flags = []
    for name in ALL_SETS:
        counted = {m: cells[m][name] for m in MODELS if cells.get(m, {}).get(name, {}).get("status") == "ok"}
        sizes = {(c.get("utterances"), c.get("ref_words")) for c in counted.values()}
        if len(sizes) > 1:
            flags.append({"kind": "size_disagreement", "set": name,
                          "sizes": {m: [c.get("utterances"), c.get("ref_words")] for m, c in counted.items()}})
    for model in MODELS:
        for name, c in cells.get(model, {}).items():
            if c.get("status") == "not_comparable":
                flags.append({"kind": "not_comparable", "model": model, "set": name, "problems": c.get("problems"),
                              "record": c.get("record")})
            elif c.get("status") == "ok" and c.get("path") not in (None, MAIN_KIND[model]):
                flags.append({"kind": "path_differs", "model": model, "set": name, "path": c["path"],
                              "row_path": MAIN_KIND[model], "record": c["record"]})
            stored, now = c.get("term_recall_stored"), c.get("term_recall")
            if c.get("status") == "ok" and stored and now and stored != now:
                flags.append({"kind": "term_recall_recomputed", "model": model, "set": name, "stored": stored,
                              "recomputed": now})
    return {"main": main, "differences": differences, "error_profile": profile, "term_recall": recall,
            "flags": flags}


def build() -> dict:
    """Every cell read and tabulated (nothing written)."""
    identity, sets = expectations()
    cells = {}
    for model in MODELS:
        cells[model] = {}
        for name in ALL_SETS:
            loc = location(model, name)
            if loc is None:
                cells[model][name] = {"status": "none"}
                continue
            expect = {"identity": identity[model], **sets[name]}
            cells[model][name] = read_cell(model, name, loc, expect)
    tables = tabulate(cells)
    gaps = [f"{m} {n}" for m, names in hp.SCORED.items() for n in names if cells[m][n]["status"] != "ok"]
    for name, entry in sets.items():
        words = {c["ref_words"] for m in MODELS for c in [cells[m][name]] if c.get("status") == "ok"}
        entry["ref_words"] = words.pop() if len(words) == 1 else None
    return {"plan": hp.PLAN, "status": "complete" if not gaps else "partial", "unscored_gaps": gaps,
            "models": {m: {"label": LABELS[m], "identity": identity[m], "path": MAIN_KIND[m]} for m in MODELS},
            "panels": {p: {"label": hp.PANEL_LABELS[p], "sets": list(names)} for p, names in hp.PANELS.items()},
            "sizes": sets, "cells": cells, "tables": tables, "flags": tables["flags"],
            "note": "descriptive; no intervals: differences are point estimates with each set's size"}


# --- markdown -------------------------------------------------------------------------------------------------------

def _f(value, signed: bool = False, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
            *["| " + " | ".join(r) + " |" for r in rows]]


def markdown(report: dict) -> str:
    t = report["tables"]
    panels = list(hp.PANELS)
    plabel = [hp.PANEL_LABELS[p] for p in panels]
    out = ["## WER per set (%)", ""]
    out += _table(["Model", *hp.SETS], [[LABELS[m], *[_f(t["main"][m]["sets"][n]) for n in hp.SETS]] for m in MODELS])
    out += ["", "## Panel means (unweighted, %) and reference rows", ""]
    ref_cols = ["H4", *REFERENCE_SETS[len(H4):]]
    out += _table(["Model", *plabel, *ref_cols],
                  [[LABELS[m], *[_f(t["main"][m]["panels"][p]) for p in panels],
                    *[_f(t["main"][m]["reference"][c]) for c in ref_cols]] for m in MODELS])
    out += ["", "## Differences (WER points, A − B; negative: A has fewer errors)", ""]
    diffs = t["differences"].values()
    out += _table(["A − B", *hp.SETS], [[d["label"], *[_f(d["sets"][n], True) for n in hp.SETS]] for d in diffs])
    out += [""]
    out += _table(["A − B", *plabel], [[d["label"], *[_f(d["panels"][p], True) for p in panels]] for d in diffs])
    out += ["", "## Error profile (per 100 reference words: substitutions / deletions / insertions)", ""]
    rows = []
    for m in ERROR_PROFILE_MODELS:
        rows.append([LABELS[m], *["-" if t["error_profile"][m][n] is None else
                                  "/".join(_f(t["error_profile"][m][n][k], digits=1)
                                           for k in ("substitutions", "deletions", "insertions")) for n in hp.SETS]])
    out += _table(["Model", *hp.SETS], rows)
    out += ["", "## Technical term recall on eval-stem (%, recomputed from the stored hypotheses)", ""]
    rows = []
    for m in MODELS:
        tr = t["term_recall"][m]
        cells = []
        for n in STEM:
            part = tr["parts"][n]
            cells.append("-" if part is None or part["recall"] is None else
                         f"{100 * part['recall']:.1f} ({part['reproduced']}/{part['occurrences']})")
        pooled = tr["pooled"]
        rows.append([LABELS[m] + ("" if m in NEW_MODELS else " (reference)"), *cells,
                     "-" if tr["mean"] is None else f"{100 * tr['mean']:.1f}",
                     "-" if not pooled or pooled["recall"] is None else f"{100 * pooled['recall']:.1f}"])
    out += _table(["Model", *STEM, "mean", "pooled"], rows)
    if report.get("sizes"):
        out += ["", "## Set sizes", ""]
        out += _table(["Set", "Panel", "Hours", "Utterances", "Reference words"],
                      [[n, ", ".join(hp.PANEL_LABELS[p] for p in panels if n in hp.PANELS[p]) or "reference",
                        f"{report['sizes'][n]['hours']:.2f}", str(report["sizes"][n]["utterances"]),
                        "-" if report["sizes"][n].get("ref_words") is None else str(report["sizes"][n]["ref_words"])]
                       for n in ALL_SETS if n in report["sizes"]])
    if report.get("cells"):
        out += ["", "## Records", ""]
        rows = []
        for m in MODELS:
            for n in ALL_SETS:
                c = report["cells"][m].get(n, {"status": "none"})
                if c["status"] == "none":
                    continue
                code = c.get("code") or {}
                rows.append([LABELS[m], n, c["status"], c.get("origin", "-"), c.get("path", "-"),
                             (c.get("identity") or "-")[:12], (c.get("record_sha256") or "-")[:12],
                             (code.get("commit") or "-")[:12] + (" (dirty)" if code.get("dirty") else ""),
                             c.get("record", "-")])
        out += _table(["Model", "Set", "Status", "Origin", "Path", "Identity", "Record SHA-256", "Commit", "Record"],
                      rows)
    if report.get("flags"):
        out += ["", "## Flags", ""] + [f"- {json.dumps(f, ensure_ascii=False)}" for f in report["flags"]]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    relabel_paths.require_mount()
    report = build()
    hp.write_json(hp.out_root() / "report.json", report)
    (hp.out_root() / "tables.md").write_text(markdown(report))
    print(markdown(report))
    print(f"status {report['status']}; unscored gaps: {report['unscored_gaps'] or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
