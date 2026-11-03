"""The Kea-B′ test table (keab-prime-test plan "Report") from saved outputs. CPU only; no decoding.

    ./python keab_report_test.py   # writes $RELABEL_ROOT/plans/RESULTS-keab-prime-test.md and
                                   # $RELABEL_ROOT/keab-prime/test/test.json; a refusal exits 1 (SystemExit),
                                   # a sanity check that pauses the write-up exits 3

Arms: NVIDIA published, B0 (parakeet-ternary eval/b0-pretrained), M1 (eval/M1-test), MB1 (parakeet-binary
eval/MB1-test), Kea-B′ K-fp and Kea-B′ K-x1 (keab_test.py --test fp / x1). The mean is over paths.MEAN_SETS (seven
sets; TED-LIUM is not in the bundle); Common Voice is reported apart, with its caveat. Empty-hypothesis rates are
computed from every arm's records with MB1's definition (binary_report_test.empty_fraction), never from a scorer's
own count. No intervals are reported (as for MB1's TEST.md).

Refuses (SystemExit): scorer files or recorded scorer hashes other than the pinned ones (keab_test_paths
SCORER_HASHES and its own SHA-256); a recorded identity other than the pinned one; a mode's attempt not complete, or
per-set files not matching its set_done hashes; a launch gate other than the one the attempt started (and resumed)
with, or one that did not pass or records other bindings than the outputs; manifest hashes differing between arms or
from the current file; a set missing, duplicated or partial, or an utterance ID duplicated or missing against the
manifest; K-fp decoding, cuDNN flags, interpreter versions or normalizer metadata differing from MB1-test's (x1:
the normalizer); a reused arm other than its pinned files and sources; any structurally invalid row (finite false or a
decode_error) in a mode, a record without its "finite" field, or a K-fp set whose encoder finiteness check is missing
or did not see every batch. Every input is read once; a file both parsed and hashed is parsed and hashed from the same
bytes.

Sanity checks (pause the write-up for investigation, never a re-score): a set where Kea-B′'s WER (either mode) is more
than 3x B0's, or where more than 1% of Kea-B′'s hypotheses are empty and it has more empty hypotheses than B0.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import keab_test_paths as tp
import paths

_BINARY_DIR = str(tp.FINETUNE / "parakeet-binary")
if _BINARY_DIR not in sys.path:
    sys.path.append(_BINARY_DIR)   # after parakeet-ternary, as score_domain.binary_on_path
import binary_report_test  # noqa: E402  (empty_fraction, MB1's definition of an empty hypothesis)

MODES = {"fp": "Kea-B' K-fp", "x1": "Kea-B' K-x1"}
REUSED = ("B0", "M1", "MB1")
ARMS = REUSED + tuple(MODES.values())
LABEL = ("Kea-B' - MB1 is the net change of stage 4's whole package (labels and teacher, activation-aware training, "
         "the 8-bit decoder, the added sources, 97,137 more steps), not a label effect. The teacher-package effect "
         "(labels, encoder target and frozen decoder together) is the stage 4 gate's matched control comparison on "
         "dev.")
CV_CAVEAT = ("Common Voice is reported apart: Kea-B' has trained on Common Voice audio (held-out clips, speakers and "
             "sentences removed), and MB1 and M1 had not, so its Common Voice difference is not like for like.")
MODE_NOTE = {"x1": "deployed: the i8x1 kernels at batch 1, the stage 4 gate's mode (the headline)",
             "fp": "MB1's test procedure: dequantized weights, FP32 activations, evaluate_set at batch 64 (the "
                   "procedure-matched row; not a deployed configuration)"}


def refuse(message: str):
    raise SystemExit(f"keab_report_test: refused: {message}")


def sha256(path: Path) -> str | None:
    try:
        with open(path, "rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError:
        return None


def snapshot(path: Path) -> tuple[bytes, str]:
    """(the file's bytes, their SHA-256), from one read."""
    try:
        data = Path(path).read_bytes()
    except OSError as error:
        refuse(f"cannot read {path}: {type(error).__name__} (missing?)")
    return data, hashlib.sha256(data).hexdigest()


def snapshot_json(path: Path):
    data, digest = snapshot(path)
    try:
        return json.loads(data), digest
    except ValueError:
        refuse(f"cannot read {path}: not JSON")


def read_manifests() -> dict[str, tuple[str, list[str]]]:
    """{test set: (manifest SHA-256, utterance IDs)}, each manifest read once."""
    out = {}
    for name in paths.TEST_SETS:
        data, digest = snapshot(tp.manifest_path(name))
        out[name] = (digest, [json.loads(line)["id"] for line in data.decode().splitlines() if line.strip()])
    return out


def check_set(label: str, name: str, doc: dict, manifests: dict) -> None:
    """A per-set output: this set, not partial, every manifest utterance exactly once."""
    if doc.get("set") != name:
        refuse(f"{label}: {name}.json holds set {doc.get('set')!r} (duplicated or misplaced)")
    if doc.get("limit") is not None:
        refuse(f"{label} {name}: a partial (--limit {doc.get('limit')}) evaluation")
    ids = [r.get("id") for r in doc.get("records") or []]
    if len(ids) != len(set(ids)):
        refuse(f"{label} {name}: duplicated utterance IDs ({len(ids) - len(set(ids))})")
    expected = set(manifests[name][1])
    if set(ids) != expected or len(ids) != len(manifests[name][1]):
        refuse(f"{label} {name}: utterance IDs differ from the manifest (missing {len(expected - set(ids))}, "
               f"extra {len(set(ids) - expected)})")
    if doc.get("utterances") != len(ids):
        refuse(f"{label} {name}: utterances {doc.get('utterances')} != {len(ids)} records (partial)")
    if doc.get("manifest_sha256") != manifests[name][0]:
        refuse(f"{label} {name}: scored manifest {doc.get('manifest_sha256')}, current {manifests[name][0]}")


def load_reused(arm: str, manifests: dict) -> tuple[dict, dict[str, dict]]:
    directory = tp.REUSED[arm]
    pins = tp.REUSED_FILES[arm]
    summary, digest = snapshot_json(directory / "summary.json")
    if digest != pins["summary.json"]:
        refuse(f"{arm}: summary.json is not the pinned file")
    if sorted((summary.get("sets") or {})) != sorted(paths.TEST_SETS):
        refuse(f"{arm}: summary.json lists sets {sorted(summary.get('sets') or {})}")
    docs = {}
    for name in paths.TEST_SETS:
        doc, digest = snapshot_json(directory / f"{name}.json")
        if digest != pins[f"{name}.json"]:
            refuse(f"{arm}: {name}.json is not the pinned file")
        check_set(arm, name, doc, manifests)
        docs[name] = doc
    sources = [summary.get("source") or {}] + [d.get("source") or {} for d in docs.values()]
    if arm == "B0":
        ok = all(s.get("kind") == "pretrained" and s.get("sha256") == tp.BASE_MODEL["nemo_sha256"]
                 and s.get("base_model") == tp.BASE_MODEL for s in sources)
    else:
        want = tp.MB1_EXPORT_SHA256 if arm == "MB1" else tp.M1_EXPORT_SHA256
        ok = {s.get("sha256") for s in sources} == {want}
    if not ok:
        refuse(f"{arm}: the outputs record a source other than its pinned model")
    return summary, docs


def complete_attempt(mode: str) -> tuple[dict[str, str], dict, list[dict]]:
    """({set: set_done SHA-256}, the started entry, the resumed entries) of the mode's one complete attempt."""
    path = tp.attempts_path(mode)
    if not path.exists():
        refuse(f"no attempt record {path}; mode {mode} is not complete")
    data, _ = snapshot(path)
    entries = []
    for line in data.decode().splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except ValueError:
                refuse(f"{path} holds a line that is not JSON")
    states = [e.get("state") for e in entries]
    if states.count("complete") != 1 or states[-1] != "complete":
        refuse(f"mode {mode}: the attempt record is not complete (exactly one final complete record)")
    if states.count("started") != 1 or states[0] != "started" or any(e.get("attempt") != 1 for e in entries):
        refuse(f"mode {mode}: the attempt record does not start with its one started record")
    done: dict[str, str] = {}
    for e in entries:
        if e.get("state") == "set_done":
            if e.get("set") in done:
                refuse(f"mode {mode}: set_done for {e.get('set')} twice")
            done[e.get("set")] = e.get("sha256")
    if set(done) != set(paths.TEST_SETS):
        refuse(f"mode {mode}: set_done records for {sorted(done)}, not all eight sets")
    return done, entries[0], [e for e in entries if e.get("state") == "resumed"]


def structural(records: list[dict]) -> list[str]:
    return [r.get("id") for r in records if r.get("finite") is False or r.get("decode_error")]


def load_mode(mode: str, manifests: dict, mb1_docs: dict[str, dict], mb1_summary: dict) -> tuple[dict, dict, dict]:
    """(summary, {set: per-set output}, the launch gate) of a complete mode, every binding checked."""
    label = MODES[mode]
    done, started, resumed = complete_attempt(mode)
    out = tp.test_dir(mode)
    docs = {}
    for name in paths.TEST_SETS:
        doc, digest = snapshot_json(out / f"{name}.json")
        if digest != done[name]:
            refuse(f"{label} {name}.json SHA-256 {digest} is not its set_done hash {done[name]} (replaced after "
                   "scoring)")
        check_set(label, name, doc, manifests)
        docs[name] = doc
    identity, own = tp.identity(), sha256(Path(tp.__file__))
    first = docs[next(iter(paths.TEST_SETS))]
    for name, doc in docs.items():
        if doc.get("identity") != identity:
            refuse(f"{label} {name}: recorded identity is not the pinned one")
        scorer = doc.get("scorer") or {}
        if scorer.get("hashes") != tp.SCORER_HASHES or scorer.get("keab_test_paths_sha256") != own:
            refuse(f"{label} {name}: recorded scorer hashes differ from the pinned ones")
        source = doc.get("source") or {}
        if (source.get("sha256") if mode == "fp" else source.get("view_sha256")) != tp.VIEW_SHA256:
            refuse(f"{label} {name}: the source is not the pinned scoring view")
        if doc.get("mode") != mode or doc.get("phase") != "test" \
                or (doc.get("keab_test") or {}).get("limit") is not None:
            refuse(f"{label} {name}: not a production {mode} test output")
        if doc.get("settings") != first.get("settings"):
            refuse(f"{label} {name}: numerical settings differ from {next(iter(paths.TEST_SETS))}'s")
        if doc.get("normalizer") != mb1_docs[name].get("normalizer"):
            refuse(f"{label} {name}: normalizer metadata differ from MB1-test's")
        records = doc.get("records") or []
        if any(not isinstance(r.get("finite"), bool) for r in records):
            refuse(f"{label} {name}: records without their finite field (the encoder finiteness check did not run)")
        if mode == "fp":
            check = doc.get("encoder_finiteness") or {}
            if not check or check.get("calls") != check.get("batches") \
                    or any(e.get("id") is None for e in check.get("not_finite") or []):
                refuse(f"{label} {name}: the encoder finiteness check is missing or did not see every batch "
                       f"({check.get('calls')} calls, {check.get('batches')} batches)")
        bad = structural(records)
        if bad or doc.get("structurally_valid") is not True:
            refuse(f"{label} {name}: {len(bad)} structurally invalid rows (finite false or a decode_error), first "
                   f"{bad[:3]}: no numerical result for mode {mode}; investigate the rows")
        if mode == "fp":
            if doc.get("decoding") != mb1_docs[name].get("decoding"):
                refuse(f"{label} {name}: decoding metadata differ from MB1-test's")
            modes = doc.get("decoder_graphs_mode") or {}
            if (modes.get("before"), modes.get("after")) != (tp.DECODER_GRAPHS_MODE, tp.DECODER_GRAPHS_MODE):
                refuse(f"{label} {name}: decoder mode {modes} is not {tp.DECODER_GRAPHS_MODE!r} before and after")
    if mode == "fp":
        mine, theirs = first.get("settings") or {}, mb1_summary.get("settings") or {}
        mv, tv = mine.get("versions") or {}, theirs.get("versions") or {}
        differ = [k for k in tp.FP_VERSION_KEYS if mv.get(k) != tv.get(k)]
        differ += [k for k in ("cudnn_deterministic", "cudnn_benchmark") if mine.get(k) != theirs.get(k)]
        if differ:
            refuse(f"{label}: interpreter versions or cuDNN flags differ from MB1-test's: {differ}")
    summary, _ = snapshot_json(out / "summary.json")
    if summary.get("per_set_sha256") != done or summary.get("attempt") != 1 or summary.get("mode") != mode:
        refuse(f"{label}: summary.json disagrees with the attempt record")
    mean = sum(100 * docs[n]["wer"]["wer"] for n in paths.MEAN_SETS) / len(paths.MEAN_SETS)
    if abs((summary.get("mean_wer") or 0.0) - mean) > 1e-9:
        refuse(f"{label}: summary.json mean {summary.get('mean_wer')} is not its per-set files' {mean}")
    gate, gate_sha = snapshot_json(tp.dev_check_dir(mode) / tp.GATE)
    if not started.get("gate_sha256") or any(e.get("gate_sha256") != started["gate_sha256"] for e in resumed) \
            or gate_sha != started["gate_sha256"]:
        refuse(f"{label}: the dev-check gate (SHA-256 {gate_sha}) is not the gate the attempt started and resumed "
               f"with ({started.get('gate_sha256')})")
    if not (gate.get("pass") is True and gate.get("limit") is None and gate.get("test_launch_gate") is True
            and gate.get("mode") == mode and gate.get("out_dir") == str(tp.dev_check_dir(mode))):
        refuse(f"{label}: the launch gate is not a passing production {mode} dev check")
    for key in ("identity", "settings", "scorer"):
        if gate.get(key) != first.get(key):
            refuse(f"{label}: the launch gate records another {key} than the test outputs")
    for name, doc in docs.items():
        if (gate.get("manifests") or {}).get(name) != doc.get("manifest_sha256"):
            refuse(f"{label}: the launch gate records another {name} test manifest than was scored")
    return summary, docs, gate


def sizes() -> dict[str, float]:
    bundle, digest = snapshot_json(tp.BUNDLE / "manifest.json")
    if digest != tp.PINNED_FILES["export/bundle/manifest.json"]:
        refuse("the bundle manifest is not the pinned file")
    return {"Kea-B'": round(bundle["bytes"]["tensor_bytes"] / 1e6, 2)}


def dev_summary(mode: str, gate: dict) -> dict:
    if mode == "x1":
        return {"pass": gate["pass"], "mean_percent": 100 * gate["mean"],
                "recorded_mean_percent": 100 * gate["recorded_mean"],
                "mean_diff_points": gate["mean_diff_points"],
                "per_set_diff_points": {n: e["diff_points"] for n, e in gate["per_set"].items()},
                "token_identity_rate": gate["token_identity"]["rate"], "tolerances": gate["tolerances"]}
    mb1, keab = gate["mb1"], gate["keab"] or {}
    return {"pass": gate["pass"], "mb1_mean_percent": 100 * mb1["mean"],
            "mb1_recorded_mean_percent": 100 * mb1["recorded_mean"],
            "mb1_mean_diff_points": mb1["mean_diff_points"],
            "mb1_per_set_diff_points": {n: e["diff_points"] for n, e in mb1["per_set"].items()},
            "keab_fp_dev_percent": {n: 100 * w for n, w in (keab.get("wer") or {}).items()},
            "keab_pytorch_fake_quant_dev_percent": {n: 100 * w
                                                    for n, w in (keab.get("pytorch_fake_quant") or {}).items()},
            "tolerances": mb1["tolerances"]}


def main() -> int:
    if not tp.SCORER_HASHES:
        refuse("SCORER_HASHES is not pinned (keab_test_paths)")
    changed = [k for k, v in tp.SCORER_HASHES.items() if sha256(tp.scorer_path(k)) != v]
    if changed:
        refuse(f"scorer sources differ from SCORER_HASHES: {changed[:6]}")
    manifests = read_manifests()
    reused = {arm: load_reused(arm, manifests) for arm in REUSED}
    mb1_summary, mb1_docs = reused["MB1"]
    for arm in ("B0", "M1"):
        for name, doc in reused[arm][1].items():
            if doc.get("normalizer") != mb1_docs[name].get("normalizer"):
                refuse(f"{arm} {name}: normalizer metadata differ from MB1-test's")
    modes = {mode: load_mode(mode, manifests, mb1_docs, mb1_summary) for mode in MODES}
    docs = {arm: reused[arm][1] for arm in REUSED} | {MODES[m]: modes[m][1] for m in MODES}

    sets = list(paths.TEST_SETS)
    cols = list(paths.MEAN_SETS) + ["mean", "common_voice"]
    rows: dict[str, dict] = {}
    pub = {k: v for k, v in paths.PUBLISHED_WER.items() if k in paths.MEAN_SETS}
    pub["mean"] = sum(pub[k] for k in paths.MEAN_SETS) / len(paths.MEAN_SETS)
    rows["NVIDIA published"] = pub
    empties, counts = {}, {}
    for arm in ARMS:
        row = {n: 100 * docs[arm][n]["wer"]["wer"] for n in sets}
        row["mean"] = sum(row[n] for n in paths.MEAN_SETS) / len(paths.MEAN_SETS)
        rows[arm] = row
        fractions = {n: binary_report_test.empty_fraction(docs[arm][n]) for n in sets}
        empties[arm] = {n: f for n, (f, _) in fractions.items()}
        counts[arm] = {n: c for n, (_, c) in fractions.items()}

    def diff(a: str, b: str, columns) -> dict:
        return {c: {"points": rows[a][c] - rows[b][c], "relative_percent": 100 * (rows[a][c] - rows[b][c]) / rows[b][c]}
                for c in columns}

    differences = {}
    for mode in ("x1", "fp"):
        differences[f"{MODES[mode]} - MB1"] = diff(MODES[mode], "MB1", sets + ["mean"])
    for mode in ("x1", "fp"):
        differences[f"{MODES[mode]} - M1"] = diff(MODES[mode], "M1", ["mean"])
    size = sizes()
    size["MB1"] = round(mb1_summary["source"]["export_manifest"]["bytes"]["file_bytes"] / 1e6, 1)
    size["M1"] = round(reused["M1"][0]["source"]["export_manifest"]["bytes"]["file_bytes"] / 1e6, 1)
    size["B0"] = round(mb1_summary["source"]["export_manifest"]["bytes"]["original_nemo_bytes"] / 1e6, 1)
    if size != tp.SIZES_MB:
        refuse(f"sizes {size} are not the plan's {tp.SIZES_MB}")
    dev = {mode: dev_summary(mode, modes[mode][2]) for mode in MODES}

    sanity = []
    for mode in ("x1", "fp"):
        arm = MODES[mode]
        sanity += [f"{s}: {arm} {rows[arm][s]:.2f} > {tp.SANITY_B0_RATIO:g}x B0 {rows['B0'][s]:.2f}" for s in sets
                   if round(rows[arm][s], 9) > round(tp.SANITY_B0_RATIO * rows["B0"][s], 9)]
        sanity += [f"{s}: {100 * empties[arm][s]:.2f}% empty {arm} hypotheses, more than B0 ({counts[arm][s]} vs "
                   f"{counts['B0'][s]})" for s in sets
                   if empties[arm][s] > tp.SANITY_EMPTY_FRACTION and counts[arm][s] > counts["B0"][s]]
    result = {"rows": rows, "columns": cols, "mean_sets": list(paths.MEAN_SETS), "separate_sets": ["common_voice"],
              "differences": differences, "empty_hypothesis_fraction": empties, "empty_hypotheses": counts,
              "sizes_mb": size, "dev_check": dev, "modes": MODE_NOTE, "label": LABEL, "common_voice_caveat": CV_CAVEAT,
              "intervals": None, "sanity_flags": sanity, "identity": tp.identity(), "scorer_hashes": tp.SCORER_HASHES}

    def fmt_row(label: str, values: dict, spec: str = ".2f", suffix: str = "") -> str:
        cells = (f"{values[c]:{spec}}{suffix}" if c in values else "n/a" for c in cols)
        return f"| {label} | " + " | ".join(cells) + " |"

    table = ["| Arm | " + " | ".join(cols) + " |", "| --- |" + " ---: |" * len(cols)]
    table += [fmt_row(label, rows[label]) for label in rows]
    for key, d in differences.items():
        table.append(fmt_row(f"{key} (points)", {c: e["points"] for c, e in d.items()}, "+.2f"))
        table.append(fmt_row(f"{key} (relative)", {c: e["relative_percent"] for c, e in d.items()}, "+.1f", "%"))
    empty_table = ["| Arm | " + " | ".join(sets) + " |", "| --- |" + " ---: |" * len(sets)]
    empty_table += [f"| {arm} | " + " | ".join(f"{100 * empties[arm][s]:.2f}%" for s in sets) + " |" for arm in ARMS]
    x1, fp, keab_mb = dev["x1"], dev["fp"], size["Kea-B'"]
    banner = ("**SANITY CHECK FAILED; investigate before any write-up:** " + "; ".join(sanity) + "\n\n"
              if sanity else "")
    text = ("# Kea-B′ test-set WER (%), Open ASR Leaderboard sets\n\n" + banner
            + "Generated by `keab_report_test.py` (keab-prime-test plan). Mean over the seven sets NVIDIA reports "
            "except "
            "TED-LIUM (not in the public bundle). B0 = FP32 original, M1 = ternary main run, MB1 = Kea-B (binary main "
            "run), all three reused as recorded (FP32 activations, evaluate.py at batch 64). Kea-B′ = stage 4 run "
            f"b100-main, scored once per mode from its scoring view: K-x1 = {MODE_NOTE['x1']}; "
            f"K-fp = {MODE_NOTE['fp']}. "
            "Same manifests and normalizer for every arm.\n\n"
            f"**{LABEL}**\n\n{CV_CAVEAT}\n\n"
            + "\n".join(table) + "\n\n## Empty hypotheses (fraction of scored utterances, MB1's definition)\n\n"
            + "\n".join(empty_table) + "\n\n## Sizes\n\n"
            f"Kea-B′ {keab_mb:.2f} MB (8-bit bundle tensors), MB1 {size['MB1']:.1f} MB, M1 {size['M1']:.1f} MB, "
            f"B0 {size['B0']:,.0f} MB.\n\n## Dev cross-checks (launch gates)\n\n"
            f"- K-x1 against stage 4's recorded x1 dev scores: mean {x1['mean_percent']:.4f}% vs "
            f"{x1['recorded_mean_percent']:.4f}% "
            f"({x1['mean_diff_points']:+.4f} points; tolerance 0.02, per set 0.10), token-sequence identity "
            f"{100 * x1['token_identity_rate']:.2f}%: {'PASS' if x1['pass'] else 'FAIL'}.\n"
            f"- K-fp validated on MB1's recorded dev check: mean {fp['mb1_mean_percent']:.4f}% vs "
            f"{fp['mb1_recorded_mean_percent']:.4f}% ({fp['mb1_mean_diff_points']:+.4f} points): "
            f"{'PASS' if fp['pass'] else 'FAIL'}. Kea-B′ K-fp dev (PyTorch fake-quant in brackets): "
            + ", ".join(f"{n} {w:.2f} ({fp['keab_pytorch_fake_quant_dev_percent'].get(n, float('nan')):.2f})"
                        for n, w in fp["keab_fp_dev_percent"].items()) + ".\n\n"
            "No intervals are reported (as for MB1's TEST.md). Nothing is selected on these numbers.\n")
    tp.TEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    tp.TEST_JSON.write_text(json.dumps(result, indent=1) + "\n")
    tp.RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    tp.RESULTS_MD.write_text(text)
    print("\n".join(table))
    if sanity:
        print("SANITY CHECK FAILED:", "; ".join(sanity))
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
