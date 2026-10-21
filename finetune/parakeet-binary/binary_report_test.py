"""Build the B0 / M1 / MB1 test table (plans/test-scoring.md "Report") from saved outputs. No GPU, no decoding.

    python binary_report_test.py   # writes results/TEST.md and results/test.json; exit 1 on a refusal,
                                   # 3 if a sanity check pauses the write-up

Arms: B0 = parakeet-ternary eval/b0-pretrained, M1 = parakeet-ternary eval/M1-test (both reused),
MB1 = eval/MB1-test (binary_test.py --test). Refuses (SystemExit) if: an MB1 set or summary records a
source SHA-256 other than the pinned weights; M1-test's summary or any per-set output records a source
other than M1's pinned export; the MB1 per-set files are not exactly the set_done files of the one
complete attempt, or MB1's summary.json disagrees with them; any manifest SHA-256 differs between arms
or from the current file; decoding or normalizer metadata differ from M1-test's; B0 is not the pinned
pretrained model; any output is partial, a set is missing or duplicated, or an utterance ID is
duplicated or missing relative to the manifest; recorded scorer hashes differ from the pinned ones or
between the dev check and the test; the dev-check gate did not pass; the attempt record is not complete.
Code review round 1: the dev-check gate.json must be the one the complete attempt launched with (its
SHA-256 = the attempt's started.gate_sha256) and record the outputs' identity, numerical settings,
test manifests and decoder mode; MB1's summary.json must equal the full projection of its per-set files.
Code review round 3: every input is read once (read_bytes); a file that is both parsed and hashed (the
gate, the attempt record, the per-set outputs, the test manifests) is parsed and hashed from those same
bytes, and the checks receive that content and digest; no path is opened twice.

The mean is over paths.MEAN_SETS (seven sets; TED-LIUM is not in the bundle); Common Voice is reported
apart. MB1 - M1 is a net change, not a causal binary cost (plans/main.md "Reported").
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import binary_test_paths as tp
import paths

ARMS = ("B0", "M1", "MB1")
LABEL = ("MB1 - M1 is the net change from converting M1 to binary and training 250,000 more steps on a new "
         "data order, not a causal binary cost (plans/main.md \"Reported\").")


def refuse(message: str):
    raise SystemExit(f"binary_report_test: refused: {message}")


def sha256(path: Path) -> str | None:
    try:
        with open(path, "rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError:
        return None


def read_bytes(path: Path) -> bytes:
    """The one place the report reads an input file."""
    return Path(path).read_bytes()


def snapshot(path: Path) -> tuple[bytes, str]:
    """(the file's bytes, their SHA-256), from one read."""
    try:
        data = read_bytes(path)
    except OSError as error:
        refuse(f"cannot read {path}: {type(error).__name__}")
    return data, hashlib.sha256(data).hexdigest()


def snapshot_json(path: Path) -> tuple[object, str]:
    """(the parsed JSON, the SHA-256 of exactly the bytes parsed)."""
    data, digest = snapshot(path)
    try:
        return json.loads(data), digest
    except ValueError:
        refuse(f"cannot read {path}: not JSON")


def arm_dirs() -> dict[str, Path]:
    return {"B0": tp.B0_TEST, "M1": tp.M1_TEST, "MB1": tp.TEST_OUT}


def manifest_path(name: str) -> Path:
    return paths.MANIFESTS / f"test_{name}.jsonl"  # evaluate.resolve_manifest for a test set


def read_manifests() -> dict[str, tuple[str, list[str]]]:
    """{test set: (manifest SHA-256, utterance IDs)}, each manifest read once."""
    out = {}
    for name in paths.TEST_SETS:
        data, digest = snapshot(manifest_path(name))
        out[name] = (digest, [json.loads(line)["id"] for line in data.decode().splitlines() if line.strip()])
    return out


def load_arm(arm: str, directory: Path, manifests: dict) -> tuple[dict, dict[str, dict], dict[str, str]]:
    """(summary.json, {set: per-set output}, {set: SHA-256 of the bytes parsed}); refuses a missing, extra,
    duplicated or partial set and utterance IDs that are duplicated or missing relative to the manifest."""
    if not (directory / "summary.json").exists():
        refuse(f"{arm}: missing {directory / 'summary.json'}")
    summary, _ = snapshot_json(directory / "summary.json")
    listed = list((summary.get("sets") or {}))
    if sorted(listed) != sorted(paths.TEST_SETS):
        refuse(f"{arm}: summary.json lists sets {sorted(listed)}; missing {sorted(set(paths.TEST_SETS) - set(listed))}, "
               f"extra {sorted(set(listed) - set(paths.TEST_SETS))}")
    docs, digests = {}, {}
    for name in paths.TEST_SETS:
        path = directory / f"{name}.json"
        if not path.exists():
            refuse(f"{arm}: missing {path}")
        doc, digests[name] = snapshot_json(path)
        if doc.get("set") != name:
            refuse(f"{arm}: {path} holds set {doc.get('set')!r} (duplicated or misplaced)")
        if doc.get("limit") is not None or summary["sets"][name].get("limit") is not None:
            refuse(f"{arm}: {name} is a partial (--limit) evaluation")
        ids = [r.get("id") for r in doc.get("records") or []]
        if len(ids) != len(set(ids)):
            refuse(f"{arm} {name}: duplicated utterance IDs ({len(ids) - len(set(ids))})")
        expected = set(manifests[name][1])
        if set(ids) != expected:
            refuse(f"{arm} {name}: utterance IDs differ from the manifest (missing {len(expected - set(ids))}, "
                   f"extra {len(set(ids) - expected)})")
        if doc.get("utterances") != len(ids):
            refuse(f"{arm} {name}: utterances {doc.get('utterances')} != {len(ids)} records (partial)")
        docs[name] = doc
    return summary, docs, digests


def sources(summary: dict, docs: dict) -> list[dict]:
    return [summary.get("source") or {}] + [d.get("source") or {} for d in docs.values()]


def complete_attempt() -> tuple[int, dict[str, str], dict]:
    """(attempt number, {set: set_done SHA-256}, its "started" entry) of the single complete attempt."""
    if not tp.ATTEMPTS.exists():
        refuse(f"no attempt record {tp.ATTEMPTS}; the test run is not complete")
    data, _ = snapshot(tp.ATTEMPTS)
    entries = []
    for line in data.decode().splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except ValueError:
                refuse(f"{tp.ATTEMPTS} holds a line that is not JSON")
    complete = {e.get("attempt") for e in entries if e.get("state") == "complete"}
    if len(complete) != 1:
        refuse(f"the attempt record has {len(complete)} complete attempts, not exactly one complete attempt")
    n = complete.pop()
    if any(e.get("state") == "failed" and e.get("attempt") == n for e in entries):
        refuse(f"attempt {n} is recorded both failed and complete")
    done: dict[str, str] = {}
    for e in entries:
        if e.get("state") == "set_done" and e.get("attempt") == n:
            if e.get("set") in done:
                refuse(f"attempt {n} records {e.get('set')} twice")
            done[e.get("set")] = e.get("sha256")
    if set(done) != set(paths.TEST_SETS):
        refuse(f"complete attempt {n} has set_done records for {sorted(done)}, not all eight sets")
    started = [e for e in entries if e.get("state") == "started" and e.get("attempt") == n]
    if len(started) != 1:
        refuse(f"complete attempt {n} has {len(started)} started records, not one")
    return n, done, started[0]


def summary_projection(docs: dict[str, dict]) -> dict:
    """What binary_test's summary (evaluate.summarize plus its own fields) must hold for these per-set
    outputs, field for field (code review round 1, finding 3)."""
    first = docs[next(iter(paths.TEST_SETS))]
    sets = {}
    for name in paths.TEST_SETS:  # binary_test scores (and evaluate.summarize sums) in this order
        r = docs[name]
        sets[name] = {"wer": 100 * r["wer"]["wer"], "utterances": r["utterances"],
                      "scored_utterances": r["wer"]["utterances"],
                      "excluded_empty_reference": r["wer"]["excluded_empty_reference"],
                      "padded_short": r.get("padded_short", 0), "audio_hours": r["audio_hours"],
                      "wall_s": r["wall_s"], "limit": r["limit"]}
    return {"source": first["source"], "sets": sets, "wall_s": sum(docs[n]["wall_s"] for n in paths.TEST_SETS),
            "mean_wer": sum(sets[n]["wer"] for n in paths.MEAN_SETS) / len(paths.MEAN_SETS),
            "mean_sets": list(paths.MEAN_SETS), "padded_short": sum(e["padded_short"] for e in sets.values()),
            "scorer": first["scorer"], "settings": first["settings"], "binary_test": first["binary_test"],
            "decoder_graphs_mode": sorted({docs[n]["decoder_graphs_mode"]["after"] for n in paths.TEST_SETS})}


def check_mb1(summary: dict, docs: dict, digests: dict[str, str]) -> tuple[int, dict]:
    for name, doc in docs.items():
        if (doc.get("source") or {}).get("sha256") != tp.WEIGHTS_SHA256:
            refuse(f"MB1 {name}: source SHA-256 {(doc.get('source') or {}).get('sha256')} is not the pinned "
                   f"weights {tp.WEIGHTS_SHA256}")
    if (summary.get("source") or {}).get("sha256") != tp.WEIGHTS_SHA256:
        refuse(f"MB1 summary.json: source SHA-256 is not the pinned weights {tp.WEIGHTS_SHA256}")
    n, done, started = complete_attempt()
    for name in paths.TEST_SETS:
        got = digests[name]  # of the bytes parsed into docs[name]
        if got != done[name]:
            refuse(f"MB1 {name}.json SHA-256 {got} is not complete attempt {n}'s set_done hash {done[name]} "
                   "(replaced after scoring, or from another attempt)")
    first = docs[next(iter(paths.TEST_SETS))]
    run = {"mode": "test", "batch_size": tp.BATCH_SIZE, "max_batch_seconds": tp.MAX_BATCH_SECONDS, "limit": None}
    for name, doc in docs.items():
        modes = doc.get("decoder_graphs_mode") or {}
        if (modes.get("before"), modes.get("after")) != (tp.DECODER_GRAPHS_MODE, tp.DECODER_GRAPHS_MODE):
            refuse(f"MB1 {name}: decoder mode {modes} is not {tp.DECODER_GRAPHS_MODE!r} before and after")
        if doc.get("settings") != first.get("settings"):
            refuse(f"MB1 {name}: numerical settings differ from {next(iter(paths.TEST_SETS))}'s")
        if doc.get("binary_test") != run:
            refuse(f"MB1 {name}: binary_test record {doc.get('binary_test')} is not the production test run {run}")
    if summary.get("attempt") != n:
        refuse(f"MB1 summary.json is from attempt {summary.get('attempt')}, not complete attempt {n}")
    if summary.get("per_set_sha256") != done:
        refuse("MB1 summary.json per-set hashes disagree with its per-set files")
    expected = summary_projection(docs)
    differ = [k for k, v in expected.items() if summary.get(k) != v]
    if differ:
        refuse(f"MB1 summary.json disagrees with its per-set files in {differ}")
    return n, started


def check_gate(gate: dict, digest: str, started: dict, attempt: int, docs: dict) -> None:
    """The dev-check gate (parsed content and the SHA-256 of the same bytes) is the one complete attempt
    `attempt` launched with, and records the same identity, numerical settings, test manifests and
    decoder mode as the test outputs."""
    if not started.get("gate_sha256") or digest != started.get("gate_sha256"):
        refuse(f"the dev-check gate.json (SHA-256 {digest}) is not the gate attempt {attempt} launched with "
               f"({started.get('gate_sha256')}); a replaced or later dev check is not the launch gate")
    for name, doc in docs.items():
        if gate.get("identity") != (doc.get("source") or {}).get("pinned_identity"):
            refuse(f"the dev-check gate records another pinned identity than MB1 {name}")
        if gate.get("settings") != doc.get("settings"):
            refuse(f"the dev-check gate records other numerical settings than MB1 {name}")
        if (gate.get("manifests") or {}).get(name) != doc.get("manifest_sha256"):
            refuse(f"the dev-check gate records another {name} test manifest than MB1 scored")
    if gate.get("decoder_graphs_mode") != tp.DECODER_GRAPHS_MODE:
        refuse(f"the dev-check gate decoded in decoder mode {gate.get('decoder_graphs_mode')!r}")


def check_scorer(mb1_summary: dict, mb1_docs: dict, gate: dict) -> None:
    """Recorded scorer hashes equal the pinned ones in every MB1 output and in the dev-check gate."""
    own = sha256(Path(tp.__file__))
    records = [("summary.json", mb1_summary.get("scorer"))] + [(f"{n}.json", d.get("scorer")) for n, d in mb1_docs.items()]
    for label, scorer in records:
        scorer = scorer or {}
        if scorer.get("hashes") != tp.SCORER_HASHES or scorer.get("binary_test_paths_sha256") != own:
            refuse(f"MB1 {label}: recorded scorer hashes differ from the pinned ones")
    if gate.get("scorer") != mb1_docs[paths.MEAN_SETS[0]].get("scorer"):
        refuse("the dev-check gate and the test run record different scorer hashes")
    if not (gate.get("pass") is True and gate.get("limit") is None and gate.get("batch_size") == tp.BATCH_SIZE
            and gate.get("test_launch_gate") is True):
        refuse(f"the dev check {tp.DEV_CHECK / tp.GATE} is not a passing batch-{tp.BATCH_SIZE} gate")


def check_comparable(arms: dict[str, tuple[dict, dict]], manifests: dict) -> None:
    """Same corpus (current manifests) and M1-test's decoding and normalizer for every arm; B0 the pinned
    pretrained model; every arm on the pinned base model."""
    for name in paths.TEST_SETS:
        current = manifests[name][0]
        for arm, (_, docs) in arms.items():
            if docs[name].get("manifest_sha256") != current:
                refuse(f"{arm} {name}: scored manifest {docs[name].get('manifest_sha256')}, current {current}")
        m1 = arms["M1"][1][name]
        for arm in ("B0", "MB1"):
            doc = arms[arm][1][name]
            if doc.get("decoding") != m1.get("decoding"):
                refuse(f"{arm} {name}: decoding metadata differ from M1-test's")
            if doc.get("normalizer") != m1.get("normalizer"):
                refuse(f"{arm} {name}: normalizer metadata differ from M1-test's")
    for source in sources(*arms["B0"]):
        if not (source.get("kind") == "pretrained" and source.get("sha256") == tp.BASE_MODEL["nemo_sha256"]
                and source.get("base_model") == tp.BASE_MODEL):
            refuse(f"B0 source {source.get('kind')} {source.get('sha256')} is not the pinned pretrained model")
    for arm in ("M1", "MB1"):
        if any(s.get("base_model") != tp.BASE_MODEL for s in sources(*arms[arm])):
            refuse(f"{arm}: base model is not the pinned {tp.BASE_MODEL['revision']}")


def empty_fraction(doc: dict) -> tuple[float, int]:
    scored = [r for r in doc["records"] if r.get("scored", True)]
    empty = sum(1 for r in scored if not (r.get("hyp_norm") or "").strip())
    return empty / max(1, len(scored)), empty


def main() -> int:
    changed = [k for k, v in tp.SCORER_HASHES.items() if sha256(tp.scorer_path(k)) != v]
    if changed:
        refuse(f"scorer sources differ from SCORER_HASHES: {changed}")
    manifests = read_manifests()
    arms, digests = {}, {}
    for arm, directory in arm_dirs().items():
        summary, docs, digests[arm] = load_arm(arm, directory, manifests)
        arms[arm] = (summary, docs)
    mb1_n, started = check_mb1(*arms["MB1"], digests["MB1"])
    m1_shas = {s.get("sha256") for s in sources(*arms["M1"])}
    if m1_shas != {tp.M1_EXPORT_SHA256}:
        refuse(f"M1-test records source SHA-256 {sorted(map(str, m1_shas))}, not M1's pinned export "
               f"{tp.M1_EXPORT_SHA256}")
    gate, gate_sha256 = snapshot_json(tp.DEV_CHECK / tp.GATE)  # read once: parsed and hashed from these bytes
    check_scorer(*arms["MB1"], gate)
    check_gate(gate, gate_sha256, started, mb1_n, arms["MB1"][1])
    check_comparable(arms, manifests)

    sets = list(paths.TEST_SETS)
    cols = paths.MEAN_SETS + ["mean", "common_voice"]
    rows: dict[str, dict] = {}
    pub = {k: v for k, v in paths.PUBLISHED_WER.items() if k in paths.TEST_SETS}
    pub["mean"] = sum(pub[k] for k in paths.MEAN_SETS) / len(paths.MEAN_SETS)
    rows["NVIDIA published"] = pub
    empties: dict[str, dict] = {}
    counts: dict[str, dict] = {}
    for arm in ARMS:
        docs = arms[arm][1]
        row = {n: 100 * docs[n]["wer"]["wer"] for n in sets}
        row["mean"] = sum(row[n] for n in paths.MEAN_SETS) / len(paths.MEAN_SETS)
        rows[arm] = row
        empties[arm] = {n: empty_fraction(docs[n])[0] for n in sets}
        counts[arm] = {n: empty_fraction(docs[n])[1] for n in sets}
    differences = {}
    for ref in ("M1", "B0"):
        differences[f"MB1 - {ref}"] = {c: {"points": rows["MB1"][c] - rows[ref][c],
                                           "relative_percent": 100 * (rows["MB1"][c] - rows[ref][c]) / rows[ref][c]}
                                       for c in sets + ["mean"]}
    mb1_bytes = arms["MB1"][0]["source"]["export_manifest"]["bytes"]
    m1_bytes = arms["M1"][0]["source"]["export_manifest"]["bytes"]
    sizes = {"MB1": round(mb1_bytes["file_bytes"] / 1e6, 1), "M1": round(m1_bytes["file_bytes"] / 1e6, 1),
             "original": round(mb1_bytes["original_nemo_bytes"] / 1e6, 1)}
    dev = {"pass": gate["pass"], "mean_percent": 100 * gate["mean"], "recorded_mean_percent": 100 * gate["recorded_mean"],
           "mean_diff_points": gate["mean_diff_points"], "batch_size": gate["batch_size"],
           "per_set_diff_points": {n: e["diff_points"] for n, e in gate["per_set"].items()},
           "tolerances": gate["tolerances"]}
    # Sanity checks (plan "Report"): pause the write-up for investigation, never re-score.
    sanity = [f"{s}: MB1 {rows['MB1'][s]:.2f} > 3x B0 {rows['B0'][s]:.2f}" for s in sets
              if rows["MB1"][s] > 3 * rows["B0"][s]]
    sanity += [f"{s}: {100 * empties['MB1'][s]:.2f}% empty MB1 hypotheses, more than B0 ({counts['B0'][s]} vs "
               f"{counts['MB1'][s]})" for s in sets
               if empties["MB1"][s] > 0.01 and counts["MB1"][s] > counts["B0"][s]]
    result = {"rows": rows, "columns": cols, "mean_sets": paths.MEAN_SETS, "separate_sets": ["common_voice"],
              "differences": differences, "empty_hypothesis_fraction": empties, "empty_hypotheses": counts,
              "export_mb": sizes, "dev_check": dev, "label": LABEL, "sanity_flags": sanity,
              "mb1_attempt": mb1_n, "mb1_weights_sha256": tp.WEIGHTS_SHA256, "m1_export_sha256": tp.M1_EXPORT_SHA256,
              "scorer_hashes": tp.SCORER_HASHES}

    def fmt_row(label: str, values: dict, spec: str = ".2f", suffix: str = "") -> str:
        return f"| {label} | " + " | ".join(f"{values[c]:{spec}}{suffix}" if c in values else "n/a" for c in cols) + " |"

    table = ["| Arm | " + " | ".join(cols) + " |", "| --- |" + " ---: |" * len(cols)]
    table += [fmt_row(label, rows[label]) for label in rows]
    for key, diff in differences.items():
        table.append(fmt_row(f"{key} (points)", {c: d["points"] for c, d in diff.items()}, "+.2f"))
        table.append(fmt_row(f"{key} (relative)", {c: d["relative_percent"] for c, d in diff.items()}, "+.1f", "%"))
    empty_table = ["| Arm | " + " | ".join(sets) + " |", "| --- |" + " ---: |" * len(sets)]
    empty_table += [f"| {arm} | " + " | ".join(f"{100 * empties[arm][s]:.2f}%" for s in sets) + " |" for arm in ARMS]
    header = ("# Test-set WER (%), Open ASR Leaderboard sets: B0 / M1 / MB1\n\n"
              + (("**SANITY CHECK FAILED; investigate before any write-up:** " + "; ".join(sanity) + "\n\n")
                 if sanity else "")
              + "Mean over the seven sets NVIDIA reports except TED-LIUM (not in the public bundle); Common Voice "
              "reported separately. B0 = FP32 original, M1 = ternary QAT main run, MB1 = binary QAT main run "
              "(final checkpoint, step 250,000, scored once on the model rebuilt from its 1-bit export), all "
              "decoded by the frozen evaluate.py at batch 64 with the same manifests and normalizer.\n\n"
              f"**{LABEL}**\n\n"
              f"Export sizes: MB1 {sizes['MB1']:.1f} MB, M1 {sizes['M1']:.1f} MB, original .nemo "
              f"{sizes['original']:.0f} MB.\n\n"
              f"Dev cross-check (launch gate): mean {dev['mean_percent']:.4f}% vs recorded "
              f"{dev['recorded_mean_percent']:.4f}% ({dev['mean_diff_points']:+.4f} points, tolerance 0.02; per set "
              "within 0.10): " + ("PASS" if dev["pass"] else "FAIL") + ".\n\n")
    text = (header + "\n".join(table) + "\n\n## Empty hypotheses (fraction of scored utterances)\n\n"
            + "\n".join(empty_table) + "\n")
    tp.RESULTS.mkdir(parents=True, exist_ok=True)
    (tp.RESULTS / "test.json").write_text(json.dumps(result, indent=1) + "\n")
    (tp.RESULTS / "TEST.md").write_text(text)
    print("\n".join(table))
    if sanity:
        print("SANITY CHECK FAILED:", "; ".join(sanity))
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
