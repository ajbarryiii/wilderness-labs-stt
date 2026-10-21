"""Build the Phase 3 test table (plans/test-scoring.md) from saved evaluation outputs. No GPU, no decoding.

    python report_test.py      # writes results/TEST.md and results/test.json; exits 1 if a check fails

B1 is merged from eval/b1-ptq (3 sets) and eval/b1-ptq-rest (5 sets) only if both scored the same
export file (identical SHA-256, equal to the current runs/b1-ptq export). Each set must come from
exactly one complete output file. All arms must have scored the current test manifests with the same
decoding and normalizer, and B0 must be the pinned pretrained model. The mean is over paths.MEAN_SETS;
Common Voice is reported apart.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import paths

ARMS = {"B0": [paths.EVAL / "b0-pretrained"],
        "B1": [paths.EVAL / "b1-ptq", paths.EVAL / "b1-ptq-rest"],
        "M1": [paths.EVAL / "M1-test"]}
B1_EXPORT = paths.RUNS / "b1-ptq" / "export"
M1_PRECHECK = paths.EVAL / "M1-precheck.json"
OUT = paths.HERE / "results"
# Sanity flags investigated after the fact (DESIGN.md "Main-run and test results"); exact flag text.
RESOLVED = {"ami: 1.43% empty M1 hypotheses":
            "The FP32 original leaves more AMI hypotheses empty (3.12%) than M1 (1.43%); AMI is full of "
            "sub-second backchannels, and M1 has fewer or equal empty outputs than B0 on every set."}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_arm(dirs: list[Path]) -> tuple[dict, set]:
    """{set: wer fraction} and the set of source SHA-256s, refusing duplicates or incomplete sets."""
    wer, shas, empty, meta = {}, set(), {}, {}
    for d in dirs:
        summary = json.loads((d / "summary.json").read_text())
        shas.add(summary["source"].get("sha256"))
        for name in summary["sets"]:  # only sets the run itself recorded; stray files are ignored
            if name not in paths.TEST_SETS:
                continue
            doc = json.loads((d / f"{name}.json").read_text())
            if name in wer:
                raise SystemExit(f"{name} scored twice for {dirs}")
            if doc.get("limit"):
                raise SystemExit(f"{d / name}.json is a partial (--limit) evaluation")
            if doc["source"].get("sha256") != summary["source"].get("sha256"):
                raise SystemExit(f"{d / name}.json scored {doc['source'].get('sha256')}, its summary "
                                 f"{summary['source'].get('sha256')}")
            shas.add(doc["source"].get("sha256"))
            meta[name] = {"manifest_sha256": doc["manifest_sha256"], "decoding": doc["decoding"],
                          "normalizer": doc["normalizer"], "source": doc["source"]}
            wer[name] = doc["wer"]["wer"]
            scored = [r for r in doc["records"] if r.get("scored", True)]
            empty[name] = sum(1 for r in scored if not (r.get("hyp_norm") or "").strip()) / max(1, len(scored))
    missing = [n for n in paths.TEST_SETS if n not in wer]
    if missing:
        raise SystemExit(f"{dirs}: missing sets {missing}")
    return wer, shas, empty, meta


def check_comparable(metas: dict[str, dict]) -> None:
    """Every arm scored the same test corpus with the same decoding and normalizer, and B0 is
    the pinned pretrained model. Raises on any mismatch."""
    lock = json.loads((paths.MODEL_DIR / "lock.json").read_text())
    pinned = lock["files"][paths.MODEL_FILE.name]
    for name in paths.TEST_SETS:
        current = sha256(paths.MANIFESTS / f"test_{name}.jsonl")
        for arm, meta in metas.items():
            m = meta[name]
            if m["manifest_sha256"] != current:
                raise SystemExit(f"{arm} {name}: scored manifest {m['manifest_sha256']}, current {current}")
            for key in ("decoding", "normalizer"):
                if m[key] != metas["B0"][name][key]:
                    raise SystemExit(f"{arm} {name}: {key} differs from B0's")
            base = m["source"].get("base_model") or {}
            if base.get("revision") != paths.MODEL_REVISION:
                raise SystemExit(f"{arm} {name}: base revision {base.get('revision')} != {paths.MODEL_REVISION}")
        b0 = metas["B0"][name]["source"]
        if b0.get("kind") != "pretrained" or b0.get("sha256") != pinned:
            raise SystemExit(f"B0 {name}: source {b0.get('kind')} {b0.get('sha256')} is not the pinned "
                             f"pretrained model {pinned}")


def main() -> int:
    precheck = json.loads(M1_PRECHECK.read_text())
    if not precheck.get("pass"):
        raise SystemExit("M1 precheck did not pass")
    rows, empties, metas = {}, {}, {}
    for arm, dirs in ARMS.items():
        wer, shas, empties[arm], metas[arm] = load_arm(dirs)
        if arm == "B1":
            current = sha256(B1_EXPORT / "export.safetensors")
            if shas != {current}:
                raise SystemExit(f"B1 outputs scored different weights: {shas} vs current {current}")
        if arm == "M1" and shas != {precheck["export_sha256"]}:
            raise SystemExit(f"M1 test outputs scored {shas}, precheck verified {precheck['export_sha256']}")
        rows[arm] = {**{k: 100 * v for k, v in wer.items()},
                     "mean": 100 * sum(wer[k] for k in paths.MEAN_SETS) / len(paths.MEAN_SETS)}
    check_comparable(metas)
    pub = dict(paths.PUBLISHED_WER)
    pub["mean"] = sum(pub[k] for k in paths.MEAN_SETS) / len(paths.MEAN_SETS)
    cols = paths.MEAN_SETS + ["mean", "common_voice"]
    lines = ["| Arm | " + " | ".join(cols) + " |", "| --- |" + " ---: |" * len(cols)]
    lines.append("| NVIDIA published | " + " | ".join(f"{pub[c]:.2f}" if c in pub else "n/a" for c in cols) + " |")
    for arm, r in rows.items():
        lines.append(f"| {arm} | " + " | ".join(f"{r[c]:.2f}" for c in cols) + " |")
    gap = ["| M1 - B0 (points) | " + " | ".join(f"{rows['M1'][c] - rows['B0'][c]:+.2f}" for c in cols) + " |",
           "| M1 / B0 (relative) | " + " | ".join(f"{rows['M1'][c] / rows['B0'][c]:.2f}x" for c in cols) + " |"]
    m1 = json.loads((paths.RUNS / "main-M1-P2-lr5e-4" / "export" / "manifest.json").read_text())
    size = m1["bytes"]["file_bytes"] / 1e6
    original = paths.MODEL_FILE.stat().st_size / 1e6
    # plans/test-scoring.md sanity checks, M1 only: pause the write-up, never re-score.
    sanity = [f"{s}: M1 {rows['M1'][s]:.2f} > 3x B0 {rows['B0'][s]:.2f}" for s in paths.TEST_SETS
              if rows["M1"][s] > 3 * rows["B0"][s]]
    sanity += [f"{s}: {100 * f:.2f}% empty M1 hypotheses" for s, f in empties["M1"].items() if f > 0.01]
    OUT.mkdir(exist_ok=True)
    (OUT / "test.json").write_text(json.dumps({"rows": rows, "published": pub, "m1_export_mb": size,
                                              "original_nemo_mb": original, "empty_hypothesis_fraction": empties,
                                              "sanity_flags": sanity,
                                              "m1_export_sha256": precheck["export_sha256"]}, indent=1))
    open_flags = [f for f in sanity if f not in RESOLVED]
    header = ("# Test-set WER (%), Open ASR Leaderboard sets\n\n"
              + (("**SANITY CHECK FAILED; investigate before any write-up:** " + "; ".join(open_flags) + "\n\n")
                 if open_flags else "")
              + "".join(f"Sanity flag `{f}`: investigated and closed. {RESOLVED[f]}\n\n"
                        for f in sanity if f in RESOLVED)
              + "Mean over the seven sets NVIDIA reports except TED-LIUM (not in the public bundle); Common "
              "Voice reported separately. B0 = FP32 original, B1 = ternary PTQ without training, M1 = ternary "
              f"QAT main run (final checkpoint, scored on the rebuilt export). M1 export {size:.1f} MB versus "
              f"{original:.0f} MB for the original .nemo file.\n\n"
              "B0 was scored on 2026-09-30 by an earlier revision of evaluate.py that predates the 30 ms "
              "padding rule and recorded no evaluator hash; the rule cannot affect any test set (shortest "
              "test utterance 40 ms) and the saved decoding and normalizer metadata match.\n\n")
    (OUT / "TEST.md").write_text(header + "\n".join(lines + gap) + "\n")
    print("\n".join(lines + gap))
    if open_flags:
        print("SANITY CHECK FAILED:", "; ".join(open_flags))
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
