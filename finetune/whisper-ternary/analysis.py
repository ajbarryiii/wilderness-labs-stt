"""Post-hoc error analysis of evaluation JSON (DESIGN.md "Secondary analyses").

The primary metric is the preregistered, uncapped corpus WER stored in each
evaluation JSON under "wer" (DESIGN.md, Evaluation). This module copies that
number unchanged and never rewrites an evaluation file. Everything else it
reports is a secondary diagnostic.

Runaway utterance: the decoder transcribes the speech and then keeps generating
fluent continuation text instead of stopping. An utterance counts as runaway if
either criterion holds (both are also reported on their own):

- by insertions: its stored insertion count I >= min_insertions (default 20);
- by length: its normalized hypothesis has more than
  ratio * (reference words) + 10 words (default ratio 1.5).

Secondary WERs, all recomputed with wer.corpus_wer / wer.edit_counts so the
counting is identical to the primary metric:

- "WER excl. runaway": corpus WER over the non-runaway utterances only;
- "WER capped": corpus WER over all utterances, with each runaway hypothesis cut
  to its first (reference words + 5) words. Uses the reference to decide.
- "WER dur-capped" (the deployable rule): every hypothesis is cut to its first
  ceil(4.5 * audio seconds) + 5 words, the duration taken from the split
  manifest paths.MANIFESTS/<split>.json. It also counts how many hypotheses were
  shortened, the words removed, and how many REFERENCES the same cap would cut
  ("refs truncated"; 0 on LibriSpeech, printed so that claim is checked on
  every run). A missing manifest or utterance id makes these fields null for
  that file, with the reason in the output.

Usage:
  analysis.py FILE [FILE ...] [--json OUT] [--markdown OUT.md]
  analysis.py trajectory RUN_DIR      RUN_DIR/dev-subset/step-*.json in step order
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

import paths
import wer

MIN_INSERTIONS = 20
RATIO = 1.5
LENGTH_SLACK = 10  # words allowed above ratio * reference words before "by length" fires
RUNAWAY_CAP_EXTRA_WORDS = 5  # a reference-capped runaway hypothesis keeps reference words + this
DEFAULT_CAP_WORDS_PER_S = 4.5  # duration cap: ceil(words/s * audio seconds) + extra words
DEFAULT_CAP_EXTRA_WORDS = 5
DURATION_FIELDS = ("wer_duration_capped", "duration_capped_truncated",
                   "duration_capped_truncated_words", "reference_truncated")
NOTE = ("'WER excl. runaway', 'WER capped' and 'WER dur-capped' are SECONDARY diagnostics; "
        "the primary metric is the uncapped corpus WER (column WER)")
MARKDOWN_HEADING = ("Secondary analyses (post hoc). Primary metric is the uncapped corpus WER; "
                    "see DESIGN.md 'Secondary analyses'.")
_STEP_FILE = re.compile(r"step-(\d+)\.json")
_SUBSET_SPLIT = re.compile(r"dev_subset\(\s*([A-Za-z0-9-]+)")


def _words(text: str) -> list[str]:
    return text.split()


def runaway_by_insertions(record: dict, min_insertions: int = MIN_INSERTIONS) -> bool:
    """I >= min_insertions, using the record's stored insertion count."""
    return record["I"] >= min_insertions


def runaway_by_ratio(record: dict, ratio: float = RATIO) -> bool:
    """More than ratio * (reference words) + 10 normalized hypothesis words."""
    return len(_words(record["hyp_norm"])) > ratio * len(_words(record["ref_norm"])) + LENGTH_SLACK


def runaway(record: dict, min_insertions: int = MIN_INSERTIONS, ratio: float = RATIO) -> bool:
    """Runaway if either criterion holds: I >= min_insertions, or hyp words > ratio * ref words + 10."""
    return runaway_by_insertions(record, min_insertions) or runaway_by_ratio(record, ratio)


def capped_hypothesis(record: dict, extra: int = RUNAWAY_CAP_EXTRA_WORDS) -> str:
    """The normalized hypothesis cut to its first len(reference words) + extra words."""
    return " ".join(_words(record["hyp_norm"])[:len(_words(record["ref_norm"])) + extra])


def duration_cap_words(duration_s: float, words_per_s: float = DEFAULT_CAP_WORDS_PER_S,
                       extra: int = DEFAULT_CAP_EXTRA_WORDS) -> int:
    """ceil(words_per_s * duration_s) + extra: the most words the duration rule keeps."""
    return math.ceil(words_per_s * duration_s) + extra


def split_of(doc: dict) -> str:
    """Top-level "split", else the split named in a dev-subset "subset" description, else dev-clean."""
    if doc.get("split"):
        return doc["split"]
    match = _SUBSET_SPLIT.search(str(doc.get("subset", "")))
    return match.group(1) if match else "dev-clean"


def load_durations(split: str, manifests: Path | None = None) -> tuple[dict[str, float] | None, str]:
    """({id: duration_s}, manifest path) or (None, why it is unavailable); never raises for a bad file."""
    path = Path(paths.MANIFESTS if manifests is None else manifests) / f"{split}.json"
    try:
        return {r["id"]: float(r["duration_s"]) for r in json.loads(path.read_text())}, str(path)
    except FileNotFoundError:
        return None, f"manifest not found: {path}"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, f"manifest unreadable: {path} ({type(exc).__name__}: {exc})"


def duration_capped(records: list[dict], durations: dict[str, float],
                    words_per_s: float = DEFAULT_CAP_WORDS_PER_S,
                    extra: int = DEFAULT_CAP_EXTRA_WORDS) -> dict:
    """Every hypothesis cut to duration_cap_words(duration); S/D/I recounted by wer.corpus_wer."""
    capped, shortened, removed, refs_cut = [], 0, 0, 0
    for record in records:
        cap = duration_cap_words(durations[record["id"]], words_per_s, extra)
        hyp = _words(record["hyp_norm"])
        shortened += len(hyp) > cap
        removed += max(0, len(hyp) - cap)
        refs_cut += len(_words(record["ref_norm"])) > cap
        capped.append({"ref_norm": record["ref_norm"], "hyp_norm": " ".join(hyp[:cap])})
    return {"wer_duration_capped": _secondary_wer(capped), "duration_capped_truncated": shortened,
            "duration_capped_truncated_words": removed, "reference_truncated": refs_cut}


def _secondary_wer(records: list[dict]) -> float | None:
    """wer.corpus_wer over records; None when they hold no reference words."""
    if not any(_words(r["ref_norm"]) for r in records):
        return None
    return wer.corpus_wer(records)["wer"]


def _brief(record: dict, is_runaway: bool) -> dict:
    return {"id": record["id"], "ref_norm": record["ref_norm"][:120],
            "hyp_norm": record["hyp_norm"][:240], "S": record["S"], "D": record["D"],
            "I": record["I"], "ref_words": len(_words(record["ref_norm"])),
            "hyp_words": len(_words(record["hyp_norm"])), "runaway": is_runaway}


def top_by_insertions(records: list[dict], n: int) -> list[dict]:
    """The n records with most insertions (ties by id), highest first."""
    return sorted(records, key=lambda r: (-r["I"], r["id"]))[:n]


def diagnose(doc: dict, min_insertions: int = MIN_INSERTIONS, ratio: float = RATIO,
             cap_words_per_s: float = DEFAULT_CAP_WORDS_PER_S,
             cap_extra_words: int = DEFAULT_CAP_EXTRA_WORDS, manifests: Path | None = None) -> dict:
    """Secondary diagnostics for one evaluation JSON; "wer" is the primary value, copied untouched."""
    records = doc["records"]
    primary = doc["wer"]
    by_ins = [runaway_by_insertions(r, min_insertions) for r in records]
    by_ratio = [runaway_by_ratio(r, ratio) for r in records]
    flags = [a or b for a, b in zip(by_ins, by_ratio)]
    kept = [r for r, f in zip(records, flags) if not f]
    runaways = [r for r, f in zip(records, flags) if f]
    capped = [{"ref_norm": r["ref_norm"], "hyp_norm": capped_hypothesis(r) if f else r["hyp_norm"]}
              for r, f in zip(records, flags)]
    errors = primary["substitutions"] + primary["deletions"] + primary["insertions"]
    ratios = [len(_words(r["hyp_norm"])) / len(_words(r["ref_norm"]))
              for r in records if _words(r["ref_norm"])]

    split = split_of(doc)
    durations, source = load_durations(split, manifests)
    missing = [] if durations is None else [r["id"] for r in records if r["id"] not in durations]
    if durations is None:
        duration, status = dict.fromkeys(DURATION_FIELDS), f"unavailable, {source}"
    elif missing:
        duration = dict.fromkeys(DURATION_FIELDS)
        status = (f"unavailable, {len(missing)} of {len(records)} ids missing from {source} "
                  f"(first {missing[0]})")
    else:
        duration = duration_capped(records, durations, cap_words_per_s, cap_extra_words)
        status = f"ok, {source}"

    return {
        "split": split,
        "utterances": len(records),
        "wer": primary["wer"],
        "substitutions": primary["substitutions"],
        "deletions": primary["deletions"],
        "insertions": primary["insertions"],
        "runaway_count": sum(flags),
        "runaway_by_insertions": sum(by_ins),
        "runaway_by_ratio": sum(by_ratio),
        "runaway_insertions": sum(r["I"] for r in runaways),
        "wer_excluding_runaway": _secondary_wer(kept),
        "wer_runaway_capped": _secondary_wer(capped),
        **duration,
        "duration_cap_status": status,
        "insertion_share_of_errors": primary["insertions"] / errors if errors else None,
        "empty_hypotheses": sum(not _words(r["hyp_norm"]) for r in records),
        "median_hyp_ref_ratio": statistics.median(ratios) if ratios else None,
        "worst": [_brief(r, runaway(r, min_insertions, ratio)) for r in top_by_insertions(records, 5)],
        "worst_runaway": [_brief(r, True) for r in top_by_insertions(runaways, 3)],
        "criteria": {"min_insertions": min_insertions, "ratio": ratio,
                     "length_slack_words": LENGTH_SLACK,
                     "runaway_cap_extra_words": RUNAWAY_CAP_EXTRA_WORDS,
                     "duration_cap_words_per_s": cap_words_per_s,
                     "duration_cap_extra_words": cap_extra_words},
    }


# ---------------------------------------------------------------- rendering

FILE_HEADER = ["file", "utts", "WER", "S", "D", "I", "runaway", "by-I", "by-len",
               "WER excl. runaway", "WER capped", "WER dur-capped", "truncated (n)",
               "refs truncated", "empty hyps"]
TRAJECTORY_HEADER = ["step", "weight_fraction", "WER", "S", "D", "I", "runaway",
                     "WER excl. runaway", "WER dur-capped", "truncated (n)", "refs truncated"]


def label(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def _pct(value: float | None) -> str:
    return "null" if value is None else f"{100 * value:.2f}%"


def _count(value: int | None) -> str:
    return "null" if value is None else str(value)


def _file_row(name: str, d: dict) -> list[str]:
    return [name, str(d["utterances"]), _pct(d["wer"]), str(d["substitutions"]),
            str(d["deletions"]), str(d["insertions"]), str(d["runaway_count"]),
            str(d["runaway_by_insertions"]), str(d["runaway_by_ratio"]),
            _pct(d["wer_excluding_runaway"]), _pct(d["wer_runaway_capped"]),
            _pct(d["wer_duration_capped"]), _count(d["duration_capped_truncated"]),
            _count(d["reference_truncated"]), str(d["empty_hypotheses"])]


def _criteria(c: dict) -> str:
    return (f"runaway = I >= {c['min_insertions']} (by-I) or hyp words > {c['ratio']} * ref words"
            f" + {c['length_slack_words']} (by-len). capped = runaway hyp cut to ref words + "
            f"{c['runaway_cap_extra_words']} (uses the reference). dur-capped = every hyp cut to "
            f"ceil({c['duration_cap_words_per_s']} * audio s) + {c['duration_cap_extra_words']} "
            f"words (deployable); truncated (n) = hyps shortened; refs truncated = references "
            f"the same cap would cut (must be 0).")


def _unavailable(name: str, d: dict) -> str | None:
    if d["duration_cap_status"].startswith("ok"):
        return None
    return f"{name}: duration cap {d['duration_cap_status']}; its dur-capped fields are null"


def _table(header: list[str], rows: list[list[str]], left: int = 1) -> str:
    """Plain-text table; the first `left` columns are left-aligned, the rest right-aligned."""
    widths = [max(len(c) for c in col) for col in zip(header, *rows)]
    lines = [header, ["-" * w for w in widths], *rows]
    return "\n".join("  ".join(c.ljust(w) if i < left else c.rjust(w)
                               for i, (c, w) in enumerate(zip(line, widths))) for line in lines)


def _markdown_table(header: list[str], rows: list[list[str]], left: int = 1) -> str:
    align = [":---" if i < left else "---:" for i in range(len(header))]
    return "\n".join("| " + " | ".join(c.replace("|", "\\|") for c in line) + " |"
                     for line in [header, align, *rows])


def _worst_summary(result: dict) -> str:
    d = result["diagnostics"]
    return (f"{result['label']}: {d['runaway_count']} runaway utterances hold "
            f"{d['runaway_insertions']} of {d['insertions']} insertions")


def _worst_head(w: dict) -> str:
    return (f"{w['id']}  S {w['S']} D {w['D']} I {w['I']}  "
            f"ref {w['ref_words']} w, hyp {w['hyp_words']} w")


def analyse_files(files: list[Path], **options) -> list[dict]:
    """[{"file", "label", "diagnostics"}] for each evaluation JSON, in the given order."""
    return [{"file": str(f), "label": label(f),
             "diagnostics": diagnose(json.loads(Path(f).read_text()), **options)} for f in files]


def render_text(results: list[dict]) -> str:
    rows = [_file_row(r["label"], r["diagnostics"]) for r in results]
    notes = [n for r in results if (n := _unavailable(r["label"], r["diagnostics"]))]
    lines = [f"Error analysis per file. {NOTE}.", _criteria(results[0]["diagnostics"]["criteria"]),
             _table(FILE_HEADER, rows), *notes, "",
             f"Worst runaway utterances by insertions (max 3 per file). {NOTE}."]
    for r in results:
        lines.append(_worst_summary(r))
        for w in r["diagnostics"]["worst_runaway"]:
            lines += [f"  {_worst_head(w)}", f"    ref: {w['ref_norm']}", f"    hyp: {w['hyp_norm']}"]
    return "\n".join(lines)


def render_markdown(results: list[dict]) -> str:
    rows = [_file_row(r["label"], r["diagnostics"]) for r in results]
    notes = [f"- {n}" for r in results if (n := _unavailable(r["label"], r["diagnostics"]))]
    lines = [f"# {MARKDOWN_HEADING}", "", _criteria(results[0]["diagnostics"]["criteria"]), "",
             _markdown_table(FILE_HEADER, rows), ""]
    if notes:
        lines += [*notes, ""]
    lines += ["## Worst runaway utterances by insertions (max 3 per file)", ""]
    for r in results:
        lines += [f"**{_worst_summary(r)}**", ""]
        for w in r["diagnostics"]["worst_runaway"]:
            lines += [f"- `{_worst_head(w)}`", f"  - ref: {w['ref_norm']}",
                      f"  - hyp: {w['hyp_norm']}"]
        lines.append("")
    return "\n".join(lines)


def trajectory(run_dir: Path, **options) -> list[dict]:
    """One diagnose row per RUN_DIR/dev-subset/step-*.json, in numeric step order."""
    files = [(int(m.group(1)), p) for p in (Path(run_dir) / "dev-subset").glob("step-*.json")
             if (m := _STEP_FILE.fullmatch(p.name))]
    rows = []
    for step, path in sorted(files):
        doc = json.loads(path.read_text())
        rows.append({"step": doc.get("step", step), "weight_fraction": doc.get("weight_fraction"),
                     "file": str(path), **diagnose(doc, **options)})
    return rows


def trajectory_text(run_dir: Path, rows: list[dict]) -> str:
    table = [[str(r["step"]),
              "-" if r["weight_fraction"] is None else f"{r['weight_fraction']:.3f}",
              _pct(r["wer"]), str(r["substitutions"]), str(r["deletions"]), str(r["insertions"]),
              str(r["runaway_count"]), _pct(r["wer_excluding_runaway"]),
              _pct(r["wer_duration_capped"]), _count(r["duration_capped_truncated"]),
              _count(r["reference_truncated"])] for r in rows]
    notes = [n for r in rows if (n := _unavailable(f"step {r['step']}", r))]
    return "\n".join([f"Dev-subset trajectory of {run_dir}. {NOTE}.", _criteria(rows[0]["criteria"]),
                      _table(TRAJECTORY_HEADER, table, left=0), *notes])


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--min-insertions", type=int, default=MIN_INSERTIONS)
    common.add_argument("--ratio", type=float, default=RATIO)
    common.add_argument("--cap-words-per-second", type=float, default=DEFAULT_CAP_WORDS_PER_S)
    common.add_argument("--cap-extra-words", type=int, default=DEFAULT_CAP_EXTRA_WORDS)
    if argv[:1] == ["trajectory"]:
        parser = argparse.ArgumentParser(prog="analysis.py trajectory", parents=[common],
                                         description="Dev-subset evaluation history of one run.")
        parser.add_argument("run_dir", type=Path)
        args = parser.parse_args(argv[1:])
    else:
        parser = argparse.ArgumentParser(prog="analysis.py", parents=[common], description=__doc__,
                                         formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("files", type=Path, nargs="+", help="evaluation JSON files")
        parser.add_argument("--json", type=Path, help="write the full diagnose results here")
        parser.add_argument("--markdown", type=Path, help="write the table and worst list as Markdown")
        args = parser.parse_args(argv)
    options = {"min_insertions": args.min_insertions, "ratio": args.ratio,
               "cap_words_per_s": args.cap_words_per_second,
               "cap_extra_words": args.cap_extra_words}
    if argv[:1] == ["trajectory"]:
        rows = trajectory(args.run_dir, **options)
        if not rows:
            parser.error(f"no dev-subset/step-*.json under {args.run_dir}")
        print(trajectory_text(args.run_dir, rows))
        return 0
    results = analyse_files(args.files, **options)
    print(render_text(results))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"note": NOTE, "results": results}, indent=1) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(results) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
