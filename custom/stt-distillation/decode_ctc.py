"""Frozen-posterior greedy/beam/3-gram comparison with auditable N-best lists."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
from pyctcdecode import build_ctcdecoder

from common import digit_string, distance, norm, save, scores
from recovery_core import DOMAINS, ROOT, artifact, read


def greedy(log_probs, labels, blank):
    ids = log_probs.argmax(-1).tolist()
    tokens = [
        labels[i]
        for j, i in enumerate(ids)
        if i != blank and (j == 0 or i != ids[j - 1])
    ]
    return norm("".join(tokens).replace("▁", " "))


def metrics(predictions):
    result = {}
    for domain in DOMAINS:
        rows = [r for r in predictions if r["domain"] == domain]
        pairs = [(r["reference"], r["prediction"]) for r in rows]
        result[domain] = scores(pairs)
        result[domain]["oracle"] = scores(
            [
                (
                    r["reference"],
                    min(
                        r["nbest"],
                        key=lambda h: distance(
                            r["reference"].split(), norm(h["text"]).split()
                        ),
                    )["text"],
                )
                for r in rows
            ]
        )
        if domain == "digits":
            result[domain].update(
                exact=sum(
                    digit_string(r["reference"]) == digit_string(r["prediction"])
                    for r in rows
                ),
                total=len(rows),
            )
    speech = [result[d]["wer"] for d in DOMAINS[:2]]
    result["speech_wer"] = (
        sum(speech) / 2 if all(v is not None for v in speech) else None
    )
    return result


def choose(results):
    complete = [
        r for r in results if r["complete"] and r["metrics"]["speech_wer"] is not None
    ]
    greedy_result = next((r for r in complete if r["name"] == "greedy"), None)
    if greedy_result is None:
        return dict(selected=None, tuning_best_lm=None, lm_screen_gate_met=False)
    digit_floor = greedy_result["metrics"]["digits"]["exact"]
    no_lm = [
        r
        for r in complete
        if not r.get("config", {}).get("lm")
        and r["metrics"]["digits"]["exact"] >= digit_floor
    ]
    baseline = min(no_lm, key=lambda r: r["metrics"]["speech_wer"])
    lm_results = [r for r in complete if r.get("config", {}).get("lm")]
    best = (
        min(lm_results, key=lambda r: r["metrics"]["speech_wer"])
        if lm_results
        else None
    )
    eligible = [
        r
        for r in lm_results
        if r["metrics"]["digits"]["exact"] >= digit_floor
        and r["metrics"]["speech_wer"] <= 0.9 * baseline["metrics"]["speech_wer"]
        and all(
            r["metrics"][d]["wer"] <= baseline["metrics"][d]["wer"] + 0.01
            for d in DOMAINS[:2]
        )
    ]
    selected = (
        min(eligible, key=lambda r: r["metrics"]["speech_wer"])
        if eligible
        else baseline
    )
    return dict(
        selected=selected["name"],
        tuning_best_lm=best["name"] if best else None,
        lm_screen_gate_met=bool(eligible),
        digit_exact_floor=digit_floor,
        acceptance_scope="Reused-validation screen only; fresh data and critical-error confirmation still required",
    )


def run(cache, output, seconds=1800, quick=False, lm_path=None):
    started = time.monotonic()
    cache = artifact(cache)
    output = artifact(output)
    output.mkdir(parents=True, exist_ok=True)
    index = read(cache / "index.json")
    entries = index["entries"]
    arrays = []
    from common import digest

    for row in entries:
        assert digest(cache / row["file"]) == row["sha256"]
        z = np.load(cache / row["file"], allow_pickle=False)
        assert z.shape == (row["frames"], len(index["labels"])) and np.isfinite(z).all()
        hyp = greedy(z, index["labels"], index["blank_id"])
        assert hyp == row["greedy"], (
            f"Greedy parity failed for {row['id']}: {hyp!r} != {row['greedy']!r}"
        )
        arrays.append(z)
    baseline = [
        dict(r, prediction=r["greedy"], nbest=[dict(text=r["greedy"])]) for r in entries
    ]
    results = [
        dict(name="greedy", complete=index["complete"], metrics=metrics(baseline))
    ]
    save(
        output / "greedy.json",
        dict(predictions=baseline, metrics=results[0]["metrics"]),
    )
    lm = read(ROOT / "models/slr11/source.json")
    model_path = str(lm_path or lm.get("binary") or lm["arpa"])
    unigrams = (ROOT / "models/slr11/unigrams.txt").read_text().splitlines()
    configs = [
        dict(name=f"beam-{w}", width=w, alpha=0.0, beta=0.0, lm=False)
        for w in ([8, 32] if quick else [8, 32, 128])
    ]
    configs += [
        dict(name=f"lm-32-a{a}-b{b}", width=32, alpha=a, beta=b, lm=True)
        for a in ([0.5] if quick else [0.25, 0.5, 1.0])
        for b in ([0] if quick else [-1, 0, 1])
    ]
    extra_added = False
    pos = 0
    while pos < len(configs):
        config = configs[pos]
        pos += 1
        if time.monotonic() - started > seconds - 30:
            break
        before = time.monotonic()
        decoder = build_ctcdecoder(
            index["labels"],
            kenlm_model_path=model_path if config["lm"] else None,
            unigrams=unigrams if config["lm"] else None,
            alpha=config["alpha"],
            beta=config["beta"],
        )
        load_seconds = time.monotonic() - before
        predictions = []
        latency = []
        for row, z in zip(entries, arrays):
            if time.monotonic() - started > seconds - 15:
                break
            tick = time.perf_counter()
            beams = decoder.decode_beams(
                z,
                beam_width=config["width"],
                beam_prune_logp=-10.0,
                token_min_logp=-5.0,
                prune_history=False,
            )
            latency.append(time.perf_counter() - tick)
            assert beams
            predictions.append(
                dict(
                    id=row["id"],
                    domain=row["domain"],
                    speaker=row.get("speaker"),
                    reference=row["reference"],
                    prediction=norm(beams[0][0]),
                    nbest=[
                        dict(
                            text=norm(b[0]),
                            acoustic_score=float(b[3]),
                            combined_score=float(b[4]),
                        )
                        for b in beams[:20]
                    ],
                )
            )
        complete = len(predictions) == len(entries) and index["complete"]
        result = dict(
            name=config["name"],
            config=config,
            complete=complete,
            examples=len(predictions),
            requested=len(entries),
            load_seconds=load_seconds,
            decode_seconds=sum(latency),
            rtf=sum(latency) / sum(r["seconds"] for r in entries[: len(predictions)])
            if predictions
            else None,
            p50_seconds=float(np.median(latency)) if latency else None,
            p95_seconds=float(np.quantile(latency, 0.95)) if latency else None,
            process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            metrics=metrics(predictions),
            pruning=dict(
                beam_prune_logp=-10.0, token_min_logp=-5.0, prune_history=False
            ),
            timing_scope="CPU whole-utterance offline search after audio; not streaming finalization or energy",
        )
        save(output / (config["name"] + ".json"), dict(result, predictions=predictions))
        results.append(result)
        print(
            json.dumps(
                dict(
                    name=config["name"],
                    complete=complete,
                    speech_wer=result["metrics"]["speech_wer"],
                )
            ),
            flush=True,
        )
        del decoder
        if pos == len(configs) and not quick and not extra_added:
            best = sorted(
                [r for r in results if r.get("config", {}).get("lm") and r["complete"]],
                key=lambda r: r["metrics"]["speech_wer"],
            )[:2]
            for r in best:
                for width in (8, 128):
                    c = dict(
                        r["config"],
                        width=width,
                        name=r["name"].replace("lm-32-", f"lm-{width}-"),
                    )
                    configs.append(c)
            extra_added = True
    decision = choose(results)
    summary = dict(
        cache=str(cache),
        cache_source=index["source"],
        lm=model_path,
        results=results,
        **decision,
        elapsed_seconds=time.monotonic() - started,
        caveats=[
            "reused development, not fresh confirmation",
            "pretrained LibriSpeech overlap",
            "RSS is process high-water mark, not isolated incremental LM memory",
            "critical natural quantities/negation, streaming and device energy unmeasured",
        ],
    )
    save(output / "summary.json", summary)
    lines = [
        "# Local CTC decoding comparison",
        "",
        "| Decoder | Complete | Speech WER | Decode RTF |",
        "| --- | --- | ---: | ---: |",
    ]
    for r in results:
        w = r["metrics"]["speech_wer"]
        lines.append(
            f"| {r['name']} | {r['complete']} | {w:.2%} | {r.get('rtf', '—')} |"
            if w is not None
            else f"| {r['name']} | {r['complete']} | unmeasured | — |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--seconds", type=float, default=1800)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--lm-path", type=Path)
    a = p.parse_args()
    run(a.cache, a.output, a.seconds, a.quick, a.lm_path)
