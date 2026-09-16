"""Read-only diagnosis of the revised 2026-09-10 four-hour pilot.

Writes derived evidence under the original run's analysis/broad-investigation;
never modifies training inputs, checkpoints, configuration, or latest pointers.
Uses the frozen run modules, not potentially edited working-tree training code.
"""

import argparse
import collections
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["static", "checkpoints"])
    parser.add_argument(
        "--run",
        type=Path,
        default=Path(
            "/mnt/hd/wilderness-labs-stt/stt-distillation/runs/pilot-4h-20260910T180602Z"
        ),
    )
    args = parser.parse_args()
    run = args.run
    sys.path.insert(0, str(run / "code"))
    from common import digest, norm, save, scores, storage

    storage()
    out = run / "analysis/broad-investigation"
    read = lambda p: json.loads(p.read_text())
    cfg = read(run / "config.json")
    rows = read(run / "manifest.json")["rows"]
    subsets = read(run / "subsets.json")
    by_id = {r["id"]: r for r in rows}
    domains = list(cfg["domain_probabilities"])
    if args.mode == "static":
        import re

        plain = lambda s: " ".join(re.sub(r"[^a-z0-9' ]", " ", norm(s)).split())
        teachers = read(run.parent / "pilot-8h-20260910T044750Z/targets.json")[
            "targets"
        ]
        summary = {
            "groups": {},
            "timelines": {},
            "replayed_exposure": {},
            "provenance_verified": {},
        }
        provenance = read(run / "provenance.json")
        summary["provenance_verified"]["sources"] = all(
            digest(run / "code" / name) == h
            for name, h in provenance["source_hashes"].items()
        )
        for name in ["config", "manifest", "subsets"]:
            summary["provenance_verified"][name] = (
                digest(run / (name + ".json")) == provenance[name + "_sha256"]
            )
        for group in ["gate", "monitor", "train", "development"]:
            selected = (
                [by_id[i] for i in subsets[group]]
                if group in subsets
                else [r for r in rows if r["split"] == group]
            )
            summary["groups"][group] = {}
            for dom in domains:
                rs = [r for r in selected if r["domain"] == dom]
                pairs = [
                    (r["text"], teachers[r["id"]]["omi"])
                    for r in rs
                    if "omi" in teachers.get(r["id"], {})
                ]
                summary["groups"][group][dom] = {
                    "count": len(rs),
                    "unique_hours": sum(r["seconds"] for r in rs) / 3600,
                    "duration_quantiles": np.quantile(
                        [r["seconds"] for r in rs], [0, 0.5, 0.95, 1]
                    ).tolist(),
                    "characters_quantiles": np.quantile(
                        [len(norm(r["text"])) for r in rs], [0, 0.5, 0.95, 1]
                    ).tolist(),
                    "unique_transcripts": len({r["text"] for r in rs}),
                    "known_speakers": sorted(
                        {r["speaker"] for r in rs if r["speaker"] is not None}
                    ),
                    "unknown_speaker_clips": sum(r["speaker"] is None for r in rs),
                    "teacher_comparison": scores(pairs) if pairs else None,
                    "teacher_plain_exact": sum(plain(a) == plain(b) for a, b in pairs),
                }
        for mode in ["gate", "train"]:
            for arm in cfg["arms"]:
                folder = run / mode / arm["name"]
                if not folder.exists():
                    continue
                timeline = []
                for path in sorted(folder.glob("eval-*.json")):
                    e = read(path)
                    item = {"step": e["step"]}
                    for split in ["training", "development"]:
                        if split not in e:
                            continue
                        m = e[split]["metrics"]
                        item[split] = {
                            k: m[k]
                            for k in [
                                "overall",
                                "blank_fraction",
                                "modal_fraction",
                                "exact_utterances",
                            ]
                        }
                        item[split]["domains"] = {
                            d: {k: m[d][k] for k in ["wer", "cer", "empty_outputs"]}
                            for d in domains
                        }
                        item[split]["digits_exact"] = m["digits"]["exact"]
                        item[split]["most_common"] = collections.Counter(
                            p["prediction"] for p in e[split]["predictions"]
                        ).most_common(5)
                    timeline.append(item)
                logs = [
                    json.loads(line)
                    for line in (folder / "metrics.jsonl").read_text().splitlines()
                ]
                summary["timelines"][mode + "/" + arm["name"]] = {
                    "evaluations": timeline,
                    "all_finite": all(
                        np.isfinite(v[k]) for v in logs for k in ["loss", "grad_norm"]
                    ),
                    "fraction_gradient_clipped": np.mean(
                        [v["grad_norm"] > cfg["grad_clip"] for v in logs]
                    ).item(),
                    "first_1000_mean_loss": np.mean(
                        [v["loss"] for v in logs[:1000]]
                    ).item(),
                    "last_1000_mean_loss": np.mean(
                        [v["loss"] for v in logs[-1000:]]
                    ).item(),
                    "gradient_quantiles": np.quantile(
                        [v["grad_norm"] for v in logs], [0, 0.5, 0.95, 1]
                    ).tolist(),
                }
        # Replay the sampler and augmentation RNG without running a model.
        groups = {
            d: [r for r in rows if r["split"] == "train" and r["domain"] == d]
            for d in domains
        }
        rng = np.random.default_rng(cfg["seed"])
        aug_rng = np.random.default_rng(cfg["augmentation"]["seed"])
        sample_hash, aug_hash = hashlib.sha256(), hashlib.sha256()
        counts = collections.Counter()
        for step in range(1, 27001):
            for _ in range(cfg["batch_size"]):
                dom = rng.choice(domains, p=list(cfg["domain_probabilities"].values()))
                row = groups[dom][int(rng.integers(len(groups[dom])))]
                counts[row["id"]] += 1
                sample_hash.update((row["id"] + "\n").encode())
                if aug_rng.random() < 0.25:
                    prefix, gain = (20 if dom == "digits" else 0), 0
                else:
                    prefix, gain = (
                        int(aug_rng.integers(0, 101)),
                        int(aug_rng.choice([0, -6, -12, -18, -24])),
                    )
                aug_hash.update(f"{row['id']} {prefix} {gain}\n".encode())
            if step in [1000, 3000, 25000, 27000]:
                summary["replayed_exposure"][str(step)] = {
                    "sample_hash": sample_hash.hexdigest(),
                    "augmentation_hash": aug_hash.hexdigest(),
                    "domains": {
                        d: {
                            "unique_seen": sum(counts[r["id"]] > 0 for r in rs),
                            "presentations": sum(counts[r["id"]] for r in rs),
                            "presentations_quantiles": np.quantile(
                                [counts[r["id"]] for r in rs], [0, 0.5, 0.95, 1]
                            ).tolist(),
                        }
                        for d, rs in groups.items()
                    },
                }
        for arm, step in [("fp_control", 25000), ("ternary_matched", 27000)]:
            actual = read(run / "train" / arm / "result.json")
            assert all(
                summary["replayed_exposure"][str(step)][k] == actual[k]
                for k in ["sample_hash", "augmentation_hash"]
            )
        probe = out / "warmstart-probe/train/fp_control/result.json"
        if probe.exists():
            actual = read(probe)
            assert all(
                summary["replayed_exposure"][str(actual["steps"])][k] == actual[k]
                for k in ["sample_hash", "augmentation_hash"]
            )
            summary["warmstart_probe_matches_original_sampling_and_augmentation"] = True
        save(out / "static.json", summary)
        print(
            json.dumps(
                {
                    "saved": str(out / "static.json"),
                    "provenance": summary["provenance_verified"],
                    "sampler_hashes_verified": True,
                }
            )
        )
        return

    import torch
    from torch.nn import functional as F
    from common import ALPHABET, decode, encode
    from pilot4_train import configured_model, evaluate

    torch.set_num_threads(4)
    selected = [by_id[i] for i in subsets["gate"] + subsets["monitor"]]
    selected += [r for r in rows if r["split"] == "development"]
    features = {r["id"]: torch.from_numpy(np.load(r["features"])) for r in selected}
    monitor = [by_id[i] for i in subsets["monitor"]]
    probes = [r for d in domains for r in [x for x in monitor if x["domain"] == d][:4]]
    result = {}
    checkpoints = [
        ("gate", "fp_control", "latest"),
        ("gate", "ternary_matched", "latest"),
        ("gate", "ternary_twostage", "latest"),
        ("train", "fp_control", "best"),
        ("train", "fp_control", "latest"),
        ("train", "ternary_matched", "latest"),
    ]
    for mode, arm, name in checkpoints:
        key = f"{mode}/{arm}/{name}"
        saved = torch.load(
            run / mode / arm / (name + ".pt"),
            mmap=True,
            map_location="cpu",
            weights_only=False,
        )
        model = configured_model(saved["config"])(
            saved["config"]["model"], saved["arm"]["precision"]
        )
        model.load_state_dict(saved["model"], strict=True)
        model.cuda().eval()
        result[key] = {"step": saved["step"], "evaluations": {}, "probes": []}
        del saved
        for group in ["gate", "monitor", "development"]:
            group_rows = (
                [by_id[i] for i in subsets[group]]
                if group in subsets
                else [r for r in rows if r["split"] == group]
            )
            result[key]["evaluations"][group] = evaluate(
                model, group_rows, features, result[key]["step"]
            )
        with torch.inference_mode():
            for row in probes:
                layer_stats, handles = [], []

                def hook(module, inputs, output):
                    z = output[:, 50:].float()
                    # Frame 50 onwards excludes the initial boundary in original clips.
                    layer_stats.append(
                        {
                            "mean_channel_temporal_std": float(z.std(dim=1).mean()),
                            "mean_vector_norm": float(z.norm(dim=-1).mean()),
                        }
                    )

                for block in model.blocks:
                    handles.append(block.register_forward_hook(hook))
                x = features[row["id"]][None].cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = model(x).float()
                for handle in handles:
                    handle.remove()
                target = torch.tensor(encode(row["text"]), device="cuda")
                loss = F.ctc_loss(
                    z.log_softmax(-1).transpose(0, 1),
                    target,
                    torch.tensor([z.shape[1]]),
                    torch.tensor([len(target)]),
                )
                probs = z.softmax(-1)
                alternatives = {}
                for label, signal in [
                    ("time_reversed", x.flip(-1)),
                    ("constant_mean", x.mean(-1, keepdim=True).expand_as(x)),
                ]:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        altered = model(signal).float()
                    alternatives[label] = {
                        "prediction": decode(altered[0].argmax(-1).tolist()),
                        "mean_abs_probability_difference_after_1s": float(
                            (probs[:, 50:] - altered.softmax(-1)[:, 50:]).abs().mean()
                        ),
                    }
                full = model(x).float()
                padded = F.pad(x, (0, 137))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pz = model(padded).float()[:, : z.shape[1]]
                # Blank-prior sweep is diagnostic only, not a tuned production decoder.
                prior_sweep = {}
                for penalty in [0, 1, 2, 4]:
                    adjusted = z.clone()
                    adjusted[..., 0] -= penalty
                    prior_sweep[str(penalty)] = decode(adjusted[0].argmax(-1).tolist())
                result[key]["probes"].append(
                    {
                        "id": row["id"],
                        "domain": row["domain"],
                        "reference": row["text"],
                        "prediction": decode(z[0].argmax(-1).tolist()),
                        "fp32_prediction": decode(full[0].argmax(-1).tolist()),
                        "padded_prediction": decode(pz[0].argmax(-1).tolist()),
                        "padded_probability_max_difference": float(
                            (pz.softmax(-1) - probs).abs().max()
                        ),
                        "ctc_loss": float(loss),
                        "mean_blank_probability": float(probs[..., 0].mean()),
                        "top_mean_symbols": [
                            (ALPHABET[int(i)], float(v))
                            for v, i in zip(*probs[0, 50:].mean(0).topk(5))
                        ],
                        "layer_statistics_after_1s": layer_stats,
                        "altered_acoustics": alternatives,
                        "blank_penalty_predictions": prior_sweep,
                    }
                )
        save(out / "checkpoints.json", result)
        print(
            json.dumps(
                {
                    "checkpoint": key,
                    "step": result[key]["step"],
                    "metrics": {
                        g: {d: round(e["metrics"][d]["cer"], 4) for d in domains}
                        for g, e in result[key]["evaluations"].items()
                    },
                    "first_probe_layer_std": [
                        round(l["mean_channel_temporal_std"], 5)
                        for l in result[key]["probes"][0]["layer_statistics_after_1s"]
                    ],
                }
            ),
            flush=True,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
