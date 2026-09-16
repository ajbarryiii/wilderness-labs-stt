"""Offline transcript targets; calibration reliability is fixed before training."""

import argparse, json, time
import numpy as np
import soundfile as sf
import torch
from common import (
    ART,
    read_manifest,
    save,
    norm,
    scores,
    distance,
    digit_string,
    encode,
    digest,
    medical_terms,
)


def main(kind, run, smoke=False):
    torch.set_num_threads(4)
    rows = (
        read_manifest()["rows"]
        if not smoke
        else json.loads((run / "smoke-input.json").read_text())
    )
    rows = sorted(
        rows,
        key=lambda r: (
            {"calibration": 0, "development": 1, "train": 2}[r["split"]],
            r["id"],
        ),
    )
    output = run / ("teacher-" + kind + ".json")
    pred = json.loads(output.read_text()) if output.exists() else {}
    if kind == "omi":
        import nemo.collections.asr as asr
        from omegaconf import OmegaConf, open_dict

        m = (
            asr.models.ASRModel.restore_from(
                str(ART / "teachers/omi/omimedstt-v1.nemo"), map_location="cpu"
            )
            .eval()
            .cuda()
        )
        conf = m.cfg.decoding
        with open_dict(conf):
            conf.greedy.use_cuda_graph_decoder = False
        m.change_decoding_strategy(conf)
        from nemo.collections.common.parts.adapter_modules import LinearAdapter

        calls = [0]
        adapters = [
            (n, x) for n, x in m.named_modules() if isinstance(x, LinearAdapter)
        ]
        assert adapters, "Medical adapter modules missing"

        def counted(*args):
            calls[0] += 1

        hooks = [x.register_forward_hook(counted) for _, x in adapters]
        save(
            run / "teacher-omi-runtime.json",
            dict(
                parameters=sum(p.numel() for p in m.parameters()),
                enabled_adapters=m.get_enabled_adapters(),
                linear_adapters=len(adapters),
                decoding=OmegaConf.to_container(conf, resolve=True),
                autocast="bfloat16",
            ),
        )

        def infer(batch):
            audio = [sf.read(r["audio"], dtype="float32")[0] for r in batch]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = m.transcribe(
                    audio, batch_size=len(batch), num_workers=0, verbose=False
                )
            assert calls[0] > 0, "Medical adapters did not execute"
            save(
                run / "teacher-omi-adapter-check.json",
                dict(forward_calls=calls[0], modules=len(adapters), executed=True),
            )
            if isinstance(out, tuple):
                out = out[0]
            return [getattr(x, "text", x) for x in out]

        size = 8
    else:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        p = WhisperProcessor.from_pretrained(
            ART / "teachers/whisper", local_files_only=True
        )
        m = (
            WhisperForConditionalGeneration.from_pretrained(
                ART / "teachers/whisper", local_files_only=True, dtype=torch.bfloat16
            )
            .eval()
            .cuda()
        )
        save(
            run / "teacher-whisper-runtime.json",
            dict(
                parameters=sum(p.numel() for p in m.parameters()),
                dtype="bfloat16",
                decoding=dict(do_sample=False, num_beams=1, max_new_tokens=256),
            ),
        )

        def infer(batch):
            audio = [sf.read(r["audio"], dtype="float32")[0] for r in batch]
            inputs = p(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True,
            )
            with torch.inference_mode():
                out = m.generate(
                    inputs.input_features.cuda().to(torch.bfloat16),
                    attention_mask=inputs.attention_mask.cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=256,
                )
            return p.batch_decode(out, skip_special_tokens=True)

        size = 8
    todo = [r for r in rows if r["id"] not in pred]
    started = time.time()
    for i in range(0, len(todo), size):
        batch = todo[i : i + size]
        hyp = infer(batch)
        if len(hyp) != len(batch):
            raise RuntimeError("Teacher batch length mismatch")
        for r, h in zip(batch, hyp):
            pred[r["id"]] = dict(
                text=norm(h), raw=h, split=r["split"], domain=r["domain"]
            )
        save(output, pred)
        if i % (size * 10) == 0:
            print(
                json.dumps(
                    dict(
                        teacher=kind,
                        done=len(pred),
                        total=len(rows),
                        elapsed=time.time() - started,
                    )
                ),
                flush=True,
            )
    save(
        run / ("teacher-" + kind + "-summary.json"),
        {
            s: {
                d: scores(
                    [
                        (r["text"], pred[r["id"]]["text"])
                        for r in rows
                        if r["split"] == s and r["domain"] == d
                    ]
                )
                for d in sorted({r["domain"] for r in rows})
            }
            for s in ["calibration", "development"]
        },
    )


def build_targets(run):
    rows = read_manifest()["rows"]
    teachers = {
        k: json.loads((run / ("teacher-" + k + ".json")).read_text())
        for k in ["omi", "whisper"]
    }
    policy = {}
    targets = {}
    accept = {k: 0 for k in teachers}
    for k, pred in teachers.items():
        summary = {}
        for split in ["calibration", "development"]:
            summary[split] = {}
            for domain in sorted({r["domain"] for r in rows}):
                rr = [
                    r
                    for r in rows
                    if r["split"] == split and r["domain"] == domain and r["id"] in pred
                ]
                pairs = [(r["text"], pred[r["id"]]["text"]) for r in rr]
                s = scores(pairs)
                if domain == "digits":
                    s["digit_sequence_accuracy"] = (
                        sum(digit_string(r) == digit_string(h) for r, h in pairs)
                        / len(pairs)
                        if pairs
                        else None
                    )
                if domain == "medical_symptoms":
                    s["medical_terms"] = medical_terms(pairs)
                summary[split][domain] = s
        save(run / ("teacher-" + k + "-summary.json"), summary)
    for domain in sorted({r["domain"] for r in rows}):
        cal = [r for r in rows if r["domain"] == domain and r["split"] == "calibration"]
        assert cal
        assert all(r["id"] in teachers[k] for r in cal for k in teachers), (
            "Incomplete calibration coverage"
        )
        errors = {}
        for k in teachers:
            if domain == "digits":
                e = sum(
                    digit_string(teachers[k][r["id"]]["text"])
                    != digit_string(r["text"])
                    for r in cal
                ) / len(cal)
            else:
                e = scores([(r["text"], teachers[k][r["id"]]["text"]) for r in cal])[
                    "wer"
                ]
            errors[k] = e
        reliability = {k: 1 / (0.05 + min(e, 3)) for k, e in errors.items()}
        total = sum(reliability.values())
        policy[domain] = dict(
            calibration_errors=errors,
            weights={k: v / total for k, v in reliability.items()},
        )
    for r in rows:
        if r["split"] != "train" or not all(r["id"] in teachers[k] for k in teachers):
            continue
        cand = {}
        for k in teachers:
            h = teachers[k][r["id"]]["text"]
            y = encode(h)
            g = encode(r["text"])
            feasible = bool(y) and (r["frames"] + 1) // 2 >= len(y) + sum(
                a == b for a, b in zip(y, y[1:])
            )
            e = distance(r["text"].split(), h.split()) / max(1, len(r["text"].split()))
            allowed = feasible and (
                digit_string(h) == digit_string(r["text"])
                if r["domain"] == "digits"
                else e <= 0.35
            )
            if allowed:
                cand[k] = h
                accept[k] += 1
        targets[r["id"]] = cand
    coverage = {
        d: sum(r["id"] in targets for r in rows if r["domain"] == d) for d in policy
    }
    assert all(n >= 30 for n in coverage.values()), (
        f"Insufficient common teacher coverage: {coverage}"
    )
    different = {
        k: sum(
            k in targets.get(r["id"], {}) and targets[r["id"]][k] != r["text"]
            for r in rows
        )
        for k in teachers
    }
    save(
        run / "targets.json",
        dict(
            common_training_coverage=coverage,
            targets_different_from_ground_truth=different,
            policy=policy,
            targets=targets,
            acceptance_counts=accept,
            teacher_files={
                k: digest(run / ("teacher-" + k + ".json")) for k in teachers
            },
            filter="GT-relative train-only WER <=0.35; exact controlled digit sequence; finite CTC alignment. Public labels are not newly audited.",
        ),
    )
    print(json.dumps(dict(policy=policy, acceptance=accept)), flush=True)


if __name__ == "__main__":
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["omi", "whisper", "combine"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    build_targets(a.run) if a.kind == "combine" else main(a.kind, a.run, a.smoke)
