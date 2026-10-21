"""Write gpu1bit/shared/shapes.json: MB1's 264 quantized layers and the benchmark clips' frame counts.

CPU only (run with CUDA_VISIBLE_DEVICES=""; no GPU, so no lock). The model is rebuilt from MB1's
export by binary_export.load_export; one eager encoder forward per benchmark clip length records,
through forward pre-hooks, every quantized layer's actual input shape and strides. The input is
zeros of the clip's exact sample count: frame counts depend only on the sample count, and no test
audio is decoded (the three LibriSpeech test-clean clips M1's recorded runs used are located as
compare_v2.py does, `min(records, key=|duration - target|)`, and only their headers are read).

    CUDA_VISIBLE_DEVICES= finetune/parakeet-binary/python -m gpu1bit.shared.make_shapes
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
from pathlib import Path

import torch

from gpu1bit.shared import env

HERE = Path(__file__).resolve().parent
OUT = HERE / "shapes.json"
CLIP_TARGETS = (3, 10, 30)  # compare_v2.py --durations default; 30 resolves to the 30.04 s clip


def benchmark_clips() -> list[dict]:
    import soundfile as sf
    import paths
    records = [json.loads(line) for line in (paths.MANIFESTS / "test_librispeech_clean.jsonl").read_text().splitlines()]
    clips = []
    for target in CLIP_TARGETS:
        rec = min(records, key=lambda r: abs(r["duration"] - target))
        info = sf.info(rec["audio_filepath"])
        if info.samplerate != 16000 or info.channels != 1:
            raise ValueError(f"unexpected audio format for {rec['id']}")
        clips.append({"target_seconds": target, "id": rec["id"], "duration": rec["duration"],
                      "samples": int(info.frames), "manifest": "test_librispeech_clean.jsonl"})
    return clips


@torch.no_grad()
def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise SystemExit('run with CUDA_VISIBLE_DEVICES="" (CPU only, no GPU lock taken)')
    import binary_export
    import evaluate as ev
    torch.set_num_threads(16)
    manifest = json.loads((env.MB1_EXPORT / "manifest.json").read_text())
    layers = manifest["quantized_layers"]
    model = binary_export.load_export(env.MB1_EXPORT, "cpu")
    records: dict[str, list] = collections.defaultdict(list)
    hooks = []
    for name in layers:
        def hook(module, args, name=name):
            x = args[0]
            records[name].append({"shape": list(x.shape), "stride": list(x.stride()),
                                  "contiguous": x.is_contiguous()})
        hooks.append(model.get_submodule(name).register_forward_pre_hook(hook))
    clips = benchmark_clips()
    with ev.inference_settings(model):
        for clip in clips:
            audio = torch.zeros(1, clip["samples"])
            length = torch.tensor([clip["samples"]])
            features, feature_length = model.preprocessor(input_signal=audio, length=length)
            encoded, encoded_length = model.encoder(audio_signal=features, length=feature_length)
            # M = rows the projections see = encoder tensor frames. NeMo's mel tensor has one frame
            # more than the valid feature length, so the encoder tensor can exceed encoded_length
            # by one (10 s: 126 tensor frames, 125 valid). Both are recorded.
            clip["feature_tensor_frames"] = int(features.shape[-1])
            clip["feature_length"] = int(feature_length[0])
            clip["encoder_frames_T"] = int(encoded.shape[-1])
            clip["encoded_length"] = int(encoded_length[0])
    for h in hooks:
        h.remove()
    t_values = [c["encoder_frames_T"] for c in clips]
    out_layers = []
    for name, layer in layers.items():
        n, k = layer["shape"]
        kind = "conv" if layer["kind"] == "pointwise_conv1d" else "linear"
        seen = records[name]
        if len(seen) != len(clips):
            raise AssertionError(f"{name}: {len(seen)} calls, expected {len(clips)}")
        rows = []
        for clip, call in zip(clips, seen):
            t = clip["encoder_frames_T"]
            if kind == "conv":
                b, kk, tt = call["shape"]
                m = b * tt
            else:
                kk = call["shape"][-1]
                m = 1
                for d in call["shape"][:-1]:
                    m *= d
            if kk != k:
                raise AssertionError(f"{name}: K {kk} != {k}")
            rows.append(m)
        rule = "2T-1" if name.endswith("self_attn.linear_pos") else "T"
        expected = [2 * t - 1 if rule == "2T-1" else t for t in t_values]
        if rows != expected:
            raise AssertionError(f"{name}: M {rows} != {rule} {expected}")
        out_layers.append({"name": name, "kind": kind, "N": n, "K": k, "bias": layer["bias"],
                           "M_rule": rule, "input_shape_3s": seen[0]["shape"],
                           "input_stride_3s": seen[0]["stride"]})
    shapes = collections.Counter((l["kind"], l["N"], l["K"]) for l in out_layers)
    distinct = []
    for (kind, n, k), count in sorted(shapes.items()):
        members = [l for l in out_layers if (l["kind"], l["N"], l["K"]) == (kind, n, k)]
        rules = sorted({l["M_rule"] for l in members})
        suffixes = sorted({".".join(l["name"].split(".")[3:]) for l in members})
        distinct.append({"kind": kind, "N": n, "K": k, "count": count, "M_rules": rules, "modules": suffixes,
                         "M_per_clip": {f"{c['duration']}s": sorted({2 * c['encoder_frames_T'] - 1 if r == "2T-1"
                                                                     else c['encoder_frames_T'] for r in rules})
                                        for c in clips},
                         "input_stride_pattern_3s": sorted({json.dumps(l["input_stride_3s"]) for l in members})})
    code_bytes = sum(l["N"] * ((l["K"] + 7) // 8) for l in out_layers)
    if code_bytes != manifest["bytes"]["packed_code_bytes"]:
        raise AssertionError(f"code bytes {code_bytes} != manifest {manifest['bytes']['packed_code_bytes']}")
    doc = {
        "about": ("MB1's 264 quantized encoder layers and the batch-1 M (rows = B*T frames) of the 3, 10 and "
                  "30.04 s LibriSpeech test-clean clips used by M1's recorded runs (compare_v2.py, "
                  "EXPERIMENTS4.md; cuda_experiments.py --rows 38 126 376). Written by "
                  "gpu1bit/shared/make_shapes.py from one eager CPU forward per clip length."),
        "export": str(env.MB1_EXPORT), "export_sha256": manifest["sha256"],
        "packed_code_bytes": code_bytes, "scale_bytes": manifest["bytes"]["scale_bytes"],
        "budget": {"factor": 1.10, "code_budget_bytes_total": (code_bytes * 11) // 10,
                   "definition": "see gpu1bit/shared/api.py weight_budget_bytes and plans/gpu-kernel.md A5"},
        "clips": clips,
        "M_rules": {"T": "every layer except linear_pos: one row per encoder tensor frame (B*T; T = encoder output tensor frames, which can exceed encoded_length by one)",
                    "2T-1": ("self_attn.linear_pos projects the relative positional embedding [1, 2T-1, 1024] "
                             "(batch 1 regardless of B); M1's optimized runtimes compute it eagerly at warm-up "
                             "and replay a cached result inside CUDA graphs (transforms.py position=True), so "
                             "it is not in the timed graph")},
        "distinct_shapes": distinct,
        "m1_runtime_calls_per_encoder_forward": {
            "about": ("Projection calls M1's optimized/expanded/hybrid encoder (encoder_transform qkv=True, "
                      "silu=True, norm_quant=True, position=True, layer_fusions) makes per forward inside "
                      "the graph; Q/K/V run as one fused [3072, 1024] projection."),
            "calls": [
                {"what": "self_attn fused QKV (norm_self_att fused into quantization)", "kind": "linear", "N": 3072, "K": 1024, "count": 24, "M": "T"},
                {"what": "self_attn.linear_out", "kind": "linear", "N": 1024, "K": 1024, "count": 24, "M": "T"},
                {"what": "feed_forward{1,2}.linear1 (+norm fused, SiLU epilogue)", "kind": "linear", "N": 4096, "K": 1024, "count": 48, "M": "T"},
                {"what": "feed_forward{1,2}.linear2", "kind": "linear", "N": 1024, "K": 4096, "count": 48, "M": "T"},
                {"what": "conv.pointwise_conv1 (+norm_conv fused)", "kind": "conv", "N": 2048, "K": 1024, "count": 24, "M": "T"},
                {"what": "conv.pointwise_conv2 (SiLU on input fused)", "kind": "conv", "N": 1024, "K": 1024, "count": 24, "M": "T"},
                {"what": "self_attn.linear_pos (cached outside the graph)", "kind": "linear", "N": 1024, "K": 1024, "count": 24, "M": "2T-1"},
            ]},
        "layers": out_layers,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"clips": clips, "distinct": [(d["kind"], d["N"], d["K"], d["count"], d["M_per_clip"]) for d in distinct]}, indent=1))


if __name__ == "__main__":
    main()
