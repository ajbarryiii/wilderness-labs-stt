"""Capture every replayed projection input (plans/cpu-benchmark-mac.md "Inputs", M1, M5) from the FP32
NeMo reconstruction of M1 and of MB1 on the NixOS GPU, for the frozen dev clips (shared/clips.json).

Per (model, group, utterance) one directory under
/mnt/hd/wilderness-labs-stt/parakeet-binary/cpu1bit/captures/<model>/<group>/<nn>/ with
  captures.f32  the 192 quantization inputs of shared/schedule.json in capture order, each [T][K]
                f32 little-endian row-major (pointwise_conv2's NeMo [1, K, T] input stored transposed;
                pointwise_conv1's transposed view stored as its contiguous [1, T, K] storage)
  index.json    clip record, T (LocalFlow frames = NeMo encoded length; rows past T of NeMo's padded
                tensor are dropped), per-capture {name, k, offset (floats), rows}, file SHA-256, the
                trace checks (linear_q/k/v received the very same tensor; linear_pos saw [1, 2T-1, K]),
                model export identity, settings and source hashes
and <model>/<group>/index.json listing them. Models: binary_export.load_export(MB1) and
export.load_export(M1) on cuda, FP32, evaluate.strict_fp32 (TF32 off), inference_settings (dither 0,
pad_to 0), batch 1 at the exact length. Dev audio only (clips.json is built from dev manifests).

    job start cpu1bit-capture --needs gpu:shared --vram 6G -- \
        finetune/parakeet-binary/python cpu1bit/shared/capture.py [--models M1,MB1] [--groups timed,exactness]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import clips  # noqa: E402

ROOT = Path("/mnt/hd/wilderness-labs-stt/parakeet-binary/cpu1bit/captures")
EXPORTS = {
    "MB1": Path("/mnt/hd/wilderness-labs-stt/parakeet-binary/runs/main-binB-m1-lr2e-4/export"),
    "M1": Path("/mnt/hd/wilderness-labs-stt/parakeet-ternary/runs/main-M1-P2-lr5e-4/export"),
}
MB1_SHA256 = "9873d2f1a53127c673d0201db98cd3ba859ba432f334fef55e9e24fd057b9651"  # gpu1bit/shared/env.py


def schedule() -> dict:
    doc = json.loads((HERE / "schedule.json").read_text())
    body = {k: v for k, v in doc.items() if k != "content_sha256"}
    if hashlib.sha256(json.dumps(body, indent=1, sort_keys=True).encode()).hexdigest() != doc["content_sha256"]:
        raise ValueError("schedule.json content hash mismatch")
    return doc


def sha256_file(p: Path) -> str:
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_model(name: str):
    export = EXPORTS[name]
    manifest = json.loads((export / "manifest.json").read_text())
    if name == "MB1":
        import binary_export
        if manifest["sha256"] != MB1_SHA256:
            raise ValueError("MB1 export changed")
        model = binary_export.load_export(export, "cuda")
    else:
        import export as ternary_export
        model = ternary_export.load_export(export, "cuda")
    ident = {"export": str(export), "format": manifest["format"], "export_safetensors_sha256": manifest["sha256"],
             "manifest_sha256": sha256_file(export / "manifest.json")}
    return model, ident


@torch.no_grad()
def capture(model_name: str, group: str, sched: dict, clip_doc: dict) -> dict:
    import evaluate as ev
    frac = os.environ.get("AGENT_JOBS_VRAM_FRACTION")
    if frac:
        torch.cuda.set_per_process_memory_fraction(float(frac))
    out_root = ROOT / model_name / group
    out_root.mkdir(parents=True, exist_ok=True)
    model, ident = load_model(model_name)
    caps = sched["captures"]
    seen: dict[str, torch.Tensor] = {}
    hooks = []
    hooked = sorted({c["nemo_module"] for c in caps}
                    | {f"encoder.layers.{l}.self_attn.{m}" for l in range(24) for m in ("linear_k", "linear_v", "linear_pos")})
    for name in hooked:
        def hook(module, args, name=name):
            seen[name] = args[0]
        hooks.append(model.get_submodule(name).register_forward_pre_hook(hook))
    entries = []
    try:
        with ev.strict_fp32("cuda"), ev.inference_settings(model):
            for n, rec in enumerate(clip_doc[group]):
                seen.clear()
                audio, length = ev.audio_batch([rec])
                if audio.shape[1] != rec["samples"]:
                    raise AssertionError(f"{rec['id']}: {audio.shape[1]} samples != {rec['samples']}")
                f, fl = model.preprocessor(input_signal=audio.cuda(), length=length.cuda())
                e, el = model.encoder(audio_signal=f, length=fl)
                torch.cuda.synchronize()
                m_tensor, t = e.shape[-1], int(el[0])
                if m_tensor != rec["M"] or t != rec["T"]:
                    raise AssertionError(f"{rec['id']}: encoder tensor frames {m_tensor} / length {t} != clips.json M {rec['M']} / T {rec['T']}")
                d = out_root / f"{n:02d}"
                d.mkdir(parents=True, exist_ok=True)
                path = d / "captures.f32"
                tmp = path.with_suffix(f".tmp{os.getpid()}")
                layout, offset, trace = [], 0, {"qkv_same_tensor": True, "pos_rows": set()}
                with open(tmp, "wb") as fh:
                    for c in caps:
                        x = seen[c["nemo_module"]]
                        k = c["k"]
                        if c["nemo_module"].endswith("pointwise_conv2"):
                            if list(x.shape) != [1, k, m_tensor] or not x.is_contiguous():
                                raise AssertionError(f"{c['name']}: pointwise_conv2 input {tuple(x.shape)}")
                            x2 = x[0].transpose(0, 1)[:t]
                        elif c["nemo_module"].endswith("pointwise_conv1"):
                            if list(x.shape) != [1, k, m_tensor] or not x.transpose(1, 2).is_contiguous():
                                raise AssertionError(f"{c['name']}: pointwise_conv1 input {tuple(x.shape)}")
                            x2 = x.transpose(1, 2)[0, :t]
                        else:
                            if list(x.shape) != [1, m_tensor, k] or not x.is_contiguous():
                                raise AssertionError(f"{c['name']}: input {tuple(x.shape)} contiguous={x.is_contiguous()}")
                            x2 = x[0, :t]
                        if c["quant"] == "att_in":
                            layer = c["layer"]
                            for other in ("linear_k", "linear_v"):
                                y = seen[f"encoder.layers.{layer}.self_attn.{other}"]
                                if y.data_ptr() != x.data_ptr() or y.shape != x.shape or y.stride() != x.stride():
                                    trace["qkv_same_tensor"] = False
                            trace["pos_rows"].add(tuple(seen[f"encoder.layers.{layer}.self_attn.linear_pos"].shape))
                        arr = x2.detach().to(torch.float32).contiguous().cpu()
                        if not torch.isfinite(arr).all():
                            raise AssertionError(f"{c['name']}: non-finite capture")
                        fh.write(arr.numpy().tobytes(order="C"))
                        layout.append({"name": c["name"], "k": k, "rows": t, "offset": offset})
                        offset += t * k
                os.replace(tmp, path)
                if not trace["qkv_same_tensor"]:
                    raise AssertionError(f"{rec['id']}: linear_k / linear_v did not receive linear_q's tensor")
                pos_rows = sorted(trace["pos_rows"])
                if pos_rows != [(1, 2 * m_tensor - 1, 1024)]:
                    raise AssertionError(f"{rec['id']}: linear_pos input shapes {pos_rows}")
                size = path.stat().st_size
                if size != offset * 4 or size != rec["capture_bytes"]:
                    raise AssertionError(f"{rec['id']}: {size} bytes != {offset * 4} / clips.json {rec['capture_bytes']}")
                entry = {"n": n, "dir": f"{n:02d}", "clip": rec, "T": t, "nemo_tensor_frames": m_tensor,
                         "padded_nemo_rows_dropped": m_tensor - t, "floats": offset, "bytes": size,
                         "sha256": sha256_file(path), "captures": layout,
                         "trace": {"qkv_same_tensor": True, "linear_pos_input_shape": [1, 2 * m_tensor - 1, 1024]}}
                (d / "index.json").write_text(json.dumps({**entry, "model": model_name, **ident}, indent=1) + "\n")
                entries.append(entry)
                print(json.dumps({"model": model_name, "group": group, "n": n, "id": rec["id"], "T": t,
                                  "MB": round(size / 1e6, 1)}), flush=True)
    finally:
        for h in hooks:
            h.remove()
        del model
        torch.cuda.empty_cache()
    doc = {"about": __doc__.split("\n\n")[0], "model": model_name, "group": group, **ident,
           "clips_content_sha256": clip_doc["content_sha256"], "schedule_content_sha256": sched["content_sha256"],
           "settings": "FP32 cuda, evaluate.strict_fp32 (TF32 off), inference_settings (dither 0, pad_to 0), batch 1",
           "torch": torch.__version__, "source_sha256": {p.name: sha256_file(p) for p in (HERE / "capture.py", HERE / "clips.py")},
           "utterances": entries, "bytes": sum(e["bytes"] for e in entries), "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out_root / "index.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", default="MB1,M1")
    p.add_argument("--groups", default="timed,exactness")
    a = p.parse_args()
    if not os.path.ismount("/mnt/hd"):
        raise SystemExit("/mnt/hd is not mounted")
    sched, clip_doc = schedule(), clips.load()
    for model_name in a.models.split(","):
        for group in a.groups.split(","):
            if (ROOT / model_name / group / "index.json").exists():
                print(f"{model_name}/{group}: already captured, skipping", flush=True)
                continue
            doc = capture(model_name, group, sched, clip_doc)
            print(json.dumps({"model": model_name, "group": group, "GB": round(doc["bytes"] / 1e9, 3)}), flush=True)


if __name__ == "__main__":
    main()
