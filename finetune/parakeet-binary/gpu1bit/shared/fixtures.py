"""Real-data fixtures (plans/gpu-kernel.md A3): MB1 weights per layer, and every quantized layer's
actual input captured from MB1's FP32 NeMo reconstruction (binary_export.load_export) on the 15
frozen dev fixture utterances (workloads.json "fixture_utterances": 5 dev corpora x 3, 10, 30 s
targets). Dev audio only. Plus the per-layer errors of M1's two production kernels on these inputs.

Layout under /mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/H/fixtures/:
  inputs/<n>.safetensors   one per utterance: tensor "<module name>" = the input as NeMo passed it,
                           stored contiguous; pointwise_conv1 inputs (NeMo passes x.transpose(1, 2)
                           of a contiguous [B, T, K]) are stored as that [B, T, K] storage and
                           rebuilt as the same transposed view by `utterance_inputs`;
                           pointwise_conv2 inputs are contiguous [B, K, T]; Q/K/V share one input
                           (stored once under linear_q, aliases recorded)
  index.json               per utterance: id, corpus, duration, samples, T, per-layer key / view /
                           shape / M / alias, file SHA-256, capture settings and source hashes
  m1_errors.json           per utterance x layer: api.errors of r_bf16x3 and r_expanded_int8x3 vs
                           the FP64 reference, keyed to index.json's hashes

    timeout 600 gpu1bit/shared/python -m gpu1bit.shared.fixtures capture
    timeout 600 gpu1bit/shared/python -m gpu1bit.shared.fixtures m1-errors [--utterances 0-4]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from gpu1bit.shared import api, env

ROOT = env.ARTIFACTS / "fixtures"
INDEX = ROOT / "index.json"
M1_ERRORS = ROOT / "m1_errors.json"


# --- MB1 weights ---------------------------------------------------------------------------------

_MANIFEST = None


def mb1_manifest() -> dict:
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = json.loads((env.MB1_EXPORT / "manifest.json").read_text())
        if _MANIFEST["sha256"] != env.MB1_SHA256:
            raise ValueError("MB1 manifest hash changed")
    return _MANIFEST


def layer_names() -> list[str]:
    return list(mb1_manifest()["quantized_layers"])


def mb1_layer(name: str) -> dict:
    """{'codes' uint8 [N, ceil(K/8)], 'scale' FP32 [N], 'bias' None, 'K', 'N', 'kind'} on the CPU."""
    from safetensors import safe_open
    info = mb1_manifest()["quantized_layers"][name]
    n, k = info["shape"]
    with safe_open(env.MB1_EXPORT / mb1_manifest()["file"], framework="pt") as f:
        codes, scale = f.get_tensor(f"{name}.codes"), f.get_tensor(f"{name}.scale")
        bias = f.get_tensor(f"{name}.bias") if info["bias"] else None
    return {"name": name, "codes": codes, "scale": scale, "bias": bias, "N": n, "K": k,
            "kind": "conv" if info["kind"] == "pointwise_conv1d" else "linear"}


def mb1_layers(names=None) -> dict[str, dict]:
    from safetensors import safe_open
    manifest = mb1_manifest()
    out = {}
    with safe_open(env.MB1_EXPORT / manifest["file"], framework="pt") as f:
        for name in (names or layer_names()):
            info = manifest["quantized_layers"][name]
            n, k = info["shape"]
            out[name] = {"name": name, "codes": f.get_tensor(f"{name}.codes"), "scale": f.get_tensor(f"{name}.scale"),
                         "bias": f.get_tensor(f"{name}.bias") if info["bias"] else None, "N": n, "K": k,
                         "kind": "conv" if info["kind"] == "pointwise_conv1d" else "linear"}
    return out


# --- captured inputs -----------------------------------------------------------------------------

def index() -> dict:
    return json.loads(INDEX.read_text())


def utterance_inputs(i: int, device="cuda", verify: bool = True) -> dict[str, torch.Tensor]:
    """{module name: input tensor exactly as NeMo passed it (same shape and stride pattern)}."""
    from safetensors.torch import load_file
    entry = index()["utterances"][i]
    path = ROOT / "inputs" / entry["file"]
    if verify and env.sha256_file(path) != entry["sha256"]:
        raise ValueError(f"{path} does not match index.json")
    stored = load_file(path, device=str(device))
    out = {}
    for name, rec in entry["layers"].items():
        t = stored[rec["key"]]
        out[name] = t.transpose(1, 2) if rec["view"] == "transpose12" else t
        if list(out[name].shape) != rec["shape"]:
            raise AssertionError(f"{name}: rebuilt shape {tuple(out[name].shape)} != {rec['shape']}")
    return out


@torch.no_grad()
def capture() -> None:
    import binary_export
    import evaluate as ev
    from safetensors.torch import save_file
    from gpu1bit.shared import workloads
    env.require_shared_wrapper()
    doc = workloads.load()
    (ROOT / "inputs").mkdir(parents=True, exist_ok=True)
    names = layer_names()
    started = time.time()
    with env.gpu_lock("H:fixture-capture") as lock:
        model = binary_export.load_export(env.MB1_EXPORT, "cuda")
        captured: dict[str, torch.Tensor] = {}
        hooks = []
        for name in names:
            def hook(module, args, name=name):
                captured[name] = args[0]
            hooks.append(model.get_submodule(name).register_forward_pre_hook(hook))
        utterances = []
        with ev.strict_fp32("cuda"), ev.inference_settings(model):
            for n, rec in enumerate(doc["fixture_utterances"]):
                env.check_hold(lock)
                captured.clear()
                audio, length = ev.audio_batch([rec])
                if audio.shape[1] != rec["samples"]:
                    raise AssertionError(f"{rec['id']}: {audio.shape[1]} samples, workloads.json says {rec['samples']}")
                f, fl = model.preprocessor(input_signal=audio.cuda(), length=length.cuda())
                e, el = model.encoder(audio_signal=f, length=fl)
                torch.cuda.synchronize()
                if e.shape[-1] != rec["M"]:
                    raise AssertionError(f"{rec['id']}: encoder frames {e.shape[-1]} != workloads M {rec['M']}")
                tensors, layers, by_ptr = {}, {}, {}
                for name in names:
                    x = captured[name]
                    kind = "conv" if name.endswith(("pointwise_conv1", "pointwise_conv2")) else "linear"
                    view = "none"
                    base = x
                    if kind == "conv" and not x.is_contiguous():
                        # pointwise_conv1: x.transpose(1, 2) of a contiguous [B, T, K]
                        if not x.transpose(1, 2).is_contiguous():
                            raise AssertionError(f"{name}: conv input is neither contiguous nor a transposed view")
                        base, view = x.transpose(1, 2), "transpose12"
                    elif not x.is_contiguous():
                        raise AssertionError(f"{name}: non-contiguous linear input")
                    ptr = (base.data_ptr(), tuple(base.shape))
                    alias = by_ptr.get(ptr)
                    if alias is None:
                        by_ptr[ptr] = name
                        tensors[name] = base.detach().contiguous().clone()
                    k = x.shape[1] if kind == "conv" else x.shape[-1]
                    layers[name] = {"key": alias or name, "alias_of": alias, "view": view, "kind": kind,
                                    "shape": list(x.shape), "M": x.numel() // k}
                file = f"{n:02d}.safetensors"
                path = ROOT / "inputs" / file
                save_file({k: v.cpu() for k, v in tensors.items()}, path)
                utterances.append({**{k: rec[k] for k in ("id", "corpus", "duration", "samples", "M", "target_seconds")},
                                   "file": file, "sha256": env.sha256_file(path), "bytes": path.stat().st_size,
                                   "encoded_length": int(el[0]), "layers": layers})
                print(json.dumps({"n": n, "id": rec["id"], "M": rec["M"], "stored": len(tensors),
                                  "MB": round(path.stat().st_size / 1e6, 1)}), flush=True)
        for h in hooks:
            h.remove()
    out = {"about": __doc__.split("\n\n")[0], "export_sha256": env.MB1_SHA256,
           "model": "binary_export.load_export(MB1, cuda) FP32, evaluate.strict_fp32 (TF32 off), inference_settings (dither 0, pad_to 0)",
           "workloads_sha256": doc["content_sha256"], "wrapper": env.wrapper(), "torch": torch.__version__,
           "source_sha256": env.source_hashes(env.SHARED), "seconds": time.time() - started, "utterances": utterances}
    INDEX.write_text(json.dumps(out, indent=1) + "\n")


# --- M1 errors -----------------------------------------------------------------------------------

def m1_errors(selection: list[int]) -> None:
    """api.errors of M1's R-bf16x3 and R-expanded INT8x3 for each selected utterance and layer."""
    from gpu1bit.shared import m1
    env.require_shared_wrapper()
    idx = index()
    existing = json.loads(M1_ERRORS.read_text()) if M1_ERRORS.exists() else {"utterances": {}}
    if existing.get("index_sha256") not in (None, env.sha256_file(INDEX)):
        raise ValueError("m1_errors.json was computed for a different index.json")
    weights = mb1_layers()
    with env.gpu_lock(f"H:m1-errors {selection[0]}-{selection[-1]}") as lock, torch.inference_mode():
        torch.backends.cuda.matmul.allow_tf32 = False
        ops = {name: m1.LayerOps(w["codes"], w["scale"], w["bias"], w["K"]) for name, w in weights.items()}
        for i in selection:
            entry = idx["utterances"][i]
            inputs = utterance_inputs(i)
            rows = {}
            for name, w in weights.items():
                env.check_hold(lock)
                x, kind = inputs[name], weights[name]["kind"]
                ref = api.reference(kind, x, w["codes"], w["scale"], w["bias"], w["K"])
                rows[name] = {op: api.errors(kind, x, f(x, kind == "conv"), ref, w["scale"])
                              for op, f in ops[name].ops().items()}
            existing["utterances"][entry["file"]] = {"id": entry["id"], "sha256": entry["sha256"], "layers": rows}
            worst = max(max(r[o]["relative_frobenius"] for o in r) for r in rows.values())
            print(json.dumps({"n": i, "id": entry["id"], "layers": len(rows), "max_rel_frob": worst,
                              "held_s": round(env.check_hold(lock), 1)}), flush=True)
            existing.update(index_sha256=env.sha256_file(INDEX), wrapper=env.wrapper(),
                            source_sha256=env.source_hashes(env.SHARED),
                            kernels={"r_bf16x3": "inference.kernels.matmul(mode='bf16x3') on derived PackedLinear",
                                     "r_expanded_int8x3": "inference.encoder4.ExpandedProjection('triton_col_tuned')(x, mod, conv, 3)"})
            tmp = M1_ERRORS.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(existing) + "\n")
            os.replace(tmp, M1_ERRORS)


def m1_thresholds() -> dict[str, dict[str, dict[str, float]]]:
    """{utterance file: {layer: api.numerics_thresholds}} from m1_errors.json (hash-checked)."""
    doc = json.loads(M1_ERRORS.read_text())
    if doc["index_sha256"] != env.sha256_file(INDEX):
        raise ValueError("m1_errors.json is stale for index.json")
    return {f: {name: api.numerics_thresholds(errs) for name, errs in u["layers"].items()}
            for f, u in doc["utterances"].items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["capture", "m1-errors"])
    p.add_argument("--utterances", default="0-14", help="inclusive range a-b of index positions")
    args = p.parse_args()
    if args.action == "capture":
        if INDEX.exists():
            raise SystemExit(f"{INDEX} exists; fixtures are captured once")
        capture()
    else:
        a, b = (int(v) for v in args.utterances.split("-"))
        m1_errors(list(range(a, b + 1)))


if __name__ == "__main__":
    main()
