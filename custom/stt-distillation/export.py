"""Portable weight export; ternary packing is storage, not a fast inference kernel."""

import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file, load_file
from common import save, digest, ALPHABET
from model import Model, Weight


def export_model(model, folder, step, packed=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tensors = {}
    layout = {}
    n = 0
    with torch.no_grad():
        for name, layer in model.named_modules():
            if not isinstance(layer, Weight):
                continue
            w = layer.effective().float().cpu().contiguous()
            n += w.numel()
            if packed and layer.precision == "ternary":
                scale = layer.weight.detach().abs().mean().float().cpu().clamp_min(1e-8)
                codes = (
                    (w / scale)
                    .round()
                    .to(torch.int8)
                    .add(1)
                    .numpy()
                    .reshape(-1)
                    .astype(np.uint8)
                )
                padded = np.pad(codes, (0, (-len(codes)) % 4)).reshape(-1, 4)
                payload = (
                    padded[:, 0]
                    | (padded[:, 1] << 2)
                    | (padded[:, 2] << 4)
                    | (padded[:, 3] << 6)
                )
                tensors[name + ".codes"] = torch.from_numpy(payload.copy())
                tensors[name + ".scale"] = scale.reshape(1)
                layout[name + ".weight"] = dict(
                    shape=list(w.shape), count=w.numel(), kind="ternary_2bit"
                )
            else:
                tensors[name + ".weight"] = w
                layout[name + ".weight"] = dict(shape=list(w.shape), kind="fp32")
    tmp = folder / "weights.safetensors.tmp"
    save_file(tensors, str(tmp))
    tmp.replace(folder / "weights.safetensors")
    meta = dict(
        schema=1,
        model=model.cfg,
        precision=model.precision,
        step=step,
        parameters=n,
        alphabet=ALPHABET,
        layout=layout,
        weights_sha256=digest(folder / "weights.safetensors"),
        bytes=(folder / "weights.safetensors").stat().st_size,
        execution="Expanded floating-point reference operations; no packed inference kernel.",
    )
    save(folder / "export.json", meta)
    return meta


def load_export(folder, device="cuda"):
    folder = Path(folder)
    meta = json.loads((folder / "export.json").read_text())
    assert meta["alphabet"] == ALPHABET
    assert digest(folder / "weights.safetensors") == meta["weights_sha256"]
    ts = load_file(str(folder / "weights.safetensors"))
    state = {}
    for name, layout in meta["layout"].items():
        if layout["kind"] == "fp32":
            state[name] = ts[name]
        else:
            prefix = name[:-7]
            p = ts[prefix + ".codes"].numpy()
            scale = ts[prefix + ".scale"].item()
            c = np.stack([(p >> shift) & 3 for shift in [0, 2, 4, 6]], axis=1).reshape(
                -1
            )[: layout["count"]]
            assert np.all(c < 3)
            state[name] = torch.from_numpy((c.astype(np.float32) - 1) * scale).reshape(
                layout["shape"]
            )
    m = Model(meta["model"], "fp")
    m.load_state_dict(state, strict=True)
    return m.to(device).eval(), meta
