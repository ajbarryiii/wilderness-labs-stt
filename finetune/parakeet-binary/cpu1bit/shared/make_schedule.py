"""Freeze the projection schedule (plans/cpu-benchmark-mac.md M1) into shared/schedule.json.

Derived by reading the measured x86 LocalFlow snapshot's encoder (lf-asr/src/encoder.rs, sha256
checked below; line numbers refer to that file):
- run_layers (434-463): the position cache covers T after warmup (`cache.ensure`, 443), so every
  layer uses LayerPositions::Cached and `linear_pos` is NOT projected per utterance (523-534 run only
  for LayerPositions::Project). It is listed here as cached / not replayed / not timed.
- run_layer (475-620), per layer in this order:
    feed_forward(ff1) 490 -> 651 linear1 (Slot::Model, quantize) ; 658 linear2 (Slot::Ff, quantize)
    496-504 linear_q (Slot::Model, quantize) ; 505-513 linear_k (reuse) ; 514-522 linear_v (reuse)
    558-566 linear_out (Slot::Model, quantize)
    575-583 pointwise_conv1 (Slot::Model, quantize; N = 2 * D_MODEL)
    600-608 pointwise_conv2 (Slot::Model, quantize)
    feed_forward(ff2) 614 -> 651 linear1 ; 658 linear2
- Projector::run (112-144): quantize unless reuse, then the INT8 GEMM with the epilogue; Q, K and V
  are three GEMMs sharing one quantization (no fused QKV).
Inputs are FP32 [rows][K] row-major with rows = T (encoder frames); outputs [rows][N] row-major,
output index o = NeMo output channel (pointwise_conv1: GLU value half o < 1024, gate half o >= 1024).

Captures (M5): one FP32 tensor per quantization, i.e. per (layer, quant id), taken from the FP32 NeMo
reconstruction as the input of the first module that uses it (forward pre-hook), stored [T][K]
row-major. NeMo passes pointwise_conv1 a transposed view of a contiguous [1, T, K] (stored as that
[T, K] storage) and pointwise_conv2 a contiguous [1, K, T] (stored transposed to [T, K]).

    finetune/parakeet-binary/python cpu1bit/shared/make_schedule.py [--check]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "schedule.json"
SNAPSHOT = Path("/mnt/hd/wilderness-labs-stt/parakeet-ternary/cpu-bitplanes/20261008T121546/snapshot")
ENCODER_RS = SNAPSHOT / "crates/lf-asr/src/encoder.rs"
ENCODER_RS_SHA256 = "2fd1c0fbfa8ae03afc2e179bad9a2e52d0ecceb1f8908ee9e5350c8695a114e3"  # snapshot-sha256.json of the run
# The unpatched LocalFlow file (metadata.json localflow_source_sha256); prepare.py only added the
# env-gated bitplanes engine (default path: gemm_i8), so the schedule is the same.
LOCALFLOW_ENCODER_RS_SHA256 = "ee0391c259aab1116863c0edc7c8db05235211b78eda369177a6385357d4b96a"
LAYERS, D_MODEL, D_FF = 24, 1024, 4096

# (quant id, K, NeMo module whose forward-pre-hook input is captured, NeMo input layout, what it is)
QUANTS = [
    ("ff1_in", D_MODEL, "feed_forward1.linear1", "[1, T, K] contiguous", "LN_ff1(x)"),
    ("ff1_mid", D_FF, "feed_forward1.linear2", "[1, T, K] contiguous", "silu(ff1.linear1(.))"),
    ("att_in", D_MODEL, "self_attn.linear_q", "[1, T, K] contiguous", "LN_att(x), also K and V input"),
    ("att_out", D_MODEL, "self_attn.linear_out", "[1, T, K] contiguous", "attention context"),
    ("conv_in", D_MODEL, "conv.pointwise_conv1", "transpose(1, 2) of contiguous [1, T, K]", "LN_conv(x)"),
    ("conv_mid", D_MODEL, "conv.pointwise_conv2", "[1, K, T] contiguous (stored transposed)", "silu(bn(depthwise(glu(.))))"),
    ("ff2_in", D_MODEL, "feed_forward2.linear1", "[1, T, K] contiguous", "LN_ff2(x)"),
    ("ff2_mid", D_FF, "feed_forward2.linear2", "[1, T, K] contiguous", "silu(ff2.linear1(.))"),
]

# (module, N, K, rows, slot, reuse, quant id or None, replayed, encoder.rs lines)
CALLS = [
    ("feed_forward1.linear1", D_FF, D_MODEL, "T", "model", False, "ff1_in", True, "490,651"),
    ("feed_forward1.linear2", D_MODEL, D_FF, "T", "ff", False, "ff1_mid", True, "490,658"),
    ("self_attn.linear_q", D_MODEL, D_MODEL, "T", "model", False, "att_in", True, "496-504"),
    ("self_attn.linear_k", D_MODEL, D_MODEL, "T", "model", True, "att_in", True, "505-513"),
    ("self_attn.linear_v", D_MODEL, D_MODEL, "T", "model", True, "att_in", True, "514-522"),
    ("self_attn.linear_pos", D_MODEL, D_MODEL, "2T-1", "pos", None, None, False, "443,523-534 (cached)"),
    ("self_attn.linear_out", D_MODEL, D_MODEL, "T", "model", False, "att_out", True, "558-566"),
    ("conv.pointwise_conv1", 2 * D_MODEL, D_MODEL, "T", "model", False, "conv_in", True, "575-583"),
    ("conv.pointwise_conv2", D_MODEL, D_MODEL, "T", "model", False, "conv_mid", True, "600-608"),
    ("feed_forward2.linear1", D_FF, D_MODEL, "T", "model", False, "ff2_in", True, "614,651"),
    ("feed_forward2.linear2", D_MODEL, D_FF, "T", "ff", False, "ff2_mid", True, "614,658"),
]


def build() -> dict:
    digest = hashlib.sha256(ENCODER_RS.read_bytes()).hexdigest()
    if digest != ENCODER_RS_SHA256:
        raise SystemExit(f"{ENCODER_RS}: sha256 {digest} != measured snapshot {ENCODER_RS_SHA256}")
    calls = []
    for layer in range(LAYERS):
        for module, n, k, rows, slot, reuse, quant, replayed, lines in CALLS:
            calls.append({
                "index": len(calls), "layer": layer, "module": f"encoder.layers.{layer}.{module}", "short": module,
                "n": n, "k": k, "rows": rows, "slot": slot, "reuse": reuse,
                "quant": None if quant is None else f"{layer}.{quant}", "replayed": replayed, "timed": replayed,
                "encoder_rs_lines": lines,
            })
    replayed = [c for c in calls if c["replayed"]]
    quantizations = [c for c in replayed if not c["reuse"]]
    captures = [{"name": f"{layer}.{q}", "layer": layer, "quant": q, "k": k, "rows": "T",
                 "nemo_module": f"encoder.layers.{layer}.{module}", "nemo_input_layout": layout, "what": what,
                 "stored_layout": "[T][K] f32 little-endian row-major"}
                for layer in range(LAYERS) for q, k, module, layout, what in QUANTS]
    # Trace invariants (also asserted by the Rust trace contracts).
    assert len(replayed) == 240 and len(quantizations) == 192 and len(captures) == 192
    assert [c["quant"] for c in quantizations] == [c["name"] for c in captures]
    for c in replayed:
        if c["reuse"]:
            prev = [p for p in replayed if p["index"] < c["index"] and p["slot"] == c["slot"]][-1]
            assert prev["quant"] == c["quant"] and prev["layer"] == c["layer"], c
    return {
        "about": __doc__.split("\n\n")[0],
        "derivation": __doc__.split("\n\n")[1],
        "source": {"file": str(ENCODER_RS), "sha256": ENCODER_RS_SHA256, "localflow_sha256": LOCALFLOW_ENCODER_RS_SHA256, "localflow_revision": "3bbbb0c9c5789efbf6c5aec4ca429a4296ea7e5f"},
        "dims": {"layers": LAYERS, "d_model": D_MODEL, "d_ff": D_FF},
        "rows": {"T": "encoder frames of the utterance", "2T-1": "relative positions (linear_pos, cached)"},
        "counts": {"calls": len(calls), "replayed_calls": len(replayed), "quantizations": len(quantizations),
                   "reused_quantizations": len(replayed) - len(quantizations), "captures": len(captures),
                   "cached_not_replayed": len(calls) - len(replayed),
                   "capture_floats_per_frame": sum(c["k"] for c in captures),
                   "macs_per_frame_per_component": sum(c["n"] * c["k"] for c in replayed)},
        "departures": ["linear_pos is cached as in LocalFlow after warmup; it is not part of the replay (M1).",
                       "Inputs are the FP32 NeMo trajectory, not LocalFlow's residual-INT8 encoder trajectory (M2)."],
        "calls": calls,
        "captures": captures,
    }


def main() -> None:
    doc = build()
    body = json.dumps(doc, indent=1, sort_keys=True)
    doc_hash = hashlib.sha256(body.encode()).hexdigest()
    text = json.dumps({**doc, "content_sha256": doc_hash}, indent=1, sort_keys=True) + "\n"
    if OUT.exists():
        old = json.loads(OUT.read_text())
        if old["content_sha256"] != doc_hash:
            raise SystemExit(f"{OUT} is frozen and differs from a fresh derivation ({old['content_sha256']} != {doc_hash})")
        print(f"{OUT} verified: {doc_hash}")
        return
    if "--check" in sys.argv:
        raise SystemExit(f"{OUT} missing")
    OUT.write_text(text)
    print(f"wrote {OUT}: {doc_hash}")


if __name__ == "__main__":
    main()
