"""NeMo FP32 golden outputs of a full-depth seeded random model, and the NeMo helpers the parity tests share.

NixOS only (NeMo env, CPU). The NeMo modules are instantiated from the pinned model config in
weight_stats.json (EncDecRNNTBPEModel.from_config_dict, optionally with fewer encoder layers) and
filled from a randomweights tensor stream by reference.load_weights (ternary modules dequantized
in place, one dense FP32 copy). Decoding is NeMo's own GreedyBatchedTDTInfer (the class the
config's strategy greedy_batch selects) with preserve_alignments and include_duration, so every
joint evaluation (blank steps included) and its log-probabilities are recorded by NeMo itself.
The per-step prediction-net input, LSTM state and raw joint logits are then recomputed by
driving NeMo's own decoder.predict / joint.project_* / joint.joint_net through reference.run_steps
with NeMo's decisions; the result must reproduce NeMo's labels, durations, token timestamps and
log-probabilities, else generation fails.

Output (default /mnt/hd/wilderness-labs-stt/parakeet-ios/golden/seed0/): one <clip>.npz per
development clip with audio, features + length, subsampling output + length, every layer's
output [layers, 1, T, d_model], encoder output [1, d_model, T] + length, the decode trace
(reference.Trace arrays, prefix trace_), per-step raw logits, NeMo log-probabilities, LSTM h and
c [steps, layers, hidden], and NeMo's tokens, timestamps and durations; plus meta.json (seed,
model digest, clips, versions).

compare() checks the reference against such outputs (DESIGN.md gate 1); tests/test_reference_nemo.py
applies it to live NeMo outputs of the real weights, tests/test_reference_golden.py to the files.

Run through the memory-capped wrapper (about 6 GB):
  ../heavy ios-wp1-golden --mem-max 12G --runtime 40min --wait -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
      MKL_NUM_THREADS=4 <repo>/finetune/parakeet-ternary/python <repo>/finetune/parakeet-ternary/ios/golden.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tests"))
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import randomweights as rw  # noqa: E402
import reference  # noqa: E402


class NemoModules(nn.Module):
    """NeMo preprocessor/encoder/decoder/joint under their EncDecRNNTBPEModel names (for load_weights).

    Built from the model config (from_config) or sharing an existing model's modules (wrap)."""

    def __init__(self, cfg: reference.Config, modules: dict[str, nn.Module]) -> None:
        super().__init__()
        self.cfg = cfg
        for name in ("preprocessor", "encoder", "decoder", "joint"):
            setattr(self, name, modules[name])
        self.preprocessor.featurizer.dither = 0.0
        self.preprocessor.featurizer.pad_to = cfg.pad_to
        self.eval()

    @classmethod
    def from_config(cls, model_config: dict, num_layers: int | None = None) -> NemoModules:
        from nemo.collections.asr.models import EncDecRNNTBPEModel
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(model_config)
        if num_layers is not None:
            cfg.encoder.n_layers = num_layers
        return cls(reference.Config.from_model_config(model_config, num_layers),
                   {name: EncDecRNNTBPEModel.from_config_dict(cfg[name])
                    for name in ("preprocessor", "encoder", "decoder", "joint")})

    @classmethod
    def wrap(cls, model: nn.Module, model_config: dict) -> NemoModules:
        return cls(reference.Config.from_model_config(model_config),
                   {name: getattr(model, name) for name in ("preprocessor", "encoder", "decoder", "joint")})


def tensor_source(stats: dict, seed: int, layers: int | None = None) -> tuple[dict, str]:
    """All tensors of a random model as torch tensors (codes int8, the rest FP32) and its manifest digest."""
    tensors, hashes = {}, {}
    for name, value in rw.generate(stats, seed, layers):
        hashes[name] = {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": rw.sha256_bytes(value.tobytes())}
        tensors[name] = torch.from_numpy(value)
    return tensors, rw._manifest(stats, seed, layers, hashes)["digest"]


def greedy_infer(nemo: NemoModules, frame_looping: bool = False):
    """NeMo's TDT greedy decoder for this config on nemo's decoder and joint, recording every step."""
    from nemo.collections.asr.parts.submodules import rnnt_greedy_decoding as g

    cfg = nemo.cfg
    if frame_looping:
        return g.GreedyTDTInfer(decoder_model=nemo.decoder, joint_model=nemo.joint, blank_index=cfg.blank,
                                durations=list(cfg.durations), max_symbols_per_step=cfg.max_symbols,
                                include_duration=True)
    return g.GreedyBatchedTDTInfer(decoder_model=nemo.decoder, joint_model=nemo.joint, blank_index=cfg.blank,
                                   durations=list(cfg.durations), max_symbols_per_step=cfg.max_symbols,
                                   preserve_alignments=True, include_duration=True, use_cuda_graph_decoder=False)


def hyp_steps(hyp) -> tuple[list[int], torch.Tensor]:
    """Labels [S] and NeMo log-probabilities [S, outputs] of every step, in order, from a hypothesis' alignments."""
    labels, logits = [], []
    for frame in hyp.alignments:
        for step_logits, label in frame:
            labels.append(int(label))
            logits.append(step_logits.float())
    return labels, torch.stack(logits) if logits else torch.empty(0)


def as_list(x) -> list[int]:
    return [int(v) for v in (x.tolist() if torch.is_tensor(x) else x)]


@torch.no_grad()
def nemo_driven(nemo: NemoModules, encoder_output: torch.Tensor, length: int, hyp
                ) -> tuple[reference.Trace, reference.StepOutputs, torch.Tensor]:
    """NeMo's decisions replayed through NeMo's own prediction and joint networks via reference.run_steps.

    Returns the trace, the per-step outputs (raw logits, LSTM state, argmax) and NeMo's recorded
    log-probabilities; RuntimeError unless the replay reproduces NeMo's hypothesis exactly."""
    cfg, decoder, joint = nemo.cfg, nemo.decoder, nemo.joint
    labels, log_probs = hyp_steps(hyp)
    durations = [cfg.durations[int(i)] for i in log_probs[:, -len(cfg.durations):].argmax(dim=-1)] if labels else []

    def decide(i: int, t: int, tok: int, dur: int) -> tuple[int, int]:
        if i >= len(labels):
            raise RuntimeError("NeMo recorded fewer steps than the frame rules need")
        return labels[i], durations[i]

    def predict(token: int, state):
        g, state = decoder.predict(torch.full((1, 1), token, dtype=torch.long), state, add_sos=False, batch_size=1)
        return joint.project_prednet(g), state

    def joint_step(f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return joint.joint_net(f.unsqueeze(2) + g.unsqueeze(1)).squeeze(1).squeeze(1)

    enc_proj = joint.project_encoder(encoder_output.transpose(1, 2))
    trace, steps = reference.run_steps(cfg, enc_proj, length, decoder.initialize_state(enc_proj), predict,
                                       joint_step, decide, record=True)
    checks = {
        "step count": len(trace) == len(labels),
        "labels": trace.token == labels,
        "tokens": trace.tokens == as_list(hyp.y_sequence),
        "timestamps": trace.timestamps == as_list(hyp.timestamp),
        "durations": trace.token_durations == as_list(hyp.token_duration),
        "argmax tokens": as_list(steps.argmax_token) == labels,
        "argmax durations": as_list(steps.argmax_duration) == durations,
        "log-probabilities": len(trace) == len(labels) and bool(
            (steps.logits.log_softmax(dim=-1) - log_probs).abs().max() <= 1e-5),
    }
    if not all(checks.values()):
        raise RuntimeError(f"NeMo-driven replay does not reproduce NeMo's decode: {checks}")
    return trace, steps, log_probs


@torch.no_grad()
def run_clip(nemo: NemoModules, audio: np.ndarray) -> dict[str, np.ndarray]:
    """Every golden output of one clip (batch of one)."""
    hidden: dict = {"layers": []}
    hooks = [nemo.encoder.pre_encode.register_forward_hook(lambda m, i, o: hidden.__setitem__("pre_encode", o))]
    hooks += [layer.register_forward_hook(lambda m, i, o: hidden["layers"].append(o)) for layer in nemo.encoder.layers]
    try:
        signal = torch.from_numpy(audio)[None]
        length = torch.tensor([len(audio)])
        features, feat_len = nemo.preprocessor(input_signal=signal, length=length)
        encoded, enc_len = nemo.encoder(audio_signal=features, length=feat_len)
    finally:
        for hook in hooks:
            hook.remove()
    (hyp,) = greedy_infer(nemo)(encoder_output=encoded, encoded_lengths=enc_len)[0]
    trace, steps, log_probs = nemo_driven(nemo, encoded, int(enc_len[0]), hyp)
    pre, pre_len = hidden["pre_encode"]
    out = {"audio": audio, "features": features, "feature_length": feat_len, "pre_encode": pre,
           "pre_encode_length": pre_len, "layers": torch.stack(hidden["layers"]), "encoder": encoded,
           "encoder_length": enc_len, "step_logits": steps.logits, "step_log_probs": log_probs,
           "step_h": steps.h, "step_c": steps.c, "nemo_tokens": torch.as_tensor(as_list(hyp.y_sequence)),
           "nemo_timestamps": torch.as_tensor(as_list(hyp.timestamp)),
           "nemo_durations": torch.as_tensor(as_list(hyp.token_duration))}
    out = {k: (v.numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in out.items()}
    out.update({f"trace_{k}": v for k, v in trace.to_arrays().items()})
    return out


def compare(ref: reference.ParakeetReference, g: dict, noise_seed: int = 0) -> tuple[dict, list[str]]:
    """The reference on one clip against golden outputs g (run_clip's dict or a loaded npz): DESIGN.md gate 1.

    Each compared output must be finite with rel <= REL_CEILING and abs <= ABS_CEILING (common.errors):
    features (reference front end on the golden audio), subsampling output, every layer, encoder
    output (encoder on the golden features), and on the golden trace replayed through the reference
    the token logits (incl. blank) and duration logits separately and LSTM h and c separately.
    Decisions must be identical: the reference's own greedy trace equals the golden trace in every
    field, and the replay's argmax equals the golden decisions. Sensitivity: 1% white noise (RMS 1%
    of the clip's RMS, torch.Generator seed noise_seed) added to the audio must move the
    reference's encoder output by rel >= 1e-2 and >= 100x the measured encoder parity error.
    Returns (metrics, failures)."""
    t = {k: torch.from_numpy(np.asarray(v)) for k, v in g.items()}
    cfg = ref.cfg
    split = cfg.vocab_size + 1
    metrics: dict = {}
    failures: list[str] = []

    def gate(name: str, a: torch.Tensor, r: torch.Tensor, store: bool = True) -> tuple[float, float]:
        rel, ab = common.errors(a, r)
        if store:
            metrics[name] = {"rel": rel, "abs": ab}
        if not common.finite(a):
            failures.append(f"{name}: not finite")
        if not rel <= common.REL_CEILING or not ab <= common.ABS_CEILING:
            failures.append(f"{name}: rel {rel:.3g}, abs {ab:.3g}")
        return rel, ab

    with torch.no_grad():
        audio, length = t["audio"][None], torch.tensor([t["audio"].numel()])
        feats, flen = ref.preprocessor(audio, length)
        gate("features", feats, t["features"])
        enc, elen, hidden = ref.encoder(t["features"], t["feature_length"], return_hidden=True)
        lengths_equal = (torch.equal(flen, t["feature_length"]) and torch.equal(elen, t["encoder_length"])
                         and torch.equal(t["pre_encode_length"].long(), elen))
        if not lengths_equal:
            failures.append("lengths differ")
        gate("pre_encode", hidden["pre_encode"], t["pre_encode"])
        layer_errors = [gate(f"layer{i}", x, t["layers"][i], store=False) for i, x in enumerate(hidden["layers"])]
        metrics["layers_max"] = {"rel": max(e[0] for e in layer_errors), "abs": max(e[1] for e in layer_errors),
                                 "count": len(layer_errors)}
        enc_rel, _ = gate("encoder", enc, t["encoder"])
        e2e, _ = ref.encoder(feats, flen)
        metrics["encoder_end_to_end"] = dict(zip(("rel", "abs"), common.errors(e2e, t["encoder"])))
        golden_trace = reference.Trace.from_arrays(g, "trace_")
        trace = reference.greedy_decode(ref, enc, elen)[0]
        differing = [k for k in reference.TRACE_FIELDS if getattr(trace, k) != getattr(golden_trace, k)]
        if differing or trace.num_frames != golden_trace.num_frames:
            failures.append(f"greedy trace differs in {differing}")
        if (trace.tokens != t["nemo_tokens"].tolist() or trace.token_durations != t["nemo_durations"].tolist()
                or trace.timestamps != t["nemo_timestamps"].tolist()):
            failures.append("greedy tokens, durations or timestamps differ from NeMo's")
        steps = reference.replay(ref, enc, int(elen[0]), golden_trace)
        gate("token_logits", steps.logits[:, :split], t["step_logits"][:, :split])
        gate("duration_logits", steps.logits[:, split:], t["step_logits"][:, split:])
        gate("lstm_h", steps.h, t["step_h"])
        gate("lstm_c", steps.c, t["step_c"])
        if steps.argmax_token.tolist() != golden_trace.token or steps.argmax_duration.tolist() != golden_trace.duration:
            failures.append("replayed argmax differs from the golden decisions")
        gen = torch.Generator().manual_seed(noise_seed)
        noise = torch.randn(audio.shape, generator=gen) * (0.01 * float(audio.pow(2).mean().sqrt()))
        perturbed, _ = ref(audio + noise, length)
        sensitivity, _ = common.errors(perturbed, t["encoder"])
        metrics["sensitivity_rel"] = sensitivity
        metrics["sensitivity_over_parity"] = sensitivity / max(enc_rel, 1e-30)
        if not (sensitivity >= 1e-2 and sensitivity >= 100 * enc_rel):
            failures.append(f"sensitivity {sensitivity:.3g} vs encoder parity {enc_rel:.3g}")
    metrics.update(steps=len(trace), tokens=len(trace.tokens), forced_advances=sum(trace.forced_advance),
                   decisions_identical=not differing, encoder_frames=int(elen[0]))
    return metrics, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=common.GOLDEN_SEED)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    out_dir = args.out or common.artifacts_dir() / "golden" / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    stats = rw.load_stats()
    tensors, digest = tensor_source(stats, args.seed)
    nemo = NemoModules.from_config(stats["model_config"])
    load = reference.load_weights(nemo, tensors)
    del tensors
    clips = common.dev_clips()
    meta = {"seed": args.seed, "model_digest": digest, "weight_stats_sha256": stats["_sha256"], "load": load,
            "layers": nemo.cfg.n_layers, "torch": torch.__version__, "clips": []}
    import nemo as nemo_pkg
    meta["nemo"] = nemo_pkg.__version__
    for record in clips:
        audio = common.load_audio(record["audio_filepath"])
        out = run_clip(nemo, audio)
        name = record["id"].split(":")[-1]
        np.savez(out_dir / f"{name}.npz", **out)
        meta["clips"].append({"id": record["id"], "file": f"{name}.npz", "audio_sha256": common.sha256_array(audio),
                              "samples": len(audio), "encoder_frames": int(out["encoder_length"][0]),
                              "steps": int(len(out["trace_frame"])), "tokens": int(len(out["nemo_tokens"])),
                              "forced_advances": int(out["trace_forced_advance"].sum())})
        common.report("golden_clip", **meta["clips"][-1])
    meta["seconds"] = round(time.time() - start, 1)
    meta["peak_rss_mb"] = common.peak_rss_mb()
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    common.report("golden", out=str(out_dir), digest=digest, seconds=meta["seconds"], peak_rss_mb=meta["peak_rss_mb"])


if __name__ == "__main__":
    main()
