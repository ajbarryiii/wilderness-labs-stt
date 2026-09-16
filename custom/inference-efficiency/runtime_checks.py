"""Check graph fidelity, logical sizes, and optional tiny packed CUDA replay.

Run with ./inference-efficiency/python inference-efficiency/runtime_checks.py
Add --cuda for a small integration/capture check (not an energy benchmark).
"""
import argparse
import json
import tempfile

import torch

from paths import ROOT, save, storage
from runtime_model import ReplayWhisper, WhisperConfig


def hf_config(config):
    from transformers import WhisperConfig as HFConfig
    return HFConfig(vocab_size=config.n_vocab, num_mel_bins=config.n_mels,
                    d_model=config.n_text_state, encoder_layers=config.n_audio_layer,
                    decoder_layers=config.n_text_layer,
                    encoder_attention_heads=config.n_audio_head,
                    decoder_attention_heads=config.n_text_head,
                    encoder_ffn_dim=4 * config.n_audio_state,
                    decoder_ffn_dim=4 * config.n_text_state,
                    max_source_positions=config.n_audio_ctx,
                    max_target_positions=config.n_text_ctx,
                    pad_token_id=0, bos_token_id=config.sot_id,
                    eos_token_id=config.sot_id - 1,
                    decoder_start_token_id=config.sot_id)


def check_reference():
    from transformers import WhisperForConditionalGeneration
    torch.manual_seed(71)
    config = WhisperConfig.tiny_smoke()
    reference = WhisperForConditionalGeneration(hf_config(config)).eval()
    candidate = ReplayWhisper(config, implementation="dense", device="cpu", dtype=torch.float32)
    root = storage()
    (root / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root / "tmp", prefix="graph-parity-") as directory:
        reference.save_pretrained(directory)
        candidate.load_hf_safetensors(directory)
        mel = torch.randn(1, config.n_mels, 2 * config.n_audio_ctx)
        tokens = [3, 4, 5, 6]
        with torch.inference_mode():
            encoded = candidate.encode(mel)
            expected_encoder = reference.model.encoder(mel).last_hidden_state
            torch.testing.assert_close(encoded, expected_encoder, rtol=2e-5, atol=2e-6)
            actual = candidate.decode(encoded, tokens)
            ids = torch.tensor([[config.sot_id, config.no_timestamps_id] + tokens[:-1]])
            expected = reference(input_features=mel, decoder_input_ids=ids).logits[:, -1:]
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            # Reusing the static buffers must not leak a preceding clip's KV cache.
            changed = candidate.run(mel * 0.3, tokens)
            repeated = candidate.run(mel, tokens)
            torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
            if torch.equal(changed, actual):
                raise AssertionError("Decoder output ignores changed encoder memory")
            result = {"encoder_max_abs_error": float((encoded - expected_encoder).abs().max()),
                      "decoder_max_abs_error": float((actual - expected).abs().max()),
                      "cache_reset": "passed"}
    medium = WhisperConfig.medium_en()
    with torch.device("meta"):
        reference_medium = WhisperForConditionalGeneration(hf_config(medium))
    learned = sum(p.numel() for p in reference_medium.parameters() if p.requires_grad)
    all_entries = sum(p.numel() for p in reference_medium.parameters())
    if learned != medium.parameter_count() or learned != 762_320_896:
        raise AssertionError(f"Whisper medium.en parameter mismatch: {learned}")
    result.update(medium_learned_parameters=learned,
                  medium_entries_including_fixed_encoder_positions=all_entries)
    return result


def check_cuda():
    results = {}
    torch.manual_seed(73)
    for distribution in ("binary", "ternary"):
        config = WhisperConfig.tiny_smoke()
        dense = ReplayWhisper(config, distribution, "dense", device="cuda")
        packed = ReplayWhisper(config, distribution, "packed", device="cuda")
        mel = torch.randn(1, config.n_mels, 2 * config.n_audio_ctx,
                          device="cuda", dtype=torch.float16)
        tokens = [3, 4, 5, 6]
        encoded_dense, encoded_packed = dense.encode(mel), packed.encode(mel)
        torch.testing.assert_close(encoded_dense, encoded_packed, rtol=0.04, atol=0.03)
        dense_logits = dense.decode(encoded_dense, tokens)
        packed_logits = packed.decode(encoded_packed, tokens)
        torch.testing.assert_close(dense_logits, packed_logits, rtol=0.04, atol=0.03)
        # Both paths receive the same graph optimization in the actual benchmark.
        for model, expected in ((dense, dense_logits), (packed, packed_logits)):
            model.capture(mel, tokens)
            actual = model.replay_graph(mel)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.003)
        results[distribution] = {
            "encoder_max_abs_error": float((encoded_dense - encoded_packed).abs().max()),
            "decoder_max_abs_error": float((dense_logits - packed_logits).abs().max()),
            "health": packed.validate_outputs(mel, tokens),
            "dense_and_packed_graph_replay": "passed"}
        del dense, packed
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    result = {"reference_graph": check_reference()}
    if args.cuda:
        result["tiny_cuda_integration"] = check_cuda()
    save(ROOT / "runtime-verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
