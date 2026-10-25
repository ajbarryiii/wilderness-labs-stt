"""The decoder-step proxy (decoder-only plan "Results": emitted tokens and decoder iterations per second of
audio, and the mean predicted duration).

greedy_steps re-runs NeMo's greedy TDT label-looping algorithm (GreedyBatchedTDTLabelLoopingComputer.torch_impl,
without fusion models or confidence) on a batch of encoder outputs and counts, per utterance:
    tokens       emitted (non-blank) labels
    iterations   joint evaluations while the utterance is active (the outer step and every inner blank step):
                 the sequential decoder steps TDT takes
    durations    the sum of the predicted durations (the argmax of the duration head, before NeMo forces a blank's
                 duration 0 to 1) over those iterations; mean predicted duration = durations / iterations
It uses the model's own decoding computer (durations, blank index, max_symbols, decoder, joint). proxy_set decodes
a manifest's records in evaluate.py's batches (descending duration, strict FP32, transcribe() settings) and checks,
batch by batch, that these tokens equal NeMo's greedy hypotheses on the same encoder output (agreement reported;
tests/test_decoder_only.py also checks the counts against NeMo's own all-step record on a small model).
"""
from __future__ import annotations

import torch


def _computer(model):
    computer = getattr(getattr(getattr(model, "decoding", None), "decoding", None), "decoding_computer", None)
    if computer is None:
        raise RuntimeError("the model has no NeMo label-looping greedy decoder")
    return computer


@torch.no_grad()
def greedy_steps(model, encoder_output: torch.Tensor, encoder_lengths: torch.Tensor) -> list[dict]:
    """encoder_output [B, D, T] (model.forward's layout). Per utterance: tokens, iterations, durations."""
    comp = _computer(model)
    decoder, joint = comp.decoder, comp.joint
    blank, max_symbols = comp._blank_index, comp.max_symbols
    enc = joint.project_encoder(encoder_output.transpose(1, 2))
    batch, _, _ = enc.shape
    device = enc.device
    model_durations = comp.durations.to(device)
    nd = model_durations.shape[0]
    rows = torch.arange(batch, device=device)
    length = encoder_lengths.to(device)
    last_t = torch.clamp(length - 1, min=0)
    time = torch.zeros_like(length)
    safe = torch.minimum(time, last_t)
    current = torch.zeros_like(time)
    active = time < length
    labels = torch.full_like(rows, blank)
    state = decoder.initialize_state(enc)
    dec_out, state, *_ = decoder.predict(labels.unsqueeze(1), state, add_sos=False, batch_size=batch)
    dec_out = joint.project_prednet(dec_out)
    iterations = torch.zeros_like(time)
    durations_sum = torch.zeros_like(time)
    last_nb_t = torch.full_like(time, -1)
    last_nb_count = torch.zeros_like(time)
    tokens: list[list[int]] = [[] for _ in range(batch)]

    def step(mask):
        logits = joint.joint_after_projection(enc[rows, safe].unsqueeze(1), dec_out).squeeze(1).squeeze(1)
        _, found = logits[:, :-nd].max(dim=-1)     # as NeMo: max(), not argmax()
        dur = model_durations[logits[:, -nd:].argmax(dim=-1)]
        iterations.add_(mask.long())
        durations_sum.add_(dur * mask)
        return found, dur

    while bool(active.any()):
        active_prev = active.clone()
        labels, dur = step(active)
        is_blank = labels == blank
        dur = dur.masked_fill(dur.eq(0) & is_blank, 1)
        current = time.clone()
        time = time + dur * active
        safe = torch.minimum(time, last_t)
        active = time < length
        advance = active & is_blank
        while bool(advance.any()):
            current = torch.where(advance, time, current)
            more, dur = step(advance)
            labels = torch.where(advance, more, labels)
            is_blank = labels == blank
            dur = dur.masked_fill(dur.eq(0) & is_blank, 1)
            time = torch.where(advance, time + dur, time)
            safe = torch.minimum(time, last_t)
            active = time < length
            advance = active & is_blank
        found = active_prev & (labels != blank)
        for i in torch.nonzero(found).flatten().tolist():
            tokens[i].append(int(labels[i]))
        last_nb_count = torch.where(found & (last_nb_t == current), last_nb_count + 1, last_nb_count)
        last_nb_count = torch.where(found & (last_nb_t != current), torch.ones_like(last_nb_count), last_nb_count)
        last_nb_t = torch.where(found, current, last_nb_t)
        prev_state, prev_out = state, dec_out
        dec_new, state, *_ = decoder.predict(labels.unsqueeze(1), prev_state, add_sos=False, batch_size=batch)
        dec_new = joint.project_prednet(dec_new)
        decoder.batch_replace_states_mask(src_states=prev_state, dst_states=state, mask=~found)
        dec_out = torch.where(found.unsqueeze(-1).unsqueeze(-1), dec_new, prev_out)
        if max_symbols is not None:
            force = active & (labels != blank) & (last_nb_count >= max_symbols) & (last_nb_t == time)
            time = time + force.long()
            safe = torch.minimum(time, last_t)
            active = time < length
    return [{"tokens": tokens[i], "iterations": int(iterations[i]), "durations": int(durations_sum[i])}
            for i in range(batch)]


def proxy_set(model, records: list[dict], batch_size: int, max_batch_seconds: float, device) -> dict:
    """The proxy over one manifest (records as evaluate.read_manifest returns them), with the NeMo token check."""
    import evaluate  # finetune/parakeet-ternary
    device = torch.device(device)
    batches = evaluate.make_batches(records, batch_size, max_batch_seconds)
    loader = torch.utils.data.DataLoader(evaluate._BatchSet(records, batches), batch_size=None, shuffle=False,
                                         num_workers=2, pin_memory=device.type == "cuda")
    totals = {"tokens": 0, "iterations": 0, "durations": 0}
    agree, mismatched = 0, []
    with evaluate.strict_fp32(device), evaluate.inference_settings(model):
        for idx, audio, lengths, _ in loader:
            enc, enc_len = model.forward(input_signal=audio.to(device), input_signal_length=lengths.to(device))
            hyps = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=enc_len,
                                                                  return_hypotheses=True)
            if isinstance(hyps, tuple):
                hyps = hyps[0]
            mine = greedy_steps(model, enc, enc_len)
            for i, h, m in zip(idx, hyps, mine):
                ref = [int(t) for t in torch.as_tensor(h.y_sequence).flatten().tolist()]
                if ref == m["tokens"]:
                    agree += 1
                else:
                    mismatched.append(records[i]["id"])
                totals["tokens"] += len(m["tokens"])
                totals["iterations"] += m["iterations"]
                totals["durations"] += m["durations"]
    seconds = sum(float(r["duration"]) for r in records)
    return {"utterances": len(records), "audio_seconds": seconds, **totals,
            "tokens_per_second": totals["tokens"] / seconds, "iterations_per_second": totals["iterations"] / seconds,
            "mean_predicted_duration": totals["durations"] / totals["iterations"] if totals["iterations"] else None,
            "nemo_token_agreement": agree, "nemo_token_mismatched_ids": mismatched[:50],
            "batch_size": batch_size, "max_batch_seconds": max_batch_seconds}


def pooled(results: dict[str, dict]) -> dict:
    """Totals over sets (per second of all their audio)."""
    s = sum(r["audio_seconds"] for r in results.values())
    t = sum(r["tokens"] for r in results.values())
    it = sum(r["iterations"] for r in results.values())
    d = sum(r["durations"] for r in results.values())
    return {"audio_seconds": s, "tokens": t, "iterations": it, "tokens_per_second": t / s,
            "iterations_per_second": it / s, "mean_predicted_duration": d / it if it else None,
            "utterances": sum(r["utterances"] for r in results.values()),
            "nemo_token_agreement": sum(r["nemo_token_agreement"] for r in results.values())}
