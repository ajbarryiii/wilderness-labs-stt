"""Cache valid-frame CTC predictions from a custom checkpoint or local NeMo reference."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from common import ALPHABET, decode, digest, norm, save
from pilot4_train import configured_model
from recovery_core import ROOT, SOURCE_RUN, artifact, read


def cache(output, checkpoint=None, manifest=None, limit=None, seconds=900):
    started = time.monotonic()
    output = artifact(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest or SOURCE_RUN / "manifest.json")
    rows = [r for r in read(manifest)["rows"] if r["split"] == "development"]
    if limit:
        rows = rows[:limit]
    torch.set_num_threads(4)
    if checkpoint:
        checkpoint = Path(checkpoint)
        saved = torch.load(
            checkpoint, map_location="cpu", mmap=True, weights_only=False
        )
        model = configured_model(saved["config"])(
            saved["config"]["model"], saved["arm"]["precision"]
        )
        model.load_state_dict(saved["model"], strict=True)
        labels = list(ALPHABET)
        labels[0] = ""
        blank = 0
        source = dict(
            kind="custom",
            checkpoint=str(checkpoint),
            sha256=digest(checkpoint),
            config=saved["config"],
            step=saved["step"],
        )
        del saved
    else:
        from nemo.collections.asr.models import EncDecCTCModelBPE

        path = ROOT / "models/conformer-ctc-small/stt_en_conformer_ctc_small.nemo"
        model = EncDecCTCModelBPE.restore_from(str(path), map_location="cpu")
        labels = list(model.decoder.vocabulary)
        blank = int(model.decoding.blank_id)
        labels.insert(blank, "")
        source = dict(
            kind="nvidia_conformer_ctc_small",
            checkpoint=str(path),
            sha256=digest(path),
            parameters=sum(p.numel() for p in model.parameters()),
            source=read(path.parent / "source.json"),
            pretraining_overlap="Current LibriSpeech development may be in pretraining",
        )
    model.cuda().eval()
    entries = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            if time.monotonic() - started > seconds - 30:
                break
            assert digest(row["audio"]) == row["audio_sha256"]
            if checkpoint:
                assert digest(row["features"]) == row["feature_sha256"]
                x = torch.from_numpy(np.load(row["features"], allow_pickle=False))[
                    None
                ].cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = model(x).float().log_softmax(-1)[0]
                greedy = decode(z.argmax(-1).tolist())
                hop = 0.02
            else:
                audio, sr = sf.read(row["audio"], dtype="float32")
                assert sr == 16000 and audio.ndim == 1
                x = torch.from_numpy(audio)[None].cuda()
                log_probs, length, _ = model(
                    input_signal=x,
                    input_signal_length=torch.tensor([len(audio)], device="cuda"),
                )
                z = log_probs[0, : int(length[0])].float().log_softmax(-1)
                hypothesis = model.decoding.ctc_decoder_predictions_tensor(
                    log_probs, length, return_hypotheses=False
                )[0]
                greedy = norm(
                    hypothesis.text if hasattr(hypothesis, "text") else hypothesis
                )
                hop = float(model.cfg.preprocessor.window_stride) * int(
                    model.cfg.encoder.subsampling_factor
                )
            assert z.ndim == 2 and z.shape[1] == len(labels) and torch.isfinite(z).all()
            assert float(z.logsumexp(-1).abs().max()) < 1e-4
            filename = f"{index:05d}.npy"
            np.save(output / filename, z.cpu().numpy(), allow_pickle=False)
            entries.append(
                dict(
                    id=row["id"],
                    domain=row["domain"],
                    reference=norm(row["text"]),
                    speaker=row.get("speaker"),
                    seconds=row["seconds"],
                    waveform_sha256=row["audio_sha256"],
                    file=filename,
                    sha256=digest(output / filename),
                    frames=z.shape[0],
                    frame_hop_seconds=hop,
                    greedy=greedy,
                )
            )
    result = dict(
        source=source,
        labels=labels,
        blank_id=blank,
        entries=entries,
        requested=len(rows),
        complete=len(entries) == len(rows),
        manifest=str(manifest),
        manifest_sha256=digest(manifest),
        normalization="common.norm; digit_string only for controlled digits",
        elapsed_seconds=time.monotonic() - started,
    )
    save(output / "index.json", result)
    print(
        json.dumps(
            dict(output=str(output), examples=len(entries), complete=result["complete"])
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--seconds", type=float, default=900)
    a = p.parse_args()
    cache(a.output, a.checkpoint, a.manifest, a.limit, a.seconds)
