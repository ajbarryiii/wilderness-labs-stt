# Prepared training runtime and real speech-presence data

Provisioning is outside the research timer. On this NixOS host, run:

```sh
custom/autoresearch/.runtime/bin/python3 custom/autoresearch/runtime_setup.py
custom/autoresearch/.runtime/bin/python3 custom/autoresearch/data_setup.py
custom/autoresearch/runtime-python custom/autoresearch/runtime_verify.py
```

`runtime.nix` pins the already available Nixpkgs source by immutable store path.
`runtime_lock.json` records its NAR hash, the resulting environment store path,
and measured package versions. Both Nixpkgs and the Python environment have
project-local GC roots. The recipe uses the PyTorch binary package plus NVIDIA's
hash-pinned NCCL 2.28.9 and NVSHMEM 3.6.5 binary wheels, avoiding a full distributed
CUDA library source build. It does not change host drivers or GPU settings.

The verified runtime is Python 3.13.13, PyTorch 2.11.0+cu128, NumPy 2.4.4,
SciPy 1.17.1 and SoundFile 0.13.1. PyTorch reports CUDA 12.8 build support and
sm120 kernels; the Nix package supplies compatible CUDA 12.9 shared libraries.
`runtime_verify.py` requires a live RTX5090 and checks three actual CUDA Conv1d
forward/backward/AdamW updates, finite gradients and changed weights. It fails if
the GC root differs from the locked environment. `runtime_evidence.json` contains
the preparation check. This is a runtime smoke test, not a model accuracy result.

`data_setup.py` verifies fixed archive SHA256 checksums and safely extracts only
Mini LibriSpeech train-clean-5 and the MUSAN point-source noises in RIRS_NOISES.
The two archives total 1,644,120,613 bytes. No official LibriSpeech development or
test audio, Eka English, or other final holdout was downloaded or used.
`data_prepare.py` deterministically builds `data_manifest.json`; the model harness
copies that manifest into its own frozen benchmark. Raw audio stays under `.data`.

The 2,903 examples contain real waveforms: complete 3–20 second English utterances
with transcripts as speech-presence positives, plus disjoint windows from real
noise recordings. All utterances from a speaker stay in one split; all windows
and exact duplicates from a noise recording stay in one split. Noise durations
are sampled from the speech-duration distribution in the same split, bounded by
the available original recording. Short clips and exact duplicate noises are
excluded. Training has 898 speech and 988 noise clips; calibration has 373 and221;
development has 211 and212. The limited development denominator cannot establish
a precise 99% recall confidence bound.

This is a whole-clip speech-presence proxy. A transcript does not label pauses as
speech, and no individual frame receives a speech label. It does not establish
streaming onset/offset accuracy. MUSAN noise labels are inherited from the corpus;
we have not performed a new human audit to rule out residual speech contamination.
Freesound contributor/session links between distinct recording IDs are absent
from this archive. Those limitations are preserved in the manifest.

## Sources and attribution

- Mini LibriSpeech, [OpenSLR31](https://www.openslr.org/31/), CC BY4.0. It is a
  subset of LibriSpeech, Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev
  Khudanpur, “Librispeech: An ASR corpus based on public domain audio books,”2015.
  Original license and speaker/chapter metadata remain under `.data/audio/LibriSpeech`.
- [OpenSLR28 RIRS_NOISES](https://www.openslr.org/28/), archive page Apache2.0.
  Only point-source noises are used. Their included LICENSE states that the
  selected Freesound recordings were marked Public Domain. The included README
  describes 843 manually foreground/background-classified MUSAN noise recordings.
- MUSAN: David Snyder, Guoguo Chen and Daniel Povey, “MUSAN: A Music, Speech, and
  Noise Corpus,”2015, [OpenSLR17](https://www.openslr.org/17/).

The manifest records archive URLs, observed SHA256 checksums, all consumed file
SHA256 checksums, source/speaker groups, sample offsets and counts. The Mini
LibriSpeech archive also matched the publisher's MD5
`5df7d4e78065366204ca6845bb08f490` before extraction. RIRS archive SHA256 was recorded
from the HTTPS origin download; no independent publisher SHA256 was available.
No synthetic feature benchmark or generated waveform is used here.
