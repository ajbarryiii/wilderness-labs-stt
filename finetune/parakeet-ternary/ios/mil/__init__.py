"""Core ML MIL programs for every encoder arm, the decoder/joint models and their gates (WP3, DESIGN.md S1).

Modules: weights (tensor sources: M_P2, surrogates, C0 for G0), encodings (constexpr chains per arm
and their effective matrices), encoder (plain-layout FastConformer encoder graph), decoder
(Decoder / JointDecision / JointLogits / DecoderJoint), build (CLI: build, compile, manifest,
compute plan), probes (one-layer probes, gate 2, C7 folding, C7/C8 stress), gates (gates 3-5),
reference_cache (FP32 reference outputs, C5 calibration).
"""
from __future__ import annotations

import sys
from pathlib import Path

MIL_DIR = Path(__file__).resolve().parent
IOS = MIL_DIR.parent
for _p in (str(IOS), str(IOS.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
