"""MLX GPU encoder arm (DESIGN.md "E. Encoder on the GPU"; WP6b). macOS only (mlx is a darwin-only dependency).

Modules: encoder (2-bit affine weights built exactly from the ternary codes, the FastConformer encoder in MLX
with the S0 masking contract), gates (gate 2 round trip, 4a FP32 path, 4b FP16 on the GPU with F2-equivalent
FP32 decoding, gate 5, stress, informational timing), record (eligibility record, NixOS).
"""
from __future__ import annotations

import sys
from pathlib import Path

MLX_DIR = Path(__file__).resolve().parent
IOS = MLX_DIR.parent
for _p in (str(IOS), str(IOS.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
