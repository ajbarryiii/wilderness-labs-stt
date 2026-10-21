"""SASS accounting of the weight expansion, frozen int8_mma vs int8_mma:b01 (CPU only: NVRTC + nvdisasm).

Counted from operands (kernel-wide static counts; the main-loop body holds one k128 block per warp):
  mask   LOP3.LUT Rd, Rw, 0x1010101   one per B register (4 weights) in both variants
  scale  IMAD Rd, Rs, R(-254), -0x1   frozen only: the +-1 map 0xFFFFFFFF - 254 * bits
  shift  SHF.R.U32.HI Rd, RZ, p, Rw  the plane shifts: one per plane p = 1..7 of each fragment word (the
                                      immediates 0x1/0x2/0x4/0x5/0x6 each occur once per word in the SASS; 0x3
                                      and 0x7 also appear in address math), so 7 per word, RN words per warp
  rowsum IDP.4A                       b01 only: exact row sums of the activation fragments
Per 4 weights = (mask + scale + shift) / B registers; b01 also reports IDP.4A per B register of a warp
(the CTA's IDP.4A are spread over its WN warps: per warp = total / WN for the balanced distribution).

    CUDA_VISIBLE_DEVICES= gpu1bit/shared/python -m gpu1bit.int8_mma.sass_b01
"""
from __future__ import annotations

import json
import re

from gpu1bit.int8_mma import configs, cuda, kernels, kernels_b01


def counts(text: str) -> dict:
    mask = len(re.findall(r"LOP3\.LUT R\d+, R\d+, 0x1010101, RZ, 0xc0", text))
    shift = {p: len(re.findall(rf"SHF\.R\.U32\.HI R\d+, RZ, 0x{p}, R\d+", text)) for p in range(1, 8)}
    scale = len(re.findall(r"IMAD R\d+, R\d+, R\d+(?:\.reuse)?, -0x1\b", text))
    return {"mask_LOP3": mask, "scale_IMAD": scale, "SHF_by_immediate": shift, "IDP.4A": len(re.findall(r"IDP\.4A", text)),
            "IMMA": len(re.findall(r"\bIMMA\.", text)), "SHFL": len(re.findall(r"\bSHFL\.", text)),
            "BAR": len(re.findall(r"\bBAR\.SYNC", text)), "instructions": len(re.findall(r"^\s+/\*[0-9a-f]{4,}\*/", text, re.M))}


def main():
    cases = [("1024x1024 M=38", 38, 1024, 1024), ("3072x1024 M=126", 126, 3072, 1024), ("4096x1024 M=256", 256, 4096, 1024),
             ("1024x4096 M=368", 368, 1024, 4096), ("2048x1024 M=368", 368, 2048, 1024)]
    for label, m, n, k in cases:
        cfg = configs.choose(m, n, k)
        b_regs = cfg.rn * 8  # per warp per k128 block (one main-loop body)
        out = {"case": label, "config": cfg.label(), "B_registers_per_warp_per_k128": b_regs}
        for variant, (src, name) in (("frozen", kernels.dot_source(cfg, False)), ("b01", kernels_b01.dot_source(cfg, False))):
            c = counts(cuda.sass(cuda.compile_cubin(src, name)))
            c["plane_SHF"] = 7 * cfg.rn  # checked: SHF_by_immediate[1, 2, 4, 5, 6] == RN each
            assert all(c["SHF_by_immediate"][p] >= cfg.rn for p in (1, 2, 4, 5, 6)), c["SHF_by_immediate"]
            c["expansion_per_4_weights"] = round((c["mask_LOP3"] + c["scale_IMAD"] + c["plane_SHF"]) / max(c["mask_LOP3"], 1), 3)
            if variant == "b01":
                c["rowsum_IDP4A_per_warp_per_B_register"] = round(c["IDP.4A"] / cfg.wn / max(c["mask_LOP3"], 1), 3)
            out[variant] = c
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
