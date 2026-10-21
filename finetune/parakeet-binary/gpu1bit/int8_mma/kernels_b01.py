"""b01: the frozen b1_dot (kernels.py, unchanged) with {0, 1} weight bytes and an exact row-sum correction.

Frozen int8_mma expands each byte plane p of a fragment word to +-1 bytes:
    expand<P>(w)   = 0xFFFFFFFF - ((w >> P) & 0x01010101) * 254        (SHF + LOP3 + IMAD; P = 0: LOP3 + IMAD)
b01 feeds the bits themselves, 0x00 / 0x01 bytes, to the same mma.m16n8k32.s8.s8 (0x01 is +1 in s8):
    expand01<P>(w) = (w >> P) & 0x01010101                             (SHF + LOP3; P = 0: LOP3)
so the INT32 accumulators hold acc01_c = sum_k q_c b, and with w = 2 b - 1
    acc_c = 2 * acc01_c - R_c,   R_c[m] = sum_k q_c[m, k]   over exactly the k this CTA reduces.
R_c is computed in the kernel from the A fragments the warps already hold after ldmatrix: lane (g, s)
of a warp holds, per (part c, m16 tile i, k32 step), the bytes of rows g / g + 8 at k columns
4s..4s+3 and 16+4s..16+4s+3. Four dp4a(a, 0x01010101) add them (each byte once). The WN warps of a
row set hold identical A fragments, so each (part, k32 step) is summed by exactly one of them
((4 c + ks) mod WN == wn; DISTRIBUTION "wn0" gives all of it to wn == 0), the four lanes s are
summed with two shuffles, every warp parks its partial in shared memory, a fixed-order INT32 sum
over the warp slots (all K groups included) gives R, and after the frozen WK reduction the group-0
warps apply acc = 2 * acc - R in INT32. Zero-filled A beyond K or M adds 0 to
both acc01 and R (padding bits are 0), so tails are exact. The correction is applied per CTA to its
own K-range partial, before any cluster exchange (XCHG 1 / 3) or deferred slice (XCHG 4): integer
sums are linear, so this equals correcting once after the full reduction, bit for bit, and every
configuration's INT32 accumulators equal the frozen kernel's, hence identical outputs.

Shared memory: the per-warp partials and the reduced R live after everything the frozen kernel uses
(pipeline, DSMEM and WK reduction scratch), (WK * WN + 1) * 3 * BMR int32. Nothing is added to global
memory.
"""
from __future__ import annotations

import dataclasses

from gpu1bit.int8_mma import kernels
from gpu1bit.int8_mma.kernels import DotConfig


@dataclasses.dataclass(frozen=True)
class B01Config(DotConfig):
    """DotConfig plus the row-sum partials' shared memory (rs region after the frozen layout)."""

    @property
    def shared_bytes(self) -> int:
        base = -(-DotConfig.shared_bytes.fget(self) // 16) * 16
        return base + rs_slots(self) * 4

    def label(self) -> str:
        return "b01_" + DotConfig.label(self)

    @classmethod
    def of(cls, config: DotConfig) -> "B01Config":
        return config if isinstance(config, cls) else cls(*[getattr(config, f.name) for f in dataclasses.fields(DotConfig)])


def _replace(source: str, old: str, new: str, count: int = 1) -> str:
    found = source.count(old)
    if found != count:
        raise AssertionError(f"b01 transform expected {count} x {old[:70]!r}, found {found}")
    return source.replace(old, new)


# Row-sum work distribution: "balanced" (default) gives the WN warps of a row set disjoint
# (part, k32 step) pairs, (4 c + ks) mod WN == wn; "wn0" puts all of it on warp wn == 0.
# "mma": every warp sums its own rows over its own k with one extra IMMA per (part, m16 tile, k32 step)
# against an all-ones B fragment (the MMA reduces over k; c0, c1 hold row g's sum, c2, c3 row g + 8)
# and corrects its own accumulators before the frozen WK reduction: no shuffles, shared memory or
# barriers, +1 IMMA per 3 * RN.
DISTRIBUTION = "balanced"


def rs_slots(config: DotConfig) -> int:
    """int32 entries of the row-sum region: per-warp partials [WK * WN][3 BMR] plus the reduced [3 BMR]."""
    if DISTRIBUTION == "mma":
        return 0
    warps = config.wk * (config.wn if DISTRIBUTION == "balanced" else 1)
    return (warps + 1) * 3 * config.bmr


def transform(source: str) -> str:
    """Frozen dot source -> b01 source (asserted edits only; see module docstring)."""
    if DISTRIBUTION == "mma":
        return transform_mma(source)
    balanced = DISTRIBUTION == "balanced"
    s = source
    # 1. {0, 1} expansion and a dp4a helper.
    s = _replace(s, "template <int P> DEVI u32 expand(u32 w) { return 0xFFFFFFFFu - ((w >> P) & 0x01010101u) * 254u; }",
                 "template <int P> DEVI u32 expand(u32 w) { return 0xFFFFFFFFu - ((w >> P) & 0x01010101u) * 254u; }\n"
                 "// b01: byte plane P as 0x00 / 0x01 bytes (bit 1 = +1 -> 1, bit 0 = -1 -> 0).\n"
                 "template <int P> DEVI u32 expand01(u32 w) { return (w >> P) & 0x01010101u; }\n"
                 "DEVI int dp4a_ones(u32 a, int c) { int r; asm(\"dp4a.s32.s32 %0, %1, %2, %3;\" : \"=r\"(r) : \"r\"(a), \"r\"(0x01010101u), \"r\"(c)); return r; }")
    for p in range(8):
        s = _replace(s, f"expand<{p}>(bw[j])", f"expand01<{p}>(bw[j])")
    # 2. per-lane row-sum partials, zeroed with the accumulators
    s = _replace(s, "    int acc[3][RM][RN][4];\n",
                 "    int acc[3][RM][RN][4];\n"
                 "    int rsl[3][RM][2];  // b01: this lane's share of R_c for rows g, g + 8 of each m16 tile\n"
                 "    #pragma unroll\n    for (int c = 0; c < 3; ++c)\n        #pragma unroll\n"
                 "        for (int i = 0; i < RM; ++i) { rsl[c][i][0] = 0; rsl[c][i][1] = 0; }\n")
    # 3. dp4a on the fragments just loaded (warps wn == 0 only; each A byte counted once)
    s = _replace(s, "                    ldmatrix_x4(a0, a1, a2, a3, st + row * 128 + ((ch ^ (row & 7)) << 4));\n",
                 "                    ldmatrix_x4(a0, a1, a2, a3, st + row * 128 + ((ch ^ (row & 7)) << 4));\n"
                 + ("                    if (((4 * c + ks) % WN) == wn) {\n" if balanced else "                    if (wn == 0) {\n") +
                 "                        rsl[c][i][0] = dp4a_ones(a2, dp4a_ones(a0, rsl[c][i][0]));\n"
                 "                        rsl[c][i][1] = dp4a_ones(a3, dp4a_ones(a1, rsl[c][i][1]));\n"
                 "                    }\n")
    # 4. after the main loop: reduce R over lanes and K groups, correct group 0's accumulators
    s = _replace(s, "    cp_wait<0>();\n    const int g = lane >> 2, s4 = lane & 3;\n",
                 "    cp_wait<0>();\n    const int g = lane >> 2, s4 = lane & 3;\n"
                 "    constexpr int RS_BASE = STAGES * STAGE_BYTES;\n"
                 "    constexpr int RS_RED = (CLUSTER > 1 && XCHG == 0) ? 3 * BMR * (BN + 8) * 4 : 0;\n"
                 "    constexpr int RS_GRP = (WK - 1) * 3 * BMR * (BN + 8) * 4;\n"
                 "    constexpr int RS_MAX = RS_BASE > RS_RED ? (RS_BASE > RS_GRP ? RS_BASE : RS_GRP) : (RS_RED > RS_GRP ? RS_RED : RS_GRP);\n"
                 "    constexpr int RS_OFF = (RS_MAX + 15) / 16 * 16;\n"
                 + ("    constexpr int RS_W = WK * WN;  // per-warp partial slots (balanced)\n"
                    "    const int rs_slot = wk * WN + wn;\n" if balanced else
                    "    constexpr int RS_W = WK;  // per-group partial slots (wn0)\n"
                    "    const int rs_slot = wk;\n") +
                 "    int *rs_part = reinterpret_cast<int *>(smem + RS_OFF);  // [RS_W][3 * BMR], then reduced [3 * BMR]\n"
                 + ("    if (true) {\n" if balanced else "    if (wn == 0) {\n") +
                 "        #pragma unroll\n        for (int c = 0; c < 3; ++c)\n            #pragma unroll\n"
                 "            for (int i = 0; i < RM; ++i)\n                #pragma unroll\n"
                 "                for (int h = 0; h < 2; ++h) {\n"
                 "                    int v = rsl[c][i][h];\n"
                 "                    v += __shfl_xor_sync(0xffffffffu, v, 1);\n"
                 "                    v += __shfl_xor_sync(0xffffffffu, v, 2);\n"
                 "                    if (s4 == 0) rs_part[rs_slot * 3 * BMR + c * BMR + (wm * RM + i) * 16 + g + 8 * h] = v;\n"
                 "                }\n"
                 "    }\n"
                 "    __syncthreads();\n"
                 "    for (int t = tid; t < 3 * BMR; t += THREADS) {  // fixed-order INT32 sum of the slots\n"
                 "        int r = 0;\n"
                 "        #pragma unroll\n        for (int w = 0; w < RS_W; ++w) r += rs_part[w * 3 * BMR + t];\n"
                 "        rs_part[RS_W * 3 * BMR + t] = r;\n"
                 "    }\n"
                 "    __syncthreads();\n")
    # 5. correction once the K groups are reduced (group 0 holds the CTA's accumulators)
    s = _replace(s, "    if constexpr (CLUSTER == 1) {\n        if (wk != 0) return;\n",
                 "    if (wk == 0) {  // b01: acc = 2 * acc01 - R, exact in INT32 (|.| <= 127 K)\n"
                 "        #pragma unroll\n        for (int c = 0; c < 3; ++c)\n            #pragma unroll\n"
                 "            for (int i = 0; i < RM; ++i)\n                #pragma unroll\n"
                 "                for (int h = 0; h < 2; ++h) {\n"
                 "                    const int r = rs_part[RS_W * 3 * BMR + c * BMR + (wm * RM + i) * 16 + g + 8 * h];\n"
                 "                    #pragma unroll\n                    for (int j = 0; j < RN; ++j) {\n"
                 "                        acc[c][i][j][2 * h] = 2 * acc[c][i][j][2 * h] - r;\n"
                 "                        acc[c][i][j][2 * h + 1] = 2 * acc[c][i][j][2 * h + 1] - r;\n"
                 "                    }\n"
                 "                }\n"
                 "    }\n"
                 "    if constexpr (CLUSTER == 1) {\n        if (wk != 0) return;\n")
    return s


def transform_mma(source: str) -> str:
    """b01 with row sums from an all-ones IMMA per warp; correction per warp before the WK reduction."""
    s = _replace(source, "template <int P> DEVI u32 expand(u32 w) { return 0xFFFFFFFFu - ((w >> P) & 0x01010101u) * 254u; }",
                 "template <int P> DEVI u32 expand(u32 w) { return 0xFFFFFFFFu - ((w >> P) & 0x01010101u) * 254u; }\n"
                 "template <int P> DEVI u32 expand01(u32 w) { return (w >> P) & 0x01010101u; }")
    for p in range(8):
        s = _replace(s, f"expand<{p}>(bw[j])", f"expand01<{p}>(bw[j])")
    s = _replace(s, "    int acc[3][RM][RN][4];\n",
                 "    int acc[3][RM][RN][4];\n"
                 "    int rsa[3][RM][4];  // b01: sum_k q_c for rows g (e 0, 1) and g + 8 (e 2, 3), via the ones IMMA\n"
                 "    #pragma unroll\n    for (int c = 0; c < 3; ++c)\n        #pragma unroll\n"
                 "        for (int i = 0; i < RM; ++i)\n            #pragma unroll\n"
                 "            for (int e = 0; e < 4; ++e) rsa[c][i][e] = 0;\n")
    s = _replace(s, "                    for (int j = 0; j < RN; ++j) mma_s8(acc[c][i][j], a0, a1, a2, a3, b0[j], b1[j]);\n",
                 "                    for (int j = 0; j < RN; ++j) mma_s8(acc[c][i][j], a0, a1, a2, a3, b0[j], b1[j]);\n"
                 "                    mma_s8(rsa[c][i], a0, a1, a2, a3, 0x01010101u, 0x01010101u);\n")
    s = _replace(s, "    cp_wait<0>();\n    const int g = lane >> 2, s4 = lane & 3;\n",
                 "    cp_wait<0>();\n    const int g = lane >> 2, s4 = lane & 3;\n"
                 "    #pragma unroll\n    for (int c = 0; c < 3; ++c)  // b01: acc = 2 * acc01 - R over this warp's own k\n"
                 "        #pragma unroll\n        for (int i = 0; i < RM; ++i)\n            #pragma unroll\n"
                 "            for (int j = 0; j < RN; ++j)\n                #pragma unroll\n"
                 "                for (int e = 0; e < 4; ++e) acc[c][i][j][e] = 2 * acc[c][i][j][e] - rsa[c][i][e];\n")
    return s


def dot_source(config: DotConfig, silu: bool, order: int = 0) -> tuple[str, str]:
    """(b01 source, kernel name) for a frozen DotConfig (its b01 shared memory: B01Config)."""
    base = DotConfig(*[getattr(config, f.name) for f in dataclasses.fields(DotConfig)])
    source, name = kernels.dot_source(base, silu, order)
    b01_name = "b01_" + name
    source = _replace(source, f"\n{name}(", f"\n{b01_name}(")
    B01Config.of(config).validate()
    return transform(source), b01_name
