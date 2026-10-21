"""CPU check of the {0,1} identity's numerics (agent A): y = alpha (2 b.x - sum x) vs y = alpha w.x.

On real captured dev inputs (H's fixtures; layers whose inputs have a large positive mean, such as
SiLU outputs feeding ff linear2, and post-LayerNorm inputs), both forms are evaluated with FP32
sequential accumulation over K (the same rounding model for both, CPU, no GPU) against FP64; the
ratio of their relative errors shows what the identity costs in accuracy. CUDA_VISIBLE_DEVICES="".
"""
import os

import torch

from gpu1bit.shared import api, fixtures


def fp32_dot_rows(x, w):
    """Sequential FP32 accumulation over K, chunked by 64 (tile sums in FP32, then running sum)."""
    acc = torch.zeros(x.shape[0], w.shape[0], dtype=torch.float32)
    for k0 in range(0, x.shape[1], 64):
        acc = acc + (x[:, k0:k0 + 64].double() @ w[:, k0:k0 + 64].double().t()).float()
    return acc


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU only"
    names = ["encoder.layers.3.feed_forward1.linear2", "encoder.layers.10.feed_forward2.linear2",
             "encoder.layers.3.feed_forward1.linear1", "encoder.layers.10.self_attn.linear_q"]
    weights = fixtures.mb1_layers(names)
    inputs = fixtures.utterance_inputs(5, device="cpu")
    for name in names:
        w, x = weights[name], inputs[name]
        x2 = x.reshape(-1, w["K"]).float()
        signs = api.unpack_signs(w["codes"], w["K"]).float()
        bits = (signs > 0).float()
        ref = x2.double() @ signs.double().t()
        pm1 = fp32_dot_rows(x2, signs)
        ident = 2 * fp32_dot_rows(x2, bits) - x2.sum(1, keepdim=True)
        e_pm1 = api.relative_error(pm1, ref)
        e_id = api.relative_error(ident, ref)
        mean_ratio = float((x2.mean(1).abs() / x2.std(1)).median())
        print(f"{name:42s} |mean|/std {mean_ratio:5.2f}  rel err +-1 {e_pm1:.2e}  {{0,1}} identity {e_id:.2e}"
              f"  ratio {e_id / e_pm1:5.1f}")


if __name__ == "__main__":
    main()
