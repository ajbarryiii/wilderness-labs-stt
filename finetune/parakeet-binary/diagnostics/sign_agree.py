"""Sign agreement and drift between M1 latents and FP32 original over the 264 quantized matrices."""
import sys, tarfile, torch
sys.path.insert(0, sys.argv[3]); from binary_diag import PATTERN
m1 = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)["student"]
with tarfile.open(sys.argv[2]) as tar:
    mem = next(m for m in tar.getmembers() if m.name.endswith("model_weights.ckpt"))
    fp = torch.load(tar.extractfile(mem), map_location="cpu", weights_only=True)
keys = [k for k in m1 if PATTERN.fullmatch(k)]; assert len(keys) == 264
n = same = d2 = w2 = 0; samebig = nbig = 0
for k in keys:
    a, b = m1[k].float().flatten(1), fp[k].float().flatten(1)
    s = b.abs().mean(1, keepdim=True)
    big = (b / s).abs() >= 0.5
    n += a.numel(); same += (torch.sign(a) == torch.sign(b)).sum().item()
    nbig += big.sum().item(); samebig += ((torch.sign(a) == torch.sign(b)) & big).sum().item()
    d2 += (a - b).pow(2).sum().item(); w2 += b.pow(2).sum().item()
print(f"sign agreement all={same/n:.4f} (FP32 nonzero-ternary weights {samebig/nbig:.4f}); relative drift ||M1-FP32||/||FP32||={(d2/w2)**.5:.4f}")
