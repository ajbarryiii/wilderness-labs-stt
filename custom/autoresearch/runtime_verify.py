"""Fail closed unless the pinned training runtime executes real CUDA gradients."""
import json, os, platform, time, pathlib
import torch
import numpy
import scipy
import soundfile

def main():
    root = pathlib.Path(__file__).resolve().parent
    lock_path = root / 'runtime_lock.json'
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        assert str((root / '.training-runtime').resolve()) == lock['runtime_store_path'], 'Runtime GC-root changed since dependency lock'
    assert torch.cuda.is_available(), "CUDA unavailable"
    assert torch.cuda.get_device_capability() == (12, 0), "Expected RTX 5090 / sm120"
    torch.manual_seed(20260909)
    torch.set_num_threads(1)
    model = torch.nn.Sequential(torch.nn.Conv1d(16, 32, 5, bias=False), torch.nn.ReLU(), torch.nn.Conv1d(32, 1, 1, bias=False)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = model[0].weight.detach().clone()
    x = torch.randn(64, 16, 200, device="cuda")
    start = time.monotonic()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model(x).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    torch.cuda.synchronize()
    assert not torch.equal(before, model[0].weight), "CUDA weights did not update"
    print(json.dumps({"ok": True, "cuda_available":True,"gpu":torch.cuda.get_device_name(),"compute_capability":list(torch.cuda.get_device_capability()),"architectures":torch.cuda.get_arch_list(),"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"python":platform.python_version(),"numpy":numpy.__version__,"scipy":scipy.__version__,"soundfile":soundfile.__version__,"cuda_forward_backward_steps":3,"weights_changed":True,"loss":float(loss.detach().cpu()),"elapsed_seconds":time.monotonic()-start,"peak_cuda_bytes":torch.cuda.max_memory_allocated()}))
if __name__ == "__main__": main()
