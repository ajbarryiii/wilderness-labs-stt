"""Compile one config of _pgemm / _bgemm and report registers, spills, smem and the TTGIR main-loop ops."""
import re, sys, collections
import torch, triton
from gpu1bit.triton_bf16.gpulock import locked
from gpu1bit.shared import env
from gpu1bit.triton_bf16 import kernels as K

def main():
    mode = sys.argv[1]; m, n, k = map(int, sys.argv[2:5]); cfg = tuple(int(v) if v.isdigit() else v for v in sys.argv[5].split(","))
    bm, bn, bk, split, warps, stages, reduce, *rest = cfg
    with locked("A:inspect"), torch.inference_mode():
        x = torch.randn(1, m, k, device="cuda")
        codes = torch.randint(0, 256, (n, k // 8), dtype=torch.uint8)
        from gpu1bit.triton_bf16.layer import TritonBF16Layer
        layer = TritonBF16Layer.prepare(codes, torch.rand(n), None, k, "cuda", mode=mode, configs={m: cfg})
        layer.linear(x); torch.cuda.synchronize()
        fn = K._pgemm if mode.endswith("p") else K._bgemm
        for key, kern in list(fn.device_caches[0][0].items())[-1:]:
            md = kern.metadata
            print("regs", kern.n_regs, "spills", kern.n_spills, "shared", md.shared, "warps", md.num_warps, "stages", md.num_stages)
            ttgir = kern.asm["ttgir"]
            ops = collections.Counter(re.findall(r"(ttg\.\w+|tt\.dot|ttng\.\w+|tt\.load|amdgpu\.\w+)", ttgir))
            print(dict(ops))
            open(f"/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/A/{mode}-{m}-{n}-{k}-{'-'.join(map(str,cfg))}.ttgir", "w").write(ttgir)
            sass = kern.asm.get("sass") or ""
            ptx = kern.asm["ptx"]
            print("ptx mma:", len(re.findall(r"mma\.sync", ptx)), "ldmatrix:", len(re.findall(r"ldmatrix", ptx)), "cp.async:", len(re.findall(r"cp\.async\.c", ptx)), "st.shared:", len(re.findall(r"st\.shared", ptx)), "ld.shared:", len(re.findall(r"ld\.shared", ptx)))

if __name__ == "__main__":
    main()
