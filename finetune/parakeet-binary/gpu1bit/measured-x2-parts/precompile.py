"""CPU-only NVRTC pass for the P-part kernels the tests use (CUDA_VISIBLE_DEVICES="")."""
import sys, os, itertools
sys.path.insert(0, "/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/B/wip/parts")
import wip  # noqa


def job(args):
    import wip  # noqa
    from gpu1bit.int8_mma import cuda, kernels_parts
    from gpu1bit.int8_mma.kernels import DotConfig
    values, silu, order, parts, b01 = args
    cfg = DotConfig(*values)
    try:
        src, name = kernels_parts.dot_source(cfg, silu, order, parts, b01)
        cuda.compile_cubin(src, name)
    except Exception as exc:
        return f"{cfg} P={parts} b01={b01}: {exc!r}"[:400]


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    import multiprocessing as mp
    from gpu1bit.int8_mma import test_parts
    jobs = []
    for cfg, silu, order, parts, b01 in itertools.product(test_parts.CONFIGS, (False, True), (0, 1), (1, 2, 3), (False, True)):
        jobs.append((tuple(getattr(cfg, f) for f in ("wm", "wn", "rm", "rn", "stages", "cluster", "xchg", "wk", "pf")), silu, order, parts, b01))
    print(len(jobs), "jobs", flush=True)
    with mp.get_context("spawn").Pool(16) as pool:
        fails = [f for f in pool.imap_unordered(job, jobs, chunksize=4) if f]
    print(len(fails), "failures", *fails[:10], sep="\n")
