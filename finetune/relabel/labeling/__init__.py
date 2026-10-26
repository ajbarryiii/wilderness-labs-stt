"""Stage 2 Part C (labeling plan): the offline labeling pass with T* over the pinned training stream.

    C1 (CPU)  replay.py      the pinned stream (pool_stream.py, config.py; batches.py = train.Batcher's sequence)
                             replayed through consumed batch 100,000 into journal + FLAC shards (shards.py)
    C2 (GPU)  c2.py          T* (engines.py: vLLM, or the stage 1 transformers decoder) and B0 over C1's shards
                             in contiguous groups, filters 1-4 (filters.py), label shards
    checks.py                the GPU probe, the consistency gate, the equivalence check, the gate to stage 3
    vllm_env.sh              the isolated vLLM environment ($RELABEL_ROOT/deps/vllm-<version>)
    launch.sh                the jobs (probes, chunks, checks), run through `job`

CLI: `./python -m labeling --help` (from finetune/relabel). Artifact locations: layout.py.
"""
