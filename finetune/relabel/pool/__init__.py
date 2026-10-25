"""Stage 2 Part B (labeling plan): domain subsets of the streamed pool and the global denylist.

Text only: the transcript columns of the pinned `yodas` and `peoples_speech` training files are read with
parquet column projection (byte-range requests for exactly the projected column chunks; audio is never
fetched) into a local text cache, and everything else runs locally from that cache.

    ./python -m pool fetch --source yodas|peoples_speech [--workers P --threads T]   text cache (network)
    ./python -m pool probe --source yodas --files N                                   timing sample (smoke path)
    ./python -m pool finalize                                                          denylist, then subsets
    ./python -m pool status

Modules: common (roots, constants, text and 8-gram helpers), remote (sparse parquet range reads), fetch
(text cache), terms_match (term densities), scan (one local pass over the cache), denylist, select.
Outputs (all under $RELABEL_ROOT/pool/): text/<source>/, denylist.json, denylist_details.jsonl,
recordings.parquet, subsets/<subset>.jsonl, subsets/summary.json, sample/<subset>.txt, summary.json.
"""
