"""The read-speech extra source of stage 3 (b0prime plan "Extra sources"; registry alias `cv27-en`).

Pipeline (CLI: ./python -m cv27 COMMAND; jobs: cv27/launch.sh):

    boilerplate   reproduce the global denylist's boilerplate 8-grams (pool scan; verified against denylist.json)
    select        held-out removal, global denylist, the talk-text 8-gram rule, caps, duration, labels; the dev
                  sample (CPU)
    decode        MP3 -> 16 kHz mono FLAC store, chunked and resumable (CPU)
    dev           the dev sample's FLAC and its scorer manifest (CPU)
    confirm       B0 greedy over every decoded clip, chunked and resumable (GPU)
    finalize      the clip source (store/clips.jsonl), B0 transcripts and reports
    probe-decode / probe-b0   a seeded 2,000-clip probe in the smoke tree

Modules: layout (roots, constants, artifact paths), heldout (held-out sets and the text rules), select (caps,
duration, labels and the selection table), store (decoding and the clip source), confirm (the B0 check), dev
(the dev sample and manifest).
"""
