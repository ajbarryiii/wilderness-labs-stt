"""Offline-label training on the stage 2 label store, shared by stage 2b (train_decoder.py) and stage 3
(train_b0prime.py). See decoder-only plan and b0prime plan.

Modules:
    labels     the label lookup by id (T* text or B0's label), with T*'s disposition mask applied to both
    replay     the journal replay from C1's local shards, batch for batch the labeled sequence, with
               consumed-batch accounting, a stop batch, and an optional extra clip source (stage 2c)
    schedule   the WSD schedule on the consumed-batch index (trunk, branches)
    devsplit   the cluster-disjoint dev split (frozen manifest, hash rule) and the dev clusters
    bootstrap  the paired cluster bootstrap (H4 and per set)
    seam       train.py replacements for offline labels: the replay batcher, the label lookup in place of the
               online teacher, the learning rate per consumed batch, the frozen-encoder train modes
    decoder_steps  the decoder-step proxy (instrumented greedy TDT, checked against NeMo's tokens)
    launch.sh  chunked GPU jobs of stage 2b
"""
