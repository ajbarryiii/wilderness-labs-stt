# Fused joint + TDT loss (`tdt_fused.py`)

NeMo's fused joint (`RNNTJoint.forward` with `fuse_loss_wer`) calls the TDT loss once per sub-batch
of `fused_batch_size` rows (4 for Parakeet-TDT), and every sub-batch keeps its FP32
`[rows, T, U+1, 1030]` gradient until `loss.backward()`. `tdt_fused` computes the same loss and
gradients in one call per batch without that tensor:

- the joint runs in chunks of at most `CHUNK_CELLS` lattice cells (16384), each reduced at once by a
  Triton kernel to per-cell scalars (log-sum-exp, blank and target log-probabilities, duration
  log-softmax);
- one Triton launch each runs the alpha and the beta recursion of every row (NeMo's TDT kernels, or
  the plain RNN-T ones on the omega draw);
- the backward pass recomputes each chunk's logits, turns them into the gradient in place and
  accumulates the joint's weight, bias and input gradients.

Memory grows with B x T x (U+1) scalars (about 92 bytes per lattice cell) instead of 1030 values per
cell. The projections, the prediction network and the encoder stay in PyTorch.

## Use: a seam on `finetune/parakeet-ternary/train.py`

`train.py` and the rest of `finetune/parakeet-ternary` are never modified (their hashes are pinned
by the M1 provenance). A consuming trainer installs the seam, the pattern of
`finetune/relabel/training/seam.py`:

    import tdt_fused
    with tdt_fused.installed():          # or add tdt_fused.replacements() to the consumer's own seam
        train.main(argv)

`replacements()` replaces `train.compute_losses` with a wrapper around the original that answers only
the joint's fused loss call (`student.joint(..., compute_wer=False)`) with `tdt_fused.joint_loss`,
and adds `tdt-fused/tdt_fused.py` to `train.TRAINING_SOURCES`, so its SHA-256 is recorded with the
other sources at the start, every resume and finishing. Install it after the consumer's own
replacements of `TRAINING_SOURCES`, so it extends them. `tdt_fused.settings()` is the record for the
consumer's configuration; the consumer should make it part of its resume key, so a run never
resumes with the seam switched. Without the seam the trainer is unchanged.

It refuses to combine with another replacement of `train.compute_losses` or of any method of the
joint or the TDT loss (they would be bypassed), and with what it does not implement: joint dropout
p > 0, `masking_prob`, adapters, an explicit joint `log_softmax`, FastEmit, gradient clamping.

## Semantics

The loss and gradients are NeMo's: the same transitions and duration-0 rules, sigma, the joint's
`RNNTLoss.reduce` (mean_volume for Parakeet-TDT-0.6B-v2), under BF16 autocast the joint in BF16 as NeMo runs it
and the loss in FP32. omega draws once per loss call, as in NeMo, so with one call per batch the
plain RNN-T branch is chosen for a whole batch (about 10% of steps) instead of per 4-row
sub-batch: the same expected loss with a larger step-to-step variance.

## Tests

    finetune/tdt-fused/python -m unittest discover -s finetune/tdt-fused/tests -v     # GPU

`tests/test_tdt_fused.py` compares the loss and every gradient with NeMo's fused joint and numba
loss (and with `TDTLossPytorch` on small shapes), with tolerances derived from the reference's own
noise floor and negative controls; it also checks the peak-memory slope per lattice cell, the seam
itself and `train.compute_losses` / `train.train_step` on a tiny decoder-only model.
