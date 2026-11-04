"""Stand-ins shared by the mutant trainer tests (CPU and CUDA): a loss through the prediction network and joint with
a Python random draw per call (as the fused loss's omega draw), deterministic in-memory batches, held-out items."""
from __future__ import annotations

import random

import torch

import mutant_data as md


def stand_in_loss(model, batch):
    """A differentiable loss through the prediction network and the joint, with one random.uniform draw per call (as
    the fused loss's omega draw) that changes the loss, so a resume that lost Python's RNG state would diverge."""
    dec, _, _ = model.decoder(targets=batch["tokens"], target_length=batch["token_len"])
    f = model.joint.project_encoder(batch["feats"].float().transpose(1, 2))
    g = model.joint.project_prednet(dec.transpose(1, 2))
    z = model.joint.joint_net(f[:, :, None, :] + g[:, None, :, :])
    scale = 1.0 if random.uniform(0, 1) >= 0.5 else 0.5
    return scale * torch.logsumexp(z.float(), dim=-1).mean()


class FakeBatches:
    """Deterministic in-memory batches of a tiny model's dimension, indexed by global step."""

    def __init__(self, dim=64, n_batches=7):
        self.dim, self.n_batches = dim, n_batches

    def batch(self, step):
        g = torch.Generator().manual_seed(1000 + step % self.n_batches)
        b = 3
        t = torch.randint(4, 9, (b,), generator=g)
        u = torch.randint(1, 5, (b,), generator=g)
        feats = torch.randn(b, self.dim, int(t.max()), generator=g).to(torch.bfloat16)
        tokens = torch.randint(0, 100, (b, int(u.max())), generator=g)
        return {"feats": feats, "feat_len": t, "tokens": tokens, "token_len": u, "ids": [f"s{step}-{i}" for i in
                range(b)], "seconds": 3.0}


def heldout_items(dim=64, n=9):
    g = torch.Generator().manual_seed(5)
    texts = ["hello world", "a b c", "", "yes", "the cat sat", "no", "ok then", "fine", "go"]
    return [{"id": f"h{i}", "feats": torch.randn(int(torch.randint(3, 12, (1,), generator=g)), dim,
                                                  generator=g).to(torch.bfloat16),
             "text": texts[i % len(texts)], "source": md.SOURCES[i % 3], "duration": 1.0 + i} for i in range(n)]
