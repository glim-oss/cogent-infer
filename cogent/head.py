"""The reader head: every option reads the pooled state, and never another option."""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import Pools

READER_D, READER_LAYERS, READER_HEADS = 192, 2, 4


class ReaderLayer(nn.Module):
    """Options read the state: cross-attention then a feed-forward, each with a residual."""

    def __init__(self, d: int, heads: int = READER_HEADS):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.a = nn.MultiheadAttention(d, heads, batch_first=True)
        self.f = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, q: torch.Tensor, ctx: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        q = q + self.a(self.n1(q), ctx, ctx, key_padding_mask=pad, need_weights=False)[0]
        return q + self.f(self.n2(q))


class ReaderHead(nn.Module):
    """logit_o = Linear(reader(po(c_o) + ty(type), ps(E))) on the encoder's pools, standardised per
    dimension with statistics stored in the weights.

    Options read the state and never each other, so neither option order nor companion questions can
    change a logit."""

    def __init__(self, hidden: int, d: int = READER_D, layers: int = READER_LAYERS, heads: int = READER_HEADS):
        super().__init__()
        for pool in "ce":
            self.register_buffer(f"{pool}_mean", torch.zeros(hidden))
            self.register_buffer(f"{pool}_std", torch.ones(hidden))
        self.ps, self.po, self.ty = nn.Linear(hidden, d), nn.Linear(hidden, d), nn.Embedding(3, d)
        self.r = nn.ModuleList(ReaderLayer(d, heads) for _ in range(layers))
        self.b = nn.Linear(d, 1)

    def forward(self, f: Pools) -> torch.Tensor:
        """One logit per option, in the caller's order."""
        c = self.po((f.c - self.c_mean) / self.c_std) + self.ty(f.kind[f.qid])
        H = self.ps((f.E.float() - self.e_mean) / self.e_std)
        q = c[None]  # one request: its options are the queries, its state the context
        for layer in self.r:
            q = layer(q, H, ~f.E_mask)
        return self.b(q[0]).squeeze(-1)


# ---- per-question operations on flat logits ---------------------------------------------------------

def _counts(qid: torch.Tensor, nq: int) -> torch.Tensor:
    return torch.zeros(nq, dtype=torch.float32).index_add_(0, qid, torch.ones_like(qid, dtype=torch.float32))


def centre(x: torch.Tensor, qid: torch.Tensor, nq: int) -> torch.Tensor:
    """Subtract the mean per question. A softmax ignores it; it fixes the gauge of the logits."""
    mean = torch.zeros(nq, dtype=x.dtype).index_add(0, qid, x) / _counts(qid, nq)
    return x - mean[qid]


def q_log_softmax(x: torch.Tensor, qid: torch.Tensor, nq: int) -> torch.Tensor:
    top = torch.full((nq,), -torch.inf, dtype=x.dtype).scatter_reduce(0, qid, x, "amax")
    return x - (torch.zeros(nq, dtype=x.dtype).index_add(0, qid, (x - top[qid]).exp()).log() + top)[qid]
