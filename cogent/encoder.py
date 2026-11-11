"""The tree-attention encoder: a stock ModernBERT-style backbone run under the tree masks."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layout import MAX_STATE_POOL, OPTION, SEQ_CAP, Request, Seq, build_seq, plan, tree_masks


@dataclass
class Pools:
    """What the reader head is given for one request. Options of a question are contiguous."""
    c: torch.Tensor  # [N,h] one pooled vector per option, float32
    qid: torch.Tensor  # [N] question of each option
    kind: torch.Tensor  # [Q] 0 choice, 1 score, 2 noul
    E: torch.Tensor  # [1,L,h] the pooled state, float32
    E_mask: torch.Tensor  # [1,L] bool

    @property
    def nq(self) -> int:
        return self.kind.shape[0]


def pool_state(h: torch.Tensor, n: int = MAX_STATE_POOL) -> torch.Tensor:
    """Stride-pool state tokens [L,h] to at most n, by window means."""
    stride = -(-h.shape[0] // n)
    return F.avg_pool1d(h.T[None], stride, stride, ceil_mode=True)[0].T if stride > 1 else h


class TreeEncoder(nn.Module):
    """A request in, pooled hidden states out. A request longer than `cap` tokens is split into several
    sequences, each with the whole state, which run as one padded batch of at most `seq_batch`."""

    def __init__(self, backbone, cap: int | None = SEQ_CAP, seq_batch: int = 16):
        super().__init__()
        self.backbone, self.cap, self.seq_batch = backbone, cap, seq_batch
        self.pad_id = backbone.config.pad_token_id or 0

    def _masked(self, seqs: list[Seq]) -> list[torch.Tensor]:
        n, t = len(seqs), max(len(s.ids) for s in seqs)
        ids = torch.full((n, t), self.pad_id)
        pos = torch.zeros((n, t), dtype=torch.long)
        A = torch.eye(t, dtype=torch.bool).repeat(n, 1, 1)  # padding reads itself, so attention stays finite
        L = A.clone()
        for b, s in enumerate(seqs):
            m = len(s.ids)
            ids[b, :m], pos[b, :m] = s.ids, s.pos
            A[b, :m, :m], L[b, :m, :m] = tree_masks(s, self.backbone.config.sliding_window)
        masks = {"full_attention": A[:, None], "sliding_attention": L[:, None]}
        h = self.backbone(input_ids=ids, attention_mask=masks, position_ids=pos).last_hidden_state
        return [h[b, :len(s.ids)] for b, s in enumerate(seqs)]

    def _run(self, seqs: list[Seq]) -> list[torch.Tensor]:
        out = []
        for i in range(0, len(seqs), self.seq_batch):
            out += self._masked(seqs[i:i + self.seq_batch])
        return [h.float() for h in out]  # the pools and the head are float32, whatever the backbone runs in

    def forward(self, req: Request) -> Pools:
        offs = list(accumulate((len(o) for o in req.options), initial=0))
        seqs = [build_seq(req, p, offs) for p in plan(req, self.cap)]
        return self._pool(req, seqs, self._run(seqs))

    @staticmethod
    def _pool(req: Request, seqs: list[Seq], hidden: list[torch.Tensor]) -> Pools:
        d, N = hidden[0].shape[1], sum(len(o) for o in req.options)
        c, counts = hidden[0].new_zeros(N, d), torch.zeros(N)
        for s, h in zip(seqs, hidden):
            opt = s.kind == OPTION
            c = c.index_add(0, s.oi[opt], h[opt])
            counts.index_add_(0, s.oi[opt], torch.ones(int(opt.sum())))
        E = pool_state(hidden[0][:seqs[0].ls])  # the state reads only itself: the same in every sequence, up to rounding
        E = E[None].clone(memory_format=torch.contiguous_format)  # a pooled state comes out strided, and a strided input rounds differently in the head
        qid = torch.tensor([i for i, o in enumerate(req.options) for _ in o])
        kind = torch.tensor([0 if k == "choice" else 1 if k == "score" else 2 for k in req.kinds])
        return Pools(c / counts[:, None], qid, kind, E, torch.ones(1, E.shape[1], dtype=torch.bool))
