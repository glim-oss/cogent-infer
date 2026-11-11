"""Request rendering, tokenisation and the tree layout.

One encoder sequence per request:

    [CLS] state [SEP] | question 1 [SEP] | option 1.1 [SEP] | option 1.2 [SEP] | question 2 [SEP] | ...

The state reads the state. A question reads the state and itself. An option reads the state, its own
question and itself. Positions follow the virtual pair "state, question, option": every question starts
right after the state, and every option starts right after its own question.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

STATE, QUESTION, OPTION = 0, 1, 2
TYPES = ("choice", "score", "noul")

# Caps, in tokens. MAX_STATE counts [CLS] and the closing [SEP]. A question or an option gets its
# [SEP] on top of its cap. SEQ_CAP is where a long request is split into several encoder sequences,
# and MAX_STATE_POOL is how many state vectors the reader head is given at most.
MAX_STATE, MAX_QUESTION, MAX_OPTION, SEQ_CAP, MAX_STATE_POOL = 2048, 512, 128, 1024, 256
NOUL_DEFAULT = {"false": "No, this does not hold.", "true": "Yes, this holds."}


@dataclass
class Question:
    """One typed question. `options` are option ids in the caller's order; for `score` that order is
    the order of the levels, lowest first. `descriptions` maps an option id to its meaning."""
    key: str
    type: str  # choice | score | noul
    instructions: str
    options: list[str]
    descriptions: dict[str, str] = field(default_factory=dict)


# ---- rendering: what the model reads ------------------------------------------------------------------

def question_text(q: Question) -> str:
    return f"{q.type}: {q.instructions}"


def option_text(q: Question, r: int) -> str:
    o = q.options[r]
    d = q.descriptions.get(o) or (NOUL_DEFAULT.get(o, "") if q.type == "noul" else "")
    level = f"level {r + 1} of {len(q.options)} " if q.type == "score" else ""
    return f"{level}{o}: {d}"


def render(questions: list[Question]) -> list[dict]:
    """The exact text of every question and option, as it is tokenised."""
    return [{"key": q.key, "question": question_text(q), "options": [option_text(q, r) for r in range(len(q.options))]}
            for q in questions]


# ---- tokens -------------------------------------------------------------------------------------------

@dataclass
class Request:
    """Tokens of one request. Options keep the caller's order."""
    state: list[int]
    questions: list[list[int]]
    options: list[list[list[int]]]  # per question, per option
    kinds: list[str]  # per question


def tokenise(tok, state: str, questions: list[Question]) -> Request:
    sep, cls = tok.sep_token_id, tok.cls_token_id
    ids = lambda texts, n: [t + [sep] for t in tok(texts, add_special_tokens=False, truncation=True, max_length=n)["input_ids"]]
    q_text = [question_text(q) for q in questions]
    o_text = [[option_text(q, r) for r in range(len(q.options))] for q in questions]
    return Request(
        state=[cls] + ids([state], MAX_STATE - 2)[0], questions=ids(q_text, MAX_QUESTION),
        options=[ids(t, MAX_OPTION) for t in o_text], kinds=[q.type for q in questions])


@dataclass
class Seq:
    """One encoder sequence."""
    ids: torch.Tensor
    pos: torch.Tensor
    kind: torch.Tensor
    qi: torch.Tensor  # question of each token, -1 in the state
    oi: torch.Tensor  # flat option of each token, -1 outside options
    ls: int  # state length


def plan(req: Request, cap: int | None) -> list[list[tuple[int, list[int]]]]:
    """Split a request into sequences of about `cap` tokens. Options are independent, so any split is
    exact up to float rounding. A sequence always holds at least one option.

    Every sequence re-encodes the state, so k sequences of S state and N other tokens cost about
    k S^2 + 2SN + N^2/k under a dense mask, least at k = N/S. A sequence therefore always gets room
    for S non-state tokens, however long the state: otherwise a long state degenerates into one
    sequence per option and the state is encoded once per option.
    """
    budget = None if cap is None else max(cap - len(req.state), len(req.state))
    seqs, cur, used = [], {}, 0
    for i, opts in enumerate(req.options):
        for j, o in enumerate(opts):
            need = len(o) + (0 if i in cur else len(req.questions[i]))
            if budget is not None and cur and used + need > budget:
                seqs.append(cur)
                cur, used = {}, 0
                need = len(o) + len(req.questions[i])
            cur.setdefault(i, []).append(j)
            used += need
    seqs.append(cur)
    return [list(s.items()) for s in seqs]


def build_seq(req: Request, parts: list[tuple[int, list[int]]], offsets: list[int]) -> Seq:
    ls = len(req.state)
    ids, pos, kind, qi, oi = list(req.state), list(range(ls)), [STATE] * ls, [-1] * ls, [-1] * ls

    def add(tokens, k, q, o, start):
        ids.extend(tokens), pos.extend(range(start, start + len(tokens)))
        kind.extend([k] * len(tokens)), qi.extend([q] * len(tokens)), oi.extend([o] * len(tokens))

    for i, opts in parts:
        lq = len(req.questions[i])
        add(req.questions[i], QUESTION, i, -1, ls)
        for j in opts:
            add(req.options[i][j], OPTION, i, offsets[i] + j, ls + lq)
    t = lambda v: torch.tensor(v, dtype=torch.long)
    return Seq(t(ids), t(pos), t(kind), t(qi), t(oi), ls)


def tree_masks(seq: Seq, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A[i,j]: token i may read token j, and A & (|pos_i - pos_j| <= window) for the backbone's
    sliding-attention layers. `window` is half of the config's local_attention."""
    kind, qi, oi, pos = seq.kind, seq.qi, seq.oi, seq.pos
    same_q = (qi[:, None] == qi[None, :]) & (qi[:, None] >= 0)
    a = (kind[None, :] == STATE) | (same_q & (kind[None, :] == QUESTION)) | ((oi[:, None] == oi[None, :]) & (oi[:, None] >= 0))
    return a, a & ((pos[:, None] - pos[None, :]).abs() <= window)
