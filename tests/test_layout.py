"""Tokens, positions, the tree masks and the split of a long request."""
from itertools import accumulate

import torch

from cogent import Question
from cogent.layout import MAX_OPTION, MAX_QUESTION, MAX_STATE, OPTION, QUESTION, STATE, build_seq, plan, tokenise, tree_masks
from conftest import ToyTokenizer

QS = [Question("a", "choice", "which one of these", ["x", "y", "z"], {"x": "the first", "y": "the second one"}),
      Question("b", "noul", "it holds", ["false", "true"])]


def seq_of(state="some state of five words", questions=QS):
    req = tokenise(ToyTokenizer(), state, questions)
    offs = list(accumulate((len(o) for o in req.options), initial=0))
    parts = plan(req, None)
    assert len(parts) == 1
    return req, build_seq(req, parts[0], offs)


def test_tokens_and_separators():
    tok = ToyTokenizer()
    req, _ = seq_of()
    assert req.state[0] == tok.cls_token_id and req.state[-1] == tok.sep_token_id and len(req.state) == 5 + 2
    assert all(q[-1] == tok.sep_token_id for q in req.questions)
    assert all(o[-1] == tok.sep_token_id for opts in req.options for o in opts)
    assert req.kinds == ["choice", "noul"]


def test_caps():
    long = " ".join(["w"] * 5000)
    req = tokenise(ToyTokenizer(), long, [Question("a", "choice", long, ["x"], {"x": long})])
    assert len(req.state) == MAX_STATE  # [CLS] and [SEP] included
    assert len(req.questions[0]) == MAX_QUESTION + 1
    assert len(req.options[0][0]) == MAX_OPTION + 1


def test_positions_follow_the_virtual_pair():
    req, s = seq_of()
    ls = len(req.state)
    assert s.pos[:ls].tolist() == list(range(ls))
    for i in range(len(req.questions)):
        lq = len(req.questions[i])
        q_tokens = (s.kind == QUESTION) & (s.qi == i)
        assert s.pos[q_tokens].tolist() == list(range(ls, ls + lq))  # every question starts right after the state
        for o in s.oi[(s.kind == OPTION) & (s.qi == i)].unique().tolist():
            mine = s.oi == o
            assert s.pos[mine].tolist() == list(range(ls + lq, ls + lq + int(mine.sum())))  # every option right after its question


def test_tree_masks():
    _, s = seq_of()
    full, sliding = tree_masks(s, window=4)
    n = len(s.ids)
    for i in range(n):
        reads = set(full[i].nonzero().squeeze(-1).tolist())
        state = set((s.kind == STATE).nonzero().squeeze(-1).tolist())
        if s.kind[i] == STATE:
            want = state
        elif s.kind[i] == QUESTION:
            want = state | set(((s.kind == QUESTION) & (s.qi == s.qi[i])).nonzero().squeeze(-1).tolist())
        else:
            want = state | set(((s.kind == QUESTION) & (s.qi == s.qi[i])).nonzero().squeeze(-1).tolist()) | set((s.oi == s.oi[i]).nonzero().squeeze(-1).tolist())
        assert reads == want
    assert not (sliding & ~full).any()
    near = (s.pos[:, None] - s.pos[None, :]).abs() <= 4
    assert torch.equal(sliding, full & near)


def test_no_option_reads_another_option():
    _, s = seq_of()
    full, _ = tree_masks(s, window=4)
    opt = s.kind == OPTION
    other = opt[:, None] & opt[None, :] & (s.oi[:, None] != s.oi[None, :])
    assert not (full & other).any()
    assert not full[s.kind == STATE][:, s.kind != STATE].any()  # the state reads only the state


def test_short_request_is_one_sequence():
    req, _ = seq_of()
    assert plan(req, 1024) == [[(0, [0, 1, 2]), (1, [0, 1])]]


def test_long_request_is_split_and_keeps_every_option_once():
    qs = [Question(f"q{i}", "choice", "pick the right one", [f"o{j}" for j in range(6)], {f"o{j}": "a few words here" for j in range(6)}) for i in range(5)]
    req = tokenise(ToyTokenizer(), "a short state", qs)
    parts = plan(req, 40)
    assert len(parts) > 1 and all(p for p in parts)
    seen = [(i, j) for p in parts for i, opts in p for j in opts]
    assert seen == [(i, j) for i in range(5) for j in range(6)]  # each once, in the caller's order
    budget = max(40 - len(req.state), len(req.state))
    for p in parts:
        used = sum(len(req.questions[i]) + sum(len(req.options[i][j]) for j in opts) for i, opts in p)
        assert used <= budget or sum(len(opts) for _, opts in p) == 1
    assert len(plan(req, None)) == 1
