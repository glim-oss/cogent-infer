"""Tiny random models, so that most tests need no released weights.

Tests that need real weights read the folder from the environment and skip without it:

    COGENT_MODEL_DIR=path/to/cogent-large pytest
"""
from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import pytest
import torch

from cogent import Cogent
from cogent.head import ReaderHead

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "case.json"
VOCAB, HIDDEN = 256, 32


class ToyTokenizer:
    """Whitespace words hashed into a small vocabulary, behind the one call the package makes."""
    cls_token_id, sep_token_id = 0, 1

    def __call__(self, texts, add_special_tokens=False, truncation=True, max_length=None):
        return {"input_ids": [[4 + zlib.crc32(w.encode()) % (VOCAB - 4) for w in t.split()][:max_length] for t in texts]}


def tiny_backbone(seed: int = 0):
    from transformers import ModernBertConfig, ModernBertModel

    torch.manual_seed(seed)
    cfg = ModernBertConfig(vocab_size=VOCAB, hidden_size=HIDDEN, num_attention_heads=4, intermediate_size=48, num_hidden_layers=4,
                           layer_types=["full_attention", "sliding_attention", "sliding_attention", "full_attention"], local_attention=8,
                           pad_token_id=3, cls_token_id=0, sep_token_id=1, bos_token_id=None, eos_token_id=None)
    model = ModernBertModel(cfg)
    for p in model.parameters():  # wider than the stock init, so that different inputs give clearly different outputs
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.3)
    return model.eval()


def tiny_head(seed: int = 0) -> ReaderHead:
    torch.manual_seed(1000 + seed)
    head = ReaderHead(HIDDEN, d=16, layers=2, heads=4)
    for p in head.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.3)
    return head.eval()


def tiny_model(seed: int = 0, tokenizer=None) -> Cogent:
    return Cogent(tokenizer or ToyTokenizer(), tiny_backbone(seed), tiny_head(seed))


@pytest.fixture(scope="session")
def tiny() -> Cogent:
    return tiny_model(0)


@pytest.fixture(scope="session")
def case() -> dict:
    return json.loads(EXAMPLE.read_text())


@pytest.fixture(scope="session")
def model_dir() -> Path:
    path = os.environ.get("COGENT_MODEL_DIR")
    if not path:
        pytest.skip("needs released weights: set COGENT_MODEL_DIR to a model folder")
    return Path(path)
