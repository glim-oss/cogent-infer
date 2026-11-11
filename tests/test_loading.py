"""Model folders: what from_pretrained reads, and which format it picks. Folders are written here from tiny random models."""
import json
import shutil
import warnings

import pytest
import torch
from safetensors.torch import save_file

import cogent.model as M
from cogent import Cogent
from cogent.__main__ import main
from cogent.model import weights_format
from conftest import EXAMPLE, HIDDEN, tiny_backbone, tiny_head

WORDS = sorted({w for t in json.loads(EXAMPLE.read_text())["questions"] for w in (t["instructions"] + " " + " ".join(t.get("descriptions", {}).values())).split()})


def word_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[CLS]": 0, "[SEP]": 1, "[UNK]": 2, "[PAD]": 3} | {w: i + 4 for i, w in enumerate(WORDS)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tk, cls_token="[CLS]", sep_token="[SEP]", unk_token="[UNK]", pad_token="[PAD]")


def write_folder(path, fmt="fp32", seed=0):
    """A model folder as the released ones are laid out."""
    path.mkdir(parents=True, exist_ok=True)
    backbone, head, tok = tiny_backbone(seed), tiny_head(seed), word_tokenizer()
    (backbone.to(torch.bfloat16) if fmt == "bf16" else backbone).save_pretrained(path)
    tok.save_pretrained(path)
    (path / "cogent.json").write_text(json.dumps({"format_version": 1, "name": "tiny", "hidden": HIDDEN, "graph": "tree", "head": "reader",
                                                  "reader": {"d": 16, "layers": 2, "heads": 4}, "weights": fmt, "seed": seed}))
    save_file({f"head.{k}": v for k, v in head.state_dict().items()}, path / "head.safetensors")
    return Cogent(tok, tiny_backbone(seed), head, "fp32")


def test_a_saved_folder_answers_as_the_model_in_memory(tmp_path, case):
    memory = write_folder(tmp_path / "m")
    assert weights_format(tmp_path / "m") == "fp32"
    loaded = Cogent.from_pretrained(tmp_path / "m")
    assert loaded.dtype == "fp32" and loaded.path == tmp_path / "m"
    assert loaded.decide(case["state"], case["questions"]) == memory.decide(case["state"], case["questions"])


def test_formats_and_their_copies(tmp_path, case, monkeypatch):
    root = tmp_path / "m"
    write_folder(root, "bf16"), write_folder(root / "fp32", "fp32")
    write_folder(root / "seed1", "bf16", 1), write_folder(root / "fp32" / "seed1", "fp32", 1)
    assert weights_format(root) == "bf16" and weights_format(root / "fp32") == "fp32"

    monkeypatch.setattr(M, "cpu_has_bf16", lambda: True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing to warn about on a CPU with native bfloat16
        auto = Cogent.from_pretrained(root)
        assert (auto.dtype, auto.path) == ("bf16", root)
        assert Cogent.from_pretrained(root, dtype="fp32").path == root / "fp32"
        assert Cogent.from_pretrained(root / "fp32", dtype="bf16").path == root
        assert Cogent.from_pretrained(root / "seed1", dtype="fp32").path == root / "fp32" / "seed1"
        assert Cogent.from_pretrained(root / "fp32" / "seed1", dtype="bf16").path == root / "seed1"
        assert Cogent.from_pretrained(root / "fp32").dtype == "fp32"
    p = auto.decide(case["state"], case["questions"])
    assert all(abs(sum(x.values()) - 1) < 1e-5 for x in p.values())  # the bf16 path runs, and its head is float32

    monkeypatch.setattr(M, "cpu_has_bf16", lambda: False)
    with pytest.warns(UserWarning, match="fp32 copy"):
        fallback = Cogent.from_pretrained(root)
    assert (fallback.dtype, fallback.path) == ("fp32", root / "fp32")
    with pytest.warns(UserWarning, match="without native bfloat16"):
        assert Cogent.from_pretrained(root, dtype="bf16").dtype == "bf16"


def test_weights_are_never_converted(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "cpu_has_bf16", lambda: True)
    write_folder(tmp_path / "only32")
    with pytest.raises(ValueError, match="no bf16 copy"):
        Cogent.from_pretrained(tmp_path / "only32", dtype="bf16")
    write_folder(tmp_path / "only16", "bf16")
    with pytest.raises(ValueError, match="no fp32 copy"):
        Cogent.from_pretrained(tmp_path / "only16", dtype="fp32")
    monkeypatch.setattr(M, "cpu_has_bf16", lambda: False)
    with pytest.warns(UserWarning, match="without native bfloat16"):  # no fp32 copy to fall back on
        assert Cogent.from_pretrained(tmp_path / "only16").dtype == "bf16"


def test_bad_arguments(tmp_path):
    with pytest.raises(FileNotFoundError):
        Cogent.from_pretrained(tmp_path / "nothing")
    write_folder(tmp_path / "m")
    with pytest.raises(ValueError):
        Cogent.from_pretrained(tmp_path / "m", dtype="int8")
    shutil.copytree(tmp_path / "m", tmp_path / "other")
    cfg = json.loads((tmp_path / "other" / "cogent.json").read_text()) | {"head": "something else"}
    (tmp_path / "other" / "cogent.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        Cogent.from_pretrained(tmp_path / "other")


def test_threads_argument(tmp_path):
    write_folder(tmp_path / "m")
    before = torch.get_num_threads()
    try:
        Cogent.from_pretrained(tmp_path / "m", threads=1)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)


def test_command_line(tmp_path, case, capsys):
    memory = write_folder(tmp_path / "m")
    write_folder(tmp_path / "n", seed=1)
    main([str(EXAMPLE), "--model", str(tmp_path / "m"), "--json"])
    assert json.loads(capsys.readouterr().out) == memory.decide(case["state"], case["questions"])
    main([str(EXAMPLE), "--model", str(tmp_path / "m"), "--model", str(tmp_path / "n")])
    out = capsys.readouterr().out
    assert out.startswith("Ensemble(") and "[choice] action" in out and "[noul] review" in out and "[score] risk" in out
