"""Load a Cogent model folder and answer requests."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .encoder import TreeEncoder
from .head import READER_D, READER_HEADS, READER_LAYERS, ReaderHead, centre, q_log_softmax
from .layout import TYPES, Question, tokenise

CONFIG_FILE, HEAD_FILE, WEIGHTS_FILE = "cogent.json", "head.safetensors", "model.safetensors"
FORMATS = {"fp32": torch.float32, "bf16": torch.bfloat16}
_ST_FORMATS = {"F32": "fp32", "BF16": "bf16"}


def cpu_has_bf16() -> bool | None:
    """Whether this CPU has native bfloat16 arithmetic (x86 avx512_bf16). None when it cannot be told."""
    try:
        return bool(torch.cpu._is_avx512_bf16_supported())
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            return any("avx512_bf16" in line.split() for line in f if line.startswith("flags"))
    except OSError:
        return None


def weights_format(folder) -> str:
    """fp32 or bf16, read from the header of the folder's model.safetensors."""
    with open(Path(folder) / WEIGHTS_FILE, "rb") as f:
        header = json.loads(f.read(int.from_bytes(f.read(8), "little")))
    found = {info["dtype"] for name, info in header.items() if name != "__metadata__"}
    if len(found) != 1 or next(iter(found)) not in _ST_FORMATS:
        raise ValueError(f"{folder}: expected weights that are all float32 or all bfloat16, found {sorted(found)}")
    return _ST_FORMATS[next(iter(found))]


def _other_format(folder: Path, fmt: str) -> Path | None:
    """The same model in the other format, where the published layout keeps it: fp32 copies sit under
    `fp32/`, mirroring the folders beside it (`X` and `X/fp32`, `X/seed1` and `X/fp32/seed1`)."""
    folder = folder.resolve()
    if fmt == "fp32":
        places = [folder / "fp32", folder.parent / "fp32" / folder.name]
    elif folder.name == "fp32":
        places = [folder.parent]
    elif folder.parent.name == "fp32":
        places = [folder.parent.parent / folder.name]
    else:
        places = []
    for p in places:
        if (p / WEIGHTS_FILE).is_file() and (p / CONFIG_FILE).is_file() and weights_format(p) == fmt:
            return p
    return None


def as_questions(questions) -> list[Question]:
    """Questions from `Question` objects or plain dicts, checked."""
    out = []
    for i, q in enumerate(questions):
        if not isinstance(q, Question):
            missing = [k for k in ("type", "instructions", "options") if k not in q]
            if missing:
                raise ValueError(f"question {i} lacks {missing}")
            q = Question(key=str(q.get("key", f"q{i}")), type=q["type"], instructions=str(q["instructions"]),
                         options=[str(o) for o in q["options"]],
                         descriptions={str(k): str(v) for k, v in (q.get("descriptions") or {}).items() if v is not None})
        if q.type not in TYPES:
            raise ValueError(f"question {q.key!r}: type must be one of {TYPES}, not {q.type!r}")
        if not q.options:
            raise ValueError(f"question {q.key!r} has no options")
        if len(set(q.options)) != len(q.options):
            raise ValueError(f"question {q.key!r} repeats an option id")
        out.append(q)
    if not out:
        raise ValueError("a request needs at least one question")
    if len({q.key for q in out}) != len(out):
        raise ValueError("question keys must be unique within a request")
    return out


def state_text(state) -> str:
    """The state as the model reads it: a string as it is, anything else as JSON."""
    return state if isinstance(state, str) else json.dumps(state)


class Cogent:
    """One model: tokeniser, tree-attention encoder and reader head.

        model = Cogent.from_pretrained("path/to/cogent-large")
        model.decide(state, questions)   # {question key: {option: probability}}
    """

    def __init__(self, tokenizer, backbone, head: ReaderHead, dtype: str = "fp32", path: Path | None = None):
        self.tokenizer, self.dtype, self.path = tokenizer, dtype, path
        self.encoder = TreeEncoder(backbone).eval()
        self.head = head.eval()

    @classmethod
    def from_pretrained(cls, path, dtype: str = "auto", threads: int | None = None) -> "Cogent":
        """Load a model folder (config.json, cogent.json, head.safetensors, model.safetensors, tokenizer files).

        dtype "auto" runs the folder in the format its weights are stored in. One exception: a bf16 folder
        on a CPU without native bfloat16 is slow and rounds differently, so if an fp32 copy sits beside it
        (see the model page's layout) that copy is loaded instead, with a warning.
        dtype "fp32" or "bf16" asks for that format: the folder itself if it holds it, else its copy in
        that format, else an error. Weights are never converted on load.
        threads, when given, is passed to torch.set_num_threads, which is process-wide.
        """
        from transformers import AutoModel, AutoTokenizer

        if dtype not in ("auto", *FORMATS):
            raise ValueError(f"dtype must be auto, fp32 or bf16, not {dtype!r}")
        folder = Path(path).expanduser()
        for name in (WEIGHTS_FILE, CONFIG_FILE, HEAD_FILE, "config.json"):
            if not (folder / name).is_file():
                raise FileNotFoundError(f"{folder} is not a Cogent model folder: no {name}")
        have = weights_format(folder)
        want = have if dtype == "auto" else dtype
        native = cpu_has_bf16() is True
        if dtype == "auto" and have == "bf16" and not native:
            other = _other_format(folder, "fp32")
            if other is not None:
                warnings.warn(f"this CPU has no native bfloat16 (avx512_bf16): loading the fp32 copy {other} instead of {folder}", stacklevel=2)
                folder, have, want = other, "fp32", "fp32"
        if want != have:
            other = _other_format(folder, want)
            if other is None:
                raise ValueError(f"{folder} holds {have} weights and no {want} copy of it was found. Weights are not converted on load")
            folder = other
        if want == "bf16" and not native:
            warnings.warn("running bf16 weights on a CPU without native bfloat16 (avx512_bf16): expect it to be slow, "
                          "and its answers to differ slightly from the reported ones. Use the fp32 weights instead", stacklevel=2)
        if threads is not None:
            torch.set_num_threads(int(threads))

        cfg = json.loads((folder / CONFIG_FILE).read_text())
        if cfg.get("head", "reader") != "reader" or cfg.get("graph", "tree") != "tree":
            raise ValueError(f"{folder}: this package reads the reader head on the tree graph only")
        reader = cfg.get("reader", {})
        head = ReaderHead(cfg["hidden"], reader.get("d", READER_D), reader.get("layers", READER_LAYERS), reader.get("heads", READER_HEADS))
        state = load_file(folder / HEAD_FILE)
        head.load_state_dict({k.removeprefix("head."): v for k, v in state.items()})
        backbone = AutoModel.from_pretrained(folder, attn_implementation="sdpa", dtype=FORMATS[want], local_files_only=True)
        if backbone.config.hidden_size != cfg["hidden"]:
            raise ValueError(f"{folder}: the head expects hidden size {cfg['hidden']}, the backbone has {backbone.config.hidden_size}")
        return cls(AutoTokenizer.from_pretrained(folder, local_files_only=True), backbone, head, want, folder)

    @torch.inference_mode()
    def _logits(self, state, questions):
        qs = as_questions(questions)
        f = self.encoder(tokenise(self.tokenizer, state_text(state), qs))
        return qs, f, self.head(f)

    def scores(self, state, questions) -> dict[str, dict[str, float]]:
        """{question key: {option: logit}}, before the softmax over a question's options. An option's
        logit depends on the state, its question and itself only."""
        qs, f, a = self._logits(state, questions)
        return {q.key: dict(zip(q.options, a[f.qid == i].tolist())) for i, q in enumerate(qs)}

    @torch.inference_mode()
    def decide(self, state, questions) -> dict[str, dict[str, float]]:
        """{question key: {option: probability}}. The probabilities of a question sum to 1."""
        qs, f, a = self._logits(state, questions)
        p = q_log_softmax(centre(a, f.qid, f.nq), f.qid, f.nq).exp()
        return {q.key: dict(zip(q.options, p[f.qid == i].tolist())) for i, q in enumerate(qs)}

    @staticmethod
    def ensemble(models) -> "Ensemble":
        return Ensemble(models)

    def __repr__(self) -> str:
        return f"Cogent({str(self.path) if self.path else 'in memory'}, dtype={self.dtype})"


class Ensemble:
    """Several models as one: the mean of the members' probabilities, option by option."""

    def __init__(self, models):
        self.models = list(models)
        if not self.models:
            raise ValueError("an ensemble needs at least one model")

    @classmethod
    def from_pretrained(cls, paths, dtype: str = "auto", threads: int | None = None) -> "Ensemble":
        return cls(Cogent.from_pretrained(p, dtype, threads) for p in paths)

    def decide(self, state, questions) -> dict[str, dict[str, float]]:
        qs = as_questions(questions)
        outs = [m.decide(state, qs) for m in self.models]
        return {q.key: dict(zip(q.options, np.mean([[o[q.key][x] for x in q.options] for o in outs], axis=0).tolist())) for q in qs}

    def __repr__(self) -> str:
        return f"Ensemble({self.models})"
