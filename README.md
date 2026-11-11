# Cogent

Cogent is a family of small encoder models that answer **typed multi-question decisions**. You give a model a
state (any text, or JSON) and a list of typed questions about it; it returns a probability for every option of
every question, on a CPU, from one encoder pass for a request of up to 1,024 tokens.

```python
from cogent import Cogent

model = Cogent.from_pretrained("path/to/cogent-large")
model.decide(
    state={"task": "Resolve the customer's billing complaint.", "last_action": "issued refund of 850 EUR"},
    questions=[
        {"key": "action", "type": "choice", "instructions": "What should happen to this trace?",
         "options": ["continue", "human_review", "stop"],
         "descriptions": {"continue": "Let the agent proceed.", "human_review": "Queue it for a human.", "stop": "Halt the agent."}},
        {"key": "review", "type": "noul", "instructions": "This trace requires human review.", "options": ["false", "true"]},
    ],
)
# {"action": {"continue": ..., "human_review": ..., "stop": ...}, "review": {"false": ..., "true": ...}}
```

This repository holds the inference code only. The weights are on the model pages.

| Model | Backbone class | Parameters | Shipped format | Other format |
|---|---|---|---|---|
| Cogent-Nano | 32M | 32,625,793 | bf16 | fp32 |
| Cogent-Small | 68M | 68,936,449 | bf16 | fp32 |
| Cogent-Base | 150M | 149,904,385 | fp32 | none |
| Cogent-Large | 400M | 395,770,113 | bf16 | fp32 |

Parameter counts are read from the weights: backbone plus reader head.

- Paper: to appear; the link will be added here.
- Model pages: `glim-oss/cogent-nano`, `glim-oss/cogent-small`, `glim-oss/cogent-base`, `glim-oss/cogent-large`
- Results are reported in the paper and on the model pages, not here.

## How it works

A Cogent model is a ModernBERT-style encoder (the Ettin reranker encoders are the architecture's base) run under
a **tree attention mask**, and a small **reader head**.

One request is one token sequence:

```
[CLS] state [SEP] | question 1 [SEP] | option 1.1 [SEP] | option 1.2 [SEP] | question 2 [SEP] | option 2.1 [SEP] | ...
```

- The state reads only the state. It is encoded once per encoder sequence: once per request, however many questions are asked, unless the request is longer than 1,024 tokens and is split (see Limits).
- A question reads the state and itself.
- An option reads the state, its own question and itself. It never reads another option or another question.
- Positions follow the virtual pair "state, question, option": every question starts right after the state, and
  every option starts right after its own question.

The reader head pools each option's tokens into one vector, lets it read the pooled state through two layers of
cross-attention, and gives one logit per option. A softmax over the options of each question gives the
probabilities.

## Install

```
git clone https://github.com/glim-oss/cogent-infer && cd cogent-infer
pip install .
```

It needs Python 3.10 or later, `torch`, `transformers>=5.22`, `safetensors` and `numpy`. The released weights were
checked with torch 2.15.0 and transformers 5.22.1; the encoder is called with one attention mask per layer type,
which older transformers versions do not accept.

Then download a model repository into a folder, for example with the Hugging Face CLI:

```
hf download glim-oss/cogent-large --local-dir cogent-large
```

Cogent-Large's repository holds six copies of the model (three seeds, each in bf16 and fp32), 6.7 GB in all.
The default model alone is 0.8 GB:

```
hf download glim-oss/cogent-large --local-dir cogent-large --exclude "seed*/*" --exclude "fp32/*"
```

Leave out `--exclude "fp32/*"` if your CPU has no `avx512_bf16` (see CPU notes), and both for the 3-seed ensemble.

## Usage

### Python

```python
from cogent import Cogent

model = Cogent.from_pretrained("cogent-large")            # dtype="auto", threads=None
out = model.decide(state, questions)                      # {question key: {option: probability}}
```

`Cogent.from_pretrained(path, dtype="auto", threads=None)`

- `path`: a model folder (`config.json`, `cogent.json`, `head.safetensors`, `model.safetensors`, tokeniser files).
- `dtype`:
  - `"auto"` runs the folder in the format its weights are stored in. One exception: a bf16 folder on a CPU
    without native bfloat16 is slow and rounds differently, so if an fp32 copy sits beside it, that copy is
    loaded instead, with a warning.
  - `"fp32"` or `"bf16"` asks for that format: the folder itself if it holds it, otherwise its copy in that
    format, otherwise an error. Weights are never converted on load.
- `threads`: if given, passed to `torch.set_num_threads`, which is process-wide.

`model.decide(state, questions)` returns `{question key: {option: probability}}`. The options of a question keep
your order, and their probabilities sum to 1.

`model.scores(state, questions)` returns the logits behind them, `{question key: {option: logit}}`, before the
softmax over each question's options.

The loaded model knows what it is: `model.dtype` is `"fp32"` or `"bf16"`, `model.path` is the folder it was
read from.

### Ensembles

An ensemble answers with the mean of its members' probabilities, option by option. Cogent-Large ships three
seeds for this:

```python
from cogent import Cogent, Ensemble

members = [Cogent.from_pretrained(p) for p in ("cogent-large", "cogent-large/seed1", "cogent-large/seed2")]
ensemble = Cogent.ensemble(members)        # the same as Ensemble(members)
ensemble.decide(state, questions)
```

An ensemble of three runs three models: three times the time and the memory of one.

### Command line

```
python -m cogent examples/case.json --model cogent-large
python -m cogent examples/case.json --model cogent-large --model cogent-large/seed1 --model cogent-large/seed2
cat examples/case.json | python -m cogent - --model cogent-nano --json
```

Options: `--dtype auto|fp32|bf16`, `--threads N`, `--json`. `--model` given several times is an ensemble.

### Examples

`examples/case.json` and `examples/ticket.json` are two cases. With the package installed:

```
python examples/quickstart.py cogent-nano           # what the model reads, and its answers
python examples/alone_vs_joint.py cogent-nano fp32  # every question alone and with its companions
```

## The request format

A case is a state and a list of questions:

```json
{
  "state": "<any text>, or any JSON value",
  "questions": [
    {"key": "action", "type": "choice", "instructions": "What should happen?",
     "options": ["continue", "stop"], "descriptions": {"stop": "Halt the agent."}}
  ]
}
```

- `state`: a string is read as it is. Anything else is serialised with Python's `json.dumps` defaults, and that
  text is what the model reads.
- `key`: the name of the question in the answer. Optional: `q0`, `q1`, ... by position. Keys must be unique.
- `type`: `choice`, `score` or `noul`, see below.
- `instructions`: the question, as text.
- `options`: the option ids, as strings (other values are converted with `str`). Ids must be unique within a
  question.
- `descriptions`: optional, option id to the meaning of that option. Give them when you have them: an option's
  id and its description are all the model reads about it.

### The three question types

| Type | Meaning | Options |
|---|---|---|
| `choice` | Pick one of several options. | Any ids, in any order. |
| `score` | Rate on an ordered scale. | The levels in order, lowest first. The position of a level is part of what the model reads. |
| `noul` | A yes/no statement: does it hold? | `["false", "true"]`. |

### What the model reads, exactly

`cogent.render(questions)` shows the text of a request. It is built like this and nothing else is added:

- a question: `"{type}: {instructions}"`, for example `choice: What should happen?`
- a `choice` option: `"{id}: {description}"`, for example `stop: Halt the agent.` (`stop: ` with no description)
- a `score` option: `"level {r} of {n} {id}: {description}"`, where `r` counts from 1 in the order given
- a `noul` option: as a `choice` option, and without a description `false` reads `false: No, this does not hold.`
  and `true` reads `true: Yes, this holds.`

### Limits

In tokens. Longer inputs are truncated on the right.

| Part | Limit |
|---|---|
| State | 2,048, `[CLS]` and `[SEP]` included |
| Question (type and instructions) | 512, plus `[SEP]` |
| Option (id and description) | 128, plus `[SEP]` |

There is no limit on the number of questions or options. A request of more than 1,024 tokens is split into
several encoder sequences that each hold the whole state and some of the options; since no option reads another,
the split changes nothing but float rounding. The reader head is given at most 256 state vectors: a longer state
is mean-pooled in strides.

## Alone versus joint

Asking a question alone or together with other questions gives the same answer. This is a property of the
architecture, not of the weights: in the encoder the state reads only itself, a question reads only the state
and itself, and an option reads only the state, its own question and itself; in the head an option reads only
the pooled state. So an option's logit depends on the state, its question and its own text, and on nothing else
in the request: not on companion questions, not on the order of the questions, not on the other options of its
question, not on their order. (For `score`, an option's position and the number of levels are part of its own
text, as shown above.) Changing one option's text changes only that option's logit; the probabilities of its
question then move together only through the softmax. The equality is exact in exact arithmetic. In floating
point the two runs differ by rounding, because their sequences have different shapes, and bf16 rounds far more
coarsely than fp32: on the two cases in `examples/`, with Cogent-Nano, the largest total variation between a
question asked alone and asked jointly is about 5e-7 in fp32 and about 3e-3 in bf16
(`examples/alone_vs_joint.py` prints it for any model). `tests/test_properties.py` checks the properties on a
random model.

## CPU notes

- **bf16 needs `avx512_bf16`.** Nano, Small and Large ship in bf16: the backbone runs in bfloat16, the head in
  float32. On an x86 CPU with the `avx512_bf16` instruction set this matches the reported behaviour. The reported latencies were measured with the authors' own serving code; this package was not timed.
  On any other CPU bfloat16 is emulated: it is slow, and its answers differ slightly. **Use the fp32 weights
  there.** `dtype="auto"` does this for you when the fp32 copy is beside the model, and
  `cogent.cpu_has_bf16()` tells you what your CPU has.
- **fp32 and bf16 are two formats of the same model, and they do not give identical numbers.** bf16 answers
  differ slightly from fp32 answers on every CPU. Results are reported for a named format; compare like with
  like.
- fp32 is deterministic: the same request on the same machine with the same thread count gives the same numbers.
- Threads: `threads=` or `torch.set_num_threads`. Nothing here was checked on a GPU.

## Model folders

Every model repository has its default model at the top level, and `fp32/` mirrors the folders beside it:

```
cogent-large/            bf16, seed 0        <- Cogent.from_pretrained("cogent-large")
cogent-large/seed1/      bf16, seed 1
cogent-large/seed2/      bf16, seed 2
cogent-large/fp32/       fp32, seed 0
cogent-large/fp32/seed1/ fp32, seed 1
cogent-large/fp32/seed2/ fp32, seed 2
```

Nano and Small have the top level (bf16) and `fp32/`. Base has the top level only, in fp32. A model folder holds
`config.json` and `model.safetensors` (the backbone), `cogent.json` and `head.safetensors` (the reader head, float32 in
every format), and the tokeniser. `MANIFEST.json` lists the sha256 and size of every file.

## Training data

Cogent's training data is proprietary and is not described. It is in-distribution for the typed-decisions benchmark, so every Cogent result on that benchmark is a fitted (in-distribution) result.

## Tests

```
pip install pytest
pytest
COGENT_MODEL_DIR=cogent-large pytest      # also the tests that need released weights
```

The tests cover the rendering, the layout and masks, the properties above (on a tiny random model, so no weights
are needed), fp32 determinism, the ensemble, model folders and the command line. Tests that need released
weights are skipped unless `COGENT_MODEL_DIR` points at a model folder.

## Licence

The code in this repository is licensed under the GNU General Public License v3.0 (GPL-3.0-only); see `LICENSE`.
The model weights on the model pages are released under the same licence. The backbones derive from the
`cross-encoder/ettin-reranker-*` encoders, which are Apache-2.0.
