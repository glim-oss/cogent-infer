"""Answer one case from the command line.

    python -m cogent case.json --model path/to/cogent-large
    python -m cogent case.json --model A --model B --model C     # an ensemble: the mean of the three
    cat case.json | python -m cogent - --model path/to/cogent-nano --json

A case is JSON: {"state": <text or any JSON value>, "questions": [{"key", "type", "instructions", "options", "descriptions"}]}.
"""
from __future__ import annotations

import argparse
import json
import sys

from .model import Cogent, Ensemble, as_questions


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m cogent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", help="case JSON file, or - for stdin")
    ap.add_argument("--model", action="append", required=True, metavar="DIR", help="a model folder; give it several times for an ensemble")
    ap.add_argument("--dtype", choices=("auto", "fp32", "bf16"), default="auto")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: torch's own)")
    ap.add_argument("--json", action="store_true", help="print the probabilities as JSON")
    a = ap.parse_args(argv)

    with (sys.stdin if a.case == "-" else open(a.case)) as f:
        case = json.load(f)
    questions = as_questions(case["questions"])
    models = [Cogent.from_pretrained(m, a.dtype, a.threads) for m in a.model]
    model = models[0] if len(models) == 1 else Ensemble(models)
    out = model.decide(case["state"], questions)
    if a.json:
        print(json.dumps(out, indent=1))
        return
    print(f"{model}\n")
    for q in questions:
        p = out[q.key]
        best = max(p, key=p.get)
        print(f"[{q.type}] {q.key}: {q.instructions}")
        width = max(len(o) for o in q.options)
        for o in q.options:
            print(f"  {'>' if o == best else ' '} {o:<{width}}  {p[o]:.3f}  {'#' * round(p[o] * 30)}")
        print()


if __name__ == "__main__":
    main()
