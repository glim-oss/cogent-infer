"""Answer a case with one model, or with an ensemble of several.

    python examples/quickstart.py path/to/cogent-large
    python examples/quickstart.py path/to/cogent-large path/to/cogent-large/seed1 path/to/cogent-large/seed2
"""
import json
import sys
from pathlib import Path

from cogent import Cogent, render
from cogent.model import as_questions

case = json.loads((Path(__file__).parent / "ticket.json").read_text())

print("What the model reads for each question:")
for r in render(as_questions(case["questions"])):
    print(" ", r["question"])
    for o in r["options"]:
        print("     ", o)

models = [Cogent.from_pretrained(p) for p in sys.argv[1:]]
if not models:
    sys.exit(__doc__)
model = models[0] if len(models) == 1 else Cogent.ensemble(models)
print(f"\n{model}")
for key, options in model.decide(case["state"], case["questions"]).items():
    best = max(options, key=options.get)
    print(f"  {key}: {best}   " + "  ".join(f"{o}={p:.3f}" for o, p in options.items()))
