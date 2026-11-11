"""Ask every question of the example cases alone and with its companions, and print how far the answers are apart.

    python examples/alone_vs_joint.py path/to/cogent-nano [fp32|bf16]

The two runs agree up to float rounding (see "Alone versus joint" in the README). The distance printed is the
total variation between the two probability vectors of a question, the largest over the questions of a case.
"""
import json
import sys
from pathlib import Path

from cogent import Cogent

if len(sys.argv) < 2:
    sys.exit(__doc__)
model = Cogent.from_pretrained(sys.argv[1], dtype=sys.argv[2] if len(sys.argv) > 2 else "auto")
print(model)
for path in sorted(Path(__file__).parent.glob("*.json")):
    case = json.loads(path.read_text())
    joint = model.decide(case["state"], case["questions"])
    worst = 0.0
    for q in case["questions"]:
        alone = model.decide(case["state"], [q])[q["key"]]
        worst = max(worst, 0.5 * sum(abs(alone[o] - joint[q["key"]][o]) for o in alone))
    print(f"  {path.name}: largest total variation between alone and joint {worst:.2g}")
