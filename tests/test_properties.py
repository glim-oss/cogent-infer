"""What the masks and the head guarantee, on a tiny random model: the guarantees are structural, so they hold for any weights."""
import copy

import pytest

from cogent.layout import tokenise
from conftest import tiny_model

TOL = 1e-4  # float32 rounding: the padded batch and the sequence length change with the request


def flat(d):
    return {(k, o): v for k, opts in d.items() for o, v in opts.items()}


def test_changing_one_option_text_changes_only_that_option_score(tiny, case):
    base = flat(tiny.scores(case["state"], case["questions"]))
    changed = copy.deepcopy(case["questions"])
    changed[0]["descriptions"]["observe"] = "Something else entirely, and quite a bit longer than before."
    new = flat(tiny.scores(case["state"], changed))
    assert abs(new[("action", "observe")] - base[("action", "observe")]) > 100 * TOL  # the test has power
    for k in base:
        if k != ("action", "observe"):
            assert new[k] == pytest.approx(base[k], abs=TOL)


def test_a_question_alone_answers_as_it_does_with_companions(tiny, case):
    joint = tiny.decide(case["state"], case["questions"])
    for q in case["questions"]:
        alone = tiny.decide(case["state"], [q])
        assert list(alone) == [q["key"]]
        for o, p in alone[q["key"]].items():
            assert p == pytest.approx(joint[q["key"]][o], abs=TOL)


def test_question_order_does_not_matter(tiny, case):
    a = flat(tiny.decide(case["state"], case["questions"]))
    b = flat(tiny.decide(case["state"], case["questions"][::-1]))
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=TOL)


def test_choice_option_order_does_not_matter(tiny, case):
    q = copy.deepcopy(case["questions"][0])
    a = tiny.decide(case["state"], [q])["action"]
    q["options"] = q["options"][::-1]
    b = tiny.decide(case["state"], [q])["action"]
    assert list(b) == list(a)[::-1]
    for o in a:
        assert a[o] == pytest.approx(b[o], abs=TOL)


def test_a_new_option_leaves_the_other_scores_alone(tiny, case):
    q = copy.deepcopy(case["questions"][0])
    base = tiny.scores(case["state"], [q])["action"]
    q["options"].append("escalate")
    q["descriptions"]["escalate"] = "Page the on-call engineer."
    new = tiny.scores(case["state"], [q])["action"]
    for o in base:
        assert new[o] == pytest.approx(base[o], abs=TOL)


def test_the_state_matters(tiny, case):
    a = flat(tiny.scores(case["state"], case["questions"]))
    b = flat(tiny.scores("a completely different state", case["questions"]))
    assert max(abs(a[k] - b[k]) for k in a) > 100 * TOL


def test_probabilities_sum_to_one(tiny, case):
    out = tiny.decide(case["state"], case["questions"])
    assert [list(out[q["key"]]) for q in case["questions"]] == [q["options"] for q in case["questions"]]
    for p in out.values():
        assert sum(p.values()) == pytest.approx(1.0, abs=1e-5)
        assert all(0.0 <= v <= 1.0 for v in p.values())


def test_a_split_request_answers_as_an_unsplit_one(case):
    whole, split = tiny_model(0), tiny_model(0)
    whole.encoder.cap, split.encoder.cap = None, 48
    from cogent.layout import plan
    from cogent.model import as_questions, state_text
    req = tokenise(split.tokenizer, state_text(case["state"]), as_questions(case["questions"]))
    assert len(plan(req, 48)) > 1  # really split, and so really padded into one batch
    a, b = flat(whole.scores(case["state"], case["questions"])), flat(split.scores(case["state"], case["questions"]))
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=TOL)
