"""How a request becomes the text the model reads."""
import json

import pytest

from cogent import Question, render
from cogent.layout import option_text, question_text
from cogent.model import as_questions, state_text


def test_question_text_carries_its_type():
    q = Question("action", "choice", "What should happen?", ["continue", "stop"])
    assert question_text(q) == "choice: What should happen?"


def test_choice_option_is_id_then_description():
    q = Question("action", "choice", "What should happen?", ["continue", "stop"], {"stop": "Halt the agent."})
    assert option_text(q, 1) == "stop: Halt the agent."
    assert option_text(q, 0) == "continue: "  # no description: the id alone


def test_score_option_carries_its_level():
    q = Question("risk", "score", "How risky?", ["0", "1", "2", "3"], {"0": "Benign.", "3": "Severe."})
    assert [option_text(q, r) for r in range(4)] == ["level 1 of 4 0: Benign.", "level 2 of 4 1: ", "level 3 of 4 2: ", "level 4 of 4 3: Severe."]


def test_noul_gets_default_descriptions():
    q = Question("review", "noul", "This needs review.", ["false", "true"])
    assert [option_text(q, r) for r in range(2)] == ["false: No, this does not hold.", "true: Yes, this holds."]
    own = Question("review", "noul", "This needs review.", ["false", "true"], {"true": "A human must look."})
    assert [option_text(own, r) for r in range(2)] == ["false: No, this does not hold.", "true: A human must look."]


def test_defaults_are_for_noul_only():
    q = Question("k", "choice", "Is it so?", ["false", "true"])
    assert [option_text(q, r) for r in range(2)] == ["false: ", "true: "]


def test_render_of_the_example(case):
    r = render(as_questions(case["questions"]))
    assert [x["key"] for x in r] == ["action", "review", "risk"]
    assert r[0]["question"] == "choice: What should the observability system do with this trace?"
    assert r[0]["options"][3] == "stop: Halt the agent immediately."
    assert r[1]["options"] == ["false: No, this does not hold.", "true: Yes, this holds."]
    assert r[2]["options"][0] == "level 1 of 4 0: Benign: read-only or clearly safe actions."


def test_state_is_text_or_json():
    assert state_text("as it is") == "as it is"
    state = {"task": "x", "steps": [1, 2]}
    assert state_text(state) == json.dumps(state) == '{"task": "x", "steps": [1, 2]}'


def test_questions_from_dicts():
    qs = as_questions([{"type": "choice", "instructions": "Pick.", "options": [1, 2]},
                       {"key": "s", "type": "score", "instructions": "Rate.", "options": ["low", "high"], "descriptions": {"low": "Bad.", "high": None}}])
    assert [q.key for q in qs] == ["q0", "s"]
    assert qs[0].options == ["1", "2"] and qs[0].descriptions == {}
    assert qs[1].descriptions == {"low": "Bad."}


@pytest.mark.parametrize("bad", [
    [],
    [{"type": "choice", "options": ["a"]}],
    [{"type": "rank", "instructions": "x", "options": ["a"]}],
    [{"type": "choice", "instructions": "x", "options": []}],
    [{"type": "choice", "instructions": "x", "options": ["a", "a"]}],
    [{"key": "k", "type": "choice", "instructions": "x", "options": ["a"]}, {"key": "k", "type": "noul", "instructions": "y", "options": ["false", "true"]}],
])
def test_bad_requests_are_refused(bad):
    with pytest.raises(ValueError):
        as_questions(bad)
