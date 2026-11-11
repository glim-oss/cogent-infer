"""fp32 is deterministic: the same request gives the same numbers, bit for bit."""
from cogent import Cogent
from cogent.model import weights_format
from conftest import tiny_model


def test_tiny_fp32_is_deterministic(tiny, case):
    first = tiny.decide(case["state"], case["questions"])
    assert all(tiny.decide(case["state"], case["questions"]) == first for _ in range(3))


def test_two_builds_of_the_same_weights_agree(case):
    a, b = tiny_model(7), tiny_model(7)
    assert a.decide(case["state"], case["questions"]) == b.decide(case["state"], case["questions"])


def test_released_fp32_is_deterministic(model_dir, case):
    model = Cogent.from_pretrained(model_dir, dtype="fp32")  # the folder itself, or its fp32 copy
    assert model.dtype == "fp32" and weights_format(model.path) == "fp32"
    first = model.decide(case["state"], case["questions"])
    assert model.decide(case["state"], case["questions"]) == first
    again = Cogent.from_pretrained(model.path, dtype="fp32")
    assert again.decide(case["state"], case["questions"]) == first
    for p in first.values():
        assert abs(sum(p.values()) - 1.0) < 1e-5
