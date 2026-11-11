"""An ensemble is the mean of its members' probabilities."""
import numpy as np
import pytest

from cogent import Cogent, Ensemble
from conftest import tiny_model


def mean_of(outs):
    return {k: {o: float(np.mean([x[k][o] for x in outs])) for o in outs[0][k]} for k in outs[0]}


def test_ensemble_is_the_mean(case):
    members = [tiny_model(s) for s in (0, 1, 2)]
    outs = [m.decide(case["state"], case["questions"]) for m in members]
    assert outs[0] != outs[1]  # different models
    got = Cogent.ensemble(members).decide(case["state"], case["questions"])
    want = mean_of(outs)
    assert got.keys() == want.keys()
    for k in want:
        assert list(got[k]) == list(want[k])
        for o in want[k]:
            assert got[k][o] == pytest.approx(want[k][o], abs=1e-12)
        assert sum(got[k].values()) == pytest.approx(1.0, abs=1e-5)


def test_ensemble_helper_and_class_are_the_same_thing(tiny):
    e = Cogent.ensemble([tiny, tiny])
    assert isinstance(e, Ensemble) and e.models == [tiny, tiny]


def test_ensemble_of_one_is_that_model(tiny, case):
    alone = tiny.decide(case["state"], case["questions"])
    got = Ensemble([tiny]).decide(case["state"], case["questions"])
    assert got == {k: {o: pytest.approx(v, abs=1e-12) for o, v in p.items()} for k, p in alone.items()}


def test_empty_ensemble_is_refused():
    with pytest.raises(ValueError):
        Ensemble([])


def test_released_seeds_ensemble_is_the_mean(model_dir, case):
    folders = [model_dir, model_dir / "seed1", model_dir / "seed2"]
    if not all((f / "model.safetensors").is_file() for f in folders):
        pytest.skip("needs a model folder with seed1/ and seed2/ beside it (Cogent-Large)")
    members = [Cogent.from_pretrained(f) for f in folders]
    outs = [m.decide(case["state"], case["questions"]) for m in members]
    got = Cogent.ensemble(members).decide(case["state"], case["questions"])
    want = mean_of(outs)
    for k in want:
        for o in want[k]:
            assert got[k][o] == pytest.approx(want[k][o], abs=1e-6)  # the members answer twice here: once alone, once inside the ensemble
