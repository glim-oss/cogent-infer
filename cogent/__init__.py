"""Cogent: small encoder models that answer typed multi-question decisions.

    from cogent import Cogent
    model = Cogent.from_pretrained("path/to/cogent-large")
    model.decide(state, questions)   # {question key: {option: probability}}
"""
from .layout import Question, render
from .model import Cogent, Ensemble, cpu_has_bf16

__version__ = "0.1.0"
__all__ = ["Cogent", "Ensemble", "Question", "render", "cpu_has_bf16", "__version__"]
