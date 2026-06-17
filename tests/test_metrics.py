"""Unit tests for evaluation + aggregation metrics and Setting-D analysis."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import evaluate
from eval.aggregate import (aggregate_runs, transfer_degradation, format_table)
from analysis.representation import (
    mmd_rbf, domain_classifier_accuracy, prototype_js_divergence)


def test_perfect_prediction_metrics():
    y = np.array([0, 1, 2, 3, 4, 0, 1])
    m = evaluate(y, y)
    assert abs(m["acc"] - 1.0) < 1e-9
    assert abs(m["mf1"] - 1.0) < 1e-9
    assert abs(m["kappa"] - 1.0) < 1e-9


def test_aggregate_mean_std():
    runs = [
        {"ablation": "A1", "best": {"acc": 0.8, "mf1": 0.7, "kappa": 0.6}},
        {"ablation": "A1", "best": {"acc": 0.9, "mf1": 0.8, "kappa": 0.7}},
    ]
    agg = aggregate_runs(runs, ("ablation",))
    s = agg[("A1",)]["mf1"]
    assert abs(s["mean"] - 0.75) < 1e-9
    assert s["n"] == 2
    assert isinstance(format_table(agg, ("ablation",)), str)


def test_transfer_degradation():
    assert abs(transfer_degradation(0.8, 0.6) - 0.2) < 1e-9


def test_mmd_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((100, 8))
    b = rng.standard_normal((100, 8))
    same = mmd_rbf(a, b)
    shifted = mmd_rbf(a, b + 5.0)
    assert shifted > same
    assert same < 0.1


def test_domain_classifier_separable():
    rng = np.random.default_rng(0)
    src = rng.standard_normal((100, 4))
    tgt = rng.standard_normal((100, 4)) + 5.0   # clearly separated
    acc = domain_classifier_accuracy(src, tgt)
    assert acc > 0.9


def test_prototype_js_zero_when_identical():
    q = np.random.default_rng(0).random((20, 8))
    q = q / q.sum(-1, keepdims=True)
    assert prototype_js_divergence(q, q) < 1e-9


if __name__ == "__main__":
    for fn in [test_perfect_prediction_metrics, test_aggregate_mean_std,
               test_transfer_degradation, test_mmd_zero_for_same_distribution,
               test_domain_classifier_separable,
               test_prototype_js_zero_when_identical]:
        fn()
    print("test_metrics OK")
