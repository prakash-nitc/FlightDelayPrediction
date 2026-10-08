import numpy as np
import pytest

from src.evaluate import best_threshold, compute_metrics


def test_compute_metrics_perfect_separation():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, s, threshold=0.5)
    assert m["accuracy"] == 1.0 and m["f1"] == 1.0 and m["roc_auc"] == 1.0


def test_best_threshold_lands_between_classes():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.6, 0.7])
    t = best_threshold(y, s)
    assert 0.3 < t <= 0.6
    assert compute_metrics(y, s, t)["f1"] == pytest.approx(1.0)
