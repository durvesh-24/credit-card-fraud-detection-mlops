import numpy as np
import pytest

from src.train import best_f1_threshold, compute_metrics


def test_best_f1_threshold_separates_perfectly_ranked_scores():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    thr = best_f1_threshold(y, proba)
    assert 0.3 < thr <= 0.8
    assert compute_metrics(y, proba, thr)["f1"] == pytest.approx(1.0)


def test_compute_metrics_keys_and_ranges():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.05).astype(int)
    proba = np.clip(y * 0.6 + rng.random(500) * 0.5, 0, 1)
    m = compute_metrics(y, proba, 0.5)
    assert set(m) == {"pr_auc", "roc_auc", "precision", "recall", "f1"}
    assert all(0.0 <= v <= 1.0 for v in m.values())


def test_no_positive_predictions_does_not_crash():
    y = np.array([0, 1, 0, 1])
    m = compute_metrics(y, np.array([0.1, 0.2, 0.1, 0.2]), threshold=0.99)
    assert m["precision"] == 0.0 and m["recall"] == 0.0
