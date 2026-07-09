import numpy as np

from trustlens.metrics import (
    evaluate_at_fpr,
    threshold_for_fpr,
    precision_recall_at_threshold,
    roc_auc,
)


def test_perfect_separation_gives_full_recall_at_zero_fpr():
    # Negatives well below positives -> a threshold exists with FPR 0, recall 1.
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.15, 0.05, 0.8, 0.9, 0.85, 0.95])
    res = evaluate_at_fpr(y, s, target_fpr=0.0)
    assert res.recall == 1.0
    assert res.fpr == 0.0
    assert res.fp == 0
    assert res.tp == 4


def test_fpr_cap_is_respected():
    rng = np.random.default_rng(0)
    y = np.array([0] * 500 + [1] * 500)
    s = np.concatenate([rng.normal(0.3, 0.1, 500), rng.normal(0.7, 0.1, 500)])
    for target in (0.01, 0.05, 0.1):
        res = evaluate_at_fpr(y, s, target_fpr=target)
        assert res.fpr <= target + 1e-9


def test_threshold_monotonicity():
    # A looser FPR cap must not reduce recall.
    rng = np.random.default_rng(1)
    y = np.array([0] * 300 + [1] * 300)
    s = np.concatenate([rng.normal(0.4, 0.15, 300), rng.normal(0.6, 0.15, 300)])
    r_strict = evaluate_at_fpr(y, s, target_fpr=0.01).recall
    r_loose = evaluate_at_fpr(y, s, target_fpr=0.10).recall
    assert r_loose >= r_strict


def test_precision_recall_at_threshold_counts():
    y = [0, 1, 1, 0]
    s = [0.2, 0.9, 0.4, 0.6]
    out = precision_recall_at_threshold(y, s, threshold=0.5)
    # predicted positive: idx1(0.9), idx3(0.6) -> tp=1 (idx1), fp=1 (idx3), fn=1 (idx2)
    assert out["tp"] == 1 and out["fp"] == 1 and out["fn"] == 1 and out["tn"] == 1


def test_roc_auc_bounds_and_ordering():
    y = np.array([0, 0, 1, 1])
    good = np.array([0.1, 0.2, 0.8, 0.9])
    rand = np.array([0.5, 0.5, 0.5, 0.5])
    assert roc_auc(y, good) == 1.0
    assert 0.0 <= roc_auc(y, rand) <= 1.0


def test_threshold_for_fpr_returns_finite_or_inf():
    y = np.array([0, 1])
    s = np.array([0.2, 0.8])
    thr = threshold_for_fpr(y, s, target_fpr=0.0)
    assert np.isfinite(thr) or thr == np.inf
