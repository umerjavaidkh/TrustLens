"""Torch-free tests for the ID-vs-OOD reporting logic (the drop exhibit)."""
import numpy as np

from trustlens.training.evaluate import (
    compute_set_metrics, accuracy_drop, EvalReport,
)


def test_compute_set_metrics_perfect():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    preds = np.array([0, 0, 1, 1])
    m = compute_set_metrics("id", y, scores, preds, target_fpr=0.01)
    assert m.accuracy == 1.0
    assert m.auc == 1.0
    assert m.n == 4


def test_accuracy_drop_sign_and_magnitude():
    y = np.array([0, 0, 1, 1])
    id_m = compute_set_metrics("id", y, [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    # OOD: half wrong -> accuracy 0.5
    ood_m = compute_set_metrics("ood", y, [0.1, 0.2, 0.4, 0.3], [0, 0, 0, 0])
    assert id_m.accuracy == 1.0
    assert ood_m.accuracy == 0.5
    assert abs(accuracy_drop(id_m, ood_m) - 0.5) < 1e-9


def test_eval_report_relative_drop():
    y = np.array([0, 1])
    id_m = compute_set_metrics("id", y, [0.2, 0.8], [0, 1])   # acc 1.0
    ood_m = compute_set_metrics("ood", y, [0.6, 0.4], [1, 0])  # acc 0.0
    report = EvalReport(id_test=id_m, ood=ood_m, ood_generator="BigGAN")
    assert report.accuracy_drop == 1.0
    assert report.relative_drop == 1.0
    assert "BigGAN" in report.summary_line()


def test_single_class_set_yields_nan_auc():
    y = np.array([1, 1, 1])
    m = compute_set_metrics("ood", y, [0.7, 0.8, 0.9], [1, 1, 1])
    assert m.accuracy == 1.0
    assert np.isnan(m.auc)
