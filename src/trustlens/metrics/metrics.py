"""Asymmetric-cost evaluation metrics.

Fraud detection cares about *recall at a tolerable false-positive rate*, not
accuracy. A model that flags 99% of images clean scores high accuracy while
missing every fraud. The primary contract here is `evaluate_at_fpr`: pick the
threshold that holds FPR at (or just under) a target, then report the recall and
precision achieved there. Positive class (label 1) = fake/fraud.

Pure-numpy so it is dependency-light and exactly unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np


@dataclass
class EvalResult:
    threshold: float
    target_fpr: float
    fpr: float          # achieved FPR (<= target_fpr)
    recall: float       # TPR at that threshold
    precision: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int(np.sum(y_pred & y_true))
    fp = int(np.sum(y_pred & ~y_true))
    fn = int(np.sum(~y_pred & y_true))
    tn = int(np.sum(~y_pred & ~y_true))
    return tp, fp, fn, tn


def precision_recall_at_threshold(
    y_true, scores, threshold: float
) -> Dict[str, float]:
    """Confusion counts and rates when predicting positive iff score >= threshold."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    y_pred = scores >= threshold
    tp, fp, fn, tn = _confusion(y_true, y_pred)

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": recall, "precision": precision, "fpr": fpr, "f1": f1,
    }


def threshold_for_fpr(y_true, scores, target_fpr: float = 0.01) -> float:
    """Lowest threshold whose FPR is <= target_fpr (i.e. max recall at that cap).

    Sweeps candidate thresholds (each observed score, descending). Lowering the
    threshold only ever raises FPR, so we take the smallest threshold that still
    satisfies the cap.
    """
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)
    if not np.any(~y_true):
        # No negatives -> FPR is undefined/zero for any threshold.
        return float(scores.min())

    # Candidate thresholds: just above each unique score, plus +inf (predict none).
    uniq = np.unique(scores)
    candidates = np.concatenate([uniq, [np.inf]])

    best_thr = np.inf
    best_recall = -1.0
    for thr in candidates:
        stats = precision_recall_at_threshold(y_true, scores, thr)
        if stats["fpr"] <= target_fpr and stats["recall"] >= best_recall:
            # Prefer higher recall; on ties prefer the *lower* threshold (looser).
            if stats["recall"] > best_recall or thr < best_thr:
                best_recall = stats["recall"]
                best_thr = thr
    return float(best_thr)


def evaluate_at_fpr(y_true, scores, target_fpr: float = 0.01) -> EvalResult:
    """The headline metric: recall/precision at a fixed false-positive rate."""
    thr = threshold_for_fpr(y_true, scores, target_fpr=target_fpr)
    s = precision_recall_at_threshold(y_true, scores, thr)
    return EvalResult(
        threshold=thr,
        target_fpr=float(target_fpr),
        fpr=s["fpr"],
        recall=s["recall"],
        precision=s["precision"],
        f1=s["f1"],
        tp=s["tp"], fp=s["fp"], fn=s["fn"], tn=s["tn"],
    )


def roc_points(y_true, scores):
    """Return (fpr, tpr) arrays sorted by threshold for plotting/AUC."""
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)
    thresholds = np.concatenate([[np.inf], np.unique(scores)[::-1]])
    fpr, tpr = [], []
    for thr in thresholds:
        s = precision_recall_at_threshold(y_true, scores, thr)
        fpr.append(s["fpr"])
        tpr.append(s["recall"])
    return np.array(fpr), np.array(tpr)


def pr_points(y_true, scores):
    """Return (recall, precision) arrays for a PR curve."""
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)
    thresholds = np.concatenate([[np.inf], np.unique(scores)[::-1]])
    recall, precision = [], []
    for thr in thresholds:
        s = precision_recall_at_threshold(y_true, scores, thr)
        recall.append(s["recall"])
        precision.append(s["precision"])
    return np.array(recall), np.array(precision)


def roc_auc(y_true, scores) -> float:
    """Trapezoidal AUC from the ROC points."""
    fpr, tpr = roc_points(y_true, scores)
    order = np.argsort(fpr)
    trapezoid = getattr(np, "trapezoid", np.trapz)  # np>=2 renamed trapz
    return float(trapezoid(tpr[order], fpr[order]))
