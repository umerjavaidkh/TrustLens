"""Evaluation: in-distribution vs. held-out-generator (OOD) performance.

The headline Day-2 number is the **OOD accuracy drop** — how much accuracy falls
when the model meets a generator family it never trained on. That gap is the
distribution-shift exhibit.

The pure-python `accuracy_drop` / `EvalReport` helpers are torch-free so the
reporting logic is unit-testable without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

from ..metrics import evaluate_at_fpr, roc_auc


@dataclass
class SetMetrics:
    name: str
    n: int
    accuracy: float
    auc: float
    recall_at_fpr: float
    target_fpr: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_set_metrics(name: str, y_true, scores, preds,
                        target_fpr: float = 0.01) -> SetMetrics:
    """Metrics for one evaluation set. `scores` = P(fake); `preds` = argmax label."""
    y_true = np.asarray(y_true)
    preds = np.asarray(preds)
    scores = np.asarray(scores, dtype=float)
    acc = float((preds == y_true).mean()) if len(y_true) else 0.0
    # AUC/recall need both classes present.
    if len(np.unique(y_true)) == 2:
        auc = roc_auc(y_true, scores)
        rec = evaluate_at_fpr(y_true, scores, target_fpr=target_fpr).recall
    else:
        auc, rec = float("nan"), float("nan")
    return SetMetrics(name, len(y_true), acc, auc, rec, target_fpr)


def accuracy_drop(id_metrics: SetMetrics, ood_metrics: SetMetrics) -> float:
    """Absolute accuracy drop from in-distribution to OOD (>=0 means OOD is worse)."""
    return float(id_metrics.accuracy - ood_metrics.accuracy)


@dataclass
class EvalReport:
    id_test: SetMetrics
    ood: SetMetrics
    ood_generator: str

    @property
    def accuracy_drop(self) -> float:
        return accuracy_drop(self.id_test, self.ood)

    @property
    def relative_drop(self) -> float:
        if self.id_test.accuracy == 0:
            return float("nan")
        return self.accuracy_drop / self.id_test.accuracy

    def as_dict(self) -> Dict[str, object]:
        return {
            "ood_generator": self.ood_generator,
            "id_test": self.id_test.as_dict(),
            "ood": self.ood.as_dict(),
            "accuracy_drop": self.accuracy_drop,
            "relative_drop": self.relative_drop,
        }

    def summary_line(self) -> str:
        return (
            f"ID acc={self.id_test.accuracy:.3f}  "
            f"OOD[{self.ood_generator}] acc={self.ood.accuracy:.3f}  "
            f"drop={self.accuracy_drop:.3f} ({self.relative_drop:.1%})"
        )


# ---------------------------------------------------------------- torch inference
def predict_loader(model, loader, device: str = "cpu"):
    """Run a model over a DataLoader, returning (y_true, fake_scores, preds)."""
    import torch  # local import keeps this module importable without torch

    model.eval()
    ys: List[int] = []
    scores: List[float] = []
    preds: List[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)[:, 1]  # P(fake)
            scores.extend(prob.cpu().numpy().tolist())
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            ys.extend(yb.numpy().tolist())
    return np.array(ys), np.array(scores), np.array(preds)


def evaluate_id_vs_ood(model, id_loader, ood_loader, ood_generator: str,
                       device: str = "cpu", target_fpr: float = 0.01) -> EvalReport:
    y_id, s_id, p_id = predict_loader(model, id_loader, device)
    y_ood, s_ood, p_ood = predict_loader(model, ood_loader, device)
    return EvalReport(
        id_test=compute_set_metrics("id_test", y_id, s_id, p_id, target_fpr),
        ood=compute_set_metrics("ood", y_ood, s_ood, p_ood, target_fpr),
        ood_generator=ood_generator,
    )
