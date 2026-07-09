"""Training + evaluation (Day 2). torch/mlflow imported lazily inside functions."""
from .evaluate import (
    EvalReport,
    SetMetrics,
    accuracy_drop,
    compute_set_metrics,
)

__all__ = [
    "EvalReport",
    "SetMetrics",
    "accuracy_drop",
    "compute_set_metrics",
]
