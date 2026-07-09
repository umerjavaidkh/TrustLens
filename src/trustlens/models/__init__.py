"""Deep-learning models and the generator-family split (Day 2)."""
from .splits import (
    Record,
    SplitResult,
    index_dataset,
    index_multi,
    index_kaggle_inputs,
    list_generators,
    make_ood_split,
)

__all__ = [
    "Record",
    "SplitResult",
    "index_dataset",
    "index_multi",
    "index_kaggle_inputs",
    "list_generators",
    "make_ood_split",
]

# torch-dependent symbols (efficientnet, dataset, transforms) are imported
# directly from their modules so that trustlens.models.splits stays usable and
# testable without PyTorch installed.
