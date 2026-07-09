"""Torch Dataset wrapping the torch-free Record list from splits.py."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from .splits import Record


class RecordDataset(Dataset):
    """Serves (image_tensor, label) from a list of Records.

    Robust to unreadable files: returns a black image rather than crashing a
    long training run on one bad JPEG.
    """

    def __init__(self, records: Sequence[Record], transform: Optional[Callable] = None,
                 size: int = 224):
        self.records = list(records)
        self.transform = transform
        self.size = size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        try:
            img = Image.open(rec.path).convert("RGB")
        except Exception:  # noqa: BLE001
            img = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(rec.label, dtype=torch.long)

    @property
    def labels(self):
        return [r.label for r in self.records]
