"""EfficientNet-B0 transfer-learning model for real-vs-fake classification.

Strategy:
  Phase 1 - freeze the ImageNet backbone, train only a fresh 2-class head. Fast,
            stable, and enough to get a decent baseline.
  Phase 2 - unfreeze the last N feature blocks and fine-tune with a low LR so the
            backbone adapts to generator fingerprints without catastrophic drift.

torchvision's EfficientNet exposes `.features` as a Sequential of 9 blocks
(stem=0 ... head-conv=8) and `.classifier` as [Dropout, Linear].
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


def build_model(num_classes: int = 2, pretrained: bool = True,
                dropout: float = 0.3) -> nn.Module:
    """EfficientNet-B0 with a fresh classifier head."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze every feature block; leave the classifier head trainable."""
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, n_blocks: int = 2) -> None:
    """Unfreeze the last `n_blocks` of model.features (plus the head)."""
    n_total = len(model.features)
    unfreeze_from = max(0, n_total - n_blocks)
    for idx, block in enumerate(model.features):
        requires = idx >= unfreeze_from
        for p in block.parameters():
            p.requires_grad = requires
    for p in model.classifier.parameters():
        p.requires_grad = True


def trainable_parameter_count(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
