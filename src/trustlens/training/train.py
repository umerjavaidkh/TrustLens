"""Two-phase EfficientNet-B0 training with MLflow tracking.

Phase 1: frozen backbone, train the head (higher LR).
Phase 2: unfreeze the last N blocks, fine-tune (low LR).

Everything is logged to MLflow: config params, per-epoch train/val loss+acc, the
best checkpoint, and finally the ID-vs-OOD evaluation with the accuracy drop.

Designed to run on a Colab/Kaggle GPU in ~2-3h on a GenImage subset. AMP + a
configurable subsample keep it inside that budget.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..models.efficientnet import (
    build_model, freeze_backbone, trainable_parameter_count, unfreeze_last_blocks)
from ..models.dataset import RecordDataset
from ..models.transforms import train_transforms, eval_transforms
from ..models.splits import index_dataset, make_ood_split
from .evaluate import evaluate_id_vs_ood


@dataclass
class TrainConfig:
    data_root: str
    ood_generator: str
    img_size: int = 224
    batch_size: int = 64
    # Phase 1 (frozen backbone).
    head_epochs: int = 3
    head_lr: float = 1e-3
    # Phase 2 (fine-tune last blocks).
    finetune_epochs: int = 5
    finetune_lr: float = 1e-4
    unfreeze_blocks: int = 2
    weight_decay: float = 1e-4
    num_workers: int = 2
    target_fpr: float = 0.01
    max_per_class_per_gen: Optional[int] = 2000   # cap for Colab time budget
    early_stop_patience: int = 3
    seed: int = 42
    experiment: str = "trustlens-deepfake"
    out_dir: str = "models"


@dataclass
class EpochLog:
    epoch: int
    phase: str
    train_loss: float
    val_loss: float
    val_acc: float


def _make_loaders(cfg: TrainConfig):
    import torch
    from torch.utils.data import DataLoader

    records = index_dataset(cfg.data_root)
    if not records:
        raise RuntimeError(f"No labelled images found under {cfg.data_root}")
    split = make_ood_split(
        records, ood_generator=cfg.ood_generator, seed=cfg.seed,
        max_per_class_per_gen=cfg.max_per_class_per_gen,
    )

    tt = train_transforms(cfg.img_size)
    et = eval_transforms(cfg.img_size)
    g = torch.Generator().manual_seed(cfg.seed)

    def dl(records, transform, shuffle):
        ds = RecordDataset(records, transform=transform, size=cfg.img_size)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers, pin_memory=True,
                          generator=g if shuffle else None)

    return split, {
        "train": dl(split.train, tt, True),
        "val": dl(split.val, et, False),
        "id_test": dl(split.id_test, et, False),
        "ood": dl(split.ood, et, False),
    }


def _run_epoch(model, loader, criterion, optimizer, device, scaler, train: bool):
    import torch

    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.split(":")[0],
                                enabled=(device != "cpu")):
                logits = model(xb)
                loss = criterion(logits, yb)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss) * xb.size(0)
            correct += int((logits.argmax(1) == yb).sum())
            n += xb.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def train(cfg: TrainConfig):
    """Full two-phase training + ID/OOD eval. Returns (model, EvalReport, logs)."""
    import torch
    import torch.nn as nn
    import mlflow

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    split, loaders = _make_loaders(cfg)
    model = build_model(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device != "cpu"))

    logs: List[EpochLog] = []
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    best_path = str(Path(cfg.out_dir) / "efficientnet_b0_best.pt")

    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run():
        mlflow.log_params(asdict(cfg))
        mlflow.log_param("device", device)
        mlflow.log_param("train_generators", ",".join(split.train_generators))
        mlflow.log_params({f"n_{k}": v for k, v in split.summary().items()})

        best_val, patience, global_epoch = float("inf"), 0, 0

        def optimize_phase(phase, epochs, lr):
            nonlocal best_val, patience, global_epoch
            trn, tot = trainable_parameter_count(model)
            mlflow.log_metric(f"{phase}_trainable_params", trn)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
            for _ in range(epochs):
                global_epoch += 1
                tl, _ = _run_epoch(model, loaders["train"], criterion, optimizer,
                                   device, scaler, train=True)
                vl, va = _run_epoch(model, loaders["val"], criterion, optimizer,
                                    device, scaler, train=False)
                logs.append(EpochLog(global_epoch, phase, tl, vl, va))
                mlflow.log_metrics({"train_loss": tl, "val_loss": vl, "val_acc": va},
                                   step=global_epoch)
                print(f"[{phase}] epoch {global_epoch} "
                      f"train_loss={tl:.4f} val_loss={vl:.4f} val_acc={va:.4f}")
                if vl < best_val:
                    best_val, patience = vl, 0
                    torch.save(model.state_dict(), best_path)
                else:
                    patience += 1
                    if patience >= cfg.early_stop_patience:
                        print("Early stopping.")
                        break

        # Phase 1: frozen backbone.
        freeze_backbone(model)
        optimize_phase("head", cfg.head_epochs, cfg.head_lr)

        # Phase 2: unfreeze last blocks.
        unfreeze_last_blocks(model, cfg.unfreeze_blocks)
        optimize_phase("finetune", cfg.finetune_epochs, cfg.finetune_lr)

        # Restore best and evaluate ID vs OOD.
        model.load_state_dict(torch.load(best_path, map_location=device))
        report = evaluate_id_vs_ood(
            model, loaders["id_test"], loaders["ood"],
            ood_generator=cfg.ood_generator, device=device, target_fpr=cfg.target_fpr)

        mlflow.log_metrics({
            "id_test_acc": report.id_test.accuracy,
            "id_test_auc": report.id_test.auc,
            "ood_acc": report.ood.accuracy,
            "ood_auc": report.ood.auc,
            "ood_accuracy_drop": report.accuracy_drop,
            "ood_relative_drop": report.relative_drop,
        })
        mlflow.log_artifact(best_path)
        print("\n=== Distribution-shift exhibit ===")
        print(report.summary_line())

    return model, report, logs
