#!/usr/bin/env python3
"""Train the Day-2 EfficientNet-B0 deepfake classifier with a generator-family
OOD hold-out. Runs on GPU (Colab/Kaggle) or CPU (tiny smoke runs).

    python scripts/train_deepfake.py \
        --data-root /content/genimage \
        --ood-generator BigGAN \
        --head-epochs 3 --finetune-epochs 5 --batch-size 64 \
        --max-per-class-per-gen 2000

List the generators found in a dataset root without training:

    python scripts/train_deepfake.py --data-root /content/genimage --list-generators
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustlens.models.splits import index_dataset, list_generators  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ood-generator", default=None)
    ap.add_argument("--list-generators", action="store_true")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--head-epochs", type=int, default=3)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--finetune-epochs", type=int, default=5)
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--unfreeze-blocks", type=int, default=2)
    ap.add_argument("--max-per-class-per-gen", type=int, default=2000)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--experiment", default="trustlens-deepfake")
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()

    if args.list_generators:
        gens = list_generators(index_dataset(args.data_root))
        print("Generators found:")
        for g in gens:
            print(f"  - {g}")
        return

    if not args.ood_generator:
        ap.error("--ood-generator is required for training (or use --list-generators)")

    # Import here so --list-generators works without torch/mlflow installed.
    from trustlens.training.train import TrainConfig, train

    cfg = TrainConfig(
        data_root=args.data_root,
        ood_generator=args.ood_generator,
        img_size=args.img_size,
        batch_size=args.batch_size,
        head_epochs=args.head_epochs,
        head_lr=args.head_lr,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        unfreeze_blocks=args.unfreeze_blocks,
        max_per_class_per_gen=args.max_per_class_per_gen,
        target_fpr=args.target_fpr,
        experiment=args.experiment,
        out_dir=args.out_dir,
    )
    _, report, _ = train(cfg)
    print("\nFinal:", report.summary_line())


if __name__ == "__main__":
    main()
