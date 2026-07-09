#!/usr/bin/env python3
"""Fit the Tier-1 classifier, set its threshold at a target FPR, and save it.

Trains on synthetic fixtures by default (offline). Point --data at a real
dataset directory (expects real/ and fake/ subfolders) once downloaded.

    python scripts/train_baseline.py                 # fixtures
    python scripts/train_baseline.py --data data/raw/140k_faces/.../train

The saved model is loaded automatically by the API when TRUSTLENS_MODEL is set:
    TRUSTLENS_MODEL=models/baseline.joblib uvicorn trustlens.api.main:app
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustlens.data.fixtures import generate_fixtures  # noqa: E402
from trustlens.heuristics.classifier import (  # noqa: E402
    HeuristicClassifier, extract_features, features_to_vector)
from trustlens.metrics import evaluate_at_fpr, threshold_for_fpr, roc_auc  # noqa: E402


def _load_dir(root: Path):
    paths, labels = [], []
    for lab, sub in [(0, "real"), (1, "fake")]:
        for p in sorted((root / sub).rglob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(p)); labels.append(lab)
    return paths, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="dir with real/ and fake/ subfolders")
    ap.add_argument("--out", default="models/baseline.joblib")
    ap.add_argument("--target-fpr", type=float, default=0.01)
    args = ap.parse_args()

    if args.data:
        paths, labels = _load_dir(Path(args.data))
    else:
        print("No --data given; generating synthetic fixtures.")
        manifest = generate_fixtures("data/interim/fixtures", n_per_class=60, seed=0)
        paths = [p for p, _ in manifest]; labels = [l for _, l in manifest]

    y = np.array(labels)
    X = np.array([features_to_vector(extract_features(Image.open(p))) for p in paths])
    print(f"Loaded {len(paths)} images (real={np.sum(y==0)}, fake={np.sum(y==1)})")

    clf = HeuristicClassifier().fit(X, y)
    scores = clf.predict_proba_features(X)
    clf.set_threshold(threshold_for_fpr(y, scores, target_fpr=args.target_fpr))

    res = evaluate_at_fpr(y, scores, target_fpr=args.target_fpr)
    print(f"AUC={roc_auc(y, scores):.3f}  recall@{args.target_fpr:.0%}FPR={res.recall:.3f} "
          f"precision={res.precision:.3f}  threshold={clf.threshold:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    clf.save(args.out)
    print(f"Saved model -> {args.out}")


if __name__ == "__main__":
    main()
