#!/usr/bin/env python3
"""Generate synthetic real/fake fixtures. Usage:

    python scripts/make_fixtures.py --out data/interim/fixtures --n 40
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustlens.data.fixtures import generate_fixtures  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/interim/fixtures")
    ap.add_argument("--n", type=int, default=40, help="images per class")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = generate_fixtures(args.out, n_per_class=args.n, size=args.size, seed=args.seed)
    print(f"Wrote {len(manifest)} images to {args.out} "
          f"({args.n} real + {args.n} fake)")


if __name__ == "__main__":
    main()
