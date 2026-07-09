"""Dataset indexing + the generator-family OOD split.

Kept deliberately **torch-free** so the split logic (the part that actually
defines the distribution-shift experiment) is unit-testable without a GPU or
even PyTorch installed.

GenImage layout (per generator subset):

    <root>/<generator>/{train,val}/{ai,nature}/*.png|jpg

- `generator` is the fingerprint we split on (e.g. stable_diffusion_v_1_4,
  BigGAN, Midjourney, ADM, glide, wukong, VQDM).
- `nature` = real (label 0), `ai` = fake (label 1).

The OOD experiment: pick one generator family, exclude it from train/val/test,
and evaluate on it separately. The accuracy gap between the in-distribution test
set and this held-out family is the distribution-shift exhibit.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
LABEL_DIRS = {"nature": 0, "real": 0, "ai": 1, "fake": 1}


@dataclass(frozen=True)
class Record:
    path: str
    label: int          # 0 = real/nature, 1 = fake/ai
    generator: str


def _generator_from_path(path: Path, root: Path) -> str:
    """Generator family = first path component under the dataset root."""
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "unknown"


def _label_from_path(path: Path) -> Optional[int]:
    for part in path.parts:
        if part.lower() in LABEL_DIRS:
            return LABEL_DIRS[part.lower()]
    return None


def index_dataset(root: str) -> List[Record]:
    """Walk `root` and return one Record per image with label + generator."""
    root_p = Path(root)
    records: List[Record] = []
    for p in root_p.rglob("*"):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        label = _label_from_path(p)
        if label is None:
            continue
        records.append(Record(str(p), label, _generator_from_path(p, root_p)))
    return records


def index_multi(roots: Sequence[str],
                generator_names: Optional[Sequence[str]] = None) -> List[Record]:
    """Index several roots, one generator per root.

    Use when each generator lives in a separate directory tree (e.g. Kaggle
    mounts each attached dataset under its own /kaggle/input/<slug>/). The
    generator label is `generator_names[i]` if given, else the root's dir name.
    """
    records: List[Record] = []
    for i, root in enumerate(roots):
        rp = Path(root)
        gen = generator_names[i] if generator_names else rp.name
        for p in rp.rglob("*"):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            label = _label_from_path(p)
            if label is None:
                continue
            records.append(Record(str(p), label, gen))
    return records


def index_kaggle_inputs(input_root: str = "/kaggle/input") -> List[Record]:
    """Index every attached Kaggle dataset as its own generator family.

    Treats each immediate subdirectory of `/kaggle/input` as one generator and
    keeps only those that actually contain ai/nature (or real/fake) images, so
    unrelated attached datasets are ignored.
    """
    root = Path(input_root)
    if not root.exists():
        return []
    records: List[Record] = []
    for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        found = 0
        for p in gen_dir.rglob("*"):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            label = _label_from_path(p)
            if label is None:
                continue
            records.append(Record(str(p), label, gen_dir.name))
            found += 1
        # (dirs with no labelled images contribute nothing)
    return records


def list_generators(records: Sequence[Record]) -> List[str]:
    return sorted({r.generator for r in records})


@dataclass
class SplitResult:
    train: List[Record]
    val: List[Record]
    id_test: List[Record]          # in-distribution test (seen generators)
    ood: List[Record]              # held-out generator family
    ood_generator: str
    train_generators: List[str]

    def summary(self) -> Dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "id_test": len(self.id_test),
            "ood": len(self.ood),
        }


def make_ood_split(
    records: Sequence[Record],
    ood_generator: str,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    max_per_class_per_gen: Optional[int] = None,
) -> SplitResult:
    """Hold out one generator family as OOD; split the rest into train/val/test.

    - `ood_generator` is excluded from train/val/id_test entirely.
    - `max_per_class_per_gen` optionally caps images per (generator, label) to fit
      a Colab time budget (GenImage is huge).
    """
    gens = list_generators(records)
    if ood_generator not in gens:
        raise ValueError(
            f"ood_generator '{ood_generator}' not found. Available: {gens}"
        )

    rng = random.Random(seed)

    # Optional balanced subsampling per (generator, label).
    if max_per_class_per_gen is not None:
        buckets: Dict[Tuple[str, int], List[Record]] = {}
        for r in records:
            buckets.setdefault((r.generator, r.label), []).append(r)
        sampled: List[Record] = []
        for recs in buckets.values():
            rng.shuffle(recs)
            sampled.extend(recs[:max_per_class_per_gen])
        records = sampled

    ood = [r for r in records if r.generator == ood_generator]
    in_dist = [r for r in records if r.generator != ood_generator]

    # Shuffle deterministically, then carve val/test out of the in-distribution pool.
    in_dist = list(in_dist)
    rng.shuffle(in_dist)
    n = len(in_dist)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val = in_dist[:n_val]
    id_test = in_dist[n_val:n_val + n_test]
    train = in_dist[n_val + n_test:]

    return SplitResult(
        train=train,
        val=val,
        id_test=id_test,
        ood=ood,
        ood_generator=ood_generator,
        train_generators=[g for g in gens if g != ood_generator],
    )
