"""Torch-free tests for the generator-family OOD split (the core Day-2 logic)."""
from pathlib import Path

import pytest
from PIL import Image

from trustlens.models.splits import (
    Record, index_dataset, list_generators, make_ood_split,
)


def _make_genimage_tree(root: Path, generators, n_per_class=6):
    """Create a tiny GenImage-shaped tree: <gen>/train/{ai,nature}/*.png."""
    for gen in generators:
        for label_dir in ("ai", "nature"):
            d = root / gen / "train" / label_dir
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                Image.new("RGB", (16, 16), (i, i, i)).save(d / f"{i}.png")


@pytest.fixture
def genimage_root(tmp_path):
    _make_genimage_tree(tmp_path, ["BigGAN", "stable_diffusion", "ADM"])
    return tmp_path


def test_index_assigns_label_and_generator(genimage_root):
    recs = index_dataset(str(genimage_root))
    assert len(recs) == 3 * 2 * 6  # 3 gens x 2 classes x 6
    assert set(list_generators(recs)) == {"BigGAN", "stable_diffusion", "ADM"}
    # nature -> 0, ai -> 1
    ai = [r for r in recs if "ai" in Path(r.path).parts]
    nat = [r for r in recs if "nature" in Path(r.path).parts]
    assert all(r.label == 1 for r in ai)
    assert all(r.label == 0 for r in nat)


def test_ood_family_excluded_from_train(genimage_root):
    recs = index_dataset(str(genimage_root))
    split = make_ood_split(recs, ood_generator="BigGAN", seed=0)
    # No BigGAN anywhere except the OOD set.
    for subset in (split.train, split.val, split.id_test):
        assert all(r.generator != "BigGAN" for r in subset)
    assert len(split.ood) > 0
    assert all(r.generator == "BigGAN" for r in split.ood)
    assert "BigGAN" not in split.train_generators


def test_split_is_disjoint_and_covers(genimage_root):
    recs = index_dataset(str(genimage_root))
    split = make_ood_split(recs, ood_generator="ADM", seed=1)
    paths = ([r.path for r in split.train] + [r.path for r in split.val]
             + [r.path for r in split.id_test] + [r.path for r in split.ood])
    assert len(paths) == len(set(paths))          # disjoint
    assert len(paths) == len(recs)                # covers everything


def test_unknown_generator_raises(genimage_root):
    recs = index_dataset(str(genimage_root))
    with pytest.raises(ValueError):
        make_ood_split(recs, ood_generator="DoesNotExist")


def test_max_per_class_per_gen_caps(genimage_root):
    recs = index_dataset(str(genimage_root))
    split = make_ood_split(recs, ood_generator="BigGAN", seed=0,
                           max_per_class_per_gen=3)
    # OOD (BigGAN) has 2 classes x 3 = 6 max.
    assert len(split.ood) <= 6


def test_deterministic_with_seed(genimage_root):
    recs = index_dataset(str(genimage_root))
    a = make_ood_split(recs, ood_generator="ADM", seed=7)
    b = make_ood_split(recs, ood_generator="ADM", seed=7)
    assert [r.path for r in a.train] == [r.path for r in b.train]
