"""Shared fixtures: synthetic real/fake images generated once per session."""
from pathlib import Path

import pytest
from PIL import Image

from trustlens.data.fixtures import (
    generate_fixtures,
    save_real_image,
    save_fake_image,
)


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("fixtures")
    generate_fixtures(str(d), n_per_class=15, size=192, seed=7)
    return d


@pytest.fixture(scope="session")
def dataset(fixture_dir):
    """(paths, labels) over the generated fixtures."""
    reals = sorted((fixture_dir / "real").glob("*.jpg"))
    fakes = sorted((fixture_dir / "fake").glob("*.jpg"))
    paths = [str(p) for p in reals] + [str(p) for p in fakes]
    labels = [0] * len(reals) + [1] * len(fakes)
    return paths, labels


@pytest.fixture
def real_image(tmp_path) -> Image.Image:
    p = tmp_path / "r.jpg"
    save_real_image(str(p), seed=1, size=192)
    return Image.open(p)


@pytest.fixture
def fake_image(tmp_path) -> Image.Image:
    p = tmp_path / "f.jpg"
    save_fake_image(str(p), seed=1, size=192)
    return Image.open(p)
