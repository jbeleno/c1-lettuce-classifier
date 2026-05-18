import pandas as pd
import pytest

from src.config import CLASSES, SPLITS_DIR


@pytest.fixture(scope="module")
def splits() -> dict[str, pd.DataFrame]:
    out = {}
    for s in ("train", "val", "test"):
        p = SPLITS_DIR / f"{s}.csv"
        if not p.exists():
            pytest.skip("run `make split` first")
        out[s] = pd.read_csv(p)
    return out


def test_no_source_frame_leakage(splits):
    frames = {k: set(v["source_frame"]) for k, v in splits.items()}
    assert not (frames["train"] & frames["val"])
    assert not (frames["train"] & frames["test"])
    assert not (frames["val"] & frames["test"])


def test_split_proportions_within_tolerance(splits):
    total = sum(len(v) for v in splits.values())
    fracs = {k: len(v) / total for k, v in splits.items()}
    # group-aware splitting can drift a couple of points off; widen the band
    assert 0.62 <= fracs["train"] <= 0.78, fracs
    assert 0.08 <= fracs["val"] <= 0.22, fracs
    assert 0.08 <= fracs["test"] <= 0.22, fracs


def test_every_class_present_in_every_split(splits):
    for name, df in splits.items():
        present = set(df["class"].unique())
        missing = set(CLASSES) - present
        assert not missing, f"{name} is missing class(es): {missing}"


def test_disjoint_filepaths(splits):
    paths = {k: set(v["filepath"]) for k, v in splits.items()}
    assert not (paths["train"] & paths["val"])
    assert not (paths["train"] & paths["test"])
    assert not (paths["val"] & paths["test"])
