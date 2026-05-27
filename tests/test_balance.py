"""Tests for the sample-level balancing step (src/data/balance.py)."""
import subprocess
import sys

import pandas as pd
import pytest

from src.config import CLASSES, SPLITS_DIR


BALANCED_CSV = SPLITS_DIR / "train_balanced.csv"


@pytest.fixture(scope="module")
def balanced_df() -> pd.DataFrame:
    if not BALANCED_CSV.exists():
        pytest.skip("run `make balance` first")
    return pd.read_csv(BALANCED_CSV)


def test_all_5_classes_present(balanced_df):
    assert set(balanced_df["class"].unique()) == set(CLASSES)


def test_all_classes_have_same_count(balanced_df):
    counts = balanced_df["class"].value_counts()
    assert counts.nunique() == 1, (
        f"sample-level balancing failed — classes have different counts: "
        f"{counts.to_dict()}"
    )


def test_target_is_majority_count(balanced_df):
    """The balancer brings every class up to the count of the originally
    majority class — no downsampling, no data loss."""
    train = pd.read_csv(SPLITS_DIR / "train.csv")
    target = int(train["class"].value_counts().max())
    per_class = balanced_df["class"].value_counts().iloc[0]
    assert per_class == target, f"expected {target} per class, got {per_class}"


def test_reproducible():
    """Running balance.py twice produces a bit-identical CSV."""
    if not BALANCED_CSV.exists():
        pytest.skip("run `make balance` first")
    before = BALANCED_CSV.read_bytes()
    subprocess.run(
        [sys.executable, "-m", "src.data.balance"],
        check=True,
        capture_output=True,
    )
    after = BALANCED_CSV.read_bytes()
    assert before == after, "balance.py is not deterministic under SPLIT_SEED"


def test_columns_preserved(balanced_df):
    assert {"filepath", "class", "source_frame", "split"} <= set(balanced_df.columns)
