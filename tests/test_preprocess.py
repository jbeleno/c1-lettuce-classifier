from pathlib import Path

import pandas as pd
import pytest

from src.config import CLASSES, PROCESSED_DIR


@pytest.fixture(scope="module")
def all_df() -> pd.DataFrame:
    p = PROCESSED_DIR / "all.csv"
    if not p.exists():
        pytest.skip("run `make preprocess` first")
    return pd.read_csv(p)


def test_columns(all_df):
    assert set(all_df.columns) >= {"filepath", "class", "source_frame", "roboflow_split"}


def test_single_class_per_row(all_df):
    assert all_df["class"].isin(CLASSES).all()


def test_filepaths_exist(all_df):
    sample = all_df.sample(min(50, len(all_df)), random_state=0)
    missing = [fp for fp in sample["filepath"] if not Path(fp).exists()]
    assert not missing, f"sampled files missing: {missing[:3]}"


def test_no_duplicate_filepaths(all_df):
    assert all_df["filepath"].is_unique
