"""Group-aware 70/15/15 split with a deterministic seed. Crops that share the
same ``source_frame`` are kept together — putting half the crops of one frame
in train and the other half in test would be leakage."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import PROCESSED_DIR, SPLITS_DIR, SPLIT_SEED
from src.utils import configure_logging, seed_everything

log = logging.getLogger(__name__)


def _group_split(
    df: pd.DataFrame, frac_target: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull off roughly ``frac_target`` of the rows as a held-out set, keeping
    every ``source_frame`` whole and trying to preserve the class distribution.

    Strategy: pick ``k`` such that one of ``k`` stratified-group folds is
    ~``frac_target`` of the data, then peel off fold 0 and return the rest."""
    if not 0.0 < frac_target < 0.5:
        raise ValueError("frac_target must be in (0, 0.5)")
    k = max(2, round(1 / frac_target))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    splits = list(
        sgkf.split(df, y=df["class"], groups=df["source_frame"])
    )
    rest_idx, held_idx = splits[0]
    return df.iloc[rest_idx].copy(), df.iloc[held_idx].copy()


def main() -> dict[str, pd.DataFrame]:
    configure_logging()
    seed_everything(SPLIT_SEED)

    all_csv = PROCESSED_DIR / "all.csv"
    df = pd.read_csv(all_csv)
    n_total = len(df)

    # 1) carve out test ~15%
    rest, test = _group_split(df, frac_target=0.15, seed=SPLIT_SEED)
    # 2) of the remaining 85%, carve out val so that val/(train+val) ≈ 0.176
    #    yields ~15% of original. Equivalent target on the *rest*: 0.15/0.85.
    train, val = _group_split(rest, frac_target=0.15 / 0.85, seed=SPLIT_SEED)

    splits = {"train": train, "val": val, "test": test}

    # Leakage guard
    train_frames = set(train["source_frame"])
    val_frames = set(val["source_frame"])
    test_frames = set(test["source_frame"])
    assert not (train_frames & val_frames), "leak: train/val share source frames"
    assert not (train_frames & test_frames), "leak: train/test share source frames"
    assert not (val_frames & test_frames), "leak: val/test share source frames"

    for name, sub in splits.items():
        sub = sub.copy()
        sub["split"] = name
        out = SPLITS_DIR / f"{name}.csv"
        sub[["filepath", "class", "source_frame", "split"]].to_csv(out, index=False)
        log.info(
            "%s: %d rows (%.1f%%) | %d source frames | class dist:\n%s",
            name,
            len(sub),
            100 * len(sub) / n_total,
            sub["source_frame"].nunique(),
            sub["class"].value_counts().to_string(),
        )
    return splits


if __name__ == "__main__":
    main()
