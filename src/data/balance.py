"""Sample-level class balancing for the training split.

Reads ``data/splits/train.csv`` and produces ``data/splits/train_balanced.csv``
where every class has the same number of rows, achieved by **random
oversampling with replacement** of the minority classes up to the count of
the majority class. The validation and test splits are deliberately NOT
balanced — those must remain representative of the true class distribution
so the reported metrics generalize.

This is the *dataset-level* counterpart to the *loss-level* class weighting
applied during training. The two work together:

  - ``balance.py`` (here) — every class sees the same number of rows per
    epoch. Combined with the augmentation pipeline, duplicate rows actually
    see different transformations on every visit, so this is closer to
    "oversampling with synthetic variation" than to literal duplication.
  - ``class_weight = "balanced"`` (in ``_tf_lib`` / ``_torch_lib``) — even
    after this step, sklearn-computed weights are passed to the loss as a
    second-layer safety net.

Reproducibility: all sampling is keyed on ``SPLIT_SEED`` so two runs
produce bit-identical ``train_balanced.csv`` files.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import CLASSES, SPLITS_DIR, SPLIT_SEED
from src.utils import configure_logging

log = logging.getLogger(__name__)
IN_PATH = SPLITS_DIR / "train.csv"
OUT_PATH = SPLITS_DIR / "train_balanced.csv"


def main() -> pd.DataFrame:
    configure_logging()
    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"{IN_PATH} not found — run `make split` first."
        )
    df = pd.read_csv(IN_PATH)

    counts_before = df["class"].value_counts().reindex(CLASSES, fill_value=0)
    target = int(counts_before.max())
    log.info(
        "input counts (target = %d per class):\n%s",
        target,
        counts_before.to_string(),
    )

    rng = np.random.default_rng(SPLIT_SEED)
    parts: list[pd.DataFrame] = []
    for cls in CLASSES:
        cls_df = df[df["class"] == cls]
        n = len(cls_df)
        if n == 0:
            log.warning("class %s has 0 samples in train.csv — skipping", cls)
            continue
        if n < target:
            extra_idx = rng.choice(cls_df.index.to_numpy(), size=target - n, replace=True)
            extras = df.loc[extra_idx].copy()
            cls_df = pd.concat([cls_df, extras], ignore_index=True)
        parts.append(cls_df)

    balanced = (
        pd.concat(parts, ignore_index=True)
        # Shuffle so the trainer doesn't see all of one class then all of
        # another. Seeded for reproducibility.
        .sample(frac=1.0, random_state=SPLIT_SEED)
        .reset_index(drop=True)
    )

    balanced.to_csv(OUT_PATH, index=False)
    counts_after = balanced["class"].value_counts().reindex(CLASSES, fill_value=0)
    assert counts_after.nunique() == 1, "post-condition: every class must end with the same count"
    log.info(
        "wrote %d rows -> %s\noutput counts:\n%s",
        len(balanced),
        OUT_PATH,
        counts_after.to_string(),
    )
    return balanced


if __name__ == "__main__":
    main()
