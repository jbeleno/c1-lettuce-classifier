"""Read Roboflow's three one-hot ``_classes.csv`` files, validate that every
crop has exactly one positive label, and write a unified
``data/processed/all.csv`` with columns ``filepath, class, source_frame``."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import CLASSES, PROCESSED_DIR, RAW_DIR
from src.utils import configure_logging, source_frame_id

log = logging.getLogger(__name__)
ROBOFLOW_ROOT = RAW_DIR / "lettuce_pallets"
SPLITS = ("train", "valid", "test")
OUT_PATH = PROCESSED_DIR / "all.csv"


def _load_split(split: str) -> pd.DataFrame:
    split_dir = ROBOFLOW_ROOT / split
    csv_path = split_dir / "_classes.csv"
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in CLASSES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing class columns in {csv_path}: {missing}")

    label_sums = df[CLASSES].sum(axis=1)
    bad = df[label_sums != 1]
    if len(bad):
        raise ValueError(
            f"{csv_path}: {len(bad)} rows do not have exactly one positive class"
        )

    df["class"] = df[CLASSES].idxmax(axis=1)
    df["filepath"] = df["filename"].map(lambda n: str(split_dir / n))
    df["source_frame"] = df["filename"].map(source_frame_id)
    df["roboflow_split"] = split
    return df[["filepath", "class", "source_frame", "roboflow_split"]]


def main() -> pd.DataFrame:
    configure_logging()
    frames = [_load_split(s) for s in SPLITS]
    df = pd.concat(frames, ignore_index=True)

    missing = [fp for fp in df["filepath"] if not Path(fp).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} files referenced in CSVs do not exist on disk "
            f"(e.g. {missing[0]})"
        )

    df.to_csv(OUT_PATH, index=False)
    log.info("wrote %d rows -> %s", len(df), OUT_PATH)
    log.info("class counts (raw):\n%s", df["class"].value_counts().to_string())
    log.info(
        "unique source frames: %d (avg %.1f crops/frame)",
        df["source_frame"].nunique(),
        len(df) / df["source_frame"].nunique(),
    )
    return df


if __name__ == "__main__":
    main()
