"""Render per-split class-distribution charts.

Produces two figures consumed by the documentation:

* ``reports/class_distribution.png`` — the train/val/test histogram on the
  original (imbalanced) split. Confirms group-aware stratification kept
  the proportions reasonable.
* ``reports/class_distribution_balanced.png`` — the same train column
  before and after sample-level oversampling
  (:mod:`src.data.balance`). Documents that
  ``data/splits/train_balanced.csv`` does what it claims: equal counts per
  class.

Both charts use the slate-900 palette so they drop into the documentation
and slide deck without further styling.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import CLASS_COLORS, CLASSES, REPORTS_DIR, SPLITS_DIR
from src.utils import configure_logging

log = logging.getLogger(__name__)
OUT_PNG = REPORTS_DIR / "class_distribution.png"
OUT_CSV = REPORTS_DIR / "class_distribution.csv"
OUT_BAL_PNG = REPORTS_DIR / "class_distribution_balanced.png"
OUT_BAL_CSV = REPORTS_DIR / "class_distribution_balanced.csv"


# ── shared chart styling ─────────────────────────────────────────────────────


def _style_axes(fig, ax) -> None:
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.tick_params(colors="#CBD5E1")
    ax.yaxis.label.set_color("#CBD5E1")
    ax.xaxis.label.set_color("#CBD5E1")
    ax.title.set_color("#F8FAFC")


def _legend(ax) -> None:
    leg = ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#CBD5E1")
    for text in leg.get_texts():
        text.set_color("#CBD5E1")


# ── per-split histogram ─────────────────────────────────────────────────────


def _render_split_histogram() -> Path:
    frames = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    counts = df.groupby(["split", "class"]).size().unstack(fill_value=0)[CLASSES]
    counts.to_csv(OUT_CSV)
    log.info("counts per split:\n%s", counts.to_string())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    counts.loc[["train", "val", "test"]].plot(
        kind="bar",
        stacked=False,
        ax=ax,
        color=[CLASS_COLORS[c] for c in CLASSES],
        edgecolor="#0F172A",
        width=0.85,
    )
    _style_axes(fig, ax)
    ax.set_title("Class distribution per split (group-aware 70/15/15)")
    ax.set_ylabel("# crops")
    ax.set_xlabel("split")
    _legend(ax)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("chart -> %s", OUT_PNG)
    return OUT_PNG


# ── before/after balancing histogram ────────────────────────────────────────


def _render_balanced_histogram() -> Path | None:
    train_csv = SPLITS_DIR / "train.csv"
    balanced_csv = SPLITS_DIR / "train_balanced.csv"
    if not balanced_csv.exists():
        log.warning(
            "train_balanced.csv not found — run `make balance` to generate it. "
            "Skipping the balanced-distribution chart."
        )
        return None

    raw = pd.read_csv(train_csv)["class"].value_counts().reindex(CLASSES, fill_value=0)
    bal = pd.read_csv(balanced_csv)["class"].value_counts().reindex(CLASSES, fill_value=0)
    counts = pd.DataFrame({"original train": raw, "after balance.py": bal}).T
    counts.to_csv(OUT_BAL_CSV)
    log.info("counts before vs after balancing:\n%s", counts.to_string())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    counts.plot(
        kind="bar",
        ax=ax,
        color=[CLASS_COLORS[c] for c in CLASSES],
        edgecolor="#0F172A",
        width=0.85,
    )
    _style_axes(fig, ax)
    ax.set_title(
        "Sample-level balancing — train set before vs after balance.py "
        "(random oversampling, seed=42)"
    )
    ax.set_ylabel("# crops")
    ax.set_xlabel("")
    _legend(ax)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_BAL_PNG, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("chart -> %s", OUT_BAL_PNG)
    return OUT_BAL_PNG


def main() -> Path:
    configure_logging()
    out = _render_split_histogram()
    _render_balanced_histogram()
    return out


if __name__ == "__main__":
    main()
