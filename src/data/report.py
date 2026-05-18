"""Render a per-split class-distribution chart. The PNG is consumed by the
documentation later and also serves as an at-a-glance sanity check that the
split routine kept the proportions reasonable."""
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


def main() -> Path:
    configure_logging()
    frames = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(SPLITS_DIR / f"{split}.csv")
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    counts = (
        df.groupby(["split", "class"]).size().unstack(fill_value=0)[CLASSES]
    )
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
    ax.set_facecolor("#0F172A")
    fig.patch.set_facecolor("#0F172A")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.tick_params(colors="#CBD5E1")
    ax.yaxis.label.set_color("#CBD5E1")
    ax.xaxis.label.set_color("#CBD5E1")
    ax.title.set_color("#F8FAFC")
    ax.set_title("Class distribution per split (group-aware 70/15/15)")
    ax.set_ylabel("# crops")
    ax.set_xlabel("split")
    leg = ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#CBD5E1")
    for text in leg.get_texts():
        text.set_color("#CBD5E1")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130, facecolor=fig.get_facecolor())
    log.info("chart -> %s", OUT_PNG)
    return OUT_PNG


if __name__ == "__main__":
    main()
