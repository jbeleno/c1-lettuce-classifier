"""Walk through ``models_saved`` and produce a single Markdown + CSV table
comparing every backbone (and the ensemble) on test accuracy and macro-F1.
Used both as a CLI for development and as the source of truth that the
``GET /metrics`` backend endpoint will eventually serve."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import MODELS_DIR, REPORTS_DIR
from src.utils import configure_logging

log = logging.getLogger(__name__)
OUT_CSV = REPORTS_DIR / "model_comparison.csv"
OUT_MD = REPORTS_DIR / "model_comparison.md"


def collect() -> pd.DataFrame:
    rows = []
    for child in sorted(MODELS_DIR.iterdir()):
        meta = child / "metadata.json"
        if not meta.exists():
            continue
        with open(meta) as f:
            data = json.load(f)
        cr = data.get("classification_report", {})
        macro = cr.get("macro avg", {})
        rows.append(
            {
                "model": data["name"],
                "test_accuracy": data.get("test_accuracy"),
                "macro_f1": macro.get("f1-score"),
                "macro_precision": macro.get("precision"),
                "macro_recall": macro.get("recall"),
                "best_val_accuracy": data.get("best_val_accuracy"),
                "components": ",".join(data.get("components", [])) or "—",
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "model",
                "test_accuracy",
                "macro_f1",
                "macro_precision",
                "macro_recall",
                "best_val_accuracy",
                "components",
            ]
        )
    return pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)


def main() -> pd.DataFrame:
    configure_logging()
    df = collect()
    if df is None or df.empty:
        log.warning("no metadata.json files under %s — train something first", MODELS_DIR)
        return df
    df.to_csv(OUT_CSV, index=False)
    with open(OUT_MD, "w") as f:
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
    log.info("comparison saved to %s and %s", OUT_CSV, OUT_MD)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
