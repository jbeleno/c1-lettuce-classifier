"""Heterogeneous soft-voting ensemble. Each backbone wrote its test-set
probabilities to ``models_saved/<name>/test_probs.parquet`` during training;
this module loads them, averages, evaluates, and saves the result the same
way an individual model would. That keeps downstream code (the FastAPI
backend, the comparison table) blind to whether a "model" is a single
backbone or the ensemble."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.config import CLASSES, MODELS_DIR, NUM_CLASSES
from src.utils import configure_logging

log = logging.getLogger(__name__)
NAME = "ensemble_avg"
COMPONENTS = ("mobilenet_v3_small", "efficientnet_b0", "resnet50", "vit_b_16", "swin_t")


def _load_probs(name: str) -> pd.DataFrame | None:
    path = MODELS_DIR / name / "test_probs.parquet"
    if not path.exists():
        log.warning("missing %s -> skipping in ensemble", path)
        return None
    return pd.read_parquet(path)


def main() -> dict | None:
    configure_logging()
    frames = {n: _load_probs(n) for n in COMPONENTS}
    frames = {n: f for n, f in frames.items() if f is not None}
    if len(frames) < 2:
        log.error("need at least 2 component models with test_probs; found %d", len(frames))
        return None

    keys = list(frames.keys())
    ref = frames[keys[0]][["filepath", "true_label"]].reset_index(drop=True)
    prob_cols = [f"p_{c}" for c in CLASSES]
    stacked = np.zeros((len(ref), NUM_CLASSES), dtype=np.float64)

    for name in keys:
        df = frames[name].set_index("filepath").loc[ref["filepath"]].reset_index()
        stacked += df[prob_cols].to_numpy()
    probs = stacked / len(keys)

    y_true = ref["true_label"].map(CLASSES.index).to_numpy()
    y_pred = probs.argmax(1)
    test_acc = float((y_pred == y_true).mean())

    out_dir = MODELS_DIR / NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(probs, columns=prob_cols)
    out_df["filepath"] = ref["filepath"].values
    out_df["true_label"] = ref["true_label"].values
    out_df.to_parquet(out_dir / "test_probs.parquet", index=False)

    report = classification_report(
        y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    metadata = {
        "name": NAME,
        "components": keys,
        "strategy": "soft-voting (mean of softmax outputs)",
        "test_accuracy": test_acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("[%s] components=%s test_acc=%.4f", NAME, keys, test_acc)
    return metadata


if __name__ == "__main__":
    main()
