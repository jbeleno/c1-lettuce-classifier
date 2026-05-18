"""K-fold cross-validation. Topic 3 of the syllabus marks K-Fold with three
asterisks — this module covers that requirement on top of the standard hold-out
split. For each model the routine:

  1. Reads ``data/processed/all.csv`` (the unified dataset before split).
  2. Builds ``n_folds`` stratified, group-aware folds (groups = source frame)
     so crops from the same original image never cross fold boundaries.
  3. In every fold: trains the backbone head from scratch for a fixed number
     of epochs and reports test metrics on the held-out fold.
  4. Aggregates fold metrics into mean ± std and writes them to
     ``reports/cv_<model>/``.

We deliberately keep the per-fold training short (head-only, fixed epochs):
CV's job is to estimate variance across folds, not to chase the best score —
that's what ``make train-all`` is for."""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    BATCH_SIZE,
    CLASSES,
    NUM_CLASSES,
    PROCESSED_DIR,
    REPORTS_DIR,
    SPLIT_SEED,
)
from src.utils import configure_logging, seed_everything

log = logging.getLogger(__name__)

MODELS: dict[str, dict] = {
    "mobilenet_v3_small": {
        "framework": "tf",
        "module": "src.models.mobilenet",
        "preprocess_module": "tensorflow.keras.applications.mobilenet_v3",
    },
    "efficientnet_b0": {
        "framework": "tf",
        "module": "src.models.efficientnet",
        "preprocess_module": "tensorflow.keras.applications.efficientnet",
    },
    "resnet50": {
        "framework": "tf",
        "module": "src.models.resnet",
        "preprocess_module": "tensorflow.keras.applications.resnet",
    },
    "vit_b_16": {"framework": "torch", "module": "src.models.vit"},
    "swin_t": {"framework": "torch", "module": "src.models.swin"},
}


@dataclass
class CVConfig:
    model_name: str
    n_folds: int = 5
    epochs: int = 6
    learning_rate: float = 1e-3
    smoke: bool = False

    def __post_init__(self):
        if self.smoke:
            self.n_folds = 2
            self.epochs = 1


# ── Fold metric helpers ───────────────────────────────────────────────────────


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    accuracy = float((y_pred == y_true).mean())
    labels = list(range(NUM_CLASSES))
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
        labels=labels,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    macro = report["macro avg"]
    return {
        "accuracy": accuracy,
        "macro_f1": macro["f1-score"],
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


# ── TensorFlow fold runner ────────────────────────────────────────────────────


def _run_fold_tf(
    cfg: CVConfig, train_df: pd.DataFrame, val_df: pd.DataFrame, fold_idx: int
) -> dict:
    import tensorflow as tf

    from src.models._tf_lib import _make_dataset
    from sklearn.utils.class_weight import compute_class_weight

    spec = MODELS[cfg.model_name]
    model_module = importlib.import_module(spec["module"])
    preprocess = importlib.import_module(spec["preprocess_module"]).preprocess_input

    seed_everything(SPLIT_SEED + fold_idx)

    train_df = train_df.assign(
        label_idx=train_df["class"].map(CLASSES.index).astype(np.int32)
    )
    val_df = val_df.assign(
        label_idx=val_df["class"].map(CLASSES.index).astype(np.int32)
    )

    cap = 200 if cfg.smoke else None
    train_ds = _make_dataset(
        train_df,
        preprocess=preprocess,
        augment=True,
        shuffle=True,
        batch_size=BATCH_SIZE,
        smoke_cap=cap,
    )
    val_ds = _make_dataset(
        val_df,
        preprocess=preprocess,
        augment=False,
        shuffle=False,
        batch_size=BATCH_SIZE,
        smoke_cap=cap,
    )

    model, base = model_module.build_model()
    base.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    y_train = train_df["label_idx"].to_numpy()
    cw_values = compute_class_weight(
        class_weight="balanced", classes=np.arange(NUM_CLASSES), y=y_train
    )
    class_weight = {i: float(w) for i, w in enumerate(cw_values)}

    log.info(
        "[%s][fold %d] train=%d val=%d epochs=%d",
        cfg.model_name,
        fold_idx,
        len(train_df),
        len(val_df),
        cfg.epochs,
    )
    model.fit(
        train_ds,
        epochs=cfg.epochs,
        class_weight=class_weight,
        verbose=2,
    )

    probs = model.predict(val_ds, verbose=0)
    if cap is not None:
        val_df = val_df.head(cap)
    y_true = val_df["label_idx"].to_numpy()
    y_pred = probs.argmax(axis=1)
    tf.keras.backend.clear_session()
    return _fold_metrics(y_true, y_pred)


# ── PyTorch fold runner ───────────────────────────────────────────────────────


def _run_fold_torch(
    cfg: CVConfig, train_df: pd.DataFrame, val_df: pd.DataFrame, fold_idx: int
) -> dict:
    import torch
    import torch.nn as nn
    from sklearn.utils.class_weight import compute_class_weight
    from torch.utils.data import DataLoader

    from src.config import IMG_SIZE, NUM_WORKERS
    from src.models._torch_lib import (
        LettuceDataset,
        _eval_transforms,
        _train_transforms,
        select_device,
    )

    spec = MODELS[cfg.model_name]
    model_module = importlib.import_module(spec["module"])
    device = select_device()
    seed_everything(SPLIT_SEED + fold_idx)

    train_df = train_df.assign(
        label_idx=train_df["class"].map(CLASSES.index).astype(np.int64)
    )
    val_df = val_df.assign(
        label_idx=val_df["class"].map(CLASSES.index).astype(np.int64)
    )

    cap = 200 if cfg.smoke else None
    if cap is not None:
        train_df = train_df.head(cap)
        val_df = val_df.head(cap)

    train_loader = DataLoader(
        LettuceDataset(train_df, _train_transforms()),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        LettuceDataset(val_df, _eval_transforms()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = model_module.build_model().to(device)

    y_train = train_df["label_idx"].to_numpy()
    cw_values = compute_class_weight(
        class_weight="balanced", classes=np.arange(NUM_CLASSES), y=y_train
    )
    cw_tensor = torch.tensor(cw_values, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)

    log.info(
        "[%s][fold %d] train=%d val=%d epochs=%d device=%s",
        cfg.model_name,
        fold_idx,
        len(train_df),
        len(val_df),
        cfg.epochs,
        device,
    )
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        log.info("[%s][fold %d] epoch %d/%d done", cfg.model_name, fold_idx, epoch, cfg.epochs)

    model.eval()
    probs_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            probs_chunks.append(torch.softmax(model(x), dim=1).cpu().numpy())
    probs = np.vstack(probs_chunks)
    y_true = val_df["label_idx"].to_numpy()
    y_pred = probs.argmax(axis=1)

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return _fold_metrics(y_true, y_pred)


# ── Orchestrator ──────────────────────────────────────────────────────────────


def cross_validate(cfg: CVConfig) -> dict:
    configure_logging()

    if cfg.model_name not in MODELS:
        raise KeyError(
            f"unknown model: {cfg.model_name}. available: {list(MODELS)}"
        )

    df = pd.read_csv(PROCESSED_DIR / "all.csv")
    df["label_idx"] = df["class"].map(CLASSES.index).astype(np.int64)

    sgkf = StratifiedGroupKFold(
        n_splits=cfg.n_folds, shuffle=True, random_state=SPLIT_SEED
    )
    folds = list(sgkf.split(df, y=df["class"], groups=df["source_frame"]))

    out_dir = REPORTS_DIR / f"cv_{cfg.model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    framework = MODELS[cfg.model_name]["framework"]
    run_fold = _run_fold_tf if framework == "tf" else _run_fold_torch

    per_fold: list[dict] = []
    for i, (train_idx, val_idx) in enumerate(folds):
        log.info(
            "=== CV fold %d/%d  (%s) ===", i + 1, cfg.n_folds, cfg.model_name
        )
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        # Leakage guard inside the fold itself
        leak = set(train_df["source_frame"]) & set(val_df["source_frame"])
        assert not leak, f"fold {i}: {len(leak)} source frames leaked"
        fold_metrics = run_fold(cfg, train_df, val_df, i)
        fold_metrics["fold"] = i
        fold_metrics["n_train"] = int(len(train_df))
        fold_metrics["n_val"] = int(len(val_df))
        per_fold.append(fold_metrics)
        with open(out_dir / f"fold_{i}.json", "w") as f:
            json.dump(fold_metrics, f, indent=2)

    summary = _summarise(per_fold, cfg)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _write_markdown(summary, out_dir / "summary.md")
    log.info(
        "[%s] CV done  acc=%.4f±%.4f  macroF1=%.4f±%.4f",
        cfg.model_name,
        summary["mean"]["accuracy"],
        summary["std"]["accuracy"],
        summary["mean"]["macro_f1"],
        summary["std"]["macro_f1"],
    )
    return summary


def _summarise(per_fold: list[dict], cfg: CVConfig) -> dict:
    keys = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]

    def _stat(fn, key):
        values = [f[key] for f in per_fold]
        return float(fn(values)) if len(values) > 1 else float(values[0])

    means = {k: _stat(statistics.fmean, k) for k in keys}
    stds = {
        k: float(statistics.stdev([f[k] for f in per_fold]))
        if len(per_fold) > 1
        else 0.0
        for k in keys
    }
    return {
        "model": cfg.model_name,
        "n_folds": cfg.n_folds,
        "epochs": cfg.epochs,
        "mean": means,
        "std": stds,
        "folds": [
            {k: f[k] for k in ("fold", "n_train", "n_val", *keys)} for f in per_fold
        ],
    }


def _write_markdown(summary: dict, path: Path) -> None:
    keys = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    lines = [
        f"# Cross-validation — {summary['model']}",
        "",
        f"- folds: **{summary['n_folds']}**  | epochs/fold: **{summary['epochs']}**",
        "",
        "## Summary (mean ± std across folds)",
        "",
        "| metric | mean | std |",
        "|---|---|---|",
    ]
    for k in keys:
        lines.append(
            f"| {k} | {summary['mean'][k]:.4f} | {summary['std'][k]:.4f} |"
        )
    lines += ["", "## Per-fold", "", "| fold | n_train | n_val | acc | macro_f1 |", "|---|---|---|---|---|"]
    for f in summary["folds"]:
        lines.append(
            f"| {f['fold']} | {f['n_train']} | {f['n_val']} | "
            f"{f['accuracy']:.4f} | {f['macro_f1']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="K-fold CV for one backbone.")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    cfg = CVConfig(
        model_name=args.model,
        n_folds=args.n_folds,
        epochs=args.epochs,
        smoke=args.smoke,
    )
    cross_validate(cfg)


if __name__ == "__main__":
    main()
