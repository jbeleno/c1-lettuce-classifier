"""Shared TensorFlow plumbing used by the three CNN trainers (MobileNetV3,
EfficientNet-B0, ResNet50). Keeps each backbone file short and focused on the
parts that actually differ (the base model + the preprocess function)."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    BATCH_SIZE,
    CLASSES,
    CLASS_COLORS,
    IMG_SIZE,
    MIXED_PRECISION,
    MODELS_DIR,
    NUM_CLASSES,
    SPLITS_DIR,
    SPLIT_SEED,
    TF_FINE_TUNE_EPOCHS,
    TF_HEAD_EPOCHS,
)
from src.utils import seed_everything

log = logging.getLogger(__name__)
AUTOTUNE = tf.data.AUTOTUNE


def _maybe_enable_mixed_precision() -> bool:
    """Turn on fp16 mixed precision when running on a CUDA GPU. The lab
    workstation runs an RTX 2090 (Turing) which gets a real wall-time win
    from this; on CPU or Apple Metal we leave float32 alone."""
    if MIXED_PRECISION == "0":
        return False
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return False
    # Memory growth keeps TF from grabbing every byte of the card up-front,
    # which matters when other students share the workstation.
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        log.info("TF mixed precision enabled (mixed_float16) on %d GPU(s)", len(gpus))
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("could not enable mixed precision: %s", exc)
        return False


_MP_ENABLED = _maybe_enable_mixed_precision()


# ── Dataset construction ──────────────────────────────────────────────────────


def _label_to_int(label: str) -> int:
    return CLASSES.index(label)


def _read_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(SPLITS_DIR / f"{name}.csv")
    df["label_idx"] = df["class"].map(_label_to_int).astype(np.int32)
    return df


def _decode_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    raw = tf.io.read_file(path)
    img = tf.io.decode_jpeg(raw, channels=3, try_recover_truncated=True)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32)
    return img, label


def _make_dataset(
    df: pd.DataFrame,
    *,
    preprocess: Callable[[tf.Tensor], tf.Tensor],
    augment: bool,
    shuffle: bool,
    batch_size: int,
    smoke_cap: int | None = None,
) -> tf.data.Dataset:
    if smoke_cap is not None:
        df = df.sample(min(smoke_cap, len(df)), random_state=SPLIT_SEED).reset_index(
            drop=True
        )
    paths = df["filepath"].tolist()
    labels = df["label_idx"].tolist()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(8192, len(df)), seed=SPLIT_SEED)
    ds = ds.map(_decode_image, num_parallel_calls=AUTOTUNE)

    if augment:
        aug = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.15),
                tf.keras.layers.RandomBrightness(0.15),
                tf.keras.layers.RandomContrast(0.15),
                tf.keras.layers.RandomZoom(0.10),
            ],
            name="augment",
        )
        ds = ds.map(
            lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE
        )

    ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def load_splits(
    preprocess: Callable[[tf.Tensor], tf.Tensor],
    *,
    batch_size: int = BATCH_SIZE,
    smoke: bool = False,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, pd.DataFrame]:
    cap = 500 if smoke else None
    train_df = _read_split("train")
    val_df = _read_split("val")
    test_df = _read_split("test")
    train_ds = _make_dataset(
        train_df,
        preprocess=preprocess,
        augment=True,
        shuffle=True,
        batch_size=batch_size,
        smoke_cap=cap,
    )
    val_ds = _make_dataset(
        val_df,
        preprocess=preprocess,
        augment=False,
        shuffle=False,
        batch_size=batch_size,
        smoke_cap=cap,
    )
    test_ds = _make_dataset(
        test_df,
        preprocess=preprocess,
        augment=False,
        shuffle=False,
        batch_size=batch_size,
        smoke_cap=cap,
    )
    return train_ds, val_ds, test_ds, test_df.head(cap) if cap else test_df


def maybe_float32_head(model: tf.keras.Model) -> tf.keras.Model:
    """Sanity check: when mixed precision is enabled, the last layer's
    dtype must stay float32 for numerically stable softmax + loss."""
    if not _MP_ENABLED:
        return model
    last = model.layers[-1]
    if hasattr(last, "dtype") and last.dtype != "float32":
        log.warning("last layer dtype is %s, expected float32", last.dtype)
    return model


def class_weights() -> dict[int, float]:
    df = _read_split("train")
    y = df["label_idx"].to_numpy()
    classes = np.arange(NUM_CLASSES)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(wi) for c, wi in zip(classes, w)}


# ── Training run helpers ──────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    model_name: str
    epochs: int = TF_HEAD_EPOCHS
    fine_tune_epochs: int = TF_FINE_TUNE_EPOCHS
    learning_rate: float = 1e-3
    fine_tune_lr: float = 1e-5
    smoke: bool = False
    mixed_precision: bool = _MP_ENABLED

    def __post_init__(self):
        if self.smoke:
            self.epochs = 1
            self.fine_tune_epochs = 0


def make_callbacks(name: str) -> list[tf.keras.callbacks.Callback]:
    out_dir = MODELS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.CSVLogger(str(out_dir / "training_log.csv")),
    ]


def evaluate_and_save(
    model: tf.keras.Model,
    name: str,
    test_ds: tf.data.Dataset,
    test_df: pd.DataFrame,
    history: tf.keras.callbacks.History,
    config: TrainConfig,
) -> dict:
    """Run the test set through the model, persist probabilities for the
    ensemble, confusion matrix PNG and a metadata.json with headline metrics."""
    out_dir = MODELS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    probs = model.predict(test_ds, verbose=0)
    y_true = test_df["label_idx"].to_numpy()
    y_pred = probs.argmax(axis=1)
    test_acc = float((y_pred == y_true).mean())

    prob_df = pd.DataFrame(probs, columns=[f"p_{c}" for c in CLASSES])
    prob_df["filepath"] = test_df["filepath"].values
    prob_df["true_label"] = test_df["class"].values
    prob_df.to_parquet(out_dir / "test_probs.parquet", index=False)

    report = classification_report(
        y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    _save_confusion_png(cm, name)

    best_val = max(history.history.get("val_accuracy", [0.0]))
    metadata = {
        "name": name,
        "config": asdict(config),
        "test_accuracy": test_acc,
        "best_val_accuracy": float(best_val),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("[%s] test_acc=%.4f  best_val_acc=%.4f", name, test_acc, best_val)
    return metadata


def _save_confusion_png(cm: np.ndarray, name: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    out = MODELS_DIR / name / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="mako",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar=False,
        ax=ax,
        annot_kws={"color": "#F8FAFC"},
    )
    ax.set_title(f"{name} — confusion matrix", color="#F8FAFC")
    ax.set_xlabel("predicted", color="#CBD5E1")
    ax.set_ylabel("true", color="#CBD5E1")
    ax.tick_params(colors="#CBD5E1")
    plt.tight_layout()
    plt.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def fit_transfer(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config: TrainConfig,
) -> tf.keras.callbacks.History:
    """Two-phase fit: freeze the backbone and train the head, then unfreeze
    the top of the backbone and fine-tune at a small learning rate."""
    seed_everything(SPLIT_SEED)

    base_model.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    cw = class_weights()
    log.info("[%s] head training (%d epochs)", config.model_name, config.epochs)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        class_weight=cw,
        callbacks=make_callbacks(config.model_name),
        verbose=2,
    )
    if config.fine_tune_epochs > 0:
        log.info(
            "[%s] fine-tune (%d epochs, lr=%.0e)",
            config.model_name,
            config.fine_tune_epochs,
            config.fine_tune_lr,
        )
        base_model.trainable = True
        # Only unfreeze the top 30% of layers; freeze BN to keep statistics
        n = len(base_model.layers)
        for layer in base_model.layers[: int(n * 0.7)]:
            layer.trainable = False
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.fine_tune_lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.fine_tune_epochs,
            class_weight=cw,
            callbacks=make_callbacks(config.model_name),
            verbose=2,
        )
        for k, v in ft.history.items():
            history.history.setdefault(k, []).extend(v)
    return history
