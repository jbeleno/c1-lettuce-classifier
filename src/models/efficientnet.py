"""EfficientNet-B0 — strong accuracy/parameters tradeoff. The midsize CNN of
the comparison."""
from __future__ import annotations

import argparse

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from src.config import IMG_SIZE, NUM_CLASSES
from src.models._tf_lib import (
    TrainConfig,
    evaluate_and_save,
    fit_transfer,
    load_splits,
)
from src.utils import configure_logging

NAME = "efficientnet_b0"


def build_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(
        NUM_CLASSES, activation="softmax", dtype="float32"
    )(x)
    model = tf.keras.Model(inputs, outputs, name=NAME)
    return model, base


def main(smoke: bool = False) -> None:
    configure_logging()
    cfg = TrainConfig(model_name=NAME, smoke=smoke)
    train_ds, val_ds, test_ds, test_df = load_splits(preprocess_input, smoke=smoke)
    model, base = build_model()
    history = fit_transfer(model, base, train_ds, val_ds, cfg)
    evaluate_and_save(model, NAME, test_ds, test_df, history, cfg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    main(smoke=args.smoke)
