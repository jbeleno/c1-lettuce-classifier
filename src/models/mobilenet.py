"""MobileNetV3-Small — lightweight CNN baseline. Fast to train and small
enough to ship to an edge device, useful as the lower bound everyone else
must beat."""
from __future__ import annotations

import argparse

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

from src.config import IMG_SIZE, NUM_CLASSES
from src.models._tf_lib import (
    TrainConfig,
    evaluate_and_save,
    fit_transfer,
    load_splits,
)
from src.utils import configure_logging

NAME = "mobilenet_v3_small"


def build_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    # `include_preprocessing=True` lets the network apply its own Rescaling
    # layer (maps [0, 255] -> [-1, 1]). Earlier we set this to False without
    # adding manual normalization, which fed the ImageNet-trained weights
    # pixel values ~127x larger than expected -- loss exploded into the 1k
    # range and accuracy never broke ~35 %.
    base = MobileNetV3Small(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
        include_preprocessing=True,
    )
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.Dropout(0.3)(x)
    # dtype="float32" pins the softmax + loss to fp32 even when global mixed
    # precision is on; the rest of the graph still runs in fp16 on GPU.
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
