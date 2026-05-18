"""Shared helpers — deterministic seeding, logging, lightweight stopwatch."""
from __future__ import annotations

import logging
import os
import random
import time
from contextlib import contextmanager

import numpy as np


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except Exception:
        pass


@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    yield
    logging.getLogger("timer").info("%s: %.2fs", label, time.perf_counter() - t0)


def source_frame_id(filename: str) -> str:
    """Roboflow names crops as ``<source>_jpg.rf.<hash>.jpg``. The portion
    before ``_jpg.rf.`` uniquely identifies the original frame the bbox was
    cropped from. We group by it to keep the train/val/test split free of
    same-frame leakage."""
    marker = "_jpg.rf."
    idx = filename.find(marker)
    return filename[:idx] if idx >= 0 else filename
