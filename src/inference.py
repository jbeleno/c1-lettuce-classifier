"""Unified inference API. Every backbone (TF or PyTorch) and the ensemble are
exposed through a single ``predict(image, model_name)`` function. Models are
loaded lazily on first request and cached for the lifetime of the process.

Note on devices: TensorFlow is pinned to CPU inside this module so it cannot
fight PyTorch for the Apple MPS device. Single-image latency on CPU is well
under half a second for these backbones, which is comfortable for the GUI."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Callable

import numpy as np
from PIL import Image

# Pin TF to CPU before importing it anywhere — must run first.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src.config import CLASSES, IMG_SIZE, MODELS_DIR, NUM_CLASSES  # noqa: E402

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    model: str

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "model": self.model,
        }


# ── Model registry ────────────────────────────────────────────────────────────


@dataclass
class ModelSpec:
    name: str
    framework: str  # "tf" | "torch" | "ensemble"
    backbone: str | None = None  # for torchvision lookup


REGISTRY: dict[str, ModelSpec] = {
    "mobilenet_v3_small": ModelSpec("mobilenet_v3_small", "tf"),
    "efficientnet_b0": ModelSpec("efficientnet_b0", "tf"),
    "resnet50": ModelSpec("resnet50", "tf"),
    "vit_b_16": ModelSpec("vit_b_16", "torch", backbone="vit_b_16"),
    "swin_t": ModelSpec("swin_t", "torch", backbone="swin_t"),
    "ensemble_avg": ModelSpec("ensemble_avg", "ensemble"),
}

# RLock (not Lock) so the ensemble loader can recursively call
# _get_predictor() for each component without self-deadlocking.
_lock = RLock()
_cache: dict[str, Callable[[np.ndarray], np.ndarray]] = {}


def available_models() -> list[str]:
    """Return only the models that have a trained artifact on disk."""
    out = []
    for name, spec in REGISTRY.items():
        if spec.framework == "ensemble":
            # ensemble is virtual: needs at least 2 components on disk
            ready = sum(
                1
                for n, s in REGISTRY.items()
                if s.framework != "ensemble"
                and (MODELS_DIR / n / _artifact_filename(s)).exists()
            )
            if ready >= 2:
                out.append(name)
        elif (MODELS_DIR / name / _artifact_filename(spec)).exists():
            out.append(name)
    return out


def _artifact_filename(spec: ModelSpec) -> str:
    return "model.keras" if spec.framework == "tf" else "model.pt"


def model_metadata(name: str) -> dict | None:
    meta = MODELS_DIR / name / "metadata.json"
    if not meta.exists():
        return None
    with open(meta) as f:
        return json.load(f)


# ── TF backbones ──────────────────────────────────────────────────────────────


def _load_tf_predictor(name: str):
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    path = MODELS_DIR / name / "model.keras"
    model = tf.keras.models.load_model(path, compile=False)

    if name == "mobilenet_v3_small":
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
    elif name == "efficientnet_b0":
        from tensorflow.keras.applications.efficientnet import preprocess_input
    elif name == "resnet50":
        from tensorflow.keras.applications.resnet import preprocess_input
    else:
        raise ValueError(f"no TF preprocess for {name}")

    def predict(img: np.ndarray) -> np.ndarray:
        x = preprocess_input(img.astype(np.float32))[None, ...]
        return model(x, training=False).numpy()[0]

    return predict


# ── Torch backbones ───────────────────────────────────────────────────────────


def _load_torch_predictor(name: str, backbone: str):
    import torch
    import torch.nn as nn
    from torchvision import models as tvm
    from torchvision import transforms

    if backbone == "vit_b_16":
        model = tvm.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    elif backbone == "swin_t":
        model = tvm.swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
    else:
        raise ValueError(f"unknown torch backbone {backbone}")

    state = torch.load(
        MODELS_DIR / name / "model.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state)
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )

    def predict(img: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(img.astype(np.uint8))
        x = tfm(pil).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).numpy()[0]
        return probs

    return predict


# ── Ensemble (delegates to components) ───────────────────────────────────────


def _load_ensemble_predictor():
    components = [
        n
        for n, s in REGISTRY.items()
        if s.framework != "ensemble"
        and (MODELS_DIR / n / _artifact_filename(s)).exists()
    ]

    if len(components) < 2:
        raise RuntimeError("ensemble needs >= 2 trained component models")

    component_predictors = [_get_predictor(n) for n in components]

    def predict(img: np.ndarray) -> np.ndarray:
        accum = np.zeros(NUM_CLASSES, dtype=np.float64)
        for pred in component_predictors:
            accum += pred(img)
        return accum / len(component_predictors)

    return predict


def _get_predictor(name: str):
    if name in _cache:
        return _cache[name]
    with _lock:
        if name in _cache:
            return _cache[name]
        if name not in REGISTRY:
            raise KeyError(f"unknown model: {name}")
        spec = REGISTRY[name]
        log.info("loading %s (%s) ...", name, spec.framework)
        if spec.framework == "tf":
            predictor = _load_tf_predictor(name)
        elif spec.framework == "torch":
            predictor = _load_torch_predictor(name, spec.backbone)
        else:
            predictor = _load_ensemble_predictor()
        _cache[name] = predictor
        log.info("loaded %s", name)
        return predictor


# ── Public API ────────────────────────────────────────────────────────────────


def _to_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = np.asarray(image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
        img = np.asarray(
            Image.fromarray(img.astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))
        )
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def predict(image: Image.Image | np.ndarray, model_name: str) -> Prediction:
    arr = _to_array(image)
    probs = _get_predictor(model_name)(arr)
    idx = int(np.argmax(probs))
    return Prediction(
        label=CLASSES[idx],
        confidence=float(probs[idx]),
        probabilities={c: float(probs[i]) for i, c in enumerate(CLASSES)},
        model=model_name,
    )


def warmup(names: list[str] | None = None) -> None:
    """Force-load the listed models (or every available model) so the first
    real request doesn't pay the cold-start cost."""
    for n in names or available_models():
        _get_predictor(n)
