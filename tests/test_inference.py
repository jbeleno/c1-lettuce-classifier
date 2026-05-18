import numpy as np
import pytest
from PIL import Image

from src.inference import _to_array, REGISTRY, available_models


def test_registry_has_all_expected_models():
    assert set(REGISTRY.keys()) == {
        "mobilenet_v3_small",
        "efficientnet_b0",
        "resnet50",
        "vit_b_16",
        "swin_t",
        "ensemble_avg",
    }


def test_to_array_handles_pil_image():
    img = Image.new("RGB", (300, 200), (128, 64, 32))
    arr = _to_array(img)
    assert arr.shape == (224, 224, 3)


def test_to_array_handles_rgba():
    img = Image.new("RGBA", (224, 224), (10, 20, 30, 200))
    arr = _to_array(np.asarray(img))
    assert arr.shape[-1] == 3


def test_to_array_handles_grayscale():
    arr = _to_array(np.zeros((224, 224), dtype=np.uint8))
    assert arr.shape == (224, 224, 3)


def test_available_models_returns_list():
    # No models trained yet -> empty list (or just trained ones). Either way,
    # it should be a list and a subset of the registry.
    avail = available_models()
    assert isinstance(avail, list)
    for name in avail:
        assert name in REGISTRY
