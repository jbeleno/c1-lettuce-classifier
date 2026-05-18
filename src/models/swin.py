"""Swin-Tiny from torchvision with ImageNet pretrained weights. Hierarchical
self-attention — a useful counterpoint to ViT's flat patch attention."""
from __future__ import annotations

import argparse

import torch.nn as nn
from torchvision import models as tvm

from src.config import NUM_CLASSES
from src.models._torch_lib import (
    TorchTrainConfig,
    evaluate_and_save_torch,
    load_splits,
    train_torch_model,
)
from src.utils import configure_logging

NAME = "swin_t"


def build_model() -> nn.Module:
    model = tvm.swin_t(weights=tvm.Swin_T_Weights.IMAGENET1K_V1)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, NUM_CLASSES)
    return model


def main(smoke: bool = False) -> None:
    configure_logging()
    cfg = TorchTrainConfig(model_name=NAME, smoke=smoke)
    train_loader, val_loader, test_loader, test_df = load_splits(smoke=smoke)
    model = build_model()
    history = train_torch_model(model, cfg, train_loader, val_loader)
    evaluate_and_save_torch(model, cfg, test_loader, test_df, history)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    main(smoke=args.smoke)
