"""Shared PyTorch plumbing for the two transformer backbones (ViT-B/16 and
Swin-Tiny). Mirrors what ``_tf_lib`` does for the TensorFlow side: dataset,
train loop, evaluation + persistence."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import (
    BATCH_SIZE,
    CLASSES,
    IMG_SIZE,
    MIXED_PRECISION,
    MODELS_DIR,
    NUM_CLASSES,
    NUM_WORKERS,
    SPLITS_DIR,
    SPLIT_SEED,
    TORCH_EPOCHS,
)
from src.utils import seed_everything

log = logging.getLogger(__name__)


def select_device() -> torch.device:
    """Prefer CUDA when available — it benefits from mixed precision in a way
    that Apple Metal doesn't. Falls back to MPS for laptop dev, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mixed_precision_enabled(device: torch.device) -> bool:
    if MIXED_PRECISION == "0":
        return False
    return device.type == "cuda"


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _train_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _eval_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class LettuceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        return self.transform(img), int(row["label_idx"])


def _read_split(name: str) -> pd.DataFrame:
    """Reads ``data/splits/<name>.csv``. For the training split, prefers
    ``train_balanced.csv`` when present so the sample-level balancing
    survives the trip into the PyTorch DataLoader."""
    csv = SPLITS_DIR / f"{name}.csv"
    if name == "train":
        balanced = SPLITS_DIR / "train_balanced.csv"
        if balanced.exists():
            csv = balanced
    df = pd.read_csv(csv)
    df["label_idx"] = df["class"].map(CLASSES.index).astype(np.int64)
    return df


def load_splits(
    *,
    batch_size: int = BATCH_SIZE,
    smoke: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    def _maybe_cap(df):
        return (
            df.sample(min(500, len(df)), random_state=SPLIT_SEED).reset_index(drop=True)
            if smoke
            else df
        )

    train_df = _maybe_cap(_read_split("train"))
    val_df = _maybe_cap(_read_split("val"))
    test_df = _maybe_cap(_read_split("test"))

    pin = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": NUM_WORKERS,
        "pin_memory": pin,
        "persistent_workers": NUM_WORKERS > 0,
    }

    train_loader = DataLoader(
        LettuceDataset(train_df, _train_transforms()), shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        LettuceDataset(val_df, _eval_transforms()), shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        LettuceDataset(test_df, _eval_transforms()), shuffle=False, **loader_kwargs
    )
    return train_loader, val_loader, test_loader, test_df


def class_weight_tensor(device: torch.device) -> torch.Tensor:
    df = _read_split("train")
    y = df["label_idx"].to_numpy()
    classes = np.arange(NUM_CLASSES)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return torch.tensor(w, dtype=torch.float32, device=device)


@dataclass
class TorchTrainConfig:
    model_name: str
    epochs: int = TORCH_EPOCHS
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    smoke: bool = False

    def __post_init__(self):
        if self.smoke:
            self.epochs = 1


def _epoch_loop(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion,
    optimizer,
    device,
    train: bool,
    scaler: "torch.amp.GradScaler | None" = None,
) -> tuple[float, float]:
    use_amp = scaler is not None and train
    model.train(mode=train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train_torch_model(
    model: nn.Module,
    config: TorchTrainConfig,
    train_loader,
    val_loader,
) -> dict[str, list[float]]:
    seed_everything(SPLIT_SEED)
    device = select_device()
    use_amp = mixed_precision_enabled(device)
    log.info(
        "[%s] device=%s  mixed_precision=%s  batch_size=%d  num_workers=%d",
        config.model_name,
        device,
        use_amp,
        train_loader.batch_size,
        train_loader.num_workers,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    out_dir = MODELS_DIR / config.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_log_path = out_dir / "training_log.csv"
    with open(csv_log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    for epoch in range(1, config.epochs + 1):
        tr_loss, tr_acc = _epoch_loop(
            model,
            train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
            scaler=scaler,
        )
        v_loss, v_acc = _epoch_loop(
            model,
            val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=False,
            scaler=None,
        )
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        log.info(
            "[%s] epoch %d/%d  train_acc=%.4f  val_acc=%.4f",
            config.model_name,
            epoch,
            config.epochs,
            tr_acc,
            v_acc,
        )
        with open(csv_log_path, "a") as f:
            f.write(f"{epoch},{tr_loss:.4f},{tr_acc:.4f},{v_loss:.4f},{v_acc:.4f}\n")
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), out_dir / "model.pt")
    return history


def evaluate_and_save_torch(
    model: nn.Module,
    config: TorchTrainConfig,
    test_loader,
    test_df: pd.DataFrame,
    history: dict[str, list[float]],
) -> dict:
    device = select_device()
    out_dir = MODELS_DIR / config.model_name
    state = torch.load(out_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    probs_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            logits = model(x)
            probs_chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.vstack(probs_chunks)

    y_true = test_df["label_idx"].to_numpy()
    y_pred = probs.argmax(1)
    test_acc = float((y_pred == y_true).mean())

    prob_df = pd.DataFrame(probs, columns=[f"p_{c}" for c in CLASSES])
    prob_df["filepath"] = test_df["filepath"].values
    prob_df["true_label"] = test_df["class"].values
    prob_df.to_parquet(out_dir / "test_probs.parquet", index=False)

    report = classification_report(
        y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    _save_confusion_png_torch(cm, config.model_name)

    metadata = {
        "name": config.model_name,
        "config": asdict(config),
        "test_accuracy": test_acc,
        "best_val_accuracy": float(max(history.get("val_acc", [0.0]))),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(
        "[%s] test_acc=%.4f  best_val_acc=%.4f",
        config.model_name,
        test_acc,
        metadata["best_val_accuracy"],
    )
    return metadata


def _save_confusion_png_torch(cm: np.ndarray, name: str) -> None:
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
